"""Prefill execution in eager mode or with an adapter-owned CUDA graph."""
from contextlib import nullcontext
import torch
from bfla import BFLAPrefillPlan
from .capture import capture_context
from .collectives import feature_reducer
from .metadata import integer_tensor


class PreparedExecution:
    def __init__(self, q, k, v, scale, *, request_lengths, head_layout, group,
                 settings, kv_lengths=None, prompt_lengths=None,
                 copy_inputs=True, kv_page_table=None):
        self.heads, self.group, self.settings = head_layout, group, settings
        self.inputs = (tuple(x.clone(memory_format=torch.contiguous_format) for x in (q, k, v))
                       if copy_inputs else (q, k, v))
        self.plan = BFLAPrefillPlan(*self.inputs, scale, request_lengths=request_lengths,
            head_layout=head_layout, all_reduce=feature_reducer(group),
            kv_lengths=kv_lengths, prompt_lengths=prompt_lengths,
            kv_page_table=kv_page_table, config=settings.config)
        self.incremental = kv_page_table is not None
        self.output = self.plan.out
        self.key_states = None
        self.owner = None
        if settings.execution == 'localgraph':
            from .capture_lifetime import CaptureLifetime
            self.owner = CaptureLifetime(q.device)
        self.captures = self.replays = 0
        self.last_use = self.last_stream = None
        self.last_caller = None

    def wait_for_inputs(self):
        if self.owner is not None:
            self.owner.wait_for_last_use(torch.cuda.current_stream(self.output.device))
        elif self.last_stream is not None:
            current=torch.cuda.current_stream(self.output.device)
            if current.cuda_stream != self.last_stream: current.wait_event(self.last_use)

    def record_use(self):
        if self.owner is not None:
            self.owner.record_use(torch.cuda.current_stream(self.output.device))
        else:
            current=torch.cuda.current_stream(self.output.device)
            if self.last_use is None: self.last_use=torch.cuda.Event()
            self.last_use.record(current); self.last_stream=current.cuda_stream
            self.last_caller=current

    def bind_paged_kv(self, k, v, table, rows, key_states):
        self.wait_for_inputs()
        if getattr(self,'_native_rows',None) != tuple(rows):
            self._native_rows=tuple(rows)
            self._native_row_ids=integer_tensor(rows,device=k.device)
        self.plan.bind_paged_kv(k,v,table,request_rows=self._native_row_ids)
        # Reordering can replace the row IDs while this stream still reads them.
        self._native_row_ids.record_stream(torch.cuda.current_stream(k.device))
        self.inputs=(self.inputs[0],k,v)
        self.key_states=key_states

    def unbind_kv(self):
        # Selection graphs do not capture K/V pointers. Do not retain finished
        # requests' native KV views through the shape reservation cache.
        if self.incremental:
            self.inputs=(self.inputs[0],None,None)
            self.key_states=None

    def __call__(self, q, k, v, out):
        self.wait_for_inputs()
        for dst, src in zip(self.inputs, (q, k, v)):
            dst.copy_(src)
        return self.replay(out)

    def replay(self, out):
        if out.shape != self.output.shape or out.dtype != self.output.dtype or out.device != self.output.device:
            raise ValueError('caller output differs from reserved attention geometry')
        if self.incremental:
            self.plan.prepare_keys(self.inputs[1],key_states=self.key_states)
        operation = (lambda: self.plan.run_selection(self.inputs[0])) if self.incremental else (
            lambda: self.plan.run(*self.inputs))
        if self.settings.execution == 'eager':
            if self.incremental: operation()
            else: self.plan.run(*self.inputs, out=out, key_states=self.key_states)
        else:
            if self.key_states is not None and not self.incremental:
                raise ValueError('captured full pipelines cannot mutate incremental index state')
            current = torch.cuda.current_stream(self.output.device)
            if self.owner.graph is None:
                self.owner.stream.wait_stream(current)
                with torch.cuda.stream(self.owner.stream):
                    for _ in range(2):
                        operation()
                    self.owner.graph = torch.cuda.CUDAGraph()
                    context = capture_context(self.owner.stream) if self.heads.tp_size > 1 else nullcontext()
                    with context:
                        with torch.cuda.graph(self.owner.graph, stream=self.owner.stream):
                            operation()
                current.wait_stream(self.owner.stream)
                self.captures += 1
            self.owner.graph.replay()
            if not self.incremental and out.data_ptr() != self.output.data_ptr():
                out.copy_(self.output)
        if self.incremental:
            self.plan.run_attention(*self.inputs,out=out)
        self.record_use()
        self.replays += 1
        return out

    def close(self):
        if self.plan is None:
            return
        if self.owner is not None:
            self.owner.close()
        elif self.last_stream is not None:
            # Eager raw-pointer consumers are asynchronous. Record allocator
            # ownership instead of adding a host synchronize at every eviction.
            objects=[self.plan,*self.plan.selector_batches]
            if hasattr(self.plan,'features'): objects.append(self.plan.features)
            tensors=list(self.inputs)+[getattr(self,"_native_row_ids",None)]
            for obj in objects: tensors.extend(vars(obj).values())
            for tensor in tensors:
                if isinstance(tensor,torch.Tensor): tensor.record_stream(self.last_caller)
        self.plan = None
        self.output = None
        self.key_states = None
        self.inputs = ()
