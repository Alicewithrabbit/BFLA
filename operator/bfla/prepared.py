"""Explicit prepare/run API over original, packed Q/K/V.

Preparation owns bounded metadata and workspace; run
accepts the same geometry and computes fresh supports on the caller's stream.
"""
from dataclasses import asdict
from math import isfinite
from types import SimpleNamespace
import torch
from . import build
from .config import BFLAConfig
from .capabilities import supports
from .features import make_features
from .planning import make_plan
from .state import pool_features, IndexPoolState
from ._limits import KEY_READER_ROWS
from .metadata import MetadataCache

METADATA = MetadataCache()


def _plan_metadata(spec, heads):
    tensor = lambda x: torch.tensor(x, device='cpu', dtype=torch.int32)
    csr = [0]
    for _ in range(heads.local_hkv):
        for count in spec.counts: csr.append(csr[-1]+count)
    values = dict(csr=tensor(csr), used=tensor(spec.layout['used_k']),
                  cu_q=tensor(spec.layout['cu_q']), cu_tiles=tensor(spec.layout['cu_tiles']))
    if spec.layout['table'] is not None:
        values['table'] = tensor(spec.layout['table'])
    values['segment_requests'] = torch.tensor(spec.layout['segment_requests'], device='cpu', dtype=torch.int64)
    for i, b in enumerate(spec.batches):
        for name, data in dict(query_ids=b.query_indices, rows=b.rows,
                sizes=[g.size for g in b.groups], lengths=[g.count for g in b.groups],
                offsets=[g.consumer_tiles[0] for g in b.groups],
                maps=[(g.consumer_tiles[0],len(g.consumer_tiles),g.last//64+1) for g in b.groups]).items():
            values[f'{i}:{name}'] = tensor(data)
    return values


def _device_capability(tensor):
    if not tensor.is_cuda:
        raise ValueError('BFLA execution requires CUDA tensors')
    return torch.cuda.get_device_capability(tensor.device)


def _load_kernels():
    import triton
    from ._kernels.runtime import identity
    from ._kernels.selection import score_rows, publish_rows
    from ._kernels.features import score_index
    from ._kernels.publish import publish_groups
    return SimpleNamespace(identity=identity, score_rows=score_rows, score_index=score_index,
        publish_rows=publish_rows, publish_groups=publish_groups,
        next_power_of_2=triton.next_power_of_2, cdiv=triton.cdiv)


def _signature(x):
    return tuple(x.shape), tuple(x.stride()), x.dtype, x.device


def _tma_aligned(x):
    """CUDA tensor-map contract, shared by initial preparation and rebinding."""
    return (x.stride(-1) == 1 and x.data_ptr() % 16 == 0 and
            all(0 < s*x.element_size() < 2**40 and s*x.element_size() % 16 == 0
                for s in x.stride()[:-1]))


def _overlap(a, b):
    # Inputs have nonnegative strides. Output is contiguous and disjoint.
    end = b.data_ptr() + (1 + sum((s-1)*t for s,t in zip(b.shape,b.stride()))) * b.element_size()
    return a.data_ptr() < end and b.data_ptr() < a.data_ptr() + a.numel()*a.element_size()


class BFLAPrefillPlan:
    def __init__(self, q, k, v, scale, *, request_lengths, head_layout, all_reduce=None,
                 kv_lengths=None, prompt_lengths=None, kv_padded=False,
                 kv_page_offsets=None, kv_page_table=None, config=BFLAConfig()):
        self.config, self.heads = config, head_layout
        self.paged = kv_page_table is not None
        if self.paged and (kv_padded or kv_page_offsets is not None):
            raise ValueError("native pages and contiguous padded segments are mutually exclusive")
        self.lengths = tuple(request_lengths)
        self.kv_lengths = self.lengths if kv_lengths is None else tuple(kv_lengths)
        self.prompt_lengths = self.kv_lengths if prompt_lengths is None else tuple(prompt_lengths)
        supports(head_layout, config, device_capability=_device_capability(q),
            dtype=str(q.dtype).removeprefix('torch.'), request_lengths=self.lengths,
            kv_lengths=self.kv_lengths, prompt_lengths=self.prompt_lengths,
            kv_layout='paged' if self.paged else ('padded' if kv_padded else 'ragged')).require()
        self.n, self.hq, self.hk, self.d = sum(self.lengths), head_layout.local_hq, head_layout.local_hkv, head_layout.head_dim
        key_rows = sum((n+63)//64*64 if kv_padded else n for n in self.kv_lengths)
        self.arena = kv_page_offsets is not None
        if self.arena:
            if not kv_padded:
                raise ValueError('explicit page offsets require padded block64 storage')
            key_rows = k.shape[0]
        if (q.shape != (self.n,self.hq,self.d) or (not self.paged and k.shape != (key_rows,self.hk,self.d)) or
            (self.paged and (k.ndim != 4 or k.shape[1]<1 or k.shape[2:] != (self.hk,self.d))) or v.shape != k.shape or
            any(x.device != q.device or x.dtype != torch.bfloat16 or not _tma_aligned(x)
                for x in (q,k,v))):
            raise ValueError('Q/K/V must match packed BF16 geometry and TMA alignment')
        if (self.d == 512 and any(not x.is_contiguous() for x in ((q,) if self.paged else (q,k,v)))) or (
            head_layout.tp_size == 1 and not q.is_contiguous()):
            raise ValueError('this feature/attention variant requires contiguous inputs')
        if head_layout.tp_size > 1 and all_reduce is None:
            raise ValueError('TP features require an all_reduce callback')
        if not isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and positive')
        self.scale, self.device = scale, q.device
        self._signatures = tuple(_signature(x) for x in (q,k,v))
        self.spec = make_plan(self.lengths,head_layout,config,kv_lengths=self.kv_lengths,
                              prompt_lengths=self.prompt_lengths,native_paged=self.paged)
        self.layout, self.tiles = dict(self.spec.layout), len(self.spec.counts)
        self.kernels = _load_kernels()
        i32 = dict(device=q.device,dtype=torch.int32)
        fp = dict(device=q.device,dtype=torch.float32)
        metadata_key = (head_layout, config, self.lengths, self.kv_lengths,
                        tuple(r.sparse_rows for r in self.spec.requests), self.paged)
        metadata = METADATA.get(metadata_key, lambda: _plan_metadata(self.spec, head_layout), q.device)
        self.csr, self.ids = metadata['csr'], torch.empty(self.hk*sum(self.spec.counts),**i32)
        self.table = None if self.paged else metadata['table']
        if self.arena: self.table = self.table.clone()
        self.used = metadata['used']
        self.cu_q, self.cu_tiles = metadata['cu_q'], metadata['cu_tiles']
        self.scheduler = torch.empty(self.layout['scheduler_words'],**i32)
        self.out = torch.empty(q.shape,device=q.device,dtype=q.dtype)
        self.lse = torch.empty((self.hq,self.n),**fp)
        self.num_sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        self.selector_batches = tuple(SimpleNamespace(spans=b.spans,
            **{name: metadata[f'{i}:{name}'] for name in
               ('query_ids','rows','sizes','lengths','offsets','maps')})
            for i,b in enumerate(self.spec.batches))
        self.selector_invocations = len(self.selector_batches)
        self.key_starts = []
        start = 0
        for n in self.kv_lengths:
            self.key_starts.append(start)
            start += (n+63)//64*64 if kv_padded else n
        self.kpad = self.vpad = None
        partial_block64 = not self.paged and any(n % 64 for n in self.kv_lengths)
        # The paged consumer zero-fills invalid V lanes on load. Packed TMA
        # can read through the tail only when the plan owns its zero padding.
        self._uses_paged_attention = self.paged or (kv_padded and partial_block64)
        if partial_block64 and not kv_padded:
            self.kpad = torch.zeros((self.layout['total_pages']*64,self.hk,self.d),device=q.device,dtype=q.dtype)
            self.vpad = torch.zeros_like(self.kpad)
        self.capacity = config.block_count(max(self.kv_lengths))
        self.nb = (max(self.kv_lengths)+config.pool_block_size-1)//config.pool_block_size
        with torch.cuda.device(q.device):
            self.attn = (build.paged_attention if self._uses_paged_attention else build.attention)(self.d)
            if self.selector_batches:
                maxq = max(len(b.query_indices) for b in self.spec.batches)
                maxgroups = max(len(b.groups) for b in self.spec.batches)
                key_capacity = min(KEY_READER_ROWS,max(self.kv_lengths)) if self.paged else max(self.kv_lengths)
                self.features = make_features(head_layout,key_capacity,maxq,all_reduce,config,q.device)
                self.key_features = torch.empty((key_capacity,config.index_dim),device=q.device,dtype=torch.bfloat16)
                self.key_scratch = (torch.empty((key_capacity,self.hk,self.d),device=q.device,dtype=q.dtype)
                                    if self.paged else None)
                self.qfeatures = torch.empty((maxq,config.index_dim),**fp)
                blocks = sum((n+config.pool_block_size-1)//config.pool_block_size for n in self.kv_lengths)
                dtype = torch.bfloat16 if config.index_dtype == 'bf16' else torch.float8_e4m3fn
                self.pool = torch.empty((blocks,config.index_dim),device=q.device,dtype=dtype)
                self.scales = torch.empty(blocks,**fp)
                self.scores = torch.empty(maxgroups*self.nb,**fp)
                self.values = torch.empty((maxgroups,self.capacity),**fp)
                self.selected = torch.empty((maxgroups,self.capacity),**i32)
                self.sel = build.selector(self.capacity, candidates=self.nb)
                self.workspace = torch.empty(self.sel.workspace_bytes(maxgroups,self.nb),device=q.device,dtype=torch.uint8)
            self.kernels.identity[(self.tiles,self.hk)](self.csr,self.ids,self.tiles,
                self.kernels.next_power_of_2((max(self.kv_lengths)+63)//64))
        self.statistics = dict(config=asdict(config),selection_groups=len(self.spec.groups),
            consumer_tiles=self.tiles,dense_tiles=len(self.spec.dense_tiles),
            selector_invocations=self.selector_invocations,selector_capacity=self.capacity,
            pool_blocks=sum((n+config.pool_block_size-1)//config.pool_block_size for n in self.kv_lengths),
            logical_budget_min=min((g.count*config.pool_block_size for g in self.spec.groups),default=0),
            logical_budget_max=max((g.count*config.pool_block_size for g in self.spec.groups),default=0))
        self.last_key_refresh_rows = (0,) * len(self.lengths)
        self._page_offsets = None
        if self.arena:
            self.bind_kv(k,v,kv_page_offsets)
        if self.paged:
            self.page_size = k.shape[1]
            width = (max(self.kv_lengths)+self.page_size-1)//self.page_size
            self.native_table = torch.empty((len(self.lengths),width),**i32)
            self.table = torch.empty((len(self.layout['used_k']),width),**i32)
            self.segment_requests = metadata['segment_requests']
            self.bind_paged_kv(k,v,kv_page_table)

    def bind_paged_kv(self, k, v, table, *, request_rows=None):
        """Bind physical pages; selections and causal boundaries remain logical."""
        if (not self.paged or k.ndim != 4 or k.shape != v.shape or
                k.shape[1:] != (self.page_size,self.hk,self.d) or
                any(x.dtype != torch.bfloat16 or x.device != self.device or not _tma_aligned(x)
                    for x in (k,v)) or
                table.ndim!=2 or (request_rows is None and table.shape[0]!=len(self.lengths)) or
                table.shape[1]<self.native_table.shape[1] or table.dtype!=torch.int32 or table.device!=self.device):
            raise ValueError('native KV/page-table geometry differs from prepared plan')
        source=table[:,:self.native_table.shape[1]]
        if request_rows is None:
            self.native_table.copy_(source)
        else:
            if request_rows.shape!=(len(self.lengths),) or request_rows.dtype!=torch.int64 or request_rows.device!=self.device:
                raise ValueError('one device request-row index required per request')
            torch.index_select(source,0,request_rows,out=self.native_table)
        torch.index_select(self.native_table,0,self.segment_requests,out=self.table)
        self.layout['total_pages']=k.shape[0]
        self._signatures=(self._signatures[0],_signature(k),_signature(v))

    def bind_kv(self, k, v, page_offsets):
        """Bind caller-owned block64 segments without changing logical supports.

        Rebinding is host preparation, outside graph capture. Only selection
        may be captured across different backing tensors; attention receives
        the current K/V pointers on every launch.
        """
        if not self.arena:
            raise ValueError('plan was not prepared for caller-owned page offsets')
        offsets = tuple(page_offsets)
        if (k.ndim != 3 or k.shape != v.shape or k.shape[1:] != (self.hk,self.d)
                or k.shape[0] % 64 or len(offsets) != len(self.lengths)
                or any(type(x) is not int or x < 0 for x in offsets)
                or any(not x.is_contiguous() or x.dtype != torch.bfloat16
                       or x.device != self.device or x.data_ptr() % 16 for x in (k,v))):
            raise ValueError('invalid block64 arena geometry or offsets')
        pages = k.shape[0]//64
        ranges = sorted((off,off+(n+63)//64) for off,n in zip(offsets,self.kv_lengths))
        if any(end > pages for _,end in ranges) or any(a[1] > b[0] for a,b in zip(ranges,ranges[1:])):
            raise ValueError('arena segments overlap or exceed backing capacity')
        if self._page_offsets != offsets:
            mapping = dict(zip(self.spec.layout['request_page_offsets'],offsets))
            rows = [[page+mapping[row[0]]-row[0] if page >= 0 else -1 for page in row]
                    for row in self.spec.layout['table']]
            metadata = METADATA.get(('arena_table', tuple(map(tuple, rows))),
                lambda: {'table': torch.tensor(rows,device='cpu',dtype=torch.int32)}, self.device)
            self.table.copy_(metadata['table'])
            self.layout['request_page_offsets'] = list(offsets)
            self.key_starts = [off*64 for off in offsets]
            self._page_offsets = offsets
        self.layout['total_pages'] = pages
        self._signatures = (self._signatures[0],_signature(k),_signature(v))

    def run(self, q, k, v, *, out=None, key_states=None):
        if (tuple(_signature(x) for x in (q,k,v)) != self._signatures or
            any(x.data_ptr() % 16 for x in (q,k,v))):
            raise ValueError('input geometry/dtype/device differs from prepared plan')
        target = self.out if out is None else out
        if (_signature(target) != _signature(self.out) or not target.is_contiguous() or target.data_ptr() % 16 or
            any(_overlap(target,x) for x in (q,k,v))):
            raise ValueError('output must be contiguous, geometry-compatible and disjoint from Q/K/V')
        with torch.cuda.device(self.device):
            self.prepare_keys(k,key_states=key_states)
            self.run_selection(q)
            return self.run_attention(q,k,v,out=target)

    def prepare_keys(self, k, *, key_states=None):
        """Populate stable compact index inputs, optionally incrementally.

        Call once before replay; selection and attention never update these
        states. Ownership and stream ordering belong to the caller.
        """
        if _signature(k) != self._signatures[1]:
            raise ValueError('key geometry differs from prepared binding')
        if key_states is not None and (len(key_states) != len(self.lengths) or
                any(s.config != self.config for s in key_states)):
            raise ValueError('one matching IndexPoolState is required per request')
        c = self.config
        refreshed = []
        for ri,(r,start) in enumerate(zip(self.spec.requests,self.key_starts)):
            n = r.key_length
            if r.sparse_rows:
                blocks = (n+c.pool_block_size-1)//c.pool_block_size
                pool, scales = self.pool[r.pool_start:r.pool_start+blocks], self.scales[r.pool_start:r.pool_start+blocks]
                if self.paged:
                    from ._kernels.paged import read_key_window
                    state = IndexPoolState(c) if key_states is None else key_states[ri]
                    def reader(a,b):
                        return read_key_window(k,self.native_table[ri],a,b,self.key_scratch)
                    refreshed.append(state.populate_from_reader(self.features,n,reader,self.key_features,
                        pool,scales,key_shape=(self.hk,self.d),key_dtype=k.dtype,window=KEY_READER_ROWS))
                elif key_states is None:
                    self.features.key(k[start:start+n],self.key_features)
                    pool_features(self.key_features,pool,scales,n,c)
                    refreshed.append(n)
                else:
                    refreshed.append(key_states[ri].populate(self.features,k[start:start+n],self.key_features,pool,scales))
            else: refreshed.append(0)
        self.last_key_refresh_rows = tuple(refreshed)

    def run_selection(self, q):
        """Refresh query-dependent supports from already prepared index keys."""
        if _signature(q) != self._signatures[0] or q.data_ptr() % 16:
            raise ValueError('query geometry differs from prepared plan')
        c, kernels = self.config, self.kernels
        stream = torch.cuda.current_stream(self.device).cuda_stream
        for b in self.selector_batches:
            rows = b.rows.shape[0]
            if self.heads.tp_size == 1:
                self.features.query_batch(q,self.qfeatures,b.spans)
            else:
                self.features.query_indexed(q,b.query_ids,self.qfeatures)
            args = (self.qfeatures,self.pool,self.scales,self.scores,b.rows,self.heads.selection_ratio,self.nb)
            grid = (rows,kernels.cdiv(self.nb,32))
            if (c.index_dim,c.query_tile_rows,c.pool_block_size) == (128,128,64):
                kernels.score_rows[grid](*args,PRUNED=self.d==256)
            else:
                kernels.score_index[grid](*args,c.index_dim,c.query_tile_rows,c.pool_block_size,PRUNED=self.d==256)
            self.sel.topk(self.scores.data_ptr(),self.values.data_ptr(),self.selected.data_ptr(),
                b.sizes.data_ptr(),b.lengths.data_ptr(),self.workspace.data_ptr(),self.workspace.numel(),
                rows,self.nb,self.capacity,stream)
            if (c.query_tile_rows,c.pool_block_size) == (128,64):
                kernels.publish_rows[(rows,self.hk)](self.selected,b.lengths,b.offsets,self.csr,self.ids,
                    self.tiles,self.capacity,kernels.next_power_of_2(self.capacity))
            else:
                kernels.publish_groups[(rows,self.hk)](self.selected,b.lengths,b.maps,self.csr,self.ids,
                    self.tiles,self.capacity,c.pool_block_size//64,c.query_tile_rows//128,
                    kernels.next_power_of_2(self.capacity))

    def run_attention(self, q, k, v, *, out=None):
        """Consume current original Q/K/V and the prepared logical supports."""
        target = self.out if out is None else out
        if (tuple(_signature(x) for x in (q,k,v)) != self._signatures or
                any(x.data_ptr() % 16 for x in (q,k,v)) or
                _signature(target) != _signature(self.out) or not target.is_contiguous()
                or target.data_ptr() % 16 or any(_overlap(target,x) for x in (q,k,v))):
            raise ValueError('attention inputs/output differ from prepared binding')
        if self.kpad is not None:
            for r,start,page in zip(self.spec.requests,self.key_starts,self.layout['request_page_offsets']):
                self.kpad[page*64:page*64+r.key_length].copy_(k[start:start+r.key_length])
                self.vpad[page*64:page*64+r.key_length].copy_(v[start:start+r.key_length])
            k,v = self.kpad,self.vpad
        stream = torch.cuda.current_stream(self.device).cuda_stream
        z = [self.n,self.layout['max_rows'],max(self.kv_lengths),self.hq,self.hk,self.tiles,self.num_sm,
             len(self.layout['used_k']),self.layout['total_pages'],self.table.shape[1]]
        if self._uses_paged_attention:
            if not self.paged:
                # Split the row dimension into block64 pages without copying.
                k = k.unflatten(0,(-1,64))
                v = v.unflatten(0,(-1,64))
            z += [k.shape[1],q.stride(0),q.stride(1),k.stride(1),k.stride(2),k.stride(0),
                  v.stride(1),v.stride(2),v.stride(0)]
        elif self.d == 256: z += [q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1)]
        self.attn.forward([q.data_ptr(),k.data_ptr(),v.data_ptr(),target.data_ptr(),self.lse.data_ptr(),
            self.table.data_ptr(),self.csr.data_ptr(),self.ids.data_ptr(),self.used.data_ptr(),
            self.cu_tiles.data_ptr(),self.scheduler.data_ptr(),self.cu_q.data_ptr()],z,self.scale,stream)
        return target
