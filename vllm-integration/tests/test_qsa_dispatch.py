"""Exercise QSA projection, routing and cache ownership without GPU kernels."""
import os
import sys
import unittest
from contextlib import ExitStack
from copy import copy
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

import bfla_vllm.execution as execution
from bfla_vllm.qsa import adapter, native_index


class UnquantizedLinearMethod:
    _gemm_impl = staticmethod(torch.nn.functional.linear)


class ReplicatedLinear:
    bias = None
    quant_method = UnquantizedLinearMethod()

    def __init__(self):
        self.weight = torch.zeros((4, 2), dtype=torch.bfloat16)
        self.projected_rows = []

    def __call__(self, hidden):
        self.projected_rows.append(len(hidden))
        return torch.nn.functional.linear(hidden, self.weight), None


class WithLoRA:
    """A wrapper exposing its base weight and quant method, as in vLLM."""

    def __init__(self, base_layer):
        self.base_layer = base_layer

    def __getattr__(self, name):
        return getattr(self.base_layer, name)

    def __call__(self, hidden):
        base, _ = self.base_layer(hidden)
        # A=[1, 0], B=[0, 0, 2, 3]. Last two columns are the index K.
        return base + hidden[:, :1] * hidden.new_tensor([0, 0, 2, 3]), None


class Indexer:
    raw_key_cache = NS(prefix='side')
    use_fused_pre_indexer = True
    index_kv_heads = index_n_heads = 1
    index_head_dim = 2
    skip_topk = False

    def __init__(self):
        self.index_qk_proj = ReplicatedLinear()
        self.keys = []

    def __call__(self, projected, positions, out):
        self.keys.append(projected[:, 2:].clone())
        return out


class QsaDispatch(unittest.TestCase):
    def exercise(self, *, tp=2, cache_dtype=torch.bfloat16, k_only=True,
                 lora=False, capture=False, capability=(9, 0), pynccl=True,
                 microbatch=False, check_retirement=False, draft=False):
        events = []
        comm = None if pynccl is None else NS(disabled=not pynccl)
        group = NS(world_size=tp, rank_in_group=0, device_communicator=NS(pynccl_comm=comm))
        main = NS(max_query_len=2, bfla_partition=(1, 2), bfla_rows=((1, 1, 3, 2),),
                  bfla_keys=('p',), bfla_prompts=(2,), num_actual_tokens=3,
                  block_table=torch.zeros((2, 1), dtype=torch.int32),
                  slot_mapping=torch.arange(3))
        side = NS(num_decode_tokens=1, num_prefill_tokens=2, num_actual_tokens=3,
                  token_to_req=torch.tensor([0, 1, 1]))
        indexer = Indexer()
        hidden = torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=torch.bfloat16)
        if not lora:
            indexer.index_qk_proj.weight[2:, 0] = hidden.new_tensor([5, 1])

        class Attention:
            layer_name = 'attn'
            head_dim, total_num_heads, total_num_kv_heads = 256, 24, 2
            num_heads, num_kv_heads = 24 // tp, max(1, 2 // tp)
            scaling = 256 ** -0.5
            kv_cache = torch.zeros((1, num_kv_heads, 16, 512), dtype=cache_dtype)
            topk_indices_buffer = torch.zeros((3, 4), dtype=torch.int32)

            def qkv_proj(self, x):
                events.append('qkv')
                return x, None

            def _project_qkv_gate(self, x, positions):
                q = x.new_zeros((len(x), self.num_heads, self.head_dim))
                k = q[:, :self.num_kv_heads]
                return q, k, k, q

            def o_proj(self, output):
                return output, None

            def forward(self, positions, x):
                qkv, _ = self.qkv_proj(x)
                q, k, v, gate = self._project_qkv_gate(qkv, positions)
                projected, _ = self.indexer.index_qk_proj(x)
                output = torch.empty_like(q)
                self._run_qsa(projected, positions, q, k, v, output, gate)
                return self.o_proj(output.view(len(x), -1))[0]

            def _run_qsa(self, projected, positions, q, k, v, output, gate):
                events.append('native')
                self.indexer(projected, positions, self.topk_indices_buffer)
                self.impl.do_kv_cache_update(self, k, v, self.kv_cache, main.slot_mapping)
                output.fill_(1)

        def k_only_update(indexer, inputs, positions, out):
            events.append('k-only')
            self.assertEqual(tuple(inputs.q.shape), (1, 2))
            indexer.keys.append(inputs.k.clone())

        def mixed_decode(*args, **kwargs):
            events.append('mixed decode')
            args[7].fill_(2)

        def consume(query, gate, cache, table, rows, heads, group, scale, output, **kw):
            events.append('sparse prefill')
            output[1:].fill_(3)

        layer = Attention()
        layer.is_draft_layer = draft
        layer.indexer = indexer
        layer.impl = NS(do_kv_cache_update=lambda *args: events.append('write KV'))
        class Builder:
            def build(self, common_prefix_len, common, fast_build=False):
                return copy(main)

        class Runner:
            input_batch = NS(req_ids=('d1', 'p1', 'd2', 'p2'),
                             num_prompt_tokens_cpu_tensor=torch.tensor([5, 8, 5, 8]))
            requests = input_batch.req_ids

            def _build_attention_metadata(self, num_tokens, num_reqs, max_query_len, ubatch_slices=None):
                # Both slices have the same geometry. vLLM can copy the first
                # builder result to the second slice, replacing just its KV table.
                common = NS(num_reqs=2, max_query_len=2, query_start_loc_cpu=torch.tensor([0, 1, 3]),
                            seq_lens_cpu_upper_bound=torch.tensor([9, 2]))
                return copy(Builder().build(0, common))

        class ReplayRunner:
            execute_model_state = None
            model_state = NS(_bfla_prompt_lengths={'p': 8, 'p2': 16, 'partial': 32})
            def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
                if scheduler_output == 'fail':
                    raise RuntimeError('replay failed')
                self.execute_model_state = NS(input_batch=NS(has_prefill=True, num_reqs=3,
                    req_ids=['p','p2','partial'], is_prefilling_np=[True]*3,
                    num_computed_prefill_tokens_np=[7,15,4], num_scheduled_tokens=[1]*3,
                    num_draft_tokens_per_req=None))
                return 'native'

        qsa = NS(Qwen4ExpQSAAttention=Attention, Qwen4ExpQSAMetadataBuilder=Builder)
        modules = {
            'vllm': NS(envs=NS(VLLM_BATCH_INVARIANT=False)),
            'vllm.compilation.breakable_cudagraph': NS(
                BreakableCUDAGraphCapture=NS(is_active=lambda: False),
                eager_break_during_capture=lambda fn: fn),
            'vllm.models.qwen4_exp.nvidia': NS(qsa=qsa, model=NS()),
            'vllm.models.qwen4_exp.nvidia.ops': NS(qsa_indexer=NS(
                qsa_select_paged_prefill=None, expand_qsa_block_indices=None)),
            'vllm.models.qwen4_exp.nvidia.ops.qsa': NS(qsa_sparse_paged_attention=mixed_decode),
            'vllm.v1.worker.gpu_model_runner': NS(GPUModelRunner=Runner),
            'vllm.v1.worker.gpu.model_runner': NS(GPUModelRunner=ReplayRunner),
            'vllm.v1.worker.gpu.ubatch_utils': NS(UBatchRunner=NS()),
            'vllm.v1.worker.gpu.model_states.mamba_hybrid': NS(MambaHybridModelState=NS()),
            'vllm.distributed': NS(get_tp_group=lambda: group),
            'vllm.forward_context': NS(get_forward_context=lambda: NS(attn_metadata={
                'attn': main, 'side': side})),
            'vllm.model_executor.layers.linear': NS(ReplicatedLinear=ReplicatedLinear,
                                                    UnquantizedLinearMethod=UnquantizedLinearMethod),
            'vllm.model_executor.layers.utils': NS(default_unquantized_gemm=torch.nn.functional.linear),
            'bfla_vllm.qsa.consumer': NS(consume_prefills=consume),
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(os.environ, {'BFLA_K_ONLY_INDEX': str(int(k_only))}))
            stack.enter_context(patch('bfla_vllm.requests.install_v2'))
            stack.enter_context(patch.object(torch.Tensor, 'is_cuda', property(lambda t: True)))
            stack.enter_context(patch.object(torch.cuda, 'get_device_capability', return_value=capability))
            stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=capture))
            stack.enter_context(patch.object(native_index, 'update_and_select', k_only_update))
            install_ubatches = stack.enter_context(patch('bfla_vllm.requests.install_v2_ubatches'))
            adapter.register()
            if check_retirement:
                runner = ReplayRunner()
                pools = {(1,k):object() for k in ('p','p2','partial')}
                stack.enter_context(patch.object(execution, 'POOLS', pools))
                with self.assertRaisesRegex(RuntimeError, 'replay failed'):
                    runner.execute_model('fail')
                self.assertEqual(len(pools), 3)
                self.assertEqual(runner.execute_model(None), 'native')
                self.assertEqual(set(pools), {(1,'partial')})
            install_ubatches.assert_called_once_with(modules['vllm.v1.worker.gpu.ubatch_utils'].UBatchRunner)
            if microbatch:
                pools = {('attn', key): object() for key in Runner.requests}
                stack.enter_context(patch.object(execution, 'POOLS', pools))
                main = Runner()._build_attention_metadata(6, 4, 2, (slice(0, 2), slice(2, 4)))
                self.assertEqual(set(key for _, key in pools), set(Runner.requests))
            if lora:
                # Replacing an already-inspected native projection must invalidate
                # the K-only decision as well.
                self.assertTrue(native_index.supported(indexer))
                indexer.index_qk_proj = WithLoRA(indexer.index_qk_proj)
            output = layer.forward(torch.arange(3), hidden)
        self.assertEqual(len(indexer.keys), 1)
        self.assertFalse(indexer.skip_topk)
        return events, indexer.index_qk_proj.projected_rows, indexer.keys[0].tolist(), output[:, 0].tolist()

    def test_draft_keeps_native_index_projection_and_selection(self):
        for k_only in (False, True):
            with self.subTest(k_only=k_only):
                events, projected_rows, _, output = self.exercise(draft=True, k_only=k_only)
                self.assertEqual(events, ['qkv', 'native', 'write KV'])
                self.assertEqual(projected_rows, [3])
                self.assertEqual(output, [1, 1, 1])

    def test_qsa_native_v2_replay_retires_completed_pools_without_builder(self):
        self.exercise(check_retirement=True)

    def test_lora_updates_index_cache_with_complete_projection(self):
        events, projected_rows, keys, output = self.exercise(lora=True)
        self.assertEqual(projected_rows, [3])
        self.assertEqual(keys, [[2, 3], [4, 6], [6, 9]])
        self.assertEqual(events, ['qkv', 'write KV', 'mixed decode', 'sparse prefill'])
        self.assertEqual(output, [2, 3, 3])

    def test_unsupported_mixed_batches_use_native_once_in_both_hooks(self):
        for options in ({'tp': 4}, {'cache_dtype': torch.uint8}, {'capability': (8, 0)},
                        {'capture': True}, {'pynccl': False}, {'pynccl': None}, {'microbatch': True}):
            for k_only in (True, False):
                with self.subTest(options=options, k_only=k_only):
                    events, projected_rows, keys, output = self.exercise(**options, k_only=k_only)
                    self.assertEqual(events, ['qkv', 'native', 'write KV'])
                    self.assertEqual(projected_rows, [3])
                    self.assertEqual(keys, [[5, 1], [10, 2], [15, 3]])
                    self.assertEqual(output, [1, 1, 1])

    def test_bf16_preserves_k_only_and_mixed_routing_with_available_reducer(self):
        for options in ({'tp': 1, 'pynccl': None}, {'tp': 2, 'pynccl': True}):
            with self.subTest(options=options):
                events, projected_rows, keys, output = self.exercise(**options)
                self.assertEqual(events, ['qkv', 'k-only', 'write KV', 'mixed decode', 'sparse prefill'])
                self.assertEqual(projected_rows, [1])
                self.assertEqual(keys, [[5, 1], [10, 2], [15, 3]])
                self.assertEqual(output, [2, 3, 3])
