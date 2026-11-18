"""Run the vLLM 0.31 QSA forward and breakpoint lifecycle on CPU tensors."""
import importlib.util
import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


@unittest.skipUnless(importlib.util.find_spec('vllm'), 'requires the vLLM 0.31 QSA fork')
class PiecewiseReplay(unittest.TestCase):
    def exercise(self, *, k_only, capture_owned, early_registration=False):
        try:
            from vllm.models.qwen4_exp.nvidia import model, qsa
        except ModuleNotFoundError as err:
            if err.name and 'vllm.models.qwen4_exp.nvidia'.startswith(err.name):
                self.skipTest('requires the vLLM 0.31 QSA fork')
            raise
        from vllm.compilation import breakable_cudagraph as cg
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import override_forward_context
        from bfla_vllm.qsa import adapter, consumer, native_index
        from bfla_vllm.qsa.protocol import uses_sparse_prefill

        num_tokens = 8
        hidden = torch.arange(2 * num_tokens, dtype=torch.float32).view(num_tokens, 2)
        positions = torch.arange(num_tokens)
        routes, projections = [], []
        capture = cg.BreakableCUDAGraphCapture()

        class Projection:
            weight = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [0., 3.]])

            def __call__(self, x):
                out = x.new_empty((len(x), len(self.weight)))

                def project():
                    projections.append(('full', len(x)))
                    torch.mm(x, self.weight.T, out=out)

                project()
                # Stand in for the recorded CUDA GEMM, with real tensor storage.
                if capture._capturing:
                    capture.segments.append(project)
                return out, None

        class Indexer:
            raw_key_cache = NS(prefix='side')
            index_qk_proj = Projection()
            index_n_heads, index_head_dim, packed_output_width = 1, 2, 4
            skip_topk = False

            def __call__(self, projected, pos, out):
                self.queries = projected[:, :2].clone()
                self.keys = projected[:, 2:].clone()
                return out

        def update(indexer, inputs, pos, out):
            projections.append(('k-only', len(inputs.k)))
            self.assertEqual(len(inputs.q), 0)
            indexer.keys = inputs.k.clone()

        def native_attention(layer, query, key, value, cache, meta, output, **kwargs):
            routes.append('native')
            output.copy_(query + layer.indexer.queries.view_as(query) + layer.indexer.keys.view_as(query))

        def sparse_attention(query, gate, cache, table, rows, heads, group, scale,
                             output, *, request_keys, **kwargs):
            routes.append(request_keys[0])
            output.copy_(query + layer.indexer.keys.view_as(query) + table[0, 0])

        def contract(layer, query, main, side):
            if uses_sparse_prefill(main, side) and not capture._capturing:
                return main, side, None, None, None

        def context(kind, marker):
            decode = kind == 'decode'
            main = NS(max_query_len=1 if decode else num_tokens, num_actual_tokens=num_tokens,
                      bfla_partition=(0, num_tokens) if kind == 'prefill' else None,
                      bfla_rows=((0, 0, num_tokens, 16384 + num_tokens),),
                      bfla_prompts=(65536,), bfla_keys=(str(marker),),
                      block_table=torch.tensor([[marker]]), slot_mapping=positions)
            side = NS(num_decode_tokens=num_tokens if decode else 0,
                      num_prefill_tokens=0 if decode else num_tokens,
                      num_actual_tokens=num_tokens, token_to_req=positions)
            return override_forward_context(NS(attn_metadata={'attn': main, 'side': side},
                                               cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE))

        with ExitStack() as stack:
            stack.enter_context(patch.object(cg.envs, 'VLLM_USE_BREAKABLE_CUDAGRAPH', True))
            native = qsa.Qwen4ExpQSAAttention
            native_run = getattr(native._run_qsa, '__wrapped__', native._run_qsa)

            class Attention:
                # Use the installed fork's complete projection and native D path.
                forward = native.forward
                _run_qsa = native_run if early_registration else cg.eager_break_during_capture(native_run)
                num_heads = num_kv_heads = 1
                head_dim = 2
                layer_name = 'attn'
                scaling = 1.
                _max_decode_query_len = 1
                indexer = Indexer()
                kv_cache = hidden.new_empty((1, 1, num_tokens, 4))
                topk_indices_buffer = positions[:, None].expand(-1, 4).clone()
                impl = NS(do_kv_cache_update=lambda *args: None, forward_qsa=native_attention)
                qkv_proj = o_proj = staticmethod(lambda x: (x, None))
                _project_qkv_gate = staticmethod(lambda x, pos: (x, x, x, x))

            class Builder:
                def build(self, *args, **kwargs):
                    raise AssertionError('metadata supplied by the runner')

            stack.enter_context(patch.object(qsa, 'Qwen4ExpQSAAttention', Attention))
            stack.enter_context(patch.object(qsa, 'Qwen4ExpQSAMetadataBuilder', Builder))
            for hook in ('install_v1', 'install_v2', 'install_v2_retirement', 'install_v2_ubatches'):
                stack.enter_context(patch('bfla_vllm.requests.' + hook))
            stack.enter_context(patch('bfla_vllm.qsa.decode_selection.install'))
            stack.enter_context(patch.dict(os.environ, {'BFLA_K_ONLY_INDEX': str(int(k_only))}))
            stack.enter_context(patch.object(adapter, 'prefill_context', contract))
            stack.enter_context(patch.object(native_index, 'prefill_context', contract))
            stack.enter_context(patch.object(native_index, 'supported', return_value=True))
            stack.enter_context(patch.object(native_index, 'update_and_select', update))
            stack.enter_context(patch.object(consumer, 'consume_prefills', sparse_attention))
            # Only the CUDA graph and allocator mechanics are simulated on CPU.
            stack.enter_context(patch.object(cg, '_weak_ref_capture_arg', lambda arg: arg))
            stack.enter_context(patch.object(capture, '_begin_segment',
                                             lambda: setattr(capture, '_capturing', True)))
            stack.enter_context(patch.object(capture, '_end_segment',
                                             lambda: setattr(capture, '_capturing', False)))
            stack.enter_context(patch.object(torch.Tensor, 'is_cuda', property(lambda tensor: True)))
            stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing',
                                             lambda: capture._capturing))
            with patch.object(cg.envs, 'VLLM_USE_BREAKABLE_CUDAGRAPH', not early_registration):
                adapter.register()
            layer = Attention()
            with context('prefill' if capture_owned else 'dummy', 1), capture:
                graph_output = layer.forward(positions, hidden)
            self.assertEqual(capture.num_eager_breaks, 1)
            self.assertEqual(projections, [('full', num_tokens)])

            for marker, kind in ((2, 'prefill'), (3, 'decode'), (4, 'prefill')):
                hidden.add_(10)
                routes.clear()
                projections.clear()
                with context(kind, marker):
                    capture.replay()
                    self.assertEqual(routes, ['native' if kind == 'decode' else str(marker)])
                    self.assertEqual(projections, [('full', num_tokens)])
                    torch.testing.assert_close(layer.indexer.keys, hidden * hidden.new_tensor([2, 3]))
                    if kind == 'decode':
                        torch.testing.assert_close(layer.indexer.queries, hidden, rtol=0, atol=0)
                    expected = layer.forward(positions, hidden)
                    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
                    self.assertEqual(projections[-1],
                                     ('k-only' if k_only and kind == 'prefill' else 'full', num_tokens))
                    self.assertEqual(routes[0], routes[1])

    def test_dummy_capture_rechecks_each_live_request(self):
        for k_only in (False, True):
            with self.subTest(k_only=k_only):
                self.exercise(k_only=k_only, capture_owned=False)

    def test_owned_capture_keeps_a_replayable_complete_projection(self):
        self.exercise(k_only=True, capture_owned=True)

    def test_default_graph_mode_can_be_selected_after_plugin_registration(self):
        self.exercise(k_only=True, capture_owned=False, early_registration=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
