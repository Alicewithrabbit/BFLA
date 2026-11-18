"""Exercise request lifetime and decode selection with CPU protocol objects."""
from types import ModuleType, SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch


class NativeStateOwnership(unittest.TestCase):
    def test_v2_context_is_scoped_and_release_follows_request_removal(self):
        from bfla_vllm.qsa.protocol import owned_prefills
        from bfla_vllm.requests import install_v2, current_requests
        from bfla_vllm import execution
        removed = []
        class State:
            def prepare_attn(self, batch, *args, **kwargs):
                return current_requests.get()
            def add_request(self, index, data):
                return index
            def remove_request(self, request_id):
                removed.append(request_id)
                return 'removed'
        pools = {('layer','a'): object(), ('layer','b'): object()}
        install_v2(State)
        batch = NS(has_prefill=True, num_reqs=2, req_ids=['a', 'b'], prefill_len_np=[520, 65540])
        state = State()
        state.add_request(0,NS(req_id='a',prompt_len=512))
        state.add_request(1,NS(req_id='b',prompt_len=65536))
        context = state.prepare_attn(batch, None, None, None, None, None)
        rows, prompts, keys = owned_prefills((0, 2, 4), (512, 65536), context)
        self.assertEqual((prompts, keys), ((512, 65536), ('a', 'b')))
        self.assertEqual(len(rows), 2)
        self.assertIsNone(current_requests.get())
        self.assertIsNone(state.prepare_attn(batch, None, None, None, None, None, True))
        with patch.object(execution,'POOLS',pools), \
                patch.object(execution,'release_request',wraps=execution.release_request) as release:
            self.assertEqual(state.remove_request('a'), 'removed')
            release.assert_called_once_with('a')
        self.assertEqual(list(pools), [('layer','b')])
        self.assertNotIn('a',state._bfla_prompt_lengths)
        self.assertEqual(removed, ['a'])

    def test_verify_rows_keep_native_projection_and_attention(self):
        from bfla_vllm.qsa import native_index
        class Attention:
            layer_name = 'attention'
            indexer = NS(raw_key_cache=NS(prefix='side'))
            def forward(self, positions, hidden_states):
                return hidden_states
        native_index.install(Attention)
        # Both query intervals have multiple tokens, but the fork owns the
        # first interval as target verification.
        metadata = dict(attention=NS(max_query_len=97, bfla_partition=(0, 101)),
                        side=NS(num_decode_tokens=4, num_prefill_tokens=97))
        context = ModuleType('vllm.forward_context')
        context.get_forward_context = lambda: NS(attn_metadata=metadata)
        hidden = object()
        with patch.dict(sys.modules, {'vllm.forward_context': context}), \
                patch.object(native_index, 'project') as project:
            self.assertIs(Attention().forward(None, hidden), hidden)
            project.assert_not_called()

    def test_decode_scope_leaves_prefill_output_untouched_and_restores_native(self):
        from bfla_vllm.qsa.decode_selection import install, decode_selection_only
        events = []
        def prefill():
            events.append('prefill')
        def expand(blocks, positions, visible, ratio, budget, out):
            for i, (block, pos) in enumerate(zip(blocks, positions)):
                out[i][0] = block * ratio + pos
        ops = NS(qsa_select_paged_prefill=prefill, expand_qsa_block_indices=expand)
        install(ops)
        out = [[-1], [-1], [-1]]
        with decode_selection_only(1):
            ops.qsa_select_paged_prefill()
            ops.expand_qsa_block_indices([10, 20, 30], [1, 2, 3], [9]*3, 4, 2048, out)
        self.assertEqual(out, [[41], [-1], [-1]])
        self.assertEqual(events, [])
        ops.qsa_select_paged_prefill()
        ops.expand_qsa_block_indices([10, 20, 30], [1, 2, 3], [9]*3, 4, 2048, out)
        self.assertEqual(out, [[41], [82], [123]])
        self.assertEqual(events, ['prefill'])

if __name__ == '__main__':
    unittest.main()
