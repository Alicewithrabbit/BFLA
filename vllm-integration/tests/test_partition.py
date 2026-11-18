"""Mixed P/D must match independent native-D and sparse-P execution."""
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import torch
from bfla_vllm.protocol import classify_prefill


class Partition(unittest.TestCase):
    def test_tp_without_feature_reducer_preserves_whole_native_mixed_batch(self):
        from bfla_vllm.backend import PrefillMixin

        class NativeAttention:
            def forward(self, layer, query, key, value, cache, meta, *args):
                return query, meta

        class Attention(PrefillMixin, NativeAttention):
            attn_type = 'decoder'
            sliding_window = (-1, -1)
            kv_cache_dtype = 'auto'
            num_heads, num_kv_heads, head_size = 12, 1, 256

        plan = classify_prefill((0, 1, 3), (100, 2), True, request_keys=('d', 'p'),
                                prompt_lengths=(80, 2), is_prefilling=(False, True))
        self.assertEqual(plan.reason, 'mixed_prefill')
        meta = NS(max_query_len=2, bfla_plan=plan, bfla_model_config=None, bfla_native_metadata=NS())
        layer = NS(total_num_heads=24, total_num_kv_heads=2)
        query = torch.empty((3, 12, 256), dtype=torch.bfloat16)
        for comm in (None, NS(disabled=True)):
            group = NS(world_size=2, rank_in_group=0, device_communicator=NS(pynccl_comm=comm))
            with self.subTest(comm=comm), patch.dict(sys.modules, {'vllm.distributed': NS(get_tp_group=lambda: group)}), \
                    patch.object(torch.Tensor, 'is_cuda', property(lambda t: True)), \
                    patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
                    patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                    patch('bfla_vllm.execution.forward', return_value=(None, None)) as sparse:
                native_query, native_meta = Attention().forward(layer, query, None, None, None, meta)
                sparse.assert_not_called()
                self.assertIs(native_query, query)
                self.assertIs(native_meta, meta)

    def test_rswa_keeps_both_prefixes_and_all_query_tokens_on_native_attention(self):
        from bfla_vllm.backend import MetadataMixin, PrefillMixin
        from bfla_vllm.requests import RequestContext, current_requests

        class NativeBuilder:
            def build(self, common_prefix_len, common, fast_build=False):
                return NS(max_query_len=3, num_actual_tokens=4,
                          query_start_loc=common.query_start_loc_cpu, seq_lens=torch.tensor([92, 9]),
                          block_table=torch.tensor([[1], [0]]), slot_mapping=torch.arange(4),
                          rswa_window=self.window,
                          rswa_prefix_lens=common.rswa_prefix_lens if self.window else None)

        class Builder(MetadataMixin, NativeBuilder):
            vllm_config = NS(model_config=None, scheduler_config=NS(enable_chunked_prefill=True))

        class NativeAttention:
            def forward(self, layer, query, key, value, kv_cache, meta, *args):
                return query, meta

        class Attention(PrefillMixin, NativeAttention):
            attn_type = 'decoder'
            sliding_window = (-1, -1)
            kv_cache_dtype = 'auto'

        common = NS(max_query_len=3, num_reqs=2, causal=True, query_start_loc_cpu=torch.tensor([0, 3, 4]),
                    seq_lens_cpu_upper_bound=torch.tensor([92, 9]), is_prefilling=torch.tensor([True, False]),
                    rswa_prefix_lens=torch.tensor([92, 6]))
        token = current_requests.set(RequestContext(('prefill', 'decode'), (92, 6)))
        self.addCleanup(current_requests.reset, token)
        builder = Builder()
        builder.window = 4
        masked = builder.build(0, common)
        self.assertFalse(hasattr(masked, 'bfla_native_metadata'))
        self.assertFalse(masked.bfla_plan.rows)
        query = torch.empty((4, 1, 8), dtype=torch.bfloat16)
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            native_query, native_meta = Attention().forward(None, query, None, None, None, masked)
        self.assertIs(native_query, query)
        self.assertIs(native_meta, masked)
        self.assertEqual(native_meta.rswa_prefix_lens.tolist(), [92, 6])

        # The hybrid runner can provide prompt lengths even when the native
        # backend has no active R-SWA window.
        builder.window = None
        plain = builder.build(0, common)
        self.assertEqual(plain.bfla_plan.rows, ((0, 0, 3, 92),))
        self.assertEqual(plain.bfla_plan.reason, 'mixed_prefill')
        self.assertEqual(plain.bfla_native_metadata.seq_lens.tolist(), [9])

    def test_mixed_selects_only_p_ignoring_optimistic_decode_bound(self):
        p = classify_prefill((0,1,258,259,324),(9000,4353,8192,8257),True,
            prompt_lengths=(8000,8192,8192,16384),request_keys=('d','p0','p1','p2'),
            is_prefilling=(False,True,True,True),chunking_enabled=True)
        self.assertEqual(p.reason,'mixed_prefill')
        self.assertEqual(p.rows,((1,1,258,4353),(2,258,259,8192),(3,259,324,8257)))
        self.assertEqual(p.native_rows,((0,0,1,9000),))
        self.assertEqual(p.lengths,(257,1,65))
        self.assertEqual(p.request_keys,('p0','p1','p2'))
        self.assertEqual(p.prompt_lengths,(8192,8192,16384))

    def test_shared_row_geometry_matches_existing_qsa_and_explicit_p_flags(self):
        from bfla_vllm.partition import prefill_rows
        from bfla_vllm.qsa.protocol import prefill_rows as qsa_rows
        starts=(0,1,98,99,164); lens=(9000,4097,8192,8257)
        self.assertEqual(qsa_rows(starts,lens),prefill_rows(starts,lens))
        self.assertEqual(prefill_rows(starts,lens,(False,True,True,True)),
                         ((1,1,98,4097),(2,98,99,8192),(3,99,164,8257)))

    def test_interleaved_rows_pack_and_scatter_without_touching_other_outputs(self):
        from bfla_vllm.partition import take_rows, write_rows
        x=torch.arange(20).reshape(10,2)
        rows=((1,1,4,30),(3,7,9,40))
        packed=take_rows(x,rows)
        self.assertTrue(torch.equal(packed,torch.cat((x[1:4],x[7:9]))))
        out=torch.full_like(x,-1);write_rows(out,rows,packed)
        self.assertTrue(torch.equal(out[1:4],x[1:4]))
        self.assertTrue(torch.equal(out[7:9],x[7:9]))
        self.assertTrue(torch.all(out[[0,4,5,6,9]]==-1))
        one=take_rows(x,((1,2,7,30),))
        self.assertEqual(one.data_ptr(),x[2:].data_ptr())

    def test_decode_subset_uses_device_lengths_and_discards_full_batch_schedule(self):
        from bfla_vllm.partition import native_metadata
        meta=NS(num_actual_tokens=259,max_query_len=257,max_seq_len=9000,
            query_start_loc=torch.tensor([0,1,258,259]),seq_lens=torch.tensor([8997,4353,8191]),
            block_table=torch.tensor([[7,2],[1,4],[9,6]]),slot_mapping=torch.arange(259),
            use_cascade=True,common_prefix_len=64,cu_prefix_query_lens=torch.tensor([0,259]),
            prefix_kv_lens=torch.tensor([64]),suffix_kv_lens=torch.tensor([8933,4289,8127]),
            scheduler_metadata=torch.tensor([11]),prefix_scheduler_metadata=torch.tensor([12]),
            max_num_splits=0,softmax_segm_output=torch.empty(3,8,16,512))
        child=native_metadata(meta,((0,0,1,9000),(2,258,259,8192)))
        self.assertEqual(child.seq_lens.tolist(),[8997,8191])
        self.assertEqual(child.block_table.tolist(),[[7,2],[9,6]])
        self.assertEqual(child.query_start_loc.tolist(),[0,1,2])
        self.assertEqual(child.slot_mapping.tolist(),[0,258])
        self.assertEqual((child.num_actual_tokens,child.max_query_len),(2,1))
        self.assertFalse(child.use_cascade)
        self.assertIsNone(child.scheduler_metadata)
        self.assertIs(child.softmax_segm_output,meta.softmax_segm_output)
        self.assertTrue(meta.use_cascade)
        self.assertEqual(meta.query_start_loc.tolist(),[0,1,258,259])

    def test_multitoken_decode_does_not_guess_speculative_token_layout(self):
        p=classify_prefill((0,3,260),(9000,4353),True,
            prompt_lengths=(8000,8192),request_keys=('d','p'),
            is_prefilling=(False,True),chunking_enabled=True)
        self.assertEqual(p.reason,'mixed_spec_decode')


if __name__=='__main__':unittest.main()
