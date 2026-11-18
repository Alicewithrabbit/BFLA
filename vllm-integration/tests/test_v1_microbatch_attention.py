"""Compare V1 double-batch attention with two independent BF16 paged forwards."""
from importlib.util import find_spec
from types import SimpleNamespace as NS
import unittest

import torch


@unittest.skipUnless(find_spec('vllm') and torch.cuda.is_available(), 'CUDA vLLM required')
class MicrobatchAttention(unittest.TestCase):
    def test_unequal_kv_histories_and_query_lengths(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.worker.ubatch_utils import UBatchSlice
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from bfla_vllm.requests import install_v1
        from bfla_vllm.v031 import FlashMetadataBuilder, FlashPrefillImpl

        class Runner:
            _build_attention_metadata = GPUModelRunner._build_attention_metadata
            initialize_metadata_builders = GPUModelRunner.initialize_metadata_builders
            _get_encoder_seq_lens = staticmethod(lambda *args, **kw: (None, None))
            calculate_reorder_batch_threshold = staticmethod(lambda: None)

        class Group:
            layer_names = ('attn',)

            def create_metadata_builders(self, config, device, block_size, num_metadata_builders):
                self.metadata_builders = []
                for _ in range(num_metadata_builders):
                    builder = object.__new__(FlashMetadataBuilder)
                    builder.aot_schedule = True
                    builder.use_full_cuda_graph = False
                    builder.headdim, builder.num_heads_q, builder.num_heads_kv = dim, heads, kv_heads
                    builder.block_size = page
                    builder.cache_config = NS(cache_dtype='auto')
                    builder.kv_cache_dtype = torch.bfloat16
                    builder.aot_sliding_window = (-1, -1)
                    builder.dcp_world_size = 1
                    builder.rswa_window = None
                    builder.vllm_config = config
                    self.metadata_builders.append(builder)

            def get_metadata_builder(self, index=0):
                return self.metadata_builders[index]

        install_v1(Runner)
        heads, kv_heads, dim, page = 24, 4, 256, 16
        worker = Runner()
        worker.device = torch.device('cuda')
        worker.vllm_config = NS(model_config=None, scheduler_config=NS(enable_chunked_prefill=True))
        worker.parallel_config = NS(use_ubatching=True, num_ubatches=2)
        worker.model_config = NS(rswa_window=None)
        worker.cache_config = NS(use_replayssm=False)
        worker.speculative_config = None
        worker.is_mm_prefix_lm = False
        worker.dcp_world_size = 1
        worker.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec='global')])
        worker.attn_groups = [[Group()]]
        worker.initialize_metadata_builders(worker.kv_cache_config, [16])
        self.assertTrue(worker.attn_groups[0][0].get_metadata_builder(0).supports_update_block_table)

        def build(query_lens, visible, pages, slots, slices=None):
            offsets = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0, dtype=torch.int32)
            worker.query_start_loc = NS(cpu=offsets, gpu=offsets.cuda())
            worker.optimistic_seq_lens_cpu = torch.tensor(visible, dtype=torch.int32)
            worker.seq_lens = worker.optimistic_seq_lens_cpu.cuda()
            worker.positions = slots
            worker.requests = tuple(range(len(query_lens)))
            worker.input_batch = NS(req_ids=worker.requests,
                num_prompt_tokens_cpu_tensor=worker.optimistic_seq_lens_cpu,
                num_computed_tokens_cpu_tensor=worker.optimistic_seq_lens_cpu - torch.tensor(query_lens),
                block_table=[NS(get_device_tensor=lambda n: pages[:n])])
            return worker._build_attention_metadata(sum(query_lens), len(query_lens), max(query_lens),
                ubatch_slices=slices, slot_mappings={0: slots})[0]

        device = worker.device
        gen = torch.Generator(device=device).manual_seed(1931)
        def inputs(*shape):
            return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)

        impl = FlashPrefillImpl(heads, dim, dim ** -0.5, kv_heads, None, None, 'auto')
        scale = torch.ones(1, device=device)
        layer = NS(_q_scale=scale, _k_scale=scale, _v_scale=scale)
        pages = torch.arange(1, 25, dtype=torch.int32, device=device).view(2, 12)
        cache = inputs(25, kv_heads, page, 2 * dim)
        for query_lens, visible in (((8, 8), (64, 192)), ((5, 13), (192, 80))):
            with self.subTest(query_lens=query_lens, visible=visible):
                query = inputs(sum(query_lens), heads, dim)
                offsets = (0, query_lens[0], sum(query_lens))
                slices = [UBatchSlice(slice(i, i+1), slice(offsets[i], offsets[i+1])) for i in range(2)]
                slots = []
                for i, (count, length) in enumerate(zip(query_lens, visible)):
                    positions = torch.arange(length-count, length, device=device)
                    slots.append(pages[i, positions // page].long() * page + positions % page)
                slots = torch.cat(slots)
                batches = build(query_lens, visible, pages, slots, slices)
                for i, sliced in enumerate(batches):
                    meta = sliced['attn']
                    self.assertEqual(meta.bfla_plan.reason, 'microbatch')
                    span = slices[i].token_slice
                    q = query[span]
                    ref = build([query_lens[i]], [visible[i]], pages[i:i+1], slots[span])['attn']
                    # The paged backend reads K/V from the same already-filled owner cache.
                    actual = impl.forward(layer, q, None, None, cache, meta, torch.full_like(q, float('nan')))
                    expected = FlashAttentionImpl.forward(impl, layer, q, None, None, cache, ref, torch.empty_like(q))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(meta.seq_lens.tolist(), [visible[i]])
                    self.assertEqual(meta.query_start_loc.tolist(), [0, query_lens[i]])
                    torch.testing.assert_close(meta.slot_mapping, ref.slot_mapping)
                    torch.testing.assert_close(meta.block_table, ref.block_table)

        # A single request can straddle two microbatches. The first slice must
        # stop at its own causal boundary; the second resumes from that point.
        q = inputs(18, heads, dim)
        positions = torch.arange(62, 80, device=device)
        slots = pages[0, positions // page].long() * page + positions % page
        spans = (slice(0, 5), slice(5, 18))
        batches = build([18], [80], pages[:1], slots,
                        [UBatchSlice(slice(0, 1), span) for span in spans])
        for span, length, batch in zip(spans, (67, 80), batches):
            meta = batch['attn']
            ref = build([span.stop - span.start], [length], pages[:1], slots[span])['attn']
            actual = impl.forward(layer, q[span], None, None, cache, meta, torch.empty_like(q[span]))
            expected = FlashAttentionImpl.forward(impl, layer, q[span], None, None, cache, ref, torch.empty_like(q[span]))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(meta.seq_lens.tolist(), [length])
