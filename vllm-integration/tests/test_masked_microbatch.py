"""V1 metadata slicing and native masked attention on a paged BF16 cache."""
from importlib.util import find_spec
from types import SimpleNamespace as NS
import unittest

import torch


class SlicePolicy(unittest.TestCase):
    def test_phase_follows_request_rows_including_a_straddling_request(self):
        from bfla_vllm.v1_masks import _masked_split
        cuts = [NS(request_slice=slice(0, 2), token_slice=slice(0, 256)),
                NS(request_slice=slice(1, 3), token_slice=slice(256, 513))]
        # The deployed native splitter's fresh metadata defaults is_prefilling to None.
        def native(slices, common):
            return [NS(is_prefilling=None) for _ in slices]
        split = _masked_split(native)
        for flags in (torch.tensor([True, True, False]), None):
            for ranges in (None, {0: [(0, 0)]}):
                cm = NS(causal=True, mm_req_doc_ranges=ranges, rswa_prefix_lens=None,
                        is_prefilling=flags)
                parts = split(cuts, cm)
                for cut, part in zip(cuts, parts):
                    if flags is None:
                        self.assertIsNone(part.is_prefilling)
                    else:
                        self.assertIsNotNone(part.is_prefilling)
                        self.assertEqual(part.is_prefilling.tolist(), flags[cut.request_slice].tolist())

    def test_vote_uses_the_ranges_kept_by_native_attention(self):
        from bfla_vllm.v1_masks import _may_split_image
        def veto(ranges, *, offset=0, length=10, window=None, clamp=False, modulus=0, cap=20):
            position = NS(offset=offset, length=length, extract_embeds_range=lambda: ranges)
            config = NS(sliding_window=window, mm_prefix_span_leading_pad_modulus=modulus)
            runner = NS(is_mm_prefix_lm=True, parallel_config=NS(use_ubatching=True, num_ubatches=2),
                model=NS(mm_prefix_clamp_sliding_window=clamp), model_config=NS(hf_text_config=config),
                max_num_tokens=cap, _pad_for_sequence_parallelism=lambda n: n,
                compilation_config=NS(max_cudagraph_capture_size=None),
                input_batch=NS(req_ids=['r'], num_computed_tokens_cpu=[0]),
                requests={'r': NS(mm_features=[NS(modality='image', mm_position=position)])})
            return _may_split_image(runner, [10])
        self.assertFalse(veto([]))
        self.assertFalse(veto([(0, 3), (7, 7)]))
        self.assertTrue(veto([(3, 6)]))
        self.assertFalse(veto([(2, 9)], window=4))
        self.assertTrue(veto([(2, 9)], window=4, clamp=True))
        # For this model the inclusive sentinel range starts at 7, not at offset 0.
        self.assertFalse(veto([], length=8, modulus=8))
        self.assertTrue(veto([], length=10, modulus=8))
        # At capacity both ranks must cut at 5. Spare capacity permits a peer
        # to move that cut inside the same late image span.
        self.assertFalse(veto([(7, 9)], cap=10))
        self.assertTrue(veto([(7, 9)], cap=20))
        self.assertFalse(veto([(5, 9)], cap=10))
        self.assertTrue(veto([(4, 5)], cap=10))


@unittest.skipUnless(find_spec("vllm") and torch.cuda.is_available(), "CUDA vLLM required")
class MaskedMicrobatch(unittest.TestCase):
    def setUp(self):
        from vllm.v1.worker import gpu_model_runner as v1
        from bfla_vllm.requests import install_v1

        class Runner:
            _allow_microbatching = v1.GPUModelRunner._allow_microbatching
            _build_attention_metadata = v1.GPUModelRunner._build_attention_metadata

        install_v1(Runner)
        self.runner = Runner()
        self.runner.parallel_config = NS(use_ubatching=True, num_ubatches=2)
        self.runner.is_mm_prefix_lm = True
        self.runner.cache_config = NS(mamba_cache_mode='none')
        self.runner.max_num_tokens = 12
        self.runner._pad_for_sequence_parallelism = lambda n: n
        self.runner.compilation_config = NS(max_cudagraph_capture_size=None)
        self.runner.model = NS()
        self.runner.model_config = NS(hf_text_config=NS())
        self.split = v1.split_attn_metadata

    def common(self, lengths, pages, ranges):
        from vllm.v1.attention.backend import CommonAttentionMetadata
        n = torch.tensor(lengths, dtype=torch.int32)
        starts = torch.tensor([0, *lengths]).cumsum(0, dtype=torch.int32)
        return CommonAttentionMetadata(query_start_loc=starts.cuda(), query_start_loc_cpu=starts,
            seq_lens=n.cuda(), seq_lens_cpu_upper_bound=n, num_reqs=len(n),
            num_actual_tokens=sum(lengths), max_query_len=max(lengths), max_seq_len=max(lengths),
            block_table_tensor=torch.tensor(pages, dtype=torch.int32, device="cuda").view(-1, 1),
            slot_mapping=torch.arange(sum(lengths), device="cuda"), mm_req_doc_ranges=ranges)

    def test_safe_row_slices_preserve_special_mask_fields(self):
        from vllm.v1.worker.ubatch_utils import UBatchSlice
        cm = self.common([4, 8], [1, 2], {1: [(1, 6)]})
        cm.causal = torch.tensor([True, False], device="cuda")
        cm.rswa_prefix_lens = torch.tensor([4, 7])
        cuts = [UBatchSlice(slice(0, 1), slice(0, 4)), UBatchSlice(slice(1, 2), slice(4, 12))]
        first, second = self.split(cuts, cm)
        self.assertEqual(second.mm_req_doc_ranges, {0: [(1, 6)]})
        self.assertFalse(first.mm_req_doc_ranges)
        self.assertEqual(second.causal.tolist(), [False])
        self.assertEqual(second.rswa_prefix_lens.tolist(), [7])
        self.assertEqual(cm.mm_req_doc_ranges, {1: [(1, 6)]})
        cm.causal = True
        crossing = [UBatchSlice(slice(0, 2), slice(0, 8)), UBatchSlice(slice(1, 2), slice(8, 12))]
        with self.assertRaisesRegex(ValueError, 'bidirectional range'):
            self.split(crossing, cm)

    def test_masked_outputs_match_unsliced_native_request(self):
        import numpy as np
        from vllm.v1.worker.ubatch_utils import UBatchSlice, maybe_create_ubatch_slices
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl, TritonAttentionMetadataBuilder
        from bfla_vllm.v031 import TritonMetadataBuilder, TritonPrefillImpl

        b = object.__new__(TritonAttentionMetadataBuilder)
        b.seq_threshold_3D, b.num_par_softmax_segments, b.rswa_window = 0, 16, None
        b.softmax_segm_output = b.softmax_segm_max = b.softmax_segm_expsum = None
        impl = TritonAttentionImpl(4, 128, 128 ** -0.5, 1, None, None, 'auto')
        adapter = TritonPrefillImpl(4, 128, 128 ** -0.5, 1, None, None, 'auto')
        adapted_builder = object.__new__(TritonMetadataBuilder)
        adapted_builder.__dict__.update(b.__dict__)
        adapted_builder.vllm_config = NS(model_config=None)
        scale = torch.ones(1, device='cuda')
        layer = NS(_q_scale=scale, _k_scale=scale, _v_scale=scale)
        cache = torch.zeros((3, 1, 16, 256), dtype=torch.bfloat16, device='cuda')
        cache[2, :, 4:7, 128:] = 8
        q = torch.zeros((12, 4, 128), dtype=torch.bfloat16, device='cuda')
        def run(common, queries, adapted=False):
            meta = (adapted_builder if adapted else b).build(0, common)
            if adapted:
                self.assertEqual(meta.bfla_plan.reason, 'unsupported_attention_contract')
            return (adapter if adapted else impl).forward(
                layer, queries, None, None, cache, meta, torch.empty_like(queries))

        cm = self.common([4, 8], [1, 2], {1: [(1, 6)]})
        expected = run(self.common([8], [2], {0: [(1, 6)]}), q[4:])
        plain = run(self.common([8], [2], None), q[4:])
        self.assertFalse(torch.equal(plain[1], expected[1]))
        # Safe request-aligned slice: request 1 becomes local request 0.
        cuts = [UBatchSlice(slice(0, 1), slice(0, 4)), UBatchSlice(slice(1, 2), slice(4, 12))]
        torch.testing.assert_close(run(self.split(cuts, cm)[1], q[4:], True), expected, rtol=0, atol=0)

        # Native equal-token splitting would cut the second request inside its image.
        lens = np.array([4, 8])
        runner = self.runner
        runner.reorder_batch_threshold = 1
        runner.input_batch = NS(req_ids=['text', 'image'], num_computed_tokens_cpu=np.array([0, 0]))
        feat = NS(modality='image', mm_position=NS(offset=1, length=6, extract_embeds_range=lambda: [(1, 6)]))
        runner.requests = {'text': NS(mm_features=[]), 'image': NS(mm_features=[feat])}
        allowed = runner._allow_microbatching(2, lens)
        cuts, _ = maybe_create_ubatch_slices(allowed, lens, 12, 2, 2)
        if cuts:
            actual = torch.cat([run(m, q[s.token_slice]) for m, s in zip(self.split(cuts, cm), cuts)])
        else:
            actual = run(cm, q, True)
        torch.testing.assert_close(actual[4:], expected, rtol=0, atol=0)
        self.assertFalse(allowed)
        runner.requests['image'].mm_features.clear()
        self.assertTrue(runner._allow_microbatching(2, lens))
        # DP padding can only move the first cut to the right of token 6.
        runner.requests['text'].mm_features = [NS(modality='image', mm_position=NS(offset=0, length=4, extract_embeds_range=lambda: [(0, 3)]))]
        self.assertTrue(runner._allow_microbatching(2, lens))
        runner.input_batch.req_ids = ['image']
        runner.requests['image'].mm_features.append(feat)
        self.assertFalse(runner._allow_microbatching(1, np.array([8])))
