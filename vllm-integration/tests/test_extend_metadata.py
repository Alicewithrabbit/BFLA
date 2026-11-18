"""Exact prefill metadata, distinct from optimistic decode lengths."""
import unittest
from types import SimpleNamespace as NS
import torch
from bfla_vllm.protocol import classify_prefill


class ExtendMetadata(unittest.TestCase):
    def test_chunk_and_prefix_rows_keep_original_prompt_and_owners(self):
        plan = classify_prefill((0, 256, 320), (4352, 8256), True,
            chunking_enabled=True, prompt_lengths=(8192, 16384),
            request_keys=('a', 'b'), is_prefilling=(True, True))
        self.assertEqual(plan.reason, 'extend_prefill')
        self.assertEqual(plan.lengths, (256, 64))
        self.assertEqual(plan.kv_lengths, (4352, 8256))
        self.assertEqual(plan.prompt_lengths, (8192, 16384))
        self.assertEqual(plan.request_keys, ('a', 'b'))

    def test_last_single_token_of_cached_prompt_remains_prefill(self):
        p = classify_prefill((0, 1), (8192,), True, prompt_lengths=(8192,),
                            request_keys=('a',), is_prefilling=(True,))
        self.assertEqual(p.reason, 'extend_prefill')

    def test_mixed_decode_bound_is_not_mistaken_for_cached_prefill(self):
        p = classify_prefill((0, 1, 257), (9000, 4352), True,
            prompt_lengths=(8192, 8192), request_keys=('d', 'p'),
            is_prefilling=(False, True), chunking_enabled=True)
        self.assertEqual(p.reason, 'mixed_prefill')
        self.assertEqual(p.kv_lengths,(4352,))
        self.assertEqual(p.native_rows,((0,0,1,9000),))

    def test_missing_prompt_owner_does_not_guess_chunk_end(self):
        p = classify_prefill((0, 256), (4352,), True, chunking_enabled=True)
        self.assertEqual(p.reason, 'chunking_enabled')

    def test_preemption_recompute_crossing_prompt_keeps_native_semantics(self):
        p = classify_prefill((0,300), (8300,), True,
            prompt_lengths=(8192,), request_keys=('resumed',), is_prefilling=(True,),
            chunking_enabled=True)
        self.assertEqual(p.reason, 'recompute_beyond_prompt')

    def test_metadata_builder_uses_runner_context_not_layer_gpu_readback(self):
        from bfla_vllm.requests import RequestContext, current_requests
        from bfla_vllm.backend import MetadataMixin
        class Native:
            def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                return NS(native=True)
        class Builder(MetadataMixin, Native):
            vllm_config = NS(model_config=None, scheduler_config=NS(enable_chunked_prefill=True))
        common = NS(max_query_len=256, num_reqs=2, causal=True,
                    query_start_loc_cpu=torch.tensor([0,256,320]),
                    seq_lens_cpu_upper_bound=torch.tensor([4352,8256]),
                    is_prefilling=torch.tensor([True,True]))
        token = current_requests.set(RequestContext(('a','b'), (8192,16384)))
        try: result = Builder().build(0,common)
        finally: current_requests.reset(token)
        self.assertEqual(result.bfla_plan.reason, 'extend_prefill')
        self.assertEqual(result.bfla_plan.prompt_lengths, (8192,16384))
        self.assertTrue(result.native)


if __name__ == '__main__': unittest.main()
