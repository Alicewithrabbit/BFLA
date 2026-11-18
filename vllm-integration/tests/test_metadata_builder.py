"""Exercise the adapter mixin against CPU metadata, without a vLLM runtime."""
from types import SimpleNamespace as NS
import unittest
import torch
from bfla_vllm.backend import MetadataMixin


class NativeBuilder:
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return NS(native_result='preserved', max_query_len=common_attn_metadata.max_query_len)


class Builder(MetadataMixin, NativeBuilder):
    def __init__(self, chunking=False):
        self.vllm_config = NS(model_config=NS(hf_text_config=NS(num_attention_heads=24)),
                              scheduler_config=NS(enable_chunked_prefill=chunking))


class HostMetadata(unittest.TestCase):
    def test_microbatch_with_fresh_lengths_keeps_native_plan_without_owners(self):
        from bfla_vllm.requests import install_v1

        class Runner:
            input_batch = NS(req_ids=('first', 'second'), num_prompt_tokens_cpu_tensor=torch.tensor([20, 20]))
            requests = input_batch.req_ids

            def _build_attention_metadata(self, num_tokens, num_reqs, max_query_len, ubatch_slices=None):
                common = NS(max_query_len=20, num_reqs=1, causal=True,
                            query_start_loc_cpu=torch.tensor([0, 20]), seq_lens_cpu_upper_bound=torch.tensor([20]))
                return Builder().build(0, common)

        install_v1(Runner)
        result = Runner()._build_attention_metadata(40, 2, 20, ubatch_slices=(slice(0, 1), slice(1, 2)))
        self.assertEqual(result.bfla_plan.reason, 'microbatch')
        self.assertEqual(result.native_result, 'preserved')

    def test_031_cpu_upper_bound_proves_fresh_without_reading_device_lengths(self):
        common = NS(max_query_len=30, num_reqs=2, causal=True,
                    query_start_loc_cpu=torch.tensor([0, 20, 50]),
                    seq_lens_cpu_upper_bound=torch.tensor([20, 30]))
        builder = Builder()
        result = builder.build(0, common)
        self.assertEqual(result.bfla_plan.lengths, (20, 30))
        self.assertIs(result.bfla_model_config, builder.vllm_config.model_config)
        self.assertEqual(result.native_result, 'preserved')

    def test_missing_request_ownership_keeps_prefix_and_chunking_native(self):
        common = NS(max_query_len=20, num_reqs=1, causal=True,
                    query_start_loc_cpu=torch.tensor([0, 20]), seq_lens_cpu_upper_bound=torch.tensor([40]))
        self.assertEqual(Builder().build(0, common).bfla_plan.reason, 'prefix_or_chunk')
        common.seq_lens_cpu_upper_bound = torch.tensor([20])
        self.assertEqual(Builder(chunking=True).build(0, common).bfla_plan.reason,
                         'chunking_enabled')

    def test_decode_and_missing_cpu_bound_do_not_guess_request_lengths(self):
        common = NS(max_query_len=1)
        self.assertEqual(Builder().build(0, common).bfla_plan.reason, 'decode')
        common = NS(max_query_len=20, num_reqs=1, causal=True,
                    query_start_loc_cpu=torch.tensor([0, 20]))
        self.assertEqual(Builder().build(0, common).bfla_plan.reason,
                         'missing_cpu_bound')


if __name__ == '__main__':
    unittest.main()
