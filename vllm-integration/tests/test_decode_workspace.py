"""Exercise the real migration metadata builder on CPU, before CUDA capture."""
from types import SimpleNamespace as NS
import unittest
import torch
import importlib.util


def builder(group_dim):
    from importlib.metadata import version
    if not version('vllm').startswith('0.31.'):
        raise unittest.SkipTest('vLLM 0.31 metadata contract required')
    from vllm.config import CUDAGraphMode
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from bfla_vllm.v031 import TritonMetadataBuilder
    # Gemma has D256 local layers but D512 global layers. Native metadata must
    # allocate for the attention group's spec, not the model-wide default.
    model = NS(get_num_attention_heads=lambda _: 8,
               get_num_kv_heads=lambda _: 2,
               get_head_size=lambda: 256, rswa_window=None)
    config = NS(model_config=model, parallel_config=NS(),
                compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                    cudagraph_capture_sizes=[1], static_forward_context={}),
                scheduler_config=NS(max_num_seqs=1))
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2,
                             head_size=group_dim, dtype=torch.bfloat16)
    return TritonMetadataBuilder(spec, [], config, torch.device('cpu'))


@unittest.skipUnless(importlib.util.find_spec('vllm'), 'vLLM runtime required')
class DecodeWorkspace(unittest.TestCase):
    def test_global_d512_workspace_covers_native_decode_store(self):
        b = builder(512)
        self.assertEqual(tuple(b.softmax_segm_output.shape), (1, 8, 16, 512))
        self.assertEqual(b.headdim, 512)

    def test_local_d256_workspace_is_unchanged(self):
        b = builder(256)
        self.assertEqual(tuple(b.softmax_segm_output.shape), (1, 8, 16, 256))
        self.assertEqual(tuple(b.softmax_segm_max.shape), (1, 8, 16))
        self.assertEqual(tuple(b.softmax_segm_expsum.shape), (1, 8, 16))


if __name__ == '__main__':
    unittest.main(verbosity=2)
