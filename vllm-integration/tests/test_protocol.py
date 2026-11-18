"""Host contracts: reject ambiguous rows and preserve native cache ownership."""
from types import SimpleNamespace as NS
import unittest


class MetadataContracts(unittest.TestCase):
    def test_fresh_proof_does_not_treat_upper_bounds_as_exact_prefix(self):
        from bfla_vllm.protocol import classify_prefill
        result = classify_prefill((0, 20, 50), (20, 30), True)
        self.assertEqual(result.lengths, (20, 30))
        self.assertEqual(result.reason, 'fresh_prefill')
        self.assertEqual(classify_prefill((0, 20), (40,), True).reason, 'prefix_or_chunk')
        self.assertEqual(classify_prefill((0, 20), (20,), True,
                                         chunking_enabled=True).reason, 'chunking_enabled')

    def test_masks_decode_and_padding_stay_native(self):
        from bfla_vllm.protocol import classify_prefill, has_prefix_mask
        self.assertNotEqual(classify_prefill((0, 20), (20,), False).reason, 'fresh_prefill')
        self.assertNotEqual(classify_prefill((0, 1, 21), (1, 20), True).reason, 'fresh_prefill')
        self.assertTrue(has_prefix_mask(NS(mm_prefix_query_range_tensor=[])))
        self.assertFalse(has_prefix_mask(NS()))
        with self.assertRaises(ValueError):
            classify_prefill((0, 21), (20,), True)

    def test_only_plain_decoder_attention_is_eligible(self):
        from bfla_vllm.protocol import attention_contract_reason
        values = dict(attn_type='decoder', sliding_window=(-1, -1), logits_soft_cap=0,
                      alibi_slopes=None, sinks=None, kv_cache_dtype='auto')
        self.assertIsNone(attention_contract_reason(NS(**values), NS()))
        for changes in ({'attn_type': 'encoder_decoder'}, {'attn_type': None},
                        {'sliding_window': (255, 0)}, {'logits_soft_cap': 30},
                        {'alibi_slopes': [1]}, {'sinks': [0]}, {'kv_cache_dtype': 'fp8'}):
            self.assertIsNotNone(attention_contract_reason(NS(**(values | changes)), NS()))
        for meta in (NS(use_cascade=True), NS(mm_prefix_range=[]), NS(mm_prefix_query_range_tensor=[]),
                     NS(rswa_prefix_lens=[92, 6]), NS(rswa_window=64), NS(rswa_window_tensor=[64])):
            self.assertIsNotNone(attention_contract_reason(NS(**values), meta))
        self.assertIsNotNone(attention_contract_reason(NS(**values), NS(), output_scaling=True))

    def test_generic_layout_uses_global_kv_when_heads_are_replicated(self):
        from bfla_vllm.protocol import resolve_head_layout
        impl = NS(num_heads=6, num_kv_heads=1, head_size=256)
        config = NS(hf_text_config=NS(num_attention_heads=24, num_key_value_heads=2))
        layout = resolve_head_layout(impl, NS(), config, tp_size=4, tp_rank=3)
        self.assertEqual((layout.global_hq, layout.global_hkv, layout.local_hq,
                          layout.local_hkv, layout.head_dim), (24, 2, 6, 1, 256))
        self.assertEqual(layout.key_weight, .5)

    def test_actual_layer_geometry_precedes_generic_config(self):
        from bfla_vllm.protocol import resolve_head_layout
        impl = NS(num_heads=4, num_kv_heads=1, head_size=512)
        layer = NS(total_num_heads=8, total_num_kv_heads=2)
        config = NS(hf_text_config=NS(num_attention_heads=8, num_key_value_heads=4))
        layout = resolve_head_layout(impl, layer, config, tp_size=2, tp_rank=1)
        self.assertEqual((layout.global_hkv, layout.head_dim), (2, 512))

    def test_config_global_attention_fields_do_not_require_model_name(self):
        from bfla_vllm.protocol import resolve_head_layout
        impl = NS(num_heads=4, num_kv_heads=1, head_size=512, sliding_window=(-1, -1))
        text = NS(model_type='a_new_name', num_attention_heads=8,
                  num_key_value_heads=4, attention_k_eq_v=True,
                  num_global_key_value_heads=2, global_head_dim=512)
        layout = resolve_head_layout(impl, NS(), NS(hf_text_config=text), tp_size=2, tp_rank=0)
        self.assertEqual(layout.global_hkv, 2)

    def test_ambiguous_replication_or_mismatched_local_geometry_is_rejected(self):
        from bfla_vllm.protocol import resolve_head_layout
        impl = NS(num_heads=6, num_kv_heads=1, head_size=256)
        with self.assertRaises(ValueError):
            resolve_head_layout(impl, NS(), None, tp_size=4, tp_rank=0)
        config = NS(hf_text_config=NS(num_attention_heads=16, num_key_value_heads=2))
        with self.assertRaises(ValueError):
            resolve_head_layout(impl, NS(), config, tp_size=4, tp_rank=0)

    def test_mixed_qsa_rows_retain_history_prompt_and_request_order(self):
        from bfla_vllm.qsa.protocol import owned_prefills, uses_sparse_prefill
        from bfla_vllm.requests import RequestContext
        context = RequestContext(('d', 'a', 'b'), (7000, 8192, 2048))
        starts, lengths = (0, 1, 98, 163), (7000, 4097, 1600)
        rows, prompts, keys = owned_prefills(starts, lengths, context)
        self.assertEqual(rows, ((1, 1, 98, 4097), (2, 98, 163, 1600)))
        self.assertEqual((prompts, keys), ((8192, 2048), ('a', 'b')))
        resumed = RequestContext(context.keys, (7000, 2048, 2048))
        self.assertEqual(owned_prefills(starts, lengths, resumed), ((), (), ()))
        main = NS(bfla_partition=(1, 162))
        self.assertTrue(uses_sparse_prefill(main, NS(num_decode_tokens=1, num_prefill_tokens=162)))
        self.assertFalse(uses_sparse_prefill(main, NS(num_decode_tokens=98, num_prefill_tokens=65)))
        self.assertFalse(uses_sparse_prefill(NS(), NS(num_decode_tokens=1, num_prefill_tokens=162)))


class CacheLayouts(unittest.TestCase):
    def test_alias_cache_031_preserves_request_page_order(self):
        import torch
        from bfla_vllm.protocol import materialize_alias_kv
        cache = torch.arange(3*2*4*6).reshape(3, 2, 4, 6)
        k, v = materialize_alias_kv(cache, torch.tensor([2, 0]), 6, 3)
        self.assertEqual(k.shape, (6, 2, 3))
        self.assertEqual(k[0, 0].tolist(), [96, 97, 98])
        self.assertEqual(v[4, 1].tolist(), [27, 28, 29])
        self.assertTrue(k.is_contiguous() and v.is_contiguous())

    def test_alias_context_must_fit_the_owner_page_table(self):
        import torch
        from bfla_vllm.protocol import materialize_alias_kv
        cache = torch.empty(3, 2, 4, 6)
        with self.assertRaisesRegex(ValueError, 'block table capacity'):
            materialize_alias_kv(cache, torch.tensor([0]), 5, 3)


class Configuration(unittest.TestCase):
    def test_environment_populates_public_config_and_execution(self):
        from bfla_vllm.settings import Settings
        result = Settings.from_env({'BFLA_EXECUTION': 'localgraph', 'BFLA_INDEX_DIM': '64',
                                    'BFLA_INDEX_DTYPE': 'bf16', 'BFLA_K_MAX': '8192'})
        self.assertEqual((result.execution, result.config.index_dim, result.config.index_dtype,
                          result.config.k_max), ('localgraph', 64, 'bf16', 8192))
        self.assertEqual(Settings.from_env({}).execution, 'eager')
        with self.assertRaises(ValueError):
            Settings.from_env({'BFLA_EXECUTION': 'other'})

    def test_uncapped_selection_is_forwarded_as_none(self):
        from bfla_vllm.settings import Settings
        self.assertIsNone(Settings.from_env({'BFLA_K_MAX': 'none'}).config.k_max)


if __name__ == '__main__':
    unittest.main()
