import importlib.util
import unittest


class ImportContract(unittest.TestCase):
    def test_operator_is_an_importable_package(self):
        self.assertIsNotNone(importlib.util.find_spec('bfla'),
                             'operator must be independently importable')


class ConfigContract(unittest.TestCase):
    def setUp(self):
        from bfla import BFLAConfig, HeadLayout, supports
        self.Config, self.Heads, self.supports = BFLAConfig, HeadLayout, supports

    def test_budget_rounding_and_independent_short_threshold(self):
        p = self.Config(k_min=1024)
        self.assertEqual([p.budget(n) for n in (100, 3073, 100000)], [100, 1025, 32768])
        self.assertEqual(p.short_full_threshold, 2048)
        self.assertEqual(p.block_count(3073), 17)

    def test_valid_combined_parameters_are_not_ablation_gated(self):
        p = self.Config(index_dim=64, index_dtype='bf16', k_max=8192,
                        ratio_denominator=5, query_tile_rows=256, pool_block_size=128)
        self.assertEqual(p.budget(100000), 8192)
        h = self.Heads.create(24, 4, 256)
        self.assertTrue(self.supports(h, p, request_lengths=(4096, 5000)).supported)

    def test_new_head_counts_are_judged_by_geometry_not_model_name(self):
        h = self.Heads.create(48, 8, 256)
        self.assertTrue(self.supports(h, request_lengths=(8192,)).supported)

    def test_unsupported_semantics_return_specific_reasons(self):
        h = self.Heads.create(24, 4, 256)
        r = self.supports(h, request_lengths=(8192,), dtype='float16', causal=False)
        self.assertFalse(r.supported)
        self.assertTrue(any('dtype' in reason for reason in r.reasons))
        self.assertTrue(any('causal' in reason for reason in r.reasons))

    def test_dimension_helper_is_not_a_kernel_support_claim(self):
        h = self.Heads.create(32, 8, 128)
        self.assertFalse(self.supports(h, request_lengths=(4096,)).supported)

    def test_d512_native_guard_and_extend_limits_are_explicit(self):
        h = self.Heads.create(8, 2, 512)
        self.assertTrue(self.supports(h, request_lengths=(8192, 5000)).supported)
        self.assertTrue(self.supports(h, request_lengths=(4096,), kv_lengths=(8192,)).supported)
        self.assertFalse(self.supports(h, request_lengths=(8192,), kv_lengths=(4096,)).supported)
        h2 = self.Heads.create(16, 4, 512)
        self.assertFalse(self.supports(h2, request_lengths=(8192,)).supported)

    def test_tp_replication_mapping(self):
        h = self.Heads.create(8, 2, 512, 4, 3)
        self.assertEqual((h.query_head_ids, h.kv_head_ids), ((6, 7), (1,)))
        self.assertEqual((h.selection_ratio, h.consumer_ratio, h.key_weight), (4, 2, .5))

    def test_unbounded_policy_does_not_bypass_selector_capacity(self):
        h = self.Heads.create(24, 4, 256)
        p = self.Config(k_max=None)
        self.assertTrue(self.supports(h, p, request_lengths=(131072,)).supported)
        self.assertFalse(self.supports(h, p, request_lengths=(524288,)).supported)

    def test_invalid_ratio_or_budget_is_rejected(self):
        for kwargs in (dict(ratio_denominator=0), dict(ratio_numerator=4),
                       dict(k_min=0), dict(k_max=1024), dict(index_dim=32)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.Config(**kwargs)

    def test_mandatory_quota_is_a_joint_capability_constraint(self):
        h = self.Heads.create(24,4,256)
        result = self.supports(h,self.Config(k_min=1,ratio_denominator=1000),request_lengths=(4096,))
        self.assertFalse(result.supported)
        self.assertTrue(any('mandatory' in r for r in result.reasons))


if __name__ == '__main__':
    unittest.main()
