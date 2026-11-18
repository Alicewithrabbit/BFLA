"""Page metadata and quota checks retain the exact logical support policy."""
from dataclasses import replace
import unittest
from bfla import BFLAConfig, HeadLayout
from bfla import planning


class PlanEfficiencyTests(unittest.TestCase):
    def test_capability_validation_visits_only_current_chunk(self):
        from unittest.mock import patch
        from bfla.capabilities import supports, _supports
        _supports.cache_clear()
        with patch.object(planning, 'mandatory_count', wraps=planning.mandatory_count) as count:
            result = supports(HeadLayout.create(24, 4, 256), request_lengths=(129,),
                              kv_lengths=(65536,), prompt_lengths=(131072,), kv_layout='paged')
        self.assertTrue(result.supported)
        self.assertGreater(count.call_count, 0)
        self.assertTrue(all(65536-129 <= c.args[0] <= c.args[1] < 65536
                            for c in count.call_args_list))

    def test_equivalent_chunk_prompts_reuse_immutable_geometry_only(self):
        h = HeadLayout.create(24, 4, 256)
        kwargs = dict(kv_lengths=(32768, 49152), native_paged=True)
        a = planning.make_plan((8192, 8192), h, prompt_lengths=(65536, 98304), **kwargs)
        b = planning.make_plan((8192, 8192), h, prompt_lengths=(98304, 131072), **kwargs)
        self.assertEqual(a, b)
        self.assertTrue(a.groups is b.groups)
        a.layout['used_k'][0] = -1
        a.layout['total_pages'] = -1
        c = planning.make_plan((8192, 8192), h, prompt_lengths=(65536, 98304), **kwargs)
        self.assertEqual(c, b)
        tail = planning.make_plan((8192, 8192), h, prompt_lengths=(32768, 49152), **kwargs)
        self.assertNotEqual(tail.groups, b.groups)

    def test_returned_contiguous_tables_are_independently_mutable(self):
        h = HeadLayout.create(8, 2, 512)
        a = planning.make_plan((4096,), h)
        b = planning.make_plan((4096,), h)
        a.layout['table'][0][0] = -1
        a.layout['request_page_offsets'][0] = -1
        c = planning.make_plan((4096,), h)
        self.assertEqual(c, b)

    def test_large_full_prompts_do_not_accumulate_cached_plans(self):
        h = HeadLayout.create(24, 4, 256)
        a = planning.make_plan((16385,), h, native_paged=True)
        b = planning.make_plan((16385,), h, native_paged=True)
        self.assertEqual(a, b)
        self.assertIsNot(a.groups, b.groups)

    def test_cpu_plan_cache_has_a_fixed_entry_bound(self):
        h = HeadLayout.create(8, 2, 512)
        kwargs = dict(kv_lengths=(4096,), prompt_lengths=(8192,), native_paged=True)
        first = planning.make_plan((17,), h, **kwargs)
        for n in range(18, 83):
            planning.make_plan((n,), h, **kwargs)
        repeated = planning.make_plan((17,), h, **kwargs)
        self.assertEqual(first, repeated)
        self.assertIsNot(first.groups, repeated.groups)

    def test_interval_count_matches_mandatory_union_at_boundaries(self):
        self.assertTrue(hasattr(planning, 'mandatory_count'))
        for block in (64, 128, 256):
            for sink in (0, 1, 63, 256, 8192):
                for alignment, local in ((64, 1), (128, 4), (257, 3)):
                    cfg = BFLAConfig(pool_block_size=block, sink_tokens=sink,
                                     local_alignment=alignment, local_blocks=local)
                    for first in (0, 1, 63, 64, 127, 128, 511, 512, 4095, 4096, 8191):
                        for width in (1, 2, 32, 129, 512):
                            last = first + width - 1
                            with self.subTest(block=block, sink=sink, first=first, last=last):
                                self.assertEqual(planning.mandatory_count(first, last, cfg),
                                                 len(planning.mandatory_blocks(first, last, cfg)))

    def test_native_page_plan_omits_only_contiguous_table(self):
        for heads in (HeadLayout.create(8, 2, 512), HeadLayout.create(24, 4, 256),
                      HeadLayout.create(24, 2, 256, 2, 0)):
            for batch in (1, 2, 3, 8):
                q = tuple(129 + i * 17 for i in range(batch))
                k = tuple(8191 + i * 784 for i in range(batch))
                prompts = tuple(n + 1 for n in k)
                ordinary = planning.make_plan(q, heads, kv_lengths=k, prompt_lengths=prompts)
                native = planning.make_plan(q, heads, kv_lengths=k, prompt_lengths=prompts,
                                            native_paged=True)
                self.assertIsNone(native.layout['table'])
                restored = dict(native.layout, table=ordinary.layout['table'])
                self.assertEqual(replace(native, layout=restored), ordinary)


if __name__ == '__main__':
    unittest.main()
