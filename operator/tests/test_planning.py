import unittest


class PlanningContract(unittest.TestCase):
    def setUp(self):
        from bfla import BFLAConfig, HeadLayout
        from bfla.planning import make_plan, tail_boundary, mandatory_blocks
        self.Config, self.Heads = BFLAConfig, HeadLayout
        self.plan, self.tail, self.mandatory = make_plan, tail_boundary, mandatory_blocks

    def test_tail_is_prompt_relative_and_short_threshold_is_not_kmin(self):
        p = self.Config(k_min=8192)
        self.assertEqual(self.tail(8192, 4, p), 7936)
        self.assertEqual(self.tail(2048, 4, p), 0)
        self.assertEqual(self.tail(8192, 6, p), 8021)
        self.assertEqual(self.tail(8192, 4, self.Config(tail_tiles=0)), 8192)

    def test_mandatory_union_counts_overlap_once(self):
        self.assertEqual(self.mandatory(0, 31, self.Config()), (0,))
        self.assertEqual(self.mandatory(1024, 1055, self.Config()),
                         (0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16))

    def test_default_full_support_suffix_and_causal_counts(self):
        p = self.plan((8192,), self.Heads.create(8, 2, 512))
        self.assertEqual(p.requests[0].sparse_rows, 7936)
        self.assertEqual(len(p.counts), 256)
        self.assertEqual(p.counts[-8:], (125, 125, 126, 126, 127, 127, 128, 128))
        self.assertEqual(p.groups[0].count, 1)
        self.assertEqual(p.groups[63].count, 32)
        self.assertEqual(p.groups[-1].count, 42)

    def test_two_request_metadata_never_crosses_request(self):
        p = self.plan((4096, 6144), self.Heads.create(24, 4, 256))
        self.assertEqual(p.layout['request_page_offsets'], [0, 64])
        self.assertEqual(p.layout['total_pages'], 160)
        self.assertEqual(p.layout['cu_q'][-1], 10240)
        self.assertEqual(len(p.counts), p.layout['cu_tiles'][-1])
        self.assertEqual({g.request for g in p.groups}, {0, 1})

    def test_extend_early_chunk_does_not_protect_its_own_tail(self):
        p = self.plan((4096,), self.Heads.create(24, 2, 256, 2, 0),
                      kv_lengths=(8192,), prompt_lengths=(16384,))
        self.assertEqual(p.requests[0].sparse_rows, 4096)
        self.assertEqual(p.groups[0].first, 4096)
        self.assertEqual(p.groups[-1].last, 8191)

    def test_last_extend_chunk_uses_full_prompt_tail(self):
        p = self.plan((4096,), self.Heads.create(24, 2, 256, 2, 1),
                      kv_lengths=(16384,), prompt_lengths=(16384,))
        self.assertEqual(p.requests[0].sparse_rows, 4010)

    def test_replicated_heads_preserve_support_tiles(self):
        p = self.plan((8192,), self.Heads.create(8, 2, 512, 4, 0))
        self.assertEqual(len(p.counts), p.layout['cu_tiles'][-1])
        self.assertEqual(len(p.counts), 256)
        self.assertLessEqual(p.layout['max_rows'], 32)

    def test_nondefault_combined_policy_preserves_mandatory_and_counts(self):
        cfg = self.Config(ratio_denominator=5, query_tile_rows=256, pool_block_size=128,
                          index_dtype='bf16', k_max=8192)
        p = self.plan((9001,), self.Heads.create(24, 4, 256), cfg)
        for g in p.groups:
            self.assertGreaterEqual(g.count, len(self.mandatory(g.first, g.last, cfg)))
            self.assertLessEqual(g.count, g.size)
            self.assertTrue(all(p.counts[t] > 0 for t in g.consumer_tiles))


if __name__ == '__main__':
    unittest.main()
