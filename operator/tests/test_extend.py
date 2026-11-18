"""Prompt-relative supports for cached and chunked D512 requests."""
import unittest
from bfla import HeadLayout, supports
from bfla.planning import make_plan


class ExtendPlanning(unittest.TestCase):
    def test_d512_prefix_uses_absolute_positions_without_chunk_tail(self):
        heads = HeadLayout.create(8, 2, 512)
        self.assertTrue(supports(heads, request_lengths=(257, 65),
            kv_lengths=(4353, 8257), prompt_lengths=(8192, 16384)).supported)
        p = make_plan((257, 65), heads, kv_lengths=(4353, 8257),
                      prompt_lengths=(8192, 16384))
        self.assertEqual([r.prefix for r in p.requests], [4096, 8192])
        self.assertEqual([r.sparse_rows for r in p.requests], [257, 65])
        self.assertEqual(p.groups[0].first, 4096)
        self.assertEqual(p.groups[-1].last, 8256)
        self.assertEqual(p.layout['used_k'], [4353, 8257])

    def test_d512_last_chunk_keeps_original_prompt_tail(self):
        p = make_plan((512,), HeadLayout.create(8, 2, 512),
                      kv_lengths=(8192,), prompt_lengths=(8192,))
        self.assertEqual(p.requests[0].sparse_rows, 256)
        self.assertEqual(len(p.dense_tiles), 8)
        self.assertEqual(p.counts[-1], 128)

    def test_one_query_cached_prefix_is_not_a_short_context(self):
        p = make_plan((1,), HeadLayout.create(8, 2, 512),
                      kv_lengths=(4097,), prompt_lengths=(8192,))
        self.assertEqual(p.requests[0].sparse_rows, 1)
        self.assertEqual(p.groups[0].first, 4096)
        self.assertEqual(p.groups[0].size, 65)

    def test_unaligned_prefix_does_not_share_sparse_support_with_protected_tail(self):
        for qlen,sparse in ((4095,3839),(257,1)):
            p = make_plan((qlen,),HeadLayout.create(8,2,512),
                          kv_lengths=(8192,),prompt_lengths=(8192,))
            self.assertEqual(p.layout['cu_q'],[0,sparse,qlen])
            self.assertEqual(p.counts[p.dense_tiles[0]],125)


if __name__ == '__main__': unittest.main()
