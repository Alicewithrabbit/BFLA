"""Default scoring must retain the packed-head reduction and partial-tile policy."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
class QueryReuseTests(unittest.TestCase):
    def test_graph_replay_reads_updated_query_and_pool(self):
        from bfla._kernels.selection import score_rows
        from bfla._kernels.features import score_index
        torch.manual_seed(713)
        q = torch.randn(33, 128, device='cuda')
        pool = torch.randn(129, 128, device='cuda').to(torch.float8_e4m3fn)
        scales = torch.ones(129, device='cuda')
        rows = torch.tensor([(0, 0, 129, 4096, 33, t) for t in range(4)],
                            device='cuda', dtype=torch.int32)
        actual = torch.empty((4, 129), device='cuda')
        reference = torch.empty_like(actual)
        def run():
            score_rows[(4, 5)](q, pool, scales, actual, rows, 12, 129, PRUNED=True)
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            q.normal_()
            pool.copy_(torch.randn(129, 128, device='cuda').to(pool.dtype))
            graph.replay()
            score_index[(4, 5)](q, pool, scales, reference, rows, 12, 129,
                                128, 128, 64, PRUNED=True)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_scores_and_selected_sets_match_packed_reference(self):
        import triton
        from bfla._kernels.selection import score_rows
        from bfla._kernels.features import score_index
        torch.manual_seed(289)
        for g in (4, 6, 12):
            for dtype in (torch.float8_e4m3fn, torch.bfloat16):
                for tied in (False, True):
                    q = torch.randn(139, 128, device='cuda')
                    pool = torch.randn(2053, 128, device='cuda').to(dtype)
                    if tied:
                        pool.zero_()
                    scales = torch.rand(2053, device='cuda') + .1
                    records = []
                    # Three independent requests with partial packed head groups.
                    for qbase, n, prefix, poolbase, nb in ((0, 67, 65531, 0, 1029),
                                                          (67, 65, 122877, 11, 2041),
                                                          (132, 7, 16384, 7, 311)):
                        records.extend((qbase, poolbase, nb, prefix, n, tile)
                                       for tile in range(triton.cdiv(n*g, 128)))
                    descriptors = torch.tensor(records, device='cuda', dtype=torch.int32)
                    shape = (len(records), 2053)
                    actual = torch.empty(shape, device='cuda')
                    reference = torch.empty_like(actual)
                    for pruned in (False, True):
                        with self.subTest(g=g, dtype=dtype, tied=tied, pruned=pruned):
                            grid = (len(records), triton.cdiv(shape[1], 32))
                            score_rows[grid](q, pool, scales, actual, descriptors, g, shape[1], PRUNED=pruned)
                            score_index[grid](q, pool, scales, reference, descriptors, g, shape[1],
                                              128, 128, 64, PRUNED=pruned)
                            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                            for k in (32, 128, 256):
                                a = actual.topk(k, sorted=False).indices.sort().values
                                b = reference.topk(k, sorted=False).indices.sort().values
                                torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
