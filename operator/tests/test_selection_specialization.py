"""Request lengths must reuse compiled indexing kernels within a tile class."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
class SelectionSpecializationTests(unittest.TestCase):
    def test_score_stride_reuses_kernel_and_matches_scores(self):
        import triton
        from bfla._kernels.selection import score_rows

        torch.manual_seed(127)
        q = torch.randn(33, 128, device='cuda')
        pool = torch.randn(131, 128, device='cuda').to(torch.float8_e4m3fn)
        scales = torch.rand(131, device='cuda') + .5
        for pruned in (False, True):
            hashes = set()
            for stride in (129, 130, 131):
                rows = torch.tensor([[0, 0, stride, 4096, 33, t] for t in (0, 1)],
                                    device='cuda', dtype=torch.int32)
                scores = torch.empty(2, stride, device='cuda')
                kernel = score_rows[(2, triton.cdiv(stride, 32))](
                    q, pool, scales, scores, rows, 4, stride, PRUNED=pruned)
                hashes.add(kernel.hash)
                block_ids = torch.arange(stride, device='cuda')
                for tile in (0, 1):
                    packed = torch.arange(tile*128, min((tile+1)*128, 33*4), device='cuda')
                    queries = q[packed//4].bfloat16().float()
                    logits = (queries @ pool[:stride].bfloat16().float().T).relu()
                    logits *= 128**-.5 * scales[:stride]
                    expected = logits.logsumexp(0)
                    first, last = 4096+tile*32, 4096+int(packed[-1])//4
                    mandatory = min(first//64, max(0, (first//128-4+1)*128)//64)
                    expected[(block_ids < 4) | (block_ids >= mandatory)] = float('inf')
                    expected[block_ids > last//64] = -float('inf')
                    torch.testing.assert_close(scores[tile], expected, atol=.003, rtol=.003)
            self.assertEqual(len(hashes), 1, 'changing score stride compiled another kernel')

    def test_publish_runtime_tile_count_and_capacity(self):
        from bfla._kernels.selection import publish_rows

        hashes = set()
        for tiles, capacity in ((9, 5), (10, 6), (17, 7)):
            selected = torch.full((2, capacity), -1, device='cuda', dtype=torch.int32)
            selected[0, :3] = torch.tensor([7, 2, 5], device='cuda')
            selected[1, :2] = torch.tensor([9, 1], device='cuda')
            lengths = torch.tensor([3, 2], device='cuda', dtype=torch.int32)
            offsets = torch.tensor([1, tiles-1], device='cuda', dtype=torch.int32)
            csr = torch.arange(2*tiles+1, device='cuda', dtype=torch.int32)*4
            ids = torch.full((2*tiles*4+8,), -99, device='cuda', dtype=torch.int32)
            kernel = publish_rows[(2, 2)](selected, lengths, offsets, csr, ids, tiles, capacity, 8)
            hashes.add(kernel.hash)
            expected = torch.full_like(ids, -99)
            for head in (0, 1):
                for tile, values in ((1, [2, 5, 7]), (tiles-1, [1, 9])):
                    start = (head*tiles+tile)*4
                    expected[start:start+len(values)] = torch.tensor(values, device='cuda')
            torch.testing.assert_close(ids, expected, atol=0, rtol=0)
        self.assertEqual(len(hashes), 1, 'runtime publish dimensions compiled another kernel')

    def test_identity_runtime_tile_count(self):
        from bfla._kernels.runtime import identity

        hashes = set()
        for tiles in (9, 10, 17):
            counts = torch.arange(2*tiles, device='cuda', dtype=torch.int32) % 8 + 1
            csr = torch.cat((torch.zeros(1, device='cuda', dtype=torch.int32), counts.cumsum(0).int()))
            total = int(csr[-1])
            ids = torch.full((total+8,), -99, device='cuda', dtype=torch.int32)
            kernel = identity[(tiles, 2)](csr, ids, tiles, 8)
            hashes.add(kernel.hash)
            expected = torch.cat([torch.arange(n, device='cuda', dtype=torch.int32)
                                  for n in counts.cpu().tolist()])
            torch.testing.assert_close(ids[:total], expected, atol=0, rtol=0)
            self.assertTrue(bool((ids[total:] == -99).all()))
        self.assertEqual(len(hashes), 1, 'changing tile count compiled another identity kernel')


if __name__ == '__main__':
    unittest.main()
