import unittest

import torch

from bfla import build


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
                     'SM90 required')
class SelectorCapacityTests(unittest.TestCase):
    def test_ragged_topk_across_candidate_bounds(self):
        sizes = (500, 8192, 8193, 15625, 16384)
        n = max(sizes)
        generator = torch.Generator(device='cuda').manual_seed(71)
        scores = torch.randint(64, (len(sizes), n), generator=generator, device='cuda').float()
        scores[:, :4] = torch.inf
        sizes_gpu = torch.tensor(sizes, device='cuda', dtype=torch.int32)
        for capacity in (512, 2048):
            with self.subTest(capacity=capacity):
                ks = (capacity, 32, 1, capacity - 1, capacity)
                ks_gpu = torch.tensor(ks, device='cuda', dtype=torch.int32)
                op = build.selector(capacity, candidates=n)
                values = torch.empty((len(sizes), capacity), device='cuda')
                ids = torch.full_like(values, -1, dtype=torch.int32)
                workspace = torch.empty(op.workspace_bytes(len(sizes), n),
                                        dtype=torch.uint8, device='cuda')
                op.topk(scores.data_ptr(), values.data_ptr(), ids.data_ptr(),
                        sizes_gpu.data_ptr(), ks_gpu.data_ptr(), workspace.data_ptr(),
                        workspace.numel(), len(sizes), n, capacity,
                        torch.cuda.current_stream().cuda_stream)
                for row, (size, k) in enumerate(zip(sizes, ks)):
                    count = min(k, size)
                    expected = scores[row, :size].argsort(descending=True, stable=True)[:count]
                    got = ids[row, :count].long()
                    torch.testing.assert_close(got.sort().values, expected.sort().values)
                    torch.testing.assert_close(values[row, :count], scores[row, got])
                    self.assertTrue(ids[row, count:].eq(-1).all().item())
