"""Only immutable geometry is reused; native D lengths stay device-owned."""
from types import SimpleNamespace as NS
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeMetadataUpload(unittest.TestCase):
    def test_reuse_and_reorder_preserve_current_device_lengths(self):
        from bfla_vllm.partition import native_metadata
        meta=NS(query_start_loc=torch.tensor([0,1,12,13,24],device='cuda',dtype=torch.int32),
                seq_lens=torch.tensor([91,32,73,64],device='cuda',dtype=torch.int32),
                block_table=torch.arange(12,device='cuda',dtype=torch.int32).reshape(4,3),
                slot_mapping=torch.arange(24,device='cuda'),use_cascade=False)
        rows=((0,0,1,100),(2,12,13,100))
        first=native_metadata(meta,rows)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profile:
            meta.seq_lens.add_(2)
            second=native_metadata(meta,rows)
            third=native_metadata(meta,((1,1,2,100),(3,13,14,100)))
        self.assertEqual(first.query_start_loc.data_ptr(),second.query_start_loc.data_ptr())
        self.assertEqual(second.seq_lens.cpu().tolist(),[93,75])
        self.assertEqual(third.seq_lens.cpu().tolist(),[34,66])
        self.assertEqual(second.query_start_loc.cpu().tolist(),[0,1,2])
        self.assertEqual(third.block_table.cpu().tolist(),[[3,4,5],[9,10,11]])
        self.assertNotIn('cudaStreamSynchronize',{e.key for e in profile.key_averages()})


if __name__=='__main__':unittest.main()
