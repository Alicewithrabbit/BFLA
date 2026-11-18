"""Metadata contents and lifetime must survive asynchronous publication."""
import importlib.util
import unittest

try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False


@unittest.skipUnless(GPU, 'CUDA required')
class MetadataUploadTests(unittest.TestCase):
    def cache(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('bfla.metadata'),
                             'missing reusable asynchronous metadata publisher')
        from bfla.metadata import MetadataCache
        return MetadataCache(**kwargs)

    @staticmethod
    def arrays(n):
        return {'rows': torch.arange(n, dtype=torch.int32).reshape(-1, 2),
                'indices': torch.tensor([3, 1, 0], dtype=torch.int64)}

    def test_reuse_dtype_shape_and_cache_bound(self):
        cache = self.cache(max_entries=2, max_bytes=1024)
        device = torch.device('cuda', torch.cuda.current_device())
        a = cache.get(('a',), lambda: self.arrays(8), device)
        b = cache.get(('a',), lambda: self.fail('hit rebuilt metadata'), device)
        self.assertEqual(a['rows'].data_ptr(), b['rows'].data_ptr())
        self.assertTrue(torch.equal(a['rows'].cpu(), self.arrays(8)['rows']))
        self.assertEqual(a['indices'].dtype, torch.int64)
        for n in (10, 12, 14):
            cache.get((n,), lambda n=n: self.arrays(n), device)
        info = cache.info()
        self.assertLessEqual(info['entries'], 2)
        self.assertLessEqual(info['bytes'], 1024)
        self.assertTrue(torch.equal(a['rows'].cpu(), self.arrays(8)['rows']))

    def test_cross_stream_and_eviction_do_not_overwrite_inflight_values(self):
        cache = self.cache(max_entries=1, max_bytes=1024)
        device = torch.device('cuda', torch.cuda.current_device())
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            torch.cuda._sleep(20_000_000)
            a = cache.get('a', lambda: self.arrays(8), device)
        with torch.cuda.stream(consumer):
            same = cache.get('a', lambda: self.fail('hit rebuilt'), device)
            copied = same['rows'].clone()
            cache.get('b', lambda: self.arrays(16), device)
            cache.clear()
        del a, same
        consumer.synchronize()
        self.assertTrue(torch.equal(copied.cpu(), self.arrays(8)['rows']))

    def test_publication_does_not_host_synchronize(self):
        cache = self.cache()
        device = torch.device('cuda', torch.cuda.current_device())
        cache.get('warm', lambda: self.arrays(8), device)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profile:
            torch.cuda._sleep(20_000_000)
            result = cache.get('new', lambda: self.arrays(16), device)
        names = {event.key for event in profile.key_averages()}
        self.assertNotIn('cudaStreamSynchronize', names)
        self.assertTrue(torch.equal(result['rows'].cpu(), self.arrays(16)['rows']))

    def test_arena_rebinding_does_not_mutate_another_plan(self):
        from bfla import BFLAPrefillPlan, HeadLayout
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest('packaged consumer requires SM90')
        q = torch.zeros((64, 4, 512), device='cuda', dtype=torch.bfloat16)
        k = torch.zeros((256, 1, 512), device='cuda', dtype=q.dtype)
        v = torch.zeros_like(k)
        kwargs = dict(request_lengths=(64,), kv_lengths=(128,),
                      head_layout=HeadLayout.create(4, 1, 512),
                      kv_padded=True, kv_page_offsets=(0,))
        first = BFLAPrefillPlan(q, k, v, 512**-.5, **kwargs)
        second = BFLAPrefillPlan(q, k, v, 512**-.5, **kwargs)
        self.assertEqual(first.csr.data_ptr(), second.csr.data_ptr())
        self.assertNotEqual(first.table.data_ptr(), second.table.data_ptr())
        original = second.table.clone()
        first.bind_kv(k, v, (2,))
        self.assertTrue(torch.equal(second.table, original))
        self.assertEqual(first.table[0, 0].item(), 2)
        self.assertEqual(second.table[0, 0].item(), 0)


if __name__ == '__main__':
    unittest.main()
