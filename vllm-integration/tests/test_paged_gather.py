import unittest
import torch


class NativeViews(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
    def test_sm90_tp2_singleton_stride_matches_native_paged_attention(self):
        from bfla import BFLAPrefillPlan, HeadLayout
        from bfla_vllm.paged import native_views
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest('packaged consumer requires SM90')
        torch.manual_seed(927)
        heads = HeadLayout.create(8, 2, 512, tp_size=2, tp_rank=0)
        cache = torch.randn(65, 1, 64, 1024, device='cuda', dtype=torch.bfloat16)
        query = torch.randn(65, 4, 512, device=cache.device, dtype=cache.dtype)
        table = torch.randperm(65, device=cache.device).int().view(1, -1)
        # Two identical rank contributions keep this stride check on one GPU.
        kwargs = dict(request_lengths=(65,), kv_lengths=(4097,), prompt_lengths=(8192,),
                      head_layout=heads, all_reduce=lambda partial: partial * 2, kv_page_table=table)
        regular_k, regular_v = cache.transpose(1, 2).split(512, -1)
        reference = BFLAPrefillPlan(query, regular_k, regular_v, 512**-.5, **kwargs)
        expected = reference.run(query, regular_k, regular_v)

        unusual = cache.as_strided(cache.shape, (cache.stride(0), 1, cache.stride(2), 1))
        key, value = native_views(unusual, 512)
        actual = BFLAPrefillPlan(query, key, value, 512**-.5, **kwargs)
        self.assertGreater(actual.selector_invocations, 0)
        output = actual.run(query, key, value)
        torch.testing.assert_close(actual.csr, reference.csr, atol=0, rtol=0)
        torch.testing.assert_close(actual.ids, reference.ids, atol=0, rtol=0)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_tp2_d512_singleton_strides_keep_owner_addresses_and_values(self):
        from bfla.prepared import _tma_aligned
        from bfla_vllm.paged import native_views

        # Nonzero cache offset and gaps between pages and tokens are intentional.
        storage = torch.arange(4 * 2 * 16 * 1040).to(torch.bfloat16)
        owner = storage.reshape(4, 2, 16, 1040)[1:4:2, :1, :, 8:1032]
        for cache in (owner, owner[:1], owner[:, :, :1]):
            with self.subTest(shape=tuple(cache.shape)):
                strides = tuple(1 if size == 1 else stride
                                for size, stride in zip(cache.shape, cache.stride()))
                unusual = cache.as_strided(cache.shape, strides)
                key, value = native_views(unusual, 512)
                for view, expected in zip((key, value), cache.transpose(1, 2).split(512, -1)):
                    self.assertEqual(view.data_ptr(), expected.data_ptr())
                    self.assertEqual(view.shape, expected.shape)
                    self.assertTrue(torch.equal(view, expected))
                    self.assertTrue(_tma_aligned(view), view.stride())
                    for axis, size in enumerate(view.shape):
                        if size > 1:
                            self.assertEqual(view.stride(axis), expected.stride(axis))

    def test_packed_native_layout_keeps_distinct_k_v_and_strides(self):
        from bfla_vllm.paged import native_views
        cache = torch.arange(3*2*784*16).reshape(3,2,784,16)
        k,v = native_views(cache,8)
        self.assertEqual(k.shape,(3,784,2,8))
        self.assertEqual(k[2,783,1,7],cache[2,1,783,7])
        self.assertEqual(v[2,783,1,7],cache[2,1,783,15])
        self.assertEqual(k.untyped_storage().data_ptr(),cache.untyped_storage().data_ptr())
        self.assertNotEqual(k.data_ptr(),v.data_ptr())


if __name__ == '__main__': unittest.main()
