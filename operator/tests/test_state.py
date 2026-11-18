import unittest
from bfla import BFLAConfig, IndexPoolState
from bfla.state import refresh_start


class StateTests(unittest.TestCase):
    def test_partial_block_recomputed_from_original_keys(self):
        self.assertEqual(refresh_start(67, 70, 64), 64)
        self.assertEqual(refresh_start(128, 130, 64), 128)
        self.assertEqual(refresh_start(70, 70, 64), 70)
        self.assertEqual(refresh_start(70, 50, 64), 0)

    def test_reset_invalidates_same_length_prefix(self):
        state = IndexPoolState()
        state.filled, state.identity = 256, ('old',)
        state.reset()
        self.assertEqual(state.filled, 0)
        self.assertIsNone(state.identity)
        self.assertEqual(state.logical_bytes, 0)

    def test_incremental_state_equivalent_to_full_pool(self):
        import torch
        config = BFLAConfig(index_dtype='bf16')
        calls = []
        class Features:
            layout = 'cpu-test'
            def key(self, x, scratch):
                calls.append(len(x))
                scratch[:len(x)].copy_(x[:, 0])
        def pooler(features, pool, scales, n, cfg):
            for b in range((n + 63) // 64):
                pool[b].copy_(features[b*64:min(n, (b+1)*64)].float().mean(0))
                scales[b] = 1
        state = IndexPoolState(config, _pooler=pooler)
        keys = torch.arange(140 * 128, dtype=torch.float32).reshape(140, 1, 128).to(torch.bfloat16)
        scratch = torch.empty((140, 128), dtype=torch.bfloat16)
        for n, refreshed in ((67, 67), (70, 6), (70, 0), (140, 76), (50, 50)):
            pool = torch.empty(((n+63)//64, 128), dtype=torch.bfloat16)
            scales = torch.empty(len(pool))
            self.assertEqual(state.populate(Features(), keys[:n], scratch, pool, scales), refreshed)
            expected, expected_scales = torch.empty_like(pool), torch.empty_like(scales)
            pooler(keys[:n, 0], expected, expected_scales, n, config)
            torch.testing.assert_close(pool, expected)
        self.assertEqual(calls, [67, 6, 76, 50])


class ReaderTests(unittest.TestCase):
    def test_bounded_reader_keeps_only_pooled_prefix_and_recomputes_partial_block(self):
        import torch
        cfg=BFLAConfig(index_dtype='bf16')
        reads=[]
        keys=torch.randn(333,1,128,dtype=torch.bfloat16)
        class Features:
            layout='reader-test'
            def key(self,x,out): out[:len(x)].copy_(x[:,0])
        def pooler(x,p,s,n,c):
            for i in range((n+63)//64):
                p[i].copy_(x[i*64:min(n,(i+1)*64)].float().mean(0)); s[i]=1
        state=IndexPoolState(cfg,_pooler=pooler)
        scratch=torch.empty((128,128),dtype=torch.bfloat16)
        def reader(a,b): reads.append((a,b)); return keys[a:b]
        for n,expected in ((67,67),(70,6),(333,269),(333,0),(50,50)):
            p=torch.empty(((n+63)//64,128),dtype=torch.bfloat16); s=torch.empty(len(p))
            count=state.populate_from_reader(Features(),n,reader,scratch,p,s,
                key_shape=(1,128),key_dtype=keys.dtype,window=128)
            self.assertEqual(count,expected)
            ref=torch.empty_like(p); scale=torch.empty_like(s)
            pooler(keys[:n,0],ref,scale,n,cfg)
            torch.testing.assert_close(p,ref,rtol=0,atol=0)
        self.assertEqual(reads,[(0,67),(64,70),(64,192),(192,320),(320,333),(0,50)])
        self.assertFalse(any(isinstance(v,torch.Tensor) and v.ndim==3 for v in vars(state).values()))


if __name__ == '__main__': unittest.main()
