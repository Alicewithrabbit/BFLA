"""Small support-conditioned attention oracle; not a performance matrix."""
import unittest
import torch
from bfla import BFLAConfig, HeadLayout


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
class GPUAttentionTests(unittest.TestCase):
    def check_case(self, heads, lengths, config=BFLAConfig(), kv_lengths=None, prompt_lengths=None):
        from bfla import BFLAPrefillPlan
        if torch.cuda.get_device_capability() != (9,0):
            self.skipTest('packaged consumer requires SM90')
        torch.manual_seed(42)
        klens = lengths if kv_lengths is None else kv_lengths
        q = torch.randn((sum(lengths),heads.local_hq,heads.head_dim),device='cuda',dtype=torch.bfloat16)
        k = torch.randn((sum(klens),heads.local_hkv,heads.head_dim),device='cuda',dtype=torch.bfloat16)
        v = torch.randn_like(k)
        plan = BFLAPrefillPlan(q,k,v,heads.head_dim**-.5,request_lengths=lengths,
                              head_layout=heads,kv_lengths=klens,prompt_lengths=prompt_lengths,config=config)
        out = plan.run(q,k,v)
        csr, ids = plan.csr.cpu().tolist(), plan.ids.cpu().tolist()
        layout = plan.layout
        # Sample boundaries and tail, not just the first full-support tile.
        qstarts = [sum(lengths[:i]) for i in range(len(lengths))]
        kstarts = [sum(klens[:i]) for i in range(len(klens))]
        for seg,(start,stop) in enumerate(zip(layout['cu_q'],layout['cu_q'][1:])):
            req = max(i for i,s in enumerate(qstarts) if s <= start)
            for row in sorted({0,min(31,stop-start-1),min(32,stop-start-1),stop-start-1}):
                position = klens[req]-lengths[req]+start-qstarts[req]+row
                for h in range(heads.local_hq):
                    kh = h//heads.consumer_ratio
                    tile = layout['cu_tiles'][seg]+(row*heads.consumer_ratio+h%heads.consumer_ratio)//128
                    address = kh*plan.tiles+tile
                    blocks = ids[csr[address]:csr[address+1]]
                    tokens = [b*64+j for b in blocks for j in range(64) if b*64+j<=position]
                    self.assertEqual(len(tokens),len(set(tokens)))
                    from bfla.planning import tail_boundary
                    prompt = (klens if prompt_lengths is None else prompt_lengths)[req]
                    if position >= tail_boundary(prompt,heads.selection_ratio,config):
                        self.assertEqual(sorted(tokens),list(range(position+1)),
                                         'protected tail must retain full causal support')
                    ti = torch.tensor(tokens,device='cuda')+kstarts[req]
                    logits = k[ti,kh].float() @ q[start+row,h].float() * heads.head_dim**-.5
                    expected = logits.softmax(0) @ v[ti,kh].float()
                    torch.testing.assert_close(out[start+row,h].float(),expected,atol=.02,rtol=.02)

    def test_d256_two_requests(self):
        self.check_case(HeadLayout.create(24,4,256),(2112,2176))

    def test_d512_two_requests_and_combined_configuration(self):
        self.check_case(HeadLayout.create(8,2,512),(2112,2176),
            BFLAConfig(index_dim=64,index_dtype='bf16',pool_block_size=128,query_tile_rows=256))

    def test_extend_d512_and_d256_ragged_causal_boundaries(self):
        for heads in (HeadLayout.create(8,2,512), HeadLayout.create(24,4,256)):
            with self.subTest(dim=heads.head_dim):
                self.check_case(heads,(257,65,1),kv_lengths=(4353,8192,4097),
                                prompt_lengths=(8192,8192,16384))

    def test_d512_unaligned_prefix_retains_all_tail_keys(self):
        self.check_case(HeadLayout.create(8,2,512),(4095,257),
                        kv_lengths=(8192,8192),prompt_lengths=(8192,8192))

    def test_nan_value_tail_in_caller_padded_and_rebound_arena(self):
        from bfla import BFLAPrefillPlan
        if torch.cuda.get_device_capability() != (9,0):
            self.skipTest('packaged consumer requires SM90')
        torch.manual_seed(518)
        lengths, klens = (65,31), (4097,3519)
        for heads in (HeadLayout.create(24,4,256), HeadLayout.create(4,1,512)):
            d, hk = heads.head_dim, heads.local_hkv
            q = torch.randn(sum(lengths),heads.local_hq,d,device='cuda',dtype=torch.bfloat16)
            pk = torch.randn(sum(klens),hk,d,device=q.device,dtype=q.dtype)
            pv = torch.randn_like(pk)
            kwargs = dict(request_lengths=lengths,kv_lengths=klens,prompt_lengths=(8192,8192),
                          head_layout=heads)
            packed = BFLAPrefillPlan(q,pk,pv,d**-.5,**kwargs)
            expected = packed.run(q,pk,pv)
            arena = None
            # The requests occupy 65 and 55 pages, respectively.
            for offsets,pages,sign in ((None,120,1),((58,1),126,1),((2,71),132,-1)):
                with self.subTest(dim=d,offsets=offsets):
                    k = torch.zeros(pages*64,hk,d,device=q.device,dtype=q.dtype)
                    v = torch.full_like(k,float('nan'))
                    for off,key,value in zip(offsets or (0,65),pk.split(klens),pv.split(klens)):
                        k[off*64:off*64+len(key)].copy_(key)
                        v[off*64:off*64+len(value)].copy_(value*sign)
                    before = v.clone()
                    if offsets is None:
                        plan = BFLAPrefillPlan(q,k,v,d**-.5,kv_padded=True,**kwargs)
                    elif arena is None:
                        arena = BFLAPrefillPlan(q,k,v,d**-.5,kv_padded=True,
                                               kv_page_offsets=offsets,**kwargs)
                        plan = arena
                    else:
                        arena.bind_kv(k,v,offsets)
                        plan = arena
                    actual = plan.run(q,k,v)
                    torch.testing.assert_close(actual,expected*sign,atol=.002,rtol=.01)
                    torch.testing.assert_close(plan.csr,packed.csr,atol=0,rtol=0)
                    torch.testing.assert_close(plan.ids,packed.ids,atol=0,rtol=0)
                    torch.testing.assert_close(v,before,atol=0,rtol=0,equal_nan=True)

    def test_d512_value_partitions_paged_and_replay(self):
        from bfla import BFLAPrefillPlan
        if torch.cuda.get_device_capability() != (9,0):
            self.skipTest('packaged consumer requires SM90')
        heads = HeadLayout.create(4,1,512)
        torch.manual_seed(1401)
        # Different selected-tile counts exercise both pipeline ring stages.
        for page,batch,edge in ((16,1,63),(64,2,785),(784,3,4097),(784,8,8193)):
            with self.subTest(page=page,batch=batch,edge=edge):
                klens = tuple(edge+33*i for i in range(batch))
                lengths = tuple(min(n,65+7*i) for i,n in enumerate(klens))
                width = (max(klens)+page-1)//page
                table = torch.randperm(batch*width,device='cuda').int().view(batch,width)
                cache = torch.full((batch*width,1,page,1024),float('nan'),
                                   device='cuda',dtype=torch.bfloat16)
                keys,values = [],[]
                for i,n in enumerate(klens):
                    kv = torch.randn(n,1,1024,device='cuda',dtype=cache.dtype)
                    # Each V partition must receive the same final normalization.
                    kv[...,768:] = kv[...,512:768]*2
                    pos = torch.arange(n,device='cuda')
                    cache[table[i,pos//page].long(),:,pos%page] = kv
                    keys.append(kv[...,:512]);values.append(kv[...,512:])
                k,v = cache[...,:512].permute(0,2,1,3),cache[...,512:].permute(0,2,1,3)
                q = torch.randn(sum(lengths),4,512,device='cuda',dtype=cache.dtype)
                kwargs = dict(request_lengths=lengths,kv_lengths=klens,
                              prompt_lengths=klens,head_layout=heads)
                paged = BFLAPrefillPlan(q,k,v,512**-.5,kv_page_table=table,**kwargs)
                pk,pv = torch.cat(keys),torch.cat(values)
                packed = BFLAPrefillPlan(q,pk,pv,512**-.5,**kwargs)
                expected = packed.run(q,pk,pv).clone()
                actual = paged.run(q,k,v)
                torch.testing.assert_close(paged.ids,packed.ids,atol=0,rtol=0)
                torch.testing.assert_close(paged.csr,packed.csr,atol=0,rtol=0)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                torch.testing.assert_close(actual[...,256:],actual[...,:256]*2,atol=0,rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    paged.run_attention(q,k,v)
                q.mul_(0.97)
                expected = packed.run_attention(q,pk,pv).clone()
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(paged.out,expected,atol=0,rtol=0)
                del graph,paged,packed

    def test_native_d256_partial_blocks_and_page_crossings(self):
        from bfla import BFLAPrefillPlan
        if torch.cuda.get_device_capability() != (9,0):
            self.skipTest('packaged consumer requires SM90')
        torch.manual_seed(724)
        heads = HeadLayout.create(24,4,256)
        for page in (16,64,784):
            for batch in (1,2,3,8):
                for edge in (63,785,4097):
                    with self.subTest(page=page,batch=batch,edge=edge):
                        klens = tuple(edge+33*i for i in range(batch))
                        lengths = tuple(min(n,65+7*i) for i,n in enumerate(klens))
                        width = (max(klens)+page-1)//page
                        table = torch.randperm(batch*width,device='cuda').int().view(batch,width)
                        # Poison unused physical rows: an incorrectly unmasked
                        # final V block must not silently pass the output check.
                        cache = torch.full((batch*width,4,page,512),float('nan'),
                                           device='cuda',dtype=torch.bfloat16)
                        keys, values = [], []
                        for i,n in enumerate(klens):
                            kv = torch.randn(n,4,512,device='cuda',dtype=cache.dtype)
                            pos = torch.arange(n,device='cuda')
                            cache[table[i,pos//page].long(),:,pos%page] = kv
                            keys.append(kv[...,:256]); values.append(kv[...,256:])
                        k = cache[...,:256].permute(0,2,1,3)
                        v = cache[...,256:].permute(0,2,1,3)
                        q = torch.randn(sum(lengths),24,256,device='cuda',dtype=cache.dtype)
                        kwargs = dict(request_lengths=lengths,kv_lengths=klens,
                                      prompt_lengths=klens,head_layout=heads)
                        paged = BFLAPrefillPlan(q,k,v,256**-.5,kv_page_table=table,**kwargs)
                        actual = paged.run(q,k,v).clone()
                        pk,pv = torch.cat(keys),torch.cat(values)
                        packed = BFLAPrefillPlan(q,pk,pv,256**-.5,**kwargs)
                        expected = packed.run(q,pk,pv)
                        torch.testing.assert_close(paged.ids,packed.ids,atol=0,rtol=0)
                        torch.testing.assert_close(paged.csr,packed.csr,atol=0,rtol=0)
                        torch.testing.assert_close(actual,expected,atol=.002,rtol=.01)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            paged.run_attention(q,k,v)
                        graph.replay(); torch.cuda.synchronize()
                        torch.testing.assert_close(paged.out,actual,atol=0,rtol=0)
                        del graph,paged,packed


if __name__ == '__main__': unittest.main()
