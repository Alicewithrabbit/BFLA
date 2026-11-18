"""Host orchestration tests use real CPU tensors and recording device kernels."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from bfla import BFLAConfig, HeadLayout


class Launch:
    def __init__(self): self.calls = []
    def __getitem__(self, grid):
        def run(*args, **kwargs): self.calls.append((grid, args, kwargs))
        return run


class PreparedTests(unittest.TestCase):
    def setUp(self):
        from bfla import prepared
        self.prepared = prepared
        self.attn = SimpleNamespace(calls=[])
        self.attn.forward = lambda x, z, scale, stream: self.attn.calls.append((x,z,scale,stream))
        self.sel = SimpleNamespace(workspace_bytes=lambda *a: 1, topk=lambda *a: None)
        self.features = SimpleNamespace(key=lambda *a: None, query_batch=lambda *a: None,
                                        query_indexed=lambda *a: None)
        self.identity, self.score, self.publish = Launch(), Launch(), Launch()
        self.modules = SimpleNamespace(identity=self.identity, score_rows=self.score, score_index=self.score,
            publish_rows=self.publish, publish_groups=self.publish,
            next_power_of_2=lambda n: 1 << (n-1).bit_length(), cdiv=lambda a,b:(a+b-1)//b)
        for p in (patch.object(prepared, '_device_capability', return_value=(9,0)),
                  patch.object(prepared, '_load_kernels', return_value=self.modules),
                  patch.object(prepared, 'make_features', return_value=self.features),
                  patch.object(prepared, 'pool_features'),
                  patch.object(prepared.build, 'attention', return_value=self.attn),
                  patch.object(prepared.build, 'paged_attention', return_value=self.attn),
                  patch.object(prepared.build, 'selector', return_value=self.sel),
                  patch.object(torch.cuda, 'device', side_effect=lambda *a:nullcontext()),
                  patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(multi_processor_count=132)),
                  patch.object(torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=0))):
            p.start(); self.addCleanup(p.stop)

    def tensors(self, lengths, heads, klens=None, padded=False):
        klens = klens or lengths
        n = sum((k+63)//64*64 if padded else k for k in klens)
        return (torch.empty((sum(lengths),heads.local_hq,heads.head_dim),dtype=torch.bfloat16),
                torch.empty((n,heads.local_hkv,heads.head_dim),dtype=torch.bfloat16),
                torch.empty((n,heads.local_hkv,heads.head_dim),dtype=torch.bfloat16))

    def test_two_requests_separate_padded_pages_direct_output(self):
        h=HeadLayout.create(24,4,256); lengths=(2090,4100)
        q,k,v=self.tensors(lengths,h)
        p=self.prepared.BFLAPrefillPlan(q,k,v,.0625,request_lengths=lengths,head_layout=h)
        out=torch.empty_like(q)
        self.assertIs(p.run(q,k,v,out=out),out)
        pointers, z, _, _=self.attn.calls[-1]
        self.assertEqual(pointers[3],out.data_ptr())
        self.assertEqual(z[0],sum(lengths))
        self.assertEqual(z[8],sum((n+63)//64 for n in lengths))
        self.assertEqual(p.last_key_refresh_rows,lengths)
        self.assertEqual(self.publish.calls[0][0][1],4)

    def test_prefix_and_tp_use_global_selection_local_consumer(self):
        h=HeadLayout.create(24,2,256,tp_size=2,tp_rank=0)
        q,k,v=self.tensors((1024,),h,(8192,),True)
        p=self.prepared.BFLAPrefillPlan(q,k,v,.0625,request_lengths=(1024,),kv_lengths=(8192,),
            prompt_lengths=(16384,),kv_padded=True,head_layout=h,all_reduce=lambda x:x)
        p.run(q,k,v)
        self.assertEqual(p.last_key_refresh_rows,(8192,))
        self.assertEqual(self.score.calls[0][1][5],12)
        self.assertEqual(self.attn.calls[-1][1][3:5],[12,1])
        self.prepared.build.selector.assert_called_once_with(p.capacity,candidates=p.nb)

    def test_partial_caller_padding_passes_real_block64_kv_strides(self):
        h = HeadLayout.create(24,4,256)
        q = torch.empty((9,24,256),dtype=torch.bfloat16)
        k = torch.empty((192,4,512),dtype=q.dtype)[...,:256]
        v = torch.empty((192,12,256),dtype=q.dtype)[:,1::3]
        plan = self.prepared.BFLAPrefillPlan(q,k,v,.0625,request_lengths=(4,5),
            kv_lengths=(65,31),kv_padded=True,head_layout=h)
        plan.run_attention(q,k,v)
        pointers, dims, _, _ = self.attn.calls[-1]
        self.assertEqual(pointers[1:3],[k.data_ptr(),v.data_ptr()])
        self.assertEqual(dims[10:],[64,6144,256,2048,512,131072,3072,768,196608])
        self.assertEqual(plan.table.tolist(),[[0,1],[2,-1]])
        self.prepared.build.paged_attention.assert_called_once_with(256)

    def test_no_sparse_query_does_not_build_indexer(self):
        h=HeadLayout.create(8,2,512)
        q,k,v=self.tensors((128,),h)
        p=self.prepared.BFLAPrefillPlan(q,k,v,.04,request_lengths=(128,),head_layout=h)
        p.run(q,k,v)
        self.prepared.make_features.assert_not_called()
        self.prepared.build.selector.assert_not_called()
        self.assertEqual(p.last_key_refresh_rows,(0,))

    def test_shape_change_rejected_before_pointer_launch(self):
        h=HeadLayout.create(8,2,512)
        q,k,v=self.tensors((128,),h)
        p=self.prepared.BFLAPrefillPlan(q,k,v,.04,request_lengths=(128,),head_layout=h)
        with self.assertRaises(ValueError): p.run(q[:-1],k,v)
        with self.assertRaises(ValueError): p.run(q,k,v,out=q)
        self.assertEqual(self.attn.calls,[])

    def test_rebound_storage_must_keep_tma_alignment(self):
        h=HeadLayout.create(8,2,512)
        q,k,v=self.tensors((128,),h)
        p=self.prepared.BFLAPrefillPlan(q,k,v,.04,request_lengths=(128,),head_layout=h)
        unaligned=torch.empty(q.numel()+1,dtype=q.dtype)[1:].view_as(q)
        with self.assertRaises(ValueError): p.run(unaligned,k,v)
        with self.assertRaises(ValueError): p.run(q,k,v,out=unaligned)

    def test_arena_binding_preserves_logical_support_and_handles_reorder(self):
        h=HeadLayout.create(8,2,512); lengths=(128,256); klens=(4096,8192)
        q,k,v=self.tensors(lengths,h,klens,True)
        base=self.prepared.BFLAPrefillPlan(q,k,v,.04,request_lengths=lengths,
            kv_lengths=klens,prompt_lengths=(16384,16384),kv_padded=True,head_layout=h)
        arena_k=torch.empty((24576,2,512),dtype=k.dtype); arena_v=torch.empty_like(arena_k)
        p=self.prepared.BFLAPrefillPlan(q,arena_k,arena_v,.04,request_lengths=lengths,
            kv_lengths=klens,prompt_lengths=(16384,16384),kv_padded=True,
            kv_page_offsets=(256,0),head_layout=h)
        self.assertEqual(p.spec.groups,base.spec.groups)
        self.assertEqual(p.spec.counts,base.spec.counts)
        self.assertEqual(p.key_starts,[16384,0])
        self.assertEqual(p.table[0,0].item(),256)
        self.assertEqual(p.table[-1,0].item(),0)
        p.run(q,arena_k,arena_v)
        self.assertEqual(self.attn.calls[-1][1][8],384)
        with self.assertRaises(ValueError): p.bind_kv(arena_k,arena_v,(0,32))
        with self.assertRaises(ValueError): p.bind_kv(arena_k,arena_v,(380,0))

    def test_prepared_keys_not_refreshed_by_selection_or_attention(self):
        h=HeadLayout.create(24,4,256)
        q,k,v=self.tensors((128,),h,(8192,),True)
        p=self.prepared.BFLAPrefillPlan(q,k,v,.0625,request_lengths=(128,),
            kv_lengths=(8192,),prompt_lengths=(16384,),kv_padded=True,
            kv_page_offsets=(0,),head_layout=h)
        refresh=SimpleNamespace(config=p.config,calls=0)
        def populate(*args): refresh.calls+=1; return 64
        refresh.populate=populate
        p.prepare_keys(k,key_states=(refresh,))
        for _ in range(4):
            p.run_selection(q)
            p.run_attention(q,k,v)
        self.assertEqual(refresh.calls,1)
        self.assertEqual(p.last_key_refresh_rows,(64,))
        self.assertEqual(len(self.attn.calls),4)

    def test_native_pages_keep_supports_and_map_each_wave_to_its_request(self):
        h=HeadLayout.create(8,2,512); lengths=(256,128); klens=(4097,3500)
        q,k,v=self.tensors(lengths,h,klens,True)
        base=self.prepared.BFLAPrefillPlan(q,k,v,.04,request_lengths=lengths,
            kv_lengths=klens,prompt_lengths=(8192,8192),kv_padded=True,head_layout=h)
        for page in (16,64,784):
            width=(max(klens)+page-1)//page
            cache=torch.empty((2*width,2,page,1024),dtype=k.dtype)
            nk,nv=cache.transpose(1,2).split(512,-1)
            table=torch.stack((torch.arange(width-1,-1,-1),torch.arange(width,2*width))).int()
            with patch.object(self.prepared.build,'paged_attention',return_value=self.attn):
                plan=self.prepared.BFLAPrefillPlan(q,nk,nv,.04,request_lengths=lengths,
                    kv_lengths=klens,prompt_lengths=(8192,8192),head_layout=h,kv_page_table=table)
            self.assertIsNone(plan.layout['table'])
            self.assertEqual(plan.spec.groups,base.spec.groups)
            self.assertEqual(plan.spec.counts,base.spec.counts)
            self.assertIsNone(plan.kpad)
            self.assertLessEqual(plan.key_scratch.shape[0],16384)
            for i,r in enumerate(plan.layout['segment_requests']):
                torch.testing.assert_close(plan.table[i],table[r])
            plan.run_attention(q,nk,nv)
            self.assertEqual(self.attn.calls[-1][1][10],page)
            self.assertEqual(self.attn.calls[-1][1][13:16],list((nk.stride(1),nk.stride(2),nk.stride(0))))
            replacement=table.flip(0).contiguous()
            plan.bind_paged_kv(nk,nv,replacement)
            for i,r in enumerate(plan.layout['segment_requests']):
                torch.testing.assert_close(plan.table[i],replacement[r])

    def test_paged_rebinding_enforces_the_same_tma_stride_limit_as_prepare(self):
        h=HeadLayout.create(24,4,256)
        q=torch.empty((8,24,256),dtype=torch.bfloat16)
        cache=torch.empty((1,4,16,512),dtype=q.dtype)
        k,v=cache.transpose(1,2).split(256,-1)
        table=torch.zeros((1,1),dtype=torch.int32)
        with patch.object(self.prepared.build,'paged_attention',return_value=self.attn):
            p=self.prepared.BFLAPrefillPlan(q,k,v,.0625,request_lengths=(8,),
                head_layout=h,kv_page_table=table)
        # Size-one page dimension lets us test the descriptor's byte-stride
        # limit without allocating huge storage or launching a device kernel.
        invalid=k.as_strided(k.shape,(2**39,*k.stride()[1:]))
        with self.assertRaises(ValueError):
            self.prepared.BFLAPrefillPlan(q,invalid,v,.0625,request_lengths=(8,),
                head_layout=h,kv_page_table=table)
        with self.assertRaises(ValueError):
            p.bind_paged_kv(invalid,v,table)


if __name__=='__main__': unittest.main()
