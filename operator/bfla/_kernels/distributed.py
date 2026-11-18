"""Distributed DI64/256 features with a dedicated DI128 implementation."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def partial(X,Out,H:tl.constexpr,BH:tl.constexpr,D:tl.constexpr,DI:tl.constexpr,
            RS:tl.constexpr,HS:tl.constexpr,GLOBAL_H:tl.constexpr,WEIGHT:tl.constexpr,
            RowIds,INDEXED:tl.constexpr=False):
    row=tl.program_id(0).to(tl.int64)
    source=tl.load(RowIds+row).to(tl.int64) if INDEXED else row
    h=tl.arange(0,BH); d=tl.arange(0,D)
    x=tl.load(X+source*RS+h[:,None]*HS+d[None,:],h[:,None]<H,0).to(tl.float32)
    norm=tl.rsqrt(tl.sum(x*x,1)/D+1.e-6)
    summed=tl.sum(x*norm[:,None],0)
    compact=tl.sum(tl.reshape(summed,(DI,D//DI)),1)
    compact=compact*(WEIGHT/GLOBAL_H)/(D//DI)**.5
    tl.store(Out+row*DI+tl.arange(0,DI),compact)


@tr.jit
def finish(X,Out,DI:tl.constexpr,LOG_DI:tl.constexpr):
    row=tl.program_id(0).to(tl.int64); d=tl.arange(0,DI)
    y=tl.load(X+row*DI+d)
    y=y*tl.rsqrt(tl.sum(y*y,0)/DI+1.e-6)
    for bit in tl.static_range(0,LOG_DI):
        pair=tl.gather(y,d^(1<<bit),0)
        y=tl.where((d&(1<<bit))==0,y+pair,pair-y)
    tl.store(Out+row*DI+d,y*(DI**-.5))


class DistributedFeatures:
    def __init__(self,layout,key_rows,query_rows,all_reduce,*,index_dim,device):
        if index_dim not in (64,256) or layout.head_dim not in (256,512):
            raise ValueError('DI64/256 features require native D256/D512')
        self.layout=layout; self.all_reduce=all_reduce; self.di=index_dim
        self.key_partial=torch.empty((key_rows,index_dim),device=device,dtype=torch.float32)
        self.query_partial=torch.empty((query_rows,index_dim),device=device,dtype=torch.float32)

    def _run(self,x,out,scratch,heads,global_heads,weight,row_ids=None):
        n=x.shape[0] if row_ids is None else row_ids.numel()
        if (x.ndim!=3 or x.shape[1:]!=(heads,self.layout.head_dim) or n<1 or n>scratch.shape[0] or
            out.shape[0]<n or out.shape[1]!=self.di):
            raise ValueError('TP feature geometry differs from reserved capacity')
        active=scratch[:n]
        partial[(n,)](x,active,heads,tr.next_power_of_2(heads),self.layout.head_dim,self.di,
                      x.stride(0),x.stride(1),global_heads,weight,row_ids,
                      INDEXED=row_ids is not None,num_warps=1,enable_fp_fusion=False)
        reduced=self.all_reduce(active)
        finish[(n,)](reduced,out,self.di,self.di.bit_length()-1,num_warps=1,enable_fp_fusion=False)
        return out

    def query(self,q,out):
        return self._run(q,out,self.query_partial,self.layout.local_hq,self.layout.global_hq,1.)

    def key(self,k,out):
        return self._run(k,out,self.key_partial,self.layout.local_hkv,self.layout.global_hkv,
                         self.layout.key_weight)

    def query_indexed(self,q,row_ids,out):
        return self._run(q,out,self.query_partial,self.layout.local_hq,self.layout.global_hq,1.,row_ids)


def make_features(layout,key_rows,query_rows,all_reduce,config,device):
    if config.index_dim==128:
        from .distributed128 import DistributedFeatures as FrozenFeatures
        return FrozenFeatures(layout,key_rows,query_rows,all_reduce,device=device)
    return DistributedFeatures(layout,key_rows,query_rows,all_reduce,index_dim=config.index_dim,device=device)
