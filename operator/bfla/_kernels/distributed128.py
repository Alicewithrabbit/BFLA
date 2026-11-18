"""Reduce DI128 partial head sums before nonlinear normalization/FWHT.

No full Q/K activation all-gather. The caller supplies the all-reduce function
and owns its execution lifetime.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partial(X,Out,H:tl.constexpr,BH:tl.constexpr,D:tl.constexpr,
             RS:tl.constexpr,HS:tl.constexpr,GLOBAL_H:tl.constexpr,WEIGHT:tl.constexpr,
             RowIds,INDEXED:tl.constexpr=False):
    row=tl.program_id(0).to(tl.int64)
    source=tl.load(RowIds+row).to(tl.int64) if INDEXED else row
    h=tl.arange(0,BH);d=tl.arange(0,D)
    x=tl.load(X+source*RS+h[:,None]*HS+d[None,:],h[:,None]<H,0).to(tl.float32)
    norm=tl.rsqrt(tl.sum(x*x,1)/D+1.e-6)
    summed=tl.sum(x*norm[:,None],0)
    compact=tl.sum(tl.reshape(summed,(128,D//128)),1)
    compact=compact*(WEIGHT/GLOBAL_H)/(D//128)**.5
    tl.store(Out+row*128+tl.arange(0,128),compact)


@tr.jit
def _finish(X,Out):
    row=tl.program_id(0).to(tl.int64);d=tl.arange(0,128)
    y=tl.load(X+row*128+d)
    y=y*tl.rsqrt(tl.sum(y*y,0)/128+1.e-6)
    for bit in tl.static_range(0,7):
        pair=tl.gather(y,d^(1<<bit),0)
        y=tl.where((d&(1<<bit))==0,y+pair,pair-y)
    tl.store(Out+row*128+d,y*(128**-.5))


class DistributedFeatures:
    def __init__(self,layout,key_rows,query_rows,all_reduce,*,device=None):
        if key_rows<1 or query_rows<1:raise ValueError('positive feature capacities required')
        self.layout=layout;self.all_reduce=all_reduce
        options=dict(device=device or torch.device('cuda',torch.cuda.current_device()),dtype=torch.float32)
        self.key_partial=torch.empty((key_rows,128),**options)
        self.query_partial=torch.empty((query_rows,128),**options)

    def _run(self,x,out,partial,heads,global_heads,weight,row_ids=None):
        n=x.shape[0] if row_ids is None else row_ids.numel()
        if (x.ndim!=3 or x.shape[1:]!=(heads,self.layout.head_dim) or
                n<1 or n>partial.shape[0] or out.shape[0]<n or out.shape[1]!=128):
            raise ValueError('TP feature input/output differs from reserved capacity or geometry')
        active=partial[:n]
        _partial[(n,)](x,active,heads,tr.next_power_of_2(heads),self.layout.head_dim,
                       x.stride(0),x.stride(1),global_heads,weight,row_ids,INDEXED=row_ids is not None,
                       num_warps=1,enable_fp_fusion=False)
        reduced=self.all_reduce(active) if self.layout.tp_size>1 else active
        _finish[(n,)](reduced,out,num_warps=1,enable_fp_fusion=False)
        return out

    def query(self,q,out):
        return self._run(q,out,self.query_partial,self.layout.local_hq,
                         self.layout.global_hq,1.)

    def key(self,k,out):
        return self._run(k,out,self.key_partial,self.layout.local_hkv,
                         self.layout.global_hkv,self.layout.key_weight)

    def query_indexed(self,q,row_ids,out):
        return self._run(q,out,self.query_partial,self.layout.local_hq,
                         self.layout.global_hq,1.,row_ids)
