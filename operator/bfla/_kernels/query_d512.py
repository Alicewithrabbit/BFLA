"""D512/DI128 fusion preserving RMS, head mean, interval reduction and FWHT."""
import triton as tr
import triton.language as tl


@tr.jit
def fused_query_features(Q,Out,H:tl.constexpr,BH:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    h=tl.arange(0,BH);d=tl.arange(0,512)
    q=tl.load(Q+(row*H+h[:,None])*512+d[None,:],h[:,None]<H,0).to(tl.float32)
    norm=tl.rsqrt(tl.sum(q*q,1)/512+1.e-6)
    mean=tl.sum(q*norm[:,None],0)/H
    # Four adjacent dimensions, scaled by 1/sqrt(4), not an ordinary mean.
    reduced=tl.sum(tl.reshape(mean,(128,4)),1)*.5
    transformed=reduced*tl.rsqrt(tl.sum(reduced*reduced,0)/128+1.e-6)
    index=tl.arange(0,128)
    for bit in tl.static_range(0,7):
        pair=tl.gather(transformed,index^(1<<bit),0)
        transformed=tl.where((index&(1<<bit))==0,transformed+pair,pair-transformed)
    tl.store(Out+row*128+index,transformed*(128**-.5))


def query_features(q,out):
    if q.ndim!=3 or q.shape[-1]!=512 or not q.is_contiguous():
        raise ValueError('contiguous D512 Q required')
    fused_query_features[(q.shape[0],)](q,out,q.shape[1],tr.next_power_of_2(q.shape[1]),
                                      num_warps=1,enable_fp_fusion=False)
