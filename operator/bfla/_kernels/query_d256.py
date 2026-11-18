"""D256/DI128 fusion: identical RMS, head mean, adjacent-pair reduction and FWHT."""
import triton as tr
import triton.language as tl


@tr.jit
def fused_query_features(Q,Out,H:tl.constexpr,BH:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    h=tl.arange(0,BH);d=tl.arange(0,256)
    q=tl.load(Q+(row*H+h[:,None])*256+d[None,:],h[:,None]<H,0).to(tl.float32)
    norm=tl.rsqrt(tl.sum(q*q,1)/256+1.e-6)
    mean=tl.sum(q*norm[:,None],0)/H
    # D=256 and DI=128 partition the dimension into exactly 128 disjoint pairs.
    reduced=tl.sum(tl.reshape(mean,(128,2)),1)/tl.sqrt(2.)
    transformed=reduced*tl.rsqrt(tl.sum(reduced*reduced,0)/128+1.e-6)
    index=tl.arange(0,128)
    for bit in tl.static_range(0,7):
        pair=tl.gather(transformed,index^(1<<bit),0)
        transformed=tl.where((index&(1<<bit))==0,transformed+pair,pair-transformed)
    tl.store(Out+row*128+index,transformed*(128**-.5))


def query_features(q,out):
    if q.ndim!=3 or q.shape[-1]!=256 or not q.is_contiguous():
        raise ValueError('contiguous D256 Q required')
    fused_query_features[(q.shape[0],)](q,out,q.shape[1],tr.next_power_of_2(q.shape[1]),
                                      num_warps=1,enable_fp_fusion=False)
