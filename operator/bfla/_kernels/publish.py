"""Expand selected logical blocks into block64 consumer CSR lists."""
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['TILES','CAP'])
def publish_groups(Selected,Lengths,Maps,CSR,Ids,TILES,CAP,
                   FACTOR:tl.constexpr,SHARE:tl.constexpr,SORT:tl.constexpr):
    row=tl.program_id(0); head=tl.program_id(1); i=tl.arange(0,SORT)
    count=tl.load(Lengths+row)
    selected=tl.load(Selected+row*CAP+i,(i<CAP)&(i<count),0x7fffffff)
    selected=tl.sort(selected,descending=False)
    tile=tl.load(Maps+row*3); nt=tl.load(Maps+row*3+1); end=tl.load(Maps+row*3+2)
    child=tl.arange(0,FACTOR)
    # Use int64 to avoid overflow when multiplying inactive INT_MAX sentinels.
    physical=selected[:,None].to(tl.int64)*FACTOR+child[None,:]
    position=i[:,None]*FACTOR+child[None,:]
    valid=(i[:,None]<count)&(physical<end)
    for j in tl.static_range(SHARE):
        if j<nt:
            a=tl.load(CSR+head*TILES+tile+j)
            tl.store(Ids+a+position,physical,valid)
