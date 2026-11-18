"""Bounded K-only reads for index construction from caller-owned paged KV."""
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['START','N'])
def _read(K,Table,Out,START,N,H:tl.constexpr,D:tl.constexpr,PAGE:tl.constexpr,
          PS:tl.constexpr,TS:tl.constexpr,HS:tl.constexpr,BLOCK:tl.constexpr=1024):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    token=START+i//(H*D)
    physical=tl.load(Table+token//PAGE,i<N*H*D,0).to(tl.int64)
    offset=physical*PS+(token%PAGE)*TS+(i//D%H)*HS+i%D
    value=tl.load(K+offset,i<N*H*D,0)
    tl.store(Out+i,value,i<N*H*D)


def read_key_window(key,table,start,end,out):
    n=end-start
    _read[(tr.cdiv(n*key.shape[2]*key.shape[3],1024),)](
        key,table,out,start,n,key.shape[2],key.shape[3],key.shape[1],*key.stride()[:3])
    return out[:n]
