"""Initialize full-support block lists; sparse lists are replaced each run."""
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['TILES'])
def identity(CSR,Ids,TILES,BLOCK:tl.constexpr):
    tile=tl.program_id(0);h=tl.program_id(1);i=tl.arange(0,BLOCK)
    a=tl.load(CSR+h*TILES+tile);b=tl.load(CSR+h*TILES+tile+1)
    tl.store(Ids+a+i,i,i<b-a)
