# SPDX-FileCopyrightText: 2026 Chong WU
# SPDX-License-Identifier: Apache-2.0

"""Score request-local M128 query tiles and publish selected block IDs.

Each descriptor maps a query group to its request's pooled keys and CSR output.
"""
import triton as tr
import triton.language as tl


# Request dimensions vary between chunks; only compute tile widths specialize.
@tr.jit(do_not_specialize=['STRIDE'])
def score_rows(Q,Pool,Scales,Scores,Rows,G:tl.constexpr,STRIDE,
               PRUNED:tl.constexpr,BN:tl.constexpr=32):
    output=tl.program_id(0)
    qbase=tl.load(Rows+output*6);poolbase=tl.load(Rows+output*6+1)
    nb=tl.load(Rows+output*6+2);start=tl.load(Rows+output*6+3)
    rows=tl.load(Rows+output*6+4);tile=tl.load(Rows+output*6+5)
    first=start+tile*128//G
    last=start+(tl.minimum((tile+1)*128,rows*G)-1)//G
    block_start=tl.program_id(1)*BN
    b=block_start+tl.arange(0,BN)
    mandatory_start=tl.minimum(first//64,tl.maximum(0,(first//128-4+1)*128)//64)
    if block_start>=nb:
        energy=tl.full((BN,),-float('inf'),tl.float32)
    elif PRUNED and block_start>last//64:
        energy=tl.full((BN,),-float('inf'),tl.float32)
    elif PRUNED and ((block_start+BN<=4) | (block_start>=mandatory_start)):
        energy=tl.where(b<=last//64,float('inf'),-float('inf'))
    else:
        packed=tile*128+tl.arange(0,128)
        row=packed//G;valid=packed<rows*G
        d=tl.arange(0,128)
        # Repeated packed head rows share a query feature. Compute its dot
        # product once, then restore the original reduction order below.
        # At least 64 rows keeps SM90 warp-group MMA accumulation; smaller
        # tiles select a different MMA instruction and can perturb score ties.
        BQ:tl.constexpr=max(64,tr.next_power_of_2(tr.cdiv(128,G)+(128%G!=0)))
        first_row=tile*128//G
        unique_row=first_row+tl.arange(0,BQ)
        q=tl.load(Q+(qbase+unique_row[:,None])*128+d[None,:],
                  ((unique_row<rows)&(unique_row<=(tile*128+127)//G))[:,None],0.).to(tl.bfloat16)
        k=tl.load(Pool+(poolbase+b[None,:])*128+d[:,None],b[None,:]<nb,0.).to(tl.bfloat16)
        logits=tl.dot(q,k)
        indices=tl.broadcast_to((row-first_row)[:,None],(128,BN))
        s=tl.maximum(tl.gather(logits,indices,axis=0),0)*(128**-.5)
        s*=tl.load(Scales+poolbase+b,b<nb,0.)[None,:]
        s=tl.where(valid[:,None],s,-float('inf'))
        peak=tl.max(s,0)
        values=tl.exp(s-peak[None,:])
        # Preserve the M128 four-warp reduction: register rows 0/8/64/72,
        # then lane rows 4/2/1, then warp rows 32/16. A plain sum after
        # gather chooses another layout and can change near-equal rankings.
        lane=tl.arange(0,32)
        bases=(lane//8)*16+lane%8
        gather=tl.broadcast_to(bases[:,None],(32,BN))
        mass=tl.gather(values,gather,axis=0)+tl.gather(values,gather+8,axis=0)
        mass=mass+tl.gather(values,gather+64,axis=0)
        mass=mass+tl.gather(values,gather+72,axis=0)
        for shift in tl.static_range(3):
            indices=tl.broadcast_to((lane^(4>>shift))[:,None],(32,BN))
            mass=mass+tl.gather(mass,indices,axis=0)
        for shift in tl.static_range(2):
            indices=tl.broadcast_to((lane^(16>>shift))[:,None],(32,BN))
            mass=mass+tl.gather(mass,indices,axis=0)
        mass=tl.sum(tl.where(lane[:,None]==0,mass,0.),axis=0)
        energy=peak+tl.log(mass)
        energy=tl.where((b<4)|(b>=mandatory_start),float('inf'),energy)
        energy=tl.where(b<=last//64,energy,-float('inf'))
    tl.store(Scores+output.to(tl.int64)*STRIDE+b,energy,b<STRIDE)


@tr.jit(do_not_specialize=['TILES','CAP'])
def publish_rows(Selected,Lengths,Offsets,CSR,Ids,TILES,
                 CAP,BLOCK:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1);i=tl.arange(0,BLOCK)
    count=tl.load(Lengths+row);tile=tl.load(Offsets+row)
    ids=tl.load(Selected+row*CAP+i,(i<CAP)&(i<count),0x7fffffff)
    ids=tl.sort(ids,descending=False)
    a=tl.load(CSR+head*TILES+tile)
    tl.store(Ids+a+i,ids,i<count)
