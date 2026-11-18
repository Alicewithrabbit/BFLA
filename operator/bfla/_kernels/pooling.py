# SPDX-FileCopyrightText: 2026 Chong WU
# SPDX-License-Identifier: Apache-2.0

import triton as tr
import triton.language as tl

@tr.jit(do_not_specialize=['N'])
def index_pool(Features,Pool,Scales,N):
    b=tl.program_id(0);t=b*64+tl.arange(0,64);d=tl.arange(0,128)
    x=tl.load(Features+t[:,None]*128+d[None,:],t[:,None]<N,0).to(tl.float32)
    mean=tl.sum(x,0)/tl.minimum(64,N-b*64)
    scale=tl.maximum(tl.max(tl.abs(mean),0),1.e-6)/448.
    tl.store(Pool+b*128+d,(mean/scale).to(tl.float8e4nv))
    tl.store(Scales+b,scale)

