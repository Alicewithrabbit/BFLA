// D512 consumer: one QK/softmax stream feeds two cooperating V partitions.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "flash_fwd_launch_template.h"
#ifndef BFLA_BATCHED_TMA
#define BFLA_BATCHED_TMA 1
#endif

void batched_d512_forward(std::vector<uint64_t> const& x, std::vector<int64_t> const& z,
             double scale, uint64_t stream_ptr) {
    if (x.size()!=12 || z.size()!=10 || z[7]<1 ||
        !((z[3]==8 && z[4]==2) || (z[3]==4 && z[4]==1) || (z[3]==2 && z[4]==1)))
        throw std::runtime_error("D512 contiguous varlen ABI requires local (Hq,Hkv) of (8,2), (4,1), or (2,1)");
    int rows=z[0], max_rows=z[1], max_k=z[2], hq=z[3], hk=z[4], tiles=z[5];
    int segments=z[7], pages=z[8], table_width=z[9];
    Flash_fwd_params p{};
    p.q_ptr=reinterpret_cast<void*>(x[0]); p.k_ptr=reinterpret_cast<void*>(x[1]);
    p.softmax_lse_ptr=reinterpret_cast<void*>(x[4]);
    p.page_table=reinterpret_cast<int*>(x[5]);
    p.block_sparse_cu=reinterpret_cast<int*>(x[6]); p.block_sparse_idx=reinterpret_cast<int*>(x[7]);
    p.seqused_k=reinterpret_cast<int*>(x[8]); p.cu_q_tiles=reinterpret_cast<int*>(x[9]);
    p.cu_seqlens_q=reinterpret_cast<int*>(x[11]);
    p.b=p.b_k=segments; p.h=hq; p.h_k=hk; p.d=p.d_rounded=512; p.dv=p.dv_rounded=512;
    p.seqlen_q=max_rows; p.seqlen_k=table_width*64; p.total_q=rows; p.total_k=pages*64;
    p.seqlen_q_rounded=((max_rows+127)/128)*128; p.seqlen_k_rounded=p.seqlen_k;
    p.q_row_stride=p.o_row_stride=int64_t(hq)*512; p.q_head_stride=p.o_head_stride=512;
    p.q_batch_stride=p.o_batch_stride=0;
    p.k_row_stride=p.v_row_stride=int64_t(hk)*512;
    p.k_batch_stride=p.v_batch_stride=int64_t(64)*hk*512;
    p.k_head_stride=p.v_head_stride=512; p.v_dim_stride=1;
    p.page_size=64; p.num_pages=pages; p.page_table_batch_stride=table_width;
    p.has_block_sparse=true; p.total_q_tiles=tiles; p.mean_k_block_size=0; p.mean_n_sub=2;
    p.scale_softmax=float(scale); p.is_bf16=true; p.is_causal=true; p.pack_gqa=true;
    p.window_size_left=max_k; p.window_size_right=0; p.num_splits=1; p.arch=90; p.num_sm=z[6];
    p.pagedkv_tma=bool(BFLA_BATCHED_TMA);
    int stride=(segments+3)/4*4;
    auto meta=reinterpret_cast<int*>(x[10]);
    p.num_splits_dynamic_ptr=meta; p.num_m_blocks_ptr=meta+stride;
    p.varlen_batch_idx_ptr=meta+2*stride; p.num_nheads_in_l2_ptr=meta+3*stride;
    p.tile_count_semaphore=meta+4*stride; p.tile_count_semaphore_offset=4*stride;
    p.varlen_sort_batches=true; p.head_swizzle=true;
    p.skip_scheduler_metadata_computation=false; p.prepare_varlen_pdl=false;
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    p.v_ptr=reinterpret_cast<void*>(x[2]);
    p.o_ptr=reinterpret_cast<void*>(x[3]);
    run_flash_fwd<90,512,512,1,cutlass::bfloat16_t,cutlass::bfloat16_t,
        true,false,false,true,!bool(BFLA_BATCHED_TMA),false,false,true,false,false,true,false>(p,stream);
    CHECK_CUDA(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("forward", &batched_d512_forward); }
