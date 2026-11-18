// SPDX-FileCopyrightText: 2026 Chong WU
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT
#include <cub/device/device_batched_topk.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/argument>
#include <cuda/std/__execution/env.h>
#include <cstdint>
#include <stdexcept>
using Id=thrust::counting_iterator<int>;
struct View { float const** in; float** out; Id* ids; int** selected; int* sizes; int* ks; void* temp; };
size_t aligned(size_t n){return (n+255)/256*256;}
size_t header(int r){return 6*aligned(size_t(r)*8);}
View view(uint8_t* p,int r){size_t a=aligned(size_t(r)*8);return {
    reinterpret_cast<float const**>(p),reinterpret_cast<float**>(p+a),reinterpret_cast<Id*>(p+2*a),
    reinterpret_cast<int**>(p+3*a),reinterpret_cast<int*>(p+4*a),reinterpret_cast<int*>(p+5*a),p+6*a};}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void prepare(float const* s,float* vals,int* ids,int const* sizes,int const* lengths,
                       int rows,int stride,int cap,View w){
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;
    w.in[r]=s+int64_t(r)*stride;w.out[r]=vals+int64_t(r)*cap;
    w.ids[r]=Id(0);w.selected[r]=ids+int64_t(r)*cap;
    w.sizes[r]=sizes[r];w.ks[r]=min(cap,min(lengths[r],sizes[r]));
}
cudaError_t invoke(View w,void* p,size_t& bytes,int rows,cudaStream_t stream){
    auto sizes=cuda::args::deferred_sequence{w.sizes,cuda::args::bounds<int{0},int{2048}>()};
    auto ks=cuda::args::deferred_sequence{w.ks,cuda::args::bounds<int{0},int{1024}>()};
    auto req=cuda::execution::require(cuda::execution::determinism::gpu_to_gpu,
        cuda::execution::tie_break::prefer_smaller_index,cuda::execution::output_ordering::unsorted);
    auto env=cuda::std::execution::env{req,cuda::stream_ref{stream}};
    return cub::DeviceBatchedTopK::MaxPairs(p,bytes,w.in,w.out,w.ids,w.selected,sizes,ks,rows,env);
}
size_t workspace_bytes(int rows,int n){
    if(rows<1||n<1||n>2048)throw std::runtime_error("128K/block64 bound required");
    size_t bytes=0;View empty{};check(invoke(empty,nullptr,bytes,rows,nullptr));
    return header(rows)+aligned(bytes)+256;
}
void topk(uint64_t s,uint64_t v,uint64_t ids,uint64_t sizes,uint64_t lengths,uint64_t workspace,
          size_t bytes,int rows,int n,int cap,uint64_t stream_ptr){
    if(rows<1||n<1||n>2048||cap<1||cap>1024)throw std::runtime_error("invalid CUB geometry");
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);auto w=view(reinterpret_cast<uint8_t*>(workspace),rows);
    prepare<<<(rows+127)/128,128,0,stream>>>(reinterpret_cast<float const*>(s),reinterpret_cast<float*>(v),
        reinterpret_cast<int*>(ids),reinterpret_cast<int const*>(sizes),reinterpret_cast<int const*>(lengths),rows,n,cap,w);
    size_t available=bytes-header(rows);check(invoke(w,w.temp,available,rows,stream));check(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("workspace_bytes",&workspace_bytes);m.def("topk",&topk);}
