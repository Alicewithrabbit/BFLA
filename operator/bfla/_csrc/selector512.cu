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

using Id = thrust::counting_iterator<int>;
inline void checked(cudaError_t e) {
    if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct View {
    float const** in; float** out; Id* ids; int** selected;
    int* sizes; int* ks; void* temp;
};
size_t align256(size_t n) { return (n+255)/256*256; }
size_t header_bytes(int rows) { return 6*align256(size_t(rows)*8); }
View view(uint8_t* p, int r) {
    size_t a=align256(size_t(r)*8);
    return {reinterpret_cast<float const**>(p),reinterpret_cast<float**>(p+a),
        reinterpret_cast<Id*>(p+2*a),reinterpret_cast<int**>(p+3*a),
        reinterpret_cast<int*>(p+4*a),reinterpret_cast<int*>(p+5*a),p+6*a};
}
__global__ void prepare(float const* scores,float* values,int* selected,
    int const* sizes,int const* lengths,int rows,int stride,int cap,View w) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows)return;
    w.in[r]=scores+int64_t(r)*stride; w.out[r]=values+int64_t(r)*cap;
    w.ids[r]=Id(0); w.selected[r]=selected+int64_t(r)*cap;
    w.sizes[r]=sizes[r]; w.ks[r]=min(cap,min(lengths[r],sizes[r]));
}
template<int Bound>
cudaError_t invoke(View w,void* temp,size_t& bytes,int rows,cudaStream_t stream) {
    auto sizes=cuda::args::deferred_sequence{w.sizes,cuda::args::bounds<int{0},Bound>()};
    auto ks=cuda::args::deferred_sequence{w.ks,cuda::args::bounds<int{0},int{512}>()};
    auto requirements=cuda::execution::require(cuda::execution::determinism::gpu_to_gpu,
        cuda::execution::tie_break::prefer_smaller_index,cuda::execution::output_ordering::unsorted);
    auto env=cuda::std::execution::env{requirements,cuda::stream_ref{stream}};
    return cub::DeviceBatchedTopK::MaxPairs(temp,bytes,w.in,w.out,w.ids,w.selected,sizes,ks,rows,env);
}
cudaError_t dispatch(View w,void* temp,size_t& bytes,int rows,int n,cudaStream_t stream) {
    if(n<=2048)return invoke<2048>(w,temp,bytes,rows,stream);
    if(n<=8192)return invoke<8192>(w,temp,bytes,rows,stream);
    throw std::runtime_error("block candidate capacity exceeds 8192 (512K tokens)");
}
size_t workspace_bytes(int rows,int n) {
    size_t bytes=0; View empty{};
    checked(dispatch(empty,nullptr,bytes,rows,n,nullptr));
    return header_bytes(rows)+align256(bytes)+256;
}
void topk(uint64_t scores,uint64_t values,uint64_t selected,uint64_t sizes,uint64_t lengths,
          uint64_t workspace,size_t bytes,int rows,int n,int cap,uint64_t stream_ptr) {
    if(rows<=0 || n<=0 || cap<=0 || cap>512)throw std::runtime_error("invalid TopK geometry");
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    auto w=view(reinterpret_cast<uint8_t*>(workspace),rows);
    prepare<<<(rows+127)/128,128,0,stream>>>(reinterpret_cast<float const*>(scores),
        reinterpret_cast<float*>(values),reinterpret_cast<int*>(selected),
        reinterpret_cast<int const*>(sizes),reinterpret_cast<int const*>(lengths),rows,n,cap,w);
    size_t temp_bytes=bytes-header_bytes(rows);
    checked(dispatch(w,w.temp,temp_bytes,rows,n,stream));
    checked(cudaGetLastError());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("workspace_bytes",&workspace_bytes);m.def("topk",&topk);
}
