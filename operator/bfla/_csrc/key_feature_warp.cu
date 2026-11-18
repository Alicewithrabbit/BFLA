// D256/Hkv4/DI128 specialization of the existing deterministic feature map.
// A warp owns one token; four warps replace four block-synchronized CTAs.
#include <pybind11/pybind11.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace {
__device__ __forceinline__ float sum128(float a, float b, float c, float d) {
  // Match the original 128-thread reduction: offsets 64,32,16,8,4,2,1.
  a += c; b += d; a += b;
  #pragma unroll
  for (int delta=16; delta; delta>>=1)
    a += __shfl_down_sync(0xffffffff, a, delta);
  return __shfl_sync(0xffffffff, a, 0);
}

__global__ void key_feature_d256_warp(
    const __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ out,
    int64_t n, int64_t stride_token, int64_t stride_head, float index_width) {
  const int lane=threadIdx.x&31;
  const int64_t token=static_cast<int64_t>(blockIdx.x)*4+(threadIdx.x>>5);
  if (token>=n) return;
  float feature[4]={0,0,0,0};
  #pragma unroll
  for (int h=0; h<4; ++h) {
    const auto* head=key+token*stride_token+h*stride_head;
    float square[4];
    #pragma unroll
    for (int j=0; j<4; ++j) {
      const float a=__bfloat162float(head[lane+j*32]);
      const float b=__bfloat162float(head[lane+j*32+128]);
      square[j]=a*a+b*b;
    }
    const float inverse=rsqrtf(sum128(square[0],square[1],square[2],square[3])/256.0f+1.e-6f);
    #pragma unroll
    for (int j=0; j<4; ++j) {
      const int dim=2*(lane+j*32);
      const float sum=__bfloat162float(head[dim])+__bfloat162float(head[dim+1]);
      // The generic kernel divides by a runtime denominator. Prevent NVCC
      // from replacing this constant denominator with a rounded reciprocal.
      feature[j] += __fdiv_rn(__fmul_rn(sum,inverse),4.0f*sqrtf(2.0f));
    }
  }
  // The reference stores these products to shared memory before reduction.
  // Keep that rounding point: fusing a square into the first reduction add
  // can move a value across a BF16 rounding boundary.
  const float inverse=rsqrtf(sum128(__fmul_rn(feature[0],feature[0]),__fmul_rn(feature[1],feature[1]),
                                  __fmul_rn(feature[2],feature[2]),__fmul_rn(feature[3],feature[3]))/128.0f+1.e-6f);
  #pragma unroll
  for (int j=0; j<4; ++j) feature[j]*=inverse;
  #pragma unroll
  for (int bit=1; bit<32; bit<<=1) {
    #pragma unroll
    for (int j=0; j<4; ++j) {
      const float pair=__shfl_xor_sync(0xffffffff,feature[j],bit);
      feature[j]=(lane&bit)?pair-feature[j]:feature[j]+pair;
    }
  }
  float a=feature[0],b=feature[1],c=feature[2],d=feature[3];
  feature[0]=a+b;feature[1]=a-b;feature[2]=c+d;feature[3]=c-d;
  a=feature[0];b=feature[1];c=feature[2];d=feature[3];
  feature[0]=a+c;feature[1]=b+d;feature[2]=a-c;feature[3]=b-d;
  // The reference evaluates rsqrt at runtime. NVCC folds rsqrtf(128.f)
  // to 0x3db504f3, while SM90 MUFU.RSQ produces 0x3db504f2. Preserve
  // the reference operation rather than moving BF16 rounding boundaries.
  const float hadamard_scale=rsqrtf(index_width);
  #pragma unroll
  for (int j=0; j<4; ++j)
    out[token*128+lane+j*32]=__float2bfloat16_rn(feature[j]*hadamard_scale);
}
}

void forward(uint64_t key, uint64_t out, int64_t n, int64_t stride_token,
             int64_t stride_head, uint64_t stream) {
  if (n==0) return;
  key_feature_d256_warp<<<(n+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(key),reinterpret_cast<__nv_bfloat16*>(out),
      n,stride_token,stride_head,128.0f);
  const auto error=cudaGetLastError();
  if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&forward);}
