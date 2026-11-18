#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "indexer_key_feature.h"

namespace bfla {
namespace {

constexpr int kThreads = 128;
constexpr int kIndexDim = 128;

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* pointer) {
  return static_cast<float>(*pointer);
}

template <>
__device__ __forceinline__ float load_float(const __half* pointer) {
  return __half2float(*pointer);
}

template <>
__device__ __forceinline__ float load_float(
    const __nv_bfloat16* pointer) {
  return __bfloat162float(*pointer);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_float(float value);

template <>
__device__ __forceinline__ __half store_float(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_float(float value) {
  return __float2bfloat16_rn(value);
}

template <typename scalar_t>
__global__ void indexer_key_feature_kernel(
    const scalar_t* __restrict__ key,
    scalar_t* __restrict__ out,
    int64_t tokens,
    int kv_heads,
    int head_dim,
    int query_heads,
    int index_q_heads,
    float eps,
    int64_t key_stride_token,
    int64_t key_stride_head,
    int64_t key_stride_dim,
    int64_t out_stride_token,
    int64_t out_stride_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t token = row / index_q_heads;
  const int group = static_cast<int>(row - token * index_q_heads);
  if (token >= tokens) {
    return;
  }

  const int width = kIndexDim / index_q_heads;
  const int query_start = group * query_heads / index_q_heads;
  int query_end =
      ((group + 1) * query_heads + index_q_heads - 1) / index_q_heads;
  query_end = max(query_start + 1, query_end);
  const int queries_per_kv = query_heads / kv_heads;
  const int kv_start = query_start / queries_per_kv;
  const int kv_end = (query_end - 1) / queries_per_kv + 1;
  const int kv_count = kv_end - kv_start;

  __shared__ float slice[kIndexDim];
  __shared__ float reduction[kThreads];
  __shared__ float inverse_head_rms;

  slice[threadIdx.x] = 0.0f;
  __syncthreads();

  for (int head = kv_start; head < kv_end; ++head) {
    float local_square_sum = 0.0f;
    const int64_t head_base =
        token * key_stride_token + head * key_stride_head;
    for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
      const float value =
          load_float(key + head_base + dim * key_stride_dim);
      local_square_sum += value * value;
    }
    reduction[threadIdx.x] = local_square_sum;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta >>= 1) {
      if (threadIdx.x < delta) {
        reduction[threadIdx.x] += reduction[threadIdx.x + delta];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      inverse_head_rms =
          rsqrtf(reduction[0] / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    if (threadIdx.x < width) {
      const int output = threadIdx.x;
      const int dim_start = output * head_dim / width;
      int dim_end = ((output + 1) * head_dim + width - 1) / width;
      dim_end = max(dim_start + 1, dim_end);
      float sum = 0.0f;
      for (int dim = dim_start; dim < dim_end; ++dim) {
        sum += load_float(key + head_base + dim * key_stride_dim);
      }
      slice[output] +=
          sum * inverse_head_rms /
          (static_cast<float>(kv_count) *
           sqrtf(static_cast<float>(dim_end - dim_start)));
    }
    __syncthreads();
  }

  const float value =
      threadIdx.x < width ? slice[threadIdx.x] : 0.0f;
  reduction[threadIdx.x] = value * value;
  __syncthreads();
  for (int delta = blockDim.x / 2; delta > 0; delta >>= 1) {
    if (threadIdx.x < delta) {
      reduction[threadIdx.x] += reduction[threadIdx.x + delta];
    }
    __syncthreads();
  }
  if (threadIdx.x < width) {
    slice[threadIdx.x] *=
        rsqrtf(reduction[0] / static_cast<float>(width) + eps);
  }
  __syncthreads();

  for (int butterfly = 1; butterfly < width; butterfly <<= 1) {
    if (threadIdx.x < width / 2) {
      const int pair = (threadIdx.x / butterfly) * 2 * butterfly +
                       threadIdx.x % butterfly;
      const float left = slice[pair];
      const float right = slice[pair + butterfly];
      slice[pair] = left + right;
      slice[pair + butterfly] = left - right;
    }
    __syncthreads();
  }

  if (threadIdx.x < width) {
    const int output_dim = group * width + threadIdx.x;
    out[token * out_stride_token + output_dim * out_stride_dim] =
        store_float<scalar_t>(slice[threadIdx.x] * rsqrtf(width));
  }
}

template <typename scalar_t>
void launch_typed(const at::Tensor& key,
                  const at::Tensor& out,
                  int query_heads,
                  int index_q_heads,
                  float eps,
                  cudaStream_t stream) {
  const int64_t tokens = key.size(0);
  if (tokens == 0) {
    return;
  }
  const int64_t rows = tokens * index_q_heads;
  indexer_key_feature_kernel<scalar_t><<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const scalar_t*>(key.const_data_ptr()),
      reinterpret_cast<scalar_t*>(out.mutable_data_ptr()), tokens,
      key.size(1), key.size(2), query_heads, index_q_heads, eps,
      key.stride(0), key.stride(1), key.stride(2), out.stride(0),
      out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void launch_indexer_key_feature(const at::Tensor& key,
                                const at::Tensor& out,
                                int query_heads,
                                int index_q_heads,
                                float eps) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (key.scalar_type() == at::kHalf) {
    launch_typed<__half>(
        key, out, query_heads, index_q_heads, eps, stream);
  } else {
    launch_typed<__nv_bfloat16>(
        key, out, query_heads, index_q_heads, eps, stream);
  }
}

}  // namespace bfla
