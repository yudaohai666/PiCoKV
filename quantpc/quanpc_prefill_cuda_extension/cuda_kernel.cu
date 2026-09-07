#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cuda_kernel.h"

namespace {

constexpr int kThreads = 128;
constexpr int kWarpsPerBlock = 4;

inline int64_t ceil_div_i64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

template <typename scalar_t>
__device__ __forceinline__ float load_scalar(const scalar_t *ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_scalar<__nv_bfloat16>(const __nv_bfloat16 *ptr, int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_scalar(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_scalar<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename scalar_t>
__device__ __forceinline__ float round_like_scalar(float value) {
  return value;
}

template <>
__device__ __forceinline__ float round_like_scalar<__nv_bfloat16>(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int Subwarp>
__device__ __forceinline__ float subwarp_sum(float value, unsigned mask) {
  #pragma unroll
  for (int offset = Subwarp / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, Subwarp);
  }
  return value;
}

template <int Subwarp>
__device__ __forceinline__ unsigned subwarp_mask(int thread_idx) {
  return ((uint32_t{1} << Subwarp) - 1u) << (thread_idx & (32 - Subwarp));
}

template <typename scalar_t, int CodeDim, int Subwarp>
__global__ void quantize_all_bits_subwarp_d128_codedim_kernel(
    const scalar_t *__restrict__ full_states,
    const scalar_t *__restrict__ projection,
    int8_t *__restrict__ codes,
    int64_t *__restrict__ packed_codes,
    float *__restrict__ err,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int zerorope,
    bool apply_zerorope,
    int pack_layout) {
  static_assert(Subwarp == 2 || Subwarp == 4 || Subwarp == 8 || Subwarp == 16,
                "Subwarp must divide a CUDA warp");
  static_assert(CodeDim % 64 == 0, "CodeDim must be divisible by 64");
  constexpr int kDim = 128;
  constexpr int kValsPerLane = kDim / Subwarp;
  constexpr int kGroupsPerBlock = kThreads / Subwarp;
  constexpr int kPackedBlocks = CodeDim / 64;
  const int sublane = threadIdx.x & (Subwarp - 1);
  const int group = threadIdx.x / Subwarp;
  const int token = blockIdx.x * kGroupsPerBlock + group;
  const int row = blockIdx.y;
  if (row >= rows || token >= glob_len) {
    return;
  }
  const unsigned mask = subwarp_mask<Subwarp>(threadIdx.x);
  const int half_dim = kDim >> 1;
  const int src_token = glob_start + token;
  const int64_t full_base = (static_cast<int64_t>(row) * total_len + src_token) * kDim;

  float vals[kValsPerLane];
  #pragma unroll
  for (int i = 0; i < kValsPerLane; ++i) {
    const int d = sublane + i * Subwarp;
    const bool zero_dim =
        apply_zerorope &&
        zerorope > 0 &&
        (d < zerorope || (d >= half_dim && d < half_dim + zerorope));
    vals[i] = zero_dim ? 0.0f : load_scalar(full_states, full_base + d);
  }

  uint64_t packed_word = 0;
  #pragma unroll 1
  for (int bit_idx = 0; bit_idx < CodeDim; ++bit_idx) {
    const int64_t proj_base = (static_cast<int64_t>(row) * CodeDim + bit_idx) * kDim;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      const int d = sublane + i * Subwarp;
      dot += vals[i] * load_scalar(projection, proj_base + d);
    }
    dot = subwarp_sum<Subwarp>(dot, mask);
    const float group_dot = __shfl_sync(mask, dot, 0, Subwarp);
    const int sign = (group_dot >= 0.0f) ? 1 : -1;
    if (sublane == 0) {
      if (codes != nullptr) {
        codes[(static_cast<int64_t>(row) * glob_len + token) * CodeDim + bit_idx] = static_cast<int8_t>(sign);
      }
      if (packed_codes != nullptr) {
        const int bit = bit_idx & 63;
        if ((pack_layout == 1 && sign < 0) || (pack_layout == 2 && sign > 0)) {
          packed_word |= (uint64_t{1} << bit);
        }
        if (bit == 63) {
          const int block = bit_idx >> 6;
          int64_t out_idx;
          if (pack_layout == 1) {
            out_idx = (static_cast<int64_t>(row) * kPackedBlocks + block) * glob_len + token;
          } else {
            out_idx = (static_cast<int64_t>(row) * glob_len + token) * kPackedBlocks + block;
          }
          packed_codes[out_idx] = static_cast<int64_t>(packed_word);
          packed_word = 0;
        }
      }
    }
    const float coeff = static_cast<float>(sign);
    #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      const int d = sublane + i * Subwarp;
      vals[i] -= coeff * load_scalar(projection, proj_base + d);
    }
    if ((bit_idx & 31) == 31 && bit_idx + 1 < CodeDim) {
      #pragma unroll
      for (int i = 0; i < kValsPerLane; ++i) {
        vals[i] = round_like_scalar<scalar_t>(vals[i]);
      }
    }
  }

  float norm2 = 0.0f;
  #pragma unroll
  for (int i = 0; i < kValsPerLane; ++i) {
    norm2 += vals[i] * vals[i];
  }
  norm2 = subwarp_sum<Subwarp>(norm2, mask);
  if (sublane == 0) {
    err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
  }
}


constexpr int kLearnDim = 128;
constexpr int kLearnWarps = 8;
constexpr int kLearnTileTokensSmall = 128;
constexpr int kLearnTileTokensLarge = 256;

template <typename scalar_t, typename vector_t, bool VectorByBit, int LearnTileTokens>
__global__ void learn_ata_vec_partial_d128_kernel(
    const scalar_t *__restrict__ residual,
    const vector_t *__restrict__ vector,
    float *__restrict__ partial,
    int rows,
    int glob_len,
    int code_dim,
    int bit_idx,
    int num_tiles) {
  __shared__ float warp_acc[kLearnWarps][kLearnDim];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  if (row >= rows || tile >= num_tiles) {
    return;
  }
  const int tile_start = tile * LearnTileTokens;
  const int tile_end = min(tile_start + LearnTileTokens, glob_len);

  float local_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int token = tile_start + warp; token < tile_end; token += kLearnWarps) {
    const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * kLearnDim;
    float vals[4];
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      const float val = load_scalar(residual, residual_base + d);
      int64_t vec_idx;
      if constexpr (VectorByBit) {
        vec_idx = (static_cast<int64_t>(row) * code_dim + bit_idx) * kLearnDim + d;
      } else {
        vec_idx = static_cast<int64_t>(row) * kLearnDim + d;
      }
      vals[i] = val;
      dot += val * load_scalar(vector, vec_idx);
    }
    dot = warp_sum(dot);
    dot = __shfl_sync(0xffffffffu, dot, 0);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      local_acc[i] += vals[i] * dot;
    }
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    warp_acc[warp][lane + i * 32] = local_acc[i];
  }
  __syncthreads();

  if (threadIdx.x < kLearnDim) {
    float sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < kLearnWarps; ++w) {
      sum += warp_acc[w][threadIdx.x];
    }
    partial[(static_cast<int64_t>(row) * num_tiles + tile) * kLearnDim + threadIdx.x] = sum;
  }
}

__global__ void learn_reduce_partial_float_d128_kernel(
    const float *__restrict__ partial,
    float *__restrict__ out,
    int rows,
    int num_tiles) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= rows || d >= kLearnDim) {
    return;
  }
  float sum = 0.0f;
  for (int tile = 0; tile < num_tiles; ++tile) {
    sum += partial[(static_cast<int64_t>(row) * num_tiles + tile) * kLearnDim + d];
  }
  out[static_cast<int64_t>(row) * kLearnDim + d] = sum;
}

template <typename scalar_t, int LearnTileTokens>
__global__ void learn_sign_vh_partial_d128_kernel(
    const scalar_t *__restrict__ residual,
    const float *__restrict__ direction,
    int8_t *__restrict__ signs,
    float *__restrict__ partial,
    int rows,
    int glob_len,
    int num_tiles,
    float inv_len) {
  __shared__ float warp_acc[kLearnWarps][kLearnDim];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  if (row >= rows || tile >= num_tiles) {
    return;
  }
  const int tile_start = tile * LearnTileTokens;
  const int tile_end = min(tile_start + LearnTileTokens, glob_len);

  float local_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int token = tile_start + warp; token < tile_end; token += kLearnWarps) {
    const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * kLearnDim;
    float vals[4];
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      const float val = load_scalar(residual, residual_base + d);
      vals[i] = val;
      dot += val * direction[static_cast<int64_t>(row) * kLearnDim + d];
    }
    dot = warp_sum(dot);
    const int sign = (__shfl_sync(0xffffffffu, dot, 0) >= 0.0f) ? 1 : -1;
    if (lane == 0) {
      signs[static_cast<int64_t>(row) * glob_len + token] = static_cast<int8_t>(sign);
    }
    const float coeff = static_cast<float>(sign) * inv_len;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      local_acc[i] += vals[i] * coeff;
    }
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    warp_acc[warp][lane + i * 32] = local_acc[i];
  }
  __syncthreads();

  if (threadIdx.x < kLearnDim) {
    float sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < kLearnWarps; ++w) {
      sum += warp_acc[w][threadIdx.x];
    }
    partial[(static_cast<int64_t>(row) * num_tiles + tile) * kLearnDim + threadIdx.x] = sum;
  }
}

template <typename scalar_t>
__global__ void learn_reduce_partial_projection_d128_kernel(
    const float *__restrict__ partial,
    scalar_t *__restrict__ projection,
    int rows,
    int num_tiles,
    int code_dim,
    int bit_idx) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= rows || d >= kLearnDim) {
    return;
  }
  float sum = 0.0f;
  for (int tile = 0; tile < num_tiles; ++tile) {
    sum += partial[(static_cast<int64_t>(row) * num_tiles + tile) * kLearnDim + d];
  }
  projection[(static_cast<int64_t>(row) * code_dim + bit_idx) * kLearnDim + d] = store_scalar<scalar_t>(sum);
}

constexpr int kLearnUpdateTileTokens = 128;
constexpr int kLearnUpdateThreads = kLearnWarps * 32;

template <typename scalar_t>
__global__ void learn_update_residual_from_sign_d128_kernel(
    scalar_t *__restrict__ residual,
    const int8_t *__restrict__ signs,
    const scalar_t *__restrict__ projection,
    int rows,
    int glob_len,
    int code_dim,
    int bit_idx,
    int num_update_tiles) {
  __shared__ float proj_sh[kLearnDim];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  if (row >= rows || tile >= num_update_tiles) {
    return;
  }
  if (threadIdx.x < kLearnDim) {
    proj_sh[threadIdx.x] = load_scalar(
        projection,
        (static_cast<int64_t>(row) * code_dim + bit_idx) * kLearnDim + threadIdx.x);
  }
  __syncthreads();

  const int tile_start = tile * kLearnUpdateTileTokens;
  const int tile_end = min(tile_start + kLearnUpdateTileTokens, glob_len);
  for (int token = tile_start + warp; token < tile_end; token += kLearnWarps) {
    float sign = 0.0f;
    if (lane == 0) {
      sign = static_cast<float>(signs[static_cast<int64_t>(row) * glob_len + token]);
    }
    sign = __shfl_sync(0xffffffffu, sign, 0);
    const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * kLearnDim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      const int64_t idx = residual_base + d;
      const float updated = load_scalar(residual, idx) - sign * proj_sh[d];
      residual[idx] = store_scalar<scalar_t>(updated);
    }
  }
}

template <typename scalar_t>
__global__ void init_residual_kernel(
    const scalar_t *__restrict__ full_states,
    scalar_t *__restrict__ residual,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int dim,
    int zerorope,
    bool apply_zerorope) {
  const int64_t total = static_cast<int64_t>(rows) * glob_len * dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int d = idx % dim;
    const int t = (idx / dim) % glob_len;
    const int row = idx / (static_cast<int64_t>(glob_len) * dim);
    const int half_dim = dim / 2;
    const bool zero_dim =
        apply_zerorope &&
        zerorope > 0 &&
        (d < zerorope || (d >= half_dim && d < half_dim + zerorope));
    const int64_t src = (static_cast<int64_t>(row) * total_len + (glob_start + t)) * dim + d;
    residual[idx] = zero_dim ? store_scalar<scalar_t>(0.0f) : full_states[src];
  }
}

template <typename scalar_t>
__global__ void quantize_bit_kernel(
    scalar_t *__restrict__ residual,
    const scalar_t *__restrict__ projection,
    int8_t *__restrict__ codes,
    int64_t *__restrict__ packed_codes,
    float *__restrict__ err,
    int rows,
    int glob_len,
    int code_dim,
    int dim,
    int bit_idx,
    int pack_layout,
    int packed_blocks) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  const int row = blockIdx.y;
  if (row >= rows || token >= glob_len) {
    return;
  }

  const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * dim;
  const int64_t proj_base = (static_cast<int64_t>(row) * code_dim + bit_idx) * dim;

  float dot = 0.0f;
  for (int d = lane; d < dim; d += 32) {
    dot += load_scalar(residual, residual_base + d) * load_scalar(projection, proj_base + d);
  }
  dot = warp_sum(dot);

  const int sign = (__shfl_sync(0xffffffffu, dot, 0) >= 0.0f) ? 1 : -1;
  if (lane == 0) {
    if (codes != nullptr) {
      codes[(static_cast<int64_t>(row) * glob_len + token) * code_dim + bit_idx] = static_cast<int8_t>(sign);
    }
    if (packed_codes != nullptr) {
      const int block = bit_idx >> 6;
      const int bit = bit_idx & 63;
      if (pack_layout == 1) {
        if (sign < 0) {
          const int64_t out_idx = (static_cast<int64_t>(row) * packed_blocks + block) * glob_len + token;
          const uint64_t word = static_cast<uint64_t>(packed_codes[out_idx]) | (uint64_t{1} << bit);
          packed_codes[out_idx] = static_cast<int64_t>(word);
        }
      } else if (pack_layout == 2) {
        if (sign > 0) {
          const int64_t out_idx = (static_cast<int64_t>(row) * glob_len + token) * packed_blocks + block;
          const uint64_t word = static_cast<uint64_t>(packed_codes[out_idx]) | (uint64_t{1} << bit);
          packed_codes[out_idx] = static_cast<int64_t>(word);
        }
      }
    }
  }

  float norm2 = 0.0f;
  for (int d = lane; d < dim; d += 32) {
    const int64_t ridx = residual_base + d;
    const float updated =
        load_scalar(residual, ridx) -
        static_cast<float>(sign) * load_scalar(projection, proj_base + d);
    residual[ridx] = store_scalar<scalar_t>(updated);
    if (bit_idx == code_dim - 1) {
      norm2 += updated * updated;
    }
  }

  if (bit_idx == code_dim - 1) {
    norm2 = warp_sum(norm2);
    if (lane == 0) {
      err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
    }
  }
}


template <typename scalar_t, int BitsPerChunk, int MaxLaneVals>
__global__ void quantize_bits_chunk_kernel(
    scalar_t *__restrict__ residual,
    const scalar_t *__restrict__ projection,
    int8_t *__restrict__ codes,
    int64_t *__restrict__ packed_codes,
    float *__restrict__ err,
    int rows,
    int glob_len,
    int code_dim,
    int dim,
    int bit_start,
    int pack_layout,
    int packed_blocks) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  const int row = blockIdx.y;
  if (row >= rows || token >= glob_len) {
    return;
  }

  float vals[MaxLaneVals];
  int dims[MaxLaneVals];
  int nvals = 0;
  const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * dim;
  #pragma unroll
  for (int i = 0; i < MaxLaneVals; ++i) {
    const int d = lane + i * 32;
    dims[i] = d;
    if (d < dim) {
      vals[i] = load_scalar(residual, residual_base + d);
      nvals = i + 1;
    } else {
      vals[i] = 0.0f;
    }
  }

  #pragma unroll
  for (int local_bit = 0; local_bit < BitsPerChunk; ++local_bit) {
    const int bit_idx = bit_start + local_bit;
    if (bit_idx >= code_dim) {
      continue;
    }
    const int64_t proj_base = (static_cast<int64_t>(row) * code_dim + bit_idx) * dim;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < MaxLaneVals; ++i) {
      if (i < nvals) {
        dot += vals[i] * load_scalar(projection, proj_base + dims[i]);
      }
    }
    dot = warp_sum(dot);
    const int sign = (__shfl_sync(0xffffffffu, dot, 0) >= 0.0f) ? 1 : -1;
    if (lane == 0) {
      if (codes != nullptr) {
        codes[(static_cast<int64_t>(row) * glob_len + token) * code_dim + bit_idx] = static_cast<int8_t>(sign);
      }
      if (packed_codes != nullptr) {
        const int block = bit_idx >> 6;
        const int bit = bit_idx & 63;
        if (pack_layout == 1) {
          if (sign < 0) {
            const int64_t out_idx = (static_cast<int64_t>(row) * packed_blocks + block) * glob_len + token;
            const uint64_t word = static_cast<uint64_t>(packed_codes[out_idx]) | (uint64_t{1} << bit);
            packed_codes[out_idx] = static_cast<int64_t>(word);
          }
        } else if (pack_layout == 2) {
          if (sign > 0) {
            const int64_t out_idx = (static_cast<int64_t>(row) * glob_len + token) * packed_blocks + block;
            const uint64_t word = static_cast<uint64_t>(packed_codes[out_idx]) | (uint64_t{1} << bit);
            packed_codes[out_idx] = static_cast<int64_t>(word);
          }
        }
      }
    }
    const float coeff = static_cast<float>(sign);
    #pragma unroll
    for (int i = 0; i < MaxLaneVals; ++i) {
      if (i < nvals) {
        vals[i] -= coeff * load_scalar(projection, proj_base + dims[i]);
      }
    }
  }

  float norm2 = 0.0f;
  const bool final_chunk = bit_start + BitsPerChunk >= code_dim;
  #pragma unroll
  for (int i = 0; i < MaxLaneVals; ++i) {
    if (i < nvals) {
      residual[residual_base + dims[i]] = store_scalar<scalar_t>(vals[i]);
      if (final_chunk) {
        norm2 += vals[i] * vals[i];
      }
    }
  }
  if (final_chunk) {
    norm2 = warp_sum(norm2);
    if (lane == 0) {
      err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
    }
  }
}


template <typename scalar_t, int MaxLaneVals>
__global__ void quantize_all_bits_kernel(
    const scalar_t *__restrict__ full_states,
    const scalar_t *__restrict__ projection,
    int8_t *__restrict__ codes,
    int64_t *__restrict__ packed_codes,
    float *__restrict__ err,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int dim,
    int zerorope,
    bool apply_zerorope,
    int pack_layout,
    int packed_blocks) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  const int row = blockIdx.y;
  if (row >= rows || token >= glob_len) {
    return;
  }

  float vals[MaxLaneVals];
  int dims[MaxLaneVals];
  int nvals = 0;
  const int half_dim = dim >> 1;
  const int src_token = glob_start + token;
  const int64_t full_base = (static_cast<int64_t>(row) * total_len + src_token) * dim;

  #pragma unroll
  for (int i = 0; i < MaxLaneVals; ++i) {
    const int d = lane + i * 32;
    dims[i] = d;
    if (d < dim) {
      const bool zero_dim =
          apply_zerorope &&
          zerorope > 0 &&
          (d < zerorope || (d >= half_dim && d < half_dim + zerorope));
      vals[i] = zero_dim ? 0.0f : load_scalar(full_states, full_base + d);
      nvals = i + 1;
    } else {
      vals[i] = 0.0f;
    }
  }

  uint64_t packed_word = 0;
  #pragma unroll 1
  for (int bit_idx = 0; bit_idx < code_dim; ++bit_idx) {
    const int64_t proj_base = (static_cast<int64_t>(row) * code_dim + bit_idx) * dim;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < MaxLaneVals; ++i) {
      if (i < nvals) {
        dot += vals[i] * load_scalar(projection, proj_base + dims[i]);
      }
    }
    dot = warp_sum(dot);
    const int sign = (__shfl_sync(0xffffffffu, dot, 0) >= 0.0f) ? 1 : -1;
    if (lane == 0) {
      if (codes != nullptr) {
        codes[(static_cast<int64_t>(row) * glob_len + token) * code_dim + bit_idx] = static_cast<int8_t>(sign);
      }
      if (packed_codes != nullptr) {
        const int bit = bit_idx & 63;
        if ((pack_layout == 1 && sign < 0) || (pack_layout == 2 && sign > 0)) {
          packed_word |= (uint64_t{1} << bit);
        }
        if (bit == 63) {
          const int block = bit_idx >> 6;
          int64_t out_idx;
          if (pack_layout == 1) {
            out_idx = (static_cast<int64_t>(row) * packed_blocks + block) * glob_len + token;
          } else {
            out_idx = (static_cast<int64_t>(row) * glob_len + token) * packed_blocks + block;
          }
          packed_codes[out_idx] = static_cast<int64_t>(packed_word);
          packed_word = 0;
        }
      }
    }
    const float coeff = static_cast<float>(sign);
    #pragma unroll
    for (int i = 0; i < MaxLaneVals; ++i) {
      if (i < nvals) {
        vals[i] -= coeff * load_scalar(projection, proj_base + dims[i]);
      }
    }
    if ((bit_idx & 31) == 31 && bit_idx + 1 < code_dim) {
      #pragma unroll
      for (int i = 0; i < MaxLaneVals; ++i) {
        if (i < nvals) {
          vals[i] = round_like_scalar<scalar_t>(vals[i]);
        }
      }
    }
  }

  float norm2 = 0.0f;
  #pragma unroll
  for (int i = 0; i < MaxLaneVals; ++i) {
    if (i < nvals) {
      norm2 += vals[i] * vals[i];
    }
  }
  norm2 = warp_sum(norm2);
  if (lane == 0) {
    err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
  }
}


template <typename scalar_t, int Subwarp>
__global__ void quantize_all_bits_subwarp_d128_kernel(
    const scalar_t *__restrict__ full_states,
    const scalar_t *__restrict__ projection,
    int8_t *__restrict__ codes,
    int64_t *__restrict__ packed_codes,
    float *__restrict__ err,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int zerorope,
    bool apply_zerorope,
    int pack_layout,
    int packed_blocks) {
  static_assert(Subwarp == 2 || Subwarp == 4 || Subwarp == 8 || Subwarp == 16,
                "Subwarp must divide a CUDA warp");
  constexpr int kDim = 128;
  constexpr int kValsPerLane = kDim / Subwarp;
  constexpr int kGroupsPerBlock = kThreads / Subwarp;
  const int sublane = threadIdx.x & (Subwarp - 1);
  const int group = threadIdx.x / Subwarp;
  const int token = blockIdx.x * kGroupsPerBlock + group;
  const int row = blockIdx.y;
  if (row >= rows || token >= glob_len) {
    return;
  }
  const unsigned mask = subwarp_mask<Subwarp>(threadIdx.x);
  const int half_dim = kDim >> 1;
  const int src_token = glob_start + token;
  const int64_t full_base = (static_cast<int64_t>(row) * total_len + src_token) * kDim;

  float vals[kValsPerLane];
  #pragma unroll
  for (int i = 0; i < kValsPerLane; ++i) {
    const int d = sublane + i * Subwarp;
    const bool zero_dim =
        apply_zerorope &&
        zerorope > 0 &&
        (d < zerorope || (d >= half_dim && d < half_dim + zerorope));
    vals[i] = zero_dim ? 0.0f : load_scalar(full_states, full_base + d);
  }

  uint64_t packed_word = 0;
  #pragma unroll 1
  for (int bit_idx = 0; bit_idx < code_dim; ++bit_idx) {
    const int64_t proj_base = (static_cast<int64_t>(row) * code_dim + bit_idx) * kDim;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      const int d = sublane + i * Subwarp;
      dot += vals[i] * load_scalar(projection, proj_base + d);
    }
    dot = subwarp_sum<Subwarp>(dot, mask);
    float group_dot = dot;
    if constexpr (Subwarp > 1) {
      group_dot = __shfl_sync(mask, dot, 0, Subwarp);
    }
    const int sign = (group_dot >= 0.0f) ? 1 : -1;
    if (sublane == 0) {
      if (codes != nullptr) {
        codes[(static_cast<int64_t>(row) * glob_len + token) * code_dim + bit_idx] = static_cast<int8_t>(sign);
      }
      if (packed_codes != nullptr) {
        const int bit = bit_idx & 63;
        if ((pack_layout == 1 && sign < 0) || (pack_layout == 2 && sign > 0)) {
          packed_word |= (uint64_t{1} << bit);
        }
        if (bit == 63) {
          const int block = bit_idx >> 6;
          int64_t out_idx;
          if (pack_layout == 1) {
            out_idx = (static_cast<int64_t>(row) * packed_blocks + block) * glob_len + token;
          } else {
            out_idx = (static_cast<int64_t>(row) * glob_len + token) * packed_blocks + block;
          }
          packed_codes[out_idx] = static_cast<int64_t>(packed_word);
          packed_word = 0;
        }
      }
    }
    const float coeff = static_cast<float>(sign);
    #pragma unroll
    for (int i = 0; i < kValsPerLane; ++i) {
      const int d = sublane + i * Subwarp;
      vals[i] -= coeff * load_scalar(projection, proj_base + d);
    }
    if ((bit_idx & 31) == 31 && bit_idx + 1 < code_dim) {
      #pragma unroll
      for (int i = 0; i < kValsPerLane; ++i) {
        vals[i] = round_like_scalar<scalar_t>(vals[i]);
      }
    }
  }

  float norm2 = 0.0f;
  #pragma unroll
  for (int i = 0; i < kValsPerLane; ++i) {
    norm2 += vals[i] * vals[i];
  }
  norm2 = subwarp_sum<Subwarp>(norm2, mask);
  if (sublane == 0) {
    err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
  }
}

__global__ void init_indices_offsets_kernel(
    int *__restrict__ indices,
    int *__restrict__ offsets,
    int rows,
    int len) {
  const int64_t total = static_cast<int64_t>(rows) * len;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    indices[idx] = static_cast<int>(idx % len);
  }
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx <= rows; idx += blockDim.x * gridDim.x) {
    offsets[idx] = idx * len;
  }
}

__global__ void mark_topk_kernel(
    const int *__restrict__ sorted_idx,
    uint8_t *__restrict__ selected_mask,
    int rows,
    int len,
    int k_count) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  for (int i = threadIdx.x; i < k_count; i += blockDim.x) {
    const int token = sorted_idx[static_cast<int64_t>(row) * len + i];
    selected_mask[static_cast<int64_t>(row) * len + token] = 1;
  }
}

__global__ void make_value_scores_kernel(
    const float *__restrict__ value_err,
    const uint8_t *__restrict__ selected_mask,
    float *__restrict__ value_score,
    int *__restrict__ indices,
    int rows,
    int len) {
  const int64_t total = static_cast<int64_t>(rows) * len;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    indices[idx] = static_cast<int>(idx % len);
    value_score[idx] = selected_mask[idx] ? -3.4028234663852886e38f : value_err[idx];
  }
}

__global__ void make_eas_offsets_kernel(int *__restrict__ offsets, int rows, int eas_count) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx <= rows; idx += blockDim.x * gridDim.x) {
    offsets[idx] = idx * eas_count;
  }
}

__global__ void finalize_eas_kernel(
    const int *__restrict__ key_sorted_idx,
    const int *__restrict__ value_sorted_idx,
    uint8_t *__restrict__ selected_mask,
    int *__restrict__ eas_unsorted,
    int rows,
    int len,
    int k_count,
    int v_count,
    int eas_count) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  for (int i = threadIdx.x; i < k_count; i += blockDim.x) {
    eas_unsorted[static_cast<int64_t>(row) * eas_count + i] =
        key_sorted_idx[static_cast<int64_t>(row) * len + i];
  }
  for (int i = threadIdx.x; i < v_count; i += blockDim.x) {
    const int token = value_sorted_idx[static_cast<int64_t>(row) * len + i];
    selected_mask[static_cast<int64_t>(row) * len + token] = 1;
    eas_unsorted[static_cast<int64_t>(row) * eas_count + k_count + i] = token;
  }
}

__global__ void build_kept_indices_kernel(
    const uint8_t *__restrict__ selected_mask,
    int *__restrict__ kept_indices,
    int rows,
    int len,
    int kept_len) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  int out = 0;
  for (int token = 0; token < len; ++token) {
    if (!selected_mask[static_cast<int64_t>(row) * len + token]) {
      if (out < kept_len) {
        kept_indices[static_cast<int64_t>(row) * kept_len + out] = token;
      }
      ++out;
    }
  }
}

__global__ void build_kept_from_sorted_eas_kernel(
    const int *__restrict__ eas_indices,
    int *__restrict__ kept_indices,
    int rows,
    int len,
    int eas_count,
    int kept_len) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t eas_base = static_cast<int64_t>(row) * eas_count;
  const int64_t kept_base = static_cast<int64_t>(row) * kept_len;
  for (int interval = threadIdx.x; interval <= eas_count; interval += blockDim.x) {
    const int prev = (interval == 0) ? -1 : eas_indices[eas_base + interval - 1];
    const int next = (interval == eas_count) ? len : eas_indices[eas_base + interval];
    const int start = prev + 1;
    const int out_start = start - interval;
    for (int token = start; token < next; ++token) {
      kept_indices[kept_base + out_start + (token - start)] = token;
    }
  }
}

template <typename scalar_t>
__global__ void gather_dense_kernel(
    const scalar_t *__restrict__ full_states,
    const int *__restrict__ eas_indices,
    scalar_t *__restrict__ dense_states,
    int rows,
    int total_len,
    int dim,
    int sink_size,
    int local_size,
    int eas_count,
    int dense_len) {
  const int64_t total = static_cast<int64_t>(rows) * dense_len * dim;
  const int glob_start = sink_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int d = idx % dim;
    const int pos = (idx / dim) % dense_len;
    const int row = idx / (static_cast<int64_t>(dense_len) * dim);
    int src_token;
    if (pos < sink_size) {
      src_token = pos;
    } else if (pos < sink_size + eas_count) {
      src_token = glob_start + eas_indices[static_cast<int64_t>(row) * eas_count + (pos - sink_size)];
    } else {
      src_token = total_len - local_size + (pos - sink_size - eas_count);
    }
    dense_states[idx] = full_states[(static_cast<int64_t>(row) * total_len + src_token) * dim + d];
  }
}

template <typename scalar_t>
__global__ void gather_dense_kv_kernel(
    const scalar_t *__restrict__ full_key,
    const scalar_t *__restrict__ full_value,
    const int *__restrict__ eas_indices,
    scalar_t *__restrict__ dense_key,
    scalar_t *__restrict__ dense_value,
    int rows,
    int total_len,
    int dim,
    int sink_size,
    int local_size,
    int eas_count,
    int dense_len) {
  const int64_t total = static_cast<int64_t>(rows) * dense_len * dim;
  const int glob_start = sink_size;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int d = idx % dim;
    const int pos = (idx / dim) % dense_len;
    const int row = idx / (static_cast<int64_t>(dense_len) * dim);
    int src_token;
    if (pos < sink_size) {
      src_token = pos;
    } else if (pos < sink_size + eas_count) {
      src_token = glob_start + eas_indices[static_cast<int64_t>(row) * eas_count + (pos - sink_size)];
    } else {
      src_token = total_len - local_size + (pos - sink_size - eas_count);
    }
    const int64_t src = (static_cast<int64_t>(row) * total_len + src_token) * dim + d;
    dense_key[idx] = full_key[src];
    dense_value[idx] = full_value[src];
  }
}

__global__ void pack_key_kept_kernel(
    const int8_t *__restrict__ key_codes,
    const int *__restrict__ kept_indices,
    int64_t *__restrict__ packed_key,
    int rows,
    int glob_len,
    int code_dim,
    int kept_len,
    int key_blocks) {
  const int64_t total = static_cast<int64_t>(rows) * key_blocks * kept_len;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int token_out = idx % kept_len;
    const int key_block = (idx / kept_len) % key_blocks;
    const int row = idx / (static_cast<int64_t>(kept_len) * key_blocks);
    const int token_in = kept_indices[static_cast<int64_t>(row) * kept_len + token_out];
    const int64_t code_base = (static_cast<int64_t>(row) * glob_len + token_in) * code_dim + key_block * 64;
    uint64_t word = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
      if (key_codes[code_base + bit] < 0) {
        word |= (uint64_t{1} << bit);
      }
    }
    packed_key[idx] = static_cast<int64_t>(word);
  }
}

__global__ void pack_value_kept_kernel(
    const int8_t *__restrict__ value_codes,
    const int *__restrict__ kept_indices,
    int64_t *__restrict__ packed_value,
    int rows,
    int glob_len,
    int code_dim,
    int kept_len,
    int value_groups) {
  const int64_t total = static_cast<int64_t>(rows) * value_groups * code_dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int c = idx % code_dim;
    const int group = (idx / code_dim) % value_groups;
    const int row = idx / (static_cast<int64_t>(value_groups) * code_dim);
    uint64_t word = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
      const int token_out = group * 64 + bit;
      if (token_out < kept_len) {
        const int token_in = kept_indices[static_cast<int64_t>(row) * kept_len + token_out];
        const int8_t code = value_codes[(static_cast<int64_t>(row) * glob_len + token_in) * code_dim + c];
        if (code > 0) {
          word |= (uint64_t{1} << bit);
        }
      }
    }
    packed_value[idx] = static_cast<int64_t>(word);
  }
}

size_t segmented_sort_pairs_desc_temp_bytes(
    at::Tensor keys_in,
    at::Tensor keys_out,
    at::Tensor vals_in,
    at::Tensor vals_out,
    at::Tensor offsets,
    int rows,
    int total,
    cudaStream_t stream) {
  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      keys_in.data_ptr<float>(),
      keys_out.data_ptr<float>(),
      vals_in.data_ptr<int>(),
      vals_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  return temp_bytes;
}

void segmented_sort_pairs_desc_with_temp(
    at::Tensor keys_in,
    at::Tensor keys_out,
    at::Tensor vals_in,
    at::Tensor vals_out,
    at::Tensor offsets,
    at::Tensor temp,
    size_t temp_bytes,
    int rows,
    int total,
    cudaStream_t stream) {
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      keys_in.data_ptr<float>(),
      keys_out.data_ptr<float>(),
      vals_in.data_ptr<int>(),
      vals_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(float) * 8,
      stream);
}

size_t segmented_sort_keys_asc_temp_bytes(
    at::Tensor keys_in,
    at::Tensor keys_out,
    at::Tensor offsets,
    int rows,
    int total,
    cudaStream_t stream) {
  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr,
      temp_bytes,
      keys_in.data_ptr<int>(),
      keys_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(int) * 8,
      stream);
  return temp_bytes;
}

void segmented_sort_keys_asc_with_temp(
    at::Tensor keys_in,
    at::Tensor keys_out,
    at::Tensor offsets,
    at::Tensor temp,
    size_t temp_bytes,
    int rows,
    int total,
    cudaStream_t stream) {
  cub::DeviceSegmentedRadixSort::SortKeys(
      temp.data_ptr(),
      temp_bytes,
      keys_in.data_ptr<int>(),
      keys_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(int) * 8,
      stream);
}

void segmented_sort_pairs_desc(
    at::Tensor keys_in,
    at::Tensor keys_out,
    at::Tensor vals_in,
    at::Tensor vals_out,
    at::Tensor offsets,
    int rows,
    int total,
    cudaStream_t stream) {
  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_bytes,
      keys_in.data_ptr<float>(),
      keys_out.data_ptr<float>(),
      vals_in.data_ptr<int>(),
      vals_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(float) * 8,
      stream);
  auto temp = at::empty({static_cast<int64_t>(temp_bytes)}, keys_in.options().dtype(at::kByte));
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp.data_ptr(),
      temp_bytes,
      keys_in.data_ptr<float>(),
      keys_out.data_ptr<float>(),
      vals_in.data_ptr<int>(),
      vals_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(float) * 8,
      stream);
}

void segmented_sort_keys_asc(
    at::Tensor keys_in,
    at::Tensor keys_out,
    at::Tensor offsets,
    int rows,
    int total,
    cudaStream_t stream) {
  size_t temp_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr,
      temp_bytes,
      keys_in.data_ptr<int>(),
      keys_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(int) * 8,
      stream);
  auto temp = at::empty({static_cast<int64_t>(temp_bytes)}, keys_in.options().dtype(at::kByte));
  cub::DeviceSegmentedRadixSort::SortKeys(
      temp.data_ptr(),
      temp_bytes,
      keys_in.data_ptr<int>(),
      keys_out.data_ptr<int>(),
      total,
      rows,
      offsets.data_ptr<int>(),
      offsets.data_ptr<int>() + 1,
      0,
      sizeof(int) * 8,
      stream);
}

template <typename scalar_t>
void launch_quantize_typed(
    const scalar_t *full_ptr,
    const scalar_t *proj_ptr,
    scalar_t *residual_ptr,
    int8_t *codes_ptr,
    int64_t *packed_ptr,
    float *err_ptr,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int dim,
    int zerorope,
    bool apply_zerorope,
    int pack_layout,
    cudaStream_t stream) {
  (void)residual_ptr;

  dim3 block(kThreads);
  if (dim == 128) {
    constexpr int groups_per_block = kThreads / 2;
    dim3 grid(static_cast<unsigned int>(ceil_div_i64(glob_len, groups_per_block)), rows);
    if (code_dim == 128) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 128, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    }
    else if (code_dim == 192) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 192, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    }
    else if (code_dim == 256) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 256, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    }
    else if (code_dim == 320) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 320, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    }
    else if (code_dim == 384) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 384, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    }
    else if (code_dim == 448) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 448, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    }
    else if (code_dim == 512) {
      quantize_all_bits_subwarp_d128_codedim_kernel<scalar_t, 512, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          zerorope,
          apply_zerorope,
          pack_layout);
    } else {
      quantize_all_bits_subwarp_d128_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
          full_ptr,
          proj_ptr,
          codes_ptr,
          packed_ptr,
          err_ptr,
          rows,
          total_len,
          glob_start,
          glob_len,
          code_dim,
          zerorope,
          apply_zerorope,
          pack_layout,
          code_dim / 64);
    }
  } else if (dim < 128) {
    dim3 grid(static_cast<unsigned int>(ceil_div_i64(glob_len, kWarpsPerBlock)), rows);
    quantize_all_bits_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
        full_ptr,
        proj_ptr,
        codes_ptr,
        packed_ptr,
        err_ptr,
        rows,
        total_len,
        glob_start,
        glob_len,
        code_dim,
        dim,
        zerorope,
        apply_zerorope,
        pack_layout,
        code_dim / 64);
  } else {
    dim3 grid(static_cast<unsigned int>(ceil_div_i64(glob_len, kWarpsPerBlock)), rows);
    quantize_all_bits_kernel<scalar_t, 8><<<grid, block, 0, stream>>>(
        full_ptr,
        proj_ptr,
        codes_ptr,
        packed_ptr,
        err_ptr,
        rows,
        total_len,
        glob_start,
        glob_len,
        code_dim,
        dim,
        zerorope,
        apply_zerorope,
        pack_layout,
        code_dim / 64);
  }
}




__global__ void gather_key_from_packed_all_kernel(
    const int64_t *__restrict__ packed_key_all,
    const int *__restrict__ kept_indices,
    int64_t *__restrict__ packed_key,
    int rows,
    int glob_len,
    int kept_len,
    int key_blocks) {
  const int64_t total = static_cast<int64_t>(rows) * key_blocks * kept_len;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int token_out = idx % kept_len;
    const int key_block = (idx / kept_len) % key_blocks;
    const int row = idx / (static_cast<int64_t>(kept_len) * key_blocks);
    const int token_in = kept_indices[static_cast<int64_t>(row) * kept_len + token_out];
    packed_key[idx] = packed_key_all[(static_cast<int64_t>(row) * key_blocks + key_block) * glob_len + token_in];
  }
}

__global__ void pack_value_from_token_packed_kernel(
    const int64_t *__restrict__ packed_value_token,
    const int *__restrict__ kept_indices,
    int64_t *__restrict__ packed_value,
    int rows,
    int glob_len,
    int code_dim,
    int kept_len,
    int value_blocks,
    int value_groups) {
  const int64_t total = static_cast<int64_t>(rows) * value_groups * code_dim;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int c = idx % code_dim;
    const int group = (idx / code_dim) % value_groups;
    const int row = idx / (static_cast<int64_t>(value_groups) * code_dim);
    const int c_block = c >> 6;
    const int c_bit = c & 63;
    uint64_t word = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
      const int token_out = group * 64 + bit;
      if (token_out < kept_len) {
        const int token_in = kept_indices[static_cast<int64_t>(row) * kept_len + token_out];
        const uint64_t token_word = static_cast<uint64_t>(
            packed_value_token[(static_cast<int64_t>(row) * glob_len + token_in) * value_blocks + c_block]);
        if ((token_word >> c_bit) & uint64_t{1}) {
          word |= (uint64_t{1} << bit);
        }
      }
    }
    packed_value[idx] = static_cast<int64_t>(word);
  }
}

__global__ void pack_value_from_token_packed_tile_kernel(
    const int64_t *__restrict__ packed_value_token,
    const int *__restrict__ kept_indices,
    int64_t *__restrict__ packed_value,
    int rows,
    int glob_len,
    int code_dim,
    int kept_len,
    int value_blocks,
    int value_groups) {
  __shared__ uint64_t token_words[64];
  const int group = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int lane = threadIdx.x;
  if (row >= rows || group >= value_groups || c_block >= value_blocks || lane >= 64) {
    return;
  }

  const int token_out = group * 64 + lane;
  uint64_t token_word = 0;
  if (token_out < kept_len) {
    const int token_in = kept_indices[static_cast<int64_t>(row) * kept_len + token_out];
    token_word = static_cast<uint64_t>(
        packed_value_token[(static_cast<int64_t>(row) * glob_len + token_in) * value_blocks + c_block]);
  }
  token_words[lane] = token_word;
  __syncthreads();

  uint64_t out_word = 0;
  #pragma unroll
  for (int bit = 0; bit < 64; ++bit) {
    if ((token_words[bit] >> lane) & uint64_t{1}) {
      out_word |= (uint64_t{1} << bit);
    }
  }
  const int c = c_block * 64 + lane;
  packed_value[(static_cast<int64_t>(row) * value_groups + group) * code_dim + c] = static_cast<int64_t>(out_word);
}



template <typename scalar_t, int GroupBits, int LearnTileTokens>
__global__ void blockk_ata_partial_d128_kernel(
    const scalar_t *__restrict__ residual,
    const scalar_t *__restrict__ vector,
    float *__restrict__ partial,
    int rows,
    int glob_len,
    int code_dim,
    int group_start,
    int active_bits,
    int num_tiles) {
  __shared__ float shared_acc[GroupBits * kLearnDim];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  if (row >= rows || tile >= num_tiles) {
    return;
  }
  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    shared_acc[idx] = 0.0f;
  }
  __syncthreads();

  float local_acc[GroupBits][4];
  #pragma unroll
  for (int g = 0; g < GroupBits; ++g) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      local_acc[g][i] = 0.0f;
    }
  }

  const int tile_start = tile * LearnTileTokens;
  const int tile_end = min(tile_start + LearnTileTokens, glob_len);
  for (int token = tile_start + warp; token < tile_end; token += kLearnWarps) {
    const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * kLearnDim;
    float vals[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      vals[i] = load_scalar(residual, residual_base + d);
    }

    float dots[GroupBits];
    #pragma unroll
    for (int g = 0; g < GroupBits; ++g) {
      float dot = 0.0f;
      if (g < active_bits) {
        const int64_t vec_base = (static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = lane + i * 32;
          dot += vals[i] * load_scalar(vector, vec_base + d);
        }
      }
      dot = warp_sum(dot);
      dots[g] = __shfl_sync(0xffffffffu, dot, 0);
    }

    #pragma unroll
    for (int g = 0; g < GroupBits; ++g) {
      if (g < active_bits) {
        const float coeff = dots[g];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          local_acc[g][i] += vals[i] * coeff;
        }
      }
    }
  }

  #pragma unroll
  for (int g = 0; g < GroupBits; ++g) {
    if (g < active_bits) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        atomicAdd(&shared_acc[g * kLearnDim + d], local_acc[g][i]);
      }
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    partial[((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits * kLearnDim) + idx] = shared_acc[idx];
  }
}

template <typename scalar_t, int GroupBits>
__global__ void blockk_reduce_norm_store_d128_kernel(
    const float *__restrict__ partial,
    scalar_t *__restrict__ encoder,
    int rows,
    int num_tiles,
    int code_dim,
    int group_start,
    int active_bits) {
  __shared__ float vec[kLearnDim];
  __shared__ float red[kLearnDim];
  const int row = blockIdx.y;
  const int g = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= rows || g >= active_bits || d >= kLearnDim) {
    return;
  }
  float sum = 0.0f;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int64_t idx = ((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits + g) * kLearnDim + d;
    sum += partial[idx];
  }
  vec[d] = sum;
  red[d] = sum * sum;
  __syncthreads();

  for (int stride = kLearnDim >> 1; stride > 0; stride >>= 1) {
    if (d < stride) {
      red[d] += red[d + stride];
    }
    __syncthreads();
  }
  const float inv_norm = rsqrtf(fmaxf(red[0], 1.0e-20f));
  encoder[(static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d] = store_scalar<scalar_t>(vec[d] * inv_norm);
}



template <typename scalar_t, int GroupBits>
__global__ void blockk_init_orth_store_d128_kernel(
    const scalar_t *__restrict__ init_vectors,
    scalar_t *__restrict__ encoder,
    int rows,
    int code_dim,
    int group_start,
    int active_bits) {
  __shared__ float vec[GroupBits * kLearnDim];
  __shared__ float red[kLearnDim];
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    const int g = idx / kLearnDim;
    const int d = idx - g * kLearnDim;
    vec[idx] = load_scalar(init_vectors, (static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d);
  }
  __syncthreads();

  for (int g = 0; g < active_bits; ++g) {
    for (int prev = 0; prev < g; ++prev) {
      float local = 0.0f;
      for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
        local += vec[g * kLearnDim + d] * vec[prev * kLearnDim + d];
      }
      local = warp_sum(local);
      const int lane = threadIdx.x & 31;
      const int warp = threadIdx.x >> 5;
      __shared__ float warp_sums[8];
      if (lane == 0) {
        warp_sums[warp] = local;
      }
      __syncthreads();
      float dot = 0.0f;
      if (warp == 0) {
        dot = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        dot = warp_sum(dot);
        if (lane == 0) {
          red[0] = dot;
        }
      }
      __syncthreads();
      dot = red[0];
      for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
        vec[g * kLearnDim + d] -= dot * vec[prev * kLearnDim + d];
      }
      __syncthreads();
    }

    float local_norm = 0.0f;
    for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
      const float v = vec[g * kLearnDim + d];
      local_norm += v * v;
    }
    local_norm = warp_sum(local_norm);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float norm_warp_sums[8];
    if (lane == 0) {
      norm_warp_sums[warp] = local_norm;
    }
    __syncthreads();
    float norm2 = 0.0f;
    if (warp == 0) {
      norm2 = (lane < (blockDim.x >> 5)) ? norm_warp_sums[lane] : 0.0f;
      norm2 = warp_sum(norm2);
      if (lane == 0) {
        red[0] = norm2;
      }
    }
    __syncthreads();
    norm2 = red[0];
    const float inv_norm = rsqrtf(fmaxf(norm2, 1.0e-20f));
    for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
      vec[g * kLearnDim + d] *= inv_norm;
    }
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    const int g = idx / kLearnDim;
    const int d = idx - g * kLearnDim;
    encoder[(static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d] = store_scalar<scalar_t>(vec[idx]);
  }
}

template <typename scalar_t, int GroupBits>
__global__ void blockk_reduce_orth_store_d128_kernel(
    const float *__restrict__ partial,
    scalar_t *__restrict__ encoder,
    int rows,
    int num_tiles,
    int code_dim,
    int group_start,
    int active_bits) {
  __shared__ float vec[GroupBits * kLearnDim];
  __shared__ float red[kLearnDim];
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    const int g = idx / kLearnDim;
    const int d = idx - g * kLearnDim;
    float sum = 0.0f;
    for (int tile = 0; tile < num_tiles; ++tile) {
      sum += partial[((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits + g) * kLearnDim + d];
    }
    vec[idx] = sum;
  }
  __syncthreads();

  for (int g = 0; g < active_bits; ++g) {
    for (int prev = 0; prev < g; ++prev) {
      float local = 0.0f;
      for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
        local += vec[g * kLearnDim + d] * vec[prev * kLearnDim + d];
      }
      local = warp_sum(local);
      const int lane = threadIdx.x & 31;
      const int warp = threadIdx.x >> 5;
      __shared__ float warp_sums[8];
      if (lane == 0) {
        warp_sums[warp] = local;
      }
      __syncthreads();
      float dot = 0.0f;
      if (warp == 0) {
        dot = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        dot = warp_sum(dot);
        if (lane == 0) {
          red[0] = dot;
        }
      }
      __syncthreads();
      dot = red[0];
      for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
        vec[g * kLearnDim + d] -= dot * vec[prev * kLearnDim + d];
      }
      __syncthreads();
    }

    float local_norm = 0.0f;
    for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
      const float v = vec[g * kLearnDim + d];
      local_norm += v * v;
    }
    local_norm = warp_sum(local_norm);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float norm_warp_sums[8];
    if (lane == 0) {
      norm_warp_sums[warp] = local_norm;
    }
    __syncthreads();
    float norm2 = 0.0f;
    if (warp == 0) {
      norm2 = (lane < (blockDim.x >> 5)) ? norm_warp_sums[lane] : 0.0f;
      norm2 = warp_sum(norm2);
      if (lane == 0) {
        red[0] = norm2;
      }
    }
    __syncthreads();
    norm2 = red[0];
    const float inv_norm = rsqrtf(fmaxf(norm2, 1.0e-20f));
    for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
      vec[g * kLearnDim + d] *= inv_norm;
    }
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    const int g = idx / kLearnDim;
    const int d = idx - g * kLearnDim;
    encoder[(static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d] = store_scalar<scalar_t>(vec[idx]);
  }
}

template <typename scalar_t, int GroupBits, int LearnTileTokens>
__global__ void blockk_codes_rhs_partial_d128_kernel(
    const scalar_t *__restrict__ residual,
    const scalar_t *__restrict__ encoder,
    int8_t *__restrict__ codes,
    float *__restrict__ partial,
    float *__restrict__ gram_partial,
    int rows,
    int glob_len,
    int code_dim,
    int group_start,
    int active_bits,
    int num_tiles,
    float inv_len,
    bool compute_gram) {
  __shared__ float shared_acc[GroupBits * kLearnDim];
  __shared__ float shared_gram[GroupBits * GroupBits];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.y;
  const int tile = blockIdx.x;
  if (row >= rows || tile >= num_tiles) {
    return;
  }
  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    shared_acc[idx] = 0.0f;
  }
  if (compute_gram) {
    for (int idx = threadIdx.x; idx < active_bits * active_bits; idx += blockDim.x) {
      shared_gram[idx] = 0.0f;
    }
  }
  __syncthreads();

  float local_acc[GroupBits][4];
  #pragma unroll
  for (int g = 0; g < GroupBits; ++g) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      local_acc[g][i] = 0.0f;
    }
  }

  const int tile_start = tile * LearnTileTokens;
  const int tile_end = min(tile_start + LearnTileTokens, glob_len);
  for (int token = tile_start + warp; token < tile_end; token += kLearnWarps) {
    const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * kLearnDim;
    float vals[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      vals[i] = load_scalar(residual, residual_base + d);
    }

    int signs[GroupBits];
    #pragma unroll
    for (int g = 0; g < GroupBits; ++g) {
      float dot = 0.0f;
      if (g < active_bits) {
        const int64_t enc_base = (static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = lane + i * 32;
          dot += vals[i] * load_scalar(encoder, enc_base + d);
        }
      }
      dot = warp_sum(dot);
      const float score = __shfl_sync(0xffffffffu, dot, 0);
      signs[g] = (score >= 0.0f) ? 1 : -1;
      if (lane == 0 && g < active_bits) {
        codes[(static_cast<int64_t>(row) * glob_len + token) * code_dim + group_start + g] = static_cast<int8_t>(signs[g]);
      }
    }

    if (compute_gram && lane == 0) {
      #pragma unroll
      for (int gi = 0; gi < GroupBits; ++gi) {
        if (gi < active_bits) {
          const float si = static_cast<float>(signs[gi]) * inv_len;
          #pragma unroll
          for (int gj = gi; gj < GroupBits; ++gj) {
            if (gj < active_bits) {
              atomicAdd(&shared_gram[gi * GroupBits + gj], si * static_cast<float>(signs[gj]));
            }
          }
        }
      }
    }

    #pragma unroll
    for (int g = 0; g < GroupBits; ++g) {
      if (g < active_bits) {
        const float coeff = static_cast<float>(signs[g]) * inv_len;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          local_acc[g][i] += vals[i] * coeff;
        }
      }
    }
  }

  #pragma unroll
  for (int g = 0; g < GroupBits; ++g) {
    if (g < active_bits) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        atomicAdd(&shared_acc[g * kLearnDim + d], local_acc[g][i]);
      }
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    partial[((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits * kLearnDim) + idx] = shared_acc[idx];
  }
  if (compute_gram) {
    const int tri_count = active_bits * (active_bits + 1) / 2;
    for (int tri = threadIdx.x; tri < tri_count; tri += blockDim.x) {
      int remaining = tri;
      int gi = 0;
      while (gi < active_bits) {
        const int row_count = active_bits - gi;
        if (remaining < row_count) {
          break;
        }
        remaining -= row_count;
        ++gi;
      }
      const int gj = gi + remaining;
      const int idx = gi * GroupBits + gj;
      gram_partial[((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits * GroupBits) + idx] = shared_gram[idx];
    }
  }
}

template <typename scalar_t, int GroupBits>
__global__ void blockk_reduce_decoder_store_d128_kernel(
    const float *__restrict__ partial,
    scalar_t *__restrict__ decoder,
    int rows,
    int num_tiles,
    int code_dim,
    int group_start,
    int active_bits) {
  const int row = blockIdx.y;
  const int g = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= rows || g >= active_bits || d >= kLearnDim) {
    return;
  }
  float sum = 0.0f;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int64_t idx = ((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits + g) * kLearnDim + d;
    sum += partial[idx];
  }
  decoder[(static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d] = store_scalar<scalar_t>(sum);
}


template <int GroupBits>
__global__ void blockk_reduce_rhs_float_d128_kernel(
    const float *__restrict__ partial,
    float *__restrict__ rhs,
    int rows,
    int num_tiles,
    int active_bits) {
  const int row = blockIdx.y;
  const int g = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= rows || g >= active_bits || d >= kLearnDim) {
    return;
  }
  float sum = 0.0f;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int64_t idx = ((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits + g) * kLearnDim + d;
    sum += partial[idx];
  }
  rhs[(static_cast<int64_t>(row) * GroupBits + g) * kLearnDim + d] = sum;
}

template <int GroupBits>
__global__ void blockk_reduce_gram_d128_kernel(
    const float *__restrict__ gram_partial,
    float *__restrict__ gram,
    int rows,
    int num_tiles,
    int active_bits) {
  const int row = blockIdx.y;
  const int tri = blockIdx.x;
  const int tri_count = active_bits * (active_bits + 1) / 2;
  if (row >= rows || tri >= tri_count) {
    return;
  }
  int remaining = tri;
  int gi = 0;
  while (gi < active_bits) {
    const int row_count = active_bits - gi;
    if (remaining < row_count) {
      break;
    }
    remaining -= row_count;
    ++gi;
  }
  const int gj = gi + remaining;
  const int pair = gi * GroupBits + gj;
  float sum = 0.0f;
  for (int tile = threadIdx.x; tile < num_tiles; tile += blockDim.x) {
    const int64_t idx = ((static_cast<int64_t>(row) * num_tiles + tile) * GroupBits * GroupBits) + pair;
    sum += gram_partial[idx];
  }
  sum = warp_sum(sum);
  __shared__ float warp_sums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    float total = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const int64_t base = static_cast<int64_t>(row) * GroupBits * GroupBits;
      gram[base + pair] = total;
      if (gi != gj) {
        gram[base + gj * GroupBits + gi] = total;
      }
    }
  }
}

template <typename scalar_t, int GroupBits>
__global__ void blockk_solve_decoder_d128_kernel(
    const float *__restrict__ gram,
    const float *__restrict__ rhs,
    scalar_t *__restrict__ decoder,
    int rows,
    int code_dim,
    int group_start,
    int active_bits) {
  __shared__ float A[GroupBits * GroupBits];
  __shared__ float B[GroupBits * kLearnDim];
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  for (int idx = threadIdx.x; idx < GroupBits * GroupBits; idx += blockDim.x) {
    const int gi = idx / GroupBits;
    const int gj = idx - gi * GroupBits;
    float value = 0.0f;
    if (gi < active_bits && gj < active_bits) {
      value = gram[static_cast<int64_t>(row) * GroupBits * GroupBits + idx];
    } else if (gi == gj) {
      value = 1.0f;
    }
    A[idx] = value;
  }
  for (int idx = threadIdx.x; idx < GroupBits * kLearnDim; idx += blockDim.x) {
    const int g = idx / kLearnDim;
    const int d = idx - g * kLearnDim;
    B[idx] = (g < active_bits) ? rhs[(static_cast<int64_t>(row) * GroupBits + g) * kLearnDim + d] : 0.0f;
  }
  __syncthreads();

  for (int pivot_idx = 0; pivot_idx < active_bits; ++pivot_idx) {
    float pivot = A[pivot_idx * GroupBits + pivot_idx];
    if (fabsf(pivot) < 1.0e-6f) {
      pivot = (pivot < 0.0f) ? -1.0e-6f : 1.0e-6f;
    }
    const float inv_pivot = 1.0f / pivot;
    for (int c = threadIdx.x; c < active_bits; c += blockDim.x) {
      A[pivot_idx * GroupBits + c] *= inv_pivot;
    }
    for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
      B[pivot_idx * kLearnDim + d] *= inv_pivot;
    }
    __syncthreads();

    for (int r = 0; r < active_bits; ++r) {
      if (r == pivot_idx) {
        continue;
      }
      const float factor = A[r * GroupBits + pivot_idx];
      for (int c = threadIdx.x; c < active_bits; c += blockDim.x) {
        A[r * GroupBits + c] -= factor * A[pivot_idx * GroupBits + c];
      }
      for (int d = threadIdx.x; d < kLearnDim; d += blockDim.x) {
        B[r * kLearnDim + d] -= factor * B[pivot_idx * kLearnDim + d];
      }
      __syncthreads();
    }
  }

  for (int idx = threadIdx.x; idx < active_bits * kLearnDim; idx += blockDim.x) {
    const int g = idx / kLearnDim;
    const int d = idx - g * kLearnDim;
    decoder[(static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d] = store_scalar<scalar_t>(B[g * kLearnDim + d]);
  }
}

template <typename scalar_t, int GroupBits>
__global__ void blockk_update_residual_d128_kernel(
    scalar_t *__restrict__ residual,
    const int8_t *__restrict__ codes,
    const scalar_t *__restrict__ decoder,
    float *__restrict__ err,
    int rows,
    int glob_len,
    int code_dim,
    int group_start,
    int active_bits,
    bool final_group) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  const int row = blockIdx.y;
  if (row >= rows || token >= glob_len) {
    return;
  }
  const int64_t residual_base = (static_cast<int64_t>(row) * glob_len + token) * kLearnDim;
  float norm2 = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * 32;
    float val = load_scalar(residual, residual_base + d);
    float update = 0.0f;
    #pragma unroll
    for (int g = 0; g < GroupBits; ++g) {
      if (g < active_bits) {
        const float sign = static_cast<float>(codes[(static_cast<int64_t>(row) * glob_len + token) * code_dim + group_start + g]);
        update += sign * load_scalar(decoder, (static_cast<int64_t>(row) * code_dim + group_start + g) * kLearnDim + d);
      }
    }
    val -= update;
    residual[residual_base + d] = store_scalar<scalar_t>(val);
    if (final_group) {
      norm2 += val * val;
    }
  }
  if (final_group) {
    norm2 = warp_sum(norm2);
    if (lane == 0) {
      err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
    }
  }
}

template <typename scalar_t, int GroupBits, int LearnTileTokens>
void launch_blockk_fast_typed(
    const scalar_t *full_ptr,
    const scalar_t *init_ptr,
    int8_t *codes_ptr,
    scalar_t *encoder_ptr,
    scalar_t *decoder_ptr,
    scalar_t *residual_ptr,
    float *partial_ptr,
    float *gram_partial_ptr,
    float *rhs_ptr,
    float *gram_ptr,
    float *err_ptr,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int zerorope,
    bool apply_zerorope,
    int power_iters,
    int solve_mode,
    int num_tiles,
    cudaStream_t stream) {
  const int64_t residual_total = static_cast<int64_t>(rows) * glob_len * kLearnDim;
  const int init_blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(residual_total, kThreads), 65535));
  init_residual_kernel<scalar_t><<<init_blocks, kThreads, 0, stream>>>(
      full_ptr,
      residual_ptr,
      rows,
      total_len,
      glob_start,
      glob_len,
      kLearnDim,
      zerorope,
      apply_zerorope);

  dim3 partial_grid(static_cast<unsigned int>(num_tiles), rows);
  dim3 vector_block(kLearnWarps * 32);
  dim3 group_reduce_grid(GroupBits, rows);
  dim3 gram_reduce_grid(GroupBits * (GroupBits + 1) / 2, rows);
  dim3 dim_block(kLearnDim);
  const int update_tiles = static_cast<int>(ceil_div_i64(glob_len, kWarpsPerBlock));
  dim3 update_grid(static_cast<unsigned int>(update_tiles), rows);
  const float inv_len = 1.0f / static_cast<float>(glob_len);
  power_iters = max(0, min(power_iters, 2));

  for (int group_start = 0; group_start < code_dim; group_start += GroupBits) {
    const int active_bits = min(GroupBits, code_dim - group_start);
    if (power_iters == 0) {
      blockk_init_orth_store_d128_kernel<scalar_t, GroupBits><<<rows, 256, 0, stream>>>(
          init_ptr,
          encoder_ptr,
          rows,
          code_dim,
          group_start,
          active_bits);
    } else {
      for (int iter = 0; iter < power_iters; ++iter) {
        const scalar_t *vector_ptr = (iter == 0) ? init_ptr : encoder_ptr;
        blockk_ata_partial_d128_kernel<scalar_t, GroupBits, LearnTileTokens><<<partial_grid, vector_block, 0, stream>>>(
            residual_ptr,
            vector_ptr,
            partial_ptr,
            rows,
            glob_len,
            code_dim,
            group_start,
            active_bits,
            num_tiles);
        blockk_reduce_orth_store_d128_kernel<scalar_t, GroupBits><<<rows, 256, 0, stream>>>(
            partial_ptr,
            encoder_ptr,
            rows,
            num_tiles,
            code_dim,
            group_start,
            active_bits);
      }
    }

    blockk_codes_rhs_partial_d128_kernel<scalar_t, GroupBits, LearnTileTokens><<<partial_grid, vector_block, 0, stream>>>(
        residual_ptr,
        encoder_ptr,
        codes_ptr,
        partial_ptr,
        gram_partial_ptr,
        rows,
        glob_len,
        code_dim,
        group_start,
        active_bits,
        num_tiles,
        inv_len,
        solve_mode == 0);
    if (solve_mode == 0) {
      blockk_reduce_rhs_float_d128_kernel<GroupBits><<<group_reduce_grid, dim_block, 0, stream>>>(
          partial_ptr,
          rhs_ptr,
          rows,
          num_tiles,
          active_bits);
      blockk_reduce_gram_d128_kernel<GroupBits><<<gram_reduce_grid, 128, 0, stream>>>(
          gram_partial_ptr,
          gram_ptr,
          rows,
          num_tiles,
          active_bits);
      blockk_solve_decoder_d128_kernel<scalar_t, GroupBits><<<rows, 256, 0, stream>>>(
          gram_ptr,
          rhs_ptr,
          decoder_ptr,
          rows,
          code_dim,
          group_start,
          active_bits);
    } else {
      blockk_reduce_decoder_store_d128_kernel<scalar_t, GroupBits><<<group_reduce_grid, dim_block, 0, stream>>>(
          partial_ptr,
          decoder_ptr,
          rows,
          num_tiles,
          code_dim,
          group_start,
          active_bits);
    }
    blockk_update_residual_d128_kernel<scalar_t, GroupBits><<<update_grid, kWarpsPerBlock * 32, 0, stream>>>(
        residual_ptr,
        codes_ptr,
        decoder_ptr,
        err_ptr,
        rows,
        glob_len,
        code_dim,
        group_start,
        active_bits,
        group_start + active_bits >= code_dim);
  }
}

template <typename scalar_t, int LearnTileTokens>
void launch_learn_projection_typed(
    const scalar_t *full_ptr,
    const scalar_t *init_ptr,
    scalar_t *projection_ptr,
    scalar_t *residual_ptr,
    float *partial_ptr,
    float *tmp1_ptr,
    float *tmp2_ptr,
    int8_t *signs_ptr,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int zerorope,
    bool apply_zerorope,
    int num_tiles,
    cudaStream_t stream) {
  const int64_t residual_total = static_cast<int64_t>(rows) * glob_len * kLearnDim;
  const int init_blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(residual_total, kThreads), 65535));
  init_residual_kernel<scalar_t><<<init_blocks, kThreads, 0, stream>>>(
      full_ptr,
      residual_ptr,
      rows,
      total_len,
      glob_start,
      glob_len,
      kLearnDim,
      zerorope,
      apply_zerorope);

  const float inv_len = 1.0f / static_cast<float>(glob_len);
  dim3 partial_grid(static_cast<unsigned int>(num_tiles), rows);
  dim3 vector_block(kLearnWarps * 32);
  dim3 reduce_block(kLearnDim);
  const int num_update_tiles = static_cast<int>(ceil_div_i64(glob_len, kLearnUpdateTileTokens));
  dim3 update_grid(static_cast<unsigned int>(num_update_tiles), rows);

  for (int bit = 0; bit < code_dim; ++bit) {
    learn_ata_vec_partial_d128_kernel<scalar_t, scalar_t, true, LearnTileTokens><<<partial_grid, vector_block, 0, stream>>>(
        residual_ptr,
        init_ptr,
        partial_ptr,
        rows,
        glob_len,
        code_dim,
        bit,
        num_tiles);
    learn_reduce_partial_float_d128_kernel<<<rows, reduce_block, 0, stream>>>(
        partial_ptr,
        tmp1_ptr,
        rows,
        num_tiles);

    // Use one power iteration for the fast learner; this removes one full residual scan per bit.
    learn_sign_vh_partial_d128_kernel<scalar_t, LearnTileTokens><<<partial_grid, vector_block, 0, stream>>>(
        residual_ptr,
        tmp1_ptr,
        signs_ptr,
        partial_ptr,
        rows,
        glob_len,
        num_tiles,
        inv_len);
    learn_reduce_partial_projection_d128_kernel<scalar_t><<<rows, reduce_block, 0, stream>>>(
        partial_ptr,
        projection_ptr,
        rows,
        num_tiles,
        code_dim,
        bit);
    learn_update_residual_from_sign_d128_kernel<scalar_t><<<update_grid, kLearnUpdateThreads, 0, stream>>>(
        residual_ptr,
        signs_ptr,
        projection_ptr,
        rows,
        glob_len,
        code_dim,
        bit,
        num_update_tiles);
  }
}

}  // namespace

std::vector<at::Tensor> quanpc_prefill_learn_projection(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  TORCH_CHECK(full_states.is_cuda(), "full_states must be CUDA");
  TORCH_CHECK(init_vectors.is_cuda(), "init_vectors must be CUDA");
  TORCH_CHECK(full_states.is_contiguous(), "full_states must be contiguous");
  TORCH_CHECK(init_vectors.is_contiguous(), "init_vectors must be contiguous");
  TORCH_CHECK(full_states.dim() == 4, "full_states must be [B,H,L,D]");
  TORCH_CHECK(init_vectors.dim() == 4, "init_vectors must be [B,H,C,D]");
  TORCH_CHECK(full_states.scalar_type() == init_vectors.scalar_type(), "full_states/init_vectors dtype mismatch");
  TORCH_CHECK(full_states.scalar_type() == at::kBFloat16 || full_states.scalar_type() == at::kFloat,
              "projection learning currently supports bf16 and fp32");

  const int B = full_states.size(0);
  const int H = full_states.size(1);
  const int total_len = full_states.size(2);
  const int D = full_states.size(3);
  const int C = init_vectors.size(2);
  TORCH_CHECK(init_vectors.size(0) == B && init_vectors.size(1) == H && init_vectors.size(3) == D,
              "init_vectors shape must match full_states B/H/D");
  TORCH_CHECK(D == kLearnDim, "projection learning CUDA path currently supports head_dim == 128");
  TORCH_CHECK(C > 0 && C % 64 == 0, "code_dim must be positive and divisible by 64");
  TORCH_CHECK(glob_start >= 0 && glob_len > 0 && glob_start + glob_len <= total_len,
              "invalid global slice");
  TORCH_CHECK(zerorope >= 0 && zerorope <= D / 2, "invalid zerorope");

  at::cuda::CUDAGuard device_guard(full_states.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_states.device().index());
  const int rows = B * H;
  const int learn_tile_tokens = (glob_len >= 65536) ? kLearnTileTokensLarge : kLearnTileTokensSmall;
  const int num_tiles = static_cast<int>(ceil_div_i64(glob_len, learn_tile_tokens));

  auto projection = at::empty_like(init_vectors);
  auto residual = at::empty({B, H, glob_len, D}, full_states.options());
  auto float_options = full_states.options().dtype(at::kFloat);
  auto partial = at::empty({rows, num_tiles, D}, float_options);
  auto tmp1 = at::empty({rows, D}, float_options);
  auto tmp2 = at::empty({rows, D}, float_options);
  auto signs = at::empty({rows, glob_len}, full_states.options().dtype(at::kChar));

  if (full_states.scalar_type() == at::kBFloat16) {
    if (learn_tile_tokens == kLearnTileTokensLarge) {
      launch_learn_projection_typed<__nv_bfloat16, kLearnTileTokensLarge>(
          reinterpret_cast<const __nv_bfloat16 *>(full_states.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(init_vectors.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 *>(projection.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 *>(residual.data_ptr<at::BFloat16>()),
          partial.data_ptr<float>(),
          tmp1.data_ptr<float>(),
          tmp2.data_ptr<float>(),
          signs.data_ptr<int8_t>(),
          rows,
          total_len,
          static_cast<int>(glob_start),
          static_cast<int>(glob_len),
          C,
          static_cast<int>(zerorope),
          apply_zerorope,
          num_tiles,
          stream);
    } else {
      launch_learn_projection_typed<__nv_bfloat16, kLearnTileTokensSmall>(
          reinterpret_cast<const __nv_bfloat16 *>(full_states.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(init_vectors.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 *>(projection.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16 *>(residual.data_ptr<at::BFloat16>()),
          partial.data_ptr<float>(),
          tmp1.data_ptr<float>(),
          tmp2.data_ptr<float>(),
          signs.data_ptr<int8_t>(),
          rows,
          total_len,
          static_cast<int>(glob_start),
          static_cast<int>(glob_len),
          C,
          static_cast<int>(zerorope),
          apply_zerorope,
          num_tiles,
          stream);
    }
  } else {
    if (learn_tile_tokens == kLearnTileTokensLarge) {
      launch_learn_projection_typed<float, kLearnTileTokensLarge>(
          full_states.data_ptr<float>(),
          init_vectors.data_ptr<float>(),
          projection.data_ptr<float>(),
          residual.data_ptr<float>(),
          partial.data_ptr<float>(),
          tmp1.data_ptr<float>(),
          tmp2.data_ptr<float>(),
          signs.data_ptr<int8_t>(),
          rows,
          total_len,
          static_cast<int>(glob_start),
          static_cast<int>(glob_len),
          C,
          static_cast<int>(zerorope),
          apply_zerorope,
          num_tiles,
          stream);
    } else {
      launch_learn_projection_typed<float, kLearnTileTokensSmall>(
          full_states.data_ptr<float>(),
          init_vectors.data_ptr<float>(),
          projection.data_ptr<float>(),
          residual.data_ptr<float>(),
          partial.data_ptr<float>(),
          tmp1.data_ptr<float>(),
          tmp2.data_ptr<float>(),
          signs.data_ptr<int8_t>(),
          rows,
          total_len,
          static_cast<int>(glob_start),
          static_cast<int>(glob_len),
          C,
          static_cast<int>(zerorope),
          apply_zerorope,
          num_tiles,
          stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {projection};
}

std::vector<at::Tensor> quanpc_prefill_quantize(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  TORCH_CHECK(full_states.is_cuda(), "full_states must be CUDA");
  TORCH_CHECK(projection.is_cuda(), "projection must be CUDA");
  TORCH_CHECK(full_states.is_contiguous(), "full_states must be contiguous");
  TORCH_CHECK(projection.is_contiguous(), "projection must be contiguous");
  TORCH_CHECK(full_states.dim() == 4, "full_states must be [B,H,L,D]");
  TORCH_CHECK(projection.dim() == 4, "projection must be [B,H,C,D]");
  TORCH_CHECK(full_states.scalar_type() == projection.scalar_type(), "full_states/projection dtype mismatch");
  TORCH_CHECK(full_states.scalar_type() == at::kBFloat16 || full_states.scalar_type() == at::kFloat,
              "prefill quantize currently supports bf16 and fp32");

  const int B = full_states.size(0);
  const int H = full_states.size(1);
  const int total_len = full_states.size(2);
  const int D = full_states.size(3);
  const int C = projection.size(2);
  TORCH_CHECK(projection.size(0) == B && projection.size(1) == H && projection.size(3) == D,
              "projection shape must match full_states B/H/D");
  TORCH_CHECK(C > 0 && C % 64 == 0, "projection C must be positive and divisible by 64");
  TORCH_CHECK(D <= 256, "prefill quantize currently supports head_dim <= 256");
  TORCH_CHECK(glob_start >= 0 && glob_len > 0 && glob_start + glob_len <= total_len,
              "invalid global slice");
  TORCH_CHECK(zerorope >= 0 && zerorope <= D / 2, "invalid zerorope");

  at::cuda::CUDAGuard device_guard(full_states.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_states.device().index());
  const int rows = B * H;

  auto code_options = full_states.options().dtype(at::kChar);
  auto float_options = full_states.options().dtype(at::kFloat);
  auto codes = at::empty({B, H, glob_len, C}, code_options);
  auto err = at::empty({B, H, glob_len}, float_options);

  if (full_states.scalar_type() == at::kBFloat16) {
    launch_quantize_typed<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(full_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(projection.data_ptr<at::BFloat16>()),
        nullptr,
        codes.data_ptr<int8_t>(),
        nullptr,
        err.data_ptr<float>(),
        rows,
        total_len,
        static_cast<int>(glob_start),
        static_cast<int>(glob_len),
        C,
        D,
        static_cast<int>(zerorope),
        apply_zerorope,
        0,
        stream);
  } else {
    launch_quantize_typed<float>(
        full_states.data_ptr<float>(),
        projection.data_ptr<float>(),
        nullptr,
        codes.data_ptr<int8_t>(),
        nullptr,
        err.data_ptr<float>(),
        rows,
        total_len,
        static_cast<int>(glob_start),
        static_cast<int>(glob_len),
        C,
        D,
        static_cast<int>(zerorope),
        apply_zerorope,
        0,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, projection, err};
}


std::vector<at::Tensor> quanpc_prefill_quantize_pack(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  TORCH_CHECK(full_states.is_cuda(), "full_states must be CUDA");
  TORCH_CHECK(projection.is_cuda(), "projection must be CUDA");
  TORCH_CHECK(full_states.is_contiguous(), "full_states must be contiguous");
  TORCH_CHECK(projection.is_contiguous(), "projection must be contiguous");
  TORCH_CHECK(full_states.dim() == 4, "full_states must be [B,H,L,D]");
  TORCH_CHECK(projection.dim() == 4, "projection must be [B,H,C,D]");
  TORCH_CHECK(full_states.scalar_type() == projection.scalar_type(), "full_states/projection dtype mismatch");
  TORCH_CHECK(full_states.scalar_type() == at::kBFloat16 || full_states.scalar_type() == at::kFloat,
              "prefill quantize-pack currently supports bf16 and fp32");

  const int B = full_states.size(0);
  const int H = full_states.size(1);
  const int total_len = full_states.size(2);
  const int D = full_states.size(3);
  const int C = projection.size(2);
  TORCH_CHECK(projection.size(0) == B && projection.size(1) == H && projection.size(3) == D,
              "projection shape must match full_states B/H/D");
  TORCH_CHECK(C > 0 && C % 64 == 0, "projection C must be positive and divisible by 64");
  TORCH_CHECK(D <= 256, "prefill quantize currently supports head_dim <= 256");
  TORCH_CHECK(glob_start >= 0 && glob_len > 0 && glob_start + glob_len <= total_len,
              "invalid global slice");
  TORCH_CHECK(zerorope >= 0 && zerorope <= D / 2, "invalid zerorope");

  at::cuda::CUDAGuard device_guard(full_states.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_states.device().index());
  const int rows = B * H;
  const int blocks_per_code = C / 64;

  auto long_options = full_states.options().dtype(at::kLong);
  auto float_options = full_states.options().dtype(at::kFloat);
  at::Tensor packed;
  if (pack_key_layout) {
    packed = at::empty({B, H, blocks_per_code, glob_len}, long_options);
  } else {
    packed = at::empty({B, H, glob_len, blocks_per_code}, long_options);
  }
  auto err = at::empty({B, H, glob_len}, float_options);
  const int pack_layout = pack_key_layout ? 1 : 2;

  if (full_states.scalar_type() == at::kBFloat16) {
    launch_quantize_typed<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(full_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(projection.data_ptr<at::BFloat16>()),
        nullptr,
        nullptr,
        packed.data_ptr<int64_t>(),
        err.data_ptr<float>(),
        rows,
        total_len,
        static_cast<int>(glob_start),
        static_cast<int>(glob_len),
        C,
        D,
        static_cast<int>(zerorope),
        apply_zerorope,
        pack_layout,
        stream);
  } else {
    launch_quantize_typed<float>(
        full_states.data_ptr<float>(),
        projection.data_ptr<float>(),
        nullptr,
        nullptr,
        packed.data_ptr<int64_t>(),
        err.data_ptr<float>(),
        rows,
        total_len,
        static_cast<int>(glob_start),
        static_cast<int>(glob_len),
        C,
        D,
        static_cast<int>(zerorope),
        apply_zerorope,
        pack_layout,
        stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, projection, err};
}


// Fully fused fixed-projection cascade (stored-decoder path).
//
// With stored encoders AND decoders, the block-32 cascade has no cross-token
// dependency left: the decoder least-squares solve was the only global
// reduction, and it is now a constant.  Each token's group chain
// (score -> sign -> residual update) is independent, so one thread owns one
// token's whole 128-dim residual in registers and walks all groups without
// touching HBM.  This removes the per-group residual read/write traffic that
// both the PyTorch loop and the blockk_fast launcher pay (3 passes/group).
//
// Numerics deliberately replicate the reference PyTorch path's two rounding
// points so the stored decoders stay calibrated to the codes they were fit
// against:
//   1. scores are rounded to bf16 before sign (reference: bf16-output matmul);
//   2. the residual is rounded to bf16 after every group update
//      (reference: baddbmm with bf16 output).
// The fp32 accumulation ORDER differs from cuBLAS, so codes are not
// guaranteed bit-identical: scores within a few fp32 ulps of zero can flip.
// Those flips are quality-neutral (near-zero directions) but mean validation
// must use agreement rate, not torch.equal.  err is a sequential fp32 sum,
// which also differs from torch.linalg.vector_norm's tree order.
constexpr int kFixedDim = 128;
constexpr int kFixedGroupBits = 32;
constexpr int kFixedThreads = 256;

template <typename scalar_t>
__global__ void __launch_bounds__(kFixedThreads)
blockk_fixed_d128_kernel(
    const scalar_t *__restrict__ full_states,
    const scalar_t *__restrict__ encoders,
    const scalar_t *__restrict__ decoders,
    int64_t *__restrict__ packed_out,
    float *__restrict__ err,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int zerorope,
    bool apply_zerorope,
    bool key_layout) {
  const int token = blockIdx.x * kFixedThreads + threadIdx.x;
  const int row = blockIdx.y;
  const bool active = token < glob_len;
  const int groups = code_dim / kFixedGroupBits;
  const int words = code_dim >> 6;
  const int half = kFixedDim / 2;

  __shared__ float e_smem[kFixedGroupBits * kFixedDim];
  __shared__ float d_smem[kFixedGroupBits * kFixedDim];

  // Load this token's residual row once: global slice + zerorope masking,
  // fused here instead of the reference path's clone + in-place zeroing.
  float r[kFixedDim];
  const int64_t src =
      (static_cast<int64_t>(row) * total_len + glob_start + token) * kFixedDim;
  #pragma unroll
  for (int d = 0; d < kFixedDim; ++d) {
    const bool zero =
        apply_zerorope && (d < zerorope || (d >= half && d < half + zerorope));
    r[d] = (active && !zero) ? load_scalar(full_states, src + d) : 0.0f;
  }

  const int64_t mat_row = static_cast<int64_t>(row) * code_dim * kFixedDim;
  uint64_t word = 0;
  for (int g = 0; g < groups; ++g) {
    // Stage this group's encoder/decoder as fp32 in shared memory.  Every
    // thread reads the same addresses during the FMA loops, so these are
    // broadcast (conflict-free) accesses.
    const scalar_t *e_g =
        encoders + mat_row + static_cast<int64_t>(g) * kFixedGroupBits * kFixedDim;
    const scalar_t *d_g =
        decoders + mat_row + static_cast<int64_t>(g) * kFixedGroupBits * kFixedDim;
    for (int i = threadIdx.x; i < kFixedGroupBits * kFixedDim; i += kFixedThreads) {
      e_smem[i] = load_scalar(e_g, i);
      d_smem[i] = load_scalar(d_g, i);
    }
    __syncthreads();

    if (active) {
      // b-outer / d-inner-by-4: one float4 LDS feeds four FMAs, and the 32
      // independent accumulator chains give the scheduler enough ILP to hide
      // FMA latency at 2 warps/scheduler.
      float acc[kFixedGroupBits];
      #pragma unroll
      for (int b = 0; b < kFixedGroupBits; ++b) {
        float a = 0.0f;
        #pragma unroll
        for (int d = 0; d < kFixedDim; d += 4) {
          const float4 e4 =
              *reinterpret_cast<const float4 *>(&e_smem[b * kFixedDim + d]);
          a = fmaf(r[d + 0], e4.x, a);
          a = fmaf(r[d + 1], e4.y, a);
          a = fmaf(r[d + 2], e4.z, a);
          a = fmaf(r[d + 3], e4.w, a);
        }
        acc[b] = a;
      }

      // Strict {-1,+1} with the reference rounding: score through bf16 first,
      // exact zero maps to +1.  Bit packing follows the pack kernels'
      // per-side convention: key sets a bit iff the code is negative, value
      // sets it iff the code is positive.
      float c[kFixedGroupBits];
      #pragma unroll
      for (int b = 0; b < kFixedGroupBits; ++b) {
        const float score = round_like_scalar<scalar_t>(acc[b]);
        c[b] = (score < 0.0f) ? -1.0f : 1.0f;
        const bool bit = key_layout ? (score < 0.0f) : (score > 0.0f);
        if (bit) {
          word |= (uint64_t{1} << ((g & 1) * kFixedGroupBits + b));
        }
      }

      #pragma unroll
      for (int d = 0; d < kFixedDim; d += 4) {
        float4 u = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int b = 0; b < kFixedGroupBits; ++b) {
          const float4 d4 =
              *reinterpret_cast<const float4 *>(&d_smem[b * kFixedDim + d]);
          const float cb = c[b];
          u.x = fmaf(cb, d4.x, u.x);
          u.y = fmaf(cb, d4.y, u.y);
          u.z = fmaf(cb, d4.z, u.z);
          u.w = fmaf(cb, d4.w, u.w);
        }
        r[d + 0] = round_like_scalar<scalar_t>(r[d + 0] - u.x);
        r[d + 1] = round_like_scalar<scalar_t>(r[d + 1] - u.y);
        r[d + 2] = round_like_scalar<scalar_t>(r[d + 2] - u.z);
        r[d + 3] = round_like_scalar<scalar_t>(r[d + 3] - u.w);
      }
    }
    __syncthreads();

    // Two 32-bit groups make one packed uint64 word, written in exactly the
    // layouts pack_key_transposed / pack_value_token produce.
    if ((g & 1) == 1 && active) {
      const int w = g >> 1;
      const int64_t addr =
          key_layout
              ? (static_cast<int64_t>(row) * words + w) * glob_len + token
              : (static_cast<int64_t>(row) * glob_len + token) * words + w;
      packed_out[addr] = static_cast<int64_t>(word);
      word = 0;
    }
  }

  if (active) {
    float norm2 = 0.0f;
    #pragma unroll
    for (int d = 0; d < kFixedDim; ++d) {
      norm2 = fmaf(r[d], r[d], norm2);
    }
    err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
  }
}


// Tensor-core (wmma) variant of the fixed cascade.  The SIMT kernel above is
// issue/latency-bound at ~15 TFLOPS (240 regs/thread -> 1 block/SM); the mma
// path keeps the residual tile in shared memory, runs both per-group matmuls
// on bf16 tensor cores, and reaches 4 blocks/SM.  Numerics are identical in
// structure: bf16 products are exact, accumulation is fp32, scores are
// rounded to bf16 before sign, and the residual is rounded to bf16 after
// every group update.
constexpr int kFixedWmmaTile = 64;
constexpr int kFixedRStride = 136;  // bf16 elements per residual row (16B-multiple)
constexpr int kFixedCStride = 40;   // bf16 elements per code row
constexpr int kFixedYStride = 132;  // fp32 elements per score/update row
constexpr int kFixedSmemBytes =
    kFixedWmmaTile * (kFixedRStride * 2 + kFixedCStride * 2 + kFixedYStride * 4) +
    2 * 32 * kFixedRStride * 2;

__global__ void __launch_bounds__(kFixedThreads)
blockk_fixed_wmma_d128_kernel(
    const __nv_bfloat16 *__restrict__ full_states,
    const __nv_bfloat16 *__restrict__ encoders,
    const __nv_bfloat16 *__restrict__ decoders,
    int64_t *__restrict__ packed_out,
    float *__restrict__ err,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int zerorope,
    bool apply_zerorope,
    bool key_layout,
    int phase_mask) {  // debug ablation: bit0 score, bit1 sign, bit2 update, bit3 residual, bit4 staging
  constexpr int kDim = 128;
  constexpr int kGb = 32;
  const int tile_token = blockIdx.x * kFixedWmmaTile;
  const int row = blockIdx.y;
  const int groups = code_dim / kGb;
  const int words = code_dim >> 6;
  const int warp = threadIdx.x >> 5;

  extern __shared__ unsigned char fixed_smem[];
  __nv_bfloat16 *r_smem = reinterpret_cast<__nv_bfloat16 *>(fixed_smem);
  __nv_bfloat16 *c_smem = r_smem + kFixedWmmaTile * kFixedRStride;
  float *yu_smem = reinterpret_cast<float *>(c_smem + kFixedWmmaTile * kFixedCStride);
  // Per-group staged encoder/decoder tiles.  wmma B-fragment loads from
  // global memory compile to per-lane gathers with L2 latency on the mma
  // critical path; staging them in smem turns them into ldmatrix.
  __nv_bfloat16 *e_smem =
      reinterpret_cast<__nv_bfloat16 *>(yu_smem + kFixedWmmaTile * kFixedYStride);
  __nv_bfloat16 *d_smem = e_smem + kGb * kFixedRStride;

  // Load the residual tile once: global slice + zerorope masking fused.
  for (int i = threadIdx.x; i < kFixedWmmaTile * kDim; i += kFixedThreads) {
    const int t = i >> 7;
    const int d = i & (kDim - 1);
    const int token = tile_token + t;
    __nv_bfloat16 v = __float2bfloat16(0.0f);
    if (token < glob_len) {
      const bool zero =
          apply_zerorope && (d < zerorope || (d >= kDim / 2 && d < kDim / 2 + zerorope));
      if (!zero) {
        v = full_states[
            (static_cast<int64_t>(row) * total_len + glob_start + token) * kDim + d];
      }
    }
    r_smem[t * kFixedRStride + d] = v;
  }
  __syncthreads();

  // Lane 0 of each warp accumulates the packed words of the warp's tokens.
  uint64_t packed_words[kFixedWmmaTile / 8] = {};
  for (int g = 0; g < groups; ++g) {
    const __nv_bfloat16 *e_g =
        encoders + (static_cast<int64_t>(row) * code_dim + g * kGb) * kDim;
    const __nv_bfloat16 *d_g =
        decoders + (static_cast<int64_t>(row) * code_dim + g * kGb) * kDim;
    // Coalesced stage of both 32x128 tiles into padded smem rows.
    if (phase_mask & 16)
    for (int i = threadIdx.x; i < kGb * (kDim / 8); i += kFixedThreads) {
      const int rr = i >> 4;
      const int cc = i & 15;
      reinterpret_cast<uint4 *>(e_smem)[rr * (kFixedRStride / 8) + cc] =
          reinterpret_cast<const uint4 *>(e_g)[i];
      reinterpret_cast<uint4 *>(d_smem)[rr * (kFixedRStride / 8) + cc] =
          reinterpret_cast<const uint4 *>(d_g)[i];
    }
    __syncthreads();

    // Y[64x32] = R[64x128] @ E_g^T;  E_g is [32][128] row-major, i.e. the
    // matrix_b operand in col-major with ldm=128.  Warp w owns output tile
    // (w&3, w>>2) of the 4x2 tile grid.
    if (phase_mask & 1) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      const int tw = (warp & 3) * 16;
      const int tn = (warp >> 2) * 16;
      #pragma unroll
      for (int k = 0; k < kDim; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
        wmma::load_matrix_sync(af, &r_smem[tw * kFixedRStride + k], kFixedRStride);
        wmma::load_matrix_sync(bf, e_smem + tn * kFixedRStride + k, kFixedRStride);
        wmma::mma_sync(acc, af, bf, acc);
      }
      wmma::store_matrix_sync(&yu_smem[tw * kFixedYStride + tn], acc, kFixedYStride,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // Strict {-1,+1} sign with the reference bf16 score rounding; per-side
    // bit convention (key: negative; value: positive).  One warp per 8
    // tokens: lane b reads score b of the token (conflict-free), and
    // __ballot_sync assembles all 32 code bits in one instruction.  Lane 0
    // accumulates each token's word and packs two groups per uint64.
    if (phase_mask & 2) {
      const int lane = threadIdx.x & 31;
      #pragma unroll
      for (int tl = 0; tl < kFixedWmmaTile / 8; ++tl) {
        const int t = warp * (kFixedWmmaTile / 8) + tl;
        const float score = __bfloat162float(
            __float2bfloat16(yu_smem[t * kFixedYStride + lane]));
        c_smem[t * kFixedCStride + lane] =
            __float2bfloat16((score < 0.0f) ? -1.0f : 1.0f);
        const bool bit = key_layout ? (score < 0.0f) : (score > 0.0f);
        const unsigned mask = __ballot_sync(0xffffffffu, bit);
        if (lane == 0) {
          packed_words[tl] |= (static_cast<uint64_t>(mask) << ((g & 1) * kGb));
        }
      }
      if ((g & 1) == 1 && lane == 0) {
        const int w = g >> 1;
        #pragma unroll
        for (int tl = 0; tl < kFixedWmmaTile / 8; ++tl) {
          const int token = tile_token + warp * (kFixedWmmaTile / 8) + tl;
          if (token < glob_len) {
            const int64_t addr =
                key_layout
                    ? (static_cast<int64_t>(row) * words + w) * glob_len + token
                    : (static_cast<int64_t>(row) * glob_len + token) * words + w;
            packed_out[addr] = static_cast<int64_t>(packed_words[tl]);
          }
          packed_words[tl] = 0;
        }
      }
    }
    __syncthreads();

    // U[64x128] = C[64x32] @ D_g[32x128];  32 output tiles, 8 warps x 4 each.
    if (phase_mask & 4)
    for (int tile = warp; tile < 4 * 8; tile += 8) {
      const int tw = (tile & 3) * 16;
      const int tn = (tile >> 2) * 16;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      #pragma unroll
      for (int k = 0; k < kGb; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
        wmma::load_matrix_sync(af, &c_smem[tw * kFixedCStride + k], kFixedCStride);
        wmma::load_matrix_sync(bf, d_smem + k * kFixedRStride + tn, kFixedRStride);
        wmma::mma_sync(acc, af, bf, acc);
      }
      wmma::store_matrix_sync(&yu_smem[tw * kFixedYStride + tn], acc, kFixedYStride,
                              wmma::mem_row_major);
    }
    __syncthreads();

    // R = bf16(R - U), replicating the reference baddbmm rounding point.
    if (phase_mask & 8)
    for (int i = threadIdx.x; i < kFixedWmmaTile * kDim; i += kFixedThreads) {
      const int t = i >> 7;
      const int d = i & (kDim - 1);
      r_smem[t * kFixedRStride + d] = __float2bfloat16(
          __bfloat162float(r_smem[t * kFixedRStride + d]) - yu_smem[t * kFixedYStride + d]);
    }
    __syncthreads();
  }

  if (threadIdx.x < kFixedWmmaTile) {
    const int t = threadIdx.x;
    const int token = tile_token + t;
    if (token < glob_len) {
      float norm2 = 0.0f;
      #pragma unroll
      for (int d = 0; d < kDim; ++d) {
        const float v = __bfloat162float(r_smem[t * kFixedRStride + d]);
        norm2 = fmaf(v, v, norm2);
      }
      err[static_cast<int64_t>(row) * glob_len + token] = sqrtf(fmaxf(norm2, 0.0f));
    }
  }
}


// Fragment-resident fixed cascade.  The wmma kernel above still round-trips
// Y/C/U/R through shared memory every group (~240 KB per 64-token tile per
// group); ablation shows all phases contend superlinearly on the smem/MIO
// pipe.  This version keeps the residual tile in matrix_a fragments
// distributed across lanes for the kernel's whole life, exploiting the
// m16n16k16 layout identity between f32 accumulator fragments and bf16
// matrix_a fragments (the flash-attention mma-chaining trick): sign, bit
// packing, and the residual update all run elementwise in registers, and
// shared memory only stages the per-group encoder/decoder tiles.
//
// Fragment element map (m16n16k16, lane = 4*lg + lc): element i of an
// 8-wide fragment covers (row, col) =
//   i%4 in {0,1}: (lg + 8*(i>>2 & 1)... ) -- concretely:
//   x[0],x[1] -> (lg, 2lc), (lg, 2lc+1)
//   x[2],x[3] -> (lg+8, 2lc), (lg+8, 2lc+1)
//   x[4],x[5] -> (lg, 2lc+8), (lg, 2lc+9)
//   x[6],x[7] -> (lg+8, 2lc+8), (lg+8, 2lc+9)
// identical for f32 accumulators and bf16 A-fragments, which is what makes
// the register-only residual update exact.
constexpr int kFixedFragTile = 128;     // tokens per block
constexpr int kFixedFragThreads = 128;  // 4 warps, 32 rows each

__global__ void __launch_bounds__(kFixedFragThreads)
blockk_fixed_frag_d128_kernel(
    const __nv_bfloat16 *__restrict__ full_states,
    const __nv_bfloat16 *__restrict__ encoders,
    const __nv_bfloat16 *__restrict__ decoders,
    int64_t *__restrict__ packed_out,
    float *__restrict__ err,
    int rows,
    int total_len,
    int glob_start,
    int glob_len,
    int code_dim,
    int zerorope,
    bool apply_zerorope,
    bool key_layout,
    int phase_mask) {
  constexpr int kDim = 128;
  constexpr int kGb = 32;
  const int tile_token = blockIdx.x * kFixedFragTile;
  const int row = blockIdx.y;
  const int groups = code_dim / kGb;
  const int words = code_dim >> 6;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lg = lane >> 2;
  const int lc = lane & 3;

  // Union: residual staging tile during init, then two double-buffered
  // (E,D) staging pairs: 4 * 32 * 136 bf16 = exactly the same 34.8 KB.
  extern __shared__ unsigned char frag_smem[];
  __nv_bfloat16 *stage = reinterpret_cast<__nv_bfloat16 *>(frag_smem);

  for (int i = threadIdx.x; i < kFixedFragTile * kDim; i += kFixedFragThreads) {
    const int t = i >> 7;
    const int d = i & (kDim - 1);
    const int token = tile_token + t;
    __nv_bfloat16 v = __float2bfloat16(0.0f);
    if (token < glob_len) {
      const bool zero =
          apply_zerorope && (d < zerorope || (d >= kDim / 2 && d < kDim / 2 + zerorope));
      if (!zero) {
        v = full_states[
            (static_cast<int64_t>(row) * total_len + glob_start + token) * kDim + d];
      }
    }
    stage[t * 136 + d] = v;
  }
  __syncthreads();

  // Residual fragments: warp owns rows [warp*32, warp*32+32) = 2 bands of 16.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
      r_frag[2][8];
  #pragma unroll
  for (int band = 0; band < 2; ++band) {
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      wmma::load_matrix_sync(
          r_frag[band][k], &stage[(warp * 32 + band * 16) * 136 + k * 16], 136);
    }
  }
  __syncthreads();  // stage[] now serves as the e/d buffers

  // Double-buffered E/D staging with cp.async: the group g+1 tiles are
  // prefetched while group g computes, hiding the global-load latency under
  // the mma work and needing only one barrier per group.
  auto stage_group = [&](int gg, int p) {
    if (phase_mask & 16) {
      const __nv_bfloat16 *e_g =
          encoders + (static_cast<int64_t>(row) * code_dim + gg * kGb) * kDim;
      const __nv_bfloat16 *d_g =
          decoders + (static_cast<int64_t>(row) * code_dim + gg * kGb) * kDim;
      __nv_bfloat16 *e_dst = stage + p * (2 * kGb * 136);
      __nv_bfloat16 *d_dst = e_dst + kGb * 136;
      for (int i = threadIdx.x; i < kGb * (kDim / 8); i += kFixedFragThreads) {
        const int rr = i >> 4;
        const int cc = i & 15;
        __pipeline_memcpy_async(e_dst + rr * 136 + cc * 8, e_g + i * 8, 16);
        __pipeline_memcpy_async(d_dst + rr * 136 + cc * 8, d_g + i * 8, 16);
      }
    }
    __pipeline_commit();
  };

  uint64_t pword[2][2] = {};  // [band][row half]: packed bits across 2 groups
  stage_group(0, 0);
  for (int g = 0; g < groups; ++g) {
    __pipeline_wait_prior(0);
    __syncthreads();
    if (g + 1 < groups) {
      stage_group(g + 1, (g + 1) & 1);
    }
    const __nv_bfloat16 *e_smem = stage + (g & 1) * (2 * kGb * 136);
    const __nv_bfloat16 *d_smem = e_smem + kGb * 136;

    #pragma unroll
    for (int band = 0; band < 2; ++band) {
      // Y[16x32] = R_band[16x128] @ E_g^T
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
      if (phase_mask & 1) {
        #pragma unroll
        for (int ct = 0; ct < 2; ++ct) {
          wmma::fill_fragment(acc[ct], 0.0f);
          #pragma unroll
          for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(bf, e_smem + ct * 16 * 136 + k * 16, 136);
            wmma::mma_sync(acc[ct], r_frag[band][k], bf, acc[ct]);
          }
        }
      }

      // Sign in registers (reference bf16 score rounding), build C fragments
      // for the update mma, and assemble this group's 32 code bits per row.
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
          c_frag[2];
      uint32_t bits0 = 0;  // row lg of this band
      uint32_t bits1 = 0;  // row lg+8
      if (phase_mask & 2) {
        #pragma unroll
        for (int ct = 0; ct < 2; ++ct) {
          __nv_bfloat16 *cx = reinterpret_cast<__nv_bfloat16 *>(c_frag[ct].x);
          #pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float score =
                __bfloat162float(__float2bfloat16(acc[ct].x[i]));
            cx[i] = __float2bfloat16((score < 0.0f) ? -1.0f : 1.0f);
            const bool bit = key_layout ? (score < 0.0f) : (score > 0.0f);
            if (bit) {
              // col within the 32-wide code row of this group; x[0,1,4,5]
              // belong to row lg, x[2,3,6,7] to row lg+8.
              const int col = ct * 16 + (lc * 2) + (i & 1) + ((i >> 2) << 3);
              if ((i & 2) == 0) {
                bits0 |= (uint32_t{1} << col);
              } else {
                bits1 |= (uint32_t{1} << col);
              }
            }
          }
        }
        bits0 |= __shfl_xor_sync(0xffffffffu, bits0, 1);
        bits0 |= __shfl_xor_sync(0xffffffffu, bits0, 2);
        bits1 |= __shfl_xor_sync(0xffffffffu, bits1, 1);
        bits1 |= __shfl_xor_sync(0xffffffffu, bits1, 2);
      }

      // R_band = bf16(R_band - C @ D_g), elementwise on fragments.
      if (phase_mask & 4) {
        #pragma unroll
        for (int ct = 0; ct < 8; ++ct) {
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> uacc;
          wmma::fill_fragment(uacc, 0.0f);
          #pragma unroll
          for (int k = 0; k < 2; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
            wmma::load_matrix_sync(bf, d_smem + k * 16 * 136 + ct * 16, 136);
            wmma::mma_sync(uacc, c_frag[k], bf, uacc);
          }
          __nv_bfloat16 *rx = reinterpret_cast<__nv_bfloat16 *>(r_frag[band][ct].x);
          #pragma unroll
          for (int i = 0; i < 8; ++i) {
            rx[i] = __float2bfloat16(__bfloat162float(rx[i]) - uacc.x[i]);
          }
        }
      }

      // Pack two groups per uint64 word; lane lc==0 of each quad writes.
      if (phase_mask & 2) {
        if ((g & 1) == 0) {
          pword[band][0] = (pword[band][0] & ~uint64_t{0xffffffff}) | bits0;
          pword[band][1] = (pword[band][1] & ~uint64_t{0xffffffff}) | bits1;
        } else {
          pword[band][0] |= static_cast<uint64_t>(bits0) << 32;
          pword[band][1] |= static_cast<uint64_t>(bits1) << 32;
          if (lc == 0) {
            const int w = g >> 1;
            const int tok0 = tile_token + warp * 32 + band * 16 + lg;
            const int tok1 = tok0 + 8;
            if (key_layout) {
              const int64_t base =
                  (static_cast<int64_t>(row) * words + w) * glob_len;
              if (tok0 < glob_len) packed_out[base + tok0] = static_cast<int64_t>(pword[band][0]);
              if (tok1 < glob_len) packed_out[base + tok1] = static_cast<int64_t>(pword[band][1]);
            } else {
              if (tok0 < glob_len)
                packed_out[(static_cast<int64_t>(row) * glob_len + tok0) * words + w] =
                    static_cast<int64_t>(pword[band][0]);
              if (tok1 < glob_len)
                packed_out[(static_cast<int64_t>(row) * glob_len + tok1) * words + w] =
                    static_cast<int64_t>(pword[band][1]);
            }
          }
          pword[band][0] = 0;
          pword[band][1] = 0;
        }
      }
    }
  }

  // Per-token residual norm from the fragments: lane holds 4 elements of row
  // lg and 4 of row lg+8 per fragment; butterfly-reduce within the quad.
  // Norms are per band -- accumulating across bands would mix two tokens.
  #pragma unroll
  for (int band = 0; band < 2; ++band) {
    float n0 = 0.0f, n1 = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const __nv_bfloat16 *rx =
          reinterpret_cast<const __nv_bfloat16 *>(r_frag[band][k].x);
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float v = __bfloat162float(rx[i]);
        // x[0,1,4,5] -> row lg, x[2,3,6,7] -> row lg+8
        if ((i & 2) == 0) {
          n0 = fmaf(v, v, n0);
        } else {
          n1 = fmaf(v, v, n1);
        }
      }
    }
    n0 += __shfl_xor_sync(0xffffffffu, n0, 1);
    n0 += __shfl_xor_sync(0xffffffffu, n0, 2);
    n1 += __shfl_xor_sync(0xffffffffu, n1, 1);
    n1 += __shfl_xor_sync(0xffffffffu, n1, 2);
    if (lc == 0) {
      const int tok0 = tile_token + warp * 32 + band * 16 + lg;
      if (tok0 < glob_len)
        err[static_cast<int64_t>(row) * glob_len + tok0] = sqrtf(fmaxf(n0, 0.0f));
      if (tok0 + 8 < glob_len)
        err[static_cast<int64_t>(row) * glob_len + tok0 + 8] = sqrtf(fmaxf(n1, 0.0f));
    }
  }
}

std::vector<at::Tensor> quanpc_prefill_blockk_fixed(
    at::Tensor full_states,
    at::Tensor encoders,
    at::Tensor decoders,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  TORCH_CHECK(full_states.is_cuda() && encoders.is_cuda() && decoders.is_cuda(),
              "full_states/encoders/decoders must be CUDA");
  TORCH_CHECK(full_states.is_contiguous() && encoders.is_contiguous() && decoders.is_contiguous(),
              "full_states/encoders/decoders must be contiguous");
  TORCH_CHECK(full_states.dim() == 4, "full_states must be [B,H,L,D]");
  TORCH_CHECK(encoders.dim() == 4 && decoders.dim() == 4,
              "encoders/decoders must be [B,H,C,D]");
  TORCH_CHECK(full_states.scalar_type() == encoders.scalar_type() &&
                  full_states.scalar_type() == decoders.scalar_type(),
              "full_states/encoders/decoders dtype mismatch");
  TORCH_CHECK(full_states.scalar_type() == at::kBFloat16 || full_states.scalar_type() == at::kFloat,
              "blockk_fixed currently supports bf16 and fp32");

  const int B = full_states.size(0);
  const int H = full_states.size(1);
  const int total_len = full_states.size(2);
  const int D = full_states.size(3);
  const int C = encoders.size(2);
  TORCH_CHECK(encoders.size(0) == B && encoders.size(1) == H && encoders.size(3) == D &&
                  decoders.size(0) == B && decoders.size(1) == H &&
                  decoders.size(2) == C && decoders.size(3) == D,
              "encoders/decoders shape must match full_states B/H/D and share C");
  TORCH_CHECK(D == kFixedDim, "blockk_fixed currently supports head_dim == 128");
  TORCH_CHECK(C > 0 && C % 64 == 0, "code_dim must be positive and divisible by 64");
  TORCH_CHECK(glob_start >= 0 && glob_len > 0 && glob_start + glob_len <= total_len,
              "invalid global slice");
  TORCH_CHECK(zerorope >= 0 && zerorope <= D / 2, "invalid zerorope");

  at::cuda::CUDAGuard device_guard(full_states.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_states.device().index());
  const int rows = B * H;
  const int words = C >> 6;

  auto packed = pack_key_layout
      ? at::empty({B, H, words, static_cast<int>(glob_len)}, full_states.options().dtype(at::kLong))
      : at::empty({B, H, static_cast<int>(glob_len), words}, full_states.options().dtype(at::kLong));
  auto err = at::empty({B, H, static_cast<int>(glob_len)}, full_states.options().dtype(at::kFloat));

  dim3 grid(static_cast<unsigned int>(ceil_div_i64(glob_len, kFixedThreads)), rows);
  if (full_states.scalar_type() == at::kBFloat16) {
    static bool wmma_smem_ready = false;
    if (!wmma_smem_ready) {
      cudaFuncSetAttribute(blockk_fixed_wmma_d128_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kFixedSmemBytes);
      wmma_smem_ready = true;
    }
    static const int phase_mask = [] {
      const char *s = getenv("QUANPC_FIXED_PHASE_MASK");
      return s ? atoi(s) : 0x1f;
    }();
    constexpr int kFragSmemBytes = kFixedFragTile * 136 * 2;
    static bool frag_smem_configured = false;
    if (!frag_smem_configured) {
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          blockk_fixed_frag_d128_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          kFragSmemBytes));
      frag_smem_configured = true;
    }
    dim3 wmma_grid(
        static_cast<unsigned int>(ceil_div_i64(glob_len, kFixedFragTile)), rows);
    blockk_fixed_frag_d128_kernel<<<wmma_grid, kFixedFragThreads, kFragSmemBytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(full_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(encoders.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(decoders.data_ptr<at::BFloat16>()),
        packed.data_ptr<int64_t>(),
        err.data_ptr<float>(),
        rows,
        total_len,
        static_cast<int>(glob_start),
        static_cast<int>(glob_len),
        C,
        static_cast<int>(zerorope),
        apply_zerorope,
        pack_key_layout,
        phase_mask);
  } else {
    blockk_fixed_d128_kernel<float><<<grid, kFixedThreads, 0, stream>>>(
        full_states.data_ptr<float>(),
        encoders.data_ptr<float>(),
        decoders.data_ptr<float>(),
        packed.data_ptr<int64_t>(),
        err.data_ptr<float>(),
        rows,
        total_len,
        static_cast<int>(glob_start),
        static_cast<int>(glob_len),
        C,
        static_cast<int>(zerorope),
        apply_zerorope,
        pack_key_layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, err};
}

std::vector<at::Tensor> quanpc_prefill_blockk_fast(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    int64_t group_bits,
    int64_t power_iters,
    int64_t solve_mode) {
  TORCH_CHECK(full_states.is_cuda(), "full_states must be CUDA");
  TORCH_CHECK(init_vectors.is_cuda(), "init_vectors must be CUDA");
  TORCH_CHECK(full_states.is_contiguous(), "full_states must be contiguous");
  TORCH_CHECK(init_vectors.is_contiguous(), "init_vectors must be contiguous");
  TORCH_CHECK(full_states.dim() == 4, "full_states must be [B,H,L,D]");
  TORCH_CHECK(init_vectors.dim() == 4, "init_vectors must be [B,H,C,D]");
  TORCH_CHECK(full_states.scalar_type() == init_vectors.scalar_type(), "full_states/init_vectors dtype mismatch");
  TORCH_CHECK(full_states.scalar_type() == at::kBFloat16 || full_states.scalar_type() == at::kFloat,
              "block-k fast currently supports bf16 and fp32");

  const int B = full_states.size(0);
  const int H = full_states.size(1);
  const int total_len = full_states.size(2);
  const int D = full_states.size(3);
  const int C = init_vectors.size(2);
  TORCH_CHECK(init_vectors.size(0) == B && init_vectors.size(1) == H && init_vectors.size(3) == D,
              "init_vectors shape must match full_states B/H/D");
  TORCH_CHECK(D == kLearnDim, "block-k fast currently supports head_dim == 128");
  TORCH_CHECK(C > 0 && C % 64 == 0, "code_dim must be positive and divisible by 64");
  TORCH_CHECK(group_bits == 16 || group_bits == 32, "group_bits must be 16 or 32");
  TORCH_CHECK(solve_mode == 0 || solve_mode == 1, "solve_mode must be 0/full or 1/diagonal");
  TORCH_CHECK(C % group_bits == 0, "code_dim must be divisible by group_bits");
  TORCH_CHECK(glob_start >= 0 && glob_len > 0 && glob_start + glob_len <= total_len,
              "invalid global slice");
  TORCH_CHECK(zerorope >= 0 && zerorope <= D / 2, "invalid zerorope");

  at::cuda::CUDAGuard device_guard(full_states.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_states.device().index());
  const int rows = B * H;
  const int learn_tile_tokens = (glob_len >= 65536) ? kLearnTileTokensLarge : kLearnTileTokensSmall;
  const int num_tiles = static_cast<int>(ceil_div_i64(glob_len, learn_tile_tokens));

  auto code_options = full_states.options().dtype(at::kChar);
  auto float_options = full_states.options().dtype(at::kFloat);
  auto codes = at::empty({B, H, glob_len, C}, code_options);
  auto encoder = at::empty_like(init_vectors);
  auto decoder = at::empty_like(init_vectors);
  auto residual = at::empty({B, H, glob_len, D}, full_states.options());
  auto partial = at::empty({rows, num_tiles, static_cast<int>(group_bits), D}, float_options);
  auto gram_partial = at::empty({rows, num_tiles, static_cast<int>(group_bits), static_cast<int>(group_bits)}, float_options);
  auto rhs = at::empty({rows, static_cast<int>(group_bits), D}, float_options);
  auto gram = at::empty({rows, static_cast<int>(group_bits), static_cast<int>(group_bits)}, float_options);
  auto err = at::empty({B, H, glob_len}, float_options);

  if (full_states.scalar_type() == at::kBFloat16) {
    const auto *full_ptr = reinterpret_cast<const __nv_bfloat16 *>(full_states.data_ptr<at::BFloat16>());
    const auto *init_ptr = reinterpret_cast<const __nv_bfloat16 *>(init_vectors.data_ptr<at::BFloat16>());
    auto *encoder_ptr = reinterpret_cast<__nv_bfloat16 *>(encoder.data_ptr<at::BFloat16>());
    auto *decoder_ptr = reinterpret_cast<__nv_bfloat16 *>(decoder.data_ptr<at::BFloat16>());
    auto *residual_ptr = reinterpret_cast<__nv_bfloat16 *>(residual.data_ptr<at::BFloat16>());
    if (group_bits == 16) {
      if (learn_tile_tokens == kLearnTileTokensLarge) {
        launch_blockk_fast_typed<__nv_bfloat16, 16, kLearnTileTokensLarge>(
            full_ptr, init_ptr, codes.data_ptr<int8_t>(), encoder_ptr, decoder_ptr, residual_ptr,
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      } else {
        launch_blockk_fast_typed<__nv_bfloat16, 16, kLearnTileTokensSmall>(
            full_ptr, init_ptr, codes.data_ptr<int8_t>(), encoder_ptr, decoder_ptr, residual_ptr,
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      }
    } else {
      if (learn_tile_tokens == kLearnTileTokensLarge) {
        launch_blockk_fast_typed<__nv_bfloat16, 32, kLearnTileTokensLarge>(
            full_ptr, init_ptr, codes.data_ptr<int8_t>(), encoder_ptr, decoder_ptr, residual_ptr,
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      } else {
        launch_blockk_fast_typed<__nv_bfloat16, 32, kLearnTileTokensSmall>(
            full_ptr, init_ptr, codes.data_ptr<int8_t>(), encoder_ptr, decoder_ptr, residual_ptr,
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      }
    }
  } else {
    if (group_bits == 16) {
      if (learn_tile_tokens == kLearnTileTokensLarge) {
        launch_blockk_fast_typed<float, 16, kLearnTileTokensLarge>(
            full_states.data_ptr<float>(), init_vectors.data_ptr<float>(), codes.data_ptr<int8_t>(),
            encoder.data_ptr<float>(), decoder.data_ptr<float>(), residual.data_ptr<float>(),
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      } else {
        launch_blockk_fast_typed<float, 16, kLearnTileTokensSmall>(
            full_states.data_ptr<float>(), init_vectors.data_ptr<float>(), codes.data_ptr<int8_t>(),
            encoder.data_ptr<float>(), decoder.data_ptr<float>(), residual.data_ptr<float>(),
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      }
    } else {
      if (learn_tile_tokens == kLearnTileTokensLarge) {
        launch_blockk_fast_typed<float, 32, kLearnTileTokensLarge>(
            full_states.data_ptr<float>(), init_vectors.data_ptr<float>(), codes.data_ptr<int8_t>(),
            encoder.data_ptr<float>(), decoder.data_ptr<float>(), residual.data_ptr<float>(),
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      } else {
        launch_blockk_fast_typed<float, 32, kLearnTileTokensSmall>(
            full_states.data_ptr<float>(), init_vectors.data_ptr<float>(), codes.data_ptr<int8_t>(),
            encoder.data_ptr<float>(), decoder.data_ptr<float>(), residual.data_ptr<float>(),
            partial.data_ptr<float>(), gram_partial.data_ptr<float>(), rhs.data_ptr<float>(), gram.data_ptr<float>(),
            err.data_ptr<float>(), rows, total_len,
            static_cast<int>(glob_start), static_cast<int>(glob_len), C, static_cast<int>(zerorope),
            apply_zerorope, static_cast<int>(power_iters), static_cast<int>(solve_mode), num_tiles, stream);
      }
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, encoder, decoder, err};
}

std::vector<at::Tensor> quanpc_prefill_select_eas(
    at::Tensor key_err,
    at::Tensor value_err,
    int64_t eas_size) {
  TORCH_CHECK(key_err.is_cuda() && value_err.is_cuda(), "errors must be CUDA");
  TORCH_CHECK(key_err.is_contiguous() && value_err.is_contiguous(), "errors must be contiguous");
  TORCH_CHECK(key_err.scalar_type() == at::kFloat && value_err.scalar_type() == at::kFloat,
              "errors must be fp32");
  TORCH_CHECK(key_err.sizes() == value_err.sizes(), "key/value error shape mismatch");
  TORCH_CHECK(key_err.dim() == 3, "errors must be [B,H,L]");
  const int B = key_err.size(0);
  const int H = key_err.size(1);
  const int L = key_err.size(2);
  TORCH_CHECK(L > 0, "global length must be positive");
  TORCH_CHECK(eas_size > 0, "eas_size must be positive");

  const int rows = B * H;
  const int k_count = static_cast<int>(std::min<int64_t>(eas_size, L));
  const int v_count = (k_count == L) ? 0 : static_cast<int>(std::min<int64_t>(eas_size, L - k_count));
  const int eas_count = k_count + v_count;
  const int kept_len = L - eas_count;
  TORCH_CHECK(eas_count > 0, "eas_count must be positive");

  at::cuda::CUDAGuard device_guard(key_err.device());
  auto stream = at::cuda::getCurrentCUDAStream(key_err.device().index());
  auto int_options = key_err.options().dtype(at::kInt);
  auto byte_options = key_err.options().dtype(at::kByte);

  auto offsets = at::empty({rows + 1}, int_options);
  auto idx = at::empty({rows, L}, int_options);
  const int64_t total = static_cast<int64_t>(rows) * L;
  int blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(std::max<int64_t>(total, rows + 1), kThreads), 65535));
  init_indices_offsets_kernel<<<blocks, kThreads, 0, stream>>>(
      idx.data_ptr<int>(),
      offsets.data_ptr<int>(),
      rows,
      L);

  auto key_sorted_err = at::empty_like(key_err);
  auto key_sorted_idx = at::empty_like(idx);
  const size_t pair_temp_bytes = segmented_sort_pairs_desc_temp_bytes(
      key_err, key_sorted_err, idx, key_sorted_idx, offsets, rows, static_cast<int>(total), stream);
  auto pair_temp = at::empty({static_cast<int64_t>(pair_temp_bytes)}, byte_options);
  segmented_sort_pairs_desc_with_temp(
      key_err, key_sorted_err, idx, key_sorted_idx, offsets, pair_temp, pair_temp_bytes, rows, static_cast<int>(total), stream);

  auto selected_mask = at::empty({rows, L}, byte_options);
  AT_CUDA_CHECK(cudaMemsetAsync(selected_mask.data_ptr<uint8_t>(), 0, static_cast<size_t>(total), stream));
  mark_topk_kernel<<<rows, kThreads, 0, stream>>>(
      key_sorted_idx.data_ptr<int>(),
      selected_mask.data_ptr<uint8_t>(),
      rows,
      L,
      k_count);

  auto value_score = at::empty_like(value_err);
  make_value_scores_kernel<<<blocks, kThreads, 0, stream>>>(
      value_err.data_ptr<float>(),
      selected_mask.data_ptr<uint8_t>(),
      value_score.data_ptr<float>(),
      idx.data_ptr<int>(),
      rows,
      L);

  auto value_sorted_err = key_sorted_err;
  auto value_sorted_idx = at::empty_like(idx);
  segmented_sort_pairs_desc_with_temp(
      value_score, value_sorted_err, idx, value_sorted_idx, offsets, pair_temp, pair_temp_bytes, rows, static_cast<int>(total), stream);

  auto eas_unsorted = at::empty({rows, eas_count}, int_options);
  finalize_eas_kernel<<<rows, kThreads, 0, stream>>>(
      key_sorted_idx.data_ptr<int>(),
      value_sorted_idx.data_ptr<int>(),
      selected_mask.data_ptr<uint8_t>(),
      eas_unsorted.data_ptr<int>(),
      rows,
      L,
      k_count,
      v_count,
      eas_count);

  blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(rows + 1, kThreads), 65535));
  make_eas_offsets_kernel<<<blocks, kThreads, 0, stream>>>(offsets.data_ptr<int>(), rows, eas_count);
  auto eas_indices = at::empty_like(eas_unsorted);
  const int eas_total = rows * eas_count;
  const size_t eas_temp_bytes = segmented_sort_keys_asc_temp_bytes(
      eas_unsorted, eas_indices, offsets, rows, eas_total, stream);
  auto eas_temp = at::empty({static_cast<int64_t>(eas_temp_bytes)}, byte_options);
  segmented_sort_keys_asc_with_temp(
      eas_unsorted, eas_indices, offsets, eas_temp, eas_temp_bytes, rows, eas_total, stream);

  auto kept_indices = at::empty({rows, kept_len}, int_options);
  if (kept_len > 0) {
    build_kept_from_sorted_eas_kernel<<<rows, 256, 0, stream>>>(
        eas_indices.data_ptr<int>(),
        kept_indices.data_ptr<int>(),
        rows,
        L,
        eas_count,
        kept_len);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      eas_indices.view({B, H, eas_count}),
      selected_mask.view({B, H, L}),
      kept_indices.view({B, H, kept_len}),
  };
}


__global__ void pack_key_transposed_i8_kernel(
    const int8_t *codes,
    int64_t *packed_key,
    int rows,
    int L,
    int C,
    int blocks_per_code) {
  const int64_t total = static_cast<int64_t>(rows) * blocks_per_code * L;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int l = idx % L;
    const int block_id = (idx / L) % blocks_per_code;
    const int row = idx / (static_cast<int64_t>(L) * blocks_per_code);
    const int64_t code_base = (static_cast<int64_t>(row) * L + l) * C + block_id * 64;
    uint64_t word = 0;
#pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
      if (codes[code_base + bit] < 0) {
        word |= (uint64_t{1} << bit);
      }
    }
    packed_key[idx] = static_cast<int64_t>(word);
  }
}

__global__ void pack_value_token_i8_kernel(
    const int8_t *codes,
    int64_t *packed_value,
    int rows,
    int L,
    int C,
    int blocks_per_code) {
  const int64_t total = static_cast<int64_t>(rows) * L * blocks_per_code;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int block_id = idx % blocks_per_code;
    const int l = (idx / blocks_per_code) % L;
    const int row = idx / (static_cast<int64_t>(blocks_per_code) * L);
    const int64_t code_base = (static_cast<int64_t>(row) * L + l) * C + block_id * 64;
    uint64_t word = 0;
#pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
      if (codes[code_base + bit] > 0) {
        word |= (uint64_t{1} << bit);
      }
    }
    packed_value[idx] = static_cast<int64_t>(word);
  }
}

at::Tensor quanpc_prefill_pack_key_transposed(at::Tensor codes) {
  TORCH_CHECK(codes.is_cuda(), "key codes must be CUDA");
  TORCH_CHECK(codes.is_contiguous(), "key codes must be contiguous");
  TORCH_CHECK(codes.scalar_type() == at::kChar, "key codes must be int8");
  TORCH_CHECK(codes.dim() == 4, "key codes must be [B,H,L,C]");
  const int B = codes.size(0);
  const int H = codes.size(1);
  const int L = codes.size(2);
  const int C = codes.size(3);
  TORCH_CHECK(C % 64 == 0, "key code dim must be divisible by 64");
  const int rows = B * H;
  const int blocks_per_code = C / 64;
  at::cuda::CUDAGuard device_guard(codes.device());
  auto out = at::empty({B, H, blocks_per_code, L}, codes.options().dtype(at::kLong));
  const int64_t total = static_cast<int64_t>(rows) * blocks_per_code * L;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(total, threads), 65535));
  auto stream = at::cuda::getCurrentCUDAStream(codes.device().index());
  pack_key_transposed_i8_kernel<<<blocks, threads, 0, stream>>>(
      codes.data_ptr<int8_t>(),
      out.data_ptr<int64_t>(),
      rows,
      L,
      C,
      blocks_per_code);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor quanpc_prefill_pack_value_token(at::Tensor codes) {
  TORCH_CHECK(codes.is_cuda(), "value codes must be CUDA");
  TORCH_CHECK(codes.is_contiguous(), "value codes must be contiguous");
  TORCH_CHECK(codes.scalar_type() == at::kChar, "value codes must be int8");
  TORCH_CHECK(codes.dim() == 4, "value codes must be [B,H,L,C]");
  const int B = codes.size(0);
  const int H = codes.size(1);
  const int L = codes.size(2);
  const int C = codes.size(3);
  TORCH_CHECK(C % 64 == 0, "value code dim must be divisible by 64");
  const int rows = B * H;
  const int blocks_per_code = C / 64;
  at::cuda::CUDAGuard device_guard(codes.device());
  auto out = at::empty({B, H, L, blocks_per_code}, codes.options().dtype(at::kLong));
  const int64_t total = static_cast<int64_t>(rows) * L * blocks_per_code;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(total, threads), 65535));
  auto stream = at::cuda::getCurrentCUDAStream(codes.device().index());
  pack_value_token_i8_kernel<<<blocks, threads, 0, stream>>>(
      codes.data_ptr<int8_t>(),
      out.data_ptr<int64_t>(),
      rows,
      L,
      C,
      blocks_per_code);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> quanpc_prefill_gather_pack(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor key_codes,
    at::Tensor value_codes,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  TORCH_CHECK(full_key.is_cuda() && full_value.is_cuda(), "full key/value must be CUDA");
  TORCH_CHECK(key_codes.is_cuda() && value_codes.is_cuda(), "codes must be CUDA");
  TORCH_CHECK(eas_indices.is_cuda() && kept_indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(full_key.is_contiguous() && full_value.is_contiguous(), "full key/value must be contiguous");
  TORCH_CHECK(key_codes.is_contiguous() && value_codes.is_contiguous(), "codes must be contiguous");
  TORCH_CHECK(eas_indices.is_contiguous() && kept_indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(full_key.scalar_type() == full_value.scalar_type(), "full key/value dtype mismatch");
  TORCH_CHECK(full_key.scalar_type() == at::kBFloat16 || full_key.scalar_type() == at::kFloat,
              "gather currently supports bf16 and fp32");
  TORCH_CHECK(key_codes.scalar_type() == at::kChar && value_codes.scalar_type() == at::kChar,
              "codes must be int8");
  TORCH_CHECK(eas_indices.scalar_type() == at::kInt && kept_indices.scalar_type() == at::kInt,
              "indices must be int32");
  TORCH_CHECK(full_key.dim() == 4 && full_value.dim() == 4, "full key/value must be [B,H,L,D]");
  TORCH_CHECK(key_codes.dim() == 4 && value_codes.dim() == 4, "codes must be [B,H,Lglob,C]");
  TORCH_CHECK(eas_indices.dim() == 3 && kept_indices.dim() == 3, "indices must be [B,H,N]");

  const int B = full_key.size(0);
  const int H = full_key.size(1);
  const int total_len = full_key.size(2);
  const int D = full_key.size(3);
  TORCH_CHECK(full_value.size(0) == B && full_value.size(1) == H && full_value.size(2) == total_len && full_value.size(3) == D,
              "full value shape mismatch");
  TORCH_CHECK(key_codes.size(0) == B && key_codes.size(1) == H, "key code B/H mismatch");
  TORCH_CHECK(value_codes.size(0) == B && value_codes.size(1) == H, "value code B/H mismatch");
  const int glob_len = key_codes.size(2);
  const int Ck = key_codes.size(3);
  const int Cv = value_codes.size(3);
  TORCH_CHECK(value_codes.size(2) == glob_len, "K/V code global length mismatch");
  TORCH_CHECK(Ck > 0 && Ck % 64 == 0, "key code dim must be divisible by 64");
  TORCH_CHECK(sink_size >= 0 && local_size >= 0 && sink_size + local_size + glob_len == total_len,
              "sink/local/global lengths do not match full length");
  TORCH_CHECK(eas_indices.size(0) == B && eas_indices.size(1) == H, "eas index B/H mismatch");
  TORCH_CHECK(kept_indices.size(0) == B && kept_indices.size(1) == H, "kept index B/H mismatch");

  const int rows = B * H;
  const int eas_count = eas_indices.size(2);
  const int kept_len = kept_indices.size(2);
  TORCH_CHECK(eas_count + kept_len == glob_len, "EAS + kept length must equal global length");
  const int dense_len = static_cast<int>(sink_size) + eas_count + static_cast<int>(local_size);
  const int key_blocks = Ck / 64;
  const int value_groups = static_cast<int>(ceil_div_i64(kept_len, 64));

  at::cuda::CUDAGuard device_guard(full_key.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_key.device().index());
  auto long_options = full_key.options().dtype(at::kLong);
  auto dense_key = at::empty({B, H, dense_len, D}, full_key.options());
  auto dense_value = at::empty({B, H, dense_len, D}, full_value.options());
  auto packed_key = at::empty({B, H, key_blocks, kept_len}, long_options);
  auto packed_value = at::empty({B, H, value_groups, Cv}, long_options);

  const int64_t dense_total = static_cast<int64_t>(rows) * dense_len * D;
  int blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(dense_total, kThreads), 65535));
  if (full_key.scalar_type() == at::kBFloat16) {
    gather_dense_kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(full_key.data_ptr<at::BFloat16>()),
        eas_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(dense_key.data_ptr<at::BFloat16>()),
        rows,
        total_len,
        D,
        static_cast<int>(sink_size),
        static_cast<int>(local_size),
        eas_count,
        dense_len);
    gather_dense_kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(full_value.data_ptr<at::BFloat16>()),
        eas_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(dense_value.data_ptr<at::BFloat16>()),
        rows,
        total_len,
        D,
        static_cast<int>(sink_size),
        static_cast<int>(local_size),
        eas_count,
        dense_len);
  } else {
    gather_dense_kernel<float><<<blocks, kThreads, 0, stream>>>(
        full_key.data_ptr<float>(),
        eas_indices.data_ptr<int>(),
        dense_key.data_ptr<float>(),
        rows,
        total_len,
        D,
        static_cast<int>(sink_size),
        static_cast<int>(local_size),
        eas_count,
        dense_len);
    gather_dense_kernel<float><<<blocks, kThreads, 0, stream>>>(
        full_value.data_ptr<float>(),
        eas_indices.data_ptr<int>(),
        dense_value.data_ptr<float>(),
        rows,
        total_len,
        D,
        static_cast<int>(sink_size),
        static_cast<int>(local_size),
        eas_count,
        dense_len);
  }

  if (kept_len > 0) {
    const int64_t key_total = static_cast<int64_t>(rows) * key_blocks * kept_len;
    blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(key_total, kThreads), 65535));
    pack_key_kept_kernel<<<blocks, kThreads, 0, stream>>>(
        key_codes.data_ptr<int8_t>(),
        kept_indices.data_ptr<int>(),
        packed_key.data_ptr<int64_t>(),
        rows,
        glob_len,
        Ck,
        kept_len,
        key_blocks);

    const int64_t value_total = static_cast<int64_t>(rows) * value_groups * Cv;
    blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(value_total, kThreads), 65535));
    pack_value_kept_kernel<<<blocks, kThreads, 0, stream>>>(
        value_codes.data_ptr<int8_t>(),
        kept_indices.data_ptr<int>(),
        packed_value.data_ptr<int64_t>(),
        rows,
        glob_len,
        Cv,
        kept_len,
        value_groups);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dense_key, dense_value, packed_key, packed_value};
}


std::vector<at::Tensor> quanpc_prefill_gather_pack_from_packed(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor packed_key_all,
    at::Tensor packed_value_token,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  TORCH_CHECK(full_key.is_cuda() && full_value.is_cuda(), "full key/value must be CUDA");
  TORCH_CHECK(packed_key_all.is_cuda() && packed_value_token.is_cuda(), "packed codes must be CUDA");
  TORCH_CHECK(eas_indices.is_cuda() && kept_indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(full_key.is_contiguous() && full_value.is_contiguous(), "full key/value must be contiguous");
  TORCH_CHECK(packed_key_all.is_contiguous() && packed_value_token.is_contiguous(), "packed codes must be contiguous");
  TORCH_CHECK(eas_indices.is_contiguous() && kept_indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(full_key.scalar_type() == full_value.scalar_type(), "full key/value dtype mismatch");
  TORCH_CHECK(full_key.scalar_type() == at::kBFloat16 || full_key.scalar_type() == at::kFloat,
              "gather currently supports bf16 and fp32");
  TORCH_CHECK(packed_key_all.scalar_type() == at::kLong && packed_value_token.scalar_type() == at::kLong,
              "packed codes must be int64");
  TORCH_CHECK(eas_indices.scalar_type() == at::kInt && kept_indices.scalar_type() == at::kInt,
              "indices must be int32");
  TORCH_CHECK(full_key.dim() == 4 && full_value.dim() == 4, "full key/value must be [B,H,L,D]");
  TORCH_CHECK(packed_key_all.dim() == 4, "packed_key_all must be [B,H,Ck/64,Lglob]");
  TORCH_CHECK(packed_value_token.dim() == 4, "packed_value_token must be [B,H,Lglob,Cv/64]");
  TORCH_CHECK(eas_indices.dim() == 3 && kept_indices.dim() == 3, "indices must be [B,H,N]");

  const int B = full_key.size(0);
  const int H = full_key.size(1);
  const int total_len = full_key.size(2);
  const int D = full_key.size(3);
  TORCH_CHECK(full_value.size(0) == B && full_value.size(1) == H && full_value.size(2) == total_len && full_value.size(3) == D,
              "full value shape mismatch");
  TORCH_CHECK(packed_key_all.size(0) == B && packed_key_all.size(1) == H, "packed key B/H mismatch");
  TORCH_CHECK(packed_value_token.size(0) == B && packed_value_token.size(1) == H, "packed value B/H mismatch");
  const int key_blocks = packed_key_all.size(2);
  const int glob_len = packed_key_all.size(3);
  const int value_blocks = packed_value_token.size(3);
  const int Cv = value_blocks * 64;
  TORCH_CHECK(packed_value_token.size(2) == glob_len, "packed K/V global length mismatch");
  TORCH_CHECK(sink_size >= 0 && local_size >= 0 && sink_size + local_size + glob_len == total_len,
              "sink/local/global lengths do not match full length");
  TORCH_CHECK(eas_indices.size(0) == B && eas_indices.size(1) == H, "eas index B/H mismatch");
  TORCH_CHECK(kept_indices.size(0) == B && kept_indices.size(1) == H, "kept index B/H mismatch");

  const int rows = B * H;
  const int eas_count = eas_indices.size(2);
  const int kept_len = kept_indices.size(2);
  TORCH_CHECK(eas_count + kept_len == glob_len, "EAS + kept length must equal global length");
  const int dense_len = static_cast<int>(sink_size) + eas_count + static_cast<int>(local_size);
  const int value_groups = static_cast<int>(ceil_div_i64(kept_len, 64));

  at::cuda::CUDAGuard device_guard(full_key.device());
  auto stream = at::cuda::getCurrentCUDAStream(full_key.device().index());
  auto long_options = full_key.options().dtype(at::kLong);
  auto dense_key = at::empty({B, H, dense_len, D}, full_key.options());
  auto dense_value = at::empty({B, H, dense_len, D}, full_value.options());
  auto packed_key = at::empty({B, H, key_blocks, kept_len}, long_options);
  auto packed_value = at::empty({B, H, value_groups, Cv}, long_options);

  const int64_t dense_total = static_cast<int64_t>(rows) * dense_len * D;
  int blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(dense_total, kThreads), 65535));
  if (full_key.scalar_type() == at::kBFloat16) {
    gather_dense_kv_kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(full_key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(full_value.data_ptr<at::BFloat16>()),
        eas_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16 *>(dense_key.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(dense_value.data_ptr<at::BFloat16>()),
        rows,
        total_len,
        D,
        static_cast<int>(sink_size),
        static_cast<int>(local_size),
        eas_count,
        dense_len);
  } else {
    gather_dense_kv_kernel<float><<<blocks, kThreads, 0, stream>>>(
        full_key.data_ptr<float>(),
        full_value.data_ptr<float>(),
        eas_indices.data_ptr<int>(),
        dense_key.data_ptr<float>(),
        dense_value.data_ptr<float>(),
        rows,
        total_len,
        D,
        static_cast<int>(sink_size),
        static_cast<int>(local_size),
        eas_count,
        dense_len);
  }

  if (kept_len > 0) {
    const int64_t key_total = static_cast<int64_t>(rows) * key_blocks * kept_len;
    blocks = static_cast<int>(std::min<int64_t>(ceil_div_i64(key_total, kThreads), 65535));
    gather_key_from_packed_all_kernel<<<blocks, kThreads, 0, stream>>>(
        packed_key_all.data_ptr<int64_t>(),
        kept_indices.data_ptr<int>(),
        packed_key.data_ptr<int64_t>(),
        rows,
        glob_len,
        kept_len,
        key_blocks);

    dim3 value_grid(value_groups, value_blocks, rows);
    pack_value_from_token_packed_tile_kernel<<<value_grid, 64, 0, stream>>>(
        packed_value_token.data_ptr<int64_t>(),
        kept_indices.data_ptr<int>(),
        packed_value.data_ptr<int64_t>(),
        rows,
        glob_len,
        Cv,
        kept_len,
        value_blocks,
        value_groups);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dense_key, dense_value, packed_key, packed_value};
}
