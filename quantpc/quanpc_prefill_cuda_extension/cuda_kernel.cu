#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_bf16.h>
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
