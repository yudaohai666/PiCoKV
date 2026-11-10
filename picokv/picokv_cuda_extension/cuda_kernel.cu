#include "cuda_kernel.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <mutex>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kTileSize = 256;
constexpr int kGroupSize = 4;
constexpr int kQMagBits = 7;

__device__ __forceinline__ float picokv_fast_expf(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
}

__device__ __forceinline__ unsigned long long load_packed_key_word(
  const long *__restrict__ packed_key,
  const int B_idx,
  const int KVH_idx,
  const int KVH,
  const int L,
  const int KBlocks,
  const int l_idx,
  const int kb,
  const bool key_transposed
) {
  const long idx = key_transposed
    ? (((long)(B_idx * KVH + KVH_idx) * KBlocks + kb) * L + l_idx)
    : (((long)(B_idx * KVH + KVH_idx) * L + l_idx) * KBlocks + kb);
  return (unsigned long long)packed_key[idx];
}


template <int ExpBits>
__device__ __forceinline__ int picokv_weighted_popc64(
  const unsigned long long value_word,
  const unsigned long long *__restrict__ planes
) {
  if constexpr (ExpBits == 1) {
    return __popcll(value_word & planes[0]);
  } else if constexpr (ExpBits == 2) {
    return __popcll(value_word & planes[0])
      + (__popcll(value_word & planes[1]) << 1);
  } else if constexpr (ExpBits == 4) {
    return __popcll(value_word & planes[0])
      + (__popcll(value_word & planes[1]) << 1)
      + (__popcll(value_word & planes[2]) << 2)
      + (__popcll(value_word & planes[3]) << 3);
  } else {
    int pos = 0;
    #pragma unroll
    for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
      pos += __popcll(value_word & planes[bit_idx]) << bit_idx;
    }
    return pos;
  }
}

template <typename scalar_t>
__device__ __forceinline__ float picokv_to_float(scalar_t value) {
  return (float)value;
}

template <>
__device__ __forceinline__ float picokv_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t picokv_from_float(float value) {
  return (scalar_t)value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 picokv_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename scalar_t>
__device__ __forceinline__ float picokv_weight_for_matmul(float value) {
  return value;
}

template <>
__device__ __forceinline__ float picokv_weight_for_matmul<__nv_bfloat16>(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

template <typename scalar_t>
__device__ __forceinline__ float picokv_score_for_softmax(float dot, float attention_scale) {
  return dot * attention_scale;
}

template <>
__device__ __forceinline__ float picokv_score_for_softmax<__nv_bfloat16>(float dot, float attention_scale) {
  const float rounded_dot = __bfloat162float(__float2bfloat16(dot));
  return __bfloat162float(__float2bfloat16(rounded_dot * attention_scale));
}

// Dense retained-KV numerator used as the softmax anchor for global PiCoKV.
// One block handles one [B, QH, S] query row. Scores are cached once in shared
// memory so the value pass does not recompute q dot k for every output dimension.
template <typename scalar_t>
__device__ __forceinline__ float picokv_code_for_projection(float value) {
  return value;
}

template <>
__device__ __forceinline__ float picokv_code_for_projection<__nv_bfloat16>(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// Project global code output back to head_dim, add dense numerator, and apply
// the shared normalization denominator in one pass. This avoids materializing
// repeated value projection heads and a separate torch matmul for tiny S=1 rows.
template <typename scalar_t>
__global__ void picokv_fused_output_projection_kernel(
  const scalar_t *__restrict__ dense_output,      // [B,QH,S,D]
  const float *__restrict__ glob_code_output,     // [B,QH,S,Cv]
  const float *__restrict__ glob_den,             // [B,QH,S]
  const scalar_t *__restrict__ value_proj,        // [B,KVH,Cv,D]
  scalar_t *__restrict__ attn_output,             // [B,QH,S,D]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int Cv,
  const int D
) {
  const int row = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int S_idx = row % S;
  const int qh = (row / S) % QH;
  const int B_idx = row / (S * QH);
  const int group_size = QH / KVH;
  const int KVH_idx = qh / group_size;

  const long dense_base = ((long)(B_idx * QH + qh) * S + S_idx) * D;
  const long code_base = ((long)(B_idx * QH + qh) * S + S_idx) * Cv;
  const long proj_base = ((long)(B_idx * KVH + KVH_idx) * Cv) * D;
  const long den_idx = (long)(B_idx * QH + qh) * S + S_idx;
  const float inv_den = 1.0f / (1.0f + glob_den[den_idx]);

  for (int d = thread_idx; d < D; d += blockDim.x) {
    float glob_acc = 0.0f;
    for (int c = 0; c < Cv; c++) {
      const float code_value = picokv_code_for_projection<scalar_t>(glob_code_output[code_base + c]);
      glob_acc += code_value * picokv_to_float(value_proj[proj_base + (long)c * D + d]);
    }
    const float numerator = picokv_to_float(dense_output[dense_base + d]) + glob_acc;
    attn_output[dense_base + d] = picokv_from_float<scalar_t>(numerator * inv_den);
  }

}

// Round fp32 accumulated codes to bf16 for the tensor-core GEMM projection:
// the same rounding picokv_code_for_projection applies elementwise in the
// scalar kernel, so the GEMM multiplies the exact same bf16 values.
__global__ void picokv_cast_code_bf16_kernel(
  const float *__restrict__ src,
  __nv_bfloat16 *__restrict__ dst,
  const long numel
) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = __float2bfloat16(src[idx]);
  }
}

// GEMM-path epilogue: out = (dense + acc32) * (1 / (1 + den)). Identical
// arithmetic to the scalar kernel's epilogue (fp32 add, fp32 scale, one bf16
// round at the store); only the accumulated value's producer differs.
template <typename scalar_t>
__global__ void picokv_proj_gemm_epilogue_kernel(
  const scalar_t *__restrict__ dense_output,  // [B,QH,S,D]
  const float *__restrict__ glob_acc,         // [B,QH,S,D] fp32 GEMM output
  const float *__restrict__ glob_den,         // [B,QH,S]
  scalar_t *__restrict__ attn_output,         // [B,QH,S,D]
  const int total,
  const int D
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const long row = idx / D;
  const float inv_den = 1.0f / (1.0f + glob_den[row]);
  const float numerator = picokv_to_float(dense_output[idx]) + glob_acc[idx];
  attn_output[idx] = picokv_from_float<scalar_t>(numerator * inv_den);
}

template <typename scalar_t>
__global__ void picokv_dense_attention_numerator_kernel(
  const scalar_t *__restrict__ query,       // [B,QH,S,D]
  const scalar_t *__restrict__ dense_key,   // [B,KVH,L,D]
  const scalar_t *__restrict__ dense_value, // [B,KVH,L,D]
  scalar_t *__restrict__ dense_output,      // [B,QH,S,D]
  float *__restrict__ dense_lse,            // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int D,
  const float attention_scale
) {
  extern __shared__ float scores[];
  __shared__ float reduce_cache[256];
  __shared__ float row_lse;
  __shared__ float row_inv_den;

  const int row = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int S_idx = row % S;
  const int qh = (row / S) % QH;
  const int B_idx = row / (S * QH);
  const int group_size = QH / KVH;
  const int KVH_idx = qh / group_size;

  const long q_base = ((long)(B_idx * QH + qh) * S + S_idx) * D;
  const long kv_base = ((long)(B_idx * KVH + KVH_idx) * L) * D;

  float local_max = -3.402823466e38f;
  for (int l_idx = thread_idx; l_idx < L; l_idx += blockDim.x) {
    float dot = 0.0f;
    const long k_base = kv_base + (long)l_idx * D;
    #pragma unroll 4
    for (int d = 0; d < D; d++) {
      dot += picokv_to_float(query[q_base + d]) * picokv_to_float(dense_key[k_base + d]);
    }
    const float score = picokv_score_for_softmax<scalar_t>(dot, attention_scale);
    scores[l_idx] = score;
    local_max = fmaxf(local_max, score);
  }

  reduce_cache[thread_idx] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      reduce_cache[thread_idx] = fmaxf(reduce_cache[thread_idx], reduce_cache[thread_idx + stride]);
    }
    __syncthreads();
  }
  const float max_score = reduce_cache[0];

  float local_den = 0.0f;
  for (int l_idx = thread_idx; l_idx < L; l_idx += blockDim.x) {
    const float unnormalized_weight = expf(scores[l_idx] - max_score);
    scores[l_idx] = unnormalized_weight;
    local_den += unnormalized_weight;
  }

  reduce_cache[thread_idx] = local_den;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      reduce_cache[thread_idx] += reduce_cache[thread_idx + stride];
    }
    __syncthreads();
  }

  if (thread_idx == 0) {
    const float den = reduce_cache[0];
    row_lse = max_score + logf(den);
    row_inv_den = 1.0f / den;
    dense_lse[(long)(B_idx * QH + qh) * S + S_idx] = row_lse;
  }
  __syncthreads();

  for (int d = thread_idx; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int l_idx = 0; l_idx < L; l_idx++) {
      float weight = scores[l_idx] * row_inv_den;
      weight = picokv_weight_for_matmul<scalar_t>(weight);
      const long v_idx = kv_base + (long)l_idx * D + d;
      acc += weight * picokv_to_float(dense_value[v_idx]);
    }
    dense_output[q_base + d] = picokv_from_float<scalar_t>(acc);
  }
}

__global__ void picokv_global_code_output_packed_value_tiled_reduce_kernel(
  const __nv_bfloat16 *__restrict__ partial_code_output, // [B,H,S,num_tiles,Cv]
  const float *__restrict__ partial_den,                 // [B,H,S,num_tiles]
  __nv_bfloat16 *__restrict__ glob_code_output,          // [B,H,S,Cv]
  float *__restrict__ glob_den,                          // [B,H,S]
  const int B,
  const int H,
  const int S,
  const int num_tiles,
  const int Cv
) {
  const int S_idx = blockIdx.x;
  const int BH_idx = blockIdx.y;
  const int B_idx = BH_idx / H;
  const int H_idx = BH_idx - B_idx * H;
  const int thread_idx = threadIdx.x;
  const long base = ((long)(B_idx * H + H_idx) * S + S_idx);

  if ((Cv & 1) == 0) {
    const int pair_count = Cv >> 1;
    if (thread_idx < pair_count) {
      const int c = thread_idx << 1;
      __nv_bfloat162 acc = __float2bfloat162_rn(0.0f);
      for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const long partial_idx = (base * num_tiles + tile_idx) * Cv + c;
        acc = __hadd2(acc, *((const __nv_bfloat162*)(&partial_code_output[partial_idx])));
      }
      *((__nv_bfloat162*)(&glob_code_output[base * Cv + c])) = acc;
    }

    if (thread_idx == pair_count) {
      float den = 0.0f;
      for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        den += partial_den[base * num_tiles + tile_idx];
      }
      glob_den[base] = den;
    }
    return;
  }

  if (thread_idx < Cv) {
    __nv_bfloat16 acc = __float2bfloat16(0.0f);
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      const long partial_idx = (base * num_tiles + tile_idx) * Cv + thread_idx;
      acc = __hadd(acc, partial_code_output[partial_idx]);
    }
    glob_code_output[base * Cv + thread_idx] = acc;
  }

  if (thread_idx == Cv) {
    float den = 0.0f;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      den += partial_den[base * num_tiles + tile_idx];
    }
    glob_den[base] = den;
  }
}

template <int KBlocks, bool KeyTransposedFixed>
__global__ void picokv_packed_value_partial_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  __nv_bfloat16 *__restrict__ partial_code_output, // [B,QH,S,num_tiles,Cv]
  float *__restrict__ partial_den,                 // [B,QH,S,num_tiles]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int VBlocks,
  const int value_bits,
  const int num_tiles,
  const int query_scale_numel,
  const float attention_scale
) {
  const int tile_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int l_idx = tile_idx * kTileSize + thread_idx;
  const int qh_base = KVH_idx * kGroupSize;
  const int Cv = VBlocks * 64;

  __shared__ unsigned long long query_cache[kGroupSize][KBlocks * 8];
  __shared__ int query_base_dot[kGroupSize];
  __shared__ float scale_cache[kGroupSize];
  __shared__ float lse_cache[kGroupSize];
  __shared__ float weights[kGroupSize][kTileSize];
  __shared__ float den_cache[kGroupSize][8];

  for (int idx = thread_idx; idx < kGroupSize * KBlocks * 8; idx += blockDim.x) {
    const int hrel = idx / (KBlocks * 8);
    const int off = idx - hrel * KBlocks * 8;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < kGroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }
  __syncthreads();

  float local_weights[kGroupSize] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (l_idx < L) {
    unsigned long long key_words[KBlocks];
    #pragma unroll
    for (int kb = 0; kb < KBlocks; kb++) {
      key_words[kb] = load_packed_key_word(
        packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
      );
    }

    #pragma unroll
    for (int hrel = 0; hrel < kGroupSize; hrel++) {
      int negative_dot = 0;
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
        #pragma unroll
        for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
          negative_dot += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
        }
      }
      const int dot = query_base_dot[hrel] - (negative_dot << 1);
      local_weights[hrel] = picokv_fast_expf((float)dot * scale_cache[hrel] - lse_cache[hrel]);
    }
  }

  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;
  #pragma unroll
  for (int hrel = 0; hrel < kGroupSize; hrel++) {
    weights[hrel][thread_idx] = local_weights[hrel];
    float den_sum = local_weights[hrel];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      den_sum += __shfl_down_sync(0xffffffff, den_sum, offset);
    }
    if (lane_idx == 0) {
      den_cache[hrel][warp_idx] = den_sum;
    }
  }
  __syncthreads();

  if (thread_idx < kGroupSize) {
    float den = 0.0f;
    #pragma unroll
    for (int warp = 0; warp < 8; warp++) {
      den += den_cache[thread_idx][warp];
    }
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    partial_den[base * num_tiles + tile_idx] = den;
  }

  for (int task_idx = thread_idx; task_idx < kGroupSize * Cv; task_idx += blockDim.x) {
    const int hrel = task_idx / Cv;
    const int c = task_idx - hrel * Cv;
    const int vb = c >> 6;
    const int bit_offset = c & 63;
    const float zero_point = ((float)((1 << value_bits) - 1)) * 0.5f;
    float acc = 0.0f;

    #pragma unroll 4
    for (int local_l = 0; local_l < kTileSize; local_l++) {
      const int token_idx = tile_idx * kTileSize + local_l;
      if (token_idx < L) {
        int code = 0;
        #pragma unroll
        for (int bit_idx = 0; bit_idx < 8; bit_idx++) {
          if (bit_idx < value_bits) {
            const long value_idx = (((long)(B_idx * KVH + KVH_idx) * L + token_idx) * VBlocks + vb) * value_bits + bit_idx;
            const unsigned long long plane = (unsigned long long)packed_value[value_idx];
            code |= (int)((plane >> bit_offset) & 1ULL) << bit_idx;
          }
        }
        acc += weights[hrel][local_l] * ((float)code - zero_point);
      }
    }

    const int qh = qh_base + hrel;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    const long partial_idx = ((base * num_tiles + tile_idx) * Cv + c);
    partial_code_output[partial_idx] = __float2bfloat16(acc);
  }
}



template <typename scalar_t>
__global__ void picokv_project_quantize_pack_query_kernel(
  const scalar_t *__restrict__ query,        // [B,QH,S,D]
  const scalar_t *__restrict__ key_proj,     // [B,KVH,C,D]
  long *__restrict__ packed_projected_query, // [B,QH,S,C/64,8]
  float *__restrict__ query_scale,           // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int C,
  const int D
) {
  __shared__ float projected_cache[512];
  __shared__ int8_t quant_cache[512];
  __shared__ float reduce_cache[256];
  __shared__ float row_scale;

  const int row = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int S_idx = row % S;
  const int qh = (row / S) % QH;
  const int B_idx = row / (S * QH);
  const int group_size = QH / KVH;
  const int KVH_idx = qh / group_size;
  const int KBlocks = C >> 6;
  const long q_base = ((long)(B_idx * QH + qh) * S + S_idx) * D;
  const long proj_base = ((long)(B_idx * KVH + KVH_idx) * C) * D;

  float local_max = 0.0f;
  for (int c = thread_idx; c < C; c += blockDim.x) {
    float dot = 0.0f;
    const long c_base = proj_base + (long)c * D;
    #pragma unroll 4
    for (int d = 0; d < D; d++) {
      dot += picokv_to_float(query[q_base + d]) * picokv_to_float(key_proj[c_base + d]);
    }
    projected_cache[c] = dot;
    local_max = fmaxf(local_max, fabsf(dot));
  }

  reduce_cache[thread_idx] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      reduce_cache[thread_idx] = fmaxf(reduce_cache[thread_idx], reduce_cache[thread_idx + stride]);
    }
    __syncthreads();
  }

  if (thread_idx == 0) {
    const float max_abs = fmaxf(reduce_cache[0], 1.1920928955078125e-7f);
    row_scale = max_abs / 127.0f;
    query_scale[row] = row_scale;
  }
  __syncthreads();

  for (int c = thread_idx; c < C; c += blockDim.x) {
    int q = (int)nearbyintf(projected_cache[c] / row_scale);
    q = q < -127 ? -127 : q;
    q = q > 127 ? 127 : q;
    quant_cache[c] = (int8_t)q;
  }
  __syncthreads();

  for (int task = thread_idx; task < KBlocks * 8; task += blockDim.x) {
    const int kb = task >> 3;
    const int plane = task & 7;
    unsigned long long word = 0ULL;
    #pragma unroll
    for (int bit = 0; bit < 64; bit++) {
      int q = (int)quant_cache[(kb << 6) + bit];
      int selected = 0;
      if (plane == 0) {
        selected = q < 0;
      } else {
        q = q < 0 ? -q : q;
        selected = (q >> (plane - 1)) & 1;
      }
      if (selected) {
        word |= 1ULL << bit;
      }
    }
    packed_projected_query[((long)row * KBlocks + kb) * 8 + plane] = (long)word;
  }
}

template <typename scalar_t>
__global__ void picokv_quantize_pack_projected_query_kernel(
  const scalar_t *__restrict__ projected_query, // [B,H,S,C]
  long *__restrict__ packed_projected_query,    // [B,H,S,C/64,8]
  float *__restrict__ query_scale,              // [B,H,S]
  const int B,
  const int H,
  const int S,
  const int C
) {
  __shared__ float reduce_cache[256];
  __shared__ float row_scale;
  __shared__ int8_t quant_cache[512];

  const int row = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int KBlocks = C >> 6;
  const long row_base = (long)row * C;

  float local_max = 0.0f;
  for (int c = thread_idx; c < C; c += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(picokv_to_float(projected_query[row_base + c])));
  }

  reduce_cache[thread_idx] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      reduce_cache[thread_idx] = fmaxf(reduce_cache[thread_idx], reduce_cache[thread_idx + stride]);
    }
    __syncthreads();
  }

  if (thread_idx == 0) {
    const float max_abs = fmaxf(reduce_cache[0], 1.1920928955078125e-7f);
    row_scale = max_abs / 127.0f;
    query_scale[row] = row_scale;
  }
  __syncthreads();

  for (int c = thread_idx; c < C; c += blockDim.x) {
    int q = (int)nearbyintf(picokv_to_float(projected_query[row_base + c]) / row_scale);
    q = q < -127 ? -127 : q;
    q = q > 127 ? 127 : q;
    quant_cache[c] = (int8_t)q;
  }
  __syncthreads();

  for (int task = thread_idx; task < KBlocks * 8; task += blockDim.x) {
    const int kb = task >> 3;
    const int plane = task & 7;
    unsigned long long word = 0ULL;
    #pragma unroll
    for (int bit = 0; bit < 64; bit++) {
      int q = (int)quant_cache[(kb << 6) + bit];
      int selected = 0;
      if (plane == 0) {
        selected = q < 0;
      } else {
        q = q < 0 ? -q : q;
        selected = (q >> (plane - 1)) & 1;
      }
      if (selected) {
        word |= 1ULL << bit;
      }
    }
    packed_projected_query[((long)row * KBlocks + kb) * 8 + plane] = (long)word;
  }
}

// Quantize+pack with the tail accumulators zeroed in the same launch: each row
// block also clears its [Cv] code accumulator and its denominator entry, which
// is exactly what the removed cudaMemsetAsync wrote (same zeros, same stream,
// still strictly before the main kernel reads/accumulates them). Saves one
// kernel launch and the memset's full-tensor pass per decode layer.
template <typename scalar_t>
__global__ void picokv_quantize_pack_zero_projected_query_kernel(
  const scalar_t *__restrict__ projected_query, // [B,H,S,C]
  long *__restrict__ packed_projected_query,    // [B,H,S,C/64,8]
  float *__restrict__ query_scale,              // [B,H,S]
  float *__restrict__ code_zero,                // [B,H,S,Cv] accumulator to zero
  float *__restrict__ den_zero,                 // [B,H,S] accumulator to zero
  const int C,
  const int Cv
) {
  __shared__ float reduce_cache[256];
  __shared__ float row_scale;
  __shared__ int8_t quant_cache[512];

  const int row = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int KBlocks = C >> 6;
  const long row_base = (long)row * C;
  const long zero_base = (long)row * Cv;

  for (int c = thread_idx; c < Cv; c += blockDim.x) {
    code_zero[zero_base + c] = 0.0f;
  }
  if (thread_idx == 0) {
    den_zero[row] = 0.0f;
  }

  float local_max = 0.0f;
  for (int c = thread_idx; c < C; c += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(picokv_to_float(projected_query[row_base + c])));
  }

  reduce_cache[thread_idx] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      reduce_cache[thread_idx] = fmaxf(reduce_cache[thread_idx], reduce_cache[thread_idx + stride]);
    }
    __syncthreads();
  }

  if (thread_idx == 0) {
    const float max_abs = fmaxf(reduce_cache[0], 1.1920928955078125e-7f);
    row_scale = max_abs / 127.0f;
    query_scale[row] = row_scale;
  }
  __syncthreads();

  for (int c = thread_idx; c < C; c += blockDim.x) {
    int q = (int)nearbyintf(picokv_to_float(projected_query[row_base + c]) / row_scale);
    q = q < -127 ? -127 : q;
    q = q > 127 ? 127 : q;
    quant_cache[c] = (int8_t)q;
  }
  __syncthreads();

  for (int task = thread_idx; task < KBlocks * 8; task += blockDim.x) {
    const int kb = task >> 3;
    const int plane = task & 7;
    unsigned long long word = 0ULL;
    #pragma unroll
    for (int bit = 0; bit < 64; bit++) {
      int q = (int)quant_cache[(kb << 6) + bit];
      int selected = 0;
      if (plane == 0) {
        selected = q < 0;
      } else {
        q = q < 0 ? -q : q;
        selected = (q >> (plane - 1)) & 1;
      }
      if (selected) {
        word |= 1ULL << bit;
      }
    }
    packed_projected_query[((long)row * KBlocks + kb) * 8 + plane] = (long)word;
  }
}


// Warp-per-row variant of the kernel above (same math, bitwise-identical
// outputs): lane l holds elements l + 32 t, the row max is a shuffle
// reduction (max is exact in any order), each plane word of a 64-element
// block is two ballots. The block-per-row kernel spends most of its time in
// __syncthreads and its serial 64-bit pack loop with 216 of 256 threads idle
// (C = 320: 40 words per row); this one has no block barrier.
template <typename scalar_t>
__global__ void picokv_quantize_pack_zero_projected_query_warp_kernel(
  const scalar_t *__restrict__ projected_query, // [B,H,S,C]
  long *__restrict__ packed_projected_query,    // [B,H,S,C/64,8]
  float *__restrict__ query_scale,              // [B,H,S]
  float *__restrict__ code_zero,                // [B,H,S,Cv]
  float *__restrict__ den_zero,                 // [B,H,S]
  const int rows,
  const int C,
  const int Cv
) {
  constexpr int MAXT = 16;  // C <= 512
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) {
    return;
  }
  const int T = C >> 5;
  const long row_base = (long)row * C;
  const long zero_base = (long)row * Cv;
  for (int c = lane; c < Cv; c += 32) {
    code_zero[zero_base + c] = 0.0f;
  }
  if (lane == 0) {
    den_zero[row] = 0.0f;
  }
  float x[MAXT];
  float local_max = 0.0f;
  #pragma unroll
  for (int t = 0; t < MAXT; t++) {
    if (t < T) {
      x[t] = picokv_to_float(projected_query[row_base + (t << 5) + lane]);
      local_max = fmaxf(local_max, fabsf(x[t]));
    }
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
  }
  const float row_scale = fmaxf(local_max, 1.1920928955078125e-7f) / 127.0f;
  if (lane == 0) {
    query_scale[row] = row_scale;
  }
  int q[MAXT];
  #pragma unroll
  for (int t = 0; t < MAXT; t++) {
    if (t < T) {
      int v = (int)nearbyintf(x[t] / row_scale);
      v = v < -127 ? -127 : v;
      q[t] = v > 127 ? 127 : v;
    }
  }
  const int KBlocks = C >> 6;
  long *out = packed_projected_query + (long)row * KBlocks * 8;
  #pragma unroll
  for (int kb = 0; kb < MAXT / 2; kb++) {
    if (kb < KBlocks) {
      const int lo = q[2 * kb], hi = q[2 * kb + 1];
      const int alo = lo < 0 ? -lo : lo, ahi = hi < 0 ? -hi : hi;
      unsigned long long mine = 0ULL;
      #pragma unroll
      for (int plane = 0; plane < 8; plane++) {
        const unsigned blo = __ballot_sync(0xffffffffu, plane == 0 ? lo < 0 : ((alo >> (plane - 1)) & 1));
        const unsigned bhi = __ballot_sync(0xffffffffu, plane == 0 ? hi < 0 : ((ahi >> (plane - 1)) & 1));
        if (lane == plane) {
          mine = (unsigned long long)blo | ((unsigned long long)bhi << 32);
        }
      }
      if (lane < 8) {
        out[kb * 8 + lane] = (long)mine;
      }
    }
  }
}


__global__ void picokv_pack_projected_query_kernel(
  const int8_t *__restrict__ quant_projected_query, // [B,H,S,C]
  long *__restrict__ packed_projected_query,        // [B,H,S,C/64,8]
  const int total_words
) {
  const int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (word_idx >= total_words) {
    return;
  }

  const long base_in = ((long)word_idx) * 64;
  const long base_out = ((long)word_idx) * 8;
  unsigned long long planes[8] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};

  #pragma unroll
  for (int bit = 0; bit < 64; bit++) {
    int q = (int)quant_projected_query[base_in + bit];
    if (q < 0) {
      planes[0] |= 1ULL << bit;
      q = -q;
    }
    q = q > 127 ? 127 : q;
    #pragma unroll
    for (int mag_bit = 0; mag_bit < 7; mag_bit++) {
      if ((q >> mag_bit) & 1) {
        planes[mag_bit + 1] |= 1ULL << bit;
      }
    }
  }

  #pragma unroll
  for (int plane = 0; plane < 8; plane++) {
    packed_projected_query[base_out + plane] = (long)planes[plane];
  }
}

template <int GroupSize, int KBlocks, int Cv, bool KeyTransposedFixed, bool AtomicOutputFixed, int ExpBits>
__global__ void __launch_bounds__(256, 6) picokv_v1_expbits_partial_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  __nv_bfloat16 *__restrict__ partial_code_output, // [B,QH,S,num_tiles,Cv]
  float *__restrict__ partial_den,                 // [B,QH,S,num_tiles]
  float *__restrict__ atomic_code_output,          // [B,QH,S,Cv]
  float *__restrict__ atomic_den,                  // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int num_tiles,
  const int query_scale_numel,
  const float attention_scale
) {
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;

  const int tile_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int l_idx = tile_idx * kTileSize + thread_idx;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = threadIdx.x >> 5;

  __shared__ unsigned long long query_cache[GroupSize][QueryWords];
  __shared__ int query_base_dot[GroupSize];
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ float weights[GroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[GroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];

  for (int idx = thread_idx; idx < GroupSize * QueryWords; idx += blockDim.x) {
    const int hrel = idx / QueryWords;
    const int off = idx - hrel * QueryWords;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }
  __syncthreads();

  float local_weights[GroupSize] = {0.0f};
  if (l_idx < L) {
    unsigned long long key_words[KBlocks];
    #pragma unroll
    for (int kb = 0; kb < KBlocks; kb++) {
      key_words[kb] = load_packed_key_word(
        packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
      );
    }

    #pragma unroll
    for (int hrel = 0; hrel < GroupSize; hrel++) {
      int negative_dot = 0;
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
        #pragma unroll
        for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
          negative_dot += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
        }
      }
      const int dot = query_base_dot[hrel] - (negative_dot << 1);
      local_weights[hrel] = picokv_fast_expf((float)dot * scale_cache[hrel] - lse_cache[hrel]);
    }
  }

  #pragma unroll
  for (int hrel = 0; hrel < GroupSize; hrel++) {
    weights[hrel][thread_idx] = local_weights[hrel];
  }
  __syncthreads();

  #pragma unroll
  for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
    const int quant_task = warp_idx + quant_pass * QuantWarps;
    if (quant_task < GroupSize * TileGroups) {
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float max_weight = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        max_weight = fmaxf(max_weight, __shfl_down_sync(0xffffffff, max_weight, offset));
      }
      max_weight = __shfl_sync(0xffffffff, max_weight, 0);

      const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
      int q0 = 0;
      int q1 = 0;
      if (max_weight > 0.0f) {
        q0 = (int)(w0 * inv_scale + 0.5f);
        q1 = (int)(w1 * inv_scale + 0.5f);
        q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
        q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        const float scale = max_weight > 0.0f ? max_weight / (float)ExpMaxCode : 0.0f;
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    float den_sum = 0.0f;
    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      den_sum -= exp_bias[thread_idx][local_group];
    }
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    if constexpr (AtomicOutputFixed) {
      atomicAdd(&atomic_den[base], den_sum);
    } else {
      partial_den[base * num_tiles + tile_idx] = den_sum;
    }
  }

  const int pair_count = Cv >> 1;
  for (int task_idx = thread_idx; task_idx < GroupSize * pair_count; task_idx += blockDim.x) {
    const int hrel = task_idx / pair_count;
    const int pair_idx = task_idx - hrel * pair_count;
    const int c = pair_idx << 1;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      const int global_group = tile_idx * TileGroups + local_group;
      if (global_group < LGroups) {
        const float scale2 = exp_scale2[hrel][local_group];
        const float bias = exp_bias[hrel][local_group];
        if (scale2 > 0.0f) {
          const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
          const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));
          int pos0 = 0;
          int pos1 = 0;
          #pragma unroll
          for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
            const int bit_weight = 1 << bit_idx;
            const unsigned long long qplane = exp_planes[hrel][local_group][bit_idx];
            pos0 += bit_weight * __popcll(value_words.x & qplane);
            pos1 += bit_weight * __popcll(value_words.y & qplane);
          }
          acc0 += scale2 * (float)pos0 + bias;
          acc1 += scale2 * (float)pos1 + bias;
        }
      }
    }

    const int qh = qh_base + hrel;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    if constexpr (AtomicOutputFixed) {
      atomicAdd(&atomic_code_output[base * Cv + c], acc0);
      atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1);
    } else {
      const long partial_idx = ((base * num_tiles + tile_idx) * Cv + c);
      *((__nv_bfloat162*)(&partial_code_output[partial_idx])) = __floats2bfloat162_rn(acc0, acc1);
    }
  }
}

// Common Ck=256/Cv=256 path.  Each block processes two 256-token tiles before
// writing atomics, preserving enough parallelism while reducing query reloads
// and atomic traffic for the runtime-bit exp
// / 1-bit value popcount accumulator.
template <bool KeyTransposedFixed, int ExpBits>
__global__ void __launch_bounds__(256, 8) picokv_v1_expbits_atomic2_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,256]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale
) {
  constexpr int KBlocks = 4;
  constexpr int Cv = 256;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = 2;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * kGroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;

  __shared__ unsigned long long query_cache[kGroupSize][QueryWords];
  __shared__ int query_base_dot[kGroupSize];
  __shared__ float scale_cache[kGroupSize];
  __shared__ float lse_cache[kGroupSize];
  __shared__ float weights[kGroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[kGroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[kGroupSize][TileGroups];
  __shared__ float exp_bias[kGroupSize][TileGroups];

  for (int idx = thread_idx; idx < kGroupSize * QueryWords; idx += blockDim.x) {
    const int hrel = idx / QueryWords;
    const int off = idx - hrel * QueryWords;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < kGroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }
  __syncthreads();

  const int hrel0 = thread_idx >> 7;
  const int hrel1 = hrel0 + 2;
  const int pair_idx = thread_idx & 127;
  const int c = pair_idx << 1;
  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weights[kGroupSize] = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (ExpBits <= 2) {
      #pragma unroll
      for (int hrel = 0; hrel < kGroupSize; hrel++) {
        local_weights[hrel] = -3.402823466e38f;
      }
    }
    if (l_idx < L) {
      unsigned long long key_words[KBlocks];
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      #pragma unroll
      for (int hrel = 0; hrel < kGroupSize; hrel++) {
        int negative_dot = 0;
        #pragma unroll
        for (int kb = 0; kb < KBlocks; kb++) {
          const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
          #pragma unroll
          for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
            negative_dot += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
          }
        }
        const int dot = query_base_dot[hrel] - (negative_dot << 1);
        const float score = (float)dot * scale_cache[hrel];
        if constexpr (ExpBits <= 2) {
          local_weights[hrel] = score;
        } else {
          local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
        }
      }
    }

    #pragma unroll
    for (int hrel = 0; hrel < kGroupSize; hrel++) {
      weights[hrel][thread_idx] = local_weights[hrel];
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      if constexpr (ExpBits <= 2) {
        const float max_score = group_max;
        const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
        if (max_weight > 0.0f) {
          const float d0 = w0 - max_score;
          const float d1 = w1 - max_score;
          if constexpr (ExpBits == 1) {
            constexpr float t0 = -0.6931471805599453f;
            q0 = d0 >= t0 ? 1 : 0;
            q1 = d1 >= t0 ? 1 : 0;
          } else {
            constexpr float t0 = -1.7917594692280550f;
            constexpr float t1 = -0.6931471805599453f;
            constexpr float t2 = -0.1823215567939546f;
            q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
            q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
          }
          scale = max_weight / (float)ExpMaxCode;
        }
      } else {
        const float max_weight = group_max;
        const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
        if (max_weight > 0.0f) {
          q0 = (int)(w0 * inv_scale + 0.5f);
          q1 = (int)(w1 * inv_scale + 0.5f);
          q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
          q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
          scale = max_weight / (float)ExpMaxCode;
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < kGroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      const int global_group = tile_idx * TileGroups + local_group;
      if (global_group < LGroups) {
        const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
        const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

        const float scale20 = exp_scale2[hrel0][local_group];
        const float bias0 = exp_bias[hrel0][local_group];
        if (scale20 > 0.0f) {
          const unsigned long long *planes0 = &exp_planes[hrel0][local_group][0];
          const int pos00 = picokv_weighted_popc64<ExpBits>(value_words.x, planes0);
          const int pos01 = picokv_weighted_popc64<ExpBits>(value_words.y, planes0);
          acc00 += scale20 * (float)pos00 + bias0;
          acc01 += scale20 * (float)pos01 + bias0;
        }

        const float scale21 = exp_scale2[hrel1][local_group];
        const float bias1 = exp_bias[hrel1][local_group];
        if (scale21 > 0.0f) {
          const unsigned long long *planes1 = &exp_planes[hrel1][local_group][0];
          const int pos10 = picokv_weighted_popc64<ExpBits>(value_words.x, planes1);
          const int pos11 = picokv_weighted_popc64<ExpBits>(value_words.y, planes1);
          acc10 += scale21 * (float)pos10 + bias1;
          acc11 += scale21 * (float)pos11 + bias1;
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < kGroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  const int qh0 = qh_base + hrel0;
  const int qh1 = qh_base + hrel1;
  const long base0 = ((long)(B_idx * QH + qh0) * S + S_idx);
  const long base1 = ((long)(B_idx * QH + qh1) * S + S_idx);
  atomicAdd(&atomic_code_output[base0 * Cv + c], acc00);
  atomicAdd(&atomic_code_output[base0 * Cv + c + 1], acc01);
  atomicAdd(&atomic_code_output[base1 * Cv + c], acc10);
  atomicAdd(&atomic_code_output[base1 * Cv + c + 1], acc11);
}


// GQA=4 Ck=256/Cv=256 pruning experiment.  PruneMode bit 0 enables
// 64-token group pruning after group max; bit 1 enables cascaded prefix-dim
// token pruning at 64/128/192 code dims.  The output format matches
// picokv_v1_expbits_atomic2_kernel so callers can compare against baseline.
template <bool KeyTransposedFixed, int ExpBits, int PruneMode>
__global__ void __launch_bounds__(256, 8) picokv_v1_expbits_atomic2_pruned_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,256]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale,
  const float prune_log_eps
) {
  constexpr int KBlocks = 4;
  constexpr int Cv = 256;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = 2;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;
  constexpr bool DoBlockPrune = (PruneMode & 1) != 0;
  constexpr bool DoPrefixPrune = (PruneMode & 2) != 0;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * kGroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;

  __shared__ unsigned long long query_cache[kGroupSize][QueryWords];
  __shared__ int query_base_dot[kGroupSize];
  __shared__ int query_block_abs[kGroupSize][KBlocks];
  __shared__ float scale_cache[kGroupSize];
  __shared__ float lse_cache[kGroupSize];
  __shared__ float prune_dot_threshold_cache[kGroupSize];
  __shared__ float weights[kGroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[kGroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[kGroupSize][TileGroups];
  __shared__ float exp_bias[kGroupSize][TileGroups];

  for (int idx = thread_idx; idx < kGroupSize * QueryWords; idx += blockDim.x) {
    const int hrel = idx / QueryWords;
    const int off = idx - hrel * QueryWords;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < kGroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int kb = 0; kb < KBlocks; kb++) {
      int block_abs = 0;
      #pragma unroll
      for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
        block_abs += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
      }
      query_block_abs[thread_idx][kb] = block_abs;
      base_dot += block_abs;
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
    prune_dot_threshold_cache[thread_idx] = (lse_cache[thread_idx] + prune_log_eps) / fmaxf(scale_cache[thread_idx], 1.0e-30f);
  }
  __syncthreads();

  const int hrel0 = thread_idx >> 7;
  const int hrel1 = hrel0 + 2;
  const int pair_idx = thread_idx & 127;
  const int c = pair_idx << 1;
  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weights[kGroupSize] = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (ExpBits <= 2) {
      #pragma unroll
      for (int hrel = 0; hrel < kGroupSize; hrel++) {
        local_weights[hrel] = -3.402823466e38f;
      }
    }
    if (l_idx < L) {
      unsigned long long key_words[KBlocks];
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      #pragma unroll
      for (int hrel = 0; hrel < kGroupSize; hrel++) {
        int negative_dot = 0;
        int prefix_abs = 0;
        bool token_pruned = false;
        const float dot_threshold = prune_dot_threshold_cache[hrel];
        #pragma unroll
        for (int kb = 0; kb < KBlocks; kb++) {
          if (!token_pruned) {
            const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
            int negative_block = 0;
            #pragma unroll
            for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
              negative_block += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
            }
            negative_dot += negative_block;
            prefix_abs += query_block_abs[hrel][kb];
            if constexpr (DoPrefixPrune) {
              if (kb < KBlocks - 1) {
                const int partial_dot = prefix_abs - (negative_dot << 1);
                const int remaining_abs = query_base_dot[hrel] - prefix_abs;
                if ((float)(partial_dot + remaining_abs) <= dot_threshold) {
                  token_pruned = true;
                }
              }
            }
          }
        }
        if (!token_pruned) {
          const int dot = query_base_dot[hrel] - (negative_dot << 1);
          const float score = (float)dot * scale_cache[hrel];
          if constexpr (ExpBits <= 2) {
            local_weights[hrel] = score;
          } else {
            local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
          }
        }
      }
    }

    #pragma unroll
    for (int hrel = 0; hrel < kGroupSize; hrel++) {
      weights[hrel][thread_idx] = local_weights[hrel];
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      bool block_pruned = false;
      if constexpr (DoBlockPrune) {
        if constexpr (ExpBits <= 2) {
          block_pruned = group_max <= lse_cache[hrel] + prune_log_eps;
        } else {
          block_pruned = group_max <= picokv_fast_expf(prune_log_eps);
        }
      }
      if (!block_pruned) {
        if constexpr (ExpBits <= 2) {
          const float max_score = group_max;
          const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
          if (max_weight > 0.0f) {
            const float d0 = w0 - max_score;
            const float d1 = w1 - max_score;
            if constexpr (ExpBits == 1) {
              constexpr float t0 = -0.6931471805599453f;
              q0 = d0 >= t0 ? 1 : 0;
              q1 = d1 >= t0 ? 1 : 0;
            } else {
              constexpr float t0 = -1.7917594692280550f;
              constexpr float t1 = -0.6931471805599453f;
              constexpr float t2 = -0.1823215567939546f;
              q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
              q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
            }
            scale = max_weight / (float)ExpMaxCode;
          }
        } else {
          const float max_weight = group_max;
          const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
          if (max_weight > 0.0f) {
            q0 = (int)(w0 * inv_scale + 0.5f);
            q1 = (int)(w1 * inv_scale + 0.5f);
            q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
            q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
            scale = max_weight / (float)ExpMaxCode;
          }
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < kGroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      const int global_group = tile_idx * TileGroups + local_group;
      if (global_group < LGroups) {
        const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
        const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

        const float scale20 = exp_scale2[hrel0][local_group];
        const float bias0 = exp_bias[hrel0][local_group];
        if (scale20 > 0.0f) {
          const unsigned long long *planes0 = &exp_planes[hrel0][local_group][0];
          const int pos00 = picokv_weighted_popc64<ExpBits>(value_words.x, planes0);
          const int pos01 = picokv_weighted_popc64<ExpBits>(value_words.y, planes0);
          acc00 += scale20 * (float)pos00 + bias0;
          acc01 += scale20 * (float)pos01 + bias0;
        }

        const float scale21 = exp_scale2[hrel1][local_group];
        const float bias1 = exp_bias[hrel1][local_group];
        if (scale21 > 0.0f) {
          const unsigned long long *planes1 = &exp_planes[hrel1][local_group][0];
          const int pos10 = picokv_weighted_popc64<ExpBits>(value_words.x, planes1);
          const int pos11 = picokv_weighted_popc64<ExpBits>(value_words.y, planes1);
          acc10 += scale21 * (float)pos10 + bias1;
          acc11 += scale21 * (float)pos11 + bias1;
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < kGroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  const int qh0 = qh_base + hrel0;
  const int qh1 = qh_base + hrel1;
  const long base0 = ((long)(B_idx * QH + qh0) * S + S_idx);
  const long base1 = ((long)(B_idx * QH + qh1) * S + S_idx);
  atomicAdd(&atomic_code_output[base0 * Cv + c], acc00);
  atomicAdd(&atomic_code_output[base0 * Cv + c + 1], acc01);
  atomicAdd(&atomic_code_output[base1 * Cv + c], acc10);
  atomicAdd(&atomic_code_output[base1 * Cv + c + 1], acc11);
}



// Runtime-Ck/Cv path for variable PiCoKV code dimensions and templated GQA
// group size. It keeps the current Ck=256/Cv=256 specialized kernels untouched,
// and covers Ck/Cv in {128,192,256,320,384,448,512}.
template <int GroupSize, bool KeyTransposedFixed, int ExpBits, int PruneMode>
__global__ void __launch_bounds__(256, 6) picokv_v1_expbits_atomic2_runtime_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,Cv]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int KBlocks,
  const int Cv,
  const int query_scale_numel,
  const float attention_scale,
  const float prune_log_eps
) {
  constexpr int MaxKBlocks = 8;
  constexpr int MaxQueryWords = MaxKBlocks * 8;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;
  constexpr int MaxTaskSlots = GroupSize == 4 ? 2 : GroupSize; // paired GQA4 needs at most 2 slots for Cv<=512
  constexpr bool DoBlockPrune = (PruneMode & 1) != 0;
  constexpr bool DoPrefixPrune = (PruneMode & 2) != 0;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;
  const int query_words = KBlocks * 8;

  __shared__ unsigned long long query_cache[GroupSize][MaxQueryWords];
  __shared__ int query_base_dot[GroupSize];
  __shared__ int query_block_abs[GroupSize][MaxKBlocks];
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ float prune_dot_threshold_cache[GroupSize];
  __shared__ float weights[GroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[GroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];

  for (int idx = thread_idx; idx < GroupSize * query_words; idx += blockDim.x) {
    const int hrel = idx / query_words;
    const int off = idx - hrel * query_words;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    int base_dot = 0;
    for (int kb = 0; kb < KBlocks; kb++) {
      int block_abs = 0;
      #pragma unroll
      for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
        block_abs += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
      }
      query_block_abs[thread_idx][kb] = block_abs;
      base_dot += block_abs;
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
    if constexpr (PruneMode != 0) {
      prune_dot_threshold_cache[thread_idx] = (lse_cache[thread_idx] + prune_log_eps) / fmaxf(scale_cache[thread_idx], 1.0e-30f);
    }
  }
  __syncthreads();

  float acc0[MaxTaskSlots];
  float acc1[MaxTaskSlots];
  float acc2[MaxTaskSlots];
  float acc3[MaxTaskSlots];
  #pragma unroll
  for (int slot = 0; slot < MaxTaskSlots; slot++) {
    acc0[slot] = 0.0f;
    acc1[slot] = 0.0f;
    acc2[slot] = 0.0f;
    acc3[slot] = 0.0f;
  }
  float den_acc = 0.0f;
  const int pair_count = Cv >> 1;
  const int task_count = GroupSize * pair_count;
  const int pair_task_count = (GroupSize >> 1) * pair_count;


  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weights[GroupSize];
    #pragma unroll
    for (int hrel = 0; hrel < GroupSize; hrel++) {
      if constexpr (ExpBits <= 2) {
        local_weights[hrel] = -3.402823466e38f;
      } else {
        local_weights[hrel] = 0.0f;
      }
    }
    if (l_idx < L) {
      unsigned long long key_words[MaxKBlocks];
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      #pragma unroll
      for (int hrel = 0; hrel < GroupSize; hrel++) {
        int negative_dot = 0;
        int prefix_abs = 0;
        bool token_pruned = false;
        const float dot_threshold = PruneMode == 0 ? -3.402823466e38f : prune_dot_threshold_cache[hrel];
        for (int kb = 0; kb < KBlocks; kb++) {
          if (!token_pruned) {
            const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
            int negative_block = 0;
            #pragma unroll
            for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
              negative_block += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
            }
            negative_dot += negative_block;
            if constexpr (DoPrefixPrune) {
              prefix_abs += query_block_abs[hrel][kb];
              if (kb < KBlocks - 1) {
                const int partial_dot = prefix_abs - (negative_dot << 1);
                const int remaining_abs = query_base_dot[hrel] - prefix_abs;
                if ((float)(partial_dot + remaining_abs) <= dot_threshold) {
                  token_pruned = true;
                }
              }
            }
          }
        }
        if (!token_pruned) {
          const int dot = query_base_dot[hrel] - (negative_dot << 1);
          const float score = (float)dot * scale_cache[hrel];
          if constexpr (ExpBits <= 2) {
            local_weights[hrel] = score;
          } else {
            local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
          }
        }
      }
    }

    #pragma unroll
    for (int hrel = 0; hrel < GroupSize; hrel++) {
      weights[hrel][thread_idx] = local_weights[hrel];
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;
      if (quant_task >= GroupSize * TileGroups) {
        continue;
      }

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      bool block_pruned = false;
      if constexpr (DoBlockPrune) {
        if constexpr (ExpBits <= 2) {
          block_pruned = group_max <= lse_cache[hrel] + prune_log_eps;
        } else {
          block_pruned = group_max <= picokv_fast_expf(prune_log_eps);
        }
      }
      if (!block_pruned) {
        if constexpr (ExpBits <= 2) {
          const float max_score = group_max;
          const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
          if (max_weight > 0.0f) {
            const float d0 = w0 - max_score;
            const float d1 = w1 - max_score;
            if constexpr (ExpBits == 1) {
              constexpr float t0 = -0.6931471805599453f;
              q0 = d0 >= t0 ? 1 : 0;
              q1 = d1 >= t0 ? 1 : 0;
            } else {
              constexpr float t0 = -1.7917594692280550f;
              constexpr float t1 = -0.6931471805599453f;
              constexpr float t2 = -0.1823215567939546f;
              q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
              q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
            }
            scale = max_weight / (float)ExpMaxCode;
          }
        } else {
          const float max_weight = group_max;
          const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
          if (max_weight > 0.0f) {
            q0 = (int)(w0 * inv_scale + 0.5f);
            q1 = (int)(w1 * inv_scale + 0.5f);
            q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
            q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
            scale = max_weight / (float)ExpMaxCode;
          }
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < GroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    if constexpr (GroupSize == 4) {
      // Reuse each loaded value word for two GQA heads. This is the key
      // 256:256 fast-path trick generalized to every supported Cv.
      #pragma unroll
      for (int slot = 0; slot < MaxTaskSlots; slot++) {
        const int pair_task_idx = thread_idx + slot * blockDim.x;
        if (pair_task_idx < pair_task_count) {
          const int hrel0 = pair_task_idx / pair_count;
          const int hrel1 = hrel0 + 2;
          const int pair_idx = pair_task_idx - hrel0 * pair_count;
          const int c = pair_idx << 1;
          #pragma unroll
          for (int local_group = 0; local_group < TileGroups; local_group++) {
            const int global_group = tile_idx * TileGroups + local_group;
            if (global_group < LGroups) {
              const float scale20 = exp_scale2[hrel0][local_group];
              const float scale21 = exp_scale2[hrel1][local_group];
              if (scale20 > 0.0f || scale21 > 0.0f) {
                const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
                const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

                if (scale20 > 0.0f) {
                  const float bias0 = exp_bias[hrel0][local_group];
                  const unsigned long long *planes0 = &exp_planes[hrel0][local_group][0];
                  const int pos00 = picokv_weighted_popc64<ExpBits>(value_words.x, planes0);
                  const int pos01 = picokv_weighted_popc64<ExpBits>(value_words.y, planes0);
                  acc0[slot] += scale20 * (float)pos00 + bias0;
                  acc1[slot] += scale20 * (float)pos01 + bias0;
                }

                if (scale21 > 0.0f) {
                  const float bias1 = exp_bias[hrel1][local_group];
                  const unsigned long long *planes1 = &exp_planes[hrel1][local_group][0];
                  const int pos10 = picokv_weighted_popc64<ExpBits>(value_words.x, planes1);
                  const int pos11 = picokv_weighted_popc64<ExpBits>(value_words.y, planes1);
                  acc2[slot] += scale21 * (float)pos10 + bias1;
                  acc3[slot] += scale21 * (float)pos11 + bias1;
                }
              }
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int slot = 0; slot < MaxTaskSlots; slot++) {
        const int task_idx = thread_idx + slot * blockDim.x;
        if (task_idx < task_count) {
          const int hrel = task_idx / pair_count;
          const int pair_idx = task_idx - hrel * pair_count;
          const int c = pair_idx << 1;
          #pragma unroll
          for (int local_group = 0; local_group < TileGroups; local_group++) {
            const int global_group = tile_idx * TileGroups + local_group;
            if (global_group < LGroups) {
              const float scale2 = exp_scale2[hrel][local_group];
              const float bias = exp_bias[hrel][local_group];
              if (scale2 > 0.0f) {
                const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
                const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));
                const unsigned long long *planes = &exp_planes[hrel][local_group][0];
                const int pos0 = picokv_weighted_popc64<ExpBits>(value_words.x, planes);
                const int pos1 = picokv_weighted_popc64<ExpBits>(value_words.y, planes);
                acc0[slot] += scale2 * (float)pos0 + bias;
                acc1[slot] += scale2 * (float)pos1 + bias;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  if constexpr (GroupSize == 4) {
    #pragma unroll
    for (int slot = 0; slot < MaxTaskSlots; slot++) {
      const int pair_task_idx = thread_idx + slot * blockDim.x;
      if (pair_task_idx < pair_task_count) {
        const int hrel0 = pair_task_idx / pair_count;
        const int hrel1 = hrel0 + 2;
        const int pair_idx = pair_task_idx - hrel0 * pair_count;
        const int c = pair_idx << 1;
        const int qh0 = qh_base + hrel0;
        const int qh1 = qh_base + hrel1;
        const long base0 = ((long)(B_idx * QH + qh0) * S + S_idx);
        const long base1 = ((long)(B_idx * QH + qh1) * S + S_idx);
        atomicAdd(&atomic_code_output[base0 * Cv + c], acc0[slot]);
        atomicAdd(&atomic_code_output[base0 * Cv + c + 1], acc1[slot]);
        atomicAdd(&atomic_code_output[base1 * Cv + c], acc2[slot]);
        atomicAdd(&atomic_code_output[base1 * Cv + c + 1], acc3[slot]);
      }
    }
  } else {
    #pragma unroll
    for (int slot = 0; slot < MaxTaskSlots; slot++) {
      const int task_idx = thread_idx + slot * blockDim.x;
      if (task_idx < task_count) {
        const int hrel = task_idx / pair_count;
        const int pair_idx = task_idx - hrel * pair_count;
        const int c = pair_idx << 1;
        const int qh = qh_base + hrel;
        const long base = ((long)(B_idx * QH + qh) * S + S_idx);
        atomicAdd(&atomic_code_output[base * Cv + c], acc0[slot]);
        atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1[slot]);
      }
    }
  }
}
template <int GroupSize, int KBlocks, int Cv, bool KeyTransposedFixed, int ExpBits, int PruneMode>
__global__ void __launch_bounds__(256, 6) picokv_v1_expbits_atomic2_fixed_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,Cv]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale,
  const float prune_log_eps
) {
  constexpr int MaxKBlocks = KBlocks;
  constexpr int MaxQueryWords = MaxKBlocks * 8;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;
  constexpr int MaxTaskSlots = GroupSize == 4 ? 2 : GroupSize; // paired GQA4 needs at most 2 slots for Cv<=512
  constexpr bool DoBlockPrune = (PruneMode & 1) != 0;
  constexpr bool DoPrefixPrune = (PruneMode & 2) != 0;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;
  constexpr int query_words = KBlocks * 8;

  __shared__ unsigned long long query_cache[GroupSize][MaxQueryWords];
  __shared__ int query_base_dot[GroupSize];
  __shared__ int query_block_abs[GroupSize][MaxKBlocks];
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ float prune_dot_threshold_cache[GroupSize];
  __shared__ float weights[GroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[GroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];

  for (int idx = thread_idx; idx < GroupSize * query_words; idx += blockDim.x) {
    const int hrel = idx / query_words;
    const int off = idx - hrel * query_words;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int kb = 0; kb < KBlocks; kb++) {
      int block_abs = 0;
      #pragma unroll
      for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
        block_abs += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
      }
      query_block_abs[thread_idx][kb] = block_abs;
      base_dot += block_abs;
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
    if constexpr (PruneMode != 0) {
      prune_dot_threshold_cache[thread_idx] = (lse_cache[thread_idx] + prune_log_eps) / fmaxf(scale_cache[thread_idx], 1.0e-30f);
    }
  }
  __syncthreads();

  float acc0[MaxTaskSlots];
  float acc1[MaxTaskSlots];
  float acc2[MaxTaskSlots];
  float acc3[MaxTaskSlots];
  #pragma unroll
  for (int slot = 0; slot < MaxTaskSlots; slot++) {
    acc0[slot] = 0.0f;
    acc1[slot] = 0.0f;
    acc2[slot] = 0.0f;
    acc3[slot] = 0.0f;
  }
  float den_acc = 0.0f;
  constexpr int pair_count = Cv >> 1;
  constexpr int task_count = GroupSize * pair_count;
  constexpr int pair_task_count = (GroupSize >> 1) * pair_count;


  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weights[GroupSize];
    #pragma unroll
    for (int hrel = 0; hrel < GroupSize; hrel++) {
      if constexpr (ExpBits <= 2) {
        local_weights[hrel] = -3.402823466e38f;
      } else {
        local_weights[hrel] = 0.0f;
      }
    }
    if (l_idx < L) {
      unsigned long long key_words[MaxKBlocks];
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      #pragma unroll
      for (int hrel = 0; hrel < GroupSize; hrel++) {
        int negative_dot = 0;
        int prefix_abs = 0;
        bool token_pruned = false;
        const float dot_threshold = PruneMode == 0 ? -3.402823466e38f : prune_dot_threshold_cache[hrel];
        #pragma unroll
        for (int kb = 0; kb < KBlocks; kb++) {
          if (!token_pruned) {
            const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
            int negative_block = 0;
            #pragma unroll
            for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
              negative_block += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
            }
            negative_dot += negative_block;
            if constexpr (DoPrefixPrune) {
              prefix_abs += query_block_abs[hrel][kb];
              if (kb < KBlocks - 1) {
                const int partial_dot = prefix_abs - (negative_dot << 1);
                const int remaining_abs = query_base_dot[hrel] - prefix_abs;
                if ((float)(partial_dot + remaining_abs) <= dot_threshold) {
                  token_pruned = true;
                }
              }
            }
          }
        }
        if (!token_pruned) {
          const int dot = query_base_dot[hrel] - (negative_dot << 1);
          const float score = (float)dot * scale_cache[hrel];
          if constexpr (ExpBits <= 2) {
            local_weights[hrel] = score;
          } else {
            local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
          }
        }
      }
    }

    #pragma unroll
    for (int hrel = 0; hrel < GroupSize; hrel++) {
      weights[hrel][thread_idx] = local_weights[hrel];
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;
      if (quant_task >= GroupSize * TileGroups) {
        continue;
      }

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      bool block_pruned = false;
      if constexpr (DoBlockPrune) {
        if constexpr (ExpBits <= 2) {
          block_pruned = group_max <= lse_cache[hrel] + prune_log_eps;
        } else {
          block_pruned = group_max <= picokv_fast_expf(prune_log_eps);
        }
      }
      if (!block_pruned) {
        if constexpr (ExpBits <= 2) {
          const float max_score = group_max;
          const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
          if (max_weight > 0.0f) {
            const float d0 = w0 - max_score;
            const float d1 = w1 - max_score;
            if constexpr (ExpBits == 1) {
              constexpr float t0 = -0.6931471805599453f;
              q0 = d0 >= t0 ? 1 : 0;
              q1 = d1 >= t0 ? 1 : 0;
            } else {
              constexpr float t0 = -1.7917594692280550f;
              constexpr float t1 = -0.6931471805599453f;
              constexpr float t2 = -0.1823215567939546f;
              q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
              q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
            }
            scale = max_weight / (float)ExpMaxCode;
          }
        } else {
          const float max_weight = group_max;
          const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
          if (max_weight > 0.0f) {
            q0 = (int)(w0 * inv_scale + 0.5f);
            q1 = (int)(w1 * inv_scale + 0.5f);
            q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
            q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
            scale = max_weight / (float)ExpMaxCode;
          }
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < GroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    if constexpr (GroupSize == 4) {
      // Reuse each loaded value word for two GQA heads. This is the key
      // 256:256 fast-path trick generalized to every supported Cv.
      #pragma unroll
      for (int slot = 0; slot < MaxTaskSlots; slot++) {
        const int pair_task_idx = thread_idx + slot * blockDim.x;
        if (pair_task_idx < pair_task_count) {
          const int hrel0 = pair_task_idx / pair_count;
          const int hrel1 = hrel0 + 2;
          const int pair_idx = pair_task_idx - hrel0 * pair_count;
          const int c = pair_idx << 1;
          #pragma unroll
          for (int local_group = 0; local_group < TileGroups; local_group++) {
            const int global_group = tile_idx * TileGroups + local_group;
            if (global_group < LGroups) {
              const float scale20 = exp_scale2[hrel0][local_group];
              const float scale21 = exp_scale2[hrel1][local_group];
              if (scale20 > 0.0f || scale21 > 0.0f) {
                const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
                const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

                if (scale20 > 0.0f) {
                  const float bias0 = exp_bias[hrel0][local_group];
                  const unsigned long long *planes0 = &exp_planes[hrel0][local_group][0];
                  const int pos00 = picokv_weighted_popc64<ExpBits>(value_words.x, planes0);
                  const int pos01 = picokv_weighted_popc64<ExpBits>(value_words.y, planes0);
                  acc0[slot] += scale20 * (float)pos00 + bias0;
                  acc1[slot] += scale20 * (float)pos01 + bias0;
                }

                if (scale21 > 0.0f) {
                  const float bias1 = exp_bias[hrel1][local_group];
                  const unsigned long long *planes1 = &exp_planes[hrel1][local_group][0];
                  const int pos10 = picokv_weighted_popc64<ExpBits>(value_words.x, planes1);
                  const int pos11 = picokv_weighted_popc64<ExpBits>(value_words.y, planes1);
                  acc2[slot] += scale21 * (float)pos10 + bias1;
                  acc3[slot] += scale21 * (float)pos11 + bias1;
                }
              }
            }
          }
        }
      }
    } else {
      #pragma unroll
      for (int slot = 0; slot < MaxTaskSlots; slot++) {
        const int task_idx = thread_idx + slot * blockDim.x;
        if (task_idx < task_count) {
          const int hrel = task_idx / pair_count;
          const int pair_idx = task_idx - hrel * pair_count;
          const int c = pair_idx << 1;
          #pragma unroll
          for (int local_group = 0; local_group < TileGroups; local_group++) {
            const int global_group = tile_idx * TileGroups + local_group;
            if (global_group < LGroups) {
              const float scale2 = exp_scale2[hrel][local_group];
              const float bias = exp_bias[hrel][local_group];
              if (scale2 > 0.0f) {
                const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
                const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));
                const unsigned long long *planes = &exp_planes[hrel][local_group][0];
                const int pos0 = picokv_weighted_popc64<ExpBits>(value_words.x, planes);
                const int pos1 = picokv_weighted_popc64<ExpBits>(value_words.y, planes);
                acc0[slot] += scale2 * (float)pos0 + bias;
                acc1[slot] += scale2 * (float)pos1 + bias;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  if constexpr (GroupSize == 4) {
    #pragma unroll
    for (int slot = 0; slot < MaxTaskSlots; slot++) {
      const int pair_task_idx = thread_idx + slot * blockDim.x;
      if (pair_task_idx < pair_task_count) {
        const int hrel0 = pair_task_idx / pair_count;
        const int hrel1 = hrel0 + 2;
        const int pair_idx = pair_task_idx - hrel0 * pair_count;
        const int c = pair_idx << 1;
        const int qh0 = qh_base + hrel0;
        const int qh1 = qh_base + hrel1;
        const long base0 = ((long)(B_idx * QH + qh0) * S + S_idx);
        const long base1 = ((long)(B_idx * QH + qh1) * S + S_idx);
        atomicAdd(&atomic_code_output[base0 * Cv + c], acc0[slot]);
        atomicAdd(&atomic_code_output[base0 * Cv + c + 1], acc1[slot]);
        atomicAdd(&atomic_code_output[base1 * Cv + c], acc2[slot]);
        atomicAdd(&atomic_code_output[base1 * Cv + c + 1], acc3[slot]);
      }
    }
  } else {
    #pragma unroll
    for (int slot = 0; slot < MaxTaskSlots; slot++) {
      const int task_idx = thread_idx + slot * blockDim.x;
      if (task_idx < task_count) {
        const int hrel = task_idx / pair_count;
        const int pair_idx = task_idx - hrel * pair_count;
        const int c = pair_idx << 1;
        const int qh = qh_base + hrel;
        const long base = ((long)(B_idx * QH + qh) * S + S_idx);
        atomicAdd(&atomic_code_output[base * Cv + c], acc0[slot]);
        atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1[slot]);
      }
    }
  }
}





// Runtime-Ck/Cv path with fixed 8-block unrolled score loops for Ck >= 320.
// It keeps the current Ck=256/Cv=256 specialized kernels untouched, and covers
// Ck/Cv in {128,192,256,320,384,448,512}.
// MinBlocksPerSm feeds __launch_bounds__: 6 (the historical setting) caps the
// kernel at ~40 registers/thread and spills heavily at Ck=320 (measured ~250MB
// local-memory traffic per b16 decode layer). 4 allows 64 registers/thread and
// removes most of those spills at the cost of occupancy (32 vs 48 warps/SM);
// the dispatcher picks per PICOKV_TAIL_LB. Same arithmetic either way.
// KBlocksT > 0 compiles the KBlocks unrolled loops (key load / key match /
// query_block_abs) for that exact block count, dropping the dead `kb <
// KBlocks` guarded iterations and the unused key_words registers; at the
// production Ck=320 (KBlocks=5) that removes 3 of 8 predicated iterations on
// an XU-issue-bound kernel. KBlocksT == 0 keeps the fully generic runtime form.
// UseBmmaScore=1 replaces the popc key-match section with a tensor-core b1
// bmma (m8n8k128 AND+POPC) score path: A rows 0-3 hold the query's positive
// magnitude planes (mag & ~sign) and rows 4-7 the negative planes (mag & sign)
// of the same bit, so ONE bmma per (chunk, bit) covers all four GQA-4 heads;
// lanes exchange pos/neg halves with one shfl_xor(16). KBlocks=5 is 2.5 bmma
// chunks -> 3 chunks with the last one zero-padded. All-integer arithmetic,
// identical dots to the popc path (verified bitwise on single-writer shapes).
// BlockTilesT = 256-token tiles per block (grid.x = ceil(L / (256*BT))). 2 is
// the default; 1 doubles the grid for small launches (b1 at short L fills
// only ~1 block/SM at BT=2) at the cost of 2x query staging + atomics.
template <int GroupSize, bool KeyTransposedFixed, int ExpBits, int PruneMode, int MinBlocksPerSm = 6, int KBlocksT = 0, int UseBmmaScore = 0, int BlockTilesT = 2>
__global__ void __launch_bounds__(256, MinBlocksPerSm) picokv_v1_expbits_atomic2_runtime_unrolled_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,Cv]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int KBlocks,
  const int Cv,
  const int query_scale_numel,
  const float attention_scale,
  const float prune_log_eps,
  const int *__restrict__ global_lens,
  const int *__restrict__ sched_req,
  const int *__restrict__ sched_tile
) {
  constexpr int MaxKBlocks = 8;
  // Loop bound / array extent for the KBlocks-keyed loops: compile-time when
  // specialized, MaxKBlocks with runtime guards when generic.
  constexpr int KeyLoopBlocks = KBlocksT > 0 ? KBlocksT : MaxKBlocks;
  // Compile-time KBlocks value for addressing (strides); runtime arg when generic.
  constexpr int KeyAddrBlocks = KBlocksT > 0 ? KBlocksT : 0;
  constexpr int MaxQueryWords = MaxKBlocks * 8;
  constexpr int BlockTiles = BlockTilesT;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;
  constexpr int MaxTaskSlots = GroupSize; // GroupSize * (512 / 2) / 256
  constexpr bool DoBlockPrune = (PruneMode & 1) != 0;
  constexpr bool DoPrefixPrune = (PruneMode & 2) != 0;
  // bmma score path shapes: GQA-4 only (A rows 0-3 = query positive magnitude
  // planes, rows 4-7 = negative planes of the same bit), KBlocks=5 (3 bmma
  // chunks of 128 bits, chunk 2 zero-padded past kb 4), unpruned.
  static_assert(!UseBmmaScore || (GroupSize == 4 && KBlocksT == 5 && PruneMode == 0),
                "bmma score path is GQA-4 / Ck=320 / unpruned only");
  // The bmma score loop sweeps the tile in 8-token groups (m8n8 B side).
  constexpr int BmmaTokenGroups = kTileSize / 8;
  constexpr int BmmaGroupPasses = BmmaTokenGroups / QuantWarps;

  const int _sched_bx = blockIdx.x;
  // Block-scheduled (paged) grid: sched_req/sched_tile map each grid.x
  // block to a (request, local-tile) pair so only REAL blocks launch
  // (no batch-max over-launch). nullptr => legacy blockIdx derivation.
  const int tile_block_idx = (sched_tile != nullptr) ? sched_tile[_sched_bx] : _sched_bx;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = (sched_req != nullptr) ? sched_req[_sched_bx] : (BKVH_idx / KVH);
  const int KVH_idx = (sched_req != nullptr) ? BKVH_idx : (BKVH_idx - B_idx * KVH);
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;
  const int query_words = KBlocks * 8;

  // Varlen (ragged batch): batch element b has L_b <= L real global tokens
  // (L is the padded physical stride). Skip fully-padded blocks and bound the
  // in-range token test by L_b so padding is neither read nor scored.
  const int L_b = (global_lens != nullptr) ? global_lens[B_idx] : L;
  if (tile_block_idx * (BlockTiles * kTileSize) >= L_b) return;

  __shared__ unsigned long long query_cache[GroupSize][MaxQueryWords];
  __shared__ int query_base_dot[GroupSize];
  __shared__ int query_block_abs[GroupSize][MaxKBlocks];
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ float prune_dot_threshold_cache[GroupSize];
  __shared__ float weights[GroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[GroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];
  // bmma score staging. bmma_a[chunk][bit]: 8 A rows x 4 x 32-bit segments;
  // row h (0-3) = a_pos of head h (mag & ~sign), row 4+h = a_neg (mag & sign).
  // bmma_b[warp]: 8 key-sign columns x 4 segments for the warp's current
  // 8-token group. bmma_qneg: popc(q_mag & q_sign) per (head, chunk, bit).
  // All sized to 1 in popc mode so no shared memory is wasted.
  constexpr int kBmmaChunks = UseBmmaScore ? 3 : 1;
  constexpr int kBmmaBits = UseBmmaScore ? kQMagBits : 1;
  constexpr int kBmmaRowWords = UseBmmaScore ? 32 : 1;
  __shared__ __align__(128) unsigned int bmma_a[kBmmaChunks][kBmmaBits][kBmmaRowWords];
  __shared__ __align__(128) unsigned int bmma_b[UseBmmaScore ? QuantWarps : 1][kBmmaRowWords];
  __shared__ int bmma_qneg[UseBmmaScore ? GroupSize : 1][kBmmaChunks][kBmmaBits];

  for (int idx = thread_idx; idx < GroupSize * query_words; idx += blockDim.x) {
    const int hrel = idx / query_words;
    const int off = idx - hrel * query_words;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int kb = 0; kb < KeyLoopBlocks; kb++) {
      if (KBlocksT > 0 || kb < KBlocks) {
        int block_abs = 0;
        #pragma unroll
        for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
          block_abs += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
        }
        query_block_abs[thread_idx][kb] = block_abs;
        base_dot += block_abs;
      }
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
    if constexpr (PruneMode != 0) {
      prune_dot_threshold_cache[thread_idx] = (lse_cache[thread_idx] + prune_log_eps) / fmaxf(scale_cache[thread_idx], 1.0e-30f);
    }
  }
  __syncthreads();

  if constexpr (UseBmmaScore) {
    // Stage bmma_a: idx = ((chunk * kQMagBits + bit) * GroupSize + hrel) * 4 + seg.
    // Rows 0-3 = a_pos of heads 0-3, rows 4-7 = a_neg. Chunk 2 covers kb 4
    // only, so its segments 2-3 stay zero.
    for (int idx = thread_idx; idx < 3 * kQMagBits * GroupSize * 4; idx += blockDim.x) {
      const int seg = idx & 3;
      const int hrel = (idx >> 2) & 3;
      const int bit = (idx >> 4) % kQMagBits;
      const int chunk = idx / (kQMagBits * GroupSize * 4);
      const int kb = chunk * 2 + (seg >> 1);
      unsigned int pos_piece = 0;
      unsigned int neg_piece = 0;
      if (chunk < 2 || seg < 2) {
        const unsigned long long sign_word = query_cache[hrel][kb * 8];
        const unsigned long long mag_word = query_cache[hrel][kb * 8 + bit + 1];
        const unsigned int sign_piece = (unsigned int)((seg & 1) ? (sign_word >> 32) : sign_word);
        const unsigned int mag_piece = (unsigned int)((seg & 1) ? (mag_word >> 32) : mag_word);
        pos_piece = mag_piece & ~sign_piece;
        neg_piece = mag_piece & sign_piece;
      }
      bmma_a[chunk][bit][hrel * 4 + seg] = pos_piece;
      bmma_a[chunk][bit][(GroupSize + hrel) * 4 + seg] = neg_piece;
    }
    // popc(q_mag & q_sign) per (head, chunk, bit): the query's own negative
    // planes' weight, added back in the mismatch identity.
    for (int idx = thread_idx; idx < GroupSize * 3 * kQMagBits; idx += blockDim.x) {
      const int bit = idx % kQMagBits;
      const int chunk = (idx / kQMagBits) % 3;
      const int hrel = idx / (3 * kQMagBits);
      int qneg = 0;
      #pragma unroll
      for (int half = 0; half < 2; half++) {
        const int kb = chunk * 2 + half;
        if (kb < KBlocks) {
          qneg += __popcll(query_cache[hrel][kb * 8 + bit + 1] & query_cache[hrel][kb * 8]);
        }
      }
      bmma_qneg[hrel][chunk][bit] = qneg;
    }
    __syncthreads();
  }

  float acc0[MaxTaskSlots];
  float acc1[MaxTaskSlots];
  #pragma unroll
  for (int slot = 0; slot < MaxTaskSlots; slot++) {
    acc0[slot] = 0.0f;
    acc1[slot] = 0.0f;
  }
  float den_acc = 0.0f;
  const int pair_count = Cv >> 1;
  const int task_count = GroupSize * pair_count;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;

    if constexpr (UseBmmaScore) {
      // Tensor-core score path. Each warp sweeps the tile's 8-token groups;
      // ONE m8n8k128 bmma per (chunk, bit) covers all 4 heads: A rows 0-3
      // hold the query's positive magnitude planes (mag & ~sign) and rows 4-7
      // the negative planes (mag & sign) of the same bit; B columns hold the
      // 8 tokens' key sign words. The m8n8 accumulator gives lane L
      // C[L/4][2*(L%4)] in x[0] and C[L/4][2*(L%4)+1] in x[1], so lanes 0-15
      // carry positive-plane dots and lanes 16-31 negative-plane dots of the
      // same (head, token) pairs — one shfl_xor(16) reunites them and
      //   mismatch = popc(a_pos & k) + popc(a_neg) - popc(a_neg & k)
      // completes in-lane. All integer; dots identical to the popc path.
      #pragma unroll
      for (int group_pass = 0; group_pass < BmmaGroupPasses; group_pass++) {
        const int token_group = group_pass * QuantWarps + warp_idx;
        const int token_base = tile_idx * kTileSize + token_group * 8;
        const int h = (lane_idx & 15) >> 2;
        const int t0 = (lane_idx & 3) << 1;
        int dot0 = query_base_dot[h];
        int dot1 = dot0;

        #pragma unroll
        for (int chunk = 0; chunk < 3; chunk++) {
          // B side: lane -> (token column b_col, 32-bit segment b_seg); the
          // four segments tile the chunk's 128 key bits. Chunk 2 covers kb 4
          // only, so its segments 2-3 stage zero.
          const int b_col = lane_idx >> 2;
          const int b_seg = lane_idx & 3;
          const int l_idx_b = token_base + b_col;
          unsigned int key_piece = 0;
          if (l_idx_b < L_b && (chunk < 2 || b_seg < 2)) {
            const int kb = chunk * 2 + (b_seg >> 1);
            const unsigned long long key_word = load_packed_key_word(
              packed_key, B_idx, KVH_idx, KVH, L, KeyAddrBlocks > 0 ? KeyAddrBlocks : KBlocks, l_idx_b, kb, KeyTransposedFixed
            );
            key_piece = (unsigned int)((b_seg & 1) ? (key_word >> 32) : key_word);
          }
          bmma_b[warp_idx][lane_idx] = key_piece;
          __syncwarp();

          // B (key sign pieces) is bit-independent: load the fragment once
          // per chunk and reuse it across all kQMagBits A planes.
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::col_major> b_frag;
          nvcuda::wmma::load_matrix_sync(b_frag, &bmma_b[warp_idx][0], 128);

          #pragma unroll
          for (int bit = 0; bit < kQMagBits; bit++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 128, int> c_frag;
            nvcuda::wmma::fill_fragment(c_frag, 0);
            nvcuda::wmma::load_matrix_sync(a_frag, &bmma_a[chunk][bit][0], 128);
            nvcuda::wmma::bmma_sync(
              c_frag, a_frag, b_frag, c_frag,
              nvcuda::wmma::experimental::bmmaBitOpAND,
              nvcuda::wmma::experimental::bmmaAccumulateOpPOPC
            );
            // Lanes 0-15 hold popc(a_pos & k); lanes 16-31 popc(a_neg & k)
            // for the same (head, token) pairs (rows 4-7 of A).
            const int v0 = c_frag.x[0];
            const int v1 = c_frag.x[1];
            const int o0 = __shfl_xor_sync(0xffffffffu, v0, 16);
            const int o1 = __shfl_xor_sync(0xffffffffu, v1, 16);
            const bool pos_lane = lane_idx < 16;
            const int p0 = pos_lane ? v0 : o0;
            const int p1 = pos_lane ? v1 : o1;
            const int n0 = pos_lane ? o0 : v0;
            const int n1 = pos_lane ? o1 : v1;
            const int qneg = bmma_qneg[h][chunk][bit];
            dot0 -= (p0 + qneg - n0) << (bit + 1);
            dot1 -= (p1 + qneg - n1) << (bit + 1);
          }
        }

        // Lanes 0-15 already hold complete dots for every (head, token) pair
        // of this 8-token group (4 heads x 2 tokens per lane); lanes 16-31
        // mirror them, so only half the warp writes.
        if (lane_idx < 16) {
          const int local0 = token_group * 8 + t0;
          const int local1 = local0 + 1;
          const int l0 = token_base + t0;
          const int l1 = l0 + 1;
          float out0 = 0.0f;
          float out1 = 0.0f;
          if constexpr (ExpBits <= 2) {
            out0 = -3.402823466e38f;
            out1 = -3.402823466e38f;
          }
          if (l0 < L_b) {
            const float score0 = (float)dot0 * scale_cache[h];
            if constexpr (ExpBits <= 2) {
              out0 = score0;
            } else {
              out0 = picokv_fast_expf(score0 - lse_cache[h]);
            }
          }
          if (l1 < L_b) {
            const float score1 = (float)dot1 * scale_cache[h];
            if constexpr (ExpBits <= 2) {
              out1 = score1;
            } else {
              out1 = picokv_fast_expf(score1 - lse_cache[h]);
            }
          }
          weights[h][local0] = out0;
          weights[h][local1] = out1;
        }
      }
    } else {
      const int l_idx = tile_idx * kTileSize + thread_idx;

      float local_weights[GroupSize];
      #pragma unroll
      for (int hrel = 0; hrel < GroupSize; hrel++) {
        if constexpr (ExpBits <= 2) {
          local_weights[hrel] = -3.402823466e38f;
        } else {
          local_weights[hrel] = 0.0f;
        }
      }
      if (l_idx < L_b) {
        unsigned long long key_words[KeyLoopBlocks];
        #pragma unroll
        for (int kb = 0; kb < KeyLoopBlocks; kb++) {
          if (KBlocksT > 0 || kb < KBlocks) {
            key_words[kb] = load_packed_key_word(
              packed_key, B_idx, KVH_idx, KVH, L, KeyAddrBlocks > 0 ? KeyAddrBlocks : KBlocks, l_idx, kb, KeyTransposedFixed
            );
          }
        }

        #pragma unroll
        for (int hrel = 0; hrel < GroupSize; hrel++) {
          int negative_dot = 0;
          int prefix_abs = 0;
          bool token_pruned = false;
          const float dot_threshold = PruneMode == 0 ? -3.402823466e38f : prune_dot_threshold_cache[hrel];
          #pragma unroll
          for (int kb = 0; kb < KeyLoopBlocks; kb++) {
            if ((KBlocksT > 0 || kb < KBlocks) && !token_pruned) {
              const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
              int negative_block = 0;
              #pragma unroll
              for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
                negative_block += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
              }
              negative_dot += negative_block;
              if constexpr (DoPrefixPrune) {
                prefix_abs += query_block_abs[hrel][kb];
                if (kb < (KBlocksT > 0 ? KBlocksT : KBlocks) - 1) {
                  const int partial_dot = prefix_abs - (negative_dot << 1);
                  const int remaining_abs = query_base_dot[hrel] - prefix_abs;
                  if ((float)(partial_dot + remaining_abs) <= dot_threshold) {
                    token_pruned = true;
                  }
                }
              }
            }
          }
          if (!token_pruned) {
            const int dot = query_base_dot[hrel] - (negative_dot << 1);
            const float score = (float)dot * scale_cache[hrel];
            if constexpr (ExpBits <= 2) {
              local_weights[hrel] = score;
            } else {
              local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
            }
          }
        }
      }

      #pragma unroll
      for (int hrel = 0; hrel < GroupSize; hrel++) {
        weights[hrel][thread_idx] = local_weights[hrel];
      }
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;
      if (quant_task >= GroupSize * TileGroups) {
        continue;
      }

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      bool block_pruned = false;
      if constexpr (DoBlockPrune) {
        if constexpr (ExpBits <= 2) {
          block_pruned = group_max <= lse_cache[hrel] + prune_log_eps;
        } else {
          block_pruned = group_max <= picokv_fast_expf(prune_log_eps);
        }
      }
      if (!block_pruned) {
        if constexpr (ExpBits <= 2) {
          const float max_score = group_max;
          const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
          if (max_weight > 0.0f) {
            const float d0 = w0 - max_score;
            const float d1 = w1 - max_score;
            if constexpr (ExpBits == 1) {
              constexpr float t0 = -0.6931471805599453f;
              q0 = d0 >= t0 ? 1 : 0;
              q1 = d1 >= t0 ? 1 : 0;
            } else {
              constexpr float t0 = -1.7917594692280550f;
              constexpr float t1 = -0.6931471805599453f;
              constexpr float t2 = -0.1823215567939546f;
              q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
              q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
            }
            scale = max_weight / (float)ExpMaxCode;
          }
        } else {
          const float max_weight = group_max;
          const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
          if (max_weight > 0.0f) {
            q0 = (int)(w0 * inv_scale + 0.5f);
            q1 = (int)(w1 * inv_scale + 0.5f);
            q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
            q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
            scale = max_weight / (float)ExpMaxCode;
          }
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < GroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    #pragma unroll
    for (int slot = 0; slot < MaxTaskSlots; slot++) {
      const int task_idx = thread_idx + slot * blockDim.x;
      if (task_idx < task_count) {
        const int hrel = task_idx / pair_count;
        const int pair_idx = task_idx - hrel * pair_count;
        const int c = pair_idx << 1;
        #pragma unroll
        for (int local_group = 0; local_group < TileGroups; local_group++) {
          const int global_group = tile_idx * TileGroups + local_group;
          if (global_group < LGroups) {
            const float scale2 = exp_scale2[hrel][local_group];
            const float bias = exp_bias[hrel][local_group];
            if (scale2 > 0.0f) {
              const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
              const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));
              const unsigned long long *planes = &exp_planes[hrel][local_group][0];
              const int pos0 = picokv_weighted_popc64<ExpBits>(value_words.x, planes);
              const int pos1 = picokv_weighted_popc64<ExpBits>(value_words.y, planes);
              acc0[slot] += scale2 * (float)pos0 + bias;
              acc1[slot] += scale2 * (float)pos1 + bias;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  #pragma unroll
  for (int slot = 0; slot < MaxTaskSlots; slot++) {
    const int task_idx = thread_idx + slot * blockDim.x;
    if (task_idx < task_count) {
      const int hrel = task_idx / pair_count;
      const int pair_idx = task_idx - hrel * pair_count;
      const int c = pair_idx << 1;
      const int qh = qh_base + hrel;
      const long base = ((long)(B_idx * QH + qh) * S + S_idx);
      atomicAdd(&atomic_code_output[base * Cv + c], acc0[slot]);
      atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1[slot]);
    }
  }
}

// Tensor-core (mma b1 AND+POPC) score + value variant of the kernel above for
// the production GQA-4 / Ck=320 / Cv=320 shape (bitwise-identical outputs).
#include "picokv_tc_tail.cuh"


// Experimental GQA=8 split path: two GQA4-sized blocks per KV head.
template <bool KeyTransposedFixed, int ExpBits>
__global__ void __launch_bounds__(256, 8) picokv_v1_expbits_atomic2_gqa8_split4_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,256]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale
) {
  constexpr int KBlocks = 4;
  constexpr int Cv = 256;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = 2;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVHHalf_idx = blockIdx.z;
  const int half_idx = BKVHHalf_idx & 1;
  const int BKVH_idx = BKVHHalf_idx >> 1;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * 8 + half_idx * kGroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;

  __shared__ unsigned long long query_cache[kGroupSize][QueryWords];
  __shared__ int query_base_dot[kGroupSize];
  __shared__ float scale_cache[kGroupSize];
  __shared__ float lse_cache[kGroupSize];
  __shared__ float weights[kGroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[kGroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[kGroupSize][TileGroups];
  __shared__ float exp_bias[kGroupSize][TileGroups];

  for (int idx = thread_idx; idx < kGroupSize * QueryWords; idx += blockDim.x) {
    const int hrel = idx / QueryWords;
    const int off = idx - hrel * QueryWords;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < kGroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }
  __syncthreads();

  const int hrel0 = thread_idx >> 7;
  const int hrel1 = hrel0 + 2;
  const int pair_idx = thread_idx & 127;
  const int c = pair_idx << 1;
  float acc00 = 0.0f;
  float acc01 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weights[kGroupSize] = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (ExpBits <= 2) {
      #pragma unroll
      for (int hrel = 0; hrel < kGroupSize; hrel++) {
        local_weights[hrel] = -3.402823466e38f;
      }
    }
    if (l_idx < L) {
      unsigned long long key_words[KBlocks];
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      #pragma unroll
      for (int hrel = 0; hrel < kGroupSize; hrel++) {
        int negative_dot = 0;
        #pragma unroll
        for (int kb = 0; kb < KBlocks; kb++) {
          const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
          #pragma unroll
          for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
            negative_dot += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
          }
        }
        const int dot = query_base_dot[hrel] - (negative_dot << 1);
        const float score = (float)dot * scale_cache[hrel];
        if constexpr (ExpBits <= 2) {
          local_weights[hrel] = score;
        } else {
          local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
        }
      }
    }

    #pragma unroll
    for (int hrel = 0; hrel < kGroupSize; hrel++) {
      weights[hrel][thread_idx] = local_weights[hrel];
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      if constexpr (ExpBits <= 2) {
        const float max_score = group_max;
        const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
        if (max_weight > 0.0f) {
          const float d0 = w0 - max_score;
          const float d1 = w1 - max_score;
          if constexpr (ExpBits == 1) {
            constexpr float t0 = -0.6931471805599453f;
            q0 = d0 >= t0 ? 1 : 0;
            q1 = d1 >= t0 ? 1 : 0;
          } else {
            constexpr float t0 = -1.7917594692280550f;
            constexpr float t1 = -0.6931471805599453f;
            constexpr float t2 = -0.1823215567939546f;
            q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
            q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
          }
          scale = max_weight / (float)ExpMaxCode;
        }
      } else {
        const float max_weight = group_max;
        const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
        if (max_weight > 0.0f) {
          q0 = (int)(w0 * inv_scale + 0.5f);
          q1 = (int)(w1 * inv_scale + 0.5f);
          q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
          q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
          scale = max_weight / (float)ExpMaxCode;
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < kGroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      const int global_group = tile_idx * TileGroups + local_group;
      if (global_group < LGroups) {
        const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
        const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

        const float scale20 = exp_scale2[hrel0][local_group];
        const float bias0 = exp_bias[hrel0][local_group];
        if (scale20 > 0.0f) {
          const unsigned long long *planes0 = &exp_planes[hrel0][local_group][0];
          const int pos00 = picokv_weighted_popc64<ExpBits>(value_words.x, planes0);
          const int pos01 = picokv_weighted_popc64<ExpBits>(value_words.y, planes0);
          acc00 += scale20 * (float)pos00 + bias0;
          acc01 += scale20 * (float)pos01 + bias0;
        }

        const float scale21 = exp_scale2[hrel1][local_group];
        const float bias1 = exp_bias[hrel1][local_group];
        if (scale21 > 0.0f) {
          const unsigned long long *planes1 = &exp_planes[hrel1][local_group][0];
          const int pos10 = picokv_weighted_popc64<ExpBits>(value_words.x, planes1);
          const int pos11 = picokv_weighted_popc64<ExpBits>(value_words.y, planes1);
          acc10 += scale21 * (float)pos10 + bias1;
          acc11 += scale21 * (float)pos11 + bias1;
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < kGroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  const int qh0 = qh_base + hrel0;
  const int qh1 = qh_base + hrel1;
  const long base0 = ((long)(B_idx * QH + qh0) * S + S_idx);
  const long base1 = ((long)(B_idx * QH + qh1) * S + S_idx);
  atomicAdd(&atomic_code_output[base0 * Cv + c], acc00);
  atomicAdd(&atomic_code_output[base0 * Cv + c + 1], acc01);
  atomicAdd(&atomic_code_output[base1 * Cv + c], acc10);
  atomicAdd(&atomic_code_output[base1 * Cv + c + 1], acc11);
}


// GQA=1 Ck=256/Cv=256 path.  It processes two 256-token tiles per block like
// the GQA=4/8 atomic2 paths, but only one query head is attached to each KV
// head.  This reduces tile-level query reloads and atomic writes for MHA/GQA=1.
template <bool KeyTransposedFixed, int ExpBits, int BlockTiles>
__global__ void __launch_bounds__(256, 8) picokv_v1_expbits_atomic2_gqa1_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,256]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale
) {
  constexpr int KBlocks = 4;
  constexpr int Cv = 256;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;
  const int qh = KVH_idx;

  __shared__ unsigned long long query_cache[QueryWords];
  __shared__ int query_base_dot;
  __shared__ float scale_cache;
  __shared__ float lse_cache;
  __shared__ float weights[kTileSize];
  __shared__ unsigned long long exp_planes[TileGroups][ExpBits];
  __shared__ float exp_scale2[TileGroups];
  __shared__ float exp_bias[TileGroups];

  for (int idx = thread_idx; idx < QueryWords; idx += blockDim.x) {
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[idx] = (unsigned long long)packed_projected_query[q_base + idx];
  }
  __syncthreads();

  if (thread_idx == 0) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot = base_dot;

    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache = query_scale_value * attention_scale;
    lse_cache = dense_lse[lse_idx];
  }
  __syncthreads();

  const int pair_idx = thread_idx;
  const int c = pair_idx << 1;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weight = 0.0f;
    if constexpr (ExpBits <= 2) {
      local_weight = -3.402823466e38f;
    }
    if (l_idx < L) {
      unsigned long long key_words[KBlocks];
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      int negative_dot = 0;
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        const unsigned long long signs = key_words[kb] ^ query_cache[kb * 8];
        #pragma unroll
        for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
          negative_dot += __popcll(query_cache[kb * 8 + bit_idx] & signs) << (bit_idx - 1);
        }
      }
      const int dot = query_base_dot - (negative_dot << 1);
      const float score = (float)dot * scale_cache;
      if constexpr (ExpBits <= 2) {
        local_weight = score;
      } else {
        local_weight = picokv_fast_expf(score - lse_cache);
      }
    }

    weights[thread_idx] = local_weight;
    __syncthreads();

    if (warp_idx < TileGroups) {
      const int local_group = warp_idx;
      const int local_base = local_group << 6;
      const float w0 = weights[local_base + lane_idx];
      const float w1 = weights[local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      if constexpr (ExpBits <= 2) {
        const float max_score = group_max;
        const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache) : 0.0f;
        if (max_weight > 0.0f) {
          const float d0 = w0 - max_score;
          const float d1 = w1 - max_score;
          if constexpr (ExpBits == 1) {
            constexpr float t0 = -0.6931471805599453f;
            q0 = d0 >= t0 ? 1 : 0;
            q1 = d1 >= t0 ? 1 : 0;
          } else {
            constexpr float t0 = -1.7917594692280550f;
            constexpr float t1 = -0.6931471805599453f;
            constexpr float t2 = -0.1823215567939546f;
            q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
            q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
          }
          scale = max_weight / (float)ExpMaxCode;
        }
      } else {
        const float max_weight = group_max;
        const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
        if (max_weight > 0.0f) {
          q0 = (int)(w0 * inv_scale + 0.5f);
          q1 = (int)(w1 * inv_scale + 0.5f);
          q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
          q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
          scale = max_weight / (float)ExpMaxCode;
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[local_group] = scale + scale;
        exp_bias[local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx == 0) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[local_group];
      }
    }

    if (thread_idx < 128) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        const int global_group = tile_idx * TileGroups + local_group;
        if (global_group < LGroups) {
          const float scale2 = exp_scale2[local_group];
          const float bias = exp_bias[local_group];
          if (scale2 > 0.0f) {
            const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
            const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));
            const unsigned long long *planes = &exp_planes[local_group][0];
            const int pos0 = picokv_weighted_popc64<ExpBits>(value_words.x, planes);
            const int pos1 = picokv_weighted_popc64<ExpBits>(value_words.y, planes);
            acc0 += scale2 * (float)pos0 + bias;
            acc1 += scale2 * (float)pos1 + bias;
          }
        }
      }
    }
    __syncthreads();
  }

  const long base = ((long)(B_idx * QH + qh) * S + S_idx);
  if (thread_idx == 0) {
    atomicAdd(&atomic_den[base], den_acc);
  }
  if (thread_idx < 128) {
    atomicAdd(&atomic_code_output[base * Cv + c], acc0);
    atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1);
  }
}




// GQA=8 Ck=256/Cv=256 path.  It mirrors the GQA=4 atomic2 path but processes
// two 256-token tiles per block and reuses each packed-value load across four
// query heads handled by the same thread.  This keeps the existing exact
// bitwise score/value math while reducing query reloads and atomic traffic.
template <bool KeyTransposedFixed, int ExpBits>
__global__ void __launch_bounds__(256, 6) picokv_v1_expbits_atomic2_gqa8_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,256]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale
) {
  constexpr int GroupSize = 8;
  constexpr int KBlocks = 4;
  constexpr int Cv = 256;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;

  __shared__ unsigned long long query_cache[GroupSize][QueryWords];
  __shared__ int query_base_dot[GroupSize];
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ float weights[GroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[GroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];

  for (int idx = thread_idx; idx < GroupSize * QueryWords; idx += blockDim.x) {
    const int hrel = idx / QueryWords;
    const int off = idx - hrel * QueryWords;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }
  __syncthreads();

  const int hrel_base = (thread_idx >> 7) << 2; // first half handles heads 0..3, second half handles 4..7.
  const int pair_idx = thread_idx & 127;
  const int c = pair_idx << 1;
  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;
    const int l_idx = tile_idx * kTileSize + thread_idx;

    float local_weights[GroupSize] = {0.0f};
    if constexpr (ExpBits <= 2) {
      #pragma unroll
      for (int hrel = 0; hrel < GroupSize; hrel++) {
        local_weights[hrel] = -3.402823466e38f;
      }
    }
    if (l_idx < L) {
      unsigned long long key_words[KBlocks];
      #pragma unroll
      for (int kb = 0; kb < KBlocks; kb++) {
        key_words[kb] = load_packed_key_word(
          packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
        );
      }

      #pragma unroll
      for (int hrel = 0; hrel < GroupSize; hrel++) {
        int negative_dot = 0;
        #pragma unroll
        for (int kb = 0; kb < KBlocks; kb++) {
          const unsigned long long signs = key_words[kb] ^ query_cache[hrel][kb * 8];
          #pragma unroll
          for (int bit_idx = 1; bit_idx <= kQMagBits; bit_idx++) {
            negative_dot += __popcll(query_cache[hrel][kb * 8 + bit_idx] & signs) << (bit_idx - 1);
          }
        }
        const int dot = query_base_dot[hrel] - (negative_dot << 1);
        const float score = (float)dot * scale_cache[hrel];
        if constexpr (ExpBits <= 2) {
          local_weights[hrel] = score;
        } else {
          local_weights[hrel] = picokv_fast_expf(score - lse_cache[hrel]);
        }
      }
    }

    #pragma unroll
    for (int hrel = 0; hrel < GroupSize; hrel++) {
      weights[hrel][thread_idx] = local_weights[hrel];
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      if constexpr (ExpBits <= 2) {
        const float max_score = group_max;
        const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
        if (max_weight > 0.0f) {
          const float d0 = w0 - max_score;
          const float d1 = w1 - max_score;
          if constexpr (ExpBits == 1) {
            constexpr float t0 = -0.6931471805599453f;
            q0 = d0 >= t0 ? 1 : 0;
            q1 = d1 >= t0 ? 1 : 0;
          } else {
            constexpr float t0 = -1.7917594692280550f;
            constexpr float t1 = -0.6931471805599453f;
            constexpr float t2 = -0.1823215567939546f;
            q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
            q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
          }
          scale = max_weight / (float)ExpMaxCode;
        }
      } else {
        const float max_weight = group_max;
        const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
        if (max_weight > 0.0f) {
          q0 = (int)(w0 * inv_scale + 0.5f);
          q1 = (int)(w1 * inv_scale + 0.5f);
          q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
          q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
          scale = max_weight / (float)ExpMaxCode;
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < GroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      const int global_group = tile_idx * TileGroups + local_group;
      if (global_group < LGroups) {
        const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
        const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

        #pragma unroll
        for (int slot = 0; slot < 4; slot++) {
          const int hrel = hrel_base + slot;
          const float scale2 = exp_scale2[hrel][local_group];
          const float bias = exp_bias[hrel][local_group];
          if (scale2 > 0.0f) {
            const unsigned long long *planes = &exp_planes[hrel][local_group][0];
            const int pos0 = picokv_weighted_popc64<ExpBits>(value_words.x, planes);
            const int pos1 = picokv_weighted_popc64<ExpBits>(value_words.y, planes);
            acc0[slot] += scale2 * (float)pos0 + bias;
            acc1[slot] += scale2 * (float)pos1 + bias;
          }
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  #pragma unroll
  for (int slot = 0; slot < 4; slot++) {
    const int qh = qh_base + hrel_base + slot;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_code_output[base * Cv + c], acc0[slot]);
    atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1[slot]);
  }
}


// Prototype GQA=8 Ck=256/Cv=256 path that uses BMMA m8n8k128 for score
// generation only.  The exp quantization, packed-value accumulation, and atomic
// output path intentionally match picokv_v1_expbits_atomic2_gqa8_kernel so the
// benchmark isolates whether BMMA score calculation is useful.
template <bool KeyTransposedFixed, int ExpBits>
__global__ void __launch_bounds__(256, 4) picokv_v1_expbits_atomic2_gqa8_bmma_score_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,256]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int query_scale_numel,
  const float attention_scale,
  const int *__restrict__ global_lens
) {
  constexpr int GroupSize = 8;
  constexpr int KBlocks = 4;
  constexpr int Cv = 256;
  constexpr int BlockTiles = 2;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int QueryWords = KBlocks * 8;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;
  constexpr int BmmaTokenGroups = kTileSize / 8;
  constexpr int BmmaGroupPasses = BmmaTokenGroups / QuantWarps;

  const int tile_block_idx = blockIdx.x;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = BKVH_idx / KVH;
  const int KVH_idx = BKVH_idx - B_idx * KVH;
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;

  // Varlen (ragged batch): each batch element b may have a different logical
  // number of compressed global tokens L_b <= L (the padded physical stride).
  // Skip blocks whose whole tile is beyond this request's real length, and use
  // L_b (not L) for all token-in-range checks below.
  const int L_b = (global_lens != nullptr) ? global_lens[B_idx] : L;
  if (tile_block_idx * (BlockTiles * kTileSize) >= L_b) return;

  __shared__ unsigned long long query_cache[GroupSize][QueryWords];
  __shared__ int query_base_dot[GroupSize];
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ float weights[GroupSize][kTileSize];
  __shared__ unsigned long long exp_planes[GroupSize][TileGroups][ExpBits];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];
  __shared__ __align__(128) unsigned int bmma_a_pos[2][kQMagBits][32];
  __shared__ __align__(128) unsigned int bmma_a_neg[2][kQMagBits][32];
  __shared__ int query_neg_count[GroupSize][2][kQMagBits];
  __shared__ __align__(128) unsigned int bmma_b[QuantWarps][32];

  for (int idx = thread_idx; idx < GroupSize * QueryWords; idx += blockDim.x) {
    const int hrel = idx / QueryWords;
    const int off = idx - hrel * QueryWords;
    const int qh = qh_base + hrel;
    const long q_base = (((long)(B_idx * QH + qh) * S + S_idx) * KBlocks) * 8;
    query_cache[hrel][off] = (unsigned long long)packed_projected_query[q_base + off];
  }
  __syncthreads();

  if (thread_idx < GroupSize) {
    int base_dot = 0;
    #pragma unroll
    for (int local = 0; local < KBlocks * kQMagBits; local++) {
      const int kb = local / kQMagBits;
      const int bit_idx = local - kb * kQMagBits + 1;
      base_dot += __popcll(query_cache[thread_idx][kb * 8 + bit_idx]) << (bit_idx - 1);
    }
    query_base_dot[thread_idx] = base_dot;

    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }

  for (int idx = thread_idx; idx < 2 * kQMagBits * GroupSize * 4; idx += blockDim.x) {
    const int seg = idx & 3;
    const int hrel = (idx >> 2) & 7;
    const int bit = (idx >> 5) % kQMagBits;
    const int chunk = idx / (kQMagBits * GroupSize * 4);
    const int kb = chunk * 2 + (seg >> 1);
    const unsigned long long sign_word = query_cache[hrel][kb * 8];
    const unsigned long long mag_word = query_cache[hrel][kb * 8 + bit + 1];
    const unsigned int sign_piece = (unsigned int)((seg & 1) ? (sign_word >> 32) : sign_word);
    const unsigned int mag_piece = (unsigned int)((seg & 1) ? (mag_word >> 32) : mag_word);
    bmma_a_pos[chunk][bit][hrel * 4 + seg] = mag_piece & ~sign_piece;
    bmma_a_neg[chunk][bit][hrel * 4 + seg] = mag_piece & sign_piece;
  }
  for (int idx = thread_idx; idx < GroupSize * 2 * kQMagBits; idx += blockDim.x) {
    const int bit = idx % kQMagBits;
    const int chunk = (idx / kQMagBits) & 1;
    const int hrel = idx / (2 * kQMagBits);
    const int kb0 = chunk * 2;
    const int kb1 = kb0 + 1;
    const unsigned long long neg0 = query_cache[hrel][kb0 * 8 + bit + 1] & query_cache[hrel][kb0 * 8];
    const unsigned long long neg1 = query_cache[hrel][kb1 * 8 + bit + 1] & query_cache[hrel][kb1 * 8];
    query_neg_count[hrel][chunk][bit] = __popcll(neg0) + __popcll(neg1);
  }
  __syncthreads();

  const int hrel_base = (thread_idx >> 7) << 2;
  const int pair_idx = thread_idx & 127;
  const int c = pair_idx << 1;
  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = tile_block_idx * BlockTiles + block_tile;

    #pragma unroll
    for (int group_pass = 0; group_pass < BmmaGroupPasses; group_pass++) {
      const int token_group = group_pass * QuantWarps + warp_idx;
      const int token_base = tile_idx * kTileSize + token_group * 8;
      // m8n8 accumulator fragment (PTX mma layout): lane L holds
      // C[L/4][2*(L%4)] in x[0] and C[L/4][2*(L%4)+1] in x[1] — one head row,
      // two adjacent token columns per lane. Verified end-to-end against the
      // popc kernel by check_bmma_bitwise.py.
      const int h = lane_idx >> 2;
      const int t0 = (lane_idx & 3) << 1;
      int dot0 = query_base_dot[h];
      int dot1 = dot0;

      #pragma unroll
      for (int chunk = 0; chunk < 2; chunk++) {
        const int b_idx = lane_idx;
        const int b_col = b_idx >> 2;
        const int b_seg = b_idx & 3;
        const int l_idx = token_base + b_col;
        unsigned int key_piece = 0;
        if (l_idx < L_b) {
          const int kb = chunk * 2 + (b_seg >> 1);
          const unsigned long long key_word = load_packed_key_word(
            packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
          );
          key_piece = (unsigned int)((b_seg & 1) ? (key_word >> 32) : key_word);
        }
        bmma_b[warp_idx][b_idx] = key_piece;
        __syncwarp();

        // B (key sign pieces) is bit-independent: load the fragment once per
        // chunk and reuse it across all kQMagBits A planes.
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::col_major> b_frag;
        nvcuda::wmma::load_matrix_sync(b_frag, &bmma_b[warp_idx][0], 128);

        #pragma unroll
        for (int bit = 0; bit < kQMagBits; bit++) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::row_major> a_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 128, int> pos_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 128, int> neg_frag;

          nvcuda::wmma::fill_fragment(pos_frag, 0);
          nvcuda::wmma::load_matrix_sync(a_frag, &bmma_a_pos[chunk][bit][0], 128);
          nvcuda::wmma::bmma_sync(
            pos_frag, a_frag, b_frag, pos_frag,
            nvcuda::wmma::experimental::bmmaBitOpAND,
            nvcuda::wmma::experimental::bmmaAccumulateOpPOPC
          );

          nvcuda::wmma::fill_fragment(neg_frag, 0);
          nvcuda::wmma::load_matrix_sync(a_frag, &bmma_a_neg[chunk][bit][0], 128);
          nvcuda::wmma::bmma_sync(
            neg_frag, a_frag, b_frag, neg_frag,
            nvcuda::wmma::experimental::bmmaBitOpAND,
            nvcuda::wmma::experimental::bmmaAccumulateOpPOPC
          );

          // Consume both accumulators in-register — no store_matrix_sync, no
          // shared round-trip, no warp barrier inside the bit loop:
          // mismatch_b = popc(a_pos & k) + popc(a_neg) - popc(a_neg & k).
          const int qneg = query_neg_count[h][chunk][bit];
          dot0 -= (pos_frag.x[0] + qneg - neg_frag.x[0]) << (bit + 1);
          dot1 -= (pos_frag.x[1] + qneg - neg_frag.x[1]) << (bit + 1);
        }
      }

      const int local0 = token_group * 8 + t0;
      const int local1 = local0 + 1;
      const int l0 = token_base + t0;
      const int l1 = l0 + 1;
      float out0 = 0.0f;
      float out1 = 0.0f;
      if constexpr (ExpBits <= 2) {
        out0 = -3.402823466e38f;
        out1 = -3.402823466e38f;
      }
      if (l0 < L_b) {
        const float score0 = (float)dot0 * scale_cache[h];
        if constexpr (ExpBits <= 2) {
          out0 = score0;
        } else {
          out0 = picokv_fast_expf(score0 - lse_cache[h]);
        }
      }
      if (l1 < L_b) {
        const float score1 = (float)dot1 * scale_cache[h];
        if constexpr (ExpBits <= 2) {
          out1 = score1;
        } else {
          out1 = picokv_fast_expf(score1 - lse_cache[h]);
        }
      }
      weights[h][local0] = out0;
      weights[h][local1] = out1;
    }
    __syncthreads();

    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      float group_max = fmaxf(w0, w1);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        group_max = fmaxf(group_max, __shfl_down_sync(0xffffffff, group_max, offset));
      }
      group_max = __shfl_sync(0xffffffff, group_max, 0);

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      if constexpr (ExpBits <= 2) {
        const float max_score = group_max;
        const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
        if (max_weight > 0.0f) {
          const float d0 = w0 - max_score;
          const float d1 = w1 - max_score;
          if constexpr (ExpBits == 1) {
            constexpr float t0 = -0.6931471805599453f;
            q0 = d0 >= t0 ? 1 : 0;
            q1 = d1 >= t0 ? 1 : 0;
          } else {
            constexpr float t0 = -1.7917594692280550f;
            constexpr float t1 = -0.6931471805599453f;
            constexpr float t2 = -0.1823215567939546f;
            q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
            q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
          }
          scale = max_weight / (float)ExpMaxCode;
        }
      } else {
        const float max_weight = group_max;
        const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
        if (max_weight > 0.0f) {
          q0 = (int)(w0 * inv_scale + 0.5f);
          q1 = (int)(w1 * inv_scale + 0.5f);
          q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
          q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
          scale = max_weight / (float)ExpMaxCode;
        }
      }

      int qsum = q0 + q1;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
      }

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx] = ((unsigned long long)high << 32) | (unsigned long long)low;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
      }
    }
    __syncthreads();

    if (thread_idx < GroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }

    #pragma unroll
    for (int local_group = 0; local_group < TileGroups; local_group++) {
      const int global_group = tile_idx * TileGroups + local_group;
      if (global_group < LGroups) {
        const long value_idx0 = (((long)(B_idx * KVH + KVH_idx) * LGroups + global_group) * Cv + c);
        const ulonglong2 value_words = *((const ulonglong2*)(&packed_value_sign[value_idx0]));

        #pragma unroll
        for (int slot = 0; slot < 4; slot++) {
          const int hrel = hrel_base + slot;
          const float scale2 = exp_scale2[hrel][local_group];
          const float bias = exp_bias[hrel][local_group];
          if (scale2 > 0.0f) {
            const unsigned long long *planes = &exp_planes[hrel][local_group][0];
            const int pos0 = picokv_weighted_popc64<ExpBits>(value_words.x, planes);
            const int pos1 = picokv_weighted_popc64<ExpBits>(value_words.y, planes);
            acc0[slot] += scale2 * (float)pos0 + bias;
            acc1[slot] += scale2 * (float)pos1 + bias;
          }
        }
      }
    }
    __syncthreads();
  }

  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  #pragma unroll
  for (int slot = 0; slot < 4; slot++) {
    const int qh = qh_base + hrel_base + slot;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_code_output[base * Cv + c], acc0[slot]);
    atomicAdd(&atomic_code_output[base * Cv + c + 1], acc1[slot]);
  }
}


template <typename scalar_t>
__global__ void picokv_v1_fused_reduce_output_projection_kernel(
  const __nv_bfloat16 *__restrict__ partial_code_output, // [B,QH,S,num_tiles,Cv]
  const float *__restrict__ partial_den,                 // [B,QH,S,num_tiles]
  const scalar_t *__restrict__ dense_output,             // [B,QH,S,D]
  const scalar_t *__restrict__ value_proj,               // [B,KVH,Cv,D]
  scalar_t *__restrict__ attn_output,                    // [B,QH,S,D]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int num_tiles,
  const int Cv,
  const int D
) {
  __shared__ float code_acc[512];
  __shared__ float den_acc;

  const int S_idx = blockIdx.x;
  const int BH_idx = blockIdx.y;
  const int B_idx = BH_idx / QH;
  const int qh = BH_idx - B_idx * QH;
  const int thread_idx = threadIdx.x;
  const int group_size = QH / KVH;
  const int KVH_idx = qh / group_size;
  const long row_base = (long)(B_idx * QH + qh) * S + S_idx;

  if (thread_idx < Cv) {
    __nv_bfloat16 acc = __float2bfloat16(0.0f);
    #pragma unroll 1
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      const long partial_idx = (row_base * num_tiles + tile_idx) * Cv + thread_idx;
      acc = __hadd(acc, partial_code_output[partial_idx]);
    }
    code_acc[thread_idx] = __bfloat162float(acc);
  }

  if (thread_idx == 0) {
    float den = 0.0f;
    #pragma unroll 1
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      den += partial_den[row_base * num_tiles + tile_idx];
    }
    den_acc = den;
  }
  __syncthreads();

  const long dense_base = row_base * D;
  const long proj_base = ((long)(B_idx * KVH + KVH_idx) * Cv) * D;
  const float inv_den = 1.0f / (1.0f + den_acc);
  for (int d = thread_idx; d < D; d += blockDim.x) {
    float glob_acc = 0.0f;
    #pragma unroll 4
    for (int c = 0; c < Cv; c++) {
      glob_acc += code_acc[c] * picokv_to_float(value_proj[proj_base + (long)c * D + d]);
    }
    const float numerator = picokv_to_float(dense_output[dense_base + d]) + glob_acc;
    attn_output[dense_base + d] = picokv_from_float<scalar_t>(numerator * inv_den);
  }
}


// ---------------------------------------------------------------------------
// Dense-window decode attention over the per-layer ring pools (split-chunk).
//
// The flashinfer paged-decode call this replaces costs ~10-15 us/layer under
// CUDA-graph replay (split-kv kernel + merge kernel): its scheduler is sized
// for full-KV attention while the decode window is only 320-384 rows. This
// pair of kernels does the same math — softmax attention over rows [0,
// lpl[b]) of kbuf/vbuf [B, cap, KVH, D], GQA-4 block-major (q head h reads
// kv head h/4) — with the window CHUNKED over blockIdx.z so small batches
// still fill the GPU, and the 4 group heads share every K/V line.
//
// main kernel, one block per (batch, kv head, 96-row chunk), 256 threads:
//   1. logits: warp w covers chunk-local rows r0 = w, w+8, ...; lane owns
//      dims [lane*4, lane*4+4) of all 4 heads (5-step shuffle dot per head)
//   2. softmax stats per head over the chunk -> (m, l) to global scratch
//   3. probs in place: s = exp(s - m)   (unnormalized; merge rescales)
//   4. value: thread = (g, row-half, lane) owns dims [lane*4, +4) and rows
//      r0 = half, half+2, ... -> unnormalized acc to scratch
// merge kernel, one block per (batch, kv head), 256 threads = (head, dim
// pair): standard log-sum-exp combine of the chunk partials -> bf16 out +
// ln-LSE. Empty chunks (row0 >= lpl[b]) exit early; the merge reads only
// ceil(lpl[b]/CH) partials, so stale scratch is never touched.
//
// Deterministic: every reduction runs in a fixed order. lpl is read from
// device memory at run time, so a captured launch stays correct while the
// window grows (no re-plan, no graph re-capture).
// ---------------------------------------------------------------------------
constexpr int kWinChunkRows = 96;  // default rows per split-chunk block

// CH = rows per split-chunk block (template: smem logits [GROUP][CH] and the
// serial per-thread row chains scale with it). Small batches pick a smaller
// CH so more blocks share the window (the launcher's selector).
template <int GROUP, int D, int CH>
__global__ void __launch_bounds__(256, 4) picokv_window_attn_kernel(
    const __nv_bfloat16 *__restrict__ kbuf,  // [B, cap, KVH, D]
    const __nv_bfloat16 *__restrict__ vbuf,  // [B, cap, KVH, D]
    const int *__restrict__ lpl,             // [B] live rows per request
    const __nv_bfloat16 *__restrict__ q,     // [B, QH, D], QH = KVH*GROUP
    float *__restrict__ acc,                 // [B, KVH, CHUNKS, GROUP, D]
    float *__restrict__ ml,                  // [B, KVH, CHUNKS, GROUP, 2]
    float sm_scale,
    int cap,
    int KVH,
    int CHUNKS)
{
  constexpr int THREADS = 256;
  constexpr int WARPS = THREADS / 32;
  extern __shared__ float smem[];
  float *s = smem;                                 // [GROUP][CH] logits/probs
  float *wmax_s = smem + GROUP * CH;               // [WARPS][GROUP]
  float *wsum_s = wmax_s + WARPS * GROUP;          // [WARPS][GROUP]
  float *m_s = wsum_s + WARPS * GROUP;             // [GROUP]
  float *partial_s = m_s + GROUP;                  // [2][GROUP][D]

  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int chunk = blockIdx.z;
  const int row0 = chunk * CH;
  const int L = lpl[b];
  if (row0 >= L) return;            // empty chunk — uniform per block
  const int R = min(CH, L - row0);  // live rows in this chunk
  const int h0 = kvh * GROUP;       // first q head of this block
  const int QH = KVH * GROUP;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // ---- phase 0: pull this chunk's K/V lines into L2. The K pass (phase 1)
  // and V pass (phase 4) are barrier-separated; without this each pays its
  // own DRAM latency. The whole batch's window (tens of MB) fits in L2.
  {
    const __nv_bfloat16 *kp = kbuf + (((long)b * cap + row0) * KVH + kvh) * D;
    const __nv_bfloat16 *vp = vbuf + (((long)b * cap + row0) * KVH + kvh) * D;
    for (int i = tid; i < R * 4; i += THREADS) {  // R rows x {K,V} x 2 lines
      const int r = i >> 2;
      const int piece = i & 3;
      const char *p = (const char *)((piece < 2 ? kp : vp))
          + (long)r * KVH * D * 2 + (piece & 1) * 128;
      asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
    }
  }

  // ---- phase 1: logits. Lane owns dims [lane*4, lane*4+4) of every head;
  // warp w covers chunk-local rows r0 = w, w+8, ... One 8B K load per lane
  // per row, 4 rows in flight.
  float qg[GROUP][4];
  {
    const __nv_bfloat16 *qb = q + ((long)b * QH + h0) * D;
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      const uint2 raw = *reinterpret_cast<const uint2 *>(qb + g * D + lane * 4);
      const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.x);
      const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.y);
      qg[g][0] = __bfloat162float(p0.x);
      qg[g][1] = __bfloat162float(p0.y);
      qg[g][2] = __bfloat162float(p1.x);
      qg[g][3] = __bfloat162float(p1.y);
    }
  }
  {
    const __nv_bfloat16 *kb = kbuf + (((long)b * cap + row0) * KVH + kvh) * D + lane * 4;
    #pragma unroll 4
    for (int r0 = warp; r0 < R; r0 += WARPS) {
      const uint2 raw = *reinterpret_cast<const uint2 *>(kb + (long)r0 * KVH * D);
      const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.x);
      const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.y);
      float kf[4] = {__bfloat162float(p0.x), __bfloat162float(p0.y),
                     __bfloat162float(p1.x), __bfloat162float(p1.y)};
      float pd[GROUP];
      #pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        float accv = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          accv = fmaf(qg[g][j], kf[j], accv);
        pd[g] = accv;
      }
      #pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          pd[g] += __shfl_xor_sync(0xffffffffu, pd[g], off);
      }
      if (lane == 0) {
        #pragma unroll
        for (int g = 0; g < GROUP; ++g)
          s[g * CH + r0] = pd[g] * sm_scale;
      }
    }
  }
  __syncthreads();

  // ---- phase 2: per-head online max+sum-exp over the chunk (one fused
  // pass; warp-uniform values, no shuffle reduce) -> (m, l) to scratch.
  {
    float wm[GROUP], wl[GROUP];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      wm[g] = -FLT_MAX;
      wl[g] = 0.f;
    }
    for (int r0 = warp; r0 < R; r0 += WARPS) {
      #pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        const float sv = s[g * CH + r0];
        const float nm = fmaxf(wm[g], sv);
        wl[g] = wl[g] * picokv_fast_expf(wm[g] - nm) + picokv_fast_expf(sv - nm);
        wm[g] = nm;
      }
    }
    if (lane == 0) {
      #pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        wmax_s[warp * GROUP + g] = wm[g];
        wsum_s[warp * GROUP + g] = wl[g];
      }
    }
  }
  __syncthreads();
  if (tid < GROUP) {
    float m = -FLT_MAX;
    #pragma unroll
    for (int w = 0; w < WARPS; ++w)
      m = fmaxf(m, wmax_s[w * GROUP + tid]);
    float l = 0.f;
    #pragma unroll
    for (int w = 0; w < WARPS; ++w)
      l += wsum_s[w * GROUP + tid] * picokv_fast_expf(wmax_s[w * GROUP + tid] - m);
    m_s[tid] = m;
    float2 v;
    v.x = m;
    v.y = l;
    *reinterpret_cast<float2 *>(ml + (((long)b * KVH + kvh) * CHUNKS + chunk) * GROUP * 2 + tid * 2) = v;
  }
  __syncthreads();

  // ---- phase 3: unnormalized probabilities in place (merge rescales).
  {
    float *sg = s;
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      const float m = m_s[g];
      for (int r0 = tid; r0 < R; r0 += THREADS)
        sg[g * CH + r0] = picokv_fast_expf(sg[g * CH + r0] - m);
    }
  }
  __syncthreads();

  // ---- phase 4: V accumulation. Thread = (g, row-half, lane): lane owns
  // dims [lane*4, lane*4+4) via one 8B load, row-half picks rows
  // r0 = half, half+2, ... so each thread's chain is R/2 long and a warp
  // covers 256B of every row it touches.
  {
    const int g = tid >> 6;
    const int rowhalf = (tid >> 5) & 1;
    const __nv_bfloat16 *vb = vbuf + (((long)b * cap + row0) * KVH + kvh) * D + lane * 4;
    float ac[4] = {0.f, 0.f, 0.f, 0.f};
    const float *pg = s + g * CH;
    #pragma unroll 4
    for (int r0 = rowhalf; r0 < R; r0 += 2) {
      const uint2 raw = *reinterpret_cast<const uint2 *>(vb + (long)r0 * KVH * D);
      const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.x);
      const __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162 *>(&raw.y);
      const float p = pg[r0];
      ac[0] = fmaf(p, __bfloat162float(v0.x), ac[0]);
      ac[1] = fmaf(p, __bfloat162float(v0.y), ac[1]);
      ac[2] = fmaf(p, __bfloat162float(v1.x), ac[2]);
      ac[3] = fmaf(p, __bfloat162float(v1.y), ac[3]);
    }
    float *ps = partial_s + (rowhalf * GROUP + g) * D + lane * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) ps[j] = ac[j];
  }
  __syncthreads();
  if (tid < GROUP * (D / 2)) {  // fold the 2 row halves -> scratch (fixed order)
    const int g = tid >> 6;
    const int dpair = tid & 63;
    const float *p0s = partial_s + (0 * GROUP + g) * D + dpair * 2;
    const float *p1s = partial_s + (1 * GROUP + g) * D + dpair * 2;
    float2 v;
    v.x = p0s[0] + p1s[0];
    v.y = p0s[1] + p1s[1];
    *reinterpret_cast<float2 *>(acc + ((((long)b * KVH + kvh) * CHUNKS + chunk) * GROUP + g) * D + dpair * 2) = v;
  }
}


// log-sum-exp merge of the chunk partials: one block per (batch, kv head),
// thread = (head, dim pair). Reads only the ceil(lpl[b]/CH) live chunks.
template <int CH>
__global__ void __launch_bounds__(256) picokv_window_attn_merge_kernel(
    const int *__restrict__ lpl,             // [B]
    const float *__restrict__ acc,           // [B, KVH, CHUNKS, GROUP, D]
    const float *__restrict__ ml,            // [B, KVH, CHUNKS, GROUP, 2]
    __nv_bfloat16 *__restrict__ out,         // [B, QH, D]
    float *__restrict__ lse,                 // [B, QH] ln(sum exp(scale*q.k))
    int KVH,
    int CHUNKS)
{
  constexpr int GROUP = 4;
  constexpr int D = 128;
  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int QH = KVH * GROUP;
  const int tid = threadIdx.x;
  const int g = tid >> 6;
  const int dpair = tid & 63;
  const int h = kvh * GROUP + g;
  const int L = lpl[b];
  const int nch = min(CHUNKS, (L + CH - 1) / CH);
  const long cbase = ((long)b * KVH + kvh) * CHUNKS;

  float mstar = -FLT_MAX;
  for (int c = 0; c < nch; ++c)
    mstar = fmaxf(mstar, ml[(cbase + c) * GROUP * 2 + g * 2]);

  float ltot = 0.f, a0 = 0.f, a1 = 0.f;
  for (int c = 0; c < nch; ++c) {
    const float2 mlg = *reinterpret_cast<const float2 *>(ml + (cbase + c) * GROUP * 2 + g * 2);
    const float w = picokv_fast_expf(mlg.x - mstar);
    ltot = fmaf(w, mlg.y, ltot);
    const float2 ac = *reinterpret_cast<const float2 *>(acc + ((cbase + c) * GROUP + g) * D + dpair * 2);
    a0 = fmaf(w, ac.x, a0);
    a1 = fmaf(w, ac.y, a1);
  }
  const float inv_l = 1.f / ltot;
  lse[(long)b * QH + h] = mstar + logf(ltot);
  *reinterpret_cast<__nv_bfloat162 *>(out + ((long)b * QH + h) * D + dpair * 2) =
      __floats2bfloat162_rn(a0 * inv_l, a1 * inv_l);
}


} // namespace

void picokv_dense_attention_numerator(
  at::Tensor query,
  at::Tensor dense_key,
  at::Tensor dense_value,
  at::Tensor dense_output,
  at::Tensor dense_lse,
  double attention_scale
) {
  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  TORCH_CHECK(query.dim() == 4, "query must have shape [B,QH,S,D]");
  TORCH_CHECK(dense_key.dim() == 4, "dense_key must have shape [B,KVH,L,D]");
  TORCH_CHECK(dense_value.dim() == 4, "dense_value must have shape [B,KVH,L,D]");
  TORCH_CHECK(query.is_cuda() && dense_key.is_cuda() && dense_value.is_cuda(), "dense attention inputs must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(dense_key.is_contiguous(), "dense_key must be contiguous");
  TORCH_CHECK(dense_value.is_contiguous(), "dense_value must be contiguous");
  TORCH_CHECK(dense_output.is_contiguous(), "dense_output must be contiguous");
  TORCH_CHECK(dense_lse.is_contiguous(), "dense_lse must be contiguous");
  TORCH_CHECK(query.scalar_type() == dense_key.scalar_type(), "query and dense_key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == dense_value.scalar_type(), "query and dense_value dtype mismatch");
  TORCH_CHECK(query.scalar_type() == dense_output.scalar_type(), "dense_output dtype must match query dtype");
  TORCH_CHECK(dense_lse.scalar_type() == at::kFloat, "dense_lse must be float32");

  const int B = query.size(0);
  const int QH = query.size(1);
  const int S = query.size(2);
  const int D = query.size(3);
  const int KVH = dense_key.size(1);
  const int L = dense_key.size(2);
  TORCH_CHECK(dense_key.size(0) == B && dense_value.size(0) == B, "dense KV B mismatch");
  TORCH_CHECK(dense_value.size(1) == KVH, "dense KV head mismatch");
  TORCH_CHECK(dense_value.size(2) == L, "dense KV length mismatch");
  TORCH_CHECK(dense_key.size(3) == D && dense_value.size(3) == D, "dense KV head_dim mismatch");
  TORCH_CHECK(QH % KVH == 0, "QH must be divisible by KVH for GQA dense attention");
  TORCH_CHECK(L > 0, "dense attention requires at least one retained KV token");
  TORCH_CHECK(D <= 256, "dense attention currently supports head_dim <= 256");
  TORCH_CHECK(L <= 12288, "dense attention retained length is too large for the shared-score kernel");
  TORCH_CHECK(dense_output.sizes() == query.sizes(), "dense_output shape must match query");
  TORCH_CHECK(dense_lse.size(0) == B && dense_lse.size(1) == QH && dense_lse.size(2) == S, "dense_lse shape must be [B,QH,S]");

  const int threads = 256;
  const int rows = B * QH * S;
  const size_t shared_bytes = (size_t)L * sizeof(float);
  if (query.scalar_type() == at::kBFloat16) {
    picokv_dense_attention_numerator_kernel<__nv_bfloat16><<<rows, threads, shared_bytes, stream>>>(
      (const __nv_bfloat16*)query.data_ptr<at::BFloat16>(),
      (const __nv_bfloat16*)dense_key.data_ptr<at::BFloat16>(),
      (const __nv_bfloat16*)dense_value.data_ptr<at::BFloat16>(),
      (__nv_bfloat16*)dense_output.data_ptr<at::BFloat16>(),
      dense_lse.data_ptr<float>(),
      B, QH, KVH, S, L, D,
      (float)attention_scale
    );
  } else if (query.scalar_type() == at::kFloat) {
    picokv_dense_attention_numerator_kernel<float><<<rows, threads, shared_bytes, stream>>>(
      query.data_ptr<float>(),
      dense_key.data_ptr<float>(),
      dense_value.data_ptr<float>(),
      dense_output.data_ptr<float>(),
      dense_lse.data_ptr<float>(),
      B, QH, KVH, S, L, D,
      (float)attention_scale
    );
  } else {
    TORCH_CHECK(false, "dense attention currently supports torch.bfloat16 and torch.float32");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void picokv_fused_output_projection(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output
) {
  at::cuda::CUDAGuard device_guard(dense_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  TORCH_CHECK(dense_output.dim() == 4, "dense_output must have shape [B,QH,S,D]");
  TORCH_CHECK(glob_code_output.dim() == 4, "glob_code_output must have shape [B,QH,S,Cv]");
  TORCH_CHECK(glob_den.dim() == 3, "glob_den must have shape [B,QH,S]");
  TORCH_CHECK(value_proj.dim() == 4, "value_proj must have shape [B,KVH,Cv,D]");
  TORCH_CHECK(dense_output.is_cuda() && glob_code_output.is_cuda() && glob_den.is_cuda() && value_proj.is_cuda(), "fused output inputs must be CUDA tensors");
  TORCH_CHECK(dense_output.is_contiguous(), "dense_output must be contiguous");
  TORCH_CHECK(glob_code_output.is_contiguous(), "glob_code_output must be contiguous");
  TORCH_CHECK(glob_den.is_contiguous(), "glob_den must be contiguous");
  TORCH_CHECK(value_proj.is_contiguous(), "value_proj must be contiguous");
  TORCH_CHECK(attn_output.is_contiguous(), "attn_output must be contiguous");
  TORCH_CHECK(glob_code_output.scalar_type() == at::kFloat, "glob_code_output must be float32");
  TORCH_CHECK(glob_den.scalar_type() == at::kFloat, "glob_den must be float32");
  TORCH_CHECK(dense_output.scalar_type() == value_proj.scalar_type(), "dense_output and value_proj dtype mismatch");
  TORCH_CHECK(dense_output.scalar_type() == attn_output.scalar_type(), "attn_output dtype must match dense_output");

  const int B = dense_output.size(0);
  const int QH = dense_output.size(1);
  const int S = dense_output.size(2);
  const int D = dense_output.size(3);
  const int KVH = value_proj.size(1);
  const int Cv = glob_code_output.size(3);
  TORCH_CHECK(glob_code_output.size(0) == B && glob_code_output.size(1) == QH && glob_code_output.size(2) == S, "glob_code_output shape mismatch");
  TORCH_CHECK(glob_den.size(0) == B && glob_den.size(1) == QH && glob_den.size(2) == S, "glob_den shape mismatch");
  TORCH_CHECK(value_proj.size(0) == B && value_proj.size(2) == Cv && value_proj.size(3) == D, "value_proj shape mismatch");
  TORCH_CHECK(QH % KVH == 0, "QH must be divisible by KVH for GQA output projection");
  TORCH_CHECK(Cv <= 512, "fused output projection supports Cv <= 512");
  TORCH_CHECK(D <= 256, "fused output projection supports head_dim <= 256");
  TORCH_CHECK(attn_output.sizes() == dense_output.sizes(), "attn_output shape must match dense_output");

  const int rows = B * QH * S;
  const int threads = 128;
  if (dense_output.scalar_type() == at::kBFloat16) {
    picokv_fused_output_projection_kernel<__nv_bfloat16><<<rows, threads, 0, stream>>>(
      (const __nv_bfloat16*)dense_output.data_ptr<at::BFloat16>(),
      glob_code_output.data_ptr<float>(),
      glob_den.data_ptr<float>(),
      (const __nv_bfloat16*)value_proj.data_ptr<at::BFloat16>(),
      (__nv_bfloat16*)attn_output.data_ptr<at::BFloat16>(),
      B, QH, KVH, S, Cv, D
    );
  } else if (dense_output.scalar_type() == at::kFloat) {
    picokv_fused_output_projection_kernel<float><<<rows, threads, 0, stream>>>(
      dense_output.data_ptr<float>(),
      glob_code_output.data_ptr<float>(),
      glob_den.data_ptr<float>(),
      value_proj.data_ptr<float>(),
      attn_output.data_ptr<float>(),
      B, QH, KVH, S, Cv, D
    );
  } else {
    TORCH_CHECK(false, "fused output projection currently supports torch.bfloat16 and torch.float32");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// GEMM value projection: replaces the scalar per-row dot-product kernel with
// one cublas strided-batched GEMM (bf16 tensor cores, fp32 accumulate/output)
// plus a small epilogue. Same math as the scalar kernel up to fp32 accumulation
// order: codes are rounded to bf16 exactly as picokv_code_for_projection does,
// products accumulate in fp32, and the (dense + acc) * inv_den epilogue is
// identical. Selected by default for bf16; PICOKV_PROJ_GEMM=0 forces the
// scalar kernel (A/B).
void picokv_fused_output_projection_gemm(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output
) {
  at::cuda::CUDAGuard device_guard(dense_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int B = dense_output.size(0);
  const int QH = dense_output.size(1);
  const int S = dense_output.size(2);
  const int D = dense_output.size(3);
  const int KVH = value_proj.size(1);
  const int Cv = glob_code_output.size(3);
  const int G = QH / KVH;
  TORCH_CHECK(G > 0 && QH == G * KVH, "QH must be divisible by KVH");
  TORCH_CHECK(dense_output.scalar_type() == at::kBFloat16,
              "GEMM projection currently supports torch.bfloat16");

  // Persistent workspaces (stable addresses across calls; CUDA-graph safe):
  // bf16 copy of the fp32 code accumulator and the fp32 GEMM output.
  static std::mutex ws_mutex;
  static std::unordered_map<int, std::pair<at::Tensor, at::Tensor>> ws;
  at::Tensor code_bf16, acc32;
  {
    std::lock_guard<std::mutex> lock(ws_mutex);
    auto it = ws.find(dense_output.get_device());
    if (it == ws.end()) {
      auto opts_bf16 = dense_output.options();
      auto opts_f32 = dense_output.options().dtype(at::kFloat);
      code_bf16 = at::empty({B, QH, S, Cv}, opts_bf16);
      acc32 = at::empty({B, QH, S, D}, opts_f32);
      ws.emplace(dense_output.get_device(), std::make_pair(code_bf16, acc32));
    } else {
      std::tie(code_bf16, acc32) = it->second;
      if (code_bf16.numel() < B * QH * S * Cv || acc32.numel() < B * QH * S * D) {
        code_bf16 = at::empty({B, QH, S, Cv}, code_bf16.options());
        acc32 = at::empty({B, QH, S, D}, acc32.options());
        it->second = std::make_pair(code_bf16, acc32);
      }
    }
  }

  // 1. Round codes to bf16 (the scalar kernel's picokv_code_for_projection).
  {
    const long numel = (long)B * QH * S * Cv;
    const int threads = 256;
    const int blocks = (int)((numel + threads - 1) / threads);
    picokv_cast_code_bf16_kernel<<<blocks, threads, 0, stream>>>(
      glob_code_output.data_ptr<float>(),
      (__nv_bfloat16 *)code_bf16.data_ptr<at::BFloat16>(),
      numel
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 2. Batched GEMM over (b, kvh): out rows are the G*S queries of kvh's group.
  // Row-major out [G*S, D] equals col-major [D, G*S]; vp row-major [Cv, D]
  // equals col-major [D, Cv]; code row-major [G*S, Cv] equals col-major
  // [Cv, G*S] -- so a plain N,N GEMM computes out = code @ vp in place.
  {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(handle != nullptr, "cublas handle unavailable");
    cublasSetStream(handle, stream);
    const float one = 1.0f;
    const float zero = 0.0f;
    const int batch = B * KVH;
    cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      /*m=*/D, /*n=*/G * S, /*k=*/Cv,
      &one,
      value_proj.data_ptr(), CUDA_R_16BF, /*lda=*/D, /*strideA=*/(long long)Cv * D,
      code_bf16.data_ptr(), CUDA_R_16BF, /*ldb=*/Cv, /*strideB=*/(long long)G * S * Cv,
      &zero,
      acc32.data_ptr(), CUDA_R_32F, /*ldc=*/D, /*strideC=*/(long long)G * S * D,
      batch,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cublasGemmStridedBatchedEx failed: ", (int)status);
  }

  // 3. Epilogue: out = (dense + acc32) * (1 / (1 + den)).
  {
    const long total = (long)B * QH * S * D;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    picokv_proj_gemm_epilogue_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
      (const __nv_bfloat16 *)dense_output.data_ptr<at::BFloat16>(),
      acc32.data_ptr<float>(),
      glob_den.data_ptr<float>(),
      (__nv_bfloat16 *)attn_output.data_ptr<at::BFloat16>(),
      (int)total,
      D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}




void picokv_project_quantize_pack_query(
  at::Tensor query,
  at::Tensor key_proj,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
) {
  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  TORCH_CHECK(query.dim() == 4, "query must have shape [B,QH,S,D]");
  TORCH_CHECK(key_proj.dim() == 4, "key_proj must have shape [B,KVH,C,D]");
  TORCH_CHECK(query.is_cuda() && key_proj.is_cuda(), "query and key_proj must be CUDA tensors");
  TORCH_CHECK(packed_projected_query.is_cuda() && query_scale.is_cuda(), "outputs must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key_proj.is_contiguous(), "key_proj must be contiguous");
  TORCH_CHECK(packed_projected_query.is_contiguous(), "packed_projected_query must be contiguous");
  TORCH_CHECK(query_scale.is_contiguous(), "query_scale must be contiguous");
  TORCH_CHECK(query.scalar_type() == key_proj.scalar_type(), "query and key_proj dtype mismatch");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat,
              "query/key_proj must be torch.bfloat16 or torch.float32");
  TORCH_CHECK(packed_projected_query.scalar_type() == at::kLong, "packed_projected_query must be torch.long");
  TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be torch.float32");

  const int B = query.size(0);
  const int QH = query.size(1);
  const int S = query.size(2);
  const int D = query.size(3);
  const int KVH = key_proj.size(1);
  const int C = key_proj.size(2);
  TORCH_CHECK(key_proj.size(0) == B && key_proj.size(3) == D, "key_proj shape mismatch");
  TORCH_CHECK(QH % KVH == 0, "QH must be divisible by KVH");
  TORCH_CHECK(C % 64 == 0, "projected query C must be divisible by 64");
  TORCH_CHECK(C <= 512, "fused project+pack supports C <= 512");
  TORCH_CHECK(D <= 256, "fused project+pack supports D <= 256");
  const int KBlocks = C / 64;
  TORCH_CHECK(packed_projected_query.size(0) == B && packed_projected_query.size(1) == QH &&
              packed_projected_query.size(2) == S && packed_projected_query.size(3) == KBlocks &&
              packed_projected_query.size(4) == 8, "packed_projected_query shape mismatch");
  TORCH_CHECK(query_scale.size(0) == B && query_scale.size(1) == QH && query_scale.size(2) == S,
              "query_scale shape must be [B,QH,S]");

  const int rows = B * QH * S;
  const int threads = 256;
  if (query.scalar_type() == at::kBFloat16) {
    picokv_project_quantize_pack_query_kernel<__nv_bfloat16><<<rows, threads, 0, stream>>>(
      (const __nv_bfloat16*)query.data_ptr<at::BFloat16>(),
      (const __nv_bfloat16*)key_proj.data_ptr<at::BFloat16>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      B, QH, KVH, S, C, D
    );
  } else {
    picokv_project_quantize_pack_query_kernel<float><<<rows, threads, 0, stream>>>(
      query.data_ptr<float>(),
      key_proj.data_ptr<float>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      B, QH, KVH, S, C, D
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void picokv_quantize_pack_projected_query(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
) {
  at::cuda::CUDAGuard device_guard(projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  TORCH_CHECK(projected_query.dim() == 4, "projected_query must have shape [B,H,S,C]");
  TORCH_CHECK(projected_query.is_cuda(), "projected_query must be a CUDA tensor");
  TORCH_CHECK(projected_query.is_contiguous(), "projected_query must be contiguous");
  TORCH_CHECK(projected_query.scalar_type() == at::kBFloat16 || projected_query.scalar_type() == at::kFloat,
              "projected_query must be torch.bfloat16 or torch.float32");
  TORCH_CHECK(packed_projected_query.scalar_type() == at::kLong, "packed_projected_query must be torch.long");
  TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be torch.float32");
  TORCH_CHECK(packed_projected_query.is_cuda() && query_scale.is_cuda(), "outputs must be CUDA tensors");
  TORCH_CHECK(packed_projected_query.is_contiguous(), "packed_projected_query must be contiguous");
  TORCH_CHECK(query_scale.is_contiguous(), "query_scale must be contiguous");

  const int B = projected_query.size(0);
  const int H = projected_query.size(1);
  const int S = projected_query.size(2);
  const int C = projected_query.size(3);
  TORCH_CHECK(C % 64 == 0, "projected query C must be divisible by 64");
  TORCH_CHECK(C <= 512, "fused projected-query pack currently supports C <= 512");
  const int KBlocks = C / 64;
  TORCH_CHECK(packed_projected_query.size(0) == B, "packed query B mismatch");
  TORCH_CHECK(packed_projected_query.size(1) == H, "packed query H mismatch");
  TORCH_CHECK(packed_projected_query.size(2) == S, "packed query S mismatch");
  TORCH_CHECK(packed_projected_query.size(3) == KBlocks, "packed query KBlocks mismatch");
  TORCH_CHECK(packed_projected_query.size(4) == 8, "packed query last dim must be 8");
  TORCH_CHECK(query_scale.size(0) == B && query_scale.size(1) == H && query_scale.size(2) == S,
              "query_scale shape must be [B,H,S]");

  const int rows = B * H * S;
  const int threads = 256;
  if (projected_query.scalar_type() == at::kBFloat16) {
    picokv_quantize_pack_projected_query_kernel<__nv_bfloat16><<<rows, threads, 0, stream>>>(
      (const __nv_bfloat16*)projected_query.data_ptr<at::BFloat16>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      B, H, S, C
    );
  } else {
    picokv_quantize_pack_projected_query_kernel<float><<<rows, threads, 0, stream>>>(
      projected_query.data_ptr<float>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      B, H, S, C
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Quantize+pack that also zeroes the two fp32 tail accumulators in the same
// launch (see the kernel comment). The accumulators must be [B,H,S,Cv] and
// [B,H,S] contiguous fp32, matching what the removed cudaMemsetAsync cleared.
void picokv_quantize_pack_zero_projected_query(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor query_scale,
  at::Tensor code_zero,
  at::Tensor den_zero
) {
  at::cuda::CUDAGuard device_guard(projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(projected_query.dim() == 4, "projected_query must have shape [B,H,S,C]");
  TORCH_CHECK(projected_query.is_contiguous(), "projected_query must be contiguous");
  TORCH_CHECK(projected_query.scalar_type() == at::kBFloat16 || projected_query.scalar_type() == at::kFloat,
              "projected_query must be torch.bfloat16 or torch.float32");
  TORCH_CHECK(code_zero.is_contiguous() && den_zero.is_contiguous(), "accumulators must be contiguous");
  TORCH_CHECK(code_zero.scalar_type() == at::kFloat && den_zero.scalar_type() == at::kFloat,
              "accumulators must be float32");

  const int B = projected_query.size(0);
  const int H = projected_query.size(1);
  const int S = projected_query.size(2);
  const int C = projected_query.size(3);
  const int Cv = code_zero.size(3);
  TORCH_CHECK(code_zero.dim() == 4 && code_zero.size(0) == B && code_zero.size(1) == H &&
              code_zero.size(2) == S && Cv >= 1, "code_zero must be [B,H,S,Cv]");
  TORCH_CHECK(den_zero.dim() == 3 && den_zero.size(0) == B && den_zero.size(1) == H &&
              den_zero.size(2) == S, "den_zero must be [B,H,S]");

  const int rows = B * H * S;
  const int threads = 256;
  // PICOKV_QQ_WARP=0: the block-per-row kernel (A/B; outputs identical)
  static const bool warp = [] {
    const char *e = getenv("PICOKV_QQ_WARP");
    return e == nullptr || e[0] != '0';
  }();
  if (warp && C % 64 == 0 && C <= 512) {
    const int blocks = (rows + 7) / 8;
    if (projected_query.scalar_type() == at::kBFloat16) {
      picokv_quantize_pack_zero_projected_query_warp_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)projected_query.data_ptr<at::BFloat16>(),
        packed_projected_query.data_ptr<long>(), query_scale.data_ptr<float>(),
        code_zero.data_ptr<float>(), den_zero.data_ptr<float>(), rows, C, Cv);
    } else {
      picokv_quantize_pack_zero_projected_query_warp_kernel<float><<<blocks, threads, 0, stream>>>(
        projected_query.data_ptr<float>(),
        packed_projected_query.data_ptr<long>(), query_scale.data_ptr<float>(),
        code_zero.data_ptr<float>(), den_zero.data_ptr<float>(), rows, C, Cv);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (projected_query.scalar_type() == at::kBFloat16) {
    picokv_quantize_pack_zero_projected_query_kernel<__nv_bfloat16><<<rows, threads, 0, stream>>>(
      (const __nv_bfloat16*)projected_query.data_ptr<at::BFloat16>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      code_zero.data_ptr<float>(),
      den_zero.data_ptr<float>(),
      C,
      Cv
    );
  } else {
    picokv_quantize_pack_zero_projected_query_kernel<float><<<rows, threads, 0, stream>>>(
      projected_query.data_ptr<float>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      code_zero.data_ptr<float>(),
      den_zero.data_ptr<float>(),
      C,
      Cv
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void picokv_pack_projected_query(
  at::Tensor quant_projected_query,
  at::Tensor packed_projected_query
) {
  at::cuda::CUDAGuard device_guard(quant_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  TORCH_CHECK(quant_projected_query.dim() == 4, "quant_projected_query must have shape [B,H,S,C]");
  TORCH_CHECK(quant_projected_query.scalar_type() == at::kChar, "quant_projected_query must be torch.int8");
  TORCH_CHECK(packed_projected_query.scalar_type() == at::kLong, "packed_projected_query must be torch.long");
  TORCH_CHECK(quant_projected_query.is_contiguous(), "quant_projected_query must be contiguous");
  TORCH_CHECK(packed_projected_query.is_contiguous(), "packed_projected_query must be contiguous");

  const int B = quant_projected_query.size(0);
  const int H = quant_projected_query.size(1);
  const int S = quant_projected_query.size(2);
  const int C = quant_projected_query.size(3);
  TORCH_CHECK(C % 64 == 0, "projected query C must be divisible by 64");
  const int KBlocks = C / 64;
  TORCH_CHECK(packed_projected_query.size(0) == B, "packed query B mismatch");
  TORCH_CHECK(packed_projected_query.size(1) == H, "packed query H mismatch");
  TORCH_CHECK(packed_projected_query.size(2) == S, "packed query S mismatch");
  TORCH_CHECK(packed_projected_query.size(3) == KBlocks, "packed query KBlocks mismatch");
  TORCH_CHECK(packed_projected_query.size(4) == 8, "packed query last dim must be 8");

  const int total_words = B * H * S * KBlocks;
  const int threads = 256;
  const int blocks = (total_words + threads - 1) / threads;
  picokv_pack_projected_query_kernel<<<blocks, threads, 0, stream>>>(
    quant_projected_query.data_ptr<int8_t>(),
    packed_projected_query.data_ptr<long>(),
    total_words
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void picokv_global_code_output_packed_value_tiled_gqa4(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int value_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int KVH = packed_key.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  const int L = key_transposed ? packed_key.size(3) : packed_key.size(2);
  const int VBlocks = packed_value.size(3);
  const int Cv = VBlocks * 64;
  const int num_tiles = (L + tile_size - 1) / tile_size;

  TORCH_CHECK(QH == 4 * KVH, "gqa4 path requires QH == 4 * KVH");
  TORCH_CHECK(KBlocks == 2 || KBlocks == 4, "gqa4 path currently requires KBlocks == 2 or 4");
  TORCH_CHECK(value_bits == 2, "recovered gqa4 path currently supports value_bits == 2");
  TORCH_CHECK(tile_size == kTileSize, "gqa4 path currently requires tile_size == 256");

  dim3 partial_threads(kTileSize);
  dim3 partial_blocks(num_tiles, S, B * KVH);
#define LAUNCH_PACKED_VALUE(KB, KEY_T) \
  picokv_packed_value_partial_kernel<KB, KEY_T><<<partial_blocks, partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(), \
    partial_den.data_ptr<float>(), \
    B, QH, KVH, S, L, VBlocks, value_bits, num_tiles, \
    query_scale.numel(), \
    (float)attention_scale \
  )

  if (KBlocks == 2) {
    if (key_transposed) { LAUNCH_PACKED_VALUE(2, true); }
    else { LAUNCH_PACKED_VALUE(2, false); }
  } else {
    if (key_transposed) { LAUNCH_PACKED_VALUE(4, true); }
    else { LAUNCH_PACKED_VALUE(4, false); }
  }
#undef LAUNCH_PACKED_VALUE
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 reduce_threads(512);
  dim3 reduce_blocks(S, B * QH);
  picokv_global_code_output_packed_value_tiled_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
    (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(),
    partial_den.data_ptr<float>(),
    (__nv_bfloat16*)glob_code_output.data_ptr<at::BFloat16>(),
    glob_den.data_ptr<float>(),
    B, QH, S, num_tiles, Cv
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void picokv_global_code_output_packed_value_tiled_gqa4_tensorcore(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int value_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
  picokv_global_code_output_packed_value_tiled_gqa4(
    packed_projected_query,
    packed_key,
    packed_value,
    query_scale,
    dense_lse,
    partial_code_output,
    partial_den,
    glob_code_output,
    glob_den,
    value_bits,
    tile_size,
    key_transposed,
    attention_scale
  );
}

void picokv_global_code_output_v1_exp8_tiled_gqa4(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int KVH = packed_key.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  const int L = key_transposed ? packed_key.size(3) : packed_key.size(2);
  const int LGroups = packed_value_sign.size(2);
  const int Cv = packed_value_sign.size(3);
  const int num_tiles = (L + tile_size - 1) / tile_size;
  const int group_size = KVH == 0 ? 0 : QH / KVH;

  TORCH_CHECK(QH == group_size * KVH && (group_size == 1 || group_size == 4 || group_size == 8), "v1 runtime-exp path requires QH == KVH, 4 * KVH, or 8 * KVH");
  TORCH_CHECK(KBlocks == 2 || KBlocks == 4, "v1 runtime-exp path currently requires KBlocks == 2 or 4");
  TORCH_CHECK(tile_size == kTileSize, "v1 runtime-exp path currently requires tile_size == 256");
  TORCH_CHECK(Cv == 128 || Cv == 256, "v1 runtime-exp path currently requires Cv == 128 or 256");
  TORCH_CHECK(LGroups == (L + 63) / 64, "packed_value_sign must have shape [B,KVH,ceil(L/64),Cv]");
  TORCH_CHECK(exp_bits == 1 || exp_bits == 2 || exp_bits == 4 || exp_bits == 8, "exp_bits must be one of 1, 2, 4, or 8");

  dim3 partial_threads(kTileSize);
  dim3 partial_blocks(num_tiles, S, B * KVH);
#define LAUNCH_V1_EXP_BITS(EB, GS, KB, CV, KEY_T) \
  picokv_v1_expbits_partial_kernel<GS, KB, CV, KEY_T, false, EB><<<partial_blocks, partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(), \
    partial_den.data_ptr<float>(), \
    nullptr, \
    nullptr, \
    B, QH, KVH, S, L, LGroups, num_tiles, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define DISPATCH_V1_EXP_BITS_GS(EB, GS) \
  do { \
    if (KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS(EB, GS, 4, 256, true); } \
      else { LAUNCH_V1_EXP_BITS(EB, GS, 4, 256, false); } \
    } else if (KBlocks == 2 && Cv == 128) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS(EB, GS, 2, 128, true); } \
      else { LAUNCH_V1_EXP_BITS(EB, GS, 2, 128, false); } \
    } else if (KBlocks == 4 && Cv == 128) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS(EB, GS, 4, 128, true); } \
      else { LAUNCH_V1_EXP_BITS(EB, GS, 4, 128, false); } \
    } else if (KBlocks == 2 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS(EB, GS, 2, 256, true); } \
      else { LAUNCH_V1_EXP_BITS(EB, GS, 2, 256, false); } \
    } else { \
      TORCH_CHECK(false, "unsupported v1 runtime-exp KBlocks/Cv combination"); \
    } \
  } while (0)
#define DISPATCH_V1_EXP_BITS(EB) \
  do { \
    if (group_size == 1) { DISPATCH_V1_EXP_BITS_GS(EB, 1); } \
    else if (group_size == 8) { DISPATCH_V1_EXP_BITS_GS(EB, 8); } \
    else { DISPATCH_V1_EXP_BITS_GS(EB, 4); } \
  } while (0)

  if (exp_bits == 1) { DISPATCH_V1_EXP_BITS(1); }
  else if (exp_bits == 2) { DISPATCH_V1_EXP_BITS(2); }
  else if (exp_bits == 4) { DISPATCH_V1_EXP_BITS(4); }
  else { DISPATCH_V1_EXP_BITS(8); }
#undef DISPATCH_V1_EXP_BITS
#undef DISPATCH_V1_EXP_BITS_GS
#undef LAUNCH_V1_EXP_BITS
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 reduce_threads(512);
  dim3 reduce_blocks(S, B * QH);
  picokv_global_code_output_packed_value_tiled_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
    (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(),
    partial_den.data_ptr<float>(),
    (__nv_bfloat16*)glob_code_output.data_ptr<at::BFloat16>(),
    glob_den.data_ptr<float>(),
    B, QH, S, num_tiles, Cv
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void picokv_global_output_v1_exp8_tiled_gqa4_fused_reduce(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor dense_output,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int KVH = packed_key.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  const int L = key_transposed ? packed_key.size(3) : packed_key.size(2);
  const int LGroups = packed_value_sign.size(2);
  const int Cv = packed_value_sign.size(3);
  const int num_tiles = (L + tile_size - 1) / tile_size;
  const int D = dense_output.size(3);
  const int group_size = KVH == 0 ? 0 : QH / KVH;

  TORCH_CHECK(QH == group_size * KVH && (group_size == 1 || group_size == 4 || group_size == 8), "v1 fused-reduce path requires QH == KVH, 4 * KVH, or 8 * KVH");
  TORCH_CHECK(KBlocks == 2 || KBlocks == 4, "v1 fused-reduce path currently requires KBlocks == 2 or 4");
  TORCH_CHECK(tile_size == kTileSize, "v1 fused-reduce path currently requires tile_size == 256");
  TORCH_CHECK(Cv == 128 || Cv == 256, "v1 fused-reduce path currently requires Cv == 128 or 256");
  TORCH_CHECK(D <= 256, "v1 fused-reduce output projection supports D <= 256");
  TORCH_CHECK(LGroups == (L + 63) / 64, "packed_value_sign must have shape [B,KVH,ceil(L/64),Cv]");
  TORCH_CHECK(exp_bits == 1 || exp_bits == 2 || exp_bits == 4 || exp_bits == 8, "exp_bits must be one of 1, 2, 4, or 8");
  TORCH_CHECK(partial_code_output.scalar_type() == at::kBFloat16, "partial_code_output must be bf16");
  TORCH_CHECK(partial_den.scalar_type() == at::kFloat, "partial_den must be fp32");
  TORCH_CHECK(dense_output.scalar_type() == value_proj.scalar_type(), "dense_output and value_proj dtype mismatch");
  TORCH_CHECK(attn_output.scalar_type() == dense_output.scalar_type(), "attn_output dtype mismatch");
  TORCH_CHECK(dense_output.size(0) == B && dense_output.size(1) == QH && dense_output.size(2) == S, "dense_output shape mismatch");
  TORCH_CHECK(value_proj.size(0) == B && value_proj.size(1) == KVH && value_proj.size(2) == Cv && value_proj.size(3) == D, "value_proj shape mismatch");
  TORCH_CHECK(attn_output.sizes() == dense_output.sizes(), "attn_output shape mismatch");

  dim3 partial_threads(kTileSize);
  dim3 partial_blocks(num_tiles, S, B * KVH);
#define LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, KB, CV, KEY_T) \
  picokv_v1_expbits_partial_kernel<GS, KB, CV, KEY_T, false, EB><<<partial_blocks, partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(), \
    partial_den.data_ptr<float>(), \
    nullptr, \
    nullptr, \
    B, QH, KVH, S, L, LGroups, num_tiles, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define DISPATCH_V1_EXP_BITS_FUSED_REDUCE_GS(EB, GS) \
  do { \
    if (KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 4, 256, true); } \
      else { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 4, 256, false); } \
    } else if (KBlocks == 2 && Cv == 128) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 2, 128, true); } \
      else { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 2, 128, false); } \
    } else if (KBlocks == 4 && Cv == 128) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 4, 128, true); } \
      else { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 4, 128, false); } \
    } else if (KBlocks == 2 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 2, 256, true); } \
      else { LAUNCH_V1_EXP_BITS_FUSED_REDUCE(EB, GS, 2, 256, false); } \
    } else { \
      TORCH_CHECK(false, "unsupported v1 fused-reduce KBlocks/Cv combination"); \
    } \
  } while (0)
#define DISPATCH_V1_EXP_BITS_FUSED_REDUCE(EB) \
  do { \
    if (group_size == 1) { DISPATCH_V1_EXP_BITS_FUSED_REDUCE_GS(EB, 1); } \
    else if (group_size == 8) { DISPATCH_V1_EXP_BITS_FUSED_REDUCE_GS(EB, 8); } \
    else { DISPATCH_V1_EXP_BITS_FUSED_REDUCE_GS(EB, 4); } \
  } while (0)

  if (exp_bits == 1) { DISPATCH_V1_EXP_BITS_FUSED_REDUCE(1); }
  else if (exp_bits == 2) { DISPATCH_V1_EXP_BITS_FUSED_REDUCE(2); }
  else if (exp_bits == 4) { DISPATCH_V1_EXP_BITS_FUSED_REDUCE(4); }
  else { DISPATCH_V1_EXP_BITS_FUSED_REDUCE(8); }
#undef DISPATCH_V1_EXP_BITS_FUSED_REDUCE
#undef DISPATCH_V1_EXP_BITS_FUSED_REDUCE_GS
#undef LAUNCH_V1_EXP_BITS_FUSED_REDUCE
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 reduce_threads(256);
  dim3 reduce_blocks(S, B * QH);
  if (dense_output.scalar_type() == at::kBFloat16) {
    picokv_v1_fused_reduce_output_projection_kernel<__nv_bfloat16><<<reduce_blocks, reduce_threads, 0, stream>>>(
      (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(),
      partial_den.data_ptr<float>(),
      (const __nv_bfloat16*)dense_output.data_ptr<at::BFloat16>(),
      (const __nv_bfloat16*)value_proj.data_ptr<at::BFloat16>(),
      (__nv_bfloat16*)attn_output.data_ptr<at::BFloat16>(),
      B, QH, KVH, S, num_tiles, Cv, D
    );
  } else if (dense_output.scalar_type() == at::kFloat) {
    picokv_v1_fused_reduce_output_projection_kernel<float><<<reduce_blocks, reduce_threads, 0, stream>>>(
      (__nv_bfloat16*)partial_code_output.data_ptr<at::BFloat16>(),
      partial_den.data_ptr<float>(),
      dense_output.data_ptr<float>(),
      value_proj.data_ptr<float>(),
      attn_output.data_ptr<float>(),
      B, QH, KVH, S, num_tiles, Cv, D
    );
  } else {
    TORCH_CHECK(false, "v1 fused-reduce output projection supports bf16 and fp32 dense output");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// __launch_bounds__ min-blocks selector for the runtime-unrolled kernel.
// Default 4: measured 1-7% faster than the historical 6 across production
// shapes (equal-batch, ragged, long-L) with no regression; 64 regs/thread.
// PICOKV_TAIL_LB overrides per environment.
static inline int picokv_tail_min_blocks() {
  static int value = []() {
    const char *env = getenv("PICOKV_TAIL_LB");
    return env ? atoi(env) : 4;
  }();
  return value;
}

// Escape hatch / A-B switch for the KBlocks==5 compile-time specialization of
// the runtime-unrolled kernel: PICOKV_NO_KBSPEC=1 forces the generic runtime
// kernel from the same binary.
static inline bool picokv_kb_spec_disabled() {
  static int value = []() {
    const char *env = getenv("PICOKV_NO_KBSPEC");
    return env ? atoi(env) : 0;
  }();
  return value != 0;
}

// Tensor-core (bmma) score path for the GQA-4 / Ck=320 runtime-unrolled
// kernel: PICOKV_SCORE_BMMA=1 swaps the popc key-match section for the
// m8n8k128 bmma path (dots bitwise identical; KBlocks=5 chunks 0-2, chunk 2
// zero-padded past kb 4). Default off. PICOKV_NO_BMMA=1 forces the popc path
// from the same binary even when enabled.
static inline bool picokv_score_bmma_enabled() {
  static int value = []() {
    const char *env = getenv("PICOKV_SCORE_BMMA");
    const char *no = getenv("PICOKV_NO_BMMA");
    return (env != nullptr && atoi(env) != 0 && !(no != nullptr && atoi(no) != 0)) ? 1 : 0;
  }();
  return value != 0;
}

// One-tile (BlockTiles=1) blocks for the KBlocks==5 popc kernel at short
// contexts: twice the blocks, each half the serial tile work, but twice the
// atomics into each (request, kv-head) accumulator. Engine BCG ITL (ms/step,
// BT1 vs BT2): L<=16k wins at every batch (b1@8k -0.16, b16@8k -0.13,
// b1@16k -0.09, b8@16k -0.03); L>=32k loses (b1@32k +0.02, b16@32k +0.22,
// b1@64k +0.01). PICOKV_TAIL_BT1: unset/-1 = auto (L <= 16384), 0 = never,
// 1 = whenever unscheduled.
static inline int picokv_tail_bt1_mode() {
  static int value = []() {
    const char *env = getenv("PICOKV_TAIL_BT1");
    return env ? atoi(env) : -1;
  }();
  return value;
}

static inline bool picokv_tail_use_bt1(int L, bool scheduled) {
  if (scheduled) return false;  // sched tables index 512-token blocks
  const int mode = picokv_tail_bt1_mode();
  if (mode >= 0) return mode != 0;
  return L <= 16384;
}

// Tensor-core tail main kernel (picokv_tc_tail.cuh) for GQA-4 / Ck=320 /
// Cv=320: score and value inner products on mma.sync b1 AND+POPC, same math,
// bitwise-identical single-writer outputs, ~2.4-3.2x faster than the popc
// kernel. Default on; PICOKV_TC_TAIL=0 or PICOKV_NO_TC=1 restores the popc
// selector below from the same binary.
static inline bool picokv_tc_tail_enabled() {
  static int value = []() {
    const char *env = getenv("PICOKV_TC_TAIL");
    const char *no = getenv("PICOKV_NO_TC");
    if (no != nullptr && atoi(no) != 0) return 0;
    return (env == nullptr || atoi(env) != 0) ? 1 : 0;
  }();
  return value != 0;
}

// Tiles (256 tokens) per block for the unscheduled tc grid. Standalone sweep
// (H800, LB4): BT1 wins while the BT2 grid underfills the GPU (b1@8k 7.4 vs
// 8.9 us), BT4 once its grid is >= 8 waves of blocks (b16@120k 626 vs 636 us),
// BT2 otherwise. Sched grids are fixed at BT2 (tables index 512-token blocks).
// PICOKV_TC_BT=1/2/4 overrides.
static inline int picokv_tc_tail_block_tiles(int L, int BKVH, bool scheduled) {
  if (scheduled) return 2;
  static int forced = []() {
    const char *env = getenv("PICOKV_TC_BT");
    const int v = env ? atoi(env) : 0;
    return (v == 1 || v == 2 || v == 4) ? v : 0;
  }();
  if (forced) return forced;
  static int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long grid_bt2 = (long)((L + 511) / 512) * BKVH;
  const long grid_bt4 = (long)((L + 1023) / 1024) * BKVH;
  if (grid_bt2 < 2L * sms) return 1;
  if (grid_bt4 >= 8L * sms) return 4;
  return 2;
}

int picokv_tail_block_tiles(int L, int BKVH) {
  return picokv_tc_tail_block_tiles(L, BKVH, false);
}

void picokv_global_code_output_v1_exp8_tiled_gqa4_atomic(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  at::Tensor global_lens,
  at::Tensor sched_req,
  at::Tensor sched_tile,
  at::Tensor row_kptr,
  at::Tensor row_vptr,
  at::Tensor row_cap,
  at::Tensor part_code,
  at::Tensor part_den,
  at::Tensor part_rs,
  at::Tensor projected_query
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int KVH = packed_key.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  const int L = key_transposed ? packed_key.size(3) : packed_key.size(2);
  const int LGroups = packed_value_sign.size(2);
  const int Cv = packed_value_sign.size(3);
  const int num_tiles = (L + tile_size - 1) / tile_size;
  const int group_size = KVH == 0 ? 0 : QH / KVH;
  const int *gl_ptr = (global_lens.defined() && global_lens.numel() > 0)
                        ? global_lens.data_ptr<int>() : nullptr;
  if (gl_ptr != nullptr) {
    TORCH_CHECK((group_size == 4 || group_size == 8) && KBlocks >= 5,
                "picokv varlen (global_lens) is only wired for the runtime-unrolled kernel (group_size 4/8, Ck>=320)");
  }
  const int *sr_ptr = (sched_req.defined() && sched_req.numel() > 0) ? sched_req.data_ptr<int>() : nullptr;
  const int *st_ptr = (sched_tile.defined() && sched_tile.numel() > 0) ? sched_tile.data_ptr<int>() : nullptr;
  // the kernels read sched_tile[x] for every sched_req entry x
  TORCH_CHECK((sr_ptr == nullptr) == (st_ptr == nullptr) &&
                (sr_ptr == nullptr || sched_tile.numel() == sched_req.numel()),
              "picokv tail: sched_req and sched_tile must both be empty or the same length");
  const int sched_gx = (sr_ptr != nullptr) ? (int)sched_req.numel() : (L + 511) / 512;
  const int sched_gz = (sr_ptr != nullptr) ? KVH : (B * KVH);
  const bool rows = row_cap.defined() && row_cap.numel() > 0;
  const long long *rk_ptr = nullptr;
  const long long *rv_ptr = nullptr;
  const int *rc_ptr = nullptr;
  if (rows) {
    TORCH_CHECK(row_kptr.defined() && row_vptr.defined() && row_kptr.numel() >= B &&
                row_vptr.numel() >= B && row_cap.numel() >= B &&
                row_kptr.scalar_type() == at::kLong && row_vptr.scalar_type() == at::kLong &&
                row_cap.scalar_type() == at::kInt,
                "picokv row tables: int64 row_kptr/row_vptr and int32 row_cap, >= B entries");
    TORCH_CHECK(sr_ptr != nullptr && st_ptr != nullptr && gl_ptr != nullptr && key_transposed &&
                group_size == 4 && KBlocks == 5 && Cv == 320 && picokv_tc_tail_enabled(),
                "picokv row tables need the tensor-core sched path (GQA-4, Ck=Cv=320, "
                "transposed keys, global_lens + sched tables, TC tail enabled)");
    rk_ptr = reinterpret_cast<const long long *>(row_kptr.data_ptr<int64_t>());
    rv_ptr = reinterpret_cast<const long long *>(row_vptr.data_ptr<int64_t>());
    rc_ptr = row_cap.data_ptr<int>();
  }
  // Deterministic mode: per-block partials, no atomics (TC kernel only; the
  // reduction's tile size is picokv_tail_block_tiles, or 2 for sched grids).
  const bool part = part_code.defined() && part_code.numel() > 0;
  float *pc_ptr = nullptr;
  float *pd_ptr = nullptr;
  int *prs_ptr = nullptr;
  if (part) {
    TORCH_CHECK(group_size == 4 && KBlocks == 5 && Cv == 320 && picokv_tc_tail_enabled(),
                "picokv deterministic tail needs the tensor-core path (GQA-4, Ck=Cv=320, TC tail enabled)");
    const int blk = 256 * picokv_tc_tail_block_tiles(L, B * KVH, sr_ptr != nullptr);
    const long gx = (sr_ptr != nullptr) ? (long)sched_req.numel() : (long)((L + blk - 1) / blk);
    const long slots = (long)S * sched_gz * gx;
    TORCH_CHECK(part_code.scalar_type() == at::kFloat && part_den.defined() &&
                part_den.scalar_type() == at::kFloat && part_code.is_contiguous() &&
                part_den.is_contiguous() && part_code.numel() >= slots * 4 * 320 &&
                part_den.numel() >= slots * 4,
                "picokv deterministic tail: fp32 part_code >= [S*GZ*GX,4,320], part_den >= [S*GZ*GX,4]");
    pc_ptr = part_code.data_ptr<float>();
    pd_ptr = part_den.data_ptr<float>();
    if (sr_ptr != nullptr) {
      TORCH_CHECK(part_rs.defined() && part_rs.scalar_type() == at::kInt && part_rs.numel() >= B,
                  "picokv deterministic sched tail: int32 part_rs >= [B]");
      prs_ptr = part_rs.data_ptr<int>();
    }
  }

  // Fused query quantizer (TC kernel only): the bf16 projected query replaces
  // packed_projected_query / query_scale (still passed, for shapes only).
  const bool fused_q = projected_query.defined() && projected_query.numel() > 0;
  const __nv_bfloat16 *pjq_ptr = nullptr;
  if (fused_q) {
    TORCH_CHECK(group_size == 4 && KBlocks == 5 && Cv == 320 && exp_bits >= 4 &&
                picokv_tc_tail_enabled() && !part,
                "picokv fused-quantizer tail needs the tensor-core path (GQA-4, Ck=Cv=320, "
                "exp_bits >= 4, TC tail enabled, not deterministic)");
    TORCH_CHECK(projected_query.scalar_type() == at::kBFloat16 && projected_query.is_contiguous() &&
                projected_query.numel() == (long)B * QH * S * 320,
                "picokv fused-quantizer tail: contiguous bf16 projected_query [B,QH,S,320]");
    pjq_ptr = reinterpret_cast<const __nv_bfloat16 *>(projected_query.data_ptr<at::BFloat16>());
  }
  // Sched split: a small batch's sched grid (512-token entries, BT 2) runs
  // as 2 one-tile blocks per entry, the width the unscheduled grid picks below
  // 2 waves (t62/mb/tail_bt.py: B1 x 1k 7.5 -> 5.0 us, B8 x 1k 9.0 -> 7.3).
  // PICOKV_TC_SCHED_SPLIT=0 off, 1 always, default below 2 blocks per SM.
  int sched_shift = 0;
  if (sr_ptr != nullptr && !part) {
    static const int split_env = []() {
      const char *env = getenv("PICOKV_TC_SCHED_SPLIT");
      return env ? atoi(env) : -1;
    }();
    static const int split_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    sched_shift = split_env == 1 ? 1
                : split_env == 0 ? 0
                : ((long)sched_gx * S * sched_gz < 2L * split_sms ? 1 : 0);
  }
  TORCH_CHECK(QH == group_size * KVH && (group_size == 1 || group_size == 4 || group_size == 8), "v1 runtime-exp atomic path requires QH == KVH, 4 * KVH, or 8 * KVH");
  TORCH_CHECK(KBlocks == 2 || KBlocks == 3 || KBlocks == 4 || KBlocks == 5 || KBlocks == 6 || KBlocks == 7 || KBlocks == 8, "v1 runtime-exp atomic path currently requires Ck in {128,192,256,320,384,448,512}");
  TORCH_CHECK(tile_size == kTileSize, "v1 runtime-exp atomic path currently requires tile_size == 256");
  TORCH_CHECK(Cv == 128 || Cv == 192 || Cv == 256 || Cv == 320 || Cv == 384 || Cv == 448 || Cv == 512, "v1 runtime-exp atomic path currently requires Cv in {128,192,256,320,384,448,512}");
  TORCH_CHECK(LGroups == (L + 63) / 64, "packed_value_sign must have shape [B,KVH,ceil(L/64),Cv]");
  TORCH_CHECK(exp_bits == 1 || exp_bits == 2 || exp_bits == 4 || exp_bits == 8, "exp_bits must be one of 1, 2, 4, or 8");
  TORCH_CHECK(glob_code_output.scalar_type() == at::kFloat, "atomic glob_code_output must be float32");
  TORCH_CHECK(glob_den.scalar_type() == at::kFloat, "atomic glob_den must be float32");

  dim3 partial_threads(kTileSize);
#define LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, KB, CV, KEY_T) \
  picokv_v1_expbits_partial_kernel<GS, KB, CV, KEY_T, true, EB><<<dim3(num_tiles, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    nullptr, \
    nullptr, \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, num_tiles, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC2(EB, KEY_T) \
  picokv_v1_expbits_atomic2_kernel<KEY_T, EB><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, GS, KEY_T) \
  picokv_v1_expbits_atomic2_runtime_kernel<GS, KEY_T, EB, 0><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    0.0f \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, GS, KEY_T, LB, KB) \
  picokv_v1_expbits_atomic2_runtime_unrolled_kernel<GS, KEY_T, EB, 0, LB, KB><<<dim3(sched_gx, S, sched_gz), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    0.0f, \
    gl_ptr, \
    sr_ptr, \
    st_ptr \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB(EB, GS, KEY_T) \
  do { \
    if (picokv_tail_min_blocks() == 4) { \
      LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, GS, KEY_T, 4, 0); \
    } else { \
      LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, GS, KEY_T, 6, 0); \
    } \
  } while (0)
// bmma score instantiation: GQA-4, KBlocks=5 compile-time, PruneMode=0, and
// UseBmmaScore=1. Same launch config / argument list as the popc kernel.
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_BMMA(EB, KEY_T, LB) \
  picokv_v1_expbits_atomic2_runtime_unrolled_kernel<4, KEY_T, EB, 0, LB, 5, 1><<<dim3(sched_gx, S, sched_gz), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    0.0f, \
    gl_ptr, \
    sr_ptr, \
    st_ptr \
  )
// One-tile-per-block variant of the KBlocks=5 popc kernel (LB 4, unscheduled
// grid only — sched tables address 512-token blocks).
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_BT1(EB, KEY_T) \
  picokv_v1_expbits_atomic2_runtime_unrolled_kernel<4, KEY_T, EB, 0, 4, 5, 0, 1><<<dim3((L + 255) / 256, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    0.0f, \
    gl_ptr, \
    nullptr, \
    nullptr \
  )
// Tensor-core tail kernel (GQA-4, Ck=320, Cv=320, LB 4). Unscheduled grid with
// BT tiles per block, or the sched grid (BT 2, 512-token sched tiles).
#define LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID(EB, KEY_T, BT, GRID, SR, ST, SH) \
  do { \
    if (pjq_ptr != nullptr) { \
      LAUNCH_V1_EXP_BITS_ATOMIC_TC_FQ(EB, KEY_T, BT, GRID, SR, ST, SH, (EB >= 4)); \
    } else { \
      LAUNCH_V1_EXP_BITS_ATOMIC_TC_FQ(EB, KEY_T, BT, GRID, SR, ST, SH, false); \
    } \
  } while (0)
// FQ: the fused-quantizer instantiation (exp_bits >= 4 only: the op checks).
#define LAUNCH_V1_EXP_BITS_ATOMIC_TC_FQ(EB, KEY_T, BT, GRID, SR, ST, SH, FQ) \
  picokv_v1_expbits_atomic2_tc_kernel<KEY_T, EB, 320, 4, BT, FQ><<<GRID, partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    0.0f, \
    gl_ptr, \
    SR, \
    ST, \
    rk_ptr, \
    rv_ptr, \
    rc_ptr, \
    pc_ptr, \
    pd_ptr, \
    prs_ptr, \
    SH, \
    pjq_ptr \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC_TC(EB, KEY_T) \
  do { \
    if (sr_ptr != nullptr && sched_shift == 1) { \
      LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID(EB, KEY_T, 1, dim3(2 * sched_gx, S, sched_gz), sr_ptr, st_ptr, 1); \
    } else if (sr_ptr != nullptr) { \
      LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID(EB, KEY_T, 2, dim3(sched_gx, S, sched_gz), sr_ptr, st_ptr, 0); \
    } else { \
      const int tc_bt = picokv_tc_tail_block_tiles(L, B * KVH, false); \
      if (tc_bt == 1) { \
        LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID(EB, KEY_T, 1, dim3((L + 255) / 256, S, B * KVH), nullptr, nullptr, 0); \
      } else if (tc_bt == 4) { \
        LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID(EB, KEY_T, 4, dim3((L + 1023) / 1024, S, B * KVH), nullptr, nullptr, 0); \
      } else { \
        LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID(EB, KEY_T, 2, dim3((L + 511) / 512, S, B * KVH), nullptr, nullptr, 0); \
      } \
    } \
  } while (0)
// GQA-4 Ck>=320 selector: at KBlocks==5 / Cv==320 (the production shape)
// launch the tensor-core kernel (default; PICOKV_NO_TC=1 disables). Otherwise
// at KBlocks==5 launch the compile-time-specialized popc kernel — dead-guard
// elimination on the unrolled
// key loops. PICOKV_SCORE_BMMA=1 (default off) further swaps the popc
// key-match for the tensor-core bmma score path inside the same
// specialization. Other KBlocks, or PICOKV_NO_KBSPEC=1, keep the generic
// kernel.
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB_KB5(EB, KEY_T) \
  do { \
    if (KBlocks == 5 && Cv == 320 && picokv_tc_tail_enabled()) { \
      LAUNCH_V1_EXP_BITS_ATOMIC_TC(EB, KEY_T); \
    } else if (KBlocks == 5 && picokv_score_bmma_enabled()) { \
      if (picokv_tail_min_blocks() == 4) { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_BMMA(EB, KEY_T, 4); \
      } else { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_BMMA(EB, KEY_T, 6); \
      } \
    } else if (KBlocks == 5 && !picokv_kb_spec_disabled()) { \
      if (picokv_tail_min_blocks() == 4 && \
          picokv_tail_use_bt1(L, sr_ptr != nullptr)) { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_BT1(EB, KEY_T); \
      } else if (picokv_tail_min_blocks() == 4) { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 4, KEY_T, 4, 5); \
      } else { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 4, KEY_T, 6, 5); \
      } \
    } else { \
      if (picokv_tail_min_blocks() == 4) { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 4, KEY_T, 4, 0); \
      } else { \
        LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 4, KEY_T, 6, 0); \
      } \
    } \
  } while (0)
#define LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1_BT(EB, KEY_T, BT, DIV) \
  picokv_v1_expbits_atomic2_gqa1_kernel<KEY_T, EB, BT><<<dim3((L + (DIV - 1)) / DIV, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1(EB, KEY_T) \
  do { \
    if (L <= 65536) { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1_BT(EB, KEY_T, 3, 768); } \
    else { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1_BT(EB, KEY_T, 4, 1024); } \
  } while (0)
#define LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8_BASE(EB, KEY_T) \
  picokv_v1_expbits_atomic2_gqa8_kernel<KEY_T, EB><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8_SPLIT4(EB, KEY_T) \
  picokv_v1_expbits_atomic2_gqa8_split4_kernel<KEY_T, EB><<<dim3((L + 511) / 512, S, B * KVH * 2), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, \
    query_scale.numel(), \
    (float)attention_scale \
  )
#define LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8(EB, KEY_T) \
  do { \
    if (B == 1) { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8_SPLIT4(EB, KEY_T); } \
    else { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8_BASE(EB, KEY_T); } \
  } while (0)
#define DISPATCH_V1_EXP_BITS_ATOMIC_GS(EB, GS) \
  do { \
    if (KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 4, 256, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 4, 256, false); } \
    } else if (KBlocks == 2 && Cv == 128) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 2, 128, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 2, 128, false); } \
    } else if (KBlocks == 4 && Cv == 128) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 4, 128, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 4, 128, false); } \
    } else if (KBlocks == 2 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 2, 256, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC(EB, GS, 2, 256, false); } \
    } else { \
      TORCH_CHECK(false, "unsupported v1 runtime-exp atomic KBlocks/Cv combination"); \
    } \
  } while (0)
#define DISPATCH_V1_EXP_BITS_ATOMIC(EB) \
  do { \
    if (group_size == 1 && KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1(EB, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1(EB, false); } \
    } else if (group_size == 1 && KBlocks >= 5) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB(EB, 1, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB(EB, 1, false); } \
    } else if (group_size == 1) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 1, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 1, false); } \
    } else if (group_size == 4 && KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC2(EB, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC2(EB, false); } \
    } else if (group_size == 4 && KBlocks >= 5) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB_KB5(EB, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB_KB5(EB, false); } \
    } else if (group_size == 4) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 4, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 4, false); } \
    } else if (group_size == 8 && KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8(EB, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8(EB, false); } \
    } else if (group_size == 8 && KBlocks >= 5) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB(EB, 8, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB(EB, 8, false); } \
    } else if (group_size == 8) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 8, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 8, false); } \
    } else { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 4, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 4, false); } \
    } \
  } while (0)

  if (exp_bits == 1) { DISPATCH_V1_EXP_BITS_ATOMIC(1); }
  else if (exp_bits == 2) { DISPATCH_V1_EXP_BITS_ATOMIC(2); }
  else if (exp_bits == 4) { DISPATCH_V1_EXP_BITS_ATOMIC(4); }
  else { DISPATCH_V1_EXP_BITS_ATOMIC(8); }
#undef DISPATCH_V1_EXP_BITS_ATOMIC
#undef DISPATCH_V1_EXP_BITS_ATOMIC_GS
#undef LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8
#undef LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8_SPLIT4
#undef LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8_BASE
#undef LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1
#undef LAUNCH_V1_EXP_BITS_ATOMIC2_GQA1_BT
#undef LAUNCH_V1_EXP_BITS_ATOMIC2
#undef LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED
#undef LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_LB_KB5
#undef LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED_BMMA
#undef LAUNCH_V1_EXP_BITS_ATOMIC_TC
#undef LAUNCH_V1_EXP_BITS_ATOMIC_TC_GRID
#undef LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME
#undef LAUNCH_V1_EXP_BITS_ATOMIC
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  int prune_mode,
  double prune_log_eps
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int KVH = packed_key.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  const int L = key_transposed ? packed_key.size(3) : packed_key.size(2);
  const int LGroups = packed_value_sign.size(2);
  const int Cv = packed_value_sign.size(3);
  const int group_size = KVH == 0 ? 0 : QH / KVH;

  TORCH_CHECK(QH == group_size * KVH && group_size == 4, "pruned path currently requires QH == 4 * KVH");
  TORCH_CHECK(KBlocks == 2 || KBlocks == 3 || KBlocks == 4 || KBlocks == 5 || KBlocks == 6 || KBlocks == 7 || KBlocks == 8, "pruned path currently requires Ck in {128,192,256,320,384,448,512}");
  TORCH_CHECK(Cv == 128 || Cv == 192 || Cv == 256 || Cv == 320 || Cv == 384 || Cv == 448 || Cv == 512, "pruned path currently requires Cv in {128,192,256,320,384,448,512}");
  TORCH_CHECK(tile_size == kTileSize, "pruned path currently requires tile_size == 256");
  TORCH_CHECK(LGroups == (L + 63) / 64, "packed_value_sign must have shape [B,KVH,ceil(L/64),Cv]");
  TORCH_CHECK(exp_bits == 1 || exp_bits == 2 || exp_bits == 4 || exp_bits == 8, "exp_bits must be one of 1, 2, 4, or 8");
  TORCH_CHECK(prune_mode >= 1 && prune_mode <= 3, "prune_mode must be 1(block), 2(prefix), or 3(combined)");
  TORCH_CHECK(glob_code_output.scalar_type() == at::kFloat, "atomic glob_code_output must be float32");
  TORCH_CHECK(glob_den.scalar_type() == at::kFloat, "atomic glob_den must be float32");

  dim3 partial_threads(kTileSize);
#define LAUNCH_PRUNED(EB, KEY_T, MODE) \
  picokv_v1_expbits_atomic2_pruned_kernel<KEY_T, EB, MODE><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, \
    query_scale.numel(), \
    (float)attention_scale, \
    (float)prune_log_eps \
  )
#define LAUNCH_PRUNED_RUNTIME(EB, KEY_T, MODE) \
  picokv_v1_expbits_atomic2_runtime_kernel<4, KEY_T, EB, MODE><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    (float)prune_log_eps \
  )
#define LAUNCH_PRUNED_RUNTIME_UNROLLED(EB, KEY_T, MODE) \
  picokv_v1_expbits_atomic2_runtime_unrolled_kernel<4, KEY_T, EB, MODE><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, KBlocks, Cv, \
    query_scale.numel(), \
    (float)attention_scale, \
    (float)prune_log_eps, \
    nullptr, \
    nullptr, \
    nullptr \
  )
#define LAUNCH_PRUNED_FIXED43(KB, CV) \
  picokv_v1_expbits_atomic2_fixed_kernel<4, KB, CV, true, 4, 3><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
    packed_projected_query.data_ptr<long>(), \
    packed_key.data_ptr<long>(), \
    packed_value_sign.data_ptr<long>(), \
    query_scale.data_ptr<float>(), \
    dense_lse.data_ptr<float>(), \
    glob_code_output.data_ptr<float>(), \
    glob_den.data_ptr<float>(), \
    B, QH, KVH, S, L, LGroups, \
    query_scale.numel(), \
    (float)attention_scale, \
    (float)prune_log_eps \
  )
#define DISPATCH_PRUNED_FIXED43_LOWK() \
  do { \
    if (KBlocks == 2 && Cv == 128) { LAUNCH_PRUNED_FIXED43(2, 128); } \
    else if (KBlocks == 2 && Cv == 192) { LAUNCH_PRUNED_FIXED43(2, 192); } \
    else if (KBlocks == 2 && Cv == 256) { LAUNCH_PRUNED_FIXED43(2, 256); } \
    else if (KBlocks == 2 && Cv == 320) { LAUNCH_PRUNED_FIXED43(2, 320); } \
    else if (KBlocks == 2 && Cv == 384) { LAUNCH_PRUNED_FIXED43(2, 384); } \
    else if (KBlocks == 2 && Cv == 448) { LAUNCH_PRUNED_FIXED43(2, 448); } \
    else if (KBlocks == 2 && Cv == 512) { LAUNCH_PRUNED_FIXED43(2, 512); } \
    else if (KBlocks == 3 && Cv == 128) { LAUNCH_PRUNED_FIXED43(3, 128); } \
    else if (KBlocks == 3 && Cv == 192) { LAUNCH_PRUNED_FIXED43(3, 192); } \
    else if (KBlocks == 3 && Cv == 256) { LAUNCH_PRUNED_FIXED43(3, 256); } \
    else if (KBlocks == 3 && Cv == 320) { LAUNCH_PRUNED_FIXED43(3, 320); } \
    else if (KBlocks == 3 && Cv == 384) { LAUNCH_PRUNED_FIXED43(3, 384); } \
    else if (KBlocks == 3 && Cv == 448) { LAUNCH_PRUNED_FIXED43(3, 448); } \
    else if (KBlocks == 3 && Cv == 512) { LAUNCH_PRUNED_FIXED43(3, 512); } \
    else if (KBlocks == 4 && Cv == 128) { LAUNCH_PRUNED_FIXED43(4, 128); } \
    else if (KBlocks == 4 && Cv == 192) { LAUNCH_PRUNED_FIXED43(4, 192); } \
    else if (KBlocks == 4 && Cv == 320) { LAUNCH_PRUNED_FIXED43(4, 320); } \
    else if (KBlocks == 4 && Cv == 384) { LAUNCH_PRUNED_FIXED43(4, 384); } \
    else if (KBlocks == 4 && Cv == 448) { LAUNCH_PRUNED_FIXED43(4, 448); } \
    else if (KBlocks == 4 && Cv == 512) { LAUNCH_PRUNED_FIXED43(4, 512); } \
    else if (KBlocks == 5 && Cv == 128) { LAUNCH_PRUNED_FIXED43(5, 128); } \
    else if (KBlocks == 5 && Cv == 192) { LAUNCH_PRUNED_FIXED43(5, 192); } \
    else if (KBlocks == 5 && Cv == 256) { LAUNCH_PRUNED_FIXED43(5, 256); } \
    else if (KBlocks == 5 && Cv == 320) { LAUNCH_PRUNED_FIXED43(5, 320); } \
    else if (KBlocks == 5 && Cv == 384) { LAUNCH_PRUNED_FIXED43(5, 384); } \
    else if (KBlocks == 5 && Cv == 448) { LAUNCH_PRUNED_FIXED43(5, 448); } \
    else if (KBlocks == 5 && Cv == 512) { LAUNCH_PRUNED_FIXED43(5, 512); } \
    else if (KBlocks == 6 && Cv == 128) { LAUNCH_PRUNED_FIXED43(6, 128); } \
    else if (KBlocks == 6 && Cv == 192) { LAUNCH_PRUNED_FIXED43(6, 192); } \
    else if (KBlocks == 6 && Cv == 256) { LAUNCH_PRUNED_FIXED43(6, 256); } \
    else if (KBlocks == 6 && Cv == 320) { LAUNCH_PRUNED_FIXED43(6, 320); } \
    else if (KBlocks == 6 && Cv == 384) { LAUNCH_PRUNED_FIXED43(6, 384); } \
    else if (KBlocks == 6 && Cv == 448) { LAUNCH_PRUNED_FIXED43(6, 448); } \
    else if (KBlocks == 6 && Cv == 512) { LAUNCH_PRUNED_FIXED43(6, 512); } \
    else { LAUNCH_PRUNED_RUNTIME(4, true, 3); } \
  } while (0)
#define DISPATCH_PRUNED_MODE(EB, KEY_T) \
  do { \
    if (KBlocks == 4 && Cv == 256) { \
      if (prune_mode == 1) { LAUNCH_PRUNED(EB, KEY_T, 1); } \
      else if (prune_mode == 2) { LAUNCH_PRUNED(EB, KEY_T, 2); } \
      else { LAUNCH_PRUNED(EB, KEY_T, 3); } \
    } else if (KBlocks >= 5) { \
      if (prune_mode == 1) { LAUNCH_PRUNED_RUNTIME_UNROLLED(EB, KEY_T, 1); } \
      else if (prune_mode == 2) { LAUNCH_PRUNED_RUNTIME_UNROLLED(EB, KEY_T, 2); } \
      else { LAUNCH_PRUNED_RUNTIME_UNROLLED(EB, KEY_T, 3); } \
    } else { \
      if (prune_mode == 1) { LAUNCH_PRUNED_RUNTIME(EB, KEY_T, 1); } \
      else if (prune_mode == 2) { LAUNCH_PRUNED_RUNTIME(EB, KEY_T, 2); } \
      else { LAUNCH_PRUNED_RUNTIME(EB, KEY_T, 3); } \
    } \
  } while (0)
#define DISPATCH_PRUNED_EB(KEY_T) \
  do { \
    if (exp_bits == 1) { DISPATCH_PRUNED_MODE(1, KEY_T); } \
    else if (exp_bits == 2) { DISPATCH_PRUNED_MODE(2, KEY_T); } \
    else if (exp_bits == 4) { DISPATCH_PRUNED_MODE(4, KEY_T); } \
    else { DISPATCH_PRUNED_MODE(8, KEY_T); } \
  } while (0)

  if (key_transposed && exp_bits == 4 && prune_mode == 3 && (
        (KBlocks < 6 && !(KBlocks == 4 && Cv == 256)) ||
        KBlocks == 6)) {
    DISPATCH_PRUNED_FIXED43_LOWK();
  } else if (key_transposed) { DISPATCH_PRUNED_EB(true); }
  else { DISPATCH_PRUNED_EB(false); }

#undef DISPATCH_PRUNED_EB
#undef DISPATCH_PRUNED_MODE
#undef DISPATCH_PRUNED_FIXED43_LOWK
#undef LAUNCH_PRUNED_FIXED43
#undef LAUNCH_PRUNED_RUNTIME_UNROLLED
#undef LAUNCH_PRUNED_RUNTIME
#undef LAUNCH_PRUNED
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void picokv_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  int prune_mode,
  double prune_log_eps
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(packed_projected_query.dim() == 5, "packed_projected_query must have shape [B,QH,S,KBlocks,8]");
  TORCH_CHECK(packed_projected_query.scalar_type() == at::kLong, "packed_projected_query must be torch.long");
  TORCH_CHECK(packed_projected_query.is_cuda() && packed_projected_query.is_contiguous(), "packed_projected_query must be contiguous CUDA");
  TORCH_CHECK(query_scale.is_cuda() && query_scale.is_contiguous() && query_scale.scalar_type() == at::kFloat, "query_scale must be contiguous CUDA fp32");
  TORCH_CHECK(dense_lse.is_cuda() && dense_lse.is_contiguous() && dense_lse.scalar_type() == at::kFloat, "dense_lse must be contiguous CUDA fp32");
  TORCH_CHECK(dense_output.is_cuda() && dense_output.is_contiguous(), "dense_output must be contiguous CUDA");
  TORCH_CHECK(glob_code_output.is_cuda() && glob_code_output.is_contiguous() && glob_code_output.scalar_type() == at::kFloat, "glob_code_output must be contiguous CUDA fp32");
  TORCH_CHECK(glob_den.is_cuda() && glob_den.is_contiguous() && glob_den.scalar_type() == at::kFloat, "glob_den must be contiguous CUDA fp32");
  TORCH_CHECK(value_proj.is_cuda() && value_proj.is_contiguous(), "value_proj must be contiguous CUDA");
  TORCH_CHECK(attn_output.is_cuda() && attn_output.is_contiguous(), "attn_output must be contiguous CUDA");

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  TORCH_CHECK(packed_projected_query.size(4) == 8, "packed_projected_query last dim must be 8");
  TORCH_CHECK(query_scale.size(0) == B && query_scale.size(1) == QH && query_scale.size(2) == S, "query_scale shape mismatch");
  TORCH_CHECK(dense_lse.size(0) == B && dense_lse.size(1) == QH && dense_lse.size(2) == S, "dense_lse shape mismatch");
  TORCH_CHECK(dense_output.size(0) == B && dense_output.size(1) == QH && dense_output.size(2) == S, "dense_output leading shape mismatch");
  TORCH_CHECK(attn_output.sizes() == dense_output.sizes(), "attn_output shape must match dense_output");
  TORCH_CHECK(attn_output.scalar_type() == dense_output.scalar_type(), "attn_output dtype must match dense_output");
  TORCH_CHECK(value_proj.scalar_type() == dense_output.scalar_type(), "value_proj dtype must match dense_output");

  const size_t code_numel = (size_t)glob_code_output.numel();
  const size_t den_numel = (size_t)glob_den.numel();
  float *code_ptr = glob_code_output.data_ptr<float>();
  float *den_ptr = glob_den.data_ptr<float>();
  if (den_ptr == code_ptr + code_numel) {
    C10_CUDA_CHECK(cudaMemsetAsync(code_ptr, 0, (code_numel + den_numel) * sizeof(float), stream));
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(code_ptr, 0, code_numel * sizeof(float), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(den_ptr, 0, den_numel * sizeof(float), stream));
  }

  picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    glob_code_output,
    glob_den,
    exp_bits,
    tile_size,
    key_transposed,
    attention_scale,
    prune_mode,
    prune_log_eps
  );
  picokv_fused_output_projection(
    dense_output,
    glob_code_output,
    glob_den,
    value_proj,
    attn_output
  );
  (void)KBlocks;
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  int prune_mode,
  double prune_log_eps
) {
  at::cuda::CUDAGuard device_guard(projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int B = projected_query.size(0);
  const int QH = projected_query.size(1);
  const int S = projected_query.size(2);
  const int C = projected_query.size(3);
  TORCH_CHECK(C == 128 || C == 192 || C == 256 || C == 320 || C == 384 || C == 448 || C == 512, "pruned decode tail currently requires projected-query Ck in {128,192,256,320,384,448,512}");
  dim3 pack_blocks(B * QH * S);
  dim3 pack_threads(256);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, projected_query.scalar_type(), "picokv_quantize_pack_projected_query", [&] {
    picokv_quantize_pack_projected_query_kernel<scalar_t><<<pack_blocks, pack_threads, 0, stream>>>(
      (const scalar_t*)projected_query.data_ptr<scalar_t>(),
      packed_projected_query.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      B, QH, S, C
    );
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemsetAsync(glob_code_output.data_ptr<float>(), 0, glob_code_output.numel() * sizeof(float), stream);
  cudaMemsetAsync(glob_den.data_ptr<float>(), 0, glob_den.numel() * sizeof(float), stream);
  picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    glob_code_output,
    glob_den,
    exp_bits,
    tile_size,
    key_transposed,
    attention_scale,
    prune_mode,
    prune_log_eps
  );
  picokv_fused_output_projection(
    dense_output,
    glob_code_output,
    glob_den,
    value_proj,
    attn_output
  );
}


void picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
  at::cuda::CUDAGuard device_guard(packed_projected_query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); /* explicit_current_stream */

  const int B = packed_projected_query.size(0);
  const int QH = packed_projected_query.size(1);
  const int KVH = packed_key.size(1);
  const int S = packed_projected_query.size(2);
  const int KBlocks = packed_projected_query.size(3);
  const int L = key_transposed ? packed_key.size(3) : packed_key.size(2);
  const int LGroups = packed_value_sign.size(2);
  const int Cv = packed_value_sign.size(3);
  const int group_size = KVH == 0 ? 0 : QH / KVH;

  TORCH_CHECK(group_size == 8 && QH == group_size * KVH, "BMMA score prototype requires QH == 8 * KVH");
  TORCH_CHECK(KBlocks == 4 && Cv == 256, "BMMA score prototype requires Ck == 256 and Cv == 256");
  TORCH_CHECK(tile_size == kTileSize, "BMMA score prototype requires tile_size == 256");
  TORCH_CHECK(LGroups == (L + 63) / 64, "packed_value_sign must have shape [B,KVH,ceil(L/64),Cv]");
  TORCH_CHECK(exp_bits == 4, "BMMA score prototype currently supports exp_bits == 4");
  TORCH_CHECK(glob_code_output.scalar_type() == at::kFloat, "BMMA score glob_code_output must be float32");
  TORCH_CHECK(glob_den.scalar_type() == at::kFloat, "BMMA score glob_den must be float32");

  dim3 partial_threads(kTileSize);
  if (key_transposed) {
    picokv_v1_expbits_atomic2_gqa8_bmma_score_kernel<true, 4><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>(
      packed_projected_query.data_ptr<long>(),
      packed_key.data_ptr<long>(),
      packed_value_sign.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      dense_lse.data_ptr<float>(),
      glob_code_output.data_ptr<float>(),
      glob_den.data_ptr<float>(),
      B, QH, KVH, S, L, LGroups,
      query_scale.numel(),
      (float)attention_scale,
      nullptr
    );
  } else {
    picokv_v1_expbits_atomic2_gqa8_bmma_score_kernel<false, 4><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>(
      packed_projected_query.data_ptr<long>(),
      packed_key.data_ptr<long>(),
      packed_value_sign.data_ptr<long>(),
      query_scale.data_ptr<float>(),
      dense_lse.data_ptr<float>(),
      glob_code_output.data_ptr<float>(),
      glob_den.data_ptr<float>(),
      B, QH, KVH, S, L, LGroups,
      query_scale.numel(),
      (float)attention_scale,
      nullptr
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void picokv_decode_tail_v1_exp8_tiled_gqa4_atomic(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  at::Tensor global_lens,
  at::Tensor sched_req,
  at::Tensor sched_tile
) {
  at::cuda::CUDAGuard device_guard(projected_query.device());

  // Quantize+pack the projected query and zero both fp32 accumulators in ONE
  // launch (replaces the separate quantize kernel + cudaMemsetAsync pair).
  // Same zeros, same stream, still strictly before the main kernel.
  picokv_quantize_pack_zero_projected_query(
    projected_query,
    packed_projected_query,
    query_scale,
    glob_code_output,
    glob_den
  );

  picokv_global_code_output_v1_exp8_tiled_gqa4_atomic(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    glob_code_output,
    glob_den,
    exp_bits,
    tile_size,
    key_transposed,
    attention_scale,
    global_lens,
    sched_req,
    sched_tile
  );

  // Value projection: cublas GEMM path by default (bf16), scalar kernel as
  // fallback (PICOKV_PROJ_GEMM=0 or non-bf16 inputs). Same math up to fp32
  // accumulation order.
  static const bool use_proj_gemm = []() {
    const char *env = getenv("PICOKV_PROJ_GEMM");
    return env ? (atoi(env) != 0) : true;
  }();
  if (use_proj_gemm && dense_output.scalar_type() == at::kBFloat16) {
    picokv_fused_output_projection_gemm(
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output
    );
  } else {
    picokv_fused_output_projection(
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output
    );
  }
}


// Rows per window split-chunk block. PICOKV_WIN_CH in {32,48,64,96} forces
// one; unset/0 = auto by batch size. Small launches are latency-bound on the
// per-thread phase-4 V chain (CH/2 rows), so fewer rows per block win there;
// from ~128 (b, kv-head) pairs the grid is full and the merge cost dominates.
// Graph-replayed times on this box (us, cap 832, lpl 384) CH 96/48/32:
// b1 10.9/8.4/7.7, b4 12.9/11.1/10.7, b8 15.8/13.3/13.9, b16 21.7/22.9/23.3.
static inline int picokv_window_chunk_rows(int B, int KVH) {
  static int forced = []() {
    const char *env = getenv("PICOKV_WIN_CH");
    return env ? atoi(env) : 0;
  }();
  if (forced == 32 || forced == 48 || forced == 64 || forced == 96) return forced;
  const int pairs = B * KVH;
  if (pairs <= 32) return 32;
  if (pairs <= 64) return 48;
  return kWinChunkRows;
}

void picokv_window_attn(
  at::Tensor kbuf,
  at::Tensor vbuf,
  at::Tensor lpl,
  at::Tensor q,
  at::Tensor out,
  at::Tensor lse,
  double sm_scale
) {
  at::cuda::CUDAGuard device_guard(kbuf.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(kbuf.dim() == 4 && vbuf.dim() == 4, "window attn: kbuf/vbuf must be [B, cap, KVH, D]");
  TORCH_CHECK(q.dim() == 3, "window attn: q must be [B, QH, D]");
  TORCH_CHECK(kbuf.is_cuda() && vbuf.is_cuda() && lpl.is_cuda() && q.is_cuda(), "window attn: all inputs must be CUDA tensors");
  TORCH_CHECK(kbuf.is_contiguous() && vbuf.is_contiguous() && lpl.is_contiguous(), "window attn: kbuf/vbuf/lpl must be contiguous");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && lse.is_contiguous(), "window attn: q/out/lse must be contiguous");
  TORCH_CHECK(kbuf.scalar_type() == at::kBFloat16 && vbuf.scalar_type() == at::kBFloat16
              && q.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
              "window attn: kbuf/vbuf/q/out must be bf16");
  TORCH_CHECK(lpl.scalar_type() == at::kInt, "window attn: lpl must be int32");
  TORCH_CHECK(lse.scalar_type() == at::kFloat, "window attn: lse must be float32");

  const int B = kbuf.size(0);
  const int cap = kbuf.size(1);
  const int KVH = kbuf.size(2);
  const int D = kbuf.size(3);
  const int QH = q.size(1);
  TORCH_CHECK(vbuf.sizes() == kbuf.sizes(), "window attn: vbuf shape must match kbuf");
  TORCH_CHECK(q.size(0) == B && q.size(2) == D, "window attn: q must be [B, QH, D] with matching B/D");
  TORCH_CHECK(out.size(0) == B && out.size(1) == QH && out.size(2) == D, "window attn: out must be [B, QH, D]");
  TORCH_CHECK(lse.size(0) == B && lse.size(1) == QH, "window attn: lse must be [B, QH]");
  TORCH_CHECK(lpl.size(0) == B, "window attn: lpl must be [B]");
  TORCH_CHECK(D == 128, "window attn kernel currently supports head_dim 128");
  TORCH_CHECK(QH == KVH * 4, "window attn kernel currently supports GQA-4 (QH == 4*KVH)");

  // chunk split: smem holds only [4][CH] logits + stats + phase-4 partials
  // (<= 5.9 KB) — no 48 KB opt-in needed and cap is unbounded.
  const int ch = picokv_window_chunk_rows(B, KVH);
  const int CHUNKS = (cap + ch - 1) / ch;
  const size_t smem_bytes = (size_t)(4 * ch + 72 + 1024) * sizeof(float);
  auto fopts = q.options();
  at::Tensor acc = at::empty({(long)B * KVH * CHUNKS * 4 * D}, fopts.dtype(at::kFloat));
  at::Tensor ml = at::empty({(long)B * KVH * CHUNKS * 4 * 2}, fopts.dtype(at::kFloat));

  dim3 grid(B, KVH, CHUNKS);
#define PICOKV_WINDOW_LAUNCH(CHV) \
  do { \
    picokv_window_attn_kernel<4, 128, CHV><<<grid, 256, smem_bytes, stream>>>( \
        reinterpret_cast<const __nv_bfloat16 *>(kbuf.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const __nv_bfloat16 *>(vbuf.data_ptr<at::BFloat16>()), \
        lpl.data_ptr<int>(), \
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()), \
        acc.data_ptr<float>(), ml.data_ptr<float>(), \
        (float)sm_scale, cap, KVH, CHUNKS); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    picokv_window_attn_merge_kernel<CHV><<<dim3(B, KVH), 256, 0, stream>>>( \
        lpl.data_ptr<int>(), acc.data_ptr<float>(), ml.data_ptr<float>(), \
        reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), \
        lse.data_ptr<float>(), KVH, CHUNKS); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
  } while (0)
  if (ch == 32) PICOKV_WINDOW_LAUNCH(32);
  else if (ch == 48) PICOKV_WINDOW_LAUNCH(48);
  else if (ch == 64) PICOKV_WINDOW_LAUNCH(64);
  else PICOKV_WINDOW_LAUNCH(96);
#undef PICOKV_WINDOW_LAUNCH
}
