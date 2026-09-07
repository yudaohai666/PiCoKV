#include "cuda_kernel.h"

#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
template <int GroupSize, bool KeyTransposedFixed, int ExpBits, int PruneMode>
__global__ void __launch_bounds__(256, 6) picokv_v1_expbits_atomic2_runtime_unrolled_kernel(
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
  constexpr int MaxTaskSlots = GroupSize; // GroupSize * (512 / 2) / 256
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
    #pragma unroll
    for (int kb = 0; kb < MaxKBlocks; kb++) {
      if (kb < KBlocks) {
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
      for (int kb = 0; kb < MaxKBlocks; kb++) {
        if (kb < KBlocks) {
          key_words[kb] = load_packed_key_word(
            packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
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
        for (int kb = 0; kb < MaxKBlocks; kb++) {
          if (kb < KBlocks && !token_pruned) {
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
  __shared__ __align__(128) int bmma_tmp[QuantWarps][64];

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
      const int cell0 = lane_idx;
      const int cell1 = lane_idx + 32;
      const int h0 = cell0 >> 3;
      const int h1 = cell1 >> 3;
      const int col0 = cell0 & 7;
      const int col1 = cell1 & 7;
      int dot0 = query_base_dot[h0];
      int dot1 = query_base_dot[h1];

      #pragma unroll
      for (int chunk = 0; chunk < 2; chunk++) {
        const int b_idx = lane_idx;
        const int b_col = b_idx >> 2;
        const int b_seg = b_idx & 3;
        const int l_idx = token_base + b_col;
        unsigned int key_piece = 0;
        if (l_idx < L) {
          const int kb = chunk * 2 + (b_seg >> 1);
          const unsigned long long key_word = load_packed_key_word(
            packed_key, B_idx, KVH_idx, KVH, L, KBlocks, l_idx, kb, KeyTransposedFixed
          );
          key_piece = (unsigned int)((b_seg & 1) ? (key_word >> 32) : key_word);
        }
        bmma_b[warp_idx][b_idx] = key_piece;
        __syncwarp();

        #pragma unroll
        for (int bit = 0; bit < kQMagBits; bit++) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::row_major> a_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 128, nvcuda::wmma::experimental::precision::b1, nvcuda::wmma::col_major> b_frag;
          nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 128, int> c_frag;

          nvcuda::wmma::fill_fragment(c_frag, 0);
          nvcuda::wmma::load_matrix_sync(a_frag, &bmma_a_pos[chunk][bit][0], 128);
          nvcuda::wmma::load_matrix_sync(b_frag, &bmma_b[warp_idx][0], 128);
          nvcuda::wmma::bmma_sync(
            c_frag, a_frag, b_frag, c_frag,
            nvcuda::wmma::experimental::bmmaBitOpAND,
            nvcuda::wmma::experimental::bmmaAccumulateOpPOPC
          );
          nvcuda::wmma::store_matrix_sync(&bmma_tmp[warp_idx][0], c_frag, 8, nvcuda::wmma::mem_row_major);
          __syncwarp();
          const int pos0 = bmma_tmp[warp_idx][cell0];
          const int pos1 = bmma_tmp[warp_idx][cell1];

          nvcuda::wmma::fill_fragment(c_frag, 0);
          nvcuda::wmma::load_matrix_sync(a_frag, &bmma_a_neg[chunk][bit][0], 128);
          nvcuda::wmma::load_matrix_sync(b_frag, &bmma_b[warp_idx][0], 128);
          nvcuda::wmma::bmma_sync(
            c_frag, a_frag, b_frag, c_frag,
            nvcuda::wmma::experimental::bmmaBitOpAND,
            nvcuda::wmma::experimental::bmmaAccumulateOpPOPC
          );
          nvcuda::wmma::store_matrix_sync(&bmma_tmp[warp_idx][0], c_frag, 8, nvcuda::wmma::mem_row_major);
          __syncwarp();
          const int neg_key0 = bmma_tmp[warp_idx][cell0];
          const int neg_key1 = bmma_tmp[warp_idx][cell1];
          const int neg0 = pos0 + query_neg_count[h0][chunk][bit] - neg_key0;
          const int neg1 = pos1 + query_neg_count[h1][chunk][bit] - neg_key1;
          dot0 -= neg0 << (bit + 1);
          dot1 -= neg1 << (bit + 1);
        }
      }

      const int local0 = token_group * 8 + col0;
      const int local1 = token_group * 8 + col1;
      const int l0 = token_base + col0;
      const int l1 = token_base + col1;
      float out0 = 0.0f;
      float out1 = 0.0f;
      if constexpr (ExpBits <= 2) {
        out0 = -3.402823466e38f;
        out1 = -3.402823466e38f;
      }
      if (l0 < L) {
        const float score0 = (float)dot0 * scale_cache[h0];
        if constexpr (ExpBits <= 2) {
          out0 = score0;
        } else {
          out0 = picokv_fast_expf(score0 - lse_cache[h0]);
        }
      }
      if (l1 < L) {
        const float score1 = (float)dot1 * scale_cache[h1];
        if constexpr (ExpBits <= 2) {
          out1 = score1;
        } else {
          out1 = picokv_fast_expf(score1 - lse_cache[h1]);
        }
      }
      weights[h0][local0] = out0;
      weights[h1][local1] = out1;
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
#define LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, GS, KEY_T) \
  picokv_v1_expbits_atomic2_runtime_unrolled_kernel<GS, KEY_T, EB, 0><<<dim3((L + 511) / 512, S, B * KVH), partial_threads, 0, stream>>>( \
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
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 1, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 1, false); } \
    } else if (group_size == 1) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 1, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 1, false); } \
    } else if (group_size == 4 && KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC2(EB, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC2(EB, false); } \
    } else if (group_size == 4 && KBlocks >= 5) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 4, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 4, false); } \
    } else if (group_size == 4) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 4, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME(EB, 4, false); } \
    } else if (group_size == 8 && KBlocks == 4 && Cv == 256) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8(EB, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC2_GQA8(EB, false); } \
    } else if (group_size == 8 && KBlocks >= 5) { \
      if (key_transposed) { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 8, true); } \
      else { LAUNCH_V1_EXP_BITS_ATOMIC_RUNTIME_UNROLLED(EB, 8, false); } \
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
    (float)prune_log_eps \
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
      (float)attention_scale
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
      (float)attention_scale
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
  double attention_scale
) {
  at::cuda::CUDAGuard device_guard(projected_query.device());

  picokv_quantize_pack_projected_query(projected_query, packed_projected_query, query_scale);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(projected_query.device().index());
  float *code_ptr = glob_code_output.data_ptr<float>();
  float *den_ptr = glob_den.data_ptr<float>();
  const size_t code_numel = (size_t)glob_code_output.numel();
  const size_t den_numel = (size_t)glob_den.numel();
  if (den_ptr == code_ptr + code_numel) {
    C10_CUDA_CHECK(cudaMemsetAsync(
      code_ptr,
      0,
      (code_numel + den_numel) * sizeof(float),
      stream
    ));
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(
      code_ptr,
      0,
      code_numel * sizeof(float),
      stream
    ));
    C10_CUDA_CHECK(cudaMemsetAsync(
      den_ptr,
      0,
      den_numel * sizeof(float),
      stream
    ));
  }

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
    attention_scale
  );

  picokv_fused_output_projection(
    dense_output,
    glob_code_output,
    glob_den,
    value_proj,
    attn_output
  );
}
