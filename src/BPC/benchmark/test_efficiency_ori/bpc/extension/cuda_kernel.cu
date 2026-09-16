#include "cuda_kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
using namespace std;

__global__ void quant_probe_hash64_kernel(
  const __nv_bfloat162 *probe,   // [B, H, S, 32]
  int2 *quant_probe,             // [B, H, S, 32]
  const int B,
  const int H,
  const int S
) {

  const int B_idx = blockIdx.y;
  const int H_idx = blockIdx.x;
  const int S_idx = threadIdx.y;
  const int warpth_idx = threadIdx.x;

  __shared__ float cache[32];
  __shared__ float global_max;

  float2 val = __bfloat1622float2(probe[((B_idx * H + H_idx) * S + S_idx) * 32 + warpth_idx]);
  float max_val = max(abs(val.x), abs(val.y));
  // warp-level max reduction
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 1));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 2));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 4));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 8));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 16));
  // store per-probe warp max
  if (warpth_idx == 0) {
    cache[S_idx] = max_val;
  }
  __syncthreads();

  // Global max across probes; done by thread 0 (avoids shuffle issue when S=1)
  if (S_idx == 0 && warpth_idx == 0) {
    float gmax = cache[0];
    for (int i = 1; i < S; i++) {
      gmax = max(gmax, cache[i]);
    }
    global_max = gmax;
  }
  __syncthreads();

  // Quantize using global max
  max_val = global_max;

  int2 quant_val;
  quant_val.x = (int)round(val.x * 127 / max_val);
  quant_val.y = (int)round(val.y * 127 / max_val);
  quant_probe[((B_idx * H + H_idx) * S + S_idx) * 32 + warpth_idx] = quant_val;
}

void quant_probe_hash64(
  at::Tensor probe, at::Tensor quant_probe
) {
  int B = probe.size(0);
  int H = probe.size(1);
  int S = probe.size(2);

  dim3 threads(32, S);
  dim3 blocks(H, B);

  quant_probe_hash64_kernel<<<blocks, threads>>>(
    (__nv_bfloat162*)(probe.data_ptr<at::BFloat16>()),
    (int2*)(quant_probe.data_ptr<int>()),
    B, H, S
  );
}


__global__ void pack_bits_hash64_kernel(
  int *unpack_x,          // [B, H, L, 64]
  long *pack_x,           // [B, H, L]
  const int B,
  const int H,
  const int L
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int token_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  __shared__ int cache[64];

  cache[thread_idx] = unpack_x[((B_idx * H + H_idx) * L + token_idx) * 64 + thread_idx] < 0;
  cache[32 + thread_idx] = unpack_x[((B_idx * H + H_idx) * L + token_idx) * 64 + 32 + thread_idx] < 0;

  if (thread_idx == 0) {
    long packed = 0;
    for (int idx = 0; idx < 64; idx++) {
      packed = packed | (((long)cache[idx]) << idx);
    }
    pack_x[(B_idx * H + H_idx) * L + token_idx] = packed;
  }
}

__global__ void unpack_bits_hash64_kernel(
  int *unpack_x,          // [B, H, L, 64]
  long *pack_x,           // [B, H, L]
  const int B,
  const int H,
  const int L
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int token_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  long val = pack_x[(B_idx * H + H_idx) * L + token_idx];
  int *unpack_x_pt = &unpack_x[((B_idx * H + H_idx) * L + token_idx) * 64];
  unpack_x_pt[thread_idx] = 1 - (((int)((val >> thread_idx) & 1)) << 1);
  unpack_x_pt[32 + thread_idx] = 1 - (((int)((val >> (32 + thread_idx)) & 1)) << 1);
}

void format_bits_hash64(
  at::Tensor unpack_x, at::Tensor pack_x, bool pack
) {
  int B = unpack_x.size(0);
  int H = unpack_x.size(1);
  int L = unpack_x.size(2);

  dim3 threads(32);
  dim3 blocks(L, H, B);

  if (pack) {
    pack_bits_hash64_kernel<<<blocks, threads>>>(
      unpack_x.data_ptr<int>(),
      pack_x.data_ptr<long>(),
      B, H, L
    );
  } else {
    unpack_bits_hash64_kernel<<<blocks, threads>>>(
      unpack_x.data_ptr<int>(),
      pack_x.data_ptr<long>(),
      B, H, L
    );
  }
}

__global__ void pack_probe_hash64_kernel(
  int *unpack_x,       // [B, H, S, 64]
  long *pack_x,        // [B, H, S, 8]
  const int B,
  const int H,
  const int S
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int S_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  __shared__ int cache[64];

  int val = unpack_x[((B_idx * H + H_idx) * S + S_idx) * 64 + thread_idx];
  cache[thread_idx] = (((val >> 31) & 1) << 31) | abs(val);

  val = unpack_x[((B_idx * H + H_idx) * S + S_idx) * 64 + 32 + thread_idx];
  cache[32 + thread_idx] = (((val >> 31) & 1) << 31) | abs(val);

  long packed = 0;

  #pragma unroll
  for (int idx = 0; idx < 64; idx++) {
    packed = packed | (((long)((cache[idx] >> thread_idx) & 1)) << idx);
  }
  if (thread_idx == 31) {
     pack_x[((B_idx * H + H_idx) * S + S_idx) * 8] = packed;
  } else if (thread_idx < 7) {
     pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + thread_idx + 1] = packed;
  }
   
}

__global__ void unpack_probe_hash64_kernel(
  int *unpack_x,       // [B, H, S, 64]
  long *pack_x,        // [B, H, S, 8]
  const int B,
  const int H,
  const int S
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int S_idx = blockIdx.x;

  long sign = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8];
  long bit1 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 1];
  long bit2 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 2];
  long bit3 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 3];
  long bit4 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 4];
  long bit5 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 5];
  long bit6 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 6];
  long bit7 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 8 + 7];

  #pragma unroll
  for (int idx = 0; idx < 64; idx++) {
    int val = ((bit1 >> idx) & 1) | (((bit2 >> idx) & 1) << 1) | (((bit3 >> idx) & 1) << 2) | (((bit4 >> idx) & 1) << 3);
    val = val | (((bit5 >> idx) & 1) << 4) | (((bit6 >> idx) & 1) << 5) | (((bit7 >> idx) & 1) << 6);
    if ((sign >> idx) & 1) {
      val = - val;
    }
    unpack_x[((B_idx * H + H_idx) * S + S_idx) * 64 + idx] = val;
  }
}

void format_probe_hash64(
  at::Tensor unpack_x, at::Tensor pack_x, bool pack
) {
  int B = unpack_x.size(0);
  int H = unpack_x.size(1);
  int S = unpack_x.size(2);

  dim3 threads(32);
  dim3 blocks(S, H, B);

  if (pack) {
    pack_probe_hash64_kernel<<<blocks, threads>>>(
      (unpack_x.data_ptr<int>()),
      (pack_x.data_ptr<long>()),
      B, H, S
    );
  } else {
    unpack_probe_hash64_kernel<<<blocks, threads>>>(
      (unpack_x.data_ptr<int>()),
      (pack_x.data_ptr<long>()),
      B, H, S
    );
  }
}

__global__ void binary_project_dim128_hash64_kernel(
  const __nv_bfloat162 *A,           // [B, H, L, 64]
  const __nv_bfloat162 *quan_proj,   // [B, H, 64, 64]
  long *y,                           // [B, H, L]
  const int B,
  const int H,
  const int L
) {

  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int warpth_idx = threadIdx.x;

  __shared__ __nv_bfloat162 A_cache[32 * 64];
  __shared__ __nv_bfloat162 quan_proj_cache[64];

  const __nv_bfloat162 *A_pt = &A[((B_idx * H + H_idx) * L + 32 * block_idx) * 64];
  const __nv_bfloat162 *quan_proj_pt = &quan_proj[(B_idx * H + H_idx) * 64 * 64];

  #pragma unroll
  for (int offset = 0; offset < 32 * 64; offset = offset + 32) {
    A_cache[offset + warpth_idx] = A_pt[offset + warpth_idx];
  }

  long y_val = 0;
  bool sign;
  __nv_bfloat162 direction;

  #pragma unroll
  for (int h_idx = 0; h_idx < 64; h_idx++) {
    __nv_bfloat162 val = __float2bfloat162_rn(0.0);
    quan_proj_cache[warpth_idx] = quan_proj_pt[h_idx * 64 + warpth_idx];
    quan_proj_cache[32 + warpth_idx] = quan_proj_pt[h_idx * 64 + 32 + warpth_idx];

    #pragma unroll
    for (int idx = 0; idx < 64; idx++) {
      val = __hadd2(__hmul2(quan_proj_cache[idx], A_cache[warpth_idx * 64 + idx]), val);
    }
    
    sign = __hlt(__hadd(val.x, val.y), __float2bfloat16(0.0));
    direction = __float2bfloat162_rn(((float)sign) * 2 - 1);

    #pragma unroll
    for (int idx = 0; idx < 64; idx++) {
      A_cache[warpth_idx * 64 + idx] += __hmul2(direction, quan_proj_cache[idx]);
    }

    y_val = y_val | (((long)sign) << h_idx);
  }

  long *y_pt = &y[(B_idx * H + H_idx) * L + 32 * block_idx];
  y_pt[warpth_idx] = y_val;
}


void binary_project_dim128_hash64(
  at::Tensor A, at::Tensor quan_proj, at::Tensor y
) {
  int B = A.size(0);
  int H = A.size(1);
  int L = A.size(2);

  dim3 threads(32);
  dim3 blocks(L / 32, H, B);

  binary_project_dim128_hash64_kernel<<<blocks, threads>>>(
    (__nv_bfloat162*)(A.data_ptr<at::BFloat16>()),
    (__nv_bfloat162*)(quan_proj.data_ptr<at::BFloat16>()),
    y.data_ptr<long>(),
    B, H, L
  );
}

__global__ void probe_hash64_group4_block256_kernel(
  const long *probe,                 // [B, H, S, 8]
  const long *hashcode,              // [B, H, L]
  short *scores,                     // [B, H, L]
  const int B,
  const int H,
  const int L,
  const int S                        // probe count (1 or 4)
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  __shared__ long probe_cache[32];   // up to 4 probes x 8 = 32
  __shared__ int probe_popc[28];     // up to 4 probes x 7 bits = 28

  // Load probe into shared memory
  const int probe_size = S * 8;
  if (thread_idx < probe_size) {
    probe_cache[thread_idx] = probe[(B_idx * H + H_idx) * S * 8 + thread_idx];
  }
  __syncthreads();

  // Precompute popcll(prob_val) for bits 1-7 of each probe
  const int popc_size = S * 7;
  if (thread_idx < popc_size) {
    int s = thread_idx / 7;           // probe index: 0 ~ S-1
    int b = thread_idx % 7 + 1;       // bit index: 1-7
    probe_popc[thread_idx] = __popcll(probe_cache[s * 8 + b]);
  }
  __syncthreads();

  long hashcode_val = hashcode[(B_idx * H + H_idx) * L + block_idx * 256 + thread_idx];

  short max_val = -10000;
  for (int S_idx = 0; S_idx < S; S_idx++) {
    long signs = hashcode_val ^ probe_cache[S_idx * 8];
    short val = 0;
    #pragma unroll
    for (int bit_idx = 1; bit_idx < 8; bit_idx++) {
      long prob_val = probe_cache[S_idx * 8 + bit_idx];
      // popc(A & ~B) - popc(A & B) = popc(A) - 2 * popc(A & B); popc(A) precomputed
      int popc_prob = probe_popc[S_idx * 7 + bit_idx - 1];
      int popc_and = __popcll(prob_val & signs);
      val = val + ((popc_prob - 2 * popc_and) << (bit_idx - 1));
    }
    max_val = max(max_val, val);
  }
  scores[(B_idx * H + H_idx) * L + block_idx * 256 + thread_idx] = max_val;
}


void probe_hash64_group4_block256(
  at::Tensor probe, at::Tensor hashcode, at::Tensor scores
) {
  int B = hashcode.size(0);
  int H = hashcode.size(1);
  int L = hashcode.size(2);
  int S = probe.size(2);  // variable probe count

  dim3 threads(256);
  dim3 blocks(L / 256, H, B);

  probe_hash64_group4_block256_kernel<<<blocks, threads>>>(
    (probe.data_ptr<long>()),
    (hashcode.data_ptr<long>()),
    (scores.data_ptr<short>()),
    B, H, L, S
  );
}

__global__ void topk_step1_block512_kernel(
  const short *scores,               // [B, H, L]
  int *counters,                     // [B, H, 32]
  const int B,
  const int H,
  const int L
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  __shared__ int counters_cache[32];
  
  if (thread_idx < 32) {
    counters_cache[thread_idx] = 0;
  }
  __syncthreads();

  short score_val = scores[(B_idx * H + H_idx) * L + block_idx * 512 + thread_idx];
  score_val = 32 - max(1, min(31, score_val));

  atomicAdd(&counters_cache[score_val], 1);
  __syncthreads();
  if (thread_idx < 32) {
    int val = counters_cache[thread_idx];

    val += __shfl_up_sync(0xffffffff, val, 1) * (thread_idx >= 1);
    val += __shfl_up_sync(0xffffffff, val, 2) * (thread_idx >= 2);
    val += __shfl_up_sync(0xffffffff, val, 4) * (thread_idx >= 4);
    val += __shfl_up_sync(0xffffffff, val, 8) * (thread_idx >= 8);
    val += __shfl_up_sync(0xffffffff, val, 16) * (thread_idx >= 16);

    atomicAdd(&counters[(B_idx * H + H_idx) * 32 + thread_idx], val);
  }
}

void topk_step1_block512(
  at::Tensor scores, at::Tensor counters
) {
  int B = scores.size(0);
  int H = scores.size(1);
  int L = scores.size(2);

  dim3 threads(512);
  dim3 blocks(L / 512, H, B);

  topk_step1_block512_kernel<<<blocks, threads>>>(
    (scores.data_ptr<short>()),
    (counters.data_ptr<int>()),
    B, H, L
  );
}

__global__ void topk_step2_kernel(
  const int *counters,    // [B, H, 32]
  int *counters_cumsum,   // [B, H, 32]
  const int B,
  const int H
) {
  const int B_idx = blockIdx.y;
  const int H_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  int val = counters[(B_idx * H + H_idx) * 32 + thread_idx];

  val += __shfl_up_sync(0xffffffff, val, 1) * (thread_idx >= 1);
  val += __shfl_up_sync(0xffffffff, val, 2) * (thread_idx >= 2);
  val += __shfl_up_sync(0xffffffff, val, 4) * (thread_idx >= 4);
  val += __shfl_up_sync(0xffffffff, val, 8) * (thread_idx >= 8);
  val += __shfl_up_sync(0xffffffff, val, 16) * (thread_idx >= 16);

  counters_cumsum[(B_idx * H + H_idx) * 32 + thread_idx] = val;
}

void topk_step2(
  at::Tensor counters, at::Tensor counters_cumsum
) {
  int B = counters.size(0);
  int H = counters.size(1);

  dim3 threads(32);
  dim3 blocks(H, B);

  topk_step2_kernel<<<blocks, threads>>>(
    (counters.data_ptr<int>()),
    (counters_cumsum.data_ptr<int>()),
    B, H
  );
}


__global__ void topk_step3_block512_kernel(
  const short *scores,         // [B, H, L]
  int *counters_cumsum,        // [B, H, 32]
  int *indices,                // [B, H, k]
  const int B,
  const int H,
  const int L,
  const int k
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  __shared__ int cache[32];
  
  if (thread_idx < 32) {
    cache[thread_idx] = counters_cumsum[(B_idx * H + H_idx) * 32 + thread_idx];
  }
  __syncthreads();

  short score_val = scores[(B_idx * H + H_idx) * L + block_idx * 512 + thread_idx];
  score_val = 32 - max(1, min(31, score_val));
  
  int index = atomicAdd(&cache[score_val - 1], 1);
  if (index < k) {
    index = atomicAdd(&counters_cumsum[(B_idx * H + H_idx) * 32 + score_val - 1], 1);
    if (index < k) {
      indices[(B_idx * H + H_idx) * k + index] = block_idx * 512 + thread_idx;
    }
  }
}

void topk_step3_block512(
  at::Tensor scores, at::Tensor counters_cumsum, at::Tensor indices
) {
  int B = scores.size(0);
  int H = scores.size(1);
  int L = scores.size(2);
  int k = indices.size(2);

  dim3 threads(512);
  dim3 blocks(L / 512, H, B);

  topk_step3_block512_kernel<<<blocks, threads>>>(
    (scores.data_ptr<short>()),
    (counters_cumsum.data_ptr<int>()),
    (indices.data_ptr<int>()),
    B, H, L, k
  );
}

// Fused kernel: quant_probe + pack_probe + probe_hash64 (with boundary handling).
// Reduces kernel-launch overhead and intermediate memory traffic.
// Outputs scores; topk is left to PyTorch. S can be 1 (MHA) or 4 (GQA).

__global__ void fused_quant_pack_probe_hash64_kernel(
  const __nv_bfloat162 *probe,   // [B, H, S, 32] bfloat162 (i.e. [B,H,S,64] bf16)
  const long *hashcode,          // [B, H, L]
  short *scores,                 // [B, H, L]
  const int B,
  const int H,
  const int L,
  const int S,                   // probe count (1 or 4)
  const int T,                   // effective seq length
  const int hash_T               // hash-valid length
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  const int global_idx = block_idx * 256 + thread_idx;

  // Step 1: quantize and pack probe in shared memory; supports S=1 or S=4

  __shared__ int quant_cache[4][64];     // quantized probe [S, D]
  __shared__ long packed_probe[32];      // packed probe [S * 8]
  __shared__ int probe_popc[28];         // precomputed popcll
  __shared__ float max_vals[4];          // per-probe warp max
  __shared__ float global_max_val;       // global max across all S probes

  // Step 1a: quantize (first S*32 threads)
  const int quant_threads = S * 32;
  if (thread_idx < quant_threads) {
    int S_idx = thread_idx / 32;  // 0 ~ S-1
    int d_idx = thread_idx % 32;  // 0-31

    // bfloat162 packs 2 bf16 values
    float2 val = __bfloat1622float2(probe[((B_idx * H + H_idx) * S + S_idx) * 32 + d_idx]);

    // warp-level max
    float max_val = max(abs(val.x), abs(val.y));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 1));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 2));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 4));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 8));
    max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 16));

    if (d_idx == 0) {
      max_vals[S_idx] = max_val;
    }
  }
  __syncthreads();

  // Step 1a-2: cross-probe global max (S=1 reduces to a single value)
  if (thread_idx == 0) {
    float max_val = max_vals[0];
    for (int i = 1; i < S; i++) {
      max_val = max(max_val, max_vals[i]);
    }
    global_max_val = max_val;
  }
  __syncthreads();

  // Step 1a-3: quantize using global max (7-bit signed)
  if (thread_idx < quant_threads) {
    int S_idx = thread_idx / 32;
    int d_idx = thread_idx % 32;

    float2 val = __bfloat1622float2(probe[((B_idx * H + H_idx) * S + S_idx) * 32 + d_idx]);
    float max_val = global_max_val;

    int quant_x = (int)roundf(val.x * 127.0f / max_val);
    int quant_y = (int)roundf(val.y * 127.0f / max_val);

    // Pack format: sign in bit 31, magnitude in low bits
    quant_cache[S_idx][d_idx * 2] = (((quant_x >> 31) & 1) << 31) | abs(quant_x);
    quant_cache[S_idx][d_idx * 2 + 1] = (((quant_y >> 31) & 1) << 31) | abs(quant_y);
  }
  __syncthreads();

  // Step 1b: pack probe bits (first S*8 threads)
  const int pack_threads = S * 8;
  if (thread_idx < pack_threads) {
    int S_idx = thread_idx / 8;   // 0 ~ S-1
    int bit_idx = thread_idx % 8; // 0-7

    long packed = 0;
    #pragma unroll
    for (int d_idx = 0; d_idx < 64; d_idx++) {
      int val = quant_cache[S_idx][d_idx];
      int bit;
      if (bit_idx == 0) {
        // sign bit
        bit = (val >> 31) & 1;
      } else {
        // magnitude bit (bit_idx - 1)
        bit = (val >> (bit_idx - 1)) & 1;
      }
      packed |= ((long)bit) << d_idx;
    }
    packed_probe[S_idx * 8 + bit_idx] = packed;
  }
  __syncthreads();

  // Step 1c: precompute popcll (first S*7 threads)
  const int popc_threads = S * 7;
  if (thread_idx < popc_threads) {
    int s = thread_idx / 7;           // probe index: 0 ~ S-1
    int b = thread_idx % 7 + 1;       // bit index: 1-7
    probe_popc[thread_idx] = __popcll(packed_probe[s * 8 + b]);
  }
  __syncthreads();

  // Step 2: compute hash score (boundary handled inline to avoid extra launch)
  short final_score;

  if (global_idx >= T) {
    // beyond effective length: never selected
    final_score = -10000;
  } else if (global_idx >= hash_T) {
    // newly added tokens in [hash_T, T): force-selected
    final_score = 10000;
  } else {
    long hashcode_val = hashcode[(B_idx * H + H_idx) * L + global_idx];
    
    short max_score = -10000;
    for (int S_idx = 0; S_idx < S; S_idx++) {
      long signs = hashcode_val ^ packed_probe[S_idx * 8];
      short val = 0;
      #pragma unroll
      for (int bit_idx = 1; bit_idx < 8; bit_idx++) {
        long prob_val = packed_probe[S_idx * 8 + bit_idx];
        int popc_prob = probe_popc[S_idx * 7 + bit_idx - 1];
        int popc_and = __popcll(prob_val & signs);
        val = val + ((popc_prob - 2 * popc_and) << (bit_idx - 1));
      }
      max_score = max(max_score, val);
    }
    final_score = max_score;
  }
  
  scores[(B_idx * H + H_idx) * L + global_idx] = final_score;
}

// C++ entry point
void fused_quant_pack_probe_hash64(
  at::Tensor probe,      // [B, H, S, 64] bf16
  at::Tensor hashcode,   // [B, H, L] long
  at::Tensor scores,     // [B, H, L] short
  int T,                 // effective seq length
  int hash_T             // hash-valid length
) {
  int B = hashcode.size(0);
  int H = hashcode.size(1);
  int L = hashcode.size(2);
  int S = probe.size(2);  // variable probe count

  dim3 threads(256);
  dim3 blocks(L / 256, H, B);

  fused_quant_pack_probe_hash64_kernel<<<blocks, threads>>>(
    (__nv_bfloat162*)(probe.data_ptr<at::BFloat16>()),
    hashcode.data_ptr<long>(),
    scores.data_ptr<short>(),
    B, H, L, S, T, hash_T
  );
}

// Fused gather KV: combines key/value gathers into one kernel and one index load.

__global__ void fused_gather_kv_kernel(
  const __nv_bfloat162 *keys,      // [B, H, L, D/2] (bfloat162)
  const __nv_bfloat162 *values,    // [B, H, L, D/2] (bfloat162)
  const long *indices,             // [B, H, k] (int64)
  __nv_bfloat162 *selected_keys,   // [B, H, k, D/2]
  __nv_bfloat162 *selected_values, // [B, H, k, D/2]
  const int B,
  const int H,
  const int L,
  const int k,
  const int D_half                 // D/2 (uses bfloat162)
) {
  // Each block handles all k indices for one (B, H);
  // each thread handles one (k_idx, d_idx) position.
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;  // chunk of k
  const int thread_idx = threadIdx.x;

  // Host guarantees blockDim.x >= D_half
  const int k_per_block = blockDim.x / D_half;
  const int local_k_idx = thread_idx / D_half;
  const int d_idx = thread_idx % D_half;
  const int k_idx = block_idx * k_per_block + local_k_idx;

  // Cache indices in shared memory to avoid duplicate loads
  extern __shared__ long shared_indices[];
  if (d_idx == 0 && k_idx < k) {
    shared_indices[local_k_idx] = indices[(B_idx * H + H_idx) * k + k_idx];
  }
  __syncthreads();

  if (k_idx >= k) return;

  long src_idx = shared_indices[local_k_idx];

  long src_offset = ((long)(B_idx * H + H_idx) * L + src_idx) * D_half + d_idx;
  long dst_offset = ((long)(B_idx * H + H_idx) * k + k_idx) * D_half + d_idx;

  selected_keys[dst_offset] = keys[src_offset];
  selected_values[dst_offset] = values[src_offset];
}

void fused_gather_kv(
  at::Tensor keys,           // [B, H, L, D] bf16
  at::Tensor values,         // [B, H, L, D] bf16
  at::Tensor indices,        // [B, H, k] int64
  at::Tensor selected_keys,  // [B, H, k, D] bf16
  at::Tensor selected_values // [B, H, k, D] bf16
) {
  // Ensure correct GPU context (multi-GPU)
  at::cuda::CUDAGuard device_guard(keys.device());

  int B = keys.size(0);
  int H = keys.size(1);
  int L = keys.size(2);
  int D = keys.size(3);
  int k = indices.size(2);
  int D_half = D / 2;  // bfloat162 packs 2 bf16

  // Pick threads_per_block / k_per_block based on D so each block covers
  // at least one full k row of D_half elements.
  int threads_per_block;
  int k_per_block;

  if (D_half <= 256) {
    // Common case D <= 512: a block can cover multiple k rows
    threads_per_block = 256;
    k_per_block = threads_per_block / D_half;
  } else {
    // Large D: one k per block, max 1024 threads (CUDA limit)
    threads_per_block = D_half;
    k_per_block = 1;
  }

  int num_k_blocks = (k + k_per_block - 1) / k_per_block;

  // shared memory: k_per_block int64 indices per block
  int shared_mem_size = k_per_block * sizeof(long);
  
  dim3 threads(threads_per_block);
  dim3 blocks(num_k_blocks, H, B);
  
  fused_gather_kv_kernel<<<blocks, threads, shared_mem_size>>>(
    (__nv_bfloat162*)(keys.data_ptr<at::BFloat16>()),
    (__nv_bfloat162*)(values.data_ptr<at::BFloat16>()),
    indices.data_ptr<long>(),
    (__nv_bfloat162*)(selected_keys.data_ptr<at::BFloat16>()),
    (__nv_bfloat162*)(selected_values.data_ptr<at::BFloat16>()),
    B, H, L, k, D_half
  );
}
