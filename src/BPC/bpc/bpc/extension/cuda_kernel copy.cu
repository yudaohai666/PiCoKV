#include "cuda_kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
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

  float2 val = __bfloat1622float2(probe[((B_idx * H + H_idx) * S + S_idx) * 32 + warpth_idx]);
  float max_val = max(abs(val.x), abs(val.y));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 1));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 2));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 4));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 8));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 16));
  if (warpth_idx == 0) {
    cache[S_idx] = max_val;
  }
  __syncthreads();
  if (warpth_idx < S) {
    max_val = cache[warpth_idx];
  }
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 1));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 2));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 4));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 8));
  max_val = max(max_val, __shfl_xor_sync(0xffffffff, max_val, 16));

  int2 quant_val;
  quant_val.x = (int)round(val.x * 15 / max_val);
  quant_val.y = (int)round(val.y * 15 / max_val);
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
  long *pack_x,        // [B, H, S, 5]
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
  for (int idx = 0; idx < 64; idx++) {
    packed = packed | (((long)((cache[idx] >> thread_idx) & 1)) << idx);
  }
  if (thread_idx == 31) {
     pack_x[((B_idx * H + H_idx) * S + S_idx) * 5] = packed;
  } else if (thread_idx < 4) {
     pack_x[((B_idx * H + H_idx) * S + S_idx) * 5 + thread_idx + 1] = packed;
  }
   
}

__global__ void unpack_probe_hash64_kernel(
  int *unpack_x,       // [B, H, S, 64]
  long *pack_x,        // [B, H, S, 5]
  const int B,
  const int H,
  const int S
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int S_idx = blockIdx.x;

  long sign = pack_x[((B_idx * H + H_idx) * S + S_idx) * 5];
  long bit1 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 5 + 1];
  long bit2 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 5 + 2];
  long bit3 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 5 + 3];
  long bit4 = pack_x[((B_idx * H + H_idx) * S + S_idx) * 5 + 4];

  for (int idx = 0; idx < 64; idx++) {
    int val = ((bit1 >> idx) & 1) | (((bit2 >> idx) & 1) << 1) | (((bit3 >> idx) & 1) << 2) | (((bit4 >> idx) & 1) << 3);
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
  const long *probe,                 // [B, H, 4, 5]
  const long *hashcode,              // [B, H, L]
  short *scores,                     // [B, H, L]
  const int B,
  const int H,
  const int L
) {
  const int B_idx = blockIdx.z;
  const int H_idx = blockIdx.y;
  const int block_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  __shared__ long probe_cache[20];
  
  if (thread_idx < 20) {
    probe_cache[thread_idx] = probe[(B_idx * H + H_idx) * 4 * 5 + thread_idx];
  }
  long hashcode_val = hashcode[(B_idx * H + H_idx) * L + block_idx * 256 + thread_idx];
  __syncthreads();

  short max_val = -10000;
  #pragma unroll
  for (int S_idx = 0; S_idx < 4; S_idx++) {
    long signs = hashcode_val ^ probe_cache[S_idx * 5];
    short val = 0;
    #pragma unroll
    for (int bit_idx = 1; bit_idx < 5; bit_idx++) {
      long prob_val = probe_cache[S_idx * 5 + bit_idx];
      val = val + ((__popcll(prob_val & (~signs)) - __popcll(prob_val & signs)) << (bit_idx - 1));
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

  dim3 threads(256);
  dim3 blocks(L / 256, H, B);

  probe_hash64_group4_block256_kernel<<<blocks, threads>>>(
    (probe.data_ptr<long>()),
    (hashcode.data_ptr<long>()),
    (scores.data_ptr<short>()),
    B, H, L
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
