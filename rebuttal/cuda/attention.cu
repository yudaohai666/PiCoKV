#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>

namespace {
namespace wmma = nvcuda::wmma;
using bf16 = __nv_bfloat16;
using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
using ARow = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
using BRow = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major>;
using BCol = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major>;

__device__ __forceinline__ bf16 sign_bf16(unsigned negative) {
  return __ushort_as_bfloat16(static_cast<unsigned short>(0x3f80U | (negative << 15)));
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// No full BF16 code tensor or score tensor is written to global memory.
// One CTA owns four GQA query heads and BT*BlockTiles compressed tokens.
template<int BT, int Warps, int BlockTiles>
__global__ void unpacked_bf16_attention_kernel(
    const bf16* __restrict__ qp, const uint64_t* __restrict__ pk,
    const uint64_t* __restrict__ pv, const float* __restrict__ lse,
    float* __restrict__ output, float* __restrict__ den,
    int length, int ck, int cv, float scale) {
  constexpr int Threads = Warps * 32;
  constexpr int TS = BT + 8;  // 16-byte aligned, padding reduces bank conflicts.
  constexpr int ScoreTiles = (BT / 16 + Warps - 1) / Warps;
  __shared__ __align__(32) bf16 query[16 * 72];
  __shared__ __align__(32) bf16 code[64 * TS > BT * 72 ? 64 * TS : BT * 72];
  __shared__ __align__(32) bf16 weight[16 * TS];
  __shared__ __align__(32) float scratch[16 * (BT > 64 ? BT : 64)];
  const int row = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int groups = (length + 63) / 64;
  const int first = blockIdx.x * BT * BlockTiles;
  // Accumulate only this CTA's small output; only global atomics cross CTAs.
  for (int tile = 0; tile < BlockTiles; ++tile) {
    const int start = first + tile * BT;
    Acc score_acc[ScoreTiles];
    #pragma unroll
    for (int j = 0; j < ScoreTiles; ++j) wmma::fill_fragment(score_acc[j], 0.0f);
    for (int cb = 0; cb < ck / 64; ++cb) {
      for (int i = threadIdx.x; i < 16 * 64; i += Threads) {
        const int h = i / 64, c = i % 64;
        query[h * 72 + c] = h < 4 ? qp[(row * 4 + h) * ck + cb * 64 + c] : sign_bf16(0);
        if (h >= 4) query[h * 72 + c] = __float2bfloat16_rn(0.0f);
      }
      // K is packed along code dimensions, contiguous along tokens.
      for (int i = threadIdx.x; i < 64 * BT; i += Threads) {
        const int c = i / BT, t = i % BT;
        const int token = start + t;
        const uint64_t word = token < length ? pk[(int64_t(row) * (ck / 64) + cb) * length + token] : 0;
        code[c * TS + t] = token < length ? sign_bf16((word >> c) & 1ULL) : __float2bfloat16_rn(0.0f);
      }
      __syncthreads();
      #pragma unroll
      for (int j = 0; j < ScoreTiles; ++j) {
        const int n = (warp + j * Warps) * 16;
        if (n < BT) {
          #pragma unroll
          for (int k = 0; k < 64; k += 16) {
            ARow a; BRow b;
            wmma::load_matrix_sync(a, query + k, 72);
            wmma::load_matrix_sync(b, code + k * TS + n, TS);
            wmma::mma_sync(score_acc[j], a, b, score_acc[j]);
          }
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < ScoreTiles; ++j) {
      const int n = (warp + j * Warps) * 16;
      if (n < BT) wmma::store_matrix_sync(scratch + n, score_acc[j], BT, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * BT; i += Threads) {
      const int h = i / BT, t = i % BT;
      const float score = __bfloat162float(__float2bfloat16_rn(scratch[i])) * scale;
      const float w = h < 4 && start + t < length ? expf(score - lse[row * 4 + h]) : 0.0f;
      weight[h * TS + t] = __float2bfloat16_rn(w);
    }
    __syncthreads();
    if (warp < 4) {
      float sum = 0.0f;
      for (int t = lane; t < BT; t += 32) sum += __bfloat162float(weight[warp * TS + t]);
      sum = warp_sum(sum);
      if (!lane) atomicAdd(den + row * 4 + warp, sum);
    }
    // A 64-code staging window avoids storing all CV BF16 directions on chip.
    for (int cb = 0; cb < cv; cb += 64) {
      for (int i = threadIdx.x; i < BT * 64; i += Threads) {
        const int t = i / 64, c = i % 64;
        const int token = start + t;
        const uint64_t word = token < length && cb + c < cv ?
            pv[(int64_t(row) * groups + token / 64) * cv + cb + c] : 0;
        code[t * 72 + c] = token < length && cb + c < cv ?
            sign_bf16(1U ^ ((word >> (token % 64)) & 1ULL)) : __float2bfloat16_rn(0.0f);
      }
      __syncthreads();
      if (warp < 4) {
        Acc acc; wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int k = 0; k < BT; k += 16) {
          ARow a; BRow b;
          wmma::load_matrix_sync(a, weight + k, TS);
          wmma::load_matrix_sync(b, code + k * 72 + warp * 16, 72);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(scratch + warp * 16, acc, 64, wmma::mem_row_major);
      }
      __syncthreads();
      for (int i = threadIdx.x; i < 4 * 64; i += Threads) {
        const int h = i / 64, c = i % 64;
        if (cb + c < cv) atomicAdd(output + (row * 4 + h) * cv + cb + c, scratch[i]);
      }
      __syncthreads();
    }
  }
}

template<int BT, int Warps>
__device__ __forceinline__ void reconstruct_tile(
    const uint64_t* pk, const uint64_t* pv, const bf16* decoder,
    bf16* stage, bf16* restored, int row, int start, int length, int bits, bool is_key) {
  constexpr int Threads = Warps * 32;
  constexpr int NTiles = BT / 16 * 8;
  constexpr int PerWarp = (NTiles + Warps - 1) / Warps;
  bf16* codes = stage;
  bf16* d = stage + BT * 72;
  const int warp = threadIdx.x / 32;
  Acc acc[PerWarp];
  #pragma unroll
  for (int j = 0; j < PerWarp; ++j) wmma::fill_fragment(acc[j], 0.0f);
  for (int cb = 0; cb < bits; cb += 64) {
    // Each lane emits a BF16 pair directly from packed sign bits.
    for (int i = threadIdx.x; i < BT * 32; i += Threads) {
      const int t = i / 32, c = (i % 32) * 2;
      const int token = start + t;
      unsigned n0 = 0, n1 = 0;
      if (token < length) {
        if (is_key) {
          const uint64_t word = pk[(int64_t(row) * (bits / 64) + cb / 64) * length + token];
          const unsigned pair = (word >> c) & 3ULL;
          n0 = pair & 1; n1 = pair >> 1;
        } else {
          const int groups = (length + 63) / 64;
          const auto* words = pv + (int64_t(row) * groups + token / 64) * bits + cb + c;
          n0 = 1U ^ ((words[0] >> (token % 64)) & 1ULL);
          n1 = 1U ^ ((words[1] >> (token % 64)) & 1ULL);
        }
      }
      *reinterpret_cast<uint32_t*>(codes + t * 72 + c) = token < length ?
          0x3f803f80U | (n0 << 15) | (n1 << 31) : 0;
    }
    for (int i = threadIdx.x; i < 64 * 16; i += Threads) {
      const int c = i / 16, dim = (i % 16) * 8;
      *reinterpret_cast<uint4*>(d + c * 136 + dim) =
          *reinterpret_cast<const uint4*>(decoder + (int64_t(row) * bits + cb + c) * 128 + dim);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < PerWarp; ++j) {
      const int task = warp + j * Warps;
      const int m = (task / 8) * 16, n = (task % 8) * 16;
      if (task < NTiles) {
        #pragma unroll
        for (int k = 0; k < 64; k += 16) {
          ARow a; BRow b;
          wmma::load_matrix_sync(a, codes + m * 72 + k, 72);
          wmma::load_matrix_sync(b, d + k * 136 + n, 136);
          wmma::mma_sync(acc[j], a, b, acc[j]);
        }
      }
    }
    __syncthreads();
  }
  auto* fp32 = reinterpret_cast<float*>(stage);
  #pragma unroll
  for (int j = 0; j < PerWarp; ++j) {
    const int task = warp + j * Warps;
    if (task < NTiles)
      wmma::store_matrix_sync(fp32 + (task / 8) * 16 * 128 + (task % 8) * 16,
                              acc[j], 128, wmma::mem_row_major);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < BT * 128; i += Threads)
    restored[(i / 128) * 136 + i % 128] = __float2bfloat16_rn(fp32[i]);
  __syncthreads();
}

template<int BT, int Warps, int BlockTiles>
__global__ void reconstructed_bf16_attention_kernel(
    const bf16* __restrict__ q, const uint64_t* __restrict__ pk,
    const uint64_t* __restrict__ pv, const bf16* __restrict__ dk,
    const bf16* __restrict__ dv, const float* __restrict__ lse,
    float* __restrict__ output, float* __restrict__ den,
    int length, int ck, int cv, float scale) {
  constexpr int Threads = Warps * 32;
  constexpr int TS = BT + 8;
  constexpr int StageBytes = (BT * 72 + 64 * 136) * 2 > BT * 128 * 4 ?
      (BT * 72 + 64 * 136) * 2 : BT * 128 * 4;
  extern __shared__ __align__(32) unsigned char storage[];
  auto* stage = reinterpret_cast<bf16*>(storage);
  auto* kv = reinterpret_cast<bf16*>(storage + StageBytes);
  auto* query = kv + BT * 136;
  auto* weight = query + 16 * 136;
  auto* scratch = reinterpret_cast<float*>(stage);
  const int row = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  constexpr int ScoreTiles = (BT / 16 + Warps - 1) / Warps;
  for (int tile = 0; tile < BlockTiles; ++tile) {
    const int start = (blockIdx.x * BlockTiles + tile) * BT;
    reconstruct_tile<BT, Warps>(pk, pv, dk, stage, kv, row, start, length, ck, true);
    for (int i = threadIdx.x; i < 16 * 128; i += Threads) {
      const int h = i / 128, dim = i % 128;
      query[h * 136 + dim] = h < 4 ? q[(row * 4 + h) * 128 + dim] : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < ScoreTiles; ++j) {
      const int n = (warp + j * Warps) * 16;
      if (n < BT) {
        Acc acc; wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int k = 0; k < 128; k += 16) {
          ARow a; BCol b;
          wmma::load_matrix_sync(a, query + k, 136);
          wmma::load_matrix_sync(b, kv + n * 136 + k, 136);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(scratch + n, acc, BT, wmma::mem_row_major);
      }
    }
    __syncthreads();
    // Preserve the original full-reconstruction control's FP32 denominator:
    // only the PV Tensor Core operand is rounded to BF16.
    if (warp < 4) {
      float sum = 0.0f;
      for (int t = lane; t < BT; t += 32) {
        const float w = start + t < length ? expf(scratch[warp * BT + t] * scale - lse[row * 4 + warp]) : 0.0f;
        sum += w;
      }
      sum = warp_sum(sum);
      if (!lane) atomicAdd(den + row * 4 + warp, sum);
    }
    for (int i = threadIdx.x; i < 16 * BT; i += Threads) {
      const int h = i / BT, t = i % BT;
      const float w = h < 4 && start + t < length ? expf(scratch[i] * scale - lse[row * 4 + h]) : 0.0f;
      weight[h * TS + t] = __float2bfloat16_rn(w);
    }
    __syncthreads();
    reconstruct_tile<BT, Warps>(pk, pv, dv, stage, kv, row, start, length, cv, false);
    #pragma unroll
    for (int j = 0; j < (8 + Warps - 1) / Warps; ++j) {
      const int n = (warp + j * Warps) * 16;
      if (n < 128) {
        Acc acc; wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int k = 0; k < BT; k += 16) {
          ARow a; BRow b;
          wmma::load_matrix_sync(a, weight + k, TS);
          wmma::load_matrix_sync(b, kv + k * 136 + n, 136);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(scratch + n, acc, 128, wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4 * 128; i += Threads)
      atomicAdd(output + row * 4 * 128 + i, scratch[i]);
    __syncthreads();
  }
}

#include "attention_ptx.cuh"

template<int BT, int Warps, int BlockTiles, int CodeBits=0>
void launch(bool reconstruct, bool ptx, const at::Tensor& query, const at::Tensor& pk, const at::Tensor& pv,
            const at::Tensor& dk, const at::Tensor& dv, const at::Tensor& lse,
            at::Tensor& out, at::Tensor& den, float scale) {
  const int length = pk.size(3), ck = pk.size(2) * 64, cv = pv.size(3);
  const dim3 grid((length + BT * BlockTiles - 1) / (BT * BlockTiles), pk.size(0) * pk.size(1));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* q = reinterpret_cast<const bf16*>(query.data_ptr<at::BFloat16>());
  const auto* k = reinterpret_cast<const uint64_t*>(pk.data_ptr<int64_t>());
  const auto* v = reinterpret_cast<const uint64_t*>(pv.data_ptr<int64_t>());
  if (ptx && reconstruct) {
    constexpr int TS=BT+8;
    constexpr int KVSize=128*TS>BT*136?128*TS:BT*136;
    constexpr int SharedBytes=(128*72+KVSize+16*TS)*2;
    if(SharedBytes>48*1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(reconstructed_bf16_attention_kernel_ptx<BT,Warps,BlockTiles,CodeBits>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
    reconstructed_bf16_attention_kernel_ptx<BT,Warps,BlockTiles,CodeBits><<<grid,Warps*32,SharedBytes,stream>>>(
        q,k,v,reinterpret_cast<const bf16*>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(dv.data_ptr<at::BFloat16>()),lse.data_ptr<float>(),
        out.data_ptr<float>(),den.data_ptr<float>(),length,ck,cv,scale);
  } else if(ptx) {
    unpacked_bf16_attention_kernel_ptx<BT,Warps,BlockTiles,CodeBits><<<grid,Warps*32,0,stream>>>(
        q,k,v,lse.data_ptr<float>(),out.data_ptr<float>(),den.data_ptr<float>(),length,ck,cv,scale);
  } else if (reconstruct) {
    constexpr int StageBytes = (BT * 72 + 64 * 136) * 2 > BT * 128 * 4 ? (BT * 72 + 64 * 136) * 2 : BT * 128 * 4;
    constexpr int SharedBytes = StageBytes + (BT * 136 + 16 * 136 + 16 * (BT + 8)) * 2;
    if (SharedBytes > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(reconstructed_bf16_attention_kernel<BT, Warps, BlockTiles>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
    reconstructed_bf16_attention_kernel<BT, Warps, BlockTiles><<<grid, Warps * 32, SharedBytes, stream>>>(
        q, k, v, reinterpret_cast<const bf16*>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(dv.data_ptr<at::BFloat16>()), lse.data_ptr<float>(),
        out.data_ptr<float>(), den.data_ptr<float>(), length, ck, cv, scale);
  } else {
    unpacked_bf16_attention_kernel<BT, Warps, BlockTiles><<<grid, Warps * 32, 0, stream>>>(
        q, k, v, lse.data_ptr<float>(), out.data_ptr<float>(), den.data_ptr<float>(), length, ck, cv, scale);
  }
}
template<int BT,int Warps>
void launch_packed_decoder(const at::Tensor& query,const at::Tensor& pk,const at::Tensor& pv,
    const at::Tensor& dk,const at::Tensor& dv,const at::Tensor& lse,at::Tensor& out,at::Tensor& den,float scale) {
  constexpr int KVSize=128*(BT+8)>BT*136?128*(BT+8):BT*136;
  constexpr int SharedBytes=(16*136+KVSize+16*(BT+8))*2;
  const int length=pk.size(3),ck=pk.size(2)*64,cv=pv.size(3);
  const dim3 grid((length+BT-1)/BT,pk.size(0)*pk.size(1));
  reconstructed_bf16_attention_kernel_ptx<BT,Warps,1,0,true><<<grid,Warps*32,SharedBytes,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const bf16*>(query.data_ptr<at::BFloat16>()),
    reinterpret_cast<const uint64_t*>(pk.data_ptr<int64_t>()),reinterpret_cast<const uint64_t*>(pv.data_ptr<int64_t>()),
    reinterpret_cast<const bf16*>(dk.data_ptr<at::BFloat16>()),reinterpret_cast<const bf16*>(dv.data_ptr<at::BFloat16>()),
    lse.data_ptr<float>(),out.data_ptr<float>(),den.data_ptr<float>(),length,ck,cv,scale);
}
} // namespace

void bf16_code_attention_impl(
    at::Tensor query, at::Tensor pk, at::Tensor pv, at::Tensor dk, at::Tensor dv,
    at::Tensor lse, at::Tensor output, at::Tensor den, double scale, bool reconstruct,
    int64_t tile, int64_t warps, int64_t block_tiles, bool ptx) {
  for (const auto& t : {query, pk, pv, dk, dv, lse, output, den}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "inputs must be contiguous CUDA tensors");
    TORCH_CHECK(t.device() == query.device(), "inputs must share a CUDA device");
  }
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && dk.scalar_type() == at::kBFloat16 && dv.scalar_type() == at::kBFloat16, "BF16 query/decoders required");
  TORCH_CHECK(pk.scalar_type() == at::kLong && pv.scalar_type() == at::kLong, "int64 packed codes required");
  TORCH_CHECK(lse.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat && den.scalar_type() == at::kFloat, "FP32 LSE/workspace required");
  TORCH_CHECK(pk.dim() == 4 && pv.dim() == 4 && query.dim() == 4 && dk.dim() == 4 && dv.dim() == 4, "four-dimensional query/codes/decoders required");
  const auto b = pk.size(0), h = pk.size(1), length = pk.size(3), ck = pk.size(2) * 64, cv = pv.size(3);
  TORCH_CHECK(ck >= 64 && ck <= 512 && cv >= 64 && cv <= 512 && cv % 64 == 0, "code dimensions must be multiples of64 up to512");
  TORCH_CHECK(pv.sizes() == at::IntArrayRef({b,h,(length+63)/64,cv}), "packed V shape mismatch");
  if(ptx && reconstruct) {
    TORCH_CHECK((dk.sizes() == at::IntArrayRef({b,h,128,ck}) && dv.sizes() == at::IntArrayRef({b,h,128,cv})) ||
                (dk.sizes() == at::IntArrayRef({b,h,ck/16,2048}) && dv.sizes() == at::IntArrayRef({b,h,cv/16,2048})), "decoder layout shape mismatch");
  } else {
    TORCH_CHECK(dk.sizes() == at::IntArrayRef({b,h,ck,128}) && dv.sizes() == at::IntArrayRef({b,h,cv,128}), "decoder shape mismatch");
  }
  TORCH_CHECK(query.sizes() == at::IntArrayRef({b,h*4,1,reconstruct ? 128 : ck}), "GQA4 query shape mismatch");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({b,h*4,1,reconstruct ? 128 : cv}), "output workspace shape mismatch");
  TORCH_CHECK(lse.numel() == b*h*4 && den.numel() == b*h*4, "LSE/denominator shape mismatch");
  c10::cuda::CUDAGuard guard(query.device());
  if (!length) return;
  if(ptx && reconstruct && dk.size(2)==ck/16) {
#define PACKED(BT,W) if(tile==BT && warps==W && block_tiles==1) { launch_packed_decoder<BT,W>(query,pk,pv,dk,dv,lse,output,den,float(scale)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
    PACKED(32,4) PACKED(32,8) PACKED(64,4) PACKED(64,8) PACKED(128,4) PACKED(128,8)
#undef PACKED
    TORCH_CHECK(false,"unsupported configuration for packed decoder layout");
  }
  // Specialize the measured code dimensions, retain the generic path for all others.
#define SPECIAL(BT,W) if (ptx && ck==320 && cv==320 && tile==BT && warps==W && block_tiles==1) { launch<BT,W,1,320>(reconstruct,ptx,query,pk,pv,dk,dv,lse,output,den,float(scale)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  SPECIAL(32,4) SPECIAL(32,8) SPECIAL(64,4) SPECIAL(64,8) SPECIAL(128,8)
#undef SPECIAL
  bool launched = false;
#define TRY(BT,W,BLOCKS) if (tile==BT && warps==W && block_tiles==BLOCKS) { launch<BT,W,BLOCKS>(reconstruct,ptx,query,pk,pv,dk,dv,lse,output,den,float(scale)); launched=true; }
#define THREADS(BT,BLOCKS) TRY(BT,4,BLOCKS) TRY(BT,8,BLOCKS)
#define TILES(BT) THREADS(BT,1) THREADS(BT,2) THREADS(BT,4) THREADS(BT,8)
  TILES(32) TILES(64) TILES(128)
#undef TILES
#undef THREADS
#undef TRY
  TORCH_CHECK(launched, "unsupported tile/warps/block_tiles configuration");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bf16_code_attention_cuda(
    at::Tensor q, at::Tensor pk, at::Tensor pv, at::Tensor dk, at::Tensor dv,
    at::Tensor lse, at::Tensor out, at::Tensor den, double scale, bool reconstruct,
    int64_t tile, int64_t warps, int64_t block_tiles) {
  bf16_code_attention_impl(q,pk,pv,dk,dv,lse,out,den,scale,reconstruct,tile,warps,block_tiles,false);
}
void bf16_code_attention_ptx_cuda(
    at::Tensor q, at::Tensor pk, at::Tensor pv, at::Tensor dk, at::Tensor dv,
    at::Tensor lse, at::Tensor out, at::Tensor den, double scale, bool reconstruct,
    int64_t tile, int64_t warps, int64_t block_tiles) {
  bf16_code_attention_impl(q,pk,pv,dk,dv,lse,out,den,scale,reconstruct,tile,warps,block_tiles,true);
}
