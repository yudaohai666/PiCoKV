#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <vector>

namespace {
namespace wmma = nvcuda::wmma;
constexpr int kDim = 128;
constexpr int kTokens = 64;
constexpr int kCodeTile = 64;
constexpr int kAStride = 72;
constexpr int kBStride = 136;
constexpr int kOutStride = 128;
constexpr int kThreads = 128;
constexpr int kSharedBytes = kTokens * kOutStride * sizeof(float);

// Each block reconstructs 64 tokens x 128 dimensions on four tensor-core
// warps. Packed codes are unpacked ONLY in shared memory, never into a global
// BF16 C tensor. The decoder is staged in 64-code chunks. The final store also
// appends the dense window, so a separate concatenate/copy is unnecessary.
__global__ void reconstruct_kv_kernel(
    const int64_t* __restrict__ packed_key,
    const int64_t* __restrict__ packed_value,
    const __nv_bfloat16* __restrict__ key_decoder,
    const __nv_bfloat16* __restrict__ value_decoder,
    const __nv_bfloat16* __restrict__ dense_key,
    const __nv_bfloat16* __restrict__ dense_value,
    __nv_bfloat16* __restrict__ output_key,
    __nv_bfloat16* __restrict__ output_value,
    int global_len, int dense_len, int key_bits, int value_bits) {
  const bool is_key = blockIdx.z == 0;
  const int row = blockIdx.y;
  const int token_start = blockIdx.x * kTokens;
  const int total_len = global_len + dense_len;
  const int bits = is_key ? key_bits : value_bits;
  const auto* decoder = (is_key ? key_decoder : value_decoder)
      + static_cast<int64_t>(row) * bits * kDim;
  const auto* dense = (is_key ? dense_key : dense_value)
      + static_cast<int64_t>(row) * dense_len * kDim;
  auto* output = (is_key ? output_key : output_value)
      + static_cast<int64_t>(row) * total_len * kDim;

  if (token_start >= global_len) {
    for (int i = threadIdx.x; i < kTokens * (kDim / 8); i += kThreads) {
      const int token = token_start + i / (kDim / 8);
      const int dim = (i % (kDim / 8)) * 8;
      if (token < total_len) {
        *reinterpret_cast<uint4*>(output + static_cast<int64_t>(token) * kDim + dim) =
            *reinterpret_cast<const uint4*>(dense + static_cast<int64_t>(token - global_len) * kDim + dim);
      }
    }
    return;
  }

  extern __shared__ __align__(32) unsigned char storage[];
  auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
  auto* b = a + kTokens * kAStride;
  auto* words = reinterpret_cast<uint64_t*>(b + kCodeTile * kBStride);
  const int warp = threadIdx.x / 32;
  const int warp_m = warp * 16;
  const int warp_n = 0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[8];
  #pragma unroll
  for (int n = 0; n < 8; ++n) wmma::fill_fragment(accumulators[n], 0.0f);

  for (int code_start = 0; code_start < bits; code_start += kCodeTile) {
    if (threadIdx.x < 64) {
      uint64_t word = 0;
      if (is_key) {
        const int token = token_start + threadIdx.x;
        if (token < global_len) {
          word = static_cast<uint64_t>(packed_key[
              (static_cast<int64_t>(row) * (bits / 64) + code_start / 64) * global_len + token]);
        }
      } else {
        const int groups = (global_len + 63) / 64;
        word = static_cast<uint64_t>(packed_value[
            (static_cast<int64_t>(row) * groups + token_start / 64) * bits + code_start + threadIdx.x]);
      }
      words[threadIdx.x] = word;
    }
    for (int i = threadIdx.x; i < kCodeTile * (kDim / 8); i += kThreads) {
      const int r = i / (kDim / 8);
      const int d = (i % (kDim / 8)) * 8;
      *reinterpret_cast<uint4*>(b + r * kBStride + d) =
          *reinterpret_cast<const uint4*>(decoder + (code_start + r) * kDim + d);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kTokens * (kCodeTile / 2); i += kThreads) {
      const int token = i / (kCodeTile / 2);
      const int code = (i % (kCodeTile / 2)) * 2;
      unsigned negative0, negative1;
      if (is_key) {
        const unsigned pair = (words[token] >> code) & 3ULL;
        negative0 = pair & 1;
        negative1 = pair >> 1;
      } else {
        negative0 = 1U ^ ((words[code] >> token) & 1ULL);
        negative1 = 1U ^ ((words[code + 1] >> token) & 1ULL);
      }
      // BF16 +/-1 differ only in their sign bit. Emit a packed BF16 pair
      // directly, without scalar float conversion or half-word smem stores.
      *reinterpret_cast<uint32_t*>(a + token * kAStride + code) =
          0x3f803f80U | (negative0 << 15) | (negative1 << 31);
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < kCodeTile; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
      wmma::load_matrix_sync(af, a + warp_m * kAStride + k, kAStride);
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
        wmma::load_matrix_sync(bf, b + k * kBStride + warp_n + n * 16, kBStride);
        wmma::mma_sync(accumulators[n], af, bf, accumulators[n]);
      }
    }
    __syncthreads();
  }

  // Reuse the staging area for the FP32 accumulator epilogue.
  auto* result = reinterpret_cast<float*>(storage);
  #pragma unroll
  for (int n = 0; n < 8; ++n) {
    wmma::store_matrix_sync(result + warp_m * kOutStride + warp_n + n * 16,
                           accumulators[n], kOutStride, wmma::mem_row_major);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kTokens * (kDim / 8); i += kThreads) {
    const int t = i / (kDim / 8);
    const int d = (i % (kDim / 8)) * 8;
    const int token = token_start + t;
    if (token >= total_len) continue;
    if (token >= global_len) {
      *reinterpret_cast<uint4*>(output + static_cast<int64_t>(token) * kDim + d) =
          *reinterpret_cast<const uint4*>(dense + static_cast<int64_t>(token - global_len) * kDim + d);
    } else {
      alignas(16) __nv_bfloat16 converted[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) converted[j] = __float2bfloat16_rn(result[t * kOutStride + d + j]);
      *reinterpret_cast<uint4*>(output + static_cast<int64_t>(token) * kDim + d) =
          *reinterpret_cast<const uint4*>(converted);
    }
  }
}
}  // namespace

std::vector<at::Tensor> reconstruct_kv_cuda(
    at::Tensor packed_key, at::Tensor packed_value,
    at::Tensor key_decoder, at::Tensor value_decoder,
    at::Tensor dense_key, at::Tensor dense_value) {
  for (const auto& tensor : {packed_key, packed_value, key_decoder, value_decoder, dense_key, dense_value}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "all inputs must be contiguous CUDA tensors");
    TORCH_CHECK(tensor.device() == packed_key.device(), "all inputs must share a CUDA device");
    TORCH_CHECK(tensor.dim() == 4, "all inputs must have four dimensions");
  }
  TORCH_CHECK(packed_key.scalar_type() == at::kLong && packed_value.scalar_type() == at::kLong,
              "packed codes must be int64");
  for (const auto& tensor : {key_decoder, value_decoder, dense_key, dense_value})
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "decoder and dense KV must be BF16");
  const auto batch = packed_key.size(0), heads = packed_key.size(1);
  const int length = packed_key.size(3), key_bits = packed_key.size(2) * 64;
  const int value_bits = packed_value.size(3), dense_len = dense_key.size(2);
  TORCH_CHECK(key_bits > 0 && value_bits > 0 && value_bits % 64 == 0, "code lengths must be positive multiples of 64");
  TORCH_CHECK(key_decoder.sizes() == at::IntArrayRef({batch, heads, key_bits, kDim}), "key decoder shape mismatch");
  TORCH_CHECK(value_decoder.sizes() == at::IntArrayRef({batch, heads, value_bits, kDim}), "value decoder shape mismatch");
  TORCH_CHECK(packed_value.sizes() == at::IntArrayRef({batch, heads, (length + 63) / 64, value_bits}), "packed value shape mismatch");
  TORCH_CHECK(dense_key.sizes() == at::IntArrayRef({batch, heads, dense_len, kDim}), "dense key shape mismatch");
  TORCH_CHECK(dense_value.sizes() == dense_key.sizes(), "dense value shape mismatch");
  c10::cuda::CUDAGuard guard(packed_key.device());
  auto key = at::empty({batch, heads, length + dense_len, kDim}, dense_key.options());
  auto value = at::empty_like(key);
  if (length + dense_len > 0) {
    const dim3 grid((length + dense_len + kTokens - 1) / kTokens, batch * heads, 2);
    reconstruct_kv_kernel<<<grid, kThreads, kSharedBytes, at::cuda::getCurrentCUDAStream()>>>(
        packed_key.data_ptr<int64_t>(), packed_value.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(key_decoder.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(value_decoder.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(dense_key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(dense_value.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        length, dense_len, key_bits, value_bits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {key, value};
}
