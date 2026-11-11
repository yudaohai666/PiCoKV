#include <torch/extension.h>

void bf16_code_attention_cuda(
    at::Tensor query, at::Tensor packed_key, at::Tensor packed_value,
    at::Tensor key_decoder, at::Tensor value_decoder, at::Tensor dense_lse,
    at::Tensor output, at::Tensor denominator, double scale, bool reconstruct,
    int64_t tile, int64_t warps, int64_t block_tiles);
void bf16_code_attention_ptx_cuda(
    at::Tensor query, at::Tensor packed_key, at::Tensor packed_value,
    at::Tensor key_decoder, at::Tensor value_decoder, at::Tensor dense_lse,
    at::Tensor output, at::Tensor denominator, double scale, bool reconstruct,
    int64_t tile, int64_t warps, int64_t block_tiles);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &bf16_code_attention_cuda,
        "Handwritten BF16 WMMA attention, with fused binary unpack/reconstruction");
  m.def("attention_ptx", &bf16_code_attention_ptx_cuda,
        "Handwritten explicit BF16 MMA attention, with fused binary unpack/reconstruction");
}
