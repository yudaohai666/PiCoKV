#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> reconstruct_kv_cuda(
    at::Tensor packed_key, at::Tensor packed_value,
    at::Tensor key_decoder, at::Tensor value_decoder,
    at::Tensor dense_key, at::Tensor dense_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reconstruct_kv", &reconstruct_kv_cuda,
        "Fused binary unpack, BF16 tensor-core C @ D, and retained KV copy");
}
