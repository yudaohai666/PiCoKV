#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_kernel.h"

void quant_probe_hash64_fn(
  at::Tensor probe, at::Tensor quant_probe
) {
    quant_probe_hash64(probe, quant_probe);
}

void format_bits_hash64_fn(
  at::Tensor unpack_x, at::Tensor pack_x, bool pack
) {
    format_bits_hash64(unpack_x, pack_x, pack);
}

void format_probe_hash64_fn(
  at::Tensor unpack_x, at::Tensor pack_x, bool pack
) {
    format_probe_hash64(unpack_x, pack_x, pack);
}

void binary_project_dim128_hash64_fn(
  at::Tensor A, at::Tensor quan_proj, at::Tensor y
) {
  binary_project_dim128_hash64(A, quan_proj, y);
}

void probe_hash64_group4_block256_fn(
  at::Tensor probe, at::Tensor hashcode, at::Tensor scores
) {
    probe_hash64_group4_block256(probe, hashcode, scores);
}

void topk_step1_block512_fn(
  at::Tensor scores, at::Tensor counters
) {
    topk_step1_block512(scores, counters);
}

void topk_step2_fn(
  at::Tensor counters, at::Tensor counters_cumsum
) {
    topk_step2(counters, counters_cumsum);
}

void topk_step3_block512_fn(
  at::Tensor scores, at::Tensor counters_cumsum, at::Tensor indices
) {
    topk_step3_block512(scores, counters_cumsum, indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_probe_hash64_fn", &quant_probe_hash64_fn, "quant_probe_hash64_fn");
  m.def("format_bits_hash64_fn", &format_bits_hash64_fn, "format_bits_hash64_fn");
  m.def("format_probe_hash64_fn", &format_probe_hash64_fn, "format_probe_hash64_fn");
  m.def("binary_project_dim128_hash64_fn", &binary_project_dim128_hash64_fn, "binary_project_dim128_hash64_fn");
  m.def("probe_hash64_group4_block256_fn", &probe_hash64_group4_block256_fn, "probe_hash64_group4_block256_fn");
  m.def("topk_step1_block512_fn", &topk_step1_block512_fn, "topk_step1_block512_fn");
  m.def("topk_step2_fn", &topk_step2_fn, "topk_step2_fn");
  m.def("topk_step3_block512_fn", &topk_step3_block512_fn, "topk_step3_block512_fn");
}
