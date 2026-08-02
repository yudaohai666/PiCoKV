#include <torch/extension.h>
#include <ATen/ATen.h>

#include "cuda_kernel.h"

std::vector<at::Tensor> quanpc_prefill_learn_projection_fn(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  return quanpc_prefill_learn_projection(
      full_states,
      init_vectors,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope);
}


std::vector<at::Tensor> quanpc_prefill_quantize_fn(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  return quanpc_prefill_quantize(
      full_states,
      projection,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope);
}


std::vector<at::Tensor> quanpc_prefill_quantize_pack_fn(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  return quanpc_prefill_quantize_pack(
      full_states,
      projection,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      pack_key_layout);
}

std::vector<at::Tensor> quanpc_prefill_select_eas_fn(
    at::Tensor key_err,
    at::Tensor value_err,
    int64_t eas_size) {
  return quanpc_prefill_select_eas(key_err, value_err, eas_size);
}

std::vector<at::Tensor> quanpc_prefill_gather_pack_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor key_codes,
    at::Tensor value_codes,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  return quanpc_prefill_gather_pack(
      full_key,
      full_value,
      key_codes,
      value_codes,
      eas_indices,
      kept_indices,
      sink_size,
      local_size);
}


std::vector<at::Tensor> quanpc_prefill_gather_pack_from_packed_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor packed_key_all,
    at::Tensor packed_value_token,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  return quanpc_prefill_gather_pack_from_packed(
      full_key,
      full_value,
      packed_key_all,
      packed_value_token,
      eas_indices,
      kept_indices,
      sink_size,
      local_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quanpc_prefill_learn_projection_fn", &quanpc_prefill_learn_projection_fn, "QuanPC CUDA projection learning");
  m.def("quanpc_prefill_quantize_fn", &quanpc_prefill_quantize_fn, "QuanPC prefill projection quantization");
  m.def("quanpc_prefill_quantize_pack_fn", &quanpc_prefill_quantize_pack_fn, "QuanPC fused prefill quantization and token/code packing");
  m.def("quanpc_prefill_select_eas_fn", &quanpc_prefill_select_eas_fn, "QuanPC prefill EAS selection");
  m.def("quanpc_prefill_gather_pack_fn", &quanpc_prefill_gather_pack_fn, "QuanPC prefill dense gather and packed code output");
  m.def("quanpc_prefill_gather_pack_from_packed_fn", &quanpc_prefill_gather_pack_from_packed_fn, "QuanPC prefill dense gather and packed output from fused packed codes");
}
