#include <torch/extension.h>
#include <ATen/ATen.h>

#include "cuda_kernel.h"

std::vector<at::Tensor> picokv_prefill_learn_projection_fn(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  return picokv_prefill_learn_projection(
      full_states,
      init_vectors,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope);
}


std::vector<at::Tensor> picokv_prefill_quantize_fn(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope) {
  return picokv_prefill_quantize(
      full_states,
      projection,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope);
}


std::vector<at::Tensor> picokv_prefill_quantize_pack_fn(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  return picokv_prefill_quantize_pack(
      full_states,
      projection,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      pack_key_layout);
}

std::vector<at::Tensor> picokv_prefill_blockk_fast_fn(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    int64_t group_bits,
    int64_t power_iters,
    int64_t solve_mode) {
  return picokv_prefill_blockk_fast(
      full_states,
      init_vectors,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      group_bits,
      power_iters,
      solve_mode);
}

std::vector<at::Tensor> picokv_prefill_blockk_fixed_fn(
    at::Tensor full_states,
    at::Tensor encoders,
    at::Tensor decoders,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout) {
  return picokv_prefill_blockk_fixed(
      full_states,
      encoders,
      decoders,
      glob_start,
      glob_len,
      zerorope,
      apply_zerorope,
      pack_key_layout);
}

std::vector<at::Tensor> picokv_prefill_select_eas_fn(
    at::Tensor key_err,
    at::Tensor value_err,
    int64_t eas_size) {
  return picokv_prefill_select_eas(key_err, value_err, eas_size);
}

at::Tensor picokv_prefill_pack_key_transposed_fn(at::Tensor codes) {
  return picokv_prefill_pack_key_transposed(codes);
}

at::Tensor picokv_prefill_pack_value_token_fn(at::Tensor codes) {
  return picokv_prefill_pack_value_token(codes);
}

std::vector<at::Tensor> picokv_prefill_gather_pack_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor key_codes,
    at::Tensor value_codes,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  return picokv_prefill_gather_pack(
      full_key,
      full_value,
      key_codes,
      value_codes,
      eas_indices,
      kept_indices,
      sink_size,
      local_size);
}


std::vector<at::Tensor> picokv_prefill_gather_pack_from_packed_fn(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor packed_key_all,
    at::Tensor packed_value_token,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size) {
  return picokv_prefill_gather_pack_from_packed(
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
  m.def("picokv_prefill_learn_projection_fn", &picokv_prefill_learn_projection_fn, "PiCoKV CUDA projection learning");
  m.def("picokv_prefill_quantize_fn", &picokv_prefill_quantize_fn, "PiCoKV prefill projection quantization");
  m.def("picokv_prefill_quantize_pack_fn", &picokv_prefill_quantize_pack_fn, "PiCoKV fused prefill quantization and token/code packing");
  m.def("picokv_prefill_blockk_fast_fn", &picokv_prefill_blockk_fast_fn, "PiCoKV fast fused block-k prefill learning and quantization");
  m.def("picokv_prefill_blockk_fixed_fn", &picokv_prefill_blockk_fixed_fn, "PiCoKV fully fused fixed-projection cascade quantization and packing");
  m.def("picokv_prefill_select_eas_fn", &picokv_prefill_select_eas_fn, "PiCoKV prefill EAS selection");
  m.def("picokv_prefill_pack_key_transposed_fn", &picokv_prefill_pack_key_transposed_fn, "PiCoKV pack int8 key codes to [B,H,C/64,L]");
  m.def("picokv_prefill_pack_value_token_fn", &picokv_prefill_pack_value_token_fn, "PiCoKV pack int8 value codes to [B,H,L,C/64]");
  m.def("picokv_prefill_gather_pack_fn", &picokv_prefill_gather_pack_fn, "PiCoKV prefill dense gather and packed code output");
  m.def("picokv_prefill_gather_pack_from_packed_fn", &picokv_prefill_gather_pack_from_packed_fn, "PiCoKV prefill dense gather and packed output from fused packed codes");
}
