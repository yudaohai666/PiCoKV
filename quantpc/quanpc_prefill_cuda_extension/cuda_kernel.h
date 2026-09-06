#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> quanpc_prefill_learn_projection(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope);

std::vector<at::Tensor> quanpc_prefill_quantize(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope);

std::vector<at::Tensor> quanpc_prefill_quantize_pack(
    at::Tensor full_states,
    at::Tensor projection,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout);

std::vector<at::Tensor> quanpc_prefill_blockk_fast(
    at::Tensor full_states,
    at::Tensor init_vectors,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    int64_t group_bits,
    int64_t power_iters,
    int64_t solve_mode);

std::vector<at::Tensor> quanpc_prefill_blockk_fixed(
    at::Tensor full_states,
    at::Tensor encoders,
    at::Tensor decoders,
    int64_t glob_start,
    int64_t glob_len,
    int64_t zerorope,
    bool apply_zerorope,
    bool pack_key_layout);

std::vector<at::Tensor> quanpc_prefill_select_eas(
    at::Tensor key_err,
    at::Tensor value_err,
    int64_t eas_size);

at::Tensor quanpc_prefill_pack_key_transposed(at::Tensor codes);

at::Tensor quanpc_prefill_pack_value_token(at::Tensor codes);

std::vector<at::Tensor> quanpc_prefill_gather_pack(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor key_codes,
    at::Tensor value_codes,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size);


std::vector<at::Tensor> quanpc_prefill_gather_pack_from_packed(
    at::Tensor full_key,
    at::Tensor full_value,
    at::Tensor packed_key_all,
    at::Tensor packed_value_token,
    at::Tensor eas_indices,
    at::Tensor kept_indices,
    int64_t sink_size,
    int64_t local_size);
