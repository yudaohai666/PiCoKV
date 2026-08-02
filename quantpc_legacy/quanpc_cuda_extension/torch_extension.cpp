#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_kernel.h"

void quanpc_dense_attention_numerator_fn(
  at::Tensor query,
  at::Tensor dense_key,
  at::Tensor dense_value,
  at::Tensor dense_output,
  at::Tensor dense_lse,
  double attention_scale
) {
    quanpc_dense_attention_numerator(
      query,
      dense_key,
      dense_value,
      dense_output,
      dense_lse,
      attention_scale
    );
}

void quanpc_rmsnorm_fn(
  at::Tensor input,
  at::Tensor weight,
  at::Tensor output,
  double eps
) {
    quanpc_rmsnorm(input, weight, output, eps);
}

void quanpc_fused_output_projection_fn(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output
) {
    quanpc_fused_output_projection(
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output
    );
}

void quanpc_fused_output_projection_splitk_fn(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int64_t k_splits
) {
    quanpc_fused_output_projection_splitk(
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output,
      k_splits
    );
}

void quanpc_fused_output_projection_atomic_fn(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int64_t k_splits
) {
    quanpc_fused_output_projection_atomic(
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output,
      k_splits
    );
}

void quanpc_global_code_output_packed_value_tiled_gqa4_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int value_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_global_code_output_packed_value_tiled_gqa4(
      packed_projected_query,
      packed_key,
      packed_value,
      query_scale,
      dense_lse,
      partial_code_output,
      partial_den,
      glob_code_output,
      glob_den,
      value_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}

void quanpc_global_code_output_packed_value_tiled_gqa4_tensorcore_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int value_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_global_code_output_packed_value_tiled_gqa4_tensorcore(
      packed_projected_query,
      packed_key,
      packed_value,
      query_scale,
      dense_lse,
      partial_code_output,
      partial_den,
      glob_code_output,
      glob_den,
      value_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}


void quanpc_global_code_output_v1_exp8_tiled_gqa4_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_global_code_output_v1_exp8_tiled_gqa4(
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      partial_code_output,
      partial_den,
      glob_code_output,
      glob_den,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}





void quanpc_project_quantize_pack_query_fn(
  at::Tensor query,
  at::Tensor key_proj,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
) {
    quanpc_project_quantize_pack_query(query, key_proj, packed_projected_query, query_scale);
}

void quanpc_quantize_pack_projected_query_fn(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
) {
    quanpc_quantize_pack_projected_query(projected_query, packed_projected_query, query_scale);
}

void quanpc_pack_projected_query_fn(
  at::Tensor quant_projected_query,
  at::Tensor packed_projected_query
) {
    quanpc_pack_projected_query(quant_projected_query, packed_projected_query);
}



void quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor partial_code_output,
  at::Tensor partial_den,
  at::Tensor dense_output,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce(
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      partial_code_output,
      partial_den,
      dense_output,
      value_proj,
      attn_output,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}

void quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic(
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      glob_code_output,
      glob_den,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}

void quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic(
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      glob_code_output,
      glob_den,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}


void quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale
) {
    quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic(
      projected_query,
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale
    );
}

void quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  int prune_mode,
  double prune_log_eps
) {
    quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned(
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      glob_code_output,
      glob_den,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale,
      prune_mode,
      prune_log_eps
    );
}

void quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  int prune_mode,
  double prune_log_eps
) {
    quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned(
      projected_query,
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale,
      prune_mode,
      prune_log_eps
    );
}


void quanpc_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned_fn(
  at::Tensor packed_projected_query,
  at::Tensor packed_key,
  at::Tensor packed_value_sign,
  at::Tensor query_scale,
  at::Tensor dense_lse,
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int exp_bits,
  int tile_size,
  int key_transposed,
  double attention_scale,
  int prune_mode,
  double prune_log_eps
) {
    quanpc_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned(
      packed_projected_query,
      packed_key,
      packed_value_sign,
      query_scale,
      dense_lse,
      dense_output,
      glob_code_output,
      glob_den,
      value_proj,
      attn_output,
      exp_bits,
      tile_size,
      key_transposed,
      attention_scale,
      prune_mode,
      prune_log_eps
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quanpc_dense_attention_numerator_fn", &quanpc_dense_attention_numerator_fn, "quanpc_dense_attention_numerator_fn");
  m.def("quanpc_rmsnorm_fn", &quanpc_rmsnorm_fn, "quanpc_rmsnorm_fn");
  m.def("quanpc_fused_output_projection_fn", &quanpc_fused_output_projection_fn, "quanpc_fused_output_projection_fn");
  m.def("quanpc_fused_output_projection_splitk_fn", &quanpc_fused_output_projection_splitk_fn, "quanpc_fused_output_projection_splitk_fn");
  m.def("quanpc_fused_output_projection_atomic_fn", &quanpc_fused_output_projection_atomic_fn, "quanpc_fused_output_projection_atomic_fn");
  m.def("quanpc_global_code_output_packed_value_tiled_gqa4_fn", &quanpc_global_code_output_packed_value_tiled_gqa4_fn, "quanpc_global_code_output_packed_value_tiled_gqa4_fn");
  m.def("quanpc_global_code_output_packed_value_tiled_gqa4_tensorcore_fn", &quanpc_global_code_output_packed_value_tiled_gqa4_tensorcore_fn, "quanpc_global_code_output_packed_value_tiled_gqa4_tensorcore_fn");
  m.def("quanpc_global_code_output_v1_exp8_tiled_gqa4_fn", &quanpc_global_code_output_v1_exp8_tiled_gqa4_fn, "quanpc_global_code_output_v1_exp8_tiled_gqa4_fn");
  m.def("quanpc_pack_projected_query_fn", &quanpc_pack_projected_query_fn, "quanpc_pack_projected_query_fn");
  m.def("quanpc_quantize_pack_projected_query_fn", &quanpc_quantize_pack_projected_query_fn, "quanpc_quantize_pack_projected_query_fn");
  m.def("quanpc_project_quantize_pack_query_fn", &quanpc_project_quantize_pack_query_fn, "quanpc_project_quantize_pack_query_fn");
  m.def("quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce_fn", &quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce_fn, "quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce_fn");
  m.def("quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_fn", &quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_fn, "quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_fn");
  m.def("quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn", &quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn, "quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn");
  m.def("quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn", &quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn");
  m.def("quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_fn", &quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_fn, "quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_fn");
  m.def("quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn", &quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn");
  m.def("quanpc_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned_fn", &quanpc_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned_fn, "quanpc_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned_fn");
}
