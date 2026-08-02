#include <torch/extension.h>
#include <ATen/ATen.h>

// Dense retained-KV attention numerator and logsumexp for GQA decode.
// Inputs: query [B,QH,S,D], dense_key/value [B,KVH,L,D].
void quanpc_dense_attention_numerator(
  at::Tensor query,
  at::Tensor dense_key,
  at::Tensor dense_value,
  at::Tensor dense_output,
  at::Tensor dense_lse,
  double attention_scale
);


// Project global code output, add dense numerator, and normalize.
void quanpc_fused_output_projection(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output
);

// Fused RMSNorm over the last dimension. input: contiguous [..., N];
// weight: [N]; output: same shape/dtype as input. Matches HF RMSNorm dtype flow.
void quanpc_rmsnorm(
  at::Tensor input,
  at::Tensor weight,
  at::Tensor output,
  double eps
);

// Two-pass non-atomic split-K variant of the fused output projection.
void quanpc_fused_output_projection_splitk(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int64_t k_splits
);

// Single-launch atomic ("last block finalizes") split-K fused output projection.
void quanpc_fused_output_projection_atomic(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output,
  int64_t k_splits
);


// GQA-specialized L-tiled packed QuanPC global attention for QH/KVH == 4.
void quanpc_global_code_output_packed_value_tiled_gqa4(
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
);


// Compatibility entry point for the former Tensor Core experiment. The
// recovered implementation currently delegates to the scalar GQA4 path.
void quanpc_global_code_output_packed_value_tiled_gqa4_tensorcore(
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
);


// Experimental path for 1-bit value codes. Attention weights are quantized to
// 1/2/4/8 bits per 64-token group, then value accumulation is computed with bitwise
// mask intersections and popcount.
void quanpc_global_code_output_v1_exp8_tiled_gqa4(
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
);





// Experimental scalar CUDA path for query projection + int8 quantization + packing.
void quanpc_project_quantize_pack_query(
  at::Tensor query,
  at::Tensor key_proj,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
);

// Fused symmetric int8 quantization and projected-query bit packing.
void quanpc_quantize_pack_projected_query(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
);

// Pack int8 projected queries into sign + 7 magnitude bitplanes used by the
// bitwise key dot-product kernels. Input: [B,H,S,C], output: [B,H,S,C/64,8].
void quanpc_pack_projected_query(
  at::Tensor quant_projected_query,
  at::Tensor packed_projected_query
);



// V1 partial stage followed by fused reduce + value projection + normalization.
void quanpc_global_output_v1_exp8_tiled_gqa4_fused_reduce(
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
);


// Host-side fused decode tail: qpack projected query, clear global outputs,
// run atomic global packed attention, then project/normalize output.

void quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic(
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
);

// Pruned GQA4/Ck=256/Cv=256 direct-output path. prune_mode:
//   1 = 64-token block pruning after group max,
//   2 = cascaded prefix-dim token pruning at 64/128/192,
//   3 = both.
void quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned(
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
);


void quanpc_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned(
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
);

void quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned(
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
);

// Experimental direct-output v1 path. Tile kernels atomically accumulate
// fp32 code output and denominator, avoiding partial tensors and reduce kernel.

void quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic(
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
);

void quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic(
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
);






