#include <torch/extension.h>
#include <ATen/ATen.h>

void quant_probe_hash64(
  at::Tensor probe, at::Tensor quant_probe
);

void format_bits_hash64(
  at::Tensor unpack_x, at::Tensor pack_x, bool pack
);

void format_probe_hash64(
  at::Tensor unpack_x, at::Tensor pack_x, bool pack
);

void binary_project_dim128_hash64(
  at::Tensor A, at::Tensor quan_proj, at::Tensor y
);

void probe_hash64_group4_block256(
  at::Tensor probe, at::Tensor hashcode, at::Tensor scores
);

void topk_step1_block512(
  at::Tensor scores, at::Tensor counters
);

void topk_step2(
  at::Tensor counters, at::Tensor counters_cumsum
);

void topk_step3_block512(
  at::Tensor scores, at::Tensor counters_cumsum, at::Tensor indices
);

// Fused: quant_probe + pack_probe + probe_hash64 + boundary handling.
// Outputs raw scores; topk is done in PyTorch.
void fused_quant_pack_probe_hash64(
  at::Tensor probe,
  at::Tensor hashcode,
  at::Tensor scores,
  int T,        // valid sequence length
  int hash_T    // valid hashed length
);

// Fused gather of keys and values.
void fused_gather_kv(
  at::Tensor keys,           // [B, H, L, D]
  at::Tensor values,         // [B, H, L, D]
  at::Tensor indices,        // [B, H, k]
  at::Tensor selected_keys,  // [B, H, k, D]
  at::Tensor selected_values // [B, H, k, D]
);