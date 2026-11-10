#include <torch/extension.h>
#include <ATen/ATen.h>

// Dense retained-KV attention numerator and logsumexp for GQA decode.
// Inputs: query [B,QH,S,D], dense_key/value [B,KVH,L,D].
void picokv_dense_attention_numerator(
  at::Tensor query,
  at::Tensor dense_key,
  at::Tensor dense_value,
  at::Tensor dense_output,
  at::Tensor dense_lse,
  double attention_scale
);


// Project global code output, add dense numerator, and normalize.
void picokv_fused_output_projection(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output
);

// Same projection via one cublas strided-batched GEMM (bf16 tensor cores, fp32
// accumulate) + epilogue. Same math up to fp32 accumulation order.
void picokv_fused_output_projection_gemm(
  at::Tensor dense_output,
  at::Tensor glob_code_output,
  at::Tensor glob_den,
  at::Tensor value_proj,
  at::Tensor attn_output
);


// GQA-specialized L-tiled packed PiCoKV global attention for QH/KVH == 4.
void picokv_global_code_output_packed_value_tiled_gqa4(
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
void picokv_global_code_output_packed_value_tiled_gqa4_tensorcore(
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
void picokv_global_code_output_v1_exp8_tiled_gqa4(
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
void picokv_project_quantize_pack_query(
  at::Tensor query,
  at::Tensor key_proj,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
);

// Fused symmetric int8 quantization and projected-query bit packing.
void picokv_quantize_pack_projected_query(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor query_scale
);

// Same, plus zeroing the two fp32 tail accumulators ([B,H,S,Cv] codes and
// [B,H,S] denominators) in the same launch, replacing a separate memset.
void picokv_quantize_pack_zero_projected_query(
  at::Tensor projected_query,
  at::Tensor packed_projected_query,
  at::Tensor query_scale,
  at::Tensor code_zero,
  at::Tensor den_zero
);

// Pack int8 projected queries into sign + 7 magnitude bitplanes used by the
// bitwise key dot-product kernels. Input: [B,H,S,C], output: [B,H,S,C/64,8].
void picokv_pack_projected_query(
  at::Tensor quant_projected_query,
  at::Tensor packed_projected_query
);



// V1 partial stage followed by fused reduce + value projection + normalization.
void picokv_global_output_v1_exp8_tiled_gqa4_fused_reduce(
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

void picokv_decode_tail_v1_exp8_tiled_gqa4_atomic(
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
  at::Tensor global_lens,
  at::Tensor sched_req,
  at::Tensor sched_tile
);

// Pruned GQA4/Ck=256/Cv=256 direct-output path. prune_mode:
//   1 = 64-token block pruning after group max,
//   2 = cascaded prefix-dim token pruning at 64/128/192,
//   3 = both.
void picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned(
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


void picokv_decode_tail_packed_v1_exp8_tiled_gqa4_atomic_pruned(
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

void picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned(
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

void picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic(
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

void picokv_global_code_output_v1_exp8_tiled_gqa4_atomic(
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
  at::Tensor global_lens,
  at::Tensor sched_req,
  at::Tensor sched_tile,
  // Per-row packed storage (optional, TC sched path only): int64 [B] row
  // addresses of [KVH, KB, cap] keys / [KVH, cap/64, Cv] values, int32 [B]
  // caps (multiples of 512). packed_key / packed_value then only give shapes.
  at::Tensor row_kptr = at::Tensor(),
  at::Tensor row_vptr = at::Tensor(),
  at::Tensor row_cap = at::Tensor(),
  // Deterministic mode (optional, TC path only): fp32 per-block partials
  // [S*GZ*GX, 4, 320] / [S*GZ*GX, 4] instead of atomics into glob_*; int32
  // [B] first sched entry per row (sched grids). See det_tail_reduce.
  at::Tensor part_code = at::Tensor(),
  at::Tensor part_den = at::Tensor(),
  at::Tensor part_rs = at::Tensor(),
  // Fused query quantizer (optional, TC path only, not deterministic): the
  // bf16 projected query [B,QH,S,320]; each tail block quantizes + packs its
  // heads (bitwise the prep op's), packed_projected_query / query_scale only
  // give shapes. glob_* must already be zero (no prep op ran).
  at::Tensor projected_query = at::Tensor()
);

// Tiles (256 tokens) per block of the unscheduled tensor-core tail grid
// (sched grids: 2); the deterministic reduction's tile size.
int picokv_tail_block_tiles(int L, int BKVH);


// Fused dense-window decode attention over the per-layer ring pools: softmax
// attention over rows [0, lpl[b]) of kbuf/vbuf [B, cap, KVH, D] (GQA-4,
// head_dim 128, bf16). Writes out [B, QH, D] and lse [B, QH] in ln space —
// the same contract as the flashinfer paged-decode call it replaces. lpl is
// read on device at run time (CUDA-graph replay-safe as the window grows).
void picokv_window_attn(
  at::Tensor kbuf,
  at::Tensor vbuf,
  at::Tensor lpl,
  at::Tensor q,
  at::Tensor out,
  at::Tensor lse,
  double sm_scale
);






