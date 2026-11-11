"""BF16 code-domain attention with ordinary GEMMs and no low-bit Q/W quantization.

The order is (Q @ D_K.T) @ C_K.T, softmax, then (W @ C_V) @ D_V.
Codes are unpacked afresh on each call; reconstructed token K/V are never built.
"""

from __future__ import annotations

import math

import torch

from picokv.picokv_standalone.cache import PreparedKVCache
from picokv.picokv_standalone.decode import _append_decode_tokens
from .reconstruct_attention import unpack_codes


def decode_attention(
    query: torch.Tensor,
    cache: PreparedKVCache,
    *,
    decode_key: torch.Tensor | None = None,
    decode_value: torch.Tensor | None = None,
    update_cache: bool = False,
    use_pruning: bool = False,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    attention_scale: float | None = None,
) -> torch.Tensor:
    """Single-token GQA attention over BF16 +/-1 codes and retained dense KV.

    GEMM operands and outputs are BF16; softmax is computed in FP32 and its
    probabilities are cast to BF16 for GEMM. No integer quantizer, b_q, b_w,
    pruning, or per-query/per-weight quantization scale is used.
    """
    if use_pruning:
        raise ValueError("the unpacked control requires pruning disabled")
    if query.ndim != 4 or query.shape[-2] != 1:
        raise ValueError("query must have shape [B,QH,1,D]")
    if query.device != cache.device or query.dtype != torch.bfloat16 or cache.dtype != query.dtype:
        raise ValueError("query and cache must share a device and use BF16")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[-1] != cache.head_dim:
        raise ValueError("query batch/head dimension does not match the cache")
    if query.shape[1] != 4 * cache.kv_heads:
        raise ValueError("the comparison uses the production GQA=4 configuration")

    dense_key, dense_value = _append_decode_tokens(cache, decode_key, decode_value, update_cache)
    # Group the four query heads so GEMMs read each KV head's codes directly,
    # without repeating the large code matrices for every query head.
    grouped_query = query.reshape(query.shape[0], cache.kv_heads, 4, cache.head_dim)
    projected_query = torch.matmul(grouped_query, cache.key_projection_t)
    key_codes = unpack_codes(cache.packed_key, length=cache.global_len, is_key=True)
    global_scores = torch.matmul(projected_query, key_codes.transpose(-1, -2)).float()
    del key_codes, projected_query
    dense_scores = torch.matmul(grouped_query, dense_key.transpose(-1, -2)).float()
    scale = 1.0 / math.sqrt(cache.head_dim) if attention_scale is None else float(attention_scale)
    scores = torch.cat((global_scores, dense_scores), dim=-1).mul_(scale)
    del global_scores, dense_scores
    # One normalization over BOTH global and dense tokens is essential.
    weights = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    del scores
    value_codes = unpack_codes(cache.packed_value, length=cache.global_len, is_key=False)
    aggregated_codes = torch.matmul(weights[..., :cache.global_len], value_codes)
    del value_codes
    global_output = torch.matmul(aggregated_codes, cache.value_projection)
    dense_output = torch.matmul(weights[..., cache.global_len:], dense_value)
    return (global_output + dense_output).reshape(query.shape)
