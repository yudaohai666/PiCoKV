"""Single-token CUDA attention over dense-retained and packed-global KV."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .cache import PreparedKVCache
from .constants import PROJECTED_QUERY_BITS

try:
    from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension
except Exception:  # pragma: no cover - reported by _decode_cuda
    load_quanpc_extension = None


def _append_decode_tokens(
    cache: PreparedKVCache,
    decode_key: Optional[torch.Tensor],
    decode_value: Optional[torch.Tensor],
    update_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense K/V including current tokens, optionally persisting them."""
    if decode_key is None and decode_value is None:
        return cache.dense_key, cache.dense_value
    if decode_key is None or decode_value is None:
        raise ValueError("decode_key and decode_value must be provided together")
    if decode_key.dim() != 4 or tuple(decode_key.shape) != tuple(decode_value.shape):
        raise ValueError("decode_key/decode_value must have the same [B,KVH,T,D] shape")
    expected = (*cache.dense_key.shape[:2], decode_key.shape[-2], cache.head_dim)
    if tuple(decode_key.shape) != tuple(expected):
        raise ValueError(f"decode K/V shape must be {tuple(expected)}; got {tuple(decode_key.shape)}")
    if decode_key.device != cache.device or decode_value.device != cache.device:
        raise ValueError("decode_key/decode_value must be on the cache device")
    if decode_key.dtype != cache.dtype or decode_value.dtype != cache.dtype:
        raise TypeError("decode_key/decode_value dtype must match the cache")

    dense_key = torch.cat((cache.dense_key, decode_key.contiguous()), dim=-2)
    dense_value = torch.cat((cache.dense_value, decode_value.contiguous()), dim=-2)
    if update_cache:
        cache.dense_key = dense_key
        cache.dense_value = dense_value
    return dense_key, dense_value


def _repeat_kv_heads(states: torch.Tensor, query_heads: int) -> torch.Tensor:
    """Expand KV heads only for the non-FlashAttention compatibility path."""
    kv_heads = int(states.shape[1])
    if query_heads == kv_heads:
        return states
    if query_heads % kv_heads:
        raise ValueError(f"QH must be divisible by KVH; got QH={query_heads}, KVH={kv_heads}")
    return states.repeat_interleave(query_heads // kv_heads, dim=1)


def _dense_attention_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dense retained attention and LSE when the Flash op is absent."""
    key = _repeat_kv_heads(key, query.shape[1])
    value = _repeat_kv_heads(value, query.shape[1])
    scores = torch.matmul(query, key.transpose(-1, -2)) * float(scale)
    lse = torch.logsumexp(scores.float(), dim=-1, keepdim=True)
    weights = torch.exp(scores.float() - lse).to(query.dtype)
    return torch.matmul(weights, value), lse.squeeze(-1).contiguous()


def _dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact dense-retained output and log-sum-exp normalizer."""
    if not hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention"):
        return _dense_attention_fallback(query, key, value, scale)
    output, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        0.0,
        False,
        False,
        scale=float(scale),
    )
    return output, lse.contiguous()


def _project_query_into_workspace(
    query: torch.Tensor,
    cache: PreparedKVCache,
    output: torch.Tensor,
) -> None:
    """Project GQA query heads with one strided batched GEMM and no allocation."""
    batch, query_heads, seq, head_dim = query.shape
    kv_heads = cache.kv_heads
    group_size = query_heads // kv_heads
    grouped_query = query.reshape(
        batch * kv_heads,
        group_size * seq,
        head_dim,
    )
    grouped_projection = cache.key_projection_t.reshape(
        batch * kv_heads,
        head_dim,
        cache.key_bits,
    )
    grouped_output = output.reshape(
        batch * kv_heads,
        group_size * seq,
        cache.key_bits,
    )
    torch.bmm(grouped_query, grouped_projection, out=grouped_output)


def _get_workspace(
    cache: PreparedKVCache,
    query: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Reuse all decode intermediates for a stable steady-state token path."""
    batch, query_heads, seq, head_dim = query.shape
    key = (
        query.device.index,
        query.dtype,
        batch,
        query_heads,
        seq,
        head_dim,
        cache.key_bits,
        cache.value_bits,
    )
    workspace = cache._workspace.get(key)
    if workspace is None:
        workspace = (
            # Projected query before int8 quantization.
            torch.empty(
                batch,
                query_heads,
                seq,
                cache.key_bits,
                device=query.device,
                dtype=query.dtype,
            ),
            # Eight packed bitplanes for each group of 64 projected dimensions.
            torch.empty(
                batch,
                query_heads,
                seq,
                cache.key_bits // 64,
                PROJECTED_QUERY_BITS,
                device=query.device,
                dtype=torch.long,
            ),
            # Per-query scale produced by projected-query quantization.
            torch.empty(batch, query_heads, seq, device=query.device, dtype=torch.float32),
            # Accumulated value code and global softmax denominator.
            torch.empty(
                batch,
                query_heads,
                seq,
                cache.value_bits,
                device=query.device,
                dtype=torch.float32,
            ),
            torch.empty(batch, query_heads, seq, device=query.device, dtype=torch.float32),
            # Final normalized attention output.
            torch.empty(
                batch,
                query_heads,
                seq,
                head_dim,
                device=query.device,
                dtype=query.dtype,
            ),
        )
        cache._workspace[key] = workspace
    return workspace


def _decode_cuda(
    query: torch.Tensor,
    cache: PreparedKVCache,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    *,
    scale: float,
    use_pruning: bool,
    prune_mode: int,
    log_eps: float,
) -> torch.Tensor:
    """Launch the measured GQA-4 CUDA decode path."""
    if load_quanpc_extension is None:
        raise RuntimeError("QuanPC decode CUDA extension is unavailable")
    extension = load_quanpc_extension()
    projected_query, packed_query, query_scale, global_code, global_den, output = (
        _get_workspace(cache, query)
    )

    if use_pruning:
        if not hasattr(
            extension,
            "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn",
        ):
            raise RuntimeError("loaded QuanPC CUDA extension lacks the pruning kernel")

        # Dense retained attention and query projection are independent.  Their
        # existing two-stream overlap is kept verbatim because it is part of
        # steady-state decode latency, unlike the removed async prepare path.
        current_stream = torch.cuda.current_stream(query.device)
        if cache._dense_stream is None:
            cache._dense_stream = torch.cuda.Stream(device=query.device)
        cache._dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(cache._dense_stream):
            dense_output, dense_lse = _dense_attention(
                query,
                dense_key,
                dense_value,
                scale,
            )
        _project_query_into_workspace(query, cache, projected_query)
        current_stream.wait_stream(cache._dense_stream)

        extension.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
            projected_query,
            packed_query,
            cache.packed_key,
            cache.packed_value,
            query_scale,
            dense_lse.contiguous(),
            dense_output.contiguous(),
            global_code,
            global_den,
            cache.value_projection,
            output,
            cache.exp_bits,
            256,
            1,
            float(scale),
            int(prune_mode),
            float(log_eps),
        )
        return output

    # The unpruned reference path is intentionally serial and uses the matching
    # non-pruned CUDA tail kernel for accuracy/performance comparisons.
    dense_output, dense_lse = _dense_attention(query, dense_key, dense_value, scale)
    _project_query_into_workspace(query, cache, projected_query)
    extension.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
        projected_query,
        packed_query,
        cache.packed_key,
        cache.packed_value,
        query_scale,
        dense_lse.contiguous(),
        dense_output.contiguous(),
        global_code,
        global_den,
        cache.value_projection,
        output,
        cache.exp_bits,
        256,
        1,
        float(scale),
    )
    return output


def decode_attention(
    query: torch.Tensor,
    cache: PreparedKVCache,
    *,
    decode_key: Optional[torch.Tensor] = None,
    decode_value: Optional[torch.Tensor] = None,
    update_cache: bool = False,
    use_pruning: bool = False,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    attention_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run one bf16 CUDA decode-attention step over ``cache``.

    ``query`` must be ``[B, 4*KVH, 1, D]``.  Optional current-token K/V can be
    included for this call and, with ``update_cache=True``, appended to the
    mutable dense portion for later decode steps.
    """
    if query.dim() != 4 or query.shape[-2] != 1:
        raise ValueError(f"query must be [B,QH,1,D]; got {tuple(query.shape)}")
    if not query.is_cuda or query.device != cache.device:
        raise ValueError("query must be a CUDA tensor on the cache device")
    if query.dtype != torch.bfloat16 or query.dtype != cache.dtype:
        raise TypeError("query and cache must use bfloat16")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[-1] != cache.head_dim:
        raise ValueError("query batch/head dimension does not match the cache")
    if query.shape[1] != cache.kv_heads * 4:
        raise ValueError(
            f"standalone CUDA decode requires GQA=4; got QH={query.shape[1]}, "
            f"KVH={cache.kv_heads}"
        )

    dense_key, dense_value = _append_decode_tokens(
        cache,
        decode_key,
        decode_value,
        bool(update_cache),
    )
    scale = (
        1.0 / math.sqrt(float(query.shape[-1]))
        if attention_scale is None
        else float(attention_scale)
    )
    return _decode_cuda(
        query,
        cache,
        dense_key,
        dense_value,
        scale=scale,
        use_pruning=bool(use_pruning),
        prune_mode=int(prune_mode),
        log_eps=float(log_eps),
    )


__all__ = ["decode_attention"]
