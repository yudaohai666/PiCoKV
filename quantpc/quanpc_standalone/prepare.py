"""Online projection learning, EAS selection, and packed-cache construction.

This module owns the complete prefill-to-cache pipeline.  It does not contain
decode logic, model integration, or alternative projection implementations.
"""

from __future__ import annotations

from typing import Optional

import torch

from .cache import PreparedKVCache
from .constants import SUPPORTED_CODE_BITS, SUPPORTED_EXP_BITS
from .online import quantize_block32_online

try:
    from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import (
        gather_pack_from_packed_cuda,
        pack_key_transposed_cuda,
        pack_value_token_cuda,
    )
except Exception:  # pragma: no cover - reported by _require_prefill_cuda
    gather_pack_from_packed_cuda = None
    pack_key_transposed_cuda = None
    pack_value_token_cuda = None


# Persistent per-device side-stream for the K/V prepare overlap.
#
# Creating a fresh torch.cuda.Stream() on every prepare_kv_cache call (once per
# layer) makes the caching allocator reserve blocks per-stream that later layers
# cannot reuse: freed blocks stay pinned to the now-dead stream, so the reserved
# pool balloons ~1.3GB/layer (72GB reserved vs 26GB live at L=131072). Reusing a
# single stream per device lets those blocks be reused across layers. Layers are
# already serialized at each prepare boundary (the trailing wait_stream), so this
# changes neither the intra-layer K/V overlap nor the numerics.
_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    key = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SIDE_STREAMS[key] = stream
    return stream


def _validate_code_bits(name: str, bits: int) -> int:
    """Return a normalized code width accepted by the CUDA kernels."""
    bits = int(bits)
    if bits not in SUPPORTED_CODE_BITS:
        raise ValueError(f"{name} must be one of {SUPPORTED_CODE_BITS}; got {bits}")
    return bits


def _validate_exp_bits(exp_bits: int) -> int:
    """Return a normalized exponent-code width accepted by decode."""
    exp_bits = int(exp_bits)
    if exp_bits not in SUPPORTED_EXP_BITS:
        raise ValueError(f"exp_bits must be one of {SUPPORTED_EXP_BITS}; got {exp_bits}")
    return exp_bits


def _require_prefill_cuda() -> None:
    """Fail early when any required packing/gather helper is unavailable."""
    if any(
        helper is None
        for helper in (
            gather_pack_from_packed_cuda,
            pack_key_transposed_cuda,
            pack_value_token_cuda,
        )
    ):
        raise RuntimeError("QuanPC prefill CUDA extension is unavailable")


def _global_span(total_len: int, sink_size: int, local_size: int) -> tuple[int, int]:
    """Return ``(start, length)`` of the portion compressed into 1-bit codes."""
    global_len = int(total_len) - int(sink_size) - int(local_size)
    if sink_size < 0 or local_size < 0 or global_len <= 0:
        raise ValueError(
            f"invalid sink/local sizes for total_len={total_len}: "
            f"sink_size={sink_size}, local_size={local_size}"
        )
    return int(sink_size), global_len


def _apply_zerorope_(states: torch.Tensor, zerorope: int) -> None:
    """Zero paired RoPE dimensions in-place before learning key directions."""
    if zerorope <= 0:
        return
    half = states.shape[-1] // 2
    if zerorope > half:
        raise ValueError(f"zerorope ({zerorope}) must be <= half head dim ({half})")
    states[..., :zerorope] = 0
    states[..., half : half + zerorope] = 0


def _select_eas_indices(
    key_error: torch.Tensor,
    value_error: torch.Tensor,
    eas_size: int,
) -> torch.Tensor:
    """Select the union of highest-error K and V tokens for dense retention.

    K errors are selected first.  V then selects from the remaining positions,
    preventing duplicate EAS tokens while keeping the final indices sorted for
    coalesced gather.
    """
    length = int(key_error.shape[-1])
    key_count = min(int(eas_size), length)
    key_indices = torch.topk(key_error, k=key_count, dim=-1).indices
    if key_count == length:
        return key_indices.sort(dim=-1).values

    value_count = min(int(eas_size), length - key_count)
    selected = torch.zeros_like(value_error, dtype=torch.bool).scatter_(
        -1,
        key_indices,
        True,
    )
    value_indices = torch.topk(
        value_error.masked_fill(selected, -torch.inf),
        k=value_count,
        dim=-1,
    ).indices
    return torch.cat([key_indices, value_indices], dim=-1).sort(dim=-1).values


def _invert_indices(drop_indices: torch.Tensor, length: int) -> torch.Tensor:
    """Build sorted int32 indices of global tokens that remain compressed."""
    drop_indices = drop_indices.to(torch.long)
    mask = torch.ones(
        *drop_indices.shape[:2],
        int(length),
        device=drop_indices.device,
        dtype=torch.bool,
    )
    mask.scatter_(-1, drop_indices, False)
    positions = torch.arange(
        int(length),
        device=drop_indices.device,
        dtype=torch.int32,
    ).view(1, 1, int(length)).expand_as(mask)
    kept_len = int(length) - int(drop_indices.shape[-1])
    return positions[mask].reshape(*drop_indices.shape[:2], kept_len).contiguous()


def _validate_prepare_inputs(
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    eas: float,
    direction_sample_size: Optional[int],
) -> None:
    """Validate public prepare inputs before enqueuing CUDA work."""
    if full_key.dim() != 4 or tuple(full_key.shape) != tuple(full_value.shape):
        raise ValueError("full_key/full_value must have the same [B,H,L,D] shape")
    if not full_key.is_cuda or not full_value.is_cuda:
        raise ValueError("prepare_kv_cache requires CUDA full_key/full_value")
    if full_key.device != full_value.device:
        raise ValueError("full_key/full_value must be on the same CUDA device")
    if full_key.dtype != torch.bfloat16 or full_value.dtype != torch.bfloat16:
        raise TypeError("prepare_kv_cache requires bfloat16 full_key/full_value")
    if not 0.0 < float(eas) < 0.5:
        raise ValueError(f"eas must be in (0, 0.5); got {eas}")
    if direction_sample_size is not None and int(direction_sample_size) <= 0:
        raise ValueError("direction_sample_size must be positive or None")


def prepare_kv_cache(
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    *,
    key_bits: int = 256,
    value_bits: int = 256,
    direction_sample_size: Optional[int] = 8192,
    sink_size: int = 64,
    local_size: int = 1024,
    eas: float = 0.01,
    zerorope: int = 20,
    exp_bits: int = 4,
    seed: int = 0,
) -> PreparedKVCache:
    """Learn online K/V projections and build a packed CUDA cache.

    Inputs are bf16 CUDA tensors shaped ``[B, KVH, L, D]``.  K and V online
    learning overlap on two CUDA streams; the caller stream joins both before
    packing, EAS selection, and retained-cache gathering.  The function does
    not synchronize the host.
    """
    _validate_prepare_inputs(full_key, full_value, eas, direction_sample_size)
    _require_prefill_cuda()
    key_bits = _validate_code_bits("key_bits", key_bits)
    value_bits = _validate_code_bits("value_bits", value_bits)
    exp_bits = _validate_exp_bits(exp_bits)

    # Split exact dense windows from the global segment learned/compressed here.
    global_start, global_len = _global_span(
        full_key.shape[-2],
        int(sink_size),
        int(local_size),
    )
    key_global = full_key[:, :, global_start : global_start + global_len].clone()
    _apply_zerorope_(key_global, int(zerorope))
    value_global = full_value[
        :,
        :,
        global_start : global_start + global_len,
    ].contiguous()

    # This internal K/V overlap is part of the measured fast path.  It is not
    # the removed async-prepare/prefill overlap API.
    current_stream = torch.cuda.current_stream(full_key.device)
    value_stream = _side_stream(full_key.device)
    value_stream.wait_stream(current_stream)
    with torch.cuda.stream(value_stream):
        value_block = quantize_block32_online(
            value_global,
            value_bits,
            direction_sample_size=direction_sample_size,
            seed=int(seed) + 1009,
        )
    key_block = quantize_block32_online(
        key_global,
        key_bits,
        direction_sample_size=direction_sample_size,
        seed=int(seed),
    )
    current_stream.wait_stream(value_stream)

    # Pack all codes once.  The fused gather then moves EAS tokens into the
    # dense cache and removes the same positions from the packed global cache.
    packed_key_all = pack_key_transposed_cuda(key_block.codes)
    packed_value_all = pack_value_token_cuda(value_block.codes)
    eas_size = max(1, int(global_len * float(eas)))
    eas_indices = _select_eas_indices(
        key_block.err,
        value_block.err,
        eas_size,
    ).to(torch.int32).contiguous()
    kept_indices = _invert_indices(eas_indices, global_len)
    dense_key, dense_value, packed_key, packed_value = (
        gather_pack_from_packed_cuda(
            full_key,
            full_value,
            packed_key_all,
            packed_value_all,
            eas_indices,
            kept_indices,
            int(sink_size),
            int(local_size),
        )
    )

    # Decode consumes K projection transposed, so materialize it once here.
    key_projection = key_block.decoder.contiguous()
    return PreparedKVCache(
        dense_key=dense_key,
        dense_value=dense_value,
        key_projection=key_projection,
        value_projection=value_block.decoder.contiguous(),
        key_projection_t=key_projection.transpose(-1, -2).contiguous(),
        packed_key=packed_key,
        packed_value=packed_value,
        exp_bits=exp_bits,
    )


__all__ = ["prepare_kv_cache"]
