"""Dequantize-and-reconstruct ablation path (no CUDA decode kernel).

This path exists to isolate the accuracy cost of quantization from the cost of
the packed-code decode kernel.  It runs the *same* block-32 quantization, the
*same* EAS token selection, and the *same* zerorope as the production
:func:`..prepare.prepare_kv_cache`, but instead of packing codes and serving
decode through the custom CUDA kernel it reconstructs the compressed sides back
to bf16 (``codes @ decoder``) and hands full-precision ``K'``/``V'`` to plain
SDPA/FlashAttention.

``quant_target`` selects which side(s) are degraded to their reconstruction:

- ``"key"``   -- only K is quantized+reconstructed; V stays exact.
- ``"value"`` -- only V is quantized+reconstructed; K stays exact.
- ``"both"``  -- both sides reconstructed (pure quantization loss, no kernel).
- ``"none"``  -- neither; returns the exact inputs (dense-baseline self-check).
- ``"dense_zr"`` -- no quantization at all; only applies zerorope to the global K
  segment (exactly where the production path zeroes RoPE dims) and serves dense
  SDPA.  Isolates zerorope's own effect on attention quality: the existing
  ``dense_flash_attention_2`` baseline is its ``zerorope=0`` counterpart, so the
  pair differs only by zerorope on global K, with no quantization noise.

To keep the dense/global token partition byte-identical to full PiCoKV across
all modes, EAS still selects the union of high-error K and V tokens whenever
either side is quantized -- so both sides are quantized during prefill purely to
produce the errors, even though only the requested side's values are written.
Only the *global* (non-EAS) tokens of a quantized side are replaced; sink,
local, and EAS tokens keep their exact values, exactly as in the kernel path.
"""

from __future__ import annotations

from typing import Optional

import torch

from .online import quantize_block32_online, OnlineQuantizedTensor
from .prepare import (
    _apply_zerorope_,
    _global_span,
    _select_eas_and_kept_indices,
)

QUANT_TARGETS = ("key", "value", "both", "none", "dense_zr")


def _overwrite_kept_with_recon(
    dst: torch.Tensor,
    block: OnlineQuantizedTensor,
    kept_indices: torch.Tensor,
    global_start: int,
) -> None:
    """Replace the kept (compressed) global positions of ``dst`` with ``codes @ decoder``.

    ``dst`` is a full-length ``[B,H,L,D]`` clone; ``kept_indices`` are the
    global-segment positions (relative to ``global_start``) that stay compressed
    in full PiCoKV.  EAS/sink/local positions are left untouched (exact).
    """
    if kept_indices.shape[-1] == 0:
        return
    dim = dst.shape[-1]
    recon_global = torch.matmul(block.codes.to(dst.dtype), block.decoder)
    kept = kept_indices.to(torch.long)
    gather_idx = kept.unsqueeze(-1).expand(*kept.shape, dim)
    kept_recon = torch.gather(recon_global, 2, gather_idx)
    scatter_idx = (kept + int(global_start)).unsqueeze(-1).expand(*kept.shape, dim)
    dst.scatter_(2, scatter_idx, kept_recon)


def build_recon_dense_cache(
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    *,
    quant_target: str,
    key_bits: int,
    value_bits: int,
    direction_sample_size: Optional[int],
    sink_size: int,
    local_size: int,
    eas: float,
    zerorope: int,
    seed: int,
    key_encoders: Optional[torch.Tensor] = None,
    value_encoders: Optional[torch.Tensor] = None,
    key_decoders: Optional[torch.Tensor] = None,
    value_decoders: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-precision ``(K', V')`` with the requested side(s) reconstructed.

    Inputs are bf16 CUDA ``[B, KVH, L, D]`` tensors in the model's own K/V basis,
    identical to :func:`..prepare.prepare_kv_cache`.  The returned tensors are
    also ``[B, KVH, L, D]`` and are meant to be served by plain SDPA at decode.
    """
    if quant_target not in QUANT_TARGETS:
        raise ValueError(f"quant_target must be one of {QUANT_TARGETS}; got {quant_target!r}")
    if quant_target == "none":
        return full_key, full_value

    global_start, global_len = _global_span(
        full_key.shape[-2], int(sink_size), int(local_size)
    )

    if quant_target == "dense_zr":
        # Dense zerorope ablation: no quantization at all, just zero the global
        # K RoPE dims and serve SDPA. zerorope=0 reproduces the flash baseline.
        recon_key = full_key.clone()
        key_global = recon_key[:, :, global_start : global_start + global_len].clone()
        _apply_zerorope_(key_global, int(zerorope))
        recon_key[:, :, global_start : global_start + global_len] = key_global
        return recon_key, full_value

    quant_key = quant_target in ("key", "both")
    quant_value = quant_target in ("value", "both")

    # Zerorope only touches the global K clone that gets quantized, matching the
    # production path; the exact K'/V' positions keep their full RoPE.
    key_global = full_key[:, :, global_start : global_start + global_len].clone()
    _apply_zerorope_(key_global, int(zerorope))
    value_global = full_value[
        :, :, global_start : global_start + global_len
    ].contiguous()

    # Both sides are quantized regardless of quant_target so EAS sees both errors
    # and picks the same tokens as full PiCoKV.
    key_block = quantize_block32_online(
        key_global,
        key_bits,
        direction_sample_size=direction_sample_size,
        seed=int(seed),
        encoders=key_encoders,
        decoders=key_decoders,
    )
    value_block = quantize_block32_online(
        value_global,
        value_bits,
        direction_sample_size=direction_sample_size,
        seed=int(seed) + 1009,
        encoders=value_encoders,
        decoders=value_decoders,
    )

    eas_size = max(1, int(global_len * float(eas)))
    _eas_indices, kept_indices = _select_eas_and_kept_indices(
        key_block.err, value_block.err, eas_size
    )

    recon_key = full_key
    recon_value = full_value
    if quant_key:
        recon_key = full_key.clone()
        _overwrite_kept_with_recon(recon_key, key_block, kept_indices, global_start)
    if quant_value:
        recon_value = full_value.clone()
        _overwrite_kept_with_recon(
            recon_value, value_block, kept_indices, global_start
        )
    return recon_key, recon_value


__all__ = ["build_recon_dense_cache", "QUANT_TARGETS"]
