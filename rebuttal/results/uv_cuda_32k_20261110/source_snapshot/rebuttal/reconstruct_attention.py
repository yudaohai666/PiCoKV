"""Unpack binary C, reconstruct BF16 K/V with C @ D, then run dense attention.

The original PreparedKVCache is retained. Unpacked codes and reconstructed K/V
are temporary and are recomputed on every decode call, inside the timed path.
"""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from picokv.picokv_standalone.cache import PreparedKVCache
from picokv.picokv_standalone.decode import _append_decode_tokens


@triton.jit
def _unpack_codes_kernel(
    packed, codes, length: tl.constexpr, bits: tl.constexpr,
    total: tl.constexpr, is_key: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    code = offset % bits
    token = (offset // bits) % length
    row = offset // (length * bits)
    if is_key:
        # K: [B, H, Ck/64, L]; bit 1 represents -1.
        packed_offset = (row * (bits // 64) + code // 64) * length + token
        shift = code % 64
    else:
        # V: [B, H, ceil(L/64), Cv]; bit 1 represents +1.
        packed_offset = (row * triton.cdiv(length, 64) + token // 64) * bits + code
        shift = token % 64
    word = tl.load(packed + packed_offset, offset < total, other=0).to(tl.uint64)
    bit = ((word >> shift.to(tl.uint64)) & 1).to(tl.float32)
    if is_key:
        value = 1.0 - 2.0 * bit
    else:
        value = 2.0 * bit - 1.0
    tl.store(codes + offset, value, offset < total)


def unpack_codes(packed: torch.Tensor, *, length: int, is_key: bool) -> torch.Tensor:
    """Return BF16 +/-1 codes [B,H,L,C] from the native K or V layout.

    CUDA uses one Triton kernel without an expanded int64 intermediate. The CPU
    implementation is only a reference for correctness checks.
    """
    if packed.ndim != 4 or packed.dtype != torch.int64 or not packed.is_contiguous():
        raise ValueError("packed codes must be contiguous int64 [B,H,*,*]")
    if length < 0:
        raise ValueError("length must be nonnegative")
    bits = packed.shape[2] * 64 if is_key else packed.shape[3]
    if is_key and packed.shape[3] != length:
        raise ValueError("packed K length mismatch")
    if not is_key and packed.shape[2] != triton.cdiv(length, 64):
        raise ValueError("packed V token-group count mismatch")
    codes = torch.empty(
        *packed.shape[:2], length, bits, device=packed.device, dtype=torch.bfloat16,
    )
    if codes.numel() == 0:
        return codes
    if packed.is_cuda:
        with torch.cuda.device(packed.device):
            _unpack_codes_kernel[(triton.cdiv(codes.numel(), 1024),)](
                packed, codes, length, bits, codes.numel(), is_key, 1024,
            )
        return codes
    shifts = torch.arange(64, device=packed.device)
    if is_key:
        binary = (packed.transpose(-1, -2).unsqueeze(-1) >> shifts) & 1
        codes.copy_(binary.flatten(-2).to(torch.bfloat16).mul_(-2).add_(1))
    else:
        binary = (packed.unsqueeze(-2) >> shifts[:, None]) & 1
        codes.copy_(binary.flatten(2, 3)[:, :, :length].to(torch.bfloat16).mul_(2).sub_(1))
    return codes


def reconstruct_global_kv(cache: PreparedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct global K/V with BF16 GEMMs, releasing each C after use."""
    key_codes = unpack_codes(cache.packed_key, length=cache.global_len, is_key=True)
    key = torch.matmul(key_codes, cache.key_projection)
    del key_codes
    value_codes = unpack_codes(cache.packed_value, length=cache.global_len, is_key=False)
    value = torch.matmul(value_codes, cache.value_projection)
    del value_codes
    return key, value


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
    """Drop-in single-token decode: unpack -> C @ D -> FlashAttention/SDPA.

    No query/attention-weight quantization or reapplication of RoPE is needed.
    Token order may differ from the prompt: paired K/V have already had RoPE
    applied and single-token attention includes all cached tokens (causal=False).
    """
    if use_pruning:
        raise ValueError("the reconstruction control requires pruning disabled")
    if query.ndim != 4 or query.shape[-2] != 1:
        raise ValueError("query must have shape [B,QH,1,D]")
    if query.device != cache.device or query.dtype != torch.bfloat16 or cache.dtype != query.dtype:
        raise ValueError("query and cache must share a device and use BF16")
    if query.shape[0] != cache.dense_key.shape[0] or query.shape[-1] != cache.head_dim:
        raise ValueError("query batch/head dimension does not match the cache")
    if query.shape[1] != 4 * cache.kv_heads:
        raise ValueError("the comparison uses the production GQA=4 configuration")
    dense_key, dense_value = _append_decode_tokens(cache, decode_key, decode_value, update_cache)
    key, value = reconstruct_global_kv(cache)
    full_key = torch.cat((key, dense_key), dim=-2)
    del key
    full_value = torch.cat((value, dense_value), dim=-2)
    del value
    scale = 1.0 / math.sqrt(cache.head_dim) if attention_scale is None else float(attention_scale)
    kwargs = dict(dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True)
    if query.is_cuda:
        # Force an efficient full-attention backend; never silently expand KV
        # heads or fall back to a materialized QH x L score matrix on CUDA.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(query, full_key, full_value, **kwargs)
    return F.scaled_dot_product_attention(query, full_key, full_value, **kwargs)


@contextmanager
def reconstructed_decode(backend="reconstruct"):
    """Temporarily replace the HF adapter's decode function in this process."""
    from picokv.picokv_standalone import transformers as adapter

    if backend == "reconstruct_cuda":
        from .reconstruct_cuda_attention import decode_attention as replacement
        from .cuda_attention import load_attention_extension

        load_attention_extension()  # Compile/load before any timed requests.
    elif backend == "reconstruct":
        replacement = decode_attention
    elif backend == "unpacked":
        from .unpacked_attention import decode_attention as replacement
    elif backend == "unpacked_kernel":
        from .unpacked_kernel_attention import decode_attention as replacement
    elif backend == "unpacked_fused":
        from .unpacked_kernel_attention import decode_attention_fused as replacement
        from .cuda_attention import load_attention_extension

        load_attention_extension()
    else:
        raise ValueError(f"unknown reconstruction backend: {backend}")
    original = adapter.decode_attention
    adapter.decode_attention = replacement
    try:
        yield
    finally:
        adapter.decode_attention = original
