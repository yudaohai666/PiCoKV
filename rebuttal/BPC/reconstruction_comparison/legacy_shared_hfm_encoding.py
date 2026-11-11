"""BinaryPC's online fit + residual encoding, extended beyond its 64-bit hash.

Reference: src/BPC/bpc/bpc/bpc_pytorch.py::find_binary_projection and
binary_project_pytorch. The original online cache first learns directions, then
re-encodes the original states with those learned directions; it does NOT use
the intermediate fitting codes directly. Both steps are preserved here.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib

import torch

from picokv.picokv_standalone.online import OnlineQuantizedTensor


def fit_binary_projection(target: torch.Tensor, bits: int, *, seed: int, num_iters: int = 4):
    """Same BF16 alternating updates/normalization/residual order as BinaryPC.

    A private generator replaces global RNG use so overlapping K/V CUDA streams
    cannot interleave random draws. Unused intermediate fitting codes are not
    retained (the original online cache discards them immediately).
    """
    batch, heads, length, dim = target.shape
    generator = torch.Generator(device=target.device).manual_seed(int(seed))
    residual = target.clone()
    projection = torch.empty(batch, heads, bits, dim, device=target.device, dtype=target.dtype)
    for bit in range(bits):
        direction = torch.randn(
            batch, heads, dim, device=target.device, dtype=target.dtype, generator=generator,
        )
        for _ in range(num_iters):
            codes = torch.einsum("bhnd,bhd->bhn", residual, direction).sign()
            direction = torch.einsum("bhnd,bhn->bhd", residual, codes)
        direction = direction / length
        residual = residual - torch.einsum("bhn,bhd->bhnd", codes, direction)
        projection[:, :, bit, :] = direction
    return projection


def encode_with_projection(target: torch.Tensor, projection: torch.Tensor):
    """Replay BinaryPC's residual encoder; use +1 for stored exact-zero codes.

    BinaryPC's original packing maps sign()==0 to bit 0, i.e. +1. The residual
    updates retain the original sign()==0 behavior. Materializing that storage
    convention explicitly also gives PiCoKV's V packer strict +/-1 inputs.
    """
    batch, heads, length, _ = target.shape
    bits = projection.shape[-2]
    codes = torch.empty(batch, heads, length, bits, device=target.device, dtype=torch.int8)
    residual = target.clone()
    for bit in range(bits):
        direction = projection[:, :, bit, :]
        signs = torch.einsum("bhld,bhd->bhl", residual, direction).sign()
        residual = residual - torch.einsum("bhl,bhd->bhld", signs, direction)
        codes[:, :, :, bit] = torch.where(signs < 0, -1, 1).to(torch.int8)
    return codes


@torch.no_grad()
def quantize_binarypc(
    target: torch.Tensor,
    bits: int,
    *,
    direction_sample_size: int | None = 8192,
    seed: int = 0,
    encoders: torch.Tensor | None = None,
    decoders: torch.Tensor | None = None,
) -> OnlineQuantizedTensor:
    """Drop-in for prepare.quantize_block32_online, using full-context BinaryPC.

    The sampling option belongs to GPC and is deliberately unused: BinaryPC
    learns each bit from every token, with four alternating updates. No ridge,
    block QR, extra decoder refit, or additional sampling is introduced.
    """
    if target.ndim != 4 or target.shape[-2] == 0:
        raise ValueError("target must be nonempty [B,H,L,D]")
    if target.dtype != torch.bfloat16:
        raise TypeError("BinaryPC/PiCoKV comparison uses BF16 target tensors")
    if bits <= 0 or bits % 64:
        raise ValueError("bits must be a positive multiple of 64")
    if encoders is not None or decoders is not None:
        raise ValueError("this BinaryPC ablation is online; GPC offline projections are incompatible")
    projection = fit_binary_projection(target, int(bits), seed=seed)
    codes = encode_with_projection(target, projection)
    # BPC computes EAS errors from the representation actually stored, using
    # reconstruction rather than the fitting/replay residual (zero ties differ).
    reconstructed = torch.einsum("bhln,bhnd->bhld", codes.to(target.dtype), projection)
    error = torch.norm(target - reconstructed, p=2, dim=-1)
    return OnlineQuantizedTensor(codes=codes, decoder=projection, err=error, encoders=projection)


@contextmanager
def binarypc_encoding():
    """Replace only the quantizer reference looked up by the existing prepare."""
    prepare = importlib.import_module("picokv.picokv_standalone.prepare")
    original = prepare.quantize_block32_online
    prepare.quantize_block32_online = quantize_binarypc
    try:
        yield
    finally:
        prepare.quantize_block32_online = original
