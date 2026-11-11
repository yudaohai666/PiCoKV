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
from rebuttal.BPC.source_encoding import find_binary_projection, binary_project_unpacked


def fit_binary_projection(target: torch.Tensor, bits: int, *, seed: int | None = None, num_iters: int = 4):
    """Call the verbatim source, including global RNG and temporary fitting codes.

    ``seed`` is accepted for old callers only and deliberately unused. Seed the
    global generator at the experiment entry, just as in the original BPC.
    """
    _, projection = find_binary_projection(target, bits, num_iters=num_iters)
    return projection


def encode_with_projection(target: torch.Tensor, projection: torch.Tensor):
    """Replay BinaryPC's residual encoder; use +1 for stored exact-zero codes.

    BinaryPC's original packing maps sign()==0 to bit 0, i.e. +1. The residual
    updates retain the original sign()==0 behavior. Materializing that storage
    convention explicitly also gives PiCoKV's V packer strict +/-1 inputs.
    """
    unpacked = binary_project_unpacked(target, projection)
    return torch.where(unpacked < 0, -1, 1).to(torch.int8)


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
    projection = fit_binary_projection(target, int(bits))
    codes = encode_with_projection(target, projection)
    # BPC computes EAS errors from the representation actually stored, using
    # reconstruction rather than the fitting/replay residual (zero ties differ).
    reconstructed = torch.einsum("bhln,bhnd->bhld", codes.to(target.dtype), projection)
    error = torch.norm(target - reconstructed, p=2, dim=-1)
    return OnlineQuantizedTensor(codes=codes, decoder=projection, err=error, encoders=projection)


@contextmanager
def binarypc_encoding():
    """Use original BPC coding at 320 bits, without PiCoKV's HFM.

    HFM is applied upstream of the quantizer, so swapping only the quantizer
    would still feed BPC a masked key. Suppress that preprocessing too, even
    for callers that retain a legacy zerorope=20 in their cache config.
    Packing, EAS and decode are still the existing PiCoKV implementation.
    """
    prepare = importlib.import_module("picokv.picokv_standalone.prepare")
    original = prepare.quantize_block32_online
    original_mask = prepare._apply_zerorope_

    def no_hfm(states, zerorope):
        return None

    prepare.quantize_block32_online = quantize_binarypc
    prepare._apply_zerorope_ = no_hfm
    try:
        yield
    finally:
        prepare.quantize_block32_online = original
        prepare._apply_zerorope_ = original_mask
