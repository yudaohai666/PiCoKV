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


def _sample_tokens(target: torch.Tensor, sample_size: int | None) -> torch.Tensor:
    """Evenly spaced token subset, matching GPC's online._sample_residual.

    Same linspace spacing and the same ">= length means no sampling" rule, so
    both arms see an identical fit set for a given direction_sample_size. Acts
    on dim -2 of a [B,H,L,D] target instead of dim 1 of GPC's [rows,L,D] view.
    """
    length = target.shape[-2]
    if sample_size is None or int(sample_size) >= length:
        return target
    if int(sample_size) <= 0:
        raise ValueError("direction_sample_size must be positive or None")
    indices = torch.linspace(
        0,
        length - 1,
        steps=int(sample_size),
        device=target.device,
    ).to(torch.long)
    return target.index_select(-2, indices).contiguous()


@torch.no_grad()
def quantize_binarypc_sampled(
    target: torch.Tensor,
    bits: int,
    *,
    direction_sample_size: int | None = 8192,
    seed: int = 0,
    encoders: torch.Tensor | None = None,
    decoders: torch.Tensor | None = None,
) -> OnlineQuantizedTensor:
    """quantize_binarypc with the direction fit honouring direction_sample_size.

    Completes the 2x2 of {GPC, BPC} x {fit on direction_sample_size tokens, fit
    on all L}. The sample size arrives through the same config field prepare
    already passes to the GPC quantizer, so the arms are sample-matched by
    construction rather than by a separate flag.

    Only the fit is sampled. Encoding still replays the learned directions over
    all L tokens: binary_project_unpacked is O(L) and unsamplable because every
    token needs a code. This split is BPC's own offline mode --
    src/BPC/find_binary_projection/calibrate_projection.py fits from 100 samples
    at max_length 4096 and applies the stored projection at any length -- so no
    new degree of freedom is introduced. find_binary_projection reads L from
    A.shape and normalizes Vh by it, so the sampled fit divides by the sample
    size automatically.

    Caveat when reading the result: this samples strictly more of BPC than GPC
    samples of itself. GPC refits its per-group ridge decoder on all L tokens
    (online.py:310-313) and samples only the power iteration, whereas BPC's
    decoder IS the projection, so sampling degrades its directions and its
    decoder together. Pair this arm with a ridge-decoder arm before attributing
    any difference to the directions alone.
    """
    if target.ndim != 4 or target.shape[-2] == 0:
        raise ValueError("target must be nonempty [B,H,L,D]")
    if target.dtype != torch.bfloat16:
        raise TypeError("BinaryPC/PiCoKV comparison uses BF16 target tensors")
    if bits <= 0 or bits % 64:
        raise ValueError("bits must be a positive multiple of 64")
    if encoders is not None or decoders is not None:
        raise ValueError("this BinaryPC ablation is online; GPC offline projections are incompatible")
    projection = fit_binary_projection(_sample_tokens(target, direction_sample_size), int(bits))
    codes = encode_with_projection(target, projection)
    reconstructed = torch.einsum("bhln,bhnd->bhld", codes.to(target.dtype), projection)
    error = torch.norm(target - reconstructed, p=2, dim=-1)
    return OnlineQuantizedTensor(codes=codes, decoder=projection, err=error, encoders=projection)


@contextmanager
def binarypc_encoding(*, sample_directions: bool = False):
    """Use original BPC coding at 320 bits, without PiCoKV's HFM.

    HFM is applied upstream of the quantizer, so swapping only the quantizer
    would still feed BPC a masked key. Suppress that preprocessing too, even
    for callers that retain a legacy zerorope=20 in their cache config.
    Packing, EAS and decode are still the existing PiCoKV implementation.

    ``sample_directions`` selects the fit-on-direction_sample_size variant and
    defaults to False, leaving the already-measured full-L arm untouched.
    """
    prepare = importlib.import_module("picokv.picokv_standalone.prepare")
    original = prepare.quantize_block32_online
    original_mask = prepare._apply_zerorope_

    def no_hfm(states, zerorope):
        return None

    prepare.quantize_block32_online = (
        quantize_binarypc_sampled if sample_directions else quantize_binarypc
    )
    prepare._apply_zerorope_ = no_hfm
    try:
        yield
    finally:
        prepare.quantize_block32_online = original
        prepare._apply_zerorope_ = original_mask
