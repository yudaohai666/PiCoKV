"""Online block-32 projection learning used by standalone cache preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .constants import (
    ONLINE_DECODER_RIDGE,
    ONLINE_GROUP_BITS,
    ONLINE_MATMUL_DTYPE,
)


@dataclass(frozen=True, slots=True)
class OnlineQuantizedTensor:
    """Binary codes, decoder projection, per-token residual error, and encoders."""

    codes: torch.Tensor
    decoder: torch.Tensor
    err: torch.Tensor
    encoders: torch.Tensor


def _householder_qr(x: torch.Tensor) -> torch.Tensor:
    """Return stable reduced-Q after the covariance power iteration."""
    q, _ = torch.linalg.qr(x.float(), mode="reduced")
    return q


def _cholesky_qr(x: torch.Tensor, *, passes: int = 1) -> torch.Tensor:
    """Orthonormalize random initialization blocks with batched CholeskyQR.

    Initialization matrices are well-conditioned and need one pass in the
    production path.  ``passes`` remains internal so the regression test can
    verify two-pass orthogonality on deliberately ill-conditioned inputs.
    """
    q = x.float()
    for _ in range(max(1, int(passes))):
        gram = torch.matmul(q.transpose(-1, -2), q)
        factor, _ = torch.linalg.cholesky_ex(gram, check_errors=False)
        q = torch.linalg.solve_triangular(
            factor,
            q.transpose(-1, -2),
            upper=False,
        ).transpose(-1, -2)
    return q


def _strict_binary_sign_(scores: torch.Tensor) -> torch.Tensor:
    """Map scores in place to {-1,+1}, with exact zero mapped to +1."""
    scores.sign_()
    scores.masked_fill_(scores == 0, 1)
    return scores


def _solve_decoder_from_matrices(
    codes: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Fit the regularized decoder for one block without host synchronization."""
    length = int(residual.shape[-2])
    gram = torch.matmul(codes.transpose(-1, -2), codes).float() / float(length)
    rhs = torch.matmul(codes.transpose(-1, -2), residual).float() / float(length)
    gram.diagonal(dim1=-2, dim2=-1).add_(ONLINE_DECODER_RIDGE)
    solution, _ = torch.linalg.solve_ex(gram, rhs, check_errors=False)
    return solution


def _sample_residual(
    residual: torch.Tensor,
    direction_sample_size: Optional[int],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return evenly sampled residual rows and their indices, or the full input."""
    if (
        direction_sample_size is None
        or int(direction_sample_size) >= residual.shape[-2]
    ):
        return residual, None
    sample_size = int(direction_sample_size)
    if sample_size <= 0:
        raise ValueError("direction_sample_size must be positive or None")
    indices = torch.linspace(
        0,
        residual.shape[-2] - 1,
        steps=sample_size,
        device=residual.device,
    ).to(torch.long)
    return residual.index_select(1, indices).contiguous(), indices


def _prepare_fixed_block32(
    encoders: torch.Tensor,
    batch: int,
    heads: int,
    bits: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize a stored ``[H,bits,D]``/``[B,H,bits,D]`` encoder or decoder.

    Encoders and decoders share this layout: both are ``bits`` rows of length
    ``dim``, consumed one 32-row group at a time.
    """
    if encoders.dim() == 3:
        encoders = encoders.unsqueeze(0).expand(batch, heads, bits, dim)
    if encoders.dim() != 4:
        raise ValueError(
            f"stored matrix must be [H,bits,D] or [B,H,bits,D]; got {tuple(encoders.shape)}"
        )
    if tuple(encoders.shape[1:]) != (heads, bits, dim):
        raise ValueError(
            f"encoders shape {tuple(encoders.shape)} does not match "
            f"target [B={batch},H={heads},bits={bits},D={dim}]"
        )
    return (
        encoders.to(device=device, dtype=dtype)
        .reshape(batch * heads, bits // ONLINE_GROUP_BITS, ONLINE_GROUP_BITS, dim)
        .contiguous()
    )


def quantize_block32_online(
    target: torch.Tensor,
    bits: int,
    *,
    direction_sample_size: Optional[int] = 8192,
    seed: int = 0,
    encoders: Optional[torch.Tensor] = None,
    decoders: Optional[torch.Tensor] = None,
) -> OnlineQuantizedTensor:
    """Learn block-32 directions and immediately quantize ``[B,H,L,D]`` states.

    This is the single validated production path: batched CholeskyQR
    initialization, one covariance power iteration with Householder QR, bf16
    tensor-core matmuls, a ``1e-3`` decoder ridge, and residual error for EAS.

    Passing ``encoders`` reuses stored directions instead of learning them
    (offline path): CholeskyQR init, covariance power iteration, and Householder
    QR are skipped.

    Passing ``decoders`` additionally reuses a stored decoder, skipping the
    per-group least-squares solve entirely. Stored decoders are only valid
    alongside the encoders they were fitted with, because the residual chain
    that produces the codes depends on both.
    """
    if target.dim() != 4:
        raise ValueError(f"target must be [B,H,L,D]; got {tuple(target.shape)}")
    if target.dtype != torch.bfloat16:
        raise TypeError(f"target must be bfloat16; got {target.dtype}")
    bits = int(bits)
    if bits <= 0 or bits % ONLINE_GROUP_BITS:
        raise ValueError(
            f"bits must be a positive multiple of {ONLINE_GROUP_BITS}; got {bits}"
        )

    batch, heads, length, dim = target.shape
    rows = batch * heads
    groups = bits // ONLINE_GROUP_BITS
    fixed_encoders = (
        None
        if encoders is None
        else _prepare_fixed_block32(
            encoders, batch, heads, bits, dim, target.device, target.dtype
        )
    )
    if decoders is not None and fixed_encoders is None:
        raise ValueError("decoders require the matching encoders to be supplied")
    fixed_decoders = (
        None
        if decoders is None
        else _prepare_fixed_block32(
            decoders, batch, heads, bits, dim, target.device, target.dtype
        )
    )
    generator = torch.Generator(device=target.device)
    generator.manual_seed(int(seed))
    init_vectors = torch.randn(
        batch,
        heads,
        bits,
        dim,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )

    residual = target.reshape(rows, length, dim).contiguous().clone()
    if fixed_encoders is None:
        init_columns = (
            init_vectors.reshape(rows, groups, ONLINE_GROUP_BITS, dim)
            .transpose(-1, -2)
            .contiguous()
            .reshape(rows * groups, dim, ONLINE_GROUP_BITS)
        )
        init_q = _cholesky_qr(init_columns).reshape(
            rows,
            groups,
            dim,
            ONLINE_GROUP_BITS,
        )
        sample_residual, sample_indices = _sample_residual(
            residual,
            direction_sample_size,
        )
    else:
        # Fixed encoders make the power iteration and sampling dead work.
        sample_residual, sample_indices = None, None

    codes = torch.empty(rows, length, bits, device=target.device, dtype=torch.int8)
    decoder = torch.empty(rows, bits, dim, device=target.device, dtype=target.dtype)
    used_encoders = torch.empty(rows, bits, dim, device=target.device, dtype=target.dtype)

    for group_idx in range(groups):
        start = group_idx * ONLINE_GROUP_BITS
        end = start + ONLINE_GROUP_BITS

        # One covariance power step estimates the dominant residual subspace;
        # Householder QR preserves accuracy after residual groups accumulate.
        if fixed_encoders is None:
            q = init_q[:, group_idx]
            sampled = sample_residual.to(dtype=ONLINE_MATMUL_DTYPE)
            y = torch.matmul(sampled, q.to(dtype=ONLINE_MATMUL_DTYPE))
            q = torch.matmul(sampled.transpose(-1, -2), y)
            encoder = _householder_qr(q).transpose(-1, -2).contiguous()
        else:
            encoder = fixed_encoders[:, group_idx]
        used_encoders[:, start:end] = encoder.to(dtype=target.dtype)

        # Quantize to strict {-1,+1}, fit this group's decoder, then subtract
        # its reconstruction before learning the next group.
        residual_m = residual.to(dtype=ONLINE_MATMUL_DTYPE)
        group_codes_m = torch.matmul(
            residual_m,
            encoder.to(dtype=ONLINE_MATMUL_DTYPE).transpose(-1, -2),
        )
        _strict_binary_sign_(group_codes_m)
        if fixed_decoders is None:
            group_decoder = _solve_decoder_from_matrices(group_codes_m, residual_m)
        else:
            group_decoder = fixed_decoders[:, group_idx]
        decoder_m = group_decoder.to(dtype=ONLINE_MATMUL_DTYPE)

        torch.baddbmm(
            residual,
            group_codes_m,
            decoder_m,
            beta=1.0,
            alpha=-1.0,
            out=residual,
        )
        if sample_indices is not None:
            sample_codes = group_codes_m.index_select(1, sample_indices)
            torch.baddbmm(
                sample_residual,
                sample_codes,
                decoder_m,
                beta=1.0,
                alpha=-1.0,
                out=sample_residual,
            )

        codes[:, :, start:end] = group_codes_m.to(dtype=torch.int8)
        decoder[:, start:end] = group_decoder.to(dtype=target.dtype)

    return OnlineQuantizedTensor(
        codes=codes.reshape(batch, heads, length, bits).contiguous(),
        decoder=decoder.reshape(batch, heads, bits, dim).contiguous(),
        err=residual.float().norm(dim=-1).reshape(batch, heads, length).contiguous(),
        encoders=used_encoders.reshape(batch, heads, bits, dim).contiguous(),
    )


__all__ = ["OnlineQuantizedTensor", "quantize_block32_online"]
