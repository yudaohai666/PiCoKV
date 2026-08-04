"""Query/output whitening of the K/V bases (the ``find_orth_proj`` transform).

Binary codes spend their budget uniformly across directions, but the downstream
error is not uniform: a K direction only matters as far as the query
distribution reaches into it, and a V direction only as far as ``o_proj`` reads
it out.  Rotating K into ``(Q^T Q)^{1/2}`` and V into ``(W_o^T W_o)^{1/2}``
makes those sensitivities isotropic, so quantization error is spent evenly on
what the model actually consumes.

The transform is an exact change of basis.  With ``A`` symmetric,
``(Q A^{-1})(K A)^T == Q K^T`` and ``(attn V A) A^{-1} == attn V``, so an
unquantized run is numerically unchanged; only the quantization error moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Singular values below this fraction of the largest one are treated as a
# degenerate direction and left unscaled.  Without the floor, the bf16 cast of
# ``1/sqrt(s)`` overflows to inf on rank-deficient heads and poisons the whole
# layer with NaNs.
_SINGULAR_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class LayerWhitening:
    """Symmetric whitening factors and their inverses for one layer."""

    key: torch.Tensor
    key_inverse: torch.Tensor
    value: torch.Tensor
    value_inverse: torch.Tensor


def find_orth_proj(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A, A^-1)`` where ``A = (x^T x)^{1/2}`` is symmetric PSD.

    ``x`` is ``[..., N, D]``; both outputs are ``[..., D, D]`` in ``x``'s dtype.
    """
    gram = torch.matmul(x.transpose(-1, -2), x).float()
    _, singular, v_h = torch.linalg.svd(gram)
    singular = singular.sqrt()
    floor = singular[..., :1] * _SINGULAR_FLOOR
    singular = torch.maximum(singular, floor)
    v = v_h.transpose(-1, -2)
    scale = singular.unsqueeze(-2)
    forward = torch.matmul(v * scale, v_h)
    inverse = torch.matmul(v / scale, v_h)
    return forward.to(x.dtype), inverse.to(x.dtype)


def value_whitening_source(
    output_weight: torch.Tensor,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reshape ``o_proj.weight`` into the per-KV-head readout matrix.

    ``[hidden, kv_heads * group * head_dim]`` becomes
    ``[kv_heads, hidden * group, head_dim]`` so that each KV head's whitening is
    fitted against every output row that reads it.
    """
    grouped = output_weight.reshape(output_weight.shape[0], kv_heads, -1, head_dim)
    return grouped.transpose(0, 1).reshape(kv_heads, -1, head_dim)


def whiten_states(states: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Apply ``states @ factor^T`` over the trailing head dimension."""
    return torch.matmul(states, factor.transpose(-1, -2))


__all__ = [
    "LayerWhitening",
    "find_orth_proj",
    "value_whitening_source",
    "whiten_states",
]
