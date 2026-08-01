"""Prepared cache data shared by cache preparation and decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PreparedKVCache:
    """Immutable compressed state plus mutable dense decode state.

    ``dense_key`` and ``dense_value`` grow when decode appends generated
    tokens.  Projection and packed-global tensors remain fixed.  Decode-owned
    workspace and stream objects are cached here so steady-state token latency
    does not pay allocation/setup costs repeatedly.
    """

    dense_key: torch.Tensor
    dense_value: torch.Tensor
    key_projection: torch.Tensor
    value_projection: torch.Tensor
    key_projection_t: torch.Tensor
    packed_key: torch.Tensor
    packed_value: torch.Tensor
    exp_bits: int
    _workspace: dict[tuple, tuple[torch.Tensor, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    _dense_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)

    @property
    def device(self) -> torch.device:
        """CUDA device that owns every persistent cache tensor."""
        return self.dense_key.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype used by dense states and projections."""
        return self.dense_key.dtype

    @property
    def kv_heads(self) -> int:
        """Number of key/value heads stored by this cache."""
        return int(self.dense_key.shape[1])

    @property
    def head_dim(self) -> int:
        """Per-head hidden dimension of K, V, and decoder output."""
        return int(self.dense_key.shape[-1])

    @property
    def key_bits(self) -> int:
        """Number of learned binary directions used for keys."""
        return int(self.key_projection.shape[-2])

    @property
    def value_bits(self) -> int:
        """Number of learned binary directions used for values."""
        return int(self.value_projection.shape[-2])

    @property
    def dense_len(self) -> int:
        """Number of exact retained tokens, including appended decode K/V."""
        return int(self.dense_key.shape[-2])

    @property
    def global_len(self) -> int:
        """Number of compressed global tokens represented by packed keys."""
        return int(self.packed_key.shape[-1])


__all__ = ["PreparedKVCache"]
