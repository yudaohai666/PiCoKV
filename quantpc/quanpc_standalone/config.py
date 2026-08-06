"""Configuration shared by standalone cache preparation and decode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StandaloneConfig:
    """User-facing settings for the single production standalone path.

    ``prepare_kwargs`` and ``decode_kwargs`` intentionally return separate
    dictionaries so cache-construction and per-token options cannot leak into
    each other's APIs.
    """

    # Cache construction / online projection.
    key_bits: int = 256
    value_bits: int = 256
    direction_sample_size: int | None = 8192
    sink_size: int = 64
    local_size: int = 1024
    eas: float = 0.01
    zerorope: int = 20
    exp_bits: int = 4
    seed: int = 0

    # Rotate K into the query-whitened basis and V into the o_proj-whitened one
    # before quantizing. Exact for an unquantized run; it only redistributes
    # quantization error. Applies to both sides and every layer -- an offline
    # projection folds the same rotation into its stored pair instead, which is
    # the configuration to prefer: applied at runtime this costs 5-6% of prefill
    # and wins nothing on either code width.
    whiten: bool = False

    # Per-token CUDA decode.
    use_pruning: bool = True
    prune_mode: int = 3
    log_eps: float = -10.0

    def prepare_kwargs(self) -> dict[str, object]:
        """Return only options consumed while constructing the cache."""
        return {
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "direction_sample_size": self.direction_sample_size,
            "sink_size": self.sink_size,
            "local_size": self.local_size,
            "eas": self.eas,
            "zerorope": self.zerorope,
            "exp_bits": self.exp_bits,
            "seed": self.seed,
        }

    def decode_kwargs(self) -> dict[str, object]:
        """Return only options consumed by each decode-attention call."""
        return {
            "use_pruning": self.use_pruning,
            "prune_mode": self.prune_mode,
            "log_eps": self.log_eps,
        }
