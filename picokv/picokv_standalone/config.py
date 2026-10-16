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
    key_bits: int = 320
    value_bits: int = 320
    direction_sample_size: int | None = 8192
    sink_size: int = 64
    local_size: int = 256
    # EAS dense retention is OFF by default (user directive 2026-10-14):
    # window = sink+local exactly.  It also unlocks incremental chunk-append
    # (no cross-chunk error selection).  Set >0 to re-enable; prepare
    # validates [0, 0.5).
    eas: float = 0.0
    zerorope: int = 20
    exp_bits: int = 4
    seed: int = 0

    # Use the fully fused CUDA cascade kernel for sides with a stored decoder.
    # Numerically equivalent to the reference loop up to fp32 accumulation
    # order; ~5x faster prefill preparation.  PICOKV_DISABLE_FUSED_QUANT=1
    # overrides this globally.
    use_fused_quant: bool = True

    # Per-token CUDA decode. Pruning is off by default: it is a decode-time
    # approximation, and every benchmark run keeps the global scan exact.
    use_pruning: bool = False
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
            "use_fused_quant": self.use_fused_quant,
        }

    def decode_kwargs(self) -> dict[str, object]:
        """Return only options consumed by each decode-attention call."""
        return {
            "use_pruning": self.use_pruning,
            "prune_mode": self.prune_mode,
            "log_eps": self.log_eps,
        }
