"""Adapter exposing the standalone block-32 method under the legacy method API.

The benchmark harness imports ``modify``, ``QuantKVCache``, and
``load_offline_projections`` from a method module and calls them with the
argument names the greedy implementation used.  This module keeps that surface
while the algorithm underneath is :mod:`quantpc.standalone`:

- no query-derived whitening; K/V are quantized in raw post-RoPE space,
- block-32 direction learning with a per-group ridge-solved decoder,
- exact SDPA for the prompt, packed CUDA kernels for single-token decode.

Offline projections store only the orthonormal *encoders*.  The decoder is
solved from the current context in every group, which is what makes a fixed
encoder nearly free here: the context-dependent part of the fit stays online.
"""

from __future__ import annotations

from typing import Optional

import torch

from quantpc.standalone.config import StandaloneConfig
from quantpc.standalone.constants import ONLINE_GROUP_BITS, PROJECTED_QUERY_BITS
from quantpc.standalone.transformers import (
    StandaloneLlamaCache,
    modify_llama_for_standalone,
)


OFFLINE_ENCODER_FORMAT = "quanpc_standalone_block32_encoder_v1"


def modify(model, *, mlp_chunk_size: int = 8192) -> None:
    """Patch a Llama model onto the standalone prefill/decode path."""
    modify_llama_for_standalone(model, mlp_chunk_size=int(mlp_chunk_size))


def load_offline_projections(path: str) -> dict:
    """Load stored block-32 encoders and reject incompatible files.

    Greedy-era projection files have the same tensor shape but hold pursuit
    step vectors rather than orthonormal encoders, so loading one here would
    quantize against meaningless directions.  The format tag makes that a hard
    error instead of a silent accuracy loss.
    """
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "key" not in data or "value" not in data:
        raise ValueError("offline projection file must be a dict containing 'key' and 'value'")

    meta = data.get("meta") or {}
    fmt = meta.get("format")
    if fmt != OFFLINE_ENCODER_FORMAT:
        raise ValueError(
            f"{path} has format {fmt!r}, but the standalone method requires "
            f"{OFFLINE_ENCODER_FORMAT!r}. Regenerate it with "
            f"create_offline_projection.py --projection-algo block32."
        )
    for kind in ("key", "value"):
        tensor = data[kind]
        if tensor.dim() != 4:
            raise ValueError(
                f"offline '{kind}' must be [layers, heads, bits, dim]; "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2] % ONLINE_GROUP_BITS:
            raise ValueError(
                f"offline '{kind}' bits {tensor.shape[-2]} is not a multiple of "
                f"{ONLINE_GROUP_BITS}"
            )
    return data


class QuantKVCache(StandaloneLlamaCache):
    """Legacy-named cache that drives the standalone path, with offline support.

    Accepts the greedy implementation's keyword names so the benchmark runner
    does not change.  Parameters that no longer describe a choice are validated
    rather than silently ignored: the standalone path has no torch decode
    backend and no optional decoder refit.
    """

    def __init__(
        self,
        *,
        k_bits: int = 256,
        v_bits: int = 256,
        projected_query_bits: int = PROJECTED_QUERY_BITS,
        exp_bits: int = 4,
        eas: float = 0.01,
        zerorope: int = 20,
        sink_size: int = 64,
        local_size: int = 1024,
        direction_sample_size: Optional[int] = 8192,
        seed: int = 0,
        use_pruning: bool = True,
        prune_mode: int = 3,
        log_eps: float = -10.0,
        offline_projs: Optional[dict] = None,
        offline_proj_mode: str = "both",
        # Accepted for runner compatibility; see the class docstring.
        use_cuda_v1_exp8: bool = True,
        refit_decoder: bool = True,
        refit_solver: str = "ridge",
    ) -> None:
        if int(projected_query_bits) != PROJECTED_QUERY_BITS:
            raise ValueError(
                f"standalone decode packs {PROJECTED_QUERY_BITS}-bit queries; "
                f"got projected_query_bits={projected_query_bits}"
            )
        if not use_cuda_v1_exp8:
            raise ValueError(
                "standalone has no torch decode backend; use_cuda_v1_exp8 must be True"
            )

        super().__init__(
            StandaloneConfig(
                key_bits=int(k_bits),
                value_bits=int(v_bits),
                direction_sample_size=direction_sample_size,
                sink_size=int(sink_size),
                local_size=int(local_size),
                eas=float(eas),
                zerorope=int(zerorope),
                exp_bits=int(exp_bits),
                seed=int(seed),
                use_pruning=bool(use_pruning),
                prune_mode=int(prune_mode),
                log_eps=float(log_eps),
            )
        )
        self.key_encoders, self.value_encoders = _split_offline(
            offline_projs,
            offline_proj_mode,
        )
        self._pending_encoders = None

    def prepare(self, layer_idx, key_states, value_states) -> None:
        """Prepare one layer, injecting that layer's offline encoders if any."""
        layer_idx = int(layer_idx)
        heads = int(key_states.shape[1])
        self._pending_encoders = (
            _layer_encoders(self.key_encoders, layer_idx, heads),
            _layer_encoders(self.value_encoders, layer_idx, heads),
        )
        try:
            super().prepare(layer_idx, key_states, value_states)
        finally:
            self._pending_encoders = None

    def _prepare_kwargs(self) -> dict:
        """Merge config kwargs with the current layer's offline encoders."""
        kwargs = super()._prepare_kwargs()
        if self._pending_encoders is not None:
            kwargs["key_encoders"], kwargs["value_encoders"] = self._pending_encoders
        return kwargs


def _split_offline(offline_projs, mode: str):
    """Return ``(key, value)`` encoder stacks honouring the offline-side mode."""
    if offline_projs is None:
        return None, None
    mode = str(mode)
    if mode not in ("both", "key", "value"):
        raise ValueError(f"offline_proj_mode must be 'both', 'key' or 'value'; got {mode!r}")
    key = offline_projs["key"] if mode in ("both", "key") else None
    value = offline_projs["value"] if mode in ("both", "value") else None
    return key, value


def _layer_encoders(stack, layer_idx: int, heads: int):
    """Slice ``[layers, heads, bits, dim]`` down to one layer, or return None."""
    if stack is None:
        return None
    if layer_idx >= stack.shape[0]:
        raise ValueError(
            f"offline projection has {stack.shape[0]} layers; layer {layer_idx} requested"
        )
    layer = stack[layer_idx]
    if layer.shape[0] != heads:
        raise ValueError(
            f"offline projection layer {layer_idx} has {layer.shape[0]} heads; "
            f"model has {heads}"
        )
    return layer


__all__ = [
    "OFFLINE_ENCODER_FORMAT",
    "QuantKVCache",
    "load_offline_projections",
    "modify",
]
