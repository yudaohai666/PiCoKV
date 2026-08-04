"""Optional Hugging Face Llama integration for the standalone API.

The core package does not monkey-patch models on import.  Call
``modify_llama_for_standalone`` explicitly, then pass a
``StandaloneLlamaCache`` to ``model.generate``.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    apply_rotary_pos_emb,
)

from .cache import PreparedKVCache
from .config import StandaloneConfig
from .constants import ONLINE_GROUP_BITS
from .decode import decode_attention
from .prepare import prepare_kv_cache
from .whiten import (
    LayerWhitening,
    find_orth_proj,
    value_whitening_source,
    whiten_states,
)


OFFLINE_ENCODER_FORMAT = "quanpc_standalone_block32_encoder_v1"
OFFLINE_DECODER_FORMAT = "quanpc_standalone_block32_encoder_decoder_v1"
# Encoders and decoders with the whitening already folded in (``E W`` and
# ``D W^-1``).  Same shapes and same runtime path as the pair above -- the point
# of folding is that nothing downstream can tell the difference, so K is never
# rotated at prefill and ``key_projection_t`` is ``W^-1 D^T`` for free.  It needs
# its own tag only so a folded file cannot be combined with runtime whitening,
# which would rotate states whose basis is already accounted for.
OFFLINE_FOLDED_FORMAT = "quanpc_standalone_block32_folded_v1"
_FORMATS_WITH_DECODER = (OFFLINE_DECODER_FORMAT, OFFLINE_FOLDED_FORMAT)


def load_offline_projections(path: str) -> dict:
    """Load stored block-32 encoders as ``{"key": ..., "value": ..., "meta": ...}``.

    ``key``/``value`` are ``[num_layers, num_kv_heads, bits, head_dim]`` stacks of
    orthonormal scoring directions produced offline.  The format tag guards
    against loading incompatible (e.g. greedy-era) files that share the shape.
    """
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "key" not in data or "value" not in data:
        raise ValueError("offline projection file must be a dict containing 'key' and 'value'")
    meta = data.get("meta") or {}
    fmt = meta.get("format")
    if fmt not in (
        OFFLINE_ENCODER_FORMAT,
        OFFLINE_DECODER_FORMAT,
        OFFLINE_FOLDED_FORMAT,
    ):
        raise ValueError(
            f"{path} has format {fmt!r}, but the standalone method requires one of "
            f"{OFFLINE_ENCODER_FORMAT!r}, {OFFLINE_DECODER_FORMAT!r}, "
            f"{OFFLINE_FOLDED_FORMAT!r}."
        )
    if fmt in _FORMATS_WITH_DECODER:
        for kind in ("key_decoder", "value_decoder"):
            if kind not in data:
                raise ValueError(f"{path} claims {fmt!r} but is missing {kind!r}")
            if data[kind].shape != data[kind.replace("_decoder", "")].shape:
                raise ValueError(
                    f"{path}: {kind} shape {tuple(data[kind].shape)} does not match "
                    f"its encoder {tuple(data[kind.replace('_decoder', '')].shape)}"
                )
    for kind in ("key", "value"):
        tensor = data[kind]
        if tensor.dim() != 4:
            raise ValueError(
                f"offline '{kind}' must be [layers, heads, bits, dim]; got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2] % ONLINE_GROUP_BITS:
            raise ValueError(
                f"offline '{kind}' bits {tensor.shape[-2]} is not a multiple of {ONLINE_GROUP_BITS}"
            )
    return data


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
            f"offline projection layer {layer_idx} has {layer.shape[0]} heads; model has {heads}"
        )
    return layer


def _chunked_llama_mlp_forward(
    self,
    hidden_states: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Run the original Llama MLP in token chunks to bound prefill memory."""
    if hidden_states.shape[-2] <= chunk_size:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
    outputs = []
    for part in hidden_states.split(chunk_size, dim=-2):
        outputs.append(
            self.down_proj(self.act_fn(self.gate_proj(part)) * self.up_proj(part))
        )
    return torch.cat(outputs, dim=-2)


def _exact_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run exact SDPA for prompt tokens, preserving an explicit caller mask."""
    # Transformers' SDPA path normally passes None for an unpadded causal
    # prompt.  Preserve an explicit mask when one is supplied.
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=attention_mask is None and query.shape[-2] > 1,
        enable_gqa=True,
    )


class StandaloneLlamaCache(Cache):
    """One :class:`PreparedKVCache` per Llama layer.

    Prompt attention remains exact SDPA.  Once a layer's prompt K/V is
    available, it is compressed through ``prepare_kv_cache``.  Single-token
    generation then calls ``decode_attention`` and appends new K/V to the dense
    retained portion of the layer cache.
    """

    def __init__(
        self,
        config: StandaloneConfig | None = None,
        *,
        key_encoders: Optional[torch.Tensor] = None,
        value_encoders: Optional[torch.Tensor] = None,
        key_decoders: Optional[torch.Tensor] = None,
        value_decoders: Optional[torch.Tensor] = None,
    ) -> None:
        """Create an empty per-layer cache with the supplied production config.

        Optional ``key_encoders``/``value_encoders`` are offline-projection stacks
        ``[num_layers, num_kv_heads, bits, head_dim]``; when given, prefill skips
        the online Householder QR. ``None`` keeps the pure-online path.

        Supplying ``key_decoders``/``value_decoders`` as well skips the per-group
        least-squares solve. They are only meaningful next to the encoders they
        were fitted with, since the codes come from a residual chain both drive.
        """
        if key_decoders is not None and key_encoders is None:
            raise ValueError("key_decoders require key_encoders")
        if value_decoders is not None and value_encoders is None:
            raise ValueError("value_decoders require value_encoders")
        self.config = config or StandaloneConfig()
        self.layers: dict[int, PreparedKVCache] = {}
        self.key_encoders = key_encoders
        self.value_encoders = value_encoders
        self.key_decoders = key_decoders
        self.value_decoders = value_decoders
        self._pending_encoders: Optional[tuple] = None
        self._pending_decoders: Optional[tuple] = None
        # Layers whose prompt is shorter than sink+local have no global segment
        # to compress; they keep full-precision K/V here and serve decode with
        # plain SDPA instead of the packed CUDA path.
        self._dense_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.seq_lengths: dict[int, int] = {}
        self.decode_calls = 0
        self._prepare_events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        # Per-layer whitening factors, fitted once during prefill and reused by
        # every decode step.  Empty when config.whiten is False.
        self._whitening: dict[int, LayerWhitening] = {}

    def fit_whitening(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        output_weight: torch.Tensor,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        """Fit this layer's K/V whitening from prompt queries and ``o_proj``."""
        if not self.config.whiten:
            return
        if int(layer_idx) < int(self.config.whiten_min_layer):
            return
        batch = int(query_states.shape[0])
        grouped_query = query_states.reshape(batch, kv_heads, -1, head_dim)
        key_forward, key_inverse = find_orth_proj(grouped_query)
        if self.config.whiten_key_only:
            # Identity leaves V in the model's basis while K still rotates, so
            # decode's un-whitening stays a single uniform code path.
            eye = torch.eye(
                head_dim, device=key_forward.device, dtype=key_forward.dtype
            ).expand_as(key_forward).contiguous()
            value_forward, value_inverse = eye, eye
        else:
            value_forward, value_inverse = find_orth_proj(
                value_whitening_source(output_weight, kv_heads, head_dim)
            )
        self._whitening[int(layer_idx)] = LayerWhitening(
            key=key_forward,
            key_inverse=key_inverse,
            value=value_forward,
            value_inverse=value_inverse,
        )

    def whitening(self, layer_idx: int) -> Optional[LayerWhitening]:
        """Return this layer's whitening factors, or ``None`` when disabled."""
        return self._whitening.get(int(layer_idx))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return logical prompt-plus-generated length for one layer."""
        return int(self.seq_lengths.get(int(layer_idx), 0))

    def get_max_cache_shape(self):
        """Report an unbounded dynamic cache to the Transformers interface."""
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Reject the generic Cache API because attention owns both phases."""
        raise RuntimeError(
            "StandaloneLlamaCache.update requires modify_llama_for_standalone(model)"
        )

    def _prepare_kwargs(self, layer_idx: int) -> dict:
        """Return kwargs for one ``prepare_kv_cache`` call, plus offline encoders."""
        kwargs = self.config.prepare_kwargs()
        if self._pending_encoders is not None:
            kwargs["key_encoders"], kwargs["value_encoders"] = self._pending_encoders
        if self._pending_decoders is not None:
            kwargs["key_decoders"], kwargs["value_decoders"] = self._pending_decoders
        whitening = self.whitening(layer_idx)
        if whitening is not None:
            kwargs["key_whitening"] = whitening.key
            kwargs["value_whitening"] = whitening.value
        return kwargs

    def prepare(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Build and time the online-projection cache for one prompt layer."""
        layer_idx = int(layer_idx)
        if layer_idx in self.layers or layer_idx in self._dense_layers:
            raise RuntimeError(f"layer {layer_idx} was prepared twice")

        if not key_states.is_cuda:
            raise ValueError("StandaloneLlamaCache requires CUDA key/value states")

        # A prompt no longer than the sink+local dense windows leaves no global
        # segment for prepare_kv_cache to compress. Keep it fully dense and let
        # decode fall back to SDPA (the same math the dense windows use anyway).
        total_len = int(key_states.shape[-2])
        if total_len <= self.config.sink_size + self.config.local_size:
            # Decode whitens its query, so this dense fallback has to hold
            # whitened K/V as well or the two would live in different bases.
            whitening = self.whitening(layer_idx)
            if whitening is not None:
                key_states = whiten_states(key_states, whitening.key)
                value_states = whiten_states(value_states, whitening.value)
            self._dense_layers[layer_idx] = (
                key_states.contiguous(),
                value_states.contiguous(),
            )
            self.seq_lengths[layer_idx] = total_len
            return

        heads = int(key_states.shape[1])
        self._pending_encoders = (
            _layer_encoders(self.key_encoders, layer_idx, heads),
            _layer_encoders(self.value_encoders, layer_idx, heads),
        )
        self._pending_decoders = (
            _layer_encoders(self.key_decoders, layer_idx, heads),
            _layer_encoders(self.value_decoders, layer_idx, heads),
        )
        stream = torch.cuda.current_stream(key_states.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        try:
            self.layers[layer_idx] = prepare_kv_cache(
                key_states,
                value_states,
                **self._prepare_kwargs(layer_idx),
            )
        finally:
            self._pending_encoders = None
            self._pending_decoders = None
        end.record(stream)
        self._prepare_events[layer_idx] = (start, end)
        self.seq_lengths[layer_idx] = int(key_states.shape[-2])

    def decode(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one token and persist its K/V in this layer's dense cache."""
        layer_idx = int(layer_idx)
        # Short-prompt layers were kept fully dense at prepare time: append the
        # generated K/V and serve this step with exact SDPA.
        if layer_idx in self._dense_layers:
            dense_key, dense_value = self._dense_layers[layer_idx]
            dense_key = torch.cat((dense_key, key_states.contiguous()), dim=-2)
            dense_value = torch.cat((dense_value, value_states.contiguous()), dim=-2)
            self._dense_layers[layer_idx] = (dense_key, dense_value)
            output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                dense_key,
                dense_value,
                attn_mask=None,
                is_causal=False,
                enable_gqa=True,
            )
            self.seq_lengths[layer_idx] += int(key_states.shape[-2])
            self.decode_calls += 1
            return output
        if layer_idx not in self.layers:
            raise RuntimeError(f"layer {layer_idx} decode requested before prefill")
        output = decode_attention(
            query_states,
            self.layers[layer_idx],
            decode_key=key_states,
            decode_value=value_states,
            update_cache=True,
            **self.config.decode_kwargs(),
        )
        self.seq_lengths[layer_idx] += int(key_states.shape[-2])
        self.decode_calls += 1
        return output

    def prepare_timings_ms(self) -> dict[int, float]:
        """Synchronize recorded prepare events and return per-layer timings."""
        timings: dict[int, float] = {}
        for layer_idx, pair in self._prepare_events.items():
            start, end = pair
            end.synchronize()
            timings[layer_idx] = float(start.elapsed_time(end))
        return timings


def modify_llama_for_standalone(model, *, mlp_chunk_size: int = 8192) -> None:
    """Install explicit standalone prefill/decode adapters on a Llama model.

    The first pass bounds MLP prefill memory.  The second replaces only Llama
    attention: prompt attention stays exact SDPA, while single-token generation
    routes through :class:`StandaloneLlamaCache`.
    """
    # MLP chunking is orthogonal to QuanPC math; it only prevents large prompt
    # intermediates from dominating memory.
    for module in model.modules():
        if isinstance(module, LlamaMLP):
            module.forward = partial(
                _chunked_llama_mlp_forward,
                module,
                chunk_size=int(mlp_chunk_size),
            )

    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue

        def modified_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            """Dispatch a Llama attention call to exact prefill or QuanPC decode."""
            # Match the original Llama projection and RoPE ordering exactly.
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            )

            query_length = int(query_states.shape[-2])
            if isinstance(past_key_value, StandaloneLlamaCache):
                if query_length > 1:
                    # Fit whitening from the prompt queries before the cache is
                    # built; prepare_kv_cache applies it after zerorope.
                    past_key_value.fit_whitening(
                        self.layer_idx,
                        query_states,
                        self.o_proj.weight,
                        int(key_states.shape[1]),
                        self.head_dim,
                    )
                    # Cache construction happens once per layer after RoPE;
                    # prompt output itself remains exact causal SDPA.
                    past_key_value.prepare(
                        self.layer_idx,
                        key_states.contiguous(),
                        value_states.contiguous(),
                    )
                    # Prefill attention is exact, and whitening is an exact
                    # change of basis, so it stays in the model's own basis:
                    # rotating here would only add a bf16 round trip.
                    attn_output = _exact_prefill_attention(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                    )
                else:
                    # Generated K/V is included in the current attention call
                    # and appended to the retained dense cache for later calls.
                    whitening = past_key_value.whitening(self.layer_idx)
                    if whitening is not None:
                        batch, kv_heads = key_states.shape[0], key_states.shape[1]
                        head_dim = key_states.shape[-1]
                        grouped_query = query_states.reshape(
                            batch,
                            kv_heads,
                            -1,
                            head_dim,
                        )
                        query_states = whiten_states(
                            grouped_query,
                            whitening.key_inverse,
                        ).reshape(batch, -1, query_length, head_dim)
                        key_states = whiten_states(key_states, whitening.key)
                        value_states = whiten_states(value_states, whitening.value)
                    attn_output = past_key_value.decode(
                        self.layer_idx,
                        query_states.contiguous(),
                        key_states.contiguous(),
                        value_states.contiguous(),
                    )
                    if whitening is not None:
                        batch, head_dim = attn_output.shape[0], attn_output.shape[-1]
                        kv_heads = key_states.shape[1]
                        attn_output = whiten_states(
                            attn_output.reshape(batch, kv_heads, -1, head_dim),
                            whitening.value_inverse,
                        ).reshape(batch, -1, query_length, head_dim)
            else:
                attn_output = _exact_prefill_attention(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        module.forward = partial(modified_forward, module)


__all__ = [
    "OFFLINE_ENCODER_FORMAT",
    "StandaloneLlamaCache",
    "load_offline_projections",
    "modify_llama_for_standalone",
]
