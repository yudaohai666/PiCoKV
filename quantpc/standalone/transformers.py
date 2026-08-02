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
from .decode import decode_attention
from .prepare import prepare_kv_cache


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

    def __init__(self, config: StandaloneConfig | None = None) -> None:
        """Create an empty per-layer cache with the supplied production config."""
        self.config = config or StandaloneConfig()
        self.layers: dict[int, PreparedKVCache] = {}
        self.seq_lengths: dict[int, int] = {}
        self.decode_calls = 0
        self._prepare_events: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._prepare_value_streams: dict[torch.device, torch.cuda.Stream] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return logical prompt-plus-generated length for one layer."""
        return int(self.seq_lengths.get(int(layer_idx), 0))

    def _prepare_kwargs(self) -> dict:
        """Return the kwargs for one ``prepare_kv_cache`` call.

        Subclasses override this to add per-layer state, such as offline
        encoders, without reimplementing :meth:`prepare`.
        """
        return self.config.prepare_kwargs()

    def get_max_cache_shape(self):
        """Report an unbounded dynamic cache to the Transformers interface."""
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Reject the generic Cache API because attention owns both phases."""
        raise RuntimeError(
            "StandaloneLlamaCache.update requires modify_llama_for_standalone(model)"
        )

    def _prepare_value_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return the persistent side stream used by all layer prepares."""
        stream = self._prepare_value_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._prepare_value_streams[device] = stream
        return stream

    def prepare(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Build and time the online-projection cache for one prompt layer."""
        layer_idx = int(layer_idx)
        if layer_idx in self.layers:
            raise RuntimeError(f"layer {layer_idx} was prepared twice")

        if not key_states.is_cuda:
            raise ValueError("StandaloneLlamaCache requires CUDA key/value states")
        stream = torch.cuda.current_stream(key_states.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        prepare_kwargs = self._prepare_kwargs()
        prepare_kwargs["value_stream"] = self._prepare_value_stream(key_states.device)
        self.layers[layer_idx] = prepare_kv_cache(
            key_states,
            value_states,
            **prepare_kwargs,
        )
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
                    # Cache construction happens once per layer after RoPE;
                    # prompt output itself remains exact causal SDPA.
                    past_key_value.prepare(
                        self.layer_idx,
                        key_states.contiguous(),
                        value_states.contiguous(),
                    )
                    attn_output = _exact_prefill_attention(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                    )
                else:
                    # Generated K/V is included in the current attention call
                    # and appended to the retained dense cache for later calls.
                    attn_output = past_key_value.decode(
                        self.layer_idx,
                        query_states.contiguous(),
                        key_states.contiguous(),
                        value_states.contiguous(),
                    )
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


__all__ = ["StandaloneLlamaCache", "modify_llama_for_standalone"]
