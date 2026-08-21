"""Qwen3 adapter for the existing KIVI KV-cache implementation.

The KIVI cache algorithm is shared with the Llama implementation.  Qwen3
only needs its Q/K head RMSNorm and Qwen3 rotary embedding before entering the
same KIVI flash-attention/cache code.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from models.llama_kivi import (
    LlamaFlashAttention_KIVI,
    LlamaForCausalLM_KIVI,
    LlamaModel_KIVI,
)


def _project_with_qk_norm(
    projection: nn.Linear,
    norm: nn.Module,
    num_heads: int,
    head_dim: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run a Qwen3 projection and normalize each attention head."""
    projected = F.linear(hidden_states, projection.weight, projection.bias)
    shaped = projected.view(*projected.shape[:-1], num_heads, head_dim)
    return norm(shaped).reshape_as(projected)


class Qwen3FlashAttention_KIVI(LlamaFlashAttention_KIVI):
    """KIVI flash attention with Qwen3's pre-RoPE Q/K RMSNorm."""

    def __init__(self, config):
        super().__init__(config)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)

        # Keep the original Linear parameters/state-dict names while making
        # the inherited KIVI forward see normalized per-head projections.
        self.q_proj.forward = partial(
            _project_with_qk_norm,
            self.q_proj,
            self.q_norm,
            self.num_heads,
            self.head_dim,
        )
        self.k_proj.forward = partial(
            _project_with_qk_norm,
            self.k_proj,
            self.k_norm,
            self.num_key_value_heads,
            self.head_dim,
        )


class Qwen3DecoderLayer_KIVI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3FlashAttention_KIVI(config=config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen3Model_KIVI(LlamaModel_KIVI):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer_KIVI(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3ForCausalLM_KIVI(LlamaForCausalLM_KIVI):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model_KIVI(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
