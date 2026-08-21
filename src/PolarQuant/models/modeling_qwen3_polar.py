"""Qwen3 adapter for PolarQuant's existing KV-cache implementations."""

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
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from models.modeling_llama_polar import (
    LlamaPolarForCausalLM,
    LlamaPolarGroupAttention,
    LlamaPolarMixedGroupAttention,
    LlamaPolarModel,
    LlamaPolarDecoderLayer,
    LlamaSnapKVAttention,
    generate,
)


def _project_with_qk_norm(
    projection: nn.Linear,
    norm: nn.Module,
    num_heads: int,
    head_dim: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    projected = F.linear(hidden_states, projection.weight, projection.bias)
    shaped = projected.view(*projected.shape[:-1], num_heads, head_dim)
    return norm(shaped).reshape_as(projected)


def _install_qwen3_qk_norm(attention: nn.Module, config) -> None:
    attention.q_norm = Qwen3RMSNorm(attention.head_dim, eps=config.rms_norm_eps)
    attention.k_norm = Qwen3RMSNorm(attention.head_dim, eps=config.rms_norm_eps)
    attention.q_proj.forward = partial(
        _project_with_qk_norm,
        attention.q_proj,
        attention.q_norm,
        attention.num_heads,
        attention.head_dim,
    )
    attention.k_proj.forward = partial(
        _project_with_qk_norm,
        attention.k_proj,
        attention.k_norm,
        attention.num_key_value_heads,
        attention.head_dim,
    )


class Qwen3PolarGroupAttention(LlamaPolarGroupAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        _install_qwen3_qk_norm(self, config)


class Qwen3PolarMixedGroupAttention(LlamaPolarMixedGroupAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        _install_qwen3_qk_norm(self, config)


class Qwen3SnapKVAttention(LlamaSnapKVAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        _install_qwen3_qk_norm(self, config)


class Qwen3PolarDecoderLayer(LlamaPolarDecoderLayer):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        mode = getattr(config, "polar_mode", "group")
        if mode == "mixed":
            self.self_attn = Qwen3PolarMixedGroupAttention(config, layer_idx)
        elif mode == "snapkv":
            self.self_attn = Qwen3SnapKVAttention(config, layer_idx=layer_idx)
        else:
            self.self_attn = Qwen3PolarGroupAttention(config, layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)


class Qwen3PolarModel(LlamaPolarModel):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3PolarDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3PolarForCausalLM(LlamaPolarForCausalLM):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3PolarModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
