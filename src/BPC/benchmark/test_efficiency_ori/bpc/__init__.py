"""
BPC: binary-hash sparse attention (CUDA kernel only).

Public API:
- modify: patch a model to use sparse attention
- origin_modify: keep model unchanged (HF flash_attention_2 baseline)
- reset_bpc / finalize_bpc: reset / finalize hash metadata
- get_bpc_metadata / get_bpc_config: accessors

Usage:
1. model = modify(model, config)
2. Chunked prefill: repeatedly call model(chunk_input_ids, past_key_values=past_kv)
3. finalize_bpc(model, past_key_values, fix_layers)
4. Decode: model(next_token, past_key_values=past_kv)

Config keys:
- enable, fix_layers, mask_out, min_remain, token_budget, err_ratio
- use_offline_proj, offline_proj_path
- num_splits: flash_attn_with_kvcache num_splits (0=auto)
"""

from .bpc import (
    modify,
    origin_modify,
    reset_bpc,
    finalize_bpc,
    get_bpc_metadata,
    get_bpc_config,
    is_using_cuda_kernel,
    find_binary_projection,
    DEFAULT_CONFIG,
    BPCMetadata,
    LayerMetadata,
    # legacy API aliases
    reset_cache,
    get_bpc_cache,
)

__all__ = [
    'modify',
    'origin_modify',
    'reset_bpc',
    'finalize_bpc',
    'get_bpc_metadata',
    'get_bpc_config',
    'is_using_cuda_kernel',
    'find_binary_projection',
    'DEFAULT_CONFIG',
    'BPCMetadata',
    'LayerMetadata',
    # legacy API aliases
    'reset_cache',
    'get_bpc_cache',
]
