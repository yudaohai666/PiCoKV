"""
BPC - binary-hash-based sparse attention.

API:
- modify: patch model to use sparse attention
- reset_cache, get_bpc_cache, get_bpc_config
- is_using_cuda_kernel: whether the CUDA kernel is active
- find_binary_projection: learn projection matrix

Config keys:
- enable, fix_layers, mask_out, min_remain
- token_budget (overrides mask_out), err_ratio
- use_offline_proj, offline_proj_path
- use_cuda_kernel
"""

from .bpc import (
    modify,
    reset_cache,
    get_bpc_cache,
    get_bpc_config,
    is_using_cuda_kernel,
    find_binary_projection,
    DEFAULT_CONFIG,
    AllCache,
    LayerCache,
)

__all__ = [
    'modify',
    'reset_cache',
    'get_bpc_cache',
    'get_bpc_config',
    'is_using_cuda_kernel',
    'find_binary_projection',
    'DEFAULT_CONFIG',
    'AllCache',
    'LayerCache',
]
