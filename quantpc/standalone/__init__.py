"""Public online-projection QuanPC API.

Importing this package does not patch Transformers models or launch CUDA work.
"""

from .api import (
    PreparedKVCache,
    decode_attention,
    prepare_kv_cache,
)
from .config import StandaloneConfig

__all__ = [
    "PreparedKVCache",
    "StandaloneConfig",
    "decode_attention",
    "prepare_kv_cache",
]
