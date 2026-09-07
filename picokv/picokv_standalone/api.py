"""Stable public facade for standalone PiCoKV.

Implementation lives in focused modules:

- :mod:`picokv_standalone.prepare` builds an online-projection cache.
- :mod:`picokv_standalone.decode` runs single-token CUDA attention.
- :mod:`picokv_standalone.cache` defines the object passed between them.

Keeping this facade preserves existing imports without mixing implementation
responsibilities back into one file.
"""

from .cache import PreparedKVCache
from .decode import decode_attention
from .prepare import prepare_kv_cache


__all__ = ["PreparedKVCache", "decode_attention", "prepare_kv_cache"]
