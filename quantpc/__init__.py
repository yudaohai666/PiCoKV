"""QuanPC core package: quantized KV-cache attention algorithm and CUDA extensions.

Benchmark-agnostic. Public modules:
- ``quantpc.quanpc``: model patching (``modify``) and ``QuantKVCache``.
- ``quantpc.quanpc_torch``: torch-only reference implementation.
- ``quantpc.quanpc_api``: cleaned public API (learn_projection / prepare_kv_cache / decode_attention).
"""
