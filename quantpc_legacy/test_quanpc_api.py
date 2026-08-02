"""Smoke tests for the public QuanPC API."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc_api import decode_attention, learn_projection, prepare_kv_cache


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(2026)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, kv_heads, q_heads, total_len, dim = 1, 8, 32, 1536, 128
    k_bits, v_bits = 128, 128
    full_key = torch.randn(bsz, kv_heads, total_len, dim, device=device, dtype=dtype).contiguous()
    full_value = torch.randn_like(full_key)
    query = torch.randn(bsz, q_heads, 1, dim, device=device, dtype=dtype).contiguous()

    key_proj = learn_projection(full_key, k_bits, sink_size=64, local_size=128, zerorope=20, apply_zerorope=True)
    value_proj = learn_projection(full_value, v_bits, sink_size=64, local_size=128)
    cache = prepare_kv_cache(
        full_key,
        full_value,
        key_proj,
        value_proj,
        sink_size=64,
        local_size=128,
        eas=0.01,
        zerorope=20,
        exp_bits=4,
    )
    out_ref = decode_attention(query, cache, use_pruning=False)
    out_pruned = decode_attention(query, cache, use_pruning=True, log_eps=-12)
    torch.cuda.synchronize()
    if out_ref.shape != query.shape or out_pruned.shape != query.shape:
        raise AssertionError(f"unexpected output shapes: {out_ref.shape}, {out_pruned.shape}")
    if not torch.isfinite(out_ref.float()).all() or not torch.isfinite(out_pruned.float()).all():
        raise AssertionError("decode output contains non-finite values")
    diff = (out_ref.float() - out_pruned.float()).abs()
    print(
        "api smoke passed "
        f"dense_len={cache.dense_len} global_len={cache.global_len} "
        f"max_abs={float(diff.max()):.6f} mean_abs={float(diff.mean()):.6f}"
    )


if __name__ == "__main__":
    main()
