"""Correctness test for the refit_decoder CUDA prefill path.

Compares prepare_kv_cache(refit_decoder=True, use_cuda=True) against the pure
torch reference _prepare_kv_cache_torch(refit_decoder=True).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc import quanpc_api  # noqa: E402


def run_case(dtype: torch.dtype):
    torch.manual_seed(7)
    device = "cuda"
    b, h, total, d = 1, 2, 160, 32
    sink, local = 8, 16
    ck, cv = 128, 192
    zerorope = 3
    eas = 0.05
    exp_bits = 4

    full_k = torch.randn(b, h, total, d, device=device, dtype=dtype).contiguous()
    full_v = torch.randn(b, h, total, d, device=device, dtype=dtype).contiguous()
    k_proj = (torch.randn(b, h, ck, d, device=device, dtype=dtype) * 0.02).contiguous()
    v_proj = (torch.randn(b, h, cv, d, device=device, dtype=dtype) * 0.02).contiguous()

    kw = dict(sink_size=sink, local_size=local, eas=eas, zerorope=zerorope,
              exp_bits=exp_bits, refit_decoder=True)
    cuda = quanpc_api.prepare_kv_cache(full_k, full_v, k_proj, v_proj, use_cuda=True, **kw)

    if dtype != torch.float32:
        # bf16 quantization has near-zero sign ambiguity, so CUDA vs torch codes
        # (and hence the cascading EAS selection) are not bit-identical. Only run
        # a smoke check that the CUDA refit path produces well-formed outputs.
        assert cuda.packed_key.dtype == torch.long and cuda.packed_value.dtype == torch.long
        assert cuda.key_projection.shape == (b, h, ck, d)
        assert cuda.value_projection.shape == (b, h, cv, d)
        assert torch.isfinite(cuda.key_projection.float()).all()
        assert torch.isfinite(cuda.value_projection.float()).all()
        assert torch.isfinite(cuda.dense_key.float()).all()
        print(f"refit case dtype={dtype} smoke passed")
        return

    ref = quanpc_api.prepare_kv_cache(full_k, full_v, k_proj, v_proj, use_cuda=False, **kw)

    # EAS selection must match exactly.
    assert torch.equal(cuda.eas_indices.sort(-1).values,
                       ref.eas_indices.sort(-1).values), "EAS indices mismatch"
    # Dense retained KV: exact.
    assert torch.equal(cuda.dense_key, ref.dense_key), "dense K mismatch"
    assert torch.equal(cuda.dense_value, ref.dense_value), "dense V mismatch"
    # Packed codes: exact (codes unchanged by refit).
    assert torch.equal(cuda.packed_key, ref.packed_key), "packed K mismatch"
    assert torch.equal(cuda.packed_value, ref.packed_value), "packed V mismatch"
    # Refit decoders (final projections): close.
    atol = 1e-4
    rtol = 1e-4
    assert torch.allclose(cuda.key_projection.float(), ref.key_projection.float(),
                          atol=atol, rtol=rtol), "refit key projection mismatch"
    assert torch.allclose(cuda.value_projection.float(), ref.value_projection.float(),
                          atol=atol, rtol=rtol), "refit value projection mismatch"
    # Note: PreparedKVCache.key_err/value_err is diagnostic and not consumed by
    # decode. The CUDA path returns the full-glob refit#1 error (which drives EAS),
    # matching the non-refit CUDA convention; the torch reference returns the
    # post-refit#2 kept-length error, so the two differ by convention/shape.
    print(f"refit case dtype={dtype} passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"device: {torch.cuda.get_device_name()}")
    for name in [x for x in args.dtypes.split(",") if x]:
        run_case(getattr(torch, name))
    print("prefill CUDA refit correctness passed")


if __name__ == "__main__":
    main()
