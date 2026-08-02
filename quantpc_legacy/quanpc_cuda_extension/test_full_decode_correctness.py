"""High-level QuanPC full decode correctness smoke test.

The test compares the public ``quantized_cache_attention_forward`` output from:

1. the Python unpacked fallback path; and
2. the CUDA packed fast path used by decoding.

It intentionally uses small tensors so it can run before speed benchmarks.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc import MixedCacheTensor, QuantizedTensor, quantized_cache_attention_forward
from quantpc.quanpc_cuda_extension.quanpc_cuda import (
    pack_quanpc_key_transposed_torch,
    pack_quanpc_value_1bit_transposed_torch,
)


def _binary(shape, *, device, dtype):
    bits = torch.randint(0, 2, shape, device=device, dtype=torch.int8)
    return (bits * 2 - 1).to(dtype).contiguous()


def _make_pair(B: int, gqa: int, exp_bits: int, dense_len: int, global_len: int, device: str):
    torch.manual_seed(1234 + B * 17 + gqa * 31 + exp_bits)
    KVH = 8
    QH = KVH * gqa
    D = 128
    C = 256
    dtype = torch.bfloat16

    query = torch.randn(B, QH, 1, D, device=device, dtype=dtype).contiguous()
    dense_k = torch.randn(B, KVH, dense_len, D, device=device, dtype=dtype).contiguous()
    dense_v = torch.randn(B, KVH, dense_len, D, device=device, dtype=dtype).contiguous()
    key_codes = _binary((B, KVH, global_len, C), device=device, dtype=dtype)
    value_codes = _binary((B, KVH, global_len, C), device=device, dtype=dtype)
    key_proj = (torch.randn(B, KVH, C, D, device=device, dtype=dtype) * 0.02).contiguous()
    value_proj = (torch.randn(B, KVH, C, D, device=device, dtype=dtype) * 0.02).contiguous()
    err = torch.zeros(B, KVH, global_len, device=device, dtype=torch.float32)

    key_block_ref = QuantizedTensor(
        key_codes,
        key_proj,
        err,
        packed_cuda=None,
        proj_t=key_proj.transpose(-1, -2).contiguous(),
    )
    value_block_ref = QuantizedTensor(
        value_codes,
        value_proj,
        err,
        packed_cuda=None,
        proj_t=value_proj.transpose(-1, -2).contiguous(),
    )
    key_ref = MixedCacheTensor(dense_k, key_block_ref, False, 8, exp_bits)
    value_ref = MixedCacheTensor(dense_v, value_block_ref, False, 8, exp_bits)

    empty_codes = key_codes[:, :, :0, :].contiguous()
    empty_err = err[:, :, :0].contiguous()
    key_block_cuda = QuantizedTensor(
        empty_codes,
        key_proj,
        empty_err,
        packed_cuda=pack_quanpc_key_transposed_torch(key_codes).contiguous(),
        proj_t=key_proj.transpose(-1, -2).contiguous(),
    )
    value_block_cuda = QuantizedTensor(
        value_codes[:, :, :0, :].contiguous(),
        value_proj,
        empty_err,
        packed_cuda=pack_quanpc_value_1bit_transposed_torch(value_codes).contiguous(),
        proj_t=value_proj.transpose(-1, -2).contiguous(),
    )
    key_cuda = MixedCacheTensor(dense_k, key_block_cuda, True, 8, exp_bits)
    value_cuda = MixedCacheTensor(dense_v, value_block_cuda, True, 8, exp_bits)
    return query, key_ref, value_ref, key_cuda, value_cuda


def run_case(B: int, gqa: int, exp_bits: int, dense_len: int, global_len: int, atol: float, rtol: float):
    query, key_ref, value_ref, key_cuda, value_cuda = _make_pair(B, gqa, exp_bits, dense_len, global_len, "cuda")
    ref = quantized_cache_attention_forward(query, key_ref, value_ref)
    out = quantized_cache_attention_forward(query, key_cuda, value_cuda)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    print(
        f"case B={B} gqa={gqa} exp_bits={exp_bits} dense_len={dense_len} global_len={global_len} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} ok={ok}"
    )
    if not ok:
        raise AssertionError(f"full decode mismatch: max_abs={max_abs}, mean_abs={mean_abs}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--gqa", type=int, nargs="+", default=[1, 4, 8], choices=[1, 4, 8])
    parser.add_argument("--exp-bits", type=int, nargs="+", default=[1, 2, 4, 8], choices=[1, 2, 4, 8])
    parser.add_argument("--dense-len", type=int, default=256)
    parser.add_argument("--global-len", type=int, default=257)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    print(f"device: {torch.cuda.get_device_name()}")
    for B in args.batch_sizes:
        for gqa in args.gqa:
            for exp_bits in args.exp_bits:
                run_case(B, gqa, exp_bits, args.dense_len, args.global_len, args.atol, args.rtol)
    print("full decode correctness passed")


if __name__ == "__main__":
    main()
