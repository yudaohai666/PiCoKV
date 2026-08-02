"""Benchmark the isolated QuanPC prefill CUDA pipeline on synthetic tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import (  # noqa: E402
    gather_pack_from_packed_cuda,
    prepare_prefill_cuda,
    quantize_pack_cuda,
    select_eas_cuda,
)


def time_cuda(fn: Callable[[], object], device: torch.device, warmup: int, iters: int) -> float:
    with torch.no_grad(), torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            fn()
        end.record(stream)
        torch.cuda.synchronize(device)
    return start.elapsed_time(end) / max(1, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k-bits", type=int, default=256)
    parser.add_argument("--v-bits", type=int, default=256)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--local", type=int, default=1024)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1234)
    full_k = torch.randn(args.batch, args.heads, args.length, args.dim, device=device, dtype=dtype).contiguous()
    full_v = torch.randn_like(full_k)
    k_proj = (torch.randn(args.batch, args.heads, args.k_bits, args.dim, device=device, dtype=dtype) * 0.02).contiguous()
    v_proj = (torch.randn(args.batch, args.heads, args.v_bits, args.dim, device=device, dtype=dtype) * 0.02).contiguous()
    glob_len = args.length - args.sink - args.local
    eas_size = max(1, int(glob_len * args.eas))
    print(f"device={torch.cuda.get_device_name(device)}")
    print(
        f"shape full=[{args.batch},{args.heads},{args.length},{args.dim}] glob={glob_len} "
        f"Ck={args.k_bits} Cv={args.v_bits} eas_size={eas_size} dtype={args.dtype}"
    )

    def qk():
        return quantize_pack_cuda(full_k, k_proj, args.sink, glob_len, args.zerorope, True, True)

    def qv():
        return quantize_pack_cuda(full_v, v_proj, args.sink, glob_len, 0, False, False)

    pk_all, key_proj, k_err = qk()
    pv_token, value_proj, v_err = qv()
    torch.cuda.synchronize(device)

    def eas_fn():
        return select_eas_cuda(k_err, v_err, eas_size)

    eas_idx, _mask, kept_idx = eas_fn()

    def gather_pack_fn():
        return gather_pack_from_packed_cuda(full_k, full_v, pk_all, pv_token, eas_idx, kept_idx, args.sink, args.local)

    def total_fn():
        return prepare_prefill_cuda(
            full_k,
            full_v,
            k_proj,
            v_proj,
            sink_size=args.sink,
            local_size=args.local,
            eas=args.eas,
            zerorope=args.zerorope,
        )

    qk_ms = time_cuda(qk, device, args.warmup, args.iters)
    qv_ms = time_cuda(qv, device, args.warmup, args.iters)
    eas_ms = time_cuda(eas_fn, device, args.warmup, args.iters)
    gp_ms = time_cuda(gather_pack_fn, device, args.warmup, args.iters)
    total_ms = time_cuda(total_fn, device, max(1, args.warmup // 2), max(1, args.iters // 2))
    print(f"quantize_pack_k={qk_ms:.4f} ms")
    print(f"quantize_pack_v={qv_ms:.4f} ms")
    print(f"select_eas={eas_ms:.4f} ms")
    print(f"gather_pack={gp_ms:.4f} ms")
    print(f"step_sum={qk_ms + qv_ms + eas_ms + gp_ms:.4f} ms")
    print(f"total_allocating={total_ms:.4f} ms")
    print(
        f"outputs dense=[{args.batch},{args.heads},{args.sink + eas_idx.shape[-1] + args.local},{args.dim}] "
        f"packed_k={tuple(pk_all.shape)} final_k={tuple(gather_pack_fn()[2].shape)} final_v={tuple(gather_pack_fn()[3].shape)}"
    )


if __name__ == "__main__":
    main()
