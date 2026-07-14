"""Benchmark QuanPC CUDA projection learning."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import learn_projection_cuda  # noqa: E402
from quantpc.quanpc_prefill_cuda_extension.test_projection_learning import torch_learn_projection  # noqa: E402


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
    parser.add_argument("--bits", type=int, default=256)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--local", type=int, default=1024)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-reference", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1234)
    full = torch.randn(args.batch, args.heads, args.length, args.dim, device=device, dtype=dtype).contiguous()
    init = torch.randn(args.batch, args.heads, args.bits, args.dim, device=device, dtype=dtype).contiguous()
    glob_len = args.length - args.sink - args.local
    print(f"device={torch.cuda.get_device_name(device)}")
    print(
        f"shape full=[{args.batch},{args.heads},{args.length},{args.dim}] glob={glob_len} "
        f"bits={args.bits} dtype={args.dtype} zerorope={args.zerorope}"
    )

    def cuda_fn():
        return learn_projection_cuda(full, init, args.sink, glob_len, args.zerorope, True)

    cuda_ms = time_cuda(cuda_fn, device, args.warmup, args.iters)
    print(f"learn_projection_cuda={cuda_ms:.4f} ms")
    if args.torch_reference:
        def torch_fn():
            return torch_learn_projection(full, init, args.sink, glob_len, args.zerorope, True, power_iters=1)
        torch_ms = time_cuda(torch_fn, device, max(1, args.warmup), max(1, args.iters))
        print(f"learn_projection_torch={torch_ms:.4f} ms speedup={torch_ms / cuda_ms:.2f}x")


if __name__ == "__main__":
    main()
