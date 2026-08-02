"""Small Nsight Compute target for the isolated prefill CUDA kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import (  # noqa: E402
    gather_pack_from_packed_cuda,
    quantize_pack_cuda,
    select_eas_cuda,
)


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
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    full_k = torch.randn(args.batch, args.heads, args.length, args.dim, device=device, dtype=dtype).contiguous()
    full_v = torch.randn_like(full_k)
    k_proj = (torch.randn(args.batch, args.heads, args.k_bits, args.dim, device=device, dtype=dtype) * 0.02).contiguous()
    v_proj = (torch.randn(args.batch, args.heads, args.v_bits, args.dim, device=device, dtype=dtype) * 0.02).contiguous()
    glob_len = args.length - args.sink - args.local
    eas_size = max(1, int(glob_len * args.eas))

    # Warmup outside ncu kernel count is handled with --launch-skip if desired.
    pk_all, _, k_err = quantize_pack_cuda(full_k, k_proj, args.sink, glob_len, args.zerorope, True, True)
    pv_tok, _, v_err = quantize_pack_cuda(full_v, v_proj, args.sink, glob_len, 0, False, False)
    eas_idx, _, kept_idx = select_eas_cuda(k_err, v_err, eas_size)
    gather_pack_from_packed_cuda(full_k, full_v, pk_all, pv_tok, eas_idx, kept_idx, args.sink, args.local)
    torch.cuda.synchronize(device)
    print("profile target done")


if __name__ == "__main__":
    main()
