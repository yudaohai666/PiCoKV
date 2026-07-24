"""Compare refit vs non-refit prefill speed via prepare_kv_cache."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc import quanpc_api
from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import (
    quantize_unpacked_cuda, select_eas_cuda, gather_pack_cuda,
)


def time_cuda(fn, device, warmup, iters):
    with torch.no_grad(), torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize(device)
    return s.elapsed_time(e) / max(1, iters)


def run(length, args, device, dtype):
    b, h, d = args.batch, args.heads, args.dim
    torch.manual_seed(0)
    fk = torch.randn(b, h, length, d, device=device, dtype=dtype).contiguous()
    fv = torch.randn_like(fk)
    kp = (torch.randn(b, h, args.k_bits, d, device=device, dtype=dtype) * 0.02).contiguous()
    vp = (torch.randn(b, h, args.v_bits, d, device=device, dtype=dtype) * 0.02).contiguous()
    kw = dict(sink_size=args.sink, local_size=args.local, eas=args.eas,
              zerorope=args.zerorope, exp_bits=4)
    glob_start, glob_len = quanpc_api._global_span(fk, args.sink, args.local)

    non = time_cuda(lambda: quanpc_api.prepare_kv_cache(fk, fv, kp, vp, refit_decoder=False, **kw),
                    device, args.warmup, args.iters)
    ref = time_cuda(lambda: quanpc_api.prepare_kv_cache(fk, fv, kp, vp, refit_decoder=True, **kw),
                    device, args.warmup, args.iters)

    # breakdown of refit path components
    kg = fk[:, :, glob_start:glob_start + glob_len, :].clone()
    quanpc_api._apply_zerorope_(kg, args.zerorope)
    vg = fv[:, :, glob_start:glob_start + glob_len, :].contiguous()
    quant = time_cuda(lambda: (quantize_unpacked_cuda(fk, kp, glob_start, glob_len, args.zerorope, True),
                               quantize_unpacked_cuda(fv, vp, glob_start, glob_len, 0, False)),
                      device, args.warmup, args.iters)
    kc, _, _ = quantize_unpacked_cuda(fk, kp, glob_start, glob_len, args.zerorope, True)
    kcache = __import__("quantpc.quanpc", fromlist=["KQuantCache"]).KQuantCache(args.k_bits, args.sink, args.local, args.zerorope, refit_decoder=True)
    def refit1():
        dec = kcache._refit_decoder_from_codes(kc, kg)
        return kcache._reconstruction_error(kc, kg, dec)
    r1 = time_cuda(refit1, device, args.warmup, args.iters)
    print(f"L={length:>6} glob={glob_len:>6} | non-refit={non:8.3f}ms  refit={ref:8.3f}ms  "
          f"x{ref/non:5.2f} | quant2x={quant:7.3f}ms  refit#1(K only)={r1:7.3f}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--k-bits", type=int, default=256)
    p.add_argument("--v-bits", type=int, default=256)
    p.add_argument("--sink", type=int, default=64)
    p.add_argument("--local", type=int, default=1024)
    p.add_argument("--eas", type=float, default=0.01)
    p.add_argument("--zerorope", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--lengths", default="4096,8192,16384,32768")
    args = p.parse_args()
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    print(f"device={torch.cuda.get_device_name(device)} dtype={args.dtype} "
          f"heads={args.heads} dim={args.dim} Ck={args.k_bits} Cv={args.v_bits} eas={args.eas}")
    for L in [int(x) for x in args.lengths.split(",") if x]:
        run(L, args, device, dtype)


if __name__ == "__main__":
    main()
