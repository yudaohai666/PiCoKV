"""A/B benchmark + correctness for split-K fused output projection.

Compares quanpc_fused_output_projection_fn (baseline, 32-block) against
quanpc_fused_output_projection_splitk_fn (two-pass non-atomic split-K) on
decode-shaped tensors. Verifies bf16 output match and sweeps k_splits.
"""
import argparse
import torch

from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension


def make_inputs(B, QH, S, D, Cv, KVH, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    dense_output = torch.randn(B, QH, S, D, generator=g, device=device, dtype=torch.float32).to(dtype)
    glob_code_output = torch.randn(B, QH, S, Cv, generator=g, device=device, dtype=torch.float32)
    # den is a sum of positive softmax weights -> keep it non-negative.
    glob_den = torch.rand(B, QH, S, generator=g, device=device, dtype=torch.float32) * 8.0
    value_proj = torch.randn(B, KVH, Cv, D, generator=g, device=device, dtype=torch.float32).to(dtype)
    return dense_output.contiguous(), glob_code_output.contiguous(), glob_den.contiguous(), value_proj.contiguous()


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--QH", type=int, default=32)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--Cv", type=int, default=320)
    ap.add_argument("--KVH", type=int, default=8)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--ksplits", type=str, default="0,1,2,4,5,8,10,16,20,32")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    ext = load_quanpc_extension(verbose=False)

    dense, code, den, vproj = make_inputs(
        args.B, args.QH, args.S, args.D, args.Cv, args.KVH, dtype, device
    )
    rows = args.B * args.QH * args.S
    print(f"shape: B={args.B} QH={args.QH} S={args.S} D={args.D} Cv={args.Cv} KVH={args.KVH} "
          f"dtype={args.dtype} rows={rows}")

    attn_ref = torch.empty_like(dense)
    ext.quanpc_fused_output_projection_fn(dense, code, den, vproj, attn_ref)
    torch.cuda.synchronize()

    base_us = time_call(lambda: ext.quanpc_fused_output_projection_fn(dense, code, den, vproj, attn_ref),
                        args.warmup, args.iters)
    print(f"\nbaseline (32-block): {base_us:8.3f} us/call")

    print(f"\n{'k_splits':>8} {'blocks':>8} {'us/call':>10} {'speedup':>8} "
          f"{'bit-match%':>10} {'max_abs_err':>12}")
    for K in [int(x) for x in args.ksplits.split(",")]:
        attn_test = torch.empty_like(dense)
        ext.quanpc_fused_output_projection_splitk_fn(dense, code, den, vproj, attn_test, K)
        torch.cuda.synchronize()

        # bf16 exact bit-match
        ref_i = attn_ref.view(torch.int16) if dtype == torch.bfloat16 else attn_ref.view(torch.int32)
        test_i = attn_test.view(torch.int16) if dtype == torch.bfloat16 else attn_test.view(torch.int32)
        match = (ref_i == test_i).float().mean().item() * 100.0
        max_err = (attn_ref.float() - attn_test.float()).abs().max().item()

        us = time_call(lambda K=K: ext.quanpc_fused_output_projection_splitk_fn(
            dense, code, den, vproj, attn_test, K), args.warmup, args.iters)
        # K<=0 requests device-aware auto K inside the wrapper; block count is
        # then resolved on-device and unknown here.
        k_label = "auto" if K <= 0 else str(K)
        blocks = "?" if K <= 0 else str(rows * K)
        print(f"{k_label:>8} {blocks:>8} {us:>10.3f} {base_us/us:>7.2f}x "
              f"{match:>9.2f}% {max_err:>12.3e}")


if __name__ == "__main__":
    main()
