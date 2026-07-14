"""Correctness tests for QuanPC CUDA projection learning."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc_prefill_cuda_extension.quanpc_prefill_cuda import learn_projection_cuda, quantize_unpacked_cuda  # noqa: E402


def torch_learn_projection(full: torch.Tensor, init: torch.Tensor, sink: int, glob_len: int, zerorope: int, apply_zerorope: bool, power_iters: int = 2):
    residual = full[:, :, sink:sink + glob_len, :].clone()
    if apply_zerorope and zerorope > 0:
        half = residual.shape[-1] // 2
        residual[:, :, :, :zerorope] = 0
        residual[:, :, :, half:half + zerorope] = 0
    code_norm = math.sqrt(float(glob_len))
    proj = torch.empty_like(init)
    for bit in range(init.shape[-2]):
        vh0 = init[:, :, bit, :]
        ata = residual.transpose(-1, -2) @ residual
        direction = torch.einsum("bhde,bhe->bhd", ata, vh0)
        if power_iters >= 2:
            direction = torch.einsum("bhde,bhe->bhd", ata, direction)
        scores = torch.einsum("bhld,bhd->bhl", residual, direction)
        sign = torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))
        basis = sign / code_norm
        vh = torch.einsum("bhld,bhl->bhd", residual, basis)
        residual = residual - torch.einsum("bhl,bhd->bhld", basis, vh)
        proj[:, :, bit, :] = vh
    return proj.contiguous()


def run_case(dtype: torch.dtype, apply_zerorope: bool):
    torch.manual_seed(321)
    device = "cuda"
    b, h, total, d = 1, 2, 112, 128
    sink, local = 4, 8
    glob_len = total - sink - local
    bits = 64
    zerorope = 20 if apply_zerorope else 0
    full = torch.randn(b, h, total, d, device=device, dtype=dtype).contiguous()
    init = torch.randn(b, h, bits, d, device=device, dtype=dtype).contiguous()
    got = learn_projection_cuda(full, init, sink, glob_len, zerorope, apply_zerorope)
    ref_same = torch_learn_projection(full, init, sink, glob_len, zerorope, apply_zerorope, power_iters=1)
    ref_quality = torch_learn_projection(full, init, sink, glob_len, zerorope, apply_zerorope, power_iters=2)
    if dtype == torch.float32:
        torch.testing.assert_close(got, ref_same, atol=3e-3, rtol=3e-3)
    if not torch.isfinite(got.float()).all():
        raise AssertionError("CUDA projection contains non-finite values")

    _, _, got_err = quantize_unpacked_cuda(full, got, sink, glob_len, zerorope, apply_zerorope)
    _, _, ref_err = quantize_unpacked_cuda(full, ref_quality, sink, glob_len, zerorope, apply_zerorope)
    mean_rel = (got_err.mean() - ref_err.mean()).abs() / ref_err.mean().clamp_min(1e-6)
    p95_rel = (torch.quantile(got_err.float(), 0.95) - torch.quantile(ref_err.float(), 0.95)).abs() / torch.quantile(ref_err.float(), 0.95).clamp_min(1e-6)
    if dtype == torch.float32:
        if mean_rel.item() > 0.10 or p95_rel.item() > 0.12:
            raise AssertionError(f"fp32 projection error drift too large: mean={mean_rel.item():.4f} p95={p95_rel.item():.4f}")
    else:
        if mean_rel.item() > 0.15 or p95_rel.item() > 0.20:
            raise AssertionError(f"bf16 projection error drift too large: mean={mean_rel.item():.4f} p95={p95_rel.item():.4f}")
    print(f"projection learning case dtype={dtype} zerorope={apply_zerorope} passed")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"device: {torch.cuda.get_device_name()}")
    for name in [x for x in args.dtypes.split(",") if x]:
        dtype = getattr(torch, name)
        run_case(dtype, False)
        run_case(dtype, True)
    print("projection learning CUDA correctness passed")


if __name__ == "__main__":
    main()
