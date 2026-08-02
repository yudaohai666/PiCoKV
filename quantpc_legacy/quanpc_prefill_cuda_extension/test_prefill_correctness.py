"""Correctness tests for the isolated QuanPC prefill CUDA path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quanpc_prefill_cuda_extension.quanpc_prefill_cuda import (  # noqa: E402
    gather_pack_from_packed_cuda,
    quantize_pack_cuda,
    quantize_unpacked_cuda,
    select_eas_cuda,
)


def pack64(bits: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(bits.shape[:-1], device=bits.device, dtype=torch.long)
    for bit in range(64):
        out |= bits[..., bit].to(torch.long) << bit
    return out


def pack_key_ref(codes: torch.Tensor) -> torch.Tensor:
    b, h, l, c = codes.shape
    return pack64((codes < 0).reshape(b, h, l, c // 64, 64)).transpose(2, 3).contiguous()


def pack_value_token_ref(codes: torch.Tensor) -> torch.Tensor:
    b, h, l, c = codes.shape
    return pack64((codes > 0).reshape(b, h, l, c // 64, 64)).contiguous()


def pack_value_final_ref(codes: torch.Tensor) -> torch.Tensor:
    b, h, l, c = codes.shape
    groups = (l + 63) // 64
    positive = codes > 0
    if groups * 64 != l:
        padded = torch.zeros(b, h, groups * 64, c, device=codes.device, dtype=torch.bool)
        padded[:, :, :l, :] = positive
        positive = padded
    value = positive.reshape(b, h, groups, 64, c).transpose(-1, -2)
    return pack64(value).contiguous()


def torch_quantize(full: torch.Tensor, proj: torch.Tensor, sink: int, glob_len: int, zerorope: int, apply_zerorope: bool):
    residual = full[:, :, sink:sink + glob_len, :].clone()
    if apply_zerorope and zerorope > 0:
        half = residual.shape[-1] // 2
        residual[:, :, :, :zerorope] = 0
        residual[:, :, :, half:half + zerorope] = 0
    codes = torch.empty(*residual.shape[:-1], proj.shape[-2], device=full.device, dtype=torch.int8)
    for bit in range(proj.shape[-2]):
        direction = proj[:, :, bit, :]
        score = torch.einsum("bhld,bhd->bhl", residual, direction)
        sign = torch.where(score >= 0, torch.ones_like(score), -torch.ones_like(score))
        codes[:, :, :, bit] = sign.to(torch.int8)
        residual = residual - torch.einsum("bhl,bhd->bhld", sign, direction)
    err = residual.norm(dim=-1, p=2).float()
    return codes.contiguous(), proj.contiguous(), err.contiguous()


def joint_eas_ref(k_err: torch.Tensor, v_err: torch.Tensor, eas_size: int):
    l = k_err.shape[-1]
    k_count = min(eas_size, l)
    k_idx = torch.topk(k_err, k=k_count, dim=-1).indices
    if k_count == l:
        return k_idx.sort(dim=-1).values
    v_count = min(eas_size, l - k_count)
    selected = torch.zeros_like(v_err, dtype=torch.bool).scatter_(-1, k_idx, True)
    v_idx = torch.topk(v_err.masked_fill(selected, -torch.inf), k=v_count, dim=-1).indices
    return torch.cat([k_idx, v_idx], dim=-1).sort(dim=-1).values


def gather_dense_ref(full: torch.Tensor, eas: torch.Tensor, sink: int, local: int):
    b, h, total, d = full.shape
    glob_start = sink
    eas_full = torch.gather(
        full[:, :, glob_start:total - local, :],
        dim=-2,
        index=eas.unsqueeze(-1).expand(-1, -1, -1, d),
    )
    return torch.cat([full[:, :, :sink, :], eas_full, full[:, :, total - local:, :]], dim=-2).contiguous()


def drop_codes(codes: torch.Tensor, eas: torch.Tensor) -> torch.Tensor:
    b, h, l, c = codes.shape
    mask = torch.ones(b, h, l, device=codes.device, dtype=torch.bool)
    mask.scatter_(-1, eas, False)
    return codes[mask].reshape(b, h, l - eas.shape[-1], c).contiguous()


def run_case(dtype: torch.dtype):
    torch.manual_seed(123)
    device = "cuda"
    b, h, total, d = 1, 2, 96, 32
    sink, local = 4, 8
    glob_len = total - sink - local
    ck, cv = 128, 192
    zerorope = 3
    eas_size = 5
    full_k = torch.randn(b, h, total, d, device=device, dtype=dtype).contiguous()
    full_v = torch.randn(b, h, total, d, device=device, dtype=dtype).contiguous()
    k_proj = (torch.randn(b, h, ck, d, device=device, dtype=dtype) * 0.02).contiguous()
    v_proj = (torch.randn(b, h, cv, d, device=device, dtype=dtype) * 0.02).contiguous()

    k_codes, k_proj_out, k_err = quantize_unpacked_cuda(full_k, k_proj, sink, glob_len, zerorope, True)
    v_codes, v_proj_out, v_err = quantize_unpacked_cuda(full_v, v_proj, sink, glob_len, 0, False)
    pk_all, k_proj_out_fused, k_err_fused = quantize_pack_cuda(full_k, k_proj, sink, glob_len, zerorope, True, True)
    pv_token, v_proj_out_fused, v_err_fused = quantize_pack_cuda(full_v, v_proj, sink, glob_len, 0, False, False)

    assert torch.equal(pk_all, pack_key_ref(k_codes)), "fused packed K does not match unpacked CUDA codes"
    assert torch.equal(pv_token, pack_value_token_ref(v_codes)), "fused token-packed V does not match unpacked CUDA codes"
    assert torch.allclose(k_proj_out.float(), k_proj_out_fused.float(), atol=0, rtol=0)
    assert torch.allclose(v_proj_out.float(), v_proj_out_fused.float(), atol=0, rtol=0)
    assert torch.allclose(k_err, k_err_fused, atol=0, rtol=0)
    assert torch.allclose(v_err, v_err_fused, atol=0, rtol=0)

    if dtype == torch.float32:
        k_ref_codes, k_ref_proj_out, k_ref_err = torch_quantize(full_k, k_proj, sink, glob_len, zerorope, True)
        v_ref_codes, v_ref_proj_out, v_ref_err = torch_quantize(full_v, v_proj, sink, glob_len, 0, False)
        assert torch.equal(k_codes, k_ref_codes), "CUDA K codes differ from torch fp32 reference"
        assert torch.equal(v_codes, v_ref_codes), "CUDA V codes differ from torch fp32 reference"
        assert torch.allclose(k_proj_out, k_ref_proj_out, atol=1e-6, rtol=1e-6)
        assert torch.allclose(v_proj_out, v_ref_proj_out, atol=1e-6, rtol=1e-6)
        assert torch.allclose(k_err, k_ref_err, atol=1e-5, rtol=1e-5)
        assert torch.allclose(v_err, v_ref_err, atol=1e-5, rtol=1e-5)

    eas_idx, mask, kept_idx = select_eas_cuda(k_err, v_err, eas_size)
    eas_ref = joint_eas_ref(k_err, v_err, eas_size)
    assert torch.equal(eas_idx, eas_ref), "EAS indices differ from torch reference"
    dense_k, dense_v, packed_k, packed_v = gather_pack_from_packed_cuda(
        full_k, full_v, pk_all, pv_token, eas_idx, kept_idx, sink, local
    )
    assert torch.equal(dense_k, gather_dense_ref(full_k, eas_ref, sink, local)), "dense K gather mismatch"
    assert torch.equal(dense_v, gather_dense_ref(full_v, eas_ref, sink, local)), "dense V gather mismatch"
    k_kept = drop_codes(k_codes, eas_ref)
    v_kept = drop_codes(v_codes, eas_ref)
    assert torch.equal(packed_k, pack_key_ref(k_kept)), "final packed K mismatch"
    assert torch.equal(packed_v, pack_value_final_ref(v_kept)), "final packed V mismatch"
    assert mask.shape == (b, h, glob_len)
    print(f"case dtype={dtype} passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", default="float32,bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"device: {torch.cuda.get_device_name()}")
    for name in [x for x in args.dtypes.split(",") if x]:
        run_case(getattr(torch, name))
    print("prefill CUDA correctness passed")


if __name__ == "__main__":
    main()
