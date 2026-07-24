"""Isolated CUDA prefill path for QuanPC experiments.

This module is intentionally separate from ``quanpc.py`` and the existing decode
CUDA extension.  It accelerates the prefill-side steps after projections are
known: projection quantization, EAS selection, dense retained-KV construction,
and packed global-code generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

try:
    import torch
    from torch.utils.cpp_extension import load
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    load = None

_EXTENSION = None


@dataclass
class PrefillCudaOutput:
    dense_key: "torch.Tensor"
    dense_value: "torch.Tensor"
    packed_key: "torch.Tensor"
    packed_value: "torch.Tensor"
    key_proj: "torch.Tensor"
    value_proj: "torch.Tensor"
    key_err: "torch.Tensor"
    value_err: "torch.Tensor"
    eas_indices: "torch.Tensor"
    kept_indices: "torch.Tensor"
    packed_key_all: "torch.Tensor | None" = None
    packed_value_token: "torch.Tensor | None" = None


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    return torch


def _ensure_build_tool_path() -> None:
    python_bin = os.path.dirname(sys.executable)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin and python_bin not in parts:
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")


def load_quanpc_prefill_extension(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    _require_torch()
    if load is None:
        raise RuntimeError("torch.utils.cpp_extension.load is unavailable.")
    _ensure_build_tool_path()
    root = Path(__file__).resolve().parent
    _EXTENSION = load(
        "quanpc_prefill_cuda",
        [str(root / "cuda_kernel.cu"), str(root / "torch_extension.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-O3"],
        verbose=verbose,
    )
    return _EXTENSION


def _check_full_inputs(full_key, full_value, key_projection, value_projection, sink_size: int, local_size: int):
    torch_mod = _require_torch()
    for name, tensor in {
        "full_key": full_key,
        "full_value": full_value,
        "key_projection": key_projection,
        "value_projection": value_projection,
    }.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D; got {tuple(tensor.shape)}.")
    if full_key.dtype != full_value.dtype or full_key.dtype != key_projection.dtype or full_key.dtype != value_projection.dtype:
        raise TypeError("full K/V and projections must have the same dtype.")
    if full_key.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError(f"expected bf16/fp32 tensors; got {full_key.dtype}.")
    if tuple(full_key.shape) != tuple(full_value.shape):
        raise ValueError("full_key and full_value shapes must match.")
    b, h, total_len, d = full_key.shape
    if key_projection.shape[0] != b or key_projection.shape[1] != h or key_projection.shape[3] != d:
        raise ValueError("key_projection must be [B,H,Ck,D] matching full_key.")
    if value_projection.shape[0] != b or value_projection.shape[1] != h or value_projection.shape[3] != d:
        raise ValueError("value_projection must be [B,H,Cv,D] matching full_value.")
    if key_projection.shape[2] % 64 != 0 or value_projection.shape[2] % 64 != 0:
        raise ValueError("K/V code dimensions must be divisible by 64.")
    glob_len = total_len - sink_size - local_size
    if sink_size < 0 or local_size < 0 or glob_len <= 0:
        raise ValueError(f"invalid sink/local sizes for total_len={total_len}: sink={sink_size}, local={local_size}.")
    return glob_len


def learn_projection_cuda(full_states, init_vectors, glob_start: int, glob_len: int, zerorope: int = 0, apply_zerorope: bool = False, verbose: bool = False):
    ext = load_quanpc_prefill_extension(verbose)
    (projection,) = ext.quanpc_prefill_learn_projection_fn(
        full_states.contiguous(),
        init_vectors.contiguous(),
        int(glob_start),
        int(glob_len),
        int(zerorope),
        bool(apply_zerorope),
    )
    return projection


def quantize_pack_cuda(full_states, projection, glob_start: int, glob_len: int, zerorope: int = 0, apply_zerorope: bool = False, pack_key_layout: bool = True, verbose: bool = False):
    ext = load_quanpc_prefill_extension(verbose)
    return ext.quanpc_prefill_quantize_pack_fn(
        full_states.contiguous(),
        projection.contiguous(),
        int(glob_start),
        int(glob_len),
        int(zerorope),
        bool(apply_zerorope),
        bool(pack_key_layout),
    )


def quantize_unpacked_cuda(full_states, projection, glob_start: int, glob_len: int, zerorope: int = 0, apply_zerorope: bool = False, verbose: bool = False):
    ext = load_quanpc_prefill_extension(verbose)
    return ext.quanpc_prefill_quantize_fn(
        full_states.contiguous(),
        projection.contiguous(),
        int(glob_start),
        int(glob_len),
        int(zerorope),
        bool(apply_zerorope),
    )


def select_eas_cuda(key_err, value_err, eas_size: int, verbose: bool = False):
    ext = load_quanpc_prefill_extension(verbose)
    return ext.quanpc_prefill_select_eas_fn(key_err.contiguous(), value_err.contiguous(), int(eas_size))


def gather_pack_cuda(full_key, full_value, key_codes, value_codes, eas_indices, kept_indices, sink_size: int, local_size: int, verbose: bool = False):
    ext = load_quanpc_prefill_extension(verbose)
    return ext.quanpc_prefill_gather_pack_fn(
        full_key.contiguous(),
        full_value.contiguous(),
        key_codes.contiguous(),
        value_codes.contiguous(),
        eas_indices.contiguous(),
        kept_indices.contiguous(),
        int(sink_size),
        int(local_size),
    )


def gather_pack_from_packed_cuda(full_key, full_value, packed_key_all, packed_value_token, eas_indices, kept_indices, sink_size: int, local_size: int, verbose: bool = False):
    ext = load_quanpc_prefill_extension(verbose)
    return ext.quanpc_prefill_gather_pack_from_packed_fn(
        full_key.contiguous(),
        full_value.contiguous(),
        packed_key_all.contiguous(),
        packed_value_token.contiguous(),
        eas_indices.contiguous(),
        kept_indices.contiguous(),
        int(sink_size),
        int(local_size),
    )


def prepare_prefill_cuda(
    full_key,
    full_value,
    key_projection,
    value_projection,
    *,
    sink_size: int = 64,
    local_size: int = 1024,
    eas: float = 0.01,
    zerorope: int = 20,
    keep_intermediates: bool = False,
    overlap_quant: bool = True,
    verbose: bool = False,
) -> PrefillCudaOutput:
    glob_len = _check_full_inputs(full_key, full_value, key_projection, value_projection, sink_size, local_size)
    eas_size = max(1, int(glob_len * float(eas)))
    load_quanpc_prefill_extension(verbose)

    if overlap_quant and full_key.device == full_value.device and full_key.is_cuda:
        current = torch.cuda.current_stream(full_key.device)
        value_stream = torch.cuda.Stream(device=full_key.device)
        with torch.cuda.stream(value_stream):
            packed_value_token, value_proj, value_err = quantize_pack_cuda(
                full_value,
                value_projection,
                sink_size,
                glob_len,
                zerorope=0,
                apply_zerorope=False,
                pack_key_layout=False,
                verbose=verbose,
            )
        packed_key_all, key_proj, key_err = quantize_pack_cuda(
            full_key,
            key_projection,
            sink_size,
            glob_len,
            zerorope=zerorope,
            apply_zerorope=True,
            pack_key_layout=True,
            verbose=verbose,
        )
        current.wait_stream(value_stream)
    else:
        packed_key_all, key_proj, key_err = quantize_pack_cuda(
            full_key,
            key_projection,
            sink_size,
            glob_len,
            zerorope=zerorope,
            apply_zerorope=True,
            pack_key_layout=True,
            verbose=verbose,
        )
        packed_value_token, value_proj, value_err = quantize_pack_cuda(
            full_value,
            value_projection,
            sink_size,
            glob_len,
            zerorope=0,
            apply_zerorope=False,
            pack_key_layout=False,
            verbose=verbose,
        )
    eas_indices, _mask, kept_indices = select_eas_cuda(key_err, value_err, eas_size, verbose=verbose)
    dense_key, dense_value, packed_key, packed_value = gather_pack_from_packed_cuda(
        full_key,
        full_value,
        packed_key_all,
        packed_value_token,
        eas_indices,
        kept_indices,
        sink_size,
        local_size,
        verbose=verbose,
    )
    return PrefillCudaOutput(
        dense_key=dense_key,
        dense_value=dense_value,
        packed_key=packed_key,
        packed_value=packed_value,
        key_proj=key_proj,
        value_proj=value_proj,
        key_err=key_err,
        value_err=value_err,
        eas_indices=eas_indices,
        kept_indices=kept_indices,
        packed_key_all=packed_key_all if keep_intermediates else None,
        packed_value_token=packed_value_token if keep_intermediates else None,
    )
