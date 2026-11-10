"""Standalone CUDA kernels for PiCoKV experiments.

This file intentionally loads separate sources from the existing binary-projection
extension so experimental PiCoKV kernels do not modify the original extension.
"""

from __future__ import annotations

import os
import re
import sys
import importlib.util
from pathlib import Path
try:
    import torch
    from torch.utils.cpp_extension import load
except ModuleNotFoundError:
    torch = None
    load = None

_EXTENSION = None


def _prebuilt_path() -> Path:
    """Return the optional install-time build directory."""
    configured = os.environ.get("PICOKV_PREBUILT_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "picokv_extensions"


def _arch_error(so: Path):
    """A prebuilt .so whose build (the build.ninja torch writes next to it)
    has no code for the visible device: its kernels would fail at the first
    launch ("no kernel image"). None when it fits or nothing can be checked."""
    ninja = so.parent / "build.ninja"
    if torch is None or not ninja.is_file() or not torch.cuda.is_available():
        return None
    try:
        text = ninja.read_text(errors="ignore")
    except OSError:
        return None
    sass, ptx = set(), set()
    for kind, num in re.findall(r"code=\[?(sm|compute)_(\d+)a?", text):
        (sass if kind == "sm" else ptx).add(int(num))
    if not sass and not ptx:
        return None
    major, minor = torch.cuda.get_device_capability()
    dev = 10 * major + minor
    # SASS runs on the same major at >= minor; PTX JIT-compiles forward
    if any(a // 10 == major and a % 10 <= minor for a in sass) or any(a <= dev for a in ptx):
        return None
    built = ", ".join([f"sm_{a}" for a in sorted(sass)] + [f"compute_{a}" for a in sorted(ptx)])
    return (f"PiCoKV extension {so} was built for {built}, the device is sm_{dev}: "
            f"rebuild with TORCH_CUDA_ARCH_LIST={major}.{minor} python "
            f"picokv/build_cuda_extensions.py, or point PICOKV_PREBUILT_DIR at a "
            f"matching build")


def loaded_extension_path():
    """Path of the loaded extension .so (None: none loaded yet)."""
    return getattr(_EXTENSION, "__file__", None)


def _load_prebuilt_extension():
    """Load the install-time .so without invoking torch's JIT builder."""
    if os.environ.get("PICOKV_DISABLE_PREBUILT") == "1":
        return None
    root = _prebuilt_path()
    candidates = sorted(root.glob("_picokv_cuda*.so"))
    if not candidates:
        return None
    module_name = "_picokv_cuda"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    err = _arch_error(candidates[0])
    if err is not None:
        raise ImportError(err)
    spec = importlib.util.spec_from_file_location(module_name, candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prebuilt PiCoKV extension: {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required, but it is not installed in this Python environment.")
    return torch


def _require_cuda():
    torch_mod = _require_torch()
    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA is required to run the PiCoKV CUDA extension.")
    return torch_mod


def _ensure_build_tool_path() -> None:
    python_bin = os.path.dirname(sys.executable)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin and python_bin not in path_parts:
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")


def load_picokv_extension(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    prebuilt = _load_prebuilt_extension()
    if prebuilt is not None:
        _EXTENSION = prebuilt
        return _EXTENSION

    _require_torch()
    if load is None:
        raise RuntimeError("torch.utils.cpp_extension.load is unavailable.")
    _ensure_build_tool_path()

    curr_path = os.path.dirname(os.path.realpath(__file__))
    src_files = [
        os.path.join(curr_path, "cuda_kernel.cu"),
        os.path.join(curr_path, "torch_extension.cpp"),
    ]
    _EXTENSION = load(
        "picokv_cuda",
        src_files,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-O3"],
        verbose=verbose,
    )
    return _EXTENSION


def _pack_64_bits(bits):
    """Pack bool/int bits in the last dim of size 64 into torch.long words."""
    torch_mod = _require_torch()
    if bits.shape[-1] != 64:
        raise ValueError(f"last dimension must be 64; got {bits.shape[-1]}.")
    packed = torch_mod.zeros(bits.shape[:-1], device=bits.device, dtype=torch_mod.long)
    for bit_idx in range(64):
        packed = packed | (bits[..., bit_idx].to(torch_mod.long) << bit_idx)
    return packed


def pack_projected_query_torch(quant_projected_query):
    """Pack int8 projected query into sign + 7 magnitude bit-planes.

    Input shape is ``[B,H,S,Ck]`` with ``Ck`` divisible by 64.  Output shape is
    ``[B,H,S,Ck//64,8]``.  Plane 0 stores signs; planes 1..7 store absolute-value
    bits, matching the bitwise dot-product formula used by the CUDA kernel.
    """
    torch_mod = _require_torch()
    if quant_projected_query.dtype != torch_mod.int8:
        raise TypeError(f"quant_projected_query must be int8; got {quant_projected_query.dtype}.")
    if quant_projected_query.shape[-1] % 64 != 0:
        raise ValueError(f"Ck must be divisible by 64; got {quant_projected_query.shape[-1]}.")
    q = quant_projected_query.to(torch_mod.int32)
    q = q.reshape(*q.shape[:-1], q.shape[-1] // 64, 64)
    mag = q.abs()
    planes = []
    planes.append(_pack_64_bits(q < 0))
    for bit_idx in range(7):
        planes.append(_pack_64_bits(((mag >> bit_idx) & 1) != 0))
    return torch_mod.stack(planes, dim=-1).contiguous()






def project_quantize_pack_query_cuda(query, key_proj, verbose: bool = False):
    """Experimental fused scalar CUDA query projection + quantize + pack.

    This avoids the intermediate projected-query tensor, but it does not use the
    tensor-core bmm path. Keep it opt-in and compare accuracy before using it in
    model runs.
    """
    torch_mod = _require_cuda()
    if not query.is_cuda or not key_proj.is_cuda:
        raise ValueError("query and key_proj must be CUDA tensors.")
    if query.dtype != key_proj.dtype or query.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError("query and key_proj must both be bfloat16 or both be float32.")
    if not query.is_contiguous():
        query = query.contiguous()
    if not key_proj.is_contiguous():
        key_proj = key_proj.contiguous()
    if query.dim() != 4 or key_proj.dim() != 4:
        raise ValueError("query must be [B,QH,S,D] and key_proj must be [B,KVH,C,D].")
    B, QH, S, D = query.shape
    if key_proj.shape[0] != B or key_proj.shape[3] != D:
        raise ValueError("key_proj B/D dimensions must match query.")
    KVH = key_proj.shape[1]
    C = key_proj.shape[2]
    if QH % KVH != 0:
        raise ValueError(f"QH must be divisible by KVH; got QH={QH}, KVH={KVH}.")
    if C % 64 != 0:
        raise ValueError(f"C must be divisible by 64; got {C}.")
    if C > 512 or D > 256:
        raise ValueError(f"fused project+pack supports C <= 512 and D <= 256; got C={C}, D={D}.")
    packed = torch_mod.empty(B, QH, S, C // 64, 8, device=query.device, dtype=torch_mod.long)
    query_scale = torch_mod.empty(B, QH, S, device=query.device, dtype=torch_mod.float32)
    load_picokv_extension(verbose).picokv_project_quantize_pack_query_fn(query, key_proj, packed, query_scale)
    return packed, query_scale


def _check_projected_query_for_pack(projected_query):
    torch_mod = _require_cuda()
    if not projected_query.is_cuda:
        raise ValueError("projected_query must be a CUDA tensor.")
    if projected_query.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError(f"projected_query must be bfloat16 or float32; got {projected_query.dtype}.")
    if not projected_query.is_contiguous():
        projected_query = projected_query.contiguous()
    if projected_query.dim() != 4:
        raise ValueError("projected_query must have shape [B,H,S,C].")
    if projected_query.shape[-1] % 64 != 0:
        raise ValueError(f"C must be divisible by 64; got {projected_query.shape[-1]}.")
    if projected_query.shape[-1] > 512:
        raise ValueError(f"fused projected-query pack supports C <= 512; got {projected_query.shape[-1]}.")
    return torch_mod, projected_query


def quantize_pack_projected_query_cuda_out(projected_query, packed, query_scale, verbose: bool = False):
    """Write fused projected-query quantization and bit packing into caller-owned outputs."""
    torch_mod, projected_query = _check_projected_query_for_pack(projected_query)
    expected_packed = (*projected_query.shape[:-1], projected_query.shape[-1] // 64, 8)
    if tuple(packed.shape) != expected_packed or packed.dtype != torch_mod.long or not packed.is_cuda:
        raise ValueError(f"packed must be CUDA int64 with shape {expected_packed}; got {tuple(packed.shape)} {packed.dtype}.")
    if tuple(query_scale.shape) != tuple(projected_query.shape[:-1]) or query_scale.dtype != torch_mod.float32 or not query_scale.is_cuda:
        raise ValueError(f"query_scale must be CUDA fp32 with shape {tuple(projected_query.shape[:-1])}.")
    if not packed.is_contiguous():
        raise ValueError("packed must be contiguous.")
    if not query_scale.is_contiguous():
        raise ValueError("query_scale must be contiguous.")
    load_picokv_extension(verbose).picokv_quantize_pack_projected_query_fn(projected_query, packed, query_scale)
    return packed, query_scale


def quantize_pack_projected_query_cuda(projected_query, verbose: bool = False):
    """Fused CUDA quantization and bit packing for projected queries.

    Input shape is ``[B,H,S,C]`` bf16/fp32. The output packed shape is
    ``[B,H,S,C//64,8]`` and ``query_scale`` is ``[B,H,S]`` fp32, matching
    ``_quantize_projected_query_int8`` followed by ``pack_projected_query_cuda``.
    """
    torch_mod, projected_query = _check_projected_query_for_pack(projected_query)
    packed = torch_mod.empty(
        *projected_query.shape[:-1],
        projected_query.shape[-1] // 64,
        8,
        device=projected_query.device,
        dtype=torch_mod.long,
    )
    query_scale = torch_mod.empty(*projected_query.shape[:-1], device=projected_query.device, dtype=torch_mod.float32)
    return quantize_pack_projected_query_cuda_out(projected_query, packed, query_scale, verbose=verbose)


def pack_projected_query_cuda(quant_projected_query, verbose: bool = False):
    """CUDA implementation of projected-query bit packing.

    Input shape is ``[B,H,S,C]`` int8 with ``C`` divisible by 64. Output shape is
    ``[B,H,S,C//64,8]`` and matches ``pack_projected_query_torch``.
    """
    torch_mod = _require_cuda()
    if quant_projected_query.dtype != torch_mod.int8:
        raise TypeError(f"quant_projected_query must be int8; got {quant_projected_query.dtype}.")
    if not quant_projected_query.is_cuda:
        raise ValueError("quant_projected_query must be a CUDA tensor.")
    if not quant_projected_query.is_contiguous():
        quant_projected_query = quant_projected_query.contiguous()
    if quant_projected_query.shape[-1] % 64 != 0:
        raise ValueError(f"C must be divisible by 64; got {quant_projected_query.shape[-1]}.")
    packed = torch_mod.empty(
        *quant_projected_query.shape[:-1],
        quant_projected_query.shape[-1] // 64,
        8,
        device=quant_projected_query.device,
        dtype=torch_mod.long,
    )
    load_picokv_extension(verbose).picokv_pack_projected_query_fn(quant_projected_query, packed)
    return packed


def pack_picokv_key_torch(quant_key):
    """Pack unpacked 1-bit PiCoKV key codes into int64 words.

    Input shape is ``[B,H,L,Ck]`` with values ``+1`` or ``-1`` and ``Ck``
    divisible by 64.  Output shape is ``[B,H,L,Ck//64]``.  Bit=1 denotes key
    code ``-1``; bit=0 denotes ``+1``.
    """
    torch_mod = _require_torch()
    if quant_key.shape[-1] % 64 != 0:
        raise ValueError(f"Ck must be divisible by 64; got {quant_key.shape[-1]}.")
    key = quant_key.reshape(*quant_key.shape[:-1], quant_key.shape[-1] // 64, 64)
    return _pack_64_bits(key < 0).contiguous()




def pack_picokv_key_transposed_torch(quant_key):
    """Pack 1-bit PiCoKV key codes as ``[B, H, Ck//64, L]``.

    This layout makes fixed-code-block loads contiguous across token threads in
    the tiled CUDA kernel.
    """
    return pack_picokv_key_torch(quant_key).transpose(2, 3).contiguous()


def pack_picokv_value_torch(quant_value, value_bits: int):
    """Pack PiCoKV value codes as unsigned bit-planes.

    The CUDA kernel decodes each raw integer code as
    ``code - ((2 ** value_bits - 1) / 2)``.  Floating inputs are interpreted as
    centered codebook values, e.g. 2-bit ``[-1.5, -0.5, 0.5, 1.5]``.  Integer
    inputs are interpreted as raw codes in ``[0, 2 ** value_bits - 1]``.
    """
    torch_mod = _require_torch()
    if quant_value.shape[-1] % 64 != 0:
        raise ValueError(f"Cv must be divisible by 64; got {quant_value.shape[-1]}.")
    if value_bits < 2 or value_bits > 8:
        raise ValueError(f"value_bits must be in [2, 8]; got {value_bits}.")

    max_code = (1 << value_bits) - 1
    zero_point = max_code / 2.0
    if quant_value.is_floating_point():
        raw_code = torch_mod.round(quant_value.float() + zero_point).to(torch_mod.long)
        if raw_code.numel() > 0:
            if raw_code.min().item() < 0 or raw_code.max().item() > max_code:
                raise ValueError(f"floating quant_value contains entries outside the {value_bits}-bit codebook.")
            expected = raw_code.float() - zero_point
            if not torch_mod.allclose(quant_value.float(), expected, atol=1e-3, rtol=0.0):
                raise ValueError("floating quant_value must contain exact centered codebook values.")
    else:
        raw_code = quant_value.to(torch_mod.long)
        if raw_code.numel() > 0 and (raw_code.min().item() < 0 or raw_code.max().item() > max_code):
            raise ValueError(f"integer quant_value raw codes must be in [0, {max_code}].")

    value = raw_code.reshape(*raw_code.shape[:-1], raw_code.shape[-1] // 64, 64)
    planes = []
    for bit_idx in range(value_bits):
        planes.append(_pack_64_bits(((value >> bit_idx) & 1) != 0))
    return torch_mod.stack(planes, dim=-1).contiguous()



def pack_picokv_value_1bit_transposed_torch(quant_value):
    """Pack 1-bit value signs as ``[B, H, ceil(L / 64), Cv]``.

    Input shape is ``[B,H,L,Cv]`` with values centered on ``-1`` and ``+1``.
    Bit=1 denotes value code ``+1``; bit=0 denotes value code ``-1``.  The
    group-major layout makes adjacent value dimensions contiguous for the CUDA
    popcount accumulator.
    """
    torch_mod = _require_torch()
    if quant_value.shape[-1] % 2 != 0:
        raise ValueError(f"Cv must be even for bf16 pair output; got {quant_value.shape[-1]}.")
    L = quant_value.shape[-2]
    Cv = quant_value.shape[-1]
    groups = (L + 63) // 64
    positive = quant_value > 0
    if groups * 64 != L:
        padded = torch_mod.zeros(*positive.shape[:-2], groups * 64, Cv, device=positive.device, dtype=torch_mod.bool)
        padded[..., :L, :] = positive
        positive = padded
    value = positive.reshape(*positive.shape[:-2], groups, 64, Cv).transpose(-1, -2)
    return _pack_64_bits(value).contiguous()



def _check_dense_attention_inputs(query_states, dense_key, dense_value):
    torch_mod = _require_cuda()
    tensors = {
        "query_states": query_states,
        "dense_key": dense_key,
        "dense_value": dense_value,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D; got shape {tuple(tensor.shape)}.")
    if query_states.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError(f"query_states must be bfloat16 or float32; got {query_states.dtype}.")
    if dense_key.dtype != query_states.dtype or dense_value.dtype != query_states.dtype:
        raise TypeError("query_states, dense_key, and dense_value must have the same dtype.")

    B, QH, S, D = query_states.shape
    if dense_key.shape[0] != B or dense_value.shape[0] != B:
        raise ValueError("dense KV B dimension must match query_states.")
    KVH = dense_key.shape[1]
    if KVH <= 0 or QH % KVH != 0:
        raise ValueError(f"QH must be divisible by KVH for GQA dense attention; got QH={QH}, KVH={KVH}.")
    if dense_key.shape[2] <= 0:
        raise ValueError("dense_key must contain at least one retained token.")
    if tuple(dense_value.shape) != tuple(dense_key.shape):
        raise ValueError("dense_value shape must match dense_key shape.")
    if dense_key.shape[3] != D:
        raise ValueError(f"dense KV head_dim must match query head_dim; got {dense_key.shape[3]} and {D}.")
    if D > 256:
        raise ValueError(f"dense CUDA attention supports head_dim <= 256; got {D}.")
    if dense_key.shape[2] > 12288:
        raise ValueError(f"dense retained length is too large for shared-score CUDA kernel; got {dense_key.shape[2]}.")
    return torch_mod, B, QH, S, D


def picokv_dense_attention_numerator_cuda(query_states, dense_key, dense_value, attention_scale: float, verbose: bool = False):
    """CUDA dense retained-KV numerator and logsumexp for PiCoKV decode.

    This matches ``_dense_attention_numerator`` in ``picokv.py`` but keeps dense
    K/V in GQA layout ``[B, KVH, L, D]`` instead of materializing repeated heads.
    Returns ``dense_output`` with query dtype and ``dense_lse`` as ``[B,QH,S]`` fp32.
    """
    torch_mod, B, QH, S, _ = _check_dense_attention_inputs(query_states, dense_key, dense_value)
    if not query_states.is_contiguous():
        query_states = query_states.contiguous()
    if not dense_key.is_contiguous():
        dense_key = dense_key.contiguous()
    if not dense_value.is_contiguous():
        dense_value = dense_value.contiguous()

    dense_output = torch_mod.empty_like(query_states)
    dense_lse = torch_mod.empty(B, QH, S, device=query_states.device, dtype=torch_mod.float32)
    load_picokv_extension(verbose).picokv_dense_attention_numerator_fn(
        query_states,
        dense_key,
        dense_value,
        dense_output,
        dense_lse,
        float(attention_scale),
    )
    return dense_output, dense_lse



def _check_fused_output_inputs(dense_output, glob_code_output, glob_den, value_proj):
    torch_mod = _require_cuda()
    tensors = {
        "dense_output": dense_output,
        "glob_code_output": glob_code_output,
        "glob_den": glob_den,
        "value_proj": value_proj,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if dense_output.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError(f"dense_output must be bfloat16 or float32; got {dense_output.dtype}.")
    if value_proj.dtype != dense_output.dtype:
        raise TypeError("value_proj dtype must match dense_output dtype.")
    if glob_code_output.dtype != torch_mod.float32 or glob_den.dtype != torch_mod.float32:
        raise TypeError("glob_code_output and glob_den must be float32.")
    if dense_output.dim() != 4 or glob_code_output.dim() != 4 or glob_den.dim() != 3 or value_proj.dim() != 4:
        raise ValueError("expected dense_output [B,QH,S,D], glob_code_output [B,QH,S,Cv], glob_den [B,QH,S], value_proj [B,KVH,Cv,D].")
    B, QH, S, D = dense_output.shape
    Cv = glob_code_output.shape[-1]
    KVH = value_proj.shape[1]
    if tuple(glob_code_output.shape[:3]) != (B, QH, S):
        raise ValueError("glob_code_output leading dims must match dense_output.")
    if tuple(glob_den.shape) != (B, QH, S):
        raise ValueError("glob_den shape must be [B,QH,S].")
    if value_proj.shape[0] != B or value_proj.shape[2] != Cv or value_proj.shape[3] != D:
        raise ValueError("value_proj shape must be [B,KVH,Cv,D] matching dense/global outputs.")
    if KVH <= 0 or QH % KVH != 0:
        raise ValueError(f"QH must be divisible by KVH for GQA output projection; got QH={QH}, KVH={KVH}.")
    if Cv > 512:
        raise ValueError(f"fused output projection supports Cv <= 512; got {Cv}.")
    if D > 256:
        raise ValueError(f"fused output projection supports head_dim <= 256; got {D}.")
    return torch_mod


def picokv_fused_output_projection_cuda_out(dense_output, glob_code_output, glob_den, value_proj, attn_output, verbose: bool = False):
    """Write fused ``glob_code @ value_proj`` plus normalization into caller-owned output."""
    _check_fused_output_inputs(dense_output, glob_code_output, glob_den, value_proj)
    if tuple(attn_output.shape) != tuple(dense_output.shape) or attn_output.dtype != dense_output.dtype or not attn_output.is_cuda:
        raise ValueError("attn_output must be a CUDA tensor matching dense_output shape and dtype.")
    if not attn_output.is_contiguous():
        raise ValueError("attn_output must be contiguous.")
    load_picokv_extension(verbose).picokv_fused_output_projection_fn(
        dense_output,
        glob_code_output,
        glob_den,
        value_proj,
        attn_output,
    )
    return attn_output


def picokv_fused_output_projection_cuda(dense_output, glob_code_output, glob_den, value_proj, verbose: bool = False):
    """Fuse ``glob_code @ value_proj``, dense numerator add, and normalization."""
    torch_mod = _check_fused_output_inputs(dense_output, glob_code_output, glob_den, value_proj)
    attn_output = torch_mod.empty_like(dense_output)
    return picokv_fused_output_projection_cuda_out(
        dense_output,
        glob_code_output,
        glob_den,
        value_proj,
        attn_output,
        verbose=verbose,
    )


def _check_packed_value_inputs(
    packed_projected_query, packed_key, packed_value, query_scale, dense_lse, value_bits: int, key_transposed: bool = False
):
    torch_mod = _require_cuda()
    tensors = {
        "packed_projected_query": packed_projected_query,
        "packed_key": packed_key,
        "packed_value": packed_value,
        "query_scale": query_scale,
        "dense_lse": dense_lse,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if packed_projected_query.dtype != torch_mod.long or packed_key.dtype != torch_mod.long or packed_value.dtype != torch_mod.long:
        raise TypeError("packed_projected_query, packed_key, and packed_value must have dtype torch.long.")
    for name, tensor in [("query_scale", query_scale), ("dense_lse", dense_lse)]:
        if tensor.dtype != torch_mod.float32:
            raise TypeError(f"{name} must have dtype torch.float32; got {tensor.dtype}.")
    if value_bits < 2 or value_bits > 8:
        raise ValueError(f"value_bits must be in [2, 8]; got {value_bits}.")

    B, QH, S, KBlocks, planes = packed_projected_query.shape
    if planes != 8:
        raise ValueError(f"packed_projected_query last dimension must be 8; got {planes}.")
    if KBlocks > 8:
        raise ValueError(f"KBlocks must be <= 8 for the current shared query cache; got {KBlocks}.")
    if key_transposed:
        if packed_key.shape[0] != B or packed_key.shape[2] != KBlocks:
            raise ValueError("packed_key must have shape [B,KVH,KBlocks,L] when key_transposed=True.")
        L = packed_key.shape[3]
    else:
        if packed_key.shape[0] != B or packed_key.shape[-1] != KBlocks:
            raise ValueError("packed_key must have shape [B,KVH,L,KBlocks] matching packed_projected_query.")
        L = packed_key.shape[2]
    KVH = packed_key.shape[1]
    if KVH <= 0 or QH % KVH != 0:
        raise ValueError(f"QH must be divisible by KVH for GQA mapping; got QH={QH}, KVH={KVH}.")
    if packed_value.shape[0] != B or packed_value.shape[1] != KVH or packed_value.shape[2] != L or packed_value.shape[-1] != value_bits:
        raise ValueError("packed_value must have shape [B,KVH,L,VBlocks,value_bits] matching the key/value cache.")
    Cv = packed_value.shape[-2] * 64
    if Cv > 256:
        raise ValueError(f"Cv must be <= 256 for one-thread-per-code-dim accumulation; got {Cv}.")
    if dense_lse.shape != (B, QH, S):
        raise ValueError(f"dense_lse must have shape {(B, QH, S)}; got {tuple(dense_lse.shape)}.")
    if query_scale.numel() != 1 and tuple(query_scale.shape) != (B, QH, S):
        raise ValueError(f"query_scale must be scalar or shape {(B, QH, S)}; got {tuple(query_scale.shape)}.")
    return torch_mod, B, QH, S, L, Cv



def _check_v1_exp8_inputs(
    packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed: bool = False
):
    torch_mod = _require_cuda()
    tensors = {
        "packed_projected_query": packed_projected_query,
        "packed_key": packed_key,
        "packed_value_sign": packed_value_sign,
        "query_scale": query_scale,
        "dense_lse": dense_lse,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if packed_projected_query.dtype != torch_mod.long or packed_key.dtype != torch_mod.long or packed_value_sign.dtype != torch_mod.long:
        raise TypeError("packed_projected_query, packed_key, and packed_value_sign must have dtype torch.long.")
    for name, tensor in [("query_scale", query_scale), ("dense_lse", dense_lse)]:
        if tensor.dtype != torch_mod.float32:
            raise TypeError(f"{name} must have dtype torch.float32; got {tensor.dtype}.")

    B, QH, S, KBlocks, planes = packed_projected_query.shape
    if planes != 8:
        raise ValueError(f"packed_projected_query last dimension must be 8; got {planes}.")
    if key_transposed:
        if packed_key.shape[0] != B or packed_key.shape[2] != KBlocks:
            raise ValueError("packed_key must have shape [B,KVH,KBlocks,L] when key_transposed=True.")
        L = packed_key.shape[3]
    else:
        if packed_key.shape[0] != B or packed_key.shape[-1] != KBlocks:
            raise ValueError("packed_key must have shape [B,KVH,L,KBlocks] matching packed_projected_query.")
        L = packed_key.shape[2]
    KVH = packed_key.shape[1]
    if QH % KVH != 0 or QH // KVH not in (1, 4, 8):
        raise ValueError(f"v1_exp4 kernel requires QH == KVH, 4 * KVH, or 8 * KVH; got QH={QH}, KVH={KVH}.")
    if KBlocks not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"v1_exp4 kernel currently requires Ck in {{128,192,256,320,384,448,512}}; got Ck={KBlocks * 64}.")
    expected_lgroups = (L + 63) // 64
    if packed_value_sign.shape[0] != B or packed_value_sign.shape[1] != KVH or packed_value_sign.shape[2] != expected_lgroups:
        raise ValueError("packed_value_sign must have shape [B,KVH,ceil(L/64),Cv] matching the key cache.")
    Cv = packed_value_sign.shape[3]
    if Cv not in (128, 192, 256, 320, 384, 448, 512):
        raise ValueError(f"v1_exp4 kernel currently requires Cv in {{128,192,256,320,384,448,512}}; got {Cv}.")
    if dense_lse.shape != (B, QH, S):
        raise ValueError(f"dense_lse must have shape {(B, QH, S)}; got {tuple(dense_lse.shape)}.")
    if query_scale.numel() != 1 and tuple(query_scale.shape) != (B, QH, S):
        raise ValueError(f"query_scale must be scalar or shape {(B, QH, S)}; got {tuple(query_scale.shape)}.")
    return torch_mod, B, QH, S, L, Cv


def _validate_exp_bits(exp_bits: int) -> int:
    exp_bits = int(exp_bits)
    if exp_bits not in (1, 2, 4, 8):
        raise ValueError(f"exp_bits must be one of 1, 2, 4, or 8; got {exp_bits}.")
    return exp_bits


def picokv_global_code_output_packed_value_tiled_gqa4_cuda(
    packed_projected_query,
    packed_key,
    packed_value,
    query_scale,
    dense_lse,
    value_bits: int,
    attention_scale: float,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """GQA-specialized tiled PiCoKV path for ``QH / KVH == 4``."""
    torch_mod, B, QH, S, L, Cv = _check_packed_value_inputs(
        packed_projected_query, packed_key, packed_value, query_scale, dense_lse, value_bits, key_transposed=key_transposed
    )
    KVH = packed_key.shape[1]
    if QH != 4 * KVH:
        raise ValueError(f"gqa4 kernel requires QH == 4 * KVH; got QH={QH}, KVH={KVH}.")
    if tile_size != 256:
        raise ValueError("gqa4 kernel currently supports tile_size=256 only.")
    if Cv > 128:
        raise ValueError(f"gqa4 kernel currently expects Cv <= 128; got {Cv}.")

    num_tiles = (L + tile_size - 1) // tile_size
    partial_code_output = torch_mod.empty(
        B, QH, S, num_tiles, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16
    )
    partial_den = torch_mod.empty(B, QH, S, num_tiles, device=packed_projected_query.device, dtype=torch_mod.float32)
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16)
    glob_den = torch_mod.empty(B, QH, S, device=packed_projected_query.device, dtype=torch_mod.float32)

    load_picokv_extension(verbose).picokv_global_code_output_packed_value_tiled_gqa4_fn(
        packed_projected_query,
        packed_key,
        packed_value,
        query_scale,
        dense_lse,
        partial_code_output,
        partial_den,
        glob_code_output,
        glob_den,
        int(value_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
    )
    return glob_code_output, glob_den




def picokv_global_code_output_packed_value_tiled_gqa4_tensorcore_cuda(
    packed_projected_query,
    packed_key,
    packed_value,
    query_scale,
    dense_lse,
    value_bits: int,
    attention_scale: float,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """Compatibility wrapper for the former Tensor Core experiment; currently delegates to the scalar GQA4 path."""
    torch_mod, B, QH, S, L, Cv = _check_packed_value_inputs(
        packed_projected_query, packed_key, packed_value, query_scale, dense_lse, value_bits, key_transposed=key_transposed
    )
    KVH = packed_key.shape[1]
    if QH != 4 * KVH:
        raise ValueError(f"tensorcore gqa4 kernel requires QH == 4 * KVH; got QH={QH}, KVH={KVH}.")
    if packed_projected_query.shape[3] != 2:
        raise ValueError(f"tensorcore gqa4 kernel currently requires Ck == 128; got Ck={packed_projected_query.shape[3] * 64}.")
    if value_bits != 2 or tile_size != 256 or Cv != 128:
        raise ValueError("tensorcore gqa4 kernel currently requires value_bits=2, tile_size=256, Cv=128.")

    num_tiles = (L + tile_size - 1) // tile_size
    partial_code_output = torch_mod.empty(
        B, QH, S, num_tiles, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16
    )
    partial_den = torch_mod.empty(B, QH, S, num_tiles, device=packed_projected_query.device, dtype=torch_mod.float32)
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16)
    glob_den = torch_mod.empty(B, QH, S, device=packed_projected_query.device, dtype=torch_mod.float32)

    load_picokv_extension(verbose).picokv_global_code_output_packed_value_tiled_gqa4_tensorcore_fn(
        packed_projected_query,
        packed_key,
        packed_value,
        query_scale,
        dense_lse,
        partial_code_output,
        partial_den,
        glob_code_output,
        glob_den,
        int(value_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
    )
    return glob_code_output, glob_den


def picokv_global_code_output_v1_exp8_tiled_gqa4_cuda(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """Experimental GQA1/GQA4/GQA8 path for 1-bit values and runtime-configurable quantized exp weights."""
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    exp_bits = _validate_exp_bits(exp_bits)
    if tile_size != 256:
        raise ValueError("v1 runtime-exp gqa4 kernel currently supports tile_size=256 only.")

    num_tiles = (L + tile_size - 1) // tile_size
    partial_code_output = torch_mod.empty(
        B, QH, S, num_tiles, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16
    )
    partial_den = torch_mod.empty(B, QH, S, num_tiles, device=packed_projected_query.device, dtype=torch_mod.float32)
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16)
    glob_den = torch_mod.empty(B, QH, S, device=packed_projected_query.device, dtype=torch_mod.float32)

    load_picokv_extension(verbose).picokv_global_code_output_v1_exp8_tiled_gqa4_fn(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        partial_code_output,
        partial_den,
        glob_code_output,
        glob_den,
        int(exp_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
    )
    return glob_code_output, glob_den




def picokv_global_output_v1_exp8_tiled_gqa4_fused_reduce_cuda(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    dense_output,
    value_proj,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """V1 GQA1/GQA4/GQA8 partial attention plus fused tile-reduce, value projection, and normalization."""
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    exp_bits = _validate_exp_bits(exp_bits)
    if tile_size != 256:
        raise ValueError("v1 fused-reduce kernel currently supports tile_size=256 only.")
    if not dense_output.is_cuda or not value_proj.is_cuda:
        raise ValueError("dense_output and value_proj must be CUDA tensors.")
    if not dense_output.is_contiguous():
        dense_output = dense_output.contiguous()
    if not value_proj.is_contiguous():
        value_proj = value_proj.contiguous()
    if dense_output.dtype != value_proj.dtype or dense_output.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError("dense_output and value_proj must both be bfloat16 or both be float32.")
    D = dense_output.shape[-1]
    KVH = packed_key.shape[1]
    if tuple(dense_output.shape[:3]) != (B, QH, S):
        raise ValueError("dense_output shape must be [B,QH,S,D].")
    if tuple(value_proj.shape) != (B, KVH, Cv, D):
        raise ValueError(f"value_proj shape must be {(B, KVH, Cv, D)}; got {tuple(value_proj.shape)}.")

    num_tiles = (L + tile_size - 1) // tile_size
    partial_code_output = torch_mod.empty(
        B, QH, S, num_tiles, Cv, device=packed_projected_query.device, dtype=torch_mod.bfloat16
    )
    partial_den = torch_mod.empty(B, QH, S, num_tiles, device=packed_projected_query.device, dtype=torch_mod.float32)
    attn_output = torch_mod.empty_like(dense_output)
    load_picokv_extension(verbose).picokv_global_output_v1_exp8_tiled_gqa4_fused_reduce_fn(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        partial_code_output,
        partial_den,
        dense_output,
        value_proj,
        attn_output,
        int(exp_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
    )
    return attn_output


def picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda_out(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    glob_code_output,
    glob_den,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """Write direct-output v1 global code attention into caller-owned outputs."""
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    exp_bits = _validate_exp_bits(exp_bits)
    if tile_size != 256:
        raise ValueError("v1 runtime-exp atomic kernel currently supports tile_size=256 only.")
    if tuple(glob_code_output.shape) != (B, QH, S, Cv) or glob_code_output.dtype != torch_mod.float32 or not glob_code_output.is_cuda:
        raise ValueError(f"glob_code_output must be CUDA fp32 with shape {(B, QH, S, Cv)}.")
    if tuple(glob_den.shape) != (B, QH, S) or glob_den.dtype != torch_mod.float32 or not glob_den.is_cuda:
        raise ValueError(f"glob_den must be CUDA fp32 with shape {(B, QH, S)}.")
    if not glob_code_output.is_contiguous() or not glob_den.is_contiguous():
        raise ValueError("global output tensors must be contiguous.")

    torch_mod._foreach_zero_([glob_code_output, glob_den])
    load_picokv_extension(verbose).picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_fn(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        glob_code_output,
        glob_den,
        int(exp_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
    )
    return glob_code_output, glob_den


def picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_cuda_out(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    glob_code_output,
    glob_den,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """Prototype GQA8 path with BMMA score generation and the existing value/atomic path."""
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    exp_bits = _validate_exp_bits(exp_bits)
    if QH // packed_key.shape[1] != 8:
        raise ValueError("BMMA score prototype requires q_heads == 8 * kv_heads.")
    if packed_projected_query.shape[3] != 4 or Cv != 256:
        raise ValueError("BMMA score prototype requires Ck=256 and Cv=256.")
    if exp_bits != 4:
        raise ValueError("BMMA score prototype currently supports exp_bits=4 only.")
    if tile_size != 256:
        raise ValueError("BMMA score prototype currently supports tile_size=256 only.")
    if tuple(glob_code_output.shape) != (B, QH, S, Cv) or glob_code_output.dtype != torch_mod.float32 or not glob_code_output.is_cuda:
        raise ValueError(f"glob_code_output must be CUDA fp32 with shape {(B, QH, S, Cv)}.")
    if tuple(glob_den.shape) != (B, QH, S) or glob_den.dtype != torch_mod.float32 or not glob_den.is_cuda:
        raise ValueError(f"glob_den must be CUDA fp32 with shape {(B, QH, S)}.")
    if not glob_code_output.is_contiguous() or not glob_den.is_contiguous():
        raise ValueError("global output tensors must be contiguous.")

    torch_mod._foreach_zero_([glob_code_output, glob_den])
    load_picokv_extension(verbose).picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_fn(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        glob_code_output,
        glob_den,
        int(exp_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
    )
    return glob_code_output, glob_den


def picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_cuda(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """Allocate outputs and run the BMMA-score GQA8 prototype."""
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=packed_projected_query.device, dtype=torch_mod.float32)
    glob_den = torch_mod.empty(B, QH, S, device=packed_projected_query.device, dtype=torch_mod.float32)
    return picokv_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_cuda_out(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        glob_code_output,
        glob_den,
        attention_scale,
        exp_bits=exp_bits,
        tile_size=tile_size,
        key_transposed=key_transposed,
        verbose=verbose,
    )


def picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    verbose: bool = False,
):
    """Experimental direct-output v1 GQA1/GQA4/GQA8 path using fp32 atomics and configurable exp bits."""
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=packed_projected_query.device, dtype=torch_mod.float32)
    glob_den = torch_mod.empty(B, QH, S, device=packed_projected_query.device, dtype=torch_mod.float32)
    return picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda_out(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        glob_code_output,
        glob_den,
        attention_scale,
        exp_bits=exp_bits,
        tile_size=tile_size,
        key_transposed=key_transposed,
        verbose=verbose,
    )




def _validate_prune_mode(prune_mode: int) -> int:
    prune_mode = int(prune_mode)
    if prune_mode not in (1, 2, 3):
        raise ValueError(f"prune_mode must be 1(block), 2(prefix), or 3(combined); got {prune_mode}.")
    return prune_mode


def _check_pruned_gqa4_inputs(packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed: bool):
    torch_mod, B, QH, S, L, Cv = _check_v1_exp8_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed=key_transposed
    )
    KVH = packed_key.shape[1]
    if QH != 4 * KVH:
        raise ValueError(f"pruned kernel currently requires QH == 4 * KVH; got QH={QH}, KVH={KVH}.")
    if packed_projected_query.shape[3] not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError(
            f"pruned kernel currently requires Ck in {{128,192,256,320,384,448,512}}; "
            f"got Ck={packed_projected_query.shape[3] * 64}."
        )
    if Cv not in (128, 192, 256, 320, 384, 448, 512):
        raise ValueError(f"pruned kernel currently requires Cv in {{128,192,256,320,384,448,512}}; got {Cv}.")
    return torch_mod, B, QH, S, L, Cv


def picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_cuda_out(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    glob_code_output,
    glob_den,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    prune_mode: int = 3,
    prune_log_eps: float = -12.0,
    verbose: bool = False,
):
    """Write pruned GQA4 global code attention into caller-owned fp32 outputs.

    ``prune_mode`` selects block-only (1), cascaded prefix-dim token pruning
    (2), or both (3).  The implementation is intentionally opt-in so the
    baseline atomic path remains available for direct speed/accuracy comparison.
    """
    torch_mod, B, QH, S, _, Cv = _check_pruned_gqa4_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed
    )
    exp_bits = _validate_exp_bits(exp_bits)
    prune_mode = _validate_prune_mode(prune_mode)
    if tile_size != 256:
        raise ValueError("pruned atomic kernel currently supports tile_size=256 only.")
    if tuple(glob_code_output.shape) != (B, QH, S, Cv) or glob_code_output.dtype != torch_mod.float32 or not glob_code_output.is_cuda:
        raise ValueError(f"glob_code_output must be CUDA fp32 with shape {(B, QH, S, Cv)}.")
    if tuple(glob_den.shape) != (B, QH, S) or glob_den.dtype != torch_mod.float32 or not glob_den.is_cuda:
        raise ValueError(f"glob_den must be CUDA fp32 with shape {(B, QH, S)}.")
    if not glob_code_output.is_contiguous() or not glob_den.is_contiguous():
        raise ValueError("global output tensors must be contiguous.")

    torch_mod._foreach_zero_([glob_code_output, glob_den])
    load_picokv_extension(verbose).picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_fn(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        glob_code_output,
        glob_den,
        int(exp_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
        int(prune_mode),
        float(prune_log_eps),
    )
    return glob_code_output, glob_den


def picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_cuda(
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    prune_mode: int = 3,
    prune_log_eps: float = -12.0,
    verbose: bool = False,
):
    """Allocate outputs and run the opt-in pruned GQA4 global attention path."""
    torch_mod, B, QH, S, _, Cv = _check_pruned_gqa4_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed
    )
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=packed_projected_query.device, dtype=torch_mod.float32)
    glob_den = torch_mod.empty(B, QH, S, device=packed_projected_query.device, dtype=torch_mod.float32)
    return picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_pruned_cuda_out(
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        glob_code_output,
        glob_den,
        attention_scale,
        exp_bits=exp_bits,
        tile_size=tile_size,
        key_transposed=key_transposed,
        prune_mode=prune_mode,
        prune_log_eps=prune_log_eps,
        verbose=verbose,
    )


def picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_cuda_out(
    projected_query,
    packed_projected_query,
    packed_key,
    packed_value_sign,
    query_scale,
    dense_lse,
    dense_output,
    glob_code_output,
    glob_den,
    value_proj,
    attn_output,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    prune_mode: int = 3,
    prune_log_eps: float = -12.0,
    verbose: bool = False,
):
    """Run pruned decode tail from projected query to normalized attention output."""
    torch_mod, B, QH, S, _, Cv = _check_pruned_gqa4_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed
    )
    exp_bits = _validate_exp_bits(exp_bits)
    prune_mode = _validate_prune_mode(prune_mode)
    if tile_size != 256:
        raise ValueError("pruned decode-tail kernel currently supports tile_size=256 only.")
    if not projected_query.is_cuda or not projected_query.is_contiguous():
        raise ValueError("projected_query must be a contiguous CUDA tensor.")
    if projected_query.dtype not in (torch_mod.bfloat16, torch_mod.float32):
        raise TypeError(f"projected_query must be bf16 or fp32; got {projected_query.dtype}.")
    if tuple(projected_query.shape) != (B, QH, S, packed_projected_query.shape[3] * 64):
        raise ValueError("projected_query must have shape [B,QH,S,Ck] matching packed_projected_query.")
    if tuple(dense_output.shape[:3]) != (B, QH, S) or dense_output.dtype != projected_query.dtype or not dense_output.is_cuda:
        raise ValueError("dense_output must be CUDA and match projected_query leading shape/dtype.")
    D = dense_output.shape[-1]
    KVH = packed_key.shape[1]
    if tuple(value_proj.shape) != (B, KVH, Cv, D) or value_proj.dtype != dense_output.dtype or not value_proj.is_cuda:
        raise ValueError(f"value_proj must be CUDA with shape {(B, KVH, Cv, D)} and dense_output dtype.")
    if tuple(glob_code_output.shape) != (B, QH, S, Cv) or glob_code_output.dtype != torch_mod.float32 or not glob_code_output.is_cuda:
        raise ValueError(f"glob_code_output must be CUDA fp32 with shape {(B, QH, S, Cv)}.")
    if tuple(glob_den.shape) != (B, QH, S) or glob_den.dtype != torch_mod.float32 or not glob_den.is_cuda:
        raise ValueError(f"glob_den must be CUDA fp32 with shape {(B, QH, S)}.")
    if tuple(attn_output.shape) != tuple(dense_output.shape) or attn_output.dtype != dense_output.dtype or not attn_output.is_cuda:
        raise ValueError("attn_output must be CUDA and match dense_output shape/dtype.")
    for name, tensor in [
        ("packed_projected_query", packed_projected_query),
        ("query_scale", query_scale),
        ("dense_lse", dense_lse),
        ("dense_output", dense_output),
        ("glob_code_output", glob_code_output),
        ("glob_den", glob_den),
        ("value_proj", value_proj),
        ("attn_output", attn_output),
    ]:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")

    load_picokv_extension(verbose).picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
        projected_query,
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        dense_output,
        glob_code_output,
        glob_den,
        value_proj,
        attn_output,
        int(exp_bits),
        int(tile_size),
        int(bool(key_transposed)),
        float(attention_scale),
        int(prune_mode),
        float(prune_log_eps),
    )
    return attn_output


def picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_cuda(
    projected_query,
    packed_key,
    packed_value_sign,
    dense_lse,
    dense_output,
    value_proj,
    attention_scale: float,
    exp_bits: int = 4,
    tile_size: int = 256,
    key_transposed: bool = False,
    prune_mode: int = 3,
    prune_log_eps: float = -12.0,
    verbose: bool = False,
):
    """Allocate workspaces and run the opt-in pruned GQA4 decode-tail path."""
    torch_mod = _require_cuda()
    if not projected_query.is_cuda or not projected_query.is_contiguous():
        raise ValueError("projected_query must be a contiguous CUDA tensor.")
    B, QH, S, C = projected_query.shape
    if C not in (128, 192, 256, 320, 512):
        raise ValueError(f"pruned decode-tail path currently requires projected-query Ck in {{128,192,256,320,384,448,512}}; got {C}.")
    packed_projected_query = torch_mod.empty(B, QH, S, C // 64, 8, device=projected_query.device, dtype=torch_mod.long)
    query_scale = torch_mod.empty(B, QH, S, device=projected_query.device, dtype=torch_mod.float32)
    _, _, _, _, _, Cv = _check_pruned_gqa4_inputs(
        packed_projected_query, packed_key, packed_value_sign, query_scale, dense_lse, key_transposed
    )
    glob_code_output = torch_mod.empty(B, QH, S, Cv, device=projected_query.device, dtype=torch_mod.float32)
    glob_den = torch_mod.empty(B, QH, S, device=projected_query.device, dtype=torch_mod.float32)
    attn_output = torch_mod.empty_like(dense_output)
    return picokv_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_cuda_out(
        projected_query,
        packed_projected_query,
        packed_key,
        packed_value_sign,
        query_scale,
        dense_lse,
        dense_output,
        glob_code_output,
        glob_den,
        value_proj,
        attn_output,
        attention_scale,
        exp_bits=exp_bits,
        tile_size=tile_size,
        key_transposed=key_transposed,
        prune_mode=prune_mode,
        prune_log_eps=prune_log_eps,
        verbose=verbose,
    )




def smoke_test(verbose: bool = False) -> None:
    torch_mod = _require_cuda()
    torch_mod.manual_seed(0)
    B, QH, KVH, S, L, C = 1, 4, 1, 2, 257, 128
    attention_scale = 0.08838834764831845

    q = torch_mod.randint(-127, 128, (B, QH, S, C), device="cuda", dtype=torch_mod.int8).contiguous()
    key_bits = torch_mod.randint(0, 2, (B, KVH, L, C), device="cuda", dtype=torch_mod.int8)
    k = (key_bits * 2 - 1).contiguous()
    v_raw = torch_mod.randint(0, 4, (B, KVH, L, C), device="cuda", dtype=torch_mod.int8).contiguous()
    v = (v_raw.float() - 1.5).contiguous()
    v1_bits = torch_mod.randint(0, 2, (B, KVH, L, C), device="cuda", dtype=torch_mod.int8)
    v1 = (v1_bits * 2 - 1).contiguous()
    q_scale = (torch_mod.rand(B, QH, S, device="cuda") * 0.01).contiguous()
    dense_lse = torch_mod.randn(B, QH, S, device="cuda", dtype=torch_mod.float32).contiguous()
    dense_q = torch_mod.randn(B, QH, S, C, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    dense_k = torch_mod.randn(B, KVH, 97, C, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    dense_v = torch_mod.randn(B, KVH, 97, C, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    dense_cuda_out, dense_cuda_lse = picokv_dense_attention_numerator_cuda(
        dense_q, dense_k, dense_v, attention_scale, verbose=verbose
    )
    dense_k_exp = dense_k.repeat_interleave(QH // KVH, dim=1)
    dense_v_exp = dense_v.repeat_interleave(QH // KVH, dim=1)
    dense_scores = torch_mod.matmul(dense_q, dense_k_exp.transpose(-1, -2)) * attention_scale
    dense_ref_lse = torch_mod.logsumexp(dense_scores.float(), dim=-1)
    dense_weights = torch_mod.exp(dense_scores.float() - dense_ref_lse[..., None]).to(dense_q.dtype)
    dense_ref_out = torch_mod.matmul(dense_weights, dense_v_exp)

    fused_code = torch_mod.randn(B, QH, S, 256, device="cuda", dtype=torch_mod.float32).contiguous()
    fused_den = (torch_mod.rand(B, QH, S, device="cuda", dtype=torch_mod.float32) * 4).contiguous()
    fused_proj = (torch_mod.randn(B, KVH, 256, C, device="cuda", dtype=torch_mod.bfloat16) * 0.02).contiguous()
    fused_cuda_out = picokv_fused_output_projection_cuda(dense_cuda_out, fused_code, fused_den, fused_proj, verbose=verbose)
    fused_proj_exp = fused_proj.repeat_interleave(QH // KVH, dim=1)
    fused_ref_out = (
        dense_cuda_out + torch_mod.matmul(fused_code.to(fused_proj.dtype), fused_proj_exp)
    ) / (1 + fused_den[..., None].to(dense_cuda_out.dtype))

    packed_q = pack_projected_query_torch(q).contiguous()
    packed_q_cuda = pack_projected_query_cuda(q, verbose=verbose).contiguous()
    if not torch_mod.equal(packed_q_cuda, packed_q):
        raise AssertionError("CUDA projected-query pack mismatch")
    packed_k = pack_picokv_key_torch(k).contiguous()
    packed_v = pack_picokv_value_torch(v, 2).contiguous()
    packed_v1 = pack_picokv_value_1bit_transposed_torch(v1).contiguous()
    cuda_out, cuda_den = picokv_global_code_output_packed_value_tiled_gqa4_cuda(
        packed_q, packed_k, packed_v, q_scale, dense_lse, 2, attention_scale, tile_size=256, verbose=verbose
    )
    tc_out, tc_den = picokv_global_code_output_packed_value_tiled_gqa4_tensorcore_cuda(
        packed_q, packed_k, packed_v, q_scale, dense_lse, 2, attention_scale, tile_size=256, verbose=verbose
    )
    v1_out, v1_den = picokv_global_code_output_v1_exp8_tiled_gqa4_cuda(
        packed_q, packed_k, packed_v1, q_scale, dense_lse, attention_scale, exp_bits=4, tile_size=256, verbose=verbose
    )
    v1_atomic_out, v1_atomic_den = picokv_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda(
        packed_q, packed_k, packed_v1, q_scale, dense_lse, attention_scale, exp_bits=4, tile_size=256, verbose=verbose
    )
    torch_mod.cuda.synchronize()

    expanded_k = k.repeat_interleave(QH // KVH, dim=1)
    expanded_v = v.repeat_interleave(QH // KVH, dim=1)
    scores = torch_mod.einsum("bhsc,bhlc->bhsl", q.float(), expanded_k.float()) * q_scale[..., None] * attention_scale
    weights = torch_mod.exp(scores - dense_lse[..., None])
    ref_den = weights.sum(dim=-1)

    ref_out = torch_mod.zeros_like(cuda_out)
    for tile_start in range(0, L, 256):
        tile_end = min(tile_start + 256, L)
        tile_terms = weights[..., tile_start:tile_end, None] * expanded_v[:, :, tile_start:tile_end, :].float().unsqueeze(2)
        tile_out = tile_terms.sum(dim=-2).to(torch_mod.bfloat16)
        ref_out = (ref_out + tile_out).to(torch_mod.bfloat16)

    expanded_v1 = v1.float().repeat_interleave(QH // KVH, dim=1)
    ref_v1_out = torch_mod.zeros_like(v1_out)
    ref_v1_den = torch_mod.zeros_like(v1_den)
    exact_v1_out = torch_mod.zeros_like(v1_out, dtype=torch_mod.float32)
    for tile_start in range(0, L, 256):
        tile_v1_out = torch_mod.zeros_like(v1_out, dtype=torch_mod.float32)
        for group_start in range(tile_start, min(tile_start + 256, L), 64):
            group_end = min(group_start + 64, L)
            group_weights = weights[..., group_start:group_end]
            max_weight = group_weights.max(dim=-1).values
            quant = torch_mod.round(group_weights / max_weight.clamp_min(1.0e-30)[..., None] * 15.0).clamp_(0, 15)
            group_scale = max_weight / 15.0
            ref_v1_den += group_scale * quant.sum(dim=-1)
            group_v = expanded_v1[:, :, group_start:group_end, :].unsqueeze(2)
            tile_v1_out += group_scale[..., None] * (quant[..., None] * group_v).sum(dim=-2)
            exact_v1_out += (group_weights[..., None] * group_v).sum(dim=-2)
        ref_v1_out = (ref_v1_out + tile_v1_out.to(torch_mod.bfloat16)).to(torch_mod.bfloat16)

    code_abs = (cuda_out.float() - ref_out.float()).abs()
    code_diff = code_abs.max().item()
    code_mean_diff = code_abs.mean().item()
    den_diff = (cuda_den - ref_den).abs().max().item()
    tc_code_abs = (tc_out.float() - ref_out.float()).abs()
    tc_code_diff = tc_code_abs.max().item()
    tc_code_mean_diff = tc_code_abs.mean().item()
    tc_den_diff = (tc_den - ref_den).abs().max().item()
    tc_vs_scalar_diff = (tc_out.float() - cuda_out.float()).abs().max().item()
    v1_code_abs = (v1_out.float() - ref_v1_out.float()).abs()
    v1_code_diff = v1_code_abs.max().item()
    v1_code_mean_diff = v1_code_abs.mean().item()
    v1_den_diff = (v1_den - ref_v1_den).abs().max().item()
    v1_atomic_code_abs = (v1_atomic_out.float() - ref_v1_out.float()).abs()
    v1_atomic_code_diff = v1_atomic_code_abs.max().item()
    v1_atomic_code_mean_diff = v1_atomic_code_abs.mean().item()
    v1_atomic_den_diff = (v1_atomic_den - ref_v1_den).abs().max().item()
    v1_quant_code_diff = (ref_v1_out.float() - exact_v1_out).abs().max().item()
    v1_quant_den_diff = (ref_v1_den - ref_den).abs().max().item()
    dense_out_diff = (dense_cuda_out.float() - dense_ref_out.float()).abs().max().item()
    dense_out_mean_diff = (dense_cuda_out.float() - dense_ref_out.float()).abs().mean().item()
    dense_lse_diff = (dense_cuda_lse - dense_ref_lse).abs().max().item()
    fused_out_diff = (fused_cuda_out.float() - fused_ref_out.float()).abs().max().item()
    fused_out_mean_diff = (fused_cuda_out.float() - fused_ref_out.float()).abs().mean().item()
    # The optimized 2-bit value path uses an equivalent den-offset formula,
    # which intentionally changes fp32 accumulation order before bf16 storage.
    if not torch_mod.allclose(dense_cuda_lse, dense_ref_lse, atol=1e-3, rtol=1e-5):
        raise AssertionError(f"dense attention lse mismatch: max diff {dense_lse_diff}")
    if not torch_mod.allclose(dense_cuda_out.float(), dense_ref_out.float(), atol=2e-2, rtol=2e-2):
        raise AssertionError(f"dense attention output mismatch: max diff {dense_out_diff}, mean diff {dense_out_mean_diff}")
    if not torch_mod.allclose(fused_cuda_out.float(), fused_ref_out.float(), atol=3e-2, rtol=3e-2):
        raise AssertionError(f"fused output projection mismatch: max diff {fused_out_diff}, mean diff {fused_out_mean_diff}")
    if not torch_mod.allclose(cuda_out.float(), ref_out.float(), atol=5e-1, rtol=5e-3):
        raise AssertionError(f"gqa4 output mismatch: max diff {code_diff}, mean diff {code_mean_diff}")
    if not torch_mod.allclose(cuda_den, ref_den, atol=1e-3, rtol=1e-5):
        raise AssertionError(f"gqa4 denominator mismatch: max diff {den_diff}")
    if not torch_mod.allclose(tc_den, ref_den, atol=1e-3, rtol=1e-5):
        raise AssertionError(f"tensorcore gqa4 denominator mismatch: max diff {tc_den_diff}")
    if not torch_mod.allclose(tc_out.float(), ref_out.float(), atol=1.0, rtol=2e-2):
        raise AssertionError(f"tensorcore gqa4 output mismatch: max diff {tc_code_diff}, mean diff {tc_code_mean_diff}")
    if not torch_mod.allclose(v1_out.float(), ref_v1_out.float(), atol=5e-1, rtol=5e-3):
        raise AssertionError(f"v1_exp4 output mismatch: max diff {v1_code_diff}, mean diff {v1_code_mean_diff}")
    if not torch_mod.allclose(v1_den, ref_v1_den, atol=1e-3, rtol=1e-5):
        raise AssertionError(f"v1_exp4 denominator mismatch: max diff {v1_den_diff}")
    if not torch_mod.allclose(v1_atomic_out.float(), ref_v1_out.float(), atol=5e-1, rtol=5e-3):
        raise AssertionError(f"v1_exp4 atomic output mismatch: max diff {v1_atomic_code_diff}, mean diff {v1_atomic_code_mean_diff}")
    if not torch_mod.allclose(v1_atomic_den, ref_v1_den, atol=1e-3, rtol=1e-5):
        raise AssertionError(f"v1_exp4 atomic denominator mismatch: max diff {v1_atomic_den_diff}")

    print(
        "picokv_cuda smoke_test passed: "
        f"dense_out_diff={dense_out_diff}, dense_out_mean_diff={dense_out_mean_diff}, dense_lse_diff={dense_lse_diff}; "
        f"fused_out_diff={fused_out_diff}, fused_out_mean_diff={fused_out_mean_diff}; "
        f"gqa4 code max diff={code_diff}, mean diff={code_mean_diff}, gqa4 den diff={den_diff}; "
        f"tensorcore_fallback code max diff={tc_code_diff}, mean diff={tc_code_mean_diff}, "
        f"tensorcore_fallback den diff={tc_den_diff}, fallback_vs_scalar max diff={tc_vs_scalar_diff}; "
        f"v1_exp4 code max diff={v1_code_diff}, mean diff={v1_code_mean_diff}, den diff={v1_den_diff}, "
        f"v1_atomic code max diff={v1_atomic_code_diff}, mean diff={v1_atomic_code_mean_diff}, den diff={v1_atomic_den_diff}, "
        f"quant_vs_exact code max diff={v1_quant_code_diff}, den diff={v1_quant_den_diff}"
    )


if __name__ == "__main__":
    smoke_test()
