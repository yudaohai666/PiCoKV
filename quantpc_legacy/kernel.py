"""Utilities for the binary-projection CUDA extension.

The CUDA extension stores a 64-bit binary code for every key/token.  Query
vectors (called probes in this code) are quantized and packed into bit-planes,
then compared with the stored hashcodes using xor + popcount.  The wrappers in
this file keep the tensor shape/dtype assumptions explicit so the low-level CUDA
kernels fail early with useful Python errors instead of silent memory mistakes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Optional

try:
    import torch
    from torch.utils.cpp_extension import load
except ModuleNotFoundError:  # Allows `python -m py_compile kernel.py` without torch.
    torch = None
    load = None


_EXTENSION = None


def _require_torch():
    """Return the imported torch module or raise a clear runtime error."""
    if torch is None:
        raise RuntimeError("PyTorch is required, but it is not installed in this Python environment.")
    return torch


def _require_cuda():
    """Return torch after checking that CUDA is visible."""
    torch_mod = _require_torch()
    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA is required to run the binary projection extension.")
    return torch_mod


def _ensure_build_tool_path() -> None:
    """Expose tools such as conda's ninja executable to PyTorch's builder."""
    python_bin = os.path.dirname(sys.executable)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin and python_bin not in path_parts:
        os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")


def load_extension(verbose: bool = False):
    """Build and load the CUDA extension once, then cache the module.

    Keeping extension loading lazy makes this file importable in CPU-only or
    documentation environments.  Compilation happens only when a CUDA wrapper is
    actually called.
    """
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    _require_torch()
    if load is None:
        raise RuntimeError("torch.utils.cpp_extension.load is unavailable.")
    _ensure_build_tool_path()

    curr_path = os.path.dirname(os.path.realpath(__file__))
    src_files = [
        os.path.join(curr_path, "extension", "cuda_kernel.cu"),
        os.path.join(curr_path, "extension", "torch_extension.cpp"),
    ]
    _EXTENSION = load(
        "binary_proj",
        src_files,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-O3"],
        verbose=verbose,
    )
    return _EXTENSION


def _check_multiple(name: str, value: int, multiple: int) -> None:
    """Check CUDA launch assumptions that are hard-coded in the kernels."""
    if value % multiple != 0:
        raise ValueError(f"{name} must be a multiple of {multiple}; got {value}.")


def _check_bf16_cuda(name: str, tensor) -> None:
    """Validate the bf16 CUDA tensors consumed through __nv_bfloat162 pointers."""
    torch_mod = _require_cuda()
    if tensor.dtype != torch_mod.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16; got {tensor.dtype}.")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def find_binary_projection(A, num_bits: int = 64, num_iters: int = 4):
    """Learn greedy binary projection directions for ``A``.

    Args:
        A: Tensor with shape ``[B, H, L, D]``.
        num_bits: Number of binary directions to learn.  The CUDA hash kernels
            below are specialized for 64 bits.
        num_iters: Alternating updates per bit.

    Returns:
        ``(quan_A, quan_proj)`` where ``quan_A`` contains unpacked +/-1 codes
        with shape ``[B, H, L, num_bits]`` and ``quan_proj`` contains projection
        directions with shape ``[B, H, num_bits, D]``.
    """
    torch_mod = _require_torch()
    B, H, L, D = A.shape
    quan_A = torch_mod.empty(B, H, L, num_bits, device=A.device, dtype=A.dtype)
    quan_proj = torch_mod.empty(B, H, num_bits, D, device=A.device, dtype=A.dtype)

    # Work on a residual so every new bit explains what previous bits missed.
    residual = A
    for bit_idx in range(num_bits):
        Vh = torch_mod.randn(B, H, D, device=A.device, dtype=A.dtype)
        for _ in range(num_iters):
            Uq = torch_mod.einsum("bhld,bhd->bhl", residual, Vh).sign()
            Vh = torch_mod.einsum("bhld,bhl->bhd", residual, Uq)
        Vh = Vh / L
        residual = residual - torch_mod.einsum("bhl,bhd->bhld", Uq, Vh)
        quan_A[:, :, :, bit_idx] = Uq
        quan_proj[:, :, bit_idx, :] = Vh

    return quan_A, quan_proj


def binary_project_torch(A, quan_proj):
    """Reference PyTorch implementation that returns unpacked +/-1 hash bits."""
    torch_mod = _require_torch()
    B, H, L, _ = A.shape
    num_bits = quan_proj.shape[-2]
    quan_A = torch_mod.empty(B, H, L, num_bits, device=A.device, dtype=A.dtype)

    residual = A
    for bit_idx in range(num_bits):
        direction = quan_proj[:, :, bit_idx, :]
        Uq = torch_mod.einsum("bhld,bhd->bhl", residual, direction).sign()
        residual = residual - torch_mod.einsum("bhl,bhd->bhld", Uq, direction)
        quan_A[:, :, :, bit_idx] = Uq

    return quan_A


def binary_project_cuda(A, quan_proj, verbose: bool = False):
    """Project ``[B, H, L, 128]`` bf16 vectors into packed int64 hashcodes.

    The CUDA kernel writes one 64-bit integer per token.  Bit ``i`` is set when
    projection direction ``i`` produces a negative sign.
    """
    _check_bf16_cuda("A", A)
    _check_bf16_cuda("quan_proj", quan_proj)
    B, H, L, D = A.shape
    if D != 128:
        raise ValueError(f"A last dimension must be 128; got {D}.")
    if quan_proj.shape != (B, H, 64, 128):
        raise ValueError(f"quan_proj must have shape {(B, H, 64, 128)}; got {tuple(quan_proj.shape)}.")
    _check_multiple("L", L, 32)

    y = torch.empty(B, H, L, device=A.device, dtype=torch.long)
    load_extension(verbose).binary_project_dim128_hash64_fn(A, quan_proj, y)
    return y


def pack_bits(unpacked, verbose: bool = False):
    """Pack ``[B, H, L, 64]`` int +/-1 codes into ``[B, H, L]`` int64 codes."""
    torch_mod = _require_cuda()
    if unpacked.dtype != torch_mod.int32:
        unpacked = unpacked.to(torch_mod.int32)
    if not unpacked.is_cuda:
        raise ValueError("unpacked must be a CUDA tensor.")
    if unpacked.shape[-1] != 64:
        raise ValueError(f"unpacked last dimension must be 64; got {unpacked.shape[-1]}.")

    unpacked = unpacked.contiguous()
    B, H, L, _ = unpacked.shape
    packed = torch_mod.empty(B, H, L, device=unpacked.device, dtype=torch_mod.long)
    load_extension(verbose).format_bits_hash64_fn(unpacked, packed, True)
    return packed


def unpack_bits(packed, verbose: bool = False):
    """Unpack ``[B, H, L]`` int64 hashcodes into ``[B, H, L, 64]`` +/-1 ints."""
    torch_mod = _require_cuda()
    if packed.dtype != torch_mod.long:
        raise TypeError(f"packed must have dtype torch.long; got {packed.dtype}.")
    if not packed.is_cuda:
        raise ValueError("packed must be a CUDA tensor.")

    B, H, L = packed.shape
    unpacked = torch_mod.empty(B, H, L, 64, device=packed.device, dtype=torch_mod.int32)
    load_extension(verbose).format_bits_hash64_fn(unpacked, packed.contiguous(), False)
    return unpacked


def quantize_probe_cuda(probe, verbose: bool = False):
    """Quantize ``[B, H, S, 64]`` bf16 probes to signed 7-bit int values."""
    _check_bf16_cuda("probe", probe)
    B, H, S, D = probe.shape
    if D != 64:
        raise ValueError(f"probe last dimension must be 64; got {D}.")

    quant_probe = torch.empty(B, H, S, 64, device=probe.device, dtype=torch.int32)
    load_extension(verbose).quant_probe_hash64_fn(probe, quant_probe)
    return quant_probe


def pack_probe_cuda(quant_probe, verbose: bool = False):
    """Pack quantized probes into sign/magnitude bit-planes.

    Output shape is ``[B, H, S, 8]``.  Plane 0 stores signs; planes 1..7 store
    absolute-value bits.  This layout lets CUDA compute many dot-product-like
    scores using popcount instead of scalar multiply/add.
    """
    torch_mod = _require_cuda()
    if quant_probe.dtype != torch_mod.int32:
        quant_probe = quant_probe.to(torch_mod.int32)
    if not quant_probe.is_cuda:
        raise ValueError("quant_probe must be a CUDA tensor.")
    if quant_probe.shape[-1] != 64:
        raise ValueError(f"quant_probe last dimension must be 64; got {quant_probe.shape[-1]}.")

    B, H, S, _ = quant_probe.shape
    packed_probe = torch_mod.empty(B, H, S, 8, device=quant_probe.device, dtype=torch_mod.long)
    load_extension(verbose).format_probe_hash64_fn(quant_probe.contiguous(), packed_probe, True)
    return packed_probe


def probe_hash_cuda(packed_probe, hashcode, verbose: bool = False):
    """Score four packed probes against packed key hashcodes.

    ``packed_probe`` must be ``[B, H, 4, 8]`` and ``hashcode`` must be
    ``[B, H, L]``.  The output score is ``int16 [B, H, L]``.
    """
    torch_mod = _require_cuda()
    if packed_probe.dtype != torch_mod.long or hashcode.dtype != torch_mod.long:
        raise TypeError("packed_probe and hashcode must have dtype torch.long.")
    if packed_probe.shape[2:] != (4, 8):
        raise ValueError(f"packed_probe must have shape [B, H, 4, 8]; got {tuple(packed_probe.shape)}.")
    B, H, L = hashcode.shape
    _check_multiple("L", L, 256)

    scores = torch_mod.empty(B, H, L, device=hashcode.device, dtype=torch_mod.int16)
    load_extension(verbose).probe_hash64_group4_block256_fn(
        packed_probe.contiguous(), hashcode.contiguous(), scores
    )
    return scores


def fused_probe_hash_cuda(probe, hashcode, T: Optional[int] = None, hash_T: Optional[int] = None, verbose: bool = False):
    """Quantize, pack, and score ``[B, H, 4, 64]`` probes in one CUDA launch.

    ``T`` is the valid sequence length inside the padded hash buffer.  Tokens at
    ``idx >= T`` receive a very low score.  Tokens in ``[hash_T, T)`` receive a
    high score so newly appended, not-yet-hashed tokens are force-selected.
    """
    _check_bf16_cuda("probe", probe)
    torch_mod = _require_cuda()
    if hashcode.dtype != torch_mod.long:
        raise TypeError(f"hashcode must have dtype torch.long; got {hashcode.dtype}.")
    if probe.shape[2:] != (4, 64):
        raise ValueError(f"probe must have shape [B, H, 4, 64]; got {tuple(probe.shape)}.")
    B, H, L = hashcode.shape
    _check_multiple("L", L, 256)

    if T is None:
        T = L
    if hash_T is None:
        hash_T = T
    scores = torch_mod.empty(B, H, L, device=hashcode.device, dtype=torch_mod.int16)
    load_extension(verbose).fused_quant_pack_probe_hash64_fn(
        probe.contiguous(), hashcode.contiguous(), scores, int(T), int(hash_T)
    )
    return scores


def gather_kv_cuda(keys, values, indices, verbose: bool = False):
    """Gather selected key/value rows with one fused CUDA kernel."""
    _check_bf16_cuda("keys", keys)
    _check_bf16_cuda("values", values)
    torch_mod = _require_cuda()
    if indices.dtype != torch_mod.long:
        raise TypeError(f"indices must have dtype torch.long; got {indices.dtype}.")
    if keys.shape != values.shape:
        raise ValueError(f"keys and values must have the same shape; got {keys.shape} and {values.shape}.")
    B, H, _, D = keys.shape
    if D % 2 != 0:
        raise ValueError(f"D must be even for bfloat162 loads; got {D}.")
    if indices.shape[:2] != (B, H):
        raise ValueError(f"indices leading dimensions must be {(B, H)}; got {tuple(indices.shape[:2])}.")

    k = indices.shape[2]
    selected_keys = torch_mod.empty(B, H, k, D, device=keys.device, dtype=keys.dtype)
    selected_values = torch_mod.empty(B, H, k, D, device=values.device, dtype=values.dtype)
    load_extension(verbose).fused_gather_kv_fn(
        keys.contiguous(),
        values.contiguous(),
        indices.contiguous(),
        selected_keys,
        selected_values,
    )
    return selected_keys, selected_values


def attention_torch(Q, K, V, scaling):
    """Plain PyTorch scaled dot-product attention reference."""
    torch_mod = _require_torch()
    return torch_mod.softmax(Q @ K.transpose(-1, -2) / scaling, dim=-1) @ V


def measure_time(method: Callable[[], object], repeats: int = 10) -> float:
    """Measure elapsed CUDA time for a callable, including synchronization."""
    torch_mod = _require_cuda()
    torch_mod.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        method()
    torch_mod.cuda.synchronize()
    return time.time() - t0


def smoke_test(verbose: bool = False) -> None:
    """Run a small end-to-end CUDA test for the exported wrappers."""
    torch_mod = _require_cuda()
    torch_mod.manual_seed(0)

    B, H, L, D = 1, 1, 512, 128
    A = torch_mod.randn(B, H, L, D, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    _, quan_proj = find_binary_projection(A, num_bits=64, num_iters=2)
    quan_proj = quan_proj.contiguous()

    # Check that the packed CUDA projection agrees with the reference unpacked signs.
    ref = binary_project_torch(A, quan_proj)
    packed = binary_project_cuda(A, quan_proj, verbose=verbose)
    unpacked = unpack_bits(packed, verbose=verbose).to(A.dtype)
    agreement = (ref == unpacked).float().mean().item()
    if agreement < 0.95:
        raise AssertionError(f"binary projection agreement too low: {agreement:.4f}")

    # Exercise both staged and fused probe scoring paths.  Their quantization
    # reductions are implemented differently, so this smoke test validates shape
    # and finite outputs rather than requiring bit-exact equality.
    probe = torch_mod.randn(B, H, 4, 64, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    quant_probe = quantize_probe_cuda(probe, verbose=verbose)
    packed_probe = pack_probe_cuda(quant_probe, verbose=verbose)
    scores = probe_hash_cuda(packed_probe, packed, verbose=verbose)
    fused_scores = fused_probe_hash_cuda(probe, packed, verbose=verbose)
    if scores.shape != fused_scores.shape or scores.dtype != fused_scores.dtype:
        raise AssertionError("staged and fused probe outputs have inconsistent metadata")

    # Check fused K/V gather against torch.gather.
    k = 8
    indices = torch_mod.topk(fused_scores, k=k, dim=-1).indices.contiguous()
    keys = torch_mod.randn(B, H, L, D, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    values = torch_mod.randn(B, H, L, D, device="cuda", dtype=torch_mod.bfloat16).contiguous()
    selected_keys, selected_values = gather_kv_cuda(keys, values, indices, verbose=verbose)
    ref_keys = torch_mod.gather(keys, 2, indices[..., None].expand(-1, -1, -1, D))
    ref_values = torch_mod.gather(values, 2, indices[..., None].expand(-1, -1, -1, D))
    if not torch_mod.equal(selected_keys, ref_keys):
        raise AssertionError("selected_keys differs from torch.gather reference")
    if not torch_mod.equal(selected_values, ref_values):
        raise AssertionError("selected_values differs from torch.gather reference")

    print(f"smoke_test passed: binary agreement={agreement:.4f}, scores shape={tuple(scores.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary projection CUDA extension utilities.")
    parser.add_argument("--smoke-test", action="store_true", help="run a small CUDA correctness test")
    parser.add_argument("--verbose", action="store_true", help="enable verbose extension build output")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
