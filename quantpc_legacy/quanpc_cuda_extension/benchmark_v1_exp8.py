"""Benchmark the packed QuanPC v1 configurable-exp CUDA path against torch SDPA.

The printed shapes are part of the benchmark contract. Keep them in the
output whenever reporting throughput so CUDA timings can be interpreted
against the exact dense and packed input sizes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from quantpc.quanpc_cuda_extension.quanpc_cuda import (
    pack_projected_query_cuda,
    quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda,
    quanpc_global_code_output_v1_exp8_tiled_gqa4_cuda,
    quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_cuda,
)


def _shape(tensor: torch.Tensor) -> str:
    return "[" + ", ".join(str(dim) for dim in tensor.shape) + "]"


def _time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    with torch.no_grad():
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
    return start.elapsed_time(end) / iters


def _make_case(
    batch_size: int,
    length: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    code_dim: int,
    value_code_dim: int,
):
    if q_heads % kv_heads != 0 or q_heads // kv_heads not in (1, 4, 8):
        raise ValueError(f"v1 configurable-exp kernel requires q_heads == kv_heads, 4 * kv_heads, or 8 * kv_heads; got {q_heads}, {kv_heads}.")
    if code_dim not in (128, 256):
        raise ValueError(f"v1 configurable-exp kernel supports code_dim 128 or 256; got {code_dim}.")
    if value_code_dim not in (128, 256):
        raise ValueError(f"v1 configurable-exp kernel supports value_code_dim 128 or 256; got {value_code_dim}.")
    if code_dim % 64 != 0:
        raise ValueError(f"code_dim must be divisible by 64; got {code_dim}.")

    B, L, QH, KVH = batch_size, length, q_heads, kv_heads
    S = 1
    k_blocks = code_dim // 64
    l_groups = (L + 63) // 64

    q_bf16 = torch.randn(B, QH, S, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    k_bf16 = torch.randn(B, KVH, L, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()
    v_bf16 = torch.randn(B, KVH, L, head_dim, device="cuda", dtype=torch.bfloat16).contiguous()

    quant_projected_query = torch.randint(-127, 128, (B, QH, S, code_dim), device="cuda", dtype=torch.int8).contiguous()
    packed_projected_query = pack_projected_query_cuda(quant_projected_query).contiguous()
    packed_key = torch.empty(B, KVH, k_blocks, L, device="cuda", dtype=torch.long).random_().contiguous()
    packed_value_sign = torch.empty(B, KVH, l_groups, value_code_dim, device="cuda", dtype=torch.long).random_().contiguous()
    query_scale = torch.full((B, QH, S), 0.005, device="cuda", dtype=torch.float32).contiguous()
    dense_lse = torch.zeros(B, QH, S, device="cuda", dtype=torch.float32).contiguous()

    return {
        "q_bf16": q_bf16,
        "k_bf16": k_bf16,
        "v_bf16": v_bf16,
        "quant_projected_query": quant_projected_query,
        "packed_projected_query": packed_projected_query,
        "packed_key": packed_key,
        "packed_value_sign": packed_value_sign,
        "query_scale": query_scale,
        "dense_lse": dense_lse,
    }


def _print_shapes(case: dict[str, torch.Tensor], attention_scale: float, tile_size: int, value_threads: int, exp_bits: int) -> None:
    print("shapes:")
    print(
        "  torch_sdpa: "
        f"Q={_shape(case['q_bf16'])} bf16, "
        f"K={_shape(case['k_bf16'])} bf16, "
        f"V={_shape(case['v_bf16'])} bf16, enable_gqa=True"
    )
    print(
        "  quanpc_core: "
        f"packed_q={_shape(case['packed_projected_query'])} int64, "
        f"packed_k={_shape(case['packed_key'])} int64 [B,KVH,Ck/64,L], "
        f"packed_v={_shape(case['packed_value_sign'])} int64 [B,KVH,ceil(L/64),Cv], "
        f"q_scale={_shape(case['query_scale'])} fp32, "
        f"dense_lse={_shape(case['dense_lse'])} fp32"
    )
    print(
        "  quanpc_pack_core: "
        f"quant_projected_query={_shape(case['quant_projected_query'])} int8 -> packed_q, "
        f"attention_scale={attention_scale:.8f}, tile_size={tile_size}, exp_bits={exp_bits}, query_bits=8, value_threads={value_threads}"
    )


def benchmark_case(
    batch_size: int,
    length: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    code_dim: int,
    value_code_dim: int,
    warmup: int,
    iters: int,
    use_atomic: bool,
    use_bmma_score: bool,
    value_threads: int,
    exp_bits: int,
) -> None:
    case = _make_case(batch_size, length, q_heads, kv_heads, head_dim, code_dim, value_code_dim)
    attention_scale = 1.0 / math.sqrt(head_dim)
    tile_size = 256
    if value_threads != 256:
        raise ValueError(f"value_threads must be 256; got {value_threads}.")
    if exp_bits not in (1, 2, 4, 8):
        raise ValueError(f"exp_bits must be one of 1, 2, 4, or 8; got {exp_bits}.")
    if use_bmma_score:
        if not use_atomic:
            raise ValueError("--bmma-score requires --atomic.")
        if q_heads // kv_heads != 8 or code_dim != 256 or value_code_dim != 256 or exp_bits != 4:
            raise ValueError("--bmma-score currently requires q_heads == 8 * kv_heads, Ck=Cv=256, and exp_bits=4.")
        kernel = quanpc_global_code_output_v1_exp8_tiled_gqa8_bmma_score_atomic_cuda
    else:
        kernel = (
            quanpc_global_code_output_v1_exp8_tiled_gqa4_atomic_cuda
            if use_atomic
            else quanpc_global_code_output_v1_exp8_tiled_gqa4_cuda
        )

    def torch_sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            case["q_bf16"],
            case["k_bf16"],
            case["v_bf16"],
            is_causal=False,
            enable_gqa=True,
        )

    def quanpc_core():
        return kernel(
            case["packed_projected_query"],
            case["packed_key"],
            case["packed_value_sign"],
            case["query_scale"],
            case["dense_lse"],
            attention_scale,
            exp_bits=exp_bits,
            tile_size=tile_size,
            key_transposed=True,
        )

    def quanpc_pack_core():
        packed_q = pack_projected_query_cuda(case["quant_projected_query"])
        return kernel(
            packed_q,
            case["packed_key"],
            case["packed_value_sign"],
            case["query_scale"],
            case["dense_lse"],
            attention_scale,
            exp_bits=exp_bits,
            tile_size=tile_size,
            key_transposed=True,
        )

    print(f"\ncase: B={batch_size}, L={length}, QH={q_heads}, KVH={kv_heads}, D={head_dim}, Ck={code_dim}, Cv={value_code_dim}, exp_bits={exp_bits}")
    _print_shapes(case, attention_scale, tile_size, value_threads, exp_bits)

    torch_ms = _time_cuda(torch_sdpa, warmup, iters)
    core_ms = _time_cuda(quanpc_core, warmup, iters)
    pack_core_ms = _time_cuda(quanpc_pack_core, warmup, iters)
    print(
        "timing: "
        f"torch_sdpa={torch_ms:.4f} ms, "
        f"quanpc_core={core_ms:.4f} ms, "
        f"quanpc_pack_core={pack_core_ms:.4f} ms, "
        f"core_speedup={torch_ms / core_ms:.3f}x, "
        f"pack_core_speedup={torch_ms / pack_core_ms:.3f}x"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[65536, 100000])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=256)
    parser.add_argument("--value-code-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--atomic", action="store_true", help="benchmark the experimental atomic direct-output kernel")
    parser.add_argument("--bmma-score", action="store_true", help="benchmark the GQA8 BMMA-score prototype; requires --atomic, exp_bits=4, Ck=Cv=256")
    parser.add_argument("--exp-bits", type=int, nargs="+", default=[4], choices=[1, 2, 4, 8], help="exp-weight quantization bits to benchmark")
    parser.add_argument("--value-threads", type=int, default=256, choices=[256], help="threads per block used by the value accumulation stage")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    args = parse_args()
    print(f"device: {torch.cuda.get_device_name()}")
    mode = "bmma_score_atomic" if args.bmma_score else ('atomic' if args.atomic else 'partial_reduce')
    print(f"mode: quanpc_v1_expbits_{mode}_v{args.value_threads}")
    for exp_bits in args.exp_bits:
        for length in args.lengths:
            for batch_size in args.batch_sizes:
                benchmark_case(
                    batch_size=batch_size,
                    length=length,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    code_dim=args.code_dim,
                    value_code_dim=args.value_code_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    use_atomic=args.atomic,
                    use_bmma_score=args.bmma_score,
                    value_threads=args.value_threads,
                    exp_bits=exp_bits,
                )


if __name__ == "__main__":
    main()
