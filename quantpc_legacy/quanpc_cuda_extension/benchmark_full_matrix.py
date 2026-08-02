"""Full decode-path benchmark matrix for QuanPC CUDA.

This benchmark compares torch bf16 SDPA against the complete QuanPC decoding
attention path:

1. dense retained-KV flash attention;
2. query projection by torch.bmm;
3. CUDA decode tail: projected-query quantize/pack, packed global attention,
   value projection, and final normalization.

Projection learning, KV quantization, and KV bit packing are prefill/offline
work and are intentionally excluded from the decoding latency.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_extension(so_path: str | None):
    if so_path is None:
        from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension

        return load_quanpc_extension(verbose=False)
    spec = importlib.util.spec_from_file_location("quanpc_cuda", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load extension from {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape(tensor: torch.Tensor) -> str:
    return "[" + ",".join(str(dim) for dim in tensor.shape) + "]"


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


def _dense_flash(query, key, value, attention_scale: float):
    out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        query,
        key,
        value,
        0.0,
        False,
        False,
        scale=float(attention_scale),
    )
    return out, lse


def _project_query_gqa_out(query, key_proj_t, out):
    B, QH, S, D = query.shape
    KVH = key_proj_t.shape[1]
    C = key_proj_t.shape[3]
    group_size = QH // KVH
    query_grouped = query.reshape(B, KVH, group_size, S, D).reshape(B * KVH, group_size * S, D)
    proj_grouped = key_proj_t.reshape(B * KVH, D, C)
    projected_grouped = out.reshape(B, KVH, group_size, S, C).reshape(B * KVH, group_size * S, C)
    torch.bmm(query_grouped, proj_grouped, out=projected_grouped)
    return out


def _make_case(batch_size: int, global_len: int, dense_len: int, gqa: int):
    if gqa not in (1, 4, 8):
        raise ValueError(f"gqa must be one of 1, 4, or 8; got {gqa}.")

    B = batch_size
    KVH = 8
    QH = KVH * gqa
    S = 1
    D = 128
    Ck = 256
    Cv = 256
    k_blocks = Ck // 64
    l_groups = (global_len + 63) // 64

    query = torch.randn(B, QH, S, D, device="cuda", dtype=torch.bfloat16).contiguous()
    dense_key = torch.randn(B, KVH, dense_len, D, device="cuda", dtype=torch.bfloat16).contiguous()
    dense_value = torch.randn(B, KVH, dense_len, D, device="cuda", dtype=torch.bfloat16).contiguous()
    global_key_dense = torch.randn(B, KVH, global_len, D, device="cuda", dtype=torch.bfloat16).contiguous()
    global_value_dense = torch.randn(B, KVH, global_len, D, device="cuda", dtype=torch.bfloat16).contiguous()
    torch_key = torch.cat([dense_key, global_key_dense], dim=2).contiguous()
    torch_value = torch.cat([dense_value, global_value_dense], dim=2).contiguous()

    key_proj = torch.randn(B, KVH, Ck, D, device="cuda", dtype=torch.bfloat16).contiguous()
    key_proj_t = key_proj.transpose(-1, -2).contiguous()
    value_proj = torch.randn(B, KVH, Cv, D, device="cuda", dtype=torch.bfloat16).contiguous()
    packed_key = torch.empty(B, KVH, k_blocks, global_len, device="cuda", dtype=torch.long).random_().contiguous()
    packed_value = torch.empty(B, KVH, l_groups, Cv, device="cuda", dtype=torch.long).random_().contiguous()

    projected_query = torch.empty(B, QH, S, Ck, device="cuda", dtype=torch.bfloat16)
    packed_query = torch.empty(B, QH, S, k_blocks, 8, device="cuda", dtype=torch.long)
    query_scale = torch.empty(B, QH, S, device="cuda", dtype=torch.float32)
    dense_output = torch.empty_like(query)
    dense_lse = torch.empty(B, QH, S, device="cuda", dtype=torch.float32)
    glob_code_numel = B * QH * S * Cv
    glob_den_numel = B * QH * S
    glob_scratch = torch.empty(glob_code_numel + glob_den_numel, device="cuda", dtype=torch.float32)
    glob_code = glob_scratch[:glob_code_numel].view(B, QH, S, Cv)
    glob_den = glob_scratch[glob_code_numel:].view(B, QH, S)
    attn_output = torch.empty_like(query)

    return {
        "B": B,
        "QH": QH,
        "KVH": KVH,
        "S": S,
        "D": D,
        "Ck": Ck,
        "Cv": Cv,
        "global_len": global_len,
        "dense_len": dense_len,
        "gqa": gqa,
        "query": query,
        "dense_key": dense_key,
        "dense_value": dense_value,
        "torch_key": torch_key,
        "torch_value": torch_value,
        "key_proj_t": key_proj_t,
        "value_proj": value_proj,
        "packed_key": packed_key,
        "packed_value": packed_value,
        "projected_query": projected_query,
        "packed_query": packed_query,
        "query_scale": query_scale,
        "dense_output": dense_output,
        "dense_lse": dense_lse,
        "glob_code": glob_code,
        "glob_den": glob_den,
        "attn_output": attn_output,
    }


def _benchmark_case(ext, batch_size: int, global_len: int, dense_len: int, gqa: int, exp_bits: int, warmup: int, iters: int):
    case = _make_case(batch_size, global_len, dense_len, gqa)
    attention_scale = 1.0 / math.sqrt(case["D"])
    dense_stream = torch.cuda.Stream()
    dense_state = {
        "output": case["dense_output"],
        "lse": case["dense_lse"],
    }

    def torch_full():
        return torch.nn.functional.scaled_dot_product_attention(
            case["query"],
            case["torch_key"],
            case["torch_value"],
            is_causal=False,
            enable_gqa=True,
        )

    def dense_only():
        out, lse = _dense_flash(case["query"], case["dense_key"], case["dense_value"], attention_scale)
        dense_state["output"] = out
        dense_state["lse"] = lse
        return out

    def project_only():
        return _project_query_gqa_out(case["query"], case["key_proj_t"], case["projected_query"])

    def tail_only():
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            case["projected_query"],
            case["packed_query"],
            case["packed_key"],
            case["packed_value"],
            case["query_scale"],
            dense_state["lse"],
            dense_state["output"].contiguous(),
            case["glob_code"],
            case["glob_den"],
            case["value_proj"],
            case["attn_output"],
            int(exp_bits),
            256,
            1,
            float(attention_scale),
        )
        return case["attn_output"]

    def quanpc_full_serial():
        dense_only()
        project_only()
        tail_only()
        return case["attn_output"]

    def quanpc_full_overlap():
        current = torch.cuda.current_stream()
        with torch.cuda.stream(dense_stream):
            out, lse = _dense_flash(case["query"], case["dense_key"], case["dense_value"], attention_scale)
            dense_state["output"] = out
            dense_state["lse"] = lse
        project_only()
        current.wait_stream(dense_stream)
        tail_only()
        return case["attn_output"]

    def quanpc_full_auto():
        if batch_size == 1:
            return quanpc_full_serial()
        return quanpc_full_overlap()

    dense_only()
    project_only()
    torch.cuda.synchronize()

    torch_ms = _time_cuda(torch_full, warmup, iters)
    quanpc_ms = _time_cuda(quanpc_full_auto, warmup, iters)
    speedup = torch_ms / quanpc_ms

    print(
        "result "
        f"B={batch_size} global_len={global_len} dense_len={dense_len} "
        f"gqa={gqa} exp_bits={exp_bits} "
        f"torch_Q={_shape(case['query'])} torch_KV={_shape(case['torch_key'])} "
        f"packed_k={_shape(case['packed_key'])} packed_v={_shape(case['packed_value'])} "
        f"torch_full_ms={torch_ms:.4f} quanpc_full_ms={quanpc_ms:.4f} speedup={speedup:.3f}x"
    )
    sys.stdout.flush()

    del case
    gc.collect()
    torch.cuda.empty_cache()
    return torch_ms, quanpc_ms, speedup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", default=None, help="Optional prebuilt quanpc_cuda .so path.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--global-lengths", type=int, nargs="+", default=[32768, 65536, 100000])
    parser.add_argument("--gqa", type=int, nargs="+", default=[1, 4, 8], choices=[1, 4, 8])
    parser.add_argument("--exp-bits", type=int, nargs="+", default=[1, 2, 4, 8], choices=[1, 2, 4, 8])
    parser.add_argument("--dense-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    ext = _load_extension(args.so)
    print(f"device: {torch.cuda.get_device_name()}")
    print("mode: full decode path matrix")
    print(
        "columns: B, global_len, dense_len, gqa, exp_bits, shapes, "
        "torch_full_ms, quanpc_full_ms, speedup"
    )
    for global_len in args.global_lengths:
        for gqa in args.gqa:
            for exp_bits in args.exp_bits:
                for batch_size in args.batch_sizes:
                    _benchmark_case(
                        ext,
                        batch_size=batch_size,
                        global_len=global_len,
                        dense_len=args.dense_len,
                        gqa=gqa,
                        exp_bits=exp_bits,
                        warmup=args.warmup,
                        iters=args.iters,
                    )


if __name__ == "__main__":
    main()
