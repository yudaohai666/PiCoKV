"""Compare real-QKV QuanPC decode attention against torch SDPA.

The script captures a real Llama-3.1/QuanPC one-token decode step at the
requested length, then measures per-layer CUDA-event sums.  SDPA uses the real
captured query shape/device and a same-shape bf16 full KV tensor with length
``dense_len + global_len``; the original full global KV is not retained after
QuanPC prefill compression, and SDPA speed is shape/dtype dominated.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc.quanpc_cuda_extension.benchmark_llama_pruned_real_qkv import (
    LayerCase,
    RealDecodeCapture,
    PromptProvider,
    run_full,
    run_tail,
    summarize_cases,
    sync_case_devices,
)
from quantpc.quanpc_cuda_extension.diagnose_llama_pruned_real_qkv_timing import capture_real_cases
from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension


def time_event_sum(case_fn: Callable[[LayerCase], object], cases: list[LayerCase], warmup: int, iters: int) -> dict[str, object]:
    per_layer = []
    per_device = {}
    with torch.no_grad():
        for case in cases:
            device = case.query.device
            with torch.cuda.device(device):
                for _ in range(warmup):
                    case_fn(case)
                torch.cuda.synchronize(device)
                stream = torch.cuda.current_stream(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                for _ in range(iters):
                    case_fn(case)
                end.record(stream)
                torch.cuda.synchronize(device)
                ms = start.elapsed_time(end) / max(1, iters)
            per_layer.append({"layer_idx": case.layer_idx, "device": str(device), "ms": ms})
            per_device[str(device)] = per_device.get(str(device), 0.0) + ms
    return {
        "sum_ms": sum(item["ms"] for item in per_layer),
        "max_layer_ms": max(item["ms"] for item in per_layer),
        "max_device_sum_ms": max(per_device.values()) if per_device else 0.0,
        "per_device_sum_ms": per_device,
        "per_layer_ms": per_layer,
    }


def time_sdpa_event_sum(cases: list[LayerCase], warmup: int, iters: int) -> dict[str, object]:
    per_layer = []
    per_device = {}
    with torch.no_grad():
        for case in cases:
            device = case.query.device
            B, QH, S, D = case.query.shape
            KVH = case.dense_key.shape[1]
            L = case.dense_len + case.global_len
            with torch.cuda.device(device):
                key = torch.randn(B, KVH, L, D, device=device, dtype=torch.bfloat16).contiguous()
                value = torch.randn(B, KVH, L, D, device=device, dtype=torch.bfloat16).contiguous()

                def sdpa_once():
                    return torch.nn.functional.scaled_dot_product_attention(
                        case.query,
                        key,
                        value,
                        is_causal=False,
                        enable_gqa=True,
                    )

                for _ in range(warmup):
                    sdpa_once()
                torch.cuda.synchronize(device)
                stream = torch.cuda.current_stream(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                for _ in range(iters):
                    sdpa_once()
                end.record(stream)
                torch.cuda.synchronize(device)
                ms = start.elapsed_time(end) / max(1, iters)
                del key, value
                torch.cuda.empty_cache()
            per_layer.append({"layer_idx": case.layer_idx, "device": str(device), "ms": ms, "L": L})
            per_device[str(device)] = per_device.get(str(device), 0.0) + ms
    return {
        "sum_ms": sum(item["ms"] for item in per_layer),
        "max_layer_ms": max(item["ms"] for item in per_layer),
        "max_device_sum_ms": max(per_device.values()) if per_device else 0.0,
        "per_device_sum_ms": per_device,
        "per_layer_ms": per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--k-bits", type=int, default=256)
    parser.add_argument("--v-bits", type=int, default=256)
    parser.add_argument("--projected-query-bits", type=int, default=8)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--prune-mode", type=int, default=3)
    parser.add_argument("--prune-log-eps", type=float, default=-10.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    ext = load_quanpc_extension(verbose=False)
    if not hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn"):
        raise RuntimeError("Pruned CUDA extension entry point is missing; rebuild the extension.")

    cases, actual_input_tokens = capture_real_cases(args)
    summary = summarize_cases(cases)
    print("captured:", json.dumps(summary, sort_keys=True))
    gc.collect()
    torch.cuda.empty_cache()

    def base_tail(case: LayerCase):
        return run_tail(ext, case, args.exp_bits, False, 0, 0.0)

    def pruned_tail(case: LayerCase):
        return run_tail(ext, case, args.exp_bits, True, args.prune_mode, args.prune_log_eps)

    def base_full(case: LayerCase):
        return run_full(ext, case, args.exp_bits, False, 0, 0.0)

    def pruned_full(case: LayerCase):
        return run_full(ext, case, args.exp_bits, True, args.prune_mode, args.prune_log_eps)

    # Prime all QuanPC paths once before measurement.
    for fn in (base_tail, pruned_tail, base_full, pruned_full):
        for case in cases:
            fn(case)
    sync_case_devices(cases)

    measurements = {
        "sdpa_full": time_sdpa_event_sum(cases, args.warmup, args.iters),
        "quanpc_base_tail": time_event_sum(base_tail, cases, args.warmup, args.iters),
        "quanpc_pruned_tail": time_event_sum(pruned_tail, cases, args.warmup, args.iters),
        "quanpc_base_full": time_event_sum(base_full, cases, args.warmup, args.iters),
        "quanpc_pruned_full": time_event_sum(pruned_full, cases, args.warmup, args.iters),
    }
    summary_rows = {
        "sdpa_full_sum_ms": measurements["sdpa_full"]["sum_ms"],
        "quanpc_base_tail_sum_ms": measurements["quanpc_base_tail"]["sum_ms"],
        "quanpc_pruned_tail_sum_ms": measurements["quanpc_pruned_tail"]["sum_ms"],
        "quanpc_base_full_sum_ms": measurements["quanpc_base_full"]["sum_ms"],
        "quanpc_pruned_full_sum_ms": measurements["quanpc_pruned_full"]["sum_ms"],
    }
    summary_rows["base_full_speedup_vs_sdpa"] = summary_rows["sdpa_full_sum_ms"] / summary_rows["quanpc_base_full_sum_ms"]
    summary_rows["pruned_full_speedup_vs_sdpa"] = summary_rows["sdpa_full_sum_ms"] / summary_rows["quanpc_pruned_full_sum_ms"]
    summary_rows["base_tail_speedup_vs_sdpa"] = summary_rows["sdpa_full_sum_ms"] / summary_rows["quanpc_base_tail_sum_ms"]
    summary_rows["pruned_tail_speedup_vs_sdpa"] = summary_rows["sdpa_full_sum_ms"] / summary_rows["quanpc_pruned_tail_sum_ms"]

    print("event_sum", json.dumps(summary_rows, sort_keys=True))
    print("sdpa_per_device", json.dumps(measurements["sdpa_full"]["per_device_sum_ms"], sort_keys=True))
    print("quanpc_pruned_full_per_device", json.dumps(measurements["quanpc_pruned_full"]["per_device_sum_ms"], sort_keys=True))

    if args.output_json:
        payload = {
            "model": args.model,
            "requested_length": args.length,
            "actual_input_tokens": actual_input_tokens,
            "config": {
                "k_bits": args.k_bits,
                "v_bits": args.v_bits,
                "projected_query_bits": args.projected_query_bits,
                "exp_bits": args.exp_bits,
                "eas": args.eas,
                "zerorope": args.zerorope,
                "prune_mode": args.prune_mode,
                "prune_log_eps": args.prune_log_eps,
                "warmup": args.warmup,
                "iters": args.iters,
            },
            "captured": summary,
            "event_sum": summary_rows,
            "measurements": measurements,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
