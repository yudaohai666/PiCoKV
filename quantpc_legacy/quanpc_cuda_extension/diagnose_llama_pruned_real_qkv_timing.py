"""Diagnose real-QKV pruning benchmark timing artifacts.

This script reuses the real Llama/QuanPC capture path, then compares two replay
measurements:

* overlap_wall: current benchmark style, launching all captured layer cases and
  synchronizing after the whole list. With device_map=auto this can overlap work
  from independent layers on different GPUs.
* event_sum_serial: CUDA-event time per captured layer, summed across layers.
  This approximates the serial per-layer attention cost and removes cross-GPU
  replay overlap from the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc import quanpc
from benchmarks.Needle.needle import LLMNeedleHaystackTester
from quantpc.quanpc import MixedCacheTensor, QuantKVCache, modify
from quantpc.quanpc_cuda_extension.benchmark_llama_pruned_real_qkv import (
    LayerCase,
    PromptProvider,
    RealDecodeCapture,
    run_full,
    run_tail,
    summarize_cases,
    sync_case_devices,
)
from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension


def time_overlap_wall(case_fn: Callable[[LayerCase], object], cases: list[LayerCase], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        for case in cases:
            case_fn(case)
    sync_case_devices(cases)
    start = time.perf_counter()
    for _ in range(iters):
        for case in cases:
            case_fn(case)
    sync_case_devices(cases)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def time_serial_wall(case_fn: Callable[[LayerCase], object], cases: list[LayerCase], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        for case in cases:
            case_fn(case)
            torch.cuda.synchronize(case.query.device)
    start = time.perf_counter()
    for _ in range(iters):
        for case in cases:
            case_fn(case)
            torch.cuda.synchronize(case.query.device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def time_event_sum(case_fn: Callable[[LayerCase], object], cases: list[LayerCase], warmup: int, iters: int) -> dict[str, object]:
    per_layer = []
    per_device = {}
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


def dense_case(case: LayerCase):
    return quanpc._dense_attention_numerator_flash_gqa(
        case.query,
        case.dense_key,
        case.dense_value,
        case.attention_scale,
    )


def project_case(case: LayerCase):
    return quanpc._project_query_gqa(case.query, case.key_proj, case.key_proj_t, out=case.projected_query)


def capture_real_cases(args) -> tuple[list[LayerCase], int]:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    provider = PromptProvider(tokenizer)
    tester = LLMNeedleHaystackTester(provider)
    context = tester.generate_context(args.length, args.depth)
    prompt = provider.generate_prompt(context, tester.retrieval_question)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt_text], return_tensors="pt")

    print(f"device_count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"device[{idx}]: {torch.cuda.get_device_name(idx)}")
    print(f"model: {args.model}")
    print(f"requested_length={args.length} actual_input_tokens={model_inputs.input_ids.shape[-1]}")
    print(
        f"quanpc: k_bits={args.k_bits} v_bits={args.v_bits} q_bits={args.projected_query_bits} "
        f"exp_bits={args.exp_bits} eas={args.eas} zerorope={args.zerorope}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    modify(model)
    first_device = next(model.parameters()).device
    model_inputs = model_inputs.to(first_device)

    capture = RealDecodeCapture()
    original_attention = quanpc.quantized_cache_attention_forward

    def capturing_attention(query_states, key_states, value_states):
        if isinstance(key_states, MixedCacheTensor) and isinstance(value_states, MixedCacheTensor):
            capture.capture_layer(query_states, key_states, value_states)
        return original_attention(query_states, key_states, value_states)

    quanpc.quantized_cache_attention_forward = capturing_attention
    try:
        with torch.no_grad():
            past_key_values = QuantKVCache(
                k_bits=args.k_bits,
                v_bits=args.v_bits,
                projected_query_bits=args.projected_query_bits,
                exp_bits=args.exp_bits,
                eas=args.eas,
                zerorope=args.zerorope,
                use_cuda_v1_exp8=True,
            )
            prefill_out = model(
                **model_inputs,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
                logits_to_keep=1,
            )
            next_token = prefill_out.logits[:, -1:].argmax(dim=-1)
            seq_len = model_inputs.input_ids.shape[-1]
            decode_attention_mask = torch.ones(
                model_inputs.input_ids.shape[0],
                seq_len + 1,
                device=first_device,
                dtype=torch.long,
            )
            cache_position = torch.arange(seq_len, seq_len + 1, device=first_device, dtype=torch.long)
            _ = model(
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            sync_case_devices(capture.layers)
    finally:
        quanpc.quantized_cache_attention_forward = original_attention

    if not capture.layers:
        raise RuntimeError("No decode layers were captured.")
    return capture.layers, int(model_inputs.input_ids.shape[-1])


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

    def base_tail(case: LayerCase):
        return run_tail(ext, case, args.exp_bits, False, 0, 0.0)

    def pruned_tail(case: LayerCase):
        return run_tail(ext, case, args.exp_bits, True, args.prune_mode, args.prune_log_eps)

    def base_full(case: LayerCase):
        return run_full(ext, case, args.exp_bits, False, 0, 0.0)

    def pruned_full(case: LayerCase):
        return run_full(ext, case, args.exp_bits, True, args.prune_mode, args.prune_log_eps)

    # Prime all paths once before measurement to avoid first-path warmup bias.
    for fn in (dense_case, project_case, base_tail, pruned_tail, base_full, pruned_full):
        for case in cases:
            fn(case)
    sync_case_devices(cases)

    overlap = {
        "base_tail_ms": time_overlap_wall(base_tail, cases, args.warmup, args.iters),
        "pruned_tail_ms": time_overlap_wall(pruned_tail, cases, args.warmup, args.iters),
        "base_full_ms": time_overlap_wall(base_full, cases, args.warmup, args.iters),
        "pruned_full_ms": time_overlap_wall(pruned_full, cases, args.warmup, args.iters),
    }
    overlap["tail_speedup"] = overlap["base_tail_ms"] / overlap["pruned_tail_ms"]
    overlap["full_speedup"] = overlap["base_full_ms"] / overlap["pruned_full_ms"]

    serial_wall = {
        "base_tail_ms": time_serial_wall(base_tail, cases, max(1, args.warmup // 2), max(1, args.iters // 2)),
        "pruned_tail_ms": time_serial_wall(pruned_tail, cases, max(1, args.warmup // 2), max(1, args.iters // 2)),
        "base_full_ms": time_serial_wall(base_full, cases, max(1, args.warmup // 2), max(1, args.iters // 2)),
        "pruned_full_ms": time_serial_wall(pruned_full, cases, max(1, args.warmup // 2), max(1, args.iters // 2)),
    }
    serial_wall["tail_speedup"] = serial_wall["base_tail_ms"] / serial_wall["pruned_tail_ms"]
    serial_wall["full_speedup"] = serial_wall["base_full_ms"] / serial_wall["pruned_full_ms"]

    event_sum = {
        "dense": time_event_sum(dense_case, cases, args.warmup, args.iters),
        "project": time_event_sum(project_case, cases, args.warmup, args.iters),
        "base_tail": time_event_sum(base_tail, cases, args.warmup, args.iters),
        "pruned_tail": time_event_sum(pruned_tail, cases, args.warmup, args.iters),
        "base_full": time_event_sum(base_full, cases, args.warmup, args.iters),
        "pruned_full": time_event_sum(pruned_full, cases, args.warmup, args.iters),
    }
    event_summary = {
        "dense_sum_ms": event_sum["dense"]["sum_ms"],
        "project_sum_ms": event_sum["project"]["sum_ms"],
        "base_tail_sum_ms": event_sum["base_tail"]["sum_ms"],
        "pruned_tail_sum_ms": event_sum["pruned_tail"]["sum_ms"],
        "base_full_sum_ms": event_sum["base_full"]["sum_ms"],
        "pruned_full_sum_ms": event_sum["pruned_full"]["sum_ms"],
    }
    event_summary["tail_speedup"] = event_summary["base_tail_sum_ms"] / event_summary["pruned_tail_sum_ms"]
    event_summary["full_speedup"] = event_summary["base_full_sum_ms"] / event_summary["pruned_full_sum_ms"]
    event_summary["tail_delta_ms"] = event_summary["base_tail_sum_ms"] - event_summary["pruned_tail_sum_ms"]
    event_summary["full_delta_ms"] = event_summary["base_full_sum_ms"] - event_summary["pruned_full_sum_ms"]

    print("overlap_wall", json.dumps(overlap, sort_keys=True))
    print("serial_wall", json.dumps(serial_wall, sort_keys=True))
    print("event_sum_serial", json.dumps(event_summary, sort_keys=True))
    print("event_sum_base_tail_per_device", json.dumps(event_sum["base_tail"]["per_device_sum_ms"], sort_keys=True))
    print("event_sum_pruned_tail_per_device", json.dumps(event_sum["pruned_tail"]["per_device_sum_ms"], sort_keys=True))

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
            "overlap_wall": overlap,
            "serial_wall": serial_wall,
            "event_sum_serial": event_summary,
            "event_detail": event_sum,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
