"""Benchmark CUDA pruning on real Llama-3.1 QuanPC decode tensors.

The benchmark builds a real QuanPC cache with a long NIAH prompt, captures the
single-token decode Q/K/V-derived tensors for every layer, then replays only the
attention workspaces.  Prefill/model generation is intentionally excluded from
latency measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc import quanpc
from benchmarks.Needle.needle import LLMNeedleHaystackTester, ModelProvider
from quantpc.quanpc import MixedCacheTensor, QuantKVCache, modify
from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension


class PromptProvider(ModelProvider):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate_model(self, prompt, needle=None) -> str:
        raise NotImplementedError

    def generate_prompt(self, context: str, retrieval_question: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI bot that answers questions for a user. "
                    "Keep your response short and direct"
                ),
            },
            {"role": "user", "content": context},
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


@dataclass
class LayerCase:
    layer_idx: int
    query: torch.Tensor
    dense_key: torch.Tensor
    dense_value: torch.Tensor
    key_proj: torch.Tensor
    key_proj_t: torch.Tensor
    packed_key: torch.Tensor
    packed_value: torch.Tensor
    value_proj: torch.Tensor
    projected_query: torch.Tensor
    packed_query: torch.Tensor
    query_scale: torch.Tensor
    dense_output: torch.Tensor
    dense_lse: torch.Tensor
    glob_code: torch.Tensor
    glob_den: torch.Tensor
    attn_output: torch.Tensor
    attention_scale: float
    global_len: int
    dense_len: int

    @property
    def shape_summary(self) -> dict[str, int]:
        B, QH, S, D = self.query.shape
        KVH = self.dense_key.shape[1]
        return {
            "B": B,
            "QH": QH,
            "KVH": KVH,
            "S": S,
            "D": D,
            "global_len": self.global_len,
            "dense_len": self.dense_len,
            "Ck": self.projected_query.shape[-1],
            "Cv": self.value_proj.shape[-2],
        }


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def sync_case_devices(cases: list[LayerCase]) -> None:
    devices = sorted({case.query.device.index for case in cases if case.query.is_cuda})
    for device_idx in devices:
        torch.cuda.synchronize(device_idx)


def time_cuda_workload(fn: Callable[[], None], cases: list[LayerCase], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync_case_devices(cases)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_case_devices(cases)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


_RUN_FULL_DENSE_STREAMS: dict[tuple[str, int], torch.cuda.Stream] = {}


def _get_run_full_dense_stream(tensor: torch.Tensor) -> torch.cuda.Stream:
    device = tensor.device
    key = (device.type, device.index if device.index is not None else torch.cuda.current_device())
    stream = _RUN_FULL_DENSE_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _RUN_FULL_DENSE_STREAMS[key] = stream
    return stream


class RealDecodeCapture:
    def __init__(self):
        self.layers: list[LayerCase] = []
        self.call_idx = 0

    def capture_layer(self, query_states: torch.Tensor, key_states: MixedCacheTensor, value_states: MixedCacheTensor) -> None:
        if query_states.shape[-2] != 1:
            return
        if key_states.glob.packed_cuda is None or value_states.glob.packed_cuda is None:
            return
        if query_states.shape[1] != 4 * key_states.dense.shape[1]:
            return

        layer_idx = self.call_idx
        self.call_idx += 1
        B, QH, S, D = query_states.shape
        Ck = key_states.glob.proj.shape[-2]
        Cv = value_states.glob.proj.shape[-2]
        attention_scale = 1.0 / math.sqrt(D)

        dense_output, dense_lse = quanpc._dense_attention_numerator_flash_gqa(
            query_states,
            key_states.dense,
            value_states.dense,
            attention_scale,
        )
        projected_query = quanpc._project_query_gqa(
            query_states,
            key_states.glob.proj,
            key_states.glob.proj_t,
        ).contiguous()

        packed_query = torch.empty(B, QH, S, Ck // 64, 8, device=query_states.device, dtype=torch.long)
        query_scale = torch.empty(B, QH, S, device=query_states.device, dtype=torch.float32)
        glob_code = torch.empty(B, QH, S, Cv, device=query_states.device, dtype=torch.float32)
        glob_den = torch.empty(B, QH, S, device=query_states.device, dtype=torch.float32)
        attn_output = torch.empty_like(query_states)

        self.layers.append(
            LayerCase(
                layer_idx=layer_idx,
                query=query_states.detach().contiguous(),
                dense_key=key_states.dense.detach().contiguous(),
                dense_value=value_states.dense.detach().contiguous(),
                key_proj=key_states.glob.proj.detach().contiguous(),
                key_proj_t=key_states.glob.proj_t.detach().contiguous(),
                packed_key=key_states.glob.packed_cuda.detach().contiguous(),
                packed_value=value_states.glob.packed_cuda.detach().contiguous(),
                value_proj=value_states.glob.proj.detach().contiguous(),
                projected_query=projected_query.detach().contiguous(),
                packed_query=packed_query,
                query_scale=query_scale,
                dense_output=dense_output.detach().contiguous(),
                dense_lse=dense_lse.detach().contiguous(),
                glob_code=glob_code,
                glob_den=glob_den,
                attn_output=attn_output,
                attention_scale=attention_scale,
                global_len=int(key_states.glob.packed_cuda.shape[-1]),
                dense_len=int(key_states.dense.shape[-2]),
            )
        )


def run_tail(ext, case: LayerCase, exp_bits: int, pruned: bool, prune_mode: int, prune_log_eps: float) -> torch.Tensor:
    if pruned:
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn(
            case.projected_query,
            case.packed_query,
            case.packed_key,
            case.packed_value,
            case.query_scale,
            case.dense_lse,
            case.dense_output,
            case.glob_code,
            case.glob_den,
            case.value_proj,
            case.attn_output,
            int(exp_bits),
            256,
            1,
            float(case.attention_scale),
            int(prune_mode),
            float(prune_log_eps),
        )
    else:
        ext.quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_fn(
            case.projected_query,
            case.packed_query,
            case.packed_key,
            case.packed_value,
            case.query_scale,
            case.dense_lse,
            case.dense_output,
            case.glob_code,
            case.glob_den,
            case.value_proj,
            case.attn_output,
            int(exp_bits),
            256,
            1,
            float(case.attention_scale),
        )
    return case.attn_output


def run_full(ext, case: LayerCase, exp_bits: int, pruned: bool, prune_mode: int, prune_log_eps: float) -> torch.Tensor:
    if case.query.is_cuda:
        current_stream = torch.cuda.current_stream(case.query.device)
        dense_stream = _get_run_full_dense_stream(case.query)
        dense_stream.wait_stream(current_stream)
        with torch.cuda.stream(dense_stream):
            dense_output, dense_lse = quanpc._dense_attention_numerator_flash_gqa(
                case.query,
                case.dense_key,
                case.dense_value,
                case.attention_scale,
            )
        quanpc._project_query_gqa(case.query, case.key_proj, case.key_proj_t, out=case.projected_query)
        current_stream.wait_stream(dense_stream)
    else:
        dense_output, dense_lse = quanpc._dense_attention_numerator_flash_gqa(
            case.query,
            case.dense_key,
            case.dense_value,
            case.attention_scale,
        )
        quanpc._project_query_gqa(case.query, case.key_proj, case.key_proj_t, out=case.projected_query)

    old_dense_output = case.dense_output
    old_dense_lse = case.dense_lse
    case.dense_output = dense_output.contiguous()
    case.dense_lse = dense_lse.contiguous()
    try:
        return run_tail(ext, case, exp_bits, pruned, prune_mode, prune_log_eps)
    finally:
        case.dense_output = old_dense_output
        case.dense_lse = old_dense_lse


def collect_reference_outputs(ext, cases: list[LayerCase], exp_bits: int) -> list[torch.Tensor]:
    refs = []
    for case in cases:
        refs.append(run_tail(ext, case, exp_bits, False, 0, 0.0).detach().clone())
    sync_case_devices(cases)
    return refs


def compare_to_refs(ext, cases: list[LayerCase], refs: list[torch.Tensor], exp_bits: int, prune_mode: int, prune_log_eps: float) -> tuple[float, float]:
    max_abs = 0.0
    sum_abs = 0.0
    numel = 0
    for case, ref in zip(cases, refs):
        out = run_tail(ext, case, exp_bits, True, prune_mode, prune_log_eps)
        diff = (out.float() - ref.float()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        sum_abs += float(diff.sum().item())
        numel += diff.numel()
    sync_case_devices(cases)
    return max_abs, sum_abs / max(1, numel)


def summarize_cases(cases: list[LayerCase]) -> dict[str, object]:
    if not cases:
        return {}
    first = cases[0].shape_summary
    global_lens = [case.global_len for case in cases]
    dense_lens = [case.dense_len for case in cases]
    devices = sorted({str(case.query.device) for case in cases})
    return {
        **first,
        "layers": len(cases),
        "devices": devices,
        "global_len_min": min(global_lens),
        "global_len_max": max(global_lens),
        "dense_len_min": min(dense_lens),
        "dense_len_max": max(dense_lens),
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
    parser.add_argument("--prune-log-eps", default="-12,-10")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    ext = load_quanpc_extension(verbose=False)
    if not hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn"):
        raise RuntimeError("Pruned CUDA extension entry point is missing; rebuild the extension.")

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

    cases = capture.layers
    if not cases:
        raise RuntimeError("No decode layers were captured.")

    summary = summarize_cases(cases)
    print("captured:", json.dumps(summary, sort_keys=True))

    refs = collect_reference_outputs(ext, cases, args.exp_bits)

    def baseline_tail_all():
        for case in cases:
            run_tail(ext, case, args.exp_bits, False, 0, 0.0)

    def baseline_full_all():
        for case in cases:
            run_full(ext, case, args.exp_bits, False, 0, 0.0)

    baseline_tail_ms = time_cuda_workload(baseline_tail_all, cases, args.warmup, args.iters)
    baseline_full_ms = time_cuda_workload(baseline_full_all, cases, args.warmup, args.iters)
    print(f"baseline tail_ms={baseline_tail_ms:.4f} full_ms={baseline_full_ms:.4f}")

    mode_names = {1: "block", 2: "prefix", 3: "combined"}
    results = []
    for log_eps in parse_float_list(args.prune_log_eps):
        for mode in (1, 2, 3):
            max_abs, mean_abs = compare_to_refs(ext, cases, refs, args.exp_bits, mode, log_eps)

            def pruned_tail_all(mode=mode, log_eps=log_eps):
                for case in cases:
                    run_tail(ext, case, args.exp_bits, True, mode, log_eps)

            def pruned_full_all(mode=mode, log_eps=log_eps):
                for case in cases:
                    run_full(ext, case, args.exp_bits, True, mode, log_eps)

            tail_ms = time_cuda_workload(pruned_tail_all, cases, args.warmup, args.iters)
            full_ms = time_cuda_workload(pruned_full_all, cases, args.warmup, args.iters)
            row = {
                "mode": mode_names[mode],
                "prune_mode": mode,
                "log_eps": log_eps,
                "tail_ms": tail_ms,
                "tail_speedup_vs_baseline": baseline_tail_ms / tail_ms,
                "full_ms": full_ms,
                "full_speedup_vs_baseline": baseline_full_ms / full_ms,
                "max_abs_vs_baseline_tail": max_abs,
                "mean_abs_vs_baseline_tail": mean_abs,
            }
            results.append(row)
            print(
                "result "
                f"mode={row['mode']} log_eps={log_eps:g} "
                f"tail={tail_ms:.4f}ms speedup={row['tail_speedup_vs_baseline']:.3f}x "
                f"full={full_ms:.4f}ms speedup={row['full_speedup_vs_baseline']:.3f}x "
                f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
            )

    if args.output_json:
        payload = {
            "model": args.model,
            "requested_length": args.length,
            "actual_input_tokens": int(model_inputs.input_ids.shape[-1]),
            "config": {
                "k_bits": args.k_bits,
                "v_bits": args.v_bits,
                "projected_query_bits": args.projected_query_bits,
                "exp_bits": args.exp_bits,
                "eas": args.eas,
                "zerorope": args.zerorope,
                "warmup": args.warmup,
                "iters": args.iters,
            },
            "captured": summary,
            "baseline": {
                "tail_ms": baseline_tail_ms,
                "full_ms": baseline_full_ms,
            },
            "results": results,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
