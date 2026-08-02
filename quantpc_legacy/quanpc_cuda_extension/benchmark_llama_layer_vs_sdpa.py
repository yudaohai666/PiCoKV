"""Compare one real Llama layer QuanPC decode attention against torch SDPA.

By default this benchmarks model layer 1, i.e. the second transformer layer.  It
captures the layer's real prefill K/V before QuanPC compression, captures the
real one-token decode query and QuanPC cache, then times only this layer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
from quantpc.quanpc_cuda_extension.benchmark_llama_pruned_real_qkv import LayerCase, PromptProvider, run_full, run_tail
from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension


def sync_tensor_device(tensor: torch.Tensor) -> None:
    torch.cuda.synchronize(tensor.device)


def time_cuda_event(fn: Callable[[], object], device: torch.device, warmup: int, iters: int) -> float:
    with torch.no_grad(), torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            fn()
        end.record(stream)
        torch.cuda.synchronize(device)
    return start.elapsed_time(end) / max(1, iters)


def make_layer_case(
    layer_idx: int,
    query_states: torch.Tensor,
    key_states: MixedCacheTensor,
    value_states: MixedCacheTensor,
) -> LayerCase:
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
    return LayerCase(
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
        packed_query=torch.empty(B, QH, S, Ck // 64, 8, device=query_states.device, dtype=torch.long),
        query_scale=torch.empty(B, QH, S, device=query_states.device, dtype=torch.float32),
        dense_output=dense_output.detach().contiguous(),
        dense_lse=dense_lse.detach().contiguous(),
        glob_code=torch.empty(B, QH, S, Cv, device=query_states.device, dtype=torch.float32),
        glob_den=torch.empty(B, QH, S, device=query_states.device, dtype=torch.float32),
        attn_output=torch.empty_like(query_states),
        attention_scale=attention_scale,
        global_len=int(key_states.glob.packed_cuda.shape[-1]),
        dense_len=int(key_states.dense.shape[-2]),
    )


def build_prompt(tokenizer, length: int, depth: int):
    provider = PromptProvider(tokenizer)
    tester = LLMNeedleHaystackTester(provider)
    context = tester.generate_context(length, depth)
    prompt = provider.generate_prompt(context, tester.retrieval_question)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return tokenizer([prompt_text], return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--layer-idx", type=int, default=1, help="0-based model layer index; 1 is the second layer.")
    parser.add_argument("--k-bits", type=int, default=256)
    parser.add_argument("--v-bits", type=int, default=256)
    parser.add_argument("--projected-query-bits", type=int, default=8)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--prune-mode", type=int, default=3)
    parser.add_argument("--prune-log-eps", type=float, default=-10.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    ext = load_quanpc_extension(verbose=False)
    if not hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn"):
        raise RuntimeError("Pruned CUDA extension entry point is missing; rebuild the extension.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_inputs = build_prompt(tokenizer, args.length, args.depth)

    print(f"device_count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"device[{idx}]: {torch.cuda.get_device_name(idx)}")
    print(f"model: {args.model}")
    print(f"target_layer_idx={args.layer_idx}")
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

    captures: dict[str, torch.Tensor | LayerCase] = {}
    original_attention = quanpc.quantized_cache_attention_forward

    past_key_values = QuantKVCache(
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        projected_query_bits=args.projected_query_bits,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        use_cuda_v1_exp8=True,
    )
    original_update = past_key_values.update

    def update_capture(keys, vals, layer_idx, cache_kwargs):
        if layer_idx == args.layer_idx:
            if keys.shape[-2] > 1:
                captures["prefill_key"] = keys.detach().contiguous()
                captures["prefill_value"] = vals.detach().contiguous()
            else:
                captures["decode_key"] = keys.detach().contiguous()
                captures["decode_value"] = vals.detach().contiguous()
        return original_update(keys, vals, layer_idx, cache_kwargs)

    past_key_values.update = update_capture
    decode_call_idx = 0

    def capturing_attention(query_states, key_states, value_states):
        nonlocal decode_call_idx
        if isinstance(key_states, MixedCacheTensor) and isinstance(value_states, MixedCacheTensor):
            if decode_call_idx == args.layer_idx:
                captures["case"] = make_layer_case(args.layer_idx, query_states, key_states, value_states)
            decode_call_idx += 1
        return original_attention(query_states, key_states, value_states)

    quanpc.quantized_cache_attention_forward = capturing_attention
    try:
        with torch.no_grad():
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
    finally:
        quanpc.quantized_cache_attention_forward = original_attention

    required = ["prefill_key", "prefill_value", "decode_key", "decode_value", "case"]
    missing = [name for name in required if name not in captures]
    if missing:
        raise RuntimeError(f"Missing captures for target layer {args.layer_idx}: {missing}")

    case = captures["case"]
    assert isinstance(case, LayerCase)
    full_key = torch.cat([captures["prefill_key"], captures["decode_key"]], dim=-2).contiguous()
    full_value = torch.cat([captures["prefill_value"], captures["decode_value"]], dim=-2).contiguous()
    device = case.query.device
    sync_tensor_device(case.query)

    def sdpa_full():
        return torch.nn.functional.scaled_dot_product_attention(
            case.query,
            full_key,
            full_value,
            is_causal=False,
            enable_gqa=True,
        )

    def base_tail():
        return run_tail(ext, case, args.exp_bits, False, 0, 0.0)

    def pruned_tail():
        return run_tail(ext, case, args.exp_bits, True, args.prune_mode, args.prune_log_eps)

    def base_full():
        return run_full(ext, case, args.exp_bits, False, 0, 0.0)

    def pruned_full():
        return run_full(ext, case, args.exp_bits, True, args.prune_mode, args.prune_log_eps)

    for fn in (sdpa_full, base_tail, pruned_tail, base_full, pruned_full):
        fn()
    sync_tensor_device(case.query)

    timings = {
        "sdpa_full_ms": time_cuda_event(sdpa_full, device, args.warmup, args.iters),
        "quanpc_base_tail_ms": time_cuda_event(base_tail, device, args.warmup, args.iters),
        "quanpc_pruned_tail_ms": time_cuda_event(pruned_tail, device, args.warmup, args.iters),
        "quanpc_base_full_ms": time_cuda_event(base_full, device, args.warmup, args.iters),
        "quanpc_pruned_full_ms": time_cuda_event(pruned_full, device, args.warmup, args.iters),
    }
    timings["base_full_speedup_vs_sdpa"] = timings["sdpa_full_ms"] / timings["quanpc_base_full_ms"]
    timings["pruned_full_speedup_vs_sdpa"] = timings["sdpa_full_ms"] / timings["quanpc_pruned_full_ms"]
    timings["base_tail_speedup_vs_sdpa"] = timings["sdpa_full_ms"] / timings["quanpc_base_tail_ms"]
    timings["pruned_tail_speedup_vs_sdpa"] = timings["sdpa_full_ms"] / timings["quanpc_pruned_tail_ms"]

    shape = {
        "query": list(case.query.shape),
        "sdpa_key": list(full_key.shape),
        "sdpa_value": list(full_value.shape),
        "quanpc_dense_key": list(case.dense_key.shape),
        "quanpc_packed_key": list(case.packed_key.shape),
        "quanpc_packed_value": list(case.packed_value.shape),
        "global_len": case.global_len,
        "dense_len": case.dense_len,
        "full_len": full_key.shape[-2],
        "device": str(device),
    }
    print("shape", json.dumps(shape, sort_keys=True))
    print("timing", json.dumps(timings, sort_keys=True))

    if args.output_json:
        payload = {
            "model": args.model,
            "requested_length": args.length,
            "actual_input_tokens": int(model_inputs.input_ids.shape[-1]),
            "target_layer_idx": args.layer_idx,
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
            "shape": shape,
            "timing": timings,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
