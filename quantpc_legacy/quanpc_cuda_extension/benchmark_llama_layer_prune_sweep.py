"""Sweep pruning modes and thresholds for one real Llama layer.

Captures one real Llama-3.1 QuanPC decode layer, then benchmarks multiple
pruning configurations against the no-pruning QuanPC CUDA path and torch SDPA.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

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


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(',') if part]


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(',') if part]


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


def make_layer_case(layer_idx: int, query_states: torch.Tensor, key_states: MixedCacheTensor, value_states: MixedCacheTensor) -> LayerCase:
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
    return tokenizer([prompt_text], return_tensors='pt')


def capture_layer(args):
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
        device_map='auto',
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
                captures['prefill_key'] = keys.detach().contiguous()
                captures['prefill_value'] = vals.detach().contiguous()
            else:
                captures['decode_key'] = keys.detach().contiguous()
                captures['decode_value'] = vals.detach().contiguous()
        return original_update(keys, vals, layer_idx, cache_kwargs)

    past_key_values.update = update_capture
    decode_call_idx = 0

    def capturing_attention(query_states, key_states, value_states):
        nonlocal decode_call_idx
        if isinstance(key_states, MixedCacheTensor) and isinstance(value_states, MixedCacheTensor):
            if decode_call_idx == args.layer_idx:
                captures['case'] = make_layer_case(args.layer_idx, query_states, key_states, value_states)
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

    missing = [name for name in ('prefill_key', 'prefill_value', 'decode_key', 'decode_value', 'case') if name not in captures]
    if missing:
        raise RuntimeError(f"Missing captures for layer {args.layer_idx}: {missing}")

    case = captures['case']
    assert isinstance(case, LayerCase)
    full_key = torch.cat([captures['prefill_key'], captures['decode_key']], dim=-2).contiguous()
    full_value = torch.cat([captures['prefill_value'], captures['decode_value']], dim=-2).contiguous()
    return case, full_key, full_value, int(model_inputs.input_ids.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/Llama-3.1-8B-Instruct')
    parser.add_argument('--length', type=int, default=100000)
    parser.add_argument('--depth', type=int, default=50)
    parser.add_argument('--layer-idx', type=int, default=1)
    parser.add_argument('--k-bits', type=int, default=256)
    parser.add_argument('--v-bits', type=int, default=256)
    parser.add_argument('--projected-query-bits', type=int, default=8)
    parser.add_argument('--exp-bits', type=int, default=4)
    parser.add_argument('--eas', type=float, default=0.01)
    parser.add_argument('--zerorope', type=int, default=20)
    parser.add_argument('--modes', default='1,2,3')
    parser.add_argument('--log-eps', default='-12,-11,-10,-9,-8')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    ext = load_quanpc_extension(verbose=False)
    if not hasattr(ext, 'quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn'):
        raise RuntimeError('Pruned CUDA extension entry point is missing; rebuild the extension.')

    modes = parse_int_list(args.modes)
    log_eps_values = parse_float_list(args.log_eps)
    mode_names = {1: 'block', 2: 'prefix', 3: 'combined'}

    case, full_key, full_value, actual_tokens = capture_layer(args)
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

    def base_full():
        return run_full(ext, case, args.exp_bits, False, 0, 0.0)

    # Prime all major paths and keep no-pruning references for error checks.
    sdpa_full()
    ref_tail = base_tail().detach().clone()
    ref_full = base_full().detach().clone()
    sync_tensor_device(case.query)

    sdpa_ms = time_cuda_event(sdpa_full, device, args.warmup, args.iters)
    base_tail_ms = time_cuda_event(base_tail, device, args.warmup, args.iters)
    base_full_ms = time_cuda_event(base_full, device, args.warmup, args.iters)

    rows = []
    for log_eps in log_eps_values:
        for mode in modes:
            def pruned_tail(mode=mode, log_eps=log_eps):
                return run_tail(ext, case, args.exp_bits, True, mode, log_eps)

            def pruned_full(mode=mode, log_eps=log_eps):
                return run_full(ext, case, args.exp_bits, True, mode, log_eps)

            out_tail = pruned_tail().detach().clone()
            out_full = pruned_full().detach().clone()
            sync_tensor_device(case.query)
            tail_diff = (out_tail.float() - ref_tail.float()).abs()
            full_diff = (out_full.float() - ref_full.float()).abs()
            tail_ms = time_cuda_event(pruned_tail, device, args.warmup, args.iters)
            full_ms = time_cuda_event(pruned_full, device, args.warmup, args.iters)
            row = {
                'mode': mode_names.get(mode, str(mode)),
                'prune_mode': mode,
                'log_eps': log_eps,
                'tail_ms': tail_ms,
                'full_ms': full_ms,
                'tail_speedup_vs_base': base_tail_ms / tail_ms,
                'full_speedup_vs_base': base_full_ms / full_ms,
                'tail_speedup_vs_sdpa': sdpa_ms / tail_ms,
                'full_speedup_vs_sdpa': sdpa_ms / full_ms,
                'tail_max_abs_vs_base': float(tail_diff.max().item()),
                'tail_mean_abs_vs_base': float(tail_diff.mean().item()),
                'full_max_abs_vs_base': float(full_diff.max().item()),
                'full_mean_abs_vs_base': float(full_diff.mean().item()),
            }
            rows.append(row)
            print(
                'result '
                f"mode={row['mode']} log_eps={log_eps:g} "
                f"tail={tail_ms:.6f}ms full={full_ms:.6f}ms "
                f"full_vs_sdpa={row['full_speedup_vs_sdpa']:.3f}x "
                f"tail_vs_base={row['tail_speedup_vs_base']:.3f}x "
                f"max_abs={row['full_max_abs_vs_base']:.6g} mean_abs={row['full_mean_abs_vs_base']:.6g}"
            )
            sys.stdout.flush()

    shape = {
        'query': list(case.query.shape),
        'sdpa_key': list(full_key.shape),
        'sdpa_value': list(full_value.shape),
        'quanpc_dense_key': list(case.dense_key.shape),
        'quanpc_packed_key': list(case.packed_key.shape),
        'quanpc_packed_value': list(case.packed_value.shape),
        'global_len': case.global_len,
        'dense_len': case.dense_len,
        'full_len': full_key.shape[-2],
        'device': str(device),
    }
    baseline = {
        'sdpa_full_ms': sdpa_ms,
        'quanpc_base_tail_ms': base_tail_ms,
        'quanpc_base_full_ms': base_full_ms,
        'base_tail_speedup_vs_sdpa': sdpa_ms / base_tail_ms,
        'base_full_speedup_vs_sdpa': sdpa_ms / base_full_ms,
    }
    print('shape', json.dumps(shape, sort_keys=True))
    print('baseline', json.dumps(baseline, sort_keys=True))

    if args.output_json:
        payload = {
            'model': args.model,
            'requested_length': args.length,
            'actual_input_tokens': actual_tokens,
            'target_layer_idx': args.layer_idx,
            'config': {
                'k_bits': args.k_bits,
                'v_bits': args.v_bits,
                'projected_query_bits': args.projected_query_bits,
                'exp_bits': args.exp_bits,
                'eas': args.eas,
                'zerorope': args.zerorope,
                'warmup': args.warmup,
                'iters': args.iters,
            },
            'shape': shape,
            'baseline': baseline,
            'results': rows,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
