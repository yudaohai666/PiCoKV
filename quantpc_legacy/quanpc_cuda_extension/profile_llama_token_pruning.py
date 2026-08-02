"""Profile single-token score pruning on a real Llama-3.1 decode step.

This is the real-model counterpart of ``profile_score_pruning.py``.  It builds a
NIAH prompt, runs one QuanPC prefill, then manually decodes one token and hooks
the decode attention call.  For every layer it profiles:

1. Prefix code dimensions, e.g. first 64/128/192 of Ck=256.
2. High magnitude bits of the int8 projected query.

Both modes use the safe upper bound:

    partial_dot + remaining_abs_bound <= threshold_dot

where:

    threshold_dot = (dense_lse + log_eps) / (query_scale * attention_scale)

The script also reports a heuristic-only version and its false-skip rate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc import quanpc
from benchmarks.Needle.needle import LLMNeedleHaystackTester, ModelProvider
from quantpc.quanpc import MixedCacheTensor, QuantKVCache, modify
from profile_llama_block_pruning import unpack_packed_key_chunk


@dataclass
class Counts:
    exact_small: int = 0
    safe_skip: int = 0
    heuristic_skip: int = 0
    heuristic_false_skip: int = 0
    total: int = 0

    def update(self, full_small: torch.Tensor, safe_skip: torch.Tensor, heuristic_skip: torch.Tensor) -> None:
        false_skip = heuristic_skip & ~full_small
        self.exact_small += int(full_small.sum().item())
        self.safe_skip += int(safe_skip.sum().item())
        self.heuristic_skip += int(heuristic_skip.sum().item())
        self.heuristic_false_skip += int(false_skip.sum().item())
        self.total += full_small.numel()

    def merge(self, other: "Counts") -> None:
        self.exact_small += other.exact_small
        self.safe_skip += other.safe_skip
        self.heuristic_skip += other.heuristic_skip
        self.heuristic_false_skip += other.heuristic_false_skip
        self.total += other.total

    def as_dict(self) -> dict[str, float]:
        return {
            "exact_small": self.exact_small / self.total if self.total else 0.0,
            "safe_skip": self.safe_skip / self.total if self.total else 0.0,
            "capture": self.safe_skip / self.exact_small if self.exact_small else 0.0,
            "heuristic_skip": self.heuristic_skip / self.total if self.total else 0.0,
            "heuristic_false": self.heuristic_false_skip / self.heuristic_skip if self.heuristic_skip else 0.0,
        }


@dataclass
class LayerStats:
    layer_idx: int
    global_len: int
    dense_len: int
    stats: dict[str, dict[float, Counts]] = field(default_factory=dict)


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


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _dot_chunk(q_int: torch.Tensor, key_chunk: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhsc,bhlc->bhsl", q_int.float(), key_chunk.float())


class LlamaTokenPruneProfiler:
    def __init__(
        self,
        log_eps: list[float],
        prefix_dims: list[int],
        high_bits: list[int],
        chunk_size: int,
    ):
        self.log_eps = log_eps
        self.prefix_dims = prefix_dims
        self.high_bits = high_bits
        self.chunk_size = chunk_size
        self.layers: list[LayerStats] = []
        self.call_idx = 0

    def profile_layer(self, query_states: torch.Tensor, key_states: MixedCacheTensor) -> None:
        if query_states.shape[-2] != 1 or key_states.glob.packed_cuda is None:
            return

        layer_idx = self.call_idx
        self.call_idx += 1

        B, QH, S, head_dim = query_states.shape
        KVH = key_states.dense.shape[1]
        repeat_factor = QH // KVH
        packed_key = key_states.glob.packed_cuda
        global_len = packed_key.shape[-1]
        dense_len = key_states.dense.shape[-2]
        attn_scale = 1.0 / math.sqrt(head_dim)

        projected_query = quanpc._project_query_gqa(query_states, key_states.glob.proj, key_states.glob.proj_t)
        q_int, query_scale = quanpc._quantize_projected_query_int8(projected_query)
        q_int = q_int.to(torch.int16).contiguous()
        q_abs = q_int.abs()
        q_sign = torch.sign(q_int).to(torch.int16)

        dense_key = key_states.dense.repeat_interleave(repeat_factor, dim=1)
        dense_scores = torch.matmul(query_states, dense_key.transpose(-1, -2)) * attn_scale
        dense_lse = torch.logsumexp(dense_scores.float(), dim=-1).contiguous()
        threshold_dot = {
            eps: ((dense_lse + eps) / (query_scale.clamp_min(1.0e-30) * attn_scale)).unsqueeze(-1)
            for eps in self.log_eps
        }

        prefix_dims = [dim for dim in self.prefix_dims if 0 < dim < q_int.shape[-1]]
        high_bits = [bits for bits in self.high_bits if 0 < bits <= 7]
        layer_stats: dict[str, dict[float, Counts]] = {}
        for dim in prefix_dims:
            layer_stats[f"prefix:{dim}"] = {eps: Counts() for eps in self.log_eps}
        for bits in high_bits:
            layer_stats[f"high_bits:{bits}"] = {eps: Counts() for eps in self.log_eps}

        prefix_remaining = {dim: q_abs[..., dim:].sum(dim=-1).float() for dim in prefix_dims}
        high_q = {}
        high_remaining = {}
        for bits in high_bits:
            low_bits = 7 - bits
            high_abs = ((q_abs >> low_bits) << low_bits).to(torch.int16)
            high_q[bits] = (q_sign * high_abs).contiguous()
            high_remaining[bits] = (q_abs - high_abs).sum(dim=-1).float()

        for start in range(0, global_len, self.chunk_size):
            end = min(start + self.chunk_size, global_len)
            key_chunk = unpack_packed_key_chunk(packed_key, start, end).repeat_interleave(repeat_factor, dim=1)
            full_dot = _dot_chunk(q_int, key_chunk)
            full_small = {eps: full_dot <= threshold_dot[eps] for eps in self.log_eps}

            for dim in prefix_dims:
                partial = _dot_chunk(q_int[..., :dim], key_chunk[..., :dim])
                rem = prefix_remaining[dim][..., None]
                by_eps = layer_stats[f"prefix:{dim}"]
                for eps in self.log_eps:
                    safe = partial + rem <= threshold_dot[eps]
                    heuristic = partial <= threshold_dot[eps]
                    by_eps[eps].update(full_small[eps], safe, heuristic)

            for bits in high_bits:
                partial = _dot_chunk(high_q[bits], key_chunk)
                rem = high_remaining[bits][..., None]
                by_eps = layer_stats[f"high_bits:{bits}"]
                for eps in self.log_eps:
                    safe = partial + rem <= threshold_dot[eps]
                    heuristic = partial <= threshold_dot[eps]
                    by_eps[eps].update(full_small[eps], safe, heuristic)

            del key_chunk, full_dot

        self.layers.append(LayerStats(layer_idx, global_len, dense_len, layer_stats))

    def aggregate(self) -> dict[str, dict[float, Counts]]:
        merged: dict[str, dict[float, Counts]] = {}
        for layer in self.layers:
            for label, by_eps in layer.stats.items():
                merged.setdefault(label, {eps: Counts() for eps in self.log_eps})
                for eps, counts in by_eps.items():
                    merged[label][eps].merge(counts)
        return merged

    @staticmethod
    def _print_counts(label: str, eps: float, counts: Counts) -> None:
        row = counts.as_dict()
        print(
            f"{label:>14}  {eps:8g}  "
            f"{row['exact_small']:8.4%}  "
            f"{row['safe_skip']:8.4%}  "
            f"{row['capture']:8.4%}  "
            f"{row['heuristic_skip']:8.4%}  "
            f"{row['heuristic_false']:8.4%}"
        )

    def print_summary(self, detail_eps: list[float]) -> None:
        aggregate = self.aggregate()
        print("\nAGGREGATE")
        print("         method   log_eps     exact      safe   capture      heur  heur_false")
        for eps in self.log_eps:
            for label in sorted(aggregate):
                self._print_counts(label, eps, aggregate[label][eps])

        if not detail_eps:
            return
        detail_set = set(detail_eps)
        print("\nPER_LAYER")
        print(" layer  global_len  dense_len          method   log_eps     exact      safe   capture  heur_false")
        for layer in self.layers:
            for label in sorted(layer.stats):
                for eps in self.log_eps:
                    if eps not in detail_set:
                        continue
                    row = layer.stats[label][eps].as_dict()
                    print(
                        f"{layer.layer_idx:6d}  {layer.global_len:10d}  {layer.dense_len:9d}  "
                        f"{label:>14}  {eps:8g}  "
                        f"{row['exact_small']:8.4%}  "
                        f"{row['safe_skip']:8.4%}  "
                        f"{row['capture']:8.4%}  "
                        f"{row['heuristic_false']:8.4%}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--k-bits", type=int, default=256)
    parser.add_argument("--v-bits", type=int, default=256)
    parser.add_argument("--projected-query-bits", type=int, default=8)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--prefix-dims", default="64,128,192")
    parser.add_argument("--high-bits", default="4,5,6")
    parser.add_argument("--log-eps", default="-5,-5.5,-6,-6.5,-7,-8,-10,-12")
    parser.add_argument("--detail-log-eps", default="-6,-8,-10,-12")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    log_eps = parse_float_list(args.log_eps)
    detail_eps = parse_float_list(args.detail_log_eps)
    profiler = LlamaTokenPruneProfiler(
        log_eps=log_eps,
        prefix_dims=parse_int_list(args.prefix_dims),
        high_bits=parse_int_list(args.high_bits),
        chunk_size=args.chunk_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    provider = PromptProvider(tokenizer)
    tester = LLMNeedleHaystackTester(provider)
    context = tester.generate_context(args.length, args.depth)
    prompt = provider.generate_prompt(context, tester.retrieval_question)
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt_text], return_tensors="pt")

    print(f"device: {torch.cuda.get_device_name()}")
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

    original_attention = quanpc.quantized_cache_attention_forward

    def profiled_attention(query_states, key_states, value_states):
        if isinstance(key_states, MixedCacheTensor):
            profiler.profile_layer(query_states, key_states)
        return original_attention(query_states, key_states, value_states)

    quanpc.quantized_cache_attention_forward = profiled_attention
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
            torch.cuda.synchronize()
    finally:
        quanpc.quantized_cache_attention_forward = original_attention

    profiler.print_summary(detail_eps)
    if args.output_json:
        aggregate = profiler.aggregate()
        payload = {
            "model": args.model,
            "requested_length": args.length,
            "actual_input_tokens": int(model_inputs.input_ids.shape[-1]),
            "log_eps": log_eps,
            "aggregate": {
                label: {str(eps): counts.as_dict() for eps, counts in by_eps.items()}
                for label, by_eps in aggregate.items()
            },
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "global_len": layer.global_len,
                    "dense_len": layer.dense_len,
                    "stats": {
                        label: {str(eps): counts.as_dict() for eps, counts in by_eps.items()}
                        for label, by_eps in layer.stats.items()
                    },
                }
                for layer in profiler.layers
            ],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
