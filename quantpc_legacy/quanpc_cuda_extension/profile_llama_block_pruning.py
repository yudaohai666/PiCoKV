"""Profile 64-token block pruning on a real Llama-3.1 decode step.

The synthetic profile scripts are useful for validating formulas, but the score
distribution that matters is produced by the real model and real QuanPC cache.
This script:

1. Builds a NIAH prompt at the requested length.
2. Runs one-token greedy generation with ``QuantKVCache``.
3. Hooks QuanPC global attention during the decode token.
4. Computes 64-token group-max pruning statistics from the real projected query
   and packed binary key cache for every layer.

It leaves the model path unchanged; the hook only records statistics, then calls
the original attention implementation.
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


@dataclass
class EpsStats:
    groups: int = 0
    skipped_groups: int = 0
    tokens: int = 0
    small_tokens: int = 0
    skipped_den: float = 0.0
    global_den: float = 0.0
    final_den: float = 0.0
    max_row_final_frac: float = 0.0

    def merge(self, other: "EpsStats") -> None:
        self.groups += other.groups
        self.skipped_groups += other.skipped_groups
        self.tokens += other.tokens
        self.small_tokens += other.small_tokens
        self.skipped_den += other.skipped_den
        self.global_den += other.global_den
        self.final_den += other.final_den
        self.max_row_final_frac = max(self.max_row_final_frac, other.max_row_final_frac)

    def as_dict(self) -> dict[str, float]:
        return {
            "groups_skip": self.skipped_groups / self.groups if self.groups else 0.0,
            "token_small": self.small_tokens / self.tokens if self.tokens else 0.0,
            "skip_global_total": self.skipped_den / self.global_den if self.global_den > 0 else 0.0,
            "skip_final_total": self.skipped_den / self.final_den if self.final_den > 0 else 0.0,
            "skip_final_max_row": self.max_row_final_frac,
        }


@dataclass
class LayerStats:
    layer_idx: int
    global_len: int
    dense_len: int
    per_eps: dict[float, EpsStats] = field(default_factory=dict)


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


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def unpack_packed_key_chunk(packed_key: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Unpack packed key [B, KVH, C//64, L] into int8 codes [B, KVH, Lc, C]."""
    words = packed_key[:, :, :, start:end].permute(0, 1, 3, 2).contiguous()
    shifts = torch.arange(64, device=packed_key.device, dtype=torch.long)
    bits = ((words.unsqueeze(-1) >> shifts) & 1).to(torch.int8)
    return (1 - 2 * bits).reshape(words.shape[0], words.shape[1], end - start, -1).contiguous()


class LlamaBlockPruneProfiler:
    def __init__(self, log_eps: list[float], block_size: int, chunk_size: int):
        self.log_eps = log_eps
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.layers: list[LayerStats] = []
        self.call_idx = 0

    @staticmethod
    def _dot_chunk(q_int: torch.Tensor, key_chunk: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bhsc,bhlc->bhsl", q_int.float(), key_chunk.float())

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

        dense_key = key_states.dense.repeat_interleave(repeat_factor, dim=1)
        dense_scores = torch.matmul(query_states, dense_key.transpose(-1, -2)) * attn_scale
        dense_lse = torch.logsumexp(dense_scores.float(), dim=-1).contiguous()

        per_eps = {eps: EpsStats() for eps in self.log_eps}
        row_den = torch.zeros(B, QH, S, device=query_states.device, dtype=torch.float32)
        row_skip_den = {eps: torch.zeros_like(row_den) for eps in self.log_eps}

        for start in range(0, global_len, self.chunk_size):
            end = min(start + self.chunk_size, global_len)
            key_chunk = unpack_packed_key_chunk(packed_key, start, end)
            key_chunk = key_chunk.repeat_interleave(repeat_factor, dim=1)
            full_dot = self._dot_chunk(q_int, key_chunk)
            scores = full_dot * query_scale[..., None] * attn_scale
            weights = torch.exp(scores - dense_lse[..., None])
            row_den += weights.sum(dim=-1)

            valid_len = end - start
            pad = (-valid_len) % self.block_size
            if pad:
                scores = torch.nn.functional.pad(scores, (0, pad), value=-float("inf"))
                weights = torch.nn.functional.pad(weights, (0, pad), value=0.0)
            group_count = scores.shape[-1] // self.block_size
            scores_grouped = scores.view(B, QH, S, group_count, self.block_size)
            weights_grouped = weights.view(B, QH, S, group_count, self.block_size)
            group_max_score = scores_grouped.amax(dim=-1)
            group_den = weights_grouped.sum(dim=-1)

            for log_eps in self.log_eps:
                eps = math.exp(log_eps)
                threshold = dense_lse[..., None] + log_eps
                prune = group_max_score <= threshold
                stats = per_eps[log_eps]
                stats.groups += prune.numel()
                stats.skipped_groups += int(prune.sum().item())
                stats.tokens += valid_len * B * QH * S
                stats.small_tokens += int((weights[..., :valid_len] <= eps).sum().item())
                row_skip_den[log_eps] += torch.where(prune, group_den, torch.zeros_like(group_den)).sum(dim=-1)

            del key_chunk, full_dot, scores, weights

        final_den = row_den + 1.0
        for log_eps in self.log_eps:
            stats = per_eps[log_eps]
            skipped = row_skip_den[log_eps]
            row_frac = skipped / final_den
            stats.skipped_den = float(skipped.sum().item())
            stats.global_den = float(row_den.sum().item())
            stats.final_den = float(final_den.sum().item())
            stats.max_row_final_frac = float(row_frac.max().item())

        self.layers.append(LayerStats(layer_idx, global_len, dense_len, per_eps))

    def print_summary(self, detail_eps: list[float]) -> None:
        aggregate = {eps: EpsStats() for eps in self.log_eps}
        for layer in self.layers:
            for eps, stats in layer.per_eps.items():
                aggregate[eps].merge(stats)

        print("\nAGGREGATE")
        print("     log_eps  groups_skip  token_small  skip_global_total  skip_final_total  skip_final_max_row")
        for eps in self.log_eps:
            row = aggregate[eps].as_dict()
            print(
                f"{eps:12g}  "
                f"{row['groups_skip']:8.4%}  "
                f"{row['token_small']:8.4%}  "
                f"{row['skip_global_total']:8.4%}  "
                f"{row['skip_final_total']:8.4%}  "
                f"{row['skip_final_max_row']:8.4%}"
            )

        if not detail_eps:
            return
        print("\nPER_LAYER")
        print(" layer  global_len  dense_len  log_eps  groups_skip  skip_final_total  skip_final_max_row")
        detail_set = set(detail_eps)
        for layer in self.layers:
            for eps in self.log_eps:
                if eps not in detail_set:
                    continue
                row = layer.per_eps[eps].as_dict()
                print(
                    f"{layer.layer_idx:6d}  {layer.global_len:10d}  {layer.dense_len:9d}  {eps:7g}  "
                    f"{row['groups_skip']:8.4%}  {row['skip_final_total']:8.4%}  {row['skip_final_max_row']:8.4%}"
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
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--log-eps", default="-4,-5,-5.5,-6,-6.5,-7,-8,-10,-12")
    parser.add_argument("--detail-log-eps", default="-5.5,-6")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    log_eps = parse_float_list(args.log_eps)
    detail_eps = parse_float_list(args.detail_log_eps)
    profiler = LlamaBlockPruneProfiler(log_eps, args.block_size, args.chunk_size)

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
        payload = {
            "model": args.model,
            "requested_length": args.length,
            "actual_input_tokens": int(model_inputs.input_ids.shape[-1]),
            "block_size": args.block_size,
            "log_eps": log_eps,
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "global_len": layer.global_len,
                    "dense_len": layer.dense_len,
                    "per_eps": {str(eps): stats.as_dict() for eps, stats in layer.per_eps.items()},
                }
                for layer in profiler.layers
            ],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
