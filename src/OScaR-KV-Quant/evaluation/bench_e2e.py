"""End-to-end wall-clock decode benchmark for OScaR, mirroring the methodology of
QuantPC/benchmarks/Efficiency/run_efficiency.py so numbers are directly comparable.

Unlike bench_throughput.py (which times 256 steps in one block, no repeats/median),
this:
  * prefills ONCE (untimed) to populate the KV cache,
  * runs `warmup` untimed decode steps,
  * times `decode_tokens` single-token forwards, repeated `repeats` times, taking the
    MEDIAN per-token latency,
  * reports prefill ms, decode ms/tok, decode tok/s, and peak GB.

Each decode step runs the FULL model forward (RMSNorm/RoPE/linears + attention), so
this is true end-to-end wall-clock — same as run_efficiency.py — not an
attention-isolated ratio.

Drives the model via inputs_embeds + past_key_values threading, exactly like
bench_throughput.py (this is the path OScaR's custom cache supports). Feeding random
embeds vs. argmax-sampled tokens is latency-irrelevant: both push 1 token/step through
the identical stack.

Run ONE backend per process (pass --attn-backend), so memory is fully released between
backends — run oscar and flash_attention_2 in separate invocations on the same GPU.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from transformers import AutoConfig  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_model_cls(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    mt = getattr(config, "model_type", None)
    if mt == "qwen3":
        from qwen3 import Qwen3ForCausalLM
        return config, Qwen3ForCausalLM
    if mt == "llama":
        from llama import LlamaForCausalLM
        return config, LlamaForCausalLM
    raise ValueError(f"Unsupported model_type: {mt}")


def load_model(args, dtype):
    config, model_cls = resolve_model_cls(args.model_path)
    torch.set_default_dtype(dtype)
    config._attn_implementation = "flash_attention_2"
    config.attn_backend = args.attn_backend
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.kv_rotation = args.kv_rotation
    config.kv_norm = args.kv_norm
    config.residual_block_size = 128
    if args.residual_evict_size is not None:
        config.residual_evict_size = args.residual_evict_size
    model = model_cls.from_pretrained(
        args.model_path, config=config,
        device_map={"": args.device}, torch_dtype=dtype,
    )
    model.eval()
    return model


def _decode_step(model, hidden_states, past):
    out = model(inputs_embeds=hidden_states, past_key_values=past,
                use_cache=True, logits_to_keep=1)
    return out.past_key_values


@torch.inference_mode()
def measure(model, hidden_size, dtype, device, prompt_len, decode_tokens, warmup, repeats):
    def rand_prompt():
        return torch.randn(1, prompt_len, hidden_size, dtype=dtype, device=device)

    def rand_tok():
        return torch.randn(1, 1, hidden_size, dtype=dtype, device=device)

    def sync():
        torch.cuda.synchronize(device)

    # ---- Warmup (kernel compile / autotune / allocator growth), not timed. ----
    for _ in range(warmup):
        past = model(inputs_embeds=rand_prompt(), use_cache=True, logits_to_keep=1).past_key_values
        for _ in range(decode_tokens):
            past = _decode_step(model, rand_tok(), past)
        del past
    torch.cuda.empty_cache()

    # ---- Prefill latency (median over repeats). ----
    torch.cuda.reset_peak_memory_stats(device)
    prefill_ms = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        past = model(inputs_embeds=rand_prompt(), use_cache=True, logits_to_keep=1).past_key_values
        sync(); prefill_ms.append((time.perf_counter() - t0) * 1000.0)
        del past
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9

    # ---- Decode: prefill ONCE (untimed), then time decode_tokens single-token steps. ----
    per_token = []
    for _ in range(repeats):
        past = model(inputs_embeds=rand_prompt(), use_cache=True, logits_to_keep=1).past_key_values
        sync(); t0 = time.perf_counter()
        for _ in range(decode_tokens):
            past = _decode_step(model, rand_tok(), past)
        sync(); per_token.append((time.perf_counter() - t0) * 1000.0 / decode_tokens)
        del past
    ms_tok = statistics.median(per_token)
    return {
        "prefill_ms": statistics.median(prefill_ms),
        "decode_ms_per_token": ms_tok,
        "decode_toks_per_s": 1000.0 / ms_tok,
        "peak_gb": peak_gb,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
    p.add_argument("--attn-backend", default="oscar", choices=["oscar", "flash_attention_2", "flash_decoding", "eager"])
    p.add_argument("--lengths", default="131072", help="comma-separated prompt lengths")
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--quant-mode", default="k-channel")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--kv-rotation", default="none")
    p.add_argument("--kv-norm", default="0")
    p.add_argument("--residual-evict-size", type=int, default=None)
    args = p.parse_args()

    seed_everything(42)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    model = load_model(args, dtype)
    hidden_size = model.config.hidden_size

    tag = args.attn_backend
    if args.attn_backend == "oscar":
        tag += f"(int{args.num_bits}/{args.quant_mode}/g{args.group_size})"
    for length in (int(x) for x in args.lengths.split(",")):
        r = measure(model, hidden_size, dtype, device, length,
                    args.decode_tokens, args.warmup, args.repeats)
        print(f"[{tag}] L={length}: prefill={r['prefill_ms']:.1f}ms  "
              f"decode={r['decode_ms_per_token']:.4f}ms/tok "
              f"({r['decode_toks_per_s']:.2f} tok/s)  peak={r['peak_gb']:.3f}GB")


if __name__ == "__main__":
    main()
