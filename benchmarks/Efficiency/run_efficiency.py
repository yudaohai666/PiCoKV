"""Prefill vs decoding efficiency benchmark across KV-cache methods.

For each (method, prompt_length) it measures:
  * prefill latency (time-to-first-token, TTFT) via ``generate(max_new_tokens=1)``
  * decode latency per token, derived from ``generate(max_new_tokens=N)`` minus TTFT
  * decode throughput (tokens/s)
  * peak GPU memory for the prefill phase and for the full generation

The same ``generate()``-based methodology works uniformly across quantpc (a
fresh QuantKVCache injected per call), kivi (internal quantized cache), and the
dense flash_attention_2 / sdpa / eager baselines, so numbers are comparable.

Run one method:
    python run_efficiency.py --method kivi --k-bits 2 --v-bits 2 \
        --lengths 4096,16384,32768 --output results/efficiency/kivi_k2_v2.json

Aggregate a directory of per-method JSONs into a comparison table:
    python run_efficiency.py --compare results/efficiency
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Make `benchmarks.common` / `import quantpc` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import load_quantpc_model, output_tag, seed_everything  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    return [int(p) for p in value.split(",") if p]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0


def build_inputs(vocab_size: int, batch_size: int, prompt_len: int, device) -> torch.Tensor:
    # Content is irrelevant for latency; use random in-range ids (avoid id 0/pad).
    return torch.randint(1, vocab_size, (batch_size, prompt_len), device=device, dtype=torch.long)


def _time_generate(model, input_ids, max_new_tokens, make_cache) -> float:
    """Return wall time (ms) for one greedy generate of exactly ``max_new_tokens``."""
    eos = model.config.eos_token_id
    pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos  # Llama-3.1 eos may be a list
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),  # no padding; silences HF mask warning
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # force exactly N tokens (no early EOS) for stable timing
        do_sample=False,
        use_cache=True,
        pad_token_id=pad_token_id,  # silence "Setting pad_token_id to eos_token_id" warning
    )
    # quantpc needs a fresh QuantKVCache per call; kivi/dense keep make_cache=None.
    pkv = make_cache() if make_cache is not None else None
    if pkv is not None:
        gen_kwargs["past_key_values"] = pkv
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**gen_kwargs)
    _sync()
    elapsed = (time.perf_counter() - t0) * 1000.0
    del pkv
    return elapsed


def _decode_ms_per_token(model, input_ids, decode_tokens, make_cache, repeats) -> float:
    """Median per-token decode latency, timing ONLY the decode steps.

    Prefill once (untimed) to populate the KV cache, then time a loop of
    single-token forwards. This isolates decode from prefill, so it stays
    reliable even when prefill >> decode (where ``total - ttft`` subtraction
    goes negative). Each step goes through ``prepare_inputs_for_generation`` so
    position_ids / cache_position / logits_to_keep match ``generate`` exactly,
    uniformly across quantpc / kivi / dense.

    ``cache_position`` is required by the stock (quantpc/dense) prepare path;
    ``logits_to_keep=1`` avoids materializing full-sequence fp32 logits at
    prefill (a 131072-len prompt would otherwise need ~67GB). KIVI's custom
    prepare ignores both (it already forces logits_to_keep=1).
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    per_token = []
    for _ in range(repeats):
        pkv = make_cache() if make_cache is not None else None
        gen_ids = input_ids
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            # Prefill (NOT timed): build the full-prompt KV cache.
            mi = model.prepare_inputs_for_generation(
                gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
                cache_position=torch.arange(prompt_len, device=device), logits_to_keep=1)
            out = model(**mi)
            pkv = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
            # Timed decode: `decode_tokens` single-token forwards on the cache.
            _sync()
            t0 = time.perf_counter()
            for step in range(decode_tokens):
                pos = prompt_len + step  # absolute position of the token to feed
                mi = model.prepare_inputs_for_generation(
                    gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
                    cache_position=torch.arange(pos, pos + 1, device=device), logits_to_keep=1)
                out = model(**mi)
                pkv = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
                attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
            _sync()
        per_token.append((time.perf_counter() - t0) * 1000.0 / decode_tokens)
        del pkv
    return statistics.median(per_token)


def measure(model, make_cache, vocab_size, batch_size, prompt_len, decode_tokens, warmup, repeats, device) -> dict:
    input_ids = build_inputs(vocab_size, batch_size, prompt_len, device)

    # Warmup (kernel compile / cuBLAS autotune / allocator growth), not timed.
    for _ in range(warmup):
        _time_generate(model, input_ids, decode_tokens, make_cache)

    # Prefill peak memory: reset, then a single prefill (+1 token).
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _time_generate(model, input_ids, 1, make_cache)
    prefill_peak = _peak_gb()

    # Prefill latency (time-to-first-token): median over repeats.
    ttft = [_time_generate(model, input_ids, 1, make_cache) for _ in range(repeats)]
    ttft_ms = statistics.median(ttft)

    # Overall peak memory for a full generate(decode_tokens).
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _time_generate(model, input_ids, decode_tokens, make_cache)
    total_peak = _peak_gb()

    # Decode per-token latency: timed directly (prefill excluded), so it stays
    # reliable even when prefill >> decode.
    decode_ms_per_token = _decode_ms_per_token(model, input_ids, decode_tokens, make_cache, repeats)
    decode_toks_per_s = (1000.0 / decode_ms_per_token * batch_size) if decode_ms_per_token > 0 else float("nan")

    return {
        "prompt_len": prompt_len,
        "prefill_ms": round(ttft_ms, 3),
        "decode_ms_per_token": round(decode_ms_per_token, 4),
        "decode_toks_per_s": round(decode_toks_per_s, 2),
        "prefill_peak_gb": round(prefill_peak, 3),
        "total_peak_gb": round(total_peak, 3),
    }


def run(args) -> None:
    seed_everything(42)
    model, _tokenizer, make_cache = load_quantpc_model(
        args.model,
        method=args.method,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        sink_size=args.sink_size,
        local_size=args.local_size,
        projected_query_bits=args.projected_query_bits,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        cuda_v1_exp8=args.cuda_v1_exp8,
        refit_decoder=args.refit_decoder,
        use_pruning=args.use_pruning,
        prune_mode=args.prune_mode,
        log_eps=args.log_eps,
        group_size=args.group_size,
        residual_length=args.residual_length,
    )
    device = model.device
    vocab_size = model.config.vocab_size

    results = {}
    for length in parse_int_list(args.lengths):
        r = measure(
            model, make_cache, vocab_size, args.batch_size, length,
            args.decode_tokens, args.warmup, args.repeats, device,
        )
        results[str(length)] = r
        print(
            f"[{args.method}] L={length}: prefill={r['prefill_ms']}ms  "
            f"decode={r['decode_ms_per_token']}ms/tok ({r['decode_toks_per_s']} tok/s)  "
            f"peak prefill/total={r['prefill_peak_gb']}/{r['total_peak_gb']}GB"
        )

    tag = output_tag(args.method, args.k_bits, args.v_bits, args.refit_decoder)
    payload = {
        "method": args.method,
        "tag": tag,
        "model": args.model,
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "config": {
            "k_bits": args.k_bits, "v_bits": args.v_bits, "group_size": args.group_size,
            "residual_length": args.residual_length, "exp_bits": args.exp_bits,
            "projected_query_bits": args.projected_query_bits, "eas": args.eas,
            "zerorope": args.zerorope, "cuda_v1_exp8": args.cuda_v1_exp8,
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out}")


def compare(directory: str) -> None:
    d = Path(directory)
    data = []
    for fp in sorted(d.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    if not data:
        print(f"[compare] no result JSONs found under {d}")
        return

    lengths = sorted({int(L) for p in data for L in p["results"]})
    for length in lengths:
        print(f"\n=== prompt_len = {length} ===")
        header = (
            f"{'method':<22}{'prefill_ms':>12}{'dec_ms/tok':>12}"
            f"{'tok/s':>10}{'prefill_GB':>12}{'total_GB':>10}"
        )
        print(header)
        print("-" * len(header))
        for p in data:
            r = p["results"].get(str(length))
            if not r:
                continue
            print(
                f"{p.get('tag', p.get('method', '?')):<22}"
                f"{r['prefill_ms']:>12}{r['decode_ms_per_token']:>12}"
                f"{r['decode_toks_per_s']:>10}{r['prefill_peak_gb']:>12}{r['total_peak_gb']:>10}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill vs decode efficiency benchmark.")
    parser.add_argument("--compare", default=None, help="aggregate all *.json under this dir into a table and exit")
    parser.add_argument("--model", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--method", default="quantpc", help="quantpc | kivi | flash_attention_2 | sdpa | eager")
    parser.add_argument("--lengths", default="4096,16384,32768", help="comma-separated prompt lengths")
    parser.add_argument("--decode-tokens", type=int, default=64, help="new tokens to generate for decode timing")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="results/efficiency/quantpc.json")
    # KV-cache config (shared with the NIAH runner semantics).
    parser.add_argument("--k-bits", type=int, default=256)
    parser.add_argument("--v-bits", type=int, default=256)
    parser.add_argument("--sink-size", type=int, default=64)
    parser.add_argument("--local-size", type=int, default=1024)
    parser.add_argument("--projected-query-bits", type=int, default=8)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--cuda-v1-exp8", action="store_true")
    parser.add_argument("--refit-decoder", action="store_true")
    parser.add_argument("--no-pruning", dest="use_pruning", action="store_false",
                        help="disable decode-time global pruning (gqa4 CUDA path); pruning is on by default")
    parser.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    parser.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    parser.set_defaults(use_pruning=True)
    parser.add_argument("--group-size", type=int, default=32, help="KIVI per-group quant size (method=kivi)")
    parser.add_argument("--residual-length", type=int, default=32, help="KIVI recent fp16 tokens (method=kivi)")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return
    run(args)


if __name__ == "__main__":
    main()
