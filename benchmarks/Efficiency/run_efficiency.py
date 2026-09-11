"""Prefill vs decoding efficiency benchmark across KV-cache methods.

For each (method, prompt_length) it measures:
  * prefill latency (time-to-first-token, TTFT) via ``generate(max_new_tokens=1)``
  * decode latency per token, derived from ``generate(max_new_tokens=N)`` minus TTFT
  * decode throughput (tokens/s)
  * peak GPU memory for the prefill phase and for the full generation

The same ``generate()``-based methodology works uniformly across picokv (a
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
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Make `benchmarks.common` / `import picokv` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import load_model, output_tag, resolve_offline_proj_path, seed_everything  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    return [int(p) for p in value.split(",") if p]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free() -> None:
    """Release freed tensors AND the caching allocator's reserved pool.

    ``measure()`` chains many ``generate``/forward passes; without returning the
    allocator's reserved blocks between them, the pool balloons (e.g. ~70GB on a
    bs48/L4096 fp16 run) and later phases OOM from fragmentation even though the
    per-pass live peak (~60GB) fits. Mirrors benchmark_kitty.py's per-iter
    ``gc.collect()`` hygiene.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return 0.0


def build_inputs(vocab_size: int, batch_size: int, prompt_len: int, device) -> torch.Tensor:
    # Content is irrelevant for latency; use random in-range ids (avoid id 0/pad).
    return torch.randint(1, vocab_size, (batch_size, prompt_len), device=device, dtype=torch.long)


def build_inputs_from_dataset(
    dataset_path: str,
    tokenizer,
    batch_size: int,
    prompt_len: int,
    device,
) -> torch.Tensor:
    """Build a real-text input_ids tensor from a .jsonl dataset.

    Concatenates all ``text`` fields, tokenizes, and slices exactly
    ``prompt_len`` tokens. Falls back to random ids if the dataset does
    not contain enough tokens.
    """
    texts = []
    with open(dataset_path) as f:
        for line in f:
            obj = json.loads(line)
            if "text" in obj:
                texts.append(obj["text"])
    combined = " ".join(texts)
    token_ids = tokenizer.encode(combined, add_special_tokens=False)
    if len(token_ids) < prompt_len:
        # Not enough real text: fall back to random ids.
        return build_inputs(tokenizer.vocab_size, batch_size, prompt_len, device)
    ids = torch.tensor(token_ids[:prompt_len], dtype=torch.long, device=device)
    return ids.unsqueeze(0).expand(batch_size, -1).contiguous()


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
        temperature=None,  # override model's generation_config defaults (silences greedy warning)
        top_p=None,
        use_cache=True,
        pad_token_id=pad_token_id,  # silence "Setting pad_token_id to eos_token_id" warning
    )
    # picokv needs a fresh QuantKVCache per call; kivi/dense keep make_cache=None.
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
    _free()
    return elapsed


def _decode_ms_per_token(model, input_ids, decode_tokens, make_cache, repeats, decode_warmup=2, track_peak: bool = False) -> float:
    """Median per-token STEADY-STATE decode latency, timing ONLY the decode steps.

    Prefill once (untimed) to populate the KV cache, run ``decode_warmup`` untimed
    decode steps to absorb the post-prefill startup transient, then time a loop of
    single-token forwards. This isolates decode from prefill, so it stays
    reliable even when prefill >> decode (where ``total - ttft`` subtraction
    goes negative). Each step goes through ``prepare_inputs_for_generation`` so
    position_ids / cache_position / logits_to_keep match ``generate`` exactly,
    uniformly across picokv / kivi / dense.

    ``track_peak`` resets the peak counter right after the untimed prefill, so
    the caller's ``_peak_gb()`` after this function returns is the decode-phase
    peak (mirrors ``_decode_peak_gb``). Counter ops are host-side and do not
    touch the timed region, so latency numbers are unaffected.

    The first decode step after a long prefill can be 10-50x a steady step (one-off
    workspace allocation / cuBLAS autotune / kernel init, and cudaMalloc under a
    tight caching-allocator pool). Timing it would smear a startup transient into
    the per-token average and mis-rank methods; ``decode_warmup`` steps move it out
    of the timed region so the number reflects steady-state throughput.

    ``cache_position`` is required by the stock (picokv/dense) prepare path;
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
            pos = prompt_len  # absolute position of the next token to feed
            # Reset AFTER the untimed prefill so only decode-step allocations
            # are counted (mirrors _decode_peak_gb); caller reads _peak_gb()
            # after this function returns.
            if track_peak and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            # Warmup decode (NOT timed): absorb the one-time post-prefill startup
            # transient so the timed loop reflects STEADY-STATE decode.
            for _ in range(decode_warmup):
                mi = model.prepare_inputs_for_generation(
                    gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
                    cache_position=torch.arange(pos, pos + 1, device=device), logits_to_keep=1)
                out = model(**mi)
                pkv = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
                attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
                pos += 1
            # Timed decode: `decode_tokens` single-token forwards on the cache.
            _sync()
            t0 = time.perf_counter()
            for _ in range(decode_tokens):
                mi = model.prepare_inputs_for_generation(
                    gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
                    cache_position=torch.arange(pos, pos + 1, device=device), logits_to_keep=1)
                out = model(**mi)
                pkv = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                gen_ids = torch.cat([gen_ids, next_tok], dim=1)
                attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
                pos += 1
            _sync()
        per_token.append((time.perf_counter() - t0) * 1000.0 / decode_tokens)
        del pkv, gen_ids, attn, out, next_tok, mi
        _free()
    return statistics.median(per_token)


def _decode_peak_gb(model, input_ids, decode_tokens, make_cache, device) -> float:
    """Peak GPU memory during the DECODE phase only.

    Prefill once (untimed) to build the KV cache, reset the peak counter, then
    run ``decode_tokens`` single-token forwards. Mirrors ``_decode_ms_per_token``
    but records memory instead of time, so the prefill peak is excluded.
    """
    prompt_len = input_ids.shape[1]
    pkv = make_cache() if make_cache is not None else None
    gen_ids = input_ids
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        # Prefill (excluded from the decode peak).
        mi = model.prepare_inputs_for_generation(
            gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
            cache_position=torch.arange(prompt_len, device=device), logits_to_keep=1)
        out = model(**mi)
        pkv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=1)
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
        # Reset AFTER prefill so only decode-step allocations count.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for step in range(decode_tokens):
            pos = prompt_len + step
            mi = model.prepare_inputs_for_generation(
                gen_ids, past_key_values=pkv, attention_mask=attn, use_cache=True,
                cache_position=torch.arange(pos, pos + 1, device=device), logits_to_keep=1)
            out = model(**mi)
            pkv = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
    peak = _peak_gb()
    del pkv, gen_ids, attn, out, next_tok, mi
    _free()
    return peak


def measure(model, make_cache, vocab_size, batch_size, prompt_len, decode_tokens, warmup, repeats, device, mem_only=False, input_ids: Optional[torch.Tensor] = None) -> dict:
    """Benchmark one prompt length.

    ``mem_only=True``  -> memory-only mode: measure peak GPU memory, skip latency.
    ``mem_only=False`` -> latency + peak GPU memory in the same pass (peaks come
                          from host-side counter reads; no extra generate passes).
    ``input_ids``      -> pre-built real-text token ids; falls back to random if None.
    """
    if input_ids is None:
        input_ids = build_inputs(vocab_size, batch_size, prompt_len, device)

    # Warmup (kernel compile / cuBLAS autotune / allocator growth), not timed.
    for _ in range(warmup):
        _time_generate(model, input_ids, decode_tokens, make_cache)

    prefill_peak = decode_peak = None
    ttft_ms = decode_ms_per_token = decode_toks_per_s = None

    if mem_only:
        # Prefill-phase peak: reset -> single prefill (+1 token).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _time_generate(model, input_ids, 1, make_cache)
        prefill_peak = _peak_gb()

        # Decode-phase peak: prefill (untimed) -> reset -> decode loop.
        decode_peak = _decode_peak_gb(model, input_ids, decode_tokens, make_cache, device)
    else:
        # Latency + peaks in one pass: reset before the TTFT loop so the
        # prefill peak excludes weights/warmup; _decode_ms_per_token resets
        # after its untimed prefill for the decode peak. Both are host-side
        # counter ops — no extra generate passes, latency untouched.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        ttft = [_time_generate(model, input_ids, 1, make_cache) for _ in range(repeats)]
        ttft_ms = statistics.median(ttft)
        prefill_peak = _peak_gb()
        decode_ms_per_token = _decode_ms_per_token(model, input_ids, decode_tokens, make_cache, repeats, track_peak=True)
        decode_peak = _peak_gb()
        decode_toks_per_s = (1000.0 / decode_ms_per_token * batch_size) if decode_ms_per_token > 0 else float("nan")

    return {
        "prompt_len": prompt_len,
        "prefill_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "decode_ms_per_token": round(decode_ms_per_token, 4) if decode_ms_per_token is not None else None,
        "decode_toks_per_s": round(decode_toks_per_s, 2) if decode_toks_per_s is not None else None,
        "prefill_peak_gb": round(prefill_peak, 3) if prefill_peak is not None else None,
        "decode_peak_gb": round(decode_peak, 3) if decode_peak is not None else None,
    }


def run(args) -> None:
    seed_everything(42)
    offline_proj_path = resolve_offline_proj_path(
        args.method, args.offline, args.offline_proj_path,
        model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
    )
    model, tokenizer, make_cache = load_model(
        args.model,
        method=args.method,
        # picokv KV-cache params (--method picokv)
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        sink_size=args.sink_size,
        local_size=args.local_size,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        use_pruning=args.use_pruning,
        prune_mode=args.prune_mode,
        log_eps=args.log_eps,
        offline_proj_path=offline_proj_path,
        # KIVI params (--method kivi)
        group_size=args.group_size,
        residual_length=args.residual_length,
        # OScaR params (--method oscar)
        num_bits=args.num_bits,
        quant_mode=args.quant_mode,
        kv_rotation=args.kv_rotation,
        kv_norm=args.kv_norm,
        residual_evict_size=args.residual_evict_size,
        offline_v_hadamard=args.offline_v_hadamard,
        # PolarQuant params (--method polar)
        polar_mode=args.polar_mode,
        polar_rbits=args.polar_rbits,
        polar_tbits=args.polar_tbits,
        polar_vbits=args.polar_vbits,
    )
    device = model.device
    vocab_size = model.config.vocab_size

    # Pre-tokenize real dataset once (shared across all lengths).
    dataset_input_ids: dict[int, torch.Tensor] = {}
    if args.dataset:
        for length in parse_int_list(args.lengths):
            dataset_input_ids[length] = build_inputs_from_dataset(
                args.dataset, tokenizer, args.batch_size, length, device,
            )

    tag = output_tag(args.method, args.k_bits, args.v_bits, args.num_bits,
                     offline=(args.offline or args.offline_proj_path is not None))
    results = {}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: the length-outer launcher calls this process once per length, so
    # keep results from previous invocations instead of dropping them. An OOM
    # entry for the same length gets re-measured and replaced on a re-run.
    if out.exists():
        try:
            with open(out, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and "results" in prev:
                results = {str(k): v for k, v in prev["results"].items()}
        except (json.JSONDecodeError, OSError):
            pass
    payload = {
        "method": args.method,
        "tag": tag,
        "model": args.model,
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "config": {
            "k_bits": args.k_bits, "v_bits": args.v_bits, "group_size": args.group_size,
            "residual_length": args.residual_length, "exp_bits": args.exp_bits,
            "eas": args.eas, "zerorope": args.zerorope,
        },
        "results": results,
    }

    def _save() -> None:
        # Incremental persistence: every completed length is written immediately,
        # so an OOM / Ctrl-C / preemption mid-run keeps all results so far.
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    for length in parse_int_list(args.lengths):
        real_ids = dataset_input_ids.get(length) if args.dataset else None
        try:
            r = measure(
                model, make_cache, vocab_size, args.batch_size, length,
                args.decode_tokens, args.warmup, args.repeats, device,
                mem_only=args.mem_only,
                input_ids=real_ids,
            )
        except torch.cuda.OutOfMemoryError:
            # Hard-fail used to lose the whole run: the JSON below is only
            # written after the loop, so one OOM at a long length killed every
            # result for the method. Record the OOM (compare() renders None
            # fields as "-") and keep going so the file is still saved.
            _free()
            r = {
                "prompt_len": length,
                "error": "out_of_memory",
                "prefill_ms": None,
                "decode_ms_per_token": None,
                "decode_toks_per_s": None,
                "prefill_peak_gb": None,
                "decode_peak_gb": None,
            }
            print(f"[{args.method}] L={length}: OUT OF MEMORY (recorded, continuing)")
        results[str(length)] = r
        _save()
        if r.get("error"):
            continue
        if args.mem_only:
            print(
                f"[{args.method}] L={length}: "
                f"peak prefill/decode={r['prefill_peak_gb']}/{r['decode_peak_gb']}GB"
            )
        else:
            print(
                f"[{args.method}] L={length}: prefill={r['prefill_ms']}ms  "
                f"decode={r['decode_ms_per_token']}ms/tok ({r['decode_toks_per_s']} tok/s)  "
                f"peak={r['prefill_peak_gb']}/{r['decode_peak_gb']}GB"
            )

    _save()
    print(f"[saved] {out}")


def compare(directory: str) -> None:
    d = Path(directory)
    data = []
    for fp in sorted(d.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                p = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "results" not in p:  # skip the merged length->method file
            continue
        data.append(p)
    if not data:
        print(f"[compare] no result JSONs found under {d}")
        return

    lengths = sorted({int(L) for p in data for L in p["results"]})
    for length in lengths:
        print(f"\n=== prompt_len = {length} ===")
        header = (
            f"{'method':<22}{'prefill_ms':>12}{'dec_ms/tok':>12}"
            f"{'tok/s':>10}{'prefill_GB':>12}{'decode_GB':>10}"
        )
        print(header)
        print("-" * len(header))
        for p in data:
            r = p["results"].get(str(length))
            if not r:
                continue
            def _c(key, width):
                v = r.get(key)
                return f"{('-' if v is None else v):>{width}}"
            label = p.get("tag", p.get("method", "?"))
            if p.get("batch_size", 1) != 1:
                label = f"{label} b{p['batch_size']}"
            print(
                f"{label:<22}"
                f"{_c('prefill_ms', 12)}{_c('decode_ms_per_token', 12)}"
                f"{_c('decode_toks_per_s', 10)}{_c('prefill_peak_gb', 12)}{_c('decode_peak_gb', 10)}"
            )


def merge(directory: str, out_path: str) -> None:
    """Transpose per-method JSONs into one length -> method file.

    Each per-method file is ``{model, batch_size, ..., results: {L: metrics}}``.
    This writes ``{model, batch_size, decode_tokens, lengths: {L: {tag: metrics}}}``
    so all methods of the same prompt-length live in one file, e.g.
    ``lengths["8192"] == {"picokv_offline_k320_v320": ..., "flash_attention_2": ...}``.
    Idempotent: call again after each method finishes and the file simply grows.
    """
    d = Path(directory)
    lengths: dict[str, dict] = {}
    meta: dict = {}
    for fp in sorted(d.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                p = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "results" not in p:  # skip an existing merged file
            continue
        if not meta:
            meta = {k: p.get(k) for k in ("model", "batch_size", "decode_tokens")}
        tag = p.get("tag") or p["method"]
        for L, r in p.get("results", {}).items():
            lengths.setdefault(str(L), {})[tag] = r
    if not lengths:
        print(f"[merge] no result JSONs found under {d}")
        return
    payload = {**meta, "lengths": lengths}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    n_methods = max(len(v) for v in lengths.values())
    print(f"[merge] {out} <- {len(lengths)} lengths x up to {n_methods} methods")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill vs decode efficiency benchmark.")
    parser.add_argument("--compare", default=None, help="aggregate all *.json under this dir into a table and exit")
    parser.add_argument("--merge-dir", default=None,
                        help="transpose all per-method JSONs under this dir into one length->method file (see --merge-out)")
    parser.add_argument("--merge-out", default=None, help="output path for --merge-dir")
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--method", default="picokv", help="picokv | oscar | kivi | flash_attention_2 | sdpa | eager")
    parser.add_argument("--dataset", default=None,
                        help="path to a .jsonl file with 'text' fields; uses real text instead of random token ids")
    parser.add_argument("--lengths", default="4096,16384,32768", help="comma-separated prompt lengths")
    parser.add_argument("--decode-tokens", type=int, default=64, help="new tokens to generate for decode timing")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mem-only", action="store_true",
                        help="memory-only mode: measure peak GPU memory only, skip all latency timing; "
                             "when off, run latency benchmark and also record prefill/decode peak GPU memory")
    parser.add_argument("--output", default="results/efficiency/picokv.json")
    # ── picokv KV-cache params (--method picokv) ──
    parser.add_argument("--k-bits", type=int, default=None, help="PiCoKV K projection bits; KIVI defaults to 2")
    parser.add_argument("--v-bits", type=int, default=None, help="PiCoKV V projection bits; KIVI defaults to 2")
    parser.add_argument("--sink-size", type=int, default=64, help="initial (sink) tokens kept in full precision")
    parser.add_argument("--local-size", type=int, default=256, help="most-recent (local) tokens kept in full precision")
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--pruning", dest="use_pruning", action=argparse.BooleanOptionalAction, default=False,
                        help="decode-time global pruning (gqa4 CUDA path); off by default")
    parser.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    parser.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    # ── KIVI params (--method kivi; also uses --k-bits/--v-bits as 2/4-bit KV) ──
    parser.add_argument("--group-size", type=int, default=32, help="KIVI per-group quant size; OScaR group size (INT2 必须=32)")
    parser.add_argument("--residual-length", type=int, default=32, help="KIVI recent fp16 tokens kept unquantized")
    # ── OScaR params (--method oscar) ──
    parser.add_argument("--num-bits", type=int, default=2, help="OScaR KV quant bits (INT2 默认)")
    parser.add_argument("--quant-mode", default="k-channel", help="OScaR quant mode")
    parser.add_argument("--kv-rotation", default="hadamard", help="OScaR kv rotation (INT2 默认 hadamard)")
    parser.add_argument("--kv-norm", default="1", help="OScaR kv norm (INT2 默认 1)")
    parser.add_argument("--residual-evict-size", type=int, default=256, help="OScaR residual evict size (INT2 默认 256)")
    parser.add_argument("--offline-v-hadamard", action=argparse.BooleanOptionalAction, default=True, help="OScaR: offline per-head Hadamard into v_proj/o_proj (INT2 默认开启)")
    parser.add_argument("--offline", action="store_true", help="picokv: use offline encoders (default path offline_projection/projections/{model}_k{K}_v{V}.pt)")
    parser.add_argument("--offline-proj-path", default=None, help="picokv: explicit offline encoder .pt (overrides the --offline default path)")
    # ── PolarQuant params (--method polar) ──
    parser.add_argument("--polar-mode", default="mixed", choices=["mixed", "group", "snapkv"], help="PolarQuant attention variant: mixed (K polar + V KIVI-quant) | group (K-only) | snapkv")
    parser.add_argument("--polar-rbits", type=int, default=3, help="PolarQuant K radius bits (rbits+tbits<=8)")
    parser.add_argument("--polar-tbits", type=int, default=3, help="PolarQuant K angle bits (rbits+tbits<=8)")
    parser.add_argument("--polar-vbits", type=int, default=2, help="PolarQuant V bits (mixed only; KIVI kernel supports 2 or 4)")
    args = parser.parse_args()
    if args.k_bits is None:
        args.k_bits = 2 if args.method == "kivi" else 320
    if args.v_bits is None:
        args.v_bits = 2 if args.method == "kivi" else 320

    if args.merge_dir:
        merge(args.merge_dir, args.merge_out or f"{args.merge_dir.rstrip('/')}/merged_efficiency.json")
        return
    if args.compare:
        compare(args.compare)
        return
    run(args)


if __name__ == "__main__":
    main()
