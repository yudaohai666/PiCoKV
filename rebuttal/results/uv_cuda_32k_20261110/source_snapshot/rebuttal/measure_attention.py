"""Measure existing decode attention in the real model, without changing kernels.

CUDA events measure the attention call after cache append. A separate profiler
pass records only CUDA kernel durations within attention/cache scopes. The
steady-state generation loop is the original efficiency runner's helper.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from benchmarks.common import load_model, seed_everything
from benchmarks.Efficiency.run_efficiency import (
    _decode_ms_per_token, _free, build_inputs_from_dataset,
)
from picokv.picokv_standalone import decode as production
from picokv.picokv_standalone import transformers as adapter
from rebuttal import reconstruct_cuda_attention, unpacked_kernel_attention


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recorder:
    def __init__(self, backend, layers, tokens, warmup, repeats, implementation="cuda"):
        self.backend, self.layers = backend, layers
        self.implementation = implementation
        self.tokens, self.warmup, self.repeats = tokens, warmup, repeats
        self.repeat, self.step, self.layer = -1, -1, 0
        self.trace = False
        self.enabled = True
        self.current_attention = None
        self.samples = []
        self.shapes = {}
        self.pool = []
        # Prime event handles outside all measured forward calls.
        for _ in range(repeats * tokens * layers * 2):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            end.record()
            self.pool.append((start, end))
        torch.cuda.synchronize()
        self.used = 0

    def pre_forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids.shape[-1] > 1:
            self.repeat += 1
            self.step = -1
            print(f"[{self.backend}] prefill {self.repeat + 1}", flush=True)
        else:
            self.step += 1
        self.layer = 0

    @property
    def timed(self):
        return self.enabled and self.warmup <= self.step < self.warmup + self.tokens

    def pair(self, stage, layer):
        if not self.timed:
            return None
        pair = self.pool[self.used]
        self.used += 1
        self.samples.append((stage, self.repeat, self.step - self.warmup, layer, pair))
        return pair

    def scope(self, stage, layer):
        if self.trace:
            return torch.profiler.record_function(f"{stage}/{self.backend}/{self.step}/{layer}")
        return nullcontext()

    def summary(self):
        result = {}
        for stage in ("attention", "cache_append"):
            rows = [s for s in self.samples if s[0] == stage]
            assert len(rows) == self.repeats * self.tokens * self.layers, (stage, len(rows))
            totals = defaultdict(float)
            layer_values = defaultdict(list)
            for _, repeat, step, layer, (start, end) in rows:
                value = start.elapsed_time(end)
                totals[repeat, step] += value
                layer_values[layer].append(value)
            per_repeat = [statistics.mean(totals[r, t] for t in range(self.tokens))
                          for r in range(self.repeats)]
            result[stage] = {
                "median_ms_per_token_all_layers": statistics.median(per_repeat),
                "repeat_mean_ms_per_token_all_layers": per_repeat,
                "mean_ms_per_layer": statistics.median(per_repeat) / self.layers,
                "per_layer_mean_ms": {str(k): statistics.mean(v) for k, v in layer_values.items()},
                "calls": len(rows),
            }
        return result


@contextmanager
def instrument(model, recorder):
    restore = []

    def replace(obj, name, value):
        restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    hook = model.register_forward_pre_hook(recorder.pre_forward, with_kwargs=True)
    if recorder.backend == "dense":
        from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS
        from transformers.cache_utils import DynamicCache

        original_attention = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        original_update = DynamicCache.update

        def attention(module, query, key, value, *args, **kwargs):
            if query.shape[2] != 1:
                return original_attention(module, query, key, value, *args, **kwargs)
            layer = module.layer_idx
            pair = recorder.pair("attention", layer)
            if pair:
                pair[0].record()
            with recorder.scope("attention", layer):
                output = original_attention(module, query, key, value, *args, **kwargs)
            if pair:
                pair[1].record()
            if recorder.timed and recorder.step == recorder.warmup and recorder.repeat == 0:
                recorder.shapes[str(layer)] = {"query": list(query.shape), "key": list(key.shape),
                                               "value": list(value.shape)}
            return output

        def update(cache, key_states, value_states, layer_idx, *args, **kwargs):
            if key_states.shape[2] != 1:
                return original_update(cache, key_states, value_states, layer_idx, *args, **kwargs)
            pair = recorder.pair("cache_append", layer_idx)
            if pair:
                pair[0].record()
            with recorder.scope("cache_append", layer_idx):
                output = original_update(cache, key_states, value_states, layer_idx, *args, **kwargs)
            if pair:
                pair[1].record()
            return output

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = attention
        replace(DynamicCache, "update", update)
    else:
        module, forward = {
            "packed": (production, production.decode_attention),
            "unpacked_fused": (unpacked_kernel_attention, unpacked_kernel_attention.decode_attention_fused),
            "reconstruct_cuda": (reconstruct_cuda_attention, reconstruct_cuda_attention.decode_attention),
        }[recorder.backend]
        original_append = module._append_decode_tokens

        def append(cache, *args, **kwargs):
            layer = recorder.layer
            pair = recorder.pair("cache_append", layer)
            if pair:
                pair[0].record()
            with recorder.scope("cache_append", layer):
                output = original_append(cache, *args, **kwargs)
            if pair:
                pair[1].record()
            # Start attention after the original append, preserving all original
            # operators, argument values, cache mutations, and their ordering.
            if recorder.current_attention:
                recorder.current_attention[0].record()
            return output

        def attention(query, cache, **kwargs):
            layer = recorder.layer
            pair = recorder.pair("attention", layer)
            recorder.current_attention = pair
            # The profiler's attention scope starts after append via a separate
            # context entered by append below, so cache kernels stay excluded.
            try:
                output = forward(query, cache, **kwargs)
                if pair:
                    pair[1].record()
                if recorder.timed and recorder.step == recorder.warmup and recorder.repeat == 0:
                    recorder.shapes[str(layer)] = {
                        "query": list(query.shape), "global_len": cache.global_len,
                        "dense_len": cache.dense_len, "packed_key": list(cache.packed_key.shape),
                        "packed_value": list(cache.packed_value.shape),
                    }
                return output
            finally:
                if active_scope[0] is not None:
                    active_scope[0].__exit__(None, None, None)
                    active_scope[0] = None
                recorder.current_attention = None
                recorder.layer += 1

        active_scope = [None]
        event_append = append

        def scoped_append(cache, *args, **kwargs):
            output = event_append(cache, *args, **kwargs)
            if recorder.trace:
                active_scope[0] = recorder.scope("attention", recorder.layer)
                active_scope[0].__enter__()
            return output

        replace(module, "_append_decode_tokens", scoped_append)
        replace(adapter, "decode_attention", attention)
    try:
        yield
    finally:
        hook.remove()
        if recorder.backend == "dense":
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = original_attention
        for obj, name, original in reversed(restore):
            setattr(obj, name, original)


def profile_decode(model, make_cache, inputs, recorder, tokens):
    """Separate untimed-prefill pass; profile only warmed single-token forwards."""
    cache = make_cache() if make_cache else None
    ids, mask = inputs, torch.ones_like(inputs)
    pos = inputs.shape[1]

    def step(prefill=False):
        nonlocal cache, ids, mask, pos
        positions = torch.arange(pos, device=inputs.device) if prefill else torch.arange(pos, pos + 1, device=inputs.device)
        mi = model.prepare_inputs_for_generation(ids, past_key_values=cache, attention_mask=mask,
                                                use_cache=True, cache_position=positions, logits_to_keep=1)
        output = model(**mi)
        cache = output.past_key_values
        next_token = output.logits[:, -1:, :].argmax(-1)
        ids = torch.cat((ids, next_token), 1)
        mask = torch.cat((mask, torch.ones_like(next_token)), 1)
        if not prefill:
            pos += 1

    recorder.enabled = False
    with torch.no_grad():
        step(prefill=True)
        for _ in range(recorder.warmup):
            step()
        torch.cuda.synchronize()
        recorder.trace = True
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(tokens):
                    step()
                torch.cuda.synchronize()
        finally:
            recorder.trace = False
    # Match each actual GPU kernel to its CUDA launch timestamp, then to our
    # CPU scope. ATen parent/child attribution is insufficient for custom C++
    # extensions: kernels without an ATen operator may inherit a previous op's
    # external correlation. Runtime correlation IDs identify the actual launch.
    with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
        prof.export_chrome_trace(temporary.name)
        events = json.loads(Path(temporary.name).read_text())["traceEvents"]
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "correlation" in e.get("args", {})}
    launch_threads = {(e["pid"], e["tid"]) for e in launches.values()}
    scopes = [e for e in events if e.get("cat") == "user_annotation"
              and (e.get("pid"), e.get("tid")) in launch_threads
              and any(e.get("name", "").startswith(f"{stage}/{recorder.backend}/")
                      for stage in ("attention", "cache_append"))]
    scopes.sort(key=lambda e: e["ts"])
    starts = [e["ts"] for e in scopes]
    attributed = defaultdict(list)
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            continue
        index = bisect_right(starts, launch["ts"]) - 1
        if index >= 0:
            scope = scopes[index]
            if (scope["pid"], scope["tid"]) == (launch["pid"], launch["tid"]) and launch["ts"] <= scope["ts"] + scope["dur"]:
                attributed[scope["name"]].append(event)
    result = {}
    for stage in ("attention", "cache_append"):
        roots = [event for event in scopes if event["name"].startswith(f"{stage}/{recorder.backend}/")]
        assert len(roots) == tokens * recorder.layers, (stage, len(roots))
        kernels = defaultdict(lambda: {"count": 0, "total_us": 0.0})
        token_totals = defaultdict(float)
        layer_totals = defaultdict(float)
        for event in roots:
            duration = 0.0
            for kernel in attributed[event["name"]]:
                kernels[kernel["name"]]["count"] += 1
                kernels[kernel["name"]]["total_us"] += kernel["dur"]
                duration += kernel["dur"]
            assert duration > 0, event["name"]
            _, _, token, layer = event["name"].split("/")
            token_totals[int(token)] += duration / 1000
            layer_totals[int(layer)] += duration / 1000
        assert sum(k["count"] for k in kernels.values()) > 0
        if stage == "attention" and recorder.backend != "dense":
            marker = {"packed": "picokv_v1_expbits_atomic2_runtime_unrolled_kernel",
                      "unpacked_fused": "unpacked_bf16_attention_kernel",
                      "reconstruct_cuda": "reconstructed_bf16_attention_kernel"}[recorder.backend]
            if recorder.implementation == "triton" and recorder.backend != "packed":
                marker = "_global_attention" if recorder.backend == "unpacked_fused" else "_reconstruct_attention_block"
            main_calls = sum(k["count"] for name, k in kernels.items() if marker in name)
            output_calls = sum(k["count"] for name, k in kernels.items() if "picokv_fused_output_projection_kernel" in name)
            assert main_calls == tokens * recorder.layers, ("main kernel", main_calls)
            assert output_calls == tokens * recorder.layers, ("output kernel", output_calls)
        result[stage] = {
            "mean_kernel_ms_per_token_all_layers": sum(token_totals.values()) / tokens,
            "median_kernel_ms_per_token_all_layers": statistics.median(token_totals.values()),
            "per_token_kernel_ms_all_layers": dict(token_totals),
            "per_layer_mean_kernel_ms": {str(k): v / tokens for k, v in layer_totals.items()},
            "kernels": [{"name": name, **data, "ms_per_token_all_layers": data["total_us"] / tokens / 1000}
                        for name, data in sorted(kernels.items(), key=lambda x: x[1]["total_us"], reverse=True)],
            "calls": len(roots),
        }
    return result


def main():
    global reconstruct_cuda_attention, unpacked_kernel_attention
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["dense", "packed", "unpacked_fused", "reconstruct_cuda"], required=True)
    parser.add_argument("--model", default="/home/zxh/zxh/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--decode-warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-tokens", type=int, default=8)
    parser.add_argument("--control-implementation", choices=["cuda", "triton"], default="cuda",
                        help="Use archived Triton controls only for implementation comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.control_implementation == "triton":
        from rebuttal import reconstruct_cuda_attention_triton, unpacked_kernel_attention_triton
        reconstruct_cuda_attention = reconstruct_cuda_attention_triton
        unpacked_kernel_attention = unpacked_kernel_attention_triton
    seed_everything(42)
    source_files = [__file__, "picokv/picokv_standalone/decode.py", "picokv/picokv_cuda_extension/cuda_kernel.cu",
                    "rebuttal/reconstruct_cuda_attention.py", "rebuttal/unpacked_kernel_attention.py",
                    "benchmarks/Efficiency/run_efficiency.py", "rebuttal/cuda_attention.py",
                    "rebuttal/cuda/attention.cu", "rebuttal/cuda/attention.cpp", "rebuttal/cuda/attention_ptx.cuh",
                    "rebuttal/reconstruct_cuda_attention_triton.py", "rebuttal/unpacked_kernel_attention_triton.py"]
    before = {str(Path(p).resolve().relative_to(ROOT)): digest(p) for p in source_files}
    start = time.time()
    if args.backend in ("unpacked_fused", "reconstruct_cuda") and args.control_implementation == "cuda":
        from rebuttal.cuda_attention import load_attention_extension
        load_attention_extension()
    model, tokenizer, make_cache = load_model(args.model, method="flash_attention_2" if args.backend == "dense" else "picokv",
                                             k_bits=320, v_bits=320)
    inputs = build_inputs_from_dataset(str(ROOT / "benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl"),
                                       tokenizer, 1, args.length, model.device)
    recorder = Recorder(args.backend, model.config.num_hidden_layers, args.decode_tokens,
                        args.decode_warmup, args.repeats, args.control_implementation)
    with instrument(model, recorder):
        model_ms = _decode_ms_per_token(model, inputs, args.decode_tokens, make_cache, args.repeats,
                                        decode_warmup=args.decode_warmup)
        torch.cuda.synchronize()
        event_results = recorder.summary()
        # Release CUDA event handles before the independent profiler pass.
        recorder.samples.clear()
        recorder.pool.clear()
        _free()
        profile_results = profile_decode(model, make_cache, inputs, recorder, args.profile_tokens)
    assert all(digest(ROOT / p) == sha for p, sha in before.items()), "sources changed during measurement"
    output = {
        "backend": args.backend, "control_implementation": args.control_implementation,
        "model": args.model, "prompt_len": args.length,
        "layers": recorder.layers, "batch_size": 1, "decode_tokens": args.decode_tokens,
        "decode_warmup": args.decode_warmup, "repeats": args.repeats, "profile_tokens": args.profile_tokens,
        "gpu": torch.cuda.get_device_name(), "input_sha256": hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
        "scope": "Attention after cache append: includes compressed query projection/quantization, unpack/reconstruction, retained dense attention and output fusion; excludes QKV projection/RoPE/cache append/model o_proj/MLP",
        "cuda_events": event_results, "kernel_profile": profile_results,
        "instrumented_model_ms_per_token": model_ms,
        "model_timing_note": "Includes event instrumentation overhead; use previous uninstrumented efficiency results for model throughput",
        "event_timing_note": "Median of five repeat means (64 tokens x all layers); CUDA event interval may include host launch gaps; no synchronization within attention calls",
        "kernel_timing_note": "Separate warmed profiler pass; actual CUDA kernel/memory durations matched by runtime launch correlation and CPU scope timestamps; excludes host gaps and avoids ATen parent/child attribution errors for custom extensions",
        "first_measured_token_shapes_by_layer": recorder.shapes,
        "source_sha256": before, "sources_unchanged_during_measurement": True,
        "elapsed_seconds": time.time() - start,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"backend": args.backend, "attention_event_ms": event_results["attention"]["median_ms_per_token_all_layers"],
                      "attention_kernel_ms": profile_results["attention"]["mean_kernel_ms_per_token_all_layers"],
                      "cache_append_kernel_ms": profile_results["cache_append"]["mean_kernel_ms_per_token_all_layers"],
                      "output": str(path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
