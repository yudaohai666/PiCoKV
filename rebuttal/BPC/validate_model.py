"""Real-model checks before efficiency: same-cache packed vs reconstructed KV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from benchmarks.common import load_model, seed_everything
from benchmarks.Efficiency.run_efficiency import _kv_cache_bytes, build_inputs_from_dataset
from picokv.picokv_standalone import transformers as adapter
from rebuttal.BPC.encoding import binarypc_encoding
from rebuttal.reconstruct_attention import decode_attention as dense_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/zxh/zxh/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_everything(42)
    model, tokenizer, factory = load_model(
        args.model, method="picokv", k_bits=320, v_bits=320,
        sink_size=64, local_size=256, exp_bits=4, eas=0.01, zerorope=0,
        use_pruning=False, prune_mode=3, log_eps=-10.0,
    )
    inputs = build_inputs_from_dataset(
        str(ROOT / "benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl"),
        tokenizer, 1, args.length, model.device,
    )
    cache = factory()
    records = []
    original = adapter.decode_attention

    def checked(query, state, **kwargs):
        packed = original(query, state, **kwargs)
        # original() has already appended the new token exactly once.
        reference = dense_reference(query, state)
        error = (packed.float() - reference.float())
        relative = error.norm() / reference.float().norm().clamp_min(1e-8)
        assert torch.isfinite(packed).all() and torch.isfinite(reference).all()
        records.append({
            "call": len(records), "relative_l2": relative.item(),
            "max_abs": error.abs().max().item(), "rmse": error.square().mean().sqrt().item(),
        })
        return packed

    adapter.decode_attention = checked
    start = time.perf_counter()
    try:
        with torch.inference_mode(), binarypc_encoding():
            generated = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), past_key_values=cache,
                min_new_tokens=args.tokens, max_new_tokens=args.tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
    finally:
        adapter.decode_attention = original
    expected_calls = (args.tokens - 1) * model.config.num_hidden_layers
    assert len(records) == expected_calls, (len(records), expected_calls)
    for state in cache.layers.values():
        assert state.key_bits == state.value_bits == 320
        assert torch.isfinite(state.key_projection).all() and torch.isfinite(state.value_projection).all()
    # Approximate packed queries/weights need not equal ordinary BF16 attention.
    # Reject gross disagreement; preserve every layer/step's error for review.
    worst = max(record["relative_l2"] for record in records)
    payload = {
        "model": args.model, "encoding": "binarypc", "k_bits": 320, "v_bits": 320,
        "coding_version": "bpc320_source_global_rng_v3", "zerorope": 0,
        "prompt_len": args.length, "new_tokens": args.tokens, "elapsed_sec": time.perf_counter() - start,
        "checks": {"finite_cache_and_outputs": True, "all_layers_320_bits": True,
                   "decode_calls": len(records), "expected_decode_calls": expected_calls,
                   "packed_vs_same_cache_dense_max_relative_l2": worst,
                   "gross_error_threshold": 0.35, "passed": worst < 0.35},
        "cache_gib_after_decode": _kv_cache_bytes(cache) / 1024**3,
        "generated_text": tokenizer.decode(generated[0, args.length:], skip_special_tokens=True),
        "generated_token_ids": generated[0, args.length:].tolist(), "per_call_errors": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["checks"], indent=2), flush=True)
    assert payload["checks"]["passed"], f"same-cache packed/dense error too large: {worst}"


if __name__ == "__main__":
    main()
