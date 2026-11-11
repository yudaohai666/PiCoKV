"""Compare packed and both BF16 controls on identical real-model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from benchmarks.common import load_model, seed_everything
from benchmarks.Efficiency.run_efficiency import build_inputs_from_dataset
from picokv.picokv_standalone import transformers as adapter
from rebuttal.reconstruct_attention import decode_attention as composed_decode
from rebuttal.reconstruct_cuda_attention import decode_attention as cuda_decode, load_reconstruct_extension
from rebuttal.unpacked_attention import decode_attention as factorized_reference
from rebuttal.unpacked_kernel_attention import decode_attention_fused as factorized_decode


def errors(reference, actual):
    ref, out = reference.float(), actual.float()
    if not torch.isfinite(ref).all() or not torch.isfinite(out).all():
        raise AssertionError("non-finite attention output")
    difference = out - ref
    return {
        "max_abs": difference.abs().max().item(),
        "relative_rmse": (difference.norm() / ref.norm().clamp_min(1e-12)).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--output", default="rebuttal/results/cuda_reconstruct_64k/real_model_output_error.json")
    args = parser.parse_args()
    seed_everything(42)
    load_reconstruct_extension()
    model, tokenizer, make_cache = load_model(args.model, method="picokv", k_bits=320, v_bits=320)
    inputs = build_inputs_from_dataset(
        str(REPO_ROOT / "benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl"),
        tokenizer, 1, args.length, model.device,
    )
    records = []
    original = adapter.decode_attention

    def compare(query, cache, **kwargs):
        # Append current KV exactly once, then reuse the updated SAME cache for
        # both reconstruction controls. Return packed output to preserve the
        # model's packed generation trajectory across layers.
        packed = original(query, cache, **kwargs).clone()
        options = {"attention_scale": kwargs.get("attention_scale")}
        composed = composed_decode(query, cache, **options)
        custom = cuda_decode(query, cache, **options)
        error = errors(composed, custom)
        # Split attention and the reused output introduce BF16 roundings that
        # differ from one full concatenated FlashAttention call.
        if error["relative_rmse"] > 0.01:
            raise AssertionError(f"CUDA reconstruction error exceeds BF16 tolerance: {error}")
        factored_ref = factorized_reference(query, cache, **options)
        factored_out = factorized_decode(query, cache, **options)
        factorized_error = errors(factored_ref, factored_out)
        if factorized_error["relative_rmse"] > 0.02:
            raise AssertionError(f"factorized attention error exceeds BF16 tolerance: {factorized_error}")
        records.append({
            "layer": len(records), "global_len": cache.global_len,
            "dense_len": cache.dense_len, "cuda_vs_composed": error,
            "factorized_vs_composed": factorized_error,
            "packed_vs_composed": errors(composed, packed),
        })
        return packed

    adapter.decode_attention = compare
    try:
        with torch.inference_mode():
            model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                past_key_values=make_cache(), max_new_tokens=2, min_new_tokens=2,
                do_sample=False, temperature=None, top_p=None, use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        adapter.decode_attention = original
    assert len(records) == model.config.num_hidden_layers
    result = {
        "model": args.model, "prompt_len": args.length, "k_bits": 320, "v_bits": 320,
        "input_sha256": hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
        "scope": "First decode token, all layers; identical cache/query, packed model trajectory",
        "cuda_vs_composed_max_relative_rmse": max(r["cuda_vs_composed"]["relative_rmse"] for r in records),
        "cuda_vs_composed_max_abs": max(r["cuda_vs_composed"]["max_abs"] for r in records),
        "packed_vs_composed_max_relative_rmse": max(r["packed_vs_composed"]["relative_rmse"] for r in records),
        "factorized_vs_composed_max_relative_rmse": max(r["factorized_vs_composed"]["relative_rmse"] for r in records),
        "factorized_vs_composed_max_abs": max(r["factorized_vs_composed"]["max_abs"] for r in records),
        "error_tolerances_relative_rmse": {"reconstructed": 0.01, "factorized": 0.02},
        "layers": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
