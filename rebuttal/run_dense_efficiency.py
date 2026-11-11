"""Reuse dense efficiency measurement with memory-bounded chunked prefill.

Full BF16 DynamicCache and the stock single-token forward are retained. Only
long prompt forwards are split; this entry point is for last-logit, unpadded
generation efficiency measurement, not full-sequence logits/hidden-state APIs.
"""

from __future__ import annotations

import argparse
from functools import wraps
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def chunk_dense_prefill(model, chunk_size):
    if chunk_size <= 0:
        raise ValueError("prefill chunk size must be positive")
    original_forward = model.forward

    @wraps(original_forward)
    def forward(input_ids=None, **kwargs):
        if (
            input_ids is None or input_ids.shape[-1] <= chunk_size
            or kwargs.get("inputs_embeds") is not None
            or kwargs.get("use_cache", model.config.use_cache) is False
            or kwargs.get("return_dict", True) is False
        ):
            return original_forward(input_ids=input_ids, **kwargs)
        cache = kwargs.get("past_key_values")
        past_length = cache.get_seq_length() if cache is not None else 0
        for start in range(0, input_ids.shape[-1], chunk_size):
            end = min(start + chunk_size, input_ids.shape[-1])
            current = dict(kwargs, past_key_values=cache, logits_to_keep=1)
            for name in ("cache_position", "position_ids"):
                if kwargs.get(name) is not None:
                    current[name] = kwargs[name][..., start:end]
            mask = kwargs.get("attention_mask")
            if mask is not None:
                if mask.ndim != 2:
                    raise ValueError("chunked dense benchmark expects a 2D attention mask")
                current["attention_mask"] = mask[:, :past_length + end]
            try:
                output = original_forward(input_ids=input_ids[:, start:end], **current)
            except torch.cuda.OutOfMemoryError:
                print(f"[dense chunked] OOM in prompt chunk [{start}:{end}], cached prefix={past_length + start}", flush=True)
                raise
            cache = output.past_key_values
        return output

    model.forward = forward


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    options, forwarded = parser.parse_known_args()
    if "--help" in forwarded or "-h" in forwarded:
        parser.print_help()
        print("\nOriginal dense efficiency arguments:")
    from benchmarks.Efficiency import run_efficiency as original

    if not any(arg == "--method" or arg.startswith("--method=") for arg in forwarded):
        forwarded += ["--method", "flash_attention_2"]
    if not any(arg == "--output" or arg.startswith("--output=") for arg in forwarded):
        forwarded += ["--output", str(REPO_ROOT / "rebuttal/results/dense_flash_attention_2_chunked.json")]
    original_load, original_tag, original_run = original.load_model, original.output_tag, original.run
    original_measure = original.measure
    previous_argv = sys.argv

    def load_model(*args, **kwargs):
        model, tokenizer, factory = original_load(*args, **kwargs)
        chunk_dense_prefill(model, options.prefill_chunk_size)
        return model, tokenizer, factory

    def run(args):
        if args.method != "flash_attention_2":
            raise ValueError("this entry point measures dense FlashAttention-2 only")
        return original_run(args)

    original.load_model = load_model
    original.output_tag = lambda *args, **kwargs: original_tag(*args, **kwargs) + f"_chunkprefill{options.prefill_chunk_size}"
    original.run = run
    def measure(*args, **kwargs):
        try:
            return original_measure(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as error:
            print(f"[dense chunked] {error}", flush=True)
            raise
    original.measure = measure
    sys.argv = [__file__, *forwarded]
    try:
        original.main()
    finally:
        original.load_model, original.output_tag, original.run = original_load, original_tag, original_run
        original.measure = original_measure
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
