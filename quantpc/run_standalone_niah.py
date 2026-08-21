"""Run end-to-end NIAH generation through the current standalone API."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from needle import LLMNeedleHaystackTester, ModelProvider
from quantpc.model_config import load_model_config
from quantpc.quanpc_standalone import StandaloneConfig
from quantpc.quanpc_standalone.transformers import (
    StandaloneLlamaCache,
    load_offline_projections,
    modify_llama_for_standalone,
    modify_qwen3_for_standalone,
)


def contains_needle(response: str) -> tuple[bool, dict[str, bool]]:
    text = response.lower()
    hits = {
        "sandwich": "sandwich" in text,
        "dolores": "dolores" in text,
        "sunny_day": "sunny" in text and "day" in text,
    }
    return all(hits.values()), hits


class StandaloneTransformer(ModelProvider):
    def __init__(self, args, config: StandaloneConfig, offline: dict | None = None) -> None:
        self.args = args
        self.config = config
        offline = offline or {}
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        model_config = load_model_config(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=model_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model.eval()
        modify_fn = (
            modify_qwen3_for_standalone
            if getattr(model_config, "model_type", None) == "qwen3"
            else modify_llama_for_standalone
        )
        modify_fn(self.model, mlp_chunk_size=args.mlp_chunk_size)
        self.offline = offline
        self.last_actual_tokens = 0
        self.last_cache_stats: dict[str, object] = {}

    def evaluate_model(self, prompt, needle=None) -> str:
        # Tokenize inside apply_chat_template. Rendering to a string first and
        # re-tokenizing prepends a second <|begin_of_text|>: the template already
        # emits one, and the tokenizer adds special tokens again by default. The
        # duplicate perturbs the first position every attention sink depends on.
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(self.model.device)
        self.last_actual_tokens = int(input_ids.shape[-1])
        cache = StandaloneLlamaCache(self.config, **self.offline)

        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                past_key_values=cache,
            )
        elapsed = time.perf_counter() - started
        generated_ids = generated.sequences[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        prepare_per_layer_ms = cache.prepare_timings_ms()
        self.last_cache_stats = {
            "elapsed_sec": elapsed,
            "layers": len(cache.layers),
            "prepare_total_ms": sum(prepare_per_layer_ms.values()),
            "prepare_per_layer_ms": prepare_per_layer_ms,
            "decode_calls": cache.decode_calls,
            "generated_tokens": int(generated_ids.shape[-1]),
        }

        del generated, generated_ids, input_ids, cache
        gc.collect()
        for device_idx in range(torch.cuda.device_count()):
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
        return response

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
                "content": (
                    f"{retrieval_question} Don't give information outside the "
                    "document or repeat your findings"
                ),
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(
        self,
        tokens: list[int],
        context_length: Optional[int] = None,
    ) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--lengths", default="64000,110000")
    parser.add_argument("--depths", default="50")
    parser.add_argument("--cells", default=None,
                        help="explicit 'depth:length,...' pairs; overrides the "
                             "--depths x --lengths cross product so a handful of "
                             "cells can be rechecked without the full sweep")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--k-bits", type=int, default=320)
    parser.add_argument("--v-bits", type=int, default=320)
    parser.add_argument("--direction-sample-size", type=int, default=8192)
    parser.add_argument("--sink-size", type=int, default=64)
    parser.add_argument("--local-size", type=int, default=256)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--prune-mode", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=-10.0)
    parser.add_argument("--no-pruning", dest="use_pruning", action="store_false")
    parser.set_defaults(use_pruning=True)
    parser.add_argument("--mlp-chunk-size", type=int, default=8192)
    # 50 matches benchmarks/Needle/run_bin.py; at 30 the model was still
    # mid-preamble on several cells and got cut off before naming the needle.
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline-proj-path", default=None,
                        help="stored block-32 encoders (and decoders, if the file has them)")
    parser.add_argument(
        "--output-json",
        default="results/niah/standalone_current_64k_110k_sample8192.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for end-to-end standalone NIAH")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = StandaloneConfig(
        key_bits=args.k_bits,
        value_bits=args.v_bits,
        direction_sample_size=args.direction_sample_size,
        sink_size=args.sink_size,
        local_size=args.local_size,
        eas=args.eas,
        zerorope=args.zerorope,
        exp_bits=args.exp_bits,
        seed=args.seed,
        use_pruning=args.use_pruning,
        prune_mode=args.prune_mode,
        log_eps=args.log_eps,
        whiten=args.whiten,
    )
    offline: dict = {}
    if args.offline_proj_path is not None:
        if args.whiten:
            raise SystemExit(
                "offline projection and runtime --whiten cannot be combined; "
                "use a folded offline PT without --whiten, or omit the offline "
                "projection when testing online whitening."
            )
        stored = load_offline_projections(args.offline_proj_path)
        # An offline projection is used whole: its decoder was fitted against
        # the codes its own encoder produces, so the two only make sense together.
        for side in ("key", "value"):
            if f"{side}_decoder" not in stored:
                raise SystemExit(
                    f"{args.offline_proj_path} stores encoders only; rebuild it "
                    "with offline_projection/create_offline_projections.py"
                )
            offline[f"{side}_encoders"] = stored[f"{side}_encoder"]
            offline[f"{side}_decoders"] = stored[f"{side}_decoder"]
        print(f"offline: {args.offline_proj_path}", flush=True)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provider = StandaloneTransformer(args, config, offline)
    tester = LLMNeedleHaystackTester(provider)
    results: dict[str, object] = {
        "config": vars(args),
        "needle": tester.needle.strip(),
        "rows": [],
    }

    if args.cells:
        grid = [
            tuple(int(v) for v in pair.split(":"))
            for pair in args.cells.split(",") if pair
        ]
    else:
        grid = [
            (depth, length)
            for depth in parse_int_list(args.depths)
            for length in parse_int_list(args.lengths)
        ]

    for depth, requested_length in grid:
            # One cell running out of memory must not discard the cells that
            # already succeeded, so failures are recorded and the sweep goes on.
            try:
                response = tester.evaluate(requested_length, depth)
                found, hits = contains_needle(response)
                error = None
            except torch.cuda.OutOfMemoryError as exc:
                response, found, hits, error = "", False, {}, f"OOM: {exc}"
                torch.cuda.empty_cache()
            row = {
                "requested_length": requested_length,
                "actual_tokens": provider.last_actual_tokens,
                "depth": depth,
                "whiten": bool(args.whiten),
                "found": found,
                "hits": hits,
                "error": error,
                "response": response,
                **provider.last_cache_stats,
            }
            results["rows"].append(row)
            output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False)
            )
            print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
