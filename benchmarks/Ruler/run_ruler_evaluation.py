#!/usr/bin/env python3
"""RULER evaluation runner for the QuanPC quantized KV-cache algorithm.

Mirrors benchmarks/Needle/run_bin_v4.py in HOW the QuanPC method is applied:
  * load a plain HF model + tokenizer
  * modify(model) monkey-patches the attention
  * a fresh QuantKVCache is injected as past_key_values for every generation

Backend selection is driven by the --cuda-v1-exp8 flag (CUDA vs torch reference),
exactly like the Needle runner.

Usage:
    python benchmarks/Ruler/run_ruler_evaluation.py \
        --model /path/to/Llama-3.1-8B-Instruct \
        --tasks ruler \
        --max_seq_lengths 131072 \
        --max_length 131072 \
        --k-bits 256 --v-bits 320 \
        --cuda-v1-exp8 \
        --apply_chat_template
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from lm_eval import evaluator
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `import src.lmeval` / `import quantpc` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lmeval import MyModel  # noqa: E402


def seed_everything(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RULER evaluation with the QuanPC quantized KV cache"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
        help="Path or name of the HF model to evaluate",
    )

    # Method dispatch
    #   quantpc            -> my quantized KV-cache method (modify + QuantKVCache)
    #   flash_attention_2  -> dense full-precision baseline (flash-attn-2)
    #   sdpa               -> dense full-precision baseline (torch SDPA)
    #   (more methods to be added in load_model_and_tokenizer)
    parser.add_argument(
        "--method",
        type=str,
        default="quantpc",
        help=(
            "evaluation method: 'quantpc' (quantized KV cache), or a dense "
            "baseline attn implementation 'flash_attention_2' / 'sdpa'"
        ),
    )

    # QuanPC KV-cache parameters (only used when --method quantpc;
    # mirrors benchmarks/Needle/run_bin_v4.py)
    parser.add_argument("--k-bits", type=int, default=256)
    parser.add_argument("--v-bits", type=int, default=320)
    parser.add_argument(
        "--projected-query-bits",
        type=int,
        default=8,
        help="signed quantization bits for projected query",
    )
    parser.add_argument(
        "--exp-bits",
        type=int,
        default=4,
        help="unsigned per-64-token quantization bits for global exp weights",
    )
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument(
        "--cuda-v1-exp8",
        action="store_true",
        help=(
            "select the CUDA backend (quantpc.quanpc) with experimental "
            "v_bpd=1 + uint8-exp global attention; without this flag the "
            "torch-only reference backend (quantpc.quanpc_torch) is used"
        ),
    )

    # Task configuration
    parser.add_argument(
        "--tasks",
        type=str,
        default="ruler",
        help="Comma-separated list of tasks to evaluate (default: ruler)",
    )
    parser.add_argument(
        "--max_seq_lengths",
        type=str,
        default="131072",
        help="Comma-separated list of max sequence lengths to evaluate",
    )

    # Runtime configuration
    parser.add_argument("--max_length", type=int, default=131072,
                        help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_gen_toks", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (for testing)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Output configuration
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results (default: auto-generated)")
    parser.add_argument("--write_out", action="store_true", default=True)

    # Chat template configuration
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--fewshot_as_multiturn", action="store_true", default=False)

    # Tokenizer options
    parser.add_argument("--add_bos_token", action="store_true")
    parser.add_argument("--prefix_token_id", type=int, default=None)

    return parser.parse_args()


# Dense-baseline attn implementations selectable via --method.
BASELINE_ATTN = {"flash_attention_2", "sdpa", "eager"}


def load_model_and_tokenizer(args):
    """Load the HF model + tokenizer for the requested --method.

    - quantpc: apply modify(model) and (later) inject a QuantKVCache, exactly
      like benchmarks/Needle/run_bin_v4.py. Backend (CUDA vs torch reference)
      is selected by --cuda-v1-exp8.
    - flash_attention_2 / sdpa: dense full-precision baseline. No modify, no
      QuantKVCache injection (model uses its default cache).
    """
    logger.info(f"Loading model: {args.model} (method={args.method})")

    from_pretrained_kwargs = {"torch_dtype": "bfloat16"}
    QuantKVCache = None
    modify = None

    if args.method == "quantpc":
        if args.cuda_v1_exp8:
            from quantpc.quanpc import QuantKVCache, modify
        else:
            from quantpc.quanpc_torch import QuantKVCache, modify
    elif args.method in BASELINE_ATTN:
        from_pretrained_kwargs["attn_implementation"] = args.method
    else:
        raise ValueError(
            f"Unknown method '{args.method}'. Supported: 'quantpc', "
            f"{sorted(BASELINE_ATTN)}. Add new methods in load_model_and_tokenizer."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        import accelerate  # noqa: F401
        device_map = "auto"
    except ImportError:
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")

    if modify is not None:
        modify(model)
        logger.info("QuanPC method applied (modify)")
    else:
        logger.info(f"Baseline (dense), attn_implementation={args.method}")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, QuantKVCache


def make_kv_cache_factory(QuantKVCache, args):
    """Return a zero-arg callable that builds a fresh QuantKVCache per request."""
    def factory():
        return QuantKVCache(
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            projected_query_bits=args.projected_query_bits,
            exp_bits=args.exp_bits,
            eas=args.eas,
            zerorope=args.zerorope,
            use_cuda_v1_exp8=args.cuda_v1_exp8,
        )
    return factory


def get_output_path(args):
    """Auto-generate an output path keyed by model name and method."""
    if args.output_path:
        return args.output_path
    model_key = Path(args.model).name
    if args.method == "quantpc":
        return f"results/ruler/{model_key}_quantpc_k{args.k_bits}_v{args.v_bits}/"
    return f"results/ruler/{model_key}_{args.method}/"


def create_ruler_model(model, tokenizer, kv_cache_factory, args):
    """Create the MyModel wrapper wired with the QuanPC KV-cache factory."""
    ruler_model = MyModel(
        model=model,
        tokenizer=tokenizer,
        model_max_length=args.max_length,
        batch_size=args.batch_size,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        max_gen_toks=args.max_gen_toks,
        kv_cache_factory=kv_cache_factory,
    )
    logger.info(
        f"MyModel created (max_length={args.max_length}, "
        f"batch_size={args.batch_size}, max_gen_toks={args.max_gen_toks})"
    )
    return ruler_model


def run_evaluation(ruler_model, args, output_path):
    """Run RULER evaluation via lm_eval."""
    tasks = [task.strip() for task in args.tasks.split(",")]
    max_seq_lengths = [int(x.strip()) for x in args.max_seq_lengths.split(",")]

    logger.info(f"Evaluating tasks: {tasks} with sequence lengths: {max_seq_lengths}")

    max_seq_length_needed = max(max_seq_lengths)
    if args.max_length < max_seq_length_needed:
        logger.warning(
            f"Model max_length ({args.max_length}) < longest sequence "
            f"({max_seq_length_needed})"
        )

    metadata = {
        "max_seq_lengths": max_seq_lengths,
        "pretrained": args.model,
        "tokenizer": args.model,
    }
    model_args_str = f"pretrained={args.model},max_length={args.max_length}"

    os.makedirs(output_path, exist_ok=True)

    logger.info("Starting RULER evaluation...")
    if args.apply_chat_template:
        logger.info("Chat template enabled")

    results = evaluator.simple_evaluate(
        model=ruler_model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        write_out=args.write_out,
        log_samples=True,
        metadata=metadata,
        model_args=model_args_str,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
    )
    return results


def save_results(results, args, output_path):
    """Save evaluation results to files."""
    model_key = Path(args.model).name
    tag = f"quantpc_k{args.k_bits}_v{args.v_bits}" if args.method == "quantpc" else args.method

    results_lite = {
        "results": results.get("results", {}),
        "config": results.get("config", {}),
    }
    results_file = os.path.join(output_path, f"results_{model_key}_{tag}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_lite, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_file}")

    if "samples" in results:
        jsonl_file = os.path.join(output_path, f"predictions_{model_key}_{tag}.jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for task_name, task_samples in results["samples"].items():
                for sample in task_samples:
                    doc = sample.get("doc", {})
                    pred_data = {
                        "task": task_name,
                        "pred": sample.get("resps", [""])[0] if sample.get("resps") else "",
                        "target": sample.get("target", ""),
                        "index": doc.get("index", 0),
                        "length": doc.get("length", 0),
                        "max_length": doc.get("max_length", 0),
                        "outputs": doc.get("outputs", []),
                    }
                    f.write(json.dumps(pred_data, ensure_ascii=False) + "\n")
        logger.info(f"Predictions saved to: {jsonl_file}")


def print_results_summary(results):
    """Print evaluation results summary."""
    logger.info("\n" + "=" * 60)
    logger.info("RULER EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)

    for task_name, task_results in results.get("results", {}).items():
        logger.info(f"\nTask: {task_name}")
        logger.info("-" * 40)
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                if metric_value == -1:
                    logger.info(f"  {metric_name:20s}: N/A (no samples)")
                else:
                    logger.info(f"  {metric_name:20s}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name:20s}: {metric_value}")


def main():
    """Main function."""
    args = parse_arguments()
    seed_everything(args.seed)

    logger.info("RULER Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Method: {args.method}")
    if args.method == "quantpc":
        backend = "CUDA (quantpc.quanpc)" if args.cuda_v1_exp8 else "torch (quantpc.quanpc_torch)"
        logger.info(f"Backend: {backend}")
        logger.info(f"KV cache: k_bits={args.k_bits}, v_bits={args.v_bits}, "
                    f"projected_query_bits={args.projected_query_bits}, "
                    f"exp_bits={args.exp_bits}, eas={args.eas}, zerorope={args.zerorope}")
    logger.info(f"Tasks: {args.tasks}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")

    try:
        model, tokenizer, QuantKVCache = load_model_and_tokenizer(args)

        output_path = get_output_path(args)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output path: {output_path}")

        # Baseline methods (QuantKVCache is None) use the model's default cache.
        kv_cache_factory = (
            make_kv_cache_factory(QuantKVCache, args)
            if QuantKVCache is not None
            else None
        )
        ruler_model = create_ruler_model(model, tokenizer, kv_cache_factory, args)

        results = run_evaluation(ruler_model, args, output_path)

        save_results(results, args, output_path)
        print_results_summary(results)

        logger.info("\nEVALUATION COMPLETED SUCCESSFULLY!")
        return 0

    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())




