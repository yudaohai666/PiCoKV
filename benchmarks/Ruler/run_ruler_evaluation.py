
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
import sys
from pathlib import Path

import torch
from lm_eval import evaluator
from loguru import logger

# Make `import src.lmeval` / `import quantpc` / `benchmarks.common` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lmeval import MyModel  # noqa: E402
from benchmarks.common import (  # noqa: E402
    load_quantpc_model,
    output_tag,
    seed_everything,
)


def _make_local_read_hotpotqa(module):
    """Build a read_hotpotqa bound to `module` that prefers the local JSON file.

    Reuses the module's own `download_json` for the network fallback so behavior
    matches upstream when HOTPOTQA_JSON is absent.
    """
    from functools import cache

    @cache
    def read_hotpotqa(
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    ):
        local = os.environ.get("HOTPOTQA_JSON")
        if local and os.path.exists(local):
            logger.info(f"Loading hotpotqa from local file: {local}")
            with open(local, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            logger.info(f"HOTPOTQA_JSON not set/found; downloading from {url}")
            data = module.download_json(url)

        total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
        total_docs = sorted(list(set(total_docs)))
        total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

        total_qas = []
        for d in data:
            total_qas.append(
                {
                    "query": d["question"],
                    "outputs": [d["answer"]],
                    "context": [
                        total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                    ],
                }
            )
        return total_qas, total_docs

    return read_hotpotqa


def patch_ruler_hotpotqa_local():
    """Redirect RULER hotpotqa loading to a local JSON file.

    Upstream ``lm_eval.tasks.ruler.qa_utils.read_hotpotqa`` hardcodes the CMU URL
    (http://curtis.ml.cmu.edu/...), which is frequently 502. When ``HOTPOTQA_JSON``
    points to an existing file, load from disk instead of the network.

    lm_eval resolves ``!function qa_utils.get_hotpotqa`` through
    ``lm_eval.tasks._yaml_loader._load_module_with_cache``, which re-executes
    qa_utils.py into a fresh module object at task-load time. Patching the
    already-imported module is therefore useless (it gets overwritten). Instead
    we wrap the loader so that whatever qa_utils module object it returns gets our
    local-file read_hotpotqa injected before the task uses it (reload-proof).
    """
    from lm_eval.tasks import _yaml_loader as yl

    if getattr(yl, "_hotpotqa_local_patched", False):
        return

    orig_loader = yl._load_module_with_cache

    def loader(module_path):
        module = orig_loader(module_path)
        norm = str(module_path).replace("\\", "/")
        if norm.endswith("/ruler/qa_utils.py"):
            module.read_hotpotqa = _make_local_read_hotpotqa(module)
        return module

    yl._load_module_with_cache = loader
    yl._hotpotqa_local_patched = True


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
    #   (dispatch handled by benchmarks.common.load_quantpc_model)
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
    parser.add_argument(
        "--refit-decoder",
        action="store_true",
        help="enable decode-matrix refit (CUDA prefill falls back to torch)",
    )
    parser.add_argument(
        "--no-pruning",
        dest="use_pruning",
        action="store_false",
        help="disable decode-time global pruning (gqa4 CUDA path); pruning is on by default",
    )
    parser.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    parser.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    parser.set_defaults(use_pruning=True)

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


def get_output_path(args):
    """Auto-generate an output path keyed by model name and method."""
    if args.output_path:
        return args.output_path
    model_key = Path(args.model).name
    tag = output_tag(args.method, args.k_bits, args.v_bits, args.refit_decoder)
    return f"results/ruler/{model_key}_{tag}/"


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
        # Override lm_eval's default seeds (0 / 1234) so args.seed applies
        # consistently across python/numpy/torch and the fewshot sampler.
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    return results


def save_results(results, args, output_path):
    """Save evaluation results to files."""
    model_key = Path(args.model).name
    tag = output_tag(args.method, args.k_bits, args.v_bits, args.refit_decoder)

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

    # Use local hotpotqa file (HOTPOTQA_JSON) instead of the flaky CMU URL.
    patch_ruler_hotpotqa_local()

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
        model, tokenizer, kv_cache_factory = load_quantpc_model(
            args.model,
            method=args.method,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            projected_query_bits=args.projected_query_bits,
            exp_bits=args.exp_bits,
            eas=args.eas,
            zerorope=args.zerorope,
            cuda_v1_exp8=args.cuda_v1_exp8,
            refit_decoder=args.refit_decoder,
            use_pruning=args.use_pruning,
            prune_mode=args.prune_mode,
            log_eps=args.log_eps,
        )
        # RULER's lm_eval harness pads batches; ensure a pad token exists.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        output_path = get_output_path(args)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output path: {output_path}")

        # Baseline methods yield kv_cache_factory=None -> model's default cache.
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




