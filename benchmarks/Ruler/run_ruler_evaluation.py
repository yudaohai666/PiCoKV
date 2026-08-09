
#!/usr/bin/env python3
"""RULER evaluation runner for the QuanPC quantized KV-cache algorithm.

Mirrors benchmarks/Needle/run_bin.py in HOW the QuanPC method is applied:
  * load a plain HF model + tokenizer
  * modify(model) monkey-patches the attention
  * a fresh QuantKVCache is injected as past_key_values for every generation

The QuanPC method uses the CUDA-only quanpc_standalone backend.

Usage:
    python benchmarks/Ruler/run_ruler_evaluation.py \
        --model /path/to/Llama-3.1-8B-Instruct \
        --tasks ruler \
        --max_seq_lengths 131072 \
        --max_length 131072 \
        --k-bits 256 --v-bits 320 \
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
    load_model,
    output_tag,
    resolve_offline_proj_path,
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

    # Method dispatch (handled by benchmarks.common.load_model):
    #   quantpc            -> quantized KV-cache method (modify + StandaloneLlamaCache)
    #   kivi               -> KIVI quantized KV cache
    #   oscar              -> OScaR quantized KV cache
    #   flash_attention_2  -> dense full-precision baseline (flash-attn-2)
    #   sdpa / eager       -> dense full-precision baseline
    parser.add_argument(
        "--method",
        type=str,
        default="quantpc",
        help=(
            "evaluation method: 'quantpc' / 'kivi' / 'oscar' (quantized KV cache), "
            "or a dense baseline attn impl 'flash_attention_2' / 'sdpa' / 'eager'"
        ),
    )

    # ── quantpc KV-cache params (--method quantpc) ──
    parser.add_argument("--k-bits", type=int, default=320, help="key projection bits (also KIVI KV bits)")
    parser.add_argument("--v-bits", type=int, default=256, help="value projection bits (also KIVI KV bits)")
    parser.add_argument("--sink-size", type=int, default=64, help="initial (sink) tokens kept in full precision")
    parser.add_argument("--local-size", type=int, default=256, help="most-recent (local) tokens kept in full precision")
    parser.add_argument(
        "--exp-bits",
        type=int,
        default=4,
        help="unsigned per-64-token quantization bits for global exp weights",
    )
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument(
        "--no-pruning",
        dest="use_pruning",
        action="store_false",
        help="disable decode-time global pruning (gqa4 CUDA path); pruning is on by default",
    )
    parser.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    parser.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    parser.set_defaults(use_pruning=True)

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
    parser.add_argument("--offline", action="store_true", help="quantpc: use offline block-32 encoders (default path offline_projection/projections/{model}_k{K}_v{V}_block32_folded.pt)")
    parser.add_argument("--offline-proj-path", default=None, help="quantpc: explicit offline encoder .pt (overrides the --offline default path)")
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="quantpc: rotate K/V into the query-whitened basis at runtime; a "
             "folded projection already has this baked into its stored pair",
    )

    # ── PolarQuant params (--method polar) ──
    parser.add_argument("--polar-mode", default="mixed", choices=["mixed", "group", "snapkv"], help="PolarQuant attention variant: mixed (K polar + V KIVI-quant) | group (K-only) | snapkv")
    parser.add_argument("--polar-rbits", type=int, default=3, help="PolarQuant K radius bits (rbits+tbits<=8)")
    parser.add_argument("--polar-tbits", type=int, default=3, help="PolarQuant K angle bits (rbits+tbits<=8)")
    parser.add_argument("--polar-vbits", type=int, default=2, help="PolarQuant V bits (mixed only; KIVI kernel supports 2 or 4)")

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
    tag = output_tag(args.method, args.k_bits, args.v_bits, args.num_bits,
                     offline=(args.offline or args.offline_proj_path is not None))
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


def _base_len(metric_key):
    """Return the leading sequence-length int of an lm_eval metric key, or None.

    RULER metric keys look like "16384,none" or "16384_stderr,none"; the length
    is the leading integer before the first ``,`` / ``_``.
    """
    head = metric_key.split(",")[0].split("_")[0]
    try:
        return int(head)
    except ValueError:
        return None


def _merge_length_results(per_length):
    """Merge single-length simple_evaluate outputs into one results dict.

    Each per-length call only yields real metric values for its own length (all
    other lengths aggregate to -1), so for every task we keep exactly the metric
    keys whose leading length equals the length that call was run at. ``config``
    is taken from the first call; ``samples`` are concatenated across lengths so
    the predictions .jsonl still spans every length.
    """
    merged_results, merged_samples, config = {}, {}, {}
    for length, res in per_length:
        if not config:
            config = res.get("config", {})
        for task, metrics in res.get("results", {}).items():
            tgt = merged_results.setdefault(task, {})
            for mk, mv in metrics.items():
                if _base_len(mk) == length:
                    tgt[mk] = mv
        for task, samps in res.get("samples", {}).items():
            merged_samples.setdefault(task, []).extend(samps)
    return {"results": merged_results, "config": config, "samples": merged_samples}


def run_evaluation(ruler_model, args, output_path):
    """Run RULER evaluation via lm_eval.

    lm_eval concatenates every sequence length into a single length-ascending doc
    list per task, and ``--limit N`` truncates that combined list from the front,
    so one global limit lands entirely in the shortest length. To make ``--limit``
    mean N samples PER length, evaluate one length at a time (each call regenerates
    only that length's synthetic docs, so the limit applies within it) and merge
    the per-length metrics back into one results dict.
    """
    tasks = [task.strip() for task in args.tasks.split(",")]
    max_seq_lengths = [int(x.strip()) for x in args.max_seq_lengths.split(",")]

    logger.info(f"Evaluating tasks: {tasks} with sequence lengths: {max_seq_lengths}")

    max_seq_length_needed = max(max_seq_lengths)
    if args.max_length < max_seq_length_needed:
        logger.warning(
            f"Model max_length ({args.max_length}) < longest sequence "
            f"({max_seq_length_needed})"
        )

    model_args_str = f"pretrained={args.model},max_length={args.max_length}"

    os.makedirs(output_path, exist_ok=True)

    logger.info("Starting RULER evaluation...")
    if args.apply_chat_template:
        logger.info("Chat template enabled")
    if args.limit:
        logger.info(f"Per-length sample limit: {args.limit} (evaluated one length at a time)")

    per_length = []
    for length in max_seq_lengths:
        logger.info(f"=== RULER length {length} (limit={args.limit} samples/length) ===")
        metadata = {
            "max_seq_lengths": [length],
            "pretrained": args.model,
            "tokenizer": args.model,
        }
        res = evaluator.simple_evaluate(
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
        per_length.append((length, res))

    return _merge_length_results(per_length)


def save_results(results, args, output_path):
    """Save evaluation results to files."""
    model_key = Path(args.model).name
    tag = output_tag(args.method, args.k_bits, args.v_bits, args.num_bits,
                     offline=(args.offline or args.offline_proj_path is not None))

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
        logger.info("Backend: CUDA (quantpc.quanpc_standalone)")
        logger.info(f"KV cache: k_bits={args.k_bits}, v_bits={args.v_bits}, "
                    f"exp_bits={args.exp_bits}, eas={args.eas}, zerorope={args.zerorope}, "
                    f"whiten={args.whiten}")
    logger.info(f"Tasks: {args.tasks}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")

    try:
        offline_proj_path = resolve_offline_proj_path(
            args.method, args.offline, args.offline_proj_path,
            model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
        )
        # A folded projection already carries the whitening in its stored pair;
        # rotating again at runtime would account for the same basis twice.
        if offline_proj_path is not None and args.whiten:
            from quantpc.quanpc_standalone.transformers import load_offline_projections
            if load_offline_projections(offline_proj_path).get("meta", {}).get("whitened"):
                raise SystemExit(
                    f"{offline_proj_path} already has the whitening folded in; drop --whiten."
                )
        model, tokenizer, kv_cache_factory = load_model(
            args.model,
            method=args.method,
            # quantpc KV-cache params (--method quantpc)
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
            whiten=args.whiten,
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




