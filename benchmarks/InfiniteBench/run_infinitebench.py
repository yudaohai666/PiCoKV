#!/usr/bin/env python3
"""InfiniteBench evaluation runner for the QuanPC quantized KV-cache algorithm.

Mirrors benchmarks/Ruler/run_ruler_evaluation.py in HOW the QuanPC method is
applied:
  * load a plain HF model + tokenizer via ``benchmarks.common.load_model``
  * for ``--method quantpc``, ``modify_llama_for_standalone`` patches the
    attention and a fresh ``StandaloneLlamaCache`` is injected as
    ``past_key_values`` for every generation
  * baseline methods (flash_attention_2 / sdpa / eager / kivi / oscar / polar)
    reuse the same dispatch

InfiniteBench (Zhang et al., 2024) is a long-context benchmark with average
context length >100K tokens. Unlike RULER it does not sweep sequence lengths:
each task ships fixed-length examples from ``xinrongzhang2022/InfiniteBench``
on the HuggingFace Hub. Per-task ``max_gen_toks`` are declared in the task
yamls (5..1200) and honoured by ``MyModel.generate_until``.

Usage:
    python benchmarks/InfiniteBench/run_infinitebench.py \\
        --model /path/to/Llama-3.1-8B-Instruct \\
        --tasks infinitebench_passkey,infinitebench_longbook_qa_en \\
        --max_length 131072 \\
        --k-bits 320 --v-bits 320
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch  # noqa: F401  (kept for parity with run_ruler_evaluation.py)
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


# The 11 InfiniteBench tasks lm_eval ships (math_calc is excluded upstream due
# to evaluation complexity). Kept as a comma-separated string so it plugs
# straight into --tasks.
DEFAULT_TASKS = ",".join((
    "infinitebench_passkey",
    "infinitebench_kv_retrieval",
    "infinitebench_number_string",
    "infinitebench_code_run",
    "infinitebench_code_debug",
    "infinitebench_math_find",
    "infinitebench_longdialogue_qa_en",
    "infinitebench_longbook_qa_en",
    "infinitebench_longbook_sum_en",
    "infinitebench_longbook_choice_en",
    "infinitebench_longbook_qa_chn",
))


def parse_arguments():
    """Parse command line arguments (mirrors run_ruler_evaluation.parse_arguments)."""
    parser = argparse.ArgumentParser(
        description="Run InfiniteBench evaluation with the QuanPC quantized KV cache"
    )

    # Model + method dispatch (handled by benchmarks.common.load_model).
    parser.add_argument("--model", type=str, default="/data/Llama-3.1-8B-Instruct",
                        help="Path or name of the HF model to evaluate")
    parser.add_argument("--method", type=str, default="quantpc",
                        help=("evaluation method: 'quantpc' / 'kivi' / 'oscar' / "
                              "'polar' (quantized KV cache), or a dense baseline attn "
                              "impl 'flash_attention_2' / 'sdpa' / 'eager'"))

    # ── quantpc KV-cache params (--method quantpc) ──
    parser.add_argument("--k-bits", type=int, default=None,
                        help="QuanPC K projection bits; KIVI defaults to 2")
    parser.add_argument("--v-bits", type=int, default=None,
                        help="QuanPC V projection bits; KIVI defaults to 2")
    parser.add_argument("--sink-size", type=int, default=64)
    parser.add_argument("--local-size", type=int, default=256)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=0)
    parser.add_argument("--no-pruning", dest="use_pruning", action="store_false")
    parser.add_argument("--prune-mode", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=-10.0)
    parser.set_defaults(use_pruning=True)

    # ── KIVI params (--method kivi; also uses --k-bits/--v-bits as 2/4-bit KV) ──
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)

    # ── OScaR params (--method oscar) ──
    parser.add_argument("--num-bits", type=int, default=2)
    parser.add_argument("--quant-mode", default="k-channel")
    parser.add_argument("--kv-rotation", default="hadamard")
    parser.add_argument("--kv-norm", default="1")
    parser.add_argument("--residual-evict-size", type=int, default=256)
    parser.add_argument("--offline-v-hadamard",
                        action=argparse.BooleanOptionalAction, default=True)

    # ── quantpc offline block-32 encoders (--method quantpc) ──
    parser.add_argument("--offline", action="store_true",
                        help="quantpc: use offline encoders "
                             "(default path offline_projection/projections/"
                             "{model}_k{K}_v{V}_nowhiten.pt)")
    parser.add_argument("--online-decoder-refit", default="none",
                        choices=("none", "k", "v", "both"),
                        help="quantpc offline only: online decoder refit side")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--offline-proj-path", default=None,
                        help="quantpc: explicit offline encoder .pt")

    # ── PolarQuant params (--method polar) ──
    parser.add_argument("--polar-mode", default="mixed",
                        choices=["mixed", "group", "snapkv"])
    parser.add_argument("--polar-rbits", type=int, default=3)
    parser.add_argument("--polar-tbits", type=int, default=3)
    parser.add_argument("--polar-vbits", type=int, default=2)

    # Task configuration. Unlike RULER, InfiniteBench does not sweep lengths --
    # each task carries fixed-length examples from the shipped dataset.
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help="Comma-separated task list (default: all 11 tasks)")

    # Runtime configuration.
    parser.add_argument("--max_length", type=int, default=131072,
                        help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=1)
    # 1200 is the ceiling used by longbook_sum_en; every other InfiniteBench
    # task overrides this via its yaml (5..50). MyModel.generate_until reads
    # per-task max_gen_toks from gen_kwargs first and only falls back here.
    parser.add_argument("--max_gen_toks", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (for testing)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Output configuration.
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results (default: auto-generated)")
    parser.add_argument("--write_out", action="store_true", default=True)

    # Chat template configuration.
    parser.add_argument("--apply_chat_template",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fewshot_as_multiturn", action="store_true", default=False)

    # Tokenizer options.
    parser.add_argument("--add_bos_token", action="store_true")
    parser.add_argument("--prefix_token_id", type=int, default=None)

    # Match run_ruler_evaluation: allow overriding the online block-32 decoder
    # ridge without editing the shipped constant.
    parser.add_argument("--decoder-ridge", type=float, default=1e-2)

    args = parser.parse_args()
    if args.k_bits is None:
        args.k_bits = 2 if args.method == "kivi" else 320
    if args.v_bits is None:
        args.v_bits = 2 if args.method == "kivi" else 320
    return args


def run_tag(args):
    """Reproducible run tag (same scheme as run_ruler_evaluation.run_tag)."""
    offline = args.offline or args.offline_proj_path is not None
    whiten = args.whiten
    if offline and args.offline_proj_path:
        from quantpc.quanpc_standalone.transformers import load_offline_projections
        meta = load_offline_projections(args.offline_proj_path).get("meta", {})
        whiten = bool(meta.get("whitened"))
    if args.method == "quantpc":
        mode = "offline" if offline else "online"
        tag = (f"quantpc_{mode}_k{args.k_bits}_v{args.v_bits}_"
               f"{'whiten' if whiten else 'nowhiten'}")
        if offline:
            suffix = {"k": "krefit", "v": "vrefit", "both": "refit"}.get(
                args.online_decoder_refit)
            if suffix:
                tag += f"_{suffix}"
        if float(args.decoder_ridge) != 1e-2:
            tag += f"_ridge{float(args.decoder_ridge):g}"
        if args.exp_bits != 4:
            tag += f"_exp{args.exp_bits}"
        if args.zerorope != 20:
            tag += f"_zr{args.zerorope}"
        if float(args.eas) != 0.01:
            tag += f"_eas{float(args.eas):g}"
        return tag
    return output_tag(args.method, args.k_bits, args.v_bits, args.num_bits)


def get_output_path(args):
    """Auto-generate one output directory for the complete configuration."""
    if args.output_path:
        return args.output_path
    model_key = Path(args.model).name
    return f"results/infinitebench/{model_key}/{run_tag(args)}/"


def create_ib_model(model, tokenizer, kv_cache_factory, args):
    """Create the MyModel wrapper wired with the KV-cache factory."""
    ib_model = MyModel(
        model=model,
        tokenizer=tokenizer,
        model_max_length=args.max_length,
        batch_size=args.batch_size,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        max_gen_toks=args.max_gen_toks,
        kv_cache_factory=kv_cache_factory,
    )
    logger.info(f"MyModel created (max_length={args.max_length}, "
                f"batch_size={args.batch_size}, max_gen_toks={args.max_gen_toks})")
    return ib_model


def _patch_datasets_features():
    """Convert dict-shorthand ``features`` into a real ``datasets.Features``.

    lm_eval (0.4.12) forwards ``dataset_kwargs.features`` from the task yaml to
    ``datasets.load_dataset`` verbatim. datasets>=4 requires a ``Features``
    object there; a plain dict crashes deep in the arrow writer with
    ``'dict' object has no attribute 'arrow_schema'``. The InfiniteBench yamls
    must declare features (schema inference fails with "Couldn't cast string to
    null" on empty ``options``), so we wrap ``load_dataset`` and upgrade the
    shorthand via ``Features.from_dict`` before the real call.
    """
    import datasets

    orig = datasets.load_dataset

    def load_dataset(*a, **kw):
        feats = kw.get("features")
        if isinstance(feats, dict) and not isinstance(feats, datasets.Features):
            kw["features"] = datasets.Features.from_dict(feats)
        return orig(*a, **kw)

    datasets.load_dataset = load_dataset


def run_evaluation(ib_model, args):
    """Run the requested InfiniteBench tasks in a single evaluator call.

    Unlike RULER (which sweeps sequence lengths and stitches metrics per length),
    InfiniteBench tasks are keyed by dataset split, not by context length, so
    one simple_evaluate call is sufficient.
    """
    _patch_datasets_features()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    logger.info(f"Evaluating tasks: {tasks}")

    # lm_eval's bundled infinitebench yamls declare dataset_kwargs.features in a
    # legacy shorthand that current `datasets` no longer parses (crashes with
    # "'str' object has no attribute 'items'"). We ship fixed copies under
    # tasks/infinitebench/ and register them via include_path so they take
    # precedence over the broken built-ins.
    from lm_eval.tasks import TaskManager
    local_tasks = str(Path(__file__).resolve().parent / "tasks")
    task_manager = TaskManager(include_path=local_tasks)

    metadata = {"pretrained": args.model, "tokenizer": args.model}
    model_args_str = f"pretrained={args.model},max_length={args.max_length}"
    return evaluator.simple_evaluate(
        model=ib_model,
        tasks=tasks,
        task_manager=task_manager,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        write_out=args.write_out,
        log_samples=True,
        metadata=metadata,
        model_args=model_args_str,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )


def save_results(results, args, output_path):
    """Persist a lite results.json plus per-sample predictions.jsonl."""
    results_lite = {
        "results": results.get("results", {}),
        "config": results.get("config", {}),
        # lm_eval's own config only records model+max_length; keep run_config so
        # method/bits/etc. are recoverable from the results directory.
        "run_config": vars(args),
    }
    results_file = os.path.join(output_path, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_lite, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_file}")

    if "samples" in results:
        jsonl_file = os.path.join(output_path, "predictions.jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for task_name, task_samples in results["samples"].items():
                for sample in task_samples:
                    doc = sample.get("doc", {})
                    pred_data = {
                        "task": task_name,
                        "pred": sample.get("resps", [""])[0] if sample.get("resps") else "",
                        "target": sample.get("target", ""),
                        "doc_id": doc.get("id"),
                        # InfiniteBench docs carry a short `input` (question) plus a
                        # 100K+-token `context`; keep `input` for debugging, drop
                        # `context` so predictions.jsonl stays scannable.
                        "input": doc.get("input", ""),
                        "answer": doc.get("answer", []),
                    }
                    f.write(json.dumps(pred_data, ensure_ascii=False) + "\n")
        logger.info(f"Predictions saved to: {jsonl_file}")


def print_results_summary(results):
    """Log a formatted per-task metric summary."""
    logger.info("\n" + "=" * 60)
    logger.info("INFINITEBENCH EVALUATION RESULTS SUMMARY")
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
    """Load model, run InfiniteBench, and persist results."""
    args = parse_arguments()
    seed_everything(args.seed)

    if args.method == "quantpc":
        # The online solve reads this as a module global; rebinding leaves
        # the shipped constant alone but changes what this run fits.
        import quantpc.quanpc_standalone.online as online_mod
        online_mod.ONLINE_DECODER_RIDGE = float(args.decoder_ridge)
        logger.info(f"online decoder ridge overridden to {args.decoder_ridge}")

    logger.info("InfiniteBench Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Method: {args.method}")
    if args.method == "quantpc":
        logger.info("Backend: CUDA (quantpc.quanpc_standalone)")
        logger.info(f"KV cache: k_bits={args.k_bits}, v_bits={args.v_bits}, "
                    f"exp_bits={args.exp_bits}, eas={args.eas}, "
                    f"zerorope={args.zerorope}")
    logger.info(f"Tasks: {args.tasks}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")

    try:
        offline_proj_path = resolve_offline_proj_path(
            args.method, args.offline, args.offline_proj_path,
            model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
            whiten=args.whiten,
        )
        model, tokenizer, kv_cache_factory = load_model(
            args.model,
            method=args.method,
            k_bits=args.k_bits, v_bits=args.v_bits,
            sink_size=args.sink_size, local_size=args.local_size,
            exp_bits=args.exp_bits, eas=args.eas, zerorope=args.zerorope,
            use_pruning=args.use_pruning, prune_mode=args.prune_mode,
            log_eps=args.log_eps,
            offline_proj_path=offline_proj_path,
            online_decoder_refit=args.online_decoder_refit,
            whiten=args.whiten,
            group_size=args.group_size,
            residual_length=args.residual_length,
            num_bits=args.num_bits,
            quant_mode=args.quant_mode,
            kv_rotation=args.kv_rotation,
            kv_norm=args.kv_norm,
            residual_evict_size=args.residual_evict_size,
            offline_v_hadamard=args.offline_v_hadamard,
            polar_mode=args.polar_mode,
            polar_rbits=args.polar_rbits,
            polar_tbits=args.polar_tbits,
            polar_vbits=args.polar_vbits,
        )
        # InfiniteBench harness may pad batches; ensure a pad token exists.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Baseline methods yield kv_cache_factory=None -> model's default cache.
        ib_model = create_ib_model(model, tokenizer, kv_cache_factory, args)

        output_path = get_output_path(args)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output path: {output_path}")

        results = run_evaluation(ib_model, args)
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
