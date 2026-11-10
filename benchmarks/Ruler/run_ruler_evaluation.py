
#!/usr/bin/env python3
"""RULER evaluation runner for the PiCoKV quantized KV-cache algorithm.

Mirrors benchmarks/Needle/run_bin.py in HOW the PiCoKV method is applied:
  * load a plain HF model + tokenizer
  * modify(model) monkey-patches the attention
  * a fresh QuantKVCache is injected as past_key_values for every generation

The PiCoKV method uses the CUDA-only picokv_standalone backend.

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

# Make `import src.lmeval` / `import picokv` / `benchmarks.common` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lmeval import MyModel  # noqa: E402
from benchmarks.common import (  # noqa: E402
    BPC_DEFAULT_OFFLINE_PROJ_PATH,
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
        local = os.environ.get("HOTPOTQA_JSON", str(
            Path(__file__).resolve().parent / "ruler_qa" / "hotpot_dev_distractor_v1.json"
        ))
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
        description="Run RULER evaluation with the PiCoKV quantized KV cache"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="/data/Llama-3.1-8B-Instruct",
        help="Path or name of the HF model to evaluate",
    )

    # Method dispatch (handled by benchmarks.common.load_model):
    #   picokv            -> quantized KV-cache method (modify + StandaloneLlamaCache)
    #   kivi               -> KIVI quantized KV cache
    #   oscar              -> OScaR quantized KV cache
    #   bitdecoding        -> BitDecoding low-bit KV cache (4-bit/g128 only; see
    #                         common._load_bitdecoding_model)
    #   flash_attention_2  -> dense full-precision baseline (flash-attn-2)
    #   sdpa / eager       -> dense full-precision baseline
    parser.add_argument(
        "--method",
        type=str,
        default="picokv",
        help=(
            "evaluation method: 'picokv' (CUDA kernel) / 'picokv_recon' (dequant+flash ablation, "
            "see --quant-target) / 'kivi' / 'oscar' / 'polar' / 'bitdecoding' (quantized KV cache), "
            "or a dense baseline attn impl 'flash_attention_2' / 'sdpa' / 'eager'"
        ),
    )

    # ── picokv KV-cache params (--method picokv) ──
    parser.add_argument("--k-bits", type=int, default=None, help="PiCoKV K projection bits; KIVI defaults to 2")
    parser.add_argument("--v-bits", type=int, default=None, help="PiCoKV V projection bits; KIVI defaults to 2")
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
        "--pruning",
        dest="use_pruning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="decode-time global pruning (gqa4 CUDA path); off by default",
    )
    parser.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    parser.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    parser.add_argument(
        "--quant-target",
        default="both",
        choices=["key", "value", "both", "none", "dense_zr"],
        help="picokv_recon: which side(s) to quantize+reconstruct (dequant+flash, no CUDA kernel); 'dense_zr'=dense zerorope-only ablation (no quantization)",
    )

    # ── KIVI params (--method kivi; also uses --k-bits/--v-bits as 2/4-bit KV) ──
    parser.add_argument("--group-size", type=int, default=None, help="KIVI per-group quant size; OScaR group size (INT2 必须=32); BitDecoding 必须=128 (default 32, or 128 for bitdecoding)")
    parser.add_argument("--residual-length", type=int, default=32, help="KIVI recent fp16 tokens kept unquantized")

    # ── OScaR params (--method oscar) ──
    parser.add_argument("--num-bits", type=int, default=None, help="OScaR KV quant bits (INT2 默认); BitDecoding bits (default 2, or 4 for bitdecoding)")
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

    # ── BPC params (--method bpc); default = offline hash projection ──
    parser.add_argument("--bpc-mask-out", type=float, default=0.98, help="BPC: token_budget = context_len * (1 - mask_out)")
    parser.add_argument("--bpc-min-remain", type=int, default=128, help="BPC: minimum token budget floor")
    parser.add_argument("--bpc-token-budget", type=int, default=None, help="BPC: fixed token budget (overrides mask_out/min_remain)")
    parser.add_argument("--bpc-err-ratio", type=float, default=0.1, help="BPC: err_topk = token_budget * err_ratio")
    parser.add_argument("--bpc-fix-layers", default="0,1", help="BPC: comma-separated layer indices that skip sparse attention (empty string for none)")
    parser.add_argument("--bpc-use-cuda-kernel", action=argparse.BooleanOptionalAction, default=True, help="BPC: use the compiled CUDA kernel; falls back to PyTorch on failure")
    parser.add_argument("--bpc-offline-proj-path", default=BPC_DEFAULT_OFFLINE_PROJ_PATH, help="BPC: learned offline hash-projection .pt; pass '' for the online (random) projection")
    # ── BitDecoding params (--method bitdecoding); reuses --num-bits/--group-size/--quant-mode ──
    parser.add_argument("--bitdecoding-residual-block-size", type=int, default=None,
                        help="BitDecoding fp16 residual block (default 128 at 4-bit, 256 at 2-bit; must match the kernel's kBlockN_pack)")
    parser.add_argument("--bitdecoding-allow-unvalidated", action="store_true",
                        help="BitDecoding: allow a (num_bits, group_size) other than (4, 128). Those configs index the wrong "
                             "quantization scale (pack_num is derived from group_size), so they are EFFICIENCY-ONLY and must "
                             "NOT be used for a RULER accuracy row")

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
    parser.add_argument(
        "--apply_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the tokenizer chat template (enabled by default)",
    )
    parser.add_argument("--fewshot_as_multiturn", action="store_true", default=False)

    # Tokenizer options
    parser.add_argument("--add_bos_token", action="store_true")
    parser.add_argument("--prefix_token_id", type=int, default=None)

    args = parser.parse_args()
    if args.k_bits is None:
        args.k_bits = 2 if args.method == "kivi" else 320
    if args.v_bits is None:
        args.v_bits = 2 if args.method == "kivi" else 320
    # BitDecoding's only correct config is 4-bit/g128 (see _load_bitdecoding_model);
    # make that the default so the common case needs no extra flags.
    if args.num_bits is None:
        args.num_bits = 4 if args.method == "bitdecoding" else 2
    if args.group_size is None:
        args.group_size = 128 if args.method == "bitdecoding" else 32
    if args.bpc_offline_proj_path == "":
        args.bpc_offline_proj_path = None
    return args


def run_tag(args):
    """Return a reproducible tag for this evaluation configuration."""
    offline = args.offline or args.offline_proj_path is not None
    if args.method in ("picokv", "picokv_recon"):
        mode = "offline" if offline else "online"
        prefix = "picokv_recon" if args.method == "picokv_recon" else "picokv"
        tag = f"{prefix}_{mode}_k{args.k_bits}_v{args.v_bits}"
        if args.method == "picokv_recon":
            tag += f"_{args.quant_target}"
        if args.exp_bits != 4:
            tag += f"_exp{args.exp_bits}"
        if args.zerorope != 20:
            tag += f"_zr{args.zerorope}"
        if float(args.eas) != 0.01:
            tag += f"_eas{float(args.eas):g}"
        return tag
    return output_tag(args.method, args.k_bits, args.v_bits, args.num_bits,
                      group_size=args.group_size)


def get_output_path(args):
    """Auto-generate one output directory for the complete configuration."""
    if args.output_path:
        return args.output_path
    model_key = Path(args.model).name
    return f"results/ruler/{model_key}/{run_tag(args)}/"


def create_ruler_model(model, tokenizer, kv_cache_factory, args):
    """Create the MyModel wrapper wired with the PiCoKV KV-cache factory."""
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
    """Return the leading sequence-length integer from an lm_eval metric key."""
    head = metric_key.split(",")[0].split("_")[0]
    try:
        return int(head)
    except ValueError:
        return None


def _merge_length_results(per_length):
    """Merge per-length results while retaining only each call's real metrics."""
    merged_results, merged_samples, config = {}, {}, {}
    for length, res in per_length:
        if not config:
            config = res.get("config", {})
        for task, metrics in res.get("results", {}).items():
            target = merged_results.setdefault(task, {})
            for metric_key, metric_value in metrics.items():
                if _base_len(metric_key) == length:
                    target[metric_key] = metric_value
        for task, samples in res.get("samples", {}).items():
            merged_samples.setdefault(task, []).extend(samples)
    return {
        "results": merged_results,
        "config": config,
        "samples": merged_samples,
    }


def run_evaluation(ruler_model, args, output_path):
    """Run each requested length serially, then merge the results."""
    tasks = [task.strip() for task in args.tasks.split(",")]
    lengths = [int(x.strip()) for x in args.max_seq_lengths.split(",")]

    logger.info(f"Evaluating tasks: {tasks} with sequence lengths: {lengths}")

    if args.max_length < max(lengths):
        logger.warning(
            f"Model max_length ({args.max_length}) < longest sequence ({max(lengths)})"
        )

    model_args_str = f"pretrained={args.model},max_length={args.max_length}"
    os.makedirs(output_path, exist_ok=True)
    logger.info("Starting RULER evaluation...")
    if args.apply_chat_template:
        logger.info("Chat template enabled")

    per_length = []
    for length in lengths:
        logger.info(f"=== RULER length {length} (limit={args.limit} per task) ===")
        metadata = {
            "max_seq_lengths": [length],
            "pretrained": args.model,
            "tokenizer": args.model,
        }
        result = evaluator.simple_evaluate(
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
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        per_length.append((length, result))
    return _merge_length_results(per_length)


def save_results(results, args, output_path):
    """Save evaluation results to files."""
    results_lite = {
        "results": results.get("results", {}),
        "config": results.get("config", {}),
        # lm_eval's own config records only the model and max_length, so a run's
        # method settings were unrecoverable from its results. Keep them here.
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
    if args.method in ("picokv", "picokv_recon"):
        if args.method == "picokv_recon":
            logger.info(f"Backend: dequant+SDPA recon ablation (quant_target={args.quant_target})")
        else:
            logger.info("Backend: CUDA (picokv.picokv_standalone)")
        logger.info(f"KV cache: k_bits={args.k_bits}, v_bits={args.v_bits}, "
                    f"exp_bits={args.exp_bits}, eas={args.eas}, zerorope={args.zerorope}")
    logger.info(f"Tasks: {args.tasks}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")

    try:
        offline_proj_path = resolve_offline_proj_path(
            args.method, args.offline, args.offline_proj_path,
            model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
        )
        model, tokenizer, kv_cache_factory = load_model(
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
            quant_target=args.quant_target,
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
            # BPC params (--method bpc)
            bpc_mask_out=args.bpc_mask_out,
            bpc_min_remain=args.bpc_min_remain,
            bpc_token_budget=args.bpc_token_budget,
            bpc_err_ratio=args.bpc_err_ratio,
            bpc_fix_layers=[int(x) for x in args.bpc_fix_layers.split(",") if x],
            bpc_use_cuda_kernel=args.bpc_use_cuda_kernel,
            bpc_offline_proj_path=args.bpc_offline_proj_path,
            # BitDecoding params (--method bitdecoding; also uses --num_bits/--group_size/--quant_mode)
            bitdecoding_residual_block_size=args.bitdecoding_residual_block_size,
            bitdecoding_allow_unvalidated=args.bitdecoding_allow_unvalidated,
        )
        # RULER's lm_eval harness pads batches; ensure a pad token exists.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Baseline methods yield kv_cache_factory=None -> model's default cache.
        ruler_model = create_ruler_model(model, tokenizer, kv_cache_factory, args)

        output_path = get_output_path(args)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output path: {output_path}")

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
