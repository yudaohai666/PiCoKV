"""LongBench runner for the QuanPC quantized KV-cache algorithm.

Adapted from ``benchmarks/Needle/run_bin.py``: loads a HuggingFace model
directly and injects a fresh ``QuantKVCache`` per request (``quantpc`` method),
or runs a dense full-precision baseline attn implementation
(``flash_attention_2`` / ``sdpa`` / ``eager``).

Run from anywhere as long as the repo root is on ``PYTHONPATH`` (see
``run_longbench.sh``) so that ``import quantpc`` resolves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# Make `import quantpc` / `from benchmarks.common import ...` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common import load_model, output_tag, resolve_offline_proj_path, seed_everything  # noqa: E402

# Config files live next to this script (config/*.json).
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Datasets whose prompts are fed raw (no chat template), matching upstream
# LongBench behaviour.
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def post_process(response: str, model_name: str) -> str:
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


class QuantPCModel:
    """Thin harness wrapper: delegates model construction to the shared loader.

    Holds the loaded model/tokenizer plus a ``make_cache`` factory (a zero-arg
    callable returning a fresh QuantKVCache for quantpc, or None for baselines).
    """

    def __init__(
        self,
        model_name: str,
        method: str = "quantpc",
        # quantpc KV-cache params (--method quantpc)
        k_bits: int = 320,
        v_bits: int = 320,
        sink_size: int = 64,
        local_size: int = 256,
        exp_bits: int = 4,
        eas: float = 0.01,
        zerorope: int = 20,
        use_pruning: bool = False,
        prune_mode: int = 3,
        log_eps: float = -10.0,
        # KIVI params (--method kivi)
        group_size: int = 32,
        residual_length: int = 32,
        # OScaR params (--method oscar) —— 默认 = INT2 官方配方
        num_bits: int = 2,
        quant_mode: str = "k-channel",
        kv_rotation: str = "hadamard",
        kv_norm: str = "1",
        residual_evict_size: int | None = 256,
        offline_v_hadamard: bool = True,
        # quantpc offline block-32 encoders (--method quantpc)
        offline_proj_path: str | None = None,
        # PolarQuant params (--method polar)
        polar_mode: str = "mixed",
        polar_rbits: int = 3,
        polar_tbits: int = 3,
        polar_vbits: int = 2,
    ):
        self.method = method
        self.model, self.tokenizer, self._make_cache = load_model(
            model_name,
            method=method,
            # quantpc KV-cache params (--method quantpc)
            k_bits=k_bits,
            v_bits=v_bits,
            sink_size=sink_size,
            local_size=local_size,
            exp_bits=exp_bits,
            eas=eas,
            zerorope=zerorope,
            use_pruning=use_pruning,
            prune_mode=prune_mode,
            log_eps=log_eps,
            offline_proj_path=offline_proj_path,
            # KIVI params (--method kivi)
            group_size=group_size,
            residual_length=residual_length,
            # OScaR params (--method oscar)
            num_bits=num_bits,
            quant_mode=quant_mode,
            kv_rotation=kv_rotation,
            kv_norm=kv_norm,
            residual_evict_size=residual_evict_size,
            offline_v_hadamard=offline_v_hadamard,
            # PolarQuant params (--method polar)
            polar_mode=polar_mode,
            polar_rbits=polar_rbits,
            polar_tbits=polar_tbits,
            polar_vbits=polar_vbits,
        )

    def build_cache(self):
        """Fresh QuantKVCache per request for quantpc; None for baselines."""
        return self._make_cache() if self._make_cache is not None else None


def build_input(model: QuantPCModel, prompt: str, dataset: str, device):
    """Tokenize a prompt, applying the chat template for QA-style datasets."""
    if dataset not in NO_CHAT_TEMPLATE and model.tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        input_ids = model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(device)
        return {"input_ids": input_ids}
    return model.tokenizer(prompt, truncation=False, return_tensors="pt").to(device)


@torch.no_grad()
def generate_pred(model: QuantPCModel, prompt: str, dataset: str, max_gen: int) -> str:
    device = model.model.device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inputs = build_input(model, prompt, dataset, device)
    context_length = inputs["input_ids"].shape[-1]

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
    )
    if dataset == "samsum":
        generate_kwargs["min_length"] = context_length + 1
        generate_kwargs["eos_token_id"] = [
            model.tokenizer.eos_token_id,
            model.tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]

    # quantpc injects a fresh QuantKVCache per request; baselines use the
    # model's default cache (no past_key_values override).
    past_key_values = model.build_cache()
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values

    output = model.model.generate(**generate_kwargs)[0]

    # Some quantpc backends keep per-request state; reset if available.
    if hasattr(model.model, "reset"):
        model.model.reset()

    pred = model.tokenizer.decode(output[context_length:], skip_special_tokens=True)

    del past_key_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def run_dataset(model, dataset, data_all, prompt_format, max_gen, max_length,
                model_key, out_path):
    """Iterate one dataset, writing one jsonl line per sample (with resume)."""
    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            start_idx = len(f.readlines())
        if start_idx >= len(data_all):
            print(f"[skip] {dataset} already complete ({start_idx}/{len(data_all)})")
            return
        if start_idx > 0:
            print(f"[resume] {dataset} from {start_idx}/{len(data_all)}")

    tokenizer = model.tokenizer
    for idx in tqdm(range(start_idx, len(data_all)), desc=f"Processing {dataset}"):
        json_obj = data_all[idx]
        prompt = prompt_format.format(**json_obj)

        # Middle-truncation to max_length (upstream LongBench behaviour).
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = max_length // 2
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        pred = generate_pred(model, prompt, dataset, max_gen)
        pred = post_process(pred, model_key)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run LongBench with quantized KV cache.")
    p.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--method",
        default="quantpc",
        help="'quantpc' / 'kivi' / 'oscar' / 'polar' (quantized KV cache) or a dense "
             "baseline attn impl ('flash_attention_2' / 'sdpa' / 'eager')",
    )
    p.add_argument("--save_dir", default="results/longbench", help="base output dir")
    p.add_argument("--output-tag", default=None,
                   help="explicit output subdir name under save_dir/<model>")
    p.add_argument("--output-path", default=None,
                   help="explicit output dir, bypassing save_dir/<model>/<tag>. "
                        "Needed when one method is split across processes, which "
                        "would otherwise share a save_path.")
    p.add_argument("--seed", type=int, default=42)
    # ── quantpc KV-cache params (--method quantpc) ──
    p.add_argument("--k-bits", type=int, default=None, help="QuanPC K projection bits; KIVI defaults to 2")
    p.add_argument("--v-bits", type=int, default=None, help="QuanPC V projection bits; KIVI defaults to 2")
    p.add_argument("--sink-size", type=int, default=64, help="initial (sink) tokens kept in full precision")
    p.add_argument("--local-size", type=int, default=256, help="most-recent (local) tokens kept in full precision")
    p.add_argument("--exp-bits", type=int, default=4)
    p.add_argument("--eas", type=float, default=0.01)
    p.add_argument("--zerorope", type=int, default=20)
    p.add_argument("--pruning", dest="use_pruning", action=argparse.BooleanOptionalAction, default=False,
                   help="decode-time global pruning (gqa4 CUDA path); off by default")
    p.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    p.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N samples of each dataset")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset subset (default: config file list)")
    p.add_argument("--data-dir", required=True,
                   help="directory holding the unzipped LongBench *.jsonl files")
    # ── KIVI params (--method kivi; also uses --k-bits/--v-bits as 2/4-bit KV) ──
    p.add_argument("--group-size", type=int, default=32, help="KIVI per-group quant size; OScaR group size (INT2 必须=32)")
    p.add_argument("--residual-length", type=int, default=32, help="KIVI recent fp16 tokens kept unquantized")
    # ── OScaR params (--method oscar) —— 默认 = INT2 官方配方 ──
    p.add_argument("--num-bits", type=int, default=2, help="OScaR KV quant bits (INT2 默认)")
    p.add_argument("--quant-mode", default="k-channel", help="OScaR quant mode")
    p.add_argument("--kv-rotation", default="hadamard", help="OScaR kv rotation (INT2 默认 hadamard)")
    p.add_argument("--kv-norm", default="1", help="OScaR kv norm (INT2 默认 1)")
    p.add_argument("--residual-evict-size", type=int, default=256, help="OScaR residual evict size (INT2 默认 256)")
    p.add_argument("--offline-v-hadamard", action=argparse.BooleanOptionalAction, default=True, help="OScaR: offline per-head Hadamard into v_proj/o_proj (INT2 默认开启)")
    p.add_argument("--offline", action="store_true", help="quantpc: use offline encoders (default path offline_projection/projections/{model}_k{K}_v{V}.pt)")
    p.add_argument("--offline-proj-path", default=None, help="quantpc: explicit offline encoder .pt (overrides the --offline default path)")
    # ── PolarQuant params (--method polar) ──
    p.add_argument("--polar-mode", default="mixed", choices=["mixed", "group", "snapkv"],
                   help="PolarQuant attention variant")
    p.add_argument("--polar-rbits", type=int, default=3)
    p.add_argument("--polar-tbits", type=int, default=3)
    p.add_argument("--polar-vbits", type=int, default=2)
    args = p.parse_args(argv)
    if args.k_bits is None:
        args.k_bits = 2 if args.method == "kivi" else 320
    if args.v_bits is None:
        args.v_bits = 2 if args.method == "kivi" else 320
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model2maxlen = json.load(open(CONFIG_DIR / "model2maxlen.json"))
    datasets = json.load(open(CONFIG_DIR / "longbench_eval_subdatasets.json"))["datasets"]
    if args.datasets:
        datasets = [d for d in args.datasets.split(",") if d]
    dataset2prompt = json.load(open(CONFIG_DIR / "dataset2prompt.json"))
    dataset2maxlen = json.load(open(CONFIG_DIR / "dataset2maxlen.json"))

    model_key = Path(args.model).name if "/" in args.model else args.model
    max_length = model2maxlen.get(model_key, 31500)
    print(f"Model: {args.model} | Method: {args.method} | max_length: {max_length}")

    tag = args.output_tag or output_tag(
        args.method, args.k_bits, args.v_bits, args.num_bits,
        offline=(args.offline or args.offline_proj_path is not None),
        full_tag=(args.method == "quantpc"),
    )
    save_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.save_dir) / model_key / tag
    )
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {save_path}")

    offline_proj_path = resolve_offline_proj_path(
        args.method, args.offline, args.offline_proj_path,
        model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
    )
    model = QuantPCModel(
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
        offline_proj_path=offline_proj_path,
        polar_mode=args.polar_mode,
        polar_rbits=args.polar_rbits,
        polar_tbits=args.polar_tbits,
        polar_vbits=args.polar_vbits,
    )

    for dataset in datasets:
        print("=" * 50)
        print(f"Evaluating dataset: {dataset}")
        # datasets>=4 dropped script-based loaders, so the official LongBench
        # loader no longer works. Read the same jsonl the release ships instead.
        local_jsonl = Path(args.data_dir) / f"{dataset}.jsonl"
        if not local_jsonl.exists():
            raise FileNotFoundError(
                f"{local_jsonl} missing; unzip THUDM/LongBench data.zip into --data-dir"
            )
        with open(local_jsonl, encoding="utf-8") as handle:
            data_all = [json.loads(line) for line in handle if line.strip()]
        if args.limit:
            data_all = data_all[: int(args.limit)]
        out_path = str(save_path / f"{dataset}.jsonl")
        run_dataset(
            model, dataset, data_all,
            dataset2prompt[dataset], dataset2maxlen[dataset],
            max_length, model_key, out_path,
        )
        print(f"Finished {dataset} -> {out_path}")


if __name__ == "__main__":
    main()
