"""LongBench runner for the QuanPC quantized KV-cache algorithm.

Adapted from ``benchmarks/Needle/run_bin_v4.py``: loads a HuggingFace model
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

from benchmarks.common import load_quantpc_model, output_tag, seed_everything  # noqa: E402

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
        k_bits: int = 128,
        v_bits: int = 256,
        projected_query_bits: int = 8,
        exp_bits: int = 4,
        eas: float = 0.01,
        zerorope: int = 30,
        cuda_v1_exp8: bool = False,
        refit_decoder: bool = False,
        use_pruning: bool = True,
        prune_mode: int = 3,
        log_eps: float = -10.0,
    ):
        self.method = method
        self.model, self.tokenizer, self._make_cache = load_quantpc_model(
            model_name,
            method=method,
            k_bits=k_bits,
            v_bits=v_bits,
            projected_query_bits=projected_query_bits,
            exp_bits=exp_bits,
            eas=eas,
            zerorope=zerorope,
            cuda_v1_exp8=cuda_v1_exp8,
            refit_decoder=refit_decoder,
            use_pruning=use_pruning,
            prune_mode=prune_mode,
            log_eps=log_eps,
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
    p.add_argument("--model", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--method",
        default="quantpc",
        help="'quantpc' (quantized KV cache) or a dense baseline attn impl "
             "('flash_attention_2' / 'sdpa' / 'eager')",
    )
    p.add_argument("--save_dir", default="results/longbench", help="base output dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-bits", type=int, default=128)
    p.add_argument("--v-bits", type=int, default=256)
    p.add_argument("--projected-query-bits", type=int, default=8)
    p.add_argument("--exp-bits", type=int, default=4)
    p.add_argument("--eas", type=float, default=0.01)
    p.add_argument("--zerorope", type=int, default=30)
    p.add_argument("--cuda-v1-exp8", action="store_true",
                   help="use CUDA backend (quantpc.quanpc); else torch reference")
    p.add_argument("--refit-decoder", action="store_true",
                   help="enable decode-matrix refit (CUDA prefill falls back to torch)")
    p.add_argument("--no-pruning", dest="use_pruning", action="store_false",
                   help="disable decode-time global pruning (gqa4 CUDA path); pruning is on by default")
    p.add_argument("--prune-mode", type=int, default=3, help="decode pruning mode (CUDA v1_exp8 path)")
    p.add_argument("--log-eps", type=float, default=-10.0, help="decode pruning log-epsilon threshold")
    p.set_defaults(use_pruning=True)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    model2maxlen = json.load(open(CONFIG_DIR / "model2maxlen.json"))
    datasets = json.load(open(CONFIG_DIR / "longbench_eval_subdatasets.json"))["datasets"]
    dataset2prompt = json.load(open(CONFIG_DIR / "dataset2prompt.json"))
    dataset2maxlen = json.load(open(CONFIG_DIR / "dataset2maxlen.json"))

    model_key = Path(args.model).name if "/" in args.model else args.model
    max_length = model2maxlen.get(model_key, 31500)
    print(f"Model: {args.model} | Method: {args.method} | max_length: {max_length}")

    tag = output_tag(args.method, args.k_bits, args.v_bits, args.refit_decoder)
    save_path = Path(args.save_dir) / model_key / tag
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {save_path}")

    model = QuantPCModel(
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

    for dataset in datasets:
        print("=" * 50)
        print(f"Evaluating dataset: {dataset}")
        data = load_dataset("THUDM/LongBench", dataset, split="test",trust_remote_code=True,)
        data_all = [x for x in data]
        out_path = str(save_path / f"{dataset}.jsonl")
        run_dataset(
            model, dataset, data_all,
            dataset2prompt[dataset], dataset2maxlen[dataset],
            max_length, model_key, out_path,
        )
        print(f"Finished {dataset} -> {out_path}")


if __name__ == "__main__":
    main()
