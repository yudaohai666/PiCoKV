"""Experiment runner for the QuanPC quantized KV-cache algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from needle import LLMNeedleHaystackTester, ModelProvider


# Dense-baseline attn implementations selectable via --method (mirrors
# benchmarks/Ruler/run_ruler_evaluation.py). Anything not in this set and not
# "quantpc" is rejected.
BASELINE_ATTN = {"flash_attention_2", "sdpa", "eager"}


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Transformer(ModelProvider):
    def __init__(
        self,
        model_name: str = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
        method: str = "quantpc",
        k_bits: int = 128,
        v_bits: int = 256,
        projected_query_bits: int = 8,
        exp_bits: int = 4,
        eas: float = 0.01,
        zerorope: int = 30,
        cuda_v1_exp8: bool = False,
        refit_decoder: bool = False,
        refit_solver: str = "lstsq",
    ):
        self.method = method
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.projected_query_bits = projected_query_bits
        self.exp_bits = exp_bits
        self.eas = eas
        self.zerorope = zerorope
        self.cuda_v1_exp8 = cuda_v1_exp8
        self.refit_decoder = refit_decoder
        self.refit_solver = refit_solver

        from_pretrained_kwargs = {"torch_dtype": "bfloat16"}

        # Method dispatch (mirrors the Ruler runner):
        #   quantpc            -> apply modify() + inject QuantKVCache per request.
        #                         Backend (CUDA vs torch reference) via cuda_v1_exp8.
        #   flash_attention_2  -> dense full-precision baseline (flash-attn-2)
        #   sdpa / eager       -> dense full-precision baseline (torch attn)
        # For baselines we DO NOT modify the model and use its default KV cache.
        if method == "quantpc":
            if cuda_v1_exp8:
                from quantpc.quanpc import QuantKVCache, modify
            else:
                from quantpc.quanpc_torch import QuantKVCache, modify
            self._QuantKVCache = QuantKVCache
            self._modify = modify
        elif method in BASELINE_ATTN:
            self._QuantKVCache = None
            self._modify = None
            from_pretrained_kwargs["attn_implementation"] = method
        else:
            raise ValueError(
                f"Unknown method '{method}'. Supported: 'quantpc', "
                f"{sorted(BASELINE_ATTN)}."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            import accelerate  # noqa: F401
            device_map = "auto"
        except ImportError:
            device_map = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            **from_pretrained_kwargs,
        )
        if device_map is None and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        if self._modify is not None:
            self._modify(self.model)
        self.model.eval()

    def evaluate_model(self, prompt, needle=None) -> str:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        ).to(self.model.device)

        with torch.no_grad():
            generate_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
            )
            # quantpc injects a fresh QuantKVCache per request; baselines use the
            # model's default cache (no past_key_values override).
            past_key_values = None
            if self._QuantKVCache is not None:
                cache_kwargs = dict(
                    k_bits=self.k_bits,
                    v_bits=self.v_bits,
                    projected_query_bits=self.projected_query_bits,
                    exp_bits=self.exp_bits,
                    eas=self.eas,
                    zerorope=self.zerorope,
                    use_cuda_v1_exp8=self.cuda_v1_exp8,
                    refit_decoder=self.refit_decoder,
                )
                # refit_solver is only supported by the CUDA-runtime QuantKVCache
                # (quantpc.quanpc); the torch reference keeps its default solver.
                if self.cuda_v1_exp8:
                    cache_kwargs["refit_solver"] = self.refit_solver
                past_key_values = self._QuantKVCache(**cache_kwargs)
                generate_kwargs["past_key_values"] = past_key_values
            output = self.model.generate(**generate_kwargs)
            generated_ids = output.sequences[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        del past_key_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI bot that answers questions for a user. "
                    "Keep your response short and direct"
                ),
            },
            {
                "role": "user",
                "content": context,
            },
            {
                "role": "user",
                # "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it"
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def run_experiment(
    model_name: str,
    lengths: list[int],
    depths: list[int],
    output_path: str,
    method: str,
    k_bits: int,
    v_bits: int,
    projected_query_bits: int,
    exp_bits: int,
    eas: float,
    zerorope: int,
    cuda_v1_exp8: bool,
    refit_decoder: bool = False,
    refit_solver: str = "lstsq",
) -> dict[tuple[int, int], str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = Transformer(
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
        refit_solver=refit_solver,
    )
    tester = LLMNeedleHaystackTester(model)
    results = {}

    # Resume: load previously completed (length, depth) results so we skip them.
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for key, value in prev.items():
                length_str, _, depth_str = key.partition("-")
                results[(int(length_str), int(depth_str))] = value
            print(f"[resume] loaded {len(results)} completed cases from {output_path}")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            print(f"[resume] ignoring unreadable checkpoint {output_path}: {exc}")

    for length in lengths:
        for depth in depths:
            if (length, depth) in results:
                print((length, depth), "[skip] already done")
                continue
            results[(length, depth)] = tester.evaluate(length, depth)
            print((length, depth), results[(length, depth)])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({f"{k[0]}-{k[1]}": v for k, v in results.items()}, f, indent=4)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run needle-in-a-haystack experiments with quantized KV cache.")
    parser.add_argument("--model", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method",
        default="quantpc",
        help=(
            "evaluation method: 'quantpc' (quantized KV cache), or a dense "
            "baseline attn implementation 'flash_attention_2' / 'sdpa' / 'eager'"
        ),
    )
    parser.add_argument("--lengths", default="64000,80000,100000", help="comma-separated context lengths")
    parser.add_argument("--depths", default="50", help="comma-separated needle depths")
    parser.add_argument("--output", default="results/niah/results_shared_kv.json")
    parser.add_argument("--k-bits", type=int, default=128)
    parser.add_argument("--v-bits", type=int, default=256)
    parser.add_argument(
        "--projected-query-bits",
        type=int,
        default=8,
        help="signed quantization bits for projected query; quantization is always enabled",
    )
    parser.add_argument(
        "--exp-bits",
        type=int,
        default=4,
        help="unsigned per-64-token quantization bits for global exp weights; quantization is always enabled",
    )
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=30)
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
        help="启用解码矩阵重拟合；开启后 CUDA prefill 回退 torch",
    )
    parser.add_argument(
        "--refit-solver",
        default="lstsq",
        choices=["lstsq", "pinv", "cholesky"],
        help="refit 最小二乘求解器：lstsq(QR，快且稳，默认)、pinv(SVD，慢) 或 cholesky(正规方程，最快但可能掉精度)；仅 CUDA 后端生效",
    )
    args = parser.parse_args()

    seed_everything(42)
    run_experiment(
        model_name=args.model,
        lengths=parse_int_list(args.lengths),
        depths=parse_int_list(args.depths),
        output_path=args.output,
        method=args.method,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        projected_query_bits=args.projected_query_bits,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        cuda_v1_exp8=args.cuda_v1_exp8,
        refit_decoder=args.refit_decoder,
        refit_solver=args.refit_solver,
    )


if __name__ == "__main__":
    main()
