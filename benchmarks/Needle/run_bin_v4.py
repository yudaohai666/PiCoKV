"""Experiment runner for the QuanPC quantized KV-cache algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from needle import LLMNeedleHaystackTester, ModelProvider


def seed_everything(seed: int = 1234) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Transformer(ModelProvider):
    def __init__(
        self,
        model_name: str = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
        k_bits: int = 128,
        v_bits: int = 256,
        projected_query_bits: int = 8,
        exp_bits: int = 4,
        eas: float = 0.01,
        zerorope: int = 30,
        cuda_v1_exp8: bool = False,
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.projected_query_bits = projected_query_bits
        self.exp_bits = exp_bits
        self.eas = eas
        self.zerorope = zerorope
        self.cuda_v1_exp8 = cuda_v1_exp8

        # Backend selection is driven by the --cuda-v1-exp8 flag: with the flag
        # we use the CUDA backend, without it the torch-only reference backend.
        if cuda_v1_exp8:
            from quantpc.quanpc import QuantKVCache, modify
        else:
            from quantpc.quanpc_torch import QuantKVCache, modify
        self._QuantKVCache = QuantKVCache
        self._modify = modify

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            import accelerate  # noqa: F401
            device_map = "auto"
        except ImportError:
            device_map = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="bfloat16",
        )
        if device_map is None and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self._modify(self.model)

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
            past_key_values = self._QuantKVCache(
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                projected_query_bits=self.projected_query_bits,
                exp_bits=self.exp_bits,
                eas=self.eas,
                zerorope=self.zerorope,
                use_cuda_v1_exp8=self.cuda_v1_exp8,
            )
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
                past_key_values=past_key_values,
            )
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
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
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
    k_bits: int,
    v_bits: int,
    projected_query_bits: int,
    exp_bits: int,
    eas: float,
    zerorope: int,
    cuda_v1_exp8: bool,
) -> dict[tuple[int, int], str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = Transformer(
        model_name,
        k_bits=k_bits,
        v_bits=v_bits,
        projected_query_bits=projected_query_bits,
        exp_bits=exp_bits,
        eas=eas,
        zerorope=zerorope,
        cuda_v1_exp8=cuda_v1_exp8,
    )
    tester = LLMNeedleHaystackTester(model)
    results = {}

    for length in lengths:
        for depth in depths:
            if (length, depth) in results:
                continue
            results[(length, depth)] = tester.evaluate(length, depth)
            print((length, depth), results[(length, depth)])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({f"{k[0]}-{k[1]}": v for k, v in results.items()}, f, indent=4)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run needle-in-a-haystack experiments with quantized KV cache.")
    parser.add_argument("--model", default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct")
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
    args = parser.parse_args()

    seed_everything(42)
    run_experiment(
        model_name=args.model,
        lengths=parse_int_list(args.lengths),
        depths=parse_int_list(args.depths),
        output_path=args.output,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        projected_query_bits=args.projected_query_bits,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        cuda_v1_exp8=args.cuda_v1_exp8,
    )


if __name__ == "__main__":
    main()
