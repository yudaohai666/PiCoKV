"""Offline end-to-end check for the BitDecoding attention backend.

Mirrors evaluation/example.py but (a) builds the long prompt from a local text
file instead of downloading gsm8k, and (b) caps per-GPU memory so the model can
be sharded onto a partially occupied node.

Run it once per backend and diff the decoded text:

    python validate_e2e.py --attn_backend flash_attention_2   # fp16 reference
    python validate_e2e.py --attn_backend bit_decoding --num_bits 4
"""

import argparse
import json
import random
import warnings

warnings.filterwarnings("ignore")

import torch

from bit_decode import Cache, DynamicCache, StaticCache
import transformers.cache_utils

transformers.cache_utils.DynamicCache = DynamicCache
transformers.cache_utils.StaticCache = StaticCache
transformers.cache_utils.Cache = Cache

from llama import LlamaForCausalLM  # noqa: E402
from transformers import AutoTokenizer, LlamaConfig  # noqa: E402

NEEDLE = (
    "\nThe secret access code for the Helsinki research cluster is 74219.\n"
)
QUESTION = (
    "\n\nQuestion: What is the secret access code for the Helsinki research "
    "cluster? Answer with the number only.\nAnswer:"
)


def build_prompt(haystack_path, enc, target_tokens):
    """Needle-in-a-haystack prompt of roughly `target_tokens` tokens."""
    chunks = []
    with open(haystack_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line)["text"])
    haystack = "\n\n".join(chunks)
    ids = enc(haystack, add_special_tokens=False).input_ids
    ids = ids[:target_tokens]
    text = enc.decode(ids)
    mid = len(text) // 2
    return text[:mid] + NEEDLE + text[mid:] + QUESTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/bd_models/Llama-3.1-8B-Instruct")
    parser.add_argument("--haystack", type=str,
                        default="/root/paddlejob/system-public/yudaohai/ICLR2027/PiCoKV/"
                                "benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl")
    parser.add_argument("--prompt_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_bits", type=int, default=4)
    parser.add_argument("--quant_mode", type=str, default="k-channel")
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--attn_backend", type=str, default="flash_attention_2")
    parser.add_argument("--gpu_gib", type=float, default=8.0,
                        help="per-GPU budget for accelerate's device_map")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    if args.group_size is None:
        args.group_size = 32 if args.num_bits == 2 else 128

    config = LlamaConfig.from_pretrained(args.model_path)
    config._attn_implementation = "flash_attention_2"
    config.attn_backend = args.attn_backend
    config.num_bits = args.num_bits
    config.quant_mode = args.quant_mode
    config.group_size = args.group_size
    config.residual_block_size = 128 if args.num_bits == 4 else 256

    max_memory = {i: f"{args.gpu_gib}GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "64GiB"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()

    enc = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    prompt = build_prompt(args.haystack, enc, args.prompt_tokens)
    inputs = enc(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=enc.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    new_ids = output[0].tolist()[inputs.input_ids.shape[1]:]
    print(f"[cfg] backend={args.attn_backend} bits={args.num_bits} "
          f"mode={args.quant_mode} group={args.group_size} "
          f"prompt_tokens={inputs.input_ids.shape[1]}")
    print(f"[out] {enc.decode(new_ids, skip_special_tokens=True)!r}")
    print(f"[ids] {new_ids}")


if __name__ == "__main__":
    main()
