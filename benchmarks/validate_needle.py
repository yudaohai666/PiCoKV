"""Single-prompt needle-in-a-haystack sanity check for any --method in common.load_model.

Shares the exact prompt used by src/BitDecoding/evaluation/validate_e2e.py so that
KIVI / BitDecoding / picokv / dense outputs are directly comparable token-for-token.

    python validate_needle.py --method flash_attention_2
    python validate_needle.py --method kivi --k-bits 2 --v-bits 2 --group-size 32 --residual-length 256
    python validate_needle.py --method bitdecoding --num-bits 4 --group-size 128
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_model, seed_everything  # noqa: E402

NEEDLE = "\nThe secret access code for the Helsinki research cluster is 74219.\n"
QUESTION = (
    "\n\nQuestion: What is the secret access code for the Helsinki research "
    "cluster? Answer with the number only.\nAnswer:"
)
DEFAULT_HAYSTACK = (
    Path(__file__).resolve().parent / "Efficiency" / "datasets" / "PaulGrahamEssays.jsonl"
)
DEFAULT_MODEL = "/root/paddlejob/pfsl3space/picokv_baseline_t83/models/llama31_8b_instruct"


def build_prompt(haystack_path, enc, target_tokens):
    chunks = []
    with open(haystack_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line)["text"])
    text = enc.decode(
        enc("\n\n".join(chunks), add_special_tokens=False).input_ids[:target_tokens]
    )
    mid = len(text) // 2
    return text[:mid] + NEEDLE + text[mid:] + QUESTION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--method", default="flash_attention_2")
    p.add_argument("--haystack", default=str(DEFAULT_HAYSTACK))
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--k-bits", type=int, default=None)
    p.add_argument("--v-bits", type=int, default=None)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=32)
    # BitDecoding (--method bitdecoding): --num-bits/--group-size/--quant-mode.
    p.add_argument("--num-bits", type=int, default=4)
    p.add_argument("--quant-mode", default="k-channel")
    p.add_argument("--bitdecoding-allow-unvalidated", action="store_true")
    args = p.parse_args()

    seed_everything(42)
    model, tokenizer, make_cache = load_model(
        args.model,
        args.method,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        num_bits=args.num_bits,
        quant_mode=args.quant_mode,
        bitdecoding_allow_unvalidated=args.bitdecoding_allow_unvalidated,
    )

    prompt = build_prompt(args.haystack, tokenizer, args.prompt_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    if make_cache is not None:
        gen_kwargs["past_key_values"] = make_cache()

    with torch.inference_mode():
        out = model.generate(inputs.input_ids, **gen_kwargs)

    new_ids = out[0].tolist()[inputs.input_ids.shape[1]:]
    print(
        f"[cfg] method={args.method} k_bits={args.k_bits} v_bits={args.v_bits} "
        f"group={args.group_size} residual={args.residual_length} "
        f"num_bits={args.num_bits} quant_mode={args.quant_mode} "
        f"prompt_tokens={inputs.input_ids.shape[1]}"
    )
    print(f"[out] {tokenizer.decode(new_ids, skip_special_tokens=True)!r}")
    print(f"[ids] {new_ids}")


if __name__ == "__main__":
    main()
