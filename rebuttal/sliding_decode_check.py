"""Model-level smoke check of PiCoKV offline incremental decode.

Runs greedy generation on one long prompt with the static global cache
(``slide_tokens=0``) and with sliding (``slide_tokens=64``), then records
token agreement, cache sizes, and wall time.

    CUDA_VISIBLE_DEVICES=6 python rebuttal/sliding_decode_check.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from picokv.picokv_standalone import StandaloneConfig
from picokv.picokv_standalone.transformers import (
    StandaloneLlamaCache,
    load_offline_projections,
    modify_llama_for_standalone,
)

REPO = Path(__file__).resolve().parents[1]


def build_prompt(tokenizer, context_tokens: int) -> str:
    text = ""
    for path in sorted((REPO / "picokv" / "PaulGrahamEssays").glob("*.txt")):
        text += path.read_text() + "\n\n"
        if len(tokenizer(text).input_ids) > context_tokens:
            break
    ids = tokenizer(text).input_ids[:context_tokens]
    context = tokenizer.decode(ids, skip_special_tokens=True)
    messages = [{
        "role": "user",
        "content": context + "\n\nWrite a detailed, multi-paragraph summary of the essays above.",
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, inputs, projs, config: StandaloneConfig, max_new_tokens: int):
    cache = StandaloneLlamaCache(
        config,
        key_encoders=projs["key_encoder"],
        value_encoders=projs["value_encoder"],
        key_decoders=projs["key_decoder"],
        value_decoders=projs["value_decoder"],
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(
        **inputs, past_key_values=cache, do_sample=False,
        max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
        temperature=None, top_p=None,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    layer0 = cache.layers[0]
    return out[0, inputs["input_ids"].shape[1]:].tolist(), {
        "seconds": elapsed,
        "slides_total": cache.slide_count,
        "layer0_dense_len": layer0.dense_len,
        "layer0_global_len": layer0.global_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--proj",
        default=str(REPO / "offline_projection/projections/Llama-3.1-8B-Instruct_k320_v320.pt"),
    )
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--new-tokens", type=int, default=512)
    parser.add_argument("--slide", type=int, default=64)
    parser.add_argument(
        "--output", default=str(REPO / "rebuttal/results/sliding_decode/check.json")
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    modify_llama_for_standalone(model)
    projs = load_offline_projections(args.proj)
    bits = (int(projs["key_encoder"].shape[-2]), int(projs["value_encoder"].shape[-2]))

    prompt = build_prompt(tokenizer, args.context)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    results = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "runs": {}}
    tokens = {}
    with torch.inference_mode():
        for name, slide in (
            ("warmup", 0), ("static", 0), ("static_repeat", 0), ("sliding", args.slide)
        ):
            config = StandaloneConfig(key_bits=bits[0], value_bits=bits[1], slide_tokens=slide)
            n = 8 if name == "warmup" else args.new_tokens
            tokens[name], stats = generate(model, inputs, projs, config, n)
            if name != "warmup":
                stats["text"] = tokenizer.decode(tokens[name], skip_special_tokens=True)
                results["runs"][name] = stats
                print(name, {k: v for k, v in stats.items() if k != "text"}, flush=True)

    # static vs static_repeat is the run-to-run noise floor: the decode kernel
    # accumulates with atomics, so greedy decoding can diverge without sliding.
    for other in ("static_repeat", "sliding"):
        a, b = tokens["static"], tokens[other]
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        agreement = sum(x == y for x, y in zip(a, b)) / len(a)
        results[f"static_vs_{other}"] = {
            "first_divergent_token": first_diff,
            "token_agreement": agreement,
        }
        print(f"static vs {other}: first divergent token {first_diff}, agreement {agreement:.3f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
