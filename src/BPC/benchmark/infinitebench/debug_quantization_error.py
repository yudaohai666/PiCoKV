"""Debug script: visualize SparseTopk quantization error at the passkey
position in number_string / passkey tasks.

Usage:
    python -m infinitebench.debug_quantization_error \
        --env_conf config/llama3-1-8b-ins-sparsetopk.json \
        --task number_string \
        --data_dir infinitebench/data \
        --example_idx 0 \
        --output_dir outputs/debug
"""

import json
import re
import os
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, GenerationConfig
from loguru import logger

from benchmark.infinitebench.eval_utils import load_data, create_prompt, DATA_NAME_TO_MAX_NEW_TOKENS
from bpc.misc import get_model_and_tokenizer, get_env_conf


def find_passkey_indices(tokenizer, prompt: str, context: str, task: str):
    """Locate token indices of the digit sequence in passkey/number_string prompts.

    Returns (passkey_indices, digit_string).
    """
    if task == "number_string":
        pattern = r"The sequence of digits is (\d+)\. Remember it\. \1 is the sequence of digits\."
    elif task == "passkey":
        pattern = r"The pass key is (\d+)\. Remember it\. \1 is the pass key\."
    else:
        print(f"[Warning] Unsupported task: {task}, trying both patterns...")
        pattern = r"The sequence of digits is (\d+)\. Remember it\. \1 is the sequence of digits\."
        match = re.search(pattern, context)
        if not match:
            pattern = r"The pass key is (\d+)\. Remember it\. \1 is the pass key\."

    match = re.search(pattern, context)

    if not match:
        print(f"[Warning] Could not find {task} pattern in context")
        return [], None

    digit_string = match.group(1)
    print(f"[Debug] Found digit string: {digit_string} (task: {task})")

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # Digits appear twice; collect all matching token positions
    passkey_indices = []

    digit_tokens = tokenizer.encode(digit_string, add_special_tokens=False)
    print(f"[Debug] Digit tokens: {digit_tokens}")
    print(f"[Debug] Digit tokens decoded: {[tokenizer.decode([t]) for t in digit_tokens]}")

    for i in range(len(tokens) - len(digit_tokens) + 1):
        if tokens[i:i+len(digit_tokens)] == digit_tokens:
            passkey_indices.extend(range(i, i + len(digit_tokens)))
            print(f"[Debug] Found digit sequence at token indices: {i} to {i + len(digit_tokens) - 1}")

    passkey_indices = sorted(list(set(passkey_indices)))
    
    print(f"[Debug] Total passkey indices: {len(passkey_indices)}")
    print(f"[Debug] Passkey indices: {passkey_indices[:20]}{'...' if len(passkey_indices) > 20 else ''}")
    
    return passkey_indices, digit_string


def parse_args():
    p = argparse.ArgumentParser(description="Debug quantization error visualization")
    p.add_argument("--env_conf", type=str, required=True, 
                   help="Config file path (e.g., config/llama3-1-8b-ins-sparsetopk.json)")
    p.add_argument("--task", type=str, default="number_string",
                   choices=["number_string", "passkey"],
                   help="Task name (number_string or passkey)")
    p.add_argument("--data_dir", type=str, default="infinitebench/data",
                   help="Data directory")
    p.add_argument("--example_idx", type=int, default=0,
                   help="Example index to analyze")
    p.add_argument("--output_dir", type=str, default="outputs/debug",
                   help="Output directory for figures")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--layer_idx", type=int, default=None,
                   help="Specific layer to visualize (default: average all layers)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_conf = get_env_conf(args.env_conf)
    model_conf = env_conf.get("model", {})
    method = model_conf.get("model_method", "sparsetopk")

    if method != "sparsetopk":
        print(f"[Warning] This script is designed for sparsetopk method, but got: {method}")

    print("Loading model and tokenizer...")
    tok, model = get_model_and_tokenizer(**model_conf)
    tok.pad_token = tok.eos_token

    device = torch.device(args.device)
    if hasattr(model, 'model'):
        model.model = model.model.to(device)
    else:
        model = model.to(device)
    model.eval()

    data_name = args.task
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.example_idx >= len(examples):
        print(f"[Error] example_idx {args.example_idx} out of range (total: {len(examples)})")
        return

    eg = examples[args.example_idx]
    context = eg["context"]

    model_name = model_conf.get("model_name", "")
    prompt = create_prompt(eg, data_name, model_name, args.data_dir)

    passkey_indices, digit_string = find_passkey_indices(tok, prompt, context, data_name)

    if not passkey_indices:
        print("[Error] Could not find passkey indices")
        return

    from bpc.svdsparse.sparse_topk import set_debug_mode, visualize_quantization_error, reset_cache

    reset_cache()
    set_debug_mode(enable=True, passkey_idxes=passkey_indices)

    # Only prefill stage is needed
    print("\nRunning inference to collect quantization error data...")

    if tok.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
    else:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    
    print(f"[Debug] Input sequence length: {input_ids.shape[-1]}")

    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    
    with torch.no_grad():
        outputs = model.generate(input_ids, generation_config=generation_config)

    output_text = tok.decode(outputs[0, input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"\n[Debug] Model output: {output_text}")
    print(f"[Debug] Expected answer: {eg['answer']}")

    print("\nVisualizing quantization error...")

    save_path = output_dir / f"quantization_error_{data_name}_example{args.example_idx}_digit{digit_string}.png"

    layer_errors, layer_passkey_errors, layer_non_passkey_errors = visualize_quantization_error(
        save_path=str(save_path),
        layer_idx=args.layer_idx,
        show_avg=True
    )

    from bpc.svdsparse.sparse_topk import get_debug_data
    debug_data = get_debug_data()

    error_stats = {}
    for layer_idx, data in debug_data.items():
        error = data["error"]  # [B, H, L, D]
        token_error = torch.norm(error.float(), dim=-1).mean(dim=(0, 1))  # [L]
        
        all_mean = token_error.mean().item()
        all_std = token_error.std().item()
        
        if passkey_indices:
            passkey_mean = token_error[passkey_indices].mean().item()
            passkey_std = token_error[passkey_indices].std().item()
        else:
            passkey_mean = passkey_std = 0
        
        error_stats[layer_idx] = {
            "all_mean": all_mean,
            "all_std": all_std,
            "passkey_mean": passkey_mean,
            "passkey_std": passkey_std,
            "ratio": passkey_mean / all_mean if all_mean > 0 else 0,
        }
    
    stats_path = output_dir / f"error_stats_{data_name}_example{args.example_idx}.json"
    with open(stats_path, "w") as f:
        json.dump({
            "example_idx": args.example_idx,
            "digit_string": digit_string,
            "passkey_indices": passkey_indices,
            "seq_length": input_ids.shape[-1],
            "model_output": output_text,
            "expected_answer": eg["answer"],
            "layer_stats": {str(k): v for k, v in error_stats.items()},
        }, f, indent=2)
    
    print(f"\n[Debug] Error stats saved to: {stats_path}")
    print(f"[Debug] Figure saved to: {save_path}")

    print("\n" + "=" * 60)
    print("Per-layer Quantization Error Statistics:")
    print("=" * 60)
    print(f"{'Layer':<8} {'All Mean':<12} {'Passkey Mean':<14} {'Ratio':<8}")
    print("-" * 60)
    for layer_idx in sorted(error_stats.keys()):
        stats = error_stats[layer_idx]
        print(f"{layer_idx:<8} {stats['all_mean']:<12.6f} {stats['passkey_mean']:<14.6f} {stats['ratio']:<8.4f}")

    set_debug_mode(enable=False)
    reset_cache()


if __name__ == "__main__":
    main()
