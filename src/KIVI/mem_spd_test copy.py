# LLaMA memory & speed test: KIVI vs. standard attention
import argparse
import gc
import time

import torch
from transformers import LlamaConfig, AutoTokenizer

from models.llama_kivi import LlamaForCausalLM_KIVI

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 32
BATCH_SIZE = 200
PATH_TO_YOUR_SAVE_DIR = './cached_models'

PROMPT_LENGTH = 160
OUTPUT_LENGTH = 338
NUM_REPEATS = 3


def build_inputs(tokenizer, batch_size, prompt_length):
    context = []
    for _ in range(batch_size):
        string = 't,' * (prompt_length // 2)
        context.append(string[:-1])
    return tokenizer(context, return_tensors="pt").to('cuda')


def run_benchmark(name, model, inputs, output_length, num_repeats):
    input_ids = inputs['input_ids']
    print(f"\n===== {name} =====")
    print(f"bs: {input_ids.shape[0]}, seqlen: {input_ids.shape[1]}+{output_length}")
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(num_repeats):
                model.generate(**inputs, max_new_tokens=output_length)
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - st) / num_repeats * 1000
        used_mem = torch.cuda.max_memory_allocated()
        print(f'used time: {elapsed_ms:.2f} ms')
        print(f'peak mem: {used_mem / 1024 ** 3:.3f} GB')
        return elapsed_ms, used_mem / 1024 ** 3
    except torch.cuda.OutOfMemoryError as e:
        print(f'OOM: {e}')
        return None, None


def load_kivi(model_name_or_path, cache_dir):
    config = LlamaConfig.from_pretrained(model_name_or_path)
    config.k_bits = K_BITS
    config.v_bits = V_BITS
    config.group_size = GROUP_SIZE
    config.residual_length = RESIDUAL_LENGTH
    config.use_flash = True
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return model.cuda().eval()


def load_baseline(model_name_or_path, cache_dir):
    from transformers import LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    )
    return model.eval()


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path or HF id of the base LLaMA model.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache_dir", type=str, default=PATH_TO_YOUR_SAVE_DIR)
    args = parser.parse_args()

    print(f"model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True)
    inputs = build_inputs(tokenizer, args.batch_size, PROMPT_LENGTH)

    results = {}

    kivi_model = load_kivi(args.model_name_or_path, args.cache_dir)
    results['KIVI'] = run_benchmark(
        f"KIVI (k{K_BITS}v{V_BITS}, group {GROUP_SIZE}, residual {RESIDUAL_LENGTH})",
        kivi_model, inputs, OUTPUT_LENGTH, NUM_REPEATS)
    free(kivi_model)

    base_model = load_baseline(args.model_name_or_path, args.cache_dir)
    results['Baseline'] = run_benchmark(
        "Standard attention (fp16 KV cache)",
        base_model, inputs, OUTPUT_LENGTH, NUM_REPEATS)
    free(base_model)

    print("\n===== summary =====")
    for name, (t, m) in results.items():
        if t is None:
            print(f"{name:>10}: OOM")
        else:
            print(f"{name:>10}: {t:.2f} ms, {m:.3f} GB")

    kivi_t, kivi_m = results['KIVI']
    base_t, base_m = results['Baseline']
    if kivi_m is not None and base_m is not None:
        print(f"\nmem saving: {(1 - kivi_m / base_m) * 100:.1f}% "
              f"({base_m:.3f} GB -> {kivi_m:.3f} GB)")


if __name__ == "__main__":
    main()
