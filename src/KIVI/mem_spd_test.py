# LLaMA memory & speed test: KIVI vs. standard attention
# Measures prefill and decode latency separately via a manual forward loop.
import argparse
import gc

import torch
from transformers import LlamaConfig, AutoTokenizer

from models.llama_kivi import LlamaForCausalLM_KIVI

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 32
BATCH_SIZE = 400
PATH_TO_YOUR_SAVE_DIR = './cached_models'

PROMPT_LENGTH = 160
OUTPUT_LENGTH = 338
NUM_REPEATS = 3


def build_inputs(tokenizer, batch_size, prompt_length):
    # tokenize an over-long string, then truncate to exactly prompt_length tokens
    text = 't,' * (prompt_length * 2)
    ids = tokenizer(text, return_tensors="pt").input_ids
    assert ids.shape[1] >= prompt_length, \
        f"only got {ids.shape[1]} tokens, need {prompt_length}; increase the multiplier"
    ids = ids[:, :prompt_length]
    input_ids = ids.repeat(batch_size, 1).contiguous().to('cuda')
    attention_mask = torch.ones_like(input_ids)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def _decode_step(model, tok, past):
    out = model(input_ids=tok, past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1:].argmax(dim=-1)


def _warmup(model, input_ids, attention_mask, steps=3):
    # first calls JIT-compile triton/cuda kernels; exclude them from timing
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(steps):
            past, tok = _decode_step(model, tok, past)
    torch.cuda.synchronize()


def prefill_decode_benchmark(name, model, inputs, output_length, num_repeats):
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)
    bsz, prompt_len = input_ids.shape
    decode_steps = output_length - 1
    print(f"\n===== {name} =====")
    print(f"bs: {bsz}, seqlen: {prompt_len}+{output_length}")

    def one_pass():
        pre_s, pre_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dec_s, dec_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            # prefill: single forward over the whole prompt
            pre_s.record()
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            pre_e.record()
            past = out.past_key_values
            tok = out.logits[:, -1:].argmax(dim=-1)
            # decode: feed one token at a time, reusing the KV cache
            dec_s.record()
            for _ in range(decode_steps):
                past, tok = _decode_step(model, tok, past)
            dec_e.record()
        torch.cuda.synchronize()
        return pre_s.elapsed_time(pre_e), dec_s.elapsed_time(dec_e)

    try:
        _warmup(model, input_ids, attention_mask)
        torch.cuda.reset_peak_memory_stats()
        prefills, decodes = [], []
        for _ in range(num_repeats):
            p, d = one_pass()
            prefills.append(p)
            decodes.append(d)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    except torch.cuda.OutOfMemoryError as e:
        print(f'OOM: {e}')
        return None

    prefill_ms = sum(prefills) / len(prefills)
    decode_ms = sum(decodes) / len(decodes)
    per_tok = decode_ms / decode_steps if decode_steps > 0 else float('nan')
    print(f'prefill:      {prefill_ms:9.2f} ms   ({bsz * prompt_len / prefill_ms * 1000:9.0f} tok/s)')
    print(f'decode total: {decode_ms:9.2f} ms   over {decode_steps} steps')
    print(f'decode/token: {per_tok:9.2f} ms   ({bsz / per_tok * 1000:9.1f} tok/s)')
    print(f'peak mem:     {peak:9.3f} GB')
    return {'prefill_ms': prefill_ms, 'decode_ms': decode_ms,
            'per_tok_ms': per_tok, 'peak_gb': peak}


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
        attn_implementation="flash_attention_2"
    )
    return model.cuda().eval()


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
    results['KIVI'] = prefill_decode_benchmark(
        f"KIVI (k{K_BITS}v{V_BITS}, group {GROUP_SIZE}, residual {RESIDUAL_LENGTH})",
        kivi_model, inputs, OUTPUT_LENGTH, NUM_REPEATS)
    free(kivi_model)

    base_model = load_baseline(args.model_name_or_path, args.cache_dir)
    results['Baseline'] = prefill_decode_benchmark(
        "Standard attention (fp16 KV cache)",
        base_model, inputs, OUTPUT_LENGTH, NUM_REPEATS)
    free(base_model)

    print("\n===== summary =====")
    print(f"{'':>10} | {'prefill(ms)':>12} | {'decode/tok(ms)':>14} | {'peak(GB)':>9}")
    for name, r in results.items():
        if r is None:
            print(f"{name:>10} | {'OOM':>12} | {'OOM':>14} | {'OOM':>9}")
        else:
            print(f"{name:>10} | {r['prefill_ms']:>12.2f} | {r['per_tok_ms']:>14.2f} | {r['peak_gb']:>9.3f}")

    kivi, base = results['KIVI'], results['Baseline']
    if kivi is not None and base is not None:
        print(f"\nmem saving:      {(1 - kivi['peak_gb'] / base['peak_gb']) * 100:.1f}% "
              f"({base['peak_gb']:.3f} -> {kivi['peak_gb']:.3f} GB)")
        print(f"prefill speedup: {base['prefill_ms'] / kivi['prefill_ms']:.2f}x")
        print(f"decode  speedup: {base['per_tok_ms'] / kivi['per_tok_ms']:.2f}x")


if __name__ == "__main__":
    main()
