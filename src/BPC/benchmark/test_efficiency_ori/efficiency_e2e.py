import os
import argparse
import time
import json
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

from bpc import modify as bpc_modify, reset_bpc, finalize_bpc


def parse_args():
    parser = argparse.ArgumentParser(description="BPC Multi-GPU Efficiency Benchmark")
    
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--method', type=str, default="origin",
                        choices=['origin', 'bpc', 'bpc-offline'])
    parser.add_argument('--model_structure', type=str, default="llama",
                        choices=['llama', 'mistral', 'qwen2'])
    
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--P', type=int, default=256000, help='Prefill length')
    parser.add_argument('--G', type=int, default=128, help='Generation length')
    parser.add_argument('--chunk_size', type=int, default=16384)
    parser.add_argument('--no_chunking', action='store_true', default=False)
    
    parser.add_argument('--token_budget', type=int, default=None)
    parser.add_argument('--mask_out', type=float, default=0.98)
    parser.add_argument('--min_remain', type=int, default=128)
    parser.add_argument('--err_ratio', type=float, default=0.1)
    parser.add_argument('--num_splits', type=int, default=0)
    parser.add_argument('--fix_layers', type=str, default="0,1")
    parser.add_argument('--offline_proj_path', type=str, default=None)
    
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--max_memory_per_gpu', type=str, default=None)
    parser.add_argument('--device_map_strategy', type=str, default="auto",
                        choices=['auto', 'balanced', 'balanced_low_0', 'sequential'])
    
    parser.add_argument('--warm_up', type=int, default=32)
    parser.add_argument('--data_path', type=str, default=None, help='Path to data file (jsonl)')
    parser.add_argument('--dtype', type=str, default="bf16", choices=['bf16', 'fp16'])
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--use_dynamic_cache', action='store_true', default=False)
    
    return parser.parse_args()


def get_dummy_input(tokenizer, prefix_len, device):
    """Generate dummy input token ids"""
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(base_text, add_special_tokens=False)
    while len(tokens) < prefix_len:
        tokens = tokens * 2
    return torch.tensor([tokens[:prefix_len]], dtype=torch.long, device=device)


def load_data_input(data_path, tokenizer, prefix_len, device):
    """Load input from jsonl data file"""
    import jsonlines
    
    with open(data_path) as f:
        reader = jsonlines.Reader(f)
        for item in reader:
            data = item
            break
    
    text = data.get("input", data.get("text", ""))
    input_ids = tokenizer.encode(text=text, return_tensors="pt").to(device=device)
    return input_ids[:, :prefix_len]


def get_default_offline_proj_path(model_name):
    """Get default offline projection file path"""
    project_root = Path(__file__).resolve().parent
    while project_root.name != "BPC" and project_root.parent != project_root:
        project_root = project_root.parent
    
    model_proj_map = {
        "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b-ins-mixlen-mixdata-projection.pt",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b-mixlen-mixdata-projection.pt",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b-ins-mixlen-mixdata-projection.pt",
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b-ins-mixlen-mixdata-projection.pt",
    }
    proj_file = model_proj_map.get(model_name)
    return str(project_root / "config" / "projections" / proj_file) if proj_file else None


def build_method_config(args):
    """Build BPC configuration"""
    if args.method not in ["bpc", "bpc-offline"]:
        return {}
    
    fix_layers = [int(x.strip()) for x in args.fix_layers.split(",") if x.strip()]
    use_offline = args.method == "bpc-offline"
    
    offline_proj_path = args.offline_proj_path
    if use_offline and not offline_proj_path:
        offline_proj_path = get_default_offline_proj_path(args.model)
        if offline_proj_path:
            print(f"[Config] Using default offline projection: {offline_proj_path}")
    
    return {
        "enable": True,
        "fix_layers": fix_layers,
        "mask_out": args.mask_out,
        "min_remain": args.min_remain,
        "token_budget": args.token_budget,
        "err_ratio": args.err_ratio,
        "use_offline_proj": use_offline,
        "offline_proj_path": offline_proj_path,
        "use_cuda_kernel": True,
    }


def print_gpu_memory(stage="", num_gpus=None):
    """Print GPU memory usage"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    total_allocated = sum(torch.cuda.memory_allocated(i) for i in range(num_gpus)) / 1024**3
    print(f"[GPU Memory] {stage}: total_allocated={total_allocated:.2f}GB")


def sync_all_gpus(num_gpus=None):
    """Synchronize all GPUs"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        torch.cuda.synchronize(i)


def get_model_first_device(model):
    """Get the first device of the model"""
    if hasattr(model, 'hf_device_map'):
        for key in ['model.embed_tokens', 'transformer.wte', 'embed_tokens']:
            if key in model.hf_device_map:
                device = model.hf_device_map[key]
                return f"cuda:{device}" if isinstance(device, int) else device
        first_device = list(model.hf_device_map.values())[0]
        return f"cuda:{first_device}" if isinstance(first_device, int) else first_device
    return getattr(model, 'device', "cuda:0")


def chunked_prefill_multi_gpu(model, input_ids, chunk_size, num_gpus, use_chunking=True, 
                              method="origin", fix_layers=None, past_key_values=None):
    """Multi-GPU chunked prefill with StaticCache support."""
    B, seq_len = input_ids.shape
    fix_layers = fix_layers or []
    
    first_device = get_model_first_device(model)
    input_ids = input_ids.to(first_device)
    
    if not use_chunking or seq_len <= chunk_size:
        print(f"  Direct prefill: seq_len={seq_len}")
        outputs = model(input_ids=input_ids, past_key_values=past_key_values,
                       use_cache=True, return_dict=True)
        if method in ["bpc", "bpc-offline"]:
            finalize_bpc(model, outputs.past_key_values, fix_layers=fix_layers)
        return outputs.past_key_values, outputs.logits
    
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    print(f"  Chunked prefill: seq_len={seq_len}, chunks={num_chunks}")
    
    logits = None
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, seq_len)
        chunk_input_ids = input_ids[:, start_idx:end_idx]
        
        past_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else start_idx
        cache_position = torch.arange(past_len, past_len + chunk_input_ids.shape[1], device=first_device)
        
        outputs = model(input_ids=chunk_input_ids, past_key_values=past_key_values,
                       cache_position=cache_position, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
            print(f"    Chunk {chunk_idx + 1}/{num_chunks}, processed {end_idx} tokens")
        
        del outputs
        if chunk_idx < num_chunks - 1:
            gc.collect()
            torch.cuda.empty_cache()
    
    if method in ["bpc", "bpc-offline"]:
        print("  Finalizing BPC...")
        finalize_bpc(model, past_key_values, fix_layers=fix_layers)
    
    return past_key_values, logits


def main():
    args = parse_args()
    use_static_cache = not args.use_dynamic_cache
    num_gpus = args.num_gpus or torch.cuda.device_count()
    
    print("=" * 60)
    print(f"Model: {args.model} | Method: {args.method}")
    print(f"GPUs: {num_gpus} | Cache: {'Static' if use_static_cache else 'Dynamic'}")
    print(f"P={args.P}, G={args.G}, B={args.B}, chunk={args.chunk_size}")
    print("=" * 60)
    
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fix_layers = [int(x.strip()) for x in args.fix_layers.split(",") if x.strip()]
    MAX_SEQ_LEN = args.P + args.G + args.warm_up + 10
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_memory = {i: args.max_memory_per_gpu for i in range(num_gpus)} if args.max_memory_per_gpu else None
    if max_memory:
        max_memory["cpu"] = "0GiB"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device_map_strategy,
        max_memory=max_memory, trust_remote_code=True,
        attn_implementation="flash_attention_2", use_cache=True,
    )
    
    if args.method in ["bpc", "bpc-offline"]:
        config = build_method_config(args)
        config['use_static_cache'] = use_static_cache
        config['num_splits'] = args.num_splits
        model = bpc_modify(model, config)
    
    model.eval()
    first_device = get_model_first_device(model)
    print_gpu_memory("Model loaded", num_gpus)
    
    # Prepare input
    if args.data_path and os.path.exists(args.data_path):
        input_ids = load_data_input(args.data_path, tokenizer, args.P, first_device)
    else:
        input_ids = get_dummy_input(tokenizer, args.P, first_device)
    input_ids = input_ids.repeat(args.B, 1)
    
    # Initialize cache
    past_key_values = None
    if use_static_cache:
        layer_device_map = {}
        if hasattr(model, 'hf_device_map'):
            for name, device in model.hf_device_map.items():
                if 'layers.' in name:
                    layer_idx = int(name.split('layers.')[1].split('.')[0])
                    layer_device_map[layer_idx] = device
        
        past_key_values = StaticCache(
            config=model.config, max_batch_size=args.B, max_cache_len=MAX_SEQ_LEN,
            device=first_device, dtype=dtype, layer_device_map=layer_device_map or None,
        )
    
    # Prefill
    print(f"\n[Prefill] chunk_size={args.chunk_size}")
    sync_all_gpus(num_gpus)
    prefill_start = time.time()
    
    with torch.inference_mode():
        if args.method in ["bpc", "bpc-offline"]:
            reset_bpc(model)
        past_key_values, logits = chunked_prefill_multi_gpu(
            model, input_ids, args.chunk_size, num_gpus,
            use_chunking=not args.no_chunking, method=args.method,
            fix_layers=fix_layers, past_key_values=past_key_values,
        )
    
    sync_all_gpus(num_gpus)
    prefill_time = time.time() - prefill_start
    print(f"Prefill: {prefill_time:.3f}s, {(args.P * args.B) / prefill_time:.0f} tok/s")
    
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True).to(first_device)
    current_cache_len = past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else args.P
    
    # Warm up
    print(f"\n[Warm up] {args.warm_up} steps")
    with torch.inference_mode():
        for _ in range(args.warm_up):
            cache_position = torch.tensor([current_cache_len], device=first_device)
            outputs = model(input_ids=next_token, past_key_values=past_key_values,
                          cache_position=cache_position, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).to(first_device)
            current_cache_len += 1
    
    # Decode
    print(f"\n[Decode] {args.G} tokens")
    sync_all_gpus(num_gpus)
    decode_start = time.time()
    
    with torch.inference_mode():
        for _ in range(args.G):
            cache_position = torch.tensor([current_cache_len], device=first_device)
            outputs = model(input_ids=next_token, past_key_values=past_key_values,
                          cache_position=cache_position, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True).to(first_device)
            current_cache_len += 1
    
    sync_all_gpus(num_gpus)
    decode_time = time.time() - decode_start
    decode_latency = 1000 * decode_time / args.G
    decode_throughput = args.B * args.G / decode_time
    
    # Results
    print(f"\n{'='*60}")
    print(f"Results: {args.method}")
    print(f"Prefill: {prefill_time:.3f}s, {(args.P * args.B) / prefill_time:.0f} tok/s")
    print(f"Decode: {decode_latency:.2f} ms/tok, {decode_throughput:.2f} tok/s")
    print(f"{'='*60}")
    
    results = {
        "method": args.method, "model": args.model, "num_gpus": num_gpus,
        "batch_size": args.B, "prefix_len": args.P, "gen_len": args.G,
        "prefill_time_sec": prefill_time,
        "prefill_throughput": (args.P * args.B) / prefill_time,
        "decode_latency_ms": decode_latency,
        "decode_throughput": decode_throughput,
    }
    
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    
    return results


if __name__ == "__main__":
    main()
