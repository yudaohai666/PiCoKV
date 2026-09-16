"""Multi-GPU parallel inference for InfiniteBench (mirrors longbench_v2/pred_v2.py)."""
import json
from pathlib import Path
import time
import os
import torch
import torch.multiprocessing as mp
import argparse
import numpy as np
import random
from transformers import AutoTokenizer, GenerationConfig
from loguru import logger
from tqdm import tqdm

from benchmark.infinitebench.eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
    iter_jsonl,
)

from bpc.misc import get_model_and_tokenizer, get_env_conf


MAX_POSITION_ID = 128 * 1024
TRUNCATE_LEN = 128 * 1024


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def precompile_cuda_kernel_if_needed(env_conf):
    """Pre-compile CUDA kernel in the main process to avoid concurrent build races."""
    extra_config_path = env_conf.get("model", {}).get("config")
    if not extra_config_path:
        return
    
    try:
        with open(extra_config_path, 'r') as f:
            extra_config = json.load(f)
        
        if extra_config.get("use_cuda_kernel", False):
            logger.info("Pre-compiling CUDA kernel in main process (GPU 0)...")
            torch.cuda.set_device(0)
            from bpc.bpc.bpc import _load_cuda_kernel
            _load_cuda_kernel(device_id=0)
            logger.info("CUDA kernel pre-compilation done")
    except Exception as e:
        logger.warning(f"Failed to pre-compile CUDA kernel: {e}")


def build_chat(tokenizer, prompt, model_name):
    model_name_lower = model_name.lower()
    is_instruct = any(kw in model_name_lower for kw in ['instruct', 'chat', '-ins'])

    if 'llama-2' in model_name_lower and is_instruct:
        prompt = f"[INST]{prompt}[/INST]"
    elif ('llama-3' in model_name_lower or 'llama3' in model_name_lower) and is_instruct:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    device,
    generation_config: GenerationConfig = None,
    model_name: str = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    truncated_text = tok.decode(input_tokens, skip_special_tokens=True)

    if tok.chat_template is not None:
        logger.info("using chat_template")
        messages = [{"role": "user", "content": truncated_text}]
        input_ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        context_length = input_ids.shape[-1]
        input_tensors = {"input_ids": input_ids}
    else:
        prompt = build_chat(tok, truncated_text, model_name)
        input_tensors = tok(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_tensors["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(**input_tensors, generation_config=generation_config)

    if hasattr(model, 'reset'):
        model.reset()

    output = outputs[0, context_length:]
    output = tok.decode(output, skip_special_tokens=True)
    output = output.strip()

    return output


def get_pred_worker(rank, world_size, data_queue, result_queue, args_all, data_name):
    """Worker per GPU: pull samples from data_queue and write to result_queue."""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    env_conf = get_env_conf(args_all.env_conf)
    if rank == 0:
        logger.info(f"Loaded config from: {args_all.env_conf}")

    extra_config_path = env_conf.get("model", {}).get("config")
    method = env_conf.get("model", {}).get("model_method", "origin")
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]

    need_override = (extra_config_path and args_all.max_capacity_prompt is not None) or \
                    (extra_config_path and method == "rocketkv")
    
    if need_override:
        try:
            with open(extra_config_path, 'r') as f:
                extra_config = json.load(f)
            
            if args_all.max_capacity_prompt is not None:
                if method in ["quest", "rocketkv", "clusterkv"]:
                    extra_config["token_budget"] = args_all.max_capacity_prompt
                else:
                    extra_config["max_capacity_prompt"] = args_all.max_capacity_prompt
            
            if method == "rocketkv":
                extra_config["max_new_tokens"] = max_tokens
            
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_{method}_{rank}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            if rank == 0:
                logger.warning(f"Extra config file not found: {extra_config_path}")
    
    model_conf = env_conf.get("model", {})
    model_conf["device_map"] = {"": f"cuda:{rank}"}
    model_name = model_conf.get("model_name", "")
    
    if rank == 0:
        logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
    
    tok, model = get_model_and_tokenizer(**model_conf)
    tok.pad_token = tok.eos_token
    model.eval()
    
    if rank == 0:
        logger.info(f"Model loaded, ready to process samples")
    
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    
    while True:
        task = data_queue.get()
        if task is None:
            if rank == 0:
                logger.info(f"Received stop signal, exiting")
            break
        
        idx, eg = task
        
        try:
            input_text = create_prompt(eg, data_name, model_name, args_all.data_dir)
            pred = get_pred(
                model, tok, input_text, max_tokens=max_tokens,
                device=device, generation_config=generation_config, model_name=model_name
            )
            
            result_queue.put({
                "idx": idx,
                "item": {
                    "id": idx,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            })
            
        except Exception as e:
            logger.error(f"[GPU {rank}] Error processing sample {idx}: {e}")
            result_queue.put({"idx": idx, "error": str(e)})


def result_writer(result_queue, out_path, total_samples, pbar_queue, pending_indices, existing_preds=None):
    """Writer process; flushes results in pending_indices order."""
    results_buffer = {}
    written_count = 0
    pending_sorted = sorted(pending_indices)
    current_pos = 0

    preds = list(existing_preds) if existing_preds else []

    while written_count < total_samples:
        result = result_queue.get()
        if result is None:
            break

        idx = result["idx"]
        results_buffer[idx] = result

        while current_pos < len(pending_sorted) and pending_sorted[current_pos] in results_buffer:
            next_idx = pending_sorted[current_pos]
            res = results_buffer.pop(next_idx)
            if "error" not in res:
                preds.append(res["item"])
                dump_jsonl(preds, out_path)
            current_pos += 1
            written_count += 1
            pbar_queue.put(1)
    
    logger.info(f"Result writer finished, wrote {written_count} samples")


def run_parallel_inference(examples, world_size, out_path, args_all, data_name, start_idx=0, stop_idx=None, completed_ids=None, existing_preds=None):
    """Parallel inference: batch_size = world_size, one sample per GPU."""
    data_queue = mp.Queue()
    result_queue = mp.Queue()
    pbar_queue = mp.Queue()

    if completed_ids is None:
        completed_ids = set()
    
    if stop_idx is None:
        stop_idx = len(examples)
    
    pending_samples = []
    pending_indices = []
    for idx in range(start_idx, stop_idx):
        if idx not in completed_ids:
            pending_samples.append((idx, examples[idx]))
            pending_indices.append(idx)
    
    total_samples = len(pending_samples)
    
    if total_samples == 0:
        logger.info("All samples already completed")
        return
    
    for idx, eg in pending_samples:
        data_queue.put((idx, eg))

    # Sentinel to terminate each worker
    for _ in range(world_size):
        data_queue.put(None)

    workers = []
    for rank in range(world_size):
        p = mp.Process(
            target=get_pred_worker,
            args=(rank, world_size, data_queue, result_queue, args_all, data_name)
        )
        p.start()
        workers.append(p)
    
    writer = mp.Process(
        target=result_writer,
        args=(result_queue, out_path, total_samples, pbar_queue, pending_indices, existing_preds)
    )
    writer.start()

    with tqdm(total=total_samples, desc=f"Processing InfiniteBench ({data_name})") as pbar:
        completed = 0
        while completed < total_samples:
            try:
                pbar_queue.get(timeout=1)
                pbar.update(1)
                completed += 1
            except:
                all_done = all(not p.is_alive() for p in workers)
                if all_done and completed < total_samples:
                    while not pbar_queue.empty():
                        pbar_queue.get()
                        pbar.update(1)
                        completed += 1
                    break
    
    for p in workers:
        p.join()
    writer.join()
    
    logger.info(f"Parallel inference completed")


def load_existing_results(output_path):
    """Load existing predictions; return set of completed sample IDs."""
    completed_ids = set()
    existing_preds = []
    if os.path.exists(output_path):
        try:
            existing_preds = list(iter_jsonl(output_path))
            completed_ids = {pred["id"] for pred in existing_preds}
            print(f"Found existing results with {len(completed_ids)} completed samples")
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
            existing_preds = []
            completed_ids = set()
    return completed_ids, existing_preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Which task to use.",
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="./results_infinite", help="Where to dump the prediction results.")
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on.")
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on.")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--env_conf", type=str, required=True,
                   help="Config file path (e.g., config/llama3-1-8b-ins-snapkv.json)")

    p.add_argument("--max_capacity_prompt", type=int, default=None,
                   help="Max capacity prompt (overrides config)")
    
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    
    print(json.dumps(vars(args), indent=4))
    data_name = args.task
    
    world_size = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {world_size}")
    logger.info(f"Batch size (default = GPU count): {world_size}, each GPU processes 1 sample")
    
    env_conf = get_env_conf(args.env_conf)
    model_path = env_conf.get("model", {}).get("model_name", "")
    method = env_conf.get("model", {}).get("model_method", "origin")
    
    model_key = Path(model_path).name if "/" in model_path else model_path
    logger.info(f"Model: {model_path}")
    logger.info(f"Method: {method}")
    
    is_offline = "offline" in args.env_conf.lower()
    offline_suffix = "_offline" if is_offline else ""
    
    result_dir = Path(args.output_dir, model_key, f"{method}{offline_suffix}")
    if method not in ["origin", "sparse", "spotlight", "sparsetopk", "magicpig", "bpc"]:
        max_cap = args.max_capacity_prompt if args.max_capacity_prompt else 512
        result_dir = result_dir / str(max_cap)
    
    result_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Result dir: {result_dir}")
    
    examples = load_data(data_name, data_dir=args.data_dir)
    
    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = result_dir / f"preds_{data_name}.jsonl"
    else:
        output_path = result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"
    
    # Resume from previous run
    completed_ids, existing_preds = load_existing_results(output_path)
    
    logger.info(f"==== Evaluation ====")
    logger.info(f"# examples: {len(examples)}")
    logger.info(f"Start index: {args.start_idx}")
    logger.info(f"Stop index: {args.stop_idx}")
    logger.info(f"Already completed: {len(completed_ids)}")
    logger.info(f"Max tokens: {DATA_NAME_TO_MAX_NEW_TOKENS[data_name]}")
    logger.info(f"Method: {method}")
    logger.info(f"Output file: {output_path}")
    
    precompile_cuda_kernel_if_needed(env_conf)

    mp.set_start_method('spawn', force=True)

    run_parallel_inference(
        examples, world_size, str(output_path), args, data_name,
        start_idx=args.start_idx, stop_idx=args.stop_idx, 
        completed_ids=completed_ids, existing_preds=existing_preds
    )
    
    logger.info(f"Finished, results saved to {output_path}")


if __name__ == "__main__":
    main()
