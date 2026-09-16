"""
Multi-GPU Needle in a Haystack Test
Adapted from run_needle_in_haystack.py with multi-GPU support
Reference: benchmark/longbench/pred.py
"""

import os 
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
from rouge_score import rouge_scorer

from datetime import datetime, timezone
import time
import torch
import torch.multiprocessing as mp
import random
import tqdm
import logging
from loguru import logger

from bpc.misc import get_model_and_tokenizer, get_env_conf, get_tokenizer


NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
HAYSTACK_DIR = "benchmark/needle/PaulGrahamEssays"
RETRIEVAL_QUESTION = "What is the best thing to do in San Francisco?"
FINAL_CONTEXT_LENGTH_BUFFER = 200


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def precompile_cuda_kernel_if_needed(env_conf):
    """Pre-compile CUDA kernel in main process to avoid concurrent build races."""
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


def get_results_dir(save_dir, model_version, method, max_capacity_prompts, is_offline):
    """Build results dir path."""
    offline_suffix = "_offline" if is_offline else ""
    
    if method in ["origin", "spotlight", "magicpig", "bpc"]:
        return f'{save_dir}/{model_version}_{method}{offline_suffix}'
    else:
        if max_capacity_prompts is not None:
            return f'{save_dir}/{model_version}_{method}{offline_suffix}_{max_capacity_prompts}'
        else:
            return f'{save_dir}/{model_version}_{method}{offline_suffix}'


def build_haystack_context(model_name, context_lengths):
    """Build haystack context once in main process to avoid repeated IO."""
    logger.info("Building haystack context...")

    # Temporary tokenizer just for length measurement
    tokenizer = get_tokenizer(model_name)
    
    context = ""
    max_context_length = max(context_lengths)
    
    iteration = 0
    current_len = 0
    while current_len < max_context_length:
        for file in glob.glob(f"{HAYSTACK_DIR}/*.txt"):
            with open(file, 'r') as f:
                context += f.read()
        iteration += 1
        current_len = len(tokenizer.encode(context))
    
    logger.info(f"Context built: {current_len} tokens")
    return context, tokenizer


def generate_test_tasks(context_lengths, document_depth_percents, base_context, tokenizer,
                        results_dir, model_path, s_len, e_len):
    """Build the full set of test tasks."""
    tasks = []

    for context_length in context_lengths:
        if context_length < s_len or context_length > e_len:
            continue
        for depth_percent in document_depth_percents:
            if result_exists(results_dir, context_length, depth_percent, model_path):
                logger.info(f"Skipping context_length={context_length}, depth={depth_percent} (already exists)")
                continue

            context = generate_context_with_needle(
                base_context, tokenizer, NEEDLE, 
                context_length, depth_percent, FINAL_CONTEXT_LENGTH_BUFFER
            )
            
            tasks.append({
                'context_length': context_length,
                'depth_percent': depth_percent,
                'context': context
            })
    
    logger.info(f"Generated {len(tasks)} test tasks")
    return tasks


def generate_context_with_needle(base_context, tokenizer, needle, context_length, depth_percent, buffer):
    """Build a context with the needle inserted (encode_and_trim, then insert_needle)."""
    # Step 1: trim to target length
    tokens_context = tokenizer.encode(base_context)
    if len(tokens_context) > context_length:
        tokens_context = tokens_context[:context_length]
        base_context = tokenizer.decode(tokens_context)

    # Step 2: insert needle
    tokens_needle = tokenizer.encode(needle)
    tokens_context = tokenizer.encode(base_context)

    # Reserve buffer (matches single-GPU final_context_length_buffer)
    adjusted_length = context_length - buffer

    if len(tokens_context) + len(tokens_needle) > adjusted_length:
        tokens_context = tokens_context[:adjusted_length - len(tokens_needle)]

    if depth_percent == 100:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_new_context = tokens_context[:insertion_point]

        # Snap insertion point back to a period for sentence alignment
        period_tokens = tokenizer.encode('.')
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    return tokenizer.decode(tokens_new_context)


def result_exists(results_dir, context_length, depth_percent, model_path):
    """Return True if a matching result file already exists."""
    if not os.path.exists(results_dir):
        return False
    
    for filename in os.listdir(results_dir):
        if filename == 'result.json':
            continue
        if filename.endswith('.json'):
            try:
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    if (result['context_length'] == context_length and 
                        result['depth_percent'] == depth_percent and
                        result['model'] == model_path):
                        return True
            except:
                continue
    return False


def generate_prompt(tokenizer, model, context, retrieval_question):
    """Build tokenized test prompt."""
    if hasattr(model, 'model') and hasattr(model.model, 'device'):
        device = model.model.device
    else:
        device = next(iter(model.parameters())).device
    
    if tokenizer.chat_template is None:
        prompt = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {retrieval_question}\nAnswer:"
        tokenized = tokenizer(prompt, return_tensors="pt").to(device)
        model_inputs = {"input_ids": tokenized.input_ids}
    else:
        logger.info(f"Using chat template")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
            },
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        model_inputs = {"input_ids": input_ids}
    
    return model_inputs


def needle_worker(rank, world_size, data_queue, result_queue, model_path, args):
    """Worker per GPU; pulls tasks from data_queue."""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Same seed per worker ensures determinism
    set_seed(args.seed)

    # Each process needs its own scorer instance
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    env_conf = get_env_conf(args.env_conf)
    if rank == 0:
        logger.info(f"Worker {rank}: Loaded config from {args.env_conf}")

    if args.method:
        env_conf.get("model", {})["model_method"] = args.method
    env_conf.get("model", {})["model_name"] = model_path

    method = env_conf.get("model", {}).get("model_method", "origin")

    extra_config_path = env_conf.get("model", {}).get("config")
    if extra_config_path and args.max_capacity_prompts is not None:
        try:
            with open(extra_config_path, 'r') as f:
                extra_config = json.load(f)

            # quest/clusterkv use token_budget; other methods use max_capacity_prompt
            if method in ["quest", "clusterkv"]:
                extra_config["token_budget"] = args.max_capacity_prompts
                if rank == 0:
                    logger.info(f"Override: token_budget={args.max_capacity_prompts}")
            else:
                extra_config["max_capacity_prompt"] = args.max_capacity_prompts
                if rank == 0:
                    logger.info(f"Override: max_capacity_prompt={args.max_capacity_prompts}")
            
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_needle_{method}_{rank}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            if rank == 0:
                logger.warning(f"Extra config file not found: {extra_config_path}")
    
    model_conf = env_conf.get("model", {})
    model_conf["device_map"] = {"": f"cuda:{rank}"}
    if rank == 0:
        logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
    
    tokenizer, model = get_model_and_tokenizer(**model_conf)
    model.eval()
    
    if rank == 0:
        logger.info(f"Worker {rank}: Model loaded, ready to process")
    
    while True:
        task = data_queue.get()
        if task is None:  # sentinel
            if rank == 0:
                logger.info(f"Worker {rank}: Received stop signal")
            break
        
        idx, task_data = task
        context_length = task_data['context_length']
        depth_percent = task_data['depth_percent']
        context = task_data['context']
        
        try:
            test_start_time = time.time()

            model_inputs = generate_prompt(tokenizer, model, context, RETRIEVAL_QUESTION)

            actual_context_length = model_inputs["input_ids"].shape[-1]

            output = model.generate(
                **model_inputs,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

            # Reset post-generation state
            if hasattr(model, 'reset'):
                model.reset()
            elif method == "cakekv" and hasattr(model, 'model'):
                layers = len(model.model.layers)
                for i in range(layers):
                    model.model.layers[i].self_attn.config.prefill = [True] * layers
                    model.model.layers[i].self_attn.config.decoding_evict = [None] * layers
            
            response = tokenizer.decode(output[actual_context_length:], skip_special_tokens=True).strip()
            
            test_elapsed_time = time.time() - test_start_time
            score = scorer.score(NEEDLE, response)['rouge1'].recall * 100
            
            result = {
                'idx': idx,
                'model': model_path,
                'context_length': int(context_length),
                'actual_context_length': int(actual_context_length),
                'depth_percent': float(depth_percent),
                'needle': NEEDLE,
                'model_response': response,
                'score': score,
                'test_duration_seconds': test_elapsed_time,
                'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
            }
            result_queue.put(result)
            
        except Exception as e:
            logger.error(f"Worker {rank}: Error processing task {idx}: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put({
                'idx': idx,
                'error': str(e),
                'context_length': context_length,
                'depth_percent': depth_percent
            })


def result_writer(result_queue, results_dir, model_version, total_tasks, pbar_queue):
    """Writer process."""
    os.makedirs(results_dir, exist_ok=True)
    
    written_count = 0
    
    while written_count < total_tasks:
        result = result_queue.get()
        if result is None:
            break
        
        if "error" not in result:
            context_length = result['context_length']
            depth_percent = result['depth_percent']

            context_file_location = f'{model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent)}'
            result_path = f'{results_dir}/{context_file_location}_results.json'

            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4)

            summary_path = os.path.join(results_dir, 'result.json')
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                except:
                    summary = {}
            else:
                summary = {}
            
            key = f"{context_length}-{int(depth_percent)}"
            summary[key] = result['model_response']
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Saved result: context={context_length}, depth={depth_percent}%, score={result['score']:.1f}")
        else:
            logger.error(f"Task failed: context={result['context_length']}, depth={result['depth_percent']}, error={result['error']}")
        
        written_count += 1
        pbar_queue.put(1)
    
    logger.info(f"Result writer finished, wrote {written_count} results")


def run_parallel_needle_test(tasks, results_dir, model_version, model_path, args):
    """Run the needle test across all GPUs."""
    world_size = torch.cuda.device_count()
    logger.info(f"Running parallel needle test with {world_size} GPUs")

    data_queue = mp.Queue()
    result_queue = mp.Queue()
    pbar_queue = mp.Queue()

    total_tasks = len(tasks)

    for idx, task in enumerate(tasks):
        data_queue.put((idx, task))

    # One sentinel per worker
    for _ in range(world_size):
        data_queue.put(None)

    workers = []
    for rank in range(world_size):
        p = mp.Process(
            target=needle_worker,
            args=(rank, world_size, data_queue, result_queue, model_path, args)
        )
        p.start()
        workers.append(p)
    
    writer = mp.Process(
        target=result_writer,
        args=(result_queue, results_dir, model_version, total_tasks, pbar_queue)
    )
    writer.start()

    with tqdm.tqdm(total=total_tasks, desc="Needle Test") as pbar:
        completed = 0
        while completed < total_tasks:
            try:
                pbar_queue.get(timeout=1)
                pbar.update(1)
                completed += 1
            except:
                all_done = all(not p.is_alive() for p in workers)
                if all_done and completed < total_tasks:
                    while not pbar_queue.empty():
                        pbar_queue.get()
                        pbar.update(1)
                        completed += 1
                    break

    for p in workers:
        p.join()
    writer.join()
    
    logger.info("Parallel needle test completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', default=0, metavar='N', type=int, help='Start context length')
    parser.add_argument('-e', '--e_len', default=131072, metavar='N', type=int, help='End context length')
    parser.add_argument('--model_path', type=str, default=None, help='Model key (e.g., llama3-1-8b-ins)')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='Model provider type')
    parser.add_argument('--step', type=int, default=1000, help='Step size for context lengths')
    parser.add_argument("--context_length", nargs='+', type=int, 
                        default=[8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 72000, 80000, 88000])
    parser.add_argument('--save_dir', type=str, default="outputs/needle", help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument("--env_conf", type=str, required=True,
                        help="Config file path (e.g., config/llama3-1-8b-ins-origin.json)")

    parser.add_argument('--method', type=str, default=None,
                        choices=["origin", "pyramidkv", "snapkv", "cakekv", "compresskv",
                                "streamingllm", "spotlight", "quest",
                                "clusterkv", "bpc", "magicpig"],
                        help='Method for KV cache compression (default: use config file)')

    parser.add_argument('--max_capacity_prompts', type=int, default=None,
                        help="Max capacity prompt (overrides config)")
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    world_size = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {world_size}")
    logger.info(f"Batch size (default = GPU count): {world_size}, each GPU processes 1 sample")

    env_conf = get_env_conf(args.env_conf)
    logger.info(f"Loaded config from: {args.env_conf}")

    model_path = env_conf.get("model", {}).get("model_name", args.model_path)
    if model_path is None:
        raise ValueError("Model path must be specified either in config file (model.model_name) or via --model_path argument")

    # CLI method takes precedence over config
    method = args.method if args.method else env_conf.get("model", {}).get("model_method", "origin")
    is_offline = "offline" in args.env_conf.lower()

    if args.model_path:
        model_version = args.model_path
    elif model_path and "/" in model_path:
        model_version = model_path.split("/")[-1]
    else:
        model_version = model_path

    if args.context_length is not None:
        context_lengths = np.array(args.context_length)
    else:
        context_lengths = np.arange(args.s_len, args.e_len + 1, step=args.step)

    document_depth_percents = np.round(np.linspace(0, 100, num=10, endpoint=True)).astype(int)

    results_dir = get_results_dir(args.save_dir, model_version, method, args.max_capacity_prompts, is_offline)
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Method: {method}")
    logger.info(f"Max capacity prompts: {args.max_capacity_prompts}")
    logger.info(f"Context lengths: {context_lengths}")
    logger.info(f"Results dir: {results_dir}")

    precompile_cuda_kernel_if_needed(env_conf)

    base_context, tokenizer = build_haystack_context(model_path, context_lengths)

    tasks = generate_test_tasks(
        context_lengths, document_depth_percents, base_context, tokenizer,
        results_dir, model_path, args.s_len, args.e_len
    )

    if len(tasks) == 0:
        logger.info("All tasks already completed, nothing to do")
    else:
        mp.set_start_method('spawn', force=True)

        run_parallel_needle_test(tasks, results_dir, model_version, model_path, args)
        
        logger.info("All tests completed!")
