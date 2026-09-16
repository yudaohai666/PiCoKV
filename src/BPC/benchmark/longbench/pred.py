import os
import sys
from pathlib import Path
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp
from loguru import logger

from bpc.misc import get_model_and_tokenizer, get_env_conf, get_tokenizer


def precompile_cuda_kernel_if_needed(env_conf):
    """Pre-compile CUDA kernel in main process to avoid concurrent build races.
    Each subprocess will still load the compiled kernel on its own GPU.
    """
    extra_config_path = env_conf.get("model", {}).get("config")
    if not extra_config_path:
        return

    try:
        with open(extra_config_path, 'r') as f:
            extra_config = json.load(f)

        if extra_config.get("use_cuda_kernel", False):
            logger.info("Pre-compiling CUDA kernel in main process (GPU 0)...")
            # Compile on GPU 0 so children can just import it
            torch.cuda.set_device(0)
            from bpc.bpc.bpc import _load_cuda_kernel
            _load_cuda_kernel(device_id=0)
            logger.info("CUDA kernel pre-compilation done")
    except Exception as e:
        logger.warning(f"Failed to pre-compile CUDA kernel: {e}")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    parser.add_argument("--env_conf", type=str, required=True,
                        help="Config file path (e.g., config/llama3-1-8b-ins-snapkv.json)")

    # Also overrides quest's token_budget
    parser.add_argument("--max_capacity_prompt", type=int, default=None,
                        help="Max capacity prompt (overrides config, also used as token_budget for quest)")
    
    return parser.parse_args(args)

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred_single_gpu(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_key, out_path, args_all, start_idx=0):
    """Single-GPU processing of a data subset (legacy, kept for compatibility)."""
    device = torch.device(f'cuda:{rank}')

    env_conf = get_env_conf(args_all.env_conf)
    logger.info(f"Loaded config from: {args_all.env_conf}")
    logger.info(f"Rank {rank}: Starting from index {start_idx}, total samples: {len(data)}")

    extra_config_path = env_conf.get("model", {}).get("config")
    method = env_conf.get("model", {}).get("model_method", "origin")

    if extra_config_path and args_all.max_capacity_prompt is not None:
        try:
            with open(extra_config_path, 'r') as f:
                extra_config = json.load(f)
            
            if method in ["quest", "rocketkv", "clusterkv"]:
                extra_config["token_budget"] = args_all.max_capacity_prompt
                logger.info(f"Override token_budget={args_all.max_capacity_prompt}")
            else:
                extra_config["max_capacity_prompt"] = args_all.max_capacity_prompt
                logger.info(f"Override max_capacity_prompt={args_all.max_capacity_prompt}")
            
            if method == "rocketkv":
                extra_config["max_new_tokens"] = max_gen
                logger.info(f"Override max_new_tokens={max_gen}")
            
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_{method}_{rank}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            if rank == 0:
                logger.warning(f"Extra config file not found: {extra_config_path}")
    
    model_conf = env_conf.get("model", {})
    logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
    tokenizer, model = get_model_and_tokenizer(**model_conf)
    
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'device'):
            current_device = model.model.device
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'device'):
            current_device = model.decoder.device
        else:
            current_device = next(iter(model.parameters())).device
        
        if current_device != device:
            model = model.to(device)
    except (StopIteration, AttributeError):
        model = model.to(device)
    
    model.eval()
    
    for idx, json_obj in enumerate(tqdm(data)):
        if idx < start_idx:
            continue
            
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            messages = [{"role": "user", "content": prompt}]
            if tokenizer.chat_template is not None:
                logger.info(f"Using chat_template")
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)
                input = {"input_ids": input_ids}
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input["input_ids"].shape[-1] if isinstance(input, dict) else input.input_ids.shape[-1]
        
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        
        if hasattr(model, 'reset'):
            model.reset()
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_key)
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def get_pred_worker(rank, world_size, data_queue, result_queue, max_length, max_gen, prompt_format, dataset, model_key, args_all):
    """Worker per GPU; processes one sample at a time from data_queue."""
    # set_device must precede model loading so CUDA kernels load on the right device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    env_conf = get_env_conf(args_all.env_conf)
    if rank == 0:
        logger.info(f"Loaded config from: {args_all.env_conf}")

    extra_config_path = env_conf.get("model", {}).get("config")
    method = env_conf.get("model", {}).get("model_method", "origin")
    
    if extra_config_path and args_all.max_capacity_prompt is not None:
        try:
            with open(extra_config_path, 'r') as f:
                extra_config = json.load(f)
            
            if method in ["quest", "rocketkv", "clusterkv"]:
                extra_config["token_budget"] = args_all.max_capacity_prompt
            else:
                extra_config["max_capacity_prompt"] = args_all.max_capacity_prompt
            
            if method == "rocketkv":
                extra_config["max_new_tokens"] = max_gen
            
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_{method}_{rank}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            if rank == 0:
                logger.warning(f"Extra config file not found: {extra_config_path}")
    
    model_conf = env_conf.get("model", {})
    # Pin device_map so dispatch_model loads model onto this rank's GPU
    model_conf["device_map"] = {"": f"cuda:{rank}"}
    if rank == 0:
        logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
    tokenizer, model = get_model_and_tokenizer(**model_conf)

    # Flash Attention 2.0 requires the model to be on a GPU
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'device'):
            current_device = model.model.device
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'device'):
            current_device = model.decoder.device
        else:
            current_device = next(iter(model.parameters())).device
        
        if current_device != device:
            model = model.to(device)
    except (StopIteration, AttributeError):
        model = model.to(device)
    
    model.eval()
    if rank == 0:
        logger.info(f"Model loaded, ready to process samples")
    
    while True:
        task = data_queue.get()
        if task is None:  # sentinel
            if rank == 0:
                logger.info(f"Received stop signal, exiting")
            break
        
        idx, json_obj = task
        
        try:
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

            if len(tokenized_prompt) > max_length:
                half = int(max_length//2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                messages = [{"role": "user", "content": prompt}]
                if tokenizer.chat_template is not None:
                    logger.info(f"Using chat template")
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(device)
                    input_data = {"input_ids": input_ids}
                else:
                    input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input_data = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input_data["input_ids"].shape[-1] if isinstance(input_data, dict) and "input_ids" in input_data else input_data.input_ids.shape[-1]
            
            if dataset == "samsum":
                output = model.generate(
                    **input_data,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input_data,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            
            if hasattr(model, 'reset'):
                model.reset()
            
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_key)

            result = {
                "idx": idx,
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"]
            }
            result_queue.put(result)
            
        except Exception as e:
            import traceback
            logger.error(f"[GPU {rank}] Error processing sample {idx}: {e}\n{traceback.format_exc()}")
            result_queue.put({"idx": idx, "error": str(e)})


def result_writer(result_queue, out_path, total_samples, pbar_queue, start_idx=0):
    """Writer process; writes results in order. start_idx supports resume."""
    results_buffer = {}
    next_idx = start_idx
    written_count = 0

    while written_count < total_samples:
        result = result_queue.get()
        if result is None:
            break

        idx = result["idx"]
        results_buffer[idx] = result

        while next_idx in results_buffer:
            res = results_buffer.pop(next_idx)
            if "error" not in res:
                with open(out_path, "a", encoding="utf-8") as f:
                    json.dump({
                        "pred": res["pred"],
                        "answers": res["answers"],
                        "all_classes": res["all_classes"],
                        "length": res["length"]
                    }, f, ensure_ascii=False)
                    f.write('\n')
            next_idx += 1
            written_count += 1
            pbar_queue.put(1)

    logger.info(f"Result writer finished, wrote {written_count} samples")

def count_existing_results(out_path, world_size):
    """Count completed samples for resume; returns per-rank skip counts (round-robin)."""
    if not os.path.exists(out_path):
        return [0] * world_size

    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        total_completed = len(lines)
        logger.info(f"Found {total_completed} completed samples in {out_path}")

        completed_per_rank = [0] * world_size
        for i in range(total_completed):
            rank = i % world_size
            completed_per_rank[rank] += 1

        return completed_per_rank
    except Exception as e:
        logger.warning(f"Error reading existing results: {e}")
        return [0] * world_size


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def run_parallel_inference(data_all, world_size, max_length, max_gen, prompt_format, dataset, model_key, out_path, args_all, start_idx=0):
    """Parallel inference: queue-based dispatch, one sample per GPU."""
    data_queue = mp.Queue()
    result_queue = mp.Queue()
    pbar_queue = mp.Queue()

    total_samples = len(data_all) - start_idx

    for idx in range(start_idx, len(data_all)):
        data_queue.put((idx, data_all[idx]))

    # One sentinel per worker
    for _ in range(world_size):
        data_queue.put(None)

    workers = []
    for rank in range(world_size):
        p = mp.Process(
            target=get_pred_worker,
            args=(rank, world_size, data_queue, result_queue, max_length, max_gen, prompt_format, dataset, model_key, args_all)
        )
        p.start()
        workers.append(p)
    
    writer = mp.Process(
        target=result_writer,
        args=(result_queue, out_path, total_samples, pbar_queue, start_idx)
    )
    writer.start()

    with tqdm(total=total_samples, desc=f"Processing {dataset}") as pbar:
        completed = 0
        while completed < total_samples:
            try:
                pbar_queue.get(timeout=1)
                pbar.update(1)
                completed += 1
            except:
                all_done = all(not p.is_alive() for p in workers)
                if all_done and completed < total_samples:
                    # Drain leftover events on early failure
                    while not pbar_queue.empty():
                        pbar_queue.get()
                        pbar.update(1)
                        completed += 1
                    break

    for p in workers:
        p.join()
    writer.join()
    
    logger.info(f"Parallel inference completed for {dataset}")


if __name__ == '__main__':
    
    seed_everything(42)
    args_all = parse_args()
    world_size = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {world_size}")
    logger.info(f"Batch size (default = GPU count): {world_size}, each GPU processes 1 sample")
    
    env_conf = get_env_conf(args_all.env_conf)
    model2maxlen = json.load(open("benchmark/longbench/config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = env_conf.get("model", {}).get("model_name", "")
    method = env_conf.get("model", {}).get("model_method", "origin")

    model_key = Path(model_path).name if "/" in model_path else model_path
    logger.info(f"Model: {model_path}")
    logger.info(f"Method: {method}")

    if model_key in model2maxlen:
        max_length = model2maxlen[model_key]
    else:
        max_length = 31500
    logger.info(f"Max length: {max_length}")

    datasets_config = json.load(open("benchmark/longbench/config/longbench_eval_subdatasets.json", "r"))
    datasets = datasets_config["datasets"]
    dataset2prompt = json.load(open("benchmark/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/longbench/config/dataset2maxlen.json", "r"))

    # Append _offline suffix when env_conf filename indicates offline mode
    is_offline = "offline" in args_all.env_conf.lower()
    offline_suffix = "_offline" if is_offline else ""

    if method in ["origin", "spotlight", 'magicpig', 'bpc', "topk"]:
        save_path = f"outputs/longbench/{model_key}/{method}{offline_suffix}/"
    else:
        # Methods that take max_capacity_prompt (incl. quest's token_budget)
        max_cap = args_all.max_capacity_prompt if args_all.max_capacity_prompt else 1024
        save_path = f"outputs/longbench/{model_key}/{method}{offline_suffix}/{max_cap}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.info(f"Output path: {save_path}")

    precompile_cuda_kernel_if_needed(env_conf)

    mp.set_start_method('spawn', force=True)
    
    for dataset in datasets:
        logger.info("=" * 50)
        logger.info(f"Evaluating on dataset: {dataset}")
        logger.info("=" * 50)
        
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        out_path = os.path.join(save_path, f"{dataset}.jsonl")

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        # Resume from existing partial results
        start_idx = 0
        if os.path.exists(out_path):
            try:
                with open(out_path, 'r', encoding='utf-8') as f:
                    start_idx = len(f.readlines())
                logger.info(f"Found {start_idx} completed samples, resuming from index {start_idx}")
            except Exception as e:
                logger.warning(f"Error reading existing results: {e}")
        
        total_samples = len(data_all)
        
        if start_idx >= total_samples:
            logger.info(f"Dataset {dataset} already completed ({start_idx}/{total_samples}), skipping...")
            continue
        elif start_idx > 0:
            logger.info(f"Resuming {dataset} from {start_idx}/{total_samples} samples")

        run_parallel_inference(
            data_all, world_size, max_length, max_gen, 
            prompt_format, dataset, model_key, out_path, args_all, start_idx
        )
        
        logger.info(f"Finished {dataset}, results saved to {out_path}")

