import os, json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import re
import torch
import torch.multiprocessing as mp
import numpy as np
import random
from loguru import logger

from bpc.misc import get_model_and_tokenizer, get_env_conf

maxlen_map = json.loads(open(f'benchmark/longbench_v2/config/model2maxlen.json', encoding='utf-8').read())

template_0shot = open('benchmark/longbench_v2/prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('benchmark/longbench_v2/prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('benchmark/longbench_v2/prompts/0shot_cot_ans.txt', encoding='utf-8').read()
template_rag = open('benchmark/longbench_v2/prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('benchmark/longbench_v2/prompts/0shot_no_context.txt', encoding='utf-8').read()


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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def query_llm_local(prompt, model, tokenizer, device, max_len, max_new_tokens=128):
    """Run local-model inference."""
    messages = [
        {"role": "user", "content": prompt}
    ]

    if tokenizer.chat_template is not None:
        logger.info(f"Using chat template")
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Truncate by keeping head and tail halves
        if input_ids.shape[-1] > max_len:
            input_ids = torch.cat([
                input_ids[:, :max_len//2],
                input_ids[:, -max_len//2:]
            ], dim=-1)

        context_length = input_ids.shape[-1]
        inputs = {"input_ids": input_ids}
    else:
        # Fallback when there is no chat template
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]
    
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred_worker(rank, world_size, data_queue, result_queue, max_len, args_all):
    """Worker per GPU; processes one sample per iteration."""
    # set_device must run before model load so kernels go on the right device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    env_conf = get_env_conf(args_all.env_conf)
    if rank == 0:
        logger.info(f"Loaded config from: {args_all.env_conf}")

    extra_config_path = env_conf.get("model", {}).get("config")
    method = env_conf.get("model", {}).get("model_method", "origin")

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
                max_new_tokens = 1024 if args_all.cot else 128
                extra_config["max_new_tokens"] = max_new_tokens
            
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_{method}_{rank}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            if rank == 0:
                logger.warning(f"Extra config file not found: {extra_config_path}")
    
    model_conf = env_conf.get("model", {})
    # Pin device_map to this rank's GPU
    model_conf["device_map"] = {"": f"cuda:{rank}"}
    if rank == 0:
        logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
    tokenizer, model = get_model_and_tokenizer(**model_conf)
    
    model.eval()
    if rank == 0:
        logger.info(f"Model loaded, ready to process samples")
    
    while True:
        task = data_queue.get()
        if task is None:  # sentinel
            if rank == 0:
                logger.info(f"Received stop signal, exiting")
            break
        
        idx, item = task
        
        try:
            context = item['context']
            if args_all.rag > 0:
                template = template_rag
                retrieved = item["retrieved_context"][:args_all.rag]
                retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
                context = '\n\n'.join([f"Retrieved chunk {i+1}: {x['content']}" for i, x in enumerate(retrieved)])
            elif args_all.no_context:
                template = template_no_context
            elif args_all.cot:
                template = template_0shot_cot
            else:
                template = template_0shot
            
            prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())

            if args_all.cot:
                output = query_llm_local(prompt, model, tokenizer, device, max_len, max_new_tokens=1024)
            else:
                output = query_llm_local(prompt, model, tokenizer, device, max_len, max_new_tokens=128)

            # Reset state after each generation
            if hasattr(model, 'reset'):
                model.reset()
            
            if output == '':
                result_queue.put({"idx": idx, "error": "empty_output"})
                continue
            
            if args_all.cot:  # extract answer
                response = output.strip()
                item['response_cot'] = response
                prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
                output = query_llm_local(prompt, model, tokenizer, device, max_len, max_new_tokens=128)
                
                if hasattr(model, 'reset'):
                    model.reset()
                
                if output == '':
                    result_queue.put({"idx": idx, "error": "empty_cot_output"})
                    continue
            
            response = output.strip()
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['context'] = context[:1000]

            result_queue.put({"idx": idx, "item": item})
            
        except Exception as e:
            logger.error(f"[GPU {rank}] Error processing sample {idx}: {e}")
            result_queue.put({"idx": idx, "error": str(e)})


def result_writer(result_queue, out_path, total_samples, pbar_queue, start_idx=0):
    """Writer process; writes results in order."""
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
                    json.dump(res["item"], ensure_ascii=False, fp=f)
                    f.write('\n')
            next_idx += 1
            written_count += 1
            pbar_queue.put(1)
    
    logger.info(f"Result writer finished, wrote {written_count} samples")


def run_parallel_inference(data_all, world_size, max_len, out_path, args_all, start_idx=0):
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
            args=(rank, world_size, data_queue, result_queue, max_len, args_all)
        )
        p.start()
        workers.append(p)
    
    writer = mp.Process(
        target=result_writer,
        args=(result_queue, out_path, total_samples, pbar_queue, start_idx)
    )
    writer.start()

    with tqdm(total=total_samples, desc="Processing LongBench-v2") as pbar:
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


def main():
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Args: {args}")
    
    world_size = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {world_size}")
    logger.info(f"Batch size (default = GPU count): {world_size}, each GPU processes 1 sample")
    
    env_conf = get_env_conf(args.env_conf)
    model_path = env_conf.get("model", {}).get("model_name", "")
    method = env_conf.get("model", {}).get("model_method", "origin")

    model_key = Path(model_path).name if "/" in model_path else model_path
    logger.info(f"Model: {model_path}")
    logger.info(f"Method: {method}")

    if model_key in maxlen_map:
        max_len = maxlen_map[model_key]
    else:
        max_len = 120000
    logger.info(f"Max length: {max_len}")

    is_offline = "offline" in args.env_conf.lower()
    offline_suffix = "_offline" if is_offline else ""

    if method in ["origin", "sparse", "spotlight", "sparsetopk", "magicpig", "bpc"]:
        method_suffix = f"_{method}{offline_suffix}"
    else:
        # Methods that take max_capacity_prompt
        max_cap = args.max_capacity_prompt if args.max_capacity_prompt else 512
        method_suffix = f"_{method}{offline_suffix}_{max_cap}"
    
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, model_key + f"_rag_{str(args.rag)}{method_suffix}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, model_key + f"_no_context{method_suffix}.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, model_key + f"_cot{method_suffix}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, model_key + f"{method_suffix}.jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], 
                 "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], 
                 "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], 
                 "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} 
                for item in dataset]

    # Resume: count completed samples from existing output
    start_idx = 0
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            lines = f.readlines()
            start_idx = len(lines)
        logger.info(f"Found {start_idx} completed samples, resuming from index {start_idx}")
    
    total_samples = len(data_all)
    
    if start_idx >= total_samples:
        logger.info(f"Already completed ({start_idx}/{total_samples}), skipping...")
        return
    elif start_idx > 0:
        logger.info(f"Resuming from {start_idx}/{total_samples} samples")

    logger.info(f"Total samples: {total_samples}, Remaining: {total_samples - start_idx}")
    logger.info(f"Output file: {out_file}")

    precompile_cuda_kernel_if_needed(env_conf)

    mp.set_start_method('spawn', force=True)

    run_parallel_inference(
        data_all, world_size, max_len, out_file, args, start_idx
    )
    
    logger.info(f"Finished, results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="outputs/longbench_v2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cot", "-cot", action='store_true')  # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true')  # set to True if using no context
    parser.add_argument("--rag", "-rag", type=int, default=0)  # set to N when using top-N retrieved context

    parser.add_argument("--env_conf", type=str, required=True,
                        help="Config file path (e.g., config/llama3-1-8b-ins-snapkv.json)")

    parser.add_argument("--max_capacity_prompt", type=int, default=None,
                        help="Max capacity prompt (overrides config)")
    
    args = parser.parse_args()
    main()
