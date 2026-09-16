import json
from pathlib import Path
import time
import os
import torch
import argparse
from transformers import AutoTokenizer, GenerationConfig
from loguru import logger



from benchmark.infinitebench.eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
    iter_jsonl,
)

from bpc.misc import get_model_and_tokenizer, get_env_conf


MAX_POSITION_ID = 128*1024  # Determined by the model
TRUNCATE_LEN = 128*1024


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name_lower = model_name.lower()

    # Match instruct/chat models only (must contain instruct/chat/-ins)
    is_instruct = any(kw in model_name_lower for kw in ['instruct', 'chat', '-ins'])

    if 'llama-2' in model_name_lower and is_instruct:
        prompt = f"[INST]{prompt}[/INST]"
    elif ('llama-3' in model_name_lower or 'llama3' in model_name_lower) and is_instruct:
        # Accepts forms like "Llama-3.1-8B-Instruct" / "llama3-1-8b-ins";
        # rejects base models like "llama3-8b".
        print("Using Llama-3 instruct model chat")
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def parse_args():
    """Parse CLI args."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",
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
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--env_conf", type=str, required=True,
                   help="Config file path (e.g., config/llama3-1-8b-ins-snapkv.json)")

    # CLI override; other params come from the config file
    p.add_argument("--max_capacity_prompt", type=int, default=None,
                   help="Max capacity prompt (overrides config)")
    
    return p.parse_args()


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
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
    model_name: str = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    # Handle device for Modifier wrappers (e.g. Sparse)
    if hasattr(model, 'model') and hasattr(model.model, 'device'):
        device = model.model.device
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = "cuda"

    print("Truncating...")
    input_tokens = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    truncated_text = tok.decode(input_tokens, skip_special_tokens=True)
    
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")

    if tok.chat_template is not None:
        print("chat model detected, using chat_template")
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
        # Fallback to manual chat template
        prompt = build_chat(tok, truncated_text, model_name)
        input_tensors = tok(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_tensors["input_ids"].shape[-1]

    outputs = model.generate(**input_tensors, generation_config=generation_config)

    # Reset post-generation state (e.g. CakeKV)
    if hasattr(model, 'reset'):
        model.reset()

    output = outputs[0, context_length:]
    output = tok.decode(output, skip_special_tokens=True)
    output = output.strip()

    print("Chunked generation:", output)
    return output


def load_model(args=None, max_new_tokens=None):
    """Load all methods via get_model_and_tokenizer; CLI args override extra_config."""
    print("Loading tokenizer and model")
    start_time = time.time()

    env_conf = get_env_conf(args.env_conf)
    logger.info(f"Loaded config from: {args.env_conf}")

    extra_config_path = env_conf.get("model", {}).get("config")
    method = env_conf.get("model", {}).get("model_method", "origin")

    need_override = (extra_config_path and args.max_capacity_prompt is not None) or \
                    (extra_config_path and method == "rocketkv" and max_new_tokens is not None)

    if need_override:
        try:
            with open(extra_config_path, 'r') as f:
                extra_config = json.load(f)

            # quest/rocketkv/clusterkv use token_budget; other methods use max_capacity_prompt
            if args.max_capacity_prompt is not None:
                if method in ["quest", "rocketkv", "clusterkv"]:
                    extra_config["token_budget"] = args.max_capacity_prompt
                    logger.info(f"Override token_budget={args.max_capacity_prompt}")
                else:
                    extra_config["max_capacity_prompt"] = args.max_capacity_prompt
                    logger.info(f"Override max_capacity_prompt={args.max_capacity_prompt}")

            if method == "rocketkv" and max_new_tokens is not None:
                extra_config["max_new_tokens"] = max_new_tokens
                logger.info(f"Override max_new_tokens={max_new_tokens}")

            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_{method}_infinitebench.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            logger.warning(f"Extra config file not found: {extra_config_path}")

    model_conf = env_conf.get("model", {})
    model_name = model_conf.get("model_name", "")
    logger.info(f"Model path: {model_name}")
    logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
    tok, llm = get_model_and_tokenizer(**model_conf)
    tok.pad_token = tok.eos_token

    device = torch.device(args.device if args.device else 'cuda')
    try:
        if hasattr(llm, 'model') and hasattr(llm.model, 'device'):
            current_device = llm.model.device
        elif hasattr(llm, 'decoder') and hasattr(llm.decoder, 'device'):
            current_device = llm.decoder.device
        else:
            current_device = next(iter(llm.parameters())).device

        if current_device.type != device.type:
            llm = llm.to(device)
    except (StopIteration, AttributeError):
        llm = llm.to(device)
    
    llm.eval()
    
    print("Time taken:", round(time.time() - start_time))
    return llm, tok, model_name, method


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


if __name__ == "__main__":
    
    args = parse_args()

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok, model_name, method = load_model(args=args, max_new_tokens=max_tokens)
    
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    
    # Data
    model_key = Path(model_name).name

    # Append _offline suffix if config filename contains "offline"
    is_offline = "offline" in args.env_conf.lower()
    offline_suffix = "_offline" if is_offline else ""

    result_dir = Path(args.output_dir, model_key, f"{method}{offline_suffix}")
    if method not in ["origin", "spotlight", "magicpig", "bpc"]:
        # Methods that take max_capacity_prompt nest under that value
        max_cap = args.max_capacity_prompt if args.max_capacity_prompt else 512
        result_dir = result_dir / str(max_cap)
    print("result_dir: ", result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"
        )

    # Resume: load completed sample IDs to skip
    completed_ids, preds = load_existing_results(output_path)
    
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    print(f"Method: {method}")
    
    for i in range(args.start_idx, args.stop_idx):
        if i in completed_ids:
            print(f"====== Example {i} ====== [SKIPPED - already completed]")
            continue
            
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose, 
            generation_config=generation_config, model_name=model_name
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
