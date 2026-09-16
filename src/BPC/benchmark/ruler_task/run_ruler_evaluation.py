#!/usr/bin/env python3
"""
RULER Evaluation Runner for MySparse

This script provides a simplified interface for running RULER evaluations
using the existing MyModel class from mysparse.lmeval.

Usage:
    python ruler_task/run_ruler_evaluation.py \
        --env_conf config/llama3-1-8b-ins-origin.json \
        --tasks ruler_cwe \
        --max_seq_lengths 4096 \
        --output_path outputs/ruler_lmeval
        
    # With method override and KV cache parameters
    python ruler_task/run_ruler_evaluation.py \
        --env_conf config/llama3-1-8b-ins-snapkv.json \
        --model llama3-1-8b-ins \
        --method snapkv \
        --max_capacity_prompt 512 \
        --tasks ruler_cwe \
        --max_seq_lengths 4096
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import random


import torch
from lm_eval import evaluator
from loguru import logger

from bpc.lmeval import MyModel
from bpc.misc import get_model_and_tokenizer, get_env_conf



def seed_everything(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RULER evaluation using MySparse framework"
    )
    
    # Required arguments
    parser.add_argument(
        "--env_conf", 
        type=str, 
        required=True,
        help="Path to environment configuration JSON file (contains model and method)"
    )
    
    # Task configuration
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="ruler_cwe",
        help="Comma-separated list of tasks to evaluate (default: ruler_cwe)"
    )
    parser.add_argument(
        "--max_seq_lengths",
        type=str,
        default="4096",
        help="Comma-separated list of max sequence lengths to evaluate (default: 4096)"
    )
    
    # Model configuration
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=131072,
        help="Maximum sequence length for the model (default: 131072)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--max_gen_toks", 
        type=int, 
        default=128,
        help="Maximum number of tokens to generate (default: 128)"
    )
    
    # KV cache method parameters (overrides extra_config)
    # Only max_capacity_prompt is commonly changed, other params load from config
    parser.add_argument(
        "--max_capacity_prompt", 
        type=int, 
        default=None,
        help="Max capacity prompt (overrides config)"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples per task (for testing)"
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0,
        help="Number of few-shot examples (default: 0)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save results (default: auto-generated based on model and method)"
    )
    parser.add_argument(
        "--write_out", 
        action="store_true",
        default=True,
        help="Write out individual predictions (default: True)"
    )
    
    # Chat template configuration
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Apply chat template for instruct/chat models (default: False)"
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="Format few-shot examples as multi-turn conversation (default: False)"
    )
    
    # Additional options (consistent with general_tasks)
    parser.add_argument(
        "--add_bos_token", 
        action="store_true",
        help="Add BOS token to inputs"
    )
    parser.add_argument(
        "--prefix_token_id", 
        type=int, 
        default=None,
        help="Custom prefix token ID"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """
    Load model and tokenizer from environment configuration.
    """
    logger.info(f"Loading model using env_conf: {args.env_conf}")
    
    # Load configuration
    env_conf = get_env_conf(args.env_conf)
    
    # Check if we need to override extra_config parameters
    extra_config_path = env_conf.get("model", {}).get("config")
    
    # Get method from config
    method = env_conf.get("model", {}).get("model_method", "unknown")
    
    need_override = (extra_config_path and args.max_capacity_prompt is not None) or \
                    (extra_config_path and method == "rocketkv")

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

            # rocketkv requires max_new_tokens (ruler default 128)
            if method == "rocketkv":
                extra_config["max_new_tokens"] = args.max_gen_toks
                logger.info(f"Override max_new_tokens={args.max_gen_toks}")
            
            # Write to temporary config file
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_ruler_{method}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            logger.warning(f"Extra config file not found: {extra_config_path}")
    
    try:
        # Load model and tokenizer
        model_conf = env_conf.get("model", {})
        logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
        tokenizer, model = get_model_and_tokenizer(**model_conf)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        logger.info("Model loaded successfully")
        
        return model, tokenizer, env_conf
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_output_path(args, env_conf):
    """Generate output path based on model and method from config."""
    if args.output_path:
        return args.output_path
    
    # Get model name from config
    model_name = env_conf.get("model", {}).get("model_name", "unknown")
    # Extract model key from path
    model_key = Path(model_name).name if "/" in model_name else model_name
    
    # Get method from config
    method = env_conf.get("model", {}).get("model_method", "origin")
    
    # Check if bpc-offline by examining config file path
    if method == "bpc":
        extra_config = env_conf.get("model", {}).get("config", "")
        if "bpc-offline" in extra_config:
            method = "bpc-offline"
    
    # Generate output path similar to longbench
    if method in ["origin", "spotlight", "magicpig", "bpc", "bpc-offline" ,"topk"]:
        save_path = f"outputs/ruler/{model_key}/{method}/"
    else:
        # For methods that need max_capacity_prompt
        max_cap = args.max_capacity_prompt if args.max_capacity_prompt else 512
        save_path = f"outputs/ruler/{model_key}/{method}/{max_cap}/"
    
    return save_path


def create_ruler_model(model, tokenizer, args):
    """Create MyModel wrapper for RULER evaluation."""
    ruler_model = MyModel(
        model=model,
        tokenizer=tokenizer,
        model_max_length=args.max_length,
        batch_size=args.batch_size,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        max_gen_toks=args.max_gen_toks
    )
    
    logger.info(f"MyModel created with max_length={args.max_length}, batch_size={args.batch_size}, max_gen_toks={args.max_gen_toks}")
    
    return ruler_model


def run_evaluation(ruler_model, args, env_conf, output_path):
    """Run RULER evaluation."""
    # Parse tasks and sequence lengths
    tasks = [task.strip() for task in args.tasks.split(",")]
    max_seq_lengths = [int(x.strip()) for x in args.max_seq_lengths.split(",")]
    
    logger.info(f"Evaluating tasks: {tasks} with sequence lengths: {max_seq_lengths}")
    
    # Validate sequence lengths
    max_seq_length_needed = max(max_seq_lengths)
    if args.max_length < max_seq_length_needed:
        logger.warning(f"Model max_length ({args.max_length}) < longest sequence ({max_seq_length_needed})")
    
    # Get model path for tokenizer initialization in RULER tasks
    model_path = env_conf["model"].get("model_name", "")
    if not model_path:
        model_path = getattr(ruler_model.tokenizer, 'name_or_path', '')
    
    # Prepare metadata for RULER evaluation
    metadata = {
        "max_seq_lengths": max_seq_lengths,
        "pretrained": model_path,
        "tokenizer": model_path,
    }
    
    model_args_str = f"pretrained={model_path},max_length={args.max_length}"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Run evaluation using lm_eval (RULER is built-in)
        logger.info("Starting RULER evaluation...")
        logger.info(f"Model/Tokenizer path: {model_path}")
        if args.apply_chat_template:
            logger.info("Chat template enabled")
        
        results = evaluator.simple_evaluate(
            model=ruler_model,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            write_out=args.write_out,
            log_samples=True,
            metadata=metadata,
            model_args=model_args_str,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_results(results, args, env_conf, output_path):
    """Save evaluation results to files."""
    # Extract model name from config
    model_name = env_conf.get("model", {}).get("model_name", "unknown")
    model_name = Path(model_name).name if "/" in model_name else model_name
    
    # Get method from config
    method = env_conf.get("model", {}).get("model_method", "origin")
    
    # Create a lightweight copy of results (without large input contexts)
    results_lite = {
        "results": results.get("results", {}),
        "config": results.get("config", {}),
    }
    
    # Save main results (metrics only, no samples)
    results_file = os.path.join(output_path, f"results_{model_name}_{method}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_lite, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_file}")
    
    # Save lightweight predictions (without input context)
    if "samples" in results:
        jsonl_file = os.path.join(output_path, f"predictions_{model_name}_{method}.jsonl")
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for task_name, task_samples in results["samples"].items():
                for sample in task_samples:
                    doc = sample.get("doc", {})
                    # Only keep essential fields, exclude large 'input' field
                    pred_data = {
                        "task": task_name,
                        "pred": sample.get("resps", [""])[0] if sample.get("resps") else "",
                        "target": sample.get("target", ""),
                        "index": doc.get("index", 0),
                        "length": doc.get("length", 0),
                        "max_length": doc.get("max_length", 0),
                        "outputs": doc.get("outputs", []),
                    }
                    f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
        logger.info(f"Predictions saved to: {jsonl_file}")


def print_results_summary(results):
    """Print evaluation results summary."""
    logger.info("\n" + "="*60)
    logger.info("RULER EVALUATION RESULTS SUMMARY")
    logger.info("="*60)
    
    if "results" in results:
        for task_name, task_results in results["results"].items():
            logger.info(f"\nTask: {task_name}")
            logger.info("-" * 40)
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    if metric_value == -1:
                        logger.info(f"  {metric_name:20s}: N/A (no samples)")
                    else:
                        logger.info(f"  {metric_name:20s}: {metric_value:.4f}")
                else:
                    logger.info(f"  {metric_name:20s}: {metric_value}")
    
    # Print aggregate results if available
    if "results" in results and "ruler" in results["results"]:
        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATE RULER RESULTS")
        logger.info("="*60)
        ruler_results = results["results"]["ruler"]
        for metric_name, metric_value in ruler_results.items():
            if isinstance(metric_value, (int, float)):
                if metric_value == -1:
                    logger.info(f"  {metric_name:20s}: N/A (no samples)")
                else:
                    logger.info(f"  {metric_name:20s}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name:20s}: {metric_value}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed
    seed_everything(args.seed)
    
    logger.info("RULER Evaluation for MySparse")
    logger.info(f"Config: {args.env_conf}")
    logger.info(f"Tasks: {args.tasks}")
    if args.limit:
        logger.info(f"Sample limit: {args.limit}")
    
    try:
        # Load model and tokenizer (with parameter overrides)
        model, tokenizer, env_conf = load_model_and_tokenizer(args)
        
        # Generate output path
        output_path = get_output_path(args, env_conf)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output path: {output_path}")
        
        # Create RULER model wrapper
        ruler_model = create_ruler_model(model, tokenizer, args)
        
        # Run evaluation
        results = run_evaluation(ruler_model, args, env_conf, output_path)
        
        # Save results
        save_results(results, args, env_conf, output_path)
        
        # Print summary
        print_results_summary(results)
        
        logger.info("\nEVALUATION COMPLETED SUCCESSFULLY!")
        
        return 0
        
    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())