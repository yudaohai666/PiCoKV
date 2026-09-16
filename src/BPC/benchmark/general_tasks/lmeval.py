#!/usr/bin/env python3
"""
General Tasks Evaluation Runner for MySparse

This script provides a simplified interface for running general tasks evaluations
(GLUE, SuperGLUE, etc.) using the existing MyModel class from mysparse.lmeval.

Usage:
    python general_tasks/lmeval.py \
        --env_conf config/llama3-1-8b-ins-origin.json \
        --tasks cola,mnli,mrpc \
        --output_path outputs/general_tasks
        
    # With method override and KV cache parameters
    python general_tasks/lmeval.py \
        --env_conf config/llama3-1-8b-ins-snapkv.json \
        --max_capacity_prompt 512 \
        --tasks cola,mnli \
        --limit 100
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
from lm_eval import evaluator, utils
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
        description="Run general tasks evaluation using MySparse framework"
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
        default=None,
        help="Comma-separated list of tasks or path to tasks JSON file (default: use lmeval.json)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_max_length", 
        type=int, 
        default=4096,
        help="Maximum sequence length for the model (default: 4096)"
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
        default=1000,
        help="Limit number of examples per task (default: 1000)"
    )
    parser.add_argument(
        "--fewshot", 
        type=int, 
        default=None,
        help="Number of few-shot examples (default: None, use each task's default)"
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
    
    # Additional options
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
    
    return parser.parse_args()


def load_tasks(args):
    """Load tasks from arguments or default config file."""
    if args.tasks is None:
        # Use default tasks from lmeval.json
        tasks_file = os.path.join(os.path.dirname(__file__), "lmeval.json")
        with open(tasks_file, 'r') as f:
            tasks = json.load(f)
        logger.info(f"Loaded default tasks from: {tasks_file}")
    elif args.tasks.endswith('.json'):
        # Load from JSON file
        with open(args.tasks, 'r') as f:
            tasks = json.load(f)
        logger.info(f"Loaded tasks from: {args.tasks}")
    else:
        # Parse comma-separated task list
        tasks = [task.strip() for task in args.tasks.split(',')]
    
    return tasks


def print_config_info(env_conf, extra_config=None, args=None):
    """Log main config, extra config, and CLI overrides."""
    logger.info("=" * 60)
    logger.info("CONFIGURATION INFORMATION")
    logger.info("=" * 60)

    model_conf = env_conf.get("model", {})
    logger.info(f"Model Name: {model_conf.get('model_name', 'unknown')}")
    logger.info(f"Model Method: {model_conf.get('model_method', 'origin')}")
    logger.info(f"Model Structure: {model_conf.get('model_structure', 'unknown')}")
    logger.info(f"Model Dtype: {model_conf.get('model_dtype', 'unknown')}")
    logger.info(f"Device Map: {model_conf.get('device_map', 'auto')}")

    extra_config_path = model_conf.get("config")
    if extra_config_path:
        logger.info(f"Extra Config Path: {extra_config_path}")

    if extra_config:
        logger.info("-" * 40)
        logger.info("EXTRA CONFIG PARAMETERS:")
        for key, value in extra_config.items():
            logger.info(f"  {key}: {value}")

    if args:
        logger.info("-" * 40)
        logger.info("COMMAND LINE ARGUMENTS:")
        logger.info(f"  model_max_length: {args.model_max_length}")
        logger.info(f"  batch_size: {args.batch_size}")
        logger.info(f"  max_gen_toks: {args.max_gen_toks}")
        if args.fewshot is not None:
            logger.info(f"  fewshot: {args.fewshot}")
        else:
            logger.info("  fewshot: None (use each task's default)")
        logger.info(f"  limit: {args.limit}")
        logger.info(f"  seed: {args.seed}")
        if args.max_capacity_prompt is not None:
            logger.info(f"  max_capacity_prompt (override): {args.max_capacity_prompt}")
        logger.info(f"  apply_chat_template: {args.apply_chat_template}")
        logger.info(f"  fewshot_as_multiturn: {args.fewshot_as_multiturn}")
    
    logger.info("=" * 60)


def load_model_and_tokenizer(args):
    """
    Load model and tokenizer from environment configuration.
    Command line arguments can override extra_config parameters.
    """
    logger.info(f"Loading model using env_conf: {args.env_conf}")
    
    # Load configuration
    env_conf = get_env_conf(args.env_conf)
    
    # Check if we need to override extra_config parameters
    extra_config_path = env_conf.get("model", {}).get("config")
    method = env_conf.get("model", {}).get("model_method", "origin")
    
    extra_config = None
    if extra_config_path:
        try:
            with open(extra_config_path, 'r') as f:
                extra_config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Extra config file not found: {extra_config_path}")

    need_override = (extra_config_path and args.max_capacity_prompt is not None) or \
                    (extra_config_path and method == "rocketkv")

    if need_override:
        try:
            if extra_config is None:
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

            # rocketkv requires max_new_tokens (general_tasks default 128)
            if method == "rocketkv":
                extra_config["max_new_tokens"] = args.max_gen_toks
                logger.info(f"Override max_new_tokens={args.max_gen_toks}")
            
            # Write to temporary config file
            os.makedirs("/tmp/mysparse_configs", exist_ok=True)
            temp_config_path = f"/tmp/mysparse_configs/temp_general_{method}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(extra_config, f, indent=4)
            env_conf["model"]["config"] = temp_config_path
        except FileNotFoundError:
            logger.warning(f"Extra config file not found: {extra_config_path}")
    
    print_config_info(env_conf, extra_config, args)
    
    try:
        # Load model and tokenizer using unified interface
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
    
    # Generate output path similar to longbench and ruler
    if method in ["origin", "spotlight", "magicpig", "bpc", "bpc-offline"]:
        save_path = f"outputs/general_tasks/{model_key}/{method}/"
    else:
        # For methods that need max_capacity_prompt
        max_cap = args.max_capacity_prompt if args.max_capacity_prompt else 512
        save_path = f"outputs/general_tasks/{model_key}/{method}/{max_cap}/"
    
    return save_path


def create_model_adapter(model, tokenizer, args):
    """Create MyModel wrapper for evaluation."""
    adapter = MyModel(
        model=model,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        batch_size=args.batch_size,
        add_bos_token=args.add_bos_token,
        prefix_token_id=args.prefix_token_id,
        max_gen_toks=args.max_gen_toks
    )
    
    logger.info(f"MyModel created with max_length={args.model_max_length}, batch_size={args.batch_size}, max_gen_toks={args.max_gen_toks}")
    
    return adapter


def run_evaluation(adapter, tasks, args):
    """Run evaluation on specified tasks."""
    logger.info(f"Starting evaluation on tasks: {tasks}")
    if args.fewshot is not None:
        logger.info(f"Few-shot examples: {args.fewshot}")
    else:
        logger.info("Few-shot examples: using each task's default")
    logger.info(f"Sample limit per task: {args.limit}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.apply_chat_template:
        logger.info("Chat template enabled")
    
    try:
        # Build kwargs, only include num_fewshot if explicitly specified
        eval_kwargs = {
            "model": adapter,
            "batch_size": args.batch_size,
            "tasks": tasks,
            "limit": args.limit,
            "apply_chat_template": args.apply_chat_template,
            "fewshot_as_multiturn": args.fewshot_as_multiturn,
        }
        if args.fewshot is not None:
            eval_kwargs["num_fewshot"] = args.fewshot
        
        result = evaluator.simple_evaluate(**eval_kwargs)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable objects."""
    def default(self, obj):
        if callable(obj):
            return f"<function: {obj.__name__ if hasattr(obj, '__name__') else str(obj)}>"
        if hasattr(obj, '__dict__'):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def save_results(results, args, env_conf, output_path):
    """Save evaluation results to files."""
    # Extract model name from config
    model_name = env_conf.get("model", {}).get("model_name", "unknown")
    model_key = Path(model_name).name if "/" in model_name else model_name
    
    # Get method from config
    method = env_conf.get("model", {}).get("model_method", "origin")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save detailed results
    detailed_results_file = os.path.join(output_path, f"detailed_results_{model_key}_{method}.json")
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=SafeJSONEncoder)
    logger.info(f"Detailed results saved to: {detailed_results_file}")
    
    # Save metrics only
    metrics = results.get('results', {})
    metrics_file = os.path.join(output_path, f"metrics_{model_key}_{method}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info(f"Metrics saved to: {metrics_file}")
    
    # Save to all_metrics.json for aggregation
    all_metrics_file = os.path.join(output_path, "all_metrics.json")
    if os.path.exists(all_metrics_file):
        with open(all_metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    all_metrics["general_tasks"] = metrics
    
    with open(all_metrics_file, "w", encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    logger.info(f"All metrics saved to: {all_metrics_file}")


def print_results_summary(results):
    """Print evaluation results summary."""
    logger.info("\n" + "="*60)
    logger.info("GENERAL TASKS EVALUATION RESULTS SUMMARY")
    logger.info("="*60)
    
    if "results" in results:
        for task_name, task_results in results["results"].items():
            logger.info(f"\nTask: {task_name}")
            logger.info("-" * 40)
            if isinstance(task_results, dict):
                for metric_name, metric_value in task_results.items():
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
    
    logger.info("General Tasks Evaluation for MySparse")
    logger.info(f"Config: {args.env_conf}")
    
    try:
        # Load tasks
        tasks = load_tasks(args)
        logger.info(f"Tasks: {tasks}")
        
        if args.limit:
            logger.info(f"Sample limit: {args.limit}")
        
        # Load model and tokenizer (with parameter overrides)
        model, tokenizer, env_conf = load_model_and_tokenizer(args)
        
        # Generate output path
        output_path = get_output_path(args, env_conf)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output path: {output_path}")
        
        # Create model adapter
        adapter = create_model_adapter(model, tokenizer, args)
        
        # Run evaluation
        results = run_evaluation(adapter, tasks, args)
        
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
