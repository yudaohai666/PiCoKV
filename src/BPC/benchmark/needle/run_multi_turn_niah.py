"""
Multi-turn Needle In A Haystack (Multi-turn NIAH) Test

This script implements the Multi-turn NIAH test as described in the SHADOWKV paper.
The key difference from standard NIAH is that each turn queries a DIFFERENT needle,
testing whether KV cache eviction strategies can maintain accuracy across multiple turns.

Key insight from SHADOWKV paper:
- SnapKV drops significantly from the second round because it evicts tokens based on 
  the first-turn conversation
- StreamingLLM also struggles as it only keeps recent tokens
- SHADOWKV can maintain accuracy in multi-turn settings

Usage:
    python run_multi_turn_niah.py --env_conf config/xxx.json --context_length 8000 16000 32000
"""

import os
import glob
import json
import numpy as np
import argparse
import time
import torch
import random
import tqdm
import logging
from datetime import datetime, timezone
from loguru import logger
from rouge_score import rouge_scorer

from bpc.misc import get_model_and_tokenizer, get_env_conf


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# Define multiple needles for multi-turn testing
# Each needle has a unique fact that can be queried independently
MULTI_TURN_NEEDLES = [
    {
        "needle": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        "question": "What is the best thing to do in San Francisco?",
        "answer_key": "eat a sandwich and sit in Dolores Park on a sunny day"
    },
    {
        "needle": "\nThe secret recipe for the perfect pizza includes fresh basil, San Marzano tomatoes, and buffalo mozzarella.\n",
        "question": "What is the secret recipe for the perfect pizza?",
        "answer_key": "fresh basil, San Marzano tomatoes, and buffalo mozzarella"
    },
    {
        "needle": "\nThe most important programming principle is to write code that is easy to read and maintain.\n",
        "question": "What is the most important programming principle?",
        "answer_key": "write code that is easy to read and maintain"
    },
    {
        "needle": "\nThe key to successful investing is diversification and long-term thinking.\n",
        "question": "What is the key to successful investing?",
        "answer_key": "diversification and long-term thinking"
    },
    {
        "needle": "\nThe ancient library of Alexandria contained over 400,000 scrolls of knowledge.\n",
        "question": "How many scrolls did the ancient library of Alexandria contain?",
        "answer_key": "400,000 scrolls"
    },
]


class MultiTurnNIAHTester:
    """
    Multi-turn Needle In A Haystack Tester
    
    Tests model's ability to retrieve different needles across multiple conversation turns.
    This is designed to evaluate KV cache eviction strategies like SnapKV, StreamingLLM, etc.
    
    Key design for testing KV cache eviction:
    - All needles are inserted into the context at different depths
    - Each turn queries a DIFFERENT needle at a DIFFERENT position
    - KV cache is reused across turns (via past_key_values)
    - This tests whether eviction strategies preserve necessary information for future queries
    """
    
    def __init__(self,
                 args,
                 needles=MULTI_TURN_NEEDLES,
                 haystack_dir="benchmark/needle/PaulGrahamEssays",
                 num_turns=3,  # Number of conversation turns
                 context_lengths=None,
                 document_depth_percents=None,
                 model_path='',
                 model_version=None,
                 method='origin',
                 max_capacity_prompts=0,
                 final_context_length_buffer=200,
                 save_results=True,
                 print_ongoing_status=True):
        
        self.args = args
        self.needles = needles[:num_turns]  # Use only as many needles as turns
        self.num_turns = min(num_turns, len(needles))
        self.haystack_dir = haystack_dir
        self.context_lengths = context_lengths
        self.document_depth_percents = document_depth_percents
        self.model_path = model_path
        self.method = method
        self.max_capacity_prompts = max_capacity_prompts
        self.final_context_length_buffer = final_context_length_buffer
        self.save_results = save_results
        self.print_ongoing_status = print_ongoing_status
        self.save_dir = args.save_dir
        self.testing_results = []
        
        # Cache for context to avoid re-reading files
        self._cached_context = None
        
        # Check if offline mode
        self.is_offline = "offline" in args.env_conf.lower()
        
        # Set model version
        if model_version:
            self.model_version = model_version
        elif "/" in model_path:
            self.model_version = model_path.split("/")[-1]
        else:
            self.model_version = model_path
            
        # Load model using unified method
        env_conf = get_env_conf(args.env_conf)
        logger.info(f"Loaded config from: {args.env_conf}")
        
        # Override config with command line args
        if args.method:
            env_conf.get("model", {})["model_method"] = args.method
        env_conf.get("model", {})["model_name"] = model_path
        
        # Handle extra config overrides
        self._setup_extra_config(args, env_conf)
        
        # Load model and tokenizer
        model_conf = env_conf.get("model", {})
        logger.info(f"Loading model with method: {model_conf.get('model_method', 'unknown')}")
        self.enc, self.model = get_model_and_tokenizer(**model_conf)
        
    def _setup_extra_config(self, args, env_conf):
        """Setup extra config with command line overrides"""
        extra_config_path = env_conf.get("model", {}).get("config")
        if extra_config_path and any([
            args.max_capacity_prompts is not None,
            args.window_size is not None,
            args.kernel_size is not None,
        ]):
            try:
                with open(extra_config_path, 'r') as f:
                    extra_config = json.load(f)
                
                method = env_conf.get("model", {}).get("model_method", "origin")
                if args.max_capacity_prompts is not None:
                    # quest/clusterkv use token_budget; other methods use max_capacity_prompt
                    if method in ["quest", "clusterkv"]:
                        extra_config["token_budget"] = args.max_capacity_prompts
                    else:
                        extra_config["max_capacity_prompt"] = args.max_capacity_prompts
                
                if args.window_size is not None:
                    extra_config["window_size"] = args.window_size
                if args.kernel_size is not None:
                    extra_config["kernel_size"] = args.kernel_size
                
                # Write temp config
                os.makedirs("/tmp/mysparse_configs", exist_ok=True)
                temp_config_path = f"/tmp/mysparse_configs/temp_multi_turn_niah_{args.method}.json"
                with open(temp_config_path, 'w') as f:
                    json.dump(extra_config, f, indent=4)
                env_conf["model"]["config"] = temp_config_path
            except FileNotFoundError:
                logger.warning(f"Extra config file not found: {extra_config_path}")
    
    def encode_text_to_tokens(self, text):
        return self.enc.encode(text)
    
    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length])
    
    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context))
    
    def read_context_files(self):
        """Read haystack context files (with caching)"""
        if self._cached_context is not None:
            return self._cached_context
            
        context = ""
        max_context_length = max(self.context_lengths)
        logger.info(f"Building context for max length: {max_context_length} tokens...")
        
        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        
        logger.info(f"Context built: {self.get_context_length_in_tokens(context)} tokens")
        self._cached_context = context
        return context
    
    def insert_needles_at_different_depths(self, context, context_length, depth_percents):
        """
        Insert multiple needles at different depth percentages.
        
        Args:
            context: The haystack context
            context_length: Target context length in tokens
            depth_percents: List of depth percentages for each needle
            
        Returns:
            Modified context with all needles inserted, and insertion_info list
            The insertion_info is ordered by turn (needle index), not by depth
        """
        tokens_context = self.encode_text_to_tokens(context)
        
        # Calculate total needle length
        total_needle_length = sum(
            len(self.encode_text_to_tokens(n["needle"])) 
            for n in self.needles
        )
        
        # Adjust context length for buffer and needles
        adjusted_length = context_length - self.final_context_length_buffer - total_needle_length
        if len(tokens_context) > adjusted_length:
            tokens_context = tokens_context[:adjusted_length]
        
        # Create pairs of (needle_index, needle_info, depth_percent)
        # We need to track the original index to maintain the question order
        needle_depth_pairs = [(i, self.needles[i], depth_percents[i]) for i in range(len(self.needles))]
        
        # Sort by depth descending for insertion (insert from end to beginning)
        needle_depth_pairs_sorted = sorted(needle_depth_pairs, key=lambda x: x[2], reverse=True)
        
        # Track insertion info by original needle index
        insertion_info_dict = {}
        
        for needle_idx, needle_info, depth_percent in needle_depth_pairs_sorted:
            tokens_needle = self.encode_text_to_tokens(needle_info["needle"])
            
            if depth_percent == 100:
                tokens_context = tokens_context + tokens_needle
                insertion_point = len(tokens_context) - len(tokens_needle)
            else:
                insertion_point = int(len(tokens_context) * (depth_percent / 100))
                
                # Find sentence boundary (period)
                period_tokens = self.encode_text_to_tokens('.')
                tokens_before = tokens_context[:insertion_point]
                while tokens_before and tokens_before[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_before = tokens_context[:insertion_point]
                
                # Insert needle
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
            
            insertion_info_dict[needle_idx] = {
                "needle": needle_info["needle"],
                "question": needle_info["question"],
                "answer_key": needle_info["answer_key"],
                "depth_percent": depth_percent,
                "insertion_point": insertion_point
            }
        
        # Return insertion_info ordered by needle index (turn order)
        insertion_info = [insertion_info_dict[i] for i in range(len(self.needles))]
        
        new_context = self.decode_tokens(tokens_context)
        return new_context, insertion_info
    
    def build_first_turn_prompt(self, context, question):
        """
        Build the complete prompt for the first turn (prefill phase).
        
        Args:
            context: The haystack context with needles inserted
            question: The first question to ask
            
        Returns:
            input_ids tensor
        """
        device = next(iter(self.model.parameters())).device
        
        if self.enc.chat_template is None:
            prompt = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n"
            prompt += f"Question: {question}\nAnswer:"
            input_ids = self.enc(prompt, return_tensors="pt").input_ids.to(device)
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct."
                },
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "assistant", 
                    "content": "I have read the document. Please ask me questions about it."
                },
                {
                    "role": "user",
                    "content": f"{question} Don't give information outside the document or repeat your findings. The document definitely contains the answer."
                }
            ]
            input_ids = self.enc.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
        
        return input_ids
    
    def build_subsequent_turn_input(self, prev_response, new_question):
        """
        Build the incremental input for subsequent turns (only new tokens).
        
        This is the key for KV cache reuse: we only encode the new parts
        (previous response + new question), not the entire conversation.
        
        Args:
            prev_response: The model's response from the previous turn
            new_question: The new question for this turn
            
        Returns:
            input_ids tensor (only the new tokens)
        """
        device = next(iter(self.model.parameters())).device
        
        if self.enc.chat_template is None:
            # Simple format: previous answer + new question
            incremental_text = f" {prev_response}\n\nQuestion: {new_question}\nAnswer:"
            input_ids = self.enc(incremental_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        else:
            # Chat template format: need to encode assistant response + user question
            # We manually construct to avoid BOS token issues
            
            # Build a minimal conversation to get the format
            dummy_messages = [
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": f"{new_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer."}
            ]
            
            try:
                # Encode the incremental part
                incremental_ids = self.enc.apply_chat_template(
                    dummy_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)
                
                # IMPORTANT: Skip BOS token if present (it's already in the KV cache)
                # Check if the first token is BOS
                if self.enc.bos_token_id is not None and incremental_ids[0, 0].item() == self.enc.bos_token_id:
                    incremental_ids = incremental_ids[:, 1:]
                
                input_ids = incremental_ids
            except Exception:
                # Fallback: manually construct
                incremental_text = f"{prev_response}</s>\n<|user|>\n{new_question} Don't give information outside the document.</s>\n<|assistant|>\n"
                input_ids = self.enc(incremental_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        return input_ids
    
    def evaluate_multi_turn(self, context_length, depth_config):
        """
        Run multi-turn evaluation with TRUE KV cache reuse.
        
        This is the correct implementation for testing KV cache eviction strategies:
        
        Turn 1: prefill(context + Q1) → eviction happens → generate A1 → keep compressed KV cache
        Turn 2: reuse KV cache + only encode (A1 + Q2) → generate A2
        Turn 3: reuse KV cache + only encode (A2 + Q3) → generate A3
        
        Key insight from SHADOWKV paper:
        - SnapKV evicts tokens during Turn 1 based on Q1's attention pattern
        - These evicted tokens may contain information needed for Q2, Q3
        - So Turn 2+ performance drops because the KV cache is already compressed
        
        Args:
            context_length: Target context length
            depth_config: List of depth percentages for each needle
        """
        # Check if result exists
        if self.save_results and self.result_exists(context_length, depth_config):
            logger.info(f"Result exists for context_length={context_length}, depths={depth_config}, skipping")
            return
        
        # Read and prepare context
        raw_context = self.read_context_files()
        tokens = self.encode_text_to_tokens(raw_context)
        if len(tokens) > context_length:
            raw_context = self.decode_tokens(tokens, context_length)
        
        # Insert all needles at specified depths
        context, insertion_info = self.insert_needles_at_different_depths(
            raw_context, context_length, depth_config
        )
        
        logger.info(f"Context length: {context_length}, Needle depths: {depth_config}")
        for i, info in enumerate(insertion_info):
            logger.info(f"  Turn {i+1} needle at {info['depth_percent']}%: {info['needle'][:50]}...")
        
        # Run multi-turn conversation with KV cache reuse
        turn_results = []
        past_key_values = None  # KV cache to be reused across turns
        prev_response = None    # Previous turn's response
        total_kv_length = 0     # Track total sequence length in KV cache
        
        for turn_idx in range(self.num_turns):
            test_start_time = time.time()
            
            needle_info = insertion_info[turn_idx]
            current_question = needle_info["question"]
            
            if turn_idx == 0:
                # ===== TURN 1: Full prefill =====
                # This is where KV cache eviction happens for methods like SnapKV
                input_ids = self.build_first_turn_prompt(context, current_question)
                actual_length = input_ids.shape[-1]
                total_kv_length = actual_length
                
                logger.info(f"Turn {turn_idx + 1} (Prefill): Input length = {actual_length} tokens")
                
                # Generate with KV cache output
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        max_new_tokens=100,
                        pad_token_id=self.enc.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        use_cache=True,
                    )
                
                # Extract response (skip input tokens)
                generated_ids = outputs.sequences[0]
                response = self.enc.decode(generated_ids[actual_length:], skip_special_tokens=True).strip()
                
                # Get the KV cache for reuse in subsequent turns
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                    past_key_values = outputs.past_key_values
                    # Update total KV length (includes generated tokens)
                    total_kv_length = past_key_values[0][0].shape[2]
                    logger.info(f"  KV cache saved: {total_kv_length} tokens")
                else:
                    logger.warning("  Model does not return past_key_values, KV cache reuse disabled")
                    past_key_values = None
                
            else:
                # ===== TURN 2+: Reuse KV cache =====
                # Only encode the new tokens (previous response + new question)
                
                if past_key_values is not None:
                    # Build incremental input (only new tokens)
                    input_ids = self.build_subsequent_turn_input(prev_response, current_question)
                    actual_length = input_ids.shape[-1]
                    
                    logger.info(f"Turn {turn_idx + 1} (KV Reuse): New tokens = {actual_length}, KV cache = {total_kv_length}, Total = {total_kv_length + actual_length}")
                    
                    # Create position_ids starting from KV cache length
                    position_ids = torch.arange(
                        total_kv_length, 
                        total_kv_length + actual_length, 
                        dtype=torch.long, 
                        device=input_ids.device
                    ).unsqueeze(0)
                    
                    # Generate with past_key_values
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            past_key_values=past_key_values,
                            position_ids=position_ids,
                            num_beams=1,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            max_new_tokens=100,
                            pad_token_id=self.enc.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=False,
                            output_attentions=False,
                            output_hidden_states=False,
                            use_cache=True,
                        )
                    
                    # Extract response
                    generated_ids = outputs.sequences[0]
                    response = self.enc.decode(generated_ids[actual_length:], skip_special_tokens=True).strip()
                    
                    # Update KV cache for next turn
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        past_key_values = outputs.past_key_values
                        total_kv_length = past_key_values[0][0].shape[2]
                        logger.info(f"  KV cache updated: {total_kv_length} tokens")
                else:
                    # Fallback: no KV cache available, rebuild full context
                    logger.warning(f"Turn {turn_idx + 1}: No KV cache, falling back to full rebuild")
                    # This shouldn't happen in normal operation
                    input_ids = self.build_first_turn_prompt(context, current_question)
                    actual_length = input_ids.shape[-1]
                    total_kv_length = actual_length
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            num_beams=1,
                            do_sample=False,
                            max_new_tokens=100,
                            pad_token_id=self.enc.eos_token_id,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                    
                    generated_ids = outputs.sequences[0]
                    response = self.enc.decode(generated_ids[actual_length:], skip_special_tokens=True).strip()
                    
                    # Try to get KV cache for next turn
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                        past_key_values = outputs.past_key_values
                        total_kv_length = past_key_values[0][0].shape[2]
            
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time
            
            # Save response for next turn
            prev_response = response
            
            # Score the response
            score = scorer.score(needle_info["answer_key"], response)['rouge1'].recall * 100
            
            turn_result = {
                "turn": turn_idx + 1,
                "question": needle_info["question"],
                "expected_answer": needle_info["answer_key"],
                "needle_depth": depth_config[turn_idx],
                "response": response,
                "score": score,
                "duration_seconds": test_elapsed_time,
                "input_length": actual_length,
                "kv_cache_length": total_kv_length if past_key_values else 0,
                "kv_cache_reused": turn_idx > 0 and past_key_values is not None
            }
            turn_results.append(turn_result)
            
            if self.print_ongoing_status:
                logger.info(f"Turn {turn_idx + 1} Results:")
                logger.info(f"  Question: {needle_info['question']}")
                logger.info(f"  Expected: {needle_info['answer_key']}")
                logger.info(f"  Response: {response}")
                logger.info(f"  Score: {score:.2f}")
                logger.info(f"  Duration: {test_elapsed_time:.1f}s")
        
        # Reset model state if needed
        if hasattr(self.model, 'reset'):
            self.model.reset()
        elif self.method == "cakekv" and hasattr(self.model, 'model'):
            layers = len(self.model.model.layers)
            for i in range(layers):
                self.model.model.layers[i].self_attn.config.prefill = [True]*layers
                self.model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
        
        # Clear KV cache
        del past_key_values
        torch.cuda.empty_cache()
        
        # Save results
        results = {
            "model": self.model_path,
            "method": self.method,
            "context_length": context_length,
            "num_turns": self.num_turns,
            "depth_config": depth_config,
            "turn_results": turn_results,
            "avg_score": np.mean([r["score"] for r in turn_results]),
            "per_turn_scores": [r["score"] for r in turn_results],
            "kv_cache_reuse": True,  # Mark that this test uses KV cache reuse
            "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }
        
        self.testing_results.append(results)
        
        if self.save_results:
            self._save_results(results, context_length, depth_config)
        
        logger.info(f"Multi-turn test completed. Average score: {results['avg_score']:.2f}")
        for i, tr in enumerate(turn_results):
            logger.info(f"  Turn {i+1}: {tr['score']:.2f} (KV reused: {tr['kv_cache_reused']})")
    
    def _get_results_dir(self):
        """Generate results directory path"""
        offline_suffix = "_offline" if self.is_offline else ""
        
        if self.method in ["origin", "spotlight", "magicpig", "bpc"]:
            return f'{self.save_dir}/{self.model_version}_{self.method}{offline_suffix}_multi_turn'
        else:
            if self.max_capacity_prompts is not None:
                return f'{self.save_dir}/{self.model_version}_{self.method}{offline_suffix}_{self.max_capacity_prompts}_multi_turn'
            else:
                return f'{self.save_dir}/{self.model_version}_{self.method}{offline_suffix}_multi_turn'
    
    def _save_results(self, results, context_length, depth_config):
        """Save results to file"""
        results_dir = self._get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        
        depth_str = "_".join([str(int(d)) for d in depth_config])
        filename = f"len_{context_length}_depths_{depth_str}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also append to summary
        summary_path = os.path.join(results_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        else:
            summary = {"results": []}
        
        summary["results"].append({
            "context_length": context_length,
            "depth_config": depth_config,
            "avg_score": results["avg_score"],
            "turn_scores": [r["score"] for r in results["turn_results"]]
        })
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
    
    def result_exists(self, context_length, depth_config):
        """Check if result already exists"""
        results_dir = self._get_results_dir()
        if not os.path.exists(results_dir):
            return False
        
        depth_str = "_".join([str(int(d)) for d in depth_config])
        filename = f"len_{context_length}_depths_{depth_str}.json"
        return os.path.exists(os.path.join(results_dir, filename))
    
    def run_test(self):
        """Run all multi-turn tests"""
        logger.info(f"Starting Multi-turn NIAH Testing...")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Method: {self.method}")
        logger.info(f"Number of turns: {self.num_turns}")
        logger.info(f"Context lengths: {self.context_lengths}")
        
        # Define depth configurations to test
        # Each config specifies where each needle is placed
        depth_configs = [
            [25, 50, 75],      # Needles at 25%, 50%, 75%
            [10, 50, 90],      # Needles spread across document
            [75, 50, 25],      # Reverse order (query later needles first)
        ]
        
        if self.num_turns == 2:
            depth_configs = [
                [25, 75],
                [50, 50],
                [75, 25],
            ]
        elif self.num_turns == 4:
            depth_configs = [
                [20, 40, 60, 80],
                [10, 30, 70, 90],
            ]
        
        for context_length in tqdm.tqdm(self.context_lengths, desc="Context lengths"):
            if context_length < self.args.s_len or context_length > self.args.e_len:
                continue
            
            for depth_config in depth_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing context_length={context_length}, depths={depth_config}")
                logger.info(f"{'='*60}")
                
                try:
                    self.evaluate_multi_turn(context_length, depth_config)
                except Exception as e:
                    logger.error(f"Error in test: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.info("All multi-turn tests completed!")
        return self.testing_results


def main():
    parser = argparse.ArgumentParser(description="Multi-turn Needle In A Haystack Test")
    parser.add_argument('-s', '--s_len', default=0, type=int, help='Start context length')
    parser.add_argument('-e', '--e_len', default=131072, type=int, help='End context length')
    parser.add_argument('--model_path', type=str, default=None, help='Model key')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='Model provider type')
    parser.add_argument('--context_length', nargs='+', type=int,
                        default=[8000, 16000, 32000, 64000])
    parser.add_argument('--save_dir', type=str, default="outputs/needle", help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_turns', type=int, default=3, help='Number of conversation turns')
    
    # Method selection
    parser.add_argument('--method', type=str, default=None,
                        choices=["origin", "pyramidkv", "snapkv", "cakekv", "compresskv",
                                "streamingllm", "spotlight", "sparse", "sparsetopk", "quest",
                                "clusterkv", "bpc", "magicpig"],
                        help='Method for KV cache compression')
    
    # Config file (required)
    parser.add_argument("--env_conf", type=str, required=True,
                        help="Config file path")
    
    # Override parameters
    parser.add_argument('--max_capacity_prompts', type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument('--kernel_size', type=int, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load config
    env_conf = get_env_conf(args.env_conf)
    model_path = env_conf.get("model", {}).get("model_name", args.model_path)
    method = args.method if args.method else env_conf.get("model", {}).get("model_method", "origin")
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Method: {method}")
    
    # Create tester
    tester = MultiTurnNIAHTester(
        args=args,
        num_turns=args.num_turns,
        context_lengths=np.array(args.context_length),
        model_path=model_path,
        model_version=args.model_path,
        method=method,
        max_capacity_prompts=args.max_capacity_prompts,
    )
    
    # Run tests
    tester.run_test()


if __name__ == "__main__":
    main()
