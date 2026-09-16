"""
Compute the offline projection matrix from a calibration set.

Steps:
1. Load model and calibration data.
2. Forward over the data, collecting per-layer key_states.
3. Solve for the optimal projection matrix from the collected keys.
4. Save the matrix for later inference use.

Usage:
    python calibrate_projection.py \
        --model_name /path/to/model \
        --calibration_data /path/to/calibration.jsonl \
        --output_path /path/to/projection.pt \
        --num_samples 100 \
        --max_length 4096
"""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralFlashAttention2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2FlashAttention2
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import numpy as np

# Supported attention classes (incl. Flash Attention 2 variants).
SUPPORTED_ATTENTION_CLASSES = (
    LlamaAttention, LlamaFlashAttention2,
    MistralAttention, MistralFlashAttention2,
    Qwen2Attention, Qwen2FlashAttention2,
)

# Global key_states accumulator, keyed by layer_idx.
collected_keys = {}


def find_binary_projection_from_samples(key_samples, num_bits=64, num_iters=4, device=None):
    """Solve for the binary projection matrix from a list of per-sample key_states.

    Args:
        key_samples: list of [1, H, L, D] tensors stored on CPU.
        num_bits: number of binary bits.
        num_iters: power-iteration steps per bit.
        device: compute device (defaults to CUDA if available).

    Returns:
        quan_proj: [H, num_bits, D] projection matrix.
    """
    first_sample = key_samples[0]
    dtype = first_sample.dtype
    H, D = first_sample.shape[1], first_sample.shape[3]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_L = sum(k.shape[2] for k in key_samples)
    print(f"  Total tokens: {total_L:,}, H={H}, D={D}, num_samples={len(key_samples)}")

    # Concatenate on CPU, then move to GPU once to minimize transfers.
    print(f"  Merging {len(key_samples)} samples on CPU...")
    merged_keys = torch.cat([k[0] for k in key_samples], dim=1)  # [H, total_L, D]

    tensor_size_gb = merged_keys.numel() * merged_keys.element_size() / (1024**3)
    print(f"  Merged tensor size: {tensor_size_gb:.2f} GB, moving to GPU...")

    A = merged_keys.to(device)  # [H, total_L, D]
    del merged_keys
    torch.cuda.empty_cache()

    quan_proj = torch.empty(H, num_bits, D, device=device, dtype=dtype)

    print(f"  Starting projection computation (num_bits={num_bits}, num_iters={num_iters})...")

    for bit_idx in range(num_bits):
        Vh = torch.randn(H, D, device=device, dtype=dtype)

        # Power-iteration refine.
        for iter_idx in range(num_iters):
            Uq = torch.einsum("hld,hd->hl", A, Vh).sign()  # [H, total_L]
            Vh = torch.einsum("hld,hl->hd", A, Uq)  # [H, D]

        Vh = Vh / total_L

        # Subtract residual captured by this bit: A -= Uq (x) Vh.
        Uq = torch.einsum("hld,hd->hl", A, Vh).sign()
        A = A - torch.einsum("hl,hd->hld", Uq, Vh)

        quan_proj[:, bit_idx, :] = Vh

        if (bit_idx + 1) % 8 == 0:
            print(f"    Bit {bit_idx + 1}/{num_bits} done")
    
    del A
    torch.cuda.empty_cache()
    
    return quan_proj


def collect_key_states_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Collect key_states then delegate to the original forward.

    Compatible with LLaMA/Qwen (position_embeddings) and Mistral (position_ids).
    """
    global collected_keys

    bsz, q_len, _ = hidden_states.size()

    key_states = self.k_proj(hidden_states)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Query is needed to derive RoPE.
    query_states = self.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    # LLaMA/Qwen pass position_embeddings; Mistral provides position_ids only.
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        # Mistral path: derive cos/sin via rotary_emb (needs a value tensor).
        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, position_ids)
        del value_states

    _, key_states_rotated = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Stash on CPU.
    if self.layer_idx not in collected_keys:
        collected_keys[self.layer_idx] = []
    collected_keys[self.layer_idx].append(key_states_rotated.detach().cpu())

    del query_states, key_states, key_states_rotated

    if position_embeddings is not None:
        # LLaMA/Qwen call signature.
        return self.original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    else:
        # Mistral call signature.
        return self.original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )


def patch_model_for_collection(model):
    """Monkey-patch attention modules to collect key_states."""
    patched_count = 0
    for module in model.modules():
        if not isinstance(module, SUPPORTED_ATTENTION_CLASSES):
            continue
        
        module.original_forward = module.forward
        module.forward = partial(collect_key_states_forward, module)
        patched_count += 1
    
    print(f"Patched {patched_count} attention layers for key collection")
    return patched_count


def restore_model(model):
    """Restore the original forward methods on attention modules."""
    for module in model.modules():
        if not isinstance(module, SUPPORTED_ATTENTION_CLASSES):
            continue
        if hasattr(module, 'original_forward'):
            module.forward = module.original_forward
            delattr(module, 'original_forward')


class CalibrationDataset(Dataset):
    """Calibration dataset wrapper."""
    def __init__(self, data, tokenizer, max_length=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return tokens.input_ids[0]


def load_calibration_data(data_source, num_samples=100, tokenizer=None, max_length=4096):
    """Load calibration data.

    Sources: JSONL path, HuggingFace dataset name, or default wikitext.
    Short texts are concatenated to reach max_length.
    """
    texts = []

    if data_source and os.path.exists(data_source):
        # JSONL file.
        print(f"Loading calibration data from {data_source}")
        raw_texts = []
        print("Reading raw texts from file...")
        with open(data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total lines in file: {len(lines)}")
            for i, line in enumerate(tqdm(lines, desc="Parsing JSON")):
                try:
                    obj = json.loads(line)
                    if 'text' in obj:
                        raw_texts.append(obj['text'])
                    elif 'content' in obj:
                        raw_texts.append(obj['content'])
                    elif 'prompt' in obj:
                        raw_texts.append(obj['prompt'])
                except:
                    continue
        print(f"Parsed {len(raw_texts)} raw texts")
        
        # Concatenate short texts up to max_length.
        print(f"Concatenating texts to reach max_length={max_length}...")
        current_text = ""
        for i, text in enumerate(tqdm(raw_texts, desc="Processing texts")):
            if len(texts) >= num_samples:
                break
            text = text.strip()
            if not text:
                continue

            test_text = current_text + " " + text if current_text else text

            if tokenizer:
                tokens = tokenizer(test_text, truncation=True, max_length=max_length, return_tensors="pt")
                token_len = tokens.input_ids.shape[1]

                if token_len >= max_length:
                    if current_text:
                        texts.append(current_text)
                        print(f"  Sample {len(texts)}: {token_len} tokens")
                        current_text = text  # carry over for next sample
                    else:
                        # Single text already exceeds max_length.
                        texts.append(text)
                        print(f"  Sample {len(texts)}: {token_len} tokens (single text)")
                        current_text = ""
                else:
                    current_text = test_text
            else:
                # No tokenizer: estimate by character count.
                if len(test_text) >= max_length * 4:
                    if current_text:
                        texts.append(current_text)
                        current_text = text
                    else:
                        texts.append(text)
                        current_text = ""
                else:
                    current_text = test_text
        
        # Append the last (possibly short) buffered text.
        if current_text and len(texts) < num_samples:
            texts.append(current_text)
            print(f"  Sample {len(texts)}: final text added")
    else:
        # Default: wikitext.
        print("Loading calibration data from wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        # Drop empty entries and concatenate to reach max_length.
        current_text = ""
        for item in dataset:
            text = item['text'].strip()
            if not text:
                continue
            current_text += " " + text

            if tokenizer:
                tokens = tokenizer(current_text, truncation=True, max_length=max_length, return_tensors="pt")
                if tokens.input_ids.shape[1] >= max_length:
                    texts.append(current_text)
                    current_text = ""
                    if len(texts) >= num_samples:
                        break
            elif len(current_text) > max_length * 4:  # rough estimate
                texts.append(current_text)
                current_text = ""
                if len(texts) >= num_samples:
                    break

        if current_text and len(texts) < num_samples:
            texts.append(current_text)
    
    print(f"Loaded {len(texts)} calibration samples")
    return texts


def parse_length_config(length_config_str):
    """Parse a "length1:num1,length2:num2,..." string into a list of (length, num) tuples.

    Example: "131072:50,65536:10,32768:10,16384:10,8192:10,4096:10".
    """
    configs = []
    for item in length_config_str.split(','):
        length_str, num_str = item.strip().split(':')
        configs.append((int(length_str), int(num_str)))
    return configs


def load_texts_from_file(file_path):
    """Load a list of texts from a single JSONL file."""
    raw_texts = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return raw_texts
    
    print(f"Loading texts from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"  Total lines in file: {len(lines)}")
        for line in tqdm(lines, desc=f"Parsing {os.path.basename(file_path)}"):
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    raw_texts.append(obj['text'])
                elif 'content' in obj:
                    raw_texts.append(obj['content'])
                elif 'prompt' in obj:
                    raw_texts.append(obj['prompt'])
            except:
                continue
    print(f"  Parsed {len(raw_texts)} texts")
    return raw_texts


def prepare_samples_from_texts(raw_texts, length_configs, tokenizer, fallback_texts=None, source_name=""):
    """Build samples for the given length config from raw texts.

    Args:
        raw_texts: list of source texts.
        length_configs: list of (length, num_samples) tuples.
        tokenizer: HF tokenizer.
        fallback_texts: extra texts used if the source runs out.
        source_name: label for log messages.

    Returns:
        list of (text, target_length) tuples.
    """
    # Sort lengths descending so we concatenate up to the largest first.
    length_configs = sorted(length_configs, key=lambda x: x[0], reverse=True)
    max_length = length_configs[0][0]
    total_samples = sum(num for _, num in length_configs)
    
    print(f"\nPreparing samples for {source_name}:")
    for length, num in length_configs:
        print(f"  - Length {length}: {num} samples")
    print(f"  Total needed: {total_samples} samples")
    
    # Concatenate to reach max_length.
    print(f"  Concatenating texts to reach max_length={max_length}...")
    long_texts = []
    current_text = ""
    
    for text in tqdm(raw_texts, desc=f"Processing {source_name}"):
        if len(long_texts) >= total_samples:
            break
        text = text.strip()
        if not text:
            continue
        
        test_text = current_text + " " + text if current_text else text
        tokens = tokenizer(test_text, truncation=True, max_length=max_length, return_tensors="pt")
        token_len = tokens.input_ids.shape[1]
        
        if token_len >= max_length:
            if current_text:
                long_texts.append(current_text)
                current_text = text
            else:
                long_texts.append(text)
                current_text = ""
        else:
            current_text = test_text
    
    # Append the last (possibly short) buffered text.
    if current_text and len(long_texts) < total_samples:
        long_texts.append(current_text)
        print(f"  Added final text (may be shorter than max_length)")

    print(f"  Generated {len(long_texts)} long texts from {source_name}")

    # Top up from fallback texts if the primary source ran out.
    if len(long_texts) < total_samples and fallback_texts:
        needed = total_samples - len(long_texts)
        print(f"  Need {needed} more samples, using fallback data...")
        
        fallback_current_text = ""
        for text in tqdm(fallback_texts, desc="Processing fallback"):
            if len(long_texts) >= total_samples:
                break
            text = text.strip()
            if not text:
                continue
            
            test_text = fallback_current_text + " " + text if fallback_current_text else text
            tokens = tokenizer(test_text, truncation=True, max_length=max_length, return_tensors="pt")
            token_len = tokens.input_ids.shape[1]
            
            if token_len >= max_length:
                if fallback_current_text:
                    long_texts.append(fallback_current_text)
                    fallback_current_text = text
                else:
                    long_texts.append(text)
                    fallback_current_text = ""
            else:
                fallback_current_text = test_text
        
        if fallback_current_text and len(long_texts) < total_samples:
            long_texts.append(fallback_current_text)
        
        print(f"  After fallback: {len(long_texts)} long texts")
    
    # Assign texts to each length bucket.
    result = []
    text_idx = 0

    for target_length, num_samples in length_configs:
        for _ in range(num_samples):
            if text_idx >= len(long_texts):
                print(f"  Warning: Not enough texts for {source_name}, reusing from beginning")
                text_idx = 0
            result.append((long_texts[text_idx], target_length))
            text_idx += 1

    print(f"  Prepared {len(result)} samples for {source_name}")
    return result


def load_calibration_data_multi_source(data_sources, fallback_data, length_configs, tokenizer):
    """Load calibration data from multiple sources, applying length_configs to each."""
    print(f"\n{'='*50}")
    print("Loading calibration data from multiple sources")
    print(f"{'='*50}")
    print(f"Data sources: {data_sources}")
    print(f"Fallback data: {fallback_data}")
    print(f"Length config per source:")
    for length, num in length_configs:
        print(f"  - Length {length}: {num} samples")
    
    fallback_texts = None
    if fallback_data and os.path.exists(fallback_data):
        fallback_texts = load_texts_from_file(fallback_data)

    all_samples = []
    for source_path in data_sources:
        source_name = os.path.basename(source_path)
        raw_texts = load_texts_from_file(source_path)
        
        if not raw_texts:
            print(f"Warning: No texts loaded from {source_path}, using fallback only")
            raw_texts = []
        
        samples = prepare_samples_from_texts(
            raw_texts, 
            length_configs, 
            tokenizer, 
            fallback_texts=fallback_texts,
            source_name=source_name
        )
        all_samples.extend(samples)
    
    print(f"\n{'='*50}")
    print(f"Total samples prepared: {len(all_samples)}")
    print(f"{'='*50}")
    
    return all_samples


def load_calibration_data_multi_length(data_source, length_configs, tokenizer):
    """Single-source variant of load_calibration_data_multi_source (back-compat)."""
    length_configs = sorted(length_configs, key=lambda x: x[0], reverse=True)
    max_length = length_configs[0][0]
    total_samples = sum(num for _, num in length_configs)

    print(f"Loading calibration data with multi-length config:")
    for length, num in length_configs:
        print(f"  - Length {length}: {num} samples")
    print(f"  Total: {total_samples} samples")

    # Load enough long texts (sized by max_length).
    raw_texts = []
    if data_source and os.path.exists(data_source):
        print(f"Loading calibration data from {data_source}")
        print("Reading raw texts from file...")
        with open(data_source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total lines in file: {len(lines)}")
            for line in tqdm(lines, desc="Parsing JSON"):
                try:
                    obj = json.loads(line)
                    if 'text' in obj:
                        raw_texts.append(obj['text'])
                    elif 'content' in obj:
                        raw_texts.append(obj['content'])
                    elif 'prompt' in obj:
                        raw_texts.append(obj['prompt'])
                except:
                    continue
        print(f"Parsed {len(raw_texts)} raw texts")
    else:
        print("Loading calibration data from wikitext-2-raw-v1")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for item in dataset:
            text = item['text'].strip()
            if text:
                raw_texts.append(text)
        print(f"Loaded {len(raw_texts)} raw texts from wikitext")

    # Concatenate to reach max_length.
    print(f"Concatenating texts to reach max_length={max_length}...")
    long_texts = []
    current_text = ""
    
    for text in tqdm(raw_texts, desc="Processing texts"):
        if len(long_texts) >= total_samples:
            break
        text = text.strip()
        if not text:
            continue
        
        test_text = current_text + " " + text if current_text else text
        tokens = tokenizer(test_text, truncation=True, max_length=max_length, return_tensors="pt")
        token_len = tokens.input_ids.shape[1]
        
        if token_len >= max_length:
            if current_text:
                long_texts.append(current_text)
                current_text = text
            else:
                long_texts.append(text)
                current_text = ""
        else:
            current_text = test_text
    
    if current_text and len(long_texts) < total_samples:
        long_texts.append(current_text)
    
    print(f"Generated {len(long_texts)} long texts")

    # Assign texts to each length bucket.
    result = []
    text_idx = 0
    
    for target_length, num_samples in length_configs:
        print(f"Assigning {num_samples} samples for length {target_length}...")
        for _ in range(num_samples):
            if text_idx >= len(long_texts):
                print(f"Warning: Not enough texts, reusing from beginning")
                text_idx = 0
            result.append((long_texts[text_idx], target_length))
            text_idx += 1
    
    print(f"Total samples prepared: {len(result)}")
    return result


def print_gpu_memory(stage=""):
    """Log allocated/reserved GPU memory."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU Memory] {stage}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def forward_with_chunking(model, tokenizer, text, max_length, chunk_size=16384):
    """Chunked forward pass that collects key_states.

    Splits long sequences into chunks and threads them via KV cache to preserve
    context. Flash Attention memory scales with query length, so chunking caps
    peak memory.
    """
    global collected_keys

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids  # [1, seq_len]
    seq_len = input_ids.shape[1]

    if seq_len <= chunk_size:
        # Short sequence: single forward.
        inputs = inputs.to(model.device)
        outputs = model(**inputs, use_cache=False)
        del outputs, inputs
        torch.cuda.empty_cache()
        return

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    print(f"    Chunking: seq_len={seq_len}, chunk_size={chunk_size}, num_chunks={num_chunks}")

    # Per-layer collected-key count before this sample, so we can merge its chunks later.
    keys_count_before = {layer_idx: len(keys) for layer_idx, keys in collected_keys.items()}

    past_key_values = None

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, seq_len)

        chunk_input_ids = input_ids[:, start_idx:end_idx].to(model.device)

        # attention_mask must cover all previously cached tokens.
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            attention_mask = torch.ones(1, past_len + chunk_input_ids.shape[1], device=model.device)
        else:
            attention_mask = torch.ones_like(chunk_input_ids)

        # Forward triggers collect_key_states_forward.
        outputs = model(
            input_ids=chunk_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

        del outputs, chunk_input_ids, attention_mask
        torch.cuda.empty_cache()

    del past_key_values
    torch.cuda.empty_cache()

    # Merge per-sample chunks: [1, H, L1, D] + [1, H, L2, D] + ... -> [1, H, sum_L, D].
    for layer_idx in list(collected_keys.keys()):
        count_before = keys_count_before.get(layer_idx, 0)
        new_keys = collected_keys[layer_idx][count_before:]

        if len(new_keys) > 1:
            merged = torch.cat(new_keys, dim=2)
            collected_keys[layer_idx] = collected_keys[layer_idx][:count_before] + [merged]
            del new_keys


def main():
    parser = argparse.ArgumentParser(description="Calibrate projection matrices for offline sparse attention")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--calibration_data", type=str, default=None, help="Path to calibration data (JSONL) - single source mode")
    parser.add_argument("--data_sources", type=str, default=None, 
                        help="Comma-separated paths to multiple data sources (multi-source mode)")
    parser.add_argument("--fallback_data", type=str, default=None,
                        help="Path to fallback data source (used when primary sources are insufficient)")
    parser.add_argument("--num_samples_per_source", type=int, default=100,
                        help="Number of samples per data source (multi-source mode)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for projection matrices")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of calibration samples (single source mode)")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length (used when length_config is not provided)")
    parser.add_argument("--length_config", type=str, default=None, 
                        help="Length configuration: 'length1:num1,length2:num2,...' e.g., '131072:50,65536:10,32768:10'")
    parser.add_argument("--num_bits", type=int, default=64, help="Number of bits for quantization")
    parser.add_argument("--num_iters", type=int, default=4, help="Number of iterations for projection optimization")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Model dtype")
    parser.add_argument("--chunk_size", type=int, default=16384, help="Chunk size for forward pass (to avoid OOM)")
    args = parser.parse_args()
    
    print_gpu_memory("程序启动")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer from {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except Exception as e:
        print(f"Warning: Fast tokenizer failed ({e}), trying slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    print_gpu_memory("tokenizer 加载后")

    use_multi_source = args.data_sources is not None
    use_multi_length = args.length_config is not None

    if use_multi_source:
        data_sources = [s.strip() for s in args.data_sources.split(',')]

        if use_multi_length:
            length_configs = parse_length_config(args.length_config)
        else:
            # Single length default.
            length_configs = [(args.max_length, args.num_samples_per_source)]

        samples_with_length = load_calibration_data_multi_source(
            data_sources,
            args.fallback_data,
            length_configs,
            tokenizer
        )
    elif use_multi_length:
        # Single source + multi-length.
        length_configs = parse_length_config(args.length_config)
        samples_with_length = load_calibration_data_multi_length(
            args.calibration_data,
            length_configs,
            tokenizer
        )
    else:
        # Legacy single-source single-length mode.
        texts = load_calibration_data(
            args.calibration_data,
            num_samples=args.num_samples,
            tokenizer=tokenizer,
            max_length=args.max_length
        )
        samples_with_length = [(text, args.max_length) for text in texts]

    print_gpu_memory("calibration 数据加载后")

    # Use Flash Attention 2 for 128K-context support.
    print(f"Loading model from {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # supports 128K context
    )
    model.eval()

    print_gpu_memory("模型加载后")

    num_layers = patch_model_for_collection(model)

    print(f"Collecting key_states from {len(samples_with_length)} samples...")
    print(f"Using chunk_size={args.chunk_size} for forward pass")
    global collected_keys
    collected_keys = {}
    
    with torch.no_grad():
        for i, (text, target_length) in enumerate(tqdm(samples_with_length, desc="Processing samples")):
            try:
                forward_with_chunking(
                    model, tokenizer, text,
                    max_length=target_length,
                    chunk_size=args.chunk_size
                )
            except Exception as e:
                print(f"Warning: Error processing sample {i} (length={target_length}): {e}")
                import traceback
                traceback.print_exc()
                continue

            if (i + 1) % 10 == 0:
                print_gpu_memory(f"Sample {i+1} (length={target_length}) 完成后")

    restore_model(model)

    print(f"Computing projection matrices for {len(collected_keys)} layers...")
    projection_matrices = {}

    for layer_idx in tqdm(sorted(collected_keys.keys()), desc="Computing projections"):
        key_samples = collected_keys[layer_idx]
        print(f"  Layer {layer_idx}: {len(key_samples)} samples")

        quan_proj = find_binary_projection_from_samples(
            key_samples,
            num_bits=args.num_bits,
            num_iters=args.num_iters,
            device=model.device
        )

        projection_matrices[layer_idx] = quan_proj.cpu()

        # Free this layer's CPU buffer.
        collected_keys[layer_idx] = None
        torch.cuda.empty_cache()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    torch.save(projection_matrices, args.output_path)
    print(f"Saved projection matrices to {args.output_path}")
    print(f"Layers saved: {list(projection_matrices.keys())}")

    total_params = 0
    for layer_idx, proj in projection_matrices.items():
        params = proj.numel()
        total_params += params
        print(f"  Layer {layer_idx}: shape={list(proj.shape)}, params={params:,}")
    print(f"Total projection parameters: {total_params:,}")


if __name__ == "__main__":
    main()
