"""Calibration-state collection for the offline block-32 pipeline.

Runs the model over calibration text and dumps per-layer post-RoPE K/V shards
under a temp directory.

This is a library, not an entry point: ``create_offline_projections.py`` drives
it and owns the fitting stages.  It used to double as a CLI that wrote
encoder-only projections in the ``picokv_standalone_block32_encoder_v1`` format,
which the encoder+decoder artifact superseded.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    apply_rotary_pos_emb,
)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATASET_DIR = BASE / "data"
DEFAULT_TMP_DIR = BASE / "offline_states.tmp"
DEFAULT_MODEL_NAME = "/data/Llama-3.1-8B-Instruct"
DEFAULT_SINK_SIZE = 64
DEFAULT_LOCAL_SIZE = 256
DEFAULT_ZEROROPE = 20
# block-32 learning only needs enough rows to resolve a 128-dim subspace, while
# the coding/residual passes scale with row count. Subsample to bound memory.
DEFAULT_BLOCK32_LEARN_TOKENS = 1048576

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

current_states = None
target_collect_layers = ()
sink_size = DEFAULT_SINK_SIZE
local_size = DEFAULT_LOCAL_SIZE
# Keep every Nth collected token.  Projection learning already subsamples rows
# evenly (``_subsample_rows``), so striding at collect time costs no fit quality
# while cutting shard bytes -- and therefore collect passes -- proportionally.
collect_stride = 1


class StopForwardAfterCollect(Exception):
    pass


def load_jsonl_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for field in ("text", "content", "prompt", "context"):
                if field in obj:
                    texts.append(obj[field])
                    break
    return texts


def parse_length_config(config):
    out = []
    for item in config.split(","):
        item = item.strip()
        if not item:
            continue
        length, num = item.split(":")
        out.append((int(length), int(num)))
    return out


def resolve_dataset_path(dataset_dir, name):
    path = Path(name)
    if path.is_absolute() or "/" in name:
        return path
    return Path(dataset_dir) / f"{name}.json"
def build_samples(tokenizer, dataset_dir, datasets, length_config, max_samples=None):
    samples = []
    length_config = sorted(length_config, key=lambda x: x[0], reverse=True)
    max_length = length_config[0][0]
    samples_per_dataset = sum(num for _, num in length_config)

    for name in datasets:
        if max_samples is not None and len(samples) >= max_samples:
            break

        path = resolve_dataset_path(dataset_dir, name)
        texts = load_jsonl_texts(path)
        long_texts = []
        now = ""
        need_this_dataset = samples_per_dataset
        if max_samples is not None:
            need_this_dataset = min(samples_per_dataset, max_samples - len(samples))

        print("loading", path, "texts", len(texts), flush=True)
        for text in tqdm(texts, desc=f"prepare {path.name}"):
            if len(long_texts) >= need_this_dataset:
                break
            text = text.strip()
            if not text:
                continue
            nxt = now + "\n" + text if now else text
            if len(tokenizer.encode(nxt, add_special_tokens=False)) >= max_length:
                long_texts.append(nxt)
                now = ""
            else:
                now = nxt

        if now and len(long_texts) < need_this_dataset:
            long_texts.append(now)
        if not long_texts:
            raise RuntimeError(f"no text loaded from {path}")

        idx = 0
        for length, num in length_config:
            for _ in range(num):
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
                samples.append((long_texts[idx % len(long_texts)], length))
                idx += 1

    return samples
def save_pt(obj, path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def get_layer_tmp_dir(args, layer_idx):
    return os.path.join(args.tmp_state_dir, f"layer_{layer_idx:02d}")


def get_layer_shard_path(args, layer_idx, sample_idx):
    return os.path.join(get_layer_tmp_dir(args, layer_idx), f"sample_{sample_idx:06d}.pt")


def get_done_sample_count(args, layer_idx, total_samples):
    if args.no_resume:
        return 0
    done = 0
    while done < total_samples and os.path.exists(get_layer_shard_path(args, layer_idx, done)):
        done += 1
    print("resume layer", layer_idx, "states from sample", done, flush=True)
    return done


def get_layer_shard_paths(args, layer_idx, total_samples):
    paths = []
    for sample_idx in range(total_samples):
        shard_path = get_layer_shard_path(args, layer_idx, sample_idx)
        if not os.path.exists(shard_path):
            raise RuntimeError(f"missing layer shard: {shard_path}")
        paths.append(shard_path)
    return paths


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_collect_cache():
    global current_states, target_collect_layers
    current_states = None
    target_collect_layers = ()
    clear_cuda()


def remove_layer_tmp(args, layer_idx):
    if args.keep_tmp:
        return
    layer_dir = get_layer_tmp_dir(args, layer_idx)
    if os.path.isdir(layer_dir):
        shutil.rmtree(layer_dir)
        print("removed tmp states for layer", layer_idx, flush=True)


def parse_layers(layers, num_layers):
    if layers is None:
        return list(range(num_layers))
    out = []
    for item in layers.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def chunk_layers(layers, chunk_size):
    chunk_size = max(1, chunk_size)
    for start_idx in range(0, len(layers), chunk_size):
        yield layers[start_idx:start_idx + chunk_size]


def format_layer_group(layer_indices):
    return ",".join(str(layer_idx) for layer_idx in layer_indices)


def eager_attention_forward(query_states, key_states, value_states):
    Lq = query_states.shape[-2]
    return torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=Lq != 1, enable_gqa=True,
    )


def collect_kv(layer_idx, key_states, value_states):
    global current_states
    if layer_idx not in target_collect_layers:
        return
    end = key_states.shape[2] - local_size
    if end <= sink_size:
        return
    stride = max(1, int(collect_stride))
    current_states[layer_idx]["key"] = key_states[0, :, sink_size:end:stride, :].detach().to("cpu", dtype=torch.bfloat16)
    current_states[layer_idx]["value"] = value_states[0, :, sink_size:end:stride, :].detach().to("cpu", dtype=torch.bfloat16)
    if layer_idx == target_collect_layers[-1]:
        raise StopForwardAfterCollect()


def set_collect_stride(stride):
    """Rows kept per token position.

    Block-32 learning already subsamples evenly (``_subsample_rows``), so
    striding here costs no fit quality while cutting shard bytes -- and the
    per-head shard re-reads during fitting -- proportionally.
    """
    global collect_stride
    collect_stride = max(1, int(stride))


def modified_mlp_forward(self, x, chunk_size=8192):
    if x.shape[-2] <= chunk_size:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    outputs = []
    for x_chunk in x.split(chunk_size, dim=-2):
        outputs.append(self.down_proj(self.act_fn(self.gate_proj(x_chunk)) * self.up_proj(x_chunk)))
    return torch.cat(outputs, dim=-2)
def modify_for_collect(model):
    """Patch attention/MLP to capture raw post-RoPE prompt K/V.

    block-32 quantizes in raw post-RoPE space (no query rotation), so
    calibration must capture K/V there.
    """
    if getattr(getattr(model, "config", None), "model_type", "llama") == "qwen3":
        modify_qwen3_for_collect(model)
        return

    for module in model.modules():
        if isinstance(module, LlamaMLP):
            module.forward = partial(modified_mlp_forward, module)

    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue

        def modified_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            collect_kv(self.layer_idx, key_states, value_states)
            attn_output = eager_attention_forward(query_states, key_states, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        module.forward = partial(modified_forward, module)


def modify_qwen3_for_collect(model):
    """Patch Qwen3 attention/MLP to capture post-RMSNorm, post-RoPE K/V."""
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3MLP,
        apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    )

    for module in model.modules():
        if isinstance(module, Qwen3MLP):
            module.forward = partial(modified_mlp_forward, module)

    for module in model.modules():
        if not isinstance(module, Qwen3Attention):
            continue

        def modified_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = qwen3_apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            collect_kv(self.layer_idx, key_states, value_states)
            attn_output = eager_attention_forward(query_states, key_states, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        module.forward = partial(modified_forward, module)


def collect_layers_states(args, model, tokenizer, samples, layer_indices):
    global current_states, target_collect_layers

    layer_indices = sorted(set(layer_indices))
    done_counts = {}
    for layer_idx in layer_indices:
        done_counts[layer_idx] = get_done_sample_count(args, layer_idx, len(samples))
        print("layer", layer_idx, "tmp state dir", get_layer_tmp_dir(args, layer_idx), flush=True)

    pending_layers = [layer_idx for layer_idx in layer_indices if done_counts[layer_idx] < len(samples)]
    if not pending_layers:
        print("layers", format_layer_group(layer_indices), "states already collected", flush=True)
        return

    start_sample = min(done_counts[layer_idx] for layer_idx in pending_layers)
    desc = f"layers {format_layer_group(layer_indices)} samples"

    try:
        with torch.no_grad():
            for sample_idx in tqdm(range(start_sample, len(samples)), desc=desc):
                active_layers = tuple(layer_idx for layer_idx in pending_layers if sample_idx >= done_counts[layer_idx])
                if not active_layers:
                    continue

                text, length = samples[sample_idx]
                current_states = {layer_idx: {"key": None, "value": None} for layer_idx in active_layers}
                target_collect_layers = active_layers

                model_inputs = tokenizer(text, truncation=True, max_length=length, return_tensors="pt").to(model.device)
                outputs = None
                try:
                    outputs = model.model(**model_inputs, use_cache=False)
                except StopForwardAfterCollect:
                    pass

                seq_len = model_inputs.input_ids.shape[1]
                for layer_idx in active_layers:
                    layer_states = current_states[layer_idx]
                    if layer_states["key"] is None or layer_states["value"] is None:
                        raise RuntimeError(f"failed to collect layer {layer_idx}, sample {sample_idx}")
                    layer_states["sample_idx"] = sample_idx
                    layer_states["layer_idx"] = layer_idx
                    layer_states["target_length"] = length
                    layer_states["seq_len"] = seq_len
                    save_pt(layer_states, get_layer_shard_path(args, layer_idx, sample_idx))
                    print("saved layer", layer_idx, "sample", sample_idx + 1, "seq_len", seq_len, flush=True)

                del outputs, model_inputs
                current_states = None
                clear_cuda()
    finally:
        target_collect_layers = ()










def apply_key_zerorope(A, zerorope):
    if zerorope <= 0:
        return
    half_head_dim = A.shape[-1] // 2
    if zerorope > half_head_dim:
        raise ValueError(f"zerorope ({zerorope}) must be <= half_head_dim ({half_head_dim})")
    A[:, :, :, :zerorope] = 0
    A[:, :, :, half_head_dim:(half_head_dim + zerorope)] = 0
def _subsample_rows(tokens, max_rows):
    """Evenly subsample concatenated calibration rows to bound peak memory."""
    if max_rows is None or tokens.shape[0] <= int(max_rows):
        return tokens
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive; got {max_rows}")
    n_rows = int(tokens.shape[0])
    if max_rows == 1:
        return tokens.index_select(0, torch.zeros(1, device=tokens.device, dtype=torch.int64))
    positions = torch.arange(max_rows, device=tokens.device, dtype=torch.int64)
    indices = (positions * (n_rows - 1) + (max_rows - 1) // 2) // (max_rows - 1)
    return tokens.index_select(0, indices)








