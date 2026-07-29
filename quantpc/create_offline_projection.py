"""Create fixed-width offline QuanPC projections.

The output file is:
    {"key": key_projs, "value": value_projs, "meta": meta}

where key_projs is [num_layers, num_kv_heads, k_bits, head_dim] and
value_projs is [num_layers, num_kv_heads, v_bits, head_dim].
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, apply_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

import quanpc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRA_CONFIG = ROOT / "config/llama3-1-8b-ins/extra_configs/llama3-1-8b-ins-quanpc.json"
DEFAULT_ENV_CONFIG = ROOT / "config/llama3-1-8b-ins/llama3-1-8b-ins-quanpc.json"
DEFAULT_DATASET_DIR = ROOT / "config/calibration"
DEFAULT_OUTPUT_DIR = ROOT / "config/projections"
DEFAULT_TMP_DIR = DEFAULT_OUTPUT_DIR / "offline_states_quanpc.tmp"
DEFAULT_MODEL_NAME = "/home/zxh/zxh/models/Llama-3.1-8B-Instruct"
DEFAULT_SINK_SIZE = 64
DEFAULT_LOCAL_SIZE = 1024
DEFAULT_K_BITS = 256
DEFAULT_V_BITS = 256
DEFAULT_ZEROROPE = 20
OFFLINE_PROJECTION_FORMAT = "quanpc_offline_projection_scaled_v2"
LEGACY_OFFLINE_PROJECTION_FORMAT = "quanpc_offline_projection_legacy_l2_normalized_v1"


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


current_states = None
target_collect_layers = ()
sink_size = DEFAULT_SINK_SIZE
local_size = DEFAULT_LOCAL_SIZE
kproj_cache = {}
vproj_cache = {}


class StopForwardAfterCollect(Exception):
    pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extra_config(path):
    if path is None or not os.path.exists(path):
        return {}
    return load_json(path)


def model_name_from_env_config(path):
    if path is None or not os.path.exists(path):
        return None
    conf = load_json(path)
    return conf.get("model", {}).get("model_name")


def load_jsonl_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "text" in obj:
                texts.append(obj["text"])
            elif "content" in obj:
                texts.append(obj["content"])
            elif "prompt" in obj:
                texts.append(obj["prompt"])
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


def save_layer_manifest(args, layer_idx, sample_idx, total_samples):
    manifest = {
        "format": "quanpc_offline_states_v1_layer_shards",
        "layer_idx": layer_idx,
        "sample_idx": sample_idx,
        "total_samples": total_samples,
        "shard_dir": get_layer_tmp_dir(args, layer_idx),
    }
    save_pt(manifest, os.path.join(get_layer_tmp_dir(args, layer_idx), "manifest.pt"))


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


def load_head_from_layer_shards(shard_paths, kind, h_idx):
    chunks = []
    for shard_path in tqdm(shard_paths, desc=f"load {kind} head {h_idx}", leave=False):
        sample = torch.load(shard_path, map_location="cpu")
        chunks.append(sample[kind][h_idx])
    return torch.cat(chunks, dim=0)


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_collect_cache():
    global current_states, target_collect_layers, kproj_cache, vproj_cache
    current_states = None
    target_collect_layers = ()
    kproj_cache = {}
    vproj_cache = {}
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
        query_states,
        key_states,
        value_states,
        is_causal=Lq != 1,
        enable_gqa=True,
    )


def collect_kv(layer_idx, key_states, value_states):
    global current_states

    if layer_idx not in target_collect_layers:
        return

    end = key_states.shape[2] - local_size
    if end <= sink_size:
        return

    current_states[layer_idx]["key"] = key_states[0, :, sink_size:end, :].detach().to("cpu", dtype=torch.bfloat16)
    current_states[layer_idx]["value"] = value_states[0, :, sink_size:end, :].detach().to("cpu", dtype=torch.bfloat16)

    if layer_idx == target_collect_layers[-1]:
        raise StopForwardAfterCollect()


def modified_mlp_forward(self, x, chunk_size=8192):
    if x.shape[-2] <= chunk_size:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    outputs = []
    for x_chunk in x.split(chunk_size, dim=-2):
        outputs.append(self.down_proj(self.act_fn(self.gate_proj(x_chunk)) * self.up_proj(x_chunk)))
    return torch.cat(outputs, dim=-2)


def modify_for_collect(model):
    for module in model.modules():
        if isinstance(module, LlamaMLP):
            module.forward = partial(modified_mlp_forward, module)

    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue

        def modified_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            B, H, L, D = key_states.shape
            query_states = query_states.reshape(B, H, -1, D)

            if L != 1:
                kproj_cache[self.layer_idx] = quanpc.find_orth_proj(query_states)
                output_proj = self.o_proj.weight.reshape(
                    self.o_proj.weight.shape[0],
                    H,
                    -1,
                    self.head_dim,
                )
                output_proj = output_proj.transpose(0, 1).reshape(H, -1, self.head_dim)
                vproj_cache[self.layer_idx] = quanpc.find_orth_proj(output_proj)

            query_states = query_states @ kproj_cache[self.layer_idx][1]
            key_states = key_states @ kproj_cache[self.layer_idx][0].transpose(-1, -2)
            value_states = value_states @ vproj_cache[self.layer_idx][0].transpose(-1, -2)

            collect_kv(self.layer_idx, key_states, value_states)

            query_states = query_states.reshape(B, -1, L, D)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output = eager_attention_forward(query_states, key_states, value_states)
            attn_output = attn_output.reshape(B, H, -1, D)
            attn_output = attn_output @ vproj_cache[self.layer_idx][1].transpose(-1, -2)
            attn_output = attn_output.reshape(B, -1, L, D)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        module.forward = partial(modified_forward, module)


def collect_layers_states(args, model, tokenizer, samples, layer_indices):
    global current_states, target_collect_layers, kproj_cache, vproj_cache

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
                kproj_cache = {}
                vproj_cache = {}

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
                    save_layer_manifest(args, layer_idx, sample_idx + 1, len(samples))
                    print("saved layer", layer_idx, "sample", sample_idx + 1, "seq_len", seq_len, flush=True)

                del outputs, model_inputs
                current_states = None
                clear_cuda()
    finally:
        target_collect_layers = ()


def collect_states(args, samples, layers):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=None if args.device_map == "none" else args.device_map,
        torch_dtype=torch.bfloat16,
    )
    if args.device_map == "none" and torch.cuda.is_available():
        model = model.to("cuda")
    modify_for_collect(model)
    model.eval()

    for layer_group in tqdm(list(chunk_layers(layers, args.collect_layers_per_forward)), desc="layer groups"):
        collect_layers_states(args, model, tokenizer, samples, layer_group)


def load_projection_state(args, num_layers, num_kv_heads, head_dim, k_bits, v_bits):
    progress_path = args.output_path + ".progress.pt"
    if (not args.no_resume) and os.path.exists(args.output_path):
        projections = torch.load(args.output_path, map_location="cpu")
        key = projections["key"]
        value = projections["value"]
        meta = projections.get("meta", legacy_projection_meta())
        if key.shape != (num_layers, num_kv_heads, k_bits, head_dim):
            raise RuntimeError(f"existing key projection has wrong shape: {tuple(key.shape)}")
        if value.shape != (num_layers, num_kv_heads, v_bits, head_dim):
            raise RuntimeError(f"existing value projection has wrong shape: {tuple(value.shape)}")
        if os.path.exists(progress_path):
            progress = torch.load(progress_path, map_location="cpu")
        else:
            progress = {
                "key_done": torch.ones(num_layers, num_kv_heads, dtype=torch.bool),
                "value_done": torch.ones(num_layers, num_kv_heads, dtype=torch.bool),
            }
        print("resume projection file", args.output_path, flush=True)
        print("projection meta", meta, flush=True)
        return {"key": key, "value": value, "meta": meta}, progress

    key = torch.zeros(num_layers, num_kv_heads, k_bits, head_dim, dtype=torch.bfloat16)
    value = torch.zeros(num_layers, num_kv_heads, v_bits, head_dim, dtype=torch.bfloat16)
    meta = projection_meta(args)
    progress = {
        "key_done": torch.zeros(num_layers, num_kv_heads, dtype=torch.bool),
        "value_done": torch.zeros(num_layers, num_kv_heads, dtype=torch.bool),
    }
    return {"key": key, "value": value, "meta": meta}, progress


def save_projection_state(args, projections, progress):
    save_pt({"key": projections["key"], "value": projections["value"], "meta": projections["meta"]}, args.output_path)
    save_pt(progress, args.output_path + ".progress.pt")


def projection_layer_done(progress, layer_idx, num_heads):
    return bool(
        progress["key_done"][layer_idx, :num_heads].all()
        and progress["value_done"][layer_idx, :num_heads].all()
    )


def apply_key_zerorope(A, zerorope):
    if zerorope <= 0:
        return
    half_head_dim = A.shape[-1] // 2
    if zerorope > half_head_dim:
        raise ValueError(f"zerorope ({zerorope}) must be <= half_head_dim ({half_head_dim})")
    A[:, :, :, :zerorope] = 0
    A[:, :, :, half_head_dim:(half_head_dim + zerorope)] = 0


def projection_meta(args):
    return {
        "format": OFFLINE_PROJECTION_FORMAT,
        "key_l2_normalize_rows": False,
        "value_l2_normalize_rows": False,
    }


def legacy_projection_meta():
    return {
        "format": LEGACY_OFFLINE_PROJECTION_FORMAT,
        "key_l2_normalize_rows": True,
        "value_l2_normalize_rows": True,
    }


def learn_head_projection(tokens, bits, device, zerorope=0, normalize_rows=False):
    cache = quanpc.QuantCache(bits=bits)
    A = tokens.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    A = A.clone()
    apply_key_zerorope(A, zerorope)
    with torch.no_grad():
        projection = cache.find_projection(A)
    out = projection.proj[0, 0].float()
    if normalize_rows:
        out = torch.nn.functional.normalize(out, dim=-1, p=2)
    out = out.detach().cpu().to(torch.bfloat16)
    del A, projection
    clear_cuda()
    return out


def project_layer(args, projections, progress, layer_idx, sample_count, device, k_bits, v_bits, zerorope):
    shard_paths = get_layer_shard_paths(args, layer_idx, sample_count)
    first_sample = torch.load(shard_paths[0], map_location="cpu")
    num_heads = first_sample["key"].shape[0]
    del first_sample
    meta = projections["meta"]
    key_l2_normalize_rows = bool(meta.get("key_l2_normalize_rows", False))
    value_l2_normalize_rows = bool(meta.get("value_l2_normalize_rows", False))

    for h_idx in tqdm(range(num_heads), desc=f"layer {layer_idx} heads"):
        if not bool(progress["key_done"][layer_idx, h_idx]):
            key_states = load_head_from_layer_shards(shard_paths, "key", h_idx)
            print("layer", layer_idx, "head", h_idx, "key_states", key_states.shape, flush=True)
            projections["key"][layer_idx, h_idx] = learn_head_projection(
                key_states,
                k_bits,
                device,
                zerorope=zerorope,
                normalize_rows=key_l2_normalize_rows,
            )
            progress["key_done"][layer_idx, h_idx] = True
            save_projection_state(args, projections, progress)
            del key_states
            clear_cuda()
        else:
            print("skip layer", layer_idx, "K", h_idx, flush=True)

        if not bool(progress["value_done"][layer_idx, h_idx]):
            value_states = load_head_from_layer_shards(shard_paths, "value", h_idx)
            print("layer", layer_idx, "head", h_idx, "value_states", value_states.shape, flush=True)
            projections["value"][layer_idx, h_idx] = learn_head_projection(
                value_states,
                v_bits,
                device,
                zerorope=0,
                normalize_rows=value_l2_normalize_rows,
            )
            progress["value_done"][layer_idx, h_idx] = True
            save_projection_state(args, projections, progress)
            del value_states
            clear_cuda()
        else:
            print("skip layer", layer_idx, "V", h_idx, flush=True)

    return num_heads


def make_projection(args, layers, num_layers, num_kv_heads, head_dim, k_bits, v_bits, zerorope):
    device = torch.device(args.projection_device if args.projection_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    projections, progress = load_projection_state(args, num_layers, num_kv_heads, head_dim, k_bits, v_bits)

    for layer_idx in tqdm(layers, desc="layers"):
        layer_dir = get_layer_tmp_dir(args, layer_idx)
        if not os.path.isdir(layer_dir):
            print("skip missing layer states", layer_idx, flush=True)
            continue
        sample_count = 0
        while os.path.exists(os.path.join(layer_dir, f"sample_{sample_count:06d}.pt")):
            sample_count += 1
        if sample_count == 0:
            continue
        project_layer(args, projections, progress, layer_idx, sample_count, device, k_bits, v_bits, zerorope)
        if projection_layer_done(progress, layer_idx, num_kv_heads):
            remove_layer_tmp(args, layer_idx)

    save_projection_state(args, projections, progress)
    print("saved projection to", args.output_path, flush=True)


def run_both(args, samples, layers, num_layers, num_kv_heads, head_dim, k_bits, v_bits, zerorope):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device(args.projection_device if args.projection_device else ("cuda" if torch.cuda.is_available() else "cpu"))
    projections, progress = load_projection_state(args, num_layers, num_kv_heads, head_dim, k_bits, v_bits)

    pending_layers = []
    for layer_idx in layers:
        if projection_layer_done(progress, layer_idx, num_kv_heads):
            print("skip finished layer", layer_idx, flush=True)
            remove_layer_tmp(args, layer_idx)
            continue
        pending_layers.append(layer_idx)

    for layer_group in tqdm(list(chunk_layers(pending_layers, args.collect_layers_per_forward)), desc="layer groups"):
        collect_layer_group = []
        for layer_idx in layer_group:
            done_samples = get_done_sample_count(args, layer_idx, len(samples))
            if done_samples < len(samples):
                collect_layer_group.append(layer_idx)
            else:
                print("layer", layer_idx, "states already collected", flush=True)

        if collect_layer_group:
            print("loading model for layers", format_layer_group(collect_layer_group), flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map=None if args.device_map == "none" else args.device_map,
                torch_dtype=torch.bfloat16,
            )
            if args.device_map == "none" and torch.cuda.is_available():
                model = model.to("cuda")
            modify_for_collect(model)
            model.eval()
            collect_layers_states(args, model, tokenizer, samples, collect_layer_group)
            del model
            clear_collect_cache()
            print("released model before projecting layers", format_layer_group(layer_group), flush=True)

        for layer_idx in layer_group:
            project_layer(args, projections, progress, layer_idx, len(samples), device, k_bits, v_bits, zerorope)
            if projection_layer_done(progress, layer_idx, num_kv_heads):
                remove_layer_tmp(args, layer_idx)

    save_projection_state(args, projections, progress)
    print("saved projection to", args.output_path, flush=True)


def infer_head_dim(config):
    return config.hidden_size // config.num_attention_heads


def default_output_path(k_bits, v_bits, zerorope):
    return DEFAULT_OUTPUT_DIR / f"k{k_bits}_v{v_bits}_zrope{zerorope}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="both", choices=["collect", "project", "both"])
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--env_config", type=str, default=None)
    parser.add_argument("--extra_config", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="codeparrot,pg19,proof-pile")
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--length_config", type=str, default="128000:20,64000:20,32000:20")
    parser.add_argument("--tmp_state_dir", type=str, default=str(DEFAULT_TMP_DIR))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--projection_device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "none"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--collect_layers_per_forward", type=int, default=16)
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--local_size", type=int, default=None)
    parser.add_argument("--k_bits", type=int, default=None)
    parser.add_argument("--v_bits", type=int, default=None)
    parser.add_argument("--zerorope", type=int, default=None)
    args = parser.parse_args()

    extra_config = load_extra_config(args.extra_config)
    args.model_name = args.model_name or model_name_from_env_config(args.env_config) or DEFAULT_MODEL_NAME
    if not args.model_name:
        raise RuntimeError("model_name is required, or env_config must contain model.model_name")

    global sink_size, local_size
    sink_size = int(args.sink_size if args.sink_size is not None else extra_config.get("sink_size", DEFAULT_SINK_SIZE))
    local_size = int(args.local_size if args.local_size is not None else extra_config.get("local_size", DEFAULT_LOCAL_SIZE))
    k_bits = int(args.k_bits if args.k_bits is not None else extra_config.get("k_bits", DEFAULT_K_BITS))
    v_bits = int(args.v_bits if args.v_bits is not None else extra_config.get("v_bits", DEFAULT_V_BITS))
    zerorope = int(args.zerorope if args.zerorope is not None else extra_config.get("zerorope", DEFAULT_ZEROROPE))
    if args.output_path is None:
        args.output_path = str(default_output_path(k_bits, v_bits, zerorope))

    config = AutoConfig.from_pretrained(args.model_name)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = infer_head_dim(config)
    layers = parse_layers(args.layers, num_layers)

    print("model", args.model_name, flush=True)
    print("layers", format_layer_group(layers), flush=True)
    print("projection shape key", (num_layers, num_kv_heads, k_bits, head_dim), flush=True)
    print("projection shape value", (num_layers, num_kv_heads, v_bits, head_dim), flush=True)
    print("output_path", args.output_path, flush=True)

    if args.stage == "collect":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        samples = build_samples(
            tokenizer,
            args.dataset_dir,
            [name.strip() for name in args.datasets.split(",") if name.strip()],
            parse_length_config(args.length_config),
            args.max_samples,
        )
        del tokenizer
        print("samples", len(samples), "length_config", args.length_config, flush=True)
        collect_states(args, samples, layers)
    elif args.stage == "project":
        make_projection(args, layers, num_layers, num_kv_heads, head_dim, k_bits, v_bits, zerorope)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        samples = build_samples(
            tokenizer,
            args.dataset_dir,
            [name.strip() for name in args.datasets.split(",") if name.strip()],
            parse_length_config(args.length_config),
            args.max_samples,
        )
        del tokenizer
        print("samples", len(samples), "length_config", args.length_config, flush=True)
        run_both(args, samples, layers, num_layers, num_kv_heads, head_dim, k_bits, v_bits, zerorope)


if __name__ == "__main__":
    main()
