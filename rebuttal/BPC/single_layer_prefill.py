"""Compare online GPC/BinaryPC in the real first Llama attention layer.

Load only embeddings on CPU and one attention layer on GPU. Layer-normalized
embedding inputs exactly reproduce layer 0's inputs; other decoder layers, MLP,
LM head, and model loading are excluded. Time the production attention forward
including QKV, RoPE application, prepare, exact causal attention and O projection.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding,
)

from benchmarks.common import seed_everything
from benchmarks.Efficiency.run_efficiency import _kv_cache_bytes
from picokv.model_config import load_model_config
from picokv.picokv_standalone import StandaloneConfig
from picokv.picokv_standalone import transformers as adapter
from rebuttal.BPC.encoding import binarypc_encoding


def now():
    return datetime.now().astimezone().isoformat()


def free():
    gc.collect()
    torch.cuda.empty_cache()


def source_hashes():
    paths = [
        Path(__file__).relative_to(ROOT), Path('rebuttal/BPC/encoding.py'), Path('rebuttal/BPC/source_encoding.py'),
        Path('picokv/picokv_standalone/online.py'),
        Path('picokv/picokv_standalone/prepare.py'),
        Path('picokv/picokv_standalone/transformers.py'),
        Path('src/BPC/bpc/bpc/bpc_pytorch.py'),
    ]
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def load_inputs_and_layer(model_path, dataset, length):
    config = load_model_config(str(model_path))
    if config.model_type != 'llama':
        raise ValueError('Only the first Llama layer is supported')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    with open(dataset) as handle:
        text = ' '.join(json.loads(line)['text'] for line in handle if line.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < length:
        raise ValueError(f'Real text has only {len(ids)} tokens; need {length}')
    ids = torch.tensor(ids[:length], dtype=torch.long)
    del text
    weight_map = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']

    def weight(name):
        with safe_open(str(model_path / weight_map[name]), framework='pt', device='cpu') as handle:
            return handle.get_tensor(name)

    attention = LlamaAttention(config, layer_idx=0).to(device='cuda', dtype=torch.bfloat16).eval()
    for name, parameter in attention.named_parameters():
        parameter.copy_(weight('model.layers.0.self_attn.' + name).to('cuda'))
    container = torch.nn.Module()
    container.add_module('attention', attention)
    adapter.modify_llama_for_standalone(container)
    norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device='cuda', dtype=torch.bfloat16)
    norm.weight.copy_(weight('model.layers.0.input_layernorm.weight').to('cuda'))
    embeddings = weight('model.embed_tokens.weight')
    hidden = torch.empty(1, length, config.hidden_size, device='cuda', dtype=torch.bfloat16)
    # Keep the vocabulary embedding table on CPU; normalize in bounded chunks.
    for start in range(0, length, 4096):
        stop = min(start + 4096, length)
        chunk = embeddings.index_select(0, ids[start:stop]).to('cuda').unsqueeze(0)
        hidden[:, start:stop] = norm(chunk)
        del chunk
    del embeddings, norm
    rotary = LlamaRotaryEmbedding(config=config).to('cuda')
    positions = torch.arange(length, device='cuda').unsqueeze(0)
    position_embeddings = rotary(hidden, positions)
    del rotary, positions
    torch.cuda.synchronize()
    free()
    return attention, hidden, position_embeddings, {
        'token_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        'normalized_hidden_shape': list(hidden.shape),
        'query_heads': config.num_attention_heads,
        'kv_heads': config.num_key_value_heads,
        'head_dim': config.hidden_size // config.num_attention_heads,
        'layer_index': 0,
        'input': 'PG real-text prefix, exact layer-0 normalized embeddings',
        'model_max_position_embeddings': config.max_position_embeddings,
        'beyond_model_context': length > config.max_position_embeddings,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/home/zxh/zxh/models/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', default=str(ROOT / 'benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl'))
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU per independent length task')
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError('Use at least 1 warmup and 3 paired repetitions')
    path = Path(args.output)
    if path.exists():
        raise ValueError('Choose a fresh output path')
    path.parent.mkdir(parents=True, exist_ok=True)
    config = StandaloneConfig(key_bits=320, value_bits=320, eas=.01, use_pruning=False)
    result = {
        'status': 'running', 'started_at': now(), 'args': vars(args),
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': source_hashes(),
        'prepare_config': config.prepare_kwargs(),
        'effective_zerorope_by_encoding': {'gpc': config.zerorope, 'binarypc': 0},
        'coding_version': 'bpc320_source_global_rng_v3',
        'timing_scope': 'one complete production attention-module prefill forward: QKV projections, RoPE application, online KV prepare, exact causal FlashAttention, output projection',
        'excluded': 'model loading, tokenization, embeddings, input RMSNorm, RoPE cos/sin construction, MLP, other layers, LM head, cleanup, validation',
        'measurement': 'synchronized wall clock plus CUDA events; GPC/BPC alternate order on the same GPU; native per-layer CUDA events for prepare; separate events for exact attention',
        'warmup': {'gpc': [], 'binarypc': []},
        'samples': {'gpc': [], 'binarypc': []},
    }

    def save():
        path.write_text(json.dumps(result, indent=2) + '\n')

    save()
    exact = adapter._exact_prefill_attention
    flash_pair = None

    def timed_exact(*positional, **kwargs):
        flash_pair[0].record()
        output = exact(*positional, **kwargs)
        flash_pair[1].record()
        return output

    try:
        seed_everything(42)
        attention, hidden, positions, inputs_meta = load_inputs_and_layer(Path(args.model), args.dataset, args.length)
        if adapter._flash_attn_func is None:
            raise RuntimeError('FlashAttention is required to avoid quadratic materialization')
        result['inputs'] = inputs_meta
        result['baseline_allocated_gib'] = torch.cuda.memory_allocated() / 1024**3
        save()
        adapter._exact_prefill_attention = timed_exact
        output_reference = None

        def measure(encoding):
            nonlocal flash_pair, output_reference
            free()
            cache = adapter.StandaloneLlamaCache(config)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            flash_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            with binarypc_encoding() if encoding == 'binarypc' else nullcontext():
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                start.record()
                output, _ = attention(
                    hidden_states=hidden, position_embeddings=positions,
                    attention_mask=None, past_key_value=cache,
                )
                end.record()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - begin) * 1000
                peak = torch.cuda.max_memory_allocated()
            # All checks, cache accounting and cleanup are outside the timing.
            prepare_ms = cache.prepare_timings_ms()[0]
            if len(cache.layers) != 1 or cache.layers[0].key_bits != 320 or cache.layers[0].value_bits != 320:
                raise AssertionError('Expected one 320/320-bit packed layer')
            state = cache.layers[0]
            for tensor in (output, state.key_projection, state.value_projection):
                if not torch.isfinite(tensor).all().item():
                    raise AssertionError('Non-finite prefill output or projection')
            # Prefill attention remains exact; changing cache coding must not
            # change the current prompt's attention output at all.
            sampled = output[:, ::max(1, args.length // 257)].clone()
            if output_reference is None:
                output_reference = sampled
            else:
                torch.testing.assert_close(sampled, output_reference, rtol=0, atol=0)
            sample = {
                'prefill_wall_ms': wall_ms,
                'prefill_cuda_ms': start.elapsed_time(end),
                'prepare_cuda_ms': prepare_ms,
                'exact_attention_cuda_ms': flash_pair[0].elapsed_time(flash_pair[1]),
                'peak_allocated_gib': peak / 1024**3,
                'incremental_peak_gib': (peak - baseline) / 1024**3,
                'cache_gib': _kv_cache_bytes(cache) / 1024**3,
                'global_len': state.global_len, 'dense_len': state.dense_len,
                'finite_outputs_and_projections': True,
                'sampled_prefill_output_matches': True,
            }
            del output, cache, state, sampled
            free()
            return sample

        for phase, count in [('warmup', args.warmup), ('samples', args.repeats)]:
            for repeat in range(count):
                order = ('gpc', 'binarypc') if repeat % 2 == 0 else ('binarypc', 'gpc')
                for encoding in order:
                    sample = measure(encoding)
                    result[phase][encoding].append(sample)
                    save()
                    print(args.length, phase, repeat, encoding, json.dumps(sample), flush=True)
        result['summary'] = {}
        for encoding, samples in result['samples'].items():
            summary = {}
            for key in ('prefill_wall_ms', 'prefill_cuda_ms', 'prepare_cuda_ms', 'exact_attention_cuda_ms', 'cache_gib'):
                values = [row[key] for row in samples]
                summary[key + '_median'] = statistics.median(values)
                summary[key + '_min_max'] = [min(values), max(values)]
            for key in ('peak_allocated_gib', 'incremental_peak_gib'):
                summary[key + '_max'] = max(row[key] for row in samples)
            result['summary'][encoding] = summary
        result['source_unchanged'] = source_hashes() == result['source_sha256']
        assert result['source_unchanged']
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='out_of_memory' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed', error=repr(error))
        raise
    finally:
        adapter._exact_prefill_attention = exact
        result['finished_at'] = now()
        save()


if __name__ == '__main__':
    main()
