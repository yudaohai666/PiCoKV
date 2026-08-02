"""Sweep QuanPC K/V code dimensions on one captured real Llama layer.

The model is run once to capture the target layer query and raw full K/V.  Then
only that layer's KV is recompressed for each K/V code dimension combination,
which isolates CUDA attention speed from full-model repeated prefill cost.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantpc import quanpc
from quantpc.quanpc import KQuantCache, VQuantCache, QuantizedTensor, MixedCacheTensor
from quantpc.quanpc_cuda_extension.benchmark_llama_layer_prune_sweep import capture_layer, make_layer_case, parse_int_list, time_cuda_event, sync_tensor_device
from quantpc.quanpc_cuda_extension.benchmark_llama_pruned_real_qkv import run_full, run_tail
from quantpc.quanpc_cuda_extension.quanpc_cuda import load_quanpc_extension, pack_quanpc_key_transposed_torch, pack_quanpc_value_1bit_transposed_torch


def parse_pairs(value: str, bits: list[int]) -> list[tuple[int, int]]:
    if value == "all":
        return [(k, v) for k in bits for v in bits]
    pairs = []
    for part in value.split(','):
        if not part:
            continue
        k, v = part.split(':')
        pairs.append((int(k), int(v)))
    return pairs


def gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    return torch.gather(x, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, d)).contiguous()


def drop_block_tokens(block: QuantizedTensor, keep_indices: torch.Tensor) -> QuantizedTensor:
    length = block.codes.shape[-2]
    mask = torch.ones(block.codes.shape[:3], device=block.codes.device, dtype=torch.bool)
    mask.scatter_(-1, keep_indices, False)
    b, h, _, c = block.codes.shape
    codes = block.codes[mask].reshape(b, h, length - keep_indices.shape[-1], c).contiguous()
    err = block.err[mask].reshape(b, h, length - keep_indices.shape[-1]).contiguous()
    return QuantizedTensor(codes, block.proj, err, proj_t=block.proj_t)


def joint_eas_indices(k_block: QuantizedTensor, v_block: QuantizedTensor, eas: float) -> torch.Tensor:
    length = k_block.err.shape[-1]
    eas_size = max(1, int(length * eas))
    k_count = min(eas_size, length)
    k_indices = torch.topk(k_block.err, k=k_count, dim=-1).indices
    if k_count == length:
        return k_indices.sort(dim=-1).values
    v_count = min(eas_size, length - k_count)
    selected = torch.zeros_like(v_block.err, dtype=torch.bool).scatter_(-1, k_indices, True)
    v_score = v_block.err.masked_fill(selected, -torch.inf)
    v_indices = torch.topk(v_score, k=v_count, dim=-1).indices
    return torch.cat([k_indices, v_indices], dim=-1).sort(dim=-1).values


def precompute_k_entries(prefill_key: torch.Tensor, bits: list[int], sink_size: int, local_size: int, zerorope: int, layer_idx: int):
    entries = {}
    for bit in bits:
        cache = KQuantCache(bit, sink_size=sink_size, local_size=local_size, zerorope=zerorope)
        raw_glob = cache.init_prefill(prefill_key, layer_idx)
        entries[bit] = {
            "sink": cache.sink[layer_idx],
            "local": cache.locl[layer_idx],
            "raw_glob": raw_glob,
            "block": cache.glob[layer_idx],
        }
        torch.cuda.synchronize(prefill_key.device)
        torch.cuda.empty_cache()
        print(f"precomputed K bits={bit}", flush=True)
    return entries


def precompute_v_entries(prefill_value: torch.Tensor, bits: list[int], sink_size: int, local_size: int, layer_idx: int):
    entries = {}
    for bit in bits:
        cache = VQuantCache(bit, sink_size=sink_size, local_size=local_size)
        raw_glob = cache.init_prefill(prefill_value, layer_idx)
        entries[bit] = {
            "sink": cache.sink[layer_idx],
            "local": cache.locl[layer_idx],
            "raw_glob": raw_glob,
            "block": cache.glob[layer_idx],
        }
        torch.cuda.synchronize(prefill_value.device)
        torch.cuda.empty_cache()
        print(f"precomputed V bits={bit}", flush=True)
    return entries


def make_case_for_bits(layer_idx: int, query: torch.Tensor, decode_key: torch.Tensor, decode_value: torch.Tensor, k_entry, v_entry, eas: float, exp_bits: int):
    eas_indices = joint_eas_indices(k_entry["block"], v_entry["block"], eas)
    eas_key = gather_tokens(k_entry["raw_glob"], eas_indices)
    eas_value = gather_tokens(v_entry["raw_glob"], eas_indices)
    dense_key = torch.cat([k_entry["sink"], eas_key, k_entry["local"], decode_key], dim=-2).contiguous()
    dense_value = torch.cat([v_entry["sink"], eas_value, v_entry["local"], decode_value], dim=-2).contiguous()

    k_glob = drop_block_tokens(k_entry["block"], eas_indices)
    v_glob = drop_block_tokens(v_entry["block"], eas_indices)
    packed_key = pack_quanpc_key_transposed_torch(k_glob.codes).contiguous()
    packed_value = pack_quanpc_value_1bit_transposed_torch(v_glob.codes).contiguous()
    k_empty_codes = k_glob.codes.new_empty(k_glob.codes.shape[0], k_glob.codes.shape[1], 0, k_glob.codes.shape[3])
    v_empty_codes = v_glob.codes.new_empty(v_glob.codes.shape[0], v_glob.codes.shape[1], 0, v_glob.codes.shape[3])
    k_empty_err = k_glob.err.new_empty(k_glob.err.shape[0], k_glob.err.shape[1], 0)
    v_empty_err = v_glob.err.new_empty(v_glob.err.shape[0], v_glob.err.shape[1], 0)
    k_tensor = MixedCacheTensor(
        dense_key,
        QuantizedTensor(k_empty_codes, k_glob.proj, k_empty_err, packed_key, k_glob.proj_t),
        True,
        quanpc.CUDA_PROJECTED_QUERY_BITS,
        exp_bits,
    )
    v_tensor = MixedCacheTensor(
        dense_value,
        QuantizedTensor(v_empty_codes, v_glob.proj, v_empty_err, packed_value, v_glob.proj_t),
        True,
        quanpc.CUDA_PROJECTED_QUERY_BITS,
        exp_bits,
    )
    return make_layer_case(layer_idx, query, k_tensor, v_tensor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--bits", default="128,192,256,320,384,448,512")
    parser.add_argument("--pairs", default="all", help="all or comma-separated K:V pairs")
    parser.add_argument("--capture-k-bits", type=int, default=256)
    parser.add_argument("--capture-v-bits", type=int, default=256)
    parser.add_argument("--projected-query-bits", type=int, default=8)
    parser.add_argument("--exp-bits", type=int, default=4)
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=20)
    parser.add_argument("--sink-size", type=int, default=64)
    parser.add_argument("--local-size", type=int, default=1024)
    parser.add_argument("--prune-mode", type=int, default=3)
    parser.add_argument("--log-eps", type=float, default=-10.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-json", default="results/benchmarks/benchmark_llama_layer1_bits_sweep_100k_logeps_m10.json")
    args = parser.parse_args()

    bits = parse_int_list(args.bits)
    pairs = parse_pairs(args.pairs, bits)
    capture_args = argparse.Namespace(**vars(args))
    capture_args.k_bits = args.capture_k_bits
    capture_args.v_bits = args.capture_v_bits

    case0, full_key, full_value, actual_tokens = capture_layer(capture_args)
    device = case0.query.device
    prefill_key = full_key[:, :, :-1, :].contiguous()
    decode_key = full_key[:, :, -1:, :].contiguous()
    prefill_value = full_value[:, :, :-1, :].contiguous()
    decode_value = full_value[:, :, -1:, :].contiguous()
    query = case0.query.detach().contiguous()

    ext = load_quanpc_extension()
    if not hasattr(ext, "quanpc_decode_tail_v1_exp8_tiled_gqa4_atomic_pruned_fn"):
        raise RuntimeError("pruned CUDA extension entry point is missing")

    def sdpa_full():
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            full_key,
            full_value,
            enable_gqa=True,
        )

    sdpa_full()
    sync_tensor_device(query)
    sdpa_ms = time_cuda_event(sdpa_full, device, args.warmup, args.iters)

    k_entries = precompute_k_entries(prefill_key, bits, args.sink_size, args.local_size, args.zerorope, args.layer_idx)
    v_entries = precompute_v_entries(prefill_value, bits, args.sink_size, args.local_size, args.layer_idx)

    rows = []
    for k_bits, v_bits in pairs:
        print(f"building case k_bits={k_bits} v_bits={v_bits}", flush=True)
        case = make_case_for_bits(args.layer_idx, query, decode_key, decode_value, k_entries[k_bits], v_entries[v_bits], args.eas, args.exp_bits)

        def base_tail():
            return run_tail(ext, case, args.exp_bits, False, 0, 0.0)

        def base_full():
            return run_full(ext, case, args.exp_bits, False, 0, 0.0)

        def pruned_tail():
            return run_tail(ext, case, args.exp_bits, True, args.prune_mode, args.log_eps)

        def pruned_full():
            return run_full(ext, case, args.exp_bits, True, args.prune_mode, args.log_eps)

        ref_tail = base_tail().detach().clone()
        ref_full = base_full().detach().clone()
        out_tail = pruned_tail().detach().clone()
        out_full = pruned_full().detach().clone()
        sync_tensor_device(query)
        tail_diff = (out_tail.float() - ref_tail.float()).abs()
        full_diff = (out_full.float() - ref_full.float()).abs()
        base_tail_ms = time_cuda_event(base_tail, device, args.warmup, args.iters)
        base_full_ms = time_cuda_event(base_full, device, args.warmup, args.iters)
        pruned_tail_ms = time_cuda_event(pruned_tail, device, args.warmup, args.iters)
        pruned_full_ms = time_cuda_event(pruned_full, device, args.warmup, args.iters)
        row = {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "log_eps": args.log_eps,
            "global_len": case.global_len,
            "dense_len": case.dense_len,
            "sdpa_full_ms": sdpa_ms,
            "base_tail_ms": base_tail_ms,
            "base_full_ms": base_full_ms,
            "pruned_tail_ms": pruned_tail_ms,
            "pruned_full_ms": pruned_full_ms,
            "pruned_tail_speedup_vs_base": base_tail_ms / pruned_tail_ms,
            "pruned_full_speedup_vs_base": base_full_ms / pruned_full_ms,
            "pruned_tail_speedup_vs_sdpa": sdpa_ms / pruned_tail_ms,
            "pruned_full_speedup_vs_sdpa": sdpa_ms / pruned_full_ms,
            "tail_max_abs_vs_base": float(tail_diff.max().item()),
            "tail_mean_abs_vs_base": float(tail_diff.mean().item()),
            "full_max_abs_vs_base": float(full_diff.max().item()),
            "full_mean_abs_vs_base": float(full_diff.mean().item()),
        }
        rows.append(row)
        print(
            "result k=%d v=%d base_tail=%.6fms pruned_tail=%.6fms base_full=%.6fms pruned_full=%.6fms full_vs_sdpa=%.3fx max_abs=%.6g mean_abs=%.6g"
            % (
                k_bits,
                v_bits,
                base_tail_ms,
                pruned_tail_ms,
                base_full_ms,
                pruned_full_ms,
                row["pruned_full_speedup_vs_sdpa"],
                row["full_max_abs_vs_base"],
                row["full_mean_abs_vs_base"],
            ),
            flush=True,
        )
        del case, ref_tail, ref_full, out_tail, out_full
        torch.cuda.empty_cache()

    result = {
        "config": vars(args),
        "actual_input_tokens": actual_tokens,
        "shape": {
            "query": list(query.shape),
            "full_key": list(full_key.shape),
            "full_value": list(full_value.shape),
            "device": str(device),
        },
        "rows": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("shape", json.dumps(result["shape"], sort_keys=True), flush=True)
    print("wrote", str(out), flush=True)


if __name__ == "__main__":
    main()
