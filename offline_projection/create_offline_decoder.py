"""Create fixed offline QuanPC block-32 encoders *and* decoders.

``create_offline_projection.py`` stores only the orthonormal encoders and leaves
the decoder to be re-solved from every context.  This script stores both, so
prefill skips the per-group least-squares solve as well.

The output file is:
    {"key": key_enc, "value": value_enc,
     "key_decoder": key_dec, "value_decoder": value_dec, "meta": meta}

with encoders and decoders both ``[num_layers, num_kv_heads, bits, head_dim]``.
The format tag is ``quanpc_standalone_block32_encoder_decoder_v1``; the file also
carries the plain encoder tag's tensors, so it loads anywhere an encoder-only
file does.

Why the decoder needs its own fitting procedure
-----------------------------------------------
Codes come from a residual chain: group ``g`` quantizes what groups ``0..g-1``
left behind, and what they leave behind depends on the decoders used to subtract
them.  Fitting each group greedily against the residual seen during calibration
therefore produces decoders that are never used on that residual at inference --
the chain there is driven by the fixed decoders instead, and the mismatch
compounds group by group.

So the fit alternates to a fixed point:
  1. run the chain with the current decoders to get every group's codes,
  2. solve one joint least squares for all bits at once against the original
     states (``min_D ||X - C D||_F``, ridge-regularized),
  3. repeat.

Step 2 targets the original states rather than per-group residuals, which is the
objective the reconstruction is actually judged by.

Stages (``--stage``):
  * collect -> reuse ``create_offline_projection``'s collector to dump per-layer
               post-RoPE K/V shards.
  * fit     -> learn per-head encoders, then fit decoders to the fixed point.
  * both    -> collect then fit, layer group by layer group (default).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from create_offline_projection import (  # noqa: E402
    DEFAULT_BLOCK32_LEARN_TOKENS,
    DEFAULT_DATASET_DIR,
    DEFAULT_LOCAL_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_SINK_SIZE,
    DEFAULT_TMP_DIR,
    DEFAULT_ZEROROPE,
    _subsample_rows,
    apply_key_zerorope,
    build_samples,
    chunk_layers,
    clear_cuda,
    collect_states,
    get_layer_shard_paths,
    get_layer_tmp_dir,
    load_head_from_layer_shards,
    parse_layers,
    parse_length_config,
    remove_layer_tmp,
    save_pt,
)
from quantpc.quanpc_standalone.constants import (  # noqa: E402
    ONLINE_DECODER_RIDGE,
    ONLINE_GROUP_BITS,
)
from quantpc.quanpc_standalone.online import (  # noqa: E402
    _strict_binary_sign_,
    quantize_block32_online,
)

BLOCK32_DECODER_FORMAT = "quanpc_standalone_block32_encoder_decoder_v1"
DEFAULT_DECODER_ITERS = 5
DEFAULT_K_BITS = 256
DEFAULT_V_BITS = 256


def chain_codes(states, encoders, decoders):
    """Replay the production residual chain with fixed encoders and decoders.

    ``states`` is ``[rows, length, dim]``; the returned codes are ``[rows,
    length, bits]`` in the same group order the cache stores them.
    """
    residual = states.clone()
    collected = []
    for start in range(0, encoders.shape[-2], ONLINE_GROUP_BITS):
        end = start + ONLINE_GROUP_BITS
        codes = torch.matmul(residual, encoders[:, start:end].transpose(-1, -2))
        _strict_binary_sign_(codes)
        collected.append(codes)
        residual -= torch.matmul(codes, decoders[:, start:end])
    return torch.cat(collected, dim=-1)


def joint_solve(codes, states):
    """Ridge-regularized ``argmin_D ||states - codes @ D||_F``."""
    length = codes.shape[-2]
    gram = torch.matmul(codes.transpose(-1, -2).float(), codes.float()) / length
    rhs = torch.matmul(codes.transpose(-1, -2).float(), states.float()) / length
    gram.diagonal(dim1=-2, dim2=-1).add_(ONLINE_DECODER_RIDGE)
    solution, _ = torch.linalg.solve_ex(gram, rhs, check_errors=False)
    return solution.to(states.dtype)


def reconstruction_error(states, encoders, decoders):
    """Relative Frobenius error of the fixed-decoder chain on ``states``."""
    codes = chain_codes(states, encoders, decoders)
    approx = torch.matmul(codes, decoders)
    return float((approx.float() - states.float()).norm() / states.float().norm())


def learn_head_encoder_decoder(
    tokens, bits, device, zerorope=0,
    block32_learn_tokens=DEFAULT_BLOCK32_LEARN_TOKENS,
    iters=DEFAULT_DECODER_ITERS,
    verbose=False,
):
    """Learn one head's encoders, then fit its decoders to the fixed point."""
    tokens = _subsample_rows(tokens, block32_learn_tokens)
    states = tokens.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.bfloat16).clone()
    apply_key_zerorope(states, zerorope)

    with torch.no_grad():
        block = quantize_block32_online(states, bits)
        encoders = block.encoders[0, 0].unsqueeze(0).contiguous()
        decoders = block.decoder[0, 0].unsqueeze(0).contiguous()
        del block

        flat = states.reshape(1, states.shape[2], states.shape[3]).contiguous()
        history = [reconstruction_error(flat, encoders, decoders)]
        for _ in range(max(0, int(iters))):
            decoders = joint_solve(chain_codes(flat, encoders, decoders), flat)
            history.append(reconstruction_error(flat, encoders, decoders))

        encoder_out = encoders[0].float().detach().cpu().to(torch.bfloat16)
        decoder_out = decoders[0].float().detach().cpu().to(torch.bfloat16)

    if verbose:
        print("    calib fit: " + " -> ".join(f"{h:.4f}" for h in history), flush=True)
    del states, flat, encoders, decoders
    clear_cuda()
    return encoder_out, decoder_out


def decoder_meta(k_bits, v_bits, zerorope, iters):
    return {
        "format": BLOCK32_DECODER_FORMAT,
        "group_bits": ONLINE_GROUP_BITS,
        "ridge": ONLINE_DECODER_RIDGE,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "zerorope": zerorope,
        "decoder_iters": iters,
    }


def load_state(args, num_layers, num_kv_heads, head_dim, k_bits, v_bits, zerorope):
    """Resume an interrupted fit, or allocate empty stacks."""
    shapes = {
        "key": (num_layers, num_kv_heads, k_bits, head_dim),
        "value": (num_layers, num_kv_heads, v_bits, head_dim),
    }
    empty = {
        "key": torch.zeros(shapes["key"], dtype=torch.bfloat16),
        "value": torch.zeros(shapes["value"], dtype=torch.bfloat16),
        "key_decoder": torch.zeros(shapes["key"], dtype=torch.bfloat16),
        "value_decoder": torch.zeros(shapes["value"], dtype=torch.bfloat16),
        "meta": decoder_meta(k_bits, v_bits, zerorope, args.decoder_iters),
    }
    progress = {
        "key_done": torch.zeros(num_layers, num_kv_heads, dtype=torch.bool),
        "value_done": torch.zeros(num_layers, num_kv_heads, dtype=torch.bool),
    }
    if args.no_resume or not os.path.exists(args.output_path):
        return empty, progress

    stored = torch.load(args.output_path, map_location="cpu")
    for field in ("key", "value", "key_decoder", "value_decoder"):
        if field not in stored or tuple(stored[field].shape) != shapes[
            field.replace("_decoder", "")
        ]:
            print(f"existing {args.output_path} incompatible; starting over", flush=True)
            return empty, progress
        empty[field] = stored[field]
    saved_progress = stored.get("progress")
    if saved_progress is not None:
        progress["key_done"] = saved_progress["key_done"]
        progress["value_done"] = saved_progress["value_done"]
    print(
        f"resuming: {int(progress['key_done'].sum())} K heads, "
        f"{int(progress['value_done'].sum())} V heads already fitted",
        flush=True,
    )
    return empty, progress


def save_state(args, state, progress):
    save_pt(
        {
            "key": state["key"],
            "value": state["value"],
            "key_decoder": state["key_decoder"],
            "value_decoder": state["value_decoder"],
            "meta": state["meta"],
            "progress": progress,
        },
        args.output_path,
    )


def fit_layer(args, state, progress, layer_idx, sample_count, device,
              k_bits, v_bits, zerorope):
    shard_paths = get_layer_shard_paths(args, layer_idx, sample_count)
    first = torch.load(shard_paths[0], map_location="cpu")
    num_heads = first["key"].shape[0]
    del first

    for head_idx in tqdm(range(num_heads), desc=f"layer {layer_idx} heads"):
        for kind, bits, rope, done_key, enc_field, dec_field in (
            ("key", k_bits, zerorope, "key_done", "key", "key_decoder"),
            ("value", v_bits, 0, "value_done", "value", "value_decoder"),
        ):
            if bool(progress[done_key][layer_idx, head_idx]):
                print(f"skip layer {layer_idx} {kind} {head_idx}", flush=True)
                continue
            states = load_head_from_layer_shards(shard_paths, kind, head_idx)
            print(f"layer {layer_idx} head {head_idx} {kind} {tuple(states.shape)}",
                  flush=True)
            encoder, decoder = learn_head_encoder_decoder(
                states, bits, device, zerorope=rope,
                block32_learn_tokens=args.block32_learn_tokens,
                iters=args.decoder_iters, verbose=args.verbose_fit,
            )
            state[enc_field][layer_idx, head_idx] = encoder
            state[dec_field][layer_idx, head_idx] = decoder
            progress[done_key][layer_idx, head_idx] = True
            save_state(args, state, progress)
            del states
            clear_cuda()

    return num_heads


def make_offline_decoder(args, layers, num_layers, num_kv_heads, head_dim,
                         k_bits, v_bits, zerorope):
    device = torch.device(
        args.projection_device
        if args.projection_device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state, progress = load_state(
        args, num_layers, num_kv_heads, head_dim, k_bits, v_bits, zerorope
    )

    for layer_idx in tqdm(layers, desc="layers"):
        layer_dir = get_layer_tmp_dir(args, layer_idx)
        if not os.path.isdir(layer_dir):
            print(f"skip missing layer states {layer_idx}", flush=True)
            continue
        sample_count = 0
        while os.path.exists(os.path.join(layer_dir, f"sample_{sample_count:06d}.pt")):
            sample_count += 1
        if sample_count == 0:
            continue
        fit_layer(args, state, progress, layer_idx, sample_count, device,
                  k_bits, v_bits, zerorope)
        done = (
            bool(progress["key_done"][layer_idx].all())
            and bool(progress["value_done"][layer_idx].all())
        )
        if done and not args.keep_tmp:
            remove_layer_tmp(args, layer_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="both", choices=["collect", "fit", "both"])
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--datasets", default="codeparrot,pg19,proof-pile")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--length_config", default="32768:4")
    parser.add_argument("--tmp_state_dir", default=str(DEFAULT_TMP_DIR))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--projection_device", default=None)
    parser.add_argument("--device_map", default="auto", choices=["auto", "none"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--collect_layers_per_forward", type=int, default=16)
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--sink_size", type=int, default=DEFAULT_SINK_SIZE)
    parser.add_argument("--local_size", type=int, default=DEFAULT_LOCAL_SIZE)
    parser.add_argument("--k_bits", type=int, default=DEFAULT_K_BITS)
    parser.add_argument("--v_bits", type=int, default=DEFAULT_V_BITS)
    parser.add_argument("--zerorope", type=int, default=DEFAULT_ZEROROPE)
    parser.add_argument("--block32-learn-tokens", type=int,
                        default=DEFAULT_BLOCK32_LEARN_TOKENS)
    parser.add_argument("--decoder-iters", type=int, default=DEFAULT_DECODER_ITERS,
                        help="alternating chain/solve rounds (0 = greedy decoder)")
    parser.add_argument("--collect-token-stride", type=int, default=1,
                        help="keep every Nth token when dumping states (bounds shard size)")
    parser.add_argument("--verbose-fit", action="store_true",
                        help="print the calibration error after each round")
    args = parser.parse_args()

    import create_offline_projection as base

    base.sink_size = args.sink_size
    base.local_size = args.local_size
    base.collect_stride = args.collect_token_stride

    config = AutoConfig.from_pretrained(args.model_name)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    layers = parse_layers(args.layers, num_layers)

    if args.output_path is None:
        tag = Path(args.model_name).name
        args.output_path = str(
            BASE / "projections" /
            f"{tag}_k{args.k_bits}_v{args.v_bits}_block32_decoder.pt"
        )
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"output: {args.output_path}", flush=True)

    if args.stage == "fit":
        make_offline_decoder(args, layers, num_layers, num_kv_heads, head_dim,
                             args.k_bits, args.v_bits, args.zerorope)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    samples = build_samples(
        tokenizer, args.dataset_dir, args.datasets.split(","),
        parse_length_config(args.length_config), args.max_samples,
    )
    print(f"{len(samples)} calibration samples", flush=True)

    if args.stage == "collect":
        collect_states(args, samples, layers)
        return

    for group in chunk_layers(layers, args.collect_layers_per_forward):
        collect_states(args, samples, group)
        make_offline_decoder(args, group, num_layers, num_kv_heads, head_dim,
                             args.k_bits, args.v_bits, args.zerorope)


if __name__ == "__main__":
    main()
