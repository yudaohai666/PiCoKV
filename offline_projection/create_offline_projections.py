"""Build the complete offline encoder/decoder pair in one pass.

The collected K/V states stay in the model's own basis. The fit stage runs the
canonical block-32 projection learning once per side and stores both encoders
and decoders, so evaluation can use the pair fully offline.

Stages (``--stage``):
  * collect -> dump per-layer post-RoPE K/V shards.
  * fit     -> run one canonical online block-32 pass per layer and store the
               complete encoder/decoder pair.
  * all     -> collect then fit, layer group by layer group (default).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from collect import (  # noqa: E402
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
    clear_collect_cache,
    collect_layers_states,
    format_layer_group,
    get_done_sample_count,
    get_layer_shard_paths,
    get_layer_tmp_dir,
    modify_for_collect,
    parse_layers,
    parse_length_config,
    remove_layer_tmp,
    save_pt,
    set_collect_stride,
)
from quantpc.quanpc_standalone.constants import (  # noqa: E402
    ONLINE_DECODER_RIDGE,
    ONLINE_GROUP_BITS,
)
from quantpc.model_config import load_model_config  # noqa: E402
from quantpc.quanpc_standalone.online import quantize_block32_online  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

FOLDED_FORMAT = "quanpc_standalone_block32_folded_v1"
DEFAULT_K_BITS = 320
DEFAULT_V_BITS = 320
def load_layer_states(shard_paths):
    """Read every shard once and return ``{kind: [heads, rows, dim]}``.

    Reading a head at a time would re-read the whole layer for each head, so
    fitting one layer costs sixteen full passes over ~5GB of shards.  Reading
    once and keeping the layer in host memory turns that into one pass; a layer
    is a few GB and the fit machine has hundreds.
    """
    keys, values = [], []
    for shard_path in tqdm(shard_paths, desc="load layer", leave=False):
        sample = torch.load(shard_path, map_location="cpu")
        keys.append(sample["key"])
        values.append(sample["value"])
        del sample
    return {
        "key": torch.cat(keys, dim=1),
        "value": torch.cat(values, dim=1),
    }


def learn_pair(states, bits):
    """Run the canonical online block-32 forward once.

    ``quantize_block32_online`` already learns each encoder, solves its
    ridge-regularized decoder, subtracts that reconstruction, and advances to
    the next group.  Re-solving from the concatenated codes here would be a
    different algorithm from the production online path, so offline fitting
    deliberately uses the same single pass.
    """
    with torch.no_grad():
        block = quantize_block32_online(states, bits)
        encoders = block.encoders[0, 0].unsqueeze(0).contiguous()
        decoders = block.decoder[0, 0].unsqueeze(0).contiguous()
        del block
    return encoders[0], decoders[0]


def folded_meta(args, k_bits, v_bits, zerorope):
    return {
        "format": FOLDED_FORMAT,
        "group_bits": ONLINE_GROUP_BITS,
        "ridge": ONLINE_DECODER_RIDGE,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "zerorope": zerorope,
        "sink_size": args.sink_size,
        "local_size": args.local_size,
        "calibration": {
            "datasets": args.datasets,
            "length_config": args.length_config,
            "collect_stride": args.collect_stride,
            "learn_tokens": args.block32_learn_tokens,
        },
    }


def load_state(args, num_layers, num_kv_heads, head_dim, k_bits, v_bits,
               zerorope):
    shapes = {
        "key_encoder": (num_layers, num_kv_heads, k_bits, head_dim),
        "value_encoder": (num_layers, num_kv_heads, v_bits, head_dim),
    }
    empty = {
        "key_encoder": torch.zeros(shapes["key_encoder"], dtype=torch.bfloat16),
        "value_encoder": torch.zeros(shapes["value_encoder"], dtype=torch.bfloat16),
        "key_decoder": torch.zeros(shapes["key_encoder"], dtype=torch.bfloat16),
        "value_decoder": torch.zeros(shapes["value_encoder"], dtype=torch.bfloat16),
        "meta": folded_meta(args, k_bits, v_bits, zerorope),
    }
    progress = {
        "key_done": torch.zeros(num_layers, num_kv_heads, dtype=torch.bool),
        "value_done": torch.zeros(num_layers, num_kv_heads, dtype=torch.bool),
    }
    if args.no_resume or not os.path.exists(args.output_path):
        return empty, progress

    stored = torch.load(args.output_path, map_location="cpu")
    for field in ("key_encoder", "value_encoder", "key_decoder", "value_decoder"):
        expected = shapes[field.replace("_decoder", "_encoder")]
        if field not in stored or tuple(stored[field].shape) != expected:
            print(f"existing {args.output_path} incompatible; starting over", flush=True)
            return empty, progress
        empty[field] = stored[field]
    saved = stored.get("progress")
    if saved is not None:
        progress["key_done"] = saved["key_done"]
        progress["value_done"] = saved["value_done"]
    print(
        f"resuming: {int(progress['key_done'].sum())} K heads, "
        f"{int(progress['value_done'].sum())} V heads already fitted",
        flush=True,
    )
    return empty, progress


def save_state(args, state, progress):
    save_pt(
        {
            "key_encoder": state["key_encoder"],
            "value_encoder": state["value_encoder"],
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
    layer_states = load_layer_states(shard_paths)
    num_heads = layer_states["key"].shape[0]
    print(f"layer {layer_idx} fitting offline projections", flush=True)
    for head_idx in tqdm(range(num_heads), desc=f"layer {layer_idx} heads"):
        for kind, bits, rope, done_key, enc_field, dec_field in (
            ("key", k_bits, zerorope, "key_done", "key_encoder", "key_decoder"),
            ("value", v_bits, 0, "value_done", "value_encoder", "value_decoder"),
        ):
            if bool(progress[done_key][layer_idx, head_idx]):
                continue
            tokens = _subsample_rows(
                layer_states[kind][head_idx], args.block32_learn_tokens
            )
            states = tokens.unsqueeze(0).unsqueeze(0).to(
                device=device, dtype=torch.bfloat16
            ).clone()
            del tokens
            apply_key_zerorope(states, rope)
            encoder, decoder = learn_pair(states, bits)
            state[enc_field][layer_idx, head_idx] = (
                encoder.float().detach().cpu().to(torch.bfloat16)
            )
            state[dec_field][layer_idx, head_idx] = (
                decoder.float().detach().cpu().to(torch.bfloat16)
            )
            progress[done_key][layer_idx, head_idx] = True
            del states, encoder, decoder
            clear_cuda()
        save_state(args, state, progress)

    del layer_states
    clear_cuda()
    return num_heads


def run(args, samples, layers, num_layers, num_kv_heads, head_dim,
        k_bits, v_bits, zerorope):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device(
        args.projection_device
        if args.projection_device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state, progress = load_state(
        args, num_layers, num_kv_heads, head_dim, k_bits, v_bits,
        zerorope,
    )

    pending = []
    for layer_idx in layers:
        done = (
            bool(progress["key_done"][layer_idx, :num_kv_heads].all())
            and bool(progress["value_done"][layer_idx, :num_kv_heads].all())
        )
        if done:
            print("skip finished layer", layer_idx, flush=True)
            remove_layer_tmp(args, layer_idx)
            continue
        pending.append(layer_idx)

    for layer_group in tqdm(
        list(chunk_layers(pending, args.collect_layers_per_forward)),
        desc="layer groups",
    ):
        needs_collect = [
            layer_idx for layer_idx in layer_group
            if get_done_sample_count(args, layer_idx, len(samples)) < len(samples)
        ]
        if needs_collect and args.stage != "fit":
            print("loading model for layers",
                  format_layer_group(needs_collect), flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=load_model_config(args.model_name),
                device_map=None if args.device_map == "none" else args.device_map,
                torch_dtype=torch.bfloat16,
            )
            if args.device_map == "none" and torch.cuda.is_available():
                model = model.to("cuda")
            modify_for_collect(model)
            model.eval()
            collect_layers_states(args, model, tokenizer, samples, needs_collect)
            del model
            clear_collect_cache()

        if args.stage == "collect":
            continue

        for layer_idx in layer_group:
            fit_layer(
                args, state, progress, layer_idx, len(samples), device,
                k_bits, v_bits, zerorope,
            )
            done = (
                bool(progress["key_done"][layer_idx].all())
                and bool(progress["value_done"][layer_idx].all())
            )
            if done:
                remove_layer_tmp(args, layer_idx)

    save_state(args, state, progress)
    print("saved", args.output_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["collect", "fit", "all"])
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--datasets", default="codeparrot,pg19,proof-pile")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--length_config", default="131072:10,98304:10,65536:10,32768:10")
    parser.add_argument("--tmp_state_dir", default=str(DEFAULT_TMP_DIR))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--projection_device", default=None)
    parser.add_argument("--device_map", default="auto", choices=["auto", "none"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--collect_layers_per_forward", type=int, default=16)
    parser.add_argument("--collect_stride", type=int, default=8,
                        help="keep every Nth collected token; block-32 learning "
                             "subsamples anyway, so this only cuts shard bytes")
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--sink_size", type=int, default=DEFAULT_SINK_SIZE)
    parser.add_argument("--local_size", type=int, default=DEFAULT_LOCAL_SIZE)
    parser.add_argument("--k_bits", type=int, default=DEFAULT_K_BITS)
    parser.add_argument("--v_bits", type=int, default=DEFAULT_V_BITS)
    parser.add_argument("--zerorope", type=int, default=DEFAULT_ZEROROPE)
    parser.add_argument("--block32-learn-tokens", dest="block32_learn_tokens",
                        type=int, default=DEFAULT_BLOCK32_LEARN_TOKENS)
    parser.add_argument("--decoder_ridge", type=float, default=1e-3,
                        help="block-32 decoder ridge for this fit (default 1e-3). "
                             "Recorded in the artifact meta.")
    args = parser.parse_args()

    if args.decoder_ridge is not None:
        # Both this module and quanpc_standalone.online read the constant as a
        # module global; learn_pair goes through the latter for its initial
        # encoders, so both bindings have to move together.
        import quantpc.quanpc_standalone.online as online_mod

        global ONLINE_DECODER_RIDGE
        ONLINE_DECODER_RIDGE = float(args.decoder_ridge)
        online_mod.ONLINE_DECODER_RIDGE = float(args.decoder_ridge)
        print(f"block-32 decoder ridge overridden to {args.decoder_ridge}", flush=True)

    # The collector reads sink/local off its own module globals: the attention
    # hook it installs has no argument channel.
    import collect as collector

    collector.sink_size = int(args.sink_size)
    collector.local_size = int(args.local_size)
    set_collect_stride(args.collect_stride)

    config = load_model_config(args.model_name)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    layers = parse_layers(args.layers, num_layers)

    if args.output_path is None:
        model_tag = Path(str(args.model_name).rstrip("/")).name
        args.output_path = str(
            BASE / "projections"
            / f"{model_tag}_k{args.k_bits}_v{args.v_bits}.pt"
        )

    print("model", args.model_name, flush=True)
    print("layers", format_layer_group(layers), flush=True)
    print("k_bits", args.k_bits, "v_bits", args.v_bits,
          "zerorope", args.zerorope, flush=True)
    print("output", args.output_path, flush=True)

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

    run(args, samples, layers, num_layers, num_kv_heads, head_dim,
        args.k_bits, args.v_bits, args.zerorope)


if __name__ == "__main__":
    main()
