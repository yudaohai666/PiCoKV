"""Build all three offline matrices in one pass: whitening, encoder, decoder.

The three cannot be built independently.  ``W`` depends only on the queries;
the encoder must be learned on ``K W``; the decoder must be fitted against the
codes that encoder produces in that same basis.  Splitting them across scripts
means three calibration passes and three chances to disagree about which W the
encoder was fitted against -- the failure already latent in this tree, where
the encoder-only script wrote ``meta["whitened"] = False`` and nothing ever read
it.  So: one pass, one artifact, one meta block.

What ships is the *folded* pair.  Writing the block-32 chain in the whitened
basis (``R_g = R'_g W``) and rewriting it in the model basis gives

    E~_g = E_g W        D~_g = D_g W^-1

with the codes unchanged (sign is invariant to the right multiplication) and the
decoder solve unchanged (the ridge sits on ``C^T C``, which never sees W).  So
whitening's entire content is *which directions the encoder picks*, and it can
ship as a pair of ordinary matrices: K is never rotated at prefill, Q is never
rotated at decode, and ``key_projection_t`` becomes ``W^-1 D^T`` for free.

Stages (``--stage``):
  * collect -> dump per-layer post-RoPE K/V shards, each carrying this sample's
               query Gram (~1.5% of the shard, so W costs no extra pass).
  * fit     -> per layer: sum the Grams into W, then run one canonical online
               block-32 pass on ``K W``/V and store the folded pair.
  * all     -> collect then fit, layer group by layer group (default).

Both sides are whitened. They do not share a source: K's comes from the
calibration query Gram, V's from ``o_proj``'s weight -- a constant of the model,
so the fit stage reads it from the checkpoint rather than through the shards.
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
    set_collect_query_grams,
    set_collect_stride,
)
from quantpc.quanpc_standalone.constants import (  # noqa: E402
    ONLINE_DECODER_RIDGE,
    ONLINE_GROUP_BITS,
)
from quantpc.quanpc_standalone.whiten import value_whitening_source  # noqa: E402
from quantpc.model_config import load_model_config  # noqa: E402
from quantpc.quanpc_standalone.online import (  # noqa: E402
    _strict_binary_sign_,
    quantize_block32_online,
)
from transformers import AutoModelForCausalLM  # noqa: E402

FOLDED_FORMAT = "quanpc_standalone_block32_folded_v1"
DEFAULT_K_BITS = 320
DEFAULT_V_BITS = 320
# Ridge on the normalized query Gram before the square root.  The online
# find_orth_proj gets this for free and by accident: it builds the Gram in
# float32, where the smallest eigenvalues of an ill-conditioned layer are below
# the representable range and end up implicitly regularized.  At float64 that
# safety net is gone, and inverting a 1e6-condition Gram exactly is what makes
# whitening lose on layer 0. Keep it fixed so whitening artifacts remain
# comparable; it is an internal numerical-stability constant, not a sweep knob.
WHITEN_RIDGE = 1e-3


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


def whitening_from_gram(gram, ridge):
    """Return ``(W, W^-1)`` in float64 from a per-head query Gram.

    The Gram is normalized to unit mean eigenvalue first, so ``ridge`` means the
    same thing on every layer.  A global scale on W is free: it cancels between
    ``E W`` and ``D W^-1``, and the codes are signs.
    """
    dim = gram.shape[-1]
    scale = gram.diagonal(dim1=-2, dim2=-1).sum(-1) / dim
    normalized = gram / scale.clamp_min(1e-30).view(-1, 1, 1)
    normalized = normalized + ridge * torch.eye(
        dim, device=gram.device, dtype=gram.dtype
    )
    values, vectors = torch.linalg.eigh(normalized)
    values = values.clamp_min(ridge).sqrt()
    forward = (vectors * values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    inverse = (vectors / values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return forward, inverse


def value_whitening_factors(model_name, layer_idx, num_kv_heads, head_dim, ridge, device):
    """Fit V whitening for one layer from that layer's ``o_proj`` weight.

    Unlike K, whose Gram comes from the calibration queries, V's whitening
    source is a constant of the model, so the fit stage reads the weight
    straight from the checkpoint rather than carrying anything through the
    shards. Only the one tensor is materialized, not the model.
    """
    from safetensors import safe_open
    import glob as _glob
    import json as _json

    index = os.path.join(model_name, "model.safetensors.index.json")
    key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    if os.path.exists(index):
        shard = _json.load(open(index))["weight_map"][key]
        files = [os.path.join(model_name, shard)]
    else:
        files = sorted(_glob.glob(os.path.join(model_name, "*.safetensors")))
    for path in files:
        with safe_open(path, framework="pt") as handle:
            if key in handle.keys():
                weight = handle.get_tensor(key)
                break
    else:
        raise RuntimeError(f"{key} not found under {model_name}")

    source = value_whitening_source(weight.to(device), num_kv_heads, head_dim)
    gram = torch.matmul(source.transpose(-1, -2).double(), source.double())
    return whitening_from_gram(gram, ridge)


def load_layer_states(shard_paths, device):
    """Read every shard once and return ``{kind: [heads, rows, dim]}`` plus the Gram.

    Reading a head at a time would re-read the whole layer for each head, so
    fitting one layer costs sixteen full passes over ~5GB of shards.  Reading
    once and keeping the layer in host memory turns that into one pass; a layer
    is a few GB and the fit machine has hundreds.
    """
    keys, values, gram = [], [], None
    for shard_path in tqdm(shard_paths, desc="load layer", leave=False):
        sample = torch.load(shard_path, map_location="cpu")
        keys.append(sample["key"])
        values.append(sample["value"])
        sample_gram = sample.get("query_gram")
        if sample_gram is None:
            raise RuntimeError(
                f"{shard_path} has no query_gram; re-run collect with this script "
                "(collect.py only stores it when set_collect_query_grams is on)"
            )
        sample_gram = sample_gram.to(device=device, dtype=torch.float64)
        gram = sample_gram if gram is None else gram + sample_gram
        del sample
    return {
        "key": torch.cat(keys, dim=1),
        "value": torch.cat(values, dim=1),
    }, gram


def rotate_states(states, forward, chunk=65536):
    """``states @ W`` at float64, chunked: a float64 copy of a 1M-row head is 1GB."""
    out = torch.empty_like(states)
    length = states.shape[-2]
    for start in range(0, length, chunk):
        block = states[:, :, start:start + chunk].double()
        out[:, :, start:start + chunk] = torch.matmul(block, forward).to(states.dtype)
    return out


def learn_pair(states, bits):
    """Run the canonical online block-32 forward once.

    ``quantize_block32_online`` already learns each encoder, solves its
    ridge-regularized decoder, subtracts that reconstruction, and advances to
    the next group.  Re-solving from the concatenated codes here would be a
    different algorithm from the production online path, so offline fitting
    deliberately has no additional decoder-refit pass.
    """
    with torch.no_grad():
        block = quantize_block32_online(states, bits)
        encoders = block.encoders[0, 0].unsqueeze(0).contiguous()
        decoders = block.decoder[0, 0].unsqueeze(0).contiguous()
        del block
    return encoders[0], decoders[0]


def fold(encoder, decoder, forward, inverse):
    """Absorb W into the stored pair: ``E W`` (renormalized) and ``D W^-1``.

    Row norms are free -- codes are signs -- so they are reset to 1 to keep the
    stored bf16 values in a sane range even where cond(W) reaches 1e3.
    """
    folded_encoder = torch.matmul(encoder.double(), forward)
    folded_encoder = folded_encoder / folded_encoder.norm(dim=-1, keepdim=True)
    folded_decoder = torch.matmul(decoder.double(), inverse)
    return folded_encoder, folded_decoder


def check_fold(states, encoder, decoder, folded_encoder, folded_decoder,
               forward, rows=65536):
    """Confirm the folded pair reconstructs as well as the rotated route.

    Both are judged in the norm both minimize, ``||(K^ - K) W||``.  Bit-for-bit
    code agreement is not the criterion: a borderline projection that rounds the
    other way in bf16 gives different codes of equal quality.
    """
    flat = states.reshape(1, states.shape[2], states.shape[3])
    flat = _subsample_rows(flat[0], rows).unsqueeze(0).contiguous()
    whitened = torch.matmul(flat.double(), forward).to(torch.bfloat16)
    rotated = torch.matmul(
        chain_codes(whitened, encoder.unsqueeze(0), decoder.unsqueeze(0)),
        decoder.unsqueeze(0),
    )
    # Cast the folded pair the way it will actually ship, so the check covers
    # the bf16 storage of matrices whose conditioning W has inflated.
    stored_encoder = folded_encoder.to(torch.bfloat16).unsqueeze(0)
    stored_decoder = folded_decoder.to(torch.bfloat16).unsqueeze(0)
    folded = torch.matmul(
        chain_codes(flat, stored_encoder, stored_decoder), stored_decoder
    )
    target = torch.matmul(flat.double(), forward)
    rotated_error = float(
        (rotated.double() - target).norm() / target.norm()
    )
    folded_error = float(
        (torch.matmul(folded.double(), forward) - target).norm() / target.norm()
    )
    return rotated_error, folded_error


def folded_meta(args, k_bits, v_bits, zerorope, whitened):
    return {
        "format": FOLDED_FORMAT,
        "group_bits": ONLINE_GROUP_BITS,
        "ridge": ONLINE_DECODER_RIDGE,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "zerorope": zerorope,
        "sink_size": args.sink_size,
        "local_size": args.local_size,
        # Whitening, when enabled, is applied to every layer and folded into
        # the stored encoder/decoder pair. The factors are not serialized.
        "whitened": bool(whitened),
        "whiten_ridge": WHITEN_RIDGE,
        "calibration": {
            "datasets": args.datasets,
            "length_config": args.length_config,
            "collect_stride": args.collect_stride,
            "learn_tokens": args.block32_learn_tokens,
        },
    }


def load_state(args, num_layers, num_kv_heads, head_dim, k_bits, v_bits,
               zerorope, whitened):
    shapes = {
        "key_encoder": (num_layers, num_kv_heads, k_bits, head_dim),
        "value_encoder": (num_layers, num_kv_heads, v_bits, head_dim),
    }
    empty = {
        "key_encoder": torch.zeros(shapes["key_encoder"], dtype=torch.bfloat16),
        "value_encoder": torch.zeros(shapes["value_encoder"], dtype=torch.bfloat16),
        "key_decoder": torch.zeros(shapes["key_encoder"], dtype=torch.bfloat16),
        "value_decoder": torch.zeros(shapes["value_encoder"], dtype=torch.bfloat16),
        "meta": folded_meta(args, k_bits, v_bits, zerorope, whitened),
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
              k_bits, v_bits, zerorope, whiten):
    shard_paths = get_layer_shard_paths(args, layer_idx, sample_count)
    layer_states, gram = load_layer_states(shard_paths, device)
    num_heads = layer_states["key"].shape[0]

    if whiten:
        forward, inverse = whitening_from_gram(gram, WHITEN_RIDGE)
        value_forward, value_inverse = value_whitening_factors(
            args.model_name, layer_idx, num_heads,
            state["key_encoder"].shape[-1], WHITEN_RIDGE, device,
        )
        condition = float(
            (torch.linalg.svdvals(gram)[..., 0]
             / torch.linalg.svdvals(gram)[..., -1].clamp_min(1e-30)).mean()
        )
        print(f"layer {layer_idx} whitening: cond(Q^T Q)={condition:.3e} "
              f"ridge={WHITEN_RIDGE}", flush=True)
    else:
        eye = torch.eye(state["key_encoder"].shape[-1], device=device, dtype=torch.float64)
        forward = eye.unsqueeze(0).expand(num_heads, -1, -1).contiguous()
        inverse = forward.clone()
        value_forward, value_inverse = forward, inverse
        print(f"layer {layer_idx} whitening: disabled (identity)", flush=True)
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

            if whiten:
                side_forward = forward if kind == "key" else value_forward
                side_inverse = inverse if kind == "key" else value_inverse
                head_forward = side_forward[head_idx]
                head_inverse = side_inverse[head_idx]
                rotated = rotate_states(states, head_forward)
                encoder, decoder = learn_pair(
                    rotated, bits
                )
                del rotated
                folded_encoder, folded_decoder = fold(
                    encoder, decoder, head_forward, head_inverse
                )
                if head_idx == 0 and kind == "key":
                    rotated_error, folded_error = check_fold(
                        states, encoder, decoder,
                        folded_encoder, folded_decoder, head_forward,
                    )
                    print(f"  layer {layer_idx} fold check: rotated "
                          f"{rotated_error:.4f} vs folded {folded_error:.4f}",
                          flush=True)
                encoder, decoder = folded_encoder, folded_decoder
            else:
                encoder, decoder = learn_pair(
                    states, bits
                )

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
        k_bits, v_bits, zerorope, whitened):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device(
        args.projection_device
        if args.projection_device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state, progress = load_state(
        args, num_layers, num_kv_heads, head_dim, k_bits, v_bits,
        zerorope, whitened,
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
                k_bits, v_bits, zerorope, whitened,
            )
            done = (
                bool(progress["key_done"][layer_idx].all())
                and bool(progress["value_done"][layer_idx].all())
            )
            if done:
                remove_layer_tmp(args, layer_idx)

    save_state(args, state, progress)
    print("saved", args.output_path, flush=True)


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true/false, got {value!r}"
    )


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
    parser.add_argument("--whiten", type=parse_bool, default=False,
                        help="whiten and fold all layers (true/false; default false)")
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
    set_collect_query_grams(True)

    config = load_model_config(args.model_name)
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    layers = parse_layers(args.layers, num_layers)

    if args.output_path is None:
        model_tag = Path(str(args.model_name).rstrip("/")).name
        whiten_tag = "whiten" if args.whiten else "nowhiten"
        args.output_path = str(
            BASE / "projections"
            / f"{model_tag}_k{args.k_bits}_v{args.v_bits}_{whiten_tag}.pt"
        )

    print("model", args.model_name, flush=True)
    print("layers", format_layer_group(layers), flush=True)
    print("whiten", "all layers" if args.whiten else "disabled", flush=True)
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
        args.k_bits, args.v_bits, args.zerorope, args.whiten)


if __name__ == "__main__":
    main()
