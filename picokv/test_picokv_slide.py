"""Bit-exact check of offline incremental decode (periodic local-window slide).

A sliding cache is prefilled on ``L`` tokens and then decodes ``steps`` tokens
one by one; every ``slide_tokens`` steps the oldest recent tokens move to the
packed global cache.  The reference cache is built directly from the full
``L + steps`` tokens with the production prefill kernels: same EAS indices as
the prefill, kept indices extended with the slid global positions, and the
final recent region as the local window.  Dense and packed tensors must match
bit for bit, and decode outputs must match up to atomic-add noise.

    python picokv/test_picokv_slide.py --proj offline_projection/projections/Llama-3.1-8B-Instruct_k320_v320.pt
"""

from __future__ import annotations

import argparse

import torch

from picokv.picokv_prefill_cuda_extension.picokv_prefill_cuda import (
    blockk_fixed_cuda,
    gather_pack_from_packed_cuda,
)
from picokv.picokv_standalone import PreparedKVCache, StandaloneConfig, decode_attention
from picokv.picokv_standalone.prepare import _select_eas_and_kept_indices
from picokv.picokv_standalone.slide import pack_value_groups, unpack_value_groups
from picokv.picokv_standalone.transformers import (
    StandaloneLlamaCache,
    load_offline_projections,
)


def check_value_pack_roundtrip(device: torch.device) -> None:
    for tokens in (1, 63, 64, 65, 200):
        bits = torch.randint(0, 2, (2, 3, tokens, 320), device=device, dtype=torch.long)
        packed = pack_value_groups(bits)
        assert packed.shape[2] == (tokens + 63) // 64
        assert torch.equal(unpack_value_groups(packed, tokens), bits)
        # Padding bits must stay zero, as the prefill gather kernel writes them.
        assert torch.equal(
            unpack_value_groups(packed, packed.shape[2] * 64)[:, :, tokens:],
            torch.zeros_like(unpack_value_groups(packed, packed.shape[2] * 64)[:, :, tokens:]),
        )
        # Offset packing places token j at bit offset+j with zeros elsewhere.
        for offset in (1, 37, 63):
            shifted = pack_value_groups(bits, offset=offset)
            padded = torch.nn.functional.pad(bits, (0, 0, offset, 0))
            assert torch.equal(shifted, pack_value_groups(padded))


def build_reference(
    sliding: StandaloneLlamaCache,
    layer: int,
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    prompt_len: int,
) -> PreparedKVCache:
    cfg = sliding.config
    cache = sliding.layers[layer]
    key_encoder, value_encoder = sliding._slider.layer_encoders(layer)
    sink, local, zr = cfg.sink_size, cfg.local_size, cfg.zerorope
    total = int(full_key.shape[-2])
    prompt_global = prompt_len - sink - local

    # EAS/kept selection exactly as prepare_kv_cache did on the prompt.
    _, key_err = blockk_fixed_cuda(
        full_key[:, :, :prompt_len].contiguous(), key_encoder, cache.key_projection,
        sink, prompt_global, zerorope=zr, apply_zerorope=True, pack_key_layout=True,
    )
    _, value_err = blockk_fixed_cuda(
        full_value[:, :, :prompt_len].contiguous(), value_encoder, cache.value_projection,
        sink, prompt_global, zerorope=0, apply_zerorope=False, pack_key_layout=False,
    )
    eas_indices, kept_indices = _select_eas_and_kept_indices(
        key_err, value_err, max(1, int(prompt_global * cfg.eas))
    )

    total_global = prompt_global + sliding._slider.slides * cfg.slide_tokens
    packed_key_all, _ = blockk_fixed_cuda(
        full_key, key_encoder, cache.key_projection,
        sink, total_global, zerorope=zr, apply_zerorope=True, pack_key_layout=True,
    )
    packed_value_all, _ = blockk_fixed_cuda(
        full_value, value_encoder, cache.value_projection,
        sink, total_global, zerorope=0, apply_zerorope=False, pack_key_layout=False,
    )
    slid = torch.arange(prompt_global, total_global, device=full_key.device, dtype=torch.int32)
    kept_indices = torch.cat(
        (kept_indices, slid.expand(*kept_indices.shape[:2], -1)), dim=-1
    ).contiguous()
    dense_key, dense_value, packed_key, packed_value = gather_pack_from_packed_cuda(
        full_key, full_value, packed_key_all, packed_value_all,
        eas_indices, kept_indices, sink, total - sink - total_global,
    )
    return PreparedKVCache(
        dense_key=dense_key,
        dense_value=dense_value,
        key_projection=cache.key_projection,
        value_projection=cache.value_projection,
        key_projection_t=cache.key_projection_t,
        packed_key=packed_key,
        packed_value=packed_value,
        exp_bits=cache.exp_bits,
    )


def run_case(
    projs: dict, bits: tuple[int, int], prompt_len: int, steps: int, slide: int, layers: int
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(prompt_len * 1000 + steps)
    batch, heads, dim = 1, int(projs["key_encoder"].shape[1]), 128
    total = prompt_len + steps
    full_key = torch.randn(layers, batch, heads, total, dim, device=device, dtype=torch.bfloat16)
    full_value = torch.randn_like(full_key)

    config = StandaloneConfig(key_bits=bits[0], value_bits=bits[1], slide_tokens=slide)
    sliding = StandaloneLlamaCache(
        config,
        key_encoders=projs["key_encoder"],
        value_encoders=projs["value_encoder"],
        key_decoders=projs["key_decoder"],
        value_decoders=projs["value_decoder"],
    )
    for layer in range(layers):
        sliding.prepare(
            layer,
            full_key[layer, :, :, :prompt_len].contiguous(),
            full_value[layer, :, :, :prompt_len].contiguous(),
        )
    prompt_kept = sliding.layers[0].global_len
    for t in range(steps):
        for layer in range(layers):
            query = torch.randn(batch, 4 * heads, 1, dim, device=device, dtype=torch.bfloat16)
            sliding.decode(
                layer, query,
                full_key[layer, :, :, prompt_len + t : prompt_len + t + 1].contiguous(),
                full_value[layer, :, :, prompt_len + t : prompt_len + t + 1].contiguous(),
            )
    assert sliding.slide_count == layers * (steps // slide), sliding.slide_count
    sliding.wait_slides()

    for layer in range(layers):
        got = sliding.layers[layer]
        ref = build_reference(sliding, layer, full_key[layer], full_value[layer], prompt_len)
        for name in ("dense_key", "dense_value", "packed_key", "packed_value"):
            a, b = getattr(got, name), getattr(ref, name)
            assert a.shape == b.shape, (layer, name, tuple(a.shape), tuple(b.shape))
            assert torch.equal(a, b), (layer, name, int((a != b).sum()))

        query = torch.randn(batch, 4 * heads, 1, dim, device=device, dtype=torch.bfloat16)
        out_got = decode_attention(query, got).clone()
        out_ref = decode_attention(query, ref).clone()
        out_ref2 = decode_attention(query, ref).clone()
        diff = (out_got.float() - out_ref.float()).abs().max().item()
        noise = (out_ref2.float() - out_ref.float()).abs().max().item()
        assert diff <= max(noise, 1e-2), (layer, diff, noise)
    print(
        f"ok L={prompt_len} steps={steps} layers={layers} slides/layer={sliding._slider.slides} "
        f"kept%64={prompt_kept % 64} global {prompt_kept}->{got.global_len} "
        f"dense={got.dense_len} out_maxdiff={diff:.3e} (rerun noise {noise:.3e})",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proj",
        default="offline_projection/projections/Llama-3.1-8B-Instruct_k320_v320.pt",
    )
    parser.add_argument("--slide", type=int, default=64)
    args = parser.parse_args()

    check_value_pack_roundtrip(torch.device("cuda"))
    print("ok value pack/unpack roundtrip", flush=True)

    projs = load_offline_projections(args.proj)
    bits = (int(projs["key_encoder"].shape[-2]), int(projs["value_encoder"].shape[-2]))
    # G=kept+EAS; the prompt lengths cover kept%64 == 0 and != 0 and steps that
    # stop mid-interval (recent region longer than the local window).  Several
    # layers check the batched (layers stacked into the batch dim) slide.
    for prompt_len, steps, layers in ((1024, 64, 3), (4096, 200, 3), (4434, 128, 2), (32768, 320, 2)):
        run_case(projs, bits, prompt_len, steps, args.slide, layers)
    print("all slide checks passed")


if __name__ == "__main__":
    main()
