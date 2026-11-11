"""Correctness checks for native packing layouts and reconstructed attention."""

import pytest
import torch
import torch.nn.functional as F

from picokv.picokv_standalone.cache import PreparedKVCache
from rebuttal.reconstruct_attention import decode_attention, reconstruct_global_kv, unpack_codes


def _make_cache(length, k_bits, v_bits, device):
    # Independent packer: K packs code dimensions, V packs token dimensions;
    # sign conventions match the production CUDA pack/gather kernels.
    generator = torch.Generator().manual_seed(23)
    key_codes = (torch.randint(2, (2, 2, length, k_bits), generator=generator) * 2 - 1).to(torch.bfloat16)
    value_codes = (torch.randint(2, (2, 2, length, v_bits), generator=generator) * 2 - 1).to(torch.bfloat16)
    weights = torch.ones(64, dtype=torch.int64) << torch.arange(64)
    packed_key = (
        (key_codes < 0).reshape(2, 2, length, k_bits // 64, 64).long() * weights
    ).sum(-1).transpose(-1, -2).contiguous()
    value_binary = F.pad((value_codes > 0).long(), (0, 0, 0, (-length) % 64))
    packed_value = (
        value_binary.reshape(2, 2, -1, 64, v_bits) * weights[:, None]
    ).sum(-2).contiguous()
    key_decoder = (torch.randn(2, 2, k_bits, 128, generator=generator) / k_bits**0.5).to(device=device, dtype=torch.bfloat16)
    value_decoder = (torch.randn(2, 2, v_bits, 128, generator=generator) / v_bits**0.5).to(device=device, dtype=torch.bfloat16)
    cache = PreparedKVCache(
        dense_key=torch.randn(2, 2, 5, 128, generator=generator).to(device=device, dtype=torch.bfloat16),
        dense_value=torch.randn(2, 2, 5, 128, generator=generator).to(device=device, dtype=torch.bfloat16),
        key_projection=key_decoder,
        value_projection=value_decoder,
        key_projection_t=key_decoder.transpose(-1, -2).contiguous(),
        packed_key=packed_key.to(device), packed_value=packed_value.to(device), exp_bits=4,
    )
    return cache, key_codes.to(device), value_codes.to(device)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length,k_bits,v_bits", [(1, 128, 192), (63, 256, 320), (64, 320, 256), (65, 320, 320), (129, 512, 128)])
def test_native_layout_and_reconstruction(device, length, k_bits, v_bits):
    cache, key_codes, value_codes = _make_cache(length, k_bits, v_bits, device)
    torch.testing.assert_close(unpack_codes(cache.packed_key, length=length, is_key=True), key_codes, rtol=0, atol=0)
    torch.testing.assert_close(unpack_codes(cache.packed_value, length=length, is_key=False), value_codes, rtol=0, atol=0)
    key, value = reconstruct_global_kv(cache)
    torch.testing.assert_close(key, key_codes @ cache.key_projection, rtol=0, atol=0)
    torch.testing.assert_close(value, value_codes @ cache.value_projection, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("update_cache", [False, True])
def test_attention_and_incremental_cache(device, update_cache):
    _check_attention_and_incremental_cache(device, update_cache, decode_attention)


def _check_attention_and_incremental_cache(device, update_cache, forward, expect_workspace=False):
    cache, key_codes, value_codes = _make_cache(65, 256, 320, device)
    query = torch.randn(2, 8, 1, 128, device=device, dtype=torch.bfloat16)
    new_key = torch.randn_like(cache.dense_key[:, :, :1])
    new_value = torch.randn_like(new_key)
    packed_key, packed_value = cache.packed_key.clone(), cache.packed_value.clone()
    key_decoder, value_decoder = cache.key_projection.clone(), cache.value_projection.clone()
    for step in range(2):
        length_before = cache.dense_len
        key = torch.cat((key_codes @ cache.key_projection, cache.dense_key, new_key), dim=-2)
        value = torch.cat((value_codes @ cache.value_projection, cache.dense_value, new_value), dim=-2)
        # Explicit FP32 dense reference, with GQA head mapping and custom scale.
        scores = query.float() @ key.repeat_interleave(4, dim=1).float().transpose(-1, -2) * 0.07
        expected = scores.softmax(-1) @ value.repeat_interleave(4, dim=1).float()
        output = forward(
            query, cache, decode_key=new_key, decode_value=new_value,
            update_cache=update_cache, attention_scale=0.07,
        )
        torch.testing.assert_close(output.float(), expected, atol=0.008, rtol=0.015)
        assert cache.dense_len == length_before + int(update_cache)
        assert cache.global_len == 65
    torch.testing.assert_close(cache.packed_key, packed_key, rtol=0, atol=0)
    torch.testing.assert_close(cache.packed_value, packed_value, rtol=0, atol=0)
    torch.testing.assert_close(cache.key_projection, key_decoder, rtol=0, atol=0)
    torch.testing.assert_close(cache.value_projection, value_decoder, rtol=0, atol=0)
    assert bool(cache._workspace) is expect_workspace


def test_reject_pruning():
    cache, _, _ = _make_cache(1, 128, 128, "cpu")
    with pytest.raises(ValueError, match="pruning"):
        decode_attention(torch.zeros(2, 8, 1, 128, dtype=torch.bfloat16), cache, use_pruning=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length,k_bits,v_bits", [(1, 128, 192), (63, 256, 320), (64, 320, 256), (65, 320, 320), (129, 512, 128)])
def test_cuda_fused_reconstruction(length, k_bits, v_bits):
    from rebuttal.reconstruct_cuda_attention import reconstruct_full_kv

    cache, key_codes, value_codes = _make_cache(length, k_bits, v_bits, "cuda")
    key, value = reconstruct_full_kv(cache)
    torch.testing.assert_close(key[:, :, :length], key_codes @ cache.key_projection, atol=2e-5, rtol=0.008)
    torch.testing.assert_close(value[:, :, :length], value_codes @ cache.value_projection, atol=2e-5, rtol=0.008)
    torch.testing.assert_close(key[:, :, length:], cache.dense_key, atol=0, rtol=0)
    torch.testing.assert_close(value[:, :, length:], cache.dense_value, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("update_cache", [False, True])
def test_cuda_fused_attention(update_cache):
    from rebuttal.reconstruct_cuda_attention import decode_attention as forward

    _check_attention_and_incremental_cache("cuda", update_cache, forward, expect_workspace=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_reconstruction_inference_mode():
    from rebuttal.reconstruct_cuda_attention import reconstruct_full_kv

    with torch.inference_mode():
        cache, key_codes, value_codes = _make_cache(65, 320, 320, "cuda")
        for _ in range(2):
            key, value = reconstruct_full_kv(cache)
            torch.testing.assert_close(key[:, :, :65], key_codes @ cache.key_projection,
                                       atol=2e-5, rtol=0.008)
            torch.testing.assert_close(value[:, :, :65], value_codes @ cache.value_projection,
                                       atol=2e-5, rtol=0.008)
        assert len(cache._workspace) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [65, 511, 512, 513, 32768])
def test_block_fusion_keeps_dense_path_and_avoids_full_reconstruction(monkeypatch, length):
    from rebuttal import reconstruct_cuda_attention as module
    from picokv.picokv_standalone.decode import _dense_attention as original_dense

    def forbid_full_reconstruction(*args, **kwargs):
        raise AssertionError("block attention must not materialize full K/V")

    dense_calls = []

    def record_dense(query, key, value, scale):
        dense_calls.append(key.shape[2])
        return original_dense(query, key, value, scale)

    monkeypatch.setattr(module, "reconstruct_full_kv", forbid_full_reconstruction)
    monkeypatch.setattr(module, "_dense_attention", record_dense)
    torch.manual_seed(97)
    with torch.inference_mode():
        cache, ck, cv = _make_cache(length, 320, 320, "cuda")
        query = torch.randn(2, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
        for step in range(2):
            new_key = torch.randn_like(cache.dense_key[:, :, :1])
            new_value = torch.randn_like(new_key)
            key = torch.cat((ck @ cache.key_projection, cache.dense_key, new_key), -2)
            value = torch.cat((cv @ cache.value_projection, cache.dense_value, new_value), -2)
            expected = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
            actual = module.decode_attention(query, cache, decode_key=new_key,
                                             decode_value=new_value, update_cache=True)
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.02)
        assert dense_calls == [6, 7]
