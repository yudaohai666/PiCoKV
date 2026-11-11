"""Check code-domain attention against an independent FP32 dense reference."""

import pytest
import torch

from rebuttal.test_reconstruct_attention import DEVICES, _make_cache
from rebuttal.unpacked_attention import decode_attention


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length,k_bits,v_bits", [(1, 128, 192), (63, 256, 320), (65, 320, 256), (129, 512, 128)])
@pytest.mark.parametrize("update_cache", [False, True])
def test_unpacked_attention(device, length, k_bits, v_bits, update_cache):
    cache, key_codes, value_codes = _make_cache(length, k_bits, v_bits, device)
    torch.manual_seed(73)
    query = torch.randn(2, 8, 1, 128, device=device, dtype=torch.bfloat16)
    new_key = torch.randn_like(cache.dense_key[:, :, :1])
    new_value = torch.randn_like(new_key)
    packed_key, packed_value = cache.packed_key.clone(), cache.packed_value.clone()
    for step in range(2):
        before = cache.dense_len
        # Build reference K/V in FP32, independently of the forward's
        # factorization order. Concatenation gives one shared softmax.
        key = torch.cat((key_codes.float() @ cache.key_projection.float(), cache.dense_key.float(), new_key.float()), dim=-2)
        value = torch.cat((value_codes.float() @ cache.value_projection.float(), cache.dense_value.float(), new_value.float()), dim=-2)
        scores = query.float() @ key.repeat_interleave(4, dim=1).transpose(-1, -2) * 0.07
        expected = scores.softmax(-1) @ value.repeat_interleave(4, dim=1)
        actual = decode_attention(
            query, cache, decode_key=new_key, decode_value=new_value,
            update_cache=update_cache, attention_scale=0.07,
        )
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.04)
        assert cache.dense_len == before + int(update_cache)
    torch.testing.assert_close(cache.packed_key, packed_key, rtol=0, atol=0)
    torch.testing.assert_close(cache.packed_value, packed_value, rtol=0, atol=0)
    assert cache._workspace == {}


def test_unpacked_reject_pruning():
    cache, _, _ = _make_cache(1, 128, 128, "cpu")
    with pytest.raises(ValueError, match="pruning"):
        decode_attention(torch.zeros(2, 8, 1, 128, dtype=torch.bfloat16), cache, use_pruning=True)
