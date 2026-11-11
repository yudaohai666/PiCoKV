"""GPU checks of custom BF16 attention against both GEMM and FP32 references."""

import pytest
import torch

from rebuttal.test_reconstruct_attention import _make_cache
from rebuttal.unpacked_attention import decode_attention as reference
from rebuttal.unpacked_kernel_attention import decode_attention, decode_attention_fused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel", [decode_attention, decode_attention_fused])
@pytest.mark.parametrize("length,k_bits,v_bits", [(1, 128, 192), (63, 256, 320), (65, 320, 256), (129, 512, 128), (65536, 320, 320)])
def test_kernel(kernel, length, k_bits, v_bits):
    torch.manual_seed(85)
    cache, codes_k, codes_v = _make_cache(length, k_bits, v_bits, "cuda")
    query = torch.randn(2, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
    for step in range(2):
        new_k = torch.randn_like(cache.dense_key[:, :, :1])
        new_v = torch.randn_like(new_k)
        expected = reference(query, cache, decode_key=new_k, decode_value=new_v, attention_scale=0.07)
        before = cache.dense_len
        actual = kernel(query, cache, decode_key=new_k, decode_value=new_v, update_cache=True, attention_scale=0.07)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.008, rtol=0.02)
        key = torch.cat((codes_k.float() @ cache.key_projection.float(), cache.dense_key.float()), dim=-2)
        value = torch.cat((codes_v.float() @ cache.value_projection.float(), cache.dense_value.float()), dim=-2)
        scores = query.float() @ key.repeat_interleave(4, dim=1).transpose(-1, -2) * 0.07
        fp32_output = scores.softmax(-1) @ value.repeat_interleave(4, dim=1)
        torch.testing.assert_close(actual.float(), fp32_output, atol=0.02, rtol=0.04)
        assert cache.dense_len == before + 1
        if kernel is decode_attention_fused:
            assert cache._workspace  # Same persistent buffers as production.
        else:
            assert cache._workspace == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_only_and_numerical_stability():
    cache, _, _ = _make_cache(1, 128, 128, "cuda")
    cache.packed_key = cache.packed_key[..., :0].contiguous()
    cache.packed_value = cache.packed_value[:, :, :0].contiguous()
    query = torch.full((2, 8, 1, 128), 10.0, device="cuda", dtype=torch.bfloat16)
    for kernel in [decode_attention, decode_attention_fused]:
        actual = kernel(query, cache)
        expected = reference(query, cache)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.008, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("restore", [False, True])
def test_production_output_kernel_is_reused(monkeypatch, restore):
    from rebuttal import reconstruct_cuda_attention, unpacked_kernel_attention

    module = reconstruct_cuda_attention if restore else unpacked_kernel_attention
    forward = module.decode_attention if restore else module.decode_attention_fused
    extension = module.load_picokv_extension()
    calls = []

    class OutputKernel:
        def picokv_fused_output_projection_fn(self, *args):
            calls.append(args[3])
            return extension.picokv_fused_output_projection_fn(*args)

    monkeypatch.setattr(module, "load_picokv_extension", lambda: OutputKernel())
    cache, _, _ = _make_cache(65, 320, 320, "cuda")
    query = torch.randn(2, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
    first = forward(query, cache).clone()
    buffers = [v.data_ptr() for v in next(iter(cache._workspace.values()))]
    second = forward(query, cache)
    assert buffers == [v.data_ptr() for v in next(iter(cache._workspace.values()))]
    assert len(calls) == 2
    torch.testing.assert_close(second, first, rtol=0.008, atol=0.008)
    if restore:
        torch.testing.assert_close(calls[0][0, 0], torch.eye(128, device="cuda", dtype=torch.bfloat16))
    else:
        assert calls[0].data_ptr() == cache.value_projection.data_ptr()
