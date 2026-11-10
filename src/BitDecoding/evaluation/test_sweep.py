"""Kernel-level accuracy sweep over (num_bits, group_size) for fwd_kvcache_int.

Same structure as test.py, but parameterized and reporting only the final
mean-absolute-error so a bad dispatch is distinguishable from genuine
quantization loss.

    python test_sweep.py
"""

import math
import sys

import torch

from bit_decode import DynamicCache, fwd_kvcache_int, kvcache_pack_int


def attention_ref(q, k, v):
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhts,bshd->bthd", attention, v)


def run(num_bits, group_size, residual_block_size, quant_mode="k-channel",
        seqlen_k=1024, rounds=16, device="cuda", dtype=torch.float16):
    pack_nums = 16 / num_bits
    layer_idx = 0
    batch_size, nheads, nheads_k, d = 1, 32, 32, 128
    seqlen_q = 1
    sm_scale = 1.0 / math.sqrt(d)

    torch.manual_seed(42)
    q = torch.rand(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k_state = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v_state = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)

    residual_len = seqlen_k % residual_block_size
    seqlen_k_pack = seqlen_k - residual_len

    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k_pack, seqlen_k_pack,
                                dtype=torch.int32, device=device)

    k_pack = torch.zeros((batch_size, int(seqlen_k_pack // pack_nums), nheads_k, d),
                         dtype=torch.uint16, device=device)
    k_params = torch.zeros((batch_size, int(seqlen_k_pack // group_size), nheads_k, d),
                           dtype=torch.float32, device=device)
    v_pack = torch.zeros((batch_size, seqlen_k_pack, nheads_k, int(d // pack_nums)),
                         dtype=torch.uint16, device=device)
    v_params = torch.zeros((batch_size, int(d // group_size), nheads_k, seqlen_k_pack),
                           dtype=torch.float32, device=device)

    past_key_value = DynamicCache()

    if residual_len > 0:
        past_key_value.update_residual(k_state[:, -residual_len:], v_state[:, -residual_len:], layer_idx)
        k_state_past = k_state[:, :-residual_len]
        v_state_past = v_state[:, :-residual_len]
    else:
        k_state_past, v_state_past = k_state, v_state

    kvcache_pack_int(k_state_past, k_pack, k_params, v_state_past, v_pack, v_params,
                     None, cu_seqlens_k, seqlen_k_pack, quant_mode, group_size, num_bits)
    past_key_value.update_pack(k_pack, k_params, v_pack, v_params, layer_idx)

    k_pack_new = torch.empty((batch_size, int(residual_block_size // pack_nums), nheads_k, k_pack.size(-1)),
                             dtype=torch.uint16, device=device)
    k_params_new = torch.empty((batch_size, int(residual_block_size // group_size), nheads_k, k_params.size(-1)),
                               dtype=torch.float32, device=device)
    v_pack_new = torch.empty((batch_size, residual_block_size, nheads_k, v_pack.size(-1)),
                             dtype=torch.uint16, device=device)
    v_params_new = torch.empty((batch_size, v_params.size(1), nheads_k, residual_block_size),
                               dtype=torch.float32, device=device)

    errs = []
    for _ in range(rounds):
        k_new = torch.randn(batch_size, 1, nheads_k, d, device=device, dtype=dtype)
        v_new = torch.randn(batch_size, 1, nheads_k, d, device=device, dtype=dtype)

        k_pack, k_params, v_pack, v_params = past_key_value.update_pack(None, None, None, None, layer_idx)
        seqlens_k = torch.full((batch_size,), v_pack.shape[1], dtype=torch.int32, device=device)

        k_residual = torch.zeros((batch_size, residual_block_size, nheads_k, d), device=device, dtype=dtype)
        v_residual = torch.zeros((batch_size, residual_block_size, nheads_k, d), device=device, dtype=dtype)
        k_res_cache, v_res_cache = past_key_value.update_residual(k_new, v_new, layer_idx)
        cur_residual_len = k_res_cache.shape[1]
        k_residual[:, :cur_residual_len] = k_res_cache
        v_residual[:, :cur_residual_len] = v_res_cache

        out, k_pack_new, k_params_new, v_pack_new, v_params_new = fwd_kvcache_int(
            q, k_pack, k_params, v_pack, v_params,
            k_residual, v_residual, seqlens_k,
            k_pack_new, k_params_new, v_pack_new, v_params_new,
            None, sm_scale, quant_mode, group_size, residual_block_size,
            cur_residual_len, num_bits,
        )

        if cur_residual_len == residual_block_size:
            past_key_value.update_pack(k_pack_new, k_params_new, v_pack_new, v_params_new, layer_idx)
            past_key_value.clear_residual(layer_idx)

        k_state = torch.cat([k_state, k_new], dim=1)
        v_state = torch.cat([v_state, v_new], dim=1)
        errs.append((out - attention_ref(q, k_state, v_state)).abs().mean().item())

    return errs


if __name__ == "__main__":
    for num_bits in (4, 2):
        residual_block_size = 128 if num_bits == 4 else 256
        for group_size in (128, 32):
            try:
                errs = run(num_bits, group_size, residual_block_size)
                print(f"bits={num_bits} group={group_size} res={residual_block_size} "
                      f"err first={errs[0]:.5f} last={errs[-1]:.5f} "
                      f"min={min(errs):.5f} max={max(errs):.5f}")
            except Exception as exc:  # noqa: BLE001
                print(f"bits={num_bits} group={group_size} res={residual_block_size} FAILED: {exc}")
            sys.stdout.flush()
