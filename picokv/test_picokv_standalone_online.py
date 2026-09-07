"""Regression checks for the production online-projection standalone path."""

from __future__ import annotations

import inspect

import torch

import picokv.picokv_standalone as picokv_standalone
from picokv.picokv_standalone import (
    PreparedKVCache,
    StandaloneConfig,
    decode_attention,
    prepare_kv_cache,
)
from picokv.picokv_standalone.online import (
    _cholesky_qr,
    _householder_qr,
    _solve_decoder_from_matrices,
    _strict_binary_sign_,
    quantize_block32_online,
)


REMOVED_PREPARE_OPTIONS = {
    "backend",
    "key_projection",
    "value_projection",
    "block_size",
    "power_iters",
    "decoder_ridge",
    "key_ridge",
    "value_ridge",
    "compute_dtype",
    "matmul_dtype",
    "error_mode",
    "refit_decoder",
    "keep_unpacked",
    "verbose",
}
REMOVED_DECODE_OPTIONS = {"backend", "new_key", "new_value", "exp_bits"}


def main() -> None:
    config = StandaloneConfig()
    if config.prepare_kwargs() != {
        "key_bits": 320,
        "value_bits": 320,
        "direction_sample_size": 8192,
        "sink_size": 64,
        "local_size": 256,
        "eas": 0.01,
        "zerorope": 20,
        "exp_bits": 4,
        "seed": 0,
        "use_fused_quant": True,
    }:
        raise AssertionError(f"unexpected prepare defaults: {config.prepare_kwargs()}")
    if config.decode_kwargs() != {
        "use_pruning": False,
        "prune_mode": 3,
        "log_eps": -10.0,
    }:
        raise AssertionError(f"unexpected decode defaults: {config.decode_kwargs()}")

    prepare_parameters = set(inspect.signature(prepare_kv_cache).parameters)
    decode_parameters = set(inspect.signature(decode_attention).parameters)
    if REMOVED_PREPARE_OPTIONS & prepare_parameters:
        raise AssertionError("a removed prepare option is public again")
    if REMOVED_DECODE_OPTIONS & decode_parameters:
        raise AssertionError("a removed decode option is public again")
    if hasattr(picokv_standalone, "prepare_kv_cache_async"):
        raise AssertionError("removed async prepare API is exported")

    scores = torch.tensor([-2.0, 0.0, 3.0])
    _strict_binary_sign_(scores)
    if scores.tolist() != [-1.0, 1.0, 1.0]:
        raise AssertionError(f"strict binary sign failed: {scores.tolist()}")

    duplicate_codes = torch.ones(2, 128, 32, dtype=torch.bfloat16)
    residual = torch.randn(2, 128, 8, dtype=torch.bfloat16)
    decoder = _solve_decoder_from_matrices(duplicate_codes, residual)
    if not bool(torch.isfinite(decoder).all()):
        raise AssertionError("regularized decoder solve returned non-finite values")

    zero_target = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)
    block = quantize_block32_online(
        zero_target,
        128,
        direction_sample_size=64,
    )
    if int((block.codes == 0).sum()) != 0:
        raise AssertionError("online quantizer emitted zero codes")
    if not bool(torch.isfinite(block.decoder).all()):
        raise AssertionError("online quantizer emitted a non-finite decoder")

    torch.manual_seed(23)
    qr_input = torch.randn(4, 32, 16, dtype=torch.float32)
    covariance = torch.randn(4, 32, 32, dtype=torch.float32)
    covariance = torch.matmul(covariance, covariance.transpose(-1, -2))
    qr_input = torch.matmul(covariance, qr_input)
    q_reference = _householder_qr(qr_input)
    q_cholesky = _cholesky_qr(qr_input, passes=2)
    identity = torch.eye(16).expand(4, -1, -1)
    torch.testing.assert_close(
        torch.matmul(q_cholesky.transpose(-1, -2), q_cholesky),
        identity,
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        torch.matmul(q_cholesky, q_cholesky.transpose(-1, -2)),
        torch.matmul(q_reference, q_reference.transpose(-1, -2)),
        rtol=2e-3,
        atol=2e-3,
    )

    if torch.cuda.is_available():
        # online.py builds err with linalg.vector_norm(dtype=fp32) instead of the
        # materialized residual.float().norm(dim=-1) it replaced, to avoid a
        # 510 MiB fp32 copy per stream at 131k.  The two are bit-identical today,
        # but that is a property of the reduction implementation rather than an
        # API guarantee, and err feeds a topk in prepare._select_eas_indices --
        # so pin it here instead of letting a torch upgrade quietly reselect EAS
        # tokens.  Compared through an int32 view because torch.equal returns
        # False for any tensor holding a NaN, including against itself.
        for probe_shape in ((8, 4096, 128), (4, 977, 127), (2, 33, 1)):
            probe = torch.randn(*probe_shape, device="cuda", dtype=torch.bfloat16)
            fused = torch.linalg.vector_norm(probe, dim=-1, dtype=torch.float32)
            materialized = probe.float().norm(dim=-1)
            if not torch.equal(fused.view(torch.int32), materialized.view(torch.int32)):
                raise AssertionError(f"err norm paths diverged at {probe_shape}")

        full_key = torch.randn(
            1,
            1,
            256,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        full_value = torch.randn_like(full_key)
        cache = prepare_kv_cache(
            full_key,
            full_value,
            key_bits=128,
            value_bits=128,
            direction_sample_size=64,
            sink_size=32,
            local_size=32,
            eas=0.01,
            zerorope=0,
        )
        query = torch.randn(
            1,
            4,
            1,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output_unpruned = decode_attention(query, cache, use_pruning=False)
        output_pruned = decode_attention(query, cache, use_pruning=True)
        torch.cuda.synchronize()
        if not isinstance(cache, PreparedKVCache):
            raise AssertionError(f"unexpected cache type: {type(cache)}")
        for name, output in {
            "unpruned": output_unpruned,
            "pruned": output_pruned,
        }.items():
            if tuple(output.shape) != (1, 4, 1, 128):
                raise AssertionError(f"unexpected {name} decode shape: {tuple(output.shape)}")
            if not bool(torch.isfinite(output).all()):
                raise AssertionError(f"{name} decode returned non-finite values")

    print("standalone online-projection regression tests passed")


if __name__ == "__main__":
    main()
