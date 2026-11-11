"""Check against the vendored BinaryPC source, not a rewritten fit reference."""

import ast
import importlib
from pathlib import Path

import pytest
import torch

from rebuttal.BPC.encoding import binarypc_encoding, encode_with_projection, fit_binary_projection, quantize_binarypc


def _original_functions():
    path = Path(__file__).resolve().parents[2] / "src/BPC/bpc/bpc/bpc_pytorch.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in (
        "find_binary_projection", "binary_project_pytorch",
    )]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    # Expose the source encoder's unpacked result before its 64-bit-only packer.
    replay = next(node for node in functions if node.name == "binary_project_pytorch")
    replay.name = "reference_unpacked"
    stop = next(i for i, node in enumerate(replay.body) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "binary_bits" for t in node.targets))
    replay.body = replay.body[:stop] + [ast.Return(value=ast.Name(id="hashcode_unpacked", ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[replay], type_ignores=[])), str(path), "exec"), namespace)
    return namespace


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bits,length", [(64, 17), (320, 65)])
def test_matches_source_fit_and_replay(device, bits, length):
    source = _original_functions()
    target = torch.randn(1, 2, length, 128, generator=torch.Generator().manual_seed(91)).to(device=device, dtype=torch.bfloat16)
    original = target.clone()
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(19)
        rng_before = torch.cuda.get_rng_state() if device == 'cuda' else torch.random.get_rng_state()
        projection = fit_binary_projection(target, bits)
        rng_after = torch.cuda.get_rng_state() if device == 'cuda' else torch.random.get_rng_state()
        assert not torch.equal(rng_before, rng_after)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(19)
        _, expected_projection = source["find_binary_projection"](target, bits)
    torch.testing.assert_close(projection, expected_projection, atol=0, rtol=0)
    replay = source["reference_unpacked"](target, projection)
    expected_codes = torch.where(replay < 0, -1, 1).to(torch.int8)
    codes = encode_with_projection(target, projection)
    torch.testing.assert_close(codes, expected_codes, atol=0, rtol=0)
    if bits == 64:
        weights = torch.ones(64, dtype=torch.int64, device=device) << torch.arange(64, device=device)
        packed = ((codes < 0).long() * weights).sum(-1)
        torch.testing.assert_close(packed, source["binary_project_pytorch"](target, projection), atol=0, rtol=0)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(19)
        result = quantize_binarypc(target, bits, seed=999)
    torch.testing.assert_close(result.decoder, expected_projection, atol=0, rtol=0)
    torch.testing.assert_close(result.codes, expected_codes, atol=0, rtol=0)
    expected_error = torch.norm(target - torch.einsum("bhln,bhnd->bhld", codes.to(target.dtype), projection), p=2, dim=-1)
    torch.testing.assert_close(result.err, expected_error, atol=0, rtol=0)
    assert torch.isfinite(result.decoder).all() and torch.isfinite(result.err).all()
    torch.testing.assert_close(target, original, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_input_and_sampling_contract(device):
    target = torch.zeros(1, 1, 65, 128, device=device, dtype=torch.bfloat16)
    first = quantize_binarypc(target, 320, direction_sample_size=8, seed=0)
    second = quantize_binarypc(target, 320, direction_sample_size=None, seed=0)
    assert (first.codes == 1).all()
    assert (first.decoder == 0).all() and (first.err == 0).all()
    torch.testing.assert_close(first.codes, second.codes, atol=0, rtol=0)
    with pytest.raises(ValueError, match="offline"):
        quantize_binarypc(target, 320, encoders=first.encoders)


def test_context_restores_on_failure():
    prepare = importlib.import_module("picokv.picokv_standalone.prepare")
    before = prepare.quantize_block32_online
    before_mask = prepare._apply_zerorope_
    with pytest.raises(RuntimeError, match="deliberate"):
        with binarypc_encoding():
            assert prepare.quantize_block32_online is quantize_binarypc
            raise RuntimeError("deliberate")
    assert prepare.quantize_block32_online is before
    assert prepare._apply_zerorope_ is before_mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_prepare_packing_decode_and_footprint():
    from benchmarks.Efficiency.run_efficiency import _kv_cache_bytes
    from picokv.picokv_standalone import decode_attention, prepare_kv_cache
    from picokv.picokv_standalone.transformers import StandaloneLlamaCache
    from rebuttal.reconstruct_attention import decode_attention as reconstructed_decode, unpack_codes

    prepare = importlib.import_module("picokv.picokv_standalone.prepare")
    original = prepare.quantize_block32_online
    calls = []

    def record(*args, **kwargs):
        result = quantize_binarypc(*args, **kwargs)
        calls.append((kwargs["seed"], result))
        return result

    torch.manual_seed(23)
    key = torch.randn(1, 1, 257, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    with binarypc_encoding():
        prepare.quantize_block32_online = record
        cache = prepare_kv_cache(key, value, key_bits=320, value_bits=320,
                                 sink_size=4, local_size=16, eas=0.01, zerorope=20, seed=3)
    assert prepare.quantize_block32_online is original
    assert sorted(seed for seed, _ in calls) == [3, 1012]
    blocks = dict(calls)
    eas, kept = prepare._select_eas_and_kept_indices(blocks[3].err, blocks[1012].err, 2)
    for is_key, block, packed in [(True, blocks[3], cache.packed_key), (False, blocks[1012], cache.packed_value)]:
        indices = kept.long().unsqueeze(-1).expand(-1, -1, -1, 320)
        expected_codes = block.codes.gather(-2, indices)
        actual_codes = unpack_codes(packed, length=cache.global_len, is_key=is_key)
        torch.testing.assert_close(actual_codes, expected_codes.to(actual_codes.dtype), rtol=0, atol=0)
    assert cache.global_len == 233 and cache.dense_len == 24
    query = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    # Small queries isolate kernel equivalence from large compression errors.
    query *= 0.1
    reference = reconstructed_decode(query, cache)
    packed = decode_attention(query, cache)
    assert torch.isfinite(packed).all()
    # Packed uses int8 queries / 4-bit weights; agreement is approximate.
    torch.testing.assert_close(packed.float(), reference.float(), atol=0.02, rtol=0.15)
    wrapped = StandaloneLlamaCache()
    wrapped.layers[0] = cache
    expected_bytes = sum(t.untyped_storage().nbytes() for t in [
        cache.dense_key, cache.dense_value, cache.key_projection, cache.key_projection_t,
        cache.value_projection, cache.packed_key, cache.packed_value,
    ])
    assert _kv_cache_bytes(wrapped) == expected_bytes  # excludes decode workspace
