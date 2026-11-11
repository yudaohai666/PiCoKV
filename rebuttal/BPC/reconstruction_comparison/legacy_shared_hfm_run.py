"""Compare original sequential BinaryPC and GPC CD on identical real layer-0 KV.

Each algorithm learns its own codes/decoder for 320 bits. Curves use prefixes
32,64,...,320, always normalized by the same original input, before EAS.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.common import seed_everything
from picokv.picokv_standalone import StandaloneConfig
from picokv.picokv_standalone import transformers as adapter
from rebuttal.BPC.reconstruction_comparison.legacy_shared_hfm_encoding import fit_binary_projection, encode_with_projection
from rebuttal.BPC.single_layer_prefill import load_inputs_and_layer


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def capture_gpc(states):
    prepare = importlib.import_module('picokv.picokv_standalone.prepare')
    original = prepare.quantize_block32_online

    def quantize(target, bits, **kwargs):
        assert bits == 320 and kwargs['seed'] in (0, 1009)
        frozen = target.cpu()
        output = original(target, bits, **kwargs)
        assert torch.equal(frozen, target.cpu())
        side = 'K' if kwargs['seed'] == 0 else 'V'
        states[side] = {'input': frozen, 'seed': kwargs['seed'],
                        'gpc': {'codes': output.codes.cpu(), 'decoder': output.decoder.cpu()}}
        return output

    prepare.quantize_block32_online = quantize
    try:
        yield
    finally:
        prepare.quantize_block32_online = original


def check_original_bpc(target):
    """Bitwise parity with source fit/replay on a real small input, no pytest."""
    path = ROOT / 'src/BPC/bpc/bpc/bpc_pytorch.py'
    tree = ast.parse(path.read_text())
    fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_binary_projection')
    replay = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'binary_project_pytorch')
    # Stop immediately before the source's uint64-only packer, preserving all
    # replay arithmetic. The source packer maps exact zero to the +1 bit value.
    stop = next(i for i, n in enumerate(replay.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'binary_bits' for t in n.targets))
    replay.body = replay.body[:stop] + [ast.Return(ast.Name('hashcode_unpacked', ast.Load()))]
    namespace = {'torch': torch}
    module = ast.fix_missing_locations(ast.Module(body=[fit, replay], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    sample = target[:, :2, :257].contiguous()
    actual = fit_binary_projection(sample, 32, seed=19)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(19)
        _, expected = namespace['find_binary_projection'](sample, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    signs = namespace['binary_project_pytorch'](sample, expected)
    expected_codes = torch.where(signs < 0, -1, 1).to(torch.int8)
    actual_codes = encode_with_projection(sample, actual)
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    return {'source_fit_bitwise_equal': True, 'source_replay_bitwise_equal': True,
            'shape': list(sample.shape), 'bits': 32, 'seed': 19}


def measure_curve(target, codes, decoder):
    assert target.device.type == codes.device.type == decoder.device.type == 'cpu'
    assert target.dtype == decoder.dtype == torch.bfloat16 and codes.dtype == torch.int8
    assert ((codes == -1) | (codes == 1)).all() and torch.isfinite(decoder).all()
    x = target.double()
    reconstruction = torch.zeros_like(x)
    head_energy = x.square().sum(dim=(-2, -1))
    total_energy = head_energy.sum().item()
    assert total_energy > 0
    result = []
    for bits in range(32, 321, 32):
        reconstruction.add_(codes[..., bits-32:bits].double() @ decoder[..., bits-32:bits, :].double())
        error = x - reconstruction
        head_sse = error.square().sum(dim=(-2, -1))
        sse = head_sse.sum().item()
        result.append({'bits': bits, 'relative_frobenius': (sse / total_energy)**.5,
                       'mse': sse / x.numel(), 'sse': sse, 'input_energy': total_energy,
                       'per_head_relative_frobenius': (head_sse / head_energy).sqrt().flatten().tolist(),
                       'per_head_sse': head_sse.flatten().tolist()})
    # Confirm that prefix summation measures the full stored C D representation.
    direct = codes.double() @ decoder.double()
    torch.testing.assert_close(reconstruction, direct, rtol=1e-12, atol=1e-12)
    return result




@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/data/zxh/models/Llama-3.1-8B-Instruct')
    parser.add_argument('--dataset', default=str(ROOT / 'benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl'))
    parser.add_argument('--length', type=int, default=16384)
    parser.add_argument('--output', default=str(Path(__file__).parent / 'results.json'))
    parser.add_argument('--gpu-memory-fraction', type=float, default=.12)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one GPU')
    if args.length != 16384:
        raise ValueError('This report is configured for the fixed 16K experiment')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracked = ['picokv/picokv_standalone/online.py', 'picokv/picokv_standalone/constants.py',
               'picokv/picokv_standalone/prepare.py', 'picokv/picokv_standalone/transformers.py',
               'rebuttal/BPC/encoding.py', 'rebuttal/BPC/single_layer_prefill.py',
               'src/BPC/bpc/bpc/bpc_pytorch.py', str(Path(__file__).relative_to(ROOT))]
    hashes = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked}
    result = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'args': vars(args),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': hashes(), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'python': sys.version,
              'uv_environment': os.environ.get('UV_PROJECT_ENVIRONMENT'), 'curves': {}, 'inputs': {},
              'scope': 'own full 320-bit fit+encoding for each method, cumulative prefixes 32:32:320, identical prepared KV input before EAS',
              'measurement': 'CPU FP64 reconstruction from stored +/-1 int8 codes and BF16 decoder; normalize every prefix by original input norm',
              'bpc_num_iters': 4, 'gpc_group_bits': 32, 'gpc_ridge': .001, 'gpc_direction_sample_size': 8192}
    def save():
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    save()
    try:
        seed_everything(42)
        attention, hidden, positions, meta = load_inputs_and_layer(Path(args.model), args.dataset, args.length)
        result['model_inputs'] = meta
        config = StandaloneConfig(key_bits=320, value_bits=320, eas=.01, use_pruning=False)
        result['prepare_config'] = config.prepare_kwargs()
        states = {}
        with capture_gpc(states):
            cache = adapter.StandaloneLlamaCache(config)
            output, _ = attention(hidden_states=hidden, position_embeddings=positions,
                                  attention_mask=None, past_key_value=cache)
        torch.cuda.synchronize()
        assert set(states) == {'K', 'V'} and torch.isfinite(output).all()
        del attention, hidden, positions, cache, output
        gc.collect()
        torch.cuda.empty_cache()
        print('Captured actual GPC inputs/codes/decoder for K and V.', flush=True)
        for side in ('K', 'V'):
            state = states[side]
            target = state['input'].cuda()
            result.setdefault('source_parity', {})[side] = check_original_bpc(target)
            decoder = fit_binary_projection(target, 320, seed=state['seed'])
            codes = encode_with_projection(target, decoder)
            assert torch.equal(target.cpu(), state['input'])
            state['bpc'] = {'codes': codes.cpu(), 'decoder': decoder.cpu()}
            del target, codes, decoder
            torch.cuda.empty_cache()
            result['inputs'][side] = {'shape': list(state['input'].shape), 'dtype': str(state['input'].dtype),
                                      'sha256': digest(state['input']), 'seed': state['seed'], 'same_input_for_both_methods': True}
            result['curves'][side] = {}
            for method in ('bpc', 'gpc'):
                encoded = state[method]
                result.setdefault('artifact_sha256', {}).setdefault(side, {})[method] = {
                    'codes': digest(encoded['codes']), 'decoder': digest(encoded['decoder'])}
                result['curves'][side][method] = measure_curve(state['input'], **encoded)
                save()
                print(side, method, json.dumps(result['curves'][side][method][-1]), flush=True)
        result['source_unchanged'] = hashes() == result['source_sha256']
        assert result['source_unchanged']
        result['status'] = 'completed'
        result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3

    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished_at'] = datetime.now().astimezone().isoformat()
        save()


if __name__ == '__main__':
    main()
