"""Check literal source copies, RNG state and 320-bit parity without pytest."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from rebuttal.BPC import source_encoding
from rebuttal.BPC.encoding import fit_binary_projection, encode_with_projection, quantize_binarypc


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source_path = ROOT / 'src/BPC/bpc/bpc/bpc_pytorch.py'
    copied_path = Path(source_encoding.__file__)
    source, copied = source_path.read_text(), copied_path.read_text()
    names = ['find_binary_projection', 'binary_project_pytorch']
    get = lambda text: {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    original_nodes, copied_nodes = get(source), get(copied)
    for name in names:
        assert ast.get_source_segment(source, original_nodes[name]) == ast.get_source_segment(copied, copied_nodes[name])
    replay = original_nodes['binary_project_pytorch']
    stop = next(i for i, n in enumerate(replay.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'binary_bits' for t in n.targets))
    expected_prefix = [ast.dump(n) for n in replay.body[:stop]]
    assert expected_prefix == [ast.dump(n) for n in copied_nodes['binary_project_unpacked'].body[:-1]]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[original_nodes[n] for n in names], type_ignores=[]), str(source_path), 'exec'), namespace)
    replay.name = 'unpacked_reference'
    replay.body = replay.body[:stop] + [ast.Return(ast.Name('hashcode_unpacked', ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[replay], type_ignores=[])), str(source_path), 'exec'), namespace)
    report = {'status': 'running', 'device': args.device, 'verbatim_functions': names,
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'copied_file_sha256': hashlib.sha256(copied.encode()).hexdigest(),
              'replay_before_packing_ast_identical': True, 'samples': []}
    devices = [torch.cuda.current_device()] if args.device == 'cuda' else []
    state = torch.cuda.get_rng_state if args.device == 'cuda' else torch.random.get_rng_state
    for bits, length in [(64, 17), (320, 65)]:
        target = torch.randn(1, 2, length, 128, generator=torch.Generator().manual_seed(91)).to(device=args.device, dtype=torch.bfloat16)
        original_target = target.clone()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(19)
            temporary_codes, expected_decoder = namespace['find_binary_projection'](target, bits)
            expected_rng = state()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(19)
            actual_temporary, actual_decoder = source_encoding.find_binary_projection(target, bits)
            torch.testing.assert_close(actual_temporary, temporary_codes, atol=0, rtol=0)
            torch.testing.assert_close(actual_decoder, expected_decoder, atol=0, rtol=0)
            assert torch.equal(state(), expected_rng)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(19)
            before = state()
            decoder = fit_binary_projection(target, bits, seed=123456)
            assert not torch.equal(before, state()) and torch.equal(state(), expected_rng)
        torch.testing.assert_close(decoder, expected_decoder, atol=0, rtol=0)
        raw = namespace['unpacked_reference'](target, expected_decoder)
        codes = encode_with_projection(target, decoder)
        torch.testing.assert_close(codes, torch.where(raw < 0, -1, 1).to(torch.int8), atol=0, rtol=0)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(19)
            actual = quantize_binarypc(target, bits, seed=999999)
            assert torch.equal(state(), expected_rng)
        torch.testing.assert_close(actual.decoder, expected_decoder, atol=0, rtol=0)
        torch.testing.assert_close(actual.codes, codes, atol=0, rtol=0)
        torch.testing.assert_close(target, original_target, atol=0, rtol=0)
        assert torch.isfinite(actual.decoder).all() and torch.isfinite(actual.err).all()
        if bits == 64:
            weights = torch.ones(64, dtype=torch.int64, device=args.device) << torch.arange(64, device=args.device)
            packed = ((codes < 0).long() * weights).sum(-1)
            torch.testing.assert_close(packed, namespace['binary_project_pytorch'](target, decoder), atol=0, rtol=0)
        report['samples'].append({'bits': bits, 'length': length, 'fitting_codes_exact': True,
                                  'decoder_exact': True, 'replay_codes_exact': True,
                                  'global_rng_consumption_exact': True, 'adapter_exact': True})
    if args.device == 'cuda':
        from rebuttal.BPC.reconstruction_comparison.run import check_no_hfm_integration
        report['no_hfm_prepare_packing_decode'] = check_no_hfm_integration()
    report['status'] = 'completed'
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
