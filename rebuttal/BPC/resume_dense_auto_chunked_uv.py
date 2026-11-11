"""Resume failed dense LongBench tasks through the original benchmark on GPU0,1."""
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'rebuttal/BPC/results/uv_retest/longbench_full'
TASKS = ['narrativeqa', 'repobench-p']


def now():
    return datetime.now().astimezone().isoformat()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise RuntimeError('Expose GPU0,1 for automatic model placement')
    if Path(sys.prefix).resolve() != Path('/data/zxh/PiCoKV/.venv').resolve():
        raise RuntimeError('Use the verified uv environment')
    from benchmarks.Longbench import pred

    meta = {'status': 'running', 'started_at': now(), 'pid': os.getpid(),
            'python': sys.executable, 'cuda_visible_devices': '0,1',
            'device_map_requested': 'auto', 'mlp_chunk_size': 8192, 'tasks': {},
            'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                              for p in ['benchmarks/common.py', 'benchmarks/Longbench/pred.py',
                                        str(Path(__file__).relative_to(ROOT))]}}
    path = BASE / 'dense_auto_chunked_resume.json'
    if path.exists():
        raise RuntimeError('Recovery metadata already exists')

    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(meta, indent=2) + '\n')
        tmp.replace(path)

    save()
    original = pred.load_model

    def load_model(*args, **kwargs):
        result = original(*args, **kwargs)
        mapping = {k: str(v) for k, v in result[0].hf_device_map.items()}
        print('Actual hf_device_map:', json.dumps(mapping), flush=True)
        if not {'0', '1'}.issubset(set(mapping.values())):
            raise RuntimeError(f'Expected both GPUs: {mapping}')
        if any(v in ['cpu', 'disk'] for v in mapping.values()):
            raise RuntimeError('Unexpected CPU/disk offload')
        from transformers.models.llama.modeling_llama import LlamaMLP
        from picokv.picokv_standalone.transformers import _chunked_llama_mlp_forward
        for module in result[0].modules():
            if isinstance(module, LlamaMLP):
                forward = partial(_chunked_llama_mlp_forward, module, chunk_size=8192)
                if hasattr(module, '_old_forward'):
                    module._old_forward = forward
                else:
                    module.forward = forward
        meta['actual_hf_device_maps'].append(mapping)
        save()
        return result

    pred.load_model = load_model
    meta['actual_hf_device_maps'] = []
    try:
        for task in TASKS:
            output = BASE / 'shards/dense' / task
            predictions = output / f'{task}.jsonl'
            before = predictions.read_bytes()
            count = len(before.splitlines())
            meta['tasks'][task] = {'status': 'running', 'resume_from': count,
                                   'prefix_sha256': hashlib.sha256(before).hexdigest(),
                                   'started_at': now()}
            argv = ['--model', '/home/zxh/zxh/models/Llama-3.1-8B-Instruct',
                    '--method', 'flash_attention_2', '--seed', '42',
                    '--data-dir', '/data/zxh/datasets/longbench/data',
                    '--datasets', task, '--output-path', str(output)]
            meta['tasks'][task]['argv'] = argv
            save()
            print(now(), 'RESUME', task, 'from', count, flush=True)
            sys.argv = ['benchmarks/Longbench/pred.py', *argv]
            pred.main()
            after = predictions.read_bytes()
            assert after[:len(before)] == before, 'Existing predictions changed'
            rows = [json.loads(line) for line in after.splitlines()]
            source = [json.loads(line) for line in
                      (Path('/data/zxh/datasets/longbench/data') / f'{task}.jsonl').read_text().splitlines()
                      if line.strip()]
            assert len(rows) == len(source)
            for row, sample in zip(rows, source):
                assert isinstance(row['pred'], str)
                assert all(row[k] == sample[k] for k in ['answers', 'all_classes', 'length'])
            meta['tasks'][task].update(status='completed', finished_at=now(),
                                       samples=len(rows), sample_order_verified=True,
                                       existing_prefix_unchanged=True)
            save()
        tasks = json.loads((ROOT / 'benchmarks/Longbench/config/longbench_eval_subdatasets.json').read_text())['datasets']
        merged = BASE / 'dense'
        merged.mkdir(exist_ok=True)
        for task in tasks:
            p = BASE / 'shards/dense' / task / f'{task}.jsonl'
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            source = [json.loads(line) for line in
                      (Path('/data/zxh/datasets/longbench/data') / f'{task}.jsonl').read_text().splitlines()
                      if line.strip()]
            assert len(rows) == len(source)
            for row, sample in zip(rows, source):
                assert all(row[k] == sample[k] for k in ['answers', 'all_classes', 'length'])
            shutil.copy2(p, merged / p.name)
        with (BASE / 'eval_dense_auto_chunked_resume.log').open('w') as handle:
            subprocess.run([sys.executable, 'benchmarks/Longbench/eval.py', '--path',
                            str(merged) + '/', '--eval_avg'], cwd=ROOT,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        meta.update(status='completed', finished_at=now(), total_dense_samples=3750,
                    dense_scores=json.loads((merged / 'result.json').read_text()))
        save()
    except BaseException as error:
        meta.update(status='failed', error=repr(error), finished_at=now())
        save()
        raise


if __name__ == '__main__':
    main()
