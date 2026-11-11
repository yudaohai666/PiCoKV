"""Run full dense/GPC/BinaryPC LongBench with two independent GPU workers per method."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
UV_ENV = '/data/zxh/PiCoKV/.venv'


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 for all samples')
    parser.add_argument('--dense-gpus', default='0,1')
    parser.add_argument('--gpc-gpus', default='2,3')
    parser.add_argument('--binarypc-gpus', default='4,5')
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError('limit must be nonnegative')
    reference = read(ROOT / 'rebuttal/BPC/results/uv_retest/environment.json')
    if Path(sys.prefix).resolve() != Path(UV_ENV).resolve():
        raise RuntimeError('Run using the verified uv environment')
    for name, expected in reference['versions'].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'Environment version changed: {name}')
    for path, expected in reference['source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Experiment source changed: {path}')
    base = (ROOT / args.output).resolve()
    base.mkdir(parents=True, exist_ok=False)
    config = ROOT / 'benchmarks/Longbench/config'
    tasks = read(config / 'longbench_eval_subdatasets.json')['datasets']
    maxgen = read(config / 'dataset2maxlen.json')
    source_data = {}
    dataset_hashes = {}
    for task in tasks:
        path = Path('/data/zxh/datasets/longbench/data') / f'{task}.jsonl'
        samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        source_data[task] = samples[:args.limit] if args.limit else samples
        dataset_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.update({
        'UV_PROJECT_ENVIRONMENT': UV_ENV,
        'UV_CACHE_DIR': '/data/zxh/.cache/uv-rebuttal',
        'CUDA_HOME': '/usr/local/cuda-12.4',
        'PATH': '/usr/local/cuda-12.4/bin:' + env['PATH'],
        'PICOKV_DISABLE_PREBUILT': '1',
        'TORCH_EXTENSIONS_DIR': '/data/zxh/.cache/picokv_bpc_uv_extensions',
        'TRITON_CACHE_DIR': '/data/zxh/.cache/picokv_bpc_uv_triton',
        'TORCH_CUDA_ARCH_LIST': '8.6', 'MAX_JOBS': '2',
        'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,garbage_collection_threshold:0.8',
    })
    uv = ['uv', 'run', '--project', str(ROOT), '--no-sync', 'python', '-u']
    source_paths = list(read(ROOT / 'rebuttal/BPC/results/uv_retest/environment.json')['source_sha256'])
    source_paths += ['benchmarks/Longbench/pred.py', 'benchmarks/Longbench/eval.py',
                     'benchmarks/Longbench/metrics.py', str(Path(__file__).relative_to(ROOT))]
    source_paths += [str(p.relative_to(ROOT)) for p in config.glob('*.json')]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    gpu_groups = {'dense': args.dense_gpus.split(','), 'gpc': args.gpc_gpus.split(','), 'binarypc': args.binarypc_gpus.split(',')}
    all_gpus = sum(gpu_groups.values(), [])
    if len(set(all_gpus)) != len(all_gpus):
        raise ValueError('Each worker needs a distinct GPU')
    manifest = {'status': 'running', 'started_at': now(), 'args': vars(args),
                'launcher_pid': os.getpid(), 'python': sys.executable,
                'versions': reference['versions'], 'dataset_sha256': dataset_hashes,
                'uv_environment': UV_ENV, 'environment': '../environment.json',
                'dense_backend': 'flash_attention_2, BF16, full DynamicCache',
                'scheduling': 'two independent single-GPU workers per method; no model sharding',
                'source_sha256': hashes, 'gpu_groups': gpu_groups,
                'expected_samples_per_encoding': sum(map(len, source_data.values())),
                'expected_counts': {t: len(source_data[t]) for t in tasks},
                'jobs': {}}
    lock = threading.Lock()

    def save():
        tmp = base / 'manifest.tmp'
        tmp.write_text(json.dumps(manifest, indent=2) + '\n')
        tmp.replace(base / 'manifest.json')

    save()
    queues = {}
    # Long tasks first; separate dynamic queues for the two encodings.
    order = sorted(tasks, key=lambda t: sum(x['length'] for x in source_data[t])
                   + len(source_data[t]) * maxgen[t] * 150, reverse=True)
    for coding in gpu_groups:
        queues[coding] = Queue()
        for task in order:
            queues[coding].put(task)

    def worker(coding, gpu):
        while True:
            try:
                task = queues[coding].get_nowait()
            except Empty:
                return
            output = base / 'shards' / coding / task
            log = base / 'logs' / f'{coding}_{task}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = uv + ['rebuttal/BPC/run.py', '--experiment', 'longbench', '--encoding', coding,
                        '--model', '/home/zxh/zxh/models/Llama-3.1-8B-Instruct', '--method', 'picokv',
                        '--k-bits', '320', '--v-bits', '320', '--sink-size', '64', '--local-size', '256',
                        '--exp-bits', '4', '--eas', '0.01', '--zerorope', '20', '--no-pruning',
                        '--seed', '42', '--data-dir', '/data/zxh/datasets/longbench/data',
                        '--datasets', task, '--output-path', str(output)]
            if coding == 'dense':
                cmd = uv + ['benchmarks/Longbench/pred.py',
                            '--model', '/home/zxh/zxh/models/Llama-3.1-8B-Instruct',
                            '--method', 'flash_attention_2', '--seed', '42',
                            '--data-dir', '/data/zxh/datasets/longbench/data',
                            '--datasets', task, '--output-path', str(output)]
            if args.limit:
                cmd += ['--limit', str(args.limit)]
            key = f'{coding}/{task}'
            with lock:
                manifest['jobs'][key] = {'status': 'running', 'gpu': gpu, 'command': cmd, 'started_at': now()}
                save()
            print(now(), 'START', key, 'GPU', gpu, flush=True)
            try:
                with log.open('w') as handle:
                    result = subprocess.run(cmd, cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': gpu},
                                            stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'exit={result.returncode}; see {log}')
                if coding != 'dense':
                    assert read(output / 'coding_run.json')['status'] == 'completed'
                predictions = [json.loads(line) for line in (output / f'{task}.jsonl').read_text().splitlines()]
                assert len(predictions) == len(source_data[task])
                for prediction, sample in zip(predictions, source_data[task]):
                    assert isinstance(prediction['pred'], str)
                    for field in ['answers', 'all_classes', 'length']:
                        assert prediction[field] == sample[field], field
                update = {'status': 'completed', 'samples': len(predictions), 'sample_order_verified': True}
            except Exception as error:
                update = {'status': 'failed', 'error': repr(error)}
            with lock:
                manifest['jobs'][key].update(update, finished_at=now())
                save()
            print(now(), update['status'].upper(), key, flush=True)

    with ThreadPoolExecutor(max_workers=len(all_gpus)) as pool:
        futures = [pool.submit(worker, coding, gpu) for coding, gpus in gpu_groups.items() for gpu in gpus]
        for future in futures:
            future.result()
    if any(job['status'] != 'completed' for job in manifest['jobs'].values()):
        manifest.update(status='failed', finished_at=now())
        save()
        raise RuntimeError('Some LongBench jobs failed; inspect manifest and logs')
    assert len(manifest['jobs']) == len(tasks) * 3
    for path, expected in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for coding in gpu_groups:
        merged = base / coding
        merged.mkdir()
        for task in tasks:
            shutil.copy2(base / 'shards' / coding / task / f'{task}.jsonl', merged / f'{task}.jsonl')
        with (base / f'eval_{coding}.log').open('w') as handle:
            subprocess.run(uv + ['benchmarks/Longbench/eval.py', '--path', str(merged) + '/', '--eval_avg'],
                           cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    dense, gpc, bpc = (read(base / name / 'result.json') for name in ['dense', 'gpc', 'binarypc'])
    comparison = {'scope': f"16 tasks, {'all' if not args.limit else args.limit} samples per task",
                  'samples_per_encoding': manifest['expected_samples_per_encoding'],
                  'dense': dense, 'gpc': gpc, 'binarypc': bpc,
                  'delta_binarypc_minus_gpc': {t: round(bpc[t] - gpc[t], 2) for t in tasks + ['avg']},
                  'sample_order_verified': True, 'source_hashes_unchanged': True}
    (base / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    manifest.update(status='completed', finished_at=now(), sample_order_verified=True, source_hashes_unchanged=True)
    save()
    print(now(), 'ALL COMPLETED', json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
