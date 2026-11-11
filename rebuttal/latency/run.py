"""Complete online prefill TTFT, with a separate non-synchronizing stage audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.common import load_model, seed_everything
from benchmarks.Efficiency.run_efficiency import (
    _free, _peak_gb, _time_generate, build_inputs_from_dataset,
)
from rebuttal.latency.instrument import Recorder

SOURCE_FILES = [
    'picokv/picokv_standalone/online.py', 'picokv/picokv_standalone/prepare.py',
    'picokv/picokv_standalone/transformers.py', 'picokv/picokv_standalone/config.py',
    'picokv/picokv_standalone/constants.py', 'benchmarks/common.py',
    'benchmarks/Efficiency/run_efficiency.py',
    'picokv/picokv_prefill_cuda_extension/picokv_prefill_cuda.py',
    'rebuttal/latency/instrument.py', 'rebuttal/latency/run.py',
]

def digest_sources():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/home/zxh/zxh/models/Llama-3.1-8B-Instruct')
    parser.add_argument('--method', choices=['online', 'dense'], default='online')
    parser.add_argument('--lengths', default='8192,32768,65536')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--stage-repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.warmup >= 1 and args.repeats >= 3 and args.stage_repeats >= 1
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    sha = digest_sources()
    result = {'status': 'running', 'args': vars(args), 'source_sha256': sha,
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'python': platform.python_version(),
              'environment': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS','MAX_JOBS','TORCH_CUDA_ARCH_LIST']},
              'scope': 'Original _time_generate(max_new_tokens=1), synchronized wall clock. Includes full model prefill and online cache fitting/GPC/EAS/packing/gather across every layer, LM head and first-token selection. Excludes model loading, JIT compilation after warmup, tokenization and host cleanup. No decoder forward.',
              'results': {}}
    def save():
        output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    seed_everything(42)
    try:
        model, tokenizer, factory = load_model(args.model, method='picokv' if args.method=='online' else 'flash_attention_2',
            k_bits=320, v_bits=320, sink_size=64, local_size=256, eas=.01,
            zerorope=20, exp_bits=4, use_pruning=False, offline_proj_path=None)
    except Exception as e:
        result.update(status='failed_before_measurement', error=repr(e))
        save()
        raise
    layers = model.config.num_hidden_layers
    held = []
    def make_cache():
        cache = factory()
        assert all(getattr(cache, k) is None for k in ['key_encoders','value_encoders','key_decoders','value_decoders'])
        assert cache.config.recon_ablation is False
        held.append(cache)
        return cache
    selected_factory = make_cache if args.method=='online' else None
    if selected_factory:
        check = make_cache()
        result['online_config'] = check.config.prepare_kwargs()
        held.clear(); del check
    def one(input_ids, instrument=False):
        held.clear(); _free()
        torch.cuda.reset_peak_memory_stats()
        rec = Recorder().install() if instrument else None
        try:
            elapsed = _time_generate(model, input_ids, 1, selected_factory)
            peak = _peak_gb()
            record = {'ttft_wall_ms': elapsed, 'peak_allocated_gib': peak}
            if selected_factory:
                assert len(held)==1 and len(held[0].layers)==layers
                cache = held[0]
                times = cache.prepare_timings_ms()
                assert len(times)==layers
                record['cache_prepare_cuda_event_ms_all_layers'] = sum(times.values())
                record['cache_prepare_per_layer_ms'] = times
                assert all(c.key_bits==320 and c.value_bits==320 for c in cache.layers.values())
            if rec:
                record['stages'] = rec.summarize()
                assert len(record['stages']['layer_shapes'])==layers
                counts={}
                for row in record['stages']['intervals']:
                    k=row['category']; counts[k]=counts.get(k,0)+1
                assert counts['packing']==2*layers
                assert counts['eas_selection']==layers
                assert counts['gather_compact_packed_cache']==layers
                assert counts['gpc_encoding_residual']==(2*10+1)*2*layers
                record['stage_interval_counts']=counts
            return record
        finally:
            if rec: rec.restore()
            held.clear(); _free()
    try:
        for length in [int(n) for n in args.lengths.split(',')]:
            ids = build_inputs_from_dataset(str(ROOT/'benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl'),tokenizer,args.batch_size,length,model.device)
            item={'input_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                  'warmup': [], 'uninstrumented': [], 'stage_diagnostics': []}
            result['results'][str(length)]=item; save()
            for i in range(args.warmup):
                r=one(ids);item['warmup'].append(r);save()
                print(args.method,length,'warmup',i,r['ttft_wall_ms'],flush=True)
            for i in range(args.repeats):
                r=one(ids);item['uninstrumented'].append(r);save()
                print(args.method,length,'formal',i,r['ttft_wall_ms'],flush=True)
            item['ttft_median_ms']=statistics.median(r['ttft_wall_ms'] for r in item['uninstrumented'])
            item['ttft_min_max_ms']=[min(r['ttft_wall_ms'] for r in item['uninstrumented']),max(r['ttft_wall_ms'] for r in item['uninstrumented'])]
            item['peak_allocated_gib']=max(r['peak_allocated_gib'] for r in item['uninstrumented'])
            if selected_factory:
                item['cache_prepare_median_ms_all_layers']=statistics.median(r['cache_prepare_cuda_event_ms_all_layers'] for r in item['uninstrumented'])
                for i in range(args.stage_repeats):
                    r=one(ids,True);item['stage_diagnostics'].append(r);save()
                    print(args.method,length,'stage',i,r['ttft_wall_ms'],flush=True)
                names=item['stage_diagnostics'][0]['stages']['stage_sum_across_streams_ms']
                item['stage_sum_median_ms_nonadditive']={k:statistics.median(r['stages']['stage_sum_across_streams_ms'][k] for r in item['stage_diagnostics']) for k in names}
                item['stage_union_median_ms_nonadditive']={k:statistics.median(r['stages']['stage_union_across_streams_ms'][k] for r in item['stage_diagnostics']) for k in names}
                item['diagnostic_ttft_median_ms']=statistics.median(r['ttft_wall_ms'] for r in item['stage_diagnostics'])
            save();del ids;_free()
        result['source_unchanged']=sha==digest_sources();assert result['source_unchanged']
        result['status']='completed'
    except Exception as e:
        result['status']='failed';result['error']=repr(e);raise
    finally:
        result['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()

if __name__=='__main__':main()
