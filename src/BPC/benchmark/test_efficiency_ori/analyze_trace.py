"""
Chrome Trace JSON analysis tool.

Compares decoding-step performance between baseline and bpc-offline traces
produced by torch.profiler.

Usage:
    python analyze_trace.py \
        --baseline /path/to/baseline_trace.json \
        --bpc /path/to/bpc_trace.json \
        --output analysis_results.json
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re


def load_trace(trace_path: str) -> dict:
    """Load a Chrome trace JSON file."""
    with open(trace_path, 'r') as f:
        return json.load(f)


def parse_trace_events(trace_data: dict) -> List[dict]:
    """Parse trace events."""
    if isinstance(trace_data, dict):
        return trace_data.get('traceEvents', [])
    elif isinstance(trace_data, list):
        return trace_data
    return []


def filter_decoding_events(events: List[dict],
                           include_cuda: bool = True,
                           include_cpu: bool = True) -> List[dict]:
    """
    Filter to decoding-related events.

    Typical decoding step covers single-token forward pass, attention,
    matmul (gemm/mm), and activations (gelu/silu).
    """
    filtered = []
    for event in events:
        if event.get('ph') not in ['X', 'B', 'E']:  # complete or begin/end
            continue

        cat = event.get('cat', '')

        if not include_cuda and 'cuda' in cat.lower():
            continue
        if not include_cpu and 'cpu' in cat.lower():
            continue
            
        filtered.append(event)
    
    return filtered


def extract_kernel_stats(events: List[dict]) -> Dict[str, dict]:
    """
    Extract per-kernel CUDA stats.

    Returns:
        Dict[kernel_name, {count, total_time_us, avg_time_us, min_time_us, max_time_us}]
    """
    kernel_stats = defaultdict(lambda: {
        'count': 0,
        'total_time_us': 0,
        'times': [],
    })

    for event in events:
        if event.get('ph') != 'X':  # complete events only
            continue

        name = event.get('name', '')
        dur = event.get('dur', 0)  # microseconds
        cat = event.get('cat', '')

        if 'kernel' in cat.lower() or 'cuda' in cat.lower():
            kernel_stats[name]['count'] += 1
            kernel_stats[name]['total_time_us'] += dur
            kernel_stats[name]['times'].append(dur)

    for name, stats in kernel_stats.items():
        times = stats['times']
        if times:
            stats['avg_time_us'] = stats['total_time_us'] / stats['count']
            stats['min_time_us'] = min(times)
            stats['max_time_us'] = max(times)
        del stats['times']

    return dict(kernel_stats)


def extract_operator_stats(events: List[dict]) -> Dict[str, dict]:
    """Extract per-operator CPU stats."""
    op_stats = defaultdict(lambda: {
        'count': 0,
        'total_time_us': 0,
        'times': [],
        'self_times': [],
    })
    
    for event in events:
        if event.get('ph') != 'X':
            continue
        
        name = event.get('name', '')
        dur = event.get('dur', 0)
        cat = event.get('cat', '')

        if 'cpu' in cat.lower() or cat in ['cpu_op', 'operator']:
            op_stats[name]['count'] += 1
            op_stats[name]['total_time_us'] += dur
            op_stats[name]['times'].append(dur)

            args = event.get('args', {})
            self_time = args.get('self_time', dur)
            op_stats[name]['self_times'].append(self_time)

    for name, stats in op_stats.items():
        times = stats['times']
        if times:
            stats['avg_time_us'] = stats['total_time_us'] / stats['count']
            stats['min_time_us'] = min(times)
            stats['max_time_us'] = max(times)
            stats['total_self_time_us'] = sum(stats['self_times'])
        del stats['times']
        del stats['self_times']
    
    return dict(op_stats)


def identify_attention_kernels(kernel_stats: Dict[str, dict]) -> Dict[str, dict]:
    """Identify attention-related kernels."""
    attention_patterns = [
        r'flash',
        r'attention',
        r'attn',
        r'softmax',
        r'fmha',  # fused multi-head attention
        r'sdpa',  # scaled dot product attention
    ]
    
    attention_kernels = {}
    for name, stats in kernel_stats.items():
        for pattern in attention_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                attention_kernels[name] = stats
                break
    
    return attention_kernels


def identify_gemm_kernels(kernel_stats: Dict[str, dict]) -> Dict[str, dict]:
    """Identify matmul/GEMM kernels."""
    gemm_patterns = [
        r'gemm',
        r'gemv',
        r'mm',
        r'matmul',
        r'cublas',
        r'cutlass',
    ]
    
    gemm_kernels = {}
    for name, stats in kernel_stats.items():
        for pattern in gemm_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                gemm_kernels[name] = stats
                break
    
    return gemm_kernels


def get_top_kernels(kernel_stats: Dict[str, dict],
                    top_k: int = 20,
                    sort_by: str = 'total_time_us') -> List[Tuple[str, dict]]:
    """Top-k kernels by time."""
    sorted_kernels = sorted(
        kernel_stats.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )
    return sorted_kernels[:top_k]


def compute_summary_stats(kernel_stats: Dict[str, dict]) -> dict:
    """Compute summary stats."""
    total_kernel_time = sum(s['total_time_us'] for s in kernel_stats.values())
    total_kernel_count = sum(s['count'] for s in kernel_stats.values())
    
    return {
        'total_kernel_time_us': total_kernel_time,
        'total_kernel_time_ms': total_kernel_time / 1000,
        'total_kernel_count': total_kernel_count,
        'unique_kernel_count': len(kernel_stats),
    }


def compare_traces(baseline_stats: dict, bpc_stats: dict) -> dict:
    """Compare two traces."""
    comparison = {
        'summary': {},
        'kernel_comparison': {},
        'attention_comparison': {},
        'gemm_comparison': {},
    }

    baseline_summary = baseline_stats['summary']
    bpc_summary = bpc_stats['summary']
    
    comparison['summary'] = {
        'baseline_total_time_ms': baseline_summary['total_kernel_time_ms'],
        'bpc_total_time_ms': bpc_summary['total_kernel_time_ms'],
        'speedup': baseline_summary['total_kernel_time_ms'] / max(bpc_summary['total_kernel_time_ms'], 1e-6),
        'time_reduction_percent': (1 - bpc_summary['total_kernel_time_ms'] / max(baseline_summary['total_kernel_time_ms'], 1e-6)) * 100,
    }

    baseline_attn = baseline_stats.get('attention_kernels', {})
    bpc_attn = bpc_stats.get('attention_kernels', {})
    
    baseline_attn_time = sum(s['total_time_us'] for s in baseline_attn.values())
    bpc_attn_time = sum(s['total_time_us'] for s in bpc_attn.values())
    
    comparison['attention_comparison'] = {
        'baseline_time_ms': baseline_attn_time / 1000,
        'bpc_time_ms': bpc_attn_time / 1000,
        'speedup': baseline_attn_time / max(bpc_attn_time, 1e-6),
        'time_reduction_percent': (1 - bpc_attn_time / max(baseline_attn_time, 1e-6)) * 100,
    }

    baseline_gemm = baseline_stats.get('gemm_kernels', {})
    bpc_gemm = bpc_stats.get('gemm_kernels', {})
    
    baseline_gemm_time = sum(s['total_time_us'] for s in baseline_gemm.values())
    bpc_gemm_time = sum(s['total_time_us'] for s in bpc_gemm.values())
    
    comparison['gemm_comparison'] = {
        'baseline_time_ms': baseline_gemm_time / 1000,
        'bpc_time_ms': bpc_gemm_time / 1000,
        'speedup': baseline_gemm_time / max(bpc_gemm_time, 1e-6),
        'time_reduction_percent': (1 - bpc_gemm_time / max(baseline_gemm_time, 1e-6)) * 100,
    }
    
    return comparison


def analyze_single_trace(trace_path: str) -> dict:
    """Analyze a single trace file."""
    print(f"Loading trace: {trace_path}")
    trace_data = load_trace(trace_path)
    events = parse_trace_events(trace_data)
    print(f"  Total events: {len(events)}")

    kernel_stats = extract_kernel_stats(events)
    print(f"  Unique kernels: {len(kernel_stats)}")

    op_stats = extract_operator_stats(events)
    print(f"  Unique operators: {len(op_stats)}")

    attention_kernels = identify_attention_kernels(kernel_stats)
    gemm_kernels = identify_gemm_kernels(kernel_stats)

    print(f"  Attention kernels: {len(attention_kernels)}")
    print(f"  GEMM kernels: {len(gemm_kernels)}")

    summary = compute_summary_stats(kernel_stats)
    top_kernels = get_top_kernels(kernel_stats, top_k=20)
    
    return {
        'trace_path': trace_path,
        'total_events': len(events),
        'summary': summary,
        'kernel_stats': kernel_stats,
        'operator_stats': op_stats,
        'attention_kernels': attention_kernels,
        'gemm_kernels': gemm_kernels,
        'top_kernels': dict(top_kernels),
    }


def print_analysis_report(baseline_stats: dict,
                          bpc_stats: dict,
                          comparison: dict):
    """Print analysis report."""
    print("\n" + "=" * 80)
    print("CHROME TRACE ANALYSIS REPORT - DECODING STEP COMPARISON")
    print("=" * 80)

    print("\n" + "-" * 40)
    print("OVERALL SUMMARY")
    print("-" * 40)
    summary = comparison['summary']
    print(f"Baseline Total Kernel Time:   {summary['baseline_total_time_ms']:.3f} ms")
    print(f"BPC Total Kernel Time:  {summary['bpc_total_time_ms']:.3f} ms")
    print(f"Speedup:                      {summary['speedup']:.2f}x")
    print(f"Time Reduction:               {summary['time_reduction_percent']:.1f}%")

    print("\n" + "-" * 40)
    print("ATTENTION KERNELS")
    print("-" * 40)
    attn = comparison['attention_comparison']
    print(f"Baseline Attention Time:      {attn['baseline_time_ms']:.3f} ms")
    print(f"BPC Attention Time:     {attn['bpc_time_ms']:.3f} ms")
    print(f"Speedup:                      {attn['speedup']:.2f}x")
    print(f"Time Reduction:               {attn['time_reduction_percent']:.1f}%")

    print("\n" + "-" * 40)
    print("GEMM KERNELS")
    print("-" * 40)
    gemm = comparison['gemm_comparison']
    print(f"Baseline GEMM Time:           {gemm['baseline_time_ms']:.3f} ms")
    print(f"BPC GEMM Time:          {gemm['bpc_time_ms']:.3f} ms")
    print(f"Speedup:                      {gemm['speedup']:.2f}x")
    print(f"Time Reduction:               {gemm['time_reduction_percent']:.1f}%")

    print("\n" + "-" * 40)
    print("TOP 10 KERNELS BY TIME (BASELINE)")
    print("-" * 40)
    print(f"{'Kernel Name':<60} {'Count':>8} {'Total(ms)':>12} {'Avg(us)':>10}")
    print("-" * 90)
    for name, stats in list(baseline_stats['top_kernels'].items())[:10]:
        short_name = name[:57] + "..." if len(name) > 60 else name
        print(f"{short_name:<60} {stats['count']:>8} {stats['total_time_us']/1000:>12.3f} {stats['avg_time_us']:>10.2f}")
    
    print("\n" + "-" * 40)
    print("TOP 10 KERNELS BY TIME (BINVORTEX)")
    print("-" * 40)
    print(f"{'Kernel Name':<60} {'Count':>8} {'Total(ms)':>12} {'Avg(us)':>10}")
    print("-" * 90)
    for name, stats in list(bpc_stats['top_kernels'].items())[:10]:
        short_name = name[:57] + "..." if len(name) > 60 else name
        print(f"{short_name:<60} {stats['count']:>8} {stats['total_time_us']/1000:>12.3f} {stats['avg_time_us']:>10.2f}")
    
    print("\n" + "=" * 80)


def analyze_single_only(trace_path: str, output_path: Optional[str] = None):
    """Analyze a single trace file (no baseline comparison)."""
    stats = analyze_single_trace(trace_path)
    
    print("\n" + "=" * 80)
    print("SINGLE TRACE ANALYSIS REPORT")
    print("=" * 80)
    
    summary = stats['summary']
    print(f"\nTotal Kernel Time: {summary['total_kernel_time_ms']:.3f} ms")
    print(f"Total Kernel Count: {summary['total_kernel_count']}")
    print(f"Unique Kernels: {summary['unique_kernel_count']}")
    
    # Attention kernels
    attn_kernels = stats['attention_kernels']
    attn_time = sum(s['total_time_us'] for s in attn_kernels.values())
    print(f"\nAttention Kernels: {len(attn_kernels)}")
    print(f"Attention Time: {attn_time/1000:.3f} ms ({100*attn_time/max(summary['total_kernel_time_us'],1):.1f}%)")
    
    # GEMM kernels
    gemm_kernels = stats['gemm_kernels']
    gemm_time = sum(s['total_time_us'] for s in gemm_kernels.values())
    print(f"\nGEMM Kernels: {len(gemm_kernels)}")
    print(f"GEMM Time: {gemm_time/1000:.3f} ms ({100*gemm_time/max(summary['total_kernel_time_us'],1):.1f}%)")
    
    # Top kernels
    print("\n" + "-" * 40)
    print("TOP 20 KERNELS BY TIME")
    print("-" * 40)
    print(f"{'Kernel Name':<60} {'Count':>8} {'Total(ms)':>12} {'Avg(us)':>10}")
    print("-" * 90)
    for name, s in stats['top_kernels'].items():
        short_name = name[:57] + "..." if len(name) > 60 else name
        print(f"{short_name:<60} {s['count']:>8} {s['total_time_us']/1000:>12.3f} {s['avg_time_us']:>10.2f}")
    
    if output_path:
        # Drop large fields when saving
        save_stats = {
            'trace_path': stats['trace_path'],
            'total_events': stats['total_events'],
            'summary': stats['summary'],
            'attention_kernels': stats['attention_kernels'],
            'gemm_kernels': stats['gemm_kernels'],
            'top_kernels': stats['top_kernels'],
        }
        with open(output_path, 'w') as f:
            json.dump(save_stats, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze Chrome trace JSON files")
    parser.add_argument('--baseline', type=str, help='Path to baseline trace JSON')
    parser.add_argument('--bpc', type=str, help='Path to bpc trace JSON')
    parser.add_argument('--trace', type=str, help='Path to single trace JSON (for single analysis)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    
    args = parser.parse_args()

    if args.trace:
        analyze_single_only(args.trace, args.output)
        return

    if not args.baseline or not args.bpc:
        print("Error: Please provide both --baseline and --bpc paths, or use --trace for single analysis")
        return

    print("Analyzing baseline trace...")
    baseline_stats = analyze_single_trace(args.baseline)

    print("\nAnalyzing bpc trace...")
    bpc_stats = analyze_single_trace(args.bpc)

    comparison = compare_traces(baseline_stats, bpc_stats)
    print_analysis_report(baseline_stats, bpc_stats, comparison)

    if args.output:
        results = {
            'baseline': {
                'trace_path': baseline_stats['trace_path'],
                'summary': baseline_stats['summary'],
                'attention_kernels': baseline_stats['attention_kernels'],
                'gemm_kernels': baseline_stats['gemm_kernels'],
                'top_kernels': baseline_stats['top_kernels'],
            },
            'bpc': {
                'trace_path': bpc_stats['trace_path'],
                'summary': bpc_stats['summary'],
                'attention_kernels': bpc_stats['attention_kernels'],
                'gemm_kernels': bpc_stats['gemm_kernels'],
                'top_kernels': bpc_stats['top_kernels'],
            },
            'comparison': comparison,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
