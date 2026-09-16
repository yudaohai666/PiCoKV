"""
Attention time-breakdown analysis tool.

Extracts per-component attention timings for baseline and BPC from profiler
summary files and renders a stacked bar chart.

Usage:
    python extract_attention_breakdown.py \
        --traces_dir outputs/traces \
        --output breakdown.json \
        --plot breakdown.png
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import glob


def parse_summary_file(file_path: str) -> Dict:
    """Parse a summary.txt file and extract kernel info."""
    result = {
        'file_path': file_path,
        'method': None,
        'model': None,
        'P': None,
        'G': None,
        'B': None,
        'GPUs': None,
        'cache': None,
        'kernels': [],
        'total_cuda_time_s': None,
    }
    
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    for line in lines:
        # Method: origin, Model: meta-llama/Llama-3.1-8B-Instruct
        if line.startswith('Method:'):
            match = re.match(r'Method:\s*(\S+),\s*Model:\s*(.+)', line.strip())
            if match:
                result['method'] = match.group(1)
                result['model'] = match.group(2)
        
        # P=131072, G=64, B=1, GPUs=8, Cache=static
        if line.startswith('P='):
            match = re.match(r'P=(\d+),\s*G=(\d+),\s*B=(\d+),\s*GPUs=(\d+),\s*Cache=(\w+)', line.strip())
            if match:
                result['P'] = int(match.group(1))
                result['G'] = int(match.group(2))
                result['B'] = int(match.group(3))
                result['GPUs'] = int(match.group(4))
                result['cache'] = match.group(5)
        
        # Self CUDA time total: 2.145s
        if 'Self CUDA time total:' in line:
            match = re.search(r'Self CUDA time total:\s*([\d.]+)([ms])', line)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                result['total_cuda_time_s'] = value if unit == 's' else value / 1000
    
    # Match table rows; example:
    # void flash_fwd_splitkv_kernel<Flash_fwd_kernel_trait...  0.00%  0.000us  0.00%  0.000us  0.000us  942.638ms  43.94%  942.638ms  460.272us  2048
    kernel_pattern = re.compile(
        r'^(.+?)\s+'  # name (non-greedy)
        r'([\d.]+)%\s+'  # Self CPU %
        r'([\d.]+)(us|ms|s)\s+'  # Self CPU
        r'([\d.]+)%\s+'  # CPU total %
        r'([\d.]+)(us|ms|s)\s+'  # CPU total
        r'([\d.]+)(us|ms|s)\s+'  # CPU time avg
        r'([\d.]+)(us|ms|s)\s+'  # Self CUDA
        r'([\d.]+)%\s+'  # Self CUDA %
        r'([\d.]+)(us|ms|s)\s+'  # CUDA total
        r'([\d.]+)(us|ms|s)\s+'  # CUDA time avg
        r'(\d+)\s*$'  # # of Calls
    )
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('-') or line.startswith('Name') or line.startswith('Top'):
            continue
        
        match = kernel_pattern.match(line)
        if match:
            name = match.group(1).strip()
            
            # Self CUDA time
            self_cuda_val = float(match.group(10))
            self_cuda_unit = match.group(11)
            self_cuda_ms = convert_to_ms(self_cuda_val, self_cuda_unit)
            
            # Self CUDA %
            self_cuda_pct = float(match.group(12))
            
            # CUDA time avg
            cuda_avg_val = float(match.group(15))
            cuda_avg_unit = match.group(16)
            cuda_avg_us = convert_to_us(cuda_avg_val, cuda_avg_unit)
            
            # # of Calls
            calls = int(match.group(17))
            
            result['kernels'].append({
                'name': name,
                'self_cuda_ms': self_cuda_ms,
                'self_cuda_pct': self_cuda_pct,
                'cuda_avg_us': cuda_avg_us,
                'calls': calls,
            })
    
    return result


def convert_to_ms(value: float, unit: str) -> float:
    """Convert to ms."""
    if unit == 'ms':
        return value
    elif unit == 'us':
        return value / 1000
    elif unit == 's':
        return value * 1000
    return value


def convert_to_us(value: float, unit: str) -> float:
    """Convert to us."""
    if unit == 'us':
        return value
    elif unit == 'ms':
        return value * 1000
    elif unit == 's':
        return value * 1000000
    return value


# Attention component classification.
# Count only low-level CUDA kernels to avoid double-counting high-level APIs.

def is_low_level_kernel(name: str) -> bool:
    """
    True for low-level CUDA kernels (vs high-level PyTorch APIs).
    Low-level kernels typically start with `void ` or are known fused kernels;
    high-level APIs look like `aten::xxx`, `flash_attn::xxx`.
    """
    name_stripped = name.strip()
    if name_stripped.startswith('void '):
        return True
    if 'fused_quant_pack_probe' in name.lower():
        return True
    return False


def classify_baseline_kernel(name: str) -> Optional[str]:
    """Classify baseline (FlashAttention) kernels; low-level kernels only."""
    if not is_low_level_kernel(name):
        return None

    name_lower = name.lower()

    if 'flash_fwd_splitkv_kernel' in name_lower or 'flash_fwd_kernel' in name_lower:
        return 'FlashAttn Kernel'

    if 'flash_fwd_splitkv_combine' in name_lower:
        return 'FlashAttn Combine'

    return None


def classify_bpc_kernel(name: str) -> Optional[str]:
    """
    Classify BPC kernels.

    For high-level APIs that wrap multiple sub-kernels (e.g. aten::topk),
    profiler Self CUDA time already reflects actual GPU time. Count those
    high-level APIs (aten::topk, aten::gather) directly, not their sub-kernels
    (radixFindKthValues, gatherTopK), to avoid miscounting.
    """
    name_lower = name.lower()
    name_stripped = name.strip()

    # SDPA / Flash Attention kernel (BPC uses PyTorch native SDPA).
    # Match low-level kernels here.
    if name_stripped.startswith('void '):
        if 'pytorch_flash::flash_fwd' in name_lower or 'flash_fwd_splitkv_kernel' in name_lower:
            return 'SDPA Kernel'

    # Count aten::topk (1920 calls = 30 layers x 64 steps x 1/layer/step);
    # do NOT count its sub-kernels (radixFindKthValues, gatherTopK).
    if name_stripped == 'aten::topk':
        return 'TopK'

    # Count aten::gather (3840 calls = 30 layers x 64 steps x 2/layer/step)
    if name_stripped == 'aten::gather':
        return 'Gather'

    # BPC-specific fused kernel (hash + quant + pack)
    if 'fused_quant_pack_probe' in name_lower:
        return 'Hash+Quant+Pack'

    return None


def extract_attention_breakdown(parsed_data: Dict, is_bpc: bool = False) -> Dict:
    """
    Extract per-component attention timings.

    Notes:
    1. Sum only low-level kernel Self CUDA time to avoid double counting.
    2. Use call count to back out sparse-layer count for per-step averages.
    """
    breakdown = defaultdict(lambda: {
        'total_ms': 0.0,
        'total_pct': 0.0,
        'kernels': [],
        'calls': 0,
    })
    
    G = parsed_data.get('G', 1)
    
    for kernel in parsed_data['kernels']:
        if is_bpc:
            component = classify_bpc_kernel(kernel['name'])
        else:
            component = classify_baseline_kernel(kernel['name'])
        
        if component:
            breakdown[component]['total_ms'] += kernel['self_cuda_ms']
            breakdown[component]['total_pct'] += kernel['self_cuda_pct']
            breakdown[component]['calls'] += kernel['calls']
            breakdown[component]['kernels'].append(kernel['name'])
    
    # Per-call/per-layer averages
    for component, data in breakdown.items():
        # calls_per_step = total_calls / steps (= sparse-layer count)
        calls_per_step = data['calls'] / G if G > 0 else 1
        data['calls_per_step'] = calls_per_step

        # Average per-layer time (ms)
        data['avg_per_layer_ms'] = data['total_ms'] / data['calls'] if data['calls'] > 0 else 0

        # Per-step average (kept for compatibility)
        data['avg_per_step_ms'] = data['total_ms'] / G if G > 0 else 0

        # Dedup kernel names
        data['kernels'] = list(set(data['kernels']))

    return dict(breakdown)


def find_summary_pairs(traces_dir: str) -> List[Tuple[str, str]]:
    """Pair baseline and bpc summary files by (P, G); timestamps may differ."""
    baseline_files = glob.glob(os.path.join(traces_dir, 'baseline_*_summary.txt'))
    bpc_files = glob.glob(os.path.join(traces_dir, 'bpc-offline_*_summary.txt'))

    def parse_config(filename):
        basename = os.path.basename(filename)
        match = re.match(r'(baseline|bpc-offline)_P(\d+)_G(\d+)', basename)
        if match:
            return (int(match.group(2)), int(match.group(3)))  # (P, G)
        return None
    
    baseline_by_config = {}
    for f in baseline_files:
        config = parse_config(f)
        if config:
            # If multiple files share a config, keep the newest (timestamp suffix)
            if config not in baseline_by_config or f > baseline_by_config[config]:
                baseline_by_config[config] = f

    bpc_by_config = {}
    for f in bpc_files:
        config = parse_config(f)
        if config:
            if config not in bpc_by_config or f > bpc_by_config[config]:
                bpc_by_config[config] = f

    pairs = []
    for config in baseline_by_config:
        if config in bpc_by_config:
            pairs.append((baseline_by_config[config], bpc_by_config[config]))
            print(f"  Found pair for P={config[0]}, G={config[1]}:")
            print(f"    Baseline: {os.path.basename(baseline_by_config[config])}")
            print(f"    BPC: {os.path.basename(bpc_by_config[config])}")
    
    return pairs


def analyze_traces(traces_dir: str) -> List[Dict]:
    """Analyze traces directory."""
    pairs = find_summary_pairs(traces_dir)
    results = []
    
    for baseline_path, bpc_path in pairs:
        print(f"Analyzing:")
        print(f"  Baseline: {os.path.basename(baseline_path)}")
        print(f"  BPC: {os.path.basename(bpc_path)}")
        
        baseline_data = parse_summary_file(baseline_path)
        bpc_data = parse_summary_file(bpc_path)
        
        baseline_breakdown = extract_attention_breakdown(baseline_data, is_bpc=False)
        bpc_breakdown = extract_attention_breakdown(bpc_data, is_bpc=True)
        
        G = baseline_data['G']
        
        result = {
            'P': baseline_data['P'],
            'G': baseline_data['G'],
            'B': baseline_data['B'],
            'GPUs': baseline_data['GPUs'],
            'baseline': {
                'method': baseline_data['method'],
                'total_cuda_time_s': baseline_data['total_cuda_time_s'],
                'attention_breakdown': baseline_breakdown,
                'total_attention_ms': sum(c['total_ms'] for c in baseline_breakdown.values()),
            },
            'bpc': {
                'method': bpc_data['method'],
                'total_cuda_time_s': bpc_data['total_cuda_time_s'],
                'attention_breakdown': bpc_breakdown,
                'total_attention_ms': sum(c['total_ms'] for c in bpc_breakdown.values()),
            },
        }
        
        # Per-step averages
        result['baseline']['avg_attention_per_step_ms'] = result['baseline']['total_attention_ms'] / G
        result['bpc']['avg_attention_per_step_ms'] = result['bpc']['total_attention_ms'] / G
        
        results.append(result)
    
    results.sort(key=lambda x: x['P'])
    return results


def print_report(results: List[Dict]):
    """Print report."""
    print("\n" + "=" * 100)
    print("ATTENTION TIME BREAKDOWN REPORT")
    print("=" * 100)
    
    for result in results:
        P = result['P']
        G = result['G']
        
        print(f"\n{'='*80}")
        print(f"Context Length P={P:,}, Generation Steps G={G}")
        print(f"{'='*80}")
        
        # Baseline
        print(f"\n--- BASELINE (FlashAttention) ---")
        baseline = result['baseline']
        print(f"Total CUDA Time: {baseline['total_cuda_time_s']:.3f}s")
        print(f"Total Attention Time: {baseline['total_attention_ms']:.3f}ms")
        print(f"Avg Attention per Step: {baseline['avg_attention_per_step_ms']:.3f}ms")
        
        print(f"\nBreakdown:")
        for comp, data in sorted(baseline['attention_breakdown'].items(), key=lambda x: -x[1]['total_ms']):
            print(f"  {comp:20s}: {data['total_ms']:8.3f}ms ({data['total_pct']:5.2f}%), "
                  f"avg={data['avg_per_layer_ms']:.3f}ms/layer, "
                  f"calls={data['calls']} ({data.get('calls_per_step', 0):.0f}/step)")
        
        # BPC
        print(f"\n--- BINVORTEX ---")
        bpc = result['bpc']
        print(f"Total CUDA Time: {bpc['total_cuda_time_s']:.3f}s")
        print(f"Total Attention Time: {bpc['total_attention_ms']:.3f}ms")
        print(f"Avg Attention per Step: {bpc['avg_attention_per_step_ms']:.3f}ms")
        
        print(f"\nBreakdown:")
        for comp, data in sorted(bpc['attention_breakdown'].items(), key=lambda x: -x[1]['total_ms']):
            print(f"  {comp:20s}: {data['total_ms']:8.3f}ms ({data['total_pct']:5.2f}%), "
                  f"avg={data['avg_per_layer_ms']:.3f}ms/layer, "
                  f"calls={data['calls']} ({data.get('calls_per_step', 0):.0f}/step)")
        
        # Comparison
        print(f"\n--- COMPARISON ---")
        baseline_attn = baseline['total_attention_ms']
        bpc_attn = bpc['total_attention_ms']
        speedup = baseline_attn / bpc_attn if bpc_attn > 0 else float('inf')
        print(f"Attention Speedup: {speedup:.2f}x")
        print(f"Time Saved: {baseline_attn - bpc_attn:.3f}ms ({(1-bpc_attn/baseline_attn)*100:.1f}%)")


def plot_attention_breakdown(results: List[Dict], output_path: str):
    """Stacked bar chart over multiple context lengths (per-layer time, ICML style)."""
    import matplotlib.pyplot as plt
    import numpy as np

    # ICML academic style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 11,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.4,
        'patch.linewidth': 0.4,
    })
    
    n_results = len(results)

    # ICML two-column width
    fig, ax = plt.subplots(figsize=(5, 4))

    # Display-name mapping
    bpc_display_names = {
        'TopK': 'Top-2% Selection',
        'Gather': 'Top-2% Gathering',
        'Hash+Quant+Pack': 'Top-2% Searching',
        'SDPA Kernel': 'Top-2% Attention',
    }
    
    colors = ['#82B366', '#6C8EBF', '#9673A6', '#E47B26']
    baseline_color = '#2E86AB'
    bpc_colors = {
        'Top-2% Searching': colors[0],
        'Top-2% Selection': colors[3],
        'Top-2% Gathering': colors[2],
        'Top-2% Attention': colors[1],
    }
    
    # X positions: paired bars with small inter-group gap
    bar_width = 0.35
    x_positions = np.arange(n_results) * (bar_width * 2 + 0.3)

    baseline_totals = []
    bpc_data = []
    context_lengths = []
    
    for result in results:
        P = result['P']
        context_lengths.append(f'{P//1024}K')
        
        baseline_breakdown = result['baseline']['attention_breakdown']
        bpc_breakdown = result['bpc']['attention_breakdown']
        
        baseline_total = sum([
            baseline_breakdown.get('FlashAttn Kernel', {}).get('avg_per_layer_ms', 0),
            baseline_breakdown.get('FlashAttn Combine', {}).get('avg_per_layer_ms', 0),
        ])
        baseline_totals.append(baseline_total)
        
        bpc_vals = {}
        for comp in ['Hash+Quant+Pack', 'TopK', 'Gather', 'SDPA Kernel']:
            val = bpc_breakdown.get(comp, {}).get('avg_per_layer_ms', 0)
            display_name = bpc_display_names[comp]
            bpc_vals[display_name] = val
        bpc_data.append(bpc_vals)
    
    # Baseline bars
    ax.bar(x_positions, baseline_totals, bar_width,
           label='FlashAttn-2.7.1', color=baseline_color,
           edgecolor='black', linewidth=0.4)

    # BPC stacked bars
    bpc_order = ['Top-2% Searching', 'Top-2% Selection', 'Top-2% Gathering', 'Top-2% Attention']
    bottoms = np.zeros(n_results)

    for comp in bpc_order:
        vals = [bpc_data[i].get(comp, 0) for i in range(n_results)]
        ax.bar(x_positions + bar_width, vals, bar_width, bottom=bottoms,
               label=comp, color=bpc_colors[comp], edgecolor='black', linewidth=0.4)
        bottoms += vals

    # Speedup annotations
    for i, (x, baseline_val) in enumerate(zip(x_positions, baseline_totals)):
        bpc_total = sum(bpc_data[i].values())
        speedup = baseline_val / bpc_total if bpc_total > 0 else 0
        max_height = max(baseline_val, bpc_total)
        ax.text(x + bar_width/2, max_height * 1.02, f'{speedup:.1f}×', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xticks(x_positions + bar_width/2)
    ax.set_xticklabels(context_lengths)
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Attention Latency (ms)')

    max_val = max(max(baseline_totals), max(sum(d.values()) for d in bpc_data))
    ax.set_ylim(0, max_val * 1.15)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.4)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.9, ncol=1, 
              handlelength=1.0, handletextpad=0.4, borderpad=0.3)
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                format='pdf' if output_path.endswith('.pdf') else 'png')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_single_comparison(result: Dict, output_path: str):
    """Single-config comparison plot (per-layer time, ICML style)."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'axes.linewidth': 0.8,
    })
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    P = result['P']
    
    baseline_breakdown = result['baseline']['attention_breakdown']
    bpc_breakdown = result['bpc']['attention_breakdown']
    
    bpc_display_names = {
        'TopK': 'Top-2% Selection',
        'Gather': 'Top-2% Gathering',
        'Hash+Quant+Pack': 'Top-2% Searching',
        'SDPA Kernel': 'Top-2% Attention',
    }
    
    colors_list = ['#82B366', '#6C8EBF', '#9673A6', '#E47B26']
    baseline_color = '#6C8EBF'
    bpc_colors = {
        'Top-2% Searching': colors_list[0],
        'Top-2% Selection': colors_list[3],
        'Top-2% Gathering': colors_list[2],
        'Top-2% Attention': colors_list[1],
    }
    
    baseline_total_val = sum([
        baseline_breakdown.get('FlashAttn Kernel', {}).get('avg_per_layer_ms', 0),
        baseline_breakdown.get('FlashAttn Combine', {}).get('avg_per_layer_ms', 0),
    ])
    
    bpc_order = ['Hash+Quant+Pack', 'TopK', 'Gather', 'SDPA Kernel']
    bpc_comps = []
    bpc_vals = []
    for comp in bpc_order:
        val = bpc_breakdown.get(comp, {}).get('avg_per_layer_ms', 0)
        if val > 0:
            bpc_comps.append(bpc_display_names[comp])
            bpc_vals.append(val)
    
    width = 0.3
    gap = 0.1
    
    # Baseline bar
    ax.bar(0, baseline_total_val, width, color=baseline_color, label='FA2',
           edgecolor='black', linewidth=0.5)

    # BPC stacked bar
    bottom = 0
    for comp, val in zip(bpc_comps, bpc_vals):
        ax.bar(1, val, width, bottom=bottom, color=bpc_colors[comp], label=comp,
               edgecolor='black', linewidth=0.5)
        bottom += val

    bpc_total = sum(bpc_vals)
    speedup = baseline_total_val / bpc_total if bpc_total > 0 else 0

    # Speedup annotation
    max_height = max(baseline_total_val, bpc_total)
    ax.text(0.5, max_height * 1.05, f'{speedup:.1f}×', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['FA2', 'Ours'])
    ax.set_ylabel('Latency (ms)')
    ax.set_ylim(0, max_height * 1.15)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def export_data(results: List[Dict], output_path: str):
    """Export JSON data."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and plot attention time breakdown")
    parser.add_argument('--traces_dir', type=str,
                        default='outputs/traces',
                        help='Directory containing trace summary files')
    parser.add_argument('--output', type=str, default='attention_breakdown.json',
                        help='Output JSON file')
    parser.add_argument('--plot', type=str, default='attention_breakdown.png',
                        help='Output plot file')
    
    args = parser.parse_args()
    
    print(f"Analyzing traces in: {args.traces_dir}")
    results = analyze_traces(args.traces_dir)
    
    if not results:
        print("No matching trace pairs found!")
        return
    
    print_report(results)
    export_data(results, args.output)

    # Multi-length comparison plot in a single chart
    plot_attention_breakdown(results, args.plot)


if __name__ == "__main__":
    main()
