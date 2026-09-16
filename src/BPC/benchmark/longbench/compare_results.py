#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare LongBench results across methods/models; render bar chart + table.

Usage:
    python compare_results.py <output_dir> [output_path]

Layout: outputs/longbench/{model}/{method}/result.json
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_result(result_path):
    """Load one result.json."""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_all_results(output_dir):
    """Discover result.json files. Handles both per-model and aggregate dirs.

    Returns (model_name_or_None, {display_name: result_dict}).
    """
    results = {}
    output_path = Path(output_dir)
    model_name = output_path.name

    has_direct_results = False
    for subdir in output_path.iterdir():
        if subdir.is_dir() and (subdir / 'result.json').exists():
            has_direct_results = True
            break

    if has_direct_results:
        # Mode 1: current dir is a model dir; subdirs are methods
        for method_dir in output_path.iterdir():
            if method_dir.is_dir():
                result_file = method_dir / 'result.json'
                if result_file.exists():
                    results[method_dir.name] = load_result(result_file)
                    print(f"已加载: {method_dir.name}")
                # Also descend into param subdirs (e.g. pyramidkv/4096/)
                for param_dir in method_dir.iterdir():
                    if param_dir.is_dir():
                        param_result = param_dir / 'result.json'
                        if param_result.exists():
                            display_name = f"{method_dir.name}-{param_dir.name}"
                            results[display_name] = load_result(param_result)
                            print(f"已加载: {display_name}")
    else:
        # Mode 2: aggregate dir; iterate model/method
        model_name = None
        for model_dir in output_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            m_name = model_dir.name
            
            for method_dir in model_dir.iterdir():
                if method_dir.is_dir():
                    result_file = method_dir / 'result.json'
                    if result_file.exists():
                        display_name = f"{m_name}/{method_dir.name}"
                        results[display_name] = load_result(result_file)
                        print(f"已加载: {display_name}")
                    for param_dir in method_dir.iterdir():
                        if param_dir.is_dir():
                            param_result = param_dir / 'result.json'
                            if param_result.exists():
                                display_name = f"{m_name}/{method_dir.name}-{param_dir.name}"
                                results[display_name] = load_result(param_result)
                                print(f"已加载: {display_name}")
    
    return model_name, results


def plot_table(results, output_path='table.png', model_name=None):
    """Render paper-style table: rows=methods, cols=tasks."""
    if not results:
        print("没有找到任何结果文件！")
        return

    all_tasks = set()
    for model_results in results.values():
        all_tasks.update(model_results.keys())
    all_tasks = sorted([t for t in all_tasks if t != 'avg']) + (['avg'] if 'avg' in all_tasks else [])

    def get_avg(method_name):
        method_results = results[method_name]
        if 'avg' in method_results:
            return method_results['avg']
        scores = [v for v in method_results.values() if isinstance(v, (int, float))]
        return np.mean(scores) if scores else 0
    
    method_names = sorted(results.keys(), key=get_avg, reverse=True)
    n_methods = len(method_names)
    n_tasks = len(all_tasks)

    # Best score per task (for bold)
    best_scores = {}
    for task in all_tasks:
        scores = []
        for method_name in method_names:
            score = results[method_name].get(task, None)
            if isinstance(score, (int, float)):
                scores.append((method_name, score))
        if scores:
            best_scores[task] = max(scores, key=lambda x: x[1])[0]
    
    cell_text = []
    for method_name in method_names:
        row = []
        for task in all_tasks:
            score = results[method_name].get(task, '-')
            if isinstance(score, (int, float)):
                row.append(f'{score:.2f}')
            else:
                row.append(str(score))
        cell_text.append(row)
    
    fig_width = max(14, n_tasks * 1.3 + 4)
    fig_height = max(3, n_methods * 0.5 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    col_labels = ['Method'] + all_tasks

    table_data = []
    for i, method_name in enumerate(method_names):
        table_data.append([method_name] + cell_text[i])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        edges='horizontal'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.8)

    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.25)

    for key, cell in table.get_celld().items():
        row, col = key
        cell.set_linewidth(0.5)
        cell.set_edgecolor('#000000')

        if row == 0:
            cell.set_text_props(fontweight='bold', fontsize=10)
            cell.set_facecolor('#f0f0f0')
            cell.set_linewidth(1.5)
        else:
            cell.set_facecolor('white')

            if col == 0:
                cell.set_text_props(ha='left', fontweight='bold')
                cell._loc = 'left'
            else:
                task = all_tasks[col - 1]
                method = method_names[row - 1]
                if best_scores.get(task) == method:
                    cell.set_text_props(fontweight='bold')

    bbox = table.get_window_extent(fig.canvas.get_renderer())
    bbox = bbox.transformed(ax.transData.inverted())

    title = f'LongBench Results - {model_name}' if model_name else 'LongBench Results'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, loc='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"表格已保存至: {output_path}")


def plot_comparison(results, output_path='comparison.png', figsize=(16, 8), model_name=None):
    """Render grouped bar chart comparing methods."""
    if not results:
        print("没有找到任何结果文件！")
        return

    all_tasks = set()
    for model_results in results.values():
        all_tasks.update(model_results.keys())
    all_tasks = sorted([t for t in all_tasks if t != 'avg']) + (['avg'] if 'avg' in all_tasks else [])

    model_names = list(results.keys())
    n_tasks = len(all_tasks)
    n_models = len(model_names)

    x = np.arange(n_tasks)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (method_name, method_results) in enumerate(results.items()):
        scores = [method_results.get(task, 0) for task in all_tasks]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=method_name, color=colors[i])

        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=6,
                        rotation=45)

    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='100')

    # Extend y-axis when scores >95 so labels fit above bars
    max_score = max(max(r.values()) for r in results.values())
    if max_score > 95:
        ax.set_ylim(0, 115)
    else:
        ax.set_ylim(0, 105)
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    title = f'LongBench Results Comparison - {model_name}' if model_name else 'LongBench Results Comparison'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存至: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python compare_results.py <output_dir> [output_path]")
        print("示例: python compare_results.py outputs/longbench/")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(output_dir, 'comparison.png')
    
    if not os.path.isdir(output_dir):
        print(f"错误: 目录不存在 - {output_dir}")
        sys.exit(1)
    
    model_name, results = find_all_results(output_dir)
    
    if not results:
        print("未找到任何 result.json 文件！")
        return
    
    print(f"\n共发现 {len(results)} 个结果")
    if model_name:
        print(f"模型: {model_name}")
    
    plot_comparison(results, output_path, model_name=model_name)

    table_path = output_path.replace('.png', '_table.png')
    plot_table(results, table_path, model_name=model_name)

    print("\n=== 结果汇总 ===")
    if model_name:
        print(f"Model: {model_name}\n")
    all_tasks = sorted(set().union(*[set(r.keys()) for r in results.values()]))
    all_tasks = [t for t in all_tasks if t != 'avg'] + (['avg'] if 'avg' in all_tasks else [])

    header = f"{'Task':<25}" + "".join(f"{name:<25}" for name in results.keys())
    print(header)
    print("-" * len(header))

    for task in all_tasks:
        row = f"{task:<25}"
        for model_results in results.values():
            score = model_results.get(task, '-')
            if isinstance(score, (int, float)):
                row += f"{score:<25.2f}"
            else:
                row += f"{score:<25}"
        print(row)
    
    print("-" * len(header))
    avg_row = f"{'Average':<25}"
    for model_results in results.values():
        scores = [v for v in model_results.values() if isinstance(v, (int, float))]
        avg = np.mean(scores) if scores else 0
        avg_row += f"{avg:<25.2f}"
    print(avg_row)


if __name__ == '__main__':
    main()
