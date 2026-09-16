#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare InfiniteBench results across methods; render bar chart + table.

Usage:
    python compare_results.py <output_dir> [output_path]

Layout: outputs/infinitebench/{model}/{method}[/{max_capacity}]/scores.json
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_scores(scores_path):
    """Load one scores.json; include 'average' as 'avg' if present."""
    with open(scores_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scores = data.get('scores', {})
    if 'average' in data:
        scores['avg'] = data['average']
    return scores


def find_all_results(output_dir):
    """Discover scores.json files. Returns (model_name, {display_name: scores})."""
    results = {}
    output_path = Path(output_dir)
    model_name = output_path.name

    for method_dir in output_path.iterdir():
        if not method_dir.is_dir():
            continue

        # Method without max_capacity_prompt: scores.json directly under method dir
        scores_file = method_dir / 'scores.json'
        if scores_file.exists():
            results[method_dir.name] = load_scores(scores_file)
            print(f"已加载: {method_dir.name}")

        # Method with max_capacity_prompt: scores.json under a param subdir
        for param_dir in method_dir.iterdir():
            if param_dir.is_dir():
                param_scores = param_dir / 'scores.json'
                if param_scores.exists():
                    display_name = f"{method_dir.name}-{param_dir.name}"
                    results[display_name] = load_scores(param_scores)
                    print(f"已加载: {display_name}")

    return model_name, results


def plot_table(results, output_path='table.png', model_name=None):
    """Render a paper-style table image."""
    if not results:
        print("没有找到任何结果文件！")
        return

    all_tasks = set()
    for method_results in results.values():
        all_tasks.update(method_results.keys())
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
                row.append(f'{score*100:.2f}')
            else:
                row.append(str(score))
        cell_text.append(row)

    fig_width = max(16, n_tasks * 1.5 + 6)
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
    table.set_fontsize(9)
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
            cell.set_text_props(fontweight='bold', fontsize=9)
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
    
    title = f'InfiniteBench Results - {model_name}' if model_name else 'InfiniteBench Results'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"表格已保存至: {output_path}")


def plot_comparison(results, output_path='comparison.png', figsize=(16, 8), model_name=None):
    """Render grouped bar chart comparison."""
    if not results:
        print("没有找到任何结果文件！")
        return

    all_tasks = set()
    for method_results in results.values():
        all_tasks.update(method_results.keys())
    all_tasks = sorted([t for t in all_tasks if t != 'avg']) + (['avg'] if 'avg' in all_tasks else [])
    
    method_names = list(results.keys())
    n_tasks = len(all_tasks)
    n_methods = len(method_names)
    
    x = np.arange(n_tasks)
    width = 0.8 / n_methods
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    for i, (method_name, method_results) in enumerate(results.items()):
        scores = [method_results.get(task, 0) * 100 for task in all_tasks]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=method_name, color=colors[i])
        
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    max_score = max(max(v * 100 for v in r.values()) for r in results.values())
    ax.set_ylim(0, min(100, max_score + 15))
    
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    title = f'InfiniteBench Results Comparison - {model_name}' if model_name else 'InfiniteBench Results Comparison'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_tasks, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"柱状图已保存至: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python compare_results.py <output_dir> [output_path]")
        print("示例: python compare_results.py outputs/infinitebench/Llama-3.1-8B-Instruct/")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(output_dir, 'comparison.png')
    
    if not os.path.isdir(output_dir):
        print(f"错误: 目录不存在 - {output_dir}")
        sys.exit(1)
    
    model_name, results = find_all_results(output_dir)
    
    if not results:
        print("未找到任何 scores.json 文件！")
        return
    
    print(f"\n共发现 {len(results)} 个结果")
    print(f"模型: {model_name}")
    
    plot_comparison(results, output_path, model_name=model_name)

    table_path = output_path.replace('.png', '_table.png')
    plot_table(results, table_path, model_name=model_name)

    print("\n=== 结果汇总 ===")
    all_tasks = sorted(set().union(*[set(r.keys()) for r in results.values()]))
    all_tasks = [t for t in all_tasks if t != 'avg'] + (['avg'] if 'avg' in all_tasks else [])
    
    header = f"{'Method':<25}" + "".join(f"{task:<15}" for task in all_tasks)
    print(header)
    print("-" * len(header))
    
    for method_name, method_results in results.items():
        row = f"{method_name:<25}"
        for task in all_tasks:
            score = method_results.get(task, '-')
            if isinstance(score, (int, float)):
                row += f"{score*100:<15.2f}"
            else:
                row += f"{score:<15}"
        print(row)


if __name__ == '__main__':
    main()
