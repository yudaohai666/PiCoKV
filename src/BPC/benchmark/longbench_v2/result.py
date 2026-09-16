import os, json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

files = os.listdir('results')
output = []
compensated = False

for file in files:
    if not file.endswith('.jsonl') and not file.endswith('.json'):
        continue
    filename = os.path.join('results', file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    
    for pred in pred_data:
        # judge may be bool or int
        judge_val = pred.get('judge', False)
        if isinstance(judge_val, bool):
            acc = 1 if judge_val else 0
        else:
            acc = int(judge_val)
        
        if compensated and pred.get("pred") is None:
            acc = 0.25
        
        if pred.get("difficulty") == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred.get('length') == "short":
            short += 1
            short_acc += acc
        elif pred.get('length') == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    name = '.'.join(file.split('.')[:-1])

    # Guard against division by zero
    overall = round(100*(easy_acc+hard_acc)/len(pred_data), 1) if pred_data else 0
    easy_pct = round(100*easy_acc/easy, 1) if easy > 0 else 0
    hard_pct = round(100*hard_acc/hard, 1) if hard > 0 else 0
    short_pct = round(100*short_acc/short, 1) if short > 0 else 0
    medium_pct = round(100*medium_acc/medium, 1) if medium > 0 else 0
    long_pct = round(100*long_acc/long, 1) if long > 0 else 0
    
    output.append([name, overall, easy_pct, hard_pct, short_pct, medium_pct, long_pct])

output.sort(key=lambda x: x[1], reverse=True)

header = ["Method", "Overall", "Easy", "Hard", "Short", "Medium", "Long"]
all_rows = [header] + output
col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(header))]

lines = []
for row in all_rows:
    line = "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
    lines.append(line)

open('result.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))


def plot_comparison(output, header):
    """Render bar chart comparison."""
    if not output:
        return

    method_names = [row[0] for row in output]
    tasks = header[1:]  # Overall, Easy, Hard, Short, Medium, Long
    n_methods = len(method_names)
    n_tasks = len(tasks)

    x = np.arange(n_tasks)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for i, row in enumerate(output):
        scores = row[1:]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=row[0], color=colors[i])
        
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    max_score = max(max(row[1:]) for row in output)
    ax.set_ylim(0, min(100, max_score + 15))
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LongBench v2 Results Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"柱状图已保存至: comparison.png")


def plot_table(output, header):
    """Render paper-style table image."""
    if not output:
        return

    n_methods = len(output)
    n_cols = len(header)

    best_idx = {}
    for col in range(1, n_cols):
        scores = [(i, row[col]) for i, row in enumerate(output)]
        best_idx[col] = max(scores, key=lambda x: x[1])[0]

    table_data = []
    for row in output:
        table_data.append([row[0]] + [f'{v:.1f}' for v in row[1:]])

    fig_width = max(14, n_cols * 1.5 + 6)
    fig_height = max(3, n_methods * 0.5 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=header,
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
            cell.set_width(0.4)

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
            elif best_idx.get(col) == row - 1:
                cell.set_text_props(fontweight='bold')
    
    ax.set_title('LongBench v2 Results', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig('comparison_table.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"表格已保存至: comparison_table.png")


if output:
    plot_comparison(output, header)
    plot_table(output, header)
