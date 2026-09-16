#!/usr/bin/env python3
"""Extract per-subtask metrics from a ruler evaluation results JSON file.

Usage: python3 extract_metrics.py path/to/results.json
"""

import json
import sys
import os
from pathlib import Path


def extract_metrics(json_path: str) -> dict:
    """Extract per-subtask metrics from a results JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', {})

    # Skip the overall 'ruler' aggregate
    subtask_metrics = {}
    for task_name, task_data in results.items():
        if task_name == 'ruler':
            continue

        metrics = {}
        for key, value in task_data.items():
            if key == 'alias' or '_stderr' in key:
                continue
            # Keys like "8192,none" carry the sequence length
            if ',none' in key:
                seq_len = key.replace(',none', '')
                if value != -1:  # -1 means not tested
                    metrics[seq_len] = value

        if metrics:
            subtask_metrics[task_name] = metrics

    return subtask_metrics


def print_metrics_table(metrics: dict):
    """Print metrics as a table."""
    all_lengths = set()
    for task_metrics in metrics.values():
        all_lengths.update(task_metrics.keys())
    all_lengths = sorted(all_lengths, key=lambda x: int(x))

    header = f"{'Task':<25}" + "".join(f"{length:>12}" for length in all_lengths)
    print(header)
    print("-" * len(header))

    for task_name in sorted(metrics.keys()):
        task_metrics = metrics[task_name]
        row = f"{task_name:<25}"
        for length in all_lengths:
            value = task_metrics.get(length, '-')
            if isinstance(value, float):
                row += f"{value:>12.4f}"
            else:
                row += f"{str(value):>12}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'Average':<25}"
    for length in all_lengths:
        values = [m.get(length) for m in metrics.values() if m.get(length) is not None]
        if values:
            avg = sum(values) / len(values)
            avg_row += f"{avg:>12.4f}"
        else:
            avg_row += f"{'-':>12}"
    print(avg_row)


def main():
    if len(sys.argv) < 2:
        # Default to the origin results
        base_dir = Path(__file__).parent
        json_path = base_dir / "Llama-3.1-8B-Instruct/origin/results_Llama-3.1-8B-Instruct_origin.json"
    else:
        json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Extracting metrics from: {json_path.name}")
    print(f"{'='*60}\n")
    
    metrics = extract_metrics(str(json_path))
    print_metrics_table(metrics)

    print(f"\n{'='*60}")
    print("JSON Format:")
    print(f"{'='*60}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
