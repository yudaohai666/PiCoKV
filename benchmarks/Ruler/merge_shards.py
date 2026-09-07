#!/usr/bin/env python3
"""Merge sharded RULER results into one table.

run_ruler_8k_shard.sh splits the 13 RULER tasks across GPUs, so each shard
writes its own results.json holding only its tasks. This collects them back
into a single {task: {seq_len: score}} view and prints the per-task scores plus
the 13-task mean, which is the number RULER reports.

Usage:
    python benchmarks/Ruler/merge_shards.py results/ruler_8k/<MODEL>/<TAG>
"""

import json
import sys
from pathlib import Path

# RULER's canonical 13-task order; the mean is only comparable when all are in.
TASK_ORDER = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_vt", "ruler_cwe", "ruler_fwe",
    "ruler_qa_squad", "ruler_qa_hotpot",
]


def collect(run_dir: Path):
    """Return ({task: {seq_len: score}}, [shard dirs with no results.json])."""
    scores, missing = {}, []
    for shard in sorted(run_dir.glob("shard*")):
        f = shard / "results.json"
        if not f.exists():
            missing.append(shard.name)
            continue
        data = json.loads(f.read_text())
        for task, metrics in data.get("results", {}).items():
            if task == "ruler":  # lm_eval's aggregate over whatever this shard ran
                continue
            for key, value in metrics.items():
                if key == "alias" or "_stderr" in key or ",none" not in key:
                    continue
                if value == -1:  # lm_eval's "no samples" sentinel
                    continue
                scores.setdefault(task, {})[key.replace(",none", "")] = value
    return scores, missing


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}")
        return 1

    scores, missing = collect(run_dir)
    if not scores:
        print(f"no results found under {run_dir}")
        return 1

    lengths = sorted({L for m in scores.values() for L in m}, key=int)
    header = f"{'task':<18}" + "".join(f"{L:>10}" for L in lengths)
    print(f"\n{run_dir}")
    print(header)
    print("-" * len(header))

    ordered = [t for t in TASK_ORDER if t in scores]
    ordered += [t for t in sorted(scores) if t not in TASK_ORDER]
    for task in ordered:
        row = f"{task:<18}"
        for L in lengths:
            v = scores[task].get(L)
            row += f"{v * 100:>10.2f}" if v is not None else f"{'-':>10}"
        print(row)

    print("-" * len(header))
    for label, keys in (("mean", ordered),):
        row = f"{label:<18}"
        for L in lengths:
            vals = [scores[t][L] for t in keys if L in scores[t]]
            row += f"{sum(vals) / len(vals) * 100:>10.2f}" if vals else f"{'-':>10}"
        print(row)

    absent = [t for t in TASK_ORDER if t not in scores]
    if absent:
        print(f"\nWARNING: {len(absent)}/13 tasks absent, mean is NOT the RULER "
              f"13-task average: {', '.join(absent)}")
    if missing:
        print(f"WARNING: shards with no results.json: {', '.join(missing)}")

    out = run_dir / "merged.json"
    out.write_text(json.dumps(scores, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
