"""Combined 1x4 efficiency figure for the paper (single shared top legend).

  1. Prefill time vs context length (bars, log-y)   [batch = 1]
  2. Decode throughput vs context length (lines)     [batch = 1]
  3. Prefill time vs batch size (bars, log-y)        [context = 16K]
  4. Decode throughput vs batch size (lines)         [context = 16K]

Usage:
    source .venv/bin/activate
    python draw/plot_efficiency_combined.py
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
LENGTH_JSON = os.path.join(RESULTS_DIR, "efficiency_b1", "b1", "Llama-3.1-8B-Instruct_efficiency.json")
BATCH_DIR = "efficiency_16k"
BATCH_LENGTH_KEY = "16384"
OUTDIR = os.path.join(HERE, "speed")
BASELINE_TAG = "flash_attention_2"
PICOKV_TAG = "picokv_k320_v320_offline"
BATCH_SIZES = [1, 2, 4, 8, 16]

# tag in results JSON -> (label, color, batch-file stem, dash pattern)
METHODS = [
    ("flash_attention_2", "FlashAttention-2", "#a9c2e8", "flash_attention_2", (1, 0)),
    ("kivi_k2_v2", "KIVI (2-bit)", "#e8829f", "kivi_k2_v2", (4, 2)),
    ("polar", "PolarQuant", "#8a7fc0", "polar_mixed_332", (5, 2, 1, 2)),
    ("oscar_int2", "OScaR (2-bit)", "#2e9998", "oscar_int2", (1, 1)),
    (PICOKV_TAG, "PiCoKV (ours)", "#474a9d", "picokv_offline_k320_v320", (1, 0)),
]


def load_length():
    with open(LENGTH_JSON, encoding="utf-8") as f:
        data = json.load(f)
    keys = ["prefill_ms", "decode_toks_per_s"]
    per = {tag: {"len": [], **{k: [] for k in keys}} for tag, *_ in METHODS}
    for length in sorted(data["lengths"].keys(), key=int):
        entry = data["lengths"][length]
        for tag, *_ in METHODS:
            m = entry.get(tag, {})
            vals = {k: m.get(k) for k in keys}
            if any(v is None for v in vals.values()) or not 16 * 1024 <= int(length) <= 256 * 1024:
                continue
            per[tag]["len"].append(int(length))
            for k, v in vals.items():
                per[tag][k].append(v / 1000 if k == "prefill_ms" else v)
    return per


def load_batch():
    per = {tag: {"batch": [], "toks_per_s": [], "prefill_ms": []} for tag, *_ in METHODS}
    for tag, _, _, stem, _ in METHODS:
        d = per[tag]
        for b in BATCH_SIZES:
            path = os.path.join(RESULTS_DIR, BATCH_DIR, f"b{b}", f"Llama-3.1-8B-Instruct_{stem}.json")
            entry = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entry = data.get("results", {}).get(BATCH_LENGTH_KEY)
            if entry is None or entry.get("error"):
                break
            toks = entry.get("decode_toks_per_s")
            prefill = entry.get("prefill_ms")
            if toks is None or prefill is None:
                break
            d["batch"].append(b)
            d["toks_per_s"].append(toks)
            d["prefill_ms"].append(prefill / 1000)
    return per


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2

    lper = load_length()
    bper = load_batch()
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.4))

    n = len(METHODS)
    bar_w = 0.8 / n

    # Panel 1: prefill time vs context length (bars) -----------------------
    ax = axes[0]
    fa2_lengths = lper[BASELINE_TAG]["len"]
    all_lengths = sorted(set(l for d in lper.values() for l in d["len"]))
    prefill_lengths = [l for l in all_lengths if l in fa2_lengths]
    x = np.arange(len(prefill_lengths))
    for i, (tag, label, color, _, _) in enumerate(METHODS):
        d = lper[tag]
        heights = [d["prefill_ms"][d["len"].index(l)] if l in d["len"] else np.nan for l in prefill_lengths]
        is_ours = tag == PICOKV_TAG
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w * 0.9, color=color, label=label,
               edgecolor="white", linewidth=1.2,
               alpha=1.0 if is_ours else 0.85, zorder=5 if is_ours else 3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l // 1024}K" for l in prefill_lengths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Prefill time (s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)

    # Panel 2: decode throughput vs context length (lines) -----------------
    ax = axes[1]
    for tag, label, color, _, dash in METHODS:
        d = lper[tag]
        if not d["len"]:
            continue
        is_ours = tag == PICOKV_TAG
        ax.plot(d["len"], d["decode_toks_per_s"], color=color, linewidth=3.8 if is_ours else 2.6,
                marker="s", markersize=8 if is_ours else 7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.8, label=label,
                alpha=0.95, zorder=5 if is_ours else 3, dashes=dash)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_lengths)
    ax.set_xticklabels([f"{l // 1024}K" for l in all_lengths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    boundary = fa2_lengths[-1]
    fa2_toks = lper[BASELINE_TAG]["decode_toks_per_s"][lper[BASELINE_TAG]["len"].index(boundary)]
    pico_toks = lper[PICOKV_TAG]["decode_toks_per_s"][lper[PICOKV_TAG]["len"].index(boundary)]
    ax.set_xlim(None, boundary * 1.35)
    bx = boundary * 1.12
    bracket_x = bx * 1.06
    ax.plot([bx, bracket_x, bracket_x, bx], [fa2_toks, fa2_toks, pico_toks, pico_toks],
            color="#d62728", linewidth=1.3, zorder=6)
    ax.text(bx * 1.02, (fa2_toks + pico_toks) / 2, f"{pico_toks / fa2_toks:.2f}x",
            fontsize=11, color="#d62728", fontweight="bold", ha="right", va="center")

    # Panel 3: prefill time vs batch size (bars) ---------------------------
    ax = axes[2]
    fa2_batches = bper[BASELINE_TAG]["batch"]
    prefill_batches = [b for b in BATCH_SIZES if any(b in bper[tag]["batch"] for tag, *_ in METHODS)]
    x = np.arange(len(prefill_batches))
    for i, (tag, label, color, _, _) in enumerate(METHODS):
        d = bper[tag]
        heights = [d["prefill_ms"][d["batch"].index(b)] if b in d["batch"] else np.nan for b in prefill_batches]
        is_ours = tag == PICOKV_TAG
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w * 0.9, color=color,
               edgecolor="white", linewidth=1.2,
               alpha=1.0 if is_ours else 0.85, zorder=5 if is_ours else 3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in prefill_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Prefill time (s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    all_heights = [v for tag, *_ in METHODS for v in bper[tag]["prefill_ms"]]
    ax.set_ylim(top=max(all_heights) * 1.8)

    # Panel 4: decode throughput vs batch size (lines) ---------------------
    ax = axes[3]
    for tag, label, color, _, dash in METHODS:
        d = bper[tag]
        if not d["batch"]:
            continue
        is_ours = tag == PICOKV_TAG
        ax.plot(d["batch"], d["toks_per_s"], color=color, linewidth=3.8 if is_ours else 2.6,
                marker="s", markersize=8 if is_ours else 7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.8, label=label,
                alpha=0.95, zorder=5 if is_ours else 3, dashes=dash)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    fa2_bt = bper[BASELINE_TAG]["toks_per_s"][-1]
    pico_bt = bper[PICOKV_TAG]["toks_per_s"][bper[PICOKV_TAG]["batch"].index(fa2_batches[-1])]
    ax.set_xlim(None, fa2_batches[-1] * 1.35)
    bx = fa2_batches[-1] * 1.12
    bracket_x = bx * 1.06
    ax.plot([bx, bracket_x, bracket_x, bx], [fa2_bt, fa2_bt, pico_bt, pico_bt],
            color="#d62728", linewidth=1.3, zorder=6, clip_on=False)
    ax.text(bx * 1.02, (fa2_bt + pico_bt) / 2, f"{pico_bt / fa2_bt:.2f}x",
            fontsize=11, color="#d62728", fontweight="bold", ha="right", va="center", clip_on=False)

    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.20, top=0.88, wspace=0.26)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(METHODS), frameon=False, fontsize=14, columnspacing=1.8,
               handlelength=2.6)

    def pair_center(a, b):
        return (axes[a].get_position().x0 + axes[b].get_position().x1) / 2

    fig.text(pair_center(0, 1), 0.03, "Batch size = 1", ha="center", va="bottom",
             fontsize=15, fontweight="bold")
    fig.text(pair_center(2, 3), 0.03, "Context = 16K", ha="center", va="bottom",
             fontsize=15, fontweight="bold")
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "efficiency_combined")
    fig.savefig(path + ".pdf", dpi=300)
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
