"""Plot prefill time, prefill peak memory, and decode throughput vs. context length.

Reads results/efficiency-1-1/Llama-3.1-8B-Instruct_efficiency.json (batch_size=1,
lengths 8K-512K) and draws three side-by-side panels comparing PiCoKV against
the dense (flash_attention_2) and KV-quant baselines (KIVI, OScaR, PolarQuant):

  - Prefill time (ms), grouped bars (one bar per method per length), truncated
    at FA2's last successful length (512K is PiCoKV-only, an OOM comparison
    point rather than a head-to-head data point).
  - Peak memory (GB), from `prefill_peak_gb`
  - Decode throughput (tokens/s), log-y

Usage:
    source .venv/bin/activate
    python draw/plot_efficiency_length.py

Figure is written to draw/speed/.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(
    HERE, "..", "results", "efficiency-1-1", "Llama-3.1-8B-Instruct_efficiency.json"
)
OUTDIR = os.path.join(HERE, "speed")
BASELINE_TAG = "flash_attention_2"
PICOKV_TAG = "picokv_k320_v320_offline"

# tag in results JSON -> (display label, color, dash pattern, hatch)
METHODS = [
    ("flash_attention_2", "FlashAttention-2", "#A6A6A6", (1, 0), "//"),
    ("kivi_k2_v2", "KIVI (2-bit)", "#E9A6A1", (4, 2), "\\\\"),
    ("oscar_int2", "OScaR (2-bit)", "#8BCF8B", (1, 1), "//"),
    ("polar", "PolarQuant", "#9A4D8E", (5, 2, 1, 2), "\\\\"),
    (PICOKV_TAG, "PiCoKV (ours)", "#0F4D92", (1, 0), "//"),
]

METRICS = [
    ("prefill_peak_gb", "Peak memory (GB)", False),
    ("decode_toks_per_s", "Decode throughput (tokens/s)", False),
]


def load_results():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    lengths = sorted(data["lengths"].keys(), key=int)
    all_keys = ["prefill_ms"] + [key for key, _, _ in METRICS]
    per_method = {tag: {"len": [], **{key: [] for key in all_keys}} for tag, _, _, _, _ in METHODS}
    for length in lengths:
        entry = data["lengths"][length]
        for tag, _, _, _, _ in METHODS:
            m = entry.get(tag, {})
            values = {key: m.get(key) for key in all_keys}
            if any(v is None for v in values.values()):
                continue
            per_method[tag]["len"].append(int(length))
            for key, v in values.items():
                per_method[tag][key].append(v / 1000 if key == "prefill_ms" else v)
    return per_method


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 17
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2.2

    per_method = load_results()
    all_lengths = sorted(set(l for d in per_method.values() for l in d["len"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    fa2_lengths = per_method[BASELINE_TAG]["len"]
    prefill_lengths = [l for l in all_lengths if l in fa2_lengths]
    n_methods = len(METHODS)
    bar_w = 0.8 / n_methods
    x = np.arange(len(prefill_lengths))
    for i, (tag, label, color, _, hatch) in enumerate(METHODS):
        d = per_method[tag]
        heights = [d["prefill_ms"][d["len"].index(l)] if l in d["len"] else np.nan for l in prefill_lengths]
        is_ours = tag == PICOKV_TAG
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w * 0.9, color=color, label=label,
               edgecolor="white", linewidth=1.2, hatch=hatch,
               alpha=1.0 if is_ours else 0.85, zorder=5 if is_ours else 3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l // 1024}K" for l in prefill_lengths])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Batch size = 1\nPrefill time (s)")
    ax.tick_params(labelsize=14, length=6, width=1.6)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=1)

    for ax, (key, ylabel, logy) in zip(axes[1:], METRICS):
        for tag, label, color, dash, _ in METHODS:
            d = per_method[tag]
            if not d["len"]:
                continue
            is_ours = tag == PICOKV_TAG
            marker = "s"
            lw = 3.8 if is_ours else 2.6
            zorder = 5 if is_ours else 3
            mfc = "white"
            ax.plot(d["len"], d[key], color=color, linewidth=lw, marker=marker,
                     markersize=8 if is_ours else 7, markerfacecolor=mfc,
                     markeredgecolor=color, markeredgewidth=1.8,
                     label=label, alpha=0.95, zorder=zorder, dashes=dash)
        ax.axvline(fa2_lengths[-1], color="#A6A6A6", linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xlabel("Context length")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_lengths)
        ax.set_xticklabels([f"{l // 1024}K" for l in all_lengths], rotation=0)
        ax.tick_params(labelsize=14, length=6, width=1.6)
        ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)

    axes[1].text(
        fa2_lengths[-1], axes[1].get_ylim()[1], "others OOM ",
        fontsize=10.5, color="#767676", ha="right", va="top", style="italic",
    )
    axes[1].legend(loc="upper left", frameon=False, fontsize=11.5)
    axes[2].text(
        fa2_lengths[-1], axes[2].get_ylim()[1], "others OOM ",
        fontsize=10.5, color="#767676", ha="right", va="top", style="italic",
    )

    fig.tight_layout(pad=2)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "efficiency_vs_length")
    fig.savefig(path + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
