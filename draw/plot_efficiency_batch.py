"""Plot decode throughput and peak memory vs. batch size, at fixed context length (16K).

Reads results/efficiency_16k/. Each batch size has its own subdirectory
(b{B}/, B in {1,2,4,8,16,32,64}), and each subdirectory has one JSON PER
METHOD, all named Llama-3.1-8B-Instruct_{method}.json (no _b{B} suffix --
the subdirectory is what disambiguates batch sizes) -- NOT the merged
Llama-3.1-8B-Instruct_efficiency.json files, which are stale
length-outer-launcher leftovers and must be ignored.
Batch axis runs 1-64 for the throughput/memory panels. A method with
"error": "out_of_memory" (or a missing file) at some batch size has its
line simply stop at the last successful batch size. Draws a single row
of 3 panels (context = 16K):

  - Prefill time vs batch size (grouped bars, one bar per method per batch
    size, truncated at FA2's last successful batch size -- prefill time is
    purely a head-to-head comparison, not a scaling claim, so points where
    the baseline has already OOM'd add no information)
  - Prefill peak memory vs batch size (line)
  - Decode throughput vs batch size (line)

Usage:
    source .venv/bin/activate
    python draw/plot_efficiency_batch.py

Figure is written to draw/speed/.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(HERE, "speed")

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

CONTEXTS = [
    ("efficiency_16k", "16384", "Context = 16K"),
]

# tag in results JSON -> (display label, color, filename stem under results/, dash pattern)
METHODS = [
    ("flash_attention_2", "FlashAttention-2", "#a9c2e8", "flash_attention_2", (1, 0)),
    ("kivi_k2_v2", "KIVI (2-bit)", "#e8829f", "kivi_k2_v2", (4, 2)),
    ("polar", "PolarQuant", "#8a7fc0", "polar_mixed_332", (5, 2, 1, 2)),
    ("oscar_int2", "OScaR (2-bit)", "#2e9998", "oscar_int2", (1, 1)),
    ("picokv_k320_v320_offline", "PiCoKV (ours)", "#474a9d", "picokv_offline_k320_v320", (1, 0)),
]


def load_context(dirname, length_key):
    """For each method, collect (batch, toks_per_s, prefill_ms, peak_gb) for
    successful runs, stopping at the first batch size (if any) where it OOMs
    or is missing."""
    per_method = {tag: {"batch": [], "toks_per_s": [], "prefill_ms": [], "peak_gb": []}
                   for tag, _, _, _, _ in METHODS}
    for tag, _, _, stem, _ in METHODS:
        d = per_method[tag]
        for b in BATCH_SIZES:
            path = os.path.join(RESULTS_DIR, dirname, f"b{b}", f"Llama-3.1-8B-Instruct_{stem}.json")
            entry = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entry = data.get("results", {}).get(length_key)
            if entry is None or entry.get("error"):
                break
            toks = entry.get("decode_toks_per_s")
            prefill = entry.get("prefill_ms")
            peak = entry.get("prefill_peak_gb")
            if toks is None or prefill is None or peak is None:
                break
            d["batch"].append(b)
            d["toks_per_s"].append(toks)
            d["prefill_ms"].append(prefill / 1000)
            d["peak_gb"].append(peak)
    return per_method


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    dirname, length_key, title = CONTEXTS[0]
    per_method = load_context(dirname, length_key)

    ax = axes[0]
    fa2_batches = per_method["flash_attention_2"]["batch"]
    prefill_batches = [b for b in BATCH_SIZES if any(b in per_method[tag]["batch"] for tag, *_ in METHODS)]
    n_methods = len(METHODS)
    bar_w = 0.8 / n_methods
    x = np.arange(len(prefill_batches))
    for i, (tag, label, color, _, _) in enumerate(METHODS):
        d = per_method[tag]
        heights = [d["prefill_ms"][d["batch"].index(b)] if b in d["batch"] else np.nan for b in prefill_batches]
        is_ours = tag == "picokv_k320_v320_offline"
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w * 0.9, color=color, label=label,
               edgecolor="white", linewidth=1.2,
               alpha=1.0 if is_ours else 0.85, zorder=5 if is_ours else 3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in prefill_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel(f"{title}\nPrefill time (s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=1)
    all_heights = [v for tag, *_ in METHODS for v in per_method[tag]["prefill_ms"]]
    y_top = max(all_heights) * 1.8
    ax.set_ylim(top=y_top)
    if fa2_batches and fa2_batches[-1] in prefill_batches:
        boundary_idx = prefill_batches.index(fa2_batches[-1])
        fa2_offset = (0 - (n_methods - 1) / 2) * bar_w
        pico_i = next(i for i, (tag, *_) in enumerate(METHODS) if tag == "picokv_k320_v320_offline")
        pico_offset = (pico_i - (n_methods - 1) / 2) * bar_w
        span_left = x[boundary_idx] + 1 + fa2_offset - bar_w * 0.9 / 2
        span_right = x[-1] + pico_offset + bar_w * 0.9 / 2
        ax.axvspan(span_left, span_right, color="#000000", alpha=0.05, zorder=0)
        ax.text(
            (span_left + span_right) / 2, y_top, "others OOM",
            fontsize=10.5, color="#767676", ha="center", va="top", style="italic",
        )

    ax = axes[1]
    for tag, label, color, _, dash in METHODS:
        d = per_method[tag]
        if not d["batch"]:
            continue
        is_ours = tag == "picokv_k320_v320_offline"
        marker = "s"
        lw = 3.8 if is_ours else 2.6
        zorder = 5 if is_ours else 3
        mfc = "white"
        ax.plot(d["batch"], d["peak_gb"], color=color, linewidth=lw, marker=marker,
                 markersize=8 if is_ours else 7, markerfacecolor=mfc,
                 markeredgecolor=color, markeredgewidth=1.8,
                 label=label, alpha=0.95, zorder=zorder, dashes=dash)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak memory (GB)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.axvspan(fa2_batches[-1], BATCH_SIZES[-1], color="#000000", alpha=0.05, zorder=0)
    ax.text(
        fa2_batches[-1], ax.get_ylim()[1], " others OOM",
        fontsize=10.5, color="#767676", ha="left", va="top", style="italic",
    )
    fa2_mem = per_method["flash_attention_2"]["peak_gb"][-1]
    pico_mem = per_method["picokv_k320_v320_offline"]["peak_gb"][
        per_method["picokv_k320_v320_offline"]["batch"].index(fa2_batches[-1])
    ]
    bx = fa2_batches[-1] * 1.12
    bracket_x = bx * 1.06
    ax.plot([bx, bracket_x, bracket_x, bx], [pico_mem, pico_mem, fa2_mem, fa2_mem],
            color="#d62728", linewidth=1.3, zorder=6, clip_on=False)
    ax.text(
        bracket_x, (pico_mem + fa2_mem) / 2 + (fa2_mem - pico_mem) * 0.22, f"{fa2_mem / pico_mem:.2f}x less",
        fontsize=11, color="#d62728", fontweight="bold", ha="center", va="center", clip_on=False,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=11.5)

    ax = axes[2]
    for tag, label, color, _, dash in METHODS:
        d = per_method[tag]
        if not d["batch"]:
            continue
        is_ours = tag == "picokv_k320_v320_offline"
        marker = "s"
        lw = 3.8 if is_ours else 2.6
        zorder = 5 if is_ours else 3
        mfc = "white"
        ax.plot(d["batch"], d["toks_per_s"], color=color, linewidth=lw, marker=marker,
                 markersize=8 if is_ours else 7, markerfacecolor=mfc,
                 markeredgecolor=color, markeredgewidth=1.8,
                 label=label, alpha=0.95, zorder=zorder, dashes=dash)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.axvspan(fa2_batches[-1], BATCH_SIZES[-1], color="#000000", alpha=0.05, zorder=0)
    ax.text(
        fa2_batches[-1], ax.get_ylim()[1], " others OOM",
        fontsize=10.5, color="#767676", ha="left", va="top", style="italic",
    )
    fa2_toks = per_method["flash_attention_2"]["toks_per_s"][-1]
    pico_toks = per_method["picokv_k320_v320_offline"]["toks_per_s"][
        per_method["picokv_k320_v320_offline"]["batch"].index(fa2_batches[-1])
    ]
    bx = fa2_batches[-1] * 1.12
    bracket_x = bx * 1.06
    ax.plot([bx, bracket_x, bracket_x, bx], [fa2_toks, fa2_toks, pico_toks, pico_toks],
            color="#d62728", linewidth=1.3, zorder=6, clip_on=False)
    ax.text(
        bx * 1.02, (fa2_toks + pico_toks) / 2, f"{pico_toks / fa2_toks:.2f}x",
        fontsize=11, color="#d62728", fontweight="bold", ha="right", va="center", clip_on=False,
    )

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.92, wspace=0.28)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "efficiency_vs_batch")
    fig.savefig(path + ".pdf", dpi=300)
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
