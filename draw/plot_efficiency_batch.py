"""Plot decode throughput and peak memory vs. batch size, at fixed context length (16K).

Reads results/efficiency_b_16k/. Each dir has one JSON PER METHOD PER BATCH SIZE
(Llama-3.1-8B-Instruct_{method}[_b{B}].json, B in {2,4,8,16,32,64}, plus the
suffix-less file for batch_size=1) -- NOT the merged
Llama-3.1-8B-Instruct_efficiency[_b{B}].json files, which are stale
length-outer-launcher leftovers that go stale past b8 and must be ignored.
Batch axis runs 1-32 for the throughput/memory panels. A method with
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

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(HERE, "speed")

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

CONTEXTS = [
    ("efficiency_b_16k", "16384", "Context = 16K"),
]

# tag in results JSON -> (display label, color, filename stem under results/, dash pattern, hatch)
METHODS = [
    ("flash_attention_2", "FlashAttention-2", "#A6A6A6", "flash_attention_2", (1, 0), "//"),
    ("kivi_k2_v2", "KIVI (2-bit)", "#E9A6A1", "kivi_k2_v2", (4, 2), "\\\\"),
    ("oscar_int2", "OScaR (2-bit)", "#8BCF8B", "oscar_int2", (1, 1), "//"),
    ("polar", "PolarQuant", "#9A4D8E", "polar_mixed_332", (5, 2, 1, 2), "\\\\"),
    ("picokv_k320_v320_offline", "PiCoKV (ours)", "#0F4D92", "picokv_offline_k320_v320", (1, 0), "//"),
]


def suffix_for(batch_size):
    return "" if batch_size == 1 else f"_b{batch_size}"


def load_context(dirname, length_key):
    """For each method, collect (batch, toks_per_s, prefill_ms, peak_gb) for
    successful runs, stopping at the first batch size (if any) where it OOMs
    or is missing."""
    per_method = {tag: {"batch": [], "toks_per_s": [], "prefill_ms": [], "peak_gb": []}
                   for tag, _, _, _, _, _ in METHODS}
    for tag, _, _, stem, _, _ in METHODS:
        d = per_method[tag]
        for b in BATCH_SIZES:
            path = os.path.join(RESULTS_DIR, dirname, f"Llama-3.1-8B-Instruct_{stem}{suffix_for(b)}.json")
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    dirname, length_key, title = CONTEXTS[0]
    per_method = load_context(dirname, length_key)

    ax = axes[0]
    fa2_batches = per_method["flash_attention_2"]["batch"]
    prefill_batches = [b for b in BATCH_SIZES if b in fa2_batches]
    n_methods = len(METHODS)
    bar_w = 0.8 / n_methods
    x = np.arange(len(prefill_batches))
    for i, (tag, label, color, _, _, hatch) in enumerate(METHODS):
        d = per_method[tag]
        heights = [d["prefill_ms"][d["batch"].index(b)] if b in d["batch"] else np.nan for b in prefill_batches]
        is_ours = tag == "picokv_k320_v320_offline"
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, heights, width=bar_w * 0.9, color=color, label=label,
               edgecolor="white", linewidth=1.2, hatch=hatch,
               alpha=1.0 if is_ours else 0.85, zorder=5 if is_ours else 3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in prefill_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel(f"{title}\nPrefill time (s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=1)

    ax = axes[1]
    for tag, label, color, _, dash, _ in METHODS:
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
    ax.axvline(fa2_batches[-1], color="#A6A6A6", linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak memory (GB)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.text(
        fa2_batches[-1], ax.get_ylim()[1], "others OOM ",
        fontsize=10.5, color="#767676", ha="right", va="top", style="italic",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=11.5)

    ax = axes[2]
    for tag, label, color, _, dash, _ in METHODS:
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
    ax.axvline(fa2_batches[-1], color="#A6A6A6", linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCH_SIZES)
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)
    ax.text(
        fa2_batches[-1], ax.get_ylim()[1], "others OOM ",
        fontsize=10.5, color="#767676", ha="right", va="top", style="italic",
    )

    fig.tight_layout(pad=2)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "efficiency_vs_batch")
    fig.savefig(path + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
