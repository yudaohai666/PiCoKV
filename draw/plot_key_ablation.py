"""Key-reconstruction ablation: average RULER score vs zerorope, dense vs key.

Usage:
    python draw/plot_key_ablation.py
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(HERE, "abla")
LEN_KEY = "131072,none"

MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]

# zerorope -> dir suffix for each variant
ZR_POINTS = [
    (0, "zr0"),
    (10, "zr10"),
    (20, ""),   # base dir (no zr suffix) uses zerorope=20
    (30, "zr30"),
]

# variant -> (base dir stem, label, color, dash, marker)
VARIANTS = [
    ("dense_zr", "Dense", "#474a9d", (1, 0), "o"),
    ("key", "Key-only", "#2e9998", (4, 2), "s"),
]


def avg_score(model, stem, zr_suffix):
    name = f"picokv_recon_online_k320_v320_{stem}"
    if zr_suffix:
        name += f"_{zr_suffix}"
    path = os.path.join(RESULTS, "test_key", model, name, "results.json")
    with open(path, encoding="utf-8") as f:
        res = json.load(f)["results"]
    vals = [v[LEN_KEY] for v in res.values()]
    return sum(vals) / len(vals)


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        for stem, label, color, dash, marker in VARIANTS:
            xs, ys = [], []
            for zr, suffix in ZR_POINTS:
                try:
                    ys.append(avg_score(model, stem, suffix))
                    xs.append(zr)
                except FileNotFoundError:
                    continue
            ax.plot(xs, ys, color=color, linewidth=3.0, marker=marker, markersize=9,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8,
                    label=label, dashes=dash, zorder=5)
        if i == 0:
            ax.set_ylabel("Avg. RULER score")
        ax.set_title(model, fontsize=16, fontweight="bold")
        ax.set_xticks([z for z, _ in ZR_POINTS])
        ax.tick_params(labelsize=13, length=6, width=1.5)
        ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=18, columnspacing=2.0, handlelength=2.6)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.84, wspace=0.24)
    fig.text(0.5, 0.035, "(b) High-frequency masking (HFM)", ha="center", va="bottom",
             fontsize=18, fontweight="bold")

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "key_ablation")
    fig.savefig(path + ".pdf", dpi=300)
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
