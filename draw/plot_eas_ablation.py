"""EAS ablation: average RULER score vs eas, online vs offline, two models.

Usage:
    python draw/plot_eas_ablation.py
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(HERE, "abla")
LEN_KEY = "131072,none"

MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]

# eas value -> results.json path template ({model}, {mode})
EAS_POINTS = [
    (1e-7, "test_eas/{model}/picokv_{mode}_k320_v320_eas1e-7", "w/o EAS"),
    (1e-4, "test_eas/{model}/picokv_{mode}_k320_v320_eas0.0001", r"$10^{-4}$"),
    (1e-3, "test_eas/{model}/picokv_{mode}_k320_v320_eas0.001", r"$10^{-3}$"),
    (5e-3, "test_eas/{model}/picokv_{mode}_k320_v320_eas0.005", r"$5{\times}10^{-3}$"),
    (1e-2, "ruler/{model}/picokv_{mode}_k320_v320", r"$10^{-2}$"),  # baseline eas=0.01
    (2e-2, "test_eas/{model}/picokv_{mode}_k320_v320_eas0.02", r"$2{\times}10^{-2}$"),
]

MODES = [
    ("online", "Online", "#474a9d", (1, 0), "o"),
    ("offline", "Offline", "#e8829f", (4, 2), "s"),
]


def avg_score(path):
    with open(os.path.join(RESULTS, path, "results.json"), encoding="utf-8") as f:
        res = json.load(f)["results"]
    vals = [v[LEN_KEY] for v in res.values()]
    return sum(vals) / len(vals)


def collect(model, mode):
    """Return evenly-spaced x indices and scores (skip missing points)."""
    xs, ys = [], []
    for i, (_, tmpl, _) in enumerate(EAS_POINTS):
        p = tmpl.format(model=model, mode=mode)
        try:
            ys.append(avg_score(p))
            xs.append(i)
        except FileNotFoundError:
            continue
    return xs, ys


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    titles = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct", "Qwen3-8B": "Qwen3-8B"}

    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        for mode, label, color, dash, marker in MODES:
            xs, ys = collect(model, mode)
            ax.plot(xs, ys, color=color, linewidth=3.0, marker=marker, markersize=9,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8,
                    label=label, dashes=dash, zorder=5)
        ax.set_xticks(range(len(EAS_POINTS)))
        ax.set_xticklabels([lab for *_, lab in EAS_POINTS])
        if i == 0:
            ax.set_ylabel("Avg. RULER score")
        ax.set_title(titles[model], fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=13, length=6, width=1.5)
        ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.25, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=18, columnspacing=2.0, handlelength=2.6)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.84, wspace=0.24)
    fig.text(0.5, 0.035, "(a) Error-aware selection (EAS)", ha="center", va="bottom",
             fontsize=18, fontweight="bold")

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "eas_ablation")
    fig.savefig(path + ".pdf", dpi=300)
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
