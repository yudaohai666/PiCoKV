"""Figure: (a) Performance-vs-Speedup scatter + (b) decode throughput bars.

Both panels use Llama-3.1-8B-Instruct (only model with efficiency data).
Performance = avg RULER score at 128K (x100). Speedup = decode throughput
at 128K relative to FlashAttention-2. Methods/colors follow
plot_efficiency_combined.py; PiCoKV = offline (matches efficiency_b1).

Usage:
    python draw/plot_ruler_radar_throughput.py
"""

import json
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
EFF_B1 = os.path.join(RESULTS, "efficiency", "b1")
OUTDIR = os.path.join(HERE, "abla")
MODEL = "Llama-3.1-8B-Instruct"
LEN128 = "131072,none"

# efficiency key -> per-method result file in results/efficiency/b1
EFF_FILES = {
    "flash_attention_2": f"{MODEL}_flash_attention_2.json",
    "kivi_k2_v2": f"{MODEL}_kivi_k2_v2.json",
    "polar": f"{MODEL}_polar_mixed_332.json",
    "oscar_int2": f"{MODEL}_oscar_int2.json",
    "picokv_k320_v320_offline": f"{MODEL}_picokv_offline_k320_v320_eas0.0000001.json",
}

# (ruler results.json rel path, efficiency key, label, color, is_ours)
# PiCoKV perf = online RULER; throughput = measured offline value (decode cost
# is identical for online/offline, projections happen at prefill).
METHODS = [
    ("dense_flash_attention_2/results.json", "flash_attention_2", "FlashAttention-2", "#8fb3d9", False),
    ("kivi_k2_v2/partB/results.json", "kivi_k2_v2", "KIVI", "#d98fa8", False),
    ("polar_mixed_332/partB/results.json", "polar", "PolarQuant", "#9b8fce", False),
    ("oscar_int2/results.json", "oscar_int2", "OScaR", "#5fae9f", False),
    ("picokv_online_k320_v320/results.json", "picokv_k320_v320_offline", "PiCoKV", "#3d4194", True),
]

# marker per label
MARKERS = {
    "FlashAttention-2": "o", "KIVI": "o", "PolarQuant": "o", "OScaR": "o",
    "PiCoKV": "*",
}

TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_vt", "ruler_cwe", "ruler_fwe",
    "ruler_qa_squad", "ruler_qa_hotpot",
]

LENGTHS = [("16384", "16K"), ("32768", "32K"), ("65536", "64K"),
           ("131072", "128K"), ("262144", "256K")]

# Panel (b) has several drawable forms (see the draw_* docstrings); this is the
# one shipped in the paper, and the only one that writes the canonical file name.
DEFAULT_PANEL_B = "gblog"

# Sub-caption size, matched to plot_groupbits_reconstruction.py's 15 pt *as
# printed*, not as declared. That figure is 19.2 in wide against this one's 13,
# so at a shared \textwidth this figure is magnified 19.2/13 = 1.48x more and an
# identical point size would come out half again as large. Scaling by the width
# ratio is what makes the two captions the same height on the page.
CAPTION_FONTSIZE = 15.0 * 13.0 / 19.2

# manual label offsets (dx, dy) in data units, per method label
LABEL_OFF = {
    "FlashAttention-2": (0.0, 0.12),
    "KIVI": (0.0, -0.15),
    "PolarQuant": (0.0, 0.12),
    "OScaR": (0.0, 0.15),
    "PiCoKV": (0.0, 0.16),
}


def avg_perf(rel):
    res = json.load(open(os.path.join(RESULTS, "ruler", MODEL, rel)))["results"]
    return 100.0 * sum(res[t][LEN128] for t in TASKS) / len(TASKS)


def eff_decode(ekey):
    """decode_toks_per_s per prompt-length key for one method (from results/efficiency/b1)."""
    res = json.load(open(os.path.join(EFF_B1, EFF_FILES[ekey])))["results"]
    return {L: res[L]["decode_toks_per_s"] for L, _ in LENGTHS}


def kv_usage(ekey):
    """prefill_kv_gb per prompt-length key for one method (from results/efficiency/b1)."""
    res = json.load(open(os.path.join(EFF_B1, EFF_FILES[ekey])))["results"]
    return {L: res[L]["prefill_kv_gb"] for L, _ in LENGTHS}


def draw_scatter(ax):
    L128 = "131072"
    decode = {ekey: eff_decode(ekey)[L128] for _, ekey, _, _, _ in METHODS}
    base = decode[METHODS[0][1]]
    pts = []
    for rel, ekey, label, color, is_ours in METHODS:
        x = avg_perf(rel)
        y = decode[ekey] / base
        pts.append((x, y, label, color, is_ours))

    flash_x = pts[0][0]
    flash_c = METHODS[0][3]
    base_line_c = "#5f8bbf"  # darker flash tone for baseline lines/text
    ax.axhline(1.0, color=flash_c, linestyle="--", linewidth=1.6, zorder=1)
    ax.axvline(flash_x, color=flash_c, linestyle="--", linewidth=1.6, zorder=1)

    handles = []
    for x, y, label, color, is_ours in pts:
        m = MARKERS[label]
        s = 560 if m == "*" else (200 if m == "D" else 170)
        h = ax.scatter(x, y, s=s, marker=m, color=color, edgecolor="white",
                       linewidth=1.4, zorder=6 if is_ours else 5, label=label)
        handles.append(h)
        dx, dy = LABEL_OFF[label]
        ax.annotate(label, (x, y), (x + dx, y + dy), fontsize=12,
                    fontweight="bold" if is_ours else "normal",
                    color=color, ha="center", va="bottom" if dy > 0 else "top",
                    zorder=7)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xlo, xhi = min(xs) - 2.0, max(xs) + 2.0
    ylo, yhi = min(ys) - 0.25, max(ys) + 0.48
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.annotate("", xy=(xhi - 0.5, yhi - 0.25), xytext=(xlo + 0.8, ylo + 0.15),
                arrowprops=dict(arrowstyle="->", color="#adadad", lw=3, alpha=0.85),
                zorder=2)
    ang = np.degrees(np.arctan2((yhi - 0.25) - (ylo + 0.15), (xhi - 0.5) - (xlo + 0.8)))
    ax.text(xlo + 4.5, ylo + 0.62, "Better", color="#7c7c7c", fontsize=15,
            fontweight="bold", rotation=ang, rotation_mode="anchor", zorder=3)
    ax.text(xhi - 0.3, 1.0, "1.0x", color=base_line_c, fontsize=12, ha="right",
            va="bottom", fontweight="bold")
    ax.set_xlabel("Performance (avg. RULER, 128K)")
    ax.set_ylabel("Speedup")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, linewidth=0.6, alpha=0.2, zorder=0)
    return handles


def gb_note():
    """The absolute scale in one line, from the same JSONs that feed the ratios.

    Both memory panels plot a ratio, which is the readable form but drops the
    magnitude a deployment is actually limited by; this restores it as text
    instead of spending an axis on it.
    """
    key = LENGTHS[-1][0]
    return (f"At {LENGTHS[-1][1]}: {kv_usage('flash_attention_2')[key]:.1f} GB "
            f"$\\rightarrow$ {kv_usage(METHODS[-1][1])[key]:.1f} GB")


def draw_ratio_lines(ax):
    """KV-memory reduction against FA-2, with the y range spent where the data is.

    FA-2 itself is not drawn: it is the denominator, so its line is 1.0 by
    definition at every length and would claim the bottom 60% of the height
    without carrying information. It is stated in the caption instead, which also
    frees the axis to cover only the range the measurements occupy.

    The three 2-bit baselines sit within 8% of each other (4.92-5.33x) and are
    flat in length by construction, so they are annotated once as a band rather
    than distinguished; the readable signal is PiCoKV climbing out of that band as
    the fixed sink+local bytes are amortized over a longer context.
    """
    x = np.arange(len(LENGTHS))
    base_kv = kv_usage("flash_attention_2")
    band, every = [], []
    for _, ekey, label, color, is_ours in METHODS:
        if ekey == "flash_attention_2":
            continue
        kv = kv_usage(ekey)
        vals = [base_kv[L] / kv[L] for L, _ in LENGTHS]
        every += vals
        if not is_ours:
            band += vals
        m = MARKERS[label]
        ax.plot(x, vals, marker=m, color=color, label=label,
                linewidth=2.4, markersize=15 if m == "*" else 8,
                markeredgecolor="white", markeredgewidth=1.2,
                zorder=6 if is_ours else 5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.3, len(LENGTHS) - 0.7)
    # Limits and both annotation anchors are derived from the data, so changing
    # LENGTHS (e.g. 8K-128K to 16K-256K) does not strand text on top of a curve.
    span = max(every) - min(every)
    y0, y1 = min(every) - 0.15 * span, max(every) + 0.18 * span
    ax.set_ylim(y0, y1)
    # Shade the baselines' envelope and label it once, *inside* the band: with
    # three lines to choose from an arrow reads as naming one method rather than
    # the group, and below the band there is no longer room for the text.
    lo, hi = min(band), max(band)
    ax.axhspan(lo, hi, color="0.5", alpha=0.10, linewidth=0, zorder=1)
    ax.text((len(LENGTHS) - 1) / 2, (lo + hi) / 2,
            "2-bit baselines $\\approx\\!5\\times$", fontsize=13,
            color="#6f6f6f", ha="center", va="center", zorder=7)
    # Upper-left: the band is below it and ours only reaches this height at the
    # right-hand end of the axis.
    ax.text(-0.22, y1 - 0.02 * (y1 - y0), gb_note(), ha="left", va="top",
            fontsize=13, color="0.35", zorder=7)
    ax.set_ylabel("Memory reduction vs. FA-2 ($\\times$)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def draw_ratio_bars(ax):
    """Same measurements as ``draw_ratio_lines``, grouped bars instead of curves.

    Bars must start at 0 -- a truncated bar misstates a ratio -- which costs the
    resolution the line version buys by starting at 3.8. What it buys back: FA-2
    can be a bar of its own instead of a caption footnote, so the unit of
    comparison is visible rather than implied; nothing suggests interpolation
    between the five measured lengths; and per-length comparison is a direct
    height read. The crossing still shows: PiCoKV is the shortest compressed bar
    at 8K and the tallest from 32K on.
    """
    x = np.arange(len(LENGTHS))
    base_kv = kv_usage("flash_attention_2")
    width = 0.80 / len(METHODS)
    for i, (_, ekey, label, color, is_ours) in enumerate(METHODS):
        kv = kv_usage(ekey)
        vals = [base_kv[L] / kv[L] for L, _ in LENGTHS]
        off = (i - (len(METHODS) - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.90, color=color, label=label,
               edgecolor="white", linewidth=0.8, zorder=5)
        if is_ours:
            # Only ours is labelled: bars invite reading exact values, and 25 of
            # them would bury the one series the panel is about.
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.08, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=11, color=color, fontweight="bold", zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.55, len(LENGTHS) - 0.45)
    peak = max(base_kv[L] / kv_usage(METHODS[-1][1])[L] for L, _ in LENGTHS)
    ax.set_ylim(0, peak * 1.13)
    # The absolute scale in one line, in the empty upper-left. A GB axis would
    # buy the same fact and lose everything else: linear GB flattens the short
    # contexts to nothing and makes the four compressed bars at the longest one
    # (2.6-3.25 GB at 128K) indistinguishable, which is the resolution this panel
    # exists for.
    ax.text(-0.45, peak * 1.11, gb_note(), ha="left", va="top", fontsize=13,
            color="0.35", zorder=7)
    ax.set_ylabel("Memory reduction vs. FA-2 ($\\times$)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def draw_both(ax):
    """Memory and decode speed in one panel, both as ratios to FA-2 on a log axis.

    The two quantities have incompatible ranges (memory 4.9-6.3x, speed
    0.48-3.45x), so a shared linear axis would flatten one of them and a twin
    axis would put two unrelated scales on one frame. Expressing both as
    "improvement over FA-2" and taking log y makes one axis legitimate: the
    y = 1 rule then reads correctly for both families at once -- above it the
    cache is smaller *and* decoding is faster.

    Line style carries the quantity (solid = memory, dashed = speed, hollow
    markers), colour carries the method, so no entry is added to the shared
    legend. The empty gap between the two families (3.5x to 4.9x) is where the
    family labels go.

    What this costs, and why the two-panel version still exists: squeezing
    4.9-6.3x into a log decade collapses the memory family back into a band, so
    PiCoKV's rise from 4.95x to 6.28x is visible but no longer readable to two
    digits. What it buys: decode speed gets the context-length axis too, which
    exposes that KIVI and PolarQuant decode *slower* than dense FA-2 at every
    length (0.48-0.83x) -- a fact panel (a) only shows at 128K.
    """
    x = np.arange(len(LENGTHS))
    base_kv = kv_usage("flash_attention_2")
    base_dec = eff_decode("flash_attention_2")
    for _, ekey, label, color, is_ours in METHODS:
        if ekey == "flash_attention_2":
            continue
        kv, dec = kv_usage(ekey), eff_decode(ekey)
        mem = [base_kv[L] / kv[L] for L, _ in LENGTHS]
        spd = [dec[L] / base_dec[L] for L, _ in LENGTHS]
        m = MARKERS[label]
        big, small = (15, 11) if m == "*" else (8, 7)
        ax.plot(x, mem, color=color, marker=m, linestyle="-", linewidth=2.4,
                markersize=big, markeredgecolor="white", markeredgewidth=1.2,
                zorder=6 if is_ours else 5)
        ax.plot(x, spd, color=color, marker=m, linestyle="--", linewidth=2.0,
                markersize=small, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.8,
                zorder=6 if is_ours else 5)

    ax.axhline(1.0, color=METHODS[0][3], linestyle=":", linewidth=1.8, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.40, 8.0)
    ax.set_yticks([0.5, 1, 2, 3, 5, 7])
    ax.set_yticklabels(["0.5", "1", "2", "3", "5", "7"])
    ax.minorticks_off()
    ax.text(-0.22, 4.25, "KV cache reduction (solid)", fontsize=12.5,
            color="0.35", ha="left", va="top", zorder=7)
    ax.text(-0.22, 1.95, "Decode speedup (dashed)", fontsize=12.5,
            color="0.35", ha="left", va="top", zorder=7)
    ax.text(len(LENGTHS) - 1.05, 1.03, "FA-2", fontsize=12, color="#5f8bbf",
            ha="right", va="bottom", fontweight="bold", zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.3, len(LENGTHS) - 0.7)
    ax.set_ylabel("Improvement over FA-2 ($\\times$)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def draw_gb_ratio(ax):
    """Absolute GB as bars, compression ratio as the label on our own bars.

    Both quantities in one frame without a second scale: bar height is GB on a
    linear axis (so FA-2's 32 GB against 5.1 GB is the shape of the panel), and
    the ratio -- which is just a comparison *within* each group -- is printed on
    the bar it belongs to. A twin ratio axis was tried and abandoned: the ratio
    curves live at 4.9-6.3x, and wherever that band is mapped on the right axis
    it crosses either the two tall FA-2 bars or the compressed cluster.

    The baselines' ratios are not labelled per bar: they are flat in length by
    construction and within 8% of each other, so one grey line of text carries
    them and keeps the 25 bars readable.
    """
    x = np.arange(len(LENGTHS))
    base_kv = kv_usage("flash_attention_2")
    width = 0.80 / len(METHODS)
    peak = max(base_kv[L] for L, _ in LENGTHS)
    band = []
    for i, (_, ekey, label, color, is_ours) in enumerate(METHODS):
        kv = kv_usage(ekey)
        vals = [kv[L] for L, _ in LENGTHS]
        off = (i - (len(METHODS) - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.90, color=color, label=label,
               edgecolor="white", linewidth=0.8, zorder=5)
        if ekey == "flash_attention_2":
            # The reference magnitude, once, at the length that motivates the
            # work; every other group repeats the same 2x-per-doubling story.
            ax.text(x[-1] + off, vals[-1] + 0.015 * peak, f"{vals[-1]:.0f} GB",
                    ha="center", va="bottom", fontsize=12, color=color, zorder=7)
            continue
        if is_ours:
            for xi, v, (L, _) in zip(x + off, vals, LENGTHS):
                ax.text(xi, v + 0.015 * peak, f"{base_kv[L] / v:.1f}$\\times$",
                        ha="center", va="bottom", fontsize=11.5, color=color,
                        fontweight="bold", rotation=90, zorder=7)
        else:
            band += [base_kv[L] / kv[L] for L, _ in LENGTHS]

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.55, len(LENGTHS) - 0.45)
    ax.set_ylim(0, peak * 1.20)
    ax.text(-0.45, peak * 1.17,
            f"PiCoKV: bold $\\times$ = reduction vs. FA-2\n"
            f"2-bit baselines: {min(band):.1f}-{max(band):.1f}$\\times$ at every length",
            ha="left", va="top", fontsize=12.5, color="0.35", zorder=7,
            linespacing=1.35)
    ax.set_ylabel("Prefill KV cache (GB)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def draw_gb_twin(ax):
    """Same two quantities, both drawn: GB bars on the left axis, reduction
    ratio as curves on a right axis.

    The two scales are deliberately detuned so the families cannot overlap: the
    left axis is stretched to 1.55x the tallest bar, which parks every bar below
    65% of the height, and the right axis puts the 4.9-6.3x band above 68%. That
    is what makes a twin axis survivable here, and also what it costs -- 35% of
    the panel is empty by construction, and the reader must check which axis a
    line belongs to.
    """
    x = np.arange(len(LENGTHS))
    base_kv = kv_usage("flash_attention_2")
    width = 0.80 / len(METHODS)
    peak = max(base_kv[L] for L, _ in LENGTHS)
    for i, (_, ekey, label, color, is_ours) in enumerate(METHODS):
        vals = [kv_usage(ekey)[L] for L, _ in LENGTHS]
        off = (i - (len(METHODS) - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.90, color=color, label=label,
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_ylim(0, peak * 1.55)
    ax.set_ylabel("Prefill KV cache (GB)")
    ax.set_xlabel("Context length")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.55, len(LENGTHS) - 0.45)
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(2)
    for _, ekey, label, color, is_ours in METHODS:
        if ekey == "flash_attention_2":
            continue
        kv = kv_usage(ekey)
        vals = [base_kv[L] / kv[L] for L, _ in LENGTHS]
        m = MARKERS[label]
        ax2.plot(x, vals, color=color, marker=m, linestyle="--",
                 linewidth=2.2, markersize=15 if m == "*" else 8,
                 markeredgecolor="white", markeredgewidth=1.2,
                 zorder=6 if is_ours else 5)
    ax2.set_ylim(0, 7.2)
    ax2.set_yticks([0, 2, 4, 6])
    ax2.set_ylabel("Reduction vs. FA-2 ($\\times$)")
    ax2.tick_params(labelsize=13, length=6, width=1.5)
    ax2.text(-0.45, 7.05, "dashed = reduction (right axis)\nbars = GB (left axis)",
             ha="left", va="top", fontsize=12.5, color="0.35", zorder=7,
             linespacing=1.35)


def draw_gb_log(ax):
    """Absolute GB on a log axis, where the vertical gap *is* the ratio.

    This is the one geometry that carries both quantities without a second axis,
    a twin scale or a text patch: on a log axis equal ratios are equal vertical
    distances, so the height you read off the axis is the footprint and the
    distance down from FA-2 is the compression. The saving is shaded rather than
    stated, and the one number the panel is about is drawn as the arrow that
    spans it.

    Bars were tried twice and are the wrong mark here: linear bars give four of
    the five groups almost no height (0.4-2.6 GB against a 32 GB tower), and log
    bars stop being proportional to what they encode.
    """
    x = np.arange(len(LENGTHS))
    gb = {ekey: kv_usage(ekey) for _, ekey, _, _, _ in METHODS}
    fa, ours = METHODS[0][1], METHODS[-1][1]
    fa_v = [gb[fa][L] for L, _ in LENGTHS]
    our_v = [gb[ours][L] for L, _ in LENGTHS]

    # The saving itself, as area. On log y this wedge widens exactly as the
    # ratio grows, so the picture and the number agree by construction.
    ax.fill_between(x, our_v, fa_v, color="#8fb3d9", alpha=0.13, linewidth=0,
                    zorder=1)

    for _, ekey, label, color, is_ours in METHODS:
        vals = [gb[ekey][L] for L, _ in LENGTHS]
        m = MARKERS[label]
        ax.plot(x, vals, color=color, marker=m, linewidth=2.4,
                markersize=16 if m == "*" else 8, markeredgecolor="white",
                markeredgewidth=1.2, zorder=6 if is_ours else 5)

    ax.set_yscale("log")
    # Decade-and-a-half of headroom on each side, derived so that dropping or
    # adding a context length (16K-128K vs 16K-256K) re-fits the axis and the
    # note below instead of clipping a curve or stranding the text.
    ylo, yhi = min(our_v) / 1.7, max(fa_v) * 1.4
    ax.set_ylim(ylo, yhi)
    ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20, 40) if ylo < t < yhi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.35, len(LENGTHS) - 1 + 1.15)

    # The ratio as the span between the two endpoints, at the length that
    # motivates the setting; a number floating in the wedge would leave the
    # reader to guess which pair of curves it refers to. Both endpoint
    # magnitudes ride on the same bracket, in the right margin, so no label has
    # to be squeezed between the compressed curves.
    xa = len(LENGTHS) - 1 + 0.28
    lo, hi = our_v[-1], fa_v[-1]
    ax.annotate("", xy=(xa, hi), xytext=(xa, lo),
                arrowprops=dict(arrowstyle="<->", color="#3d4194", lw=2.0,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(xa + 0.10, hi, f"{hi:.0f} GB", fontsize=13, color="#5f8bbf",
            ha="left", va="center", zorder=8)
    ax.text(xa + 0.10, np.sqrt(lo * hi), f"{hi / lo:.1f}$\\times$",
            fontsize=15, fontweight="bold", color="#3d4194", ha="left",
            va="center", zorder=8)
    ax.text(xa + 0.10, lo, f"{lo:.1f} GB", fontsize=13, fontweight="bold",
            color="#3d4194", ha="left", va="center", zorder=8)

    # No prose inside the axes: the five curves plus the endpoint bracket already
    # carry every number the panel claims. The baselines' own spread (3.0-3.3 GB
    # at 128K) is the one thing this form cannot resolve, and it belongs in the
    # caption rather than as a grey line competing with the "32 GB" label for the
    # same corner.
    ax.set_ylabel("Prefill KV cache (GB)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def draw_gb_compressed(ax):
    """Absolute GB for the compressed methods only; FA-2 is stated, not drawn.

    Dropping the dense curve is what buys resolution back. With FA-2 in the
    frame the axis has to span 0.4-16 GB and the four compressed methods fall
    into one thick line; without it the axis covers only 0.37-3.3 GB and the
    ordering among them -- which is the comparison the paper is actually making
    -- becomes readable, PiCoKV sitting a visible step below the three 2-bit
    baselines at every length.

    The axis stays logarithmic even though the dense outlier is gone: vertical
    distance then still reads as a ratio, so the constant gap between ours and
    the baselines is the constant ~20% saving, and FA-2's absent curve can be
    reintroduced as a single number per endpoint instead of a line.
    """
    x = np.arange(len(LENGTHS))
    gb = {ekey: kv_usage(ekey) for _, ekey, _, _, _ in METHODS}
    fa, ours = METHODS[0][1], METHODS[-1][1]
    fa_v = [gb[fa][L] for L, _ in LENGTHS]

    vals_all = []
    drawn = {}
    for _, ekey, label, color, is_ours in METHODS:
        if ekey == fa:
            continue
        vals = [gb[ekey][L] for L, _ in LENGTHS]
        vals_all += vals
        drawn[label] = vals
        m = MARKERS[label]
        # KIVI and OScaR agree to within 0.4% at every length (3.01 vs 3.02 GB at
        # 128K), so one of the two has to be dashed or it is simply invisible
        # under the other. Dashing a line is the only honest way to separate
        # coincident series -- nudging either curve would misreport the data.
        ls = "--" if label == "KIVI" else "-"
        # ...and it must also sit *on top* of OScaR, or the dashes are hidden
        # under the solid line instead of letting it show through the gaps.
        z = 6 if is_ours else (5.5 if ls == "--" else 5)
        ax.plot(x, vals, color=color, marker=m, linewidth=2.4, linestyle=ls,
                markersize=16 if m == "*" else 8, markeredgecolor="white",
                markeredgewidth=1.2, zorder=z)

    ax.set_yscale("log")
    ylo, yhi = min(vals_all) / 1.45, max(vals_all) * 2.3
    ax.set_ylim(ylo, yhi)
    ticks = [t for t in (0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 8) if ylo < t < yhi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.35, len(LENGTHS) - 1 + 0.42)

    # FA-2 as two numbers rather than a curve: it is 5-6x above everything here,
    # so drawing it would cost the whole panel its resolution to convey a
    # quantity that is exactly 2 bytes per element per token.
    ax.text(-0.25, yhi * 0.97,
            f"FA-2 (fp16, not drawn): {fa_v[0]:.0f} $\\rightarrow$ "
            f"{fa_v[-1]:.0f} GB",
            fontsize=13, color="0.35", ha="left", va="top", zorder=7)
    # Under the curves at the right-hand end, the one region no series enters.
    our_last = gb[ours][LENGTHS[-1][0]]
    ax.text(x[-1] + 0.05, our_last * 0.55,
            f"{our_last:.1f} GB\n{fa_v[-1] / our_last:.1f}$\\times$ vs. FA-2",
            fontsize=13.5, fontweight="bold", color="#3d4194", ha="right",
            va="top", linespacing=1.3, zorder=8)
    ax.set_ylabel("Prefill KV cache (GB)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def draw_gb_bars(ax):
    """Absolute prefill KV cache in GB, the quantity a deployment is limited by.

    A ratio panel answers "how much better than FA-2"; this one answers "does it
    fit", which is the question the 128K setting actually raises. Linear y, so
    FA-2's 16 GB tower against 2.6 GB is the message and the short contexts stay
    small -- appropriate, since nothing is memory-bound at 8K.

    Two things the ratio panels show and this one hides: the 2-bit baselines'
    mutual spread (3.0-3.25 GB at 128K), and that PiCoKV is slightly *worse* than
    them below 16K, where its fixed uncompressed sink and local window have not
    yet been amortized. Both belong in the caption instead.
    """
    x = np.arange(len(LENGTHS))
    width = 0.80 / len(METHODS)
    peak = max(kv_usage("flash_attention_2")[L] for L, _ in LENGTHS)
    for i, (_, ekey, label, color, is_ours) in enumerate(METHODS):
        kv = kv_usage(ekey)
        vals = [kv[L] for L, _ in LENGTHS]
        off = (i - (len(METHODS) - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.90, color=color, label=label,
               edgecolor="white", linewidth=0.8, zorder=5)
        # Only the longest-context endpoints of the reference and of ours carry a
        # number; the bars are the comparison, the labels just pin the scale.
        if is_ours or ekey == "flash_attention_2":
            ax.text(x[-1] + off, vals[-1] + 0.02 * peak, f"{vals[-1]:.1f}",
                    ha="center", va="bottom", fontsize=12, color=color,
                    fontweight="bold" if is_ours else "normal", zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LENGTHS])
    ax.set_xlim(-0.55, len(LENGTHS) - 0.45)
    ax.set_ylim(0, peak * 1.16)
    ax.set_ylabel("Prefill KV cache (GB)")
    ax.set_xlabel("Context length")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-b",
                    choices=["lines", "bars", "gb", "both", "gbx", "twin", "gblog", "gbcomp"],
                    default=DEFAULT_PANEL_B,
                    help="right panel: absolute GB curves on a log axis with the "
                         "compression ratio as the bracket between FA-2 and ours "
                         f"({DEFAULT_PANEL_B}, default), reduction-ratio curves "
                         "(lines), reduction bars, GB bars, GB bars labelled with "
                         "our ratio, GB bars + ratio on a twin axis, or memory "
                         "and decode speedup on one log ratio axis (both)")
    args = ap.parse_args()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(13, 5.6))
    handles = draw_scatter(ax_s)
    {"bars": draw_ratio_bars, "lines": draw_ratio_lines,
     "gb": draw_gb_bars, "both": draw_both, "gbx": draw_gb_ratio,
     "twin": draw_gb_twin, "gblog": draw_gb_log, "gbcomp": draw_gb_compressed}[args.panel_b](ax_b)

    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, frameon=False, fontsize=14, columnspacing=1.2, handlelength=1.4,
               scatterpoints=1)
    # The twin-axis variant needs margin for its right-hand ylabel; every other
    # panel (b) puts its annotations inside the axes.
    right = 0.92 if args.panel_b == "twin" else 0.97
    fig.subplots_adjust(left=0.07, right=right, bottom=0.21, top=0.86, wspace=0.24)
    fig.text(0.30, 0.02, "(a) Performance vs. speedup (128K)", ha="center", va="bottom",
             fontsize=CAPTION_FONTSIZE, fontweight="bold")
    cap_b = {"both": "(b) KV cache and decode speed",
             "gbx": "(b) KV cache footprint and compression",
             "twin": "(b) KV cache footprint and compression",
             "gblog": "(b) KV cache footprint and compression",
             "gbcomp": "(b) KV cache footprint and compression",
             }.get(args.panel_b, "(b) KV cache compression")
    fig.text(0.76, 0.02, cap_b, ha="center", va="bottom",
             fontsize=CAPTION_FONTSIZE, fontweight="bold")

    os.makedirs(OUTDIR, exist_ok=True)
    stem = ("ruler_scatter_throughput" if args.panel_b == DEFAULT_PANEL_B
            else f"ruler_scatter_throughput_{args.panel_b}")
    path = os.path.join(OUTDIR, stem)
    fig.savefig(path + ".pdf", dpi=300)
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)
    print(f"saved {path}.pdf / .png")


if __name__ == "__main__":
    main()
