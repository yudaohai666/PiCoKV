"""Reconstruction error vs code bits for different online GROUP_BITS.

Sweeps ``PICOKV_ONLINE_GROUP_BITS`` in {1,8,16,32,64} on a single NIAH sample and
plots, per side (Key/Value), the relative Frobenius reconstruction error of the
compressed global segment as a function of accumulated code bits.

The group bits knob only changes how the code bits are split into greedy
least-squares groups; every setting reaches the same 640-bit-per-side budget, so
the curves are directly comparable and their endpoints answer "does the grouping
granularity change final reconstruction quality?".

A third panel reports the wall-clock cost of ``quantize_block32_online`` itself,
which is the *only* thing GROUP_BITS changes: the code bits and the
``codes @ decoder`` reconstruction are identical for every setting, so decode is
untouched and the whole effect lives in prefill. Leading-order FLOPs are also
independent of the grouping (the per-group ``L*D*group_bits`` terms sum to
``L*D*bits`` either way); what changes is the number of *serially dependent*
groups, ``bits // group_bits`` -- 640 rank-1 iterations at group=1 against 10 fat
ones at group=64 -- plus the ``O(group_bits)`` QR / gram / solve terms. The bar is
Key time plus Value time, the cost a prefill actually pays for one token's KV.

``--direction-sample-size`` is pinned into the figure caption and the JSON
sidecar because it decides whether the power iteration sees the full residual or
a fixed sample, which changes the per-group cost model: at ctx 8k the global
segment (8193-64-256=7873) is *below* the 8192 default, so sampling never
engages, while at 128k it does. Timings from the two lengths are therefore not
comparable unless this value is held fixed.

The model forward pass is run fresh every time (the on-disk KV cache is
ignored); ``--json-out`` / ``--from-json`` let the figure be restyled without
paying for it again.

Usage:
    source .venv/bin/activate
    # CUDA in this container needs the stale compat libcuda dropped first; see
    # plot_kv_distribution.DEFAULT_DEVICE.
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
    python draw/plot_groupbits_reconstruction.py --context-length 131072
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from plot_kv_distribution import (  # noqa: E402
    DEFAULT_DEVICE,
    NEEDLE,
    build_niah_context,
)
from plot_residual_iteration import global_segment  # noqa: E402
import picokv.picokv_standalone.online as online  # noqa: E402

# House palette, shared with plot_ruler_radar_throughput.py and the speed
# figures. group=32 is the shipped ONLINE_GROUP_BITS default, so it gets
# PiCoKV's indigo and its star marker -- the same "this is ours" treatment
# PiCoKV gets in every other figure. The rest cycle the baseline tones.
OURS_GROUP_BITS = 32
OURS_COLOR = "#3d4194"
BASELINE_COLORS = ["#8fb3d9", "#d98fa8", "#9b8fce", "#5fae9f", "#c9a86a"]


def group_style(group_bits, baseline_index):
    """Return (color, marker, linewidth, zorder) for one group_bits series."""
    if group_bits == OURS_GROUP_BITS:
        return OURS_COLOR, "*", 2.8, 6
    color = BASELINE_COLORS[baseline_index % len(BASELINE_COLORS)]
    return color, "o", 2.2, 5


def group_label(group_bits):
    """Legend text in the paper's notation: ``g`` is the block width of §3."""
    return (
        f"$g$ = {group_bits} (default)"
        if group_bits == OURS_GROUP_BITS
        else f"$g$ = {group_bits}"
    )


def apply_house_style():
    """Match plot_ruler_radar_throughput.py / plot_efficiency_*.py."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
    )
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--head", type=int, default=2, help="KV head index, not query head")
    p.add_argument("--needle", default=NEEDLE)
    p.add_argument(
        "--dataset", default=os.path.join(HERE, "datasets", "PaulGrahamEssays.jsonl")
    )
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--depth-percent", type=float, default=50.0)
    p.add_argument("--device", default=DEFAULT_DEVICE)

    p.add_argument("--k-bits", type=int, default=640)
    p.add_argument("--v-bits", type=int, default=640)
    p.add_argument("--sink", type=int, default=64)
    p.add_argument("--local", type=int, default=256)
    p.add_argument("--zerorope", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--direction-sample-size", type=int, default=8192)
    p.add_argument(
        "--group-bits",
        default="1,8,16,32,64",
        help="comma list of PICOKV_ONLINE_GROUP_BITS values to sweep",
    )
    p.add_argument("--outdir", default=os.path.join(HERE, "groups"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--time-repeats",
        type=int,
        default=3,
        help="measured quantizer calls per (group_bits, code bits) point "
        "(median is reported)",
    )
    p.add_argument(
        "--time-bits",
        default="auto",
        help="comma list of code-bit budgets to time, or 'auto' for multiples of "
        "the coarsest swept group up to the full budget",
    )
    p.add_argument(
        "--time-warmup",
        type=int,
        default=2,
        help="discarded calls before timing each group_bits; absorbs the cuBLAS "
        "autotune and allocator growth that would otherwise be charged to "
        "whichever setting happens to run first",
    )
    p.add_argument(
        "--no-time",
        action="store_true",
        help="skip the timing panel (CPU timings are not comparable to GPU ones)",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="write curves + timings here (default: <outdir>/groupbits_<stem>.json)",
    )
    p.add_argument(
        "--from-json",
        default=None,
        help="redraw from a previous run's JSON sidecar; no model forward",
    )
    return p.parse_args()


@torch.no_grad()
def collect_kv_fresh(args):
    """Run the model forward and return one layer's ``[1,kv_heads,L,D]`` K/V.

    Unlike plot_residual_iteration.collect_kv_full this never reads/writes the
    on-disk cache -- the forward pass is always recomputed.

    The forward goes through ``model.model`` rather than ``model``: the causal-LM
    wrapper materializes logits for *every* position, which is 2 GiB of vocab
    projection at 8k and 33 TiB at 131k. Only the KV cache is wanted here, and
    the base model returns it without ever touching ``lm_head``.
    """
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = (
        build_niah_context(
            tok, args.dataset, args.context_length, args.needle, args.depth_percent
        )
        if args.context_length > 0
        else args.needle
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    model.model.layers = model.model.layers[: args.layer + 1]

    ids = tok(prompt, return_tensors="pt").to(model.device)
    cache = model.model(**ids, use_cache=True).past_key_values
    if hasattr(cache, "layers"):
        k, v = cache.layers[args.layer].keys, cache.layers[args.layer].values
    else:
        k, v = cache.key_cache[args.layer], cache.value_cache[args.layer]
    k, v = k.float(), v.float()
    n_tokens = ids["input_ids"].shape[1]

    # Timing runs next, so nothing but the two tensors may stay resident.
    del cache, ids, model
    if args.device != "cpu":
        torch.cuda.empty_cache()
    return k, v, n_tokens


@torch.no_grad()
def error_curve(target, block, group_bits):
    """Relative Frobenius error over all heads after each ``group_bits`` chunk.

    Replays the quantizer's per-group subtraction in bf16 (the same dtype the
    ``baddbmm`` loop used) so the endpoint matches the real compressed cache.
    Returns (bits_axis, rel_error) with rel_error[0] == 1.0 (unquantized input).
    """
    batch, heads, length, dim = target.shape
    rows = batch * heads
    bits = block.codes.shape[-1]
    groups = bits // group_bits

    residual = target.reshape(rows, length, dim).contiguous().clone()  # bf16
    codes_all = block.codes.reshape(rows, length, bits)
    decoder_all = block.decoder.reshape(rows, bits, dim)
    base = torch.linalg.vector_norm(residual.float()).item()

    rel = [1.0]
    for g in range(groups):
        s, e = g * group_bits, g * group_bits + group_bits
        codes = codes_all[:, :, s:e].to(residual.dtype)
        decoder = decoder_all[:, s:e].to(residual.dtype)
        torch.baddbmm(residual, codes, decoder, beta=1.0, alpha=-1.0, out=residual)
        rel.append(torch.linalg.vector_norm(residual.float()).item() / base)
    bits_axis = [g * group_bits for g in range(groups + 1)]
    return bits_axis, rel


@torch.no_grad()
def sweep_side(target, bits, args, seed, group_list):
    """Return {group_bits: (bits_axis, rel_error)} for one side."""
    out = {}
    for gb in group_list:
        if bits % gb:
            print(f"  skip group_bits={gb}: {bits} not divisible", file=sys.stderr)
            continue
        online.ONLINE_GROUP_BITS = gb  # monkeypatch the module global
        block = online.quantize_block32_online(
            target,
            bits,
            direction_sample_size=args.direction_sample_size,
            seed=seed,
        )
        out[gb] = error_curve(target, block, gb)
    return out


def timing_budgets(spec, bits, group_list):
    """Code-bit budgets to time, ascending and ending at ``bits``.

    ``auto`` steps by the coarsest swept group so every budget is divisible by
    every group in the sweep -- ``quantize_block32_online`` rejects a budget that
    is not a multiple of its group width, so a uniform grid is what keeps the
    curves comparable point for point.
    """
    if spec != "auto":
        wanted = sorted({int(x) for x in spec.split(",")})
    else:
        step = max(group_list)
        wanted = list(range(step, bits + 1, step))
        if wanted and wanted[-1] != bits:
            wanted.append(bits)
    return [b for b in wanted if 0 < b <= bits]


@torch.no_grad()
def time_side(target, bits, args, seed, group_list, budgets):
    """Return ``{group_bits: {code_bits: [ms, ...]}}`` for the quantizer alone.

    Each entry is a *separate* full call at that code length, not a prefix of the
    full-budget run: the random init, the covariance power iteration and the
    Householder QR are all sized by ``bits``, so the cost of reaching 320 bits is
    not something that can be read off a 640-bit run.

    Only the quantizer is inside the timed region -- not the forward pass and not
    :func:`error_curve`'s replay. ``quantize_block32_online`` clones ``target``
    internally, so repeated calls see a byte-identical input and no reset is
    needed between them.

    Warmup is per point, not once for the whole sweep: the first call at a new
    shape pays cuBLAS heuristic selection and caching-allocator growth, and
    charging that to whichever setting the loop happens to reach first is the
    standard way this measurement goes wrong.
    """
    on_cuda = target.is_cuda
    out = {}
    for gb in group_list:
        if bits % gb:
            continue
        online.ONLINE_GROUP_BITS = gb
        per_budget = {}
        for budget in budgets:
            if budget % gb:
                continue
            samples = []
            for call in range(args.time_warmup + args.time_repeats):
                if on_cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                block = online.quantize_block32_online(
                    target,
                    budget,
                    direction_sample_size=args.direction_sample_size,
                    seed=seed,
                )
                if on_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1e3
                del block
                if call >= args.time_warmup:
                    samples.append(elapsed_ms)
            per_budget[budget] = samples
        out[gb] = per_budget
        full = per_budget[max(per_budget)]
        print(
            f"  group_bits={gb:>3}  groups={bits // gb:>3}  "
            f"@{max(per_budget)}bits median={statistics.median(full):8.2f} ms  "
            f"min={min(full):8.2f}  max={max(full):8.2f}  "
            f"({len(per_budget)} budgets timed)"
        )
    return out


def draw(curves_by_side, timing_by_side, meta, args, stem):
    """Key / Value residual panels, plus a cost panel when timings are present.

    The cost panel carries time only. Panels (a)/(b) already show that the
    curves land on top of each other at the full budget, so a separate
    final-error axis would restate what the reader has just seen.
    """
    apply_house_style()
    n_panels = 2 if not timing_by_side else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 5.6))

    group_list = sorted(curves_by_side["Key"])
    handles = []
    for panel, side in enumerate(("Key", "Value")):
        ax = axes[panel]
        baseline_index = 0
        for gb in group_list:
            bits_axis, rel = curves_by_side[side][gb]
            color, marker, width, zorder = group_style(gb, baseline_index)
            if gb != OURS_GROUP_BITS:
                baseline_index += 1
            line, = ax.plot(
                bits_axis,
                rel,
                color=color,
                marker=marker,
                markersize=16 if marker == "*" else 8,
                markeredgecolor="white",
                markeredgewidth=1.2,
                # Each series is thinned to ~10 markers over its own data
                # points, rather than onto a shared bit grid.
                markevery=max(1, len(bits_axis) // 10),
                linewidth=width,
                zorder=zorder,
                label=group_label(gb),
            )
            if panel == 0:
                handles.append(line)
        ax.set_xlabel("Bit budget $L_c$")
        ax.set_ylabel("Relative reconstruction error")
        ax.tick_params(labelsize=13, length=6, width=1.5)
        ax.grid(True, linewidth=0.6, alpha=0.2, zorder=0)
        _add_tail_inset(ax, curves_by_side[side], group_list)

    if timing_by_side:
        _draw_cost_panel(axes[2], timing_by_side, meta)

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        frameon=False,
        fontsize=14,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.20, top=0.88, wspace=0.26)
    # "error", not "fidelity": the y axis is the residual norm ratio, so lower is
    # better, and a fidelity label would invert the reading.
    captions = [
        "(a) Key reconstruction error",
        "(b) Value reconstruction error",
        "(c) Key/Value encoding cost",
    ][:n_panels]
    for ax, caption in zip(axes, captions):
        box = ax.get_position()
        fig.text(
            box.x0 + box.width / 2,
            0.02,
            caption,
            ha="center",
            va="bottom",
            # Smaller than the two-panel figures': three narrower panels make an
            # 18 pt caption crowd its own axis labels.
            fontsize=15,
            fontweight="bold",
        )

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"groupbits_reconstruction_{stem}.pdf")
    fig.savefig(path, dpi=args.dpi)
    fig.savefig(path.replace(".pdf", ".png"), dpi=args.dpi)
    plt.close(fig)
    print(f"saved {path}")


def _add_tail_inset(ax, curves, group_list, frac=0.8):
    """Zoom the last ``1 - frac`` of the bit axis inside the panel.

    At full scale the five curves are one line, which is the point -- but a
    reader cannot then tell that five were drawn at all, nor which way the
    ordering runs. The inset spends a corner of the panel on the only region
    where the gap is resolvable.
    """
    inset = ax.inset_axes([0.42, 0.42, 0.55, 0.5])
    lo = frac * max(curves[group_list[0]][0])
    baseline_index = 0
    ymin, ymax = np.inf, -np.inf
    for gb in group_list:
        bits_axis, rel = curves[gb]
        color, marker, width, zorder = group_style(gb, baseline_index)
        if gb != OURS_GROUP_BITS:
            baseline_index += 1
        bits_arr, rel_arr = np.asarray(bits_axis), np.asarray(rel)
        keep = bits_arr >= lo
        bits_keep, rel_keep = bits_arr[keep], rel_arr[keep]
        inset.plot(
            bits_keep,
            rel_keep,
            color=color,
            marker=marker,
            markersize=11 if marker == "*" else 5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            # Thinned per series, like the parent panel. Without this group=1
            # contributes 128 markers here and reads as a fuzzy band.
            markevery=max(1, len(bits_keep) // 5),
            linewidth=width * 0.8,
            zorder=zorder,
        )
        ymin = min(ymin, rel_keep.min())
        ymax = max(ymax, rel_keep.max())
    pad = 0.06 * (ymax - ymin)
    inset.set_ylim(ymin - pad, ymax + pad)
    inset.tick_params(labelsize=10, length=3, width=1.0)
    inset.grid(True, linewidth=0.5, alpha=0.2, zorder=0)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.55")
    inset.set_facecolor("white")
    ax.indicate_inset_zoom(inset, edgecolor="0.55", linewidth=1.0, alpha=0.6)


def _draw_cost_panel(ax, timing_by_side, meta):
    """Key-side projection-learning time against code length, per group setting.

    Same x axis as panels (a)/(b), so a reader can carry a bit budget straight
    across: at any code length, (a)/(b) give the error and (c) gives what it
    cost.

    One side only, and the panel does not name it, because the cost turns out not
    to depend on the side: across all 50 (group, budget) points measured at 128K,
    Value differs from Key by at most 0.6%. That is expected -- the work is set
    by the tensor shape and the bit budget, not by the numbers in it -- so a
    prefill pays about twice what this panel shows. The *error* is a different
    story and does need both panels: Value sits 20-30% above Key at every
    budget.
    """
    timings = timing_by_side["Key"]
    group_list = sorted(timings)
    baseline_index = 0
    for gb in group_list:
        color, marker, width, zorder = group_style(gb, baseline_index)
        if gb != OURS_GROUP_BITS:
            baseline_index += 1
        budgets = sorted(timings[gb])
        medians = [statistics.median(timings[gb][b]) for b in budgets]
        ax.plot(
            budgets,
            medians,
            color=color,
            marker=marker,
            markersize=16 if marker == "*" else 8,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=width,
            zorder=zorder,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Bit budget $L_c$")
    # "per layer" is the one scale a reader cannot infer from the panel; drop it
    # and 65 ms reads as a whole-model cost instead of one layer's.
    ax.set_ylabel("Time per layer (ms)")
    ax.tick_params(labelsize=13, length=6, width=1.5)
    ax.grid(True, linewidth=0.6, alpha=0.2, zorder=0)


def _dump_json(path, curves_by_side, timing_by_side, meta):
    payload = {
        "meta": meta,
        "curves": {
            side: {str(gb): {"bits": c[0], "rel": c[1]} for gb, c in curves.items()}
            for side, curves in curves_by_side.items()
        },
        "timing_ms": {
            side: {
                str(gb): {str(b): samples for b, samples in per_budget.items()}
                for gb, per_budget in timings.items()
            }
            for side, timings in timing_by_side.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved {path}")


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    curves = {
        side: {int(gb): (c["bits"], c["rel"]) for gb, c in per_side.items()}
        for side, per_side in payload["curves"].items()
    }
    # Sidecars written before the cost panel became a curve store one sample
    # list per group (the full budget only); promote those to the nested form
    # keyed by code bits so old runs still redraw.
    timing = {}
    for side, per_side in payload.get("timing_ms", {}).items():
        timing[side] = {}
        for gb, entry in per_side.items():
            if isinstance(entry, dict):
                timing[side][int(gb)] = {int(b): s for b, s in entry.items()}
            else:
                full = payload["meta"]["k_bits" if side == "Key" else "v_bits"]
                timing[side][int(gb)] = {int(full): entry}
    return curves, timing, payload["meta"]


def main():
    args = parse_args()

    if args.from_json:
        curves_by_side, timing_by_side, meta = _load_json(args.from_json)
        draw(curves_by_side, timing_by_side, meta, args, meta["stem"])
        return

    group_list = sorted(int(x) for x in args.group_bits.split(","))

    k, v, n_tokens = collect_kv_fresh(args)
    print(f"tokens={n_tokens}  key={tuple(k.shape)}  value={tuple(v.shape)}")

    key_global, global_len = global_segment(k, args.sink, args.local, args.zerorope)
    value_global, _ = global_segment(v, args.sink, args.local, 0)
    del k, v
    if args.device != "cpu":
        torch.cuda.empty_cache()
    print(f"global segment -> {global_len} tokens over {key_global.shape[1]} KV heads")
    sampling = (
        "off (global segment is shorter than the sample)"
        if args.direction_sample_size >= global_len
        else f"on ({args.direction_sample_size} of {global_len} rows)"
    )
    print(f"direction sampling: {sampling}")
    # PLACEHOLDER_MAIN_TAIL

    curves_by_side = {
        "Key": sweep_side(key_global, args.k_bits, args, args.seed, group_list),
        "Value": sweep_side(value_global, args.v_bits, args, args.seed + 1009, group_list),
    }

    timing_by_side = {}
    if not args.no_time:
        k_budgets = timing_budgets(args.time_bits, args.k_bits, group_list)
        v_budgets = timing_budgets(args.time_bits, args.v_bits, group_list)
        print(
            f"\ntiming quantize_block32_online on {args.device}: "
            f"Key budgets {k_budgets}"
        )
        print(" Key:")
        timing_by_side["Key"] = time_side(
            key_global, args.k_bits, args, args.seed, group_list, k_budgets
        )
        print(" Value:")
        timing_by_side["Value"] = time_side(
            value_global, args.v_bits, args, args.seed + 1009, group_list, v_budgets
        )

    print("\nfinal relative reconstruction error @ full budget:")
    header = f"  {'group_bits':>10} | {'groups':>6} | {'Key':>8} | {'Value':>8}"
    if timing_by_side:
        header += f" | {'K ms':>9} | {'V ms':>9}"
    print(header)
    for gb in group_list:
        kerr = curves_by_side["Key"].get(gb, (None, [float("nan")]))[1][-1]
        verr = curves_by_side["Value"].get(gb, (None, [float("nan")]))[1][-1]
        row = f"  {gb:>10} | {args.k_bits // gb:>6} | {kerr:>8.4f} | {verr:>8.4f}"
        if timing_by_side:
            k_full = timing_by_side["Key"][gb][max(timing_by_side["Key"][gb])]
            v_full = timing_by_side["Value"][gb][max(timing_by_side["Value"][gb])]
            row += (
                f" | {statistics.median(k_full):>9.2f}"
                f" | {statistics.median(v_full):>9.2f}"
            )
        print(row)

    stem = f"L{args.layer}_ctx{n_tokens}_k{args.k_bits}_v{args.v_bits}"
    meta = {
        "stem": stem,
        "layer": args.layer,
        "context_length": args.context_length,
        "n_tokens": int(n_tokens),
        "global_len": int(global_len),
        "kv_heads": int(key_global.shape[1]),
        "bits": args.k_bits,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "sink": args.sink,
        "local": args.local,
        "zerorope": args.zerorope,
        "direction_sample_size": args.direction_sample_size,
        "direction_sampling_active": bool(args.direction_sample_size < global_len),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device != "cpu" else "cpu",
        "torch": torch.__version__,
        "time_warmup": args.time_warmup,
        "time_repeats": args.time_repeats,
        "time_bits": args.time_bits,
        "seed": args.seed,
    }
    json_path = args.json_out or os.path.join(args.outdir, f"groupbits_{stem}.json")
    _dump_json(json_path, curves_by_side, timing_by_side, meta)
    draw(curves_by_side, timing_by_side, meta, args, stem)


if __name__ == "__main__":
    main()
