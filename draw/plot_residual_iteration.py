"""Plot how PiCoKV's online block-32 residual iteration flattens |K| / |V|.

Runs the same needle-in-a-haystack prompt as ``plot_kv_distribution.py``, takes
one layer's KV cache, feeds the *global* segment (the part PiCoKV actually
compresses, i.e. tokens ``[SINK, L-LOCAL)``) through
``picokv.picokv_standalone.online.quantize_block32_online``, and renders the
residual surface after each 32-bit group.

Why the residual is *replayed* instead of re-running the quantizer per bit
budget: ``quantize_block32_online`` draws its random initialization with
``torch.randn(batch, heads, bits, dim)``, so the directions learned at bits=32
are NOT a prefix of those learned at bits=320. Calling it once at the full
budget and subtracting ``codes_g @ decoder_g`` group by group reproduces the
internal ``baddbmm`` exactly (codes are strict +-1 stored as int8, the decoder
is stored in bf16 -- the same operands the loop used), so these snapshots are
the real intermediate residuals, bit for bit.

Defaults mirror benchmarks/Needle/run_niah.sh: K_BITS=V_BITS=320, SINK=64,
LOCAL=256, EXP_BITS=4, EAS=0.01, ZEROROPE=0.

EXP_BITS is reported but does not enter this figure: it is the decode-time
exponent code width, applied after the residual iteration is finished.

Usage:
    source .venv/bin/activate
    python draw/plot_residual_iteration.py --layer 10 --head 2 --context-length 8192

Figures are written to draw/pics/.
"""

import argparse
import os
import subprocess
import sys

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
    surface,
)
from picokv.picokv_standalone.constants import ONLINE_GROUP_BITS  # noqa: E402
from picokv.picokv_standalone.online import quantize_block32_online  # noqa: E402


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
    # See plot_kv_distribution.DEFAULT_DEVICE for why this is auto-detected.
    p.add_argument("--device", default=DEFAULT_DEVICE)

    # PiCoKV parameters (run_niah.sh defaults).
    p.add_argument("--k-bits", type=int, default=int(os.environ.get("K_BITS", 320)))
    p.add_argument("--v-bits", type=int, default=int(os.environ.get("V_BITS", 320)))
    p.add_argument("--sink", type=int, default=int(os.environ.get("SINK", 64)))
    p.add_argument("--local", type=int, default=int(os.environ.get("LOCAL", 256)))
    p.add_argument("--exp-bits", type=int, default=int(os.environ.get("EXP_BITS", 4)))
    p.add_argument("--eas", type=float, default=float(os.environ.get("EAS", 0.01)))
    p.add_argument("--zerorope", type=int, default=int(os.environ.get("ZEROROPE", 0)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--direction-sample-size", type=int, default=8192)

    p.add_argument(
        "--iters",
        default="0,1,2,4",
        help="residual snapshots to draw; 0 is the unquantized input",
    )
    p.add_argument(
        "--token-stride",
        type=int,
        default=512,
        help="subsample tokens for rendering only; the quantizer sees every token",
    )
    p.add_argument("--outdir", default=os.path.join(HERE, "pics"))
    p.add_argument("--cache-dir", default=os.path.join(HERE, "cache"))
    p.add_argument("--dpi", type=int, default=300)
    # Internal flag: render exactly one panel and exit (avoids SIGSTOP from
    # sequential 3D figure creation in headless environments).
    p.add_argument("--single-panel", default=None,
                   help="INTERNAL: 'Key:0' or 'Value:2' — render one panel and exit")
    # Shared window params passed from parent when --single-panel is used.
    p.add_argument("--window-start", type=int, default=-1)
    p.add_argument("--zmax", type=float, default=-1.0)
    p.add_argument("--stem", default=None)
    return p.parse_args()


@torch.no_grad()
def collect_kv_full(args):
    """Return one layer's full ``[1, kv_heads, L, D]`` K and V, plus token count.

    The forward pass dominates the runtime, so the extracted cache is memoized to
    ``--cache-dir``; only the figure styling usually needs re-running.

    The forward dtype is part of the cache key, because it is not a free choice:
    CPU runs in fp32 and CUDA in fp16, so the two produce *different* KV values.
    Keying on ``(layer, ctx, depth)`` alone would let a CPU-fp32 cache be served
    to a GPU-fp16 run and silently change the residual curves.
    """
    forward_dtype = "fp32" if args.device == "cpu" else "fp16"
    cache_path = os.path.join(
        args.cache_dir,
        f"kv_L{args.layer}_ctx{args.context_length}"
        f"_d{args.depth_percent:g}_{forward_dtype}.pt",
    )
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        print(f"loaded {cache_path}")
        return (
            blob["key"].to(args.device),
            blob["value"].to(args.device),
            int(blob["n_tokens"]),
        )

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

    n_kv = model.config.num_key_value_heads
    if not 0 <= args.head < n_kv:
        raise SystemExit(
            f"--head must be in [0, {n_kv}): this model caches {n_kv} KV heads for "
            f"{model.config.num_attention_heads} query heads."
        )

    # Layers past the one we plot never touch its cache entry; dropping them
    # cuts the forward pass roughly proportionally.
    model.model.layers = model.model.layers[: args.layer + 1]

    ids = tok(prompt, return_tensors="pt").to(model.device)
    # model.model, not model: the causal-LM wrapper projects every position to
    # the 128k vocab, which is ~2 GiB of logits at ctx 8k and unrunnable past
    # ~32k. Only past_key_values is wanted, and the base model returns it
    # without ever entering lm_head.
    cache = model.model(**ids, use_cache=True).past_key_values
    if hasattr(cache, "layers"):  # transformers >= 4.54
        k, v = cache.layers[args.layer].keys, cache.layers[args.layer].values
    else:
        k, v = cache.key_cache[args.layer], cache.value_cache[args.layer]
    k, v, n_tokens = k.float(), v.float(), ids["input_ids"].shape[1]

    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(
        {"key": k.cpu(), "value": v.cpu(), "n_tokens": n_tokens},
        cache_path,
    )
    print(f"cached {cache_path}")
    del cache, ids, model
    if args.device != "cpu":
        torch.cuda.empty_cache()
    return k, v, n_tokens


def global_segment(states, sink, local, zerorope):
    """Slice tokens ``[sink, L-local)`` and mask RoPE pairs, as prepare.py does."""
    total = states.shape[-2]
    global_len = total - sink - local
    if global_len <= 0:
        raise SystemExit(
            f"invalid sink/local for total_len={total}: sink={sink}, local={local}"
        )
    out = states[:, :, sink : sink + global_len].clone()
    if zerorope > 0:
        half = out.shape[-1] // 2
        if zerorope > half:
            raise SystemExit(f"zerorope ({zerorope}) must be <= {half}")
        out[..., :zerorope] = 0
        out[..., half : half + zerorope] = 0
    return out.to(torch.bfloat16).contiguous(), global_len


@torch.no_grad()
def residual_snapshots(target, block, head):
    """Replay the quantizer's per-group subtraction and keep one KV head.

    The replay keeps the quantizer's own batched ``[rows, L, D]`` ``baddbmm``
    shape rather than slicing the head out first, so the arithmetic is the same
    call on the same operands and the snapshots are bit-exact intermediates.

    Returns ``groups + 1`` bf16 ``[L, D]`` residuals (index 0 = the input) and
    two relative Frobenius error curves, ``rel_head`` for the plotted head and
    ``rel_all`` over every head. They differ by more than a rounding: at layer
    10 head 2 reaches 0.159 against 0.185 for the whole segment, so the panels
    must be annotated with the per-head curve to match their own surface.
    """
    batch, heads, length, dim = target.shape
    rows = batch * heads
    groups = block.codes.shape[-1] // ONLINE_GROUP_BITS
    row = head  # batch is 1 here, so row index == head index

    residual = target.reshape(rows, length, dim).contiguous().clone()
    codes_all = block.codes.reshape(rows, length, -1)
    decoder_all = block.decoder.reshape(rows, -1, dim)
    base_all = torch.linalg.vector_norm(residual.float()).item()
    base_head = torch.linalg.vector_norm(residual[row].float()).item()

    snaps = [residual[row].clone()]
    rel_head, rel_all = [1.0], [1.0]
    for group_idx in range(groups):
        start = group_idx * ONLINE_GROUP_BITS
        end = start + ONLINE_GROUP_BITS
        codes = codes_all[:, :, start:end].to(residual.dtype)
        decoder = decoder_all[:, start:end].to(residual.dtype)
        torch.baddbmm(residual, codes, decoder, beta=1.0, alpha=-1.0, out=residual)
        snaps.append(residual[row].clone())
        rel_head.append(
            torch.linalg.vector_norm(residual[row].float()).item() / base_head
        )
        rel_all.append(torch.linalg.vector_norm(residual.float()).item() / base_all)
    return snaps, rel_head, rel_all


def draw_panels_combined(all_snaps, all_rels, iters, tag, stride, args, stem):
    """Dispatch one subprocess per panel to avoid SIGSTOP from sequential 3D renders.

    Determines the shared token window and z-scale from the in-process snapshots,
    then launches each panel as ``python plot_residual_iteration.py --single-panel
    Side:it ...``.  Each child renders exactly one figure and exits.
    """
    rng = np.random.default_rng(args.seed)

    base_argv = [
        sys.executable, __file__,
        "--model", args.model,
        "--layer", str(args.layer),
        "--head", str(args.head),
        "--context-length", str(args.context_length),
        "--depth-percent", str(args.depth_percent),
        "--device", args.device,
        "--k-bits", str(args.k_bits),
        "--v-bits", str(args.v_bits),
        "--sink", str(args.sink),
        "--local", str(args.local),
        "--exp-bits", str(args.exp_bits),
        "--eas", str(args.eas),
        "--zerorope", str(args.zerorope),
        "--seed", str(args.seed),
        "--direction-sample-size", str(args.direction_sample_size),
        "--iters", args.iters,
        "--outdir", args.outdir,
        "--cache-dir", args.cache_dir,
        "--dpi", str(args.dpi),
        "--stem", stem,
    ]

    for side, snaps in all_snaps.items():
        total_tokens = snaps[0].shape[0]
        window = 32
        start = int(rng.integers(0, max(1, total_tokens - window)))
        mats = [snaps[i][start : start + window].float().cpu().numpy() for i in iters]
        zmax = max(float(np.abs(m).max()) for m in mats)
        for it in iters:
            argv = base_argv + [
                "--single-panel", f"{side}:{it}",
                "--window-start", str(start),
                "--zmax", str(zmax),
            ]
            result = subprocess.run(argv, capture_output=False)
            if result.returncode != 0:
                print(f"[WARN] subprocess exited {result.returncode} for {side}:{it}",
                      file=sys.stderr)


def _render_single_panel(args):
    """Called when --single-panel Side:it is set. Renders one figure and exits."""
    side, it_str = args.single_panel.rsplit(":", 1)
    it = int(it_str)
    bits = it * ONLINE_GROUP_BITS
    label = "original" if it == 0 else f"{bits}bits"
    stem = args.stem

    k, v, n_tokens = collect_kv_full(args)
    key_global, global_len = global_segment(k, args.sink, args.local, args.zerorope)
    value_global, _ = global_segment(v, args.sink, args.local, 0)

    if side == "Key":
        target, seed_offset = key_global, 0
    else:
        target, seed_offset = value_global, 1009

    block = quantize_block32_online(
        target,
        args.k_bits if side == "Key" else args.v_bits,
        direction_sample_size=args.direction_sample_size,
        seed=args.seed + seed_offset,
    )
    snaps, _, _ = residual_snapshots(target, block, args.head)
    groups = len(snaps) - 1
    it_clamped = min(it, groups)

    start = args.window_start
    window = 32
    mat = snaps[it_clamped][start : start + window].float().cpu().numpy()
    zmax = args.zmax if args.zmax > 0 else float(np.abs(mat).max())
    # Per-panel colour range so blue/red contrast is always vivid.
    mat_max = float(np.abs(mat).max())

    if it == 0:
        title = f"Layer {args.layer} | KV Head {args.head}\n{side} States (Original)"
        file_prefix = f"{side.lower()}_states"
    else:
        title = f"Layer {args.layer} | KV Head {args.head}\n{side} Residual ({bits} bits)"
        file_prefix = f"{side.lower()}_residual"

    # fig = plt.figure(figsize=(5.0, 4.2))
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig = plt.figure(figsize=(5.0, 3.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    surf = surface(ax, mat, title, "", vmin=0, vmax=mat_max)
    ax.set_zlim(0, zmax)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{file_prefix}_{stem}_{label}.pdf")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
    fig.savefig(path.replace(".pdf", ".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0)
    fig.savefig(path.replace(".pdf", ".svg"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"saved {path}")


def draw_curve(curves, eas_shares, args, stem):
    """Relative reconstruction error and EAS energy concentration on one axis.

    ``eas_shares`` maps side name -> (eas_ratio, energy_share), e.g.
    {"Key": (0.01, 0.123), "Value": (0.01, 0.098)}.  An annotation is drawn at
    the final bit budget showing what fraction of residual energy the EAS tokens
    account for.
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    colors = {}
    for name, rel in curves.items():
        bits = np.arange(len(rel)) * ONLINE_GROUP_BITS
        (line,) = ax.plot(bits, rel, marker="o", markersize=3.5, label=name)
        colors[name] = line.get_color()

    # (EAS annotation removed)

    ax.set_xlabel("Code bits", fontsize=9)
    ax.set_ylabel("Relative reconstruction error", fontsize=9)
    ax.set_title("Block-32 residual iteration", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = os.path.join(args.outdir, f"residual_curve_{stem}.pdf")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    args = parse_args()

    # Single-panel mode: render exactly one figure and exit.
    if args.single_panel is not None:
        _render_single_panel(args)
        return

    os.makedirs(args.outdir, exist_ok=True)

    k, v, n_tokens = collect_kv_full(args)
    print(f"tokens={n_tokens}  key={tuple(k.shape)}  value={tuple(v.shape)}")

    # zerorope masks key directions only (prepare.py:226).
    key_global, global_len = global_segment(k, args.sink, args.local, args.zerorope)
    value_global, _ = global_segment(v, args.sink, args.local, 0)
    print(
        f"global segment [{args.sink}, {args.sink + global_len}) -> {global_len} tokens"
        f"  (sink={args.sink}, local={args.local}, exp_bits={args.exp_bits} unused here)"
    )

    key_block = quantize_block32_online(
        key_global,
        args.k_bits,
        direction_sample_size=args.direction_sample_size,
        seed=args.seed,
    )
    value_block = quantize_block32_online(
        value_global,
        args.v_bits,
        direction_sample_size=args.direction_sample_size,
        seed=args.seed + 1009,  # prepare.py:253
    )

    tag = f"Layer {args.layer} - KV Head {args.head}"
    stem = f"L{args.layer}_KV{args.head}_ctx{n_tokens}_k{args.k_bits}_v{args.v_bits}"
    curves = {}
    eas_shares = {}
    all_snaps = {}
    all_rels = {}
    iters_parsed = None
    for target, block, name in (
        (key_global, key_block, "Key"),
        (value_global, value_block, "Value"),
    ):
        snaps, rel_head, rel_all = residual_snapshots(target, block, args.head)
        curves[name] = rel_head
        all_snaps[name] = snaps
        all_rels[name] = rel_head
        groups = len(snaps) - 1
        iters_parsed = sorted({min(int(s), groups) for s in args.iters.split(",")})
        print(
            f"{name}: {groups} groups, rel. error (KV head {args.head}) "
            + " ".join(f"{r:.4f}" for r in rel_head)
        )
        print(
            f"{name}: {groups} groups, rel. error (all heads)  "
            + " ".join(f"{r:.4f}" for r in rel_all)
        )

    draw_panels_combined(all_snaps, all_rels, iters_parsed, tag, args.token_stride, args, stem)

    # EAS union: Key top-k first, then Value top-k from remaining positions,
    # mirroring _select_eas_indices in prepare.py.
    eas_size = max(1, int(global_len * args.eas))
    key_err = key_block.err[0, args.head].float()
    val_err = value_block.err[0, args.head].float()

    key_indices = torch.topk(key_err, k=eas_size).indices
    mask = torch.zeros(global_len, dtype=torch.bool, device=key_err.device)
    mask[key_indices] = True
    val_count = min(eas_size, global_len - eas_size)
    val_indices = torch.topk(val_err.masked_fill(mask, -torch.inf), k=val_count).indices
    union_indices = torch.cat([key_indices, val_indices])
    union_size = union_indices.shape[0]

    # Per-side energy share within the union.
    for block, err, name in (
        (key_block, key_err, "Key"),
        (value_block, val_err, "Value"),
    ):
        union_energy = err[union_indices].pow(2).sum()
        total_energy = err.pow(2).sum()
        share = (union_energy / total_energy).item()
        eas_ratio = union_size / global_len
        eas_shares[name] = (eas_ratio, share)
        print(
            f"EAS union {name}: {union_size} of {global_len} tokens "
            f"({eas_ratio:.1%}) hold {share:.1%} of the residual energy"
        )

    draw_curve(curves, eas_shares, args, stem)


if __name__ == "__main__":
    main()
