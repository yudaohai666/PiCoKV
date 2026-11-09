"""Plot 3D surfaces of |Key| / |Value| magnitudes over (token, channel).

Reproduces the RotateKV/KIVI-style figure: runs a needle-in-a-haystack prompt
through a HF causal LM, grabs the KV cache of one (layer, kv_head), and renders
two 3D surfaces.

  - Key states   -> ridges along the token axis (per-channel outliers)
  - Value states -> no consistent channel structure

Head indexing: under GQA the cache only holds ``num_key_value_heads`` heads
(8 for Llama-3.1-8B, against 32 query heads), so ``--head`` indexes KV heads.
Query head q reads KV head ``q // (n_query_heads // n_kv_heads)``, i.e. four
query heads share each surface plotted here.

Usage:
    source .venv/bin/activate
    python draw/plot_kv_distribution.py --layer 10 --head 2 --context-length 8192

Figures are written to draw/pics/.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

NEEDLE = (
    "\nThe best thing to do in San Francisco is eat a sandwich and sit in "
    "Dolores Park on a sunny day.\n"
)

# Shared by the other draw scripts, which import this module anyway.
#
# The GPUs on this box are sm_90 and run torch 2.6+cu124 fine. What used to make
# CUDA look unavailable -- and what an earlier version of this comment
# misattributed to sm_100 -- is the bundled compat driver: LD_LIBRARY_PATH starts
# with /usr/local/cuda/compat, whose libcuda is 575.51.03, *older* than the
# loaded kernel module (580.126.20). cudaGetDeviceCount then fails with error 803
# (driver mismatch) and torch.cuda.is_available() silently returns False. Drop
# that entry to get the system libcuda:
#
#   export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
#
# The default follows whatever actually initialized, so these scripts stay
# runnable in a CPU-only container without a flag.
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    p.add_argument(
        "--context-length",
        type=int,
        default=8192,
        help="target prompt length in tokens; 0 plots the bare needle",
    )
    p.add_argument("--depth-percent", type=float, default=50.0)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--outdir", default=os.path.join(HERE, "pics"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def build_niah_context(tok, dataset, context_length, needle, depth_percent):
    """Needle-in-a-haystack prompt, same recipe as benchmarks/Needle/needle.py
    but reading a .jsonl haystack instead of a directory of .txt files."""
    with open(dataset, encoding="utf-8") as f:
        essays = [json.loads(line)["text"] for line in f if line.strip()]

    needle_ids = tok.encode(needle, add_special_tokens=False)
    budget = context_length - len(needle_ids)
    if budget <= 0:
        raise SystemExit(f"--context-length must exceed the needle ({len(needle_ids)})")

    ids = []
    i = 0
    while len(ids) < budget:
        ids += tok.encode(essays[i % len(essays)], add_special_tokens=False)
        i += 1
    ids = ids[:budget]

    # Insert at depth_percent, snapped back to the nearest sentence break.
    cut = int(len(ids) * depth_percent / 100)
    period = tok.encode(".", add_special_tokens=False)
    while cut > 0 and ids[cut - 1] not in period:
        cut -= 1
    return tok.decode(ids[:cut] + needle_ids + ids[cut:])


@torch.no_grad()
def collect_kv(args):
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.context_length > 0:
        prompt = build_niah_context(
            tok, args.dataset, args.context_length, args.needle, args.depth_percent
        )
    else:
        prompt = args.needle

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

    # Only layers 0..layer feed the cache entry we plot; dropping the rest cuts
    # the forward pass roughly proportionally. LlamaModel.forward slices
    # self.layers[:config.num_hidden_layers], which is safe on a shorter list.
    model.model.layers = model.model.layers[: args.layer + 1]

    ids = tok(prompt, return_tensors="pt").to(model.device)
    # model.model, not model: the causal-LM wrapper projects every position to
    # the 128k vocab, ~2 GiB of logits at ctx 8k, for a tensor nothing here
    # reads. The base model returns past_key_values without entering lm_head.
    cache = model.model(**ids, use_cache=True).past_key_values
    if hasattr(cache, "layers"):  # transformers >= 4.54
        k, v = cache.layers[args.layer].keys, cache.layers[args.layer].values
    else:
        k, v = cache.key_cache[args.layer], cache.value_cache[args.layer]

    # [batch, kv_heads, seq, head_dim] -> [seq, head_dim]
    return (
        k[0, args.head].float().cpu().numpy(),
        v[0, args.head].float().cpu().numpy(),
        ids["input_ids"].shape[1],
    )


def surface(ax, mat, title, zlabel, vmin=None, vmax=None):
    """mat: [seq, head_dim] -> surface with token on x, channel on y."""
    n_token, n_chan = mat.shape
    X, Y = np.meshgrid(np.arange(n_token), np.arange(n_chan), indexing="ij")
    Z = np.abs(mat)

    surf = ax.plot_surface(
        X, Y, Z, cmap="coolwarm", rstride=1, cstride=1, linewidth=0,
        antialiased=False, alpha=0.85,
        vmin=vmin, vmax=vmax,
    )

    pane_color = (0.92, 0.92, 0.92, 1.0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(pane_color)
        pane.set_edgecolor("0.7")
    ax.grid(True, linewidth=0.4, color="white")

    ax.set_xlabel("Token", labelpad=-8, fontsize=9)
    ax.set_ylabel("Channel", labelpad=-8, fontsize=9)
    if zlabel:
        ax.set_zlabel(zlabel, labelpad=2, fontsize=9)
    else:
        ax.set_zlabel("", labelpad=2)
    if title:
        ax.text2D(0.5, 0.87, title,
                  transform=ax.transAxes,
                  ha="center", va="bottom",
                  fontsize=9, linespacing=1.2)
    ax.view_init(elev=25, azim=-60)
    ax.tick_params(labelsize=7, pad=-3)
    ax.set_box_aspect((1.0, 1.2, 0.55))
    return surf


def main():
    args = parse_args()
    k, v, n_tokens = collect_kv(args)
    print(f"tokens={n_tokens}  key={k.shape}  value={v.shape}")

    tag = f"Layer {args.layer} - KV Head {args.head}"
    os.makedirs(args.outdir, exist_ok=True)
    stem = f"L{args.layer}_KV{args.head}_ctx{n_tokens}"

    for mat, name in ((k, "Key"), (v, "Value")):
        fig = plt.figure(figsize=(4.2, 3.2))
        ax = fig.add_subplot(111, projection="3d")
        surf = surface(ax, mat, f"{tag} - {name} States", f"|{name}|")
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.20).ax.tick_params(
            labelsize=7
        )
        fig.tight_layout()
        path = os.path.join(args.outdir, f"{name.lower()}_distribution_{stem}.pdf")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(path.replace(".pdf", ".png"), dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main()


