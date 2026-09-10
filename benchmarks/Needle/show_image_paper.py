import json
import os
import glob
import argparse
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

# Use default fonts (fallback if custom fonts not available)
prop_16 = font_manager.FontProperties(size=45)
prop_13 = font_manager.FontProperties(size=38)

def round_to_nearest_k(x, k=8000):
    return int(round(x / k)) * k

def tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation before splitting into words."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def main(args):
    path = args.eval_path
    save_dir = args.save_dir

    # Two layouts are produced in this repo, both naming the model in --model:
    #   results/niah/<model>/<TAG>/         (run_niah.sh)
    #   results/niah/<model>_<TAG>/         (run_bin.py invoked by hand)
    # So strip the model prefix and let the method tag keep its underscores.
    path = os.path.normpath(path)
    dir_name = os.path.basename(path)

    model = args.model.rstrip('/').split('/')[-1]
    method = dir_name.lower()
    if method.startswith(model.lower() + '_'):
        method = method[len(model) + 1:]

    save_dir = os.path.join(save_dir, model)
    os.makedirs(save_dir, exist_ok=True)
    # Always emit both: the PNG for quick eyeballing, the PDF for the paper.
    save_paths = [
        os.path.join(save_dir, f"{dir_name}.png"),
        os.path.join(save_dir, f"{dir_name}.pdf"),
    ]

    data = []

    round_k = 8192
    if "mistral" in model.lower():
        round_k = 4096
    elif "llama-3-8b" in model.lower() or "llama3-8b" in model.lower():
        round_k = 512  # llama3-8b uses 512-step intervals

    expected_answer = tokenize(args.expected_answer)

    # --- flat format: {"length-depth": response, ...} (e.g. results_baseline_sdpa.json) ---
    flat_json = glob.glob(f"{path}/*.json")
    flat_candidates = [f for f in flat_json if not f.endswith("_results.json")]
    flat_loaded = False
    for candidate in flat_candidates:
        with open(candidate, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                continue
        # Detect flat format: all keys are "int-int"
        if json_data and all(
            re.match(r'^\d+-\d+$', k) for k in list(json_data.keys())[:5]
        ):
            for key, response in json_data.items():
                length_str, depth_str = key.split('-')
                context_length = int(length_str)
                document_depth = int(depth_str)
                resp_words = tokenize(response) if response else []
                score = len(set(resp_words).intersection(set(expected_answer))) / len(set(expected_answer)) * 100
                context_length_k = round_to_nearest_k(context_length, round_k)
                data.append({
                    "Document Depth": document_depth,
                    "Context Length": context_length_k,
                    "Score": score,
                })
            flat_loaded = True
            break

    # --- per-file format: *_results.json ---
    if not flat_loaded:
        json_files = glob.glob(f"{path}/*_results.json")
        for file in json_files:
            with open(file, 'r') as f:
                json_data = json.load(f)
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            model_response = json_data.get("model_response", None)
            resp_words = tokenize(model_response) if model_response else []
            score = len(set(resp_words).intersection(set(expected_answer))) / len(set(expected_answer)) * 100
            context_length_k = round_to_nearest_k(context_length, round_k)
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length_k,
                "Score": score,
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    # Sort context lengths numerically
    locations = sorted(df["Context Length"].unique())
    
    # Pivot with numeric context length
    pivot_table = pd.pivot_table(
        df, 
        values='Score', 
        index=['Document Depth', 'Context Length'], 
        aggfunc='mean'
    ).reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    # Create the heatmap
    plt.figure(figsize=(21, 15))
    ax = sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={'shrink': 0.4, 'location': 'right', 'pad':0.02},
        linewidths=0.5,
        linecolor='grey',
        linestyle='--',
        vmin=0, vmax=100,
    )
    ax.set_aspect('auto')

    if "mistral" in model.lower():
        tick_labels = [f"{x/1024:.1f}k" if x % 1024 != 0 else f"{int(x/1024)}k" for x in locations]
    else:
        tick_labels = [f"{x//1024}k" for x in locations]
    ax.set_xticklabels(tick_labels, rotation=45,fontproperties=prop_13)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=prop_13)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=38)  

    # Method name only -- no bit budget or window size. Run tags stay in the file
    # name (results/niah/vis/<model>/<TAG>.{png,pdf}), so ablation figures are
    # still told apart on disk; the title is what goes in the paper. The one
    # config that DOES reach the title is the projection mode, because online vs
    # offline are two named methods in the paper, not two settings of one.
    # Keys are matched as a prefix of the run tag, longest first, so
    # picokv_offline_k320_v320 -> PiCoKV-Fixed while picokv_online_k320_v320 and
    # any other picokv_* tag fall through to PiCoKV.
    name_map = {
        'picokv_offline': 'PiCoKV-Fixed',
        'picokv_online': 'PiCoKV',
        'picokv': 'PiCoKV',
        'kivi': 'KIVI',
        'oscar': 'OSCAR',
        'polar': 'PolarQuant',
        # Dense baselines: the attention implementation is not a KV method.
        'flash_attention_2': 'Full Attention',
        'sdpa': 'Full Attention',
        'eager': 'Full Attention',
    }

    method_lower = method.lower()
    method_title = None
    for key in sorted(name_map, key=len, reverse=True):
        if method_lower == key or re.match(rf'{re.escape(key)}[_-]', method_lower):
            method_title = name_map[key]
            break
    if method_title is None:
        # Unknown tag: drop whatever config tokens trail the name.
        method_title = re.split(r'[_-]', method)[0].title()

    # Bold the method name, leave the score readout at regular weight.
    # Matplotlib mixes weights inside one string only via mathtext, which needs
    # two escapes: it drops plain spaces (hence `\ ` for "Full Attention"), and it
    # reads `-` as the minus operator, which renders as a spaced en-dash --
    # "PiCoKV - Fixed" instead of "PiCoKV-Fixed". So close the math group around
    # each hyphen and let the hyphen be plain text; at title size its regular
    # weight is imperceptible next to the bold words.
    bold_name = '-'.join(
        rf"$\bf{{{seg.replace(' ', r'\ ')}}}$" for seg in method_title.split('-')
    )
    title_str = rf"{bold_name} Average Score : {df['Score'].mean():.2f}"

    plt.title(title_str,fontproperties=prop_16)
    plt.ylabel('Depth Percent',fontproperties=prop_16)
    plt.xlabel("Context Length", fontproperties=prop_16)
    plt.yticks(rotation=0)
    plt.tight_layout()
    for out in save_paths:
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"saved {out}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/Llama-3.1-8B-Instruct", help="model name or local model path")
    parser.add_argument("--save_dir", default="outputs/needle_results_vis", type=str, help="Directory the .png/.pdf pair is written to")
    parser.add_argument("--eval_path", default="", type=str, help="Result directory holding results.json / *_results.json")
    parser.add_argument("--expected_answer", type=str,
                        default="eat a sandwich and sit in Dolores Park on a sunny day.",
                        help="Needle answer; must match the one run_bin.py was given")
    args = parser.parse_args()
    main(args)
