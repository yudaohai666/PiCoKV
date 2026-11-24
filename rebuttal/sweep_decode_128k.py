"""128K-prompt decode-length sweep: PiCoKV offline (static / slide) vs flash / KIVI / OScaR.

One process per (method, decode_tokens), scheduled on a pool of GPUs, longest
jobs first.  Each decode length gets its own result directory because the
runner keys results by prompt length only.  Method settings match
benchmarks/Efficiency/run_efficiency.sh.

    python rebuttal/sweep_decode_128k.py --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[1]
MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
DATASET = REPO / "benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl"
PICOKV = ["--method", "picokv", "--k-bits", "320", "--v-bits", "320", "--sink-size", "64",
          "--local-size", "256", "--exp-bits", "4", "--eas", "0.0000001", "--zerorope", "20", "--offline"]
METHODS = {
    "picokv_slide64": ["--slide-tokens", "64", *PICOKV],
    "picokv_static": ["--slide-tokens", "0", *PICOKV],
    "flash_attention_2": ["--method", "flash_attention_2"],
    "kivi": ["--method", "kivi", "--k-bits", "2", "--v-bits", "2", "--group-size", "32", "--residual-length", "128"],
    "oscar": ["--method", "oscar", "--num-bits", "2", "--quant-mode", "k-channel", "--group-size", "32",
              "--kv-rotation", "hadamard", "--kv-norm", "1", "--residual-evict-size", "256", "--offline-v-hadamard"],
    "polar": ["--method", "polar", "--polar-mode", "mixed", "--polar-rbits", "3", "--polar-tbits", "3", "--polar-vbits", "2"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--decode-tokens", default="512,1024,2048,4096,8192")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outdir", default=str(REPO / "rebuttal/results/decode_len_128k"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    jobs = sorted(
        ((int(n), m) for n in args.decode_tokens.split(",") for m in args.methods.split(",")),
        reverse=True,
    )
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    # CPATH: Triton (KIVI) JIT-compiles a C stub that needs Python.h; same as run_efficiency.sh.
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               CPATH=":".join(filter(None, ["/usr/local/include/python3.12", os.environ.get("CPATH")])),
               PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")
    failed = []

    def worker(gpu: str) -> None:
        while True:
            try:
                tokens, method = work.get_nowait()
            except queue.Empty:
                return
            result_dir = outdir / f"dec{tokens}"
            result_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(REPO / "rebuttal/run_sliding_efficiency.py"), *METHODS[method],
                   "--model", MODEL, "--lengths", str(args.length), "--decode-tokens", str(tokens),
                   "--batch-size", "1", "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                   "--dataset", str(DATASET), "--output", str(result_dir / f"{method}.json")]
            log = outdir / "logs" / f"dec{tokens}_{method}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[gpu {gpu}] start dec={tokens} {method}", flush=True)
            with open(log, "w") as fh:
                code = subprocess.call(cmd, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                       stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
            print(f"[gpu {gpu}] done  dec={tokens} {method} exit={code}", flush=True)
            if code:
                failed.append((tokens, method))

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("failed:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
