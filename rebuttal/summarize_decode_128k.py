"""Table for rebuttal/sweep_decode_128k.py results (one row per decode length)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results/decode_len_128k")
ORDER = ["flash_attention_2", "kivi", "oscar", "polar", "picokv_static", "picokv_slide64"]


def main() -> None:
    rows = []
    for d in sorted(ROOT.glob("dec*"), key=lambda p: int(p.name[3:])):
        for method in ORDER:
            path = d / f"{method}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for r in payload["results"].values():
                rows.append((int(d.name[3:]), method, r))
    metrics = ("prefill_ms", "decode_ms_per_token", "decode_toks_per_s", "decode_peak_gb", "decode_kv_gb")
    print("| decode | method | " + " | ".join(metrics) + " |")
    print("|---" * (len(metrics) + 2) + "|")
    for tokens, method, r in rows:
        cells = ["OOM" if r.get("error") else ("-" if r.get(m) is None else f"{r[m]}") for m in metrics]
        print(f"| {tokens} | {method} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
