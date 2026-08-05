"""Re-encode a ``run_standalone_niah.py`` sweep as the flat NIAH result format.

``run_standalone_niah.py`` writes ``{"config": ..., "rows": [...]}`` with the
timing and per-cell metadata the method work needs.  ``show_image_paper.py``
reads the format ``run_bin_v4.py`` produces -- ``{"<length>-<depth>": response}``
in a directory whose name encodes model and method.  This bridges the two so a
standalone sweep can be plotted with the same script as every other method.

The output is byte-compatible with ``run_bin_v4.py``: length-major key order,
``indent=4``, ASCII-escaped.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="run_standalone_niah.py output .json")
    parser.add_argument(
        "out_dir",
        help="directory to write results.json into; its basename becomes the "
             "plot title via show_image_paper.py's name_map "
             "(e.g. Llama-3.1-8B-Instruct_k320folded)",
    )
    args = parser.parse_args()

    rows = json.load(open(args.source))["rows"]
    # run_standalone_niah loops depth-outer, so rows arrive depth-major and have
    # to be re-sorted to read like a native file.
    flat = {
        f"{length}-{depth}": response
        for length, depth, response in sorted(
            (r["requested_length"], r["depth"], r["response"] or "") for r in rows
        )
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w") as handle:
        json.dump(flat, handle, indent=4)
    print(f"wrote {path} ({len(flat)} cells)")


if __name__ == "__main__":
    main()
