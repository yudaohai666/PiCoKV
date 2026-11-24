"""Isolated entry to the original efficiency / LongBench runners."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal.BPC.encoding import binarypc_encoding


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", choices=("efficiency", "longbench", "ruler"), default="efficiency")
    parser.add_argument("--encoding", choices=("binarypc", "binarypc8192", "gpc"), default="binarypc")
    options, forwarded = parser.parse_known_args()
    # binarypc8192 is binarypc with the direction fit restricted to
    # direction_sample_size tokens, i.e. the sample-matched cell of the
    # {GPC, BPC} x {sampled fit, full-L fit} square.
    is_bpc = options.encoding in ("binarypc", "binarypc8192")
    sampled_fit = options.encoding == "binarypc8192"
    suffix = {
        "binarypc": "_coding_binarypc_nohfm",
        "binarypc8192": "_coding_binarypc_nohfm_s8192",
        "gpc": "_coding_gpc",
    }[options.encoding]
    if "--help" in forwarded or "-h" in forwarded:
        parser.print_help()
        print("\nOriginal runner arguments:")
    module = {
        "efficiency": "benchmarks.Efficiency.run_efficiency",
        "longbench": "benchmarks.Longbench.pred",
        "ruler": "benchmarks.Ruler.run_ruler_evaluation",
    }
    runner = importlib.import_module(module[options.experiment])
    previous_tag, previous_load, previous_argv = runner.output_tag, runner.load_model, sys.argv
    if "--help" in forwarded or "-h" in forwarded:
        sys.argv = [str(Path(__file__).resolve()), *forwarded]
        try:
            return runner.main()
        finally:
            sys.argv = previous_argv

    def tag(*args, **kwargs):
        return previous_tag(*args, **kwargs) + suffix

    def load_model(*args, **kwargs):
        if kwargs.get("method", "picokv") != "picokv":
            raise ValueError("use --method picokv; only the coding algorithm is replaced")
        if kwargs.get("offline_proj_path") is not None:
            raise ValueError("this comparison requires online projections for both algorithms")
        if kwargs.get("use_pruning", False):
            raise ValueError("disable pruning for this coding comparison")
        if is_bpc:
            kwargs['zerorope'] = 0
        return previous_load(*args, **kwargs)

    # The output is mandatory to keep every configuration/experiment separate.
    # All three upstream entry points support it, with different spellings.
    flag = {"efficiency": "--output", "longbench": "--output-path", "ruler": "--output_path"}[options.experiment]
    if not any(arg == flag or arg.startswith(flag + "=") for arg in forwarded):
        raise ValueError(f"specify {flag} under rebuttal/BPC/results for this run")
    at = next(i for i, arg in enumerate(forwarded) if arg == flag or arg.startswith(flag + "="))
    output = Path(forwarded[at].split("=", 1)[1] if "=" in forwarded[at] else forwarded[at + 1])
    metadata_path = output.with_suffix(".run.json") if options.experiment == "efficiency" else output / "coding_run.json"
    metadata = {
        "experiment": options.experiment, "encoding": options.encoding,
        "algorithm": "BinaryPC online fit then residual replay" if is_bpc else "PiCoKV GPC",
        "binarypc_num_iters": 4 if is_bpc else None,
        "coding_version": ("bpc320_source_sampled_fit_v1" if sampled_fit else "bpc320_source_global_rng_v3") if is_bpc else "gpc_production",
        "binarypc_rng": "original global torch RNG; quantizer seed argument unused" if is_bpc else None,
        "binarypc_fitting_codes": "original allocation and writes retained" if is_bpc else None,
        "effective_zerorope": 0 if is_bpc else "runner argument/default",
        "binarypc_fit_tokens": ("direction_sample_size, even linspace spacing as GPC" if sampled_fit else "all") if is_bpc else None,
        "binarypc_zero_code_storage": "+1 (original BPC bit=0 convention)",
        "decode": "original PiCoKV packed attention", "argv": forwarded,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path(__file__).with_name("encoding.py"), Path(__file__).with_name("source_encoding.py"), ROOT / "src/BPC/bpc/bpc/bpc_pytorch.py"]
        },
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        for key in ("encoding", "argv", "source_sha256", "git_head", "coding_version"):
            if old.get(key) != metadata[key]:
                raise ValueError(f"{metadata_path} belongs to another configuration; choose a new output")
    if output.exists() and options.experiment == "efficiency":
        existing = json.loads(output.read_text())
        if not existing.get("tag", "").endswith(suffix):
            raise ValueError(f"{output} belongs to another coding variant")
    metadata["started_at"] = datetime.now().astimezone().isoformat()
    metadata["status"] = "running"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    runner.output_tag, runner.load_model = tag, load_model
    sys.argv = [str(Path(__file__).resolve()), *forwarded]
    try:
        with binarypc_encoding(sample_directions=sampled_fit) if is_bpc else nullcontext():
            result = runner.main()
        metadata["status"] = "completed"
        return result
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = repr(error)
        raise
    finally:
        runner.output_tag, runner.load_model, sys.argv = previous_tag, previous_load, previous_argv
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
