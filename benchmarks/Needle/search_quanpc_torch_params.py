"""Parameter search for the torch-only QuanPC NIAH implementation.

The script keeps one model loaded and mutates the QuantKVCache options on the
Transformer wrapper between runs.  It is intended for accuracy search, not
throughput benchmarking.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from needle import LLMNeedleHaystackTester
from run_bin import Transformer, parse_int_list


@dataclass(frozen=True)
class Config:
    k_bits: int
    v_bits: int
    projected_query_bits: int
    exp_bits: int
    zerorope: int
    eas: float = 0.01

    @property
    def name(self) -> str:
        return (
            f"k{self.k_bits}_v{self.v_bits}_q{self.projected_query_bits}_"
            f"e{self.exp_bits}_zr{self.zerorope}_eas{self.eas:g}"
        )

    @property
    def cost_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.k_bits + self.v_bits,
            self.k_bits,
            self.v_bits,
            self.projected_query_bits + self.exp_bits,
            self.zerorope,
        )


def contains_needle(response: str) -> tuple[bool, dict[str, bool]]:
    text = response.lower()
    hits = {
        "sandwich": "sandwich" in text,
        "dolores": "dolores" in text,
        "sunny_day": "sunny" in text and "day" in text,
    }
    return all(hits.values()), hits


def load_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_results(path: Path, results: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=4, ensure_ascii=False))


def apply_config(model: Transformer, config: Config) -> None:
    model.k_bits = config.k_bits
    model.v_bits = config.v_bits
    model.projected_query_bits = config.projected_query_bits
    model.exp_bits = config.exp_bits
    model.eas = config.eas
    model.zerorope = config.zerorope


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def make_grid(args) -> list[Config]:
    configs = []
    for eas in parse_float_list(args.eas_values):
        for zerorope in parse_int_list(args.zerorope_values):
            for k_bits in parse_int_list(args.k_bits_values):
                for v_bits in parse_int_list(args.v_bits_values):
                    for q_bits in parse_int_list(args.projected_query_bits_values):
                        for exp_bits in parse_int_list(args.exp_bits_values):
                            configs.append(
                                Config(
                                    k_bits=k_bits,
                                    v_bits=v_bits,
                                    projected_query_bits=q_bits,
                                    exp_bits=exp_bits,
                                    zerorope=zerorope,
                                    eas=eas,
                                )
                            )
    if args.sort_by_cost:
        configs.sort(key=lambda cfg: cfg.cost_key)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Search torch-only QuanPC NIAH parameters.")
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="results/niah/results_torch_quanpc_param_search.json")
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--k-bits-values", default="128,192,256")
    parser.add_argument("--v-bits-values", default="128,192,256")
    parser.add_argument("--projected-query-bits-values", default="4,6,8")
    parser.add_argument("--exp-bits-values", default="4,6,8")
    parser.add_argument("--zerorope-values", default="10,15,20")
    parser.add_argument("--eas-values", default="0.01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=0, help="0 means no limit")
    parser.add_argument("--stop-after-first-ok", action="store_true")
    parser.add_argument("--sort-by-cost", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    results = load_results(output_path)
    configs = make_grid(args)

    model = Transformer(model_name=args.model)
    tester = LLMNeedleHaystackTester(model)

    new_runs = 0
    for config in configs:
        name = config.name
        if name in results:
            continue

        apply_config(model, config)
        set_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        started = time.time()
        try:
            response = tester.evaluate(args.length, args.depth)
            elapsed = time.time() - started
            ok, hits = contains_needle(response)
            result = {
                **asdict(config),
                "name": name,
                "length": args.length,
                "depth": args.depth,
                "seed": args.seed,
                "response": response,
                "hits": hits,
                "ok": ok,
                "elapsed_sec": round(elapsed, 2),
            }
            print(name, "OK" if ok else "FAIL", round(elapsed, 2), response, flush=True)
        except Exception as exc:
            elapsed = time.time() - started
            result = {
                **asdict(config),
                "name": name,
                "length": args.length,
                "depth": args.depth,
                "seed": args.seed,
                "response": "",
                "hits": {},
                "ok": False,
                "error": repr(exc),
                "elapsed_sec": round(elapsed, 2),
            }
            print(name, "ERROR", repr(exc), flush=True)

        results[name] = result
        save_results(output_path, results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        new_runs += 1
        if args.stop_after_first_ok and result.get("ok"):
            break
        if args.max_new and new_runs >= args.max_new:
            break

    ok_results = [item for item in results.values() if item.get("ok")]
    ok_results.sort(key=lambda item: Config(
        item["k_bits"],
        item["v_bits"],
        item["projected_query_bits"],
        item["exp_bits"],
        item["zerorope"],
        item.get("eas", 0.01),
    ).cost_key)
    if ok_results:
        print("BEST", ok_results[0], flush=True)
    else:
        print("BEST none", flush=True)


if __name__ == "__main__":
    main()
