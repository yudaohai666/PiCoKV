"""Experiment runner for the QuanPC quantized KV-cache algorithm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# Make `import quantpc` / `from benchmarks.common import ...` resolvable from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from needle import LLMNeedleHaystackTester, ModelProvider  # noqa: E402
from benchmarks.common import load_model, output_tag, resolve_offline_proj_path, seed_everything  # noqa: E402


class Transformer(ModelProvider):
    def __init__(
        self,
        model_name: str = "/data/Llama-3.1-8B-Instruct",
        method: str = "quantpc",
        # quantpc KV-cache params (--method quantpc)
        k_bits: int = 320,
        v_bits: int = 320,
        sink_size: int = 64,
        local_size: int = 256,
        exp_bits: int = 4,
        eas: float = 0.01,
        zerorope: int = 0,
        use_pruning: bool = True,
        prune_mode: int = 3,
        log_eps: float = -10.0,
        # KIVI params (--method kivi)
        group_size: int = 32,
        residual_length: int = 32,
        # OScaR params (--method oscar)
        # OScaR params (--method oscar) —— 默认 = INT2 官方配方
        num_bits: int = 2,
        quant_mode: str = "k-channel",
        kv_rotation: str = "hadamard",
        kv_norm: str = "1",
        residual_evict_size: int | None = 256,
        offline_v_hadamard: bool = True,
        # quantpc offline block-32 encoders (--method quantpc)
        offline_proj_path: str | None = None,
        whiten: bool = False,
        max_new_tokens: int = 50,
        online_decoder_refit: str = "none",
        # PolarQuant params (--method polar)
        polar_mode: str = "mixed",
        polar_rbits: int = 3,
        polar_tbits: int = 3,
        polar_vbits: int = 2,
    ):
        self.max_new_tokens = int(max_new_tokens)
        self.method = method
        # Shared loader handles method dispatch + modify() + KV-cache factory.
        self.model, self.tokenizer, self._make_cache = load_model(
            model_name,
            method=method,
            # quantpc KV-cache params (--method quantpc)
            k_bits=k_bits,
            v_bits=v_bits,
            sink_size=sink_size,
            local_size=local_size,
            exp_bits=exp_bits,
            eas=eas,
            zerorope=zerorope,
            use_pruning=use_pruning,
            prune_mode=prune_mode,
            log_eps=log_eps,
            offline_proj_path=offline_proj_path,
            whiten=whiten,
            # KIVI params (--method kivi)
            group_size=group_size,
            residual_length=residual_length,
            # OScaR params (--method oscar)
            num_bits=num_bits,
            quant_mode=quant_mode,
            kv_rotation=kv_rotation,
            kv_norm=kv_norm,
            residual_evict_size=residual_evict_size,
            offline_v_hadamard=offline_v_hadamard,
            online_decoder_refit=online_decoder_refit,
            # PolarQuant params (--method polar)
            polar_mode=polar_mode,
            polar_rbits=polar_rbits,
            polar_tbits=polar_tbits,
            polar_vbits=polar_vbits,
        )

    def evaluate_model(self, prompt, needle=None) -> str:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=False,
        ).to(self.model.device)

        with torch.no_grad():
            generate_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
            # quantpc injects a fresh QuantKVCache per request; baselines use the
            # model's default cache (no past_key_values override).
            past_key_values = self._make_cache() if self._make_cache is not None else None
            if past_key_values is not None:
                generate_kwargs["past_key_values"] = past_key_values
            output = self.model.generate(**generate_kwargs)
            generated_ids = output.sequences[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        del past_key_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI bot that answers questions for a user. "
                    "Keep your response short and direct"
                ),
            },
            {
                "role": "user",
                "content": context,
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
                # "content": f"{retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it"
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def run_experiment(
    model_name: str,
    lengths: list[int],
    depths: list[int],
    output_path: str,
    method: str,
    # quantpc KV-cache params (--method quantpc)
    k_bits: int,
    v_bits: int,
    sink_size: int,
    local_size: int,
    exp_bits: int,
    eas: float,
    zerorope: int,
    use_pruning: bool = True,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    # KIVI params (--method kivi)
    group_size: int = 32,
    residual_length: int = 32,
    # OScaR params (--method oscar)
    # OScaR params (--method oscar) —— 默认 = INT2 官方配方
    num_bits: int = 2,
    quant_mode: str = "k-channel",
    kv_rotation: str = "hadamard",
    kv_norm: str = "1",
    residual_evict_size: int | None = 256,
    offline_v_hadamard: bool = True,
    offline_proj_path: str | None = None,
    whiten: bool = False,
    max_new_tokens: int = 50,
    online_decoder_refit: str = "none",
    # PolarQuant params (--method polar)
    polar_mode: str = "mixed",
    polar_rbits: int = 3,
    polar_tbits: int = 3,
    polar_vbits: int = 2,
) -> dict[tuple[int, int], str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = Transformer(
        model_name,
        method=method,
        # quantpc KV-cache params (--method quantpc)
        k_bits=k_bits,
        v_bits=v_bits,
        sink_size=sink_size,
        local_size=local_size,
        exp_bits=exp_bits,
        eas=eas,
        zerorope=zerorope,
        use_pruning=use_pruning,
        prune_mode=prune_mode,
        log_eps=log_eps,
        offline_proj_path=offline_proj_path,
        whiten=whiten,
        max_new_tokens=max_new_tokens,
        # KIVI params (--method kivi)
        group_size=group_size,
        residual_length=residual_length,
        # OScaR params (--method oscar)
        num_bits=num_bits,
        quant_mode=quant_mode,
        kv_rotation=kv_rotation,
        kv_norm=kv_norm,
        residual_evict_size=residual_evict_size,
        offline_v_hadamard=offline_v_hadamard,
        online_decoder_refit=online_decoder_refit,
        polar_mode=polar_mode,
        polar_rbits=polar_rbits,
        polar_tbits=polar_tbits,
        polar_vbits=polar_vbits,
    )
    tester = LLMNeedleHaystackTester(model)
    results = {}

    # Resume: load previously completed (length, depth) results so we skip them.
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for key, value in prev.items():
                length_str, _, depth_str = key.partition("-")
                results[(int(length_str), int(depth_str))] = value
            print(f"[resume] loaded {len(results)} completed cases from {output_path}")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            print(f"[resume] ignoring unreadable checkpoint {output_path}: {exc}")

    for length in lengths:
        for depth in depths:
            if (length, depth) in results:
                print((length, depth), "[skip] already done")
                continue
            results[(length, depth)] = tester.evaluate(length, depth)
            print((length, depth), results[(length, depth)])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({f"{k[0]}-{k[1]}": v for k, v in results.items()}, f, indent=4)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run needle-in-a-haystack experiments with quantized KV cache.")
    parser.add_argument("--model", default="/data/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--method",
        default="quantpc",
        help=(
            "evaluation method: 'quantpc' / 'kivi' / 'oscar' / 'polar' (quantized KV cache), "
            "or a dense baseline attn implementation 'flash_attention_2' / 'sdpa' / 'eager'"
        ),
    )
    parser.add_argument("--lengths", default="64000,80000,100000", help="comma-separated context lengths")
    parser.add_argument("--depths", default="50", help="comma-separated needle depths")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: results/niah/{model}/{tag}/results.json)")
    # ── quantpc KV-cache params (--method quantpc) ──
    parser.add_argument("--k-bits", type=int, default=None, help="QuanPC K projection bits; KIVI defaults to 2")
    parser.add_argument("--v-bits", type=int, default=None, help="QuanPC V projection bits; KIVI defaults to 2")
    parser.add_argument(
        "--sink-size",
        type=int,
        default=64,
        help="number of initial (sink) tokens kept in full precision",
    )
    parser.add_argument(
        "--local-size",
        type=int,
        default=256,
        help="number of most-recent (local) tokens kept in full precision",
    )
    parser.add_argument(
        "--exp-bits",
        type=int,
        default=4,
        help="unsigned per-64-token quantization bits for global exp weights; quantization is always enabled",
    )
    parser.add_argument("--eas", type=float, default=0.01)
    parser.add_argument("--zerorope", type=int, default=0)
    parser.add_argument(
        "--no-pruning",
        dest="use_pruning",
        action="store_false",
        help="关闭解码时的 global 剪枝（gqa4 CUDA 路径），默认开启",
    )
    parser.add_argument("--prune-mode", type=int, default=3, help="解码剪枝模式（CUDA v1_exp8 路径）")
    parser.add_argument("--log-eps", type=float, default=-10.0, help="解码剪枝 log-epsilon 阈值")
    parser.set_defaults(use_pruning=True)
    # ── KIVI params (--method kivi; also uses --k-bits/--v-bits as 2/4-bit KV) ──
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="KIVI per-group quantization size; OScaR group size (INT2 必须=32)",
    )
    parser.add_argument(
        "--residual-length",
        type=int,
        default=32,
        help="KIVI number of recent fp16 tokens kept unquantized",
    )
    # ── OScaR params (--method oscar) —— 默认 = INT2 官方配方 ──
    parser.add_argument("--num-bits", type=int, default=2, help="OScaR KV quant bits (INT2 默认)")
    parser.add_argument("--quant-mode", default="k-channel", help="OScaR quant mode")
    parser.add_argument("--kv-rotation", default="hadamard", help="OScaR kv rotation (INT2 默认 hadamard)")
    parser.add_argument("--kv-norm", default="1", help="OScaR kv norm (INT2 默认 1)")
    parser.add_argument("--residual-evict-size", type=int, default=256, help="OScaR residual evict size (INT2 默认 256)")
    parser.add_argument(
        "--offline-v-hadamard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OScaR: absorb an offline per-head Hadamard rotation into v_proj/o_proj (INT2 默认开启; --no-offline-v-hadamard 关闭)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="quantpc: use offline encoders (default path offline_projection/projections/{model}_k{K}_v{V}_nowhiten.pt)",
    )
    parser.add_argument(
        "--offline-proj-path",
        default=None,
        help="quantpc: explicit offline encoder .pt (overrides the --offline default path)",
    )
    parser.add_argument("--online-decoder-refit", default="none",
                        choices=("none", "k", "v", "both"),
                        help="quantpc offline only: refit decoder for K, V, or both sides")
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="quantpc: rotate K/V into the query-whitened basis at runtime; a "
             "folded projection already has this baked into its stored pair",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    # ── PolarQuant params (--method polar) ──
    parser.add_argument("--polar-mode", default="mixed", choices=["mixed", "group", "snapkv"])
    parser.add_argument("--polar-rbits", type=int, default=3)
    parser.add_argument("--polar-tbits", type=int, default=3)
    parser.add_argument("--polar-vbits", type=int, default=2)
    parser.add_argument("--decoder-ridge", type=float, default=1e-2,
                        help="override the online decoder ridge (default 1e-2; offline PTs are fitted with 1e-3)")
    args = parser.parse_args()
    if args.k_bits is None:
        args.k_bits = 2 if args.method == "kivi" else 320
    if args.v_bits is None:
        args.v_bits = 2 if args.method == "kivi" else 320

    if args.method == "quantpc":
        import quantpc.quanpc_standalone.online as online_mod
        online_mod.ONLINE_DECODER_RIDGE = float(args.decoder_ridge)
        print(f"online decoder ridge set to {args.decoder_ridge}", flush=True)

    seed_everything(args.seed)
    offline_proj_path = resolve_offline_proj_path(
        args.method, args.offline, args.offline_proj_path,
        model_name=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
        whiten=args.whiten,
    )
    output_path = args.output
    if output_path is None:
        model_key = Path(args.model.rstrip("/")).name
        tag = output_tag(
            args.method, args.k_bits, args.v_bits, args.num_bits,
            offline=(args.offline or args.offline_proj_path is not None),
            whiten=args.whiten if args.method == "quantpc" else None,
            online_decoder_refit=args.online_decoder_refit,
        )
        output_path = str(Path("results/niah") / model_key / tag / "results.json")
    run_experiment(
        model_name=args.model,
        lengths=parse_int_list(args.lengths),
        depths=parse_int_list(args.depths),
        output_path=output_path,
        method=args.method,
        # quantpc KV-cache params (--method quantpc)
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        sink_size=args.sink_size,
        local_size=args.local_size,
        exp_bits=args.exp_bits,
        eas=args.eas,
        zerorope=args.zerorope,
        use_pruning=args.use_pruning,
        prune_mode=args.prune_mode,
        log_eps=args.log_eps,
        # KIVI params (--method kivi)
        group_size=args.group_size,
        residual_length=args.residual_length,
        # OScaR params (--method oscar)
        num_bits=args.num_bits,
        quant_mode=args.quant_mode,
        kv_rotation=args.kv_rotation,
        kv_norm=args.kv_norm,
        residual_evict_size=args.residual_evict_size,
        offline_v_hadamard=args.offline_v_hadamard,
        offline_proj_path=offline_proj_path,
        whiten=args.whiten,
        max_new_tokens=args.max_new_tokens,
        online_decoder_refit=args.online_decoder_refit,
        polar_mode=args.polar_mode,
        polar_rbits=args.polar_rbits,
        polar_tbits=args.polar_tbits,
        polar_vbits=args.polar_vbits,
    )


if __name__ == "__main__":
    main()
