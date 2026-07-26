
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Dense-baseline attn implementations selectable via --method. Anything not in
# this set and not "quantpc"/"kivi" is rejected by load_quantpc_model.
BASELINE_ATTN = {"flash_attention_2", "sdpa", "eager"}

# The KIVI reference implementation lives under src/KIVI with its own
# `models.*` / `quant.*` import roots, so that directory must be on sys.path
# before importing LlamaForCausalLM_KIVI / MistralForCausalLM_KIVI.
KIVI_SRC = Path(__file__).resolve().parents[1] / "src" / "KIVI"

# make_cache is a zero-arg callable returning a fresh QuantKVCache (quantpc), or
# None for dense baselines (the model uses its default cache).
CacheFactory = Optional[Callable[[], object]]


def seed_everything(seed: int = 42) -> None:
    """Seed python / numpy / torch (+ cudnn determinism) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def output_tag(method: str, k_bits: int, v_bits: int, refit_decoder: bool = False) -> str:
    """Directory/file tag keyed by method + KV bits, shared across runners."""
    if method == "kivi":
        return f"kivi_k{k_bits}_v{v_bits}"
    if method != "quantpc":
        return method
    tag = f"quantpc_k{k_bits}_v{v_bits}"
    if refit_decoder:
        tag += "_refit"
    return tag


def _load_kivi_model(
    model_name: str,
    *,
    k_bits: int,
    v_bits: int,
    group_size: int,
    residual_length: int,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load the KIVI quantized-KV model (src/KIVI/models/*_kivi.py).

    KIVI carries its quantized KV cache internally (a tuple-based state produced
    by its custom attention forward), so ``make_cache`` is ``None`` and the
    runner must NOT override ``past_key_values`` on ``generate``.
    """
    if str(KIVI_SRC) not in sys.path:
        sys.path.insert(0, str(KIVI_SRC))

    config = AutoConfig.from_pretrained(model_name)
    config.k_bits = k_bits
    config.v_bits = v_bits
    config.group_size = group_size
    config.residual_length = residual_length
    config.use_flash = True  # KIVI attention is only implemented for flash-attn.

    model_type = getattr(config, "model_type", "llama")
    if model_type == "mistral":
        from models.mistral_kivi import MistralForCausalLM_KIVI as KIVIModel
    else:
        from models.llama_kivi import LlamaForCausalLM_KIVI as KIVIModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    # KIVI kernels only support fp16; do not change to bf16.
    model = KIVIModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer, None


def load_quantpc_model(
    model_name: str,
    method: str = "quantpc",
    *,
    k_bits: int = 128,
    v_bits: int = 256,
    projected_query_bits: int = 8,
    exp_bits: int = 4,
    eas: float = 0.01,
    zerorope: int = 30,
    cuda_v1_exp8: bool = False,
    refit_decoder: bool = False,
    use_pruning: bool = True,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    group_size: int = 32,
    residual_length: int = 32,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model+tokenizer and return ``(model, tokenizer, make_cache)``.

    Method dispatch (mirrors the original three runners):
      * ``quantpc``           -> apply ``modify(model)``; ``make_cache`` builds a
                                 fresh ``QuantKVCache`` per call. Backend is the
                                 CUDA impl (``quantpc.quanpc``) when
                                 ``cuda_v1_exp8`` else the torch reference
                                 (``quantpc.quanpc_torch``).
      * ``kivi``              -> load the KIVI quantized-KV model; ``k_bits`` /
                                 ``v_bits`` are the KV quantization bits (2 or 4),
                                 with ``group_size`` / ``residual_length``.
                                 ``make_cache`` is ``None`` (KIVI keeps its own
                                 internal quantized cache).
      * flash_attention_2 / sdpa / eager -> dense full-precision baseline;
                                 ``make_cache`` is ``None`` (default KV cache).
    """
    if method == "kivi":
        return _load_kivi_model(
            model_name,
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            residual_length=residual_length,
        )

    from_pretrained_kwargs = {"torch_dtype": "bfloat16"}
    QuantKVCache = None
    modify = None

    if method == "quantpc":
        if cuda_v1_exp8:
            from quantpc.quanpc import QuantKVCache, modify
        else:
            from quantpc.quanpc_torch import QuantKVCache, modify
    elif method in BASELINE_ATTN:
        from_pretrained_kwargs["attn_implementation"] = method
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'quantpc', 'kivi', {sorted(BASELINE_ATTN)}."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        import accelerate  # noqa: F401
        device_map = "auto"
    except ImportError:
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")

    if modify is not None:
        modify(model)
    model.eval()

    make_cache: CacheFactory = None
    if QuantKVCache is not None:
        # Decode-time pruning is a CUDA-kernel feature (quantpc.quanpc); the
        # torch reference backend does not accept the pruning knobs.
        pruning_kwargs = (
            dict(use_pruning=use_pruning, prune_mode=prune_mode, log_eps=log_eps)
            if cuda_v1_exp8
            else {}
        )

        def make_cache():
            return QuantKVCache(
                k_bits=k_bits,
                v_bits=v_bits,
                projected_query_bits=projected_query_bits,
                exp_bits=exp_bits,
                eas=eas,
                zerorope=zerorope,
                use_cuda_v1_exp8=cuda_v1_exp8,
                refit_decoder=refit_decoder,
                **pruning_kwargs,
            )

    return model, tokenizer, make_cache
