
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

# OScaR's HF modeling files (evaluation/{llama,qwen3}.py) + the `oscar` package
# live in the copied tree under src/OScaR-KV-Quant. Build the oscar_cuda CUDA
# extension in-place once via src/OScaR-KV-Quant/build_oscar_cuda.sh; that drops
# oscar_cuda.*.so into this dir, so putting OSCAR_SRC on sys.path resolves both
# `import oscar` and `import oscar_cuda`.
OSCAR_SRC = Path(__file__).resolve().parents[1] / "src" / "OScaR-KV-Quant"

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


def _load_oscar_model(
    model_name: str,
    *,
    num_bits: int,
    quant_mode: str,
    group_size: int,
    kv_rotation: str,
    kv_norm: str,
    residual_evict_size: Optional[int],
    device: str = "cuda",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load the OScaR quantized-KV model from the copied src/OScaR-KV-Quant tree.

    OScaR ships its own HF modeling files (evaluation/{llama,qwen3}.py) that wire
    the oscar CUDA attention backend; the quantized KV cache is built INTERNALLY by
    the model forward, so ``make_cache`` is ``None`` (the runner must NOT override
    ``past_key_values``). Mirrors evaluation/bench_e2e.load_model.

    Requires oscar_cuda to have been built in-place first
    (``bash src/OScaR-KV-Quant/build_oscar_cuda.sh``); the resulting .so sits in
    OSCAR_SRC, which we put on sys.path alongside the oscar package + modeling.
    """
    for p in (str(OSCAR_SRC), str(OSCAR_SRC / "evaluation")):
        if p not in sys.path:
            sys.path.insert(0, p)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3":
        from qwen3 import Qwen3ForCausalLM as OscarModel
    elif model_type == "llama":
        from llama import LlamaForCausalLM as OscarModel
    else:
        raise ValueError(f"OScaR supports model_type qwen3/llama; got {model_type!r}.")

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    config._attn_implementation = "flash_attention_2"
    config.attn_backend = "oscar"
    config.num_bits = num_bits
    config.quant_mode = quant_mode
    config.group_size = group_size
    config.kv_rotation = kv_rotation
    config.kv_norm = kv_norm
    config.residual_block_size = 128
    if residual_evict_size is not None:
        config.residual_evict_size = residual_evict_size

    dev = device if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OscarModel.from_pretrained(
        model_name, config=config, device_map={"": dev}, torch_dtype=dtype,
    )
    model.eval()

    # The oscar attention backend calls `past_key_value.update_pack(...)`, which
    # only exists on oscar.DynamicCache. HF generate() would otherwise build a
    # stock transformers.DynamicCache (no update_pack), so we hand the runner a
    # factory to inject a fresh oscar cache per call (like quantpc's make_cache).
    from oscar import DynamicCache as OscarDynamicCache

    def make_cache():
        return OscarDynamicCache()

    return model, tokenizer, make_cache


def load_quantpc_model(
    model_name: str,
    method: str = "quantpc",
    *,
    k_bits: int = 128,
    v_bits: int = 256,
    sink_size: int = 64,
    local_size: int = 1024,
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
    num_bits: int = 4,
    quant_mode: str = "k-channel",
    kv_rotation: str = "none",
    kv_norm: str = "0",
    residual_evict_size: Optional[int] = None,
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
      * ``oscar``             -> load OScaR's custom modeling (src/OScaR-KV-Quant);
                                 ``num_bits`` / ``quant_mode`` / ``group_size`` set
                                 the KV quantization. ``make_cache`` is ``None``
                                 (OScaR builds its quantized cache internally).
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

    if method == "oscar":
        return _load_oscar_model(
            model_name,
            num_bits=num_bits,
            quant_mode=quant_mode,
            group_size=group_size,
            kv_rotation=kv_rotation,
            kv_norm=kv_norm,
            residual_evict_size=residual_evict_size,
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
                sink_size=sink_size,
                local_size=local_size,
                projected_query_bits=projected_query_bits,
                exp_bits=exp_bits,
                eas=eas,
                zerorope=zerorope,
                use_cuda_v1_exp8=cuda_v1_exp8,
                refit_decoder=refit_decoder,
                **pruning_kwargs,
            )

    return model, tokenizer, make_cache
