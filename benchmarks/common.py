
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Dense-baseline attn implementations selectable via --method. Anything not in
# this set and not "quantpc"/"kivi"/"oscar" is rejected by load_model.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_offline_proj_path(
    method: str,
    offline: bool,
    offline_proj_path: Optional[str],
    *,
    model_name: str,
    k_bits: int,
    v_bits: int,
) -> Optional[str]:
    """Return the effective offline block-32 encoder .pt path, or None for online.

    Offline is requested when ``--offline`` is set or an explicit path is given
    (quantpc only). With no explicit path, the default is derived from the model
    tag and bits: ``offline_projection/projections/{model_tag}_k{k}_v{v}_block32.pt``. Raises
    a clear error when the resolved file is missing.
    """
    if method != "quantpc":
        return None
    if not offline and offline_proj_path is None:
        return None
    model_tag = Path(model_name.rstrip("/")).name
    path = offline_proj_path or str(
        _REPO_ROOT / "offline_projection" / "projections" / f"{model_tag}_k{k_bits}_v{v_bits}_block32.pt"
    )
    if not Path(path).exists():
        raise FileNotFoundError(
            f"offline projection not found: {path}. Generate it with "
            f"offline_projection/create_offline_projections.py, or pass --offline-proj-path."
        )
    return path


def seed_everything(seed: int = 42) -> None:
    """Seed python / numpy / torch (+ cudnn determinism) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def output_tag(
    method: str,
    k_bits: int,
    v_bits: int,
    num_bits: int = 2,
    *,
    offline: bool = False,
    offline_mode: str = "both",
) -> str:
    """Directory/file tag keyed by method + KV bits, shared across runners.

    For quantpc, ``offline`` appends ``_offline`` (plus the side when not
    ``both``) so online and offline runs at the same bits do not overwrite.
    """
    if method == "kivi":
        return f"kivi_k{k_bits}_v{v_bits}"
    if method == "oscar":
        return f"oscar_int{num_bits}"
    if method != "quantpc":
        return method
    tag = f"quantpc_k{k_bits}_v{v_bits}"
    if offline:
        tag += "_offline" if offline_mode == "both" else f"_offline_{offline_mode}"
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
    offline_v_hadamard: bool = False,
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

    # OScaR's INT2 default recipe absorbs a per-head Hadamard rotation into the
    # v_proj/o_proj weights offline (QuaRot-style). This is an exact orthogonal
    # transform (Hadamard is orthogonal, so o_proj @ (V @ H) @ H^T == original),
    # so outputs are unchanged in full precision but V becomes far more 2-bit
    # quantization-friendly. Must run AFTER weights are loaded.
    if offline_v_hadamard:
        from example import apply_offline_v_hadamard
        had_dim = apply_offline_v_hadamard(model)
        print(f"[oscar] applied offline V Hadamard to v_proj/o_proj (had_dim={had_dim})")

    # The oscar attention backend calls `past_key_value.update_pack(...)`, which
    # only exists on oscar.DynamicCache. HF generate() would otherwise build a
    # stock transformers.DynamicCache (no update_pack), so we hand the runner a
    # factory to inject a fresh oscar cache per call (like quantpc's make_cache).
    from oscar import Cache as OscarCache, DynamicCache as OscarDynamicCache

    # HF generate()'s `_get_initial_cache_position` does `isinstance(cache, Cache)`
    # and, when False, falls back to `cache[0][0].shape[2]` -- which throws on a
    # freshly-built (0-layer) oscar cache. oscar.DynamicCache is NOT a
    # transformers.Cache subclass, so we globally rebind the `Cache`/`DynamicCache`
    # names to oscar's own classes to make that isinstance succeed (then HF takes
    # the `get_seq_length()` path, which returns 0 for an empty cache). NOTE: the
    # rebind must target `transformers.generation.utils` -- that module did
    # `from ..cache_utils import Cache, DynamicCache` at import time, so patching
    # only `transformers.cache_utils.*` would be too late for its bound names.
    # Scoped to the oscar load path (one method == one process in the runners), so
    # quantpc/kivi/dense baselines are unaffected.
    import transformers.cache_utils as _tf_cache_utils
    import transformers.generation.utils as _tf_gen_utils

    _tf_cache_utils.Cache = OscarCache
    _tf_cache_utils.DynamicCache = OscarDynamicCache
    _tf_gen_utils.Cache = OscarCache
    _tf_gen_utils.DynamicCache = OscarDynamicCache

    def make_cache():
        return OscarDynamicCache()

    return model, tokenizer, make_cache


def _auto_device_map():
    """Return "auto" when accelerate is installed (multi-GPU sharding), else None."""
    try:
        import accelerate  # noqa: F401
        return "auto"
    except ImportError:
        return None


def _load_quantpc_model(
    model_name: str,
    *,
    k_bits: int,
    v_bits: int,
    sink_size: int,
    local_size: int,
    exp_bits: int,
    eas: float,
    zerorope: int,
    use_pruning: bool,
    prune_mode: int,
    log_eps: float,
    whiten: bool = False,
    offline_proj_path: Optional[str] = None,
    offline_proj_mode: str = "both",
    offline_decoder: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model patched with the quanpc_standalone online-projection path.

    The project ships a single production standalone path (quanpc_standalone): a
    CUDA-only online-projection decode with fixed 8-bit projected queries. The old
    torch reference backend, refit decoder, and tunable projected_query_bits were
    removed. ``make_cache`` returns a fresh ``StandaloneLlamaCache`` per call (the
    model must have been patched via ``modify_llama_for_standalone``).

    When ``offline_proj_path`` is given, the stored block-32 encoders are loaded
    once and threaded into every cache so prefill skips the online Householder QR
    (the decoder is still solved per context). ``offline_proj_mode`` selects which
    side reuses stored encoders: ``both`` (default), ``key``, or ``value``.
    """
    from quantpc.quanpc_standalone import StandaloneConfig
    from quantpc.quanpc_standalone.transformers import (
        StandaloneLlamaCache,
        load_offline_projections,
        modify_llama_for_standalone,
    )

    standalone_config = StandaloneConfig(
        key_bits=k_bits,
        value_bits=v_bits,
        sink_size=sink_size,
        local_size=local_size,
        eas=eas,
        zerorope=zerorope,
        exp_bits=exp_bits,
        use_pruning=use_pruning,
        prune_mode=prune_mode,
        log_eps=log_eps,
        whiten=whiten,
    )

    key_encoders = value_encoders = None
    key_decoders = value_decoders = None
    if offline_proj_path is not None:
        if offline_proj_mode not in ("both", "key", "value"):
            raise ValueError(f"offline_proj_mode must be 'both', 'key' or 'value'; got {offline_proj_mode!r}")
        _projs = load_offline_projections(offline_proj_path)
        if offline_proj_mode in ("both", "key"):
            key_encoders = _projs["key_encoder"]
        if offline_proj_mode in ("both", "value"):
            value_encoders = _projs["value_encoder"]
        # A decoder file also carries encoders, so `offline_decoder` only
        # controls whether the stored decoder is used on top of them.
        if offline_decoder:
            if "key_decoder" not in _projs:
                raise ValueError(
                    f"{offline_proj_path} stores encoders only; rebuild it with "
                    "offline_projection/create_offline_projections.py to use --offline-decoder"
                )
            if offline_proj_mode in ("both", "key"):
                key_decoders = _projs["key_decoder"]
            if offline_proj_mode in ("both", "value"):
                value_decoders = _projs["value_decoder"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_map = _auto_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype="bfloat16",
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")
    modify_llama_for_standalone(model)
    model.eval()

    def make_cache():
        return StandaloneLlamaCache(
            standalone_config,
            key_encoders=key_encoders,
            value_encoders=value_encoders,
            key_decoders=key_decoders,
            value_decoders=value_decoders,
        )

    return model, tokenizer, make_cache


def _load_baseline_model(
    model_name: str,
    *,
    attn_implementation: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a dense full-precision baseline (flash_attention_2 / sdpa / eager).

    No quantized KV cache: ``make_cache`` is ``None`` so the runner lets the model
    use its default (DynamicCache) KV cache.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_map = _auto_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype="bfloat16",
        attn_implementation=attn_implementation,
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer, None


def load_model(
    model_name: str,
    method: str = "quantpc",
    *,
    k_bits: int = 128,
    v_bits: int = 256,
    sink_size: int = 64,
    local_size: int = 1024,
    exp_bits: int = 4,
    eas: float = 0.01,
    zerorope: int = 30,
    use_pruning: bool = True,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    whiten: bool = False,
    group_size: int = 32,
    residual_length: int = 32,
    num_bits: int = 4,
    quant_mode: str = "k-channel",
    kv_rotation: str = "none",
    kv_norm: str = "0",
    residual_evict_size: Optional[int] = None,
    offline_v_hadamard: bool = False,
    offline_proj_path: Optional[str] = None,
    offline_proj_mode: str = "both",
    offline_decoder: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model+tokenizer and return ``(model, tokenizer, make_cache)``.

    Method dispatch (each backend has its own ``_load_*`` helper):
      * ``quantpc``           -> ``_load_quantpc_model`` (quanpc_standalone
                                 online-projection path, CUDA-only, GQA=4, bf16;
                                 fresh ``StandaloneLlamaCache`` per call).
      * ``kivi``              -> ``_load_kivi_model``; ``k_bits`` / ``v_bits`` are
                                 the KV quantization bits (2 or 4), with
                                 ``group_size`` / ``residual_length``.
                                 ``make_cache`` is ``None`` (KIVI keeps its own
                                 internal quantized cache).
      * ``oscar``             -> ``_load_oscar_model`` (src/OScaR-KV-Quant);
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
            offline_v_hadamard=offline_v_hadamard,
        )

    if method == "quantpc":
        return _load_quantpc_model(
            model_name,
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
            whiten=whiten,
            offline_proj_path=offline_proj_path,
            offline_proj_mode=offline_proj_mode,
            offline_decoder=offline_decoder,
        )

    if method not in BASELINE_ATTN:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'quantpc', 'kivi', 'oscar', {sorted(BASELINE_ATTN)}."
        )
    return _load_baseline_model(model_name, attn_implementation=method)
