
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from quantpc.model_config import load_model_config

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

# PolarQuant ships a custom HF modeling file (models/modeling_llama_polar.py) plus a
# triton decode kernel (models/kernel4group.py). Its Mixed (K+V) variant reuses KIVI's
# value-quant kernels under src/KIVI/quant, so _load_polar_model puts both trees on
# sys.path. See _load_polar_model for the (models-package) name-collision handling.
POLAR_SRC = Path(__file__).resolve().parents[1] / "src" / "PolarQuant"

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
    whiten: bool = False,
) -> Optional[str]:
    """Return the effective offline encoder .pt path, or None for online.

    Offline is requested when ``--offline`` is set or an explicit path is given
    (quantpc only). With no explicit path, the default is derived from the model
    tag and bits: ``offline_projection/projections/{model_tag}_k{k}_v{v}_nowhiten.pt``. Raises
    a clear error when the resolved file is missing. Offline projections already
    contain the complete fitted basis, so runtime whitening is intentionally
    rejected rather than combining incompatible bases.
    """
    if method != "quantpc":
        return None
    if not offline and offline_proj_path is None:
        return None
    if whiten:
        raise ValueError(
            "offline projection and runtime --whiten cannot be combined: "
            "offline PTs use the model basis, while whitened PTs already fold "
            "the whitening into encoder/decoder. Use offline without --whiten "
            "or online with --whiten."
        )
    model_tag = Path(model_name.rstrip("/")).name
    path = offline_proj_path or str(
        _REPO_ROOT / "offline_projection" / "projections" / f"{model_tag}_k{k_bits}_v{v_bits}_nowhiten.pt"
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
    whiten: bool | None = None,
    online_decoder_refit: str = "none",
) -> str:
    """Directory/file tag keyed by method + KV bits, shared across runners.

    Passing ``whiten`` switches quantpc to the full naming --
    ``{online|offline}_k{K}_v{V}_{whiten|nowhiten}``. The local window is fixed
    at 256 across the current experiments, so it is intentionally omitted.
    """
    if method == "kivi":
        return f"kivi_k{k_bits}_v{v_bits}"
    if method == "oscar":
        return f"oscar_int{num_bits}"
    if method != "quantpc":
        return method
    if whiten is not None:
        mode = "offline" if offline else "online"
        tag = f"{mode}_k{k_bits}_v{v_bits}_{'whiten' if whiten else 'nowhiten'}"
        refit_suffix = {"none": "", "k": "_krefit", "v": "_vrefit", "both": "_refit"}
        if online_decoder_refit not in refit_suffix:
            raise ValueError(f"unknown online_decoder_refit: {online_decoder_refit}")
        return tag + refit_suffix[online_decoder_refit]
    tag = f"quantpc_k{k_bits}_v{v_bits}"
    if offline:
        tag += "_offline"
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

    config = load_model_config(model_name, trust_remote_code=True)
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
    online_decoder_refit: str = "none",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model patched with the quanpc_standalone online-projection path.

    The project ships a single production standalone path (quanpc_standalone): a
    CUDA-only online-projection decode with fixed 8-bit projected queries.
    ``make_cache`` returns a fresh ``StandaloneLlamaCache`` per call (the model
    must have been patched via the model-specific standalone adapter).

    When ``offline_proj_path`` is given, the stored block-32 pair is loaded
    once and threaded into every cache so prefill skips the online Householder QR
    """
    from quantpc.quanpc_standalone import StandaloneConfig
    from quantpc.quanpc_standalone.transformers import (
        StandaloneLlamaCache,
        load_offline_projections,
        modify_llama_for_standalone,
        modify_qwen3_for_standalone,
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
        if whiten:
            raise ValueError(
                "offline projection and runtime whitening cannot be combined; "
                "use a folded offline PT without --whiten, or use online whitening."
            )
        _projs = load_offline_projections(offline_proj_path)
        # The offline projection contains both encoders and decoders.  Refit is
        # controlled globally by side: the selected side(s) keep the offline
        # encoder but solve the decoder online for the current context.
        refit_sides = {
            "none": (), "k": ("key",), "v": ("value",),
            "both": ("key", "value"),
        }[online_decoder_refit]
        stored_sides = tuple(
            side for side in ("key", "value") if side not in refit_sides
        )
        for side in stored_sides:
            if f"{side}_decoder" not in _projs:
                raise ValueError(
                    f"{offline_proj_path} stores encoders only; rebuild it with "
                    "offline_projection/create_offline_projections.py"
                )
        key_encoders = _projs["key_encoder"]
        value_encoders = _projs["value_encoder"]
        if "key" in stored_sides:
            key_decoders = _projs["key_decoder"]
        if "value" in stored_sides:
            value_decoders = _projs["value_decoder"]
        print(
            f"online decoder refit sides: {refit_sides or '(none, fully offline decoder)'}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # With a single visible GPU, device_map="auto" can offload layers to CPU
    # under transient memory pressure. Pin the whole model to that GPU; keep
    # "auto" only for genuine multi-GPU sharding.
    if torch.cuda.is_available() and torch.cuda.device_count() <= 1:
        device_map = {"": 0}
    else:
        device_map = _auto_device_map()
    config = load_model_config(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        torch_dtype="bfloat16",
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")
    if getattr(config, "model_type", None) == "qwen3":
        modify_qwen3_for_standalone(model)
    else:
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


def _load_polar_model(
    model_name: str,
    *,
    polar_mode: str = "mixed",
    rbits: int = 3,
    tbits: int = 3,
    vbits: int = 2,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load PolarQuant's custom HF model (src/PolarQuant/models/modeling_llama_polar.py).

    ``polar_mode`` selects the attention variant baked into every decoder layer:
      * ``group``  -> K-only polar quant (no extra deps).
      * ``mixed``  -> K polar + V group-quant; reuses KIVI's value kernels under
                      src/KIVI/quant (triton_quantize_and_pack_along_last_dim /
                      cuda_bmm_fA_qB_outer, backed by the compiled kivi_gemv .so).
      * ``snapkv`` -> SnapKV eviction + K polar quant.
    ``rbits/tbits`` are the polar radius/angle bits; ``vbits`` the V bits (mixed only).

    PolarQuant drives generation through its own ``generate`` free function (not HF
    ``model.generate``) and keeps a tuple-of-tuples KV cache built internally from
    ``past_key_values=None``. So ``make_cache`` is ``None`` (the runner must NOT
    override ``past_key_values``), and we monkeypatch ``model.generate`` with a thin
    HF-compatible adapter. Config comes from the runner's CLI flags
    (``--polar-mode / --polar-rbits / --polar-tbits / --polar-vbits``).

    NOTE: the accuracy runners feed one prompt at a time (PolarQuant's decode skips
    padding-mask handling and tracks a single scalar kv_seq_len); the Efficiency
    runners use equal-length synthetic inputs, so batch>1 is safe there.
    """
    # Both KIVI and PolarQuant ship a top-level ``models`` package. KIVI's has an
    # __init__.py and would shadow PolarQuant's namespace, so import KIVI kernels
    # first, then remove its package root before importing PolarQuant's model.
    if polar_mode == "mixed":
        for p in (str(KIVI_SRC), str(KIVI_SRC / "quant")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import quant.new_pack  # noqa: F401
        import quant.matmul  # noqa: F401
        if str(KIVI_SRC) in sys.path:
            sys.path.remove(str(KIVI_SRC))
    if str(POLAR_SRC) not in sys.path:
        sys.path.insert(0, str(POLAR_SRC))

    from models.modeling_llama_polar import (  # noqa: E402
        LlamaPolarForCausalLM,
        generate as polar_generate,
    )

    config = AutoConfig.from_pretrained(model_name)
    config.polar_mode = polar_mode
    config.polar_rbits = rbits
    config.polar_tbits = tbits
    config.polar_vbits = vbits

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gen_cfg = GenerationConfig.from_pretrained(model_name)

    # kernel4group and the KIVI value kernels are fp16-only; do NOT use bf16.
    model = LlamaPolarForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    @torch.no_grad()
    def _generate(
        input_ids=None,
        attention_mask=None,
        max_new_tokens=32,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=False,
        **_ignored,
    ):
        cfg = gen_cfg
        if eos_token_id is not None:
            cfg = copy.deepcopy(gen_cfg)
            cfg.eos_token_id = eos_token_id
        gen_ids = polar_generate(
            model, input_ids, attention_mask, cfg,
            do_sample=do_sample, max_new_tokens=max_new_tokens,
        )
        # PolarQuant returns only new tokens; HF callers expect prompt + continuation.
        full = torch.cat([input_ids, gen_ids], dim=-1)
        if return_dict_in_generate:
            from transformers.generation.utils import GenerateDecoderOnlyOutput
            return GenerateDecoderOnlyOutput(sequences=full)
        return full

    model.generate = _generate
    return model, tokenizer, None


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
    # With a single visible GPU, device_map="auto" can offload a tiny buffer to
    # CPU, which breaks flash_attention_2 (its kernel is CUDA-only). Pin the
    # whole model to that GPU; keep "auto" only for genuine multi-GPU sharding.
    if torch.cuda.is_available() and torch.cuda.device_count() <= 1:
        device_map = {"": 0}
    else:
        device_map = _auto_device_map()
    config = load_model_config(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
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
    k_bits: Optional[int] = None,
    v_bits: Optional[int] = None,
    sink_size: int = 64,
    local_size: int = 256,
    exp_bits: int = 4,
    eas: float = 0.01,
    zerorope: int = 20,
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
    polar_mode: str = "mixed",
    polar_rbits: int = 3,
    polar_tbits: int = 3,
    polar_vbits: int = 2,
    online_decoder_refit: str = "none",
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
      * ``polar``             -> ``_load_polar_model`` (src/PolarQuant);
                                 ``polar_mode`` / ``polar_rbits`` / ``polar_tbits`` /
                                 ``polar_vbits`` set the PolarQuant configuration.
                                 ``make_cache`` is ``None`` (PolarQuant owns its cache).
      * flash_attention_2 / sdpa / eager -> dense full-precision baseline;
                                 ``make_cache`` is ``None`` (default KV cache).
    """
    if k_bits is None:
        k_bits = 2 if method == "kivi" else 320
    if v_bits is None:
        v_bits = 2 if method == "kivi" else 320

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
            online_decoder_refit=online_decoder_refit,
        )

    if method == "polar":
        return _load_polar_model(
            model_name,
            polar_mode=polar_mode,
            rbits=polar_rbits,
            tbits=polar_tbits,
            vbits=polar_vbits,
        )

    if method not in BASELINE_ATTN:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'quantpc', 'kivi', 'oscar', 'polar', {sorted(BASELINE_ATTN)}."
        )
    return _load_baseline_model(model_name, attn_implementation=method)
