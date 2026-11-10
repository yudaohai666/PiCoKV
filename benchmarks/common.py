
from __future__ import annotations

import copy
import functools
import inspect
import os
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

from picokv.model_config import load_model_config

# Dense-baseline attn implementations selectable via --method. Anything not in
# this set and not "picokv"/"kivi"/"oscar" is rejected by load_model.
BASELINE_ATTN = {"flash_attention_2", "sdpa", "eager"}

# The KIVI reference implementation lives under src/KIVI with its own
# `models.*` / `quant.*` import roots.  Its CUDA extension is imported as a
# top-level `kivi_gemv`, so both the KIVI root and quant directory are needed.
KIVI_SRC = Path(__file__).resolve().parents[1] / "src" / "KIVI"
KIVI_QUANT_SRC = KIVI_SRC / "quant"

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

# BPC (src/BPC) ships a self-contained hash-based sparse-attention patch
# (bpc.bpc.bpc.modify + AllCache) with a pure-PyTorch fallback -- no CUDA
# extension build required. _load_bpc_model puts its package root on
# sys.path so `import bpc.bpc.bpc` resolves.
BPC_SRC = Path(__file__).resolve().parents[1] / "src" / "BPC"
# Default offline hash-projection checkpoint (mixed-length + mixed-data
# calibration); BPC's online (random) projection is not evaluated here.
BPC_DEFAULT_OFFLINE_PROJ_PATH = str(
    BPC_SRC / "ckp" / "projections" / "llama3-1-8b-ins-projection-mixlen-mixdata.pt"
)

# BitDecoding (src/BitDecoding, branch main-legacy) ships its own HF modeling
# files (evaluation/{llama,qwen3}.py) plus a `bit_decode` package whose CUDA
# extension is built in-place at the repo root
# (CUDA_HOME=/opt/cuda12 CPATH=/usr/local/include/python3.12
#  python setup.py build_ext --inplace -> bit_decode_cuda.*.so). Both the root
# (for `import bit_decode` / `import bit_decode_cuda`) and evaluation/ (for the
# top-level `import llama` / `import qwen3`) go on sys.path, same as OScaR --
# which is a BitDecoding derivative and shares its config surface
# (attn_backend / num_bits / quant_mode / group_size / residual_block_size).
BITDECODING_SRC = Path(__file__).resolve().parents[1] / "src" / "BitDecoding"

# make_cache is a zero-arg callable returning a fresh QuantKVCache (picokv), or
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
    """Return the effective offline encoder .pt path, or None for online.

    Offline is requested when ``--offline`` is set or an explicit path is given
    (picokv only). With no explicit path, the default is derived from the model
    tag and bits: ``offline_projection/projections/{model_tag}_k{k}_v{v}.pt``.
    Raises a clear error when the resolved file is missing.
    """
    if method not in ("picokv", "picokv_recon"):
        return None
    if not offline and offline_proj_path is None:
        return None
    model_tag = Path(model_name.rstrip("/")).name
    path = offline_proj_path or str(
        _REPO_ROOT / "offline_projection" / "projections" / f"{model_tag}_k{k_bits}_v{v_bits}.pt"
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
    full_tag: bool = False,
    group_size: Optional[int] = None,
) -> str:
    """Directory/file tag keyed by method + KV bits, shared across runners.

    ``full_tag`` switches picokv to the full naming ``{online|offline}_k{K}_v{V}``.
    The local window is fixed at 256 across the current experiments, so it is
    intentionally omitted.

    ``group_size`` is only consumed by ``bitdecoding``, where it is NOT a free
    tuning knob but part of the identity of the result: the released kernel is
    only correct at ``group_size == 128`` with ``num_bits == 4`` (see
    ``_load_bitdecoding_model``), so the tag has to make the group size visible
    rather than let two very different numbers collide in one filename.
    """
    if method == "kivi":
        # The group size was historically absent from the KIVI tag, so a g128 run
        # would land in the g32 run's directory and the per-dataset resume logic
        # would splice two configs into one result file. Suffix only on deviation
        # from KIVI's default 32 so existing directory names stay valid.
        tag = f"kivi_k{k_bits}_v{v_bits}"
        if group_size is not None and group_size != 32:
            tag += f"_g{group_size}"
        return tag
    if method == "oscar":
        return f"oscar_int{num_bits}"
    if method == "bpc":
        return "bpc_offline"
    if method == "bitdecoding":
        return f"bitdecoding_{num_bits}bit_g{group_size if group_size is not None else '?'}"
    if method not in ("picokv", "picokv_recon"):
        return method
    if full_tag:
        mode = "offline" if offline else "online"
        return f"{mode}_k{k_bits}_v{v_bits}"
    prefix = "picokv_recon" if method == "picokv_recon" else "picokv"
    tag = f"{prefix}_k{k_bits}_v{v_bits}"
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
    for p in (str(KIVI_SRC), str(KIVI_QUANT_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Use the shared loader so Qwen3 receives the same YaRN/max-position
    # override as the PiCoKV and dense paths.  Llama/Mistral configs are
    # unchanged by load_model_config.
    config = load_model_config(model_name)
    config.k_bits = k_bits
    config.v_bits = v_bits
    config.group_size = group_size
    config.residual_length = residual_length
    config.use_flash = True  # KIVI attention is only implemented for flash-attn.

    model_type = getattr(config, "model_type", "llama")
    if model_type == "qwen3":
        # The copied KIVI implementation reads this legacy Llama attribute
        # directly; Qwen3Config does not define it.
        config.pretraining_tp = 1
        from models.qwen3_kivi import Qwen3ForCausalLM_KIVI as KIVIModel
    elif model_type == "mistral":
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

    config = load_model_config(model_name)
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
    # factory to inject a fresh oscar cache per call (like picokv's make_cache).
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
    # picokv/kivi/dense baselines are unaffected.
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


def _load_picokv_model(
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
    offline_proj_path: Optional[str] = None,
    recon_ablation: bool = False,
    quant_target: str = "both",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model patched with the picokv_standalone online-projection path.

    The project ships a single production standalone path (picokv_standalone): a
    CUDA-only online-projection decode with fixed 8-bit projected queries.
    ``make_cache`` returns a fresh ``StandaloneLlamaCache`` per call (the model
    must have been patched via the model-specific standalone adapter).

    When ``offline_proj_path`` is given, the stored block-32 pair is loaded
    once and threaded into every cache so prefill skips the online Householder QR
    """
    from picokv.picokv_standalone import StandaloneConfig
    from picokv.picokv_standalone.transformers import (
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
        recon_ablation=recon_ablation,
        quant_target=quant_target,
    )

    key_encoders = value_encoders = None
    key_decoders = value_decoders = None
    if offline_proj_path is not None:
        _projs = load_offline_projections(offline_proj_path)
        # An offline projection is used whole: its decoder was fitted against the
        # codes its own encoder produces, so the two only make sense together.
        for side in ("key", "value"):
            if f"{side}_decoder" not in _projs:
                raise ValueError(
                    f"{offline_proj_path} stores encoders only; rebuild it with "
                    "offline_projection/create_offline_projections.py"
                )
        key_encoders = _projs["key_encoder"]
        value_encoders = _projs["value_encoder"]
        key_decoders = _projs["key_decoder"]
        value_decoders = _projs["value_decoder"]
        print(f"offline projection: {offline_proj_path}", flush=True)

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

    # Load both precompiled PiCoKV extensions while the model is being made
    # ready.  This keeps request-time prefill/decode free of extension JIT or
    # shared-object loading.  The loaders still fall back to the historical
    # torch.utils.cpp_extension.load path when install-time artifacts are absent.
    if (
        torch.cuda.is_available()
        and os.environ.get("PICOKV_PRELOAD_EXTENSIONS", "1") != "0"
    ):
        from picokv.picokv_cuda_extension.picokv_cuda import load_picokv_extension
        from picokv.picokv_prefill_cuda_extension.picokv_prefill_cuda import (
            load_picokv_prefill_extension,
        )
        load_picokv_prefill_extension()
        load_picokv_extension()

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

    config = load_model_config(model_name)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3":
        from models.modeling_qwen3_polar import (  # noqa: E402
            Qwen3PolarForCausalLM as PolarModel,
            generate as polar_generate,
        )
    elif model_type == "llama":
        from models.modeling_llama_polar import (  # noqa: E402
            LlamaPolarForCausalLM as PolarModel,
            generate as polar_generate,
        )
    else:
        raise ValueError(f"PolarQuant supports model_type qwen3/llama; got {model_type!r}.")

    config.polar_mode = polar_mode
    config.polar_rbits = rbits
    config.polar_tbits = tbits
    config.polar_vbits = vbits

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gen_cfg = GenerationConfig.from_pretrained(model_name)

    # kernel4group and the KIVI value kernels are fp16-only; do NOT use bf16.
    model = PolarModel.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    @torch.no_grad()
    def _generate(
        input_ids=None,
        attention_mask=None,
        max_new_tokens=None,
        max_length=None,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=False,
        **_ignored,
    ):
        # src/lmeval.py states the budget as max_length = len(context) + max_gen_toks
        # and never passes max_new_tokens. Swallowing max_length in **_ignored capped
        # every polar generation at 32 tokens regardless of the task yaml.
        if max_new_tokens is None:
            if max_length is not None and input_ids is not None:
                max_new_tokens = max(1, int(max_length) - int(input_ids.shape[-1]))
            else:
                max_new_tokens = 32
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


def _load_bpc_model(
    model_name: str,
    *,
    mask_out: float = 0.98,
    min_remain: int = 128,
    token_budget: Optional[int] = None,
    err_ratio: float = 0.1,
    fix_layers: Optional[list[int]] = None,
    use_cuda_kernel: bool = True,
    offline_proj_path: Optional[str] = BPC_DEFAULT_OFFLINE_PROJ_PATH,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model patched with BPC hash-based sparse attention (src/BPC).

    BPC's ``modify()`` replaces each layer's attention forward: prefill
    delegates to flash_attention_2 while also caching KV + a learned 64-bit
    binary hash; decode selects top-k KV by hash score. ``make_cache`` hands
    the runner a fresh ``AllCache`` (a ``transformers.Cache``) per call, same
    shape as picokv's ``StandaloneLlamaCache`` factory. Only Llama / Mistral /
    Qwen2 attention classes are patched (see bpc.bpc.bpc.SUPPORTED_ATTENTION_CLASSES);
    Qwen3 is not yet supported by this method.

    Defaults mirror src/BPC/config/llama3-1-8b-ins/extra_configs/llama3-1-8b-ins-bpc-offline.json
    (``fix_layers=[0, 1]``, ``use_cuda_kernel=True``, offline hash projection). Pass
    ``offline_proj_path=None`` to fall back to BPC's online (random) projection instead.
    """
    if str(BPC_SRC) not in sys.path:
        sys.path.insert(0, str(BPC_SRC))
    from bpc.bpc.bpc import modify, AllCache, load_offline_projections

    config = load_model_config(model_name)
    if torch.cuda.is_available() and torch.cuda.device_count() <= 1:
        device_map = {"": 0}
    else:
        device_map = _auto_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # BPC prefill delegates to the flash forward.
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")

    use_offline_proj = offline_proj_path is not None
    bpc_cfg = dict(
        enable=True,
        fix_layers=fix_layers if fix_layers is not None else [0, 1],
        mask_out=mask_out,
        min_remain=min_remain,
        token_budget=token_budget,
        err_ratio=err_ratio,
        use_offline_proj=use_offline_proj,
        offline_proj_path=offline_proj_path,
        use_cuda_kernel=use_cuda_kernel,
    )
    modify(model, config=bpc_cfg)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load once here (not per make_cache() call) so every request's AllCache
    # shares the same projection tensors instead of re-reading the .pt file.
    offline_projs = load_offline_projections(offline_proj_path) if use_offline_proj else None

    def make_cache():
        return AllCache(mask_out, min_remain, token_budget, err_ratio, use_cuda_kernel, offline_projs)

    return model, tokenizer, make_cache


def _load_bitdecoding_model(
    model_name: str,
    *,
    num_bits: int = 4,
    group_size: int = 128,
    quant_mode: str = "k-channel",
    residual_block_size: Optional[int] = None,
    allow_unvalidated: bool = False,
    device: str = "cuda",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load the BitDecoding low-bit KV model (src/BitDecoding, branch main-legacy).

    BitDecoding is a KIVI-layout quantizer (K per-channel + V per-token, same
    asymmetric min-max) behind a tensor-core split-KV decode kernel, so its
    contribution is the kernel, not the quantization algorithm. Report it as a
    SPEED baseline; its accuracy point is already covered by KIVI.

    Only ``num_bits=4, group_size=128, quant_mode='k-channel'`` is numerically
    correct in the released code. ``dequantize.h`` derives the number of half2
    slots sharing one scale from the group size::

        kBlockN_pack = num_bits == 4 ? 128 : 256       (kernel_traits.h:83)
        num_params   = kBlockN_pack / group_size       (kernel_traits.h:111)
        pack_num     = 4 / num_params                  (dequantize.h:341, 4-bit)
        pack_num     = 4 / (num_params / 2)            (dequantize.h:270, 2-bit)

    but ``pack_num`` is a function of the bit width alone (16 / num_bits), and
    the hardcoded ``4`` is literally ``16/4`` -- written for 4-bit and never
    generalized. A dequant fragment spans 8 values at 4-bit and 16 at 2-bit, and
    ``group_size >= 32`` always, so a fragment never crosses a group and the
    ``j / pack_num`` term must be identically 0. It only is when
    ``group_size == kBlockN_pack``: 4-bit/g128. Everything else multiplies part
    of the values by another group's scale/zero. Measured against a pure-torch
    fake-quant oracle (src/BitDecoding/evaluation/ref_quant_error.py) at
    seqlen 1024:

        bits group  num_params  pack_num  needed   kernel err  oracle err
           4   128           1         4       4     0.00348     0.00349  <- 1.00x
           4    32           4         1       4     0.00943     0.00268  <- 3.5x
           2   128           2         4       8     0.03220     0.01765  <- 1.82x
           2    32           8         1       8     0.03381     0.01376  <- 2.46x

    The oracle says finer groups should be BETTER; the kernel inverts the
    ordering, which is a kernel bug and not quantization loss. 4-bit/g128 also
    reproduces fp16 token-for-token on an 8232-token needle prompt, while 2-bit
    collapses to ``'!!!!!...'`` where KIVI at the identical config
    (k=v=2, group 32, residual 256) answers correctly. 2-bit has NO valid group
    size: ``pack_num >= 8`` needs ``group_size >= 256``, which is never
    instantiated and is impossible for V (head_dim is 128).

    ``allow_unvalidated=True`` unlocks the other combinations. The intended use
    is EFFICIENCY measurement, which is unaffected: ``pack_num`` only appears in
    a scale SUBSCRIPT, so loop trip counts, tile sizes, smem usage and memory
    traffic are identical to a corrected kernel, and the pack/params tensor
    shapes come from the Python side (``16 / num_bits``, ``seqlen / group_size``),
    so the compression ratio is real. Never report those rows as accuracy.

    ``quant_mode`` is hard-restricted to ``k-channel``: on main-legacy the
    k-token / k-tensor dispatch branches are entirely commented out and the
    kernel SILENTLY RETURNS the zero-initialized output tensors -- a config that
    does not error and looks impossibly fast.
    """
    if quant_mode != "k-channel":
        raise ValueError(
            f"BitDecoding quant_mode must be 'k-channel'; got {quant_mode!r}. The "
            "k-token / k-tensor dispatch branches are commented out on main-legacy "
            "and the kernel silently returns zero-initialized outputs."
        )
    if (num_bits, group_size) != (4, 128) and not allow_unvalidated:
        raise ValueError(
            f"BitDecoding num_bits={num_bits}, group_size={group_size} is NOT "
            "numerically correct in the released code (pack_num is derived from "
            "group_size; only group_size == kBlockN_pack == 128 at 4-bit works). "
            "Use num_bits=4 group_size=128 for accuracy, or pass "
            "allow_unvalidated=True for efficiency-only measurement."
        )
    if num_bits not in (2, 4):
        raise ValueError(f"BitDecoding supports num_bits 2 or 4; got {num_bits}.")

    for p in (str(BITDECODING_SRC), str(BITDECODING_SRC / "evaluation")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from bit_decode import Cache as BDCache, DynamicCache as BDDynamicCache, StaticCache as BDStaticCache

    # BitDecoding's attention calls update_pack / update_residual / clear_residual,
    # which only exist on bit_decode.DynamicCache, and bit_decode.Cache is a
    # vendored copy rather than a transformers.Cache subclass. `generation/utils.py`
    # did `from ..cache_utils import Cache, DynamicCache` at import time, so
    # patching transformers.cache_utils alone is too late for the names generate()
    # actually resolves -- rebind both (same reasoning as _load_oscar_model).
    import transformers.cache_utils as _tf_cache_utils
    import transformers.generation.utils as _tf_gen_utils

    _tf_cache_utils.Cache = BDCache
    _tf_cache_utils.DynamicCache = BDDynamicCache
    _tf_cache_utils.StaticCache = BDStaticCache
    _tf_gen_utils.Cache = BDCache
    _tf_gen_utils.DynamicCache = BDDynamicCache

    config = load_model_config(model_name)
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3":
        from qwen3 import Qwen3ForCausalLM as BDModel
    elif model_type == "llama":
        from llama import LlamaForCausalLM as BDModel
    else:
        raise ValueError(f"BitDecoding supports model_type qwen3/llama; got {model_type!r}.")

    # The fp16 residual block buffer accumulates new tokens and is repacked every
    # residual_block_size steps; the kernel's own layout constant is
    # kBlockN_pack = 128 (4-bit) / 256 (2-bit), so it must match or the repack
    # writes a differently-shaped tile than the dequant reads.
    if residual_block_size is None:
        residual_block_size = 128 if num_bits == 4 else 256

    config._attn_implementation = "flash_attention_2"
    config.attn_backend = "bit_decoding"
    config.num_bits = num_bits
    config.quant_mode = quant_mode
    config.group_size = group_size
    config.residual_block_size = residual_block_size

    dev = device if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # BitDecoding's kernels are fp16-only (same constraint as KIVI); bf16 is not
    # instantiated. Do not "unify" this with the bf16 picokv/dense paths.
    model = BDModel.from_pretrained(
        model_name, config=config, device_map={"": dev}, torch_dtype=torch.float16,
    )
    model.eval()

    # evaluation/llama.py's forward still takes the pre-4.50 `num_logits_to_keep`
    # (evaluation/qwen3.py already uses the current `logits_to_keep`), so the
    # efficiency runner's logits_to_keep=1 would land in **kwargs and be dropped,
    # materializing full-sequence fp32 logits at prefill (~65GB at 128K tokens).
    # Bridge it here instead of editing the vendored file so the fix survives a
    # re-vendor; functools.wraps keeps the original signature visible to
    # prepare_inputs_for_generation, which introspects it to decide whether to
    # pass position_ids.
    fwd_params = inspect.signature(type(model).forward).parameters
    if "logits_to_keep" not in fwd_params and "num_logits_to_keep" in fwd_params:
        _bd_forward = model.forward

        @functools.wraps(_bd_forward)
        def _forward_with_logits_to_keep(*args, logits_to_keep=None, **kwargs):
            if logits_to_keep is not None and "num_logits_to_keep" not in kwargs:
                kwargs["num_logits_to_keep"] = logits_to_keep
            return _bd_forward(*args, **kwargs)

        model.forward = _forward_with_logits_to_keep

    def make_cache():
        return BDDynamicCache()

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
    method: str = "picokv",
    *,
    k_bits: Optional[int] = None,
    v_bits: Optional[int] = None,
    sink_size: int = 64,
    local_size: int = 256,
    exp_bits: int = 4,
    eas: float = 0.01,
    zerorope: int = 20,
    use_pruning: bool = False,
    prune_mode: int = 3,
    log_eps: float = -10.0,
    quant_target: str = "both",
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
    bpc_mask_out: float = 0.98,
    bpc_min_remain: int = 128,
    bpc_token_budget: Optional[int] = None,
    bpc_err_ratio: float = 0.1,
    bpc_fix_layers: Optional[list[int]] = None,
    bpc_use_cuda_kernel: bool = True,
    bpc_offline_proj_path: Optional[str] = BPC_DEFAULT_OFFLINE_PROJ_PATH,
    bitdecoding_residual_block_size: Optional[int] = None,
    bitdecoding_allow_unvalidated: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, CacheFactory]:
    """Load a HF model+tokenizer and return ``(model, tokenizer, make_cache)``.

    Method dispatch (each backend has its own ``_load_*`` helper):
      * ``picokv``           -> ``_load_picokv_model`` (picokv_standalone
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
      * ``bpc``               -> ``_load_bpc_model`` (src/BPC); hash-based sparse
                                 attention, ``bpc_mask_out`` / ``bpc_min_remain`` /
                                 ``bpc_token_budget`` / ``bpc_err_ratio`` /
                                 ``bpc_fix_layers`` / ``bpc_use_cuda_kernel`` configure
                                 it (defaults match config/llama3-1-8b-ins/extra_configs/
                                 llama3-1-8b-ins-bpc-offline.json). ``bpc_offline_proj_path``
                                 defaults to the learned offline hash projection
                                 (``BPC_DEFAULT_OFFLINE_PROJ_PATH``); pass ``None`` to fall
                                 back to BPC's online (random) projection.
                                 ``make_cache`` returns a fresh ``AllCache`` per call.
      * ``bitdecoding``       -> ``_load_bitdecoding_model`` (src/BitDecoding);
                                 ``num_bits`` / ``group_size`` / ``quant_mode`` set the
                                 KV quantization. ONLY ``num_bits=4, group_size=128,
                                 quant_mode='k-channel'`` is numerically correct in the
                                 released kernel; anything else needs
                                 ``bitdecoding_allow_unvalidated=True`` and is
                                 efficiency-only. ``make_cache`` returns a fresh
                                 ``bit_decode.DynamicCache`` per call.
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

    if method in ("picokv", "picokv_recon"):
        return _load_picokv_model(
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
            offline_proj_path=offline_proj_path,
            recon_ablation=(method == "picokv_recon"),
            quant_target=quant_target,
        )

    if method == "polar":
        return _load_polar_model(
            model_name,
            polar_mode=polar_mode,
            rbits=polar_rbits,
            tbits=polar_tbits,
            vbits=polar_vbits,
        )

    if method == "bpc":
        return _load_bpc_model(
            model_name,
            mask_out=bpc_mask_out,
            min_remain=bpc_min_remain,
            token_budget=bpc_token_budget,
            err_ratio=bpc_err_ratio,
            fix_layers=bpc_fix_layers,
            use_cuda_kernel=bpc_use_cuda_kernel,
            offline_proj_path=bpc_offline_proj_path,
        )

    if method == "bitdecoding":
        return _load_bitdecoding_model(
            model_name,
            num_bits=num_bits,
            group_size=group_size,
            quant_mode=quant_mode,
            residual_block_size=bitdecoding_residual_block_size,
            allow_unvalidated=bitdecoding_allow_unvalidated,
        )

    if method not in BASELINE_ATTN:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'picokv', 'picokv_recon', 'kivi', 'oscar', 'polar', 'bpc', 'bitdecoding', {sorted(BASELINE_ATTN)}."
        )
    return _load_baseline_model(model_name, attn_implementation=method)
