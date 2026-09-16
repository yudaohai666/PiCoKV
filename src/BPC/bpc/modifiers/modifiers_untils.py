from ..modifier import Modifier
import json
import numpy as np


def error_aware_layer_budget_allocation(layer_ratio, total_budget, min_tokens=32, max_tokens=1536):
    """Allocate per-layer KV budget by importance score (see pred_compresskv.py)."""
    importance = np.array(layer_ratio)
    base = np.full_like(importance, min_tokens, dtype=float)
    remaining_budget = total_budget - base.sum()
    extra = np.round(importance * remaining_budget)
    budget_per_layer = base + extra
    budget_per_layer = np.clip(budget_per_layer, min_tokens, max_tokens)
    diff = int(total_budget - budget_per_layer.sum())

    while diff != 0:
        if diff > 0:
            candidates = np.where(budget_per_layer < max_tokens)[0]
            if len(candidates) == 0:
                break
            idx_order = candidates[np.argsort(-importance[candidates])]
            for i in idx_order:
                if budget_per_layer[i] < max_tokens:
                    budget_per_layer[i] += 1
                    diff -= 1
                    if diff == 0:
                        break
        else:
            candidates = np.where(budget_per_layer > min_tokens)[0]
            if len(candidates) == 0:
                break
            idx_order = candidates[np.argsort(importance[candidates])]
            for i in idx_order:
                if budget_per_layer[i] > min_tokens:
                    budget_per_layer[i] -= 1
                    diff += 1
                    if diff == 0:
                        break
    return budget_per_layer.astype(int).tolist()


class Origin(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)

    def ft_params(self):
        return []
    
    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Sparse(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        from ..svdsparse.llama_hijack import modify
        modify(self.model)
        print("Sparse attention modification applied to model")

    def ft_params(self):
        return []

    def reset(self):
        # Clear cache between inferences.
        from ..svdsparse.llama_hijack import reset_cache
        reset_cache()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class SparseTopk(Modifier):
    """SparseTopk sparse attention. Supports LLaMA / Mistral / Qwen2 (auto-detected)."""
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)

        sparse_config = {}
        if self.conf is not None:
            sparse_config = {
                "enable": self.conf.get("enable", True),
                "fix_layers": self.conf.get("fix_layers", []),
                "mask_out": self.conf.get("mask_out", 0.90),
                "min_remain": self.conf.get("min_remain", 20),
                "local_window_size": self.conf.get("local_window_size", None),
                "token_budget": self.conf.get("token_budget", None),
                "use_offline_proj": self.conf.get("use_offline_proj", False),
                "offline_proj_path": self.conf.get("offline_proj_path", None),
            }

        from ..svdsparse.sparse_topk import modify
        modify(self.model, config=sparse_config)

    def ft_params(self):
        return []

    def reset(self):
        # Clear cache between inferences.
        from ..svdsparse.sparse_topk import reset_cache
        reset_cache()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class SnapKV(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)
       
        from ..snapkv.monkeypatch import replace_llama, replace_mistral, replace_qwen2
       
        if model_structure == 'llama':
            replace_llama()
            print("Applied SnapKV monkeypatch for LLaMA")
        elif model_structure == 'mistral':
            replace_mistral()
            print("Applied SnapKV monkeypatch for Mistral")
        elif model_structure == 'qwen2':
            replace_qwen2()
            print("Applied SnapKV monkeypatch for Qwen2")
        else:
            # Unknown structure: apply all patches (compat fallback).
            replace_llama()
            replace_mistral()
            replace_qwen2()
            print("Applied SnapKV monkeypatch for all architectures (fallback)")
       
        if hasattr(model, 'model'):
            model.model.config.window_size = self.conf.get('window_size', 8)
            model.model.config.kernel_size = self.conf.get('kernel_size', 5)
            model.model.config.max_capacity_prompt = self.conf.get('max_capacity_prompt', 512)
            model.model.config.pooling = self.conf.get('pooling', 'avgpool')
            model.model.config.fix_layers = self.conf.get('fix_layers', [])
       
        print("SnapKV modification applied to model")

    def ft_params(self):
        return []

    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PyramidKV(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)
       
        from ..pyramidkv.monkeypatch import replace_llama, replace_mistral, replace_qwen2
       
        if model_structure == 'llama':
            replace_llama()
            print("Applied PyramidKV monkeypatch for LLaMA")
        elif model_structure == 'mistral':
            replace_mistral()
            print("Applied PyramidKV monkeypatch for Mistral")
        elif model_structure == 'qwen2':
            replace_qwen2()
            print("Applied PyramidKV monkeypatch for Qwen2")
        else:
            # Unknown structure: apply all patches (compat fallback).
            replace_llama()
            replace_mistral()
            replace_qwen2()
            print("Applied PyramidKV monkeypatch for all architectures (fallback)")
       
        if hasattr(model, 'model'):
            model.model.config.window_size = self.conf.get('window_size', 8)
            model.model.config.kernel_size = self.conf.get('kernel_size', 5)
            model.model.config.max_capacity_prompt = self.conf.get('max_capacity_prompt', 512)
            model.model.config.pooling = self.conf.get('pooling', 'avgpool')
            model.model.config.pyram_beta = self.conf.get('pyram_beta', 20)
            model.model.config.fix_layers = self.conf.get('fix_layers', [])

    def ft_params(self):
        return []

    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class CakeKV(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)
       
        from ..cakekv.monkeypatch import replace_flashllama_attn_with_cakeattn, replace_flashmistral_attn_with_cakeattn, replace_flashqwen2_attn_with_cakeattn
       
        if model_structure == 'llama':
            replace_flashllama_attn_with_cakeattn()
            print("Applied CakeKV monkeypatch for LLaMA")
        elif model_structure == 'mistral':
            replace_flashmistral_attn_with_cakeattn()
            print("Applied CakeKV monkeypatch for Mistral")
        elif model_structure == 'qwen2':
            replace_flashqwen2_attn_with_cakeattn()
            print("Applied CakeKV monkeypatch for Qwen2")
        else:
            # Unknown structure: apply all patches (compat fallback).
            replace_flashllama_attn_with_cakeattn()
            replace_flashmistral_attn_with_cakeattn()
            replace_flashqwen2_attn_with_cakeattn()
            print("Applied CakeKV monkeypatch for all architectures (fallback)")
       
        from ..cakekv.cake_cache import CakeprefillKVCache
        if hasattr(model, 'model'):
            layers = model.model.config.num_hidden_layers
            window_size = self.conf.get('window_size', 8)
            max_capacity_prompt = self.conf.get('max_capacity_prompt', 512)
            fix_layers = self.conf.get('fix_layers', [])
           
            for i in range(layers):
                model.model.layers[i].self_attn.config.key_size = [max_capacity_prompt - window_size] * layers
                model.model.layers[i].self_attn.config.window_size = [window_size] * layers
                model.model.layers[i].self_attn.config.prefill = [True] * layers
                model.model.layers[i].self_attn.config.decoding_evict = [None] * layers
                model.model.layers[i].self_attn.config.tau1 = self.conf.get('tau1', 1.0)
                model.model.layers[i].self_attn.config.tau2 = self.conf.get('tau2', 1.0)
                model.model.layers[i].self_attn.config.gamma = self.conf.get('gamma', 200.0)
                model.model.layers[i].self_attn.config.fix_layers = fix_layers
                model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                    cache_size=max_capacity_prompt,
                    window_size=window_size,
                    k_seq_dim=2,
                    v_seq_dim=2,
                    num_heads=model.model.layers[i].self_attn.num_heads,
                    num_layers=layers,
                    use_cascading=True,
                    fix_layers=fix_layers
                )] * layers

    def ft_params(self):
        return []

    def reset(self):
        # Reset CakeKV prefill state.
        if hasattr(self.model, 'model'):
            layers = len(self.model.model.layers)
            for i in range(layers):
                self.model.model.layers[i].self_attn.config.prefill = [True] * layers
                self.model.model.layers[i].self_attn.config.decoding_evict = [None] * layers

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class CompressKV(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)
       
        from ..compresskv.monkeypatch import replace_llama, replace_mistral, replace_qwen2
       
        if model_structure == 'llama':
            replace_llama()
            print("Applied CompressKV monkeypatch for LLaMA")
        elif model_structure == 'mistral':
            replace_mistral()
            print("Applied CompressKV monkeypatch for Mistral")
        elif model_structure == 'qwen2':
            replace_qwen2()
            print("Applied CompressKV monkeypatch for Qwen2")
        else:
            # Unknown structure: apply all patches (compat fallback).
            replace_llama()
            replace_mistral()
            replace_qwen2()
            print("Applied CompressKV monkeypatch for all architectures (fallback)")
       
        if hasattr(model, 'model'):
            layers = model.model.config.num_hidden_layers
            model.model.config.window_size = self.conf.get('window_size', 8)
            model.model.config.kernel_size = self.conf.get('kernel_size', 5)
            model.model.config.max_capacity_prompt = self.conf.get('max_capacity_prompt', 512)
            model.model.config.pooling = self.conf.get('pooling', 'avgpool')
            model.model.config.fix_layers = self.conf.get('fix_layers', [])
           
            # Per-layer budget allocation by importance score.
            layer_importance_score_path = self.conf.get('layer_importance_score_path')
            if layer_importance_score_path is not None:
                with open(layer_importance_score_path, "r") as f:
                    layer_score = json.load(f)["avg_score"]
                max_capacity_prompt_layer_adaptive = error_aware_layer_budget_allocation(
                    layer_score, 
                    self.conf.get('max_capacity_prompt', 512) * layers, 
                    32, 
                    self.conf.get('max_capacity_prompt', 512) * 3
                )
            else:
                max_capacity_prompt_layer_adaptive = [self.conf.get('max_capacity_prompt', 512)] * layers
           
            # Important-head selection.
            importance_head_path = self.conf.get('importance_head_path')
            if importance_head_path is not None:
                with open(importance_head_path, 'r') as f:
                    important_head = json.load(f)
                important_head = [important_head[str(i)] for i in range(len(important_head))]
                model.model.config.important_heads = important_head
           
            model.model.config.first_k = self.conf.get('first_k', 4)
            model.model.config.max_capacity_prompt_layer_adaptive = max_capacity_prompt_layer_adaptive

    def ft_params(self):
        return []

    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class StreamingLLM(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)
       
        from ..streamingllm.monkeypatch import replace_llama, replace_mistral, replace_qwen2
       
        if model_structure == 'llama':
            replace_llama()
            print("Applied StreamingLLM monkeypatch for LLaMA")
        elif model_structure == 'mistral':
            replace_mistral()
            print("Applied StreamingLLM monkeypatch for Mistral")
        elif model_structure == 'qwen2':
            replace_qwen2()
            print("Applied StreamingLLM monkeypatch for Qwen2")
        else:
            # Unknown structure: apply all patches (compat fallback).
            replace_llama()
            replace_mistral()
            replace_qwen2()
            print("Applied StreamingLLM monkeypatch for all architectures (fallback)")
       
        if hasattr(model, 'model'):
            model.model.config.window_size = self.conf.get('window_size', 8)
            model.model.config.kernel_size = self.conf.get('kernel_size', 5)
            model.model.config.max_capacity_prompt = self.conf.get('max_capacity_prompt', 512)
            model.model.config.pooling = self.conf.get('pooling', 'avgpool')
            model.model.config.fix_layers = self.conf.get('fix_layers', [])

    def ft_params(self):
        return []

    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def Spotlight(model, save_ckp, load_ckp, config, model_structure=None):
    """Spotlight sparse attention. Supports LLaMA and Qwen2."""
    if model_structure == 'qwen':
        from ..spotlight.spotlight_generation_qwen2 import SpotlightQwen2
        return SpotlightQwen2(model, save_ckp, load_ckp, config, model_structure)
    else:
        from ..spotlight.spotlight_generation import Spotlight as SpotlightLLaMA
        return SpotlightLLaMA(model, save_ckp, load_ckp, config, model_structure)


class Quest(Modifier):
    """Quest sparse attention. Supports LLaMA / Mistral / Qwen2 (auto-detected)."""
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)

        quest_config = {}
        if self.conf is not None:
            quest_config = {
                "token_budget": self.conf.get("token_budget", 1024),
                "chunk_size": self.conf.get("chunk_size", 16),
                "fix_layers": self.conf.get("fix_layers", []),
            }

        from ..quest.quest_attention import modify
        modify(self.model, config=quest_config)

    def ft_params(self):
        return []

    def reset(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class RocketKV(Modifier):
    """RocketKV two-stage sparse attention (LLaMA / Mistral / Qwen2, auto-detected).

    1. Prefill: SnapKV-style KV-cache compression (permanently drops some KV).
    2. Decode: sparse attention over remaining KV via Q-dim reduction + chunked K approx.

    Reference: https://arxiv.org/abs/2501.09552
    """
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)

        rocketkv_config = {}
        if self.conf is not None:
            rocketkv_config = {
                "token_budget": self.conf.get("token_budget", 1024),
                "window_size": self.conf.get("window_size", 32),
                "kernel_size": self.conf.get("kernel_size", 63),
                "fix_layers": self.conf.get("fix_layers", []),
                "max_new_tokens": self.conf.get("max_new_tokens", 256),
            }

        from ..rocketkv import modify
        modify(self.model, config=rocketkv_config)

    def ft_params(self):
        return []

    def reset(self):
        from ..rocketkv import reset_cache
        reset_cache()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MagicPIG(Modifier):
    """MagicPIG LSH-based sparse attention (LLaMA / Mistral / Qwen2).

    Simplified pure-PyTorch implementation (no C++ extension):
    - matches original accuracy
    - uses brute-force LSH compare instead of hash-bucket lookup

    Pipeline: hash via LSH; split KV into on-device selected + off-device LSH-indexed;
    after prefill keep important tokens, then at decode match by hash codes and apply
    probability correction for the sparse-attention bias.

    Reference: MagicPIG (https://github.com/Infini-AI-Lab/MagicPIG)
    """
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)

        # Defaults match upstream MagicPIG (3rd/MagicPIG/models/magicpig_config.json).
        magicpig_config = {}
        if self.conf is not None:
            magicpig_config = {
                "device_budget": self.conf.get("device_budget", 68),
                "window_size": self.conf.get("window_size", 64),
                "fix_layers": self.conf.get("fix_layers", [0, 16]),     # upstream cache_layers
                "K": self.conf.get("K", 10),                            # lsh_K
                "L": self.conf.get("L", 150),                           # lsh_L
                "mode": self.conf.get("mode", "anns_es"),
                "threshold": self.conf.get("threshold", 2),             # LSH collision threshold
            }

        from ..magicpig.magicpig_attention import modify
        modify(self.model, config=magicpig_config)

    def ft_params(self):
        return []

    def reset(self):
        from ..magicpig.magicpig_attention import reset_cache
        reset_cache()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class BPC(Modifier):
    """BPC: binary-hash sparse attention (LLaMA / Mistral / Qwen2).

    Learns a projection that compresses keys to 64-bit hashes (LSH-like), then
    estimates Q-K similarity from hashes. Prefill caches KV + hashes and records
    high-error tokens; decode picks top-k by hash score, force-selecting err_topk
    to bound precision loss.
    """
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)

        bpc_config = {}
        if self.conf is not None:
            bpc_config = {
                "enable": self.conf.get("enable", True),
                "fix_layers": self.conf.get("fix_layers", []),
                "mask_out": self.conf.get("mask_out", 0.98),
                "min_remain": self.conf.get("min_remain", 128),
                "token_budget": self.conf.get("token_budget", None),  # if set, overrides mask_out
                "err_ratio": self.conf.get("err_ratio", 0.1),
                "use_offline_proj": self.conf.get("use_offline_proj", False),
                "offline_proj_path": self.conf.get("offline_proj_path", None),
                "use_cuda_kernel": self.conf.get("use_cuda_kernel", False),
            }

        from ..bpc.bpc import modify
        modify(self.model, config=bpc_config)

    def ft_params(self):
        return []

    def reset(self):
        from ..bpc.bpc import reset_cache
        reset_cache(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)



class TopK(Modifier):
    """TopK KV cache: full prefill, then at each decode step keep only the K key-value pairs
    with the highest attention score (default K=2). Non-topk positions are masked out."""
    def __init__(self, model, save_ckp, load_ckp, config, model_structure=None):
        super().__init__(model, save_ckp, load_ckp)
        self.get_conf(config)

        topk_config = {}
        if self.conf is not None:
            topk_config = {
                "topk": self.conf.get("topk", 2),
                "fix_layers": self.conf.get("fix_layers", []),
            }

        from ..topk.topk_attention import modify
        modify(self.model, config=topk_config)

    def ft_params(self):
        return []

    def reset(self):
        from ..topk.topk_attention import reset_cache
        reset_cache()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

