"""Shared model-config adjustments for evaluation and projection building."""

from transformers import AutoConfig


def load_model_config(model_name: str, **kwargs):
    """Load a config and apply the Qwen3-8B 128k YaRN settings."""
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    if getattr(config, "model_type", None) == "qwen3":
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
        config.max_position_embeddings = 131072
    return config
