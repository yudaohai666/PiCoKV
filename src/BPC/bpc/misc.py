import math
from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from functools import partial

from torch.utils.data import DataLoader
import json

from .modifiers import get_modifier



def get_torch_dtype(dtype: str):
    if dtype == 'fp16':
        return torch.float16
    elif dtype == 'fp32':
        return torch.float32
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype '{dtype}'")


def get_env_conf(env_conf: str):
    import json
    with open(env_conf, 'r') as f:
        env_conf = json.load(f)
    return env_conf


def get_tokenizer(
        model_name, 
        **kwargs
):
    if "tokenizer_name" in kwargs:
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs.get('tokenizer_name'), 
            use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True)

    return tokenizer


def get_model_and_tokenizer(
        model_name, 
        model_dtype, 
        model_method, 
        model_structure, 
        save_ckp, 
        load_ckp, 
        config, 
        device_map, 
        **kwargs
    ):

    from accelerate import dispatch_model
    token = "hf_xxxxxxx"

    if "tokenizer_name" in kwargs:
        tokenizer = AutoTokenizer.from_pretrained(kwargs.get('tokenizer_name'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    student_dtype = get_torch_dtype(model_dtype)
    
    # Methods whose prefill / fix_layers paths fall back to the original forward
    # and need flash_attention_2 to stay fast (e.g. clusterkv/rocketkv).
    flash_attn_methods = ['snapkv', 'pyramidkv', 'streamingllm', 'compresskv', 'cakekv',  'quest', 'origin', 'magicpig',"bpc", "topk"]
    attn_impl = 'flash_attention_2' if model_method in flash_attn_methods else None
    
    if attn_impl == 'flash_attention_2':

        student = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=student_dtype, 
            token=token, 
            device_map="auto" if device_map is None else None,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            use_cache=True
            )
    else:
        student = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=student_dtype, 
            token=token, 
            device_map="auto" if device_map is None else None,
            trust_remote_code=True,
            use_cache=True
            )

    model_structure_param, student_modifier = get_modifier(model_method, model_structure)

    if student_modifier is not None:
        student = student_modifier(
            student,
            save_ckp=save_ckp,
            load_ckp=load_ckp,
            config=config,
            model_structure=model_structure_param)

    student.eval()

    if device_map is not None:
        student.model = dispatch_model(student.model, device_map=device_map)

    return tokenizer, student


def get_optimizer_and_lr_adjuster(model, max_lr, train_iters, warmup, weight_decay, beta1, beta2, **kwargs):
    ft_params = model.ft_params() if 'extra_params' not in kwargs else model.ft_params() + kwargs['extra_params']
    optim = torch.optim.AdamW(ft_params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster