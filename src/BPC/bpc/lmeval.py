from lm_eval.api.model import TemplateLM
from lm_eval import evaluator, utils
from lm_eval.models.utils_hf import stop_sequences_criteria

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, List
from datetime import timedelta

# Optional accelerate support for multi-GPU
try:
    from accelerate import Accelerator, InitProcessGroupKwargs
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def middle_truncate(tokens: List[int], max_length: int) -> List[int]:
    """
    Truncate tokens by keeping left half and right half, removing middle.
    
    Args:
        tokens: List of token ids
        max_length: Maximum allowed length
    
    Returns:
        Truncated token list
    """
    if len(tokens) <= max_length:
        return tokens
    
    half = max_length // 2
    # Keep first half and last half
    return tokens[:half] + tokens[-(max_length - half):]


class MyModel(TemplateLM):
    def __init__(self, model, tokenizer, model_max_length, batch_size=1, 
                 add_bos_token=None, prefix_token_id=None, max_gen_toks=128,
                 data_parallel=False):
        """
        Args:
            model: The model to wrap
            tokenizer: The tokenizer
            model_max_length: Maximum sequence length
            batch_size: Batch size for inference
            add_bos_token: Whether to add BOS token
            prefix_token_id: Custom prefix token ID
            max_gen_toks: Maximum tokens to generate
            data_parallel: Enable accelerate data parallelism (multi-GPU)
        """
        super().__init__()
        self.model = model
        self._tokenizer = tokenizer
        self.model_max_length = model_max_length
        self._batch_size = batch_size
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id
        self._max_gen_toks = max_gen_toks
        
        # Initialize distributed settings
        self._rank = 0
        self._world_size = 1
        self.accelerator = None
        
        # Setup accelerate for data parallelism if requested
        if data_parallel and ACCELERATE_AVAILABLE:
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            # Only show progress bar on main process
            if self._rank != 0:
                import logging
                logging.getLogger("lm_eval").setLevel(logging.WARNING)
        
        # Configure pad token if needed
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        """Used as prefix for loglikelihood computation."""
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self._tokenizer.bos_token_id is not None:
            return self._tokenizer.bos_token_id
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model_max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def rank(self):
        """Return process rank for distributed evaluation."""
        return self._rank

    @property
    def world_size(self):
        """Return world size for distributed evaluation."""
        return self._world_size

    @property
    def tokenizer_name(self):
        """Required for chat template support."""
        return self._tokenizer.name_or_path if hasattr(self._tokenizer, 'name_or_path') else "custom_tokenizer"

    def apply_chat_template(self, chat_history: List[dict], add_generation_prompt=True) -> str:
        """
        Apply chat template to convert chat history to a string.
        Based on HFLM implementation.
        """
        # Try to use tokenizer's chat template
        if hasattr(self._tokenizer, 'apply_chat_template') and self._tokenizer.chat_template is not None:
            try:
                return self._tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=not add_generation_prompt,  # must match HFLM behavior
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
        
        # Fallback: Simple concatenation for base models
        result = ""
        for message in chat_history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                result += f"User: {content}\n"
            elif role == "assistant":
                result += f"Assistant: {content}\n"
            else:
                result += content
        
        if add_generation_prompt and not result.endswith("Assistant: "):
            result += "Assistant: "
            
        return result

    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None, 
                   left_truncate_len: Optional[int] = None, **kwargs) -> List[int]:
        """
        Encode string to tokens with proper special token handling.
        Based on HFLM implementation.
        """
        # Handle special tokens based on model type and BOS token presence
        if add_special_tokens is None:
            # Check if string already starts with BOS token
            bos_token = getattr(self._tokenizer, 'bos_token', None)
            if bos_token and string.startswith(bos_token):
                add_special_tokens = False
            elif self.add_bos_token is not None:
                add_special_tokens = self.add_bos_token
            else:
                add_special_tokens = False
        
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens, **kwargs)
        
        # Left-truncate if specified
        if left_truncate_len and len(encoding) > left_truncate_len:
            encoding = encoding[-left_truncate_len:]
            
        return encoding

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Model forward call with optional attention mask.
        For lm_eval, we need full sequence logits, so pass return_full_logits=True for custom models.
        For original HuggingFace models (like LlamaForCausalLM), don't pass this argument.
        """
        with torch.no_grad():
            # Check if model supports return_full_logits (custom sparse models)
            # Original HF models don't support this argument
            supports_full_logits = hasattr(self.model, 'return_full_logits') or \
                                   hasattr(self.model, 'supports_full_logits') or \
                                   (hasattr(self.model, 'model') and hasattr(self.model.model, 'return_full_logits'))
            
            # Try to detect if it's a wrapped model that supports return_full_logits
            # by checking the forward signature
            if not supports_full_logits:
                import inspect
                try:
                    # Get the actual forward method
                    forward_method = self.model.forward
                    sig = inspect.signature(forward_method)
                    supports_full_logits = 'return_full_logits' in sig.parameters
                except:
                    supports_full_logits = False
            
            # Also check by trying to call with return_full_logits and catching TypeError
            # This handles cases where the signature check fails but the model supports it
            if not supports_full_logits:
                # Check if model class name indicates it's a custom model (Spotlight, etc.)
                model_class_name = self.model.__class__.__name__
                if model_class_name in ['Spotlight', 'Model', 'Decoder']:
                    supports_full_logits = True
            
            if supports_full_logits:
                if attn_mask is not None:
                    output = self.model(inps, attention_mask=attn_mask, return_full_logits=True)
                else:
                    output = self.model(inps, return_full_logits=True)
            else:
                # Original HF model - don't pass return_full_logits
                if attn_mask is not None:
                    output = self.model(inps, attention_mask=attn_mask)
                else:
                    output = self.model(inps)
            
            # Handle different output types
            if hasattr(output, 'logits'):
                return output.logits
            else:
                return output

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        """
        Loglikelihood computation for batch_size=1.
        Based on HFLM implementation with LEFT truncation (same as original lm_eval).
        """
        res = []

        def _collate(req):
            """Sort by total length (descending) for better batching."""
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        for cache_key, context_enc, continuation_enc in tqdm(
            re_ord.get_reordered(), disable=disable_tqdm, desc="Running loglikelihood"
        ):
            # Sanity checks
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            # how this all works (illustrated on a causal decoder-only setup):
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:-1]
            # model  \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :] slice

            # when too long to fit in context, truncate from the left (same as HFLM)
            inp = torch.tensor(
                (list(context_enc) + list(continuation_enc))[-(self.max_length + 1):][:-1],
                dtype=torch.long,
                device=self.device,
            )
            inplen = inp.shape[0]
            
            # Forward pass
            logits = F.log_softmax(self._model_call(inp.unsqueeze(0)), dim=-1)  # [1, seq_len, vocab]
            
            # Get continuation length
            contlen = len(continuation_enc)
            
            # Take only logits for continuation tokens (from the right side)
            # logits shape: [1, inplen, vocab]
            # We want the last contlen positions
            cont_logits = logits[0, -contlen:, :]  # [contlen, vocab]
            
            # Get actual continuation tokens
            cont_toks_tensor = torch.tensor(
                list(continuation_enc), dtype=torch.long, device=cont_logits.device
            )
            
            # Check if greedy decoding matches
            greedy_tokens = cont_logits.argmax(dim=-1)
            max_equal = (greedy_tokens == cont_toks_tensor).all()
            
            # Get log probabilities for actual tokens
            cont_logprobs = torch.gather(
                cont_logits, 1, cont_toks_tensor.unsqueeze(-1)
            ).squeeze(-1)
            
            answer = (float(cont_logprobs.sum().cpu()), bool(max_equal))

            # Cache result
            if cache_key is not None:
                self.cache_hook.add_partial("loglikelihood", cache_key, answer)

            res.append(answer)

        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """
        Rolling loglikelihood computation based on HFLM implementation.
        """
        loglikelihoods = []
        
        for req in tqdm(requests, disable=disable_tqdm, desc="Computing rolling loglikelihood"):
            string = req.args[0]
            
            # Get rolling windows
            # HFLM uses max_seq_len=self.max_length (not max_length - 1)
            # because _loglikelihood_tokens handles the [:-1] internally
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            
            # Format windows for _loglikelihood_tokens
            windows = [(None,) + x for x in rolling_token_windows]
            
            # Compute loglikelihoods for all windows
            string_nll = self._loglikelihood_tokens(windows, disable_tqdm=True)
            
            # Sum up the log likelihoods (only the logprob part, ignore is_greedy)
            total_nll = sum(nll[0] for nll in string_nll)
            loglikelihoods.append(total_nll)
            
            # Cache the result
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_nll)
        
        return loglikelihoods

    def generate_until(self, requests, disable_tqdm=False):
        """
        Text generation based on HFLM implementation.
        Uses middle truncation: keep left half + right half, remove middle.
        """
        res = []
        
        for req in tqdm(requests, disable=disable_tqdm, desc="Generating text"):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            
            # Encode context
            context_enc = self.tok_encode(context)
            
            # Apply middle truncation if needed (max_length - max_gen_toks for context)
            max_ctx_len = self.max_length - max_gen_toks
            if len(context_enc) > max_ctx_len:
                context_enc = middle_truncate(context_enc, max_ctx_len)
            
            context_tensor = torch.tensor([context_enc], dtype=torch.long, device=self.device)
            
            # Prepare generation kwargs
            generation_kwargs = {
                'max_length': len(context_enc) + max_gen_toks,
                'eos_token_id': self.eot_token_id,
                'pad_token_id': self._tokenizer.pad_token_id or self.eot_token_id,
                'do_sample': gen_kwargs.get("do_sample", False),
            }
            
            # Only add temperature/top_p/top_k if sampling
            if generation_kwargs['do_sample']:
                generation_kwargs['temperature'] = gen_kwargs.get("temperature", 1.0)
                generation_kwargs['top_p'] = gen_kwargs.get("top_p", 1.0)
                generation_kwargs['top_k'] = gen_kwargs.get("top_k", 50)
            
            # Build stopping criteria if until tokens are specified
            if until:
                try:
                    stopping_criteria = stop_sequences_criteria(
                        self._tokenizer, until, context_tensor.shape[1], context_tensor.shape[0]
                    )
                    generation_kwargs['stopping_criteria'] = stopping_criteria
                except Exception:
                    pass  # Fallback: handle stopping in post-processing
            
            # Generate
            with torch.no_grad():
                try:
                    cont = self.model.generate(
                        context_tensor,
                        **generation_kwargs
                    )
                except Exception as e:
                    print(f"Generation failed: {e}")
                    # Fallback to simple generation
                    cont = self.model.generate(
                        context_tensor,
                        max_length=len(context_enc) + max_gen_toks,
                        eos_token_id=self.eot_token_id,
                        pad_token_id=self._tokenizer.pad_token_id or self.eot_token_id,
                    )
            
            # Decode generated text
            generated_tokens = cont[0].tolist()[len(context_enc):]
            s = self.tok_decode(generated_tokens)
            
            # Apply stopping sequences
            for term in until:
                if term and term in s:
                    s = s.split(term)[0]
            
            res.append(s)
            
            # Cache the result
            # Note: gen_kwargs may contain unhashable values (like lists), so we create a hashable key
            try:
                # Try to create a hashable representation of gen_kwargs
                hashable_kwargs = tuple(
                    (k, tuple(v) if isinstance(v, list) else v) 
                    for k, v in sorted(gen_kwargs.items())
                )
                self.cache_hook.add_partial("generate_until", (context, hashable_kwargs), s)
            except (TypeError, AttributeError):
                # If still unhashable, just use context as key
                self.cache_hook.add_partial("generate_until", (context,), s)
            
            # Reset model state for methods like CakeKV that need cache reset
            if hasattr(self.model, 'reset'):
                self.model.reset()
            
        return res
    
