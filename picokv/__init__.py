"""PiCoKV integrated package.

Groups the production online-projection standalone path and its two JIT CUDA
extensions under one namespace:

- ``picokv.picokv_standalone``            : public prepare/decode API + HF Llama adapter.
- ``picokv.picokv_cuda_extension``        : single-token decode CUDA kernels.
- ``picokv.picokv_prefill_cuda_extension``: cache-packing / gather CUDA kernels.
"""
