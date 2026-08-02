"""QuanPC integrated package.

Groups the production online-projection standalone path and its two JIT CUDA
extensions under one namespace:

- ``quantpc.quanpc_standalone``          : public prepare/decode API + HF Llama adapter.
- ``quantpc.quanpc_cuda_extension``      : single-token decode CUDA kernels.
- ``quantpc.quanpc_prefill_cuda_extension``: cache-packing / gather CUDA kernels.
"""
