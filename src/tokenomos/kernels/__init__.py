"""Tokenomos's GPU kernels (Triton): MXFP8 and NVFP4 producers and GEMM routing (``mxfp8``,
``nvfp4``), the any-width varlen short conv (``conv``), the 8-bit GDN state (``gdn``).

Every kernel reads its metadata from device tensors, so each one can sit inside a CUDA graph.
"""

import os

# nvcc refuses the clang-cl TileLang prefers on a PATH that holds one; cl.exe compiles
if os.name == "nt":
    os.environ.setdefault("TILELANG_DISABLE_CLANG_CL", "1")
