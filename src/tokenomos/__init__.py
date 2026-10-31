"""Tokenomos: local inference for small VLMs whose host work per step does not grow with the
batch, so the GPU is the only bottleneck.

``kernels``: Triton kernels. ``models``: the model framework (blocks, contracts, checkpoint
loading) and one declaration per family. ``core``: the engine. ``api``: ``LLM`` for in-Python
use and the HTTP server. The version lives in ``Cargo.toml`` only (maturin).
"""

from tokenomos.api.llm import LLM, SamplingParams

__all__ = ["LLM", "SamplingParams"]
