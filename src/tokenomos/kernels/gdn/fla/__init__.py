"""The chunked gated delta rule of flash-linear-attention (MIT, ``LICENSE``), as vendored by vLLM
0.32 (``vllm/third_party/flash_linear_attention/ops``), with vLLM's imports replaced and the chunk
table built from host offsets (``index.chunk_indices_host``); vLLM's fused post-conv prep (split,
q / k L2 norm, gates in one kernel)."""

from tokenomos.kernels.gdn.fla.chunk import chunk_gated_delta_rule
from tokenomos.kernels.gdn.fla.fused_gdn_prefill_post_conv import fused_post_conv_prep
from tokenomos.kernels.gdn.fla.utils import FLA_CHUNK_SIZE

__all__ = ["chunk_gated_delta_rule", "fused_post_conv_prep", "FLA_CHUNK_SIZE"]
