"""Activation functions shared by the Triton producers, in float32."""

import triton
import triton.language as tl


@triton.jit
def gelu_tanh(x):
    """``gelu(x, approximate="tanh")`` = ``x / 2 * (1 + tanh(u))``, ``u = sqrt(2 / pi) * (x +
    0.044715 x^3)``; with ``tanh(u) = 2 sigmoid(2u) - 1`` that is ``x * sigmoid(2u)``."""
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * tl.sigmoid(2.0 * u)
