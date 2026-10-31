"""Rounding an fp32 value to a 16-bit float while keeping it in fp32.

Triton folds a plain ``x.to(tl.bfloat16).to(tl.float32)`` (bitcasts included) into ``x``, so a
kernel that must round where the reference rounds goes through PTX ``cvt.rn`` here instead.
"""

import triton
import triton.language as tl


@triton.jit
def round_as(x, DTYPE: tl.constexpr):
    """``x`` (fp32) rounded to ``DTYPE`` (bf16 or fp16, nearest even; fp32: ``x``), as fp32."""
    if tl.float32 == DTYPE:
        return x
    if tl.float16 == DTYPE:
        return tl.inline_asm_elementwise(
            "{ .reg .b16 t; cvt.rn.f16.f32 t, $1; cvt.f32.f16 $0, t; }",
            "=r,r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
