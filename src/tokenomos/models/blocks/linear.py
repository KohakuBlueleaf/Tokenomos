"""A linear layer in its checkpoint's format, with an optional bias:

- ``bf16``: ``[N, K]`` in ``floats.weights()`` (bf16 or fp16; ``N <= 64`` runs the skinny kernel);
  under the fp32 work dtype an fp32 activation goes through fp16 with float32 accumulation on the
  layer's ``route``: ``tilelang`` (``kernels.linear.fp16``, the weight padded to whole tiles) or
  ``cublas`` (a 16-bit product, widened);
- ``mxfp8``: ``[N, K]`` E4M3 plus ue8m0 scales per 32 along K, swizzled at load; the activation
  quantized per call (or by a fused producer, ``mx``);
- ``nvfp4``: packed E2M1 ``[N, K / 2]`` plus E4M3 scales per 16 (swizzled at load) and a tensor
  scale; the activation quantized per call with the checkpoint's static scale (``igs``, the
  divisor-style ``1 / imul``); ``alpha = imul * wmul`` scales the product;
- ``fp8``: ``[N, K]`` E4M3 with a scale per tensor or per output row (``wmul``), the activation
  quantized per call with the static ``imul``;
- ``fp8_dynamic``: ``[N, K]`` E4M3 with scales per output row per 128 columns, the activation
  quantized per call with scales of its own (``kernels.linear.fp8``);
- ``int8``: W8A8 made from a bf16 layer at load (``to_int8``): ``[N, K]`` int8 padded to whole
  tiles with a scale per output row (``wmul``), the activation quantized per call with a scale per
  token. ``rotate`` applies a regular Hadamard rotation per group of K columns (ConvRot): the
  weight takes it at load, the activation at each call, so outlier columns spread over the group.

Fused projections (q/k/v, gate/up) are the checkpoint's parts concatenated along N, loaded as one
layer when the parts agree on format and on the tensor scales one GEMM can carry (NVFP4: both;
FP8: the activation scale); otherwise ``Stacked``: one layer per run of agreeing parts, outputs
concatenated.
"""

import torch
import torch.nn.functional as F
from torch import nn

from tokenomos.core.floats import ROUTES, weights, work
from tokenomos.kernels.linear import fp8, skinny
from tokenomos.kernels.mxfp8 import ops as mx
from tokenomos.kernels.nvfp4 import ops as nv

try:
    from tokenomos.kernels.linear import fp16 as f16
    from tokenomos.kernels.linear import int8 as i8
except ImportError:  # W8A8 and the tilelang route need TileLang
    f16 = i8 = None

FLOAT8 = (torch.float8_e4m3fn, torch.float8_e5m2)
# the largest E2M1 value times the largest E4M3 block scale
E2M1_E4M3 = 6.0 * 448.0


def _param(t: torch.Tensor | None):
    return nn.Parameter(t, requires_grad=False) if t is not None else None


def regular_hadamard(g: int, device) -> torch.Tensor:
    """The orthonormal regular Hadamard matrix of order ``g``, a power of 4: every row sums to
    the same value, so a constant vector stays constant up to a scale."""
    if g not in (4, 16, 64, 256):
        raise ValueError(f"rotation group {g}: one of 4, 16, 64, 256")
    base = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=device,
    )
    h = base
    while h.shape[0] < g:
        h = torch.kron(h, base)
    return h / g**0.5


class Linear(nn.Module):
    def __init__(
        self,
        fmt: str,
        weight: torch.Tensor,
        scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        wmul: torch.Tensor | None = None,
        imul: torch.Tensor | None = None,
        k: int | None = None,
    ):
        """``scale``: swizzled block scales; ``wmul`` / ``imul``: tensor scales (NVFP4, FP8);
        ``k``: the input width when the weight is packed."""
        super().__init__()
        if fmt == "bf16":
            weight = weight.to(weights())
        self.fmt = fmt
        self.route = "cublas"
        self.n = weight.shape[0]
        self.k = k or weight.shape[1]
        self.weight = _param(weight)
        self.scale = _param(scale)
        self.bias = _param(bias)
        self.wmul = _param(wmul)
        self.imul = _param(imul)
        self.dynamic = False
        self.rotate = 0
        if fmt == "nvfp4":
            self.igs = _param(1.0 / imul)
            self.alpha = _param(imul * wmul)

    @property
    def is_mx(self) -> bool:
        return self.fmt == "mxfp8"

    @property
    def quantizes(self) -> bool:
        """Whether this layer takes a block-quantized activation a producer can make fused."""
        return self.fmt in ("mxfp8", "nvfp4")

    def act(self, mode: str, a: torch.Tensor, b: torch.Tensor | None = None, group: int = 1):
        """This layer's quantized input made in one pass: ``plain`` ``a``, ``mul`` ``a * b``,
        ``silu_mul`` / ``gelu_mul`` (halves of ``a``), ``gate`` ``a * sigmoid(b[:, col //
        group])``."""
        if self.fmt == "nvfp4":
            return nv.act(a, b, mode, self.igs, group)
        if mode == "plain":
            return mx.quant(a)
        if mode == "mul":
            return mx.mul(a, b)
        if mode == "silu_mul":
            return mx.silu_mul(a)
        if mode == "gelu_mul":
            return mx.gelu_mul(a)
        return mx.gate(a, b, group)

    @classmethod
    def load(cls, ckpt, names: list, device) -> "Linear | Stacked":
        """The linears ``names`` (module names, no ``.weight``) fused along N; a part may be
        ``(name, rows)``: only those output rows (an index tensor) of that linear."""
        parts = []
        for p in names:
            name, rows = p if isinstance(p, tuple) else (p, None)
            part = ckpt.quant.part(ckpt, name, device)
            if rows is not None:
                for key in ("weight", "block"):
                    if key in part:
                        part[key] = part[key][rows.to(part[key].device)]
            part["bias"] = ckpt.get(f"{name}.bias", device) if ckpt.has(f"{name}.bias") else None
            part["name"] = name
            parts.append(part)
        if any(p["bias"] is not None for p in parts) and any(p["bias"] is None for p in parts):
            raise ValueError(f"fused linear {names}: bias on some parts only")
        runs = [[parts[0]]]
        for p in parts[1:]:
            if _agree(runs[-1][0], p):
                runs[-1].append(p)
            else:
                runs.append([p])
        layers = [cls._fuse(run) for run in runs]
        return layers[0] if len(layers) == 1 else Stacked(layers)

    @classmethod
    def _fuse(cls, parts: list[dict]) -> "Linear":
        fmt = parts[0]["fmt"]
        names = [p["name"] for p in parts]
        if fmt != "bf16" and work() != torch.bfloat16:
            raise ValueError(f"linear {names}: {fmt} needs the bf16 work dtype, not {work()}")
        weight = torch.cat([p["weight"] for p in parts])
        bias = None
        if parts[0]["bias"] is not None:
            bias = torch.cat([p["bias"] for p in parts]).to(work())
        if fmt == "bf16":
            if weight.dtype in FLOAT8:
                raise ValueError(f"linear {names}: {weight.dtype} weight but no quant scheme")
            return cls(fmt, weight, bias=bias)
        if fmt == "mxfp8":
            block = torch.cat([p["block"] for p in parts])
            return cls(fmt, weight.contiguous(), mx.swizzle(block.contiguous()), bias)
        if fmt == "nvfp4":
            block = torch.cat([p["block"] for p in parts])
            return cls(
                fmt,
                weight.contiguous(),
                nv.swizzle(block.contiguous()),
                bias,
                parts[0]["wmul"],
                parts[0]["imul"],
                k=weight.shape[1] * 2,
            )
        if fmt == "fp8_dynamic":
            block = torch.cat([p["block"] for p in parts]).contiguous()
            return cls(fmt, weight.contiguous(), block, bias)
        if fmt == "fp8":
            wmul = [p["wmul"].expand(p["weight"].shape[0]) for p in parts]
            same = all(torch.equal(p["wmul"], parts[0]["wmul"]) for p in parts)
            wmul = parts[0]["wmul"] if same else torch.cat(wmul).contiguous()
            return cls(fmt, weight.contiguous(), None, bias, wmul, parts[0]["imul"])
        raise ValueError(f"linear {names}: format {fmt!r}")

    @classmethod
    def nvfp4_dynamic(cls, weight: torch.Tensor) -> "Linear":
        """A bf16 weight quantized to NVFP4 at load (tensor scale from its amax); the activation
        quantized per call with a tensor scale from its own amax (no calibration): an LM head."""
        # aminmax: no temporary, whose block a small tensor would pin past the load
        lo, hi = weight.aminmax()
        gs = (E2M1_E4M3 / torch.max(-lo, hi).float().clamp(min=1e-12)).reshape(1)
        q, s = nv.nvfp4_quant(weight, gs)
        lin = cls("nvfp4", q, s, wmul=1.0 / gs, imul=torch.ones_like(gs), k=weight.shape[1])
        lin.dynamic = True
        return lin

    def to_int8(self, rotate: int = 0) -> None:
        """This bf16 layer as W8A8 in place; ``rotate``: the rotation group, 0 for none."""
        w = self.weight.float()
        self.rotate = rotate
        if rotate:
            g = regular_hadamard(rotate, w.device)
            k = self.k // rotate * rotate
            w[:, :k] = (w[:, :k].reshape(-1, rotate) @ g.t()).reshape(self.n, k)
        scale = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
        q = torch.round(w / scale).to(torch.int8)
        pad = (0, i8.pad_k(self.k) - self.k, 0, i8.pad_n(self.n) - self.n)
        self.fmt, self.weight, self.wmul = "int8", _param(F.pad(q, pad)), _param(scale[:, 0])

    def set_route(self, route: str) -> None:
        """This fp16 layer's fp32 matmul; ``tilelang`` pads a wide weight to whole tiles."""
        if route not in ROUTES:
            raise ValueError(f"matmul route {route!r}: one of {ROUTES}")
        if route == "tilelang" and f16 is None:
            raise ValueError("matmul route tilelang: TileLang not installed")
        self.route = route
        if route == "tilelang" and self.n > skinny.MAX_N:
            bn, bk = f16.pad_size(self.weight.device)
            pad = (0, -(-self.k // bk) * bk - self.k, 0, -(-self.n // bn) * bn - self.n)
            self.weight = _param(F.pad(self.weight, pad))

    def _accumulated(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 -> fp16 product -> fp32."""
        if self.route == "tilelang":
            return f16.linear(x, self.weight, self.n, self.bias)
        out = F.linear(x.to(self.weight.dtype), self.weight).float()
        return out + self.bias if self.bias is not None else out

    def _int8(self, x: torch.Tensor) -> torch.Tensor:
        q, amax = i8.quant(x, self.rotate)
        p = i8.gemm(q, self.weight)
        return i8.scale(p, x.shape[0], self.n, amax, self.wmul, self.bias, x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fmt == "int8":
            return self._int8(x)
        if self.fmt == "nvfp4" and self.dynamic:
            gs = (E2M1_E4M3 / x.float().abs().amax().clamp(min=1e-12)).reshape(1)
            out = nv.gemm(*nv.nvfp4_quant(x, gs), self.weight, self.scale, self.wmul / gs)
            return out + self.bias if self.bias is not None else out
        if self.quantizes:
            return self.quantized(*self.act("plain", x))
        if self.fmt == "fp8":
            out = fp8.gemm(fp8.quant(x, self.imul), self.imul, self.weight, self.wmul)
            return out + self.bias if self.bias is not None else out
        if self.fmt == "fp8_dynamic":
            return fp8.dynamic_gemm(*fp8.dynamic_quant(x), self.weight, self.scale, self.bias)
        if self.n <= skinny.MAX_N and x.is_cuda and x.dim() == 2 and x.stride(1) == 1:
            return skinny.linear(x, self.weight, self.bias)
        if x.dtype != self.weight.dtype:
            return self._accumulated(x)
        return F.linear(x, self.weight, self.bias)

    def quantized(self, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """The product with an activation already quantized for this layer (``act``, or a
        fused norm producer)."""
        if self.fmt == "nvfp4":
            out = nv.gemm(q, s, self.weight, self.scale, self.alpha)
        else:
            out = mx.gemm(q, s, self.weight, self.scale)
        return out + self.bias if self.bias is not None else out

    def dequantized(self) -> "Linear":
        """This layer in the work dtype (an MXFP8 weight times its scales): the reference for the
        quantized activation path."""
        if not self.is_mx:
            return self
        kb = self.k // 32
        rows = -(-self.n // 128) * 128
        tiles = self.scale.view(rows // 128, kb // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
        exps = tiles.reshape(rows, kb)[: self.n].float() - 127
        w = self.weight.float().view(self.n, kb, 32) * torch.exp2(exps)[..., None]
        return Linear("bf16", w.view(self.n, self.k).to(work()), bias=self.bias)


def _agree(a: dict, b: dict) -> bool:
    """Whether two parts fit one GEMM."""
    if a["fmt"] != b["fmt"]:
        return False
    if a["fmt"] == "nvfp4":
        return torch.equal(a["wmul"], b["wmul"]) and torch.equal(a["imul"], b["imul"])
    if a["fmt"] == "fp8":
        return torch.equal(a["imul"], b["imul"])
    return True


class Stacked(nn.Module):
    """Parts that cannot share one GEMM: each its own ``Linear``, outputs concatenated along N."""

    fmt = "stacked"
    is_mx = False
    quantizes = False

    def __init__(self, layers: list[Linear]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.n = sum(lin.n for lin in layers)
        self.k = layers[0].k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.parts(x), dim=-1)

    def parts(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Each part's output, unconcatenated (``columns`` reads a column range of them)."""
        return [lin(x) for lin in self.layers]


def columns(parts: list[torch.Tensor], lo: int, hi: int) -> torch.Tensor:
    """Columns ``lo:hi`` of the parts laid side by side: a view when one part holds them all."""
    out, start = [], 0
    for p in parts:
        end = start + p.shape[-1]
        if lo < end and hi > start:
            out.append(p[..., max(lo, start) - start : min(hi, end) - start])
        start = end
    return out[0] if len(out) == 1 else torch.cat(out, dim=-1)
