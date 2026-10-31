"""Rooflines for the decode step's two big costs, measured on this GPU.

GEMM: the MXFP8 peak (a large square product, best backend) and every LFM3 linear shape at the
decode token counts on each backend, as TFLOP/s and fraction of that peak, plus the weight-read
time at copy bandwidth. Attention: copy bandwidth (device-to-device, large buffer) and the paged
decode kernel at given contexts as GB/s of KV read and fraction of copy bandwidth.
Writes ``reports/roofline.json``.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from tokenomos.kernels.attention import paged
from tokenomos.kernels.mxfp8 import ops as mx

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"
SHAPES = {
    "conv.in_proj": (6144, 2048),
    "conv.out_proj": (2048, 2048),
    "attn.qkv": (3072, 2048),
    "attn.o": (2048, 2048),
    "ffn.gate_up": (15360, 2048),
    "ffn.down": (2048, 7680),
    "head": (128000, 2048),
}


def bench(fn) -> float:
    """Median ms of a CUDA-graph replay of ``fn`` (launch overhead excluded)."""
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return triton.testing.do_bench(g.replay, warmup=25, rep=200, return_mode="median")


def run_gemm(m, n, k, backend):
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    q, s = mx.quant(x)
    wq, ws = mx.quant(w)
    if backend == "b12x":
        if mx.mm_mxfp8 is None:
            return None
        return bench(
            lambda: mx.mm_mxfp8(q, wq.t(), s, ws, out_dtype=torch.bfloat16, backend="b12x")
        )
    sc = (s.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu))
    return bench(
        lambda: F.scaled_mm(
            q,
            wq.t(),
            sc[0],
            mx.BLOCK,
            sc[1],
            mx.BLOCK,
            mx.SWIZZLE,
            mx.SWIZZLE,
            output_dtype=torch.bfloat16,
        )
    )


def copy_bandwidth() -> float:
    a = torch.empty(1 << 30, dtype=torch.uint8, device=DEV)
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    return 2 * a.numel() / ms / 1e6


def attention(ctx: int, seqs: int, hkv=4, h=16, d=128, page=32, window=None) -> dict:
    pages_per = -(-(ctx + 1) // page)
    kc = torch.randn(seqs * pages_per + 1, page, hkv, d, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    table = torch.arange(1, seqs * pages_per + 1, device=DEV, dtype=torch.int32).view(seqs, -1)
    table = torch.cat([table[:1] * 0, table]).contiguous()
    slot = torch.arange(1, seqs + 1, device=DEV, dtype=torch.int32)
    pos = torch.full((seqs,), ctx, device=DEV, dtype=torch.int32)
    q = torch.randn(seqs, h, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(seqs, hkv, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    ms = bench(lambda: paged.decode(q, k, v, kc, vc, slot, pos, table, d**-0.5, window))
    read = min(ctx, window or ctx)
    gbs = seqs * read * hkv * d * 2 * 2 / ms / 1e6
    return {"ctx": ctx, "seqs": seqs, "window": window, "ms": round(ms, 4), "GB_s": round(gbs, 1)}


def main() -> None:
    torch.manual_seed(0)
    bw = copy_bandwidth()
    peak = {}
    for backend in ("cublas", "b12x"):
        ms = run_gemm(8192, 8192, 8192, backend)
        if ms:
            peak[backend] = 2 * 8192**3 / ms / 1e9
    top = max(peak.values())
    gemms = []
    for name, (n, k) in SHAPES.items():
        for m in (64, 128, 256, 512):
            row = {"name": name, "m": m, "n": n, "k": k, "chosen": mx.backend(n, k, m)}
            # e4m3 bytes plus 1/32 scale bytes at copy bandwidth (GB/s)
            row["weight_read_ms"] = round(n * k * (1 + 1 / 32) / bw / 1e6, 4)
            row["compute_floor_ms"] = round(2 * m * n * k / top / 1e9, 4)
            for backend in ("cublas", "b12x"):
                ms = run_gemm(m, n, k, backend)
                if ms:
                    tf = 2 * m * n * k / ms / 1e9
                    row[backend] = {
                        "ms": round(ms, 4),
                        "TFLOPs": round(tf, 1),
                        "of_peak": round(tf / top, 3),
                    }
            gemms.append(row)
            print(row)
    attn = [attention(c, 256) for c in (128, 256, 448, 1024, 4096)]
    attn += [attention(c, 256, window=256) for c in (448, 4096)]
    for a in attn:
        a["of_copy_bw"] = round(a["GB_s"] / bw, 3)
        print(a)
    out = {"copy_GB_s": round(bw, 1), "gemm_peak_TFLOPs": peak, "gemm": gemms, "attention": attn}
    path = ROOT / "reports" / "roofline.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("copy GB/s", round(bw, 1), "peak", peak)


if __name__ == "__main__":
    main()
