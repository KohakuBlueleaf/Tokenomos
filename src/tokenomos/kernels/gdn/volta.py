"""``prefill``'s tensor-core kernels on Volta (TileLang): ``gram`` writes the raw ``k @ k.T`` and
``q @ k.T`` of each chunk per value head; ``state`` runs a sequence's chunks with ``BV`` state rows
in shared memory, same recurrence as ``prefill._chunk``. Operands fp16, accumulators float32; one
warp per program, a chunk's 16 rows being one MMA tile high. Gates and block inverses stay on
Triton, vector-only there.
"""

from functools import cache

import tilelang
import tilelang.language as T

from tokenomos.kernels.linear.int8 import arch

ACC = "float32"
MM = "float16"
THREADS = 32
PASSES = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


def _compile(prim, device):
    return tilelang.compile(
        prim, target={"kind": "cuda", "arch": arch(device)}, pass_configs=PASSES
    )


@cache
def _gram_kernel(h: int, hk: int, kd: int, bk: int, bt: int, device):
    lq, nc, ncu, lo = T.dynamic("lq"), T.dynamic("nc"), T.dynamic("ncu"), T.dynamic("lo")
    policy = T.GemmWarpPolicy.FullRow

    @T.prim_func
    def kernel(
        Q: T.Tensor((lq,), MM),
        K: T.Tensor((lq,), MM),
        Chunks: T.Tensor((nc * 2,), "int32"),
        Cu: T.Tensor((ncu,), "int32"),
        KK: T.Tensor((lo,), ACC),
        QK: T.Tensor((lo,), ACC),
    ):
        with T.Kernel(h, nc, threads=THREADS) as (i_h, i_c):
            i_n = Chunks[2 * i_c]
            i_t = Chunks[2 * i_c + 1]
            bos = Cu[i_n]
            n = Cu[i_n + 1] - bos
            base = (bos * hk + i_h // (h // hk)) * kd
            q_s = T.alloc_shared((bt, bk), MM)
            k_a = T.alloc_shared((bt, bk), MM)
            k_b = T.alloc_shared((bt, bk), MM)
            kk = T.alloc_fragment((bt, bt), ACC)
            qk = T.alloc_fragment((bt, bt), ACC)
            T.clear(kk)
            T.clear(qk)
            for ib in T.serial(kd // bk):
                for i, j in T.Parallel(bt, bk):
                    t = i_t * bt + i
                    at = base + t * (hk * kd) + ib * bk + j
                    q_s[i, j] = T.if_then_else(t < n, Q[at], 0)
                    k_a[i, j] = T.if_then_else(t < n, K[at], 0)
                    k_b[i, j] = T.if_then_else(t < n, K[at], 0)
                T.gemm(k_a, k_b, kk, transpose_B=True, policy=policy)
                T.gemm(q_s, k_b, qk, transpose_B=True, policy=policy)
            for i, j in T.Parallel(bt, bt):
                at = (i_c * h + i_h) * (bt * bt) + i * bt + j
                KK[at] = kk[i, j]
                QK[at] = qk[i, j]

    return _compile(kernel, device)


def gram(q, k, cu, chunks, kk, qk, bk: int, h: int, bt: int) -> None:
    """Fills ``kk`` / ``qk`` [chunks * H, BT, BT] float32 from ``q``, ``k`` [T, Hk, K] fp16."""
    hk, kd = q.shape[1:]
    kernel = _gram_kernel(h, hk, kd, bk, bt, q.device)
    kernel(q.view(-1), k.view(-1), chunks.view(-1), cu, kk.view(-1), qk.view(-1))


@cache
def _state_kernel(h: int, hk: int, kd: int, vd: int, bv: int, bt: int, out: str, device):
    lq, lv, lo, ls, lc, ng, ncu, ns = (
        T.dynamic(x) for x in ("lq", "lv", "lo", "ls", "lc", "ng", "ncu", "ns")
    )
    policy = T.GemmWarpPolicy.FullRow

    @T.prim_func
    def kernel(
        Q: T.Tensor((lq,), MM),
        K: T.Tensor((lq,), MM),
        V: T.Tensor((lv,), MM),
        G: T.Tensor((ng,), ACC),
        Beta: T.Tensor((ng,), ACC),
        Ti: T.Tensor((lc,), MM),
        W: T.Tensor((lc,), MM),
        Out: T.Tensor((lv,), out),
        H0: T.Tensor((ls,), ACC),
        Ht: T.Tensor((ls,), ACC),
        Cu: T.Tensor((ncu,), "int32"),
        Starts: T.Tensor((ns,), "int32"),
        scale: T.float32,
    ):
        with T.Kernel(vd // bv, ns * h, threads=THREADS) as (i_v, i_nh):
            i_n = i_nh // h
            i_h = i_nh % h
            bos = Cu[i_n]
            n = Cu[i_n + 1] - bos
            start = Starts[i_n]
            base = (bos * hk + i_h // (h // hk)) * kd
            at_v = (bos * h + i_h) * vd + i_v * bv
            at_s = i_nh * (vd * kd) + i_v * bv * kd
            q_s = T.alloc_shared((bt, kd), MM)
            k_s = T.alloc_shared((bt, kd), MM)
            s_s = T.alloc_shared((bv, kd), ACC)
            sm = T.alloc_shared((bv, kd), MM)
            t_s = T.alloc_shared((bt, bt), MM)
            w_s = T.alloc_shared((bt, bt), MM)
            x_s = T.alloc_shared((bt, bv), MM)
            u_s = T.alloc_shared((bt, bv), MM)
            ur = T.alloc_shared((bv, bt), MM)
            sk = T.alloc_fragment((bt, bv), ACC)
            qs = T.alloc_fragment((bt, bv), ACC)
            u = T.alloc_fragment((bt, bv), ACC)
            wu = T.alloc_fragment((bt, bv), ACC)
            d = T.alloc_fragment((bv, kd), ACC)
            c = T.alloc_fragment((bt,), ACC)
            r = T.alloc_fragment((bt,), ACC)

            for i, j in T.Parallel(bv, kd):
                s_s[i, j] = H0[at_s + i * kd + j]
            for i_t in T.serial(T.ceildiv(n, bt)):
                g_at = (bos + T.min((i_t + 1) * bt, n) - 1) * h + i_h
                g_end = G[g_at]
                keep = T.exp(g_end)
                at_c = (start * h + i_h) * (bt * bt) + i_t * (h * bt * bt)
                for i, j in T.Parallel(bt, kd):
                    t = i_t * bt + i
                    at = base + t * (hk * kd) + j
                    q_s[i, j] = T.if_then_else(t < n, Q[at], 0)
                    k_s[i, j] = T.if_then_else(t < n, K[at], 0)
                for i, j in T.Parallel(bt, bt):
                    t_s[i, j] = Ti[at_c + i * bt + j]
                    w_s[i, j] = W[at_c + i * bt + j]
                for i, j in T.Parallel(bv, kd):
                    sm[i, j] = s_s[i, j]
                for i in T.Parallel(bt):
                    t = i_t * bt + i
                    p_g = (bos + t) * h + i_h
                    c[i] = T.if_then_else(t < n, T.exp(G[p_g]), 1.0)
                    r[i] = T.if_then_else(t < n, T.exp(g_end - G[p_g]), 0.0)
                T.clear(sk)
                T.gemm(k_s, sm, sk, transpose_B=True, policy=policy)
                for i, j in T.Parallel(bt, bv):
                    t = i_t * bt + i
                    p_v = at_v + t * (h * vd) + j
                    p_g = (bos + t) * h + i_h
                    x_s[i, j] = T.if_then_else(
                        t < n, (V[p_v].astype(ACC) - sk[i, j] * c[i]) * Beta[p_g], 0
                    )
                T.clear(u)
                T.gemm(t_s, x_s, u, policy=policy)
                for i, j in T.Parallel(bt, bv):
                    u_s[i, j] = u[i, j]
                T.clear(qs)
                T.gemm(q_s, sm, qs, transpose_B=True, policy=policy)
                T.clear(wu)
                T.gemm(w_s, u_s, wu, policy=policy)
                for i, j in T.Parallel(bt, bv):
                    t = i_t * bt + i
                    if t < n:
                        o = (qs[i, j] * c[i] + wu[i, j]) * scale
                        Out[at_v + t * (h * vd) + j] = o.astype(out)
                for i, j in T.Parallel(bt, bv):
                    ur[j, i] = u[i, j] * r[i]
                T.clear(d)
                T.gemm(ur, k_s, d, policy=policy)
                for i, j in T.Parallel(bv, kd):
                    s_s[i, j] = s_s[i, j] * keep + d[i, j]
            for i, j in T.Parallel(bv, kd):
                Ht[at_s + i * kd + j] = s_s[i, j]

    return _compile(kernel, device)


def state(q, k, v, gc, beta, Ti, W, out, initial, final, cu, starts, scale, bv: int) -> None:
    """Every sequence's chunks in order: fills ``out`` and ``final``."""
    hk, kd = q.shape[1:]
    hv, vd = v.shape[1:]
    bt = W.shape[1]
    kernel = _state_kernel(hv, hk, kd, vd, bv, bt, str(out.dtype).removeprefix("torch."), q.device)
    kernel(
        q.view(-1),
        k.view(-1),
        v.view(-1),
        gc.view(-1),
        beta.view(-1),
        Ti.view(-1),
        W.view(-1),
        out.view(-1),
        initial.view(-1),
        final.view(-1),
        cu,
        starts,
        scale,
    )
