"""Triton grouped W4A16 GEMV for the DeepSeek-V4-Flash o_proj wo_a (NVFP4, block-diagonal
BMM over n_groups). Reads 4-bit weights inline (no bf16 materialization), so wo_a stays
4-bit in RAM per recipe while being fast.

z[t,g,r] = sum_k o_g[t,g,k] * dequant(wo_a[g*RANK+r, k])
wo_a NVFP4: weight [G*RANK, K//2] uint8 (2 E2M1 nibbles/byte), weight_scale [G*RANK, K//16]
fp8-e4m3 per-16 microscale, weight_scale_2 scalar global. value = E2M1[idx]*sign * (sc*gs).
"""
import torch
import triton
import triton.language as tl

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_DEV = {}


def _e2m1(dev):
    t = _E2M1_DEV.get(dev)
    if t is None:
        t = _E2M1.to(dev); _E2M1_DEV[dev] = t
    return t


@triton.jit
def _oproj_wa_kernel(O, WQ, WS, GS, Z, G, RANK, K, T,
                     BR: tl.constexpr, BKB: tl.constexpr):
    # byte-once + arithmetic E2M1 (no LUT gather) + tl.store (no atomic). Each program
    # owns BR output rows of one group and the full K reduction.
    g = tl.program_id(0)
    rt = tl.program_id(1)
    rows = rt * BR + tl.arange(0, BR)
    rmask = rows < RANK
    wrow = g * RANK + rows
    gs = tl.load(GS)
    KB = K // 2                                        # packed bytes per row
    jj = tl.arange(0, BKB)
    for t in tl.range(0, T):
        acc = tl.zeros((BR,), tl.float32)
        for j0 in tl.range(0, KB, BKB):
            jb = j0 + jj
            km = jb < KB
            ld = rmask[:, None] & km[None, :]
            byte = tl.load(WQ + wrow[:, None] * KB + jb[None, :], mask=ld, other=0)
            for half in tl.static_range(2):            # nibble -> col 2*jb (lo) / 2*jb+1 (hi)
                nib = (byte & 0xF) if half == 0 else (byte >> 4)
                e = ((nib >> 1) & 3).to(tl.float32)
                m = (nib & 1).to(tl.float32)
                val = tl.where(e == 0, 0.5 * m, tl.exp2(e - 1.0) * (1.0 + 0.5 * m))   # E2M1 magnitude
                val = tl.where((nib & 8) != 0, -val, val)
                col = 2 * jb + half
                sc = tl.load(WS + wrow[:, None] * (K // 16) + (col // 16)[None, :], mask=ld, other=0.).to(tl.float32)
                ov = tl.load(O + (t * G + g) * K + col[None, :], mask=km[None, :], other=0.).to(tl.float32)
                acc += tl.sum(val * ov * sc, axis=1)
        tl.store(Z + (t * G + g) * RANK + rows, acc * gs, mask=rmask)


@triton.jit
def _oproj_wa_bt_kernel(O, WQ, WS, GS, Z, G, RANK, K,
                        BR: tl.constexpr, BKB: tl.constexpr, T: tl.constexpr):
    # T-BATCHED (spec-decode verify, T=K+1): read the 4-bit wo_a weight + scale ONCE per K-chunk and
    # broadcast against ALL T tokens' activations (acc [BR,T]) -> wo_a read once, not T x. Same math as
    # _oproj_wa_kernel, loops reordered. T is constexpr (compiles per captured spec size).
    g = tl.program_id(0); rt = tl.program_id(1)
    rows = rt * BR + tl.arange(0, BR); rmask = rows < RANK
    wrow = g * RANK + rows
    gs = tl.load(GS); KB = K // 2
    jj = tl.arange(0, BKB); tt = tl.arange(0, T)
    acc = tl.zeros((BR, T), tl.float32)
    for j0 in tl.range(0, KB, BKB):
        jb = j0 + jj; km = jb < KB; ld = rmask[:, None] & km[None, :]
        byte = tl.load(WQ + wrow[:, None] * KB + jb[None, :], mask=ld, other=0)
        for half in tl.static_range(2):
            nib = (byte & 0xF) if half == 0 else (byte >> 4)
            e = ((nib >> 1) & 3).to(tl.float32); m = (nib & 1).to(tl.float32)
            val = tl.where(e == 0, 0.5 * m, tl.exp2(e - 1.0) * (1.0 + 0.5 * m))
            val = tl.where((nib & 8) != 0, -val, val)
            col = 2 * jb + half
            sc = tl.load(WS + wrow[:, None] * (K // 16) + (col // 16)[None, :], mask=ld, other=0.).to(tl.float32)
            wv = val * sc                                                       # [BR,BKB] weight*scale ONCE
            ov = tl.load(O + (g * K) + col[None, :] + (tt * G * K)[:, None],
                         mask=km[None, :], other=0.).to(tl.float32)             # [T,BKB] all tokens
            acc += tl.sum(wv[:, None, :] * ov[None, :, :], axis=2)              # [BR,T]
    zp = Z + g * RANK + rows[None, :] + (tt * G * RANK)[:, None]                # [T,BR]
    tl.store(zp, tl.trans(acc) * gs, mask=rmask[None, :])


# BR=16 (small row-tile) + high SPLIT_K is the win: G=8 alone underfills the GPU, so many small
# programs raise occupancy (30% MBU vs 19% at BR=32). Tuned offline: BR16/BKB64/SK8.
_DOTCFG = [triton.Config({'BR': br, 'BKB': bkb, 'SPLIT_K': sk}, num_warps=w, num_stages=2)
           for br in (16, 32) for bkb in (64, 128) for sk in (4, 8) for w in (2, 4)]


@triton.autotune(configs=_DOTCFG, key=['RANK', 'K', 'TPAD'], reset_to_zero=['Z'])
@triton.jit
def _oproj_wa_dot_kernel(O, WQ, WS, GS, Z, G, RANK, K,
                         BR: tl.constexpr, BKB: tl.constexpr, T: tl.constexpr, TPAD: tl.constexpr,
                         SPLIT_K: tl.constexpr):
    # TENSOR-CORE T-batched verify o_proj: read 4-bit wo_a once, do the T-reduction with tl.dot
    # (not a manual outer-product, which made the old bt kernel LINEAR in T -> 9% MBU). split-K adds
    # programs (G=8 alone underfills the GPU). ~2x faster than _oproj_wa_bt_kernel at T=K+1, no bf16 cache.
    g = tl.program_id(0); rt = tl.program_id(1); ps = tl.program_id(2)
    rows = rt * BR + tl.arange(0, BR); rmask = rows < RANK
    wrow = g * RANK + rows
    gs = tl.load(GS); KB = K // 2
    jj = tl.arange(0, BKB); tt = tl.arange(0, TPAD)
    acc = tl.zeros((BR, TPAD), tl.float32)
    for j0 in tl.range(ps * BKB, KB, SPLIT_K * BKB):
        jb = j0 + jj; km = jb < KB; ld = rmask[:, None] & km[None, :]
        byte = tl.load(WQ + wrow[:, None] * KB + jb[None, :], mask=ld, other=0)
        for half in tl.static_range(2):
            nib = (byte & 0xF) if half == 0 else (byte >> 4)
            e = ((nib >> 1) & 3).to(tl.float32); m = (nib & 1).to(tl.float32)
            val = tl.where(e == 0, 0.5 * m, tl.exp2(e - 1.0) * (1.0 + 0.5 * m))
            val = tl.where((nib & 8) != 0, -val, val)
            col = 2 * jb + half
            sc = tl.load(WS + wrow[:, None] * (K // 16) + (col // 16)[None, :], mask=ld, other=0.).to(tl.float32)
            wv = (val * sc).to(tl.bfloat16)                                       # [BR,BKB]
            ov = tl.load(O + (g * K) + col[:, None] + (tt * G * K)[None, :],
                         mask=km[:, None] & (tt < T)[None, :], other=0.).to(tl.bfloat16)  # [BKB,TPAD]
            acc += tl.dot(wv, ov)                                                 # [BR,TPAD]
    zp = Z + g * RANK + rows[:, None] + (tt * G * RANK)[None, :]
    out = acc * gs; msk = rmask[:, None] & (tt < T)[None, :]
    if SPLIT_K == 1:
        tl.store(zp, out, mask=msk)
    else:
        tl.atomic_add(zp, out, mask=msk)


def oproj_wa(o_g, wa_weight, wa_scale, wa_gs, G, RANK):
    """o_g [T,G,K] -> z [T,G,RANK]. wa_weight [G*RANK,K//2] u8, wa_scale [G*RANK,K//16] fp8.
    Byte-once + arithmetic E2M1 + store (offline-tuned BR16/BKB64 ~0.245ms, ~68GB/s).
    T>1 (spec verify) uses the T-batched kernel (wo_a read once for all T)."""
    import os
    T = o_g.shape[0]
    K = o_g.shape[2]
    dev = o_g.device
    BR, BKB = 16, 64
    if T > 1 and os.environ.get("VQ2_OPROJ_DOT", "1") == "1":
        # spec-decode VERIFY (T=K+1): tensor-core T-batched kernel (read wo_a once, tl.dot over T).
        # ~2x the old manual-broadcast bt kernel (which was linear in T -> 9% MBU). No bf16 cache.
        TPAD = max(16, triton.next_power_of_2(T))
        z = torch.zeros(T, G, RANK, dtype=torch.float32, device=dev)
        grid = lambda meta: (G, triton.cdiv(RANK, meta['BR']), meta['SPLIT_K'])
        _oproj_wa_dot_kernel[grid](
            o_g.contiguous(), wa_weight, wa_scale.view(torch.float8_e4m3fn), wa_gs.float().reshape(()), z,
            G, RANK, K, T=T, TPAD=TPAD)
        return z
    z = torch.empty(T, G, RANK, dtype=torch.float32, device=dev)
    grid = (G, triton.cdiv(RANK, BR))
    if T > 1 and os.environ.get("VQ2_OPROJ_BT", "1") == "1":
        _oproj_wa_bt_kernel[grid](
            o_g.contiguous(), wa_weight, wa_scale.view(torch.float8_e4m3fn), wa_gs.float().reshape(()), z,
            G, RANK, K, BR=BR, BKB=BKB, T=T, num_warps=4,
        )
        return z
    _oproj_wa_kernel[grid](
        o_g.contiguous(), wa_weight, wa_scale.view(torch.float8_e4m3fn), wa_gs.float().reshape(()), z,
        G, RANK, K, T, BR=BR, BKB=BKB, num_warps=4,
    )
    return z
