"""Triton sm_121 sparse-MLA decode kernel for DeepSeek-V4-Flash.

Flash-decode (MQA, K==V 512-dim latent) over indexer-selected keys, reading the
fp8_ds_mla paged cache inline with NATIVE STRIDES (no reshape/contiguous copy of the
multi-GB cache — the paged blocks are padded, so flattening copied the whole cache
every step; that was ~78% of decode time). bf16 RoPE is assembled from its 2 bytes
in-kernel (avoids the .view(bf16) that forced a contiguous copy).

Per block (BB = cache.stride(0) bytes, padded): token data at pos*576 ([0:448] fp8-e4m3
NoPE in 7 UE8M0 groups of 64, [448:576] 64 bf16 RoPE); scales at bs*576 + pos*8.
attn_sink = virtual key with logit sink[h], zero value.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_decode_kernel(
    Q, OUT, SINK,
    CU8_S, SIDX, SLEN,                  # swa: uint8 cache (raw), idx[B,W], len[B]
    CU8_E, EIDX, ELEN,                  # extra (compressed): same
    scale, H, W, T, bs_s, bs_e, BB_S, BB_E,
    HAS_E: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    NOPE: tl.constexpr, ROPE: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
):
    b = tl.program_id(0)
    ht = tl.program_id(1)
    hoff = ht * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = hoff < H
    cn = tl.arange(0, D)                                 # 512 (pow2); nope = cn<NOPE
    cr = tl.arange(0, ROPE)
    nmask = cn < NOPE
    gcol = tl.minimum(cn // GROUP, 7)
    qbase = b * H * D + hoff[:, None] * D
    qn = tl.load(Q + qbase + cn[None, :], mask=hmask[:, None], other=0.0).to(tl.bfloat16)
    qr = tl.load(Q + qbase + (NOPE + cr)[None, :], mask=hmask[:, None], other=0.0).to(tl.bfloat16)

    m = tl.full((BLOCK_H,), float('-inf'), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    accn = tl.zeros((BLOCK_H, D), tl.float32)
    accr = tl.zeros((BLOCK_H, ROPE), tl.float32)

    for which in tl.static_range(2):
        if which == 0:
            CU8, IDX, LEN, NK, bs, BB = CU8_S, SIDX, SLEN, W, bs_s, BB_S
            active = True
        else:
            CU8, IDX, LEN, NK, bs, BB = CU8_E, EIDX, ELEN, T, bs_e, BB_E
            active = HAS_E
        if active:
            slen = tl.load(LEN + b)
            for k0 in tl.range(0, NK, BLOCK_K):
                kk = k0 + tl.arange(0, BLOCK_K)
                idx = tl.load(IDX + b * NK + kk, mask=kk < NK, other=-1)
                valid = (kk < slen) & (idx >= 0)
                idc = tl.where(idx >= 0, idx, 0).to(tl.int64)
                blk = idc // bs
                pos = idc % bs
                base = blk * BB + pos * 576                       # uint8 byte offset of token data (native block stride)
                soff = blk * BB + bs * 576 + pos * 8             # scales
                nm = valid[:, None] & nmask[None, :]
                fp8 = tl.load(CU8 + base[:, None] + cn[None, :], mask=nm, other=0).to(tl.uint8)
                fp8 = fp8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                sc_u8 = tl.load(CU8 + soff[:, None] + gcol[None, :], mask=nm, other=0).to(tl.float32)
                Kn = tl.where(nm, fp8 * tl.exp2(sc_u8 - 127.0), 0.0).to(tl.bfloat16)   # [BK,512]
                # RoPE bf16 assembled from 2 bytes (rope region starts at base+448)
                rb = base + 448
                rlo = tl.load(CU8 + rb[:, None] + (2 * cr)[None, :], mask=valid[:, None], other=0).to(tl.uint16)
                rhi = tl.load(CU8 + rb[:, None] + (2 * cr + 1)[None, :], mask=valid[:, None], other=0).to(tl.uint16)
                Kr = (rlo | (rhi << 8)).to(tl.uint16).to(tl.bfloat16, bitcast=True)    # [BK,64]
                qk = (tl.dot(qn, tl.trans(Kn)) + tl.dot(qr, tl.trans(Kr))) * scale     # [BH,BK] f32
                qk = tl.where(valid[None, :], qk, float('-inf'))
                m_new = tl.maximum(m, tl.max(qk, axis=1))
                alpha = tl.exp(m - m_new)
                p = tl.exp(qk - m_new[:, None])
                pb = p.to(tl.bfloat16)
                l = l * alpha + tl.sum(p, axis=1)
                accn = accn * alpha[:, None] + tl.dot(pb, Kn)
                accr = accr * alpha[:, None] + tl.dot(pb, Kr)
                m = m_new

    sink = tl.load(SINK + hoff, mask=hmask, other=float('-inf')).to(tl.float32)
    m2 = tl.maximum(m, sink)
    alpha = tl.exp(m - m2)
    l = l * alpha + tl.exp(sink - m2)
    inv = 1.0 / tl.where(l > 0, l, 1.0)
    on = accn * (alpha * inv)[:, None]
    orr = accr * (alpha * inv)[:, None]
    tl.store(OUT + qbase + cn[None, :], on.to(OUT.dtype.element_ty), mask=hmask[:, None])
    tl.store(OUT + qbase + (NOPE + cr)[None, :], orr.to(OUT.dtype.element_ty), mask=hmask[:, None])


def flash_mla_with_kvcache(q, k_cache, block_table=None, head_dim_v=512,
                           tile_scheduler_metadata=None, cache_seqlens=None,
                           is_fp8_kvcache=True, indices=None, topk_length=None,
                           softmax_scale=None, attn_sink=None, extra_k_cache=None,
                           extra_indices_in_kvcache=None, extra_topk_length=None,
                           out=None, **kw):
    """Drop-in for vLLM flash_mla_with_kvcache (decode). q [B,1,H,512]; out [B,1,H,512].
    Reads caches with NATIVE strides (no copy)."""
    B, S, H, D = q.shape
    NOPE, ROPE, GROUP = 448, 64, 64
    bs_s = k_cache.shape[1]
    BB_S = k_cache.stride(0)                              # padded block stride (bytes), no copy
    swa_idx = indices.reshape(B, -1).to(torch.int64)
    W = swa_idx.shape[1]
    has_e = extra_k_cache is not None and extra_topk_length is not None
    if has_e:
        bs_e = extra_k_cache.shape[1]
        BB_E = extra_k_cache.stride(0)
        ex_idx = extra_indices_in_kvcache.reshape(B, -1).to(torch.int64)
        T = ex_idx.shape[1]
        ex_len = extra_topk_length.to(torch.int64)
        ex_cache = extra_k_cache
    else:
        bs_e, BB_E, T = 1, 1, 1
        ex_idx = swa_idx
        ex_len = topk_length.to(torch.int64)
        ex_cache = k_cache
    out2 = out.view(B, H, D)
    BLOCK_H = 16
    grid = (B, triton.cdiv(H, BLOCK_H))
    _mla_decode_kernel[grid](
        q.view(B, H, D), out2, attn_sink.float(),
        k_cache, swa_idx, topk_length.to(torch.int64),
        ex_cache, ex_idx, ex_len,
        float(softmax_scale), H, W, T, bs_s, bs_e, BB_S, BB_E,
        HAS_E=has_e, BLOCK_H=BLOCK_H, BLOCK_K=32,
        NOPE=NOPE, ROPE=ROPE, GROUP=GROUP, D=D,
        num_stages=1, num_warps=4,
    )
    return out, None


# ===================== PREFILL: flash sparse-MLA over bf16 workspace =====================
@triton.jit
def _mla_prefill_kernel(Q, KV, IDX, OUT, SINK, TKLEN,
                        H, TOPK, scale, KVSTRIDE,
                        BH: tl.constexpr, BK: tl.constexpr, D: tl.constexpr):
    """One program = (query s, head-block). Each query attends to its own TOPK keys
    gathered from the bf16 workspace KV[idx]. Online softmax + attn_sink, fp32 accum.
    MQA: the gathered keys are shared across the BH heads (K == V latent, dim D=512)."""
    s = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BH + tl.arange(0, BH)
    hmask = hoff < H
    cd = tl.arange(0, D)
    q = tl.load(Q + s * H * D + hoff[:, None] * D + cd[None, :], mask=hmask[:, None], other=0.0).to(tl.bfloat16)
    tklen = tl.load(TKLEN + s)
    m_i = tl.full((BH,), float('-inf'), tl.float32)
    l_i = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    for k0 in tl.range(0, TOPK, BK):
        kk = k0 + tl.arange(0, BK)
        idx = tl.load(IDX + s * TOPK + kk, mask=kk < TOPK, other=-1)
        valid = (kk < tklen) & (idx >= 0)
        idc = tl.where(idx >= 0, idx, 0).to(tl.int64)
        K = tl.load(KV + idc[:, None] * KVSTRIDE + cd[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)  # [BK,D]
        qk = tl.dot(q, tl.trans(K)) * scale                  # [BH,BK] f32
        qk = tl.where(valid[None, :], qk, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), K)
        m_i = m_new
    sink = tl.load(SINK + hoff, mask=hmask, other=float('-inf')).to(tl.float32)
    m2 = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m2)
    l_i = l_i * alpha + tl.exp(sink - m2)
    acc = acc * alpha[:, None]
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(OUT + s * H * D + hoff[:, None] * D + cd[None, :], out.to(OUT.dtype.element_ty), mask=hmask[:, None])


def flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink, topk_length, out=None, **kw):
    """PREFILL drop-in. q [sq,H,512] bf16; kv [M,1,512] bf16 workspace; indices [sq,1,T]
    int32 (global offsets into kv, -1 invalid); topk_length [sq]; out [sq,H,512]."""
    sq, H, D = q.shape
    kvf = kv.reshape(-1, D)
    idx = indices.reshape(sq, -1).to(torch.int32)
    T = idx.shape[1]
    if out is None:
        out = torch.empty_like(q)
    BH = 16
    grid = (sq, triton.cdiv(H, BH))
    _mla_prefill_kernel[grid](
        q, kvf, idx, out, attn_sink.float(), topk_length.to(torch.int32),
        H, T, float(sm_scale), kvf.stride(0),
        BH=BH, BK=32, D=D, num_stages=1, num_warps=4,
    )
    return out, None, None
