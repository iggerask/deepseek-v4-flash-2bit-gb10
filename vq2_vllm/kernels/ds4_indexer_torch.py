"""Pure-torch sm_121 replacements for the DeepSeek-V4 lightning-indexer logit
kernels (DeepGEMM fp8_fp4_*_mqa_logits assert 'Unsupported architecture' on sm_121).

DS4-Flash uses the FP8 indexer path (use_fp4_cache=False; the unfused FP4 insert is
unsupported), so q/k dequant is just .float() * scale — no FP4 unpack.

Indexer logit math (MQA, weighted multi-head -> single score):
    logits[m, n] = sum_h weights[m, h] * ( q_fp8[m, h, :] . (k_fp8[n, :] * k_scale[n]) )
q per-token scale is folded into `weights` (FP8 path). clean_logits=False in DS4's
calls, so range masking is left to the top-k op (top_k_per_row_* via cu_seqlen/context_lens).
"""
import torch

_NEG = float("-inf")


def fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False):
    """PREFILL (non-paged). q=(qv [M,H,D] fp8, None); kv=(kv_v [N,D] fp8, kv_s [N] f32);
    weights [M,H] f32. Returns logits [M,N] f32."""
    qv, _ = q
    kv_v, kv_s = kv
    M, H, D = qv.shape
    N = kv_v.shape[0]
    # MQA: one K per key position, shared across all H query heads, so the per-head
    # weighting commutes into a head-reduced query: logit[m,n] = (sum_h w[m,h] q[m,h]).k[n].
    # Reduce heads FIRST (qbar [M,D]) -> single [M,D]@[D,N] matmul: ~H x less compute,
    # no [M,H,N] intermediate. fp8 stays fp8; fp32 only as matmul accumulation.
    qbar = torch.einsum("mhd,mh->md", qv.float(), weights.float())   # [M,D]
    out = (qbar @ kv_v.float().t()) * kv_s.float().reshape(1, N)     # [M,N]
    if clean_logits:
        ar = torch.arange(N, device=qv.device)
        mask = (ar[None, :] >= cu_seqlen_ks[:, None]) & (ar[None, :] < cu_seqlen_ke[:, None])
        out = out.masked_fill(~mask, _NEG)
    return out


def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
    """Stub: schedule metadata is only consumed by the DeepGEMM paged kernel, which
    we replace with a torch impl that ignores it. Buffer is [num_sms+1, 2] int32."""
    return torch.zeros(num_sms + 1, 2, dtype=torch.int32, device=context_lens.device)


def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables,
                             schedule_metadata, max_model_len, clean_logits=False):
    """DECODE (paged). q=(qv [B,next_n,H,D] fp8, None); kv_cache [num_blocks,bs,1,D+4]
    uint8 (per token: [0:D] fp8 vals, [D:D+4] f32 scale); weights [B*next_n,H] f32;
    context_lens [B] (or [B,next_n]); block_tables [B,max_blocks]. Returns [B*next_n, max_model_len]."""
    qv, _ = q
    B, next_n, H, D = qv.shape
    nb, bs = kv_cache.shape[0], kv_cache.shape[1]
    dev = qv.device
    clen = context_lens
    if clen.ndim == 2:
        clen = clen[:, -1]
    clen = clen.to(dev)
    # Gather K for all positions up to the cache extent, vectorized + graph-safe.
    # Index kv_cache DIRECTLY (native strides) -> gathers only B*L rows; NO full-cache copy.
    L = min(block_tables.shape[1] * bs, max_model_len)
    pos = torch.arange(L, device=dev)
    blk = block_tables[:, pos // bs].long().clamp(0, nb - 1)  # [B,L]
    off = (pos % bs)[None, :].expand(B, L)                    # [B,L]
    rows = kv_cache[blk, off, 0] if kv_cache.dim() == 4 else kv_cache[blk, off]  # [B,L,D+4]
    kf = rows[..., :D].contiguous().view(torch.float8_e4m3fn).float()    # [B,L,D]
    ks = rows[..., D:D + 4].contiguous().view(torch.float32)             # [B,L,1]
    kf = kf * ks
    qf = qv.float()                                           # [B,next_n,H,D]
    w = weights.float().reshape(B, next_n, H)
    # MQA head-reduction (same identity as the prefill fp8_fp4_mqa_logits 24x win): the per-head
    # weighting commutes into a head-reduced query, so reduce heads FIRST -> avoid materializing the
    # [B,next_n,H,L] per-head logits (H=64x less compute + no H*L intermediate). Exact; scales with L
    # -> a decode-at-long-context win (the old materialization grew with context).
    qbar = torch.einsum("bnhd,bnh->bnd", qf, w)             # [B,next_n,D]
    lg = torch.einsum("bnd,bld->bnl", qbar, kf).reshape(B * next_n, L)  # [B*next_n, L]
    valid = (pos[None, :] < clen[:, None])[:, None, :].expand(B, next_n, L).reshape(B * next_n, L)
    out = torch.full((B * next_n, max_model_len), _NEG, dtype=torch.float32, device=dev)
    out[:, :L] = torch.where(valid, lg, _NEG)
    return out
