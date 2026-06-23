"""Pure-torch reference for DeepSeek-V4-Flash sparse-MLA forward, to run on sm_121
(GB10) where vLLM's compiled _flashmla_C sparse kernels (sm_90a/sm_100f only) raise.

Drop-in replacements for the two gated entry points imported by
vllm/models/deepseek_v4/nvidia/flashmla.py:
  - flash_mla_with_kvcache  (DECODE): reads the fp8_ds_mla paged cache by global
    slot, over SWA keys + compressed top-k keys, online-softmax + attn_sink.
  - flash_mla_sparse_fwd    (PREFILL): SDPA over the pre-dequantized bf16 workspace
    gathered by combined indices, + attn_sink.

MLA here is MQA in the 512-dim latent (K == V latent, no kv_b up-proj). attn_sink is
a virtual key with logit attn_sink[h] and zero value: out = sum(p*K)/(sum(p)+exp(sink-m)).

Correctness reference, not optimized (per-query loop in decode, query-chunked prefill).
Triton optimization is a follow-up. fp8 layout matches cache_utils.quantize_and_insert_k_kernel:
per block, token data at pos*576 ([0:448] fp8-e4m3 NoPE in 7 UE8M0 groups of 64, [448:576]
64 bf16 RoPE); scales at cache_block_size*576 + pos*8 (7 exponents, stored = exp+127).
"""
import torch

_NEG = float("-inf")


def _dequant_latents(cache_flat, slots, bs):
    """cache_flat: [num_blocks, bs*584] uint8. slots: [n] int (global = blk*bs+pos).
    Returns [n, 512] float32 (448 NoPE dequant + 64 RoPE bf16)."""
    n = slots.numel()
    dev = cache_flat.device
    if n == 0:
        return torch.zeros(0, 512, dtype=torch.float32, device=dev)
    blk = (slots // bs).long()
    pos = (slots % bs).long()
    data_col = (pos * 576).unsqueeze(1) + torch.arange(576, device=dev)      # [n,576]
    scale_col = (bs * 576 + pos * 8).unsqueeze(1) + torch.arange(8, device=dev)  # [n,8]
    data = cache_flat[blk.unsqueeze(1), data_col]            # [n,576] uint8
    scl = cache_flat[blk.unsqueeze(1), scale_col]            # [n,8] uint8
    nope_fp8 = data[:, :448].contiguous().view(torch.float8_e4m3fn).float()  # [n,448]
    rope = data[:, 448:576].contiguous().view(torch.bfloat16).float()       # [n,64]
    exps = scl[:, :7].float() - 127.0                        # [n,7] UE8M0 exponents
    scales = torch.exp2(exps)                                # [n,7]
    nope = (nope_fp8.view(n, 7, 64) * scales[:, :, None]).reshape(n, 448)
    return torch.cat([nope, rope], dim=1)                    # [n,512]


def _dequant_latents_batched(cache_flat, idx, bs):
    """cache_flat [nb, bs*584] uint8; idx [B,T] int global slots (-1 invalid ->0, mask later).
    Returns [B,T,512] float32. Vectorized + graph-safe (no host sync)."""
    B, T = idx.shape
    dev = cache_flat.device
    idxc = idx.clamp(min=0).long()
    blk = idxc // bs
    pos = idxc % bs
    data_col = (pos * 576).unsqueeze(-1) + torch.arange(576, device=dev)     # [B,T,576]
    scale_col = (bs * 576 + pos * 8).unsqueeze(-1) + torch.arange(8, device=dev)  # [B,T,8]
    data = cache_flat[blk.unsqueeze(-1), data_col]          # [B,T,576] uint8
    scl = cache_flat[blk.unsqueeze(-1), scale_col]          # [B,T,8] uint8
    nope_fp8 = data[..., :448].contiguous().view(torch.float8_e4m3fn).float()  # [B,T,448]
    rope = data[..., 448:576].contiguous().view(torch.bfloat16).float()       # [B,T,64]
    scales = torch.exp2(scl[..., :7].float() - 127.0)                         # [B,T,7]
    nope = (nope_fp8.view(B, T, 7, 64) * scales[..., None]).reshape(B, T, 448)
    return torch.cat([nope, rope], dim=-1)                  # [B,T,512]


def _attn_sink(qf, K, scale, sink_h):
    """qf [H,512] f32, K [nk,512] f32, sink_h [H] f32 -> out [H,512] f32."""
    logits = (qf @ K.t()) * scale                            # [H,nk]
    m = torch.maximum(logits.amax(dim=-1), sink_h)           # [H]
    p = torch.exp(logits - m[:, None])                       # [H,nk]
    den = p.sum(-1) + torch.exp(sink_h - m)                  # [H]
    return (p @ K) / den.clamp_min(1e-30)[:, None]


def flash_mla_with_kvcache(q, k_cache, block_table, head_dim_v, tile_scheduler_metadata,
                           cache_seqlens, is_fp8_kvcache, indices, topk_length,
                           softmax_scale, attn_sink, extra_k_cache=None,
                           extra_indices_in_kvcache=None, extra_topk_length=None,
                           out=None, **kw):
    """DECODE. q [B,1,H,512] bf16; k_cache (SWA) [nb,bs,1,584] uint8; indices (SWA)
    [B,1,W] int32 global slots; extra_k_cache (compressed) + extra_indices/topk; out [B,1,H,512]."""
    B, S, H, D = q.shape                                      # S == 1 (one query/token)
    sink = attn_sink.float()[:H]
    swa_flat = k_cache.reshape(k_cache.shape[0], -1)
    bs_swa = k_cache.shape[1]
    swa_idx = indices.reshape(B, -1)                          # [B,W]
    W = swa_idx.shape[1]
    ar = torch.arange(W, device=q.device)
    swa_valid = (ar[None, :] < topk_length[:, None]) & (swa_idx >= 0) if topk_length is not None \
        else (swa_idx >= 0)
    Ksw = _dequant_latents_batched(swa_flat, swa_idx, bs_swa)  # [B,W,512]
    Ks = [Ksw]
    valids = [swa_valid]
    if extra_k_cache is not None and extra_topk_length is not None:
        ex_flat = extra_k_cache.reshape(extra_k_cache.shape[0], -1)
        bs_ex = extra_k_cache.shape[1]
        ex_idx = extra_indices_in_kvcache.reshape(B, -1)      # [B,T]
        T = ex_idx.shape[1]
        art = torch.arange(T, device=q.device)
        ex_valid = (art[None, :] < extra_topk_length[:, None]) & (ex_idx >= 0)
        Ks.append(_dequant_latents_batched(ex_flat, ex_idx, bs_ex))
        valids.append(ex_valid)
    K = torch.cat(Ks, dim=1)                                  # [B,Nkv,512]
    valid = torch.cat(valids, dim=1)                          # [B,Nkv]
    qf = q.reshape(B, H, D).float()
    logits = torch.einsum("bhd,bnd->bhn", qf, K) * softmax_scale     # [B,H,Nkv]
    logits = logits.masked_fill(~valid[:, None, :], _NEG)
    m = torch.maximum(logits.amax(-1), sink[None, :])        # [B,H]
    p = torch.exp(logits - m[..., None])
    den = p.sum(-1) + torch.exp(sink[None, :] - m)           # [B,H]
    o = torch.einsum("bhn,bnd->bhd", p, K) / den.clamp_min(1e-30)[..., None]
    out.reshape(B, H, D).copy_(o.to(out.dtype))
    return out, None


def flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink, topk_length, out=None, **kw):
    """PREFILL. q [sq,H,512] bf16; kv [chunk*M,1,512] bf16 (dequantized workspace);
    indices [sq,1,T] int32 (offsets into kv); topk_length [sq]; out [sq,H,512]."""
    sq, H, D = q.shape
    kv2 = kv.reshape(-1, D).float()                          # [chunk*M,512]
    idx = indices.reshape(sq, -1)                            # [sq,T]
    T = idx.shape[1]
    sink = attn_sink.float()[:H]
    ar = torch.arange(T, device=q.device)
    valid = (ar[None, :] < topk_length[:, None]) & (idx >= 0)   # [sq,T]
    CH = 128
    for s0 in range(0, sq, CH):
        s1 = min(s0 + CH, sq)
        qb = q[s0:s1].float()                                # [c,H,512]
        ib = idx[s0:s1].clamp(min=0).long()                  # [c,T]
        vb = valid[s0:s1]                                    # [c,T]
        K = kv2[ib]                                          # [c,T,512]
        logits = torch.einsum("chd,ctd->cht", qb, K) * sm_scale
        logits = logits.masked_fill(~vb[:, None, :], _NEG)
        m = torch.maximum(logits.amax(-1), sink[None, :])    # [c,H]
        p = torch.exp(logits - m[..., None])
        den = p.sum(-1) + torch.exp(sink[None, :] - m)       # [c,H]
        o = torch.einsum("cht,ctd->chd", p, K) / den.clamp_min(1e-30)[..., None]
        out[s0:s1].copy_(o.to(out.dtype))
    return out, None, None
