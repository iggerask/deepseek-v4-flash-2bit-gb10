"""vq2: W2A16 vector-quantized GEMV kernel (Triton) for the 2-bit-experts combo.

Format (group-Hadamard incoherence ON):
  Wg = W grouped [out, ngroup, group]; Wrot = Wg @ H   (H = normalized Hadamard, sym)
  scale[o,g] = amax(Wrot[o,g,:]);  codes = kmeans over (Wrot/scale) vdim-subvectors
  recon_rot = C[idx]*scale ;  W = recon_rot @ H^T
Serving GEMV:  y = W x = recon_rot @ (H^T x).  So transform the ACTIVATION once
(x' = H^T x per group, H symmetric -> x'=x@H per group) and GEMV in rotated space.

This file: pack_vq2 (export codebook/indices/scales), had_act (activation transform),
vq2_gemv (Triton kernel), and a correctness+MBU self-test vs the dequant reference.
"""
import os, torch, triton, triton.language as tl

DEV = "cuda"


def _hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / (n ** 0.5))


_HCACHE = {}


def _Hgpu(group, device, dtype):
    """Cached Hadamard on the target device -- avoids a CPU->CUDA copy each call
    (which is illegal inside CUDA-graph capture)."""
    key = (group, str(device), dtype)
    if key not in _HCACHE:
        _HCACHE[key] = _hadamard(group).to(device=device, dtype=dtype)
    return _HCACHE[key]


def pack_vq2(W, vdim=4, k=1024, group=64, iters=12):
    """Quantize W [out,in] -> (C[k,vdim], idx[out,nsub] int32, scale[out,ngroup],
    dq[out,in] reference). Group-Hadamard incoherence on."""
    W = W.float().to(DEV)
    out, inn = W.shape
    pad = (-inn) % group
    if pad:
        W = torch.cat([W, torch.zeros(out, pad, device=DEV)], 1)
    ng = W.shape[1] // group
    H = _hadamard(group).to(DEV)
    Wg = W.reshape(out, ng, group) @ H                       # rotated
    scale = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)  # [out,ng,1]
    Wn = (Wg / scale).reshape(-1, vdim)
    torch.manual_seed(0)
    samp = Wn[torch.randint(0, Wn.shape[0], (min(Wn.shape[0], 200000),), device=DEV)]
    C = samp[torch.randperm(samp.shape[0], device=DEV)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(samp, C).argmin(1)
        sums = torch.zeros_like(C).index_add_(0, a, samp)
        cnts = torch.zeros(k, device=DEV).index_add_(0, a, torch.ones(a.shape[0], device=DEV))
        nz = cnts > 0
        C[nz] = sums[nz] / cnts[nz, None]
    idx = torch.empty(Wn.shape[0], dtype=torch.long, device=DEV)
    CH = 1 << 20
    for i in range(0, Wn.shape[0], CH):
        idx[i:i + CH] = torch.cdist(Wn[i:i + CH], C).argmin(1)
    recon_rot = (C[idx].reshape(out, ng, group)) * scale
    dq = (recon_rot @ H.t()).reshape(out, -1)[:, :inn]
    nsub = ng * (group // vdim)
    # Index width = the dominant memory traffic. uint8 (k<=256) -> 8 bits / VDIM weights
    # = 2 bits/weight (HALF of NVFP4 4-bit!). int16 (k<=1024) is 4 bits/weight = SAME as
    # NVFP4 -> throws away the 2-bit advantage. Use the narrowest type that fits k.
    dt = torch.uint8 if k <= 256 else torch.int16
    assert k <= 32768
    return (C.contiguous(), idx.to(dt).reshape(out, nsub).contiguous(),
            scale.squeeze(-1).contiguous(), dq.contiguous(), inn, pad)


def had_act(x, group=64):
    """x' = H^T x per group (H symmetric -> x@H per group). x [in] -> x' [padded_in]."""
    inn = x.shape[0]
    pad = (-inn) % group
    if pad:
        x = torch.cat([x, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    H = _Hgpu(group, x.device, x.dtype)
    return (x.reshape(-1, group) @ H).reshape(-1)


@triton.autotune(
    configs=[triton.Config({'BLOCK_O': bo, 'NG_BLK': nb}, num_warps=w, num_stages=s)
             for bo in (16, 32, 64) for nb in (2, 4, 8) for w in (2, 4, 8) for s in (2, 3, 4)],
    key=['OUT', 'NGROUP'])
@triton.jit
def _vq2_gemv(x_ptr, c_ptr, idx_ptr, scale_ptr, y_ptr, OUT, NGROUP,
              BLOCK_O: tl.constexpr, VDIM: tl.constexpr, GROUP: tl.constexpr,
              SPG: tl.constexpr, NG_BLK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_O + tl.arange(0, BLOCK_O)
    rmask = rows < OUT
    NSUB = NGROUP * SPG
    KB: tl.constexpr = NG_BLK * SPG                         # subvectors per inner step
    sk = tl.arange(0, KB)
    vd = tl.arange(0, VDIM)
    gb = tl.arange(0, NG_BLK)
    acc = tl.zeros([BLOCK_O], dtype=tl.float32)
    for g0 in range(0, NGROUP, NG_BLK):
        base = g0 * SPG
        ioff = rows[:, None] * NSUB + (base + sk)[None, :]
        ids = tl.load(idx_ptr + ioff, mask=rmask[:, None], other=0).to(tl.int32)  # [BLOCK_O,KB]
        cv = tl.load(c_ptr + ids[:, :, None] * VDIM + vd[None, None, :])           # [BLOCK_O,KB,VDIM]
        xv = tl.load(x_ptr + base * VDIM + sk[:, None] * VDIM + vd[None, :])       # [KB,VDIM]
        prod = tl.sum(cv * xv[None, :, :], axis=2)                                 # [BLOCK_O,KB]
        # per-group scale: reshape KB -> [NG_BLK, SPG], sum within group, weight by scale
        prod = tl.reshape(prod, [BLOCK_O, NG_BLK, SPG])
        gsum = tl.sum(prod, axis=2)                                               # [BLOCK_O,NG_BLK]
        sc = tl.load(scale_ptr + rows[:, None] * NGROUP + (g0 + gb)[None, :],
                     mask=rmask[:, None], other=0.0)                              # [BLOCK_O,NG_BLK]
        acc += tl.sum(gsum * sc, axis=1)
    tl.store(y_ptr + rows, acc, mask=rmask)


def vq2_gemv(xp, C, idx, scale, group=64):
    out, nsub = idx.shape
    ng = scale.shape[1]
    vdim = C.shape[1]
    y = torch.empty(out, device=xp.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(out, meta['BLOCK_O']),)
    _vq2_gemv[grid](xp, C, idx, scale, y, out, ng,
                    VDIM=vdim, GROUP=group, SPG=group // vdim)
    return y


_FLUSH = torch.empty(96 * 1024 * 1024, dtype=torch.float32, device=DEV)  # >L2, to cold the cache


def _bench(out, inn, group=64, vdim=4, k=1024):
    torch.manual_seed(0)
    W = (torch.randn(out, inn) * 0.05).to(DEV)
    C, idx, scale, dq, _, _ = pack_vq2(W, vdim, k, group, iters=6)
    x = torch.randn(inn, device=DEV)
    ref = dq.float() @ x
    xp = had_act(x, group)
    rel = ((vq2_gemv(xp, C, idx, scale, group) - ref).norm() / ref.norm()).item()
    for _ in range(5):
        vq2_gemv(xp, C, idx, scale, group)
    torch.cuda.synchronize()
    # COLD timing: flush L2 before each call, time only the kernel via CUDA events
    N = 100
    times = []
    for _ in range(N):
        _FLUSH.zero_()                                       # evict L2
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); vq2_gemv(xp, C, idx, scale, group); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1e3)                # ms->s
    times.sort()
    dt = sum(times[:N // 2]) / (N // 2)                      # median-ish (lower half)
    actual = idx.numel() * 2 + scale.numel() * 4 + C.numel() * 4
    mbu = 100 * actual / dt / 1e9 / 273
    print(f"  [{out}x{inn}] relerr={rel:.1e} | {dt*1e6:6.1f} us COLD | {actual/dt/1e9:5.0f} GB/s "
          f"({mbu:.0f}% MBU) | cfg {dict(_vq2_gemv.best_config.kwargs)} w{_vq2_gemv.best_config.num_warps}")


def pack_vq2_experts(W3, vdim=4, k=1024, group=64, iters=12):
    """Pack a fused expert tensor [E, R, D] with ONE shared codebook over all experts
    (reshape [E*R, D]) -- ONE shared codebook over all experts in the tensor.
    Returns C[k,vdim], idx[E,R,nsub] int16, scale[E,R,ng], dq[E,R,D] reference."""
    E, R, D = W3.shape
    C, idx, scale, dq, _, _ = pack_vq2(W3.reshape(E * R, D), vdim, k, group, iters)
    nsub, ng = idx.shape[1], scale.shape[1]
    return C, idx.reshape(E, R, nsub), scale.reshape(E, R, ng), dq.reshape(E, R, D)


def vq2_moe(x, topk_ids, topk_w, Cgu, idx_gu, sc_gu, Cd, idx_d, sc_d, group=64):
    """MoE apply core: x[N,H] routed to top-k experts -> [N,H]. gate_up [E,2I,H] +
    SiLU-gate + down [E,H,I], shared per-layer codebooks. Correct-first (per token-
    expert vq2 calls); the batched grouped-GEMM is the perf follow-on."""
    N, H = x.shape
    twoI = idx_gu.shape[1]; I = twoI // 2
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    for n in range(N):
        xh = had_act(x[n].float(), group)                      # transform once for gate_up
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[n, j]); w = float(topk_w[n, j])
            gu = vq2_gemv(xh, Cgu, idx_gu[e], sc_gu[e], group)  # [2I]
            act = torch.nn.functional.silu(gu[:I]) * gu[I:]     # [I]
            d = vq2_gemv(had_act(act, group), Cd, idx_d[e], sc_d[e], group)  # [H]
            out[n] += w * d
    return out


def had_act_batch(X, group=64):
    """Hadamard per group over the last dim: X[..., in] -> [..., in_pad]."""
    *b, inn = X.shape
    pad = (-inn) % group
    if pad:
        X = torch.cat([X, torch.zeros(*b, pad, device=X.device, dtype=X.dtype)], -1)
    H = _Hgpu(group, X.device, X.dtype)
    return (X.reshape(*b, -1, group) @ H).reshape(*b, -1)


@triton.autotune(
    configs=[triton.Config({'BLOCK_O': bo, 'NG_BLK': nb}, num_warps=w, num_stages=s)
             for bo in (16, 32, 64) for nb in (2, 4, 8) for w in (2, 4) for s in (2, 3)],
    key=['OUT', 'NGROUP'])
@triton.jit
def _vq2_bmm(vb_ptr, eids_ptr, c_ptr, idx_ptr, scale_ptr, out_ptr,
            M, OUT, NGROUP, IN_PAD,
            BLOCK_O: tl.constexpr, VDIM: tl.constexpr, GROUP: tl.constexpr,
            SPG: tl.constexpr, NG_BLK: tl.constexpr):
    """Batched expert GEMV: out[m, :] = expert[eids[m]] @ vb[m]. idx/scale are
    [E,OUT,*]; the expert id selects the slice on-device (CUDA-graph safe)."""
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)
    e = tl.load(eids_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    rmask = rows < OUT
    NSUB = NGROUP * SPG
    KB: tl.constexpr = NG_BLK * SPG
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    idx_base = e * OUT * NSUB + rows[:, None] * NSUB
    sc_base = e * OUT * NGROUP + rows[:, None] * NGROUP
    vb_base = pid_m * IN_PAD
    acc = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(0, NGROUP, NG_BLK):
        base = g0 * SPG
        ids = tl.load(idx_ptr + idx_base + (base + sk)[None, :],
                      mask=rmask[:, None], other=0).to(tl.int32)
        cv = tl.load(c_ptr + ids[:, :, None] * VDIM + vd[None, None, :])
        xv = tl.load(vb_ptr + vb_base + base * VDIM + sk[:, None] * VDIM + vd[None, :])
        prod = tl.sum(cv * xv[None, :, :], axis=2)
        prod = tl.reshape(prod, [BLOCK_O, NG_BLK, SPG])
        gsum = tl.sum(prod, axis=2)
        sc = tl.load(scale_ptr + sc_base + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0)
        acc += tl.sum(gsum * sc, axis=1)
    tl.store(out_ptr + pid_m * OUT + rows, acc, mask=rmask)


def vq2_batched_gemv(vb, eids, C, idx, scale, group=64):
    M, IN_PAD = vb.shape
    E, OUT, nsub = idx.shape
    ng, vdim = scale.shape[2], C.shape[1]
    out = torch.empty(M, OUT, device=vb.device, dtype=torch.float32)
    grid = lambda meta: (M, triton.cdiv(OUT, meta['BLOCK_O']))
    _vq2_bmm[grid](vb, eids, C, idx, scale, out, M, OUT, ng, IN_PAD,
                   VDIM=vdim, GROUP=group, SPG=group // vdim)
    return out


def vq2_moe_fast(x, topk_ids, topk_w, Cgu, igu, sgu, Cd, idn, sdn, group=64):
    """Graph-safe batched MoE: x[N,H] -> [N,H]. 2-pass (gate_up+SiLU, down+scatter),
    no host syncs. eids/routing read on-device. ~one kernel launch per pass."""
    N, H = x.shape
    k = topk_ids.shape[1]
    M = N * k
    twoI = igu.shape[1]; I = twoI // 2
    eids = topk_ids.reshape(M).to(torch.int32)
    n_ids = torch.arange(N, device=x.device).repeat_interleave(k)
    vb_gu = had_act_batch(x.float(), group)[n_ids]                 # [M, Hpad]
    gu = vq2_batched_gemv(vb_gu, eids, Cgu, igu, sgu, group)       # [M, 2I]
    inter = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]        # [M, I]
    dn = vq2_batched_gemv(had_act_batch(inter, group), eids, Cd, idn, sdn, group)  # [M, H]
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    out.index_add_(0, n_ids, topk_w.reshape(M, 1).float() * dn)
    return out


_FCFG = [triton.Config({'BLOCK_O': bo, 'NG_BLK': nb}, num_warps=w, num_stages=s)
         for bo in (4, 8, 16, 32) for nb in (2, 4, 8) for w in (1, 2, 4) for s in (2, 3)]


@triton.autotune(configs=_FCFG, key=['M', 'I_OUT', 'NGROUP'])
@triton.jit
def _vq2_gateup_silu(xh_ptr, nids_ptr, eids_ptr, c_ptr, idx_ptr, scale_ptr, out_ptr,
                     M, I_OUT, NGROUP, IN_PAD, BLOCK_O: tl.constexpr, VDIM: tl.constexpr,
                     GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr):
    """Fused: inter[m,i] = silu(gate_e[i]·xh[n]) * (up_e[I+i]·xh[n]). Reads token act
    inline (no gather), combines gate/up in-kernel (no 2I materialization, no silu op)."""
    pid_m = tl.program_id(0); pid_o = tl.program_id(1)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < I_OUT
    NSUB = NGROUP * SPG; KB: tl.constexpr = NG_BLK * SPG; TWO_I = 2 * I_OUT
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    bg = e * TWO_I * NSUB + rows[:, None] * NSUB
    bu = e * TWO_I * NSUB + (rows + I_OUT)[:, None] * NSUB
    sg = e * TWO_I * NGROUP + rows[:, None] * NGROUP
    su = e * TWO_I * NGROUP + (rows + I_OUT)[:, None] * NGROUP
    xb = n * IN_PAD
    accg = tl.zeros([BLOCK_O], tl.float32); accu = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(0, NGROUP, NG_BLK):
        base = g0 * SPG
        xv = tl.load(xh_ptr + xb + base * VDIM + sk[:, None] * VDIM + vd[None, :])
        idg = tl.load(idx_ptr + bg + (base + sk)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        pg = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + idg[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        accg += tl.sum(pg * tl.load(scale_ptr + sg + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
        idu = tl.load(idx_ptr + bu + (base + sk)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        pu = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + idu[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        accu += tl.sum(pu * tl.load(scale_ptr + su + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    silu = accg * (1.0 / (1.0 + tl.exp(-accg)))
    tl.store(out_ptr + pid_m * I_OUT + rows, silu * accu, mask=rmask)


_DCFG = [triton.Config({'BLOCK_O': bo, 'NG_BLK': nb, 'SPLIT_K': sp}, num_warps=w, num_stages=s)
         for bo in (16, 32, 64) for nb in (2, 4) for sp in (1, 4, 8) for w in (2, 4) for s in (2, 3)]


# reset_to_zero: the kernel atomic-adds into out_ptr; the autotuner re-runs it to time
# configs, so out_ptr must be zeroed before each timing run (else it accumulates -> garbage).
# SPLIT_K: at batch-1 decode the M*H_tiles grid is tiny (occupancy-starved -> the codebook
# gather latency isn't hidden, ~4% MBU). Split the group reduction SPLIT_K ways -> SPLIT_K x
# more programs, each atomic-adds its partial. Recovers occupancy at decode.
@triton.autotune(configs=_DCFG, key=['M', 'H_OUT', 'NGROUP'], reset_to_zero=['out_ptr'])
@triton.jit
def _vq2_down_scatter(vb_ptr, nids_ptr, eids_ptr, w_ptr, c_ptr, idx_ptr, scale_ptr, out_ptr,
                      M, H_OUT, NGROUP, IN_PAD, BLOCK_O: tl.constexpr, VDIM: tl.constexpr,
                      GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr,
                      SPLIT_K: tl.constexpr):
    """Fused: out[n,:] += w[m] * (down_e @ vb[m]) via atomic add (no index_add op).
    SPLIT_K programs split the group reduction for decode occupancy."""
    pid_m = tl.program_id(0); pid_o = tl.program_id(1); pid_s = tl.program_id(2)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m); w = tl.load(w_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < H_OUT
    NSUB = NGROUP * SPG; KB: tl.constexpr = NG_BLK * SPG
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    b = e * H_OUT * NSUB + rows[:, None] * NSUB
    sb = e * H_OUT * NGROUP + rows[:, None] * NGROUP
    xb = pid_m * IN_PAD
    acc = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(pid_s * NG_BLK, NGROUP, SPLIT_K * NG_BLK):
        base = g0 * SPG
        xv = tl.load(vb_ptr + xb + base * VDIM + sk[:, None] * VDIM + vd[None, :])
        ids = tl.load(idx_ptr + b + (base + sk)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        p = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + ids[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        acc += tl.sum(p * tl.load(scale_ptr + sb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    tl.atomic_add(out_ptr + n * H_OUT + rows, w * acc, mask=rmask)


def vq2_moe_fused(x, topk_ids, topk_w, Cgu, igu, sgu, Cd, idn, sdn, group=64):
    """Fused graph-safe MoE: 2 kernels/layer (gate_up+silu, down+scatter) + 2 small
    Hadamard matmuls. No gather/silu/index_add launches, no [M,2I]/[M,H] intermediates."""
    N, H = x.shape; k = topk_ids.shape[1]; M = N * k
    I = igu.shape[1] // 2
    eids = topk_ids.reshape(M).to(torch.int32)
    nids = torch.arange(N, device=x.device).repeat_interleave(k).to(torch.int32)
    w = topk_w.reshape(M).float()
    xh = had_act_batch(x.float(), group)                       # [N, Hpad]
    inter = torch.empty(M, I, device=x.device, dtype=torch.float32)
    grid1 = lambda meta: (M, triton.cdiv(I, meta['BLOCK_O']))
    _vq2_gateup_silu[grid1](xh, nids, eids, Cgu, igu, sgu, inter, M, I, sgu.shape[2], xh.shape[1],
                            VDIM=Cgu.shape[1], GROUP=group, SPG=group // Cgu.shape[1])
    vb_dn = had_act_batch(inter, group)                        # [M, Ipad]
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    grid2 = lambda meta: (M, triton.cdiv(H, meta['BLOCK_O']), meta['SPLIT_K'])
    _vq2_down_scatter[grid2](vb_dn, nids, eids, w, Cd, idn, sdn, out, M, H, sdn.shape[2], vb_dn.shape[1],
                             VDIM=Cd.shape[1], GROUP=group, SPG=group // Cd.shape[1])
    return out


@triton.autotune(configs=_FCFG, key=['M', 'I_OUT', 'NGROUP'])
@triton.jit
def _vq2_gateup_lut(lut_ptr, nids_ptr, eids_ptr, idx_ptr, sg_ptr, su_ptr, out_ptr,
                    M, I_OUT, NGROUP, K, BLOCK_O: tl.constexpr, GROUP: tl.constexpr,
                    SPG: tl.constexpr, NG_BLK: tl.constexpr):
    """LUT-decode gate_up+silu. lut[n,j,sub]=Cgu[j]·xh[n,sub] precomputed; each weight is
    one LUT lookup (no per-weight codebook gather + VDIM MACs)."""
    pid_m = tl.program_id(0); pid_o = tl.program_id(1)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < I_OUT
    NSUB = NGROUP * SPG; KB: tl.constexpr = NG_BLK * SPG; TWO_I = 2 * I_OUT
    sk = tl.arange(0, KB); gb = tl.arange(0, NG_BLK)
    lb = n * K * NSUB
    bg = e * TWO_I * NSUB + rows[:, None] * NSUB
    bu = e * TWO_I * NSUB + (rows + I_OUT)[:, None] * NSUB
    sgb = e * TWO_I * NGROUP + rows[:, None] * NGROUP
    sub = e * TWO_I * NGROUP + (rows + I_OUT)[:, None] * NGROUP
    accg = tl.zeros([BLOCK_O], tl.float32); accu = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(0, NGROUP, NG_BLK):
        subs = (g0 * SPG + sk)[None, :]
        idg = tl.load(idx_ptr + bg + subs, mask=rmask[:, None], other=0).to(tl.int32)
        lg = tl.load(lut_ptr + lb + idg * NSUB + subs)
        lg = tl.sum(tl.reshape(lg, [BLOCK_O, NG_BLK, SPG]), 2)
        accg += tl.sum(lg * tl.load(sg_ptr + sgb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
        idu = tl.load(idx_ptr + bu + subs, mask=rmask[:, None], other=0).to(tl.int32)
        lu = tl.load(lut_ptr + lb + idu * NSUB + subs)
        lu = tl.sum(tl.reshape(lu, [BLOCK_O, NG_BLK, SPG]), 2)
        accu += tl.sum(lu * tl.load(su_ptr + sub + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    silu = accg * (1.0 / (1.0 + tl.exp(-accg)))
    tl.store(out_ptr + pid_m * I_OUT + rows, silu * accu, mask=rmask)


@triton.autotune(configs=_DCFG, key=['M', 'H_OUT', 'NGROUP'], reset_to_zero=['out_ptr'])
@triton.jit
def _vq2_down_lut(lut_ptr, nids_ptr, eids_ptr, w_ptr, idx_ptr, scale_ptr, out_ptr,
                  M, H_OUT, NGROUP, K, BLOCK_O: tl.constexpr, GROUP: tl.constexpr,
                  SPG: tl.constexpr, NG_BLK: tl.constexpr, SPLIT_K: tl.constexpr):
    """LUT-decode down+scatter. lut is per-pair (input differs per pair): lut[m,j,sub]."""
    pid_m = tl.program_id(0); pid_o = tl.program_id(1); pid_s = tl.program_id(2)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m); w = tl.load(w_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < H_OUT
    NSUB = NGROUP * SPG; KB: tl.constexpr = NG_BLK * SPG
    sk = tl.arange(0, KB); gb = tl.arange(0, NG_BLK)
    lb = pid_m * K * NSUB
    b = e * H_OUT * NSUB + rows[:, None] * NSUB
    sb = e * H_OUT * NGROUP + rows[:, None] * NGROUP
    acc = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(pid_s * NG_BLK, NGROUP, SPLIT_K * NG_BLK):
        subs = (g0 * SPG + sk)[None, :]
        ids = tl.load(idx_ptr + b + subs, mask=rmask[:, None], other=0).to(tl.int32)
        lv = tl.load(lut_ptr + lb + ids * NSUB + subs)
        lv = tl.sum(tl.reshape(lv, [BLOCK_O, NG_BLK, SPG]), 2)
        acc += tl.sum(lv * tl.load(scale_ptr + sb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    tl.atomic_add(out_ptr + n * H_OUT + rows, w * acc, mask=rmask)


def vq2_moe_lut(x, topk_ids, topk_w, Cgu, igu, sgu, Cd, idn, sdn, group=64):
    """LUT-decode MoE: precompute codebook(x)-dot LUTs (shared across rows/experts),
    then each weight is one table lookup. Targets the gather+MAC compute bottleneck."""
    N, H = x.shape; k = topk_ids.shape[1]; M = N * k
    I = igu.shape[1] // 2; vdim = Cgu.shape[1]; Kc = Cgu.shape[0]
    eids = topk_ids.reshape(M).to(torch.int32)
    nids = torch.arange(N, device=x.device).repeat_interleave(k).to(torch.int32)
    wf = topk_w.reshape(M).float()
    xh = had_act_batch(x.float(), group)                                   # [N, Hpad]
    nsg = xh.shape[1] // vdim
    lut_gu = torch.einsum('jv,nsv->njs', Cgu.float(), xh.reshape(N, nsg, vdim)).contiguous()  # [N,K,nsg]
    inter = torch.empty(M, I, device=x.device, dtype=torch.float32)
    g1 = lambda meta: (M, triton.cdiv(I, meta['BLOCK_O']))
    _vq2_gateup_lut[g1](lut_gu, nids, eids, igu, sgu, sgu, inter, M, I, sgu.shape[2], Kc,
                        GROUP=group, SPG=group // vdim)
    vb = had_act_batch(inter, group)                                       # [M, Ipad]
    nsd = vb.shape[1] // vdim
    lut_dn = torch.einsum('jv,msv->mjs', Cd.float(), vb.reshape(M, nsd, vdim)).contiguous()  # [M,K,nsd]
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    g2 = lambda meta: (M, triton.cdiv(H, meta['BLOCK_O']), meta['SPLIT_K'])
    _vq2_down_lut[g2](lut_dn, nids, eids, wf, idn, sdn, out, M, H, sdn.shape[2], Kc,
                      GROUP=group, SPG=group // vdim)
    return out


@triton.autotune(configs=_DCFG, key=['M', 'I_OUT', 'NGROUP'], reset_to_zero=['g_ptr', 'u_ptr'])
@triton.jit
def _vq2_gateup_sk(xh_ptr, nids_ptr, eids_ptr, c_ptr, idx_ptr, sg_ptr, su_ptr, g_ptr, u_ptr,
                   M, I_OUT, NGROUP, IN_PAD, BLOCK_O: tl.constexpr, VDIM: tl.constexpr,
                   GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr,
                   SPLIT_K: tl.constexpr):
    """Split-K gate_up (codebook gather): partial gate/up -> atomic into g/u scratch
    [M,I]; SiLU-combine done in torch. SPLIT_K x more programs -> hides gather latency."""
    pid_m = tl.program_id(0); pid_o = tl.program_id(1); pid_s = tl.program_id(2)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < I_OUT
    NSUB = NGROUP * SPG; KB: tl.constexpr = NG_BLK * SPG; TWO_I = 2 * I_OUT
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    bg = e * TWO_I * NSUB + rows[:, None] * NSUB
    bu = e * TWO_I * NSUB + (rows + I_OUT)[:, None] * NSUB
    sgb = e * TWO_I * NGROUP + rows[:, None] * NGROUP
    sub = e * TWO_I * NGROUP + (rows + I_OUT)[:, None] * NGROUP
    xb = n * IN_PAD
    accg = tl.zeros([BLOCK_O], tl.float32); accu = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(pid_s * NG_BLK, NGROUP, SPLIT_K * NG_BLK):
        base = g0 * SPG
        xv = tl.load(xh_ptr + xb + base * VDIM + sk[:, None] * VDIM + vd[None, :])
        idg = tl.load(idx_ptr + bg + (base + sk)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        pg = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + idg[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        accg += tl.sum(pg * tl.load(sg_ptr + sgb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
        idu = tl.load(idx_ptr + bu + (base + sk)[None, :], mask=rmask[:, None], other=0).to(tl.int32)
        pu = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + idu[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        accu += tl.sum(pu * tl.load(su_ptr + sub + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    tl.atomic_add(g_ptr + pid_m * I_OUT + rows, accg, mask=rmask)
    tl.atomic_add(u_ptr + pid_m * I_OUT + rows, accu, mask=rmask)


def vq2_moe_skg(x, topk_ids, topk_w, Cgu, igu, sgu, Cd, idn, sdn, group=64):
    """Split-K gate_up + split-K down. Codebook-gather decode, occupancy-tuned."""
    N, H = x.shape; k = topk_ids.shape[1]; M = N * k
    I = igu.shape[1] // 2; vdim = Cgu.shape[1]
    eids = topk_ids.reshape(M).to(torch.int32)
    nids = torch.arange(N, device=x.device).repeat_interleave(k).to(torch.int32)
    wf = topk_w.reshape(M).float()
    xh = had_act_batch(x.float(), group)
    gacc = torch.zeros(M, I, device=x.device, dtype=torch.float32)
    uacc = torch.zeros(M, I, device=x.device, dtype=torch.float32)
    g1 = lambda meta: (M, triton.cdiv(I, meta['BLOCK_O']), meta['SPLIT_K'])
    _vq2_gateup_sk[g1](xh, nids, eids, Cgu, igu, sgu, sgu, gacc, uacc, M, I, sgu.shape[2], xh.shape[1],
                       VDIM=vdim, GROUP=group, SPG=group // vdim)
    inter = torch.nn.functional.silu(gacc) * uacc
    vb = had_act_batch(inter, group)
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    g2 = lambda meta: (M, triton.cdiv(H, meta['BLOCK_O']), meta['SPLIT_K'])
    _vq2_down_scatter[g2](vb, nids, eids, wf, Cd, idn, sdn, out, M, H, sdn.shape[2], vb.shape[1],
                          VDIM=vdim, GROUP=group, SPG=group // vdim)
    return out


# ===================== 10-bit byte-plane path (k=1024, 2.5 bit/weight) =====================
def _pack10(idx):
    """idx int [.., nsub] (0..1023) -> lo[.., nsub] uint8 + hi[.., nsub/4] uint8."""
    lo = (idx & 0xFF).to(torch.uint8)
    h = ((idx >> 8) & 0x3).to(torch.uint8)
    *b, ns = h.shape
    h = h.reshape(*b, ns // 4, 4)
    hp = (h[..., 0] | (h[..., 1] << 2) | (h[..., 2] << 4) | (h[..., 3] << 6)).to(torch.uint8)
    return lo.contiguous(), hp.contiguous()


def pack_vq2_experts_10b(W3, vdim=4, k=1024, group=64, iters=12):
    """Like pack_vq2_experts but returns 10-bit byte-planes (lo,hi) instead of int16 idx."""
    C, idx, scale, dq = pack_vq2_experts(W3, vdim, k, group, iters)
    lo, hi = _pack10(idx.int())
    return C, lo, hi, scale, dq


# Trimmed configs + M-INDEPENDENT key: tile config is M-independent (M is the grid dim),
# so tune ONCE per static (I_OUT/H_OUT, NGROUP) and reuse for ALL prompt lengths. Keeping M
# in the key re-triggered a ~170s sweep on every new prefill length -> model unusable.
_FCFG10 = [triton.Config({'BLOCK_O': bo, 'NG_BLK': nb}, num_warps=w, num_stages=3)
           for bo in (16, 32) for nb in (4, 8) for w in (1, 2)]
_DCFG10 = [triton.Config({'BLOCK_O': bo, 'NG_BLK': nb, 'SPLIT_K': sp}, num_warps=w, num_stages=2)
           for bo in (32, 64) for nb in (2, 4) for sp in (1, 4) for w in (2,)]


@triton.autotune(configs=_FCFG10, key=['I_OUT', 'NGROUP'])
@triton.jit
def _vq2_gateup_silu_10b(xh_ptr, nids_ptr, eids_ptr, c_ptr, lo_ptr, hi_ptr, sg_ptr, su_ptr, out_ptr,
                         M, I_OUT, NGROUP, IN_PAD, BLOCK_O: tl.constexpr, VDIM: tl.constexpr,
                         GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr):
    pid_m = tl.program_id(0); pid_o = tl.program_id(1)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < I_OUT
    NSUB = NGROUP * SPG; NSUB4 = NSUB // 4; KB: tl.constexpr = NG_BLK * SPG; TWO_I = 2 * I_OUT
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    lg = e * TWO_I * NSUB + rows[:, None] * NSUB; lu = e * TWO_I * NSUB + (rows + I_OUT)[:, None] * NSUB
    hg = e * TWO_I * NSUB4 + rows[:, None] * NSUB4; hu = e * TWO_I * NSUB4 + (rows + I_OUT)[:, None] * NSUB4
    sgb = e * TWO_I * NGROUP + rows[:, None] * NGROUP; subb = e * TWO_I * NGROUP + (rows + I_OUT)[:, None] * NGROUP
    accg = tl.zeros([BLOCK_O], tl.float32); accu = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(0, NGROUP, NG_BLK):
        base = g0 * SPG; subs = (base + sk)[None, :]; sh = ((base + sk) // 4)[None, :]; bp = (((base + sk) % 4) * 2)[None, :]
        xv = tl.load(xh_ptr + n * IN_PAD + base * VDIM + sk[:, None] * VDIM + vd[None, :])
        idg = (tl.load(lo_ptr + lg + subs, mask=rmask[:, None], other=0).to(tl.int32)
               | (((tl.load(hi_ptr + hg + sh, mask=rmask[:, None], other=0).to(tl.int32) >> bp) & 3) << 8))
        pg = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + idg[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        accg += tl.sum(pg * tl.load(sg_ptr + sgb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
        idu = (tl.load(lo_ptr + lu + subs, mask=rmask[:, None], other=0).to(tl.int32)
               | (((tl.load(hi_ptr + hu + sh, mask=rmask[:, None], other=0).to(tl.int32) >> bp) & 3) << 8))
        pu = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + idu[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        accu += tl.sum(pu * tl.load(su_ptr + subb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    silu = accg * (1.0 / (1.0 + tl.exp(-accg)))
    tl.store(out_ptr + pid_m * I_OUT + rows, silu * accu, mask=rmask)


@triton.autotune(configs=_DCFG10, key=['H_OUT', 'NGROUP'], reset_to_zero=['out_ptr'])
@triton.jit
def _vq2_down_scatter_10b(vb_ptr, nids_ptr, eids_ptr, w_ptr, c_ptr, lo_ptr, hi_ptr, scale_ptr, out_ptr,
                          M, H_OUT, NGROUP, IN_PAD, BLOCK_O: tl.constexpr, VDIM: tl.constexpr,
                          GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_o = tl.program_id(1); pid_s = tl.program_id(2)
    e = tl.load(eids_ptr + pid_m); n = tl.load(nids_ptr + pid_m); w = tl.load(w_ptr + pid_m)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < H_OUT
    NSUB = NGROUP * SPG; NSUB4 = NSUB // 4; KB: tl.constexpr = NG_BLK * SPG
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    lb = e * H_OUT * NSUB + rows[:, None] * NSUB; hbb = e * H_OUT * NSUB4 + rows[:, None] * NSUB4
    sb = e * H_OUT * NGROUP + rows[:, None] * NGROUP
    acc = tl.zeros([BLOCK_O], tl.float32)
    for g0 in range(pid_s * NG_BLK, NGROUP, SPLIT_K * NG_BLK):
        base = g0 * SPG; subs = (base + sk)[None, :]; sh = ((base + sk) // 4)[None, :]; bp = (((base + sk) % 4) * 2)[None, :]
        xv = tl.load(vb_ptr + pid_m * IN_PAD + base * VDIM + sk[:, None] * VDIM + vd[None, :])
        ids = (tl.load(lo_ptr + lb + subs, mask=rmask[:, None], other=0).to(tl.int32)
               | (((tl.load(hi_ptr + hbb + sh, mask=rmask[:, None], other=0).to(tl.int32) >> bp) & 3) << 8))
        p = tl.sum(tl.reshape(tl.sum(tl.load(c_ptr + ids[:, :, None] * VDIM + vd[None, None, :]) * xv[None, :, :], 2), [BLOCK_O, NG_BLK, SPG]), 2)
        acc += tl.sum(p * tl.load(scale_ptr + sb + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0), 1)
    tl.atomic_add(out_ptr + n * H_OUT + rows, w * acc, mask=rmask)


def vq2_moe_fused_10b(x, topk_ids, topk_w, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=64):
    """10-bit byte-plane MoE (k=1024 quality, 2.5 bit/weight)."""
    N, H = x.shape; k = topk_ids.shape[1]; M = N * k
    I = sgu.shape[1] // 2; vdim = Cgu.shape[1]
    eids = topk_ids.reshape(M).to(torch.int32)
    nids = torch.arange(N, device=x.device).repeat_interleave(k).to(torch.int32)
    wf = topk_w.reshape(M).float()
    # bf16 activation path: xh bf16 is LOSSLESS (x is already bf16 in the model) and halves the
    # gate_up input read + its Hadamard. inter bf16 (the gate_up OUTPUT) adds ~+0.2% PPL but halves
    # the down-input (vb) read + its Hadamard; default on (matches the accepted fp8-prefill tradeoff).
    _xdt = torch.float32 if os.environ.get("VQ2_BF16_ACT", "1") == "0" else torch.bfloat16
    _idt = torch.bfloat16 if os.environ.get("VQ2_BF16_INTER", "1") == "1" else torch.float32
    if os.environ.get("VQ2_BF16_ACT", "1") == "0": _idt = torch.float32
    # fp8 activation was tested (like prefill) and is NOT a decode win: the 2 per-row quant passes cost
    # more than the tiny batch-1 activation-read saving (decode 19.8 fp8 vs 19.99 bf16). bf16 is the win.
    xh = had_act_batch(x.to(_xdt), group)
    inter = torch.empty(M, I, device=x.device, dtype=_idt)
    g1 = lambda meta: (M, triton.cdiv(I, meta['BLOCK_O']))
    _vq2_gateup_silu_10b[g1](xh, nids, eids, Cgu, lgu, hgu, sgu, sgu, inter, M, I, sgu.shape[2], xh.shape[1],
                             VDIM=vdim, GROUP=group, SPG=group // vdim)
    vb = had_act_batch(inter, group)
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    g2 = lambda meta: (M, triton.cdiv(H, meta['BLOCK_O']), meta['SPLIT_K'])
    _vq2_down_scatter_10b[g2](vb, nids, eids, wf, Cd, ld, hd, sdn, out, M, H, sdn.shape[2], vb.shape[1],
                              VDIM=vdim, GROUP=group, SPG=group // vdim)
    return out


# ===================== group-by-expert (amortize expert read at batch) =====================
_GCFG = [triton.Config({'BLOCK_O': bo, 'NG_BLK': nb}, num_warps=w, num_stages=2)
         for bo in (16, 32, 64) for nb in (4, 8) for w in (2, 4)]


@triton.jit
def _grp_gemm_block(c_ptr, idx_ptr, scale_ptr, xs_ptr, e, rows, rmask, base_off, sbase,
                    XB, NGROUP, NSUB, BM: tl.constexpr, BLOCK_O: tl.constexpr, VDIM: tl.constexpr,
                    SPG: tl.constexpr, NG_BLK: tl.constexpr):
    """acc[BLOCK_O,BM] = expert(rows) @ xs[block]  -- weights gathered ONCE per block."""
    KB: tl.constexpr = NG_BLK * SPG
    sk = tl.arange(0, KB); vd = tl.arange(0, VDIM); gb = tl.arange(0, NG_BLK)
    bm = tl.arange(0, BM)
    acc = tl.zeros([BLOCK_O, BM], tl.float32)
    for g0 in range(0, NGROUP, NG_BLK):
        base = g0 * SPG; subs = (base + sk)[None, :]
        ids = tl.load(idx_ptr + base_off + subs, mask=rmask[:, None], other=0).to(tl.int32)  # [BLOCK_O,KB]
        cv = tl.load(c_ptr + ids[:, :, None] * VDIM + vd[None, None, :])                      # [BLOCK_O,KB,VDIM]
        sc = tl.load(scale_ptr + sbase + (g0 + gb)[None, :], mask=rmask[:, None], other=0.0)  # [BLOCK_O,NG_BLK]
        scx = tl.reshape(tl.broadcast_to(sc[:, :, None], [BLOCK_O, NG_BLK, SPG]), [BLOCK_O, KB])
        cvf = tl.reshape(cv, [BLOCK_O, KB * VDIM]) * tl.reshape(tl.broadcast_to(scx[:, :, None], [BLOCK_O, KB, VDIM]), [BLOCK_O, KB * VDIM])
        # xs[block] for these subs: [BM, KB*VDIM]
        xoff = XB[:, None] + base * VDIM + (sk[:, None] * VDIM + vd[None, :]).reshape(1, KB * VDIM)
        xv = tl.load(xs_ptr + xoff)                                                           # [BM, KB*VDIM]
        acc += tl.dot(cvf.to(tl.float16), xv.to(tl.float16).T, out_dtype=tl.float32)          # [BLOCK_O,BM]
    return acc


@triton.autotune(configs=_GCFG, key=['I_OUT', 'NGROUP'])
@triton.jit
def _vq2_grp_gateup(xs_ptr, be_ptr, c_ptr, idx_ptr, sg_ptr, out_ptr,
                    NBLK, I_OUT, NGROUP, IN_PAD, BM: tl.constexpr, BLOCK_O: tl.constexpr,
                    VDIM: tl.constexpr, GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr):
    pid_b = tl.program_id(0); pid_o = tl.program_id(1)
    e = tl.load(be_ptr + pid_b)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < I_OUT
    bm = tl.arange(0, BM)
    NSUB = NGROUP * SPG; TWO_I = 2 * I_OUT
    XB = (pid_b * BM + bm) * IN_PAD                                    # xs is per sorted slot
    bg = e * TWO_I * NSUB + rows[:, None] * NSUB
    bu = e * TWO_I * NSUB + (rows + I_OUT)[:, None] * NSUB
    sg = e * TWO_I * NGROUP + rows[:, None] * NGROUP
    su = e * TWO_I * NGROUP + (rows + I_OUT)[:, None] * NGROUP
    accg = _grp_gemm_block(c_ptr, idx_ptr, sg_ptr, xs_ptr, e, rows, rmask, bg, sg, XB, NGROUP, NSUB, BM, BLOCK_O, VDIM, SPG, NG_BLK)
    accu = _grp_gemm_block(c_ptr, idx_ptr, sg_ptr, xs_ptr, e, rows, rmask, bu, su, XB, NGROUP, NSUB, BM, BLOCK_O, VDIM, SPG, NG_BLK)
    inter = (accg * (1.0 / (1.0 + tl.exp(-accg)))) * accu             # [BLOCK_O, BM]
    slots = pid_b * BM + bm
    tl.store(out_ptr + slots[None, :] * I_OUT + rows[:, None], inter, mask=rmask[:, None])


@triton.autotune(configs=_GCFG, key=['H_OUT', 'NGROUP'], reset_to_zero=['out_ptr'])
@triton.jit
def _vq2_grp_down(vb_ptr, tok_ptr, valid_ptr, w_ptr, be_ptr, c_ptr, idx_ptr, scale_ptr, out_ptr,
                  NBLK, H_OUT, NGROUP, IN_PAD, BM: tl.constexpr, BLOCK_O: tl.constexpr,
                  VDIM: tl.constexpr, GROUP: tl.constexpr, SPG: tl.constexpr, NG_BLK: tl.constexpr):
    pid_b = tl.program_id(0); pid_o = tl.program_id(1)
    e = tl.load(be_ptr + pid_b)
    rows = pid_o * BLOCK_O + tl.arange(0, BLOCK_O); rmask = rows < H_OUT
    bm = tl.arange(0, BM); slots = pid_b * BM + bm
    toks = tl.load(tok_ptr + slots); val = tl.load(valid_ptr + slots); ws = tl.load(w_ptr + slots)
    NSUB = NGROUP * SPG
    base_off = e * H_OUT * NSUB + rows[:, None] * NSUB
    sbase = e * H_OUT * NGROUP + rows[:, None] * NGROUP
    XB = slots * IN_PAD
    acc = _grp_gemm_block(c_ptr, idx_ptr, scale_ptr, vb_ptr, e, rows, rmask, base_off, sbase, XB, NGROUP, NSUB, BM, BLOCK_O, VDIM, SPG, NG_BLK)
    contrib = acc * (ws * val.to(tl.float32))[None, :]                # [BLOCK_O, BM]; padding->0
    tl.atomic_add(out_ptr + toks[None, :] * H_OUT + rows[:, None], contrib,
                  mask=rmask[:, None] & (val[None, :] > 0))


def vq2_moe_grouped(x, topk_ids, topk_w, Cgu, igu, sgu, Cd, idn, sdn, group=64, BM=16):
    """Group-by-expert MoE: read each active expert's weights ONCE per block of BM
    tokens (tl.dot GEMM). Amortizes the expert read at batch."""
    from moe_align import moe_align
    N, H = x.shape; k = topk_ids.shape[1]
    E = igu.shape[0]; I = igu.shape[1] // 2; vdim = Cgu.shape[1]
    st, ss, val, be, nblk = moe_align(topk_ids, E, BM)
    Pmax = st.numel()
    xh = had_act_batch(x.float(), group)
    xs = xh[st.long()]                                                # [Pmax, Hpad]
    sw = topk_w[st.long(), ss.long()].float()                        # [Pmax]
    inter = torch.zeros(Pmax, I, device=x.device, dtype=torch.float32)
    g1 = lambda m: (nblk, triton.cdiv(I, m['BLOCK_O']))
    _vq2_grp_gateup[g1](xs, be, Cgu, igu, sgu, inter, nblk, I, sgu.shape[2], xs.shape[1],
                        BM=BM, VDIM=vdim, GROUP=group, SPG=group // vdim)
    vb = had_act_batch(inter, group)                                 # [Pmax, Ipad]
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    g2 = lambda m: (nblk, triton.cdiv(H, m['BLOCK_O']))
    _vq2_grp_down[g2](vb, st, val, sw, be, Cd, idn, sdn, out, nblk, H, sdn.shape[2], vb.shape[1],
                      BM=BM, VDIM=vdim, GROUP=group, SPG=group // vdim)
    return out


def _test_moe_fast():
    torch.manual_seed(0)
    E, I, H, k, group = 8, 512, 1024, 2, 64
    gu = (torch.randn(E, 2 * I, H) * 0.05).to(DEV)
    dn = (torch.randn(E, H, I) * 0.05).to(DEV)
    Cgu, igu, sgu, _ = pack_vq2_experts(gu, k=1024, group=group, iters=6)
    Cd, idn, sdn, _ = pack_vq2_experts(dn, k=1024, group=group, iters=6)
    N = 5
    x = torch.randn(N, H, device=DEV)
    ids = torch.randint(0, E, (N, k), device=DEV)
    w = torch.softmax(torch.randn(N, k, device=DEV), -1)
    ref = vq2_moe(x, ids, w, Cgu, igu, sgu, Cd, idn, sdn, group)
    fused = vq2_moe_fused(x, ids, w, Cgu, igu, sgu, Cd, idn, sdn, group)
    lut = vq2_moe_lut(x, ids, w, Cgu, igu, sgu, Cd, idn, sdn, group)
    print(f"vq2_moe_fused vs loop: relerr={((fused - ref).norm() / ref.norm()).item():.2e}")
    print(f"vq2_moe_lut   vs loop: relerr={((lut - ref).norm() / ref.norm()).item():.2e}")


def _test_moe():
    torch.manual_seed(0)
    E, I, H, k, group = 8, 512, 1024, 2, 64
    gu = (torch.randn(E, 2 * I, H) * 0.05).to(DEV)
    dn = (torch.randn(E, H, I) * 0.05).to(DEV)
    Cgu, idx_gu, sc_gu, dq_gu = pack_vq2_experts(gu, k=k, group=group, iters=6)
    Cd, idx_d, sc_d, dq_dn = pack_vq2_experts(dn, k=k, group=group, iters=6)
    N = 4
    x = torch.randn(N, H, device=DEV)
    ids = torch.randint(0, E, (N, k), device=DEV)
    w = torch.softmax(torch.randn(N, k, device=DEV), -1)
    y = vq2_moe(x, ids, w, Cgu, idx_gu, sc_gu, Cd, idx_d, sc_d, group)
    # reference: dequant experts to BF16, standard MoE forward
    ref = torch.zeros(N, H, device=DEV)
    for n in range(N):
        for j in range(k):
            e = int(ids[n, j])
            g = dq_gu[e].float() @ x[n]
            a = torch.nn.functional.silu(g[:I]) * g[I:]
            ref[n] += float(w[n, j]) * (dq_dn[e].float() @ a)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"vq2_moe vs dequant-ref MoE: relerr={rel:.2e}  (E={E} k={k})")


def _selftest():
    print("vq2 W2A16 GEMV — correctness + MBU on GB10 (273 GB/s):")
    _bench(2048, 2048)             # small (one tensor, ~launch-bound)
    _bench(6144, 2048)             # expert gate_up-ish [2I, H]
    _bench(2048, 6144)             # expert down-ish    [H, I]
    _bench(16384, 4096)            # large linear (amortizes launch)
    _test_moe()


if __name__ == "__main__":
    _selftest()
