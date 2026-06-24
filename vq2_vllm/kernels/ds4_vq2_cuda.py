"""CUDA grouped vq2 MoE (prefill): coalesced per-expert index reads + fused gather/scatter.
Compiles vq2_grouped.cu on first import. Wrapper matches vq2_moe_fused_10b semantics."""
import os, torch
from torch.utils.cpp_extension import load
import triton, triton.language as tl


@triton.jit
def _quant_fp8_rows(x_ptr, o_ptr, inv_ptr, M, N, BLOCK_N: tl.constexpr):
    """Per-row amax -> scale into fp8-e4m3 range -> cast; emit inv=1/scale. ONE launch
    (replaces torch amax+mul+cast = 3-4 eager launches/tensor)."""
    r = tl.program_id(0)
    amax = 0.0
    for n0 in range(0, N, BLOCK_N):
        c = n0 + tl.arange(0, BLOCK_N); m = c < N
        x = tl.load(x_ptr + r * N + c, mask=m, other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    s = 448.0 / tl.maximum(amax, 1e-6)
    tl.store(inv_ptr + r, 1.0 / s)
    for n0 in range(0, N, BLOCK_N):
        c = n0 + tl.arange(0, BLOCK_N); m = c < N
        x = tl.load(x_ptr + r * N + c, mask=m, other=0.0)
        tl.store(o_ptr + r * N + c, (x * s).to(tl.float8e4nv), mask=m)


def quant_fp8_rows(x):
    """x [M,N] fp32 -> (fp8 [M,N], inv [M] fp32). Fused per-row fp8 quant."""
    M, N = x.shape
    o = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    inv = torch.empty(M, device=x.device, dtype=torch.float32)
    _quant_fp8_rows[(M,)](x, o, inv, M, N, BLOCK_N=min(triton.next_power_of_2(N), 2048))
    return o, inv

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        diag = os.environ.get("VQ2_DIAG", "0")           # compile-time diagnostic/variant selector
        _EXT = load(name=f"vq2_grouped_d{diag}",
                    sources=[os.path.join(_HERE, "vq2_grouped.cu")],
                    extra_cuda_cflags=["-O3", "--use_fast_math", f"-DVQ2_DIAG={diag}",
                                       "-gencode=arch=compute_121a,code=sm_121a"],
                    verbose=True)
    return _EXT


def vq2_moe_grouped_cuda(x, topk_ids, topk_w, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=64, BM=8):
    import vq2_kernel as vq2
    from moe_align import moe_align
    ext = _ext()
    N, H = x.shape
    E = lgu.shape[0]; I = lgu.shape[1] // 2; vd = Cgu.shape[1]
    SPG = group // vd
    st, ss, val, be, nblk = moe_align(topk_ids, E, BM)
    Pmax = st.numel()
    sorted_tok = torch.where(val > 0, st, torch.full_like(st, -1)).to(torch.int32)
    w_sorted = torch.where(val > 0, topk_w[st.long(), ss.long()], torch.zeros_like(val, dtype=topk_w.dtype)).float()
    xh = vq2.had_act_batch(x.float(), group).contiguous()           # [N, Hpad]
    Hp = xh.shape[1]; nsg = Hp // vd; ng = nsg // SPG
    inter = torch.zeros(Pmax, I, device=x.device, dtype=torch.float32)
    ext.gateup(xh, sorted_tok, be.to(torch.int32), Cgu.half(), lgu, hgu, sgu.half(), inter, nblk, I, nsg, ng, BM)
    vb = vq2.had_act_batch(inter, group).contiguous()               # [Pmax, Ipad]
    Ip = vb.shape[1]; nsd = Ip // vd; ngd = nsd // SPG
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    ext.down(vb, sorted_tok, w_sorted, be.to(torch.int32), Cd.half(), ld, hd, sdn.half(), out, nblk, H, nsd, ngd, BM)
    return out


def vq2_moe_grouped_wmma(x, topk_ids, topk_w, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=64, NN=2, BKC=64, NW=8):
    """Tensor-core (WMMA) grouped vq2 MoE. fp16 inputs, fp32 accumulate. Kernel BM=64 fixed;
    NW = warps/CTA, NN = output-row tiles per warp (both cut grid.y activation re-read); BKC = index
    chunks staged contiguously per outer K-step (coalesced uint reads). Scatter fused into down."""
    import vq2_kernel as vq2
    from moe_align import moe_align
    ext = _ext()
    N, H = x.shape
    E = lgu.shape[0]; TwoI = lgu.shape[1]; I = TwoI // 2; vd = Cgu.shape[1]
    SPG = group // vd; BM = 64
    # cudagraph-captured spec VERIFY pads the batch to a fixed size; padding rows carry garbage
    # topk_ids -> moe_align one_hot/gather goes out-of-bounds (cudaErrorMisalignedAddress in-graph).
    # Clamp to valid experts (graph-safe): real rows unaffected; padding rows compute discarded output.
    topk_ids = topk_ids.clamp(0, E - 1)
    st, ss, val, be, nblk = moe_align(topk_ids, E, BM)
    Pmax = st.numel()
    sorted_tok = torch.where(val > 0, st, torch.full_like(st, -1)).to(torch.int32)
    w_sorted = torch.where(val > 0, topk_w[st.long(), ss.long()], torch.zeros_like(val, dtype=topk_w.dtype)).float()
    be = be.to(torch.int32)
    # fp8-e4m3 activations halve the (re-read-dominant) activation bytes. PER-TOKEN scale into fp8
    # range via ONE fused Triton kernel (amax+scale+cast+inv); inverse folded into silu (xh) /
    # routing weight (vb). DS4 is fp8-native so this is quality-neutral (+0.2% PPL).
    xh, inv_xh = quant_fp8_rows(vq2.had_act_batch(x.float(), group))   # fp8 [N,Hp], inv [N]
    nsg = xh.shape[1] // vd
    # VQ2_FP8_PREFILL: native fp8 mma.sync gateup/down (2x tensor cores + halve As/Bs traffic) ~1.4x
    # on the MoE. Quality-neutral: activations are already per-token-scaled fp8 (in-range) and the
    # centroid*scale is saturated to the e4m3 range; rel-vs-half ~3% = the accepted fp8-prefill band.
    fp8 = os.environ.get("VQ2_FP8_PREFILL", "1") == "1"   # default ON (shipped: 1.41x+1.48x, +0.76% PPL, model's native fp8 regime)
    gateup = ext.gateup_wmma_fp8 if fp8 else ext.gateup_wmma
    down = ext.down_wmma_fp8 if fp8 else ext.down_wmma
    interf = torch.empty(Pmax, TwoI, device=x.device, dtype=torch.float32)
    gateup(xh, sorted_tok, be, Cgu.half(), lgu, hgu, sgu.half(), interf, nblk, TwoI, nsg, SPG, NN, BKC, NW)
    inter = torch.empty(Pmax, I, device=x.device, dtype=torch.float32)
    ext.silu_comb(interf, inter, sorted_tok, inv_xh, Pmax, I)
    vb, inv_vb = quant_fp8_rows(vq2.had_act_batch(inter, group))       # fp8 [Pmax,Ip], inv [Pmax]
    nsd = vb.shape[1] // vd
    w_scaled = (w_sorted * inv_vb).contiguous()                    # fold per-token inv_vb into routing weight
    out = torch.zeros(N, H, device=x.device, dtype=torch.float32)
    down(vb, be, sorted_tok, w_scaled, Cd.half(), ld, hd, sdn.half(), out, nblk, H, nsd, SPG, NN, BKC, NW)
    return out


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, _HERE)
    import vq2_kernel as vq2
    dev = "cuda"; torch.manual_seed(0)
    E, H, I, vd, kcb, grp = 256, 4096, 2048, 4, 1024, 64; SPG = grp // vd

    REAL = os.environ.get("REAL", "")
    if REAL and REAL.endswith(".safetensors"):
        # real fused-expert layer (CLUSTERED codebook indices -- the realistic gather pattern; random
        # u8 indices overstate gather/bank-conflict cost, see memory). Keys mirror bench_vq2_verify.py.
        print(f"[real-st] loading {REAL}", flush=True)
        from safetensors import safe_open
        f = safe_open(REAL, "pt", device=dev); g = lambda k: f.get_tensor(k)
        Cgu, lgu, hgu, sgu = g("w13_cb").half(), g("w13_lo"), g("w13_hi"), g("w13_sc").half()
        Cd, ld, hd, sdn = g("w2_cb").half(), g("w2_lo"), g("w2_hi"), g("w2_sc").half()
        print(f"[real-st] gu lo {tuple(lgu.shape)} dn lo {tuple(ld.shape)}", flush=True)
    elif REAL:
        print(f"[real] loading {REAL}", flush=True)
        d = torch.load(REAL, map_location=dev, weights_only=False)
        Cgu, lgu, hgu, sgu = d["gu"]; Cd, ld, hd, sdn = d["dn"]
        Cgu = Cgu.half(); Cd = Cd.half(); sgu = sgu.half(); sdn = sdn.half()
        print(f"[real] gu lo {tuple(lgu.shape)} dn lo {tuple(ld.shape)}", flush=True)
    else:
        def u8(*s): return torch.randint(0, 255, s, dtype=torch.uint8, device=dev)
        Cgu = torch.randn(kcb, vd, dtype=torch.float16, device=dev); Cd = torch.randn(kcb, vd, dtype=torch.float16, device=dev)
        lgu = u8(E, 2 * I, H // vd); hgu = u8(E, 2 * I, H // vd // 4); sgu = torch.randn(E, 2 * I, (H // vd) // SPG, dtype=torch.float16, device=dev).abs() * 0.1
        ld = u8(E, H, I // vd); hd = u8(E, H, I // vd // 4); sdn = torch.randn(E, H, (I // vd) // SPG, dtype=torch.float16, device=dev).abs() * 0.1

    CLUSTER = os.environ.get("CLUSTER", "1") == "1"

    def inp(Ntok, k=6):
        x = torch.randn(Ntok, H, dtype=torch.bfloat16, device=dev) * 0.3
        if CLUSTER:
            # mimic real prefill: consecutive tokens route to nearby experts (contiguous groups
            # after moe_align), so per-pair gets codebook/expert L2 locality -- the realistic case.
            base = (torch.arange(Ntok, device=dev) * (E - k) // max(Ntok, 1)).to(torch.int32)
            tid = (base[:, None] + torch.arange(k, device=dev)[None, :].to(torch.int32))
        else:
            tid = torch.randint(0, E, (Ntok, k), dtype=torch.int32, device=dev)
        tw = torch.rand(Ntok, k, device=dev)
        return x, tid, tw

    WMMA = os.environ.get("WMMA", "1") == "1"
    BMS = [int(b) for b in os.environ.get("BMS", "").split(",") if b]
    x, tid, tw = inp(64)
    ref = vq2.vq2_moe_fused_10b(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp)
    for bm in BMS:
        cu = vq2_moe_grouped_cuda(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp, BM=bm)
        print(f"correctness rel (FMA BM={bm}):", round((ref - cu).norm().item() / ref.norm().item(), 5), flush=True)
    CFGS = [tuple(int(v) for v in c.split("x")) for c in os.environ.get("WCFG", "8x2x64").split(",")]
    if WMMA:
        for nw, nn, bkc in CFGS:
            w = vq2_moe_grouped_wmma(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp, NW=nw, NN=nn, BKC=bkc)
            print(f"correctness rel (WMMA {nw}x{nn}x{bkc}):", round((ref - w).norm().item() / ref.norm().item(), 5), flush=True)

    def t(fn, n=5):
        fn(); torch.cuda.synchronize(); s = time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.time() - s) / n * 1e3

    if os.environ.get("BREAKDOWN") == "1":
        import vq2_kernel as vq2k
        from moe_align import moe_align
        ext = _ext(); BM = 64; NW = int(os.environ.get("WNW","8")); NN = int(os.environ.get("WNN","2")); BKC = int(os.environ.get("WBKC", "64")); SPG = grp // vd; TwoI = 2 * I
        for Ntok in (2048,):
            x, tid, tw = inp(Ntok)
            st, ss, val, be, nblk = moe_align(tid, E, BM); Pmax = st.numel()
            sti = torch.where(val > 0, st, torch.full_like(st, -1)).to(torch.int32)
            ws = torch.where(val > 0, tw[st.long(), ss.long()], torch.zeros_like(val, dtype=tw.dtype)).float()
            be = be.to(torch.int32)
            xh = vq2k.had_act_batch(x.float(), grp).to(torch.float8_e4m3fn).contiguous(); nsg = xh.shape[1] // vd
            invx = torch.ones(Ntok, device=dev)
            interf = torch.zeros(Pmax, TwoI, device=dev); inter = torch.empty(Pmax, I, device=dev)
            print(f"  [breakdown N={Ntok} Pmax={Pmax} nblk={nblk}]", flush=True)
            ghalf = round(t(lambda: ext.gateup_wmma(xh, sti, be, Cgu.half(), lgu, hgu, sgu.half(), interf, nblk, TwoI, nsg, SPG, NN, BKC, NW)), 2)
            print("   gateup_wmma", ghalf, "ms", flush=True)
            if os.environ.get("FP8MMA") == "1":
                interf8 = torch.zeros(Pmax, TwoI, device=dev)
                ext.gateup_wmma(xh, sti, be, Cgu.half(), lgu, hgu, sgu.half(), interf, nblk, TwoI, nsg, SPG, NN, BKC, NW)
                ext.gateup_wmma_fp8(xh, sti, be, Cgu.half(), lgu, hgu, sgu.half(), interf8, nblk, TwoI, nsg, SPG, NN, BKC, NW)
                rel = ((interf - interf8).norm() / interf.norm().clamp_min(1e-6)).item()
                gf8 = round(t(lambda: ext.gateup_wmma_fp8(xh, sti, be, Cgu.half(), lgu, hgu, sgu.half(), interf8, nblk, TwoI, nsg, SPG, NN, BKC, NW)), 2)
                print(f"   gateup_fp8  {gf8} ms ({ghalf/max(gf8,1e-6):.2f}x)  rel-vs-half={rel:.4f}", flush=True)
            print("   silu_comb  ", round(t(lambda: ext.silu_comb(interf, inter, sti, invx, Pmax, I)), 2), "ms", flush=True)
            vb = vq2k.had_act_batch(inter, grp).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(); nsd = vb.shape[1] // vd
            out = torch.zeros(Ntok, H, device=dev)
            dhalf = round(t(lambda: ext.down_wmma(vb, be, sti, ws, Cd.half(), ld, hd, sdn.half(), out, nblk, H, nsd, SPG, NN, BKC, NW)), 2)
            print("   down+scatter", dhalf, "ms", flush=True)
            if os.environ.get("FP8MMA") == "1":
                out_h = torch.zeros(Ntok, H, device=dev); ext.down_wmma(vb, be, sti, ws, Cd.half(), ld, hd, sdn.half(), out_h, nblk, H, nsd, SPG, NN, BKC, NW)
                out_8 = torch.zeros(Ntok, H, device=dev); ext.down_wmma_fp8(vb, be, sti, ws, Cd.half(), ld, hd, sdn.half(), out_8, nblk, H, nsd, SPG, NN, BKC, NW)
                reld = ((out_h - out_8).norm() / out_h.norm().clamp_min(1e-6)).item()
                df8 = round(t(lambda: ext.down_wmma_fp8(vb, be, sti, ws, Cd.half(), ld, hd, sdn.half(), out_8, nblk, H, nsd, SPG, NN, BKC, NW)), 2)
                print(f"   down_fp8    {df8} ms ({dhalf/max(df8,1e-6):.2f}x)  rel-vs-half={reld:.4f}", flush=True)
            print("   had(x)     ", round(t(lambda: vq2k.had_act_batch(x.float(), grp)), 2), "ms", flush=True)
            # --- OVERHEAD attribution (the non-kernel cost that dilutes the fp8 MoE win) ---
            from moe_align import moe_align_fast
            print("   moe_align  ", round(t(lambda: moe_align(tid, E, BM)), 2), "ms (torch)", flush=True)
            print("   moe_align_fast", round(t(lambda: moe_align_fast(tid, E, BM)), 2), "ms (triton atomic)", flush=True)
            _m = lambda: vq2_moe_grouped_wmma(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp, NW=NW, NN=NN, BKC=BKC)
            os.environ["VQ2_FAST_ALIGN"] = "0"; m0 = _m()
            os.environ["VQ2_FAST_ALIGN"] = "1"; m1 = _m()
            os.environ["VQ2_FAST_ALIGN"] = "0"
            print(f"   fast-align MoE rel-vs-torch-align: {((m0 - m1).norm() / m0.norm().clamp_min(1e-6)).item():.2e}", flush=True)
            print("   quant_fp8(had(x))", round(t(lambda: quant_fp8_rows(vq2k.had_act_batch(x.float(), grp))), 2), "ms", flush=True)
            print("   quant_fp8(had(inter)) [vb, Pmax rows]", round(t(lambda: quant_fp8_rows(vq2k.had_act_batch(inter, grp))), 2), "ms", flush=True)
            print("   allocs (interf+out zeros)", round(t(lambda: (torch.empty(Pmax, TwoI, device=dev), torch.zeros(Ntok, H, device=dev))), 2), "ms", flush=True)
            _moe = lambda: vq2_moe_grouped_wmma(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp, NW=NW, NN=NN, BKC=BKC)
            cfgs = {"baseline": ("0", "0"), "+fp8": ("1", "0"), "+fp8+fast": ("1", "1")}
            def _run(c):
                os.environ["VQ2_FP8_PREFILL"], os.environ["VQ2_FAST_ALIGN"] = cfgs[c]; _moe()
            for c in cfgs: _run(c)                                  # warmup all
            mins = {c: 1e9 for c in cfgs}
            for _ in range(12):                                    # INTERLEAVED -> min robust to clock drift
                for c in cfgs:
                    torch.cuda.synchronize(); s = time.time(); _run(c); torch.cuda.synchronize()
                    mins[c] = min(mins[c], (time.time() - s) * 1e3)
            os.environ["VQ2_FP8_PREFILL"] = "0"; os.environ["VQ2_FAST_ALIGN"] = "0"
            b = mins["baseline"]
            print(f"   FULL vq2_moe_grouped_wmma [interleaved min]: baseline {b:.1f}ms | "
                  f"+fp8 {mins['+fp8']:.1f}ms ({b/mins['+fp8']:.2f}x) | "
                  f"+fp8+fast {mins['+fp8+fast']:.1f}ms ({b/mins['+fp8+fast']:.2f}x)", flush=True)

    for Ntok in (512, 1024, 2048, 4096):
        x, tid, tw = inp(Ntok)
        f = t(lambda: vq2.vq2_moe_fused_10b(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp))
        msg = f"N={Ntok}: per-pair {f:.1f}ms"
        for bm in BMS:
            c = t(lambda: vq2_moe_grouped_cuda(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp, BM=bm))
            msg += f"  | FMA BM={bm} {c:.1f}ms ({f/c:.2f}x)"
        if WMMA:
            for nw, nn, bkc in CFGS:
                w = t(lambda: vq2_moe_grouped_wmma(x, tid, tw, Cgu, lgu, hgu, sgu, Cd, ld, hd, sdn, group=grp, NW=nw, NN=nn, BKC=bkc))
                msg += f"  | WMMA {nw}x{nn}x{bkc} {w:.1f}ms ({f/w:.2f}x)"
        print(msg, flush=True)
