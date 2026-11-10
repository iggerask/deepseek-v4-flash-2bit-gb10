#!/usr/bin/env python3
"""Back-of-envelope decode/prefill estimate for Siege-Ladder (DeepSeek-V4.1-Flash-1017, experts streamed from
NVMe) on one DGX Spark (GB10, 128 GB unified LPDDR5x, ~273 GB/s).

No dependencies. Every assumption is a flag; the defaults are the ones argued in README.md. Byte counts come from
Siege-Ladder's template/<build>/manifest.json (summed per tensor class):

    cd: routed experts 164.21 GB, dense 4.97 GB (incl. 0.455 GB token_embd + 0.455 GB output), shared experts
        0.97 GB, Engram tables 202.97 GB (row lookups only)
    qd: routed experts 152.88 GB, rest identical

    python3 estimate.py                 # the three scenarios for cd and qd
    python3 estimate.py --ssd-gbs 3.4   # what-if: a slow drive
    python3 estimate.py --ctx-kv-gib 27 # what-if: a ~1M-token q8 KV cache taking memory from the expert cache
"""
import argparse

GB, GiB = 1e9, 2**30
N_LAYER, N_EXPERT, TOPK = 40, 384, 6

BUILDS = {  # bytes from the manifests
    "cd": dict(experts=164.21 * GB, dense_read=(4.97 - 0.455) * GB, shexp=0.97 * GB),
    "qd": dict(experts=152.88 * GB, dense_read=(4.97 - 0.455) * GB, shexp=0.97 * GB),
}

# The PC's measured decode token (docs/techniques.md): of 240 expert choices ~160 hit VRAM, ~61 RAM, ~10 drive,
# ~10 dropped, with ~35% of the expert bytes resident. A Zipf(a) popularity per layer fitted to that point gives
# a = 1.21; a static top-k cache under it is the miss model below (pessimistic: real traffic has temporal locality).
ZIPF_A = 1.214


def zipf_miss(a, frac):
    w = [i ** -a for i in range(1, N_EXPERT + 1)]
    k = int(round(frac * N_EXPERT))
    return 1.0 - sum(w[:k]) / sum(w)


def decode(build, a):
    b = BUILDS[build]
    expert_mb = b["experts"] / (N_LAYER * N_EXPERT)
    bytes_tok = b["dense_read"] + b["shexp"] + TOPK * N_LAYER * expert_mb + a.kv_mb_per_tok * 1e6

    budget = (a.mem_gib - a.os_gib - a.ctx_kv_gib - a.buffers_gib) * GiB - b["dense_read"] - b["shexp"]
    resident = max(0.0, min(1.0, budget / b["experts"]))
    misses = TOPK * N_LAYER * zipf_miss(a.zipf_a, resident) * a.locality
    waited = misses * (1.0 - a.drop_share)  # the drop rule: misses with <14% routing weight are not waited for

    out = {}
    for name, bw_eff, ssd, hide in (("low", a.bw_lo, a.ssd_lo, 0.0), ("mid", a.bw_mid, a.ssd_gbs, a.hide),
                                    ("high", a.bw_hi, a.ssd_hi, 2 * a.hide)):
        t_compute = bytes_tok / (bw_eff * GB) * 1e3 + a.fixed_ms
        t_drive = waited * expert_mb / (ssd * GB) * 1e3 * (1.0 - min(hide, 0.9))
        t = t_compute + t_drive + a.host_ms
        out[name] = (1e3 / t, t_compute, t_drive)
    return dict(bytes_tok=bytes_tok, resident=resident, slots=int(resident * N_EXPERT), misses=misses,
                waited=waited, expert_mb=expert_mb, out=out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mem-gib", type=float, default=119.7, help="memory the OS reports on a Spark (128 GB)")
    p.add_argument("--os-gib", type=float, default=6.0, help="OS, desktop, CUDA context, I/O staging")
    p.add_argument("--buffers-gib", type=float, default=6.0, help="compute buffers, pinned host blocks, output head copy")
    p.add_argument("--ctx-kv-gib", type=float, default=2.0, help="KV cache (q8_0); ~2 GiB at 32K")
    p.add_argument("--kv-mb-per-tok", type=float, default=30.0, help="KV/indexer bytes read per decoded token, MB")
    p.add_argument("--bw-lo", type=float, default=140.0, help="effective GB/s of llama.cpp decode kernels, pessimistic")
    p.add_argument("--bw-mid", type=float, default=165.0, help="... central (vq2/vLLM on this Spark: ~145-150)")
    p.add_argument("--bw-hi", type=float, default=190.0, help="... optimistic (~70%% of 273 GB/s)")
    p.add_argument("--fixed-ms", type=float, default=3.0, help="launch/sync latency not hidden by bandwidth, ms")
    p.add_argument("--host-ms", type=float, default=1.0, help="host part of a token (PC: ~1 ms after G16/G22)")
    p.add_argument("--ssd-gbs", type=float, default=6.0, help="internal NVMe, 10 MB O_DIRECT reads, central GB/s")
    p.add_argument("--ssd-lo", type=float, default=3.4)
    p.add_argument("--ssd-hi", type=float, default=9.0, help="e.g. internal + a USB4/ConnectX-attached mirror")
    p.add_argument("--zipf-a", type=float, default=ZIPF_A)
    p.add_argument("--locality", type=float, default=0.85, help="miss multiplier for temporal locality (1 = static)")
    p.add_argument("--drop-share", type=float, default=0.5, help="share of misses the drop rule does not wait for")
    p.add_argument("--hide", type=float, default=0.15, help="share of drive time hidden by G31 split calls / G35 lens")
    a = p.parse_args()

    print(f"assumed: {a.mem_gib:.1f} GiB, KV {a.ctx_kv_gib:.0f} GiB, SSD {a.ssd_lo}/{a.ssd_gbs}/{a.ssd_hi} GB/s, "
          f"kernels {a.bw_lo:.0f}/{a.bw_mid:.0f}/{a.bw_hi:.0f} GB/s effective\n")
    for build in ("cd", "qd"):
        r = decode(build, a)
        print(f"[{build}] {r['bytes_tok']/GB:.2f} GB read per token, expert {r["expert_mb"]/1e6:.2f} MB, "
              f"{r['resident']*100:.0f}% of experts resident (~{r['slots']} slots/layer), "
              f"{r['misses']:.1f} misses/token, {r['waited']:.1f} waited on the drive")
        for k in ("low", "mid", "high"):
            tps, tc, td = r["out"][k]
            print(f"    {k:4s}: {tps:5.1f} tok/s   (compute {tc:5.1f} ms + drive {td:4.1f} ms + host {a.host_ms:.1f} ms)")
        print()
    # perfect cache, for scale: the DRAM wall
    b = BUILDS["cd"]
    allres = (b["dense_read"] + b["shexp"] + TOPK * N_LAYER * b["experts"] / (N_LAYER * N_EXPERT)) / GB
    print(f"DRAM wall (cd, everything resident): {allres:.2f} GB/token -> {273/allres:.1f} tok/s at the 273 GB/s peak, "
          f"{a.bw_mid/allres:.1f} at {a.bw_mid:.0f} GB/s")


def two_sparks(a):
    """Two Sparks over ConnectX-7 (200 GbE RoCE): ~239 GiB, so every expert and the dense part are resident on one of
    the nodes (Engram stays on NVMe) and the drive drops out of decode. What is left depends on how the model is split:

      pipeline - layers 0-19 on node A, 20-39 on node B (llama.cpp ggml-rpc). The nodes run one after the other, so a
                 token still reads all 8 GB at one node's bandwidth; +2 hops per token.
      ep2      - experts split by id (each node 192/layer), dense replicated: each node reads the whole dense part, and
                 the step waits for the node with more of the token's 6 experts (E[max of Binomial(6, 1/2)] = 3.94).
      tp2      - tensor parallel (vLLM/NCCL style): every matrix split in half, so each node reads half the bytes in
                 parallel; 2 all-reduces per layer (attention out, MoE out) = 80 per token.
    """
    print("=== two DGX Sparks (all experts resident, no drive misses) ===")
    print(f"assumed: kernels {a.bw_lo:.0f}/{a.bw_mid:.0f}/{a.bw_hi:.0f} GB/s per node, half-size kernels at "
          f"{a.tp_scale:.0%} of full efficiency, all-reduce {a.ar_us_hi:.0f}/{a.ar_us:.0f}/{a.ar_us_lo:.0f} us, "
          f"ggml-rpc hop {a.hop_ms_hi}/{a.hop_ms}/{a.hop_ms_lo} ms\n")
    b = BUILDS["cd"]
    expert = TOPK * N_LAYER * b["experts"] / (N_LAYER * N_EXPERT)
    dense = b["dense_read"] + b["shexp"]
    kv = a.kv_mb_per_tok * 1e6
    resident_need = (b["experts"] + dense) / GiB
    per_node = (a.mem_gib - a.os_gib - a.buffers_gib)
    print(f"memory: experts + dense {resident_need:.0f} GiB vs ~{2*per_node:.0f} GiB usable on two nodes "
          f"-> {2*per_node - resident_need:.0f} GiB left for KV and compute buffers\n")
    rows = []
    for name, bw, ar, hop, sc in (("low", a.bw_lo, a.ar_us_hi, a.hop_ms_hi, a.tp_scale - 0.05),
                                  ("mid", a.bw_mid, a.ar_us, a.hop_ms, a.tp_scale),
                                  ("high", a.bw_hi, a.ar_us_lo, a.hop_ms_lo, a.tp_scale + 0.05)):
        fixed = a.fixed_ms + a.host_ms
        t_pipe = (dense + expert + kv) / (bw * GB) * 1e3 + fixed + 2 * hop
        t_ep = (dense + expert * 3.94 / 6 + kv) / (bw * GB) * 1e3 + fixed + 2 * N_LAYER * ar / 1e3
        t_tp = (dense + expert + kv) / 2 / (bw * sc * GB) * 1e3 + fixed + 2 * N_LAYER * ar / 1e3
        rows.append((name, 1e3 / t_pipe, 1e3 / t_ep, 1e3 / t_tp))
    print("            pipeline (rpc)   expert-parallel   tensor-parallel   tp + MTP K=2 (x1.2-1.29)")
    for name, p, e, t in rows:
        print(f"    {name:4s}: {p:9.1f} tok/s {e:12.1f} tok/s {t:12.1f} tok/s {t*1.2:12.1f}-{t*1.29:.1f} tok/s")


if __name__ == "__main__":
    import sys
    if "--sparks" in sys.argv:
        i = sys.argv.index("--sparks")
        n = sys.argv[i + 1] if i + 1 < len(sys.argv) else "1"
        del sys.argv[i:i + 2]
        if n == "2":
            p = argparse.ArgumentParser()
            for k, v in (("--mem-gib", 119.7), ("--os-gib", 6.0), ("--buffers-gib", 6.0), ("--kv-mb-per-tok", 30.0),
                         ("--bw-lo", 140.0), ("--bw-mid", 165.0), ("--bw-hi", 190.0), ("--fixed-ms", 3.0),
                         ("--host-ms", 1.0), ("--tp-scale", 0.85), ("--ar-us", 30.0), ("--ar-us-lo", 15.0),
                         ("--ar-us-hi", 60.0), ("--hop-ms", 1.0), ("--hop-ms-lo", 0.3), ("--hop-ms-hi", 3.0)):
                p.add_argument(k, type=float, default=v)
            two_sparks(p.parse_args())
            sys.exit(0)
    main()
