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


if __name__ == "__main__":
    main()
