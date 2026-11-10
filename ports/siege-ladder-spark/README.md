# Siege-Ladder on one DGX Spark: port plan and performance estimate

[Siege-Ladder](https://github.com/MasherDon/Siege-Ladder) runs **DeepSeek-V4.1-Flash-1017** (373 GB at ~2.4 bits)
on a 16 GB RTX 5070 Ti with 64 GB of RAM. It uses 121 patches over JigSawPT's llama.cpp fork and streams the routed
experts through a ladder: VRAM slots, then a pinned RAM tier, then NVMe. This folder covers what it takes to run the
same thing on one DGX Spark (GB10, sm_121, 128 GB unified LPDDR5x), and what speed to expect.

**Status:** the Linux port of the patch series is done and tested on CPU. Nothing here has run on a Spark yet. The
speeds below are estimates, not measurements.

## TL;DR

- **Software changes are small.** All 121 patches apply cleanly, and the series **builds on Linux** (checked:
  CPU-only `llama-server` build on x86-64 Linux). One real gap was found and fixed. Since P8d, the Engram tables are
  read through `llama_read_scattered`, which had only a Windows body. On Linux it always returned `false`, and the
  server stopped at the first prompt. [`patches/0122`](patches/) adds a Linux body (O_DIRECT + kernel AIO, no new
  dependencies). It returns the same bytes as plain reads in a randomized test
  ([`patches/test_scatter.cpp`](patches/test_scatter.cpp)). The WDDM code is already stubbed off the Windows path
  (`llama_wddm_query_budget` returns `{}`), and the base fork's expert reads already have an O_DIRECT path.
- **The ladder gets one rung shorter.** VRAM and RAM are the same memory on a Spark, so the 30 ms/token PCIe copy
  wall goes away. About **2/3 of the experts (~250 of 384 per layer) stay resident**, against ~35% on the PC.
  Projected misses fall from ~20 to ~6 per token.
- **A new wall replaces it: DRAM bandwidth.** A token reads ~8.1 GB: 5.5 GB of Q5_K dense and shared weights plus
  2.6 GB of routed experts. 273 GB/s is about 1/3 of the 5070 Ti's VRAM bandwidth.
- **Estimated decode: ~17–18 tok/s (range 14–21)** at short context with the `default` preset. That is about the
  same as the PC's 18.4–21.3. The bottleneck moves from the bus to memory bandwidth; the Spark is not 2× faster.
  Prefill should land near the PC on agent turns. On long documents it should be somewhat slower, because it is
  compute-bound and GB10 has roughly 0.6–0.75× the 5070 Ti's tensor throughput.

## 1. Where the bytes are (from `template/cd/manifest.json`)

| class | `cd` | `qd` | read per decode token |
|---|---|---|---|
| Engram tables (2 layers, FP8 + scales) | 202.97 GB | same | a few KB of row lookups, so they stay on NVMe |
| routed experts (40 × 384, 6 per token) | 164.21 GB (10.69 MB each) | 152.88 GB (9.95 MB each) | 240 × 10.69 MB = **2.57 GB** |
| dense part (Q5_K/Q8_0/BF16/F32) | 4.97 GB | same | ~4.5 GB (all but `token_embd`) |
| shared experts (Q5_K) | 0.97 GB | same | 0.97 GB |

Over half of the 373 GB is Engram. Engram is a hash-indexed row lookup, so it never needs to be in memory. Experts
plus the dense part come to ~170 GB, which does not fit in 128 GB. That forces streaming.

## 2. Mapping the ladder onto unified memory

| PC rung | PC | Spark |
|---|---|---|
| VRAM slots | 23/layer (~10 GB) | **~230–250/layer (~94–100 GiB)**, `cudaMalloc` in LPDDR5x |
| pinned RAM tier (`--moe-stream-l2`) | 43–46 GiB, capped by WDDM | **small, ~4–8 GiB**: only a landing area for the lens' read-ahead (G35) and for dropped experts |
| tier → slot copy over PCIe 4.0 (28.8 GB/s) | ~30 ms/token, *the wall* | only on NVMe misses: a DRAM→DRAM `cudaMemcpy` of ~0.1 ms per expert |
| NVMe | Gen5 13.5 GB/s + Gen4 mirror | internal M.2 (**speed unverified**, see §5), plus an optional USB4 mirror |

Memory budget, of ~119.7 GiB visible: OS/CUDA ~6 GiB, dense+shared ~5.1 GiB, compute buffers/pinned blocks ~6 GiB,
KV ~2 GiB at 32K. That leaves **~100 GiB for experts**, which is 66% of `cd` and 71% of `qd`. At ~1M context the
q8 KV cache can take ~25+ GiB. Residency then drops to ~50% (`estimate.py --ctx-kv-gib 27`).

Leave headroom. On a Spark, overcommitting unified memory does not fail cleanly the way a discrete card does: it
gets the process OOM-killed or stalls the desktop. Start at `SLOTS=200` and work up.

Not taken (yet): zero-copy experts in host-mapped memory. ggml-cuda currently forces `integrated = false`
(upstream #15034), and device slots already avoid the bus. That option would only matter if a much bigger RAM tier
became useful.

## 3. Steps

```bash
# on the Spark (DGX OS, CUDA 13.0 - not 13.2: Siege-Ladder hit an IQ-type mul_mat_id miscompile on 13.2 / Blackwell)
./build.sh ~/siege-ladder                     # clone fork @ 8ce0968, git am 121 patches + 0122, build sm_121

# weights: 373 GB (cd) or 362 GB (qd), assembled from public HF repos by range requests, resumable
python3 ~/siege-ladder/Siege-Ladder/tools/assemble.py /data/v41-cd --plan
python3 ~/siege-ladder/Siege-Ladder/tools/assemble.py /data/v41-cd
python3 ~/siege-ladder/Siege-Ladder/tools/assemble.py /data/v41-cd --check
#   (or download `og` + g44/1017-lens-d2.gguf from MasherDon/DeepSeek-V4.1-Flash-1017-SiegeLadder-GGUF;
#    put g44/ one level above the model folder for the corrected lens)

MODEL=/data/v41-cd/DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf SLOTS=230 L2=6 ./run_server.sh
python3 bench_spark.py                         # decode + prompt tok/s from the server's own timings
python3 estimate.py --ssd-gbs <your fio number> # re-run the projection with your drive
```

Disk: the 4 TB Founders Edition has room. A 1 TB unit holds the model with ~500 GB left. Keep the model on ext4,
where O_DIRECT works. The Engram tables read many small random 4K blocks, so watch for the read-disturb slowdown
Siege-Ladder saw after weeks of use (§Hardware notes in their README).

### What to verify first on real hardware

1. **The load log** should show `MoE expert streaming uses O_DIRECT` and, for the Engram files,
   `Siege-Ladder (linux): … O_DIRECT + AIO`. If 0122 is missing, the server stops at the first prompt; the stopgap is
   `LLAMA_ENGRAM_UNMAPPED=0`, which goes through mmap and the page cache.
2. **The router mailbox (G16/G22).** The GPU writes into pinned host memory with system-scope fences, and the host
   polls it. This should be cheaper on GB10's coherent memory than under WDDM. If decode looks wrong, compare against
   `LLAMA_MOE_STREAM_MAILBOX=0`.
3. **The `cuda_host` cache and the head yield (P1).** Both were tuned to squeeze into 16 GB. They should simply have
   more room here, but check the load log for D16 "not allocated" lines.
4. **CUDA build.** Not compiled here: there is no `nvcc` in this sandbox. The ggml-cuda changes in the series contain
   no Windows-only API, but compile them with `-DCMAKE_CUDA_ARCHITECTURES=121` before relying on that.

Their `bench/` scripts are Windows-only as written: process groups, PowerShell counters. `bench_spark.py` covers the
basic numbers. For A B B A teacher-forced decode, port `tf_run.py`'s process handling.

## 4. Performance estimate

The model is in `estimate.py`, with every assumption as a flag:

- **Compute:** bytes per token divided by the effective bandwidth of llama.cpp kernels on GB10. The range used is
  140 / 165 / 190 GB/s, which is 51–70% of 273. Two anchors support it: this repo's vq2 stack on the same Spark
  decodes DeepSeek-V4-Flash non-spec at 17.4 tok/s (~57 ms for a similar byte count), and llama.cpp reaches
  ~50–59 tok/s on gpt-oss-120b. A fixed 3 ms covers launches and syncs.
- **Misses:** a Zipf popularity per layer (a = 1.21) fitted to the PC's measured hits (~20 of 240 choices not
  resident at ~35% residency). A 0.85 factor accounts for temporal locality. The drop rule skips about half of the
  remaining misses, as on the PC (10 drive, 10 dropped).
- **Drive:** each waited miss is a ~10.7 MB O_DIRECT read. Decode waits on that read's latency and bandwidth, with
  15% hidden by split calls and the lens.

| short-context decode, `default` | pessimistic | central | optimistic |
|---|---|---|---|
| `cd` (66% resident, ~3 misses waited/token) | 14.0 tok/s | **17.3 tok/s** | 20.3 tok/s |
| `qd` (71% resident) | 14.7 | **18.0** | 21.0 |
| `qd` + `fast` preset (+~5–10%, lossy) | ~15.5 | **~19–20** | ~22.5 |
| ceiling: `cd` fully resident at 165 / 273 GB/s | | 20.5 | 33.9 (unreachable peak) |
| *PC, measured (for comparison)* | | *18.4–21.3* | |

Other workloads, extrapolated more loosely. Read these as ±30%.

| | PC measured | Spark estimate | why |
|---|---|---|---|
| decode at 256–512K | 16–20 | **12–16** | the sparse-attention gather adds bytes; at 1M the KV takes expert slots (≈16 tok/s central at 27 GiB of KV) |
| agent-turn reading (0.5–7.5K new tokens) | 418–520 tok/s | **350–500** | fewer NVMe bytes per turn (≈57 GB not resident vs ≈106) and no bus, but the drive is probably slower and compute is 0.6–0.75× |
| long document, layer-major (256K / 1M) | 1128 / 888 | **~700–850 / ~550–650** | layer-major reads each layer's experts once, so it is compute-bound; it scales with tensor throughput |
| cold 8.9K prompt | ~729 (`cd`) | 450–700 | depends mostly on SSD bandwidth |

**What would move it:**

1. **A lighter dense part.** About 70% of a token's bytes are the Q5_K dense and shared weights. IQ4_XS or Q4_K
   there saves ~1 GB/token, worth ~+12% decode, at some quality cost. It is the biggest single lever on a Spark,
   where it barely mattered on the PC.
2. **MTP spec decode.** It gives +29% in this repo's stack. Here each verified draft token also routes experts, so
   there are more misses per step and the gain would be smaller. Siege-Ladder has no MTP path yet.
3. **A faster drive or a mirror.** On a Spark this matters less than on the PC, because ~3 waited misses per token is
   ~4–10 ms. A USB4 NVMe as `LLAMA_MOE_STREAM_MIRROR` helps prefill more than decode.
4. **Two Sparks** (§5): a layer split with llama.cpp barely helps decode (~18 tok/s); tensor parallelism could reach
   ~28 tok/s, or ~34–37 with MTP, but needs a vLLM-style stack.

## 5. Two Sparks

Two Sparks over their ConnectX-7 link (200 GbE RoCE) give ~215 GiB to work with. Every expert plus the dense part is
158 GiB, so **the whole model except Engram fits**, with ~57 GiB left for KV. The drive drops out of decode, and you no
longer need Siege-Ladder's streaming. That is the point: its ladder exists only because the model doesn't fit. The
Engram tables (203 GB) stay on NVMe as row lookups, and they live on node A, because Engram layers 1 and 14 are both
in the first half.

How much faster it gets depends almost entirely on **how the model is split** (`estimate.py --sparks 2`):

| split | how | decode, central (range) | software that does it |
|---|---|---|---|
| pipeline (layers 0–19 / 20–39) | nodes run one after the other, so a token still reads all 8 GB at one node's bandwidth | **~18 tok/s** (15–21) | llama.cpp `ggml-rpc`, available today |
| expert-parallel (192 experts/layer each) | dense replicated, so each node still reads all 5.5 GB of it | ~20 (17–23) | vLLM/SGLang EP |
| **tensor-parallel (TP=2)** | every matrix halved, so each node reads ~4 GB in parallel; 80 small all-reduces per token | **~28 tok/s** (22–35) | vLLM + NCCL over RoCE |
| TP=2 + MTP K=2 | +20–29%, as measured on this repo's stack | **~34–37** (27–45) | the same, plus MTP |
| *one Spark, Siege-Ladder (§4)* | | *~17* | |

- **The easy path buys almost nothing.** Pipeline over `ggml-rpc` is the only multi-node mode llama.cpp has. It
  removes the drive misses, but decode stays bandwidth-bound at one node's 273 GB/s: ~17 → ~18 tok/s. On top of
  that, the fork's streaming code attaches to the local CUDA backend directly (mailbox, `cudaMemcpy` into slots), so
  you would run the fork *without* `--moe-stream-*`, plus ~2 network hops per token.
- **The real gain needs tensor parallelism**, and that means a vLLM-style stack, not llama.cpp. This repo's vq2
  plugin is the natural base: it already runs DeepSeek-V4 on GB10, and vLLM supports TP across two Sparks. The work
  would be:
  1. V4.1-Flash-1017 support (Engram lookups) in the model code;
  2. the 2-bit VQ experts sharded along the intermediate dimension;
  3. NCCL over the ConnectX link.
  None of this exists yet; it is a project, not a config change.
- **Or spend the second Spark on quality instead of speed.** With ~57 GiB spare you can afford ~3-bit experts (~205
  GB of experts) instead of ~2.4-bit. On Siege-Ladder's API sets, the step toward IQ3 recovered ~28% of the Q2→Q4
  gap. For this repo's own V4-Flash, two Sparks hold the ≥4.25 bpw experts (~150 GB) that
  [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) says close the +27% PPL gap.
- **Prefill gains the most.** With nothing streaming from the drive and TP doubling the compute, agent-turn reading
  should roughly double over one Spark (~700–1000 tok/s, low confidence). Long documents get the same ~2× from
  compute. 1M context fits without taking memory from the experts.
- **Unmeasured:** NCCL small-message all-reduce latency over Spark-to-Spark RoCE, assumed 15–60 µs. At 80 per token,
  it decides whether TP lands at 22 or 35 tok/s. Measure it with `nccl-tests all_reduce_perf -b 8K -e 64K` before
  committing to the TP route.

## 6. Open uncertainties

- **Internal SSD bandwidth.** Public figures disagree, from 3.4 GB/s to 11+ GB/s, and from PCIe Gen4 to Gen5 x4.
  Measure it and pass the number to `estimate.py --ssd-gbs`:
  `fio --rw=randread --bs=10M --iodepth=8 --direct=1 --filename=<model part> --runtime=30 --name=x`, and
  `--bs=4k --iodepth=64` for the Engram pattern.
- **llama.cpp kernel efficiency for IQ2_XS/Q2_K MMVQ on sm_121.** This is the largest term. If it lands near the vq2
  stack's ~145 GB/s, expect the pessimistic-to-central half of the range.
- **Hit rate.** The Zipf extrapolation from one PC operating point is the softest part of the model. The fork's own
  counters (drive/RAM/VRAM per call, printed at exit) will show the real number after a first run.
