# DeepSeek-V4-Flash 2-bit on one DGX Spark (GB10)

Serve **DeepSeek-V4-Flash** (299 B params, MoE) **2-bit-quantized** on a **single NVIDIA DGX Spark
(GB10, 128 GB, sm_121)** with **stock vLLM 0.23.0 + a small plugin — no vLLM fork**.

- **~107 GB** self-contained model (2-bit VQ experts + NVFP4 hot path) → fits one Spark *with* long-context KV.
- **~22 tok/s** single-stream decode on realistic chat (MTP K=2, **+29 %** over the 17.4 non-spec baseline; TPOT ~39 ms) — and **up to ~40 tok/s** on long, predictable generations. Coherent; measured over the OpenAI chat API with `vllm bench serve`.
- **Quality:** PPL 4.64 vs the FP4 source's 3.66 on a matched corpus (+27 %, the floor for 2-bit experts at this size; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- Runs the datacenter-Blackwell DS4 model on consumer-Blackwell sm_121 via **runtime torch/Triton kernel replacements** (the plugin monkeypatches them in; no native sm_121 build needed).

> The plugin + kernels (this repo) are **Apache-2.0**. The **model weights** are a derivative of
> [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) and are
> governed by **that model's license** — see the model card.

## Hardware & prerequisites

- **NVIDIA DGX Spark / GB10** (sm_121, aarch64), 128 GB unified memory.
- **CUDA 13** toolkit at `/usr/local/cuda` (provides `nvcc`, used to JIT-compile the prefill CUDA kernel).
- **Python 3.12** and [`uv`](https://docs.astral.sh/uv/).
- **~110 GB free disk** for the model.

## Quickstart

```bash
# 1. install the serving environment (stock vLLM 0.23.0 + cu130 torch + the plugin)
./install.sh                       # creates .venv-vllm, pins everything, installs the vq2 plugin
source .venv-vllm/bin/activate

# 2. sanity-check the plugin loaded
python -c "import vllm, vq2_vllm; print('vq2 plugin OK')"

# 3. download the model (~101 GB) from Hugging Face
python scripts/download_model.py                 # -> models/DeepSeek-V4-Flash-2bit

# 4. verify: coherence + decode tok/s (expect "Paris", finish_reason=stop, ~22 tok/s on realistic chat)
python scripts/bench.py

# 5. use it
python scripts/serve.py --prompt "Explain MoE routing in two sentences."
python scripts/serve.py                          # interactive chat REPL
```

`install.sh` uses the PyTorch **cu130** index for the torch stack and PyPI for the rest; if a wheel
fails to resolve on your box, that is the first thing to debug — the pins in
`requirements-serve.lock.txt` are exact.

## What the recipe is

| component | precision | served by |
|---|---|---|
| routed experts (gate/up/down) | **2-bit VQ** (k=1024, vdim=4, group-64 Hadamard, 10-bit byte-plane) | `vq2` Triton MoE (decode) / CUDA WMMA (prefill) |
| attn wq_a/wkv/wq_b, wo_b, shared experts, lm_head | **NVFP4 4-bit** (W4A16 Marlin) | vLLM ModelOpt NVFP4 |
| attn wo_a | NVFP4, grouped W4A16 | custom `_o_proj` (no fused FP8 path) |
| compressor, indexer, norms, router, embed | **BF16** | stock |
| KV cache | **fp8** (the DS4 FlashMLA path requires it) | stock |

Speed stack: **MTP** — the model's built-in `mtp.0` draft head, **K=2** spec-decode (the measured
optimum; K=3 over-drafts — its 3rd token rarely accepts but makes verify K+1=4, net slower), captured in
one FULL cudagraph. Measured **+29 %** over non-spec on realistic chat. A second lever, **FR-Spec**
(frequency-shortlisted draft lm_head, `frspec_nvfp4_ds4.pt`, shipped inside the model repo), is wired in
but **currently inactive** under the plugin load path — the MTP draft head ties to the unquantized
embedding, so the builder falls back to the full lm_head (measured marginal regardless).

## How it was built

DeepSeek-V4-Flash targets datacenter Blackwell (sm_100); GB10 is consumer Blackwell (**sm_121**), for
which vLLM ships **no native build** of DS4's sparse-MLA / lightning-indexer / fused-FP8 kernels. So
running it — let alone in 128 GB at speed — meant writing custom kernels and wiring them in as runtime
monkeypatches over **stock** vLLM (no fork).

**Custom kernels** (`vq2_vllm/kernels/`):
- **vq2 W2A16 VQ fused-MoE GEMV** (Triton) — the 2-bit expert path: one shared codebook + 10-bit
  byte-plane indices, with group-Hadamard incoherence folded into the activation (`y = W·x = Ŵ_rot·(Hᵀx)`).
- **`vq2_grouped.cu`** — a CUTLASS-style **WMMA grouped GEMM** for prefill (each expert's indices read
  coalesced once, tensor-core accumulate).
- **grouped W4A16 `o_proj`** (Triton `tl.dot`) — replaces DS4's fused-FP8 path so `wo_a` stays NVFP4
  (no FP8 downgrade); reads the 4-bit weight once and loops the spec-verify batch.
- **sparse-MLA forward** (Triton, online-softmax flash) + **lightning-indexer logits** (torch) —
  graph-safe sm_121 replacements for the gated `_flashmla_C` kernels.

**Decode speed trajectory** (single-stream, measured over the chat API) — from a naive torch reference (~1 tok/s):
- The biggest non-obvious win was killing a per-step **full-KV-cache `reshape().contiguous()` copy**
  that was **~78 % of decode** (the paged cache is padded/non-contiguous, so flattening copied multi-GB
  every step × 43 layers). Indexing the cache with native strides + assembling RoPE in-kernel fixed it.
- Caching the `o_proj` dequant, arithmetic-E2M1 unpacking, and bf16 activations in the vq2 kernel
  reached the **non-spec batch-1 ceiling ~17 tok/s** (decode is gather-throughput-bound, not DRAM-bound).
- **MTP** spec-decode with the verify in **one FULL cudagraph** + a **tensor-core verify `o_proj`** →
  **~22 tok/s on realistic chat (+29 %)**, up to **~40 on long predictable generations** (high draft
  acceptance). K-sweep on the same ShareGPT-chat bench: non-spec 17.4 / K=1 21.4 / **K=2 22.5** / K=3 20.6 tok/s.

**Quality investigation.** The +27 % PPL cost was traced to genuine precision loss in the 2-bit experts
(not a bug, not the NVFP4 path) and shown to be **memory-bound, not quantizer-bound**: PPL falls
steeply with more expert bits (excess ∝ rel^~2.8), but those bits don't fit one Spark, and
MXFP4-structure tricks / richer codebooks / mixed-precision don't move it. Details in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## API server

`scripts/serve.py` is an offline interactive entry (the quickest verified path). For an
OpenAI-compatible HTTP server, the plugin works under `vllm serve` too — pass the same recipe:

```bash
CUTE_DSL_ARCH=sm_121a VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VQ2_EXPERTS_DIR=$(pwd)/models/DeepSeek-V4-Flash-2bit \
vllm serve models/DeepSeek-V4-Flash-2bit \
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype fp8 \
  --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,3,4,8,16],"cudagraph_copy_inputs":true}'
```

## Notes & limitations

- **GB10-specific.** The kernel replacements target sm_121; on datacenter Blackwell (sm_100) use the
  upstream DS4 kernels instead.
- **Instruct model** — use the chat template (`llm.chat` / the server's chat endpoint), not raw completion.
- **Long context** is gated only by `--max-model-len` (KV is ~27 KB/token; verified to 392 K tokens).
- **fp8 KV is required** by the DS4 FlashMLA path (it hard-asserts fp8).
- The first prefill JIT-compiles `vq2_grouped.cu` (needs `nvcc` on PATH); subsequent runs are cached.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the 2-bit stack works and the quality/speed methodology.
