# Architecture & methodology

How the DeepSeek-V4-Flash 2-bit stack works, why it fits one GB10, and the quality/speed numbers.

## The plugin (`vq2_vllm`)

Registered via a `vllm.general_plugins` entry point, so plain `import vllm` triggers `register()`.
vLLM itself is **stock 0.23.0** — every customization is a runtime monkeypatch:

- **`Vq2Config` / `Vq2MoEMethod`** — a registered quant method named `vq2`. Routed experts load a
  shared codebook + 10-bit byte-plane VQ indices from `experts_fused_layer_*.safetensors`
  (`VQ2_EXPERTS_DIR`); the linears (attn, shared, lm_head) route to NVFP4 W4A16 Marlin.
- **`_patch_o_proj`** — DS4's `_o_proj` hardcodes a fused FP8 deep-gemm (reads `weight_scale_inv`),
  incompatible with our NVFP4 `wo_a`. We override it: inverse-RoPE + unpack-NVFP4(`wo_a`) + grouped
  einsum + native NVFP4 `wo_b`. No downgrade to FP8.
- **`_patch_sparse_attn`** — DS4's sparse-MLA forward + lightning-indexer logits are gated to
  sm_90/sm_100 (`_flashmla_C` has no sm_12x build). We replace them with graph-safe Triton/torch
  references (`ds4_sparse_triton`, `ds4_indexer_torch`) and allow the sm_121 capability.
- **`_patch_lm_head`** — DS4 builds `ParallelLMHead` without a quant_config; we inject `vq2` so the
  lm_head picks up NVFP4 W4A16.
- **`_patch_drafter_full_cudagraph`** (env `VQ2_DRAFTER_FULL_CG`) and **`_patch_frspec`**
  (env `VLLM_FRSPEC_NVFP4`) — the MTP speed levers (below).

### Runtime kernels (`vq2_vllm/kernels/`, on `sys.path`)

| file | role |
|---|---|
| `vq2_kernel.py` | Triton W2A16 VQ fused-MoE GEMV (decode) |
| `ds4_vq2_cuda.py` + `vq2_grouped.cu` | CUDA WMMA grouped GEMM (prefill ≥256 tok); JIT-compiled on first use |
| `ds4_oproj_triton.py` | Triton grouped W4A16 o_proj (NVFP4 `wo_a`) |
| `ds4_sparse_torch.py`, `ds4_sparse_triton.py` | sparse-MLA forward (sm_121 refs) |
| `ds4_indexer_torch.py` | lightning-indexer fp8/fp4 MQA logits (sm_121 refs) |
| `moe_align.py` | MoE align/scatter metadata for the grouped/verify path |

## Why it fits one Spark (and the byte budget)

The official model is FP8 + MXFP4 experts ≈ 159 GB (needs ~2 Sparks). The 2-bit recipe concentrates
precision where it matters and crushes the rarely-fired experts:

- **Experts** (277 B params, ~93 % of weights) → 2-bit VQ ≈ 97 GB. Only 8/256 fire per token.
- **Always-on path** (attn / shared / lm_head) → NVFP4 4-bit (precision-critical, kept higher).
- **Recurrence/routing-sensitive** (compressor, indexer, norms, router) → BF16.

Total ≈ 107 GB, leaving headroom for KV (~27 KB/token) — long context on one box.

## Quality

Apples-to-apples perplexity (matched concatenated 512-tok chunks):

| model | PPL |
|---|---|
| source DeepSeek-V4-Flash (FP8 + MXFP4) | 3.66 |
| **this 2-bit build** | **4.64** (+27 %) |

The +27 % is **genuine 2-bit-expert precision loss**, not a bug: a per-token NLL diff shows broad
quantization noise (≈30 % of tokens *improve*), and the gap is dominated by the 2-bit VQ experts
(weight rel-error ≈0.22) — the NVFP4 hot path contributes ~25 %. It is **memory-bound, not
quantizer-bound**: PPL responds steeply to expert bits (excess ∝ rel^~2.8), so higher-precision
experts would close it, but ≥4.25 bpw experts (~150 GB) need ~2 Sparks. On one Spark, +27 % is the
floor for 2-bit experts at this parameter count.

## Speed

Single-stream, measured over the OpenAI **chat** endpoint with `vllm bench serve` (ShareGPT prompts,
identical across rows):

| configuration | tok/s | TPOT | MTP accept-len |
|---|---|---|---|
| non-spec (no MTP) | 17.4 | 50.0 ms | — |
| MTP K=1 | 21.4 | 39.5 ms | 1.67 |
| **MTP K=2** (default) | **22.5** | 39.1 ms | 2.08 |
| MTP K=3 | 20.6 | 44.0 ms | 2.14 |
| best-case: long predictable generation | ~40 | ~25 ms | 3.6 |

**MTP** reuses the model's built-in `mtp.0` draft head; the verify batch and single-token draft are
FULL-cudagraph-captured. **K=2 is the measured optimum** — at K=3 the 3rd draft token rarely accepts
(accept-len 2.08→2.14) but makes the verify K+1=4, net slower. The speedup is **acceptance-bound**:
~+29 % over non-spec on realistic chat, rising toward ~2× on long, predictable generations
(accept-len ~3.6). The decode bottleneck is the gather-throughput-bound VQ MoE.

**FR-Spec** (frequency-shortlisted draft lm_head, `frspec_nvfp4_ds4.pt`) is wired in but **currently
inactive**: under the entry-point load path the MTP draft head ties to the unquantized embedding, so the
builder falls back to the full draft lm_head. Measured marginal regardless (the best-case ~40 was reached
with it off), so K=2 alone is the live speed lever.

## Scope

This repo is the *serving* deliverable — the plugin, kernels, and everything needed to run the
released model. The quantization pipeline that produced the weights (k-means VQ pack, AWQ folding,
NVFP4 rest-quant, MTP head quant) and the full PPL/speed measurement campaign are out of scope here.
