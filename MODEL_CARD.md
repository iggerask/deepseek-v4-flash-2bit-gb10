---
license: other
license_name: deepseek-v4-flash
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
tags:
  - deepseek
  - moe
  - 2-bit
  - vq
  - nvfp4
  - quantized
  - gb10
  - dgx-spark
  - sm_121
---

# DeepSeek-V4-Flash 2-bit (GB10 / DGX Spark)

A **2-bit quantization of [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)**
(299 B params, MoE) that serves on a **single NVIDIA DGX Spark (GB10, 128 GB, sm_121)** — where the
~159 GB FP8/MXFP4 source needs ~2 — at **~41 tok/s** single-stream decode, coherent.

> ⚠️ **This checkpoint is NOT loadable by stock vLLM/transformers.** It uses a custom `vq2`
> quantization method served by a small open-source plugin. You must install the serving stack:
> **https://github.com/iggerask/deepseek-v4-flash-2bit-gb10** (Apache-2.0). The plugin auto-registers
> in vLLM (no fork) and adds the sm_121 kernel replacements DS4 needs on consumer Blackwell.

## License

The **model weights are a derivative of DeepSeek-V4-Flash and are governed by that model's license**
(linked above). The serving **code** (plugin + kernels) is Apache-2.0.

## How to use

```bash
git clone https://github.com/iggerask/deepseek-v4-flash-2bit-gb10
cd deepseek-v4-flash-2bit-gb10
./install.sh && source .venv-vllm/bin/activate
python scripts/download_model.py              # pulls this repo -> models/DeepSeek-V4-Flash-2bit
python scripts/serve.py --prompt "What is the capital of France?"
```

It is an **instruct** model — use the chat template (`llm.chat` / a chat endpoint), not raw completion.

## Recipe

| component | precision |
|---|---|
| routed experts (gate/up/down) | 2-bit VQ (k=1024, vdim=4, group-64 Hadamard) |
| attn (wq_a/wkv/wq_b/wo_a/wo_b), shared experts, lm_head | NVFP4 4-bit (W4A16) |
| compressor, indexer, norms, router, embed | BF16 |
| KV cache | fp8 |

Plus the model's built-in **MTP** draft head (K=3 spec-decode) and **FR-Spec** (frequency-shortlisted
draft lm_head; target verifies full vocab so output is exact).

## Quality & speed

- **Perplexity** (matched concatenated 512-tok chunks): **4.64** vs the FP4 source's **3.66** (+27 %).
  This is genuine 2-bit-expert precision loss and is *memory-bound* — closing it to source needs
  ≥4.25 bpw experts (~2 Sparks). +27 % is the floor for 2-bit experts at this parameter count.
- **Decode:** ~41 tok/s (MTP K=3 + FR-Spec) / ~18–20 tok/s (no spec), single-stream on one GB10.
- Coherent instruction-following; terminates on EOS.

## Hardware

NVIDIA DGX Spark / GB10 (sm_121, aarch64), 128 GB unified memory, CUDA 13. ~107 GB resident, leaving
headroom for long-context KV (~27 KB/token; verified to 392 K tokens).

## Files

`config.json` (with the `vq2` quantization_config), `experts_fused_layer_*.safetensors` (2-bit VQ
experts), `rest_layer_*.safetensors` + `rest_globals.safetensors` (NVFP4 + BF16 rest),
`mtp_rest.safetensors` (MTP draft), `frspec_nvfp4_ds4.pt` (FR-Spec reduced draft lm_head, loaded by
the plugin via `VLLM_FRSPEC_NVFP4` — auto-resolved from the model dir), tokenizer. See the GitHub
repo for the architecture writeup.
