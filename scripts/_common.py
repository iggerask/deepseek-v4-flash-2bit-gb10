"""Shared serve recipe for DeepSeek-V4-Flash 2-bit on one DGX Spark (GB10, sm_121).

The exact, verified configuration: stock vLLM 0.23.0 + the vq2 plugin (auto-registered),
compiled CUDA graphs, fp8 KV (the DS4 FlashMLA path requires it), MTP spec-decode + FR-Spec.
"""
import os


def setup_env(model_dir, frspec=None):
    """Set env BEFORE importing vllm (the plugin registers + reads these at import/engine init)."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # in-process so the plugin reaches the worker
    # expandable_segments makes the CUDA caching allocator's reserved memory track active closely
    # (no fragmentation balloon). On a ~95GB model in 128GB, that fragmentation was the real
    # long-context prefill "memory wall" (reserved 108GB vs active 101GB); this reclaims ~4-6GB.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")  # nvcc for the vq2_grouped.cu JIT
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")              # sm_121 cutlass-DSL kernels
    os.environ.setdefault("VQ2_EXPERTS_DIR", model_dir)           # plugin loads experts_fused_layer_*.safetensors here
    if frspec:                                                    # FR-Spec reduced draft lm_head (read at register())
        os.environ["VLLM_FRSPEC_NVFP4"] = frspec


def build_llm(model_dir, max_model_len=4096, spec_mtp=2):
    """Construct the LLM with the verified recipe. spec_mtp=None/0 -> no speculative decode.
    K=2 is the measured optimum for realistic single-stream chat (see README); K=3 over-drafts
    (the 3rd token rarely accepts but makes verify K+1=4, net slower)."""
    from vllm import LLM
    extra = {}
    if spec_mtp:
        extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(spec_mtp)}
        # FULL cudagraph captures the whole K+1 verify in one graph (no PIECEWISE attn breaks);
        # capture sizes cover the single-token draft (1) and the verify batch (K+1 = 3 at K=2).
        extra["compilation_config"] = {
            "cudagraph_capture_sizes": [1, 2, 3, 4, 8, 16],
            "cudagraph_copy_inputs": True,
            "cudagraph_mode": "FULL",
        }
    # Long-context single-stream knobs (optional; defaults preserve the short-ctx chat recipe):
    #   KV_CACHE_MEMORY_BYTES caps the MLA KV pool to one stream's need (frees headroom that the
    #     default util-sized pool would waste at max-num-seqs=1) -> needed to fit 256k-512k ctx.
    #   MAX_NUM_BATCHED_TOKENS is the prefill chunk cap; pair with VQ2_ADAPTIVE_PREFILL (see plugin).
    #   GPU_UTIL ~0.85 leaves more physical headroom for the prefill transient at long ctx.
    kvbytes = os.environ.get("KV_CACHE_MEMORY_BYTES")
    if kvbytes:
        extra["kv_cache_memory_bytes"] = int(kvbytes)
    mnbt = os.environ.get("MAX_NUM_BATCHED_TOKENS")
    if mnbt:
        extra["max_num_batched_tokens"] = int(mnbt)
    return LLM(
        model=model_dir, trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.92")),
        max_model_len=max_model_len, enforce_eager=False,
        max_num_seqs=1, kv_cache_dtype="fp8", **extra,
    )
