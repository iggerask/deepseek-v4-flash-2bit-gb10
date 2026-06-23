"""Shared serve recipe for DeepSeek-V4-Flash 2-bit on one DGX Spark (GB10, sm_121).

The exact, verified configuration: stock vLLM 0.23.0 + the vq2 plugin (auto-registered),
compiled CUDA graphs, fp8 KV (the DS4 FlashMLA path requires it), MTP spec-decode + FR-Spec.
"""
import os


def setup_env(model_dir, frspec=None):
    """Set env BEFORE importing vllm (the plugin registers + reads these at import/engine init)."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # in-process so the plugin reaches the worker
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")  # nvcc for the vq2_grouped.cu JIT
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")              # sm_121 cutlass-DSL kernels
    os.environ.setdefault("VQ2_EXPERTS_DIR", model_dir)           # plugin loads experts_fused_layer_*.safetensors here
    if frspec:                                                    # FR-Spec reduced draft lm_head (read at register())
        os.environ["VLLM_FRSPEC_NVFP4"] = frspec


def build_llm(model_dir, max_model_len=4096, spec_mtp=3):
    """Construct the LLM with the verified recipe. spec_mtp=None/0 -> no speculative decode."""
    from vllm import LLM
    extra = {}
    if spec_mtp:
        extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(spec_mtp)}
        # FULL cudagraph captures the whole K+1 verify in one graph (no PIECEWISE attn breaks);
        # capture sizes cover the single-token draft (1) and the verify batch (K+1).
        extra["compilation_config"] = {
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "cudagraph_copy_inputs": True,
            "cudagraph_mode": "FULL",
        }
    return LLM(
        model=model_dir, trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.92")),
        max_model_len=max_model_len, enforce_eager=False,
        max_num_seqs=1, kv_cache_dtype="fp8", **extra,
    )
