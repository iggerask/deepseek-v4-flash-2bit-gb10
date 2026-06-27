"""vq2 quantization method for vLLM — self-contained DeepSeek-V4-Flash 2-bit serving.

Registered via the ``vllm.general_plugins`` entrypoint (see pyproject.toml), so plain
``vllm serve <model_dir>`` loads it with the standard mmap weight loader — NO runtime
monkeypatch of the loader, NO sidecar .pt, NO CPU+GPU double-load.

Recipe (config.json quantization_config, quant_method="vq2"):
  - routed experts  -> vq2 2-bit VQ (k=1024, vdim=4, group=64, 10-bit byte-plane)
  - attn wq_a/wkv/wq_b, wo_b, shared_experts -> NVFP4 W4A16 (modelopt Marlin)
  - attn wo_a -> NVFP4, served via a custom grouped W4A16 o_proj (see below)
  - everything else (compressor, indexer, norms, router gate, embed, lm_head) -> BF16

o_proj fix: vLLM's DeepseekV4FlashMLAAttention._o_proj hardcodes a fused FP8 deep-gemm
(reads wo_a.weight_scale_inv) — incompatible with our NVFP4 wo_a. We do NOT downgrade to
FP8. Instead wo_a is kept as RAW NVFP4 (no Marlin repack) and we override _o_proj to do
the grouped W4A16 projection ourselves: torch inverse-RoPE + unpack-NVFP4(wo_a) + grouped
einsum + native NVFP4 wo_b. wo_b is a normal linear (Marlin W4A16 works as-is).

The vq2 kernels are bundled in vq2_vllm/kernels/ and put on sys.path so their bare-name
cross-imports (import vq2_kernel, import moe_align, ...) resolve. Override with VQ2_TOOLS.
"""
import os
import sys

import torch

# WMMA grouped GEMM for prefill (M >= _WMMA_MIN). Off -> per-pair everywhere.
_WMMA_PREFILL = os.environ.get("VQ2_WMMA_PREFILL", "1") == "1"
_WMMA_MIN = int(os.environ.get("VQ2_WMMA_MIN", "256"))
_WMMA_NN = int(os.environ.get("VQ2_NN", "2"))
_WMMA_BKC = int(os.environ.get("VQ2_BKC", "64"))
_WMMA_NW = int(os.environ.get("VQ2_NW", "8"))

_TOOLS = os.environ.get(
    "VQ2_TOOLS",
    os.path.join(os.path.dirname(__file__), "kernels"),
)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

# E2M1 signed levels for NVFP4 unpack
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_DEV = {}  # device -> GPU copy; avoids CPU->CUDA copy during CUDA-graph capture


def _e2m1(dev):
    t = _E2M1_DEV.get(dev)
    if t is None:
        t = _E2M1.to(dev)
        _E2M1_DEV[dev] = t
    return t


def _unpack_nvfp4(packed, weight_scale, weight_scale_2, group=16):
    """Reconstruct bf16 [out, in] from raw modelopt NVFP4 tensors.
    packed [out, in/2] uint8 (2 nibbles/byte along input, low=even idx);
    weight_scale [out, in/group] fp8-e4m3 per-group micro-scale;
    weight_scale_2 scalar f32 global. value = E2M1[idx]*sign * (wscale*gs)."""
    out, half = packed.shape
    cin = half * 2
    dev = packed.device
    code = torch.empty(out, cin, dtype=torch.uint8, device=dev)
    code[:, 0::2] = packed & 0xF
    code[:, 1::2] = packed >> 4
    idx = (code & 0x7).long()
    sign = (code >> 3) & 1
    lvl = _e2m1(dev)
    mag = lvl[idx]
    val = torch.where(sign > 0, -mag, mag)                       # [out, cin]
    gs = weight_scale_2.float().reshape(())
    s_eff = weight_scale.float().view(out, cin // group, 1) * gs  # [out, ng, 1]
    W = (val.view(out, cin // group, group) * s_eff).view(out, cin)
    return W.to(torch.bfloat16)


def _inv_rope(o, positions, cos_sin_cache, rope_dim):
    """Inverse RoPE on the last ``rope_dim`` dims of o [T, H, D], interleaved pairs
    (matches fused_inv_rope_fp8_quant): out[2i]=x[2i]cos+x[2i+1]sin,
    out[2i+1]=x[2i+1]cos-x[2i]sin. cos_sin_cache[pos] = [cos(rope/2) || sin(rope/2)]."""
    half = rope_dim // 2
    cs = cos_sin_cache[positions].float()                # [T, rope_dim]
    cos = cs[:, None, :half]                              # [T, 1, half]
    sin = cs[:, None, half:rope_dim]
    out = o.clone()
    rope = o[..., -rope_dim:].float()
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    out[..., -rope_dim::2] = (even * cos + odd * sin).to(o.dtype)
    out[..., -rope_dim + 1::2] = (odd * cos - even * sin).to(o.dtype)
    return out


def _o_proj_nvfp4(self, o, positions):
    """Override of DeepseekV4FlashMLAAttention._o_proj: grouped W4A16 (NVFP4 wo_a)
    instead of the fused FP8 deep-gemm. wo_a kept raw NVFP4, wo_b native NVFP4."""
    G = self.n_local_groups
    rank = self.o_lora_rank
    D = self.head_dim
    T = o.shape[0]
    hpg = self.n_local_heads // G
    import ds4_oproj_triton as op
    o_rot = _inv_rope(o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
    wa = self.wo_a
    o_g = o_rot.reshape(T, G, hpg * D)                            # [T,G,4096]
    # SMALL T (decode T=1, AND spec-decode VERIFY T=K+1): read 4-bit wo_a directly via the Triton
    # kernel (it loops T internally). The dequant-once PyTorch path below only wins for large-T prefill;
    # at the verify's small T it was re-unpacking wo_a [8192,4096] in PyTorch EVERY layer/step (where/
    # copy/elementwise soup = ~14% of the whole spec run). Crossover ~T=8 (4-bit re-read vs unpack+GEMM).
    if T <= int(os.environ.get("VQ2_OPROJ_DIRECT_MAX", "8")):
        z = op.oproj_wa(o_g, wa.weight, wa.weight_scale, wa.weight_scale_2, G, rank)  # [T,G,rank]
    else:
        # PREFILL: dequant wo_a once -> batched GEMM (weight reused across T tokens;
        # W4A16's standard prefill form — 4-bit stays in storage, transient matmul only).
        Wbf = _unpack_nvfp4(wa.weight, wa.weight_scale, wa.weight_scale_2).view(G, rank, hpg * D)
        z = torch.einsum("tgk,grk->tgr", o_g.to(Wbf.dtype), Wbf)  # [T,G,rank]
    out = self.wo_b(z.reshape(T, G * rank).to(o.dtype))
    return out[0] if isinstance(out, tuple) else out


def _o_proj_bf16(self, o, positions):
    """BF16 o_proj (for VQ2_BF16_REST A/B): wo_a/wo_b are plain BF16 (no NVFP4 unpack, no FP8 deep-gemm).
    Same inverse-RoPE + grouped wo_a + wo_b structure as _o_proj_nvfp4."""
    G = self.n_local_groups; rank = self.o_lora_rank; D = self.head_dim
    T = o.shape[0]; hpg = self.n_local_heads // G
    o_rot = _inv_rope(o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
    o_g = o_rot.reshape(T, G, hpg * D)
    Wbf = self.wo_a.weight.reshape(G, rank, hpg * D).to(o.dtype)
    z = torch.einsum("tgk,grk->tgr", o_g.to(Wbf.dtype), Wbf)
    out = self.wo_b(z.reshape(T, G * rank).to(o.dtype))
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# vq2 MoE method
# ---------------------------------------------------------------------------
def _make_vq2_moe_method():
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
    from vllm.model_executor.utils import set_weight_attrs

    class Vq2MoEMethod(FusedMoEMethodBase):
        """Routed-expert method: shared codebook + 10-bit byte-plane VQ indices."""

        def __init__(self, quant_config, moe):
            super().__init__(moe)
            self.quant_config = quant_config

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            vdim = self.quant_config.vdim
            k = self.quant_config.k
            group = self.quant_config.group
            spg = group // vdim
            H = hidden_size
            I = intermediate_size_per_partition
            E = num_experts
            nsub_gu = H // vdim
            nsub_dn = I // vdim

            def P(shape, dtype):
                return torch.nn.Parameter(torch.empty(*shape, dtype=dtype),
                                          requires_grad=False)

            params = {
                "w13_lo": P((E, 2 * I, nsub_gu), torch.uint8),
                "w13_hi": P((E, 2 * I, nsub_gu // 4), torch.uint8),
                "w13_sc": P((E, 2 * I, nsub_gu // spg), torch.float16),
                "w2_lo": P((E, H, nsub_dn), torch.uint8),
                "w2_hi": P((E, H, nsub_dn // 4), torch.uint8),
                "w2_sc": P((E, H, nsub_dn // spg), torch.float16),
                "w13_cb": P((k, vdim), torch.float16),
                "w2_cb": P((k, vdim), torch.float16),
            }
            for name, p in params.items():
                layer.register_parameter(name, p)
                set_weight_attrs(p, {"weight_loader": self._weight_loader})

        @staticmethod
        def _weight_loader(param, loaded_weight, weight_name, shard_id,
                          expert_id, return_success=False):
            lw = loaded_weight.to(param.device)
            if weight_name.endswith("cb"):
                param.data.copy_(lw.to(param.dtype))
            elif shard_id == "w2":
                param.data[expert_id].copy_(lw.to(param.dtype))
            else:
                half = param.shape[1] // 2
                if shard_id == "w1":
                    param.data[expert_id, :half].copy_(lw.to(param.dtype))
                else:
                    param.data[expert_id, half:].copy_(lw.to(param.dtype))
            return True if return_success else None

        def get_fused_moe_quant_config(self, layer):
            return None

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                 shared_experts_input):
            import vq2_kernel as vq2
            # Prefill (large token batch, NOT cudagraph-captured) -> WMMA tensor-core grouped
            # GEMM (~3x at 2048+ tok, validated rel 5e-4 on real weights). Decode (small M,
            # cudagraphed) -> per-pair fused kernel (graph-safe, no moe_align/temps).
            if _WMMA_PREFILL and x.shape[0] >= _WMMA_MIN:
                from ds4_vq2_cuda import vq2_moe_grouped_wmma
                out = vq2_moe_grouped_wmma(
                    x, topk_ids, topk_weights,
                    layer.w13_cb, layer.w13_lo, layer.w13_hi, layer.w13_sc,
                    layer.w2_cb, layer.w2_lo, layer.w2_hi, layer.w2_sc,
                    group=self.quant_config.group, NN=_WMMA_NN, BKC=_WMMA_BKC, NW=_WMMA_NW,
                )
            else:
                out = vq2.vq2_moe_fused_10b(
                    x, topk_ids, topk_weights,
                    layer.w13_cb, layer.w13_lo, layer.w13_hi, layer.w13_sc,
                    layer.w2_cb, layer.w2_lo, layer.w2_hi, layer.w2_sc,
                    group=self.quant_config.group,
                )
            return out.to(x.dtype)

        def process_weights_after_loading(self, layer):
            # FAST LOAD: if a fused per-layer expert file exists (NOT in the model index, so vLLM
            # skipped the ~99k per-expert tensors), load the 8 big [E,...] tensors directly here.
            import os, re
            from safetensors import safe_open
            ln = getattr(layer, "layer_name", "") or getattr(layer, "prefix", "")
            m = re.search(r"layers\.(\d+)\.", str(ln))
            if m is None:
                return
            d = os.environ.get("VQ2_EXPERTS_DIR", "models/DeepSeek-V4-Flash-2bit")
            path = f"{d}/experts_fused_layer_{int(m.group(1))}.safetensors"
            if not os.path.exists(path):
                return                                   # fall back to per-expert path (if indexed)
            f = safe_open(path, "pt", device=str(layer.w13_lo.device))
            for name in ("w13_lo", "w13_hi", "w13_sc", "w2_lo", "w2_hi", "w2_sc", "w13_cb", "w2_cb"):
                getattr(layer, name).data.copy_(f.get_tensor(name))
            # VQ2_COARSEN_K=K': collapse the shared codebook to K' EFFECTIVE centroids in-memory
            # (disk-free k-sensitivity probe; usage is ~uniform so unweighted is faithful). Reconstruction
            # becomes ~k=K' WITHOUT rewriting the 97GB indices -> gives the missing rel-error->PPL link.
            ck = int(os.environ.get("VQ2_COARSEN_K", "0"))
            lr = os.environ.get("VQ2_COARSEN_LAYERS", "")   # "a-b" to coarsen only that layer range
            in_range = True
            if lr:
                a_, b_ = lr.split("-"); in_range = int(a_) <= int(m.group(1)) <= int(b_)
            if 0 < ck < getattr(layer, "w13_cb").shape[0] and in_range:
                import torch
                for cbname in ("w13_cb", "w2_cb"):
                    C = getattr(layer, cbname).data; Cf = C.float()
                    g = torch.Generator(device=Cf.device).manual_seed(0)
                    sub = Cf[torch.randperm(Cf.shape[0], generator=g, device=Cf.device)[:ck]].clone()
                    for _ in range(25):
                        a = torch.cdist(Cf, sub).argmin(1)
                        sums = torch.zeros_like(sub).index_add_(0, a, Cf)
                        cnts = torch.zeros(ck, device=Cf.device).index_add_(0, a, torch.ones(Cf.shape[0], device=Cf.device))
                        nz = cnts > 0; sub[nz] = sums[nz] / cnts[nz, None]
                    a = torch.cdist(Cf, sub).argmin(1)
                    C.copy_(sub[a].to(C.dtype))            # 1024 rows, <=K' distinct values
                print(f"[vq2] layer {int(m.group(1))}: codebook coarsened to K'={ck} (probe; range={lr or 'all'})", flush=True)

    return Vq2MoEMethod


# vLLM module-prefix substrings that are NVFP4 W4A16 Marlin (wo_a handled separately).
NVFP4_LINEARS = (
    "attn.fused_wqa_wkv",
    "attn.wq_b",
    "attn.wo_b",
    ".ffn.shared_experts.gate_up_proj",
    ".ffn.shared_experts.down_proj",
)


class Vq2Config:  # replaced by a real QuantizationConfig subclass in _bind_base
    pass


def _bind_base():
    global Vq2Config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.layers.fused_moe import FusedMoE, RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4W4A16LinearMethod,
    )

    Vq2MoEMethod = _make_vq2_moe_method()

    class Vq2RawNvFp4(ModelOptNvFp4W4A16LinearMethod):
        """NVFP4 W4A16 that keeps RAW packed weights (no Marlin repack) so the
        custom o_proj can unpack wo_a itself."""
        def process_weights_after_loading(self, layer):
            if hasattr(layer, "input_scale"):
                del layer.input_scale  # placeholder, unused in W4A16
            # keep weight / weight_scale / weight_scale_2 raw (no repack)

        def apply(self, *a, **k):
            raise RuntimeError("Vq2RawNvFp4.apply must not run; wo_a is served by _o_proj")

    class _Vq2Config(QuantizationConfig):
        def __init__(self, vdim, k, group, full_config):
            super().__init__()
            self.vdim = vdim
            self.k = k
            self.group = group
            self.full_config = full_config
            self._nvfp4 = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )

        @classmethod
        def get_name(cls):
            return "vq2"

        @classmethod
        def get_supported_act_dtypes(cls):
            return [torch.bfloat16, torch.half]

        @classmethod
        def get_min_capability(cls):
            return 80

        @staticmethod
        def get_config_filenames():
            return []

        @classmethod
        def from_config(cls, config):
            return cls(
                vdim=int(config.get("vdim", 4)),
                k=int(config.get("k", 1024)),
                group=int(config.get("group", 64)),
                full_config=config,
            )

        def get_quant_method(self, layer, prefix):
            if isinstance(layer, (FusedMoE, RoutedExperts)):
                return Vq2MoEMethod(self, layer.moe_config)   # experts stay vq2 (the A/B holds them fixed)
            _bf16 = os.environ.get("VQ2_BF16_REST") == "1"
            if isinstance(layer, LinearBase):
                if _bf16:                               # A/B: attn + shared served as plain BF16
                    return UnquantizedLinearMethod()
                if "attn.wo_a" in prefix:               # raw NVFP4, served by _o_proj
                    return Vq2RawNvFp4(self._nvfp4)
                if any(s in prefix for s in NVFP4_LINEARS):
                    return ModelOptNvFp4W4A16LinearMethod(self._nvfp4)
                return UnquantizedLinearMethod()
            if "lm_head" in prefix:                     # ParallelLMHead -> NVFP4 W4A16 (per recipe)
                if _bf16:
                    return UnquantizedLinearMethod()
                return ModelOptNvFp4W4A16LinearMethod(self._nvfp4)
            return None

    Vq2Config = _Vq2Config
    return _Vq2Config


def _patch_o_proj():
    """Override the FlashMLA o_proj to use NVFP4 wo_a (no fused FP8 path); BF16 variant under VQ2_BF16_REST."""
    try:
        from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
        bf16 = os.environ.get("VQ2_BF16_REST") == "1"
        DeepseekV4FlashMLAAttention._o_proj = _o_proj_bf16 if bf16 else _o_proj_nvfp4
        print(f"[vq2] FlashMLA _o_proj patched ({'BF16' if bf16 else 'NVFP4'})", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: could not patch FlashMLA _o_proj: {e}", flush=True)


def _patch_lm_head():
    if os.environ.get("VQ2_BF16_REST") == "1":      # lm_head served as plain BF16 (no NVFP4 injection)
        print("[vq2] BF16_REST: lm_head left unquantized (BF16)", flush=True)
        return
    """DS4 builds ParallelLMHead WITHOUT a quant_config, so lm_head defaults to
    unquantized. Inject our vq2 config so lm_head picks up NVFP4 (LogitsProcessor
    applies it via lm_head.quant_method.apply -> W4A16 Marlin)."""
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        _orig = ParallelLMHead.__init__

        def _init(self, *a, **k):
            if k.get("quant_config") is None and "quant_config" not in k:
                try:
                    from vllm.config import get_current_vllm_config
                    qc = get_current_vllm_config().quant_config
                    if qc is not None and qc.get_name() == "vq2":
                        k["quant_config"] = qc
                except Exception:
                    pass
            _orig(self, *a, **k)

        ParallelLMHead.__init__ = _init
        print("[vq2] ParallelLMHead quant_config injection enabled (NVFP4 lm_head)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: could not patch ParallelLMHead: {e}", flush=True)


def _patch_sparse_attn():
    """Replace the sm_90a/sm_100f-only sparse-MLA forward kernels with sm_121 torch
    reference (vllm/_flashmla_C has no sm_12x build). Patches the names in the
    flashmla module namespace where _forward_decode/_forward_prefill resolve them."""
    try:
        import vllm.models.deepseek_v4.nvidia.flashmla as fm
        import vllm.models.deepseek_v4.sparse_mla as smla
        import ds4_sparse_torch as st
        import ds4_sparse_triton as stri
        fm.flash_mla_with_kvcache = stri.flash_mla_with_kvcache    # decode -> Triton (graph-safe, fast)
        fm.flash_mla_sparse_fwd = stri.flash_mla_sparse_fwd        # prefill -> Triton flash (9x faster than torch ref)
        # ensure backend selection doesn't reject sm_121 (major 12) before our code runs
        smla.DeepseekV4FlashMLABackend.supports_compute_capability = classmethod(
            lambda cls, capability: True)
        print("[vq2] sparse-MLA forward patched (torch sm_121 reference)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: could not patch sparse-MLA forward: {e}", flush=True)
    # lightning-indexer logit kernels (DeepGEMM fp8_fp4_*_mqa_logits gated to sm_90/100)
    try:
        import vllm.model_executor.layers.sparse_attn_indexer as sai
        import ds4_indexer_torch as it
        sai.fp8_fp4_mqa_logits = it.fp8_fp4_mqa_logits
        sai.fp8_fp4_paged_mqa_logits = it.fp8_fp4_paged_mqa_logits
        import vllm.v1.attention.backends.mla.indexer as idxmod
        idxmod.get_paged_mqa_logits_metadata = it.get_paged_mqa_logits_metadata
        print("[vq2] indexer logits patched (torch sm_121 reference)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: could not patch indexer logits: {e}", flush=True)


try:
    _bind_base()
except Exception:
    pass


def _patch_mhc_torch():
    """Route the MTP drafter's mhc_post through the torch reference instead of TileLang.
    The plugin already swaps sparse-MLA + indexer to torch sm_121 references (above), but
    mhc_post was missed: its TileLang kernel hits an illegal memory access during the
    drafter's cudagraph capture on sm_121, crashing spec decode. mhc_post_torch has an
    identical signature (pure einsum+mul+add => capture-safe). VQ2_MHC_TORCH=0 opts out."""
    if os.environ.get("VQ2_MHC_TORCH", "1") != "1":
        return
    try:
        import vllm.models.deepseek_v4.nvidia.mtp as mtp
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
        mtp.mhc_post_tilelang = mhc_post_torch
        print("[vq2] mhc_post patched (torch sm_121 reference)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: could not patch mhc_post: {e}", flush=True)


def _patch_drafter_full_cudagraph():
    """Let the spec-decode DRAFTER use FULL cudagraph (vLLM hard-forces it PIECEWISE-only in
    SpecDecodeBaseProposer.initialize_cudagraph_keys). On sm_121 the DS4 attention runs via graph-safe
    torch refs that the target FULL-captures fine (=> non-spec 19.99 tok/s); PIECEWISE breaks the graph
    at every attn op so the single-token MTP draft runs ~eager (~182ms x K). The BreakableCUDAGraphWrapper
    that wraps the drafter already supports FULL; it just never gets FULL dispatch keys. Give it FULL.
    In-place draft metadata updates are cudagraph-replay-safe; size-1 draft needs no padding."""
    import os
    if os.environ.get("VQ2_DRAFTER_FULL_CG", "0") != "1":
        return
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        from vllm.config.compilation import CUDAGraphMode
    except Exception:
        return
    if getattr(SpecDecodeBaseProposer, "_vq2_full_cg", False):
        return
    def _init(self, cudagraph_mode):
        if not self.speculative_config.enforce_eager and cudagraph_mode.has_full_cudagraphs():
            m = CUDAGraphMode.FULL
        elif not self.speculative_config.enforce_eager and cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
            m = CUDAGraphMode.PIECEWISE
        else:
            m = CUDAGraphMode.NONE
        self.cudagraph_dispatcher.initialize_cudagraph_keys(m)
    SpecDecodeBaseProposer.initialize_cudagraph_keys = _init
    SpecDecodeBaseProposer._vq2_full_cg = True


def _patch_workspace_overalloc():
    """Shrink the sparse-MLA + lightning-indexer PREFILL workspaces that vLLM sizes by
    max_model_len, not by the actual prefill token budget. flashmla_sparse uses 5*max_model_len
    (get_prefill_workspace_size) and the indexer uses 40*max_model_len (get_max_prefill_buffer_size)
    -- both untuned 'magic' multipliers that assume ">2GB of free MoE workspace so it's free". That
    is FALSE for the 2-bit model on a 128GB box: a ~95GB model + a KV pool leaves only single-digit
    GB, and those workspaces are ~GBs of pure slack that ate the headroom and OOM-froze long-context
    prefill (KV itself had room -- not the wall). SAFE to shrink: flashmla's split_prefill_chunks()
    sizes prefill chunks to FIT the workspace, so a smaller workspace just means more/smaller chunks
    (identical result); the indexer workspace stays >= max_model_len for any div<=~10. Gated by
    VQ2_WORKSPACE_DIV (divisor on the multiplier; default 1 = upstream behavior; 3 frees ~2GB).
    Needed only for long-context serving (max_model_len well above the default)."""
    import os
    try:
        div = float(os.environ.get("VQ2_WORKSPACE_DIV", "1"))
    except ValueError:
        div = 1.0
    if div <= 1.0:
        return
    try:
        import vllm.v1.attention.backends.mla.flashmla_sparse as fms
        import vllm.v1.attention.backends.mla.indexer as idx
        _o_fms = fms.get_prefill_workspace_size
        _o_idx = idx.get_max_prefill_buffer_size

        def _fms(max_model_len, _o=_o_fms, _d=div):
            return max(1, int(_o(max_model_len) / _d))

        def _idx(vllm_config, _o=_o_idx, _d=div):
            return max(1, int(_o(vllm_config) / _d))

        fms.get_prefill_workspace_size = _fms
        idx.get_max_prefill_buffer_size = _idx
        print(f"[vq2] prefill workspace over-alloc shrunk {div}x "
              f"(flashmla 5x->/{div}, indexer 40x->/{div}; VQ2_WORKSPACE_DIV)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: could not shrink prefill workspace over-alloc: {e}", flush=True)


def _patch_adaptive_prefill():
    """ADAPTIVE prefill chunk sizing for single-stream long context.

    Every prefill chunk re-runs the full MoE (compute/gather-bound on the 2-bit codebook
    experts), so prefill time scales with the NUMBER of chunks. A fixed small max_num_batched_tokens
    (needed to bound the per-chunk memory transient at very long context) creates too many chunks.
    This sets long_prefill_token_threshold per scheduler pass = BUDGET / (context_so_far * BPU),
    clamped to [MINCHUNK, MAXCHUNK]: large chunks early (context small -> small transient), shrinking
    only as context approaches the headroom ceiling -> far fewer chunks at no extra peak memory.
    Single-stream (max-num-seqs 1): the one in-prefill request's num_computed_tokens IS the
    context-so-far; decode is unaffected (it schedules 1 token). Gated by VQ2_ADAPTIVE_PREFILL=1;
    tune with VQ2_ADAPT_BUDGET / VQ2_ADAPT_BPU / VQ2_ADAPT_MAXCHUNK / VQ2_ADAPT_MINCHUNK. Requires
    --max-num-batched-tokens >= MAXCHUNK. Pair with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    (the long-ctx prefill transient is dominated by allocator fragmentation, not a single tensor)."""
    import os
    if os.environ.get("VQ2_ADAPTIVE_PREFILL", "0") != "1":
        return
    budget = float(os.environ.get("VQ2_ADAPT_BUDGET", "5.0e9"))
    bpu = float(os.environ.get("VQ2_ADAPT_BPU", "6.2"))
    max_chunk = int(os.environ.get("VQ2_ADAPT_MAXCHUNK", "16384"))
    min_chunk = int(os.environ.get("VQ2_ADAPT_MINCHUNK", "512"))
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN adaptive prefill: cannot import Scheduler: {e}", flush=True)
        return
    _orig = Scheduler.schedule

    def _adaptive(self, _o=_orig, _b=budget, _bpu=bpu, _mx=max_chunk, _mn=min_chunk):
        try:
            ctx = 0
            for r in self.running:
                if r.num_computed_tokens < r.num_prompt_tokens:  # still ingesting prompt
                    if r.num_computed_tokens > ctx:
                        ctx = r.num_computed_tokens
            chunk = int(_b / (ctx * _bpu)) if ctx > 0 else _mx
            if chunk < _mn:
                chunk = _mn
            elif chunk > _mx:
                chunk = _mx
            self.scheduler_config.long_prefill_token_threshold = chunk
        except Exception:
            pass
        return _o(self)

    Scheduler.schedule = _adaptive
    print(f"[vq2] ADAPTIVE prefill ON (budget={budget:.2g}B bpu={bpu} "
          f"chunk in [{min_chunk},{max_chunk}]; VQ2_ADAPTIVE_PREFILL)", flush=True)


def _patch_frspec():
    """FR-Spec: restrict the MTP DRAFT lm_head to a frequency-ranked shortlist (top-N
    most frequent tokens). The TARGET still verifies full-vocab, so quality is EXACT —
    rare tokens simply aren't drafted; the only cost is a tiny acceptance dip when the
    true token is outside the shortlist. The draft's lm_head is its single biggest matmul
    (full 129,280-vocab, ~265 MB read even at 4-bit NVFP4 W4A16); restricting it cuts the
    dominant draft byte cost.

    Gated by VLLM_FRSPEC_NVFP4 -> frspec_nvfp4_ds4.pt (shipped in the model repo):
    the shortlist rows of the target's NVFP4 lm_head, KEPT in NVFP4 so the reduced draft
    logits are byte-identical to the full draft on those rows -> acceptance preserved.

    Wraps DeepSeekMTP.load_weights (build the reduced 4-bit Marlin lm_head ONCE, outside
    cudagraph capture) + compute_logits (use it, applying shared_head's RMSNorm first).
    The proposer's _greedy_sample already remaps shortlist idx -> vocab id via
    getattr(self.model, '_frspec_ids')."""
    nv = os.environ.get("VLLM_FRSPEC_NVFP4")
    if not nv:
        return
    try:
        from vllm.models.deepseek_v4.nvidia.mtp import (
            DeepSeekV4MTP, hc_head_fused_kernel_tilelang, mtp_shared_head_rmsnorm)
    except Exception as e:  # pragma: no cover
        print(f"[vq2] WARN: FR-Spec import failed: {e}", flush=True); return
    if getattr(DeepSeekV4MTP, "_vq2_frspec", False):
        return

    _orig_load = DeepSeekV4MTP.load_weights
    _orig_cl = DeepSeekV4MTP.compute_logits

    def _build(self):
        if getattr(self, "_frspec_method", None) is not None:
            return
        d = torch.load(nv)
        cur = self.model.mtp_start_layer_idx
        head = self.model.layers[str(cur)].shared_head.head      # ParallelLMHead (NVFP4)
        dev = head.weight.device
        N, packed = d["weight"].shape
        hid = packed * 2
        # shared_head.head.quant_method is UnquantizedEmbeddingMethod: its prefix
        # "...shared_head.head" lacks "lm_head", so vq2's get_quant_method returns None
        # for it (hence "'UnquantizedEmbeddingMethod' has no attribute quant_config" at
        # load). Clone the SAME NVFP4 W4A16 method the recipe gives lm_head from any
        # NVFP4 linear already built in this MTP model (its attn projections qualify).
        qm = None
        for _m in self.model.modules():
            _q = getattr(_m, "quant_method", None)
            if type(_q).__name__ == "ModelOptNvFp4W4A16LinearMethod" \
                    and getattr(_q, "quant_config", None) is not None:
                qm = _q
                break
        if qm is None:
            raise RuntimeError("FR-Spec: no ModelOptNvFp4W4A16LinearMethod to clone for draft head")
        method = type(qm)(qm.quant_config)
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16                      # read by prepare_fp4_layer_for_marlin
        with torch.device(dev):
            method.create_weights(layer, input_size_per_partition=hid,
                                  output_partition_sizes=[N], input_size=hid, output_size=N,
                                  params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
        layer.weight.data.copy_(d["weight"].to(dev))
        layer.weight_scale.data.copy_(d["weight_scale"].to(dev))
        layer.weight_scale_2.data.copy_(d["weight_scale_2"].to(dev).reshape(1))
        method.process_weights_after_loading(layer)
        self._frspec_method = method
        self._frspec_layer = layer
        self._frspec_ids = d["ids"].to(dev).long()
        print(f"[FRSPEC-ds4] reduced 4-bit Marlin draft lm_head N={N} hid={hid} "
              f"({d['weight'].numel()/1e6:.0f} MB/draft vs {head.weight.numel()/1e6:.0f} MB full)",
              flush=True)

    def load_weights(self, weights):
        loaded = _orig_load(self, weights)
        try:
            _build(self)
        except Exception as e:  # pragma: no cover  -> full lm_head fallback, still correct
            print(f"[vq2] WARN: FR-Spec build failed (full lm_head fallback): {e}", flush=True)
        return loaded

    def compute_logits(self, hidden_states, spec_step_idx=0):
        m = getattr(self, "_frspec_method", None)
        if m is None and not getattr(self, "_frspec_tried", False):
            self._frspec_tried = True            # lazy build if load_weights wasn't our hook
            try:
                _build(self)
            except Exception as e:  # pragma: no cover -> full lm_head fallback
                print(f"[vq2] WARN: FR-Spec lazy build failed (full lm_head): {e}", flush=True)
            m = getattr(self, "_frspec_method", None)
        if m is None:
            return _orig_cl(self, hidden_states, spec_step_idx)
        # Replicate the DS4 pre-lm_head pipeline (hc_head fused kernel + shared rmsnorm),
        # then project through the REDUCED lm_head instead of the full 129K-vocab one.
        pred = self.model
        c = spec_step_idx % pred.num_mtp_layers
        L = pred.layers[str(pred.mtp_start_layer_idx + c)]
        h = hidden_states.view(-1, L.hc_mult, L.config.hidden_size)
        h = hc_head_fused_kernel_tilelang(h, L.hc_head_fn, L.hc_head_scale,
                                          L.hc_head_base, L.rms_norm_eps, L.hc_eps)
        h = mtp_shared_head_rmsnorm(h, L.shared_head.norm.weight.data,
                                    L.shared_head.norm.variance_epsilon)
        return m.apply(self._frspec_layer, h)

    DeepSeekV4MTP.load_weights = load_weights
    DeepSeekV4MTP.compute_logits = compute_logits
    DeepSeekV4MTP._vq2_frspec = True
    print("[vq2] FR-Spec draft lm_head patch enabled (VLLM_FRSPEC_NVFP4)", flush=True)


def register():
    from vllm.model_executor.layers.quantization import register_quantization_config

    cls = Vq2Config
    if not isinstance(cls, type) or cls.__name__ == "Vq2Config":
        cls = _bind_base()
    register_quantization_config("vq2")(cls)
    _patch_o_proj()
    _patch_lm_head()
    _patch_sparse_attn()
    _patch_mhc_torch()             # sm_121: mhc_post TileLang illegal-access under cudagraph -> torch ref (VQ2_MHC_TORCH)
    _patch_workspace_overalloc()   # long-ctx: shrink max_model_len-sized prefill workspaces (VQ2_WORKSPACE_DIV)
    _patch_adaptive_prefill()      # long-ctx: adaptive prefill chunk sizing (VQ2_ADAPTIVE_PREFILL)
    _patch_drafter_full_cudagraph()
    _patch_frspec()
    # NOTE: there is NO compressor KV "fix" here. An earlier session mis-diagnosed a compress_ratio=1
    # over-allocation; it was a measurement artifact (UniformTypeKVCacheSpecs group-wrapper inflation +
    # reading "33k tokens" — the max_model_len=2048 concurrency pool — as a per-token cost). DS4-Flash
    # KV is ~27KB/token: VERIFIED 392,011 tokens single-stream at max_model_len=131072 in ~10GB. Long
    # context is gated ONLY by the serve's max_model_len, not any cache bug.
