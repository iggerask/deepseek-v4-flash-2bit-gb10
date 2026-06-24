// Grouped vq2 MoE for PREFILL on GB10 (sm_121). Memory-bound. KEY: activations (xh) are
// shared across all output rows, so load them into SHARED once per chunk-tile and reuse
// across the RW warps (the per-warp re-read was ~178x the index traffic). Codebook in
// shared. Index read coalesced (uint per lane = 128B/warp). No tensor cores (DRAM-bound).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;

#define WARP 32
#define CT 128                  // chunks per tile (lane reads 4 -> uint, 128B/warp coalesced)

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// shared layout: [ cb (1024*VD half) | xs (BM*CT*VD float) ]
// gate_up+silu. xh [N,H]; sorted_tok [Pmax]; be [nblk]; Cgu [K,VD]; lo [E,2I,nsg];
// hi [E,2I,nsg/4]; sgu [E,2I,ng]; out (inter) [Pmax,I].
template <int BM, int VD, int SPG, int RW>
__global__ void __launch_bounds__(256, 3) grp_gateup(
    const float* __restrict__ xh, const int* __restrict__ sorted_tok,
    const int* __restrict__ be, const __half* __restrict__ Cgu,
    const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi, const __half* __restrict__ sgu,
    float* __restrict__ out, int N, int I, int nsg, int ng, int H) {
  int pid_b = blockIdx.x;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int i = blockIdx.y * RW + warp;               // output row (gate=i, up=I+i)
  int e = be[pid_b];
  extern __shared__ __half smem[];
  __half* cb = smem;                            // 1024*VD halfs
  float* xs = (float*)(cb + 1024 * VD);         // BM*CT*VD floats
  for (int t = threadIdx.x; t < 1024 * VD; t += blockDim.x) cb[t] = Cgu[t];
  int toks[BM]; float gacc[BM]; float uacc[BM];
#pragma unroll
  for (int b = 0; b < BM; b++) { toks[b] = sorted_tok[pid_b * BM + b]; gacc[b] = 0.f; uacc[b] = 0.f; }
  bool act = (i < I) && (e >= 0);
  long base_g = ((long)e * 2 * I + i) * nsg;
  long base_u = ((long)e * 2 * I + (I + i)) * nsg;
  long sbase_g = ((long)e * 2 * I + i) * ng;
  long sbase_u = ((long)e * 2 * I + (I + i)) * ng;
  const int CV = CT * VD;
  for (int t0 = 0; t0 < nsg; t0 += CT) {
    // cooperative load xs[b][k] = xh[tok[b], t0*VD + k]  (once per block, reused by RW warps)
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * CV; idx += blockDim.x) {
      int b = idx / CV, k = idx % CV; int tk = toks[b];
      xs[idx] = (tk >= 0) ? xh[(long)tk * H + (long)t0 * VD + k] : 0.f;
    }
    __syncthreads();
    if (!act) continue;
    // STRIDED chunk assignment: lane handles chunks c = lane, lane+32, lane+64, lane+96.
    // -> for a fixed cc, the xs float4 reads stride by 32 chunks (128 floats) so across the
    //    warp they tile all 32 banks each phase (4-phase floor, conflict-free) instead of the
    //    16-way conflict the contiguous (4*lane..) layout produced.
    for (int cc = 0; cc < 4; cc++) {
      int c = lane + cc * 32;                       // global chunk in this tile
      int idg = lo[base_g + t0 + c] | (((hi[base_g / 4 + (t0 + c) / 4] >> ((c & 3) * 2)) & 3) << 8);
      int idu = lo[base_u + t0 + c] | (((hi[base_u / 4 + (t0 + c) / 4] >> ((c & 3) * 2)) & 3) << 8);
      float scg = __half2float(sgu[sbase_g + (t0 + c) / SPG]);
      float scu = __half2float(sgu[sbase_u + (t0 + c) / SPG]);
      // WIDE shared loads: codebook as half2 (2 loads + vectorized convert vs 4 scalar)
      float2 g0 = __half22float2(*(const __half2*)(cb + idg * VD));
      float2 g1 = __half22float2(*(const __half2*)(cb + idg * VD + 2));
      float2 u0 = __half22float2(*(const __half2*)(cb + idu * VD));
      float2 u1 = __half22float2(*(const __half2*)(cb + idu * VD + 2));
      int xk = c * VD;
      for (int b = 0; b < BM; b++) {
        const float4 x4 = *(const float4*)(xs + b * CV + xk);   // conflict-free (4-phase floor)
        gacc[b] += (g0.x * x4.x + g0.y * x4.y + g1.x * x4.z + g1.y * x4.w) * scg;
        uacc[b] += (u0.x * x4.x + u0.y * x4.y + u1.x * x4.z + u1.y * x4.w) * scu;
      }
    }
  }
  if (!act) return;
#pragma unroll
  for (int b = 0; b < BM; b++) {
    float g = warp_reduce(gacc[b]); float u = warp_reduce(uacc[b]);
    if (lane == 0 && toks[b] >= 0) {
      float s = g / (1.f + __expf(-g));
      out[(long)(pid_b * BM + b) * I + i] = s * u;
    }
  }
}

// down+scatter. vb [Pmax,I]; Cd [K,VD]; lo/hi/scale down [E,H,nsd]; out [N,H] atomic.
template <int BM, int VD, int SPG, int RW>
__global__ void __launch_bounds__(256, 3) grp_down(
    const float* __restrict__ vb, const int* __restrict__ sorted_tok,
    const float* __restrict__ w_sorted, const int* __restrict__ be, const __half* __restrict__ Cd,
    const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi, const __half* __restrict__ sdn,
    float* __restrict__ out, int N, int H, int nsd, int ng, int I) {
  int pid_b = blockIdx.x;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int i = blockIdx.y * RW + warp;               // output col (0..H-1)
  int e = be[pid_b];
  extern __shared__ __half smem[];
  __half* cb = smem;
  float* vs = (float*)(cb + 1024 * VD);         // BM*CT*VD floats (vb tile, shared across rows)
  for (int t = threadIdx.x; t < 1024 * VD; t += blockDim.x) cb[t] = Cd[t];
  int toks[BM]; float acc[BM]; float wts[BM];
#pragma unroll
  for (int b = 0; b < BM; b++) { toks[b] = sorted_tok[pid_b * BM + b]; acc[b] = 0.f; wts[b] = w_sorted[pid_b * BM + b]; }
  bool act = (i < H) && (e >= 0);
  long base = ((long)e * H + i) * nsd;
  long sbase = ((long)e * H + i) * ng;
  const int CV = CT * VD;
  for (int t0 = 0; t0 < nsd; t0 += CT) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < BM * CV; idx += blockDim.x) {
      int b = idx / CV, k = idx % CV;
      vs[idx] = vb[(long)(pid_b * BM + b) * I + (long)t0 * VD + k];
    }
    __syncthreads();
    if (!act) continue;
    for (int cc = 0; cc < 4; cc++) {
      int c = lane + cc * 32;
      int id = lo[base + t0 + c] | (((hi[base / 4 + (t0 + c) / 4] >> ((c & 3) * 2)) & 3) << 8);
      float sc = __half2float(sdn[sbase + (t0 + c) / SPG]);
      float2 c0 = __half22float2(*(const __half2*)(cb + id * VD));
      float2 c1 = __half22float2(*(const __half2*)(cb + id * VD + 2));
      int xk = c * VD;
      for (int b = 0; b < BM; b++) {
        const float4 v4 = *(const float4*)(vs + b * CV + xk);   // conflict-free (4-phase floor)
        acc[b] += (c0.x * v4.x + c0.y * v4.y + c1.x * v4.z + c1.y * v4.w) * sc;
      }
    }
  }
  if (!act) return;
#pragma unroll
  for (int b = 0; b < BM; b++) {
    float r = warp_reduce(acc[b]);
    int t = toks[b];
    if (lane == 0 && t >= 0) atomicAdd(&out[(long)t * H + i], r * wts[b]);
  }
}

// ============================ WMMA (tensor-core) grouped GEMM ============================
// Tensor cores amortize the weight read in REGISTERS across a large token tile (BM), which the
// FMA kernel can't (shared/compute-bound). fp16 inputs, fp32 accumulate (act rel err ~3e-4 vs
// fp32, far below the 2-bit quant floor). gate_up writes the full 2I output to a temp; a cheap
// elementwise kernel does silu(gate)*up; down writes a [Pmax,H] temp then scatters with weights.
// One CTA computes [BM tokens] x [NW*16 rows] of one expert, K-looped over chunks (4 chunks=16 K).
#define MT (16)                          // wmma m=n=k=16
// K-slab variant: per outer step, stage BKC contiguous chunks of the raw index tensors for all
// NW*MT rows into shared via COALESCED uint reads (a warp reads 128 contiguous bytes = one full
// cache line; the old per-tile path used ~4 bytes/line -> ~3% efficiency, the reason per-pair beat
// it). The inner mma loop then dequantizes from the shared slab (fast) -> high weight-read MBU.
// NN = output-row tiles per warp. Each CTA covers NW*NN*MT rows but stages the BM-token activation
// only ONCE per K-tile (reused across all NN*NW output tiles) -> cuts the grid.y activation re-read
// (the dominant byte cost: activations were re-read ~32x) by NN.
template <int BM, int NW, int NN, int BKC>
__global__ void __launch_bounds__(NW * 32, 2) gemm_gateup_wmma(
    const __nv_fp8_e4m3* __restrict__ xh, const int* __restrict__ sorted_tok, const int* __restrict__ be,
    const __half* __restrict__ Cgu, const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi,
    const __half* __restrict__ sgu, float* __restrict__ interf, int TwoI, int nsg, int SPG, int H) {
  const int NR = NW * NN * MT;                       // rows per CTA N-tile
  int pid_b = blockIdx.x, warp = threadIdx.x >> 5;
  int e = be[pid_b];
  int rowbase = blockIdx.y * NR;
  extern __shared__ __half smem[];
  __half* cb = smem;                                 // 1024*4 half
  uint8_t* lo_s = (uint8_t*)(cb + 1024 * 4);         // [NR][BKC]
  uint8_t* hi_s = lo_s + NR * BKC;                   // [NR][BKC/4]
  __half*  sc_s = (__half*)(hi_s + NR * (BKC / 4));  // [NR][BKC/SPG]
  __half*  As   = sc_s + NR * (BKC / SPG);           // [BM][16]
  __half*  Bs   = As + BM * MT;                      // [NR][16]
  for (int t = threadIdx.x; t < 1024 * 4; t += blockDim.x) cb[t] = Cgu[t];
  __shared__ int toks[BM];
  for (int t = threadIdx.x; t < BM; t += blockDim.x) toks[t] = sorted_tok[pid_b * BM + t];
  wmma::fragment<wmma::accumulator, MT, MT, MT, float> acc[BM / MT][NN];
#pragma unroll
  for (int m = 0; m < BM / MT; m++)
#pragma unroll
    for (int j = 0; j < NN; j++) wmma::fill_fragment(acc[m][j], 0.f);
  __syncthreads();
  if (e < 0) return;
  for (int ko = 0; ko < nsg; ko += BKC) {
    __syncthreads();
    for (int u = threadIdx.x; u < NR * (BKC / 4); u += blockDim.x) {
      int col = u / (BKC / 4), uc = u % (BKC / 4);
      ((uint*)(lo_s + col * BKC))[uc] = ((const uint*)(lo + ((long)e * TwoI + rowbase + col) * nsg + ko))[uc];
    }
    for (int u = threadIdx.x; u < NR * (BKC / 16); u += blockDim.x) {
      int col = u / (BKC / 16), uc = u % (BKC / 16);
      ((uint*)(hi_s + col * (BKC / 4)))[uc] = ((const uint*)(hi + ((long)e * TwoI + rowbase + col) * (nsg / 4) + ko / 4))[uc];
    }
    for (int s = threadIdx.x; s < NR * (BKC / SPG); s += blockDim.x) {
      int col = s / (BKC / SPG), sc = s % (BKC / SPG);
      sc_s[col * (BKC / SPG) + sc] = sgu[((long)e * TwoI + rowbase + col) * (nsg / SPG) + ko / SPG + sc];
    }
    __syncthreads();
    for (int ki = 0; ki < BKC / 4; ki++) {
      int gc0 = ko + ki * 4;
      for (int idx = threadIdx.x; idx < BM * MT; idx += blockDim.x) {
        int m = idx >> 4, kk = idx & 15; int tk = toks[m];
#if VQ2_DIAG == 3
        As[idx] = __float2half(0.5f);   // skip the xh global activation read -> isolate activation reload
#else
        As[idx] = (tk >= 0) ? (__half)xh[(long)tk * H + gc0 * 4 + kk] : __float2half(0.f);
#endif
      }
      for (int idx = threadIdx.x; idx < NR * 4; idx += blockDim.x) {
        int col = idx >> 2, cq = idx & 3;
        int sc_chunk = ki * 4 + cq;
        float sc = __half2float(sc_s[col * (BKC / SPG) + sc_chunk / SPG]);
        __half2* dst = (__half2*)(Bs + col * MT + cq * 4);
#if VQ2_DIAG == 1
        // DIAG=1: skip the codebook lookup entirely (no lo/hi decode, no cb[id] gather) -> isolates
        // the TOTAL gather cost. Fill Bs with sc so the WMMA still runs on real-ish data.
        __half2 sv = __floats2half2_rn(sc, sc); dst[0] = sv; dst[1] = sv;
#else
#if VQ2_DIAG == 2
        int id = 0;   // conflict-free codebook (all lookups hit cb[0]) -> isolates BANK CONFLICTS
#else
        int id = lo_s[col * BKC + sc_chunk] |
                 (((hi_s[col * (BKC / 4) + (sc_chunk >> 2)] >> ((sc_chunk & 3) * 2)) & 3) << 8);
#endif
        const __half2* cv = (const __half2*)(cb + id * 4);
        float2 a = __half22float2(cv[0]), b = __half22float2(cv[1]);
        dst[0] = __floats2half2_rn(a.x * sc, a.y * sc);
        dst[1] = __floats2half2_rn(b.x * sc, b.y * sc);
#endif
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a, MT, MT, MT, __half, wmma::row_major> af[BM / MT];
#pragma unroll
      for (int m = 0; m < BM / MT; m++) wmma::load_matrix_sync(af[m], As + m * MT * MT, MT);
#pragma unroll
      for (int j = 0; j < NN; j++) {
        wmma::fragment<wmma::matrix_b, MT, MT, MT, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(bf, Bs + (warp * NN + j) * MT * MT, MT);
#pragma unroll
#if VQ2_DIAG != 4
        for (int m = 0; m < BM / MT; m++) wmma::mma_sync(acc[m][j], af[m], bf, acc[m][j]);
#endif
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (int m = 0; m < BM / MT; m++)
#pragma unroll
    for (int j = 0; j < NN; j++)
      wmma::store_matrix_sync(interf + (long)(pid_b * BM + m * MT) * TwoI + rowbase + (warp * NN + j) * MT,
                              acc[m][j], TwoI, wmma::mem_row_major);
}

// silu(gate)*up : interf[Pmax,2I] -> inter[Pmax,I]. Per-token undo of the fp8 gate_up activation
// scale: interf[p] = s_xh[tok_p] * true, so multiply by inv_xh[sorted_tok[p]] (on device, no host sync).
__global__ void silu_combine(const float* __restrict__ interf, float* __restrict__ inter,
                             const int* __restrict__ sorted_tok, const float* __restrict__ inv_xh,
                             int Pmax, int I) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)Pmax * I) return;
  int p = idx / I, i = idx % I; int tok = sorted_tok[p];
  float inv = (tok >= 0) ? inv_xh[tok] : 0.f;
  float g = interf[(long)p * 2 * I + i] * inv, u = interf[(long)p * 2 * I + I + i] * inv;
  inter[idx] = (g / (1.f + __expf(-g))) * u;
}

// down: vb[Pmax,Ip] @ W_down -> out[N,H] FUSED scatter (atomicAdd w/ routing weight; no outf temp).
template <int BM, int NW, int NN, int BKC>
__global__ void __launch_bounds__(NW * 32, 2) gemm_down_wmma(
    const __nv_fp8_e4m3* __restrict__ vb, const int* __restrict__ be,
    const int* __restrict__ sorted_tok, const float* __restrict__ w_sorted,
    const __half* __restrict__ Cd, const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi,
    const __half* __restrict__ sdn, float* __restrict__ out, int H, int nsd, int SPG, int Ip) {
  const int NR = NW * NN * MT;
  int pid_b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int e = be[pid_b];
  int rowbase = blockIdx.y * NR;
  extern __shared__ __half smem[];
  __half* cb = smem;
  uint8_t* lo_s = (uint8_t*)(cb + 1024 * 4);
  uint8_t* hi_s = lo_s + NR * BKC;
  __half*  sc_s = (__half*)(hi_s + NR * (BKC / 4));
  __half*  As   = sc_s + NR * (BKC / SPG);
  __half*  Bs   = As + BM * MT;
  for (int t = threadIdx.x; t < 1024 * 4; t += blockDim.x) cb[t] = Cd[t];
  wmma::fragment<wmma::accumulator, MT, MT, MT, float> acc[BM / MT][NN];
#pragma unroll
  for (int m = 0; m < BM / MT; m++)
#pragma unroll
    for (int j = 0; j < NN; j++) wmma::fill_fragment(acc[m][j], 0.f);
  __syncthreads();
  if (e < 0) return;
  for (int ko = 0; ko < nsd; ko += BKC) {
    __syncthreads();
    for (int u = threadIdx.x; u < NR * (BKC / 4); u += blockDim.x) {
      int col = u / (BKC / 4), uc = u % (BKC / 4);
      ((uint*)(lo_s + col * BKC))[uc] = ((const uint*)(lo + ((long)e * H + rowbase + col) * nsd + ko))[uc];
    }
    for (int u = threadIdx.x; u < NR * (BKC / 16); u += blockDim.x) {
      int col = u / (BKC / 16), uc = u % (BKC / 16);
      ((uint*)(hi_s + col * (BKC / 4)))[uc] = ((const uint*)(hi + ((long)e * H + rowbase + col) * (nsd / 4) + ko / 4))[uc];
    }
    for (int s = threadIdx.x; s < NR * (BKC / SPG); s += blockDim.x) {
      int col = s / (BKC / SPG), sc = s % (BKC / SPG);
      sc_s[col * (BKC / SPG) + sc] = sdn[((long)e * H + rowbase + col) * (nsd / SPG) + ko / SPG + sc];
    }
    __syncthreads();
    for (int ki = 0; ki < BKC / 4; ki++) {
      int gc0 = ko + ki * 4;
      for (int idx = threadIdx.x; idx < BM * MT; idx += blockDim.x) {
        int m = idx >> 4, kk = idx & 15;
        As[idx] = (__half)vb[(long)(pid_b * BM + m) * Ip + gc0 * 4 + kk];
      }
      for (int idx = threadIdx.x; idx < NR * 4; idx += blockDim.x) {
        int col = idx >> 2, cq = idx & 3;
        int sc_chunk = ki * 4 + cq;
        float sc = __half2float(sc_s[col * (BKC / SPG) + sc_chunk / SPG]);
        __half2* dst = (__half2*)(Bs + col * MT + cq * 4);
#if VQ2_DIAG == 1
        // DIAG=1: skip the codebook lookup entirely (no lo/hi decode, no cb[id] gather) -> isolates
        // the TOTAL gather cost. Fill Bs with sc so the WMMA still runs on real-ish data.
        __half2 sv = __floats2half2_rn(sc, sc); dst[0] = sv; dst[1] = sv;
#else
#if VQ2_DIAG == 2
        int id = 0;   // conflict-free codebook (all lookups hit cb[0]) -> isolates BANK CONFLICTS
#else
        int id = lo_s[col * BKC + sc_chunk] |
                 (((hi_s[col * (BKC / 4) + (sc_chunk >> 2)] >> ((sc_chunk & 3) * 2)) & 3) << 8);
#endif
        const __half2* cv = (const __half2*)(cb + id * 4);
        float2 a = __half22float2(cv[0]), b = __half22float2(cv[1]);
        dst[0] = __floats2half2_rn(a.x * sc, a.y * sc);
        dst[1] = __floats2half2_rn(b.x * sc, b.y * sc);
#endif
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_a, MT, MT, MT, __half, wmma::row_major> af[BM / MT];
#pragma unroll
      for (int m = 0; m < BM / MT; m++) wmma::load_matrix_sync(af[m], As + m * MT * MT, MT);
#pragma unroll
      for (int j = 0; j < NN; j++) {
        wmma::fragment<wmma::matrix_b, MT, MT, MT, __half, wmma::col_major> bf;
        wmma::load_matrix_sync(bf, Bs + (warp * NN + j) * MT * MT, MT);
#pragma unroll
#if VQ2_DIAG != 4
        for (int m = 0; m < BM / MT; m++) wmma::mma_sync(acc[m][j], af[m], bf, acc[m][j]);
#endif
      }
      __syncthreads();
    }
  }
  // FUSED scatter: store each acc tile to per-warp shared scratch (reuse Bs), atomicAdd into out
  // with the routing weight. Drops the [Pmax,H] outf temp + separate scatter kernel.
  float* scr = (float*)Bs + warp * (MT * MT);
  __syncthreads();
#pragma unroll
  for (int m = 0; m < BM / MT; m++)
#pragma unroll
    for (int j = 0; j < NN; j++) {
      wmma::store_matrix_sync(scr, acc[m][j], MT, wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < MT * MT; idx += 32) {
        int mi = idx >> 4, ni = idx & 15;
        int p = pid_b * BM + m * MT + mi; int tok = sorted_tok[p];
        if (tok >= 0) atomicAdd(&out[(long)tok * H + rowbase + (warp * NN + j) * MT + ni], scr[idx] * w_sorted[p]);
      }
      __syncwarp();
    }
}

// scatter outf[Pmax,H] -> out[N,H] with routing weight (atomicAdd; padding rows skipped)
__global__ void scatter_down(const float* __restrict__ outf, const int* __restrict__ sorted_tok,
                             const float* __restrict__ w_sorted, float* __restrict__ out, int Pmax, int H) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)Pmax * H) return;
  int p = idx / H, i = idx % H; int tok = sorted_tok[p];
  if (tok >= 0) atomicAdd(&out[(long)tok * H + i], outf[idx] * w_sorted[p]);
}

// ---- fp8 gateup: native fp8 mma.sync.m16n8k32.e4m3 (2x tensor cores; As/Bs stay fp8) ----
// Same [BM,NR] output + same smem bytes as gemm_gateup_wmma (fp8 32-K tile == half 16-K tile in bytes),
// but K=32/mma (half the mma steps) and the activation is consumed fp8 (no fp8->half up-convert) and
// the weight tile is reconstructed to fp8. Verified mma register layout: see fp8_mma_test.cu.
template <int BM, int NW, int NN, int BKC>
__global__ void __launch_bounds__(NW * 32, 2) gemm_gateup_wmma_fp8(
    const __nv_fp8_e4m3* __restrict__ xh, const int* __restrict__ sorted_tok, const int* __restrict__ be,
    const __half* __restrict__ Cgu, const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi,
    const __half* __restrict__ sgu, float* __restrict__ interf, int TwoI, int nsg, int SPG, int H) {
  const int NR = NW * NN * MT;           // output rows per CTA N-tile
  const int NN8 = NN * MT / 8;           // fp8 n8-tiles per warp (= NN*2)
  int pid_b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int g = lane >> 2, t = lane & 3;
  int e = be[pid_b]; int rowbase = blockIdx.y * NR;
  extern __shared__ __half smem[];
  __half* cb = smem;                                 // [1024*4] half (centroids kept half)
  uint8_t* lo_s = (uint8_t*)(cb + 1024 * 4);         // [NR][BKC]
  uint8_t* hi_s = lo_s + NR * BKC;                   // [NR][BKC/4]
  __half*  sc_s = (__half*)(hi_s + NR * (BKC / 4));  // [NR][BKC/SPG]
  __nv_fp8_e4m3* As = (__nv_fp8_e4m3*)(sc_s + NR * (BKC / SPG));  // [BM][32] fp8
  __nv_fp8_e4m3* Bs = As + BM * 32;                              // [NR][32] fp8
  for (int u = threadIdx.x; u < 1024 * 4; u += blockDim.x) cb[u] = Cgu[u];
  __shared__ int toks[BM];
  for (int u = threadIdx.x; u < BM; u += blockDim.x) toks[u] = sorted_tok[pid_b * BM + u];
  float acc[BM / 16][NN8][4];
#pragma unroll
  for (int m = 0; m < BM / 16; m++)
#pragma unroll
    for (int j = 0; j < NN8; j++) { acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f; }
  __syncthreads();
  if (e < 0) return;
  for (int ko = 0; ko < nsg; ko += BKC) {
    __syncthreads();
    for (int u = threadIdx.x; u < NR * (BKC / 4); u += blockDim.x) {
      int col = u / (BKC / 4), uc = u % (BKC / 4);
      ((uint*)(lo_s + col * BKC))[uc] = ((const uint*)(lo + ((long)e * TwoI + rowbase + col) * nsg + ko))[uc];
    }
    for (int u = threadIdx.x; u < NR * (BKC / 16); u += blockDim.x) {
      int col = u / (BKC / 16), uc = u % (BKC / 16);
      ((uint*)(hi_s + col * (BKC / 4)))[uc] = ((const uint*)(hi + ((long)e * TwoI + rowbase + col) * (nsg / 4) + ko / 4))[uc];
    }
    for (int s = threadIdx.x; s < NR * (BKC / SPG); s += blockDim.x) {
      int col = s / (BKC / SPG), sc = s % (BKC / SPG);
      sc_s[col * (BKC / SPG) + sc] = sgu[((long)e * TwoI + rowbase + col) * (nsg / SPG) + ko / SPG + sc];
    }
    __syncthreads();
    for (int ki = 0; ki < BKC / 8; ki++) {           // 8 sub-groups (32 K cols) per fp8 step
      int sg0 = ki * 8, gc0_sg = ko + sg0;
      // As: vectorized uint copy (4 fp8 K-cols/thread) -- xh is already fp8, 4 contiguous = one uint
      for (int idx = threadIdx.x; idx < BM * 8; idx += blockDim.x) {
        int m = idx >> 3, kq = idx & 7; int tk = toks[m];
        unsigned v = (tk >= 0) ? *(const unsigned*)(xh + (long)tk * H + gc0_sg * 4 + kq * 4) : 0u;
        *(unsigned*)(As + m * 32 + kq * 4) = v;
      }
      // Bs: decode code -> 4 centroid halfs -> *sc -> 4 fp8 packed into one uint write (vectorized)
      for (int idx = threadIdx.x; idx < NR * 8; idx += blockDim.x) {
        int col = idx >> 3, sg = idx & 7, sc_chunk = sg0 + sg;
        float sc = __half2float(sc_s[col * (BKC / SPG) + sc_chunk / SPG]);
        int id = lo_s[col * BKC + sc_chunk] |
                 (((hi_s[col * (BKC / 4) + (sc_chunk >> 2)] >> ((sc_chunk & 3) * 2)) & 3) << 8);
        const __half2* cv = (const __half2*)(cb + id * 4);
        float2 a = __half22float2(cv[0]), b = __half22float2(cv[1]);
        // saturate to the e4m3 range (+/-448): the conversion NaN-encodes out-of-range, not inf
        __nv_fp8x2_e4m3 p0(make_float2(fminf(fmaxf(a.x * sc, -448.f), 448.f), fminf(fmaxf(a.y * sc, -448.f), 448.f)));
        __nv_fp8x2_e4m3 p1(make_float2(fminf(fmaxf(b.x * sc, -448.f), 448.f), fminf(fmaxf(b.y * sc, -448.f), 448.f)));
        unsigned packed = (unsigned)(*(unsigned short*)&p0) | ((unsigned)(*(unsigned short*)&p1) << 16);
        *(unsigned*)(Bs + col * 32 + sg * 4) = packed;
      }
      __syncthreads();
#pragma unroll
      for (int m4 = 0; m4 < BM / 16; m4++) {
        unsigned a0 = *(const unsigned*)(As + (m4 * 16 + g)     * 32 + t * 4);
        unsigned a1 = *(const unsigned*)(As + (m4 * 16 + g + 8) * 32 + t * 4);
        unsigned a2 = *(const unsigned*)(As + (m4 * 16 + g)     * 32 + 16 + t * 4);
        unsigned a3 = *(const unsigned*)(As + (m4 * 16 + g + 8) * 32 + 16 + t * 4);
#pragma unroll
        for (int n4 = 0; n4 < NN8; n4++) {
          int nbase = (warp * NN8 + n4) * 8;
          unsigned b0 = *(const unsigned*)(Bs + (nbase + g) * 32 + t * 4);
          unsigned b1 = *(const unsigned*)(Bs + (nbase + g) * 32 + 16 + t * 4);
          float* d = acc[m4][n4];
          asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (int m4 = 0; m4 < BM / 16; m4++)
#pragma unroll
    for (int n4 = 0; n4 < NN8; n4++) {
      int nbase = (warp * NN8 + n4) * 8; float* d = acc[m4][n4];
      long r0 = (long)(pid_b * BM + m4 * 16 + g)     * TwoI + rowbase + nbase;
      long r8 = (long)(pid_b * BM + m4 * 16 + g + 8) * TwoI + rowbase + nbase;
      interf[r0 + 2 * t + 0] = d[0]; interf[r0 + 2 * t + 1] = d[1];
      interf[r8 + 2 * t + 0] = d[2]; interf[r8 + 2 * t + 1] = d[3];
    }
}

// ---- fp8 down: native fp8 mma + FUSED scatter (atomicAdd from registers, no scratch) ----
template <int BM, int NW, int NN, int BKC>
__global__ void __launch_bounds__(NW * 32, 2) gemm_down_wmma_fp8(
    const __nv_fp8_e4m3* __restrict__ vb, const int* __restrict__ be,
    const int* __restrict__ sorted_tok, const float* __restrict__ w_sorted,
    const __half* __restrict__ Cd, const uint8_t* __restrict__ lo, const uint8_t* __restrict__ hi,
    const __half* __restrict__ sdn, float* __restrict__ out, int H, int nsd, int SPG, int Ip) {
  const int NR = NW * NN * MT;
  const int NN8 = NN * MT / 8;
  int pid_b = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int g = lane >> 2, t = lane & 3;
  int e = be[pid_b]; int rowbase = blockIdx.y * NR;
  extern __shared__ __half smem[];
  __half* cb = smem;
  uint8_t* lo_s = (uint8_t*)(cb + 1024 * 4);
  uint8_t* hi_s = lo_s + NR * BKC;
  __half*  sc_s = (__half*)(hi_s + NR * (BKC / 4));
  __nv_fp8_e4m3* As = (__nv_fp8_e4m3*)(sc_s + NR * (BKC / SPG));
  __nv_fp8_e4m3* Bs = As + BM * 32;
  for (int u = threadIdx.x; u < 1024 * 4; u += blockDim.x) cb[u] = Cd[u];
  float acc[BM / 16][NN8][4];
#pragma unroll
  for (int m = 0; m < BM / 16; m++)
#pragma unroll
    for (int j = 0; j < NN8; j++) { acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f; }
  __syncthreads();
  if (e < 0) return;
  for (int ko = 0; ko < nsd; ko += BKC) {
    __syncthreads();
    for (int u = threadIdx.x; u < NR * (BKC / 4); u += blockDim.x) {
      int col = u / (BKC / 4), uc = u % (BKC / 4);
      ((uint*)(lo_s + col * BKC))[uc] = ((const uint*)(lo + ((long)e * H + rowbase + col) * nsd + ko))[uc];
    }
    for (int u = threadIdx.x; u < NR * (BKC / 16); u += blockDim.x) {
      int col = u / (BKC / 16), uc = u % (BKC / 16);
      ((uint*)(hi_s + col * (BKC / 4)))[uc] = ((const uint*)(hi + ((long)e * H + rowbase + col) * (nsd / 4) + ko / 4))[uc];
    }
    for (int s = threadIdx.x; s < NR * (BKC / SPG); s += blockDim.x) {
      int col = s / (BKC / SPG), sc = s % (BKC / SPG);
      sc_s[col * (BKC / SPG) + sc] = sdn[((long)e * H + rowbase + col) * (nsd / SPG) + ko / SPG + sc];
    }
    __syncthreads();
    for (int ki = 0; ki < BKC / 8; ki++) {
      int sg0 = ki * 8, gc0_sg = ko + sg0;
      for (int idx = threadIdx.x; idx < BM * 8; idx += blockDim.x) {
        int m = idx >> 3, kq = idx & 7;
        *(unsigned*)(As + m * 32 + kq * 4) =
            *(const unsigned*)(vb + (long)(pid_b * BM + m) * Ip + gc0_sg * 4 + kq * 4);
      }
      for (int idx = threadIdx.x; idx < NR * 8; idx += blockDim.x) {
        int col = idx >> 3, sg = idx & 7, sc_chunk = sg0 + sg;
        float sc = __half2float(sc_s[col * (BKC / SPG) + sc_chunk / SPG]);
        int id = lo_s[col * BKC + sc_chunk] |
                 (((hi_s[col * (BKC / 4) + (sc_chunk >> 2)] >> ((sc_chunk & 3) * 2)) & 3) << 8);
        const __half2* cv = (const __half2*)(cb + id * 4);
        float2 a = __half22float2(cv[0]), b = __half22float2(cv[1]);
        // saturate to the e4m3 range (+/-448): the conversion NaN-encodes out-of-range, not inf
        __nv_fp8x2_e4m3 p0(make_float2(fminf(fmaxf(a.x * sc, -448.f), 448.f), fminf(fmaxf(a.y * sc, -448.f), 448.f)));
        __nv_fp8x2_e4m3 p1(make_float2(fminf(fmaxf(b.x * sc, -448.f), 448.f), fminf(fmaxf(b.y * sc, -448.f), 448.f)));
        unsigned packed = (unsigned)(*(unsigned short*)&p0) | ((unsigned)(*(unsigned short*)&p1) << 16);
        *(unsigned*)(Bs + col * 32 + sg * 4) = packed;
      }
      __syncthreads();
#pragma unroll
      for (int m4 = 0; m4 < BM / 16; m4++) {
        unsigned a0 = *(const unsigned*)(As + (m4 * 16 + g)     * 32 + t * 4);
        unsigned a1 = *(const unsigned*)(As + (m4 * 16 + g + 8) * 32 + t * 4);
        unsigned a2 = *(const unsigned*)(As + (m4 * 16 + g)     * 32 + 16 + t * 4);
        unsigned a3 = *(const unsigned*)(As + (m4 * 16 + g + 8) * 32 + 16 + t * 4);
#pragma unroll
        for (int n4 = 0; n4 < NN8; n4++) {
          int nbase = (warp * NN8 + n4) * 8;
          unsigned b0 = *(const unsigned*)(Bs + (nbase + g) * 32 + t * 4);
          unsigned b1 = *(const unsigned*)(Bs + (nbase + g) * 32 + 16 + t * 4);
          float* d = acc[m4][n4];
          asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
      __syncthreads();
    }
  }
  // fused scatter: atomicAdd the 4 acc values straight from registers (token-row x H-col)
#pragma unroll
  for (int m4 = 0; m4 < BM / 16; m4++)
#pragma unroll
    for (int n4 = 0; n4 < NN8; n4++) {
      int nbase = (warp * NN8 + n4) * 8; float* d = acc[m4][n4];
      int p0 = pid_b * BM + m4 * 16 + g, p8 = pid_b * BM + m4 * 16 + g + 8;
      int tok0 = sorted_tok[p0], tok8 = sorted_tok[p8];
      int c0 = rowbase + nbase + 2 * t, c1 = c0 + 1;
      if (tok0 >= 0) { atomicAdd(&out[(long)tok0 * H + c0], d[0] * w_sorted[p0]);
                       atomicAdd(&out[(long)tok0 * H + c1], d[1] * w_sorted[p0]); }
      if (tok8 >= 0) { atomicAdd(&out[(long)tok8 * H + c0], d[2] * w_sorted[p8]);
                       atomicAdd(&out[(long)tok8 * H + c1], d[3] * w_sorted[p8]); }
    }
}

static int g_shm(int VD, int BM) { return 1024 * VD * (int)sizeof(__half) + BM * CT * VD * (int)sizeof(float); }

#define GU_CASE(BMV) case BMV: { \
    cudaFuncSetAttribute(grp_gateup<BMV, VD, 16, RW>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_shm(VD, BMV)); \
    grp_gateup<BMV, VD, 16, RW><<<grid, RW * WARP, g_shm(VD, BMV)>>>( \
        xh.data_ptr<float>(), sorted_tok.data_ptr<int>(), be.data_ptr<int>(), \
        (const __half*)Cgu.data_ptr(), lo.data_ptr<uint8_t>(), hi.data_ptr<uint8_t>(), \
        (const __half*)sgu.data_ptr(), out.data_ptr<float>(), N, I, nsg, ng, H); break; }

void gateup_launch(torch::Tensor xh, torch::Tensor sorted_tok, torch::Tensor be, torch::Tensor Cgu,
                   torch::Tensor lo, torch::Tensor hi, torch::Tensor sgu, torch::Tensor out,
                   int nblk, int I, int nsg, int ng, int BM) {
  const int RW = 8, VD = 4;
  dim3 grid(nblk, (I + RW - 1) / RW); int N = xh.size(0), H = xh.size(1);
  switch (BM) { GU_CASE(8) GU_CASE(16) GU_CASE(32) default: TORCH_CHECK(false, "bad BM"); }
}

#define DN_CASE(BMV) case BMV: { \
    cudaFuncSetAttribute(grp_down<BMV, VD, 16, RW>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_shm(VD, BMV)); \
    grp_down<BMV, VD, 16, RW><<<grid, RW * WARP, g_shm(VD, BMV)>>>( \
        vb.data_ptr<float>(), sorted_tok.data_ptr<int>(), w_sorted.data_ptr<float>(), be.data_ptr<int>(), \
        (const __half*)Cd.data_ptr(), lo.data_ptr<uint8_t>(), hi.data_ptr<uint8_t>(), \
        (const __half*)sdn.data_ptr(), out.data_ptr<float>(), N, H, nsd, ng, I); break; }

void down_launch(torch::Tensor vb, torch::Tensor sorted_tok, torch::Tensor w_sorted, torch::Tensor be,
                 torch::Tensor Cd, torch::Tensor lo, torch::Tensor hi, torch::Tensor sdn, torch::Tensor out,
                 int nblk, int H, int nsd, int ng, int BM) {
  const int RW = 8, VD = 4;
  dim3 grid(nblk, (H + RW - 1) / RW); int N = out.size(0), I = vb.size(1);
  switch (BM) { DN_CASE(8) DN_CASE(16) DN_CASE(32) default: TORCH_CHECK(false, "bad BM"); }
}

// ---- WMMA launchers (BM=64, NW=8 fixed; NN=row-tiles/warp; BKC=chunks staged per outer K-step) ----
static int wmma_shm(int NW, int NN, int BKC) {
  const int BM = 64, NR = NW * NN * MT, SPG = 16;
  return 1024 * 4 * (int)sizeof(__half) + NR * BKC + NR * (BKC / 4)
       + NR * (BKC / SPG) * (int)sizeof(__half)
       + BM * MT * (int)sizeof(__half) + NR * MT * (int)sizeof(__half);
}

#define GUW_CASE(NWV, NNV, BKCV) if (NW == NWV && NN == NNV && BKC == BKCV) { \
    cudaFuncSetAttribute(gemm_gateup_wmma<64, NWV, NNV, BKCV>, cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_shm(NWV, NNV, BKCV)); \
    dim3 grid(nblk, TwoI / (NWV * NNV * MT)); \
    gemm_gateup_wmma<64, NWV, NNV, BKCV><<<grid, NWV * 32, wmma_shm(NWV, NNV, BKCV)>>>( \
        (const __nv_fp8_e4m3*)xh.data_ptr(), sorted_tok.data_ptr<int>(), be.data_ptr<int>(), \
        (const __half*)Cgu.data_ptr(), lo.data_ptr<uint8_t>(), hi.data_ptr<uint8_t>(), \
        (const __half*)sgu.data_ptr(), interf.data_ptr<float>(), TwoI, nsg, SPG, H); return; }

void gateup_wmma(torch::Tensor xh, torch::Tensor sorted_tok, torch::Tensor be, torch::Tensor Cgu,
                 torch::Tensor lo, torch::Tensor hi, torch::Tensor sgu, torch::Tensor interf,
                 int nblk, int TwoI, int nsg, int SPG, int NN, int BKC, int NW) {
  int H = xh.size(1);
  GUW_CASE(8, 2, 64) GUW_CASE(8, 1, 128) GUW_CASE(8, 2, 128) GUW_CASE(8, 4, 64) GUW_CASE(8, 4, 32) GUW_CASE(8, 2, 32)
  GUW_CASE(16, 2, 64) GUW_CASE(16, 1, 64) GUW_CASE(16, 4, 32) GUW_CASE(16, 2, 32) GUW_CASE(4, 4, 64) GUW_CASE(4, 2, 128)
  TORCH_CHECK(false, "bad NW/NN/BKC");
}

#define GUWF_CASE(NWV, NNV, BKCV) if (NW == NWV && NN == NNV && BKC == BKCV) { \
    cudaFuncSetAttribute(gemm_gateup_wmma_fp8<64, NWV, NNV, BKCV>, cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_shm(NWV, NNV, BKCV)); \
    dim3 grid(nblk, TwoI / (NWV * NNV * MT)); \
    gemm_gateup_wmma_fp8<64, NWV, NNV, BKCV><<<grid, NWV * 32, wmma_shm(NWV, NNV, BKCV)>>>( \
        (const __nv_fp8_e4m3*)xh.data_ptr(), sorted_tok.data_ptr<int>(), be.data_ptr<int>(), \
        (const __half*)Cgu.data_ptr(), lo.data_ptr<uint8_t>(), hi.data_ptr<uint8_t>(), \
        (const __half*)sgu.data_ptr(), interf.data_ptr<float>(), TwoI, nsg, SPG, H); return; }

void gateup_wmma_fp8(torch::Tensor xh, torch::Tensor sorted_tok, torch::Tensor be, torch::Tensor Cgu,
                     torch::Tensor lo, torch::Tensor hi, torch::Tensor sgu, torch::Tensor interf,
                     int nblk, int TwoI, int nsg, int SPG, int NN, int BKC, int NW) {
  int H = xh.size(1);
  GUWF_CASE(8, 2, 64) GUWF_CASE(8, 4, 64) GUWF_CASE(8, 2, 128) GUWF_CASE(8, 1, 128) GUWF_CASE(8, 2, 32) GUWF_CASE(8, 4, 32)
  TORCH_CHECK(false, "bad NW/NN/BKC for fp8 gateup");
}

void silu_comb(torch::Tensor interf, torch::Tensor inter, torch::Tensor sorted_tok,
               torch::Tensor inv_xh, int Pmax, int I) {
  long n = (long)Pmax * I; int thr = 256;
  silu_combine<<<(n + thr - 1) / thr, thr>>>(interf.data_ptr<float>(), inter.data_ptr<float>(),
                                             sorted_tok.data_ptr<int>(), inv_xh.data_ptr<float>(), Pmax, I);
}

#define DNW_CASE(NWV, NNV, BKCV) if (NW == NWV && NN == NNV && BKC == BKCV) { \
    cudaFuncSetAttribute(gemm_down_wmma<64, NWV, NNV, BKCV>, cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_shm(NWV, NNV, BKCV)); \
    dim3 grid(nblk, H / (NWV * NNV * MT)); \
    gemm_down_wmma<64, NWV, NNV, BKCV><<<grid, NWV * 32, wmma_shm(NWV, NNV, BKCV)>>>( \
        (const __nv_fp8_e4m3*)vb.data_ptr(), be.data_ptr<int>(), sorted_tok.data_ptr<int>(), w_sorted.data_ptr<float>(), \
        (const __half*)Cd.data_ptr(), lo.data_ptr<uint8_t>(), hi.data_ptr<uint8_t>(), (const __half*)sdn.data_ptr(), \
        out.data_ptr<float>(), H, nsd, SPG, Ip); return; }

void down_wmma(torch::Tensor vb, torch::Tensor be, torch::Tensor sorted_tok, torch::Tensor w_sorted,
               torch::Tensor Cd, torch::Tensor lo, torch::Tensor hi,
               torch::Tensor sdn, torch::Tensor out, int nblk, int H, int nsd, int SPG, int NN, int BKC, int NW) {
  int Ip = vb.size(1);
  DNW_CASE(8, 2, 64) DNW_CASE(8, 1, 128) DNW_CASE(8, 2, 128) DNW_CASE(8, 4, 64) DNW_CASE(8, 4, 32) DNW_CASE(8, 2, 32)
  DNW_CASE(16, 2, 64) DNW_CASE(16, 1, 64) DNW_CASE(16, 4, 32) DNW_CASE(16, 2, 32) DNW_CASE(4, 4, 64) DNW_CASE(4, 2, 128)
  TORCH_CHECK(false, "bad NW/NN/BKC");
}

#define DNWF_CASE(NWV, NNV, BKCV) if (NW == NWV && NN == NNV && BKC == BKCV) { \
    cudaFuncSetAttribute(gemm_down_wmma_fp8<64, NWV, NNV, BKCV>, cudaFuncAttributeMaxDynamicSharedMemorySize, wmma_shm(NWV, NNV, BKCV)); \
    dim3 grid(nblk, H / (NWV * NNV * MT)); \
    gemm_down_wmma_fp8<64, NWV, NNV, BKCV><<<grid, NWV * 32, wmma_shm(NWV, NNV, BKCV)>>>( \
        (const __nv_fp8_e4m3*)vb.data_ptr(), be.data_ptr<int>(), sorted_tok.data_ptr<int>(), w_sorted.data_ptr<float>(), \
        (const __half*)Cd.data_ptr(), lo.data_ptr<uint8_t>(), hi.data_ptr<uint8_t>(), (const __half*)sdn.data_ptr(), \
        out.data_ptr<float>(), H, nsd, SPG, Ip); return; }

void down_wmma_fp8(torch::Tensor vb, torch::Tensor be, torch::Tensor sorted_tok, torch::Tensor w_sorted,
                   torch::Tensor Cd, torch::Tensor lo, torch::Tensor hi,
                   torch::Tensor sdn, torch::Tensor out, int nblk, int H, int nsd, int SPG, int NN, int BKC, int NW) {
  int Ip = vb.size(1);
  DNWF_CASE(8, 2, 64) DNWF_CASE(8, 4, 64) DNWF_CASE(8, 2, 128) DNWF_CASE(8, 1, 128) DNWF_CASE(8, 2, 32) DNWF_CASE(8, 4, 32)
  TORCH_CHECK(false, "bad NW/NN/BKC for fp8 down");
}

void scatter(torch::Tensor outf, torch::Tensor sorted_tok, torch::Tensor w_sorted, torch::Tensor out,
             int Pmax, int H) {
  long n = (long)Pmax * H; int thr = 256;
  scatter_down<<<(n + thr - 1) / thr, thr>>>(outf.data_ptr<float>(), sorted_tok.data_ptr<int>(),
                                             w_sorted.data_ptr<float>(), out.data_ptr<float>(), Pmax, H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gateup", &gateup_launch, "vq2 grouped gate_up+silu");
  m.def("down", &down_launch, "vq2 grouped down+scatter");
  m.def("gateup_wmma", &gateup_wmma, "vq2 wmma gate_up GEMM -> 2I temp");
  m.def("gateup_wmma_fp8", &gateup_wmma_fp8, "vq2 fp8-mma gate_up GEMM -> 2I temp");
  m.def("silu_comb", &silu_comb, "silu(gate)*up");
  m.def("down_wmma", &down_wmma, "vq2 wmma down GEMM -> Pmax,H temp");
  m.def("down_wmma_fp8", &down_wmma_fp8, "vq2 fp8-mma down GEMM + fused scatter");
  m.def("scatter", &scatter, "scatter outf -> out with routing weight");
}
