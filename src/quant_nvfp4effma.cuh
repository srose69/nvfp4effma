// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA quantization / dequantization kernels (the arithmetic of Transformer Engine) and the entry
// points of their instance objects (quant_nvfp4effma_inst.cu).  Host API: nvfp4effma.cu.
//
// The NVFP4 side's format is a template parameter QF (one instance per combination, no format test in a
// kernel loop): kQ_SWZ swizzled scales (sld = -padded columns), kQ_UE UE5M3 scales, kQ_AMAX the global is an
// amax (else a decode factor; NULL = 1), kQ_ROW one global per row.  The quantization kernels with one
// global take the amax / decode factor choice at run time (QuantOut::amax): it only enters the encode /
// decode factors, computed once per thread; with per-row globals (once per block) it stays in QF.
#pragma once
#include <type_traits>
#include "nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

constexpr int kQ_SWZ = 1, kQ_UE = 2, kQ_AMAX = 4, kQ_ROW = 8, kQFormats = 16;
// combinations Transformer Engine defines: 4over6 with an amax global and E4M3 scales; per-row globals on
// the rowwise 1 x 16 quantization
__host__ __device__ constexpr bool quant_valid(bool col, int tile, int fos, int qf) {
  return (fos == NVFP4A_4OVER6_OFF || (!(qf & kQ_UE) && (!(qf & kQ_ROW) || (qf & kQ_AMAX)))) &&
         (!(qf & kQ_ROW) || (!col && tile == 1)) && (!(qf & kQ_AMAX) || (qf & kQ_ROW));
}

// An NVFP4 matrix [orows][ocols] as the kernels see it.
struct QuantOut {
  uint8_t *data; int ld;  // nibbles, ld values per row
  uint8_t *sc; int sld;   // scales: ld (linear) or -padded columns (swizzled)
  const float *g;         // global (amax or decode factor), NULL = 1
  float smax;             // scale_max M
  int amax;               // the global is an amax (quantization without per-row globals; else kQ_AMAX)
};
template <int QF>
__device__ __forceinline__ size_t scale_at(const QuantOut &o, int r, int c) {
  return QF & kQ_SWZ ? swz_row(r, -o.sld) + swz_col(c) : (size_t)r * o.sld + c;
}

// Encode / decode factors of a global value gv: amax (S_enc = min(6 M / amax, FLT_MAX), 1 for a zero amax
// or S_enc; S_dec = 1 / S_enc), or a decode factor (S_dec = gv, S_enc = min(1 / gv, FLT_MAX)).
struct QuantGlobal { float gv, s_enc, s_dec; };
__device__ __forceinline__ QuantGlobal quant_global(float gv, float smax, bool amax) {
  constexpr float kMax = 3.40282347e38f;
  if (amax) {
    float s_enc = fminf(__fdiv_rn(smax * 6.f, gv), kMax);
    if (gv == 0.f || s_enc == 0.f) s_enc = 1.f;
    return {gv, s_enc, __fdiv_rn(1.f, s_enc)};
  }
  return {gv, fminf(__fdiv_rn(1.f, gv), kMax), gv};
}

__device__ __forceinline__ float group16_max(float v, unsigned mask) {
#pragma unroll
  for (int o = 8; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(mask, v, o, 16));
  return v;
}
// Transformer Engine's reduce_group_sum_16: a shfl_down tree over the 16 lanes, result of lane 0
__device__ __forceinline__ float group16_sum_tree(float v, unsigned mask) {
#pragma unroll
  for (int o = 8; o; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(mask, v, o, 16));
  return __shfl_sync(mask, v, 0, 16);
}
// 16 values * inv -> 16 E2M1 codes (RNE, satfinite), packed into 2 words
__device__ __forceinline__ uint2 e2m1_pack16(const float x[16], float inv) {
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int e = 0; e < 16; ++e) w[e / 8] |= f32_to_e2m1(__fmul_rn(x[e], inv)) << (4 * (e % 8));
  return make_uint2(w[0], w[1]);
}
__device__ __forceinline__ float e2m1_of(uint2 w, int e) { return e2m1_value<0>(e < 8 ? w.x : w.y, e % 8); }
// 4over6 error of a candidate over one 1 x 16 block, in element order (TE's accumulate_dequant_error); the
// division by denom = 6 M through its reciprocal rden, exactly (div_by)
template <int FOS>
__device__ __forceinline__ float fos_error(const float x[16], uint2 q, float sf, float amax, float denom, float rden) {
  float err = 0.f;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    const float val = div_by(__fmul_rn(__fmul_rn(e2m1_of(q, e), sf), amax), denom, rden);
    const float d = __fsub_rn(val, x[e]);
    err = __fadd_rn(err, FOS == NVFP4A_4OVER6_MSE ? __fmul_rn(d, d) : fabsf(d));
  }
  return err;
}

// 4 consecutive dense values: one 16-byte word of FP32, 8 bytes of BF16 / FP16
template <int T> struct Quad { using W = typename std::conditional<T == NVFP4A_FP32, uint4, uint2>::type; };
template <int T>
__device__ __forceinline__ void unpack4(uint4 w, float *v) {
  v[0] = __uint_as_float(w.x); v[1] = __uint_as_float(w.y); v[2] = __uint_as_float(w.z); v[3] = __uint_as_float(w.w);
}
template <int T>
__device__ __forceinline__ void unpack4(uint2 w, float *v) { unpack2<T>(w.x, v[0], v[1]); unpack2<T>(w.y, v[2], v[3]); }

// Rowwise 16 x 16 tiles, one warp per 16 rows x 8 blocks: each lane loads 4 values of each of the 16 rows (a
// warp reads 8 consecutive blocks per row) and keeps them in registers, 4 lanes per block.  4 values per
// lane, not 16 bytes: with 8 BF16 per lane the per-lane work doubles at the same occupancy (measured 2.9x
// slower).  The tile amax, the scale and the inverse are shared by the 16 rows.  Nibbles: 16 bits per lane
// and row (coalesced stores).  Without 4over6 only: 4over6 is bound by its divisions, and its 16 x 16 form
// runs best on the 16-lane units (a lane per row: no error sums passed between lanes).
template <int T, int QF>
__device__ __forceinline__ void quant_tiles16(const uint8_t *__restrict__ src, int lds, int rows, int cols, const QuantOut &o,
                                              const QuantGlobal &G, float &mx) {
  constexpr float kMax = 3.40282347e38f;
  constexpr int VPL = 4, LPB = 4, BPW = 8;
  using W = typename Quad<T>::W;
  const int lane = threadIdx.x & 31, part = lane % LPB;
  const int ocb = cols / 16, groups = (ocb + BPW - 1) / BPW, tiles = (rows + 15) / 16;
  for (long long u = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32; u < (long long)tiles * groups;
       u += (long long)gridDim.x * blockDim.x / 32) {
    const int t = (int)(u / groups), blk = (int)(u % groups) * BPW + lane / LPB;
    const bool bl = blk < ocb;
    W raw[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int r = 16 * t + i;
      raw[i] = bl && r < rows ? *(const W *)(src + ((size_t)r * lds + 16 * blk + VPL * part) * Elem<T>::bytes) : W{};
    }
    float bmax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float v[VPL];
      unpack4<T>(raw[i], v);
#pragma unroll
      for (int e = 0; e < VPL; ++e) bmax = fmaxf(bmax, fabsf(v[e]));
    }
    mx = fmaxf(mx, bmax);
#pragma unroll
    for (int ofs = 1; ofs < LPB; ofs <<= 1) bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFFu, bmax, ofs));
    constexpr bool UE = QF & kQ_UE;
    const float d = fminf(__fmul_rn(bmax, __fmul_rn(G.s_enc, 1.f / 6.f)), kMax);
    const uint32_t code = UE ? f32_to_ue5m3(d) : f32_to_e4m3(d);
    const float inv = fminf(__fdiv_rn(1.f, __fmul_rn(scale_val<UE>(code), G.s_dec)), kMax);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int r = 16 * t + i;
      float v[VPL];
      unpack4<T>(raw[i], v);
      uint32_t w = 0;
#pragma unroll
      for (int e = 0; e < VPL; ++e) w |= f32_to_e2m1_sel(__fmul_rn(v[e], inv)) << (4 * e);
      if (bl && r < rows) {
        uint8_t *d = o.data + (size_t)r * (o.ld / 2) + 8 * blk + (VPL / 2) * part;
        *(uint16_t *)d = (uint16_t)w;
        if (part == 0) o.sc[scale_at<QF>(o, r, blk)] = (uint8_t)code;
      }
    }
  }
}

// One 1 x 16 block: x, its (tile) amax bmax -> scale code and 16 nibbles.  TILE = 16: the 4over6 errors are
// summed over the 16 lanes of the unit (gmask).
template <int TILE, int FOS, int QF>
__device__ __forceinline__ void quant_block(const float x[16], float bmax, const QuantGlobal &G, float smax, float rden,
                                            unsigned gmask, uint32_t &code, uint2 &q) {
  constexpr float kMax = 3.40282347e38f;
  constexpr bool UE = QF & kQ_UE;
  if (FOS == NVFP4A_4OVER6_OFF) {
    const float d = fminf(__fmul_rn(bmax, __fmul_rn(G.s_enc, 1.f / 6.f)), kMax);
    code = UE ? f32_to_ue5m3(d) : f32_to_e4m3(d);
    q = e2m1_pack16(x, fminf(__fdiv_rn(1.f, __fmul_rn(scale_val<UE>(code), G.s_dec)), kMax));
  } else {  // 4over6 (E4M3 scales, amax global): map-to-6 and map-to-4 (1.5x scale) candidates
    const float base = __fmul_rn(__fdiv_rn(bmax, 6.f), G.s_enc);
    const uint32_t c4 = f32_to_e4m3(fminf(__fmul_rn(base, 1.5f), 448.f)), c6 = f32_to_e4m3(fminf(base, 448.f));
    const float sf4 = e4m3_value(c4), sf6 = e4m3_value(c6);
    const uint2 q4 = e2m1_pack16(x, fminf(__fdiv_rn(1.f, __fmul_rn(sf4, G.s_dec)), kMax));
    const uint2 q6 = e2m1_pack16(x, fminf(__fdiv_rn(1.f, __fmul_rn(sf6, G.s_dec)), kMax));
    const float denom = 6.f * smax;
    float e4 = fos_error<FOS>(x, q4, sf4, G.gv, denom, rden), e6 = fos_error<FOS>(x, q6, sf6, G.gv, denom, rden);
    if (TILE == 16) { e4 = group16_sum_tree(e4, gmask); e6 = group16_sum_tree(e6, gmask); }
    const bool pick4 = e4 < e6;  // ties: map-to-6
    code = pick4 ? c4 : c6;
    q = pick4 ? q4 : q6;
  }
}

// One output matrix per launch: the rowwise quantization of src [rows][cols] (COL = false), or the
// columnwise one (the quantization of src^T, [cols][rows]).
//   rowwise 1 x 16: one thread per block, consecutive threads on consecutive blocks of a row (coalesced).
//   rowwise 16 x 16 without 4over6: quant_tiles16.
//   16 x 16 with 4over6 and columnwise: a unit of 16 consecutive lanes owns one block column c of 16 consecutive
//   output rows (one 16 x 16 tile), lane i the block of output row 16t + i; tiles reduce over the unit.
//   Elements outside the matrix are 0.
template <int T, bool COL, int TILE, int FOS, int QF>
__global__ void __launch_bounds__(256)
quantize_kernel(const uint8_t *__restrict__ src, int lds, int rows, int cols, QuantOut o, float *stat) {
  constexpr bool ROW = QF & kQ_ROW;
  float mx = 0.f;
  QuantGlobal G0 = {};
  if (!ROW) G0 = quant_global(o.g ? o.g[0] : 1.f, o.smax, o.amax != 0);
  const float rden = FOS == NVFP4A_4OVER6_OFF ? 0.f : __fdiv_rn(1.f, 6.f * o.smax);  // 4over6: 1 / (6 M)
  if (!COL && TILE == 16 && FOS == NVFP4A_4OVER6_OFF) {
    quant_tiles16<T, QF>(src, lds, rows, cols, o, G0, mx);
  } else if (!COL && TILE == 1) {
    const int bpr = cols / 16;
    for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < (long long)rows * bpr;
         b += (long long)gridDim.x * blockDim.x) {
      const int r = (int)(b / bpr), c = (int)(b % bpr);
      float x[16];
      const uint8_t *p = src + ((size_t)r * lds + 16 * c) * Elem<T>::bytes;
      load8<T>(p, x); load8<T>(p + 8 * Elem<T>::bytes, x + 8);
      float bmax = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(x[i]));
      mx = fmaxf(mx, bmax);
      const QuantGlobal G = ROW ? quant_global(o.g ? o.g[r] : 1.f, o.smax, (QF & kQ_AMAX) != 0) : G0;
      uint32_t code; uint2 q;
      quant_block<1, FOS, QF>(x, bmax, G, o.smax, rden, 0u, code, q);
      *(uint2 *)(o.data + (size_t)r * (o.ld / 2) + 8 * c) = q;
      o.sc[scale_at<QF>(o, r, c)] = (uint8_t)code;
    }
  } else {
    const int orows = COL ? cols : rows, ocols = COL ? rows : cols;
    const int ocb = (ocols + 15) / 16, tiles = (orows + 15) / 16;
    const long long units = (long long)tiles * ocb;
    const int lane = threadIdx.x & 15;
    const unsigned gmask = 0xFFFFu << (threadIdx.x & 16);  // the 16 lanes of this unit
    for (long long u = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 16; u < units;
         u += (long long)gridDim.x * blockDim.x / 16) {
      const int t = (int)(u / ocb), c = (int)(u % ocb), orow = 16 * t + lane;
      const bool live = orow < orows;
      float x[16];
      if (!COL) {
        if (live) {
          const uint8_t *p = src + ((size_t)orow * lds + 16 * c) * Elem<T>::bytes;
          load8<T>(p, x); load8<T>(p + 8 * Elem<T>::bytes, x + 8);
        } else {
#pragma unroll
          for (int e = 0; e < 16; ++e) x[e] = 0.f;
        }
      } else {  // output row orow = src column; its block c covers src rows 16c .. 16c + 15
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          const int sr = 16 * c + e;
          x[e] = live && sr < rows ? load_any(T, src, (size_t)sr * lds + orow) : 0.f;
        }
      }
      float bmax = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) bmax = fmaxf(bmax, fabsf(x[e]));
      mx = fmaxf(mx, bmax);
      if (TILE == 16) bmax = group16_max(bmax, gmask);
      uint32_t code; uint2 q;
      quant_block<TILE, FOS, QF>(x, bmax, G0, o.smax, rden, gmask, code, q);
      if (live) {
        *(uint2 *)(o.data + (size_t)orow * (o.ld / 2) + 8 * c) = q;
        o.sc[scale_at<QF>(o, orow, c)] = (uint8_t)code;
      }
    }
  }
  if (stat) amax_commit(stat, mx);
}

// One thread per block: 8 bytes of nibbles and the scale -> 16 values: (e2m1 * sb) * g, or with an amax
// global e2m1 * ((sb * amax) * (1 / (6 * M))) (Transformer Engine's dequantization).
template <int T, int QF>
__global__ void dequantize_kernel(QuantOut q, uint8_t *__restrict__ dst, int ldd, int rows, int cols, float *amax) {
  constexpr bool AM = QF & kQ_AMAX, ROW = QF & kQ_ROW;
  const int bpr = cols / 16;
  const float finv = 1.0f / (6.f * q.smax), g0 = q.g ? q.g[0] : (AM ? 6.f * q.smax : 1.f);
  float mx = 0.f;
  for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < (long long)rows * bpr;
       b += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(b / bpr), c = (int)(b % bpr);
    const uint2 w = *(const uint2 *)(q.data + (size_t)r * (q.ld / 2) + 8 * c);
    const float sb = scale_val<(QF & kQ_UE) != 0>(q.sc[scale_at<QF>(q, r, c)]);
    const float gv = ROW ? q.g[r] : g0;
    const float fin = AM ? __fmul_rn(__fmul_rn(sb, gv), finv) : 0.f;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float e = e2m1_value<0>(i < 8 ? w.x : w.y, i % 8);
      v[i] = AM ? __fmul_rn(e, fin) : __fmul_rn(__fmul_rn(e, sb), gv);
      mx = fmaxf(mx, fabsf(v[i]));
    }
    uint8_t *p = dst + ((size_t)r * ldd + 16 * c) * Elem<T>::bytes;
    if (T == NVFP4A_FP32) {
#pragma unroll
      for (int k = 0; k < 4; ++k) ((float4 *)p)[k] = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
    } else {
      uint32_t h[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if (T == NVFP4A_BF16) h[k] = __byte_perm(rne_hi(__float_as_uint(v[2 * k])), rne_hi(__float_as_uint(v[2 * k + 1])), 0x7632);
        else { const __half2 t = __floats2half2_rn(v[2 * k], v[2 * k + 1]); h[k] = *reinterpret_cast<const uint32_t *>(&t); }
      }
      ((uint4 *)p)[0] = make_uint4(h[0], h[1], h[2], h[3]);
      ((uint4 *)p)[1] = make_uint4(h[4], h[5], h[6], h[7]);
    }
  }
  if (amax) amax_commit(amax, mx);
}

// Entry points of the instance objects (one per dense dtype T, COL and TILE): the quantization kernel for
// 4over6 mode fos and format qf (invalid combinations: cudaErrorInvalidValue); the object of COL = false,
// TILE = 1 also carries the dequantization of T.
template <int T, bool COL, int TILE>
cudaError_t quant_launch(int fos, int qf, const uint8_t *src, int lds, int rows, int cols, const QuantOut &o, float *stat,
                         cudaStream_t s);
template <int T>
cudaError_t dequant_launch(int qf, const QuantOut &q, uint8_t *dst, int ldd, int rows, int cols, float *amax, cudaStream_t s);

inline int quant_grid(long long threads) {
  const long long b = (threads + 255) / 256;
  return (int)(b < 16384 ? (b > 0 ? b : 1) : 16384);
}

}  // namespace kern
}  // namespace nvfp4a
