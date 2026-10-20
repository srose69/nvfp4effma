// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA core (src/nvfp4effma.cuh), exhaustively, and nvfp4a_quantize / nvfp4a_dequantize:
//  1. block  : the DP4A kernel's scale step -- y = fma(bits(0x4B400000 + p), sa, -1.5 * 2^23 * sa) ==
//              p * sa exactly, for every block sum p in [-24384, 24384] (NVFP4 x NVFP4: |p| <= 2304,
//              NVFP4 x INT8: |p| <= 16 * 12 * 127) and every finite E4M3 scale.
//  2. dec4   : the PRMT lookup (4 nibbles -> 4 int8) == 2 * value for all 2^16 inputs, and the FP32
//              decode e2m1_value<0 / 1> of every nibble at every position of a word.
//  3. prod   : the dense-B MAC operands -- mul_rn(e2m1, sa) times every BF16 / FP16 value, for every
//              nibble and finite scale (2^28 cases each), equals the exact product whenever that is a
//              normal FP32 number (checked in double).
//  4. rne    : FP32 -> BF16 rounding on every FP32 bit pattern vs an independent remainder-comparison
//              RNE (NaN -> 0x7FC0).
//  5. e2m1, e4m3 : FP32 -> E2M1 / E4M3 encoding on every FP32 bit pattern vs an independent search for
//              the nearest value (midpoint table, ties to the even code, saturation; NaN: E2M1 +-6, E4M3
//              0x7F).
//  6. nvfp4a_quantize / nvfp4a_dequantize: BF16 / FP16 / FP32 data with zero, tiny and saturating
//              blocks, padded leading dimensions, against the host emulation of the documented formulas;
//              nvfp4a_quantize to INT8 likewise.
// Build with -DNVFP4A_NEGATIVE_CONTROLS to check that deliberately broken variants are rejected.
#include "test_util.h"

struct Err { unsigned long long count; uint32_t x[16], y[16], got[16], exp[16]; };

__device__ __forceinline__ void record(Err *err, uint32_t x, uint32_t y, uint32_t got, uint32_t exp) {
  unsigned long long e = atomicAdd(&err->count, 1ull);
  if (e < 16) { err->x[e] = x; err->y[e] = y; err->got[e] = got; err->exp[e] = exp; }
}

// ---- implementations under test -------------------------------------------------------------
__device__ __forceinline__ float block_ok(int p, float sa) {  // as gemm_dp4a_kernel
  return nvfp4a::fma_rn(__int_as_float(0x4B400000 + p), sa, -12582912.f * sa);
}
__device__ __forceinline__ uint32_t dec4_ok(uint32_t w) { return nvfp4a::dec4(w); }
__device__ __forceinline__ uint32_t rne_ok(uint32_t u) { return nvfp4a::rne_hi(u) >> 16; }
__device__ __forceinline__ uint32_t e2m1_ok(uint32_t u) { return nvfp4a::f32_to_e2m1(__uint_as_float(u)); }
__device__ __forceinline__ uint32_t e2m1_sel_ok(uint32_t u) { return nvfp4a::f32_to_e2m1_sel(__uint_as_float(u)); }
__device__ __forceinline__ uint32_t e4m3_ok(uint32_t u) { return nvfp4a::f32_to_e4m3(__uint_as_float(u)); }
__device__ __forceinline__ uint32_t ue5m3_ok(uint32_t u) { return nvfp4a::f32_to_ue5m3(__uint_as_float(u)); }
#ifdef NVFP4A_NEGATIVE_CONTROLS
__device__ __forceinline__ float block_twostep(int p, float sa) {  // the magic removed after rounding
  return nvfp4a::mul_rn(__int_as_float(0x4B400000 + p), sa) - 12582912.f * sa;
}
__device__ __forceinline__ uint32_t dec4_nosign(uint32_t w) {  // magnitudes only
  return nvfp4a::prmt(0x03020100u, 0x0C080604u, w & 0x7777u);
}
__device__ __forceinline__ uint32_t rne_nonan(uint32_t u) { return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
__device__ __forceinline__ uint32_t e2m1_ties_away(uint32_t u) {  // round half away instead of even
  const float a = fabsf(__uint_as_float(u));
  const uint32_t s = (u >> 28) & 8u;
  if (!(a < 6.f)) return s | 7u;
  const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  uint32_t c = 0;
  while (c < 7 && a >= 0.5f * (g[c] + g[c + 1])) ++c;
  return s | c;
}
__device__ __forceinline__ uint32_t e4m3_ties_away(uint32_t u) {
  const float x = __uint_as_float(u), a = fabsf(x);
  if (a != a || a >= 448.f || a < 0x1p-6f) return nvfp4a::f32_to_e4m3(x);
  const uint32_t b = __float_as_uint(a), r = b + 0x80000u;
  return ((u >> 24) & 0x80u) | (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u);
}
#endif

// ---- kernels ---------------------------------------------------------------------------------
// every block sum p and every finite E4M3 scale (positive and negative): one case per thread
template <float (*OP)(int, float)>
__global__ void k_block(uint64_t base, uint64_t n, Err *err) {
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t code = (uint32_t)(k & 0xFF);
    const int p = (int)(k >> 8) - 24384;
    if ((code & 0x7F) == 0x7F) continue;  // E4M3 NaN: not a block scale
    const float sa = nvfp4a::e4m3_value(code), got = OP(p, sa);
    const double exact = (double)p * (double)sa;
    if ((double)got != exact) record(err, (uint32_t)p, code, __float_as_uint(got), __float_as_uint((float)exact));
  }
}

template <uint32_t (*OP)(uint32_t)>
__global__ void k_dec4(uint64_t base, uint64_t n, Err *err) {
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t w = (uint32_t)k, got = OP(w);
    uint32_t exp = 0;
    for (int j = 0; j < 4; ++j) {
      const uint32_t nb = (w >> (4 * j)) & 15;
      const int t[8] = {0, 1, 2, 3, 4, 6, 8, 12};
      exp |= (uint32_t)(uint8_t)(int8_t)((nb & 8) ? -t[nb & 7] : t[nb & 7]) << (8 * j);
    }
    if (got != exp) record(err, w, 0, got, exp);
  }
}

// the FP32 decode of every nibble at every position (E = 0: value, E = 1: 2 * value)
__global__ void k_e2m1_decode(Err *err) {
  const uint32_t nb = threadIdx.x & 15, j = threadIdx.x >> 4;  // 128 threads
  const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = (nb & 8) ? -t[nb & 7] : t[nb & 7];
  const uint32_t w = (nb << (4 * j)) | (0x99999999u & ~(15u << (4 * j)));  // neighbours non-zero
  if (__float_as_uint(nvfp4a::e2m1_value<0>(w, j)) != __float_as_uint(v)) record(err, nb, j, __float_as_uint(nvfp4a::e2m1_value<0>(w, j)), __float_as_uint(v));
  if (__float_as_uint(nvfp4a::e2m1_value<1>(w, j)) != __float_as_uint(2 * v)) record(err, nb, j, __float_as_uint(nvfp4a::e2m1_value<1>(w, j)), __float_as_uint(2 * v));
}

// (nibble, finite scale, BF16 or FP16 value): the product of the decoded operands is exact when it is a
// normal FP32 number (the documented exception: underflow below 2^-126)
template <int T>
__global__ void k_prod(uint64_t base, uint64_t n, Err *err) {
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t i = (uint32_t)k, nb = i & 15, code = (i >> 4) & 0xFF, x = i >> 12 & 0xFFFF;
    if ((code & 0x7F) == 0x7F) continue;
    float b;
    if (T == NVFP4A_BF16) b = __uint_as_float(x << 16);
    else b = __half2float(__ushort_as_half((unsigned short)x));
    if (!isfinite(b)) continue;
    const float a = nvfp4a::mul_rn(nvfp4a::e2m1_value<0>(nb, 0), nvfp4a::e4m3_value(code));
    const double exact = (double)a * (double)b;
    const float got = nvfp4a::fma_rn(a, b, 0.f);
    if (fabs(exact) < 0x1p-126 || fabs(exact) > 0x1.fffffep127) continue;
    if ((double)got != exact) record(err, i, 0, __float_as_uint(got), __float_as_uint((float)exact));
  }
}

template <uint32_t (*OP)(uint32_t)>
__global__ void k_rne(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    uint32_t u = (uint32_t)k, exp;
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) exp = 0x7FC0;  // NaN
    else {
      uint32_t hi = u >> 16, lo = u & 0xFFFF;
      exp = (lo > 0x8000 || (lo == 0x8000 && (hi & 1))) ? hi + 1 : hi;
    }
    uint32_t got = OP(u);
    if (got != exp) record(err, u, 0, got, exp);
  }
}

// Independent FP32 -> E2M1 / E4M3 reference: binary search over the midpoints between consecutive
// non-negative values (exact in FP32).  E2M1: codes 0..7 (max 6, NaN -> +-6); E4M3: codes 0..126 (max
// 0x7E = 448, NaN -> 0x7F).
__constant__ float c_mid2[7], c_mid4[126], c_mid5[254];  // c_mid[c] = (value(c) + value(c+1)) / 2
template <int BITS> __device__ __forceinline__ uint32_t enc_ref(float x) {
  const uint32_t u = __float_as_uint(x);
  if (BITS == 5) {  // UE5M3: codes 0..0xFE (max 114688), negative values / -0 / NaN -> 0
    if (u >= 0x80000000u || x != x) return 0u;
    if (x >= 114688.f) return 0xFEu;
    int lo = 0, hi = 254;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (x < c_mid5[mid] || (x == c_mid5[mid] && (mid & 1) == 0)) hi = mid; else lo = mid + 1;
    }
    return (uint32_t)lo;
  }
  const int top = BITS == 4 ? 7 : 126;
  const float *mids = BITS == 4 ? c_mid2 : c_mid4;
  const uint32_t s = BITS == 4 ? (u >> 28) & 8u : (u >> 24) & 0x80u;
  const float a = fabsf(x);
  if (a != a) return BITS == 4 ? s | 7u : 0x7Fu;
  if (a >= (BITS == 4 ? 6.f : 448.f)) return s | (uint32_t)top;
  int lo = 0, hi = top;  // answer c in [0, top]: first c with a < mid[c] (or == and c even)
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (a < mids[mid] || (a == mids[mid] && (mid & 1) == 0)) hi = mid; else lo = mid + 1;
  }
  return s | (uint32_t)lo;
}

template <int BITS, uint32_t (*OP)(uint32_t)>
__global__ void k_enc(uint64_t base, uint64_t n, Err *err) {
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    const uint32_t u = (uint32_t)k, got = OP(u), exp = enc_ref<BITS>(__uint_as_float(u));
    if (got != exp) record(err, u, 0, got, exp);
  }
}

// division by a constant through its reciprocal (div_by) against IEEE division, every FP32 x
template <int D>
__global__ void k_div(uint64_t base, uint64_t n, Err *err) {
  const float d = (float)D, r = __fdiv_rn(1.f, d);
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for (uint64_t k = base + blockIdx.x * blockDim.x + threadIdx.x; k < base + n; k += stride) {
    const uint32_t u = (uint32_t)k;
    const float x = __uint_as_float(u), g = nvfp4a::div_by(x, d, r), e = __fdiv_rn(x, d);
    if (__float_as_uint(g) != __float_as_uint(e) && !(g != g && e != e)) record(err, u, 0, __float_as_uint(g), __float_as_uint(e));
  }
}

template <class LAUNCH>
static int run(const char *name, LAUNCH launch, uint64_t total) {
  Err *d; cudaMalloc(&d, sizeof(Err)); cudaMemset(d, 0, sizeof(Err));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const uint64_t chunk = 1ull << 28;
  for (uint64_t b = 0; b < total; b += chunk) launch(b, total - b < chunk ? total - b : chunk, d);
  cudaEventRecord(e1);
  cudaError_t ce = cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  Err h; cudaMemcpy(&h, d, sizeof(Err), cudaMemcpyDeviceToHost); cudaFree(d);
  if (ce) { printf("%-14s CUDA error %s\n", name, cudaGetErrorString(ce)); return 1; }
  printf("%-14s %s  cases=%llu  mismatches=%llu  (%.1f s)\n", name, h.count ? "FAIL" : "PASS",
         (unsigned long long)total, h.count, ms / 1e3);
  for (unsigned e = 0; e < h.count && e < 8; ++e)
    printf("   in=%08x %08x got=%08x exp=%08x\n", h.x[e], h.y[e], h.got[e], h.exp[e]);
  return h.count != 0;
}

#ifndef NVFP4A_NEGATIVE_CONTROLS
// ---- nvfp4a_quantize_ex / nvfp4a_quantize / nvfp4a_dequantize ----------------------------------------
// Host emulation of Transformer Engine's NVFP4 quantization (the formulas of include/nvfp4effma.h).
struct QSpec {
  const char *name;
  int tile, fos;         // 1 / 16; nvfp4a_4over6
  unsigned flags;        // of both destinations (NVFP4A_GLOBAL_PER_ROW: rowwise only)
  int smax;
  bool compute, col;     // compute_amax; also a columnwise destination
};
struct HostQ {           // one quantized NVFP4 matrix [orows][ocols]
  int orows, ocols, ld, sld; bool swz, ue;
  std::vector<uint8_t> data, sc;
};
static float hscale(uint8_t b, bool ue) { return ue ? ue5(b) : e4(b); }
static uint32_t hscale_code(float d, bool ue) { return ue ? nvfp4a::f32_to_ue5m3(d) : nvfp4a::f32_to_e4m3(d); }
static uint32_t hpack(const float *x, float inv, uint32_t w[2]) {
  w[0] = w[1] = 0;
  for (int e = 0; e < 16; ++e) w[e / 8] |= nvfp4a::f32_to_e2m1(x[e] * inv) << (4 * (e % 8));
  return 0;
}
static float hfos_err(const float *x, const uint32_t w[2], float sf, float am, float denom, int fos) {
  float err = 0.f;
  for (int e = 0; e < 16; ++e) {
    const float v = ((e2((w[e / 8] >> (4 * (e % 8))) & 15) * sf) * am) / denom, d = v - x[e];
    err = err + (fos == NVFP4A_4OVER6_MSE ? d * d : fabsf(d));
  }
  return err;
}
static float htree16(float v[16]) {  // shfl_down tree (8, 4, 2, 1) over 16 lanes, value of lane 0
  for (int o = 8; o; o >>= 1) {
    float n[16];
    for (int i = 0; i < 16; ++i) n[i] = i + o < 16 ? v[i] + v[i + o] : v[i];
    memcpy(v, n, sizeof n);
  }
  return v[0];
}
// xv(orow, oc): the value to quantize (0 outside); gl(orow): the global value of the row
template <class XV, class GL>
static void href_quant(HostQ &q, XV xv, GL gl, const QSpec &sp, bool amax_mode, float M) {
  const float kMax = 3.40282347e38f;
  const int ocb = (q.ocols + 15) / 16;
  for (int t = 0; t < (q.orows + 15) / 16; ++t)
    for (int c = 0; c < ocb; ++c) {
      float x[16][16], bmax[16], tmax = 0.f;
      for (int i = 0; i < 16; ++i) {
        bmax[i] = 0.f;
        for (int e = 0; e < 16; ++e) { x[i][e] = xv(16 * t + i, 16 * c + e); bmax[i] = fmaxf(bmax[i], fabsf(x[i][e])); }
        tmax = fmaxf(tmax, bmax[i]);
      }
      uint32_t code[16], w[16][2];
      float e4v[16], e6v[16];
      uint32_t c4[16], c6[16], w4[16][2], w6[16][2];
      for (int i = 0; i < 16; ++i) {
        const int orow = 16 * t + i;
        const float bm = sp.tile == 16 ? tmax : bmax[i], gv = gl(orow < q.orows ? orow : 0);
        float se, sd;
        if (amax_mode) { se = fminf((M * 6.f) / gv, kMax); if (gv == 0.f || se == 0.f) se = 1.f; sd = 1.f / se; }
        else { sd = gv; se = fminf(1.f / gv, kMax); }
        if (!sp.fos) {
          code[i] = hscale_code(fminf(bm * (se * (1.f / 6.f)), kMax), q.ue);
          hpack(x[i], fminf(1.f / (hscale((uint8_t)code[i], q.ue) * sd), kMax), w[i]);
        } else {
          const float base = (bm / 6.f) * se;
          c4[i] = nvfp4a::f32_to_e4m3(fminf(base * 1.5f, 448.f)); c6[i] = nvfp4a::f32_to_e4m3(fminf(base, 448.f));
          const float sf4 = e4((uint8_t)c4[i]), sf6 = e4((uint8_t)c6[i]);
          hpack(x[i], fminf(1.f / (sf4 * sd), kMax), w4[i]); hpack(x[i], fminf(1.f / (sf6 * sd), kMax), w6[i]);
          e4v[i] = hfos_err(x[i], w4[i], sf4, gv, 6.f * M, sp.fos); e6v[i] = hfos_err(x[i], w6[i], sf6, gv, 6.f * M, sp.fos);
        }
      }
      if (sp.fos) {
        float t4 = 0.f, t6 = 0.f;
        if (sp.tile == 16) { float a[16], b[16]; memcpy(a, e4v, sizeof a); memcpy(b, e6v, sizeof b); t4 = htree16(a); t6 = htree16(b); }
        for (int i = 0; i < 16; ++i) {
          const bool p4 = sp.tile == 16 ? t4 < t6 : e4v[i] < e6v[i];
          code[i] = p4 ? c4[i] : c6[i]; w[i][0] = p4 ? w4[i][0] : w6[i][0]; w[i][1] = p4 ? w4[i][1] : w6[i][1];
        }
      }
      for (int i = 0; i < 16; ++i) {
        const int orow = 16 * t + i;
        if (orow >= q.orows) continue;
        memcpy(&q.data[(size_t)orow * q.ld / 2 + 8 * c], w[i], 8);
        q.sc[sc_pos(q.swz, q.sld, q.ocols, orow, c)] = (uint8_t)code[i];
      }
    }
}

static int test_quant(nvfp4a_dtype t, int rows, int cols, const QSpec &sp) {
  const int lds = cols + 16;
  std::vector<uint8_t> hx((size_t)rows * lds * esize(t));
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < cols / 16; ++b) {
      // 0: zero block, 1: tiny (scale underflows), 2: huge (saturates), else normal; row 1 all zero
      const int kind = r == 1 ? 0 : rnd() % 8;
      const float mag = kind == 1 ? 1e-9f : kind == 2 ? 3e4f : ldexpf(1.f, (int)(rnd() % 24) - 12);
      for (int i = 0; i < 16; ++i) hstore(t, hx.data(), (size_t)r * lds + 16 * b + i, kind == 0 ? 0.f : frnd() * mag);
    }
  auto X = [&](int r, int c) { return r < rows && c < cols ? hload(t, hx.data(), (size_t)r * lds + c) : 0.f; };
  const bool amax_mode = sp.flags & NVFP4A_GLOBAL_AMAX, per_row = sp.flags & NVFP4A_GLOBAL_PER_ROW;
  const bool swz = sp.flags & NVFP4A_SCALES_SWIZZLED, ue = sp.flags & NVFP4A_SCALES_UE5M3;
  const float M = sp.smax ? (float)sp.smax : ue ? 114688.f : 448.f;
  // globals: given (decode factor / amax) or computed
  float tam = 0.f; std::vector<float> ram(rows, 0.f);
  for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { tam = fmaxf(tam, fabsf(X(r, c))); ram[r] = fmaxf(ram[r], fabsf(X(r, c))); }
  std::vector<float> grow(per_row ? rows : 1), gcol(1);
  for (size_t r = 0; r < grow.size(); ++r) grow[r] = sp.compute ? 0.f : amax_mode ? 2000.f + 37.f * r : 0.0625f;
  gcol[0] = sp.compute ? 0.f : grow[0];  // a tensor's two halves share its amax (no Hadamard transform here)
  const int rows16 = (rows + 15) / 16 * 16;
  HostQ qr{rows, cols, cols + 32, cols / 16 + 3, swz, ue}, qc{cols, rows16, rows16 + 32, rows16 / 16 + 1, swz, ue};
  for (HostQ *q : {&qr, &qc}) {
    q->data.assign((size_t)q->orows * q->ld / 2, 0xEE);
    q->sc.assign(sc_bytes(swz, q->sld, q->orows, q->ocols), 0xEE);
  }
  uint8_t *dx = dev(hx), *dqr = dev(qr.data), *dsr = dev(qr.sc), *dqc = dev(qc.data), *dsc = dev(qc.sc);
  float *dgr = dev(grow), *dgc = dev(gcol), *dstr = devf(0.f), *dstc = devf(0.f);
  nvfp4a_tensor S = {dx, t, nullptr, 0, nullptr, nullptr};
  nvfp4a_tensor R = {dqr, NVFP4A_NVFP4, dsr, swz ? 0 : qr.sld, dgr, dstr, sp.flags, sp.smax};
  nvfp4a_tensor C = {dqc, NVFP4A_NVFP4, dsc, swz ? 0 : qc.sld, dgc, dstc, sp.flags & ~NVFP4A_GLOBAL_PER_ROW, sp.smax};
  const nvfp4a_quant_config cfg = {sp.tile, (nvfp4a_4over6)sp.fos, sp.compute ? 1 : 0};
  nvfp4a_status st = nvfp4a_quantize_ex(&S, lds, rows, cols, &R, qr.ld, sp.col ? &C : nullptr, qc.ld, &cfg, 0);
  cudaError_t ce = cudaDeviceSynchronize();
  if (sp.compute) {  // what the library must have written
    if (per_row) grow = ram; else grow[0] = tam;
    gcol[0] = tam;
  }
  std::vector<float> ggr(grow.size()), ggc(1); float gst[2];
  cudaMemcpy(ggr.data(), dgr, ggr.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(ggc.data(), dgc, 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(&gst[0], dstr, 4, cudaMemcpyDeviceToHost); cudaMemcpy(&gst[1], dstc, 4, cudaMemcpyDeviceToHost);
  int fails = st != NVFP4A_OK || ce != cudaSuccess;
  if (st != NVFP4A_OK) printf("   status %s: %s\n", nvfp4a_status_string(st), nvfp4a_last_error());
  bool gl_ok = ggr == grow && (!sp.col || ggc == gcol) && gst[0] == tam && (!sp.col || gst[1] == tam);
  for (int col = 0; col < (sp.col ? 2 : 1); ++col) {
    HostQ &q = col ? qc : qr;
    HostQ e = q;
    e.data.assign(q.data.size(), 0xEE); e.sc.assign(q.sc.size(), 0xEE);
    const std::vector<float> &g = col ? gcol : grow;
    href_quant(e, [&](int r, int c) { return col ? X(c, r) : X(r, c); },
               [&](int r) { return g[per_row && !col ? r : 0]; }, sp, amax_mode, M);
    std::vector<uint8_t> gd(q.data.size()), gs(q.sc.size());
    cudaMemcpy(gd.data(), col ? dqc : dqr, gd.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(gs.data(), col ? dsc : dsr, gs.size(), cudaMemcpyDeviceToHost);
    size_t bad = 0;
    for (size_t i = 0; i < gd.size(); ++i) bad += gd[i] != e.data[i];
    for (size_t i = 0; i < gs.size(); ++i) bad += gs[i] != e.sc[i];
    q.data = gd; q.sc = gs;
    printf("quantize   %-5s %4d x %5d %-26s %s: %s%s\n", DN[t], rows, cols, sp.name, col ? "col" : "row", bad ? "FAIL" : "ok",
           gl_ok ? "" : " (global / amax mismatch)");
    if (bad) printf("   %zu bytes differ\n", bad);
    fails += bad != 0 || !gl_ok;
  }
  // dequantize the rowwise result: (e2m1 * sb) * g, or e2m1 * ((sb * amax) * (1 / (6 M)))
  {
    const int ldy = cols + 48;
    std::vector<uint8_t> ey((size_t)rows * ldy * esize(t), 0xEE);
    uint8_t *dy = dev(ey); float *dam = devf(0.f);
    nvfp4a_tensor Y = {dy, t, nullptr, 0, nullptr, dam};
    nvfp4a_status s2 = nvfp4a_dequantize(&R, qr.ld, &Y, ldy, rows, cols, 0);
    std::vector<uint8_t> gy(ey.size()); float gam;
    cudaDeviceSynchronize();
    cudaMemcpy(gy.data(), dy, gy.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gam, dam, 4, cudaMemcpyDeviceToHost);
    const float finv = 1.0f / (6.f * M);
    float eam = 0.f;
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        const float en = e2((qr.data[(size_t)r * qr.ld / 2 + c / 2] >> (4 * (c & 1))) & 15);
        const float sb = hscale(qr.sc[sc_pos(swz, qr.sld, cols, r, c / 16)], ue), gv = grow[per_row ? r : 0];
        const float v = amax_mode ? en * ((sb * gv) * finv) : (en * sb) * gv;
        eam = fmaxf(eam, fabsf(v));
        hstore(t, ey.data(), (size_t)r * ldy + c, v);
      }
    size_t bad = 0;
    for (size_t i = 0; i < ey.size(); ++i) bad += ey[i] != gy[i];
    printf("dequantize %-5s %4d x %5d %-26s    : %s%s\n", DN[t], rows, cols, sp.name, bad || s2 ? "FAIL" : "ok",
           gam == eam ? "" : " (amax mismatch)");
    fails += bad != 0 || s2 != NVFP4A_OK || gam != eam;
    cudaFree(dy); cudaFree(dam);
  }
  // 16 x 16 tiles: the transpose of the rowwise result is the columnwise one, bit for bit
  if (sp.tile == 16 && sp.col && rows % 16 == 0 && !per_row) {
    std::vector<uint8_t> td(qc.data.size(), 0xEE), ts(qc.sc.size(), 0xEE);
    uint8_t *dtd = dev(td), *dts = dev(ts);
    nvfp4a_tensor T = {dtd, NVFP4A_NVFP4, dts, swz ? 0 : qc.sld, dgc, nullptr, C.flags, sp.smax};
    nvfp4a_status s3 = nvfp4a_transpose(&R, qr.ld, &T, qc.ld, rows, cols, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(td.data(), dtd, td.size(), cudaMemcpyDeviceToHost); cudaMemcpy(ts.data(), dts, ts.size(), cudaMemcpyDeviceToHost);
    // compare the matrix part (data rows of rows16 = rows values, scales of the matrix)
    size_t bad = 0;
    for (int r = 0; r < cols; ++r)
      for (int c = 0; c < rows / 2; ++c) bad += td[(size_t)r * qc.ld / 2 + c] != qc.data[(size_t)r * qc.ld / 2 + c];
    for (int r = 0; r < cols; ++r)
      for (int c = 0; c < rows / 16; ++c) bad += ts[sc_pos(swz, qc.sld, rows, r, c)] != qc.sc[sc_pos(swz, qc.sld, rows, r, c)];
    printf("transpose  %-5s %4d x %5d %-26s    : %s\n", DN[t], rows, cols, sp.name, bad || s3 ? "FAIL" : "ok");
    fails += bad != 0 || s3 != NVFP4A_OK;
    cudaFree(dtd); cudaFree(dts);
  }
  for (void *p : {(void *)dx, (void *)dqr, (void *)dsr, (void *)dqc, (void *)dsc, (void *)dgr, (void *)dgc, (void *)dstr, (void *)dstc})
    cudaFree(p);
  return fails;
}

// nvfp4a_quantize (1 x 16, the destination's global) and its INT8 destination
static int test_quant_simple(nvfp4a_dtype t, int rows, int cols) {
  int fails = 0;
  const QSpec sp = {"nvfp4a_quantize", 1, 0, 0, 0, false, false};
  // the NVFP4 path of nvfp4a_quantize is nvfp4a_quantize_ex with one rowwise destination
  fails += test_quant(t, rows, cols, sp);
  const int lds = cols + 16;
  std::vector<uint8_t> hx((size_t)rows * lds * esize(t));
  for (size_t i = 0; i < hx.size() / esize(t); ++i) hstore(t, hx.data(), i, frnd() * ldexpf(1.f, (int)(rnd() % 20) - 10));
  uint8_t *dx = dev(hx);
  nvfp4a_tensor X = {dx, t, nullptr, 0, nullptr, nullptr};
  // INT8 (W4A8 activations): sb = amax / 127, q = RNE(x / sb)
  {
    int8_t *d8; float *s8;
    const int ld8 = cols + 16, lsc8 = cols / 16 + 4;
    cudaMalloc(&d8, (size_t)rows * ld8); cudaMalloc(&s8, (size_t)rows * lsc8 * 4);
    cudaMemset(d8, 0x55, (size_t)rows * ld8); cudaMemset(s8, 0x55, (size_t)rows * lsc8 * 4);
    nvfp4a_tensor Q8 = {d8, NVFP4A_INT8, s8, lsc8, nullptr, nullptr};
    fails += nvfp4a_quantize(&X, lds, &Q8, ld8, rows, cols, 0) != NVFP4A_OK;
    std::vector<int8_t> g8((size_t)rows * ld8), e8(g8.size(), 0x55);
    std::vector<float> gs8((size_t)rows * lsc8), es8(gs8.size());
    { uint32_t u = 0x55555555u; float f; memcpy(&f, &u, 4); for (auto &x : es8) x = f; }
    cudaDeviceSynchronize();
    cudaMemcpy(g8.data(), d8, g8.size(), cudaMemcpyDeviceToHost); cudaMemcpy(gs8.data(), s8, gs8.size() * 4, cudaMemcpyDeviceToHost);
    for (int r = 0; r < rows; ++r)
      for (int b = 0; b < cols / 16; ++b) {
        float x[16], bmax = 0.f;
        for (int i = 0; i < 16; ++i) { x[i] = hload(t, hx.data(), (size_t)r * lds + 16 * b + i); bmax = fmaxf(bmax, fabsf(x[i])); }
        const float sb = bmax / 127.f;
        for (int i = 0; i < 16; ++i)
          e8[(size_t)r * ld8 + 16 * b + i] = (int8_t)(sb > 0.f ? fminf(fmaxf(rintf(x[i] / sb), -127.f), 127.f) : 0.f);
        es8[(size_t)r * lsc8 + b] = sb;
      }
    const bool bad = memcmp(g8.data(), e8.data(), g8.size()) != 0 || memcmp(gs8.data(), es8.data(), gs8.size() * 4) != 0;
    printf("quantize   %-5s %4d x %5d -> INT8: %s\n", DN[t], rows, cols, bad ? "FAIL" : "ok");
    fails += bad;
    cudaFree(d8); cudaFree(s8);
  }
  cudaFree(dx);
  return fails;
}

// nvfp4a_swizzle_scales: linear -> swizzled (padding 0) -> linear
static int test_swizzle(int rows, int cols) {
  const int ld = cols / 16 + 5;
  std::vector<uint8_t> lin((size_t)rows * ld), swz(sc_bytes(true, 0, rows, cols), 0xEE), exp(swz.size(), 0), back(lin.size(), 0xEE);
  for (auto &x : lin) x = rnd();
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols / 16; ++c) exp[sc_pos(true, 0, cols, r, c)] = lin[(size_t)r * ld + c];
  uint8_t *dl = dev(lin), *ds = dev(swz), *db = dev(back);
  nvfp4a_status s1 = nvfp4a_swizzle_scales(dl, ds, ld, rows, cols, 1, 0), s2 = nvfp4a_swizzle_scales(ds, db, ld, rows, cols, 0, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(swz.data(), ds, swz.size(), cudaMemcpyDeviceToHost); cudaMemcpy(back.data(), db, back.size(), cudaMemcpyDeviceToHost);
  size_t bad = 0;
  for (size_t i = 0; i < swz.size(); ++i) bad += swz[i] != exp[i];
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < ld; ++c) bad += back[(size_t)r * ld + c] != (c < cols / 16 ? lin[(size_t)r * ld + c] : 0xEE);
  printf("swizzle    scales %4d x %5d: %s\n", rows, cols, bad || s1 || s2 ? "FAIL" : "ok");
  cudaFree(dl); cudaFree(ds); cudaFree(db);
  return bad != 0 || s1 || s2;
}
#endif

#define BLOCK(OP) [](uint64_t b, uint64_t n, Err *d) { k_block<OP><<<1024, 256>>>(b, n, d); }
#define DEC4(OP) [](uint64_t b, uint64_t n, Err *d) { k_dec4<OP><<<256, 256>>>(b, n, d); }
#define PROD(T) [](uint64_t b, uint64_t n, Err *d) { k_prod<T><<<1024, 256>>>(b, n, d); }
#define RNE(OP) [](uint64_t b, uint64_t n, Err *d) { k_rne<OP><<<1024, 256>>>(b, n, d); }
#define ENC(B, OP) [](uint64_t b, uint64_t n, Err *d) { k_enc<B, OP><<<1024, 256>>>(b, n, d); }

int main(int argc, char **argv) {
  {  // midpoint tables for the encoder references
    float m2[7], m4[126];
    for (int c = 0; c < 7; ++c) m2[c] = 0.5f * (e2(c) + e2(c + 1));
    for (int c = 0; c < 126; ++c) m4[c] = 0.5f * (e4((uint8_t)c) + e4((uint8_t)(c + 1)));
    cudaMemcpyToSymbol(c_mid2, m2, sizeof m2);
    cudaMemcpyToSymbol(c_mid4, m4, sizeof m4);
    float m5[254];
    for (int c = 0; c < 254; ++c) m5[c] = 0.5f * (ue5((uint8_t)c) + ue5((uint8_t)(c + 1)));
    cudaMemcpyToSymbol(c_mid5, m5, sizeof m5);
  }
  const uint64_t total = (argc > 1 && !strcmp(argv[1], "--quick")) ? 1ull << 26 : 1ull << 32;
  const uint64_t blocks = (2ull * 24384 + 1) * 256;
#ifndef NVFP4A_NEGATIVE_CONTROLS
  int fails = 0;
  fails += run("block", BLOCK(block_ok), blocks);
  fails += run("dec4", DEC4(dec4_ok), 1ull << 16);
  fails += run("e2m1 decode", [](uint64_t, uint64_t, Err *d) { k_e2m1_decode<<<1, 128>>>(d); }, 128);
  fails += run("prod bf16", PROD(NVFP4A_BF16), 1ull << 28);
  fails += run("prod fp16", PROD(NVFP4A_FP16), 1ull << 28);
  fails += run("rne", RNE(rne_ok), total);
  fails += run("e2m1", ENC(4, e2m1_ok), total);
  fails += run("e2m1 sel", ENC(4, e2m1_sel_ok), total);
  fails += run("div 1536", [](uint64_t b, uint64_t n, Err *d) { k_div<1536><<<1024, 256>>>(b, n, d); }, total);
  fails += run("div 2688", [](uint64_t b, uint64_t n, Err *d) { k_div<2688><<<1024, 256>>>(b, n, d); }, total);
  fails += run("e4m3", ENC(8, e4m3_ok), total);
  fails += run("ue5m3", ENC(5, ue5m3_ok), total);
  for (nvfp4a_dtype t : {NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32}) {
    fails += test_quant_simple(t, 3, 16);
    fails += test_quant_simple(t, 517, 1056);
  }
  {
    const unsigned SW = NVFP4A_SCALES_SWIZZLED, UE = NVFP4A_SCALES_UE5M3, AM = NVFP4A_GLOBAL_AMAX, PR = NVFP4A_GLOBAL_PER_ROW;
    const QSpec specs[] = {
        {"1x16 amax given, swizzled", 1, 0, AM | SW, 0, false, true},
        {"16x16 amax computed", 16, 0, AM, 0, true, true},
        {"16x16 amax given, swizzled", 16, 0, AM | SW, 0, false, true},
        {"1x16 per-row amax computed", 1, 0, AM | PR, 0, true, true},
        {"1x16 4over6 MAE, 256", 1, NVFP4A_4OVER6_MAE, AM, 256, true, true},
        {"1x16 4over6 MSE, 448", 1, NVFP4A_4OVER6_MSE, AM | SW, 448, true, true},
        {"16x16 4over6 MAE, 256", 16, NVFP4A_4OVER6_MAE, AM | SW, 256, true, true},
        {"16x16 4over6 MSE, 256", 16, NVFP4A_4OVER6_MSE, AM, 256, false, true},
        {"1x16 per-row 4over6 MAE", 1, NVFP4A_4OVER6_MAE, AM | PR, 256, true, false},
        {"1x16 UE5M3 amax, swizzled", 1, 0, UE | AM | SW, 0, true, true},
        {"16x16 UE5M3 given g", 16, 0, UE, 0, false, true},
    };
    for (const QSpec &sp : specs)
      for (nvfp4a_dtype t : {NVFP4A_BF16, NVFP4A_FP32}) {
        fails += test_quant(t, 517, 1056, sp);
        fails += test_quant(t, 256, 320, sp);
      }
    fails += test_quant(NVFP4A_FP16, 40, 48, specs[6]);
  }
  fails += test_swizzle(517, 1056) + test_swizzle(3, 16) + test_swizzle(128, 64);
  printf(fails ? "\nFAILED\n" : "\nALL PASS\n");
  return fails != 0;
#else
  int caught = 0;
  caught += run("neg_block", BLOCK(block_twostep), blocks);
  caught += run("neg_dec4", DEC4(dec4_nosign), 1ull << 16);
  caught += run("neg_rne_nan", RNE(rne_nonan), total);
  caught += run("neg_e2m1_away", ENC(4, e2m1_ties_away), total);
  caught += run("neg_e4m3_away", ENC(8, e4m3_ties_away), total);
  printf(caught == 5 ? "\nall 5 broken variants rejected\n" : "\nNEGATIVE CONTROL NOT CAUGHT\n");
  return caught != 5;
#endif
}
