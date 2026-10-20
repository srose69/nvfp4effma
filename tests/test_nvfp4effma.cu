// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA core (src/nvfp4effma.cuh), exhaustively, and nvfp4a_quantize / nvfp4a_dequantize:
//  1. block  : the DP4A kernel's scale step -- y = fma(bits(0x4B400000 + p), sa, -1.5 * 2^23 * sa) ==
//              p * sa exactly, for every block sum p in [-2304, 2304] and every finite E4M3 scale.
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
//              blocks, padded leading dimensions, against the host emulation of the documented formulas.
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
__device__ __forceinline__ uint32_t e4m3_ok(uint32_t u) { return nvfp4a::f32_to_e4m3(__uint_as_float(u)); }
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
    const int p = (int)(k >> 8) - 2304;
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
__constant__ float c_mid2[7], c_mid4[126];  // c_mid[c] = (value(c) + value(c+1)) / 2
template <int BITS> __device__ __forceinline__ uint32_t enc_ref(float x) {
  const uint32_t u = __float_as_uint(x);
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
// ---- nvfp4a_quantize / nvfp4a_dequantize ------------------------------------------------------------
static int test_quant(nvfp4a_dtype t, int rows, int cols) {
  const int lds = cols + 16, ldq = cols + 32, ldsc = cols / 16 + 3;
  std::vector<uint8_t> hx((size_t)rows * lds * esize(t));
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < cols / 16; ++b) {
      const int kind = rnd() % 8;  // 0: zero block, 1: tiny (scale underflows), 2: huge (saturates), else normal
      const float mag = kind == 1 ? 1e-9f : kind == 2 ? 3e4f : ldexpf(1.f, (int)(rnd() % 24) - 12);
      for (int i = 0; i < 16; ++i) hstore(t, hx.data(), (size_t)r * lds + 16 * b + i, kind == 0 ? 0.f : frnd() * mag);
    }
  const float g = 0.0625f;
  uint8_t *dx = dev(hx), *dq, *dsc; cudaMalloc(&dq, (size_t)rows * ldq / 2); cudaMalloc(&dsc, (size_t)rows * ldsc);
  cudaMemset(dq, 0xEE, (size_t)rows * ldq / 2); cudaMemset(dsc, 0xEE, (size_t)rows * ldsc);
  float *dg = devf(g), *dam = devf(0.f), *dam2 = devf(0.f);
  nvfp4a_tensor X = {dx, t, nullptr, 0, nullptr, nullptr}, Q = {dq, NVFP4A_NVFP4, dsc, ldsc, dg, dam};
  int fails = nvfp4a_quantize(&X, lds, &Q, ldq, rows, cols, 0) != NVFP4A_OK;
  std::vector<uint8_t> gq((size_t)rows * ldq / 2), gsc((size_t)rows * ldsc), eq = gq, esc = gsc;
  cudaDeviceSynchronize();
  cudaMemcpy(gq.data(), dq, gq.size(), cudaMemcpyDeviceToHost); cudaMemcpy(gsc.data(), dsc, gsc.size(), cudaMemcpyDeviceToHost);
  float gam; cudaMemcpy(&gam, dam, 4, cudaMemcpyDeviceToHost);
  // host emulation: sb = E4M3(amax / (6 g)), q = E2M1(x / (sb g)); bytes outside the matrix untouched
  memset(eq.data(), 0xEE, eq.size()); memset(esc.data(), 0xEE, esc.size());
  float eam = 0.f;
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < cols / 16; ++b) {
      float x[16], bmax = 0.f;
      for (int i = 0; i < 16; ++i) { x[i] = hload(t, hx.data(), (size_t)r * lds + 16 * b + i); bmax = fmaxf(bmax, fabsf(x[i])); }
      eam = fmaxf(eam, bmax);
      const uint32_t sc = nvfp4a::f32_to_e4m3(bmax / (6.f * g));
      const float d = e4((uint8_t)sc) * g;
      uint8_t *row = eq.data() + (size_t)r * ldq / 2 + 8 * b;
      for (int i = 0; i < 16; ++i) {
        uint32_t u; memcpy(&u, &x[i], 4);
        const uint32_t q = sc ? nvfp4a::f32_to_e2m1(x[i] / d) : (u >> 28) & 8u;
        row[i / 2] = (uint8_t)((i & 1) ? (row[i / 2] & 0x0F) | (q << 4) : q);
      }
      esc[(size_t)r * ldsc + b] = (uint8_t)sc;
    }
  size_t bad = 0;
  for (size_t i = 0; i < eq.size(); ++i) bad += eq[i] != gq[i];
  for (size_t i = 0; i < esc.size(); ++i) bad += esc[i] != gsc[i];
  printf("quantize   %s %4d x %5d: %s%s\n", DN[t], rows, cols, bad ? "FAIL" : "ok", gam == eam ? "" : " (amax mismatch)");
  fails += bad != 0 || gam != eam;

  // dequantize the result back: v = (e2m1 * sb) * g, RNE to t
  const int ldy = cols + 48;
  uint8_t *dy; cudaMalloc(&dy, (size_t)rows * ldy * esize(t)); cudaMemset(dy, 0xEE, (size_t)rows * ldy * esize(t));
  nvfp4a_tensor Y = {dy, t, nullptr, 0, nullptr, dam2};
  fails += nvfp4a_dequantize(&Q, ldq, &Y, ldy, rows, cols, 0) != NVFP4A_OK;
  std::vector<uint8_t> gy((size_t)rows * ldy * esize(t)), ey(gy.size(), 0xEE);
  cudaDeviceSynchronize();
  cudaMemcpy(gy.data(), dy, gy.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gam, dam2, 4, cudaMemcpyDeviceToHost);
  eam = 0.f;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const uint32_t nb = (eq[(size_t)r * ldq / 2 + c / 2] >> (4 * (c & 1))) & 15;
      const float v = (e2(nb) * e4(esc[(size_t)r * ldsc + c / 16])) * g;
      eam = fmaxf(eam, fabsf(v));
      hstore(t, ey.data(), (size_t)r * ldy + c, v);
    }
  bad = 0;
  for (size_t i = 0; i < ey.size(); ++i) bad += ey[i] != gy[i];
  printf("dequantize %s %4d x %5d: %s%s\n", DN[t], rows, cols, bad ? "FAIL" : "ok", gam == eam ? "" : " (amax mismatch)");
  fails += bad != 0 || gam != eam;
  for (void *p : {(void *)dx, (void *)dq, (void *)dsc, (void *)dg, (void *)dam, (void *)dam2, (void *)dy}) cudaFree(p);
  return fails;
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
  }
  const uint64_t total = (argc > 1 && !strcmp(argv[1], "--quick")) ? 1ull << 26 : 1ull << 32;
  const uint64_t blocks = 4609ull * 256;
#ifndef NVFP4A_NEGATIVE_CONTROLS
  int fails = 0;
  fails += run("block", BLOCK(block_ok), blocks);
  fails += run("dec4", DEC4(dec4_ok), 1ull << 16);
  fails += run("e2m1 decode", [](uint64_t, uint64_t, Err *d) { k_e2m1_decode<<<1, 128>>>(d); }, 128);
  fails += run("prod bf16", PROD(NVFP4A_BF16), 1ull << 28);
  fails += run("prod fp16", PROD(NVFP4A_FP16), 1ull << 28);
  fails += run("rne", RNE(rne_ok), total);
  fails += run("e2m1", ENC(4, e2m1_ok), total);
  fails += run("e4m3", ENC(8, e4m3_ok), total);
  for (nvfp4a_dtype t : {NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32}) {
    fails += test_quant(t, 3, 16);
    fails += test_quant(t, 517, 1056);
  }
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
