// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Host side of the NVFP4EffMA tests: element helpers and a bit-exact emulation of nvfp4a_gemm in the
// documented operation order (include/nvfp4effma.h): exact block sums for NVFP4 x NVFP4 in the tiled
// kernels, one FFMA per MAC for a dense B, and the GEMV kernel's lane order.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/nvfp4effma.h"
#include "../src/nvfp4effma.cuh"  // f32_to_e2m1 / f32_to_e4m3 (host + device)

static uint32_t rng = 0x2545F491u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frnd() { return (float)(rnd() & 0xFFFFFF) / 16777216.0f * 2.f - 1.f; }
static const char *DN[] = {"NVFP4", "BF16", "FP16", "FP32", "INT8"};  // by nvfp4a_dtype

static float e4(uint8_t b) {  // E4M3, NaN included
  int e = (b >> 3) & 15, f = b & 7;
  float v = (b & 0x7F) == 0x7F ? NAN : e ? ldexpf(1.f + f / 8.f, e - 7) : ldexpf(f / 8.f, -6);
  return (b & 0x80) ? -v : v;
}
static float ue5(uint8_t b) {  // UE5M3 (unsigned, bias 15)
  int e = (b >> 3) & 31, f = b & 7;
  return e ? ldexpf(1.f + f / 8.f, e - 15) : ldexpf(f / 8.f, -14);
}
// position of block scale (r, c) of a [rows][cols] NVFP4 matrix: linear (ld scales per row) or swizzled
// (the cuBLASLt / TE GEMM layout, written out from its definition)
static size_t sc_pos(bool swz, int ld, int cols, int r, int c) {
  if (!swz) return (size_t)r * ld + c;
  const int sc = (cols / 16 + 3) / 4 * 4;
  return ((size_t)(r / 128) * (sc / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + c % 4;
}
static size_t sc_bytes(bool swz, int ld, int rows, int cols) {
  return swz ? (size_t)(rows + 127) / 128 * 128 * ((cols / 16 + 3) / 4 * 4) : (size_t)rows * ld;
}
static float e2(uint32_t n) {  // E2M1 nibble
  static const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (n & 8) ? -t[n & 7] : t[n & 7];
}
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t to_bf(float v) {
  uint32_t u; memcpy(&u, &v, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static size_t esize(nvfp4a_dtype t) { return t == NVFP4A_FP32 ? 4 : 2; }
static float hload(nvfp4a_dtype t, const void *p, size_t i) {
  switch (t) {
    case NVFP4A_BF16: return bf(((const uint16_t *)p)[i]);
    case NVFP4A_FP16: return __half2float(((const __half *)p)[i]);
    default: return ((const float *)p)[i];
  }
}
static void hstore(nvfp4a_dtype t, void *p, size_t i, float v) {
  switch (t) {
    case NVFP4A_BF16: ((uint16_t *)p)[i] = to_bf(v); break;
    case NVFP4A_FP16: ((__half *)p)[i] = __float2half_rn(v); break;
    default: ((float *)p)[i] = v;
  }
}
static void fill(nvfp4a_dtype t, std::vector<uint8_t> &buf, float range) {
  for (size_t i = 0; i < buf.size() / esize(t); ++i) hstore(t, buf.data(), i, frnd() * range);
}
template <class T> static T *dev(const std::vector<T> &h) {
  T *d; cudaMalloc(&d, h.size() * sizeof(T) + 16); cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
static float *devf(float v) { float *d; cudaMalloc(&d, 4); cudaMemcpy(d, &v, 4, cudaMemcpyHostToDevice); return d; }

// A host NVFP4 matrix: rows x cols values, ld values per row, ldsc scale bytes per row (linear layout);
// flags / scale_max as nvfp4a_tensor; global: one value (g) or one per row, a decode factor or an amax
struct HostFp4 {
  int rows, cols, ld, ldsc;
  unsigned flags; int smax;
  std::vector<uint8_t> data, sc;
  std::vector<float> gv;
  float g;
  HostFp4(int r, int c, int ld_, int ldsc_, float g_, unsigned fl = 0, int sm = 0)
      : rows(r), cols(c), ld(ld_), ldsc(ldsc_), flags(fl), smax(sm), data((size_t)r * ld_ / 2),
        sc(sc_bytes(fl & NVFP4A_SCALES_SWIZZLED, ldsc_, r, c)), g(g_) {
    for (auto &x : data) x = rnd();
    // finite positive scales over the whole range of the type, some zero blocks
    for (auto &x : sc) x = rnd() % 16 == 0 ? 0 : (fl & NVFP4A_SCALES_UE5M3) ? 1 + rnd() % 0xFD : 1 + rnd() % 0x7E;
    gv.assign((fl & NVFP4A_GLOBAL_PER_ROW) ? (r > 0 ? r : 1) : 1, g_);
    if (fl & NVFP4A_GLOBAL_PER_ROW)
      for (auto &x : gv) x = g_ * (0.5f + (rnd() & 255) / 256.f);
  }
  bool swz() const { return flags & NVFP4A_SCALES_SWIZZLED; }
  int ld_api() const { return swz() ? 0 : ldsc; }
  uint32_t nib(int r, int c) const { return (data[(size_t)r * ld / 2 + c / 2] >> (4 * (c & 1))) & 15; }
  uint8_t code(int r, int c) const { return sc[sc_pos(swz(), ldsc, cols, r, c / 16)]; }
  float scale(int r, int c) const { return (flags & NVFP4A_SCALES_UE5M3) ? ue5(code(r, c)) : e4(code(r, c)); }
  float val(int r, int c) const { return e2(nib(r, c)) * scale(r, c); }  // exact
  float M() const { return smax ? (float)smax : (flags & NVFP4A_SCALES_UE5M3) ? 114688.f : 448.f; }
  float global(int r) const { return gv[(flags & NVFP4A_GLOBAL_PER_ROW) ? r : 0]; }
};

// A host INT8 activation matrix: int8 values, one FP32 scale per 16 values of a row
struct HostQ8 {
  int rows, cols, ld, ldsc;
  std::vector<int8_t> data;
  std::vector<float> sc;
  float g;
  HostQ8(int r, int c, int ld_, int ldsc_, float g_) : rows(r), cols(c), ld(ld_), ldsc(ldsc_), data((size_t)r * ld_), sc((size_t)r * ldsc_), g(g_) {
    for (auto &x : data) x = (int8_t)(rnd() % 255 - 127);
    for (auto &x : sc) x = rnd() % 16 == 0 ? 0.f : ldexpf(1.f + (rnd() & 1023) / 1024.f, (int)(rnd() % 12) - 10);
  }
  int q(int r, int c) const { return data[(size_t)r * ld + c]; }
  float scale(int r, int c) const { return sc[(size_t)r * ldsc + c / 16]; }
};

// ---- nvfp4a_gemm ------------------------------------------------------------------------------
struct Case {
  int m, n, k;
  nvfp4a_dtype bt, dt; int ct;  // ct: C dtype, -1 = accumulate into D (C = NULL)
  int bias;                     // -1 none, else bias dtype
  float alpha, beta;
  unsigned fa = 0, fb = 0;      // NVFP4 flags of A / B (scale layout and type, global kind)
  int ma = 0, mb = 0;           // their scale_max
};

// Every operand format: the block scale formats of the kernel instances (src/nvfp4effma.cuh kF_*, bits 0..3)
// and the kinds of globals the epilogue handles (bits 4..6 here: amax, per-row A, per-row B) as NVFP4 flags of
// A / B; the amax bit goes to A, B or both by v, scale_max 448 / 256 alternating for E4M3.  kTestFormats for
// NVFP4 B; the other B dtypes have no B bits: test_format_a(i), i < kTestFormatsA.
constexpr int kTF_AMAX = 16, kTF_AROW = 32, kTF_BROW = 64, kTestFormats = 128, kTestFormatsA = 16;
inline int test_format_a(int i) { return (i & 3) | (i & 4 ? kTF_AMAX : 0) | (i & 8 ? kTF_AROW : 0); }
static void set_formats(Case &c, int f, int v) {
  using namespace nvfp4a;
  const bool fp4b = c.bt == NVFP4A_NVFP4;
  c.fa = (f & kF_ASWZ ? NVFP4A_SCALES_SWIZZLED : 0u) | (f & kF_AUE ? NVFP4A_SCALES_UE5M3 : 0u) |
         (f & kTF_AROW ? NVFP4A_GLOBAL_PER_ROW : 0u);
  c.fb = fp4b ? (f & kF_BSWZ ? NVFP4A_SCALES_SWIZZLED : 0u) | (f & kF_BUE ? NVFP4A_SCALES_UE5M3 : 0u) |
                (f & kTF_BROW ? NVFP4A_GLOBAL_PER_ROW : 0u) : 0u;
  if (f & kTF_AMAX) {
    const int on = fp4b ? v % 3 : 0;  // 0: A, 1: B, 2: both
    if (on != 1) c.fa |= NVFP4A_GLOBAL_AMAX;
    if (on != 0) c.fb |= NVFP4A_GLOBAL_AMAX;
  }
  c.ma = (c.fa & NVFP4A_GLOBAL_AMAX) && !(f & kF_AUE) && v % 2 ? 256 : 0;
  c.mb = (c.fb & NVFP4A_GLOBAL_AMAX) && !(f & kF_BUE) && v % 4 >= 2 ? 256 : 0;
}

// Runs nvfp4a_gemm on random data and compares D and amax bit for bit with the host emulation.
// nvfp4a_gemm_tc (B NVFP4, INT8, BF16) is tried on every case; NVFP4A_NOT_SUPPORTED from the library (no tensor-core
// path for this B on this device or build) is reported, not failed.  Its FP32 sums are the tensor cores' (order and
// rounding inside the mma) and stream-K's (a tile's parts summed), so its outputs are checked against a float64
// reference within the bound of an FP32 accumulation over k plus the output's rounding instead of bit for bit.
static bool tc_dtype(nvfp4a_dtype bt) { return bt == NVFP4A_NVFP4 || bt == NVFP4A_INT8 || bt == NVFP4A_BF16; }
// relative half-ulp of an output dtype (and its smallest normal step, as an absolute floor)
static double out_rel(nvfp4a_dtype t) { return t == NVFP4A_BF16 ? 0x1p-8 : t == NVFP4A_FP16 ? 0x1p-11 : 0x1p-24; }
static double out_abs(nvfp4a_dtype t) { return t == NVFP4A_FP16 ? 0x1p-24 : t == NVFP4A_BF16 ? 0x1p-133 : 0x1p-149; }

static int run_case(const Case &c) {
  const bool fp4b = c.bt == NVFP4A_NVFP4, q8b = c.bt == NVFP4A_INT8, blk = fp4b || q8b;
  const bool gemv = c.n <= nvfp4a::kern::kGemvMaxN[c.bt];
  const int lda = c.k + 64, ldb = c.k + (fp4b ? 128 : 64), ldd = c.m + 3, ldc = c.ct >= 0 ? c.m + 1 : ldd;
  const nvfp4a_dtype ct = c.ct >= 0 ? (nvfp4a_dtype)c.ct : c.dt;
  // globals: decode factors, or amaxes in the range a real tensor would have for its scale_max
  HostFp4 hA(c.m, c.k, lda, c.k / 16 + 4, (c.fa & NVFP4A_GLOBAL_AMAX) ? 37.5f : 0.375f, c.fa, c.ma);
  HostFp4 hBq(fp4b ? c.n : 0, c.k, ldb, c.k / 16 + 8, (c.fb & NVFP4A_GLOBAL_AMAX) ? 175.f : 1.75f, c.fb, c.mb);
  HostQ8 hB8(q8b ? c.n : 0, c.k, ldb, c.k / 16 + 8, 1.75f);
  std::vector<uint8_t> hB(blk ? 0 : (size_t)ldb * c.n * esize(c.bt));
  std::vector<uint8_t> hC((size_t)ldc * c.n * esize(ct)), hD((size_t)ldd * c.n * esize(c.dt)), hbias(c.m * 4);
  if (!blk) fill(c.bt, hB, 2.f);
  fill(ct, hC, 4.f);
  if (c.ct < 0) { fill(c.dt, hD, 4.f); hC = hD; } else memset(hD.data(), 0xA5, hD.size());
  if (c.bias >= 0) fill((nvfp4a_dtype)c.bias, hbias, 8.f);

  uint8_t *dA = dev(hA.data), *dSA = dev(hA.sc);
  void *dB = fp4b ? (void *)dev(hBq.data) : q8b ? (void *)dev(hB8.data) : (void *)dev(hB);
  void *dSB = fp4b ? (void *)dev(hBq.sc) : q8b ? (void *)dev(hB8.sc) : nullptr;
  uint8_t *dC = dev(hC), *dD = dev(hD), *dbias = dev(hbias);
  float *dga = dev(hA.gv), *dgb = fp4b ? dev(hBq.gv) : devf(hB8.g), *damax = devf(0.f);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, hA.ld_api(), dga, nullptr, c.fa, c.ma};
  nvfp4a_tensor B = {dB, c.bt, dSB, fp4b ? hBq.ld_api() : q8b ? hB8.ldsc : 0, blk ? dgb : nullptr, nullptr, fp4b ? c.fb : 0u,
                     fp4b ? c.mb : 0};
  nvfp4a_tensor C = {dC, ct, nullptr, 0, nullptr, nullptr}, D = {dD, c.dt, nullptr, 0, nullptr, damax};
  nvfp4a_tensor bias = {dbias, c.bias >= 0 ? (nvfp4a_dtype)c.bias : NVFP4A_FP32, nullptr, 0, nullptr, nullptr};
  nvfp4a_status st = nvfp4a_gemm(c.m, c.n, c.k, c.alpha, &A, lda, &B, ldb, c.beta, c.ct >= 0 ? &C : nullptr, ldc, &D, ldd,
                                 c.bias >= 0 ? &bias : nullptr, 0);
  cudaError_t ce = cudaDeviceSynchronize();
  std::vector<uint8_t> got(hD.size()); float gamax;
  cudaMemcpy(got.data(), dD, got.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gamax, damax, 4, cudaMemcpyDeviceToHost);

  // the same product through nvfp4a_gemm_tc: images of A and B, the (once zeroed) workspace, D from its initial values
  bool tc = tc_dtype(c.bt);
  nvfp4a_status st2 = NVFP4A_OK;
  std::vector<uint8_t> got2, got3;  // nvfp4a_gemm_tc, nvfp4a_gemv_tc (n <= 8: A's image, B as it is)
  float gamax2 = 0.f, gamax3 = 0.f;
  const bool gv = c.n <= 8;
  if (tc) {
    static void *ws = nullptr;
    if (!ws) { cudaMalloc(&ws, nvfp4a_tc_workspace_bytes()); cudaMemset(ws, 0, nvfp4a_tc_workspace_bytes()); }
    void *ia, *ib;
    const int ka = c.bt == NVFP4A_BF16 ? NVFP4A_TC_A16 : NVFP4A_TC_A8;  // A's image kind for B's dtype (GEMM and GEMV)
    cudaMalloc(&ia, nvfp4a_tc_image_bytes(ka, NVFP4A_NVFP4, c.m, c.k) + 16);
    cudaMalloc(&ib, nvfp4a_tc_image_bytes(NVFP4A_TC_B, c.bt, c.n, c.k) + 16);
    uint8_t *dD2 = dev(hD); float *damax2 = devf(0.f);
    st2 = nvfp4a_tc_image(ka, &A, lda, c.m, c.k, ia, 0);
    if (st2 == NVFP4A_OK) st2 = nvfp4a_tc_image(NVFP4A_TC_B, &B, ldb, c.n, c.k, ib, 0);
    nvfp4a_tensor Ai = A, Bi = B, D2 = {dD2, c.dt, nullptr, 0, nullptr, damax2};
    Ai.data = ia; Bi.data = ib;
    if (st2 == NVFP4A_OK)
      st2 = nvfp4a_gemm_tc(c.m, c.n, c.k, c.alpha, &Ai, &Bi, c.beta, c.ct >= 0 ? &C : nullptr, ldc, &D2, ldd, c.bias >= 0 ? &bias : nullptr,
                           ws, 0);
    if (cudaError_t e2 = cudaDeviceSynchronize()) ce = e2;
    got2.resize(hD.size());
    cudaMemcpy(got2.data(), dD2, got2.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gamax2, damax2, 4, cudaMemcpyDeviceToHost);
    if (gv && st2 == NVFP4A_OK) {  // the GEMV on the same image of A, B as it is
      cudaMemcpy(dD2, hD.data(), hD.size(), cudaMemcpyHostToDevice); cudaMemset(damax2, 0, 4);
      st2 = nvfp4a_gemv_tc(c.m, c.n, c.k, c.alpha, &Ai, &B, ldb, c.beta, c.ct >= 0 ? &C : nullptr, ldc, &D2, ldd,
                           c.bias >= 0 ? &bias : nullptr, 0);
      if (cudaError_t e2 = cudaDeviceSynchronize()) ce = e2;
      got3.resize(hD.size());
      cudaMemcpy(got3.data(), dD2, got3.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gamax3, damax2, 4, cudaMemcpyDeviceToHost);
    }
    for (void *q : {ia, ib, (void *)dD2, (void *)damax2}) cudaFree(q);
    if (st2 == NVFP4A_NOT_SUPPORTED) { printf("   nvfp4a_gemm_tc: %s\n", nvfp4a_last_error()); tc = false; st2 = NVFP4A_OK; }
  }

  auto Bval = [&](int j, int kk) { return fp4b ? hBq.val(j, kk) : hload(c.bt, hB.data(), (size_t)j * ldb + kk); };
  // NVFP4 / INT8 B: the exact integer block sum and the two scale steps
  auto block = [&](int i, int j, int kb, float acc) {
    int p = 0;
    for (int kk = kb; kk < kb + 16; ++kk)
      p += (int)(2 * e2(hA.nib(i, kk))) * (fp4b ? (int)(2 * e2(hBq.nib(j, kk))) : hB8.q(j, kk));
    return fmaf((float)p * hA.scale(i, kb), fp4b ? hBq.scale(j, kb) : hB8.scale(j, kb), acc);
  };
  const float kout = fp4b ? 0.25f : 0.5f;
  // output scale ((alpha * vA(i)) * vB(j)) * gf, gf from the amax globals in Transformer Engine's order
  const bool aa = c.fa & NVFP4A_GLOBAL_AMAX, ab = fp4b && (c.fb & NVFP4A_GLOBAL_AMAX);
  const float gf = aa && ab ? 1.0f / (6.f * 6.f * hA.M() * hBq.M()) : aa ? 1.0f / (6.f * hA.M()) : ab ? 1.0f / (6.f * hBq.M()) : 1.f;
  auto scale_ij = [&](int i, int j) {
    return ((c.alpha * hA.global(i)) * (fp4b ? hBq.global(j) : q8b ? hB8.g : 1.f)) * gf;
  };
  std::vector<uint8_t> exp = hD, exp2, exp3;
  float eamax = 0.f;
  double worst = 0., worst3 = 0.;  // tc: the largest |got2 / got3 - reference| / bound
  for (int j = 0; j < c.n; ++j)
    for (int i = 0; i < c.m; ++i) {
      float acc = 0.f;
      if (gemv) {
        float lanes[32];
        for (int l = 0; l < 32; ++l) {
          float p = 0.f;
          for (int kb = 16 * l; kb < c.k; kb += 512) {
            if (blk) {  // exact block sum, then the two scale steps; x 1/4 or 1/2 after the butterfly
              p = block(i, j, kb, p);
            } else {
              for (int jj = 0; jj < 16; ++jj) p = fmaf(hA.val(i, kb + jj), Bval(j, kb + jj), p);
            }
          }
          lanes[l] = p;
        }
        for (int o = 16; o; o >>= 1) {
          float nx[32];
          for (int l = 0; l < 32; ++l) nx[l] = lanes[l] + lanes[l ^ o];
          memcpy(lanes, nx, sizeof nx);
        }
        acc = blk ? lanes[0] * kout : lanes[0];
      } else if (blk) {
        for (int kb = 0; kb < c.k; kb += 16) acc = block(i, j, kb, acc);
        acc *= kout;
      } else {
        for (int kk = 0; kk < c.k; ++kk) acc = fmaf(hA.val(i, kk), Bval(j, kk), acc);
      }
      float v = acc * scale_ij(i, j);
      if (c.bias >= 0) v = v + hload((nvfp4a_dtype)c.bias, hbias.data(), i);
      if (c.beta != 0.f) v = fmaf(c.beta, hload(ct, hC.data(), (size_t)j * ldc + i), v);
      eamax = fmaxf(eamax, fabsf(v));
      hstore(c.dt, exp.data(), (size_t)j * ldd + i, v);
      if (tc) {  // float64 reference; bound: FP32 accumulation of k terms, the epilogue's FP32 steps, the output rounding
        double r = 0., a = 0.;
        for (int kk = 0; kk < c.k; ++kk) {
          const double t = (double)hA.val(i, kk) * (q8b ? hB8.q(j, kk) * (double)hB8.scale(j, kk) : (double)Bval(j, kk));
          r += t; a += fabs(t);
        }
        const double s = scale_ij(i, j), bias = c.bias >= 0 ? hload((nvfp4a_dtype)c.bias, hbias.data(), i) : 0.;
        const double cv = c.beta != 0.f ? c.beta * (double)hload(ct, hC.data(), (size_t)j * ldc + i) : 0.;
        const double v64 = r * s + bias + cv;
        const double bound = (c.k + 2) * 0x1p-24 * a * fabs(s) + 4 * 0x1p-24 * (fabs(r * s) + fabs(bias) + fabs(cv)) +
                             out_rel(c.dt) * fabs(v64) + out_abs(c.dt);
        if (exp2.empty()) exp2 = hD;  // (the padding stays as it was)
        const float g = hload(c.dt, got2.data(), (size_t)j * ldd + i);
        worst = fmax(worst, fabs(g - v64) / bound);
        hstore(c.dt, exp2.data(), (size_t)j * ldd + i, fabs(g - v64) <= bound ? g : (float)v64);
        if (!got3.empty()) {
          if (exp3.empty()) exp3 = hD;
          const float g3 = hload(c.dt, got3.data(), (size_t)j * ldd + i);
          worst3 = fmax(worst3, fabs(g3 - v64) / bound);
          hstore(c.dt, exp3.data(), (size_t)j * ldd + i, fabs(g3 - v64) <= bound ? g3 : (float)v64);
        }
      }
    }
  size_t bad = 0, bad2 = 0, bad3 = 0;
  for (size_t i = 0; i < exp.size(); ++i) bad += exp[i] != got[i];
  if (tc && !exp2.empty()) for (size_t i = 0; i < exp2.size(); ++i) bad2 += exp2[i] != got2[i];
  if (tc && !exp3.empty()) for (size_t i = 0; i < exp3.size(); ++i) bad3 += exp3[i] != got3[i];
  const bool amax_ok = gamax == eamax, amax2_ok = !tc || fabs(gamax2 - eamax) <= 0x1p-7 * eamax;
  const bool amax3_ok = !tc || got3.empty() || fabs(gamax3 - eamax) <= 0x1p-7 * eamax;
  const int fail = st != NVFP4A_OK || st2 != NVFP4A_OK || ce != cudaSuccess || bad || bad2 || bad3 || !amax_ok || !amax2_ok || !amax3_ok;
  printf("%s m=%4d n=%4d k=%5d B=%-5s D=%s C=%-5s bias=%-4s a=%.2f b=%5.2f fa=%x/%d fb=%x/%d : %s%s\n", gemv ? "gemv" : "gemm",
         c.m, c.n, c.k, DN[c.bt], DN[c.dt], c.ct >= 0 ? DN[c.ct] : "accum", c.bias >= 0 ? DN[c.bias] : "-", c.alpha, c.beta,
         c.fa, c.ma, c.fb, c.mb, fail ? "FAIL" : "ok", amax_ok ? "" : " (amax mismatch)");
  if (bad) printf("   %zu bytes differ\n", bad);
  if (tc) printf("   nvfp4a_gemm_tc: max |error| / bound = %.3f%s%s\n", worst, bad2 ? " (out of bound)" : "", amax2_ok ? "" : " (amax mismatch)");
  if (tc && !got3.empty())
    printf("   nvfp4a_gemv_tc: max |error| / bound = %.3f%s%s\n", worst3, bad3 ? " (out of bound)" : "", amax3_ok ? "" : " (amax mismatch)");
  if (st != NVFP4A_OK) printf("   status %s: %s\n", nvfp4a_status_string(st), nvfp4a_last_error());
  if (st2 != NVFP4A_OK) printf("   nvfp4a_gemm_tc status %s: %s\n", nvfp4a_status_string(st2), nvfp4a_last_error());
  if (ce) printf("   cuda %s\n", cudaGetErrorString(ce));
  for (void *p : {(void *)dA, (void *)dSA, (void *)dB, (void *)dSB, (void *)dC, (void *)dD, (void *)dbias, (void *)dga,
                  (void *)dgb, (void *)damax})
    cudaFree(p);
  return fail;
}
