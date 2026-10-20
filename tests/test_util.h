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
static const char *DN[] = {"NVFP4", "BF16", "FP16", "FP32"};  // by nvfp4a_dtype

static float e4(uint8_t b) {  // E4M3, NaN included
  int e = (b >> 3) & 15, f = b & 7;
  float v = (b & 0x7F) == 0x7F ? NAN : e ? ldexpf(1.f + f / 8.f, e - 7) : ldexpf(f / 8.f, -6);
  return (b & 0x80) ? -v : v;
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

// A host NVFP4 matrix: rows x cols values, ld values per row, ldsc scale bytes per row
struct HostFp4 {
  int rows, cols, ld, ldsc;
  std::vector<uint8_t> data, sc;
  float g;
  HostFp4(int r, int c, int ld_, int ldsc_, float g_) : rows(r), cols(c), ld(ld_), ldsc(ldsc_), data((size_t)r * ld_ / 2), sc((size_t)r * ldsc_), g(g_) {
    for (auto &x : data) x = rnd();
    // finite positive E4M3 scales over the whole range, some zero blocks
    for (auto &x : sc) x = rnd() % 16 == 0 ? 0 : 1 + rnd() % 0x7E;
  }
  uint32_t nib(int r, int c) const { return (data[(size_t)r * ld / 2 + c / 2] >> (4 * (c & 1))) & 15; }
  float scale(int r, int c) const { return e4(sc[(size_t)r * ldsc + c / 16]); }
  float val(int r, int c) const { return e2(nib(r, c)) * scale(r, c); }  // exact
};

// ---- nvfp4a_gemm ------------------------------------------------------------------------------
struct Case {
  int m, n, k;
  nvfp4a_dtype bt, dt; int ct;  // ct: C dtype, -1 = accumulate into D (C = NULL)
  int bias;                     // -1 none, else bias dtype
  float alpha, beta;
};

// Runs nvfp4a_gemm on random data and compares D and amax bit for bit with the host emulation.
static int run_case(const Case &c) {
  const bool fp4b = c.bt == NVFP4A_NVFP4, gemv = c.n <= 8;
  const int lda = c.k + 64, ldb = c.k + (fp4b ? 128 : 64), ldd = c.m + 3, ldc = c.ct >= 0 ? c.m + 1 : ldd;
  const nvfp4a_dtype ct = c.ct >= 0 ? (nvfp4a_dtype)c.ct : c.dt;
  HostFp4 hA(c.m, c.k, lda, c.k / 16 + 4, 0.375f), hBq(fp4b ? c.n : 0, c.k, ldb, c.k / 16 + 8, 1.75f);
  std::vector<uint8_t> hB(fp4b ? 0 : (size_t)ldb * c.n * esize(c.bt));
  std::vector<uint8_t> hC((size_t)ldc * c.n * esize(ct)), hD((size_t)ldd * c.n * esize(c.dt)), hbias(c.m * 4);
  if (!fp4b) fill(c.bt, hB, 2.f);
  fill(ct, hC, 4.f);
  if (c.ct < 0) { fill(c.dt, hD, 4.f); hC = hD; } else memset(hD.data(), 0xA5, hD.size());
  if (c.bias >= 0) fill((nvfp4a_dtype)c.bias, hbias, 8.f);

  uint8_t *dA = dev(hA.data), *dSA = dev(hA.sc), *dB = fp4b ? dev(hBq.data) : dev(hB), *dSB = fp4b ? dev(hBq.sc) : nullptr;
  uint8_t *dC = dev(hC), *dD = dev(hD), *dbias = dev(hbias);
  float *dga = devf(hA.g), *dgb = devf(hBq.g), *damax = devf(0.f);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, hA.ldsc, dga, nullptr};
  nvfp4a_tensor B = {dB, c.bt, dSB, fp4b ? hBq.ldsc : 0, fp4b ? dgb : nullptr, nullptr};
  nvfp4a_tensor C = {dC, ct, nullptr, 0, nullptr, nullptr}, D = {dD, c.dt, nullptr, 0, nullptr, damax};
  nvfp4a_tensor bias = {dbias, c.bias >= 0 ? (nvfp4a_dtype)c.bias : NVFP4A_FP32, nullptr, 0, nullptr, nullptr};
  nvfp4a_status st = nvfp4a_gemm(c.m, c.n, c.k, c.alpha, &A, lda, &B, ldb, c.beta, c.ct >= 0 ? &C : nullptr, ldc, &D, ldd,
                                 c.bias >= 0 ? &bias : nullptr, 0);
  cudaError_t ce = cudaDeviceSynchronize();
  std::vector<uint8_t> got(hD.size()); float gamax;
  cudaMemcpy(got.data(), dD, got.size(), cudaMemcpyDeviceToHost); cudaMemcpy(&gamax, damax, 4, cudaMemcpyDeviceToHost);

  auto Bval = [&](int j, int kk) { return fp4b ? hBq.val(j, kk) : hload(c.bt, hB.data(), (size_t)j * ldb + kk); };
  const float s = (c.alpha * hA.g) * (fp4b ? hBq.g : 1.f);
  std::vector<uint8_t> exp = hD;
  float eamax = 0.f;
  for (int j = 0; j < c.n; ++j)
    for (int i = 0; i < c.m; ++i) {
      float acc = 0.f;
      if (gemv) {
        float lanes[32];
        for (int l = 0; l < 32; ++l) {
          float p = 0.f;
          for (int kb = 16 * l; kb < c.k; kb += 512) {
            if (fp4b) {  // exact block sum, then the two scale steps; x 1/4 after the butterfly
              int q = 0;
              for (int kk = kb; kk < kb + 16; ++kk) q += (int)(2 * e2(hA.nib(i, kk))) * (int)(2 * e2(hBq.nib(j, kk)));
              p = fmaf((float)q * hA.scale(i, kb), hBq.scale(j, kb), p);
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
        acc = fp4b ? lanes[0] * 0.25f : lanes[0];
      } else if (fp4b) {
        for (int kb = 0; kb < c.k; kb += 16) {
          int p = 0;
          for (int kk = kb; kk < kb + 16; ++kk) p += (int)(2 * e2(hA.nib(i, kk))) * (int)(2 * e2(hBq.nib(j, kk)));
          acc = fmaf((float)p * hA.scale(i, kb), hBq.scale(j, kb), acc);
        }
        acc *= 0.25f;
      } else {
        for (int kk = 0; kk < c.k; ++kk) acc = fmaf(hA.val(i, kk), Bval(j, kk), acc);
      }
      float v = acc * s;
      if (c.bias >= 0) v = v + hload((nvfp4a_dtype)c.bias, hbias.data(), i);
      if (c.beta != 0.f) v = fmaf(c.beta, hload(ct, hC.data(), (size_t)j * ldc + i), v);
      eamax = fmaxf(eamax, fabsf(v));
      hstore(c.dt, exp.data(), (size_t)j * ldd + i, v);
    }
  size_t bad = 0;
  for (size_t i = 0; i < exp.size(); ++i) bad += exp[i] != got[i];
  const bool amax_ok = gamax == eamax;
  const int fail = st != NVFP4A_OK || ce != cudaSuccess || bad || !amax_ok;
  printf("%s m=%4d n=%4d k=%5d B=%-5s D=%s C=%-5s bias=%-4s a=%.2f b=%5.2f : %s%s\n", gemv ? "gemv" : "gemm", c.m, c.n,
         c.k, DN[c.bt], DN[c.dt], c.ct >= 0 ? DN[c.ct] : "accum", c.bias >= 0 ? DN[c.bias] : "-", c.alpha, c.beta,
         fail ? "FAIL" : "ok", amax_ok ? "" : " (amax mismatch)");
  if (bad) printf("   %zu bytes differ\n", bad);
  if (st != NVFP4A_OK) printf("   status %s: %s\n", nvfp4a_status_string(st), nvfp4a_last_error());
  if (ce) printf("   cuda %s\n", cudaGetErrorString(ce));
  for (void *p : {(void *)dA, (void *)dSA, (void *)dB, (void *)dSB, (void *)dC, (void *)dD, (void *)dbias, (void *)dga,
                  (void *)dgb, (void *)damax})
    cudaFree(p);
  return fail;
}
