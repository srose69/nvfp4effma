// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The extended epilogue (nvfp4a_gemm_ex, nvfp4a_gemv_tc_ex, nvfp4a_gemm_tc_ex) against the same entry without it, on the
// layers of Qwen3-0.6B and -8B with the additions each one takes: qkv the column scale, gate / up (interleaved by 8) the
// column scale and SwiGLU, o and down the second output.  Per path (CUDA cores; tensor cores where the device has them),
// x dtype and n: the plain row's time and A's bytes / s, then the extended one's and its ratio.  Median per row after an
// idle pause (bench_util.h).
#include <vector>
#include "../include/nvfp4effma.h"
#include "bench_util.h"

namespace {

struct Layer { const char *name; int m, k, mask; };  // mask: 1 column scale, 2 SwiGLU, 4 second output
const Layer kLayers[] = {{"0.6B qkv", 4096, 1024, 1},  {"0.6B gate_up", 6144, 1024, 3}, {"0.6B o", 1024, 2048, 4},
                         {"0.6B down", 1024, 3072, 4}, {"8B qkv", 6144, 4096, 1},       {"8B gate_up", 28672, 4096, 3},
                         {"8B o", 4096, 4096, 4},      {"8B down", 4096, 14336, 4}};
const int kNs[] = {1, 8, 128, 512};

template <class T> T *dev(size_t count, int byte) {
  void *p;
  cudaMalloc(&p, count * sizeof(T) + 16);
  cudaMemset(p, byte, count * sizeof(T) + 16);
  return (T *)p;
}

}  // namespace

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  Bench bench;
  printf("%s: %d SMs, %zu MiB VRAM, %d KiB L2\n", p.name, p.multiProcessorCount, p.totalGlobalMem >> 20, p.l2CacheSize >> 10);
  printf("Extended vs plain: paired AB/BA medians after row warmup; A bytes/s is effective warm-cache bandwidth.\n\n");
  printf("%-46s %9s %10s %9s %9s %9s\n", "", "plain", "A bytes/s", "extended", "", "SM MHz");
  int failures = 0;
  const int NMAX = 512;
  const char *xn[] = {"NVFP4", "BF16", "FP16", "FP32", "INT8"};
  void *ws = nullptr;
  if (nvfp4a_tc_workspace_bytes()) { ws = dev<uint8_t>(nvfp4a_tc_workspace_bytes(), 0); }
  for (const Layer &L : kLayers) {
    const int m = L.m, k = L.k, np = k / 16;
    const size_t abytes = (size_t)m * k / 2 + (size_t)m * k / 16;
    uint8_t *dA = dev<uint8_t>((size_t)m * k / 2, 0x35), *dsA = dev<uint8_t>((size_t)m * k / 16, 0x30);
    uint8_t *dX = dev<uint8_t>((size_t)k * NMAX * 2, 0x3c);
    float *dsX = dev<float>((size_t)k / 16 * NMAX, 0x3c), *parts = dev<float>((size_t)np * NMAX, 0x3c);
    uint8_t *dD = dev<uint8_t>((size_t)m * NMAX * 2, 0), *dD2 = dev<uint8_t>((size_t)m * NMAX * 2, 0), *dw2 = dev<uint8_t>((size_t)m * 2, 0x3c);
    float *dsq = dev<float>((size_t)m / 16 * NMAX, 0);
    nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dsA, 0, nullptr, nullptr, 0, 0};
    for (int tc = 0; tc < 2; ++tc) {
      if (tc && (p.major * 10 + p.minor < 75 || p.major * 10 + p.minor >= 90)) continue;
      void *ia = nullptr;
      if (tc) {
        // the image A8 for INT8 x, A16 for BF16 x: built per x dtype below
        if (nvfp4a_tc_image_bytes(NVFP4A_TC_A16, NVFP4A_NVFP4, m, k) == 0) continue;
      }
      for (nvfp4a_dtype xt : {NVFP4A_BF16, NVFP4A_INT8}) {
        nvfp4a_tensor Ai = A;
        if (tc) {
          const int kind = xt == NVFP4A_INT8 ? NVFP4A_TC_A8 : NVFP4A_TC_A16;
          cudaFree(ia);
          ia = dev<uint8_t>(nvfp4a_tc_image_bytes(kind, NVFP4A_NVFP4, m, k), 0);
          if (nvfp4a_tc_image(kind, &A, k, m, k, ia, 0)) break;  // no tensor cores here
          Ai.data = ia;
        }
        nvfp4a_tensor X = {dX, xt, xt == NVFP4A_INT8 ? dsX : nullptr, 0, nullptr, nullptr, 0, 0};
        void *ib = nullptr;
        for (int n : kNs) {
          char name[96];
          snprintf(name, sizeof name, "%s %s x=%s n=%d ex %d", L.name, tc ? "tc  " : "cuda", xn[xt], n, L.mask);
          if (!bench_want(name)) continue;
          nvfp4a_tensor Xb = X;
          if (tc && n > 8) {
            cudaFree(ib);
            ib = dev<uint8_t>(nvfp4a_tc_image_bytes(NVFP4A_TC_B, xt, n, k), 0);
            if (nvfp4a_tc_image(NVFP4A_TC_B, &X, k, n, k, ib, 0)) continue;  // (BF16 B: sm_80+)
            Xb.data = ib;
          }
          nvfp4a_epilogue ex{};
          if (L.mask & 1) { ex.norm_parts = parts; ex.norm_nparts = np; ex.norm_ld = np; ex.norm_inv_n = 1.f / k; ex.norm_eps = 1e-6f; }
          ex.swiglu = (L.mask & 2) != 0;
          if (L.mask & 4) { ex.d2 = dD2; ex.d2_dtype = NVFP4A_BF16; ex.ldd2 = m; ex.w2 = dw2; ex.w2_dtype = NVFP4A_BF16; ex.sq = dsq; ex.ld_sq = m / 16; }
          nvfp4a_tensor D = {dD, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
          nvfp4a_status s0 = NVFP4A_OK, s1 = NVFP4A_OK;
          const auto call = [&](const nvfp4a_epilogue *e, nvfp4a_status &st) {
            const int ldd = e && e->swiglu ? m / 2 : m;
            if (!tc) st = e ? nvfp4a_gemm_ex(m, n, k, 1.f, &A, k, &Xb, k, 0.f, nullptr, 0, &D, ldd, nullptr, e, 0)
                            : nvfp4a_gemm(m, n, k, 1.f, &A, k, &Xb, k, 0.f, nullptr, 0, &D, ldd, nullptr, 0);
            else if (n <= 8) st = e ? nvfp4a_gemv_tc_ex(m, n, k, 1.f, &Ai, &Xb, k, 0.f, nullptr, 0, &D, ldd, nullptr, e, 0)
                                    : nvfp4a_gemv_tc(m, n, k, 1.f, &Ai, &Xb, k, 0.f, nullptr, 0, &D, ldd, nullptr, 0);
            else st = e ? nvfp4a_gemm_tc_ex(m, n, k, 1.f, &Ai, &Xb, 0.f, nullptr, 0, &D, ldd, nullptr, e, ws, 0)
                        : nvfp4a_gemm_tc(m, n, k, 1.f, &Ai, &Xb, 0.f, nullptr, 0, &D, ldd, nullptr, ws, 0);
          };
          call(nullptr, s0); call(&ex, s1);
          if (s0 || s1) { printf("%-46s %s\n", name, nvfp4a_last_error()); ++failures; continue; }
          const BenchPair pair = bench.paired([&] { call(nullptr, s0); }, [&] { call(&ex, s1); }, 100);
          const BenchRow &a = pair.plain, &b = pair.extended;
          if (s0 || s1) { printf("%-46s %s\n", name, nvfp4a_last_error()); ++failures; continue; }
          printf("%-46s %6.3f ms %6.1f GB/s %6.3f ms   x%.3f   %5.0f\n", name, a.ms, abytes / a.ms / 1e6, b.ms, b.ms / a.ms, a.mhz);
          fflush(stdout);
        }
        cudaFree(ib);
      }
      cudaFree(ia);
    }
    for (void *q : {(void *)dA, (void *)dsA, (void *)dX, (void *)dsX, (void *)parts, (void *)dD, (void *)dD2, (void *)dw2, (void *)dsq}) cudaFree(q);
  }
  return failures != 0;
}
