// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_quantize / nvfp4a_dequantize bandwidth: bytes read + written (dense values, NVFP4 values and
// block scales) over time, 8192 x 8192.  Median time per row after an idle pause (see bench_util.h).
#include <vector>
#include "../include/nvfp4effma.h"
#include "bench_util.h"

int main() {
  const int R = 8192, C = 8192;
  const size_t n = (size_t)R * C;
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  int memory_clock_khz = 0;  // cudaDeviceProp::memoryClockRate is gone in CUDA 13
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);
  const double peak_bw = 2.0 * memory_clock_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
  Bench bench;
  void *x, *q, *s;
  float *g;
  cudaMalloc(&x, n * 4); cudaMalloc(&q, n / 2); cudaMalloc(&s, n / 16); cudaMalloc(&g, 4);
  cudaMemset(x, 0, n * 4); cudaMemset(q, 0, n / 2); cudaMemset(s, 0x38, n / 16);
  const float one = 1.f; cudaMemcpy(g, &one, 4, cudaMemcpyHostToDevice);
  printf("nvfp4a_quantize / nvfp4a_dequantize, %d x %d; DRAM peak %.0f GB/s\n", R, C, peak_bw);
  nvfp4a_tensor Q = {q, NVFP4A_NVFP4, s, 0, g, nullptr};
  for (nvfp4a_dtype t : {NVFP4A_BF16, NVFP4A_FP32}) {
    nvfp4a_tensor X = {x, t, nullptr, 0, nullptr, nullptr};
    const double bytes = n * (t == NVFP4A_FP32 ? 4 : 2) + n / 2 + n / 16;
    const char *tn = t == NVFP4A_FP32 ? "FP32" : "BF16";
    if (!bench_want("")) break;  // a filter selects TE rows only
    BenchRow r = bench.run([&] { nvfp4a_quantize(&X, C, &Q, C, R, C, 0); }, 50);
    printf("  quantize   %s -> NVFP4: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz\n", tn, r.ms, bytes / r.ms / 1e6,
           100 * bytes / r.ms / 1e6 / peak_bw, r.mhz);
    r = bench.run([&] { nvfp4a_dequantize(&Q, C, &X, C, R, C, 0); }, 50);
    printf("  dequantize NVFP4 -> %s: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz\n", tn, r.ms, bytes / r.ms / 1e6,
           100 * bytes / r.ms / 1e6 / peak_bw, r.mhz);
    fflush(stdout);
  }

  // Transformer Engine formats (nvfp4a_quantize_ex / nvfp4a_dequantize), each against "plain" of the same
  // dtype in the same run (x = time / plain): one feature at a time, then the recipes.  Bytes: the dense
  // matrix once plus every NVFP4 output written (read for dequantize).
  printf("\nTransformer Engine formats (x = time / plain of the same dtype)\n");
  {
    void *q2, *s2; float *gr, *gc; cudaMalloc(&q2, n / 2); cudaMalloc(&s2, n / 16); cudaMalloc(&gr, R * 4); cudaMalloc(&gc, 4);
    cudaMemcpy(gc, &one, 4, cudaMemcpyHostToDevice);
    { std::vector<float> h(R, 1.f); cudaMemcpy(gr, h.data(), R * 4, cudaMemcpyHostToDevice); }
    const unsigned SW = NVFP4A_SCALES_SWIZZLED, UE = NVFP4A_SCALES_UE5M3, AM = NVFP4A_GLOBAL_AMAX, PR = NVFP4A_GLOBAL_PER_ROW;
    struct QFmt { const char *name; unsigned flags; int smax, tile; nvfp4a_4over6 fos; int amax; bool row, col; };
    const QFmt qs[] = {
        {"plain", 0, 0, 1, NVFP4A_4OVER6_OFF, 0, true, false},
        {"swizzled", SW, 0, 1, NVFP4A_4OVER6_OFF, 0, true, false},
        {"UE5M3", UE, 0, 1, NVFP4A_4OVER6_OFF, 0, true, false},
        {"amax global", AM, 0, 1, NVFP4A_4OVER6_OFF, 0, true, false},
        {"compute amax", AM, 0, 1, NVFP4A_4OVER6_OFF, 1, true, false},
        {"16x16", AM, 0, 16, NVFP4A_4OVER6_OFF, 0, true, false},
        {"4over6 MAE", AM, 256, 1, NVFP4A_4OVER6_MAE, 0, true, false},
        {"4over6 MSE", AM, 256, 1, NVFP4A_4OVER6_MSE, 0, true, false},
        {"16x16 4over6 MAE", AM, 256, 16, NVFP4A_4OVER6_MAE, 0, true, false},
        {"row-scaled (compute)", AM | PR, 0, 1, NVFP4A_4OVER6_OFF, 1, true, false},
        {"columnwise", AM, 0, 1, NVFP4A_4OVER6_OFF, 0, false, true},
        {"16x16 columnwise", AM, 0, 16, NVFP4A_4OVER6_OFF, 0, false, true},
        {"TE X: row+col, swz, compute", AM | SW, 0, 1, NVFP4A_4OVER6_OFF, 1, true, true},
        {"TE W: row+col 16x16, swz, compute", AM | SW, 0, 16, NVFP4A_4OVER6_OFF, 1, true, true},
    };
    for (nvfp4a_dtype t : {NVFP4A_BF16, NVFP4A_FP32}) {
      nvfp4a_tensor X = {x, t, nullptr, 0, nullptr, nullptr};
      const char *tn = t == NVFP4A_FP32 ? "FP32" : "BF16";
      float plain = 0.f;
      for (const QFmt &f : qs) {
        if (&f != qs && !bench_want(f.name)) continue;  // "plain": the baseline of x, always
        nvfp4a_tensor Rw = {q, NVFP4A_NVFP4, s, 0, (f.flags & PR) ? gr : g, nullptr, f.flags, f.smax};
        nvfp4a_tensor Cw = {q2, NVFP4A_NVFP4, s2, 0, gc, nullptr, f.flags & ~PR, f.smax};
        const nvfp4a_quant_config cfg = {f.tile, f.fos, f.amax};
        const double bytes = n * (t == NVFP4A_FP32 ? 4 : 2) + (double)(f.row + f.col) * (n / 2 + n / 16);
        nvfp4a_status st = NVFP4A_OK;
        BenchRow r = bench.run([&] { st = nvfp4a_quantize_ex(&X, C, R, C, f.row ? &Rw : nullptr, C, f.col ? &Cw : nullptr, R, &cfg, 0); }, 50);
        if (!plain) plain = r.ms;
        printf("  TE quantize   %s %s%s: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz   x%.3f\n", tn, f.name,
               st == NVFP4A_OK ? "" : " FAILED", r.ms, bytes / r.ms / 1e6, 100 * bytes / r.ms / 1e6 / peak_bw, r.mhz, r.ms / plain);
        fflush(stdout);
      }
      plain = 0.f;
      for (const QFmt &f : qs) {  // dequantization: the formats it reads
        if (f.tile != 1 || f.fos || f.amax || !f.row || f.col || (&f != qs && !bench_want(f.name))) continue;
        nvfp4a_tensor Rw = {q, NVFP4A_NVFP4, s, 0, g, nullptr, f.flags, f.smax};
        const double bytes = n * (t == NVFP4A_FP32 ? 4 : 2) + n / 2 + n / 16;
        nvfp4a_status st = NVFP4A_OK;
        BenchRow r = bench.run([&] { st = nvfp4a_dequantize(&Rw, C, &X, C, R, C, 0); }, 50);
        if (!plain) plain = r.ms;
        printf("  TE dequantize %s %s%s: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz   x%.3f\n", tn, f.name,
               st == NVFP4A_OK ? "" : " FAILED", r.ms, bytes / r.ms / 1e6, 100 * bytes / r.ms / 1e6 / peak_bw, r.mhz, r.ms / plain);
        fflush(stdout);
      }
      if (bench_want("row-scaled")) {  // per-row globals (row-scaled tensors) on the dequantization side
        nvfp4a_tensor Rw = {q, NVFP4A_NVFP4, s, 0, gr, nullptr, AM | PR, 0};
        const double bytes = n * (t == NVFP4A_FP32 ? 4 : 2) + n / 2 + n / 16;
        nvfp4a_status st = NVFP4A_OK;
        BenchRow r = bench.run([&] { st = nvfp4a_dequantize(&Rw, C, &X, C, R, C, 0); }, 50);
        printf("  TE dequantize %s row-scaled%s: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz   x%.3f\n", tn,
               st == NVFP4A_OK ? "" : " FAILED", r.ms, bytes / r.ms / 1e6, 100 * bytes / r.ms / 1e6 / peak_bw, r.mhz, r.ms / plain);
        fflush(stdout);
      }
    }
  }
  return 0;
}
