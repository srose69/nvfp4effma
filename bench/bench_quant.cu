// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_quantize / nvfp4a_dequantize bandwidth: bytes read + written (dense values, NVFP4 values and
// block scales) over time, 8192 x 8192.  Median time per row after an idle pause (see bench_util.h).
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
    BenchRow r = bench.run([&] { nvfp4a_quantize(&X, C, &Q, C, R, C, 0); }, 50);
    printf("  quantize   %s -> NVFP4: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz\n", tn, r.ms, bytes / r.ms / 1e6,
           100 * bytes / r.ms / 1e6 / peak_bw, r.mhz);
    r = bench.run([&] { nvfp4a_dequantize(&Q, C, &X, C, R, C, 0); }, 50);
    printf("  dequantize NVFP4 -> %s: %7.3f ms  %6.1f GB/s  %3.0f%% of DRAM peak  %5.0f MHz\n", tn, r.ms, bytes / r.ms / 1e6,
           100 * bytes / r.ms / 1e6 / peak_bw, r.mhz);
    fflush(stdout);
  }
  return 0;
}
