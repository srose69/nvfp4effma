// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm throughput: NVFP4 weights x NVFP4 / INT8 / BF16 / FP16 / FP32 activations -> BF16, and FP32
// accumulation, vs cuBLAS SGEMM of the same shape (FP32 in/out).  Median time per row; % of FP32 peak =
// SMs x 128 FFMA x the SM clock measured during that very row (see bench_util.h) -- the DP4A kernel
// (NVFP4 x NVFP4 / INT8, sm_61+) does more than one MAC per FP32 issue slot and can exceed 100 %.
#include <vector>
#include <cublas_v2.h>
#include "../include/nvfp4effma.h"
#include "bench_util.h"

int main(int argc, char **argv) {
  const int S = argc > 1 ? atoi(argv[1]) : 4096, M = S, N = S, K = S;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s, %d SMs; M=N=K=%d; median of %d runs per row after %d s idle; %% of FP32 peak at the row's SM clock\n\n",
         p.name, p.multiProcessorCount, S, Bench::env("NVFP4A_BENCH_REPS", 20), Bench::env("NVFP4A_BENCH_COOL", 2));
  printf("%-50s %9s %13s %9s %12s\n", "", "time", "", "SM clock", "of FP32 peak");
  Bench bench;

  // random NVFP4 codes with scales around 1 (the values do not change the speed)
  std::vector<unsigned char> h((size_t)M * K / 2), hs((size_t)M * K / 16);
  unsigned r = 1;
  auto rnd = [&] { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r; };
  for (auto &x : h) x = rnd();
  for (auto &x : hs) x = 0x30 + rnd() % 16;
  void *dA, *dB, *dsA, *dsB, *dD;
  cudaMalloc(&dA, h.size()); cudaMalloc(&dB, h.size()); cudaMalloc(&dsA, hs.size()); cudaMalloc(&dsB, hs.size());
  cudaMalloc(&dD, (size_t)M * N * 4);
  cudaMemcpy(dA, h.data(), h.size(), cudaMemcpyHostToDevice); cudaMemcpy(dB, h.data(), h.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dsA, hs.data(), hs.size(), cudaMemcpyHostToDevice); cudaMemcpy(dsB, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dsA, 0, nullptr, nullptr}, B = {dB, NVFP4A_NVFP4, dsB, 0, nullptr, nullptr};
  nvfp4a_tensor D = {dD, NVFP4A_BF16, nullptr, 0, nullptr, nullptr}, F = {dD, NVFP4A_FP32, nullptr, 0, nullptr, nullptr};
  const double macs = (double)M * N * K;
  auto row = [&](const char *name, BenchRow x) {
    const double gmacs = macs / x.ms / 1e6, peak = p.multiProcessorCount * 128 * x.mhz / 1e3;
    if (x.mhz > 0) printf("%-50s %6.3f ms %7.1f GMAC/s %5.0f MHz %10.1f%%\n", name, x.ms, gmacs, x.mhz, 100 * gmacs / peak);
    else printf("%-50s %6.3f ms %7.1f GMAC/s %9s %11s\n", name, x.ms, gmacs, "n/a", "n/a");
    fflush(stdout);
  };
  auto gemm = [&](const char *name, const nvfp4a_tensor &b, const nvfp4a_tensor &d, float beta) {
    nvfp4a_status st = NVFP4A_OK;
    const BenchRow x = bench.run([&] { st = nvfp4a_gemm(M, N, K, 1.f, &A, K, &b, K, beta, nullptr, 0, &d, M, nullptr, 0); }, 20);
    char full[96]; snprintf(full, sizeof full, "%s%s", name, st == NVFP4A_OK ? "" : " (FAILED)");
    row(full, x);
  };
  gemm("nvfp4a_gemm NVFP4 x NVFP4 -> BF16", B, D, 0.f);
  {  // W4A8: int8 activations with an FP32 scale per 16 values
    void *q8, *s8; cudaMalloc(&q8, (size_t)N * K); cudaMalloc(&s8, (size_t)N * K / 16 * 4);
    std::vector<signed char> hq((size_t)N * K); for (auto &x : hq) x = (signed char)(rnd() % 255 - 127);
    std::vector<float> hsq((size_t)N * K / 16, 0.01f);
    cudaMemcpy(q8, hq.data(), hq.size(), cudaMemcpyHostToDevice); cudaMemcpy(s8, hsq.data(), hsq.size() * 4, cudaMemcpyHostToDevice);
    nvfp4a_tensor B8 = {q8, NVFP4A_INT8, s8, 0, nullptr, nullptr};
    gemm("nvfp4a_gemm NVFP4 x INT8 -> BF16", B8, D, 0.f);
    cudaFree(q8); cudaFree(s8);
  }
  void *dX; cudaMalloc(&dX, (size_t)K * N * 4); cudaMemset(dX, 0, (size_t)K * N * 4);
  for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32}) {
    nvfp4a_tensor X = {dX, bt, nullptr, 0, nullptr, nullptr};
    char name[64]; snprintf(name, 64, "nvfp4a_gemm NVFP4 x %s -> BF16", bt == NVFP4A_BF16 ? "BF16" : bt == NVFP4A_FP16 ? "FP16" : "FP32");
    gemm(name, X, D, 0.f);
  }
  gemm("nvfp4a_gemm NVFP4 x NVFP4, FP32 D += (accumulate)", B, F, 1.f);
  float *fA, *fB, *fC; cudaMalloc(&fA, (size_t)M * K * 4); cudaMalloc(&fB, (size_t)K * N * 4); cudaMalloc(&fC, (size_t)M * N * 4);
  cudaMemset(fA, 0, (size_t)M * K * 4); cudaMemset(fB, 0, (size_t)K * N * 4);
  cublasHandle_t hb; cublasCreate(&hb);
  const float one = 1, zero = 0;
  row("cuBLAS SGEMM NN FP32 -> FP32", bench.run([&] { cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, fA, M, fB, K, &zero, fC, M); }, 20));
  row("cuBLAS SGEMM TN FP32 -> FP32", bench.run([&] { cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, fA, K, fB, K, &zero, fC, M); }, 20));
  return 0;
}
