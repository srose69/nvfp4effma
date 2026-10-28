// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm on the GEMV path (small n): NVFP4 weights (4.5 bits per value with the block scales) read
// once per group of 8 columns, so the figure of merit is the effective bandwidth over A (n = 16 and 32 read
// it 2 and 4 times).  Compared with the DRAM peak, a
// device-to-device copy (practical bandwidth) and cuBLAS SGEMV / SGEMM on FP32 weights of the same
// shape (7.1x the bytes).  Median time per row after an idle pause (see bench_util.h).
#include <vector>
#include <cublas_v2.h>
#include "../include/nvfp4effma.h"
#include "bench_util.h"

int main(int argc, char **argv) {
  const int M = argc > 1 ? atoi(argv[1]) : 8192, K = argc > 2 ? atoi(argv[2]) : 8192;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int memory_clock_khz = 0;  // cudaDeviceProp::memoryClockRate is gone in CUDA 13
  cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, 0);
  const double peak_bw = 2.0 * memory_clock_khz * 1e3 * (p.memoryBusWidth / 8) / 1e9;
  Bench bench;
  const size_t abytes = (size_t)M * K / 2 + (size_t)M * K / 16;
  void *dA, *dsA, *dX, *dsX, *dY;
  cudaMalloc(&dA, (size_t)M * K / 2); cudaMalloc(&dsA, (size_t)M * K / 16);
  cudaMalloc(&dX, (size_t)K * 32 * 4); cudaMalloc(&dsX, (size_t)K * 32 / 16 * 4); cudaMalloc(&dY, (size_t)M * 32 * 4);
  std::vector<unsigned char> h((size_t)M * K / 2), hs((size_t)M * K / 16);
  unsigned r = 7;
  auto rnd = [&] { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r; };
  for (auto &x : h) x = rnd();
  for (auto &x : hs) x = 0x30 + rnd() % 16;
  cudaMemcpy(dA, h.data(), h.size(), cudaMemcpyHostToDevice); cudaMemcpy(dsA, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemset(dX, 0, (size_t)K * 32 * 4); cudaMemset(dsX, 0x38, (size_t)K * 32 / 16 * 4);  // E4M3 bytes / FP32 words
  void *cp0, *cp1; cudaMalloc(&cp0, abytes); cudaMalloc(&cp1, abytes); cudaMemset(cp0, 0, abytes);
  const BenchRow copy = bench.run([&] { cudaMemcpyAsync(cp1, cp0, abytes, cudaMemcpyDeviceToDevice); }, 200);
  const double copy_bw = 2.0 * abytes / copy.ms / 1e6;
  printf("%s: M=%d K=%d, A = %.0f MB NVFP4 (values + block scales); DRAM peak %.0f GB/s (%d-bit bus), D2D copy %.0f GB/s (read+write)\n\n",
         p.name, M, K, abytes / 1e6, peak_bw, p.memoryBusWidth, copy_bw);
  printf("%-56s %9s %12s %8s %8s %9s\n", "", "time", "A bytes/s", "of DRAM", "of copy", "SM clock");
  auto row = [&](const char *name, BenchRow x, double bytes, const char *note) {
    const double gbs = bytes / x.ms / 1e6;
    printf("%-56s %6.3f ms %7.1f GB/s %7.0f%% %7.0f%% %5.0f MHz%s\n", name, x.ms, gbs, 100 * gbs / peak_bw, 100 * gbs / copy_bw,
           x.mhz, note);
    fflush(stdout);
  };
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dsA, 0, nullptr, nullptr}, Y = {dY, NVFP4A_BF16, nullptr, 0, nullptr, nullptr};
  const char *xn[] = {"NVFP4", "BF16", "FP16", "FP32", "INT8"};  // by nvfp4a_dtype
  for (nvfp4a_dtype xt : {NVFP4A_BF16, NVFP4A_NVFP4, NVFP4A_INT8, NVFP4A_FP32})
    for (int n : {1, 2, 4, 8, 16, 32}) {  // n > 8: groups of 8 columns, A read once per group
      nvfp4a_tensor X = {dX, xt, dsX, 0, nullptr, nullptr};
      nvfp4a_status st = NVFP4A_OK;
      const BenchRow x = bench.run([&] { st = nvfp4a_gemm(M, n, K, 1.f, &A, K, &X, K, 0.f, nullptr, 0, &Y, M, nullptr, 0); }, 200);
      char name[64]; snprintf(name, 64, "nvfp4a gemv n=%d x=%s -> BF16%s", n, xn[xt], st == NVFP4A_OK ? "" : " FAILED");
      row(name, x, (double)abytes, "");
    }
  // the extended epilogue (nvfp4a_gemm_ex): x BF16 / INT8 (A's scales linear E4M3), every mask against the
  // plain call of the same x dtype and n (SwiGLU needs n a multiple of 8)
  printf("\nExtended epilogue (x = time / the plain row of the same x dtype and n)\n");
  {
    const int np = 37;  // a typical RMSNorm's part count (the cost does not depend on it)
    void *dparts, *dw2, *dd2, *dsq;
    cudaMalloc(&dparts, (size_t)32 * np * 4); cudaMalloc(&dw2, (size_t)M * 2);
    cudaMalloc(&dd2, (size_t)32 * M * 2); cudaMalloc(&dsq, (size_t)32 * (M / 16) * 4);
    std::vector<float> hp((size_t)32 * np, 0.01f);
    std::vector<unsigned short> hw(M, 0x3f80);  // BF16 1.0
    cudaMemcpy(dparts, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw2, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    const struct { const char *name; unsigned mask; } masks[] = {
        {"norm", 1}, {"swiglu", 2}, {"norm+swiglu", 3}, {"out2", 4}, {"norm+out2", 5}};
    for (nvfp4a_dtype xt : {NVFP4A_BF16, NVFP4A_INT8})
      for (int n : {1, 8, 32}) {
        nvfp4a_tensor X = {dX, xt, xt == NVFP4A_INT8 ? dsX : nullptr, 0, nullptr, nullptr};
        nvfp4a_status st = NVFP4A_OK;
        const float plain = bench.run([&] { st = nvfp4a_gemm(M, n, K, 1.f, &A, K, &X, K, 0.f, nullptr, 0, &Y, M, nullptr, 0); }, 200).ms;
        char pname[80]; snprintf(pname, sizeof pname, "gemv n=%d x=%s plain%s", n, xn[xt], st == NVFP4A_OK ? "" : " FAILED");
        row(pname, {plain, 0.0}, (double)abytes, "");
        for (const auto &mk : masks) {
          if ((mk.mask & 2) && n % 8) continue;
          nvfp4a_epilogue ex{};
          if (mk.mask & 1) {
            ex.norm_parts = (const float *)dparts; ex.norm_nparts = ex.norm_ld = np;
            ex.norm_inv_n = 1.f / np; ex.norm_eps = 1e-6f;
          }
          if (mk.mask & 2) ex.swiglu = 1;
          if (mk.mask & 4) {
            ex.d2 = dd2; ex.d2_dtype = NVFP4A_BF16; ex.ldd2 = M;
            ex.w2 = (const void *)dw2; ex.w2_dtype = NVFP4A_BF16;
            ex.sq = (float *)dsq; ex.ld_sq = M / 16;
          }
          nvfp4a_status s2 = NVFP4A_OK;
          const BenchRow x = bench.run([&] { s2 = nvfp4a_gemm_ex(M, n, K, 1.f, &A, K, &X, K, 0.f, nullptr, 0, &Y, M, nullptr, &ex, 0); }, 200);
          char name[80], note[32];
          snprintf(name, sizeof name, "gemv_ex n=%d x=%s ex %d (%s)%s", n, xn[xt], mk.mask, mk.name, s2 == NVFP4A_OK ? "" : " FAILED");
          snprintf(note, sizeof note, "   x%.3f", x.ms / plain);
          row(name, x, (double)abytes, note);
        }
      }
    for (void *q : {dparts, dw2, dd2, dsq}) cudaFree(q);
  }

  // Transformer Engine / cuBLASLt formats (bench_util.h): per x dtype and n every format against its "plain"
  // row.  dsA / dsX hold the swizzled layout's padding (M a multiple of 128; x: 128 rows of K / 16 bytes).
  printf("\nTransformer Engine formats, -> BF16 (x = time / plain of the same x dtype and n)\n");
  {
    float *gA = te_globals(M), *gB = te_globals(32);
    for (nvfp4a_dtype xt : {NVFP4A_NVFP4, NVFP4A_BF16, NVFP4A_INT8})
      for (int n : {1, 8, 32}) {
        float plain = 0.f;
        for (const TeFmt &f : kTeFmts) {
          if (xt != NVFP4A_NVFP4 && f.fb) continue;  // B flags exist on NVFP4 x only
          if (&f != kTeFmts && !bench_want(f.name)) continue;  // "plain": the baseline of x, always
          nvfp4a_tensor a = {dA, NVFP4A_NVFP4, dsA, 0, gA, nullptr, f.fa, 0};
          nvfp4a_tensor X = {dX, xt, dsX, 0, xt == NVFP4A_BF16 ? nullptr : gB, nullptr, f.fb, 0};
          nvfp4a_status st = NVFP4A_OK;
          const BenchRow x = bench.run([&] { st = nvfp4a_gemm(M, n, K, 1.f, &a, K, &X, K, 0.f, nullptr, 0, &Y, M, nullptr, 0); }, 200);
          if (!plain) plain = x.ms;
          char name[96], note[32];
          snprintf(name, sizeof name, "TE n=%d x=%s %s%s", n, xn[xt], f.name, st == NVFP4A_OK ? "" : " FAILED");
          snprintf(note, sizeof note, "   x%.3f", x.ms / plain);
          row(name, x, (double)abytes, note);
        }
      }
  }

  float *fA; cudaMalloc(&fA, (size_t)M * K * 4); cudaMemset(fA, 0, (size_t)M * K * 4);
  cublasHandle_t hb; cublasCreate(&hb);
  const float one = 1, zero = 0;
  row("cuBLAS SGEMV FP32 (same M, K)", bench.run([&] { cublasSgemv(hb, CUBLAS_OP_T, K, M, &one, fA, K, (float *)dX, 1, &zero, (float *)dY, 1); }, 200),
      4.0 * M * K, "   (FP32 weights: 7.1x bytes)");
  row("cuBLAS SGEMM n=8 FP32 (same M, K)",
      bench.run([&] { cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, 8, K, &one, fA, K, (float *)dX, K, &zero, (float *)dY, M); }, 200),
      4.0 * M * K, "   (FP32 weights: 7.1x bytes)");
  return 0;
}
