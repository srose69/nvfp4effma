// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// `make tune`: benchmark every GEMM / GEMV candidate (src/gemm_nvfp4effma.cuh, src/gemv_nvfp4effma.cuh)
// on this GPU and record the winners for its compute capability in src/nvfp4effma_tuned.h (entries of
// other compute capabilities already in the file are kept).  Usage: tune <path of nvfp4effma_tuned.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include "../src/gemm_nvfp4effma.cuh"
#include "../src/gemv_nvfp4effma.cuh"

using namespace nvfp4a::kern;

template <class F> static float best_ms(F f, int reps = 7) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize();
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  return best;
}

static Epilogue epi(void *d, int ldd) {
  Epilogue e = {};
  e.alpha = 1.f; e.d = d; e.ldd = ldd; e.d_dtype = NVFP4A_BF16; e.c_dtype = NVFP4A_FP32; e.bias_dtype = NVFP4A_FP32;
  return e;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "src/nvfp4effma_tuned.h";
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  const int cc = p.major * 10 + p.minor;
  printf("tuning on %s (sm_%d, %d SMs)\n", p.name, cc, p.multiProcessorCount);

  const size_t big = (size_t)8192 * 8192;
  uint8_t *dA, *dSA, *dB, *dSB; void *dD;
  cudaMalloc(&dA, big / 2); cudaMalloc(&dSA, big / 16); cudaMalloc(&dB, big * 4); cudaMalloc(&dSB, big / 16);
  cudaMalloc(&dD, big * 4);
  {  // NVFP4 codes, scales around 1; finite BF16 / FP16 / FP32 bit patterns (bytes & 0x37)
    std::vector<uint8_t> h(big * 4);
    unsigned r = 1;
    for (auto &x : h) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; x = (r & 0x37); }
    cudaMemcpy(dA, h.data(), big / 2, cudaMemcpyHostToDevice); cudaMemcpy(dB, h.data(), h.size(), cudaMemcpyHostToDevice);
    cudaMemset(dSA, 0x38, big / 16); cudaMemset(dSB, 0x38, big / 16);
  }

  // ---- dense-B GEMM (NVFP4 x BF16, BF16 out; FP16 / FP32 B run the same kernel) ----
  struct Shape { int m, n, k; };
  const Shape large = {4096, 4096, 4096};
  const Shape smalls[] = {{512, 512, 4096}, {256, 2048, 4096}};
  // A configuration replaces the default only if it is at least 2% faster: noise-level wins would
  // add kernels to the library for nothing.
  const float kMargin = 0.98f;
  int best_large = kDefaultTable.gemm_large, best_small = kDefaultTable.gemm_small;
  float tl = 1e30f, ts = 1e30f, tl_def = 0, ts_def = 0;
  printf("\nGEMM candidates (NVFP4 x BF16 -> BF16), ms\n%-24s %10s %14s\n", "config", "4096^3", "small shapes");
#define NVFP4A_TRY(cid, bm, bn, tm, tn)                                                                        \
  {                                                                                                         \
    auto run = [&](const Shape &s) {                                                                        \
      Epilogue e = epi(dD, s.m);                                                                            \
      return best_ms([&] { gemm_launch<GemmCfg##cid, NVFP4A_BF16>(dA, s.k, dSA, s.k / 16, dB, s.k, nullptr, 0, s.m, s.n, s.k, e, 0); }); \
    };                                                                                                      \
    const float a = run(large);                                                                             \
    float b = 0; for (const Shape &s : smalls) b += run(s);                                                 \
    printf("%d: %3dx%3d t%dx%d          %10.3f %14.3f\n", cid, bm, bn, tm, tn, a, b);                         \
    if (a < tl) { tl = a; best_large = cid; }                                                               \
    if (b < ts) { ts = b; best_small = cid; }                                                               \
    if (cid == kDefaultTable.gemm_large) tl_def = a;                                                        \
    if (cid == kDefaultTable.gemm_small) ts_def = b;                                                        \
  }
  NVFP4A_GEMM_CANDIDATES(NVFP4A_TRY)
#undef NVFP4A_TRY
  if (tl > kMargin * tl_def) best_large = kDefaultTable.gemm_large;
  if (ts > kMargin * ts_def) best_small = kDefaultTable.gemm_small;
  printf("-> large: %d, small: %d\n", best_large, best_small);

  // ---- GEMV (NVFP4 A 8192 x 8192, n = 1..8, x NVFP4 / BF16 / FP16 / FP32) ----
  unsigned char gv[kOperandTypes][8];
  const char *xn[] = {"NVFP4", "BF16", "FP16", "FP32"};
  printf("\nGEMV candidates, GB/s over A (8192 x 8192, values + block scales)\n");
  for (int xi = 0; xi < kOperandTypes; ++xi)
    for (int n = 1; n <= 8; ++n) {
      float bt = 1e30f, t_def = 0; int bid = 0;
      printf("x=%s n=%d:", xn[xi], n);
      auto measure = [&](auto launch) { return best_ms([&] { launch(); }, 15); };
#define NVFP4A_TRYV(cid, r, mx)                                                                                   \
      if (n <= mx) {                                                                                             \
        Epilogue e = epi(dD, 8192);                                                                              \
        float t = 1e30f;                                                                                         \
        auto go = [&](auto nbc) {                                                                                \
          constexpr int NB = decltype(nbc)::value;                                                               \
          auto go_t = [&](auto xc) {                                                                             \
            t = measure([&] { gemv_launch<GemvCfg##cid, NB, decltype(xc)::value>(dA, 8192, dSA, 512, dB, 8192, dSB, 512, 8192, 8192, e, 0); }); \
          };                                                                                                     \
          if (xi == NVFP4A_NVFP4) go_t(std::integral_constant<int, NVFP4A_NVFP4>());                             \
          else if (xi == NVFP4A_BF16) go_t(std::integral_constant<int, NVFP4A_BF16>());                          \
          else if (xi == NVFP4A_FP16) go_t(std::integral_constant<int, NVFP4A_FP16>());                          \
          else go_t(std::integral_constant<int, NVFP4A_FP32>());                                                 \
        };                                                                                                       \
        switch (n) {                                                                                             \
          case 1: go(std::integral_constant<int, 1>()); break; case 2: go(std::integral_constant<int, 2>()); break; \
          case 3: go(std::integral_constant<int, 3>()); break; case 4: go(std::integral_constant<int, 4>()); break; \
          case 5: go(std::integral_constant<int, 5>()); break; case 6: go(std::integral_constant<int, 6>()); break; \
          case 7: go(std::integral_constant<int, 7>()); break; default: go(std::integral_constant<int, 8>()); break; \
        }                                                                                                        \
        printf("  [%d] %6.1f", cid, 8192.0 * 8192 * 9 / 16 / t / 1e6);                                                    \
        if (t < bt) { bt = t; bid = cid; }                                                                       \
        if (cid == 0) t_def = t;                                                                                 \
      }
      NVFP4A_GEMV_CANDIDATES(NVFP4A_TRYV)
#undef NVFP4A_TRYV
      if (bt > kMargin * t_def) bid = 0;
      printf("  -> %d\n", bid);
      gv[xi][n - 1] = (unsigned char)bid;
    }
  if (cudaGetLastError() != cudaSuccess) { printf("CUDA error during tuning\n"); return 1; }

  // ---- write the header: keep entries of other compute capabilities ----
  std::vector<std::string> keep;
  std::set<int> gemm_used, gemv_used;
  {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      int ecc, g0, g1;
      const size_t pos = line.find("/* entry cc=");
      if (pos == std::string::npos || sscanf(line.c_str() + pos, "/* entry cc=%d gemm=%d,%d", &ecc, &g0, &g1) != 3 || ecc == cc) continue;
      keep.push_back(line);
      gemm_used.insert(g0); gemm_used.insert(g1);
      const size_t gp = line.find("gemv=");
      for (const char *q = line.c_str() + gp + 5; *q && *q != ' '; ++q) if (*q >= '0' && *q <= '9') gemv_used.insert(*q - '0');
    }
  }
  char buf[1024];
  std::string gvs, ids;
  std::set<int> mine;
  for (int xi = 0; xi < kOperandTypes; ++xi) {
    gvs += xi ? ", {" : "{";
    for (int n = 0; n < 8; ++n) { gvs += std::to_string(gv[xi][n]) + (n < 7 ? ", " : ""); mine.insert(gv[xi][n]); }
    gvs += "}";
  }
  for (int id : mine) { ids += std::to_string(id); gemv_used.insert(id); }
  gemm_used.insert(best_large); gemm_used.insert(best_small);
  snprintf(buf, sizeof buf, "  , nvfp4a::kern::ArchTable{%d, %d, %d, {%s}} /* entry cc=%d gemm=%d,%d gemv=%s %s */ \\",
           cc, best_large, best_small, gvs.c_str(), cc, best_large, best_small, ids.c_str(), p.name);
  keep.push_back(buf);
  std::ofstream out(path);
  out << "// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception\n"
         "// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).\n"
         "// Generated by `make tune` -- per-architecture winners of the candidates in src/gemm_nvfp4effma.cuh\n"
         "// and src/gemv_nvfp4effma.cuh.\n"
         "// One entry per compute capability; re-running the tuner on another GPU adds or replaces its entry.\n"
         "#pragma once\n#define NVFP4A_TUNED_TABLES \\\n";
  for (auto &l : keep) out << l << "\n";
  out << "\n";
  // configurations to instantiate beyond the defaults (GEMM 0, 1; GEMV 0)
  out << "#define NVFP4A_GEMM_USED(X)";
  for (int id : gemm_used) if (id != 0 && id != 1) out << " X(" << id << ")";
  out << "\n#define NVFP4A_GEMV_USED(X)";
  for (int id : gemv_used) if (id != 0) out << " X(" << id << ")";
  out << "\n";
  printf("\nwrote %s (sm_%d: gemm large %d, small %d)\n", path, cc, best_large, best_small);
  return 0;
}
