// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Shared measurement helper of the benchmarks.
//
// Before the first row: NVFP4A_BENCH_WARMUP (default 4) plain FP32 matrix products 2048 x 2048, 0.5 s apart,
// so that GPU Boost has raised the clocks (otherwise the first rows run at a lower SM clock).
// Every row: one untimed call (module loading), NVFP4A_BENCH_COOL seconds of idle GPU (default 2) so
// each row starts from a similar temperature, then NVFP4A_BENCH_REPS timed calls (default per bench).
// The reported time is the median.  While a row runs, a host thread samples the SM clock through NVML
// every 5 ms; percentages of the FP32 peak use that row's own mean clock, so rows that heat the GPU
// more (and run at a lower clock) are not compared against a clock they never had.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>

struct BenchRow { float ms; double mhz; };  // median time, mean SM clock (0 = unknown)

// the warm-up's plain matrix product: C = A B, n x n FP32, 16 x 16 shared-memory tiles
__global__ void bench_warmup_mm(const float *a, const float *b, float *c, int n) {
  __shared__ float ta[16][16], tb[16][17];
  const int tx = threadIdx.x, ty = threadIdx.y, row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < n; k0 += 16) {
    ta[ty][tx] = a[(size_t)row * n + k0 + tx];
    tb[ty][tx] = b[(size_t)(k0 + ty) * n + col];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) acc += ta[ty][k] * tb[k][tx];
    __syncthreads();
  }
  c[(size_t)row * n + col] = acc;
}

// NVFP4A_BENCH_FILTER: run only the rows whose name contains one of these '|'-separated substrings (all rows
// when unset), e.g. "plain|16x16" -- to measure one fix without the whole suite
static bool bench_want(const char *name) {
  const char *f = getenv("NVFP4A_BENCH_FILTER");
  if (!f || !*f) return true;
  for (const char *p = f; *p;) {
    const char *e = strchr(p, '|');
    const size_t len = e ? (size_t)(e - p) : strlen(p);
    for (const char *q = name; *q; ++q)
      if (len && !strncmp(q, p, len)) return true;
    if (!e) break;
    p = e + 1;
  }
  return false;
}

class Bench {
 public:
  Bench() {
    char bus[32];
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) == cudaSuccess && nvmlInit_v2() == NVML_SUCCESS &&
        nvmlDeviceGetHandleByPciBusId_v2(bus, &nvml_) == NVML_SUCCESS)
      ok_ = true;
    cudaEventCreate(&a_); cudaEventCreate(&b_);
    warmup();
  }
  ~Bench() { if (ok_) nvmlShutdown(); }

  // NVFP4A_BENCH_WARMUP products of 2048 x 2048, 0.5 s apart (the GPU raises its clocks under load)
  static void warmup() {
    const int count = env("NVFP4A_BENCH_WARMUP", 4), n = 2048;
    if (count <= 0) return;
    float *m; cudaMalloc(&m, (size_t)3 * n * n * 4); cudaMemset(m, 0, (size_t)3 * n * n * 4);
    for (int i = 0; i < count; ++i) {
      if (i) std::this_thread::sleep_for(std::chrono::milliseconds(500));
      bench_warmup_mm<<<dim3(n / 16, n / 16), dim3(16, 16)>>>(m, m + (size_t)n * n, m + (size_t)2 * n * n, n);
      cudaDeviceSynchronize();
    }
    cudaFree(m);
  }

  static int env(const char *name, int dflt) { const char *e = getenv(name); return e && atoi(e) >= 0 && *e ? atoi(e) : dflt; }

  template <class F> BenchRow run(F f, int reps_default) {
    const int reps = std::max(1, env("NVFP4A_BENCH_REPS", reps_default));
    f(); cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::seconds(env("NVFP4A_BENCH_COOL", 2)));
    std::atomic<bool> running{true};
    double sum = 0; long n = 0;
    std::thread sampler([&] {
      while (running) {
        unsigned int mhz;
        if (ok_ && nvmlDeviceGetClockInfo(nvml_, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) { sum += mhz; ++n; }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
    std::vector<float> t(reps);
    for (int r = 0; r < reps; ++r) {
      cudaEventRecord(a_); f(); cudaEventRecord(b_); cudaEventSynchronize(b_);
      cudaEventElapsedTime(&t[r], a_, b_);
    }
    running = false;
    sampler.join();
    unsigned int mhz;  // a row shorter than one sampling period: take the clock right after it
    if (!n && ok_ && nvmlDeviceGetClockInfo(nvml_, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) { sum = mhz; n = 1; }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return {t[reps / 2], n ? sum / n : 0.0};
  }

 private:
  nvmlDevice_t nvml_{};
  bool ok_ = false;
  cudaEvent_t a_, b_;
};

// Transformer Engine / cuBLASLt operand formats, each a kernel instance of its own: one feature at a time
// (where a regression comes from), then the combinations Transformer Engine produces.  B flags apply to an
// NVFP4 B only; rows with B flags are skipped for other B dtypes.  Every row is compared with "plain" of
// the same block (linear E4M3 scales, per-tensor decode factors), measured in the same run.
#ifdef NVFP4EFFMA_H
struct TeFmt { const char *name; unsigned fa, fb; };
static const unsigned kSW = NVFP4A_SCALES_SWIZZLED, kUE = NVFP4A_SCALES_UE5M3, kAM = NVFP4A_GLOBAL_AMAX,
                      kPR = NVFP4A_GLOBAL_PER_ROW;
static const TeFmt kTeFmts[] = {
    {"plain", 0, 0},
    {"A swizzled", kSW, 0},
    {"B swizzled", 0, kSW},
    {"A UE5M3", kUE, 0},
    {"B UE5M3", 0, kUE},
    {"A amax", kAM, 0},
    {"A+B amax", kAM, kAM},
    {"A per-row", kPR, 0},
    {"B per-row", 0, kPR},
    {"TE W: swz, amax", kSW | kAM, 0},
    {"TE W x X: swz, amax", kSW | kAM, kSW | kAM},
    {"TE row-scaled X: swz, amax, B per-row", kSW | kAM, kSW | kAM | kPR},
    {"TE UE5M3: swz, UE5M3, amax", kSW | kUE | kAM, kSW | kUE | kAM},
    {"ModelOpt W x TE X", 0, kSW | kAM},
};
// buffers of a TE row: per-row globals (1.0) for up to rows values
static float *te_globals(int rows) {
  std::vector<float> h(rows, 1.0f);
  float *d = nullptr;
  cudaMalloc(&d, h.size() * 4); cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  return d;
}
#endif

