// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Shared measurement helper of the benchmarks.
//
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
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>

struct BenchRow { float ms; double mhz; };  // median time, mean SM clock (0 = unknown)

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
  }
  ~Bench() { if (ok_) nvmlShutdown(); }

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
