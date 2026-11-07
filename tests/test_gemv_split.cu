// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The k-split GEMV items on CUDA cores (nvfp4a_gemv_split_items) and tensor cores (nvfp4a_gemv_tc_split_items): part items, then combine items,
// through an NVRTC executor.  k_parts = 1 equals nvfp4a_gemm_ex bit for bit; more parts equal it within the FP32 rounding of the regrouped sums
// (same bits for every launch shape); part items change exactly their declared partials, combine items read none outside their declared ranges.
#include <atomic>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <cuda.h>
#include <nvrtc.h>
#include "test_util.h"
#include "../include/nvfp4effma_items.h"
#include "../src/tensors/gemm_tc.cuh"

#ifndef NVFP4A_ROOT
#define NVFP4A_ROOT "."
#endif
#ifndef NVFP4A_CUDA_INCLUDE
#define NVFP4A_CUDA_INCLUDE "/usr/local/cuda/include"
#endif

namespace {

void check(bool ok, const char *what) {
  if (!ok) { fprintf(stderr, "%s\n", what); exit(1); }
}
void check_cuda(cudaError_t s, const char *what) {
  if (s != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(s)); exit(1); }
}
void check_driver(CUresult s, const char *what) {
  if (s != CUDA_SUCCESS) { fprintf(stderr, "%s: driver error %d\n", what, static_cast<int>(s)); exit(1); }
}

template <class T> std::vector<T> tohost(const T *d, size_t count) {
  std::vector<T> h(count);
  check_cuda(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return h;
}

// the executor kernel of one entry: CTAs stride over the items; one compile per distinct source (k_parts is an argument),
// shared by the threads that ask for it
CUfunction compile_executor(const nvfp4a_items &it) {
  const std::string src = "#include \"nvfp4effma_items.cuh\"\n"
                          "extern \"C\" __global__ void __launch_bounds__(" + std::to_string(it.threads) + ", 1)\n"
                          "run(const void *args, int items) {\n"
                          "  extern __shared__ __align__(16) unsigned char smem[];\n"
                          "  for (int i = blockIdx.x; i < items; i += gridDim.x) {\n"
                          "    " + std::string(it.entry) + "(*(const " + it.args_type + " *)args, i, threadIdx.x, smem);\n"
                          "    __syncthreads();\n  }\n}\n";
  const std::string arch = "--gpu-architecture=compute_" + std::to_string(it.arch);
  const char *opts[] = {arch.c_str(), "--std=c++17", "--device-as-default-execution-space", "--include-path=" NVFP4A_ROOT "/include",
                        "--include-path=" NVFP4A_ROOT "/src", "--include-path=" NVFP4A_CUDA_INCLUDE, "--include-path=" NVFP4A_CUDA_INCLUDE "/cccl"};
  nvrtcProgram p;
  nvrtcCreateProgram(&p, src.c_str(), "split.cu", 0, nullptr, nullptr);
  const nvrtcResult r = nvrtcCompileProgram(p, 7, opts);
  size_t n;
  nvrtcGetProgramLogSize(p, &n);
  std::string log(n, '\0');
  nvrtcGetProgramLog(p, &log[0]);
  if (r != NVRTC_SUCCESS) { fprintf(stderr, "NVRTC: %s\n%s\n", nvrtcGetErrorString(r), log.c_str()); exit(1); }
  nvrtcGetPTXSize(p, &n);
  std::string ptx(n, '\0');
  nvrtcGetPTX(p, &ptx[0]);
  nvrtcDestroyProgram(&p);
  CUmodule mod;
  check_driver(cuModuleLoadData(&mod, ptx.data()), "cuModuleLoadData");
  CUfunction fn;
  check_driver(cuModuleGetFunction(&fn, mod, "run"), "cuModuleGetFunction");
  return fn;
}

CUfunction executor(const nvfp4a_items &it) {
  static std::mutex lock;
  static std::map<std::string, std::shared_future<CUfunction>> compiled;
  const std::string key = std::string(it.entry) + '|' + it.args_type + '|' + std::to_string(it.threads) + '|' + std::to_string(it.arch);
  std::promise<CUfunction> promise;
  std::shared_future<CUfunction> future;
  bool mine = false;
  {
    std::lock_guard<std::mutex> guard(lock);
    const auto at = compiled.find(key);
    if (at != compiled.end()) future = at->second;
    else {
      future = promise.get_future().share();
      compiled.emplace(key, future);
      mine = true;
    }
  }
  if (mine) promise.set_value(compile_executor(it));
  return future.get();
}

void launch(const nvfp4a_items &it, CUfunction fn, unsigned grid) {
  void *dargs;
  check_cuda(cudaMalloc(&dargs, it.args_bytes), "cudaMalloc arguments");
  check_cuda(cudaMemcpy(dargs, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice), "cudaMemcpy arguments");
  int items = it.items;
  void *params[] = {&dargs, &items};
  check_driver(cuLaunchKernel(fn, grid < static_cast<unsigned>(items) ? grid : items, 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr), "launch");
  check_cuda(cudaDeviceSynchronize(), "synchronize");
  cudaFree(dargs);
}

std::vector<nvfp4a_item_range> ranges_of(const nvfp4a_items &it, int item) {
  const int n = nvfp4a_item_access(&it, item, nullptr, 0);
  check(n >= 0, "no ranges");
  std::vector<nvfp4a_item_range> r(static_cast<size_t>(n));
  check(nvfp4a_item_access(&it, item, r.data(), n) == n, "range count");
  return r;
}

// marks the bytes of the partials that the items' ranges of that argument cover
void mark(const nvfp4a_items &it, bool writes, std::vector<int> &hits, long long bytes) {
  for (int i = 0; i < it.items; ++i)
    for (const nvfp4a_item_range &r : ranges_of(it, i))
      if (r.arg == NVFP4A_ARG_PARTIALS && (r.write != 0) == writes)
        for (int row = 0; row < r.rows; ++row) {
          check(r.offset + row * r.stride + r.bytes <= bytes, "partials range beyond the workspace");
          for (long long b = 0; b < r.bytes; ++b) ++hits[static_cast<size_t>(r.offset + row * r.stride + b)];
        }
}

struct SplitCase {
  int m, n, k;
  nvfp4a_dtype bt;
  int mask;  // 1 column scale, 2 SwiGLU, 4 second output
  bool tc;   // tensor cores: A as its image
};

int run(const SplitCase &c, std::string *line) {
  const bool q8 = c.bt == NVFP4A_INT8, glu = c.mask & 2, out2 = c.mask & 4;
  const int lda = c.k, ldb = c.k, mo = glu ? c.m / 2 : c.m, ldd = mo + 3, np = 37, ldp = 40, lsq = c.m / 16 + 1, ldd2 = c.m + 5;
  HostFp4 hA(c.m, c.k, lda, c.k / 16, 0.375f);
  HostQ8 hB8(q8 ? c.n : 0, c.k, ldb, c.k / 16, 1.75f);
  std::vector<uint8_t> hB(q8 ? 0 : static_cast<size_t>(ldb) * c.n * 2), hbias(static_cast<size_t>(c.m) * 2), hw2(static_cast<size_t>(c.m) * 2),
      hC(static_cast<size_t>(ldd) * c.n * 4);
  if (!q8) fill(NVFP4A_BF16, hB, 2.f);
  fill(NVFP4A_BF16, hbias, 4.f);
  fill(NVFP4A_BF16, hw2, 2.f);
  fill(NVFP4A_FP32, hC, 4.f);
  std::vector<float> parts(static_cast<size_t>(ldp) * c.n);
  for (auto &x : parts) x = static_cast<float>(rnd() % 4096) / 64.f;
  uint8_t *dA = dev(hA.data), *dSA = dev(hA.sc), *dbias = dev(hbias), *dw2 = dev(hw2), *dC = dev(hC);
  void *dB = q8 ? static_cast<void *>(dev(hB8.data)) : static_cast<void *>(dev(hB));
  float *dSB = q8 ? dev(hB8.sc) : nullptr, *dparts = dev(parts), *dgb = devf(1.75f);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, 0, nullptr, nullptr, 0, 0};
  if (c.tc) {
    const int kind = q8 ? NVFP4A_TC_A8 : NVFP4A_TC_A16;
    void *ia;
    check_cuda(cudaMalloc(&ia, nvfp4a_tc_image_bytes(kind, NVFP4A_NVFP4, c.m, c.k) + 16), "cudaMalloc A image");
    check(nvfp4a_tc_image(kind, &A, lda, c.m, c.k, ia, 0) == NVFP4A_OK, nvfp4a_last_error());
    A.data = ia;
  }
  const nvfp4a_tensor B = {dB, c.bt, dSB, q8 ? c.k / 16 : 0, q8 ? dgb : nullptr, nullptr, 0, 0};
  const nvfp4a_tensor bias = {dbias, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
  const nvfp4a_tensor C = {dC, NVFP4A_FP32, nullptr, 0, nullptr, nullptr, 0, 0};

  const float beta = glu ? 0.f : 1.f;  // SwiGLU takes no C
  const nvfp4a_tensor *cp = glu ? nullptr : &C;
  struct Out { uint8_t *d, *d2; float *sq; };
  const auto fresh = [&] {
    Out x;
    x.d = dev(std::vector<uint8_t>(static_cast<size_t>(ldd) * c.n * 4, 0xA5));
    x.d2 = dev(std::vector<uint8_t>(static_cast<size_t>(ldd2) * c.n * 2, 0xA5));
    x.sq = dev(std::vector<float>(static_cast<size_t>(lsq) * c.n, -1.f));
    return x;
  };
  const auto epilogue = [&](const Out &x) {
    nvfp4a_epilogue e{};
    if (c.mask & 1) { e.norm_parts = dparts; e.norm_nparts = np; e.norm_ld = ldp; e.norm_inv_n = 1.f / 1024.f; e.norm_eps = 1e-6f; }
    e.swiglu = glu;
    if (out2) { e.d2 = x.d2; e.d2_dtype = NVFP4A_BF16; e.ldd2 = ldd2; e.w2 = dw2; e.w2_dtype = NVFP4A_BF16; e.sq = x.sq; e.ld_sq = lsq; }
    return e;
  };
  const auto tensor_d = [&](const Out &x) { return nvfp4a_tensor{x.d, NVFP4A_FP32, nullptr, 0, nullptr, nullptr, 0, 0}; };
  const size_t dn = static_cast<size_t>(ldd) * c.n * 4, d2n = static_cast<size_t>(ldd2) * c.n * 2, sqn = static_cast<size_t>(lsq) * c.n;

  const Out ref = fresh();
  {
    const nvfp4a_epilogue e = epilogue(ref);
    const nvfp4a_tensor D = tensor_d(ref);
    const nvfp4a_status st = c.tc ? nvfp4a_gemv_tc_ex(c.m, c.n, c.k, 0.75f, &A, &B, ldb, beta, cp, ldd, &D, ldd, &bias, c.mask ? &e : nullptr, 0)
                                  : nvfp4a_gemm_ex(c.m, c.n, c.k, 0.75f, &A, lda, &B, ldb, beta, cp, ldd, &D, ldd, &bias, c.mask ? &e : nullptr, 0);
    check(st == NVFP4A_OK, nvfp4a_last_error());
    check_cuda(cudaDeviceSynchronize(), "reference");
  }
  const auto outputs = [&](const Out &x) {
    auto v = tohost(x.d, dn);
    const auto v2 = tohost(x.d2, d2n);
    v.insert(v.end(), v2.begin(), v2.end());
    const auto s = tohost(reinterpret_cast<const uint8_t *>(x.sq), sqn * 4);
    v.insert(v.end(), s.begin(), s.end());
    return v;
  };
  const auto want = outputs(ref);
  const auto as_floats = [&](const std::vector<uint8_t> &v) { return std::vector<float>(reinterpret_cast<const float *>(v.data()), reinterpret_cast<const float *>(v.data()) + dn / 4); };

  // alpha A B^T + bias + beta C in double, and the sum of the magnitudes of its products
  std::vector<double> exact(static_cast<size_t>(ldd) * c.n, 0.0), abs_sum(exact.size(), 0.0);
  if (!c.mask)
    for (int j = 0; j < c.n; ++j)
      for (int r = 0; r < c.m; ++r) {
        double sum = 0, mag = 0;
        for (int t = 0; t < c.k; ++t) {
          const double bv = q8 ? static_cast<double>(hB8.q(j, t)) * hB8.scale(j, t) * 1.75
                               : static_cast<double>(bf(reinterpret_cast<const uint16_t *>(hB.data())[static_cast<size_t>(j) * ldb + t]));
          const double term = static_cast<double>(hA.val(r, t)) * bv;
          sum += term;
          mag += fabs(term);
        }
        const size_t i = static_cast<size_t>(j) * ldd + r;
        exact[i] = 0.75 * sum + bf(reinterpret_cast<const uint16_t *>(hbias.data())[r]) + reinterpret_cast<const float *>(hC.data())[i];
        abs_sum[i] = 0.75 * mag;
      }

  int bad = 0;
  const int max_parts = c.tc ? nvfp4a_gemv_tc_split_max_parts(c.k) : nvfp4a_gemv_split_max_parts(c.k);
  check(max_parts >= 1, "max parts");
  std::vector<std::vector<uint8_t>> by_parts;
  for (int p : {1, 2, 3, max_parts}) {
    if (p > max_parts) continue;
    const Out got = fresh();
    nvfp4a_epilogue e = epilogue(got);
    const size_t ws = c.tc ? nvfp4a_gemv_tc_split_workspace_bytes(c.m, c.n, p) : nvfp4a_gemv_split_workspace_bytes(c.m, c.n, p, c.bt, c.mask ? &e : nullptr);
    check(ws > 0 && ws % 16 == 0, "workspace bytes");
    float *partials;
    check_cuda(cudaMalloc(&partials, ws), "cudaMalloc partials");
    const nvfp4a_tensor D = tensor_d(got);
    nvfp4a_items part, combine;
    const nvfp4a_status ss = c.tc ? nvfp4a_gemv_tc_split_items(c.m, c.n, c.k, 0.75f, &A, &B, ldb, beta, cp, ldd, &D, ldd, &bias,
                                                               c.mask ? &e : nullptr, p, partials, &part, &combine)
                                  : nvfp4a_gemv_split_items(c.m, c.n, c.k, 0.75f, &A, lda, &B, ldb, beta, cp, ldd, &D, ldd, &bias,
                                                            c.mask ? &e : nullptr, p, partials, &part, &combine);
    check(ss == NVFP4A_OK, nvfp4a_last_error());
    check(part.kind == (c.tc ? NVFP4A_ITEM_GEMV_TC_PART : NVFP4A_ITEM_GEMV_PART) &&
              combine.kind == (c.tc ? NVFP4A_ITEM_GEMV_TC_COMBINE : NVFP4A_ITEM_GEMV_COMBINE) && part.items == p * combine.items,
          "item counts");
    const CUfunction fpart = executor(part), fcomb = executor(combine);
    if (part.smem_bytes > 48 * 1024) check_driver(cuFuncSetAttribute(fpart, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, part.smem_bytes), "smem");

    // the part items change exactly the bytes they declare, each once
    check_cuda(cudaMemset(partials, 0xA5, ws), "poison partials");
    launch(part, fpart, 5);
    const auto after = tohost(reinterpret_cast<const uint8_t *>(partials), ws);
    std::vector<int> wrote(ws, 0), read(ws, 0);
    mark(part, true, wrote, static_cast<long long>(ws));
    mark(combine, false, read, static_cast<long long>(ws));
    for (size_t b = 0; b < ws; ++b) {
      if (wrote[b] > 1) { ++bad; if (bad < 4) fprintf(stderr, "partials byte %zu written by %d items\n", b, wrote[b]); }
      if (wrote[b] == 0 && after[b] != 0xA5) { ++bad; if (bad < 4) fprintf(stderr, "partials byte %zu changed undeclared\n", b); }
      if (read[b] && !wrote[b]) { ++bad; if (bad < 4) fprintf(stderr, "partials byte %zu read, never written\n", b); }
    }
    // the combine items read only their declared bytes: poison the rest, same result
    {
      auto keep = after;
      for (size_t b = 0; b < ws; ++b)
        if (!read[b]) keep[b] = 0xFF;
      check_cuda(cudaMemcpy(partials, keep.data(), ws, cudaMemcpyHostToDevice), "poison unread partials");
    }
    launch(combine, fcomb, 3);
    const auto first = outputs(got);
    by_parts.push_back(first);

    // another launch shape: the same bits
    check_cuda(cudaMemcpy(partials, after.data(), ws, cudaMemcpyHostToDevice), "restore partials");
    launch(part, fpart, 1);
    check_cuda(cudaMemset(got.d, 0xA5, dn), "clear D");
    launch(combine, fcomb, 1);
    if (outputs(got) != first) { ++bad; fprintf(stderr, "parts %d: launch shapes differ\n", p); }

    if (p == 1) {
      if (first != want) { ++bad; fprintf(stderr, "parts 1: not bit-exact with nvfp4a_gemm_ex\n"); }
    } else if (!c.mask) {  // within the FP32 error of the exact sum, as the whole-k item is
      const auto a = as_floats(first), b = as_floats(want);
      for (int j = 0; j < c.n; ++j)
        for (int r = 0; r < c.m; ++r) {
          const size_t i = static_cast<size_t>(j) * ldd + r;
          const double bound = 256 * 0x1p-24 * abs_sum[i] + 0x1p-20 * fabs(exact[i]);
          if (fabs(b[i] - exact[i]) > bound) { ++bad; fprintf(stderr, "model: ref D[%zu] = %g, exact %g, bound %g\n", i, b[i], exact[i], bound); }
          if (fabs(a[i] - exact[i]) > bound) { ++bad; fprintf(stderr, "parts %d: D[%zu] = %g, exact %g, bound %g\n", p, i, a[i], exact[i], bound); }
          if (bad > 4) break;
        }
    }
    cudaFree(partials);
  }
  char text[160];
  snprintf(text, sizeof text, "[%s] split %s m=%d n=%d k=%d %s mask=%d parts<=%d\n", bad ? "FAIL" : "PASS", c.tc ? "tc  " : "core", c.m, c.n, c.k,
           q8 ? "int8" : "bf16", c.mask, max_parts);
  *line = text;
  return bad;
}

}  // namespace

int main() {
  check_cuda(cudaFree(nullptr), "cuda init");
  const auto &device = nvfp4a::kern::device_info();
  std::vector<SplitCase> cases;
  for (bool tc : {false, true})
    for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_INT8}) {
      if (tc && (!device.tc || (bt == NVFP4A_BF16 && device.ptx < 80))) continue;
      for (int n : {1, 3, 8})
        for (int mask : {0, 1, 2, 4})
          for (int k : {1536, 2560}) cases.push_back({mask & 6 ? 208 : 200, n, k, bt, mask, tc});
    }
  // the cases on one thread each (NVRTC is the time: they compile together), results printed in order
  std::vector<std::string> lines(cases.size());
  std::vector<int> bads(cases.size(), 0);
  std::atomic<size_t> next{0};
  const auto worker = [&] {
    check_cuda(cudaFree(nullptr), "cuda init (thread)");
    for (size_t i; (i = next++) < cases.size();) {
      rng = 0x2545F491u + 0x9E3779B9u * static_cast<uint32_t>(i);  // the data of a case does not depend on the schedule
      bads[i] = run(cases[i], &lines[i]);
    }
  };
  std::vector<std::thread> pool;
  const size_t threads = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), cases.size());
  for (size_t t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (std::thread &t : pool) t.join();
  int bad = 0;
  for (size_t i = 0; i < cases.size(); ++i) {
    fputs(lines[i].c_str(), stdout);
    bad += bads[i];
  }
  printf("%s\n", bad ? "FAILED" : "all passed");
  return bad ? 1 : 0;
}
