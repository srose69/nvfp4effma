// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The work items (include/nvfp4effma_items.h) as an executor runs them: an NVRTC kernel from the described entry and argument type, persistent
// over the items (fewer CTAs than items, shared memory reused), outputs against the call's bit for bit (tiles: the library kernel without stream-K);
// every byte the items change lies in their declared write ranges, every declared read range in its argument.
#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <functional>
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

namespace {

#ifndef NVFP4A_ROOT
#define NVFP4A_ROOT "."
#endif
#ifndef NVFP4A_CUDA_INCLUDE
#define NVFP4A_CUDA_INCLUDE "/usr/local/cuda/include"
#endif

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status)); exit(1); }
}
void check_driver(CUresult status, const char *what) {
  if (status != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(status, &message);
    fprintf(stderr, "%s: %s\n", what, message ? message : "unknown driver error");
    exit(1);
  }
}
// a case's report goes to its own buffer (the cases run on several threads), printed in order
thread_local std::string *sink = nullptr;
void say(const char *fmt, ...) {
  char text[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(text, sizeof text, fmt, args);
  va_end(args);
  if (sink) *sink += text;
  else fputs(text, stdout);
}

void check_nvrtc(nvrtcResult status, const char *what) {
  if (status != NVRTC_SUCCESS) { fprintf(stderr, "%s: %s\n", what, nvrtcGetErrorString(status)); exit(1); }
}

// The existing library kernel, one CTA per whole tile: no stream-K regrouping of the sums.
cudaError_t tile_reference(const nvfp4a_items &it, nvfp4a_dtype bt, int mask) {
  using namespace nvfp4a::kern;
  static const void *const q8[] = {tc_kernel_ptr<NVFP4A_INT8, 0>(), tc_kernel_ex_ptr<NVFP4A_INT8, 1>(),
      tc_kernel_ex_ptr<NVFP4A_INT8, 2>(), tc_kernel_ex_ptr<NVFP4A_INT8, 3>(), tc_kernel_ex_ptr<NVFP4A_INT8, 4>(),
      tc_kernel_ex_ptr<NVFP4A_INT8, 5>()};
  static const void *const bf16[] = {tc_kernel_ptr<NVFP4A_BF16, 0>(), tc_kernel_ex_ptr<NVFP4A_BF16, 1>(),
      tc_kernel_ex_ptr<NVFP4A_BF16, 2>(), tc_kernel_ex_ptr<NVFP4A_BF16, 3>(), tc_kernel_ex_ptr<NVFP4A_BF16, 4>(),
      tc_kernel_ex_ptr<NVFP4A_BF16, 5>()};
  const void *kernel = (bt == NVFP4A_INT8 ? q8 : bf16)[mask];
  GemmTcItemArgs a;
  if (it.args_bytes != sizeof a) { fprintf(stderr, "unexpected TC tile argument size\n"); exit(1); }
  memcpy(&a, it.args.bytes, sizeof a);
  float *workspace = nullptr;
  unsigned *flags = nullptr;
  void *params[] = {&a.ia, &a.ib, &a.m, &a.n, &a.kt, &a.e, &workspace, &flags, &a.xd};
  cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, it.smem_bytes);
  return status ? status : cudaLaunchKernel(kernel, dim3(it.items), dim3(it.threads), params, it.smem_bytes, 0);
}

// The executor kernel of one entry, compiled for the architecture of the described library instance.
struct PublishCopy { uint8_t *dst; const uint8_t *src; unsigned long long bytes; };
static_assert(sizeof(PublishCopy) == 24, "NVRTC publication descriptor ABI");

CUmodule build_executor(const nvfp4a_items &it, CUfunction *fn, bool publish, bool readonly_control) {
  std::string entry = it.entry;
  if (readonly_control) entry.insert(entry.size() - 1, ", false");
  const std::string publication = R"cuda(
struct PublishCopy { unsigned char *dst; const unsigned char *src; unsigned long long bytes; };
__device__ void signal_epoch(unsigned *p, unsigned epoch) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(epoch) : "memory");
#else
  __threadfence();
  atomicExch(p, epoch);
#endif
}
__device__ void wait_epoch(unsigned *p, unsigned epoch) {
  unsigned value;
  do {
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
#else
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
#endif
  } while (value < epoch);
#if __CUDA_ARCH__ < 700
  __threadfence();
#endif
}
)cuda";
  const std::string before = R"cuda(
  for (unsigned epoch = 1; epoch <= 2; ++epoch) {
    if (blockIdx.x == 0) {
      wait_epoch(flags, epoch);
      for (int c = 0; c < count; ++c)
        for (unsigned long long b = threadIdx.x; b < copies[c].bytes; b += blockDim.x)
          copies[c].dst[b] = epoch == 1 ? 0 : copies[c].src[b];
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) signal_epoch(flags + 1, epoch);
      wait_epoch(flags + 2, epoch);
      continue;
    }
    // Prime the consumer SM's read-only cache before another CTA overwrites these inputs.
    unsigned checksum = 0;
    for (int c = 0; c < count; ++c)
      for (unsigned long long b = 4 * threadIdx.x; b + 4 <= copies[c].bytes; b += 4 * blockDim.x) {
        unsigned value;
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(copies[c].dst + b) : "memory");
        checksum ^= value;
      }
    asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(flags + 3 + threadIdx.x), "r"(checksum) : "memory");
    __syncthreads();
    if (threadIdx.x == 0) signal_epoch(flags, epoch);
    wait_epoch(flags + 1, epoch);
)cuda";
  const std::string src = "#include \"nvfp4effma_items.cuh\"\n" + publication +
                          "extern \"C\" __global__ void __launch_bounds__(" + std::to_string(it.threads) + ", 1)\n"
                          "run(const void *args, int items, const PublishCopy *copies, int count, unsigned *flags) {\n"
                          "  extern __shared__ __align__(16) unsigned char smem[];\n" + (publish ? before : "") +
                          (publish ? "  for (int i = 0; i < items; ++i) {\n" :
                                     "  for (int i = blockIdx.x; i < items; i += gridDim.x) {\n") +
                          "    " + entry + "(*(const " + it.args_type + " *)args, i, threadIdx.x, smem);\n"
                          "    __syncthreads();\n"
                          "  }\n" +
                          (publish ? "  __threadfence();\n  __syncthreads();\n"
                                     "  if (threadIdx.x == 0) signal_epoch(flags + 2, epoch);\n  }\n" : "") +
                          "}\n";
  const std::string arch = "--gpu-architecture=compute_" + std::to_string(it.arch);
  const char *opts[] = {arch.c_str(), "--std=c++17", "--device-as-default-execution-space", "--include-path=" NVFP4A_ROOT "/include", "--include-path=" NVFP4A_ROOT "/src",
                        "--include-path=" NVFP4A_CUDA_INCLUDE, "--include-path=" NVFP4A_CUDA_INCLUDE "/cccl"};
  nvrtcProgram p;
  check_nvrtc(nvrtcCreateProgram(&p, src.c_str(), "executor.cu", 0, nullptr, nullptr), "nvrtcCreateProgram");
  const nvrtcResult r = nvrtcCompileProgram(p, 7, opts);
  size_t n;
  check_nvrtc(nvrtcGetProgramLogSize(p, &n), "nvrtcGetProgramLogSize");
  std::string log(n, '\0');
  check_nvrtc(nvrtcGetProgramLog(p, &log[0]), "nvrtcGetProgramLog");
  if (r != NVRTC_SUCCESS) { printf("NVRTC: %s\n%s\n", nvrtcGetErrorString(r), log.c_str()); exit(1); }
  check_nvrtc(nvrtcGetPTXSize(p, &n), "nvrtcGetPTXSize");
  std::string ptx(n, '\0');
  check_nvrtc(nvrtcGetPTX(p, &ptx[0]), "nvrtcGetPTX");
  if (const char *path = getenv("NVFP4A_ITEMS_DUMP_PTX")) {
    FILE *file = fopen(path, "wb");
    const size_t bytes = ptx.size() - 1;  // NVRTC's size includes the terminating NUL.
    if (!file || fwrite(ptx.data(), 1, bytes, file) != bytes || fclose(file)) {
      fprintf(stderr, "cannot write executor PTX to %s\n", path); exit(1);
    }
  }
  check_nvrtc(nvrtcDestroyProgram(&p), "nvrtcDestroyProgram");
  CUmodule mod;
  check_driver(cuModuleLoadData(&mod, ptx.data()), "cuModuleLoadData");
  check_driver(cuModuleGetFunction(fn, mod, "run"), "cuModuleGetFunction");
  return mod;
}

// One compile per distinct executor, shared by the threads that ask for it; the modules stay loaded to the end.
CUmodule executor(const nvfp4a_items &it, CUfunction *fn, bool publish, bool readonly_control) {
  struct Built { CUmodule module; CUfunction fn; };
  static std::mutex lock;
  static std::map<std::string, std::shared_future<Built>> built;
  const std::string key = std::string(it.entry) + '|' + it.args_type + '|' + std::to_string(it.threads) + '|' + std::to_string(it.arch) +
                          (publish ? "|pub" : "") + (readonly_control ? "|ro" : "");
  std::promise<Built> promise;
  std::shared_future<Built> future;
  bool mine = false;
  {
    std::lock_guard<std::mutex> guard(lock);
    const auto at = built.find(key);
    if (at != built.end()) future = at->second;
    else {
      future = promise.get_future().share();
      built.emplace(key, future);
      mine = true;
    }
  }
  if (mine) {
    Built b;
    b.module = build_executor(it, &b.fn, publish, readonly_control);
    promise.set_value(b);
  }
  const Built b = future.get();
  *fn = b.fn;
  return b.module;
}

struct ItemCase {
  int m, n, k;
  nvfp4a_dtype bt, dt;
  int mask, bias, c;  // mask: 1 column scale, 2 SwiGLU, 4 second output; 0 the plain epilogue
  bool tc;
  bool row_global = false;
  bool publish = false;
  bool readonly_control = false;
  int ncols = 0;  // > 0: the items compute this many of the n columns (the count in device memory)
};

template <class T> T *todev(const std::vector<T> &v) {
  T *d;
  check_cuda(cudaMalloc(&d, v.size() * sizeof(T) + 16), "cudaMalloc");
  check_cuda(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  return d;
}
template <class T> std::vector<T> tohost(const T *d, size_t count) {
  std::vector<T> h(count);
  check_cuda(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return h;
}

int run(const ItemCase &c) {
  const bool q8 = c.bt == NVFP4A_INT8, glu = c.mask & 2, out2 = c.mask & 4;
  const int lda = c.k ? c.k : 64, ldb = lda, mo = glu ? c.m / 2 : c.m, ldd = mo + 3, np = 37, ldp = 40, lsq = c.m / 16 + 1, ldd2 = c.m + 5;
  HostFp4 hA(c.m, c.k, lda, c.k / 16, 0.375f, c.row_global ? NVFP4A_GLOBAL_PER_ROW : 0);
  HostQ8 hB8(q8 ? c.n : 0, c.k, ldb, c.k / 16, 1.75f);
  if (!c.k) { hA.data.clear(); hA.sc.clear(); hB8.data.clear(); hB8.sc.clear(); }
  std::vector<uint8_t> hB(q8 ? 0 : (size_t)ldb * c.n * 2), hbias((size_t)c.m * 2), hw2((size_t)c.m * 2), hD((size_t)ldd * c.n * esize(c.dt));
  if (!q8) fill(NVFP4A_BF16, hB, 2.f);
  fill(NVFP4A_BF16, hbias, 4.f);
  fill(NVFP4A_BF16, hw2, 2.f);
  fill(c.dt, hD, 4.f);
  std::vector<float> parts((size_t)ldp * c.n);
  for (auto &x : parts) x = (float)(rnd() % 4096) / 64.f;
  uint8_t *dA = todev(hA.data), *dSA = todev(hA.sc), *dbias = todev(hbias), *dw2 = todev(hw2);
  void *dB = q8 ? (void *)todev(hB8.data) : (void *)todev(hB);
  float *dSB = q8 ? todev(hB8.sc) : nullptr, *dparts = todev(parts), *dgb = devf(1.75f);
  float *dga = c.row_global ? todev(hA.gv) : nullptr;
  void *ia = nullptr;
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, 0, dga, nullptr, hA.flags, 0};
  if (c.tc) {
    const int kind = q8 ? NVFP4A_TC_A8 : NVFP4A_TC_A16;
    check_cuda(cudaMalloc(&ia, nvfp4a_tc_image_bytes(kind, NVFP4A_NVFP4, c.m, c.k) + 16), "cudaMalloc A image");
    if (nvfp4a_tc_image(kind, &A, lda, c.m, c.k, ia, 0)) { fprintf(stderr, "%s\n", nvfp4a_last_error()); exit(1); }
    A.data = ia;
  }
  nvfp4a_tensor B = {dB, c.bt, dSB, q8 ? c.k / 16 : 0, q8 ? dgb : nullptr, nullptr, 0, 0};
  const bool gemm_tc = c.tc && c.n > 8;  // tile items over B's image
  void *ib = nullptr;
  if (gemm_tc) {
    check_cuda(cudaMalloc(&ib, nvfp4a_tc_image_bytes(NVFP4A_TC_B, c.bt, c.n, c.k) + 16), "cudaMalloc B image");
    if (nvfp4a_tc_image(NVFP4A_TC_B, &B, ldb, c.n, c.k, ib, 0)) { fprintf(stderr, "%s\n", nvfp4a_last_error()); exit(1); }
    B.data = ib;
  }
  nvfp4a_tensor bias = {dbias, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
  const float alpha = 0.75f, beta = c.c ? 1.f : 0.f;

  // two sets of outputs from the same initial contents: the call's and the items'
  struct Out { uint8_t *d, *d2; float *sq; } o[2];
  for (Out &x : o) {
    x.d = todev(hD);
    x.d2 = todev(std::vector<uint8_t>((size_t)ldd2 * c.n * 2, 0xA5));
    x.sq = todev(std::vector<float>((size_t)lsq * c.n, -1.f));
  }
  const auto ex_of = [&](const Out &x) {
    nvfp4a_epilogue e{};
    if (c.mask & 1) { e.norm_parts = dparts; e.norm_nparts = np; e.norm_ld = ldp; e.norm_inv_n = 1.f / 1024.f; e.norm_eps = 1e-6f; }
    e.swiglu = glu;
    if (out2) { e.d2 = x.d2; e.d2_dtype = NVFP4A_BF16; e.ldd2 = ldd2; e.w2 = dw2; e.w2_dtype = NVFP4A_BF16; e.sq = x.sq; e.ld_sq = lsq; }
    return e;
  };
  const nvfp4a_epilogue e0 = ex_of(o[0]), e1 = ex_of(o[1]);
  const nvfp4a_tensor D0 = {o[0].d, c.dt, nullptr, 0, nullptr, nullptr, 0, 0}, D1 = {o[1].d, c.dt, nullptr, 0, nullptr, nullptr, 0, 0};
  const nvfp4a_epilogue *x0 = c.mask ? &e0 : nullptr, *x1 = c.mask ? &e1 : nullptr;
  const nvfp4a_tensor *bp = c.bias ? &bias : nullptr;
  nvfp4a_items reference;
  uint8_t *zero_ws = gemm_tc && !c.k ? todev(std::vector<uint8_t>(nvfp4a_tc_workspace_bytes(), 0)) : nullptr;
  nvfp4a_status st = zero_ws ? nvfp4a_gemm_tc_ex(c.m, c.n, c.k, alpha, &A, &B, beta, nullptr, 0, &D0, ldd, bp, x0, zero_ws, 0)
                     : gemm_tc ? nvfp4a_gemm_tc_ex_items(c.m, c.n, c.k, alpha, &A, &B, beta, nullptr, 0, &D0, ldd, bp, x0, &reference)
                     : c.tc  ? nvfp4a_gemv_tc_ex(c.m, c.n, c.k, alpha, &A, &B, ldb, beta, nullptr, 0, &D0, ldd, bp, x0, 0)
                             : nvfp4a_gemm_ex(c.m, c.n, c.k, alpha, &A, lda, &B, ldb, beta, nullptr, 0, &D0, ldd, bp, x0, 0);
  if (!st && gemm_tc && c.k) check_cuda(tile_reference(reference, c.bt, c.mask), "TC tile reference launch");
  nvfp4a_items it;
  if (!st)
    st = gemm_tc ? nvfp4a_gemm_tc_ex_items(c.m, c.n, c.k, alpha, &A, &B, beta, nullptr, 0, &D1, ldd, bp, x1, &it)
         : c.tc  ? nvfp4a_gemv_tc_ex_items(c.m, c.n, c.k, alpha, &A, &B, ldb, beta, nullptr, 0, &D1, ldd, bp, x1, &it)
                 : nvfp4a_gemm_ex_items(c.m, c.n, c.k, alpha, &A, lda, &B, ldb, beta, nullptr, 0, &D1, ldd, bp, x1, &it);
  int bad = 0;
  int *dncols = nullptr;
  if (!st && c.ncols) {
    check_cuda(cudaMalloc(&dncols, sizeof(int)), "cudaMalloc ncols");
    check_cuda(cudaMemcpy(dncols, &c.ncols, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy ncols");
    st = nvfp4a_items_n_device(&it, dncols);
  }
  if (!st) {
    void *dargs;
    check_cuda(cudaMalloc(&dargs, it.args_bytes), "cudaMalloc arguments");
    check_cuda(cudaMemcpy(dargs, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice), "cudaMemcpy arguments");
    CUfunction fn;
    const CUmodule module = executor(it, &fn, c.publish, c.readonly_control);
    if (it.smem_bytes > 48 * 1024)
      check_driver(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, it.smem_bytes), "cuFuncSetAttribute");
    std::vector<PublishCopy> copies;
    if (c.publish) {
      check_cuda(cudaDeviceSynchronize(), "reference synchronize before input reuse");
      const auto save = [&](void *dst, size_t bytes) {
        uint8_t *src;
        check_cuda(cudaMalloc(&src, bytes), "cudaMalloc publication source");
        check_cuda(cudaMemcpy(src, dst, bytes, cudaMemcpyDeviceToDevice), "save publication source");
        check_cuda(cudaMemset(dst, 0xA5, bytes), "poison publication destination");
        copies.push_back(PublishCopy{(uint8_t *)dst, src, bytes});
      };
      save(B.data, gemm_tc ? nvfp4a_tc_image_bytes(NVFP4A_TC_B, c.bt, c.n, c.k) : (size_t)ldb * c.n * (q8 ? 1 : 2));
      if (q8 && !gemm_tc) save(dSB, hB8.sc.size() * sizeof(float));
      if (q8) save(dgb, sizeof(float));
      if (c.mask & 1) save(dparts, parts.size() * sizeof(float));
      if (c.c) save(o[1].d, hD.size());
    }
    PublishCopy *dcopies = copies.empty() ? nullptr : todev(copies);
    unsigned *flags = c.publish ? todev(std::vector<unsigned>(3 + it.threads, 0)) : nullptr;
    int items = it.items, count = (int)copies.size();
    void *params[] = {&dargs, &items, &dcopies, &count, &flags};
    const unsigned grid = it.items > 7 ? 7 : it.items;
    if (c.publish) {
      int blocks, sms, device_id;
      check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, it.threads, it.smem_bytes), "executor occupancy");
      check_cuda(cudaGetDevice(&device_id), "current device");
      check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id), "SM count");
      if (blocks * sms < 2) { fprintf(stderr, "publication test requires two resident CTAs\n"); exit(1); }
      check_driver(cuLaunchCooperativeKernel(fn, 2, 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params), "publication executor");
    } else {
      check_driver(cuLaunchKernel(fn, grid, 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr), "cuLaunchKernel executor");
    }
    check_cuda(cudaDeviceSynchronize(), "executor synchronize");
    check_cuda(cudaFree(dargs), "cudaFree arguments");
    check_cuda(cudaFree(dcopies), "cudaFree publication descriptors");
    check_cuda(cudaFree(flags), "cudaFree publication epochs");
    for (const auto &copy : copies) {
      if (copy.dst == o[1].d && tohost(copy.src, hD.size()) != hD) {
        fprintf(stderr, "publication C source changed\n"); exit(1);
      }
      check_cuda(cudaFree((void *)copy.src), "cudaFree publication source");
    }
    // the outputs, bit for bit
    const size_t dbytes = hD.size(), d2bytes = (size_t)ldd2 * c.n * 2, sqn = (size_t)lsq * c.n;
    const auto g0 = tohost(o[0].d, dbytes), g1 = tohost(o[1].d, dbytes);
    const auto a0 = tohost(o[0].d2, d2bytes), a1 = tohost(o[1].d2, d2bytes);
    const auto s0 = tohost(o[0].sq, sqn), s1 = tohost(o[1].sq, sqn);
    // ncols: the columns below it are the call's bits; the columns of whole tiles (groups of 8 for the GEMV) past it keep their contents
    bool differ = g0 != g1 || a0 != a1 || memcmp(s0.data(), s1.data(), sqn * 4);
    if (c.ncols) {
      const int tile = it.kind == NVFP4A_ITEM_GEMV ? 8 : 128, past = (c.ncols + tile - 1) / tile * tile;
      const std::vector<uint8_t> d2init(d2bytes, 0xA5);
      differ = false;
      for (int j = 0; j < c.n && !differ; ++j) {
        const size_t d = (size_t)j * ldd * esize(c.dt), a = (size_t)j * ldd2 * 2, s = (size_t)j * lsq;
        const bool low = j < c.ncols, high = j >= past;
        if (low) differ = memcmp(&g0[d], &g1[d], (size_t)mo * esize(c.dt)) || memcmp(&a0[a], &a1[a], (size_t)(out2 ? c.m : 0) * 2) ||
                          memcmp(&s0[s], &s1[s], (size_t)(out2 ? c.m / 16 : 0) * 4);
        if (high) differ = memcmp(&g1[d], &hD[d], (size_t)mo * esize(c.dt)) || memcmp(&a1[a], &d2init[a], (size_t)(out2 ? c.m : 0) * 2) ||
                           std::any_of(&s1[s], &s1[s] + (out2 ? c.m / 16 : 0), [](float x) { return x != -1.f; });
      }
    }
    if (differ) {
      say("   outputs differ bit for bit from the %s\n", gemm_tc ? "whole-tile library kernel" : "call");
      for (size_t i = 0; i < g0.size() / esize(c.dt); ++i)
        if (memcmp(g0.data() + i * esize(c.dt), g1.data() + i * esize(c.dt), esize(c.dt))) {
          say("   D[%zu]: got %.9g, reference %.9g\n", i, hload(c.dt, g1.data(), i), hload(c.dt, g0.data(), i));
          break;
        }
      ++bad;
    }
    // every changed byte in a declared write range; read ranges within their arguments
    std::vector<char> wd(dbytes, 0), wd2(d2bytes, 0), wsq(sqn * 4, 0);
    std::vector<nvfp4a_item_range> rg(64);
    if (nvfp4a_item_access(nullptr, 0, nullptr, 0) != -1 || nvfp4a_item_access(&it, -1, nullptr, 0) != -1 ||
        nvfp4a_item_access(&it, it.items, nullptr, 0) != -1 || nvfp4a_item_access(&it, 0, nullptr, -1) != -1 ||
        nvfp4a_item_access(&it, 0, nullptr, 1) != -1) {
      say("   invalid access query accepted\n"); ++bad;
    }
    for (int i = 0; i < it.items && !bad; ++i) {
      const int nr = nvfp4a_item_access(&it, i, rg.data(), (int)rg.size());
      nvfp4a_item_range truncated[2];
      memset(truncated, 0xA5, sizeof truncated);
      unsigned char guard[sizeof truncated[1]];
      memcpy(guard, &truncated[1], sizeof guard);
      if (nr <= 0 || nr > (int)rg.size() || nvfp4a_item_access(&it, i, nullptr, 0) != nr ||
          nvfp4a_item_access(&it, i, truncated, 1) != nr || memcmp(guard, &truncated[1], sizeof guard) ||
          rg[0].arg != truncated[0].arg || rg[0].write != truncated[0].write || rg[0].offset != truncated[0].offset ||
          rg[0].bytes != truncated[0].bytes || rg[0].stride != truncated[0].stride || rg[0].rows != truncated[0].rows) {
        say("   inconsistent or overflowing access query\n"); ++bad; break;
      }
      bool zero_a = false, zero_scales = false, zero_global = false, zero_norm = false;
      long long first_row = -1, first_col = -1;
      for (int q = 0; q < nr; ++q) {
        const nvfp4a_item_range &r = rg[q];
        if (!r.write && !r.offset) {
          zero_a |= r.arg == NVFP4A_ARG_A;
          zero_scales |= r.arg == NVFP4A_ARG_A_SCALES;
          zero_global |= r.arg == NVFP4A_ARG_A_GLOBAL;
          zero_norm |= r.arg == NVFP4A_ARG_NORM_PARTS;
        }
        if (r.arg == NVFP4A_ARG_D && r.write == 1) {
          first_row = r.offset / esize(c.dt) % ldd * (glu ? 2 : 1);
          first_col = r.offset / esize(c.dt) / ldd;
        }
        std::vector<char> *w = r.arg == NVFP4A_ARG_D ? &wd : r.arg == NVFP4A_ARG_D2 ? &wd2 : r.arg == NVFP4A_ARG_SQ ? &wsq : nullptr;
        const long long end = r.offset + (long long)(r.rows - 1) * r.stride + r.bytes;
        const long long size = r.arg == NVFP4A_ARG_D ? dbytes : r.arg == NVFP4A_ARG_D2 ? d2bytes : r.arg == NVFP4A_ARG_SQ ? sqn * 4
                               : r.arg == NVFP4A_ARG_A ? (c.tc ? (long long)nvfp4a_tc_image_bytes(q8 ? NVFP4A_TC_A8 : NVFP4A_TC_A16,
                                                                                                  NVFP4A_NVFP4, c.m, c.k)
                                                               : (long long)c.m * lda / 2)
                               : r.arg == NVFP4A_ARG_A_SCALES ? (long long)c.m * c.k / 16
                               : r.arg == NVFP4A_ARG_B ? (gemm_tc ? (long long)nvfp4a_tc_image_bytes(NVFP4A_TC_B, c.bt, c.n, c.k)
                                                                  : (long long)ldb * c.n * (q8 ? 1 : 2))
                               : r.arg == NVFP4A_ARG_B_SCALES ? (long long)c.n * c.k / 16 * 4
                               : r.arg == NVFP4A_ARG_A_GLOBAL ? (long long)c.m * 4
                               : r.arg == NVFP4A_ARG_B_GLOBAL ? 4
                               : r.arg == NVFP4A_ARG_NORM_PARTS ? (long long)ldp * c.n * 4
                               : r.arg == NVFP4A_ARG_BIAS || r.arg == NVFP4A_ARG_W2 ? (long long)c.m * 2 : 0;
        if (r.offset < 0 || r.bytes <= 0 || r.rows <= 0 || r.stride < 0 || end > size) {
          say("   item %d: invalid range of argument %d (%lld > %lld)\n", i, r.arg, end, size); ++bad;
        }
        if (!r.write || !w || bad) continue;
        for (int row = 0; row < r.rows; ++row)
          for (long long b = 0; b < r.bytes; ++b) {
            char &f = (*w)[r.offset + row * r.stride + b];
            if (f) { say("   item %d writes byte %lld of argument %d, another item too\n", i, r.offset + row * r.stride + b, r.arg); ++bad; }
            f = 1;
          }
      }
      const bool row_tail = first_row > 0 && first_row + it.rows > c.m;
      if ((it.kind == NVFP4A_ITEM_GEMV && c.k && row_tail && (!zero_a || !zero_scales)) ||
          (gemm_tc && c.row_global && row_tail && !zero_global) ||
          (gemm_tc && (c.mask & 1) && first_col > 0 && first_col + it.cols > c.n && !zero_norm)) {
        say("   item %d omits a boundary fallback read\n", i); ++bad;
      }
    }
    const auto coverage = [&](const std::vector<char> &w, int rows, int ld, int bytes, const char *what) {
      for (size_t b = 0; b < w.size() && !bad; ++b)
        if ((w[b] != 0) != (b % ((size_t)ld * bytes) < (size_t)rows * bytes)) {
          say("   %s byte %zu: incorrect declared write coverage\n", what, b); ++bad;
        }
    };
    coverage(wd, mo, ldd, (int)esize(c.dt), "D");
    coverage(wd2, out2 ? c.m : 0, ldd2, 2, "d2");
    coverage(wsq, out2 ? c.m / 16 : 0, lsq, 4, "sq");
    const auto ch = [&](const std::vector<uint8_t> &now, const std::vector<uint8_t> &was, const std::vector<char> &w, const char *what) {
      for (size_t b = 0; b < now.size() && !bad; ++b)
        if (now[b] != was[b] && !w[b]) { say("   %s byte %zu changed outside the write ranges\n", what, b); ++bad; }
    };
    ch(g1, hD, wd, "D");
    if (out2) {
      ch(a1, std::vector<uint8_t>(d2bytes, 0xA5), wd2, "d2");
      std::vector<uint8_t> sb(sqn * 4), sw(sqn * 4);
      memcpy(sb.data(), s1.data(), sqn * 4);
      const std::vector<float> init(sqn, -1.f);
      memcpy(sw.data(), init.data(), sqn * 4);
      ch(sb, sw, wsq, "sq");
    }
  }
  say("%-8s m %5d n %d k %5d B %-4s D %-4s ex %d%s%s: %d items x %d threads, %d B shared, %d regs: %s\n",
         c.publish ? (c.tc ? "tc pub" : "cuda pub") : (c.tc ? "tc" : "cuda"), c.m, c.n,
         c.k, DN[c.bt], DN[c.dt], c.mask, c.bias ? " bias" : "", c.c ? " +=" : "", st ? 0 : it.items, st ? 0 : it.threads,
         st ? 0 : it.smem_bytes, st ? 0 : it.regs, st ? nvfp4a_last_error() : bad ? "FAIL" : "ok");
  cudaFree(dncols);
  for (void *p : {(void *)dA, (void *)dSA, (void *)dbias, (void *)dw2, dB, (void *)dSB, (void *)dparts, (void *)dgb,
                  (void *)dga, ia, ib, (void *)zero_ws})
    check_cuda(cudaFree(p), "cudaFree input");
  for (Out &x : o) { cudaFree(x.d); cudaFree(x.d2); cudaFree(x.sq); }
  return bad || st ? 1 : 0;
}

// The image of a BF16 B as items against nvfp4a_tc_image: bit for bit; with ncols < n the columns past it are zeros in the block
// that holds the count and a block wholly past it is left alone; the declared ranges cover every changed byte, once.
int run_image(int n, int k, int ncols) {
  const int ldb = k + 64;
  std::vector<uint8_t> hB((size_t)ldb * n * 2);
  fill(NVFP4A_BF16, hB, 2.f);
  std::vector<uint8_t> hZ = hB;  // the columns the items compute, the rest zero (the reference's input)
  if (ncols) memset(&hZ[(size_t)ncols * ldb * 2], 0, (size_t)(n - ncols) * ldb * 2);
  const size_t bytes = nvfp4a_tc_image_bytes(NVFP4A_TC_B, NVFP4A_BF16, n, k);
  void *dB = todev(hB), *dZ = todev(hZ), *ref, *img;
  check_cuda(cudaMalloc(&ref, bytes + 16), "cudaMalloc reference image");
  check_cuda(cudaMalloc(&img, bytes + 16), "cudaMalloc image");
  const nvfp4a_tensor B = {dB, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0}, Z = {dZ, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
  check_cuda(cudaMemset(img, 0xA5, bytes), "poison the image");
  int bad = 0;
  nvfp4a_items it;
  nvfp4a_status st = nvfp4a_tc_image(NVFP4A_TC_B, &Z, ldb, n, k, ref, 0);
  if (!st) st = nvfp4a_tc_image_b_items(n, k, &B, ldb, img, &it);
  int *dn = nullptr;
  if (!st && ncols) {
    check_cuda(cudaMalloc(&dn, sizeof(int)), "cudaMalloc ncols");
    check_cuda(cudaMemcpy(dn, &ncols, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy ncols");
    st = nvfp4a_items_n_device(&it, dn);
  }
  if (!st) {
    void *dargs;
    check_cuda(cudaMalloc(&dargs, it.args_bytes), "cudaMalloc arguments");
    check_cuda(cudaMemcpy(dargs, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice), "cudaMemcpy arguments");
    CUfunction fn;
    executor(it, &fn, false, false);
    int items = it.items, count = 0;
    PublishCopy *copies = nullptr;
    unsigned *flags = nullptr;
    void *params[] = {&dargs, &items, &copies, &count, &flags};
    check_driver(cuLaunchKernel(fn, std::min(it.items, 7), 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr), "cuLaunchKernel executor");
    check_cuda(cudaDeviceSynchronize(), "executor synchronize");
    check_cuda(cudaFree(dargs), "cudaFree arguments");
    const auto got = tohost((const uint8_t *)img, bytes), want = tohost((const uint8_t *)ref, bytes);
    const size_t record = nvfp4a::kern::tc::RBH, block = (size_t)(k / 64) * record;
    std::vector<char> written(bytes, 0);
    std::vector<nvfp4a_item_range> rg(8);
    for (int i = 0; i < it.items && !bad; ++i) {
      const int nr = nvfp4a_item_access(&it, i, rg.data(), (int)rg.size());
      for (int q = 0; q < nr && !bad; ++q) {
        const nvfp4a_item_range &r = rg[q];
        const long long end = r.offset + (long long)(r.rows - 1) * r.stride + r.bytes;
        if (r.arg == NVFP4A_ARG_B && (r.write || end > (long long)ldb * n * 2)) { say("   item %d: bad read range\n", i); ++bad; }
        if (r.arg == NVFP4A_ARG_B_IMAGE)
          for (long long b = 0; b < r.bytes && !bad; ++b) {
            if (!r.write || written[r.offset + b]) { say("   item %d: image byte %lld declared twice\n", i, r.offset + b); ++bad; }
            written[r.offset + b] = 1;
          }
      }
    }
    for (size_t b = 0; b < bytes && !bad; ++b) {
      const bool computed = !ncols || (b / block) * 128 < (size_t)ncols;
      if (computed ? got[b] != want[b] : got[b] != 0xA5) { say("   image byte %zu: got %02x, want %02x\n", b, got[b], computed ? want[b] : 0xA5); ++bad; }
      if (!written[b]) { say("   image byte %zu is in no declared write range\n", b); ++bad; }
    }
  }
  say("image_b n %d k %d%s: %d items: %s\n", n, k, ncols ? (" columns " + std::to_string(ncols)).c_str() : "", st ? 0 : it.items,
      st ? nvfp4a_last_error() : bad ? "FAIL" : "ok");
  for (void *p : {dB, dZ, ref, img, (void *)dn}) cudaFree(p);
  return bad || st ? 1 : 0;
}

// The INT8 B image as items against nvfp4a_tc_image, as run_image for BF16: the columns past ncols are zero data and scales in the
// reference; a block wholly past it is left alone.
int run_image_int8(int n, int k, int ncols) {
  const int ldb = k + 64, lsc = k / 16 + 4;
  HostQ8 h(n, k, ldb, lsc, 1.f);
  HostQ8 z = h;
  for (int c = ncols; ncols && c < n; ++c) {
    std::fill(z.data.begin() + (size_t)c * ldb, z.data.begin() + (size_t)(c + 1) * ldb, 0);
    std::fill(z.sc.begin() + (size_t)c * lsc, z.sc.begin() + (size_t)(c + 1) * lsc, 0.f);
  }
  const size_t bytes = nvfp4a_tc_image_bytes(NVFP4A_TC_B, NVFP4A_INT8, n, k);
  uint8_t *dB = todev(std::vector<uint8_t>(h.data.begin(), h.data.end())), *dZ = todev(std::vector<uint8_t>(z.data.begin(), z.data.end()));
  float *dSB = todev(h.sc), *dSZ = todev(z.sc);
  void *ref, *img;
  check_cuda(cudaMalloc(&ref, bytes + 16), "cudaMalloc reference image");
  check_cuda(cudaMalloc(&img, bytes + 16), "cudaMalloc image");
  check_cuda(cudaMemset(img, 0xA5, bytes), "poison the image");
  const nvfp4a_tensor B = {dB, NVFP4A_INT8, dSB, lsc, nullptr, nullptr, 0, 0}, Z = {dZ, NVFP4A_INT8, dSZ, lsc, nullptr, nullptr, 0, 0};
  int bad = 0;
  nvfp4a_items it;
  nvfp4a_status st = nvfp4a_tc_image(NVFP4A_TC_B, &Z, ldb, n, k, ref, 0);
  if (!st) st = nvfp4a_tc_image_b_items(n, k, &B, ldb, img, &it);
  int *dn = nullptr;
  if (!st && ncols) {
    check_cuda(cudaMalloc(&dn, sizeof(int)), "cudaMalloc ncols");
    check_cuda(cudaMemcpy(dn, &ncols, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy ncols");
    st = nvfp4a_items_n_device(&it, dn);
  }
  if (!st) {
    void *dargs;
    check_cuda(cudaMalloc(&dargs, it.args_bytes), "cudaMalloc arguments");
    check_cuda(cudaMemcpy(dargs, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice), "cudaMemcpy arguments");
    CUfunction fn;
    executor(it, &fn, false, false);
    int items = it.items, count = 0;
    PublishCopy *copies = nullptr;
    unsigned *flags = nullptr;
    void *params[] = {&dargs, &items, &copies, &count, &flags};
    check_driver(cuLaunchKernel(fn, std::min(it.items, 7), 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr), "cuLaunchKernel executor");
    check_cuda(cudaDeviceSynchronize(), "executor synchronize");
    check_cuda(cudaFree(dargs), "cudaFree arguments");
    const auto got = tohost((const uint8_t *)img, bytes), want = tohost((const uint8_t *)ref, bytes);
    const size_t block = (size_t)(k / 64) * nvfp4a::kern::tc::RB8;
    std::vector<char> written(bytes, 0);
    std::vector<nvfp4a_item_range> rg(8);
    for (int i = 0; i < it.items && !bad; ++i) {
      const int nr = nvfp4a_item_access(&it, i, rg.data(), (int)rg.size());
      for (int q = 0; q < nr && !bad; ++q) {
        const nvfp4a_item_range &r = rg[q];
        const long long end = r.offset + (long long)(r.rows - 1) * r.stride + r.bytes;
        if (r.arg == NVFP4A_ARG_B && (r.write || end > (long long)ldb * n)) { say("   item %d: bad data range\n", i); ++bad; }
        if (r.arg == NVFP4A_ARG_B_SCALES && (r.write || end > (long long)lsc * n * 4)) { say("   item %d: bad scale range\n", i); ++bad; }
        if (r.arg == NVFP4A_ARG_B_IMAGE)
          for (long long b = 0; b < r.bytes && !bad; ++b) {
            if (!r.write || written[r.offset + b]) { say("   item %d: image byte %lld declared twice\n", i, r.offset + b); ++bad; }
            written[r.offset + b] = 1;
          }
      }
    }
    for (size_t b = 0; b < bytes && !bad; ++b) {
      const bool computed = !ncols || (b / block) * 128 < (size_t)ncols;
      if (computed ? got[b] != want[b] : got[b] != 0xA5) { say("   image byte %zu: got %02x, want %02x\n", b, got[b], computed ? want[b] : 0xA5); ++bad; }
      if (!written[b]) { say("   image byte %zu is in no declared write range\n", b); ++bad; }
    }
  }
  say("image_b INT8 n %d k %d%s: %d items: %s\n", n, k, ncols ? (" columns " + std::to_string(ncols)).c_str() : "", st ? 0 : it.items,
      st ? nvfp4a_last_error() : bad ? "FAIL" : "ok");
  for (void *p : {(void *)dB, (void *)dZ, (void *)dSB, (void *)dSZ, ref, img, (void *)dn}) cudaFree(p);
  return bad || st ? 1 : 0;
}

// nvfp4a_quantize to INT8 as items against the call: data and scales bit for bit; with nrows < rows the rows past it keep their
// contents; the declared ranges are disjoint, inside their arguments and cover everything written.
int run_quant(nvfp4a_dtype dt, int rows, int cols, int nrows) {
  const int lds = cols + 16, ldd = cols + 32, lsc = cols / 16 + 4;
  const size_t eb = dt == NVFP4A_FP32 ? 4 : 2;
  std::vector<uint8_t> hs((size_t)lds * rows * eb);
  fill(dt, hs, 3.f);
  uint8_t *dsrc = todev(hs), *dref = todev(std::vector<uint8_t>((size_t)ldd * rows, 0xA5)), *dgot = todev(std::vector<uint8_t>((size_t)ldd * rows, 0xA5));
  float *dsref = todev(std::vector<float>((size_t)lsc * rows, -1.f)), *dsgot = todev(std::vector<float>((size_t)lsc * rows, -1.f));
  const nvfp4a_tensor src = {dsrc, dt, nullptr, 0, nullptr, nullptr, 0, 0}, ref = {dref, NVFP4A_INT8, dsref, lsc, nullptr, nullptr, 0, 0},
                      got = {dgot, NVFP4A_INT8, dsgot, lsc, nullptr, nullptr, 0, 0};
  int bad = 0;
  nvfp4a_items it;
  nvfp4a_status st = nvfp4a_quantize(&src, lds, &ref, ldd, rows, cols, 0);
  if (!st) st = nvfp4a_quantize_int8_items(&src, lds, &got, ldd, rows, cols, &it);
  int *dn = nullptr;
  const int active = nrows ? nrows : rows;
  if (!st && nrows) {
    check_cuda(cudaMalloc(&dn, sizeof(int)), "cudaMalloc nrows");
    check_cuda(cudaMemcpy(dn, &nrows, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy nrows");
    st = nvfp4a_items_n_device(&it, dn);
  }
  if (!st) {
    void *dargs;
    check_cuda(cudaMalloc(&dargs, it.args_bytes), "cudaMalloc arguments");
    check_cuda(cudaMemcpy(dargs, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice), "cudaMemcpy arguments");
    CUfunction fn;
    executor(it, &fn, false, false);
    int items = it.items, count = 0;
    PublishCopy *copies = nullptr;
    unsigned *flags = nullptr;
    void *params[] = {&dargs, &items, &copies, &count, &flags};
    check_driver(cuLaunchKernel(fn, std::min(it.items, 7), 1, 1, it.threads, 1, 1, it.smem_bytes, 0, params, nullptr), "cuLaunchKernel executor");
    check_cuda(cudaDeviceSynchronize(), "executor synchronize");
    check_cuda(cudaFree(dargs), "cudaFree arguments");
    const auto d_ref = tohost(dref, (size_t)ldd * rows), d_got = tohost(dgot, (size_t)ldd * rows);
    const auto s_ref = tohost(dsref, (size_t)lsc * rows), s_got = tohost(dsgot, (size_t)lsc * rows);
    std::vector<char> wd(d_got.size(), 0), ws(s_got.size() * 4, 0);
    std::vector<nvfp4a_item_range> rg(8);
    for (int i = 0; i < it.items && !bad; ++i) {
      const int nr = nvfp4a_item_access(&it, i, rg.data(), (int)rg.size());
      for (int q = 0; q < nr; ++q) {
        const nvfp4a_item_range &r = rg[q];
        std::vector<char> *w = r.arg == NVFP4A_ARG_Q_DATA ? &wd : r.arg == NVFP4A_ARG_Q_SCALES ? &ws : nullptr;
        const long long end = r.offset + (long long)(r.rows - 1) * r.stride + r.bytes,
                        size = r.arg == NVFP4A_ARG_Q_SRC ? (long long)lds * rows * eb : r.arg == NVFP4A_ARG_Q_DATA ? (long long)wd.size() : (long long)ws.size();
        if (r.offset < 0 || end > size || (r.arg == NVFP4A_ARG_Q_SRC) == (r.write != 0)) { say("   item %d: bad range of argument %d\n", i, r.arg); ++bad; continue; }
        if (!w) continue;
        for (int row = 0; row < r.rows; ++row)
          for (long long b = 0; b < r.bytes; ++b) {
            char &f = (*w)[r.offset + row * r.stride + b];
            if (f) { say("   item %d: a byte of argument %d declared twice\n", i, r.arg); ++bad; }
            f = 1;
          }
      }
    }
    for (int r = 0; r < rows && !bad; ++r)
      for (int c = 0; c < ldd && !bad; ++c) {
        const size_t at = (size_t)r * ldd + c;
        const bool computed = r < active && c < cols;
        if (computed ? d_got[at] != d_ref[at] : d_got[at] != 0xA5) { say("   data[%d][%d] differs\n", r, c); ++bad; }
        if (d_got[at] != 0xA5 && !wd[at]) { say("   data[%d][%d] written outside the declared ranges\n", r, c); ++bad; }
      }
    for (int r = 0; r < rows && !bad; ++r)
      for (int c = 0; c < lsc && !bad; ++c) {
        const size_t at = (size_t)r * lsc + c;
        const bool computed = r < active && c < cols / 16;
        if (computed ? memcmp(&s_got[at], &s_ref[at], 4) : s_got[at] != -1.f) { say("   scale[%d][%d] differs\n", r, c); ++bad; }
      }
  }
  say("quantize_int8 %s rows %d cols %d%s: %d items: %s\n", DN[dt], rows, cols, nrows ? (" computed rows " + std::to_string(nrows)).c_str() : "", st ? 0 : it.items,
      st ? nvfp4a_last_error() : bad ? "FAIL" : "ok");
  for (void *p : {(void *)dsrc, (void *)dref, (void *)dgot, (void *)dsref, (void *)dsgot, (void *)dn}) cudaFree(p);
  return bad || st ? 1 : 0;
}

// The median time of f in ms
double time_ms(const std::function<void()> &f, int reps) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  std::vector<float> t(reps);
  f();
  check_cuda(cudaDeviceSynchronize(), "bench warm-up");
  for (float &x : t) {
    cudaEventRecord(a);
    f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&x, a, b);
  }
  std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
  return t[reps / 2];
}

// One launch of an item description over the device: as many CTAs as it keeps resident, each taking every gridDim-th item.
std::function<void()> launcher(const nvfp4a_items &it) {
  void *dargs;
  check_cuda(cudaMalloc(&dargs, it.args_bytes), "cudaMalloc arguments");
  check_cuda(cudaMemcpy(dargs, it.args.bytes, it.args_bytes, cudaMemcpyHostToDevice), "cudaMemcpy arguments");
  CUfunction fn;
  executor(it, &fn, false, false);
  if (it.smem_bytes > 48 * 1024)
    check_driver(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, it.smem_bytes), "cuFuncSetAttribute");
  int blocks, sms, dev;
  check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, it.threads, it.smem_bytes), "occupancy");
  check_cuda(cudaGetDevice(&dev), "device");
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SMs");
  const unsigned grid = std::min(it.items, std::max(1, blocks) * sms);
  const int items = it.items, smem = it.smem_bytes, threads = it.threads;
  return [=] {
    int count = 0;
    PublishCopy *copies = nullptr;
    unsigned *flags = nullptr;
    const void *a = dargs;
    int n = items;
    void *params[] = {(void *)&a, &n, &copies, &count, &flags};
    check_driver(cuLaunchKernel(fn, grid, 1, 1, threads, 1, 1, smem, 0, params, nullptr), "cuLaunchKernel executor");
  };
}

// The INT8 quantization of a chunk's activations as items against the library's own kernel (and the bytes it moves per second).
void bench_quantize() {
  const int rows = 2048, cols = 4096;
  std::vector<uint8_t> h((size_t)rows * cols * 2);
  fill(NVFP4A_BF16, h, 2.f);
  void *x, *q, *sc;
  check_cuda(cudaMalloc(&x, h.size()), "x");
  check_cuda(cudaMemcpy(x, h.data(), h.size(), cudaMemcpyHostToDevice), "x");
  check_cuda(cudaMalloc(&q, (size_t)rows * cols), "q");
  check_cuda(cudaMalloc(&sc, (size_t)rows * cols / 4), "scales");
  const nvfp4a_tensor src = {x, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0}, dst = {q, NVFP4A_INT8, sc, 0, nullptr, nullptr, 0, 0};
  const double stream_ms = time_ms([&] { nvfp4a_quantize(&src, cols, &dst, cols, rows, cols, 0); }, 50);
  nvfp4a_items it;
  if (nvfp4a_quantize_int8_items(&src, cols, &dst, cols, rows, cols, &it)) exit(1);
  const double ms = time_ms(launcher(it), 50), bytes = (double)rows * cols * (2 + 1 + 0.25);
  printf("INT8 quantization rows %d cols %d: the library's kernel %.3f ms, as items %.3f ms (%.0f GB/s moved)\n", rows, cols, stream_ms, ms,
         bytes / (ms * 1e-3) * 1e-9);
}

// ./test_items --bench: the quantization above; the B image (BF16 and INT8) as items against the library's own transcoder, and a
// tensor-core GEMM whose columns are in device memory against the whole -- n = 2048 columns (a prefill chunk), the Qwen3-8B qkv shape.
void bench_items() {
  const auto &device = nvfp4a::kern::device_info();
  bench_quantize();
  if (!device.tc || device.ptx < 80) { printf("bench: the tensor-core cases need sm_80 .. sm_89\n"); return; }
  const int m = 6144, n = 2048, k = 4096, ld = k;
  std::vector<uint8_t> hB((size_t)ld * n * 2);
  fill(NVFP4A_BF16, hB, 2.f);
  void *dB = todev(hB);
  HostFp4 hA(m, k, ld, k / 16, 0.375f, 0);
  uint8_t *dA = todev(hA.data), *dSA = todev(hA.sc);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, 0, nullptr, nullptr, 0, 0}, B = {dB, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
  void *ia, *ib, *dD;
  check_cuda(cudaMalloc(&ia, nvfp4a_tc_image_bytes(NVFP4A_TC_A16, NVFP4A_NVFP4, m, k) + 16), "A image");
  check_cuda(cudaMalloc(&ib, nvfp4a_tc_image_bytes(NVFP4A_TC_B, NVFP4A_BF16, n, k) + 16), "B image");
  check_cuda(cudaMalloc(&dD, (size_t)m * n * 2), "D");
  if (nvfp4a_tc_image(NVFP4A_TC_A16, &A, ld, m, k, ia, 0)) exit(1);
  const nvfp4a_tensor Ai = {ia, NVFP4A_NVFP4, nullptr, 0, nullptr, nullptr, 0, 0}, Bi = {ib, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0},
                      D = {dD, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
  const double stream_ms = time_ms([&] { nvfp4a_tc_image(NVFP4A_TC_B, &B, ld, n, k, ib, 0); }, 50);
  nvfp4a_items img;
  if (nvfp4a_tc_image_b_items(n, k, &B, ld, ib, &img)) exit(1);
  const double image_ms = time_ms(launcher(img), 50), image_bytes = (double)n * k * 2 + nvfp4a_tc_image_bytes(NVFP4A_TC_B, NVFP4A_BF16, n, k);
  printf("B image n %d k %d: the library's transcoder %.3f ms, as items %.3f ms (%.0f GB/s moved)\n", n, k, stream_ms, image_ms,
         image_bytes / (image_ms * 1e-3) * 1e-9);
  nvfp4a_items gemm;
  if (nvfp4a_gemm_tc_ex_items(m, n, k, 1.f, &Ai, &Bi, 0.f, nullptr, 0, &D, m, nullptr, nullptr, &gemm)) exit(1);
  const double flops = 2.0 * m * n * k;
  const double full_ms = time_ms(launcher(gemm), 20);
  printf("GEMM_TC m %d n %d k %d: all %d columns %.3f ms (%.1f TFLOPS)\n", m, n, k, n, full_ms, flops / (full_ms * 1e-3) * 1e-12);
  for (int cols : {1024, 256, 129}) {
    int *dn;
    check_cuda(cudaMalloc(&dn, sizeof(int)), "ncols");
    check_cuda(cudaMemcpy(dn, &cols, sizeof(int), cudaMemcpyHostToDevice), "ncols");
    nvfp4a_items part = gemm;
    if (nvfp4a_items_n_device(&part, dn)) exit(1);
    const double ms = time_ms(launcher(part), 20);
    printf("  %d columns in device memory: %.3f ms = %.1f%% of the whole, %.1f%% of the columns' share\n", cols, ms, 100 * ms / full_ms,
           100 * ms / (full_ms * cols / n));
  }

  // W4A8: the INT8 B image, then the GEMM on A's A8 image
  HostQ8 h8(n, k, ld, k / 16, 1.f);
  uint8_t *d8 = todev(std::vector<uint8_t>(h8.data.begin(), h8.data.end()));
  float *dsc = todev(h8.sc);
  const nvfp4a_tensor B8 = {d8, NVFP4A_INT8, dsc, k / 16, nullptr, nullptr, 0, 0};
  void *ia8, *ib8;
  check_cuda(cudaMalloc(&ia8, nvfp4a_tc_image_bytes(NVFP4A_TC_A8, NVFP4A_NVFP4, m, k) + 16), "A8 image");
  check_cuda(cudaMalloc(&ib8, nvfp4a_tc_image_bytes(NVFP4A_TC_B, NVFP4A_INT8, n, k) + 16), "B8 image");
  if (nvfp4a_tc_image(NVFP4A_TC_A8, &A, ld, m, k, ia8, 0)) exit(1);
  const double stream8_ms = time_ms([&] { nvfp4a_tc_image(NVFP4A_TC_B, &B8, ld, n, k, ib8, 0); }, 50);
  nvfp4a_items img8;
  if (nvfp4a_tc_image_b_items(n, k, &B8, ld, ib8, &img8)) exit(1);
  const double image8_ms = time_ms(launcher(img8), 50),
               image8_bytes = (double)n * k + (double)n * k / 4 + nvfp4a_tc_image_bytes(NVFP4A_TC_B, NVFP4A_INT8, n, k);
  printf("B image INT8 n %d k %d: the library's transcoder %.3f ms, as items %.3f ms (%.0f GB/s moved)\n", n, k, stream8_ms, image8_ms,
         image8_bytes / (image8_ms * 1e-3) * 1e-9);
  const nvfp4a_tensor Ai8 = {ia8, NVFP4A_NVFP4, nullptr, 0, nullptr, nullptr, 0, 0}, Bi8 = {ib8, NVFP4A_INT8, nullptr, 0, nullptr, nullptr, 0, 0};
  nvfp4a_items gemm8;
  if (nvfp4a_gemm_tc_ex_items(m, n, k, 1.f, &Ai8, &Bi8, 0.f, nullptr, 0, &D, m, nullptr, nullptr, &gemm8)) exit(1);
  const double full8_ms = time_ms(launcher(gemm8), 20);
  printf("GEMM_TC INT8 m %d n %d k %d: all %d columns %.3f ms (%.1f TOPS), %.2fx the BF16 B\n", m, n, k, n, full8_ms,
         flops / (full8_ms * 1e-3) * 1e-12, full_ms / full8_ms);
}

}  // namespace

// covers nvfp4a_gemm_ex (GEMV items), nvfp4a_gemv_tc_ex (slices), nvfp4a_gemm_tc_ex (tiles); every addition mask and the plain epilogue,
// B BF16 / INT8, D BF16 / FP32 (publication also FP16), bias, C.  Publication cases: two resident CTAs, a cache-primed consumer and
// a publisher, two epochs (docs/c-api.md, Work items)
int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--bench") == 0) {
    check_driver(cuInit(0), "cuInit");
    check_cuda(cudaFree(nullptr), "cuda init");
    bench_items();
    return 0;
  }
  const bool readonly_control = argc == 2 && strcmp(argv[1], "--readonly-control") == 0;
  const bool publication_only = readonly_control || (argc == 2 && strcmp(argv[1], "--coherence-only") == 0);
  if (argc != 1 && !publication_only) { fprintf(stderr, "usage: %s [--coherence-only|--readonly-control|--bench]\n", argv[0]); return 1; }
  check_driver(cuInit(0), "cuInit");
  const auto &device = nvfp4a::kern::device_info();
  int device_id, cooperative;
  check_cuda(cudaGetDevice(&device_id), "current device");
  check_cuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device_id), "cooperative launch support");
  if (!cooperative) {
    printf("SKIP publication cases: cooperative launch unsupported\n");
    if (publication_only) return 77;
  }
  std::vector<ItemCase> cases;
  int v = 0;
  const int shapes[][3] = {{64, 1, 64}, {1024, 1, 1024}, {528, 3, 576}, {6144, 8, 1024}, {2048, 5, 3072},
                           {528, 12, 576}, {512, 64, 1024}, {528, 129, 64}, {1024, 200, 512}, {1168, 137, 64},
                           {64, 1, 0}, {528, 12, 0}, {1168, 137, 0}};
  for (bool tc : {false, true})
    for (const auto &s : shapes)
      for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_INT8}) {
        if (publication_only) continue;
        if (tc && (!device.tc || (bt == NVFP4A_BF16 && device.ptx < 80))) continue;
        for (int mask = 0; mask <= 5; ++mask, ++v) {
          const bool glu = mask & 2;
          cases.push_back(ItemCase{s[0], s[1], s[2], bt, v % 2 ? NVFP4A_FP32 : NVFP4A_BF16, mask, !glu && v % 3 == 1,
                                   !glu && v % 4 == 2, tc, v % 3 == 0});
        }
      }
  const int counted[][4] = {{528, 12, 576, 9}, {528, 12, 576, 8}, {1024, 200, 512, 70}, {1168, 137, 64, 130}, {512, 64, 1024, 1}};
  for (bool tc : {false, true})
    for (const auto &s : counted)
      for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_INT8}) {
        if (publication_only) continue;
        if (tc && (!device.tc || (bt == NVFP4A_BF16 && device.ptx < 80))) continue;
        for (int mask = 0; mask <= 5; ++mask, ++v) {
          const bool glu = mask & 2;
          ItemCase x{s[0], s[1], s[2], bt, v % 2 ? NVFP4A_FP32 : NVFP4A_BF16, mask, !glu && v % 3 == 1, !glu && v % 4 == 2, tc, v % 3 == 0};
          x.ncols = s[3];
          cases.push_back(x);
        }
      }
  const int published[][3] ={{64, 1, 64}, {528, 3, 576}, {16, 12, 64}, {144, 64, 64}, {144, 129, 64}};
  for (bool tc : {false, true})
    for (const auto &s : published)
      for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_INT8}) {
        if (!cooperative) continue;
        if (tc && (!device.tc || (bt == NVFP4A_BF16 && device.ptx < 80))) continue;
        for (int mask = 0; mask <= 5; ++mask, ++v) {
          const bool glu = mask & 2;
          const nvfp4a_dtype dt[] = {NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32};
          cases.push_back(ItemCase{s[0], s[1], s[2], bt, dt[v % 3], mask, !glu && v % 3 == 1, !glu, tc, true, true, readonly_control});
        }
      }
  // the cases on one thread each (the compiles and the launches overlap); the data of a case does not depend on the schedule
  std::vector<std::string> reports(cases.size());
  std::vector<int> failed(cases.size(), 0);
  std::atomic<size_t> next{0};
  const auto worker = [&] {
    check_cuda(cudaFree(nullptr), "cuda init (thread)");
    for (size_t i; (i = next++) < cases.size();) {
      rng = 0x2545F491u + 0x9E3779B9u * static_cast<uint32_t>(i);
      sink = &reports[i];
      failed[i] = run(cases[i]);
      sink = nullptr;
    }
  };
  std::vector<std::thread> pool;
  const size_t threads = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), cases.size());
  for (size_t t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (std::thread &t : pool) t.join();
  int fails = 0;
  for (size_t i = 0; i < cases.size(); ++i) {
    fputs(reports[i].c_str(), stdout);
    fails += failed[i];
  }
  if (!publication_only && device.tc && device.ptx >= 80) {
    const int images[][3] = {{1, 64, 0}, {129, 576, 0}, {300, 1024, 0}, {1024, 512, 0}, {300, 128, 129}, {1024, 256, 129}, {260, 64, 1}, {512, 64, 256}};
    for (const auto &s : images) fails += run_image(s[0], s[1], s[2]);
  }
  if (!publication_only && device.tc && device.ptx >= 75) {
    const int images[][3] = {{1, 64, 0}, {129, 576, 0}, {300, 1024, 0}, {1024, 512, 0}, {300, 128, 129}, {1024, 192, 129}, {260, 64, 1}, {512, 64, 256}};
    for (const auto &s : images) fails += run_image_int8(s[0], s[1], s[2]);
  }
  if (!publication_only) {
    const int quants[][3] = {{1, 64, 0}, {5, 4096, 0}, {37, 1024, 0}, {300, 768, 0}, {64, 4096, 21}, {300, 1024, 129}, {9, 16, 1}};
    for (nvfp4a_dtype dt : {NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32})
      for (const auto &s : quants) fails += run_quant(dt, s[0], s[1], s[2]);
  }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
