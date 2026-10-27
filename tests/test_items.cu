// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The work items (include/nvfp4effma_items.h) as an executor runs them: a kernel compiled with NVRTC from the described
// entry and argument type, persistent over the items (fewer CTAs than items, the shared memory reused), its outputs against
// the call's bit for bit (tiles: the library kernel without stream-K, the same summation order); every byte the items
// change lies in their declared write ranges, every declared read range in its argument.
// nvfp4a_gemm_ex (GEMV items), nvfp4a_gemv_tc_ex (slices) and nvfp4a_gemm_tc_ex (tiles; tensor cores); every
// addition mask and the plain epilogue, B BF16 and INT8, D BF16 and FP32 (publication also FP16), bias and C.
// Publication cases: two resident CTAs, a cache-primed consumer and a publisher, two epochs (docs/c-api.md, Work items).
#include <string>
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

CUmodule executor(const nvfp4a_items &it, CUfunction *fn, bool publish, bool readonly_control) {
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

struct ItemCase {
  int m, n, k;
  nvfp4a_dtype bt, dt;
  int mask, bias, c;  // mask: 1 column scale, 2 SwiGLU, 4 second output; 0 the plain epilogue
  bool tc;
  bool row_global = false;
  bool publish = false;
  bool readonly_control = false;
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
    check_driver(cuModuleUnload(module), "cuModuleUnload");
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
    if (g0 != g1 || a0 != a1 || memcmp(s0.data(), s1.data(), sqn * 4)) {
      printf("   outputs differ bit for bit from the %s\n", gemm_tc ? "whole-tile library kernel" : "call");
      for (size_t i = 0; i < g0.size() / esize(c.dt); ++i)
        if (memcmp(g0.data() + i * esize(c.dt), g1.data() + i * esize(c.dt), esize(c.dt))) {
          printf("   D[%zu]: got %.9g, reference %.9g\n", i, hload(c.dt, g1.data(), i), hload(c.dt, g0.data(), i));
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
      printf("   invalid access query accepted\n"); ++bad;
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
        printf("   inconsistent or overflowing access query\n"); ++bad; break;
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
          printf("   item %d: invalid range of argument %d (%lld > %lld)\n", i, r.arg, end, size); ++bad;
        }
        if (!r.write || !w || bad) continue;
        for (int row = 0; row < r.rows; ++row)
          for (long long b = 0; b < r.bytes; ++b) {
            char &f = (*w)[r.offset + row * r.stride + b];
            if (f) { printf("   item %d writes byte %lld of argument %d, another item too\n", i, r.offset + row * r.stride + b, r.arg); ++bad; }
            f = 1;
          }
      }
      const bool row_tail = first_row > 0 && first_row + it.rows > c.m;
      if ((it.kind == NVFP4A_ITEM_GEMV && c.k && row_tail && (!zero_a || !zero_scales)) ||
          (gemm_tc && c.row_global && row_tail && !zero_global) ||
          (gemm_tc && (c.mask & 1) && first_col > 0 && first_col + it.cols > c.n && !zero_norm)) {
        printf("   item %d omits a boundary fallback read\n", i); ++bad;
      }
    }
    const auto coverage = [&](const std::vector<char> &w, int rows, int ld, int bytes, const char *what) {
      for (size_t b = 0; b < w.size() && !bad; ++b)
        if ((w[b] != 0) != (b % ((size_t)ld * bytes) < (size_t)rows * bytes)) {
          printf("   %s byte %zu: incorrect declared write coverage\n", what, b); ++bad;
        }
    };
    coverage(wd, mo, ldd, (int)esize(c.dt), "D");
    coverage(wd2, out2 ? c.m : 0, ldd2, 2, "d2");
    coverage(wsq, out2 ? c.m / 16 : 0, lsq, 4, "sq");
    const auto ch = [&](const std::vector<uint8_t> &now, const std::vector<uint8_t> &was, const std::vector<char> &w, const char *what) {
      for (size_t b = 0; b < now.size() && !bad; ++b)
        if (now[b] != was[b] && !w[b]) { printf("   %s byte %zu changed outside the write ranges\n", what, b); ++bad; }
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
  printf("%-8s m %5d n %d k %5d B %-4s D %-4s ex %d%s%s: %d items x %d threads, %d B shared, %d regs: %s\n",
         c.publish ? (c.tc ? "tc pub" : "cuda pub") : (c.tc ? "tc" : "cuda"), c.m, c.n,
         c.k, DN[c.bt], DN[c.dt], c.mask, c.bias ? " bias" : "", c.c ? " +=" : "", st ? 0 : it.items, st ? 0 : it.threads,
         st ? 0 : it.smem_bytes, st ? 0 : it.regs, st ? nvfp4a_last_error() : bad ? "FAIL" : "ok");
  for (void *p : {(void *)dA, (void *)dSA, (void *)dbias, (void *)dw2, dB, (void *)dSB, (void *)dparts, (void *)dgb,
                  (void *)dga, ia, ib, (void *)zero_ws})
    check_cuda(cudaFree(p), "cudaFree input");
  for (Out &x : o) { cudaFree(x.d); cudaFree(x.d2); cudaFree(x.sq); }
  return bad || st ? 1 : 0;
}

}  // namespace

int main(int argc, char **argv) {
  const bool readonly_control = argc == 2 && strcmp(argv[1], "--readonly-control") == 0;
  const bool publication_only = readonly_control || (argc == 2 && strcmp(argv[1], "--coherence-only") == 0);
  if (argc != 1 && !publication_only) { fprintf(stderr, "usage: %s [--coherence-only|--readonly-control]\n", argv[0]); return 1; }
  check_driver(cuInit(0), "cuInit");
  const auto &device = nvfp4a::kern::device_info();
  int device_id, cooperative;
  check_cuda(cudaGetDevice(&device_id), "current device");
  check_cuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device_id), "cooperative launch support");
  if (!cooperative) {
    printf("SKIP publication cases: cooperative launch unsupported\n");
    if (publication_only) return 77;
  }
  int fails = 0, v = 0;
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
          fails += run(ItemCase{s[0], s[1], s[2], bt, v % 2 ? NVFP4A_FP32 : NVFP4A_BF16, mask, !glu && v % 3 == 1,
                               !glu && v % 4 == 2, tc, v % 3 == 0});
        }
      }
  const int published[][3] = {{64, 1, 64}, {528, 3, 576}, {16, 12, 64}, {144, 64, 64}, {144, 129, 64}};
  for (bool tc : {false, true})
    for (const auto &s : published)
      for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_INT8}) {
        if (!cooperative) continue;
        if (tc && (!device.tc || (bt == NVFP4A_BF16 && device.ptx < 80))) continue;
        for (int mask = 0; mask <= 5; ++mask, ++v) {
          const bool glu = mask & 2;
          const nvfp4a_dtype dt[] = {NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32};
          fails += run(ItemCase{s[0], s[1], s[2], bt, dt[v % 3], mask, !glu && v % 3 == 1, !glu, tc, true, true, readonly_control});
        }
      }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
