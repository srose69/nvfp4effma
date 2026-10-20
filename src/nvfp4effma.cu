// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA core (host side): nvfp4a_quantize / nvfp4a_dequantize, nvfp4a_status_string, errors and
// the choice of the kernel table for the current device.
#include <stdarg.h>
#include <stdio.h>
#include "nvfp4effma.cuh"

namespace nvfp4a {

static thread_local char g_error[512];

void clear_error() { g_error[0] = 0; }
nvfp4a_status fail(nvfp4a_status s, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_error, sizeof g_error, fmt, ap);
  va_end(ap);
  return s;
}
nvfp4a_status cuda_status(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return NVFP4A_OK;
  return fail(NVFP4A_CUDA_ERROR, "%s: CUDA error %d (%s): %s", what, (int)e, cudaGetErrorName(e), cudaGetErrorString(e));
}

namespace kern {

// Compiled with the same architectures as the kernels: tells which SASS the device actually runs.
__global__ void probe_kernel() {}

const DeviceInfo &device_info() {
  static DeviceInfo cache[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  DeviceInfo &d = cache[dev & 63];
  if (!d.table) {
    int ma = 0, mi = 0;
    cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    d.cc = ma * 10 + mi;
    // A tuned table only applies when the device runs SASS built for exactly its compute capability:
    // other architectures carry kernel bodies for their own table and the defaults only.
    cudaFuncAttributes fa;
    const bool loads = cudaFuncGetAttributes(&fa, probe_kernel) == cudaSuccess;
    if (!loads) cudaGetLastError();  // "no kernel image" is reported by check_device, not left sticky
    d.sass = loads ? fa.binaryVersion : 0;
    d.supported = loads && d.cc >= 50 && d.cc < 100;
    d.table = &kTables[loads && d.sass == d.cc ? table_index(d.cc) : 0];
  }
  return d;
}

// One thread per block of 16 values of a row: 16 dense loads, block amax, E4M3 scale, 16 nibbles.
template <int T>
__global__ void quantize_kernel(const uint8_t *__restrict__ src, int lds, uint8_t *__restrict__ dst, int ldd,
                                uint8_t *__restrict__ scales, int ld_scale, const float *g_p, int rows, int cols,
                                float *amax) {
  const float g = scale_or_one(g_p), six_g = __fmul_rn(6.f, g);
  const int bpr = cols / 16;
  float mx = 0.f;
  for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < (long long)rows * bpr;
       b += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(b / bpr), c = (int)(b % bpr);
    float x[16];
    const uint8_t *p = src + ((size_t)r * lds + 16 * c) * Elem<T>::bytes;
    load8<T>(p, x); load8<T>(p + 8 * Elem<T>::bytes, x + 8);
    float bmax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(x[i]));
    mx = fmaxf(mx, bmax);
    const uint32_t sc = f32_to_e4m3(__fdiv_rn(bmax, six_g));
    const float d = __fmul_rn(e4m3_value(sc), g);
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t q = sc ? f32_to_e2m1(__fdiv_rn(x[i], d)) : (__float_as_uint(x[i]) >> 28) & 8u;
      w[i / 8] |= q << (4 * (i % 8));
    }
    *(uint2 *)(dst + (size_t)r * (ldd / 2) + 8 * c) = make_uint2(w[0], w[1]);
    scales[(size_t)r * ld_scale + c] = (uint8_t)sc;
  }
  if (amax) amax_commit(amax, mx);
}

// One thread per block: 8 bytes of nibbles and the scale -> 16 values (e2m1 * sb) * g.
template <int T>
__global__ void dequantize_kernel(const uint8_t *__restrict__ src, int lds, const uint8_t *__restrict__ scales,
                                  int ld_scale, const float *g_p, uint8_t *__restrict__ dst, int ldd, int rows,
                                  int cols, float *amax) {
  const float g = scale_or_one(g_p);
  const int bpr = cols / 16;
  float mx = 0.f;
  for (long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x; b < (long long)rows * bpr;
       b += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(b / bpr), c = (int)(b % bpr);
    const uint2 w = *(const uint2 *)(src + (size_t)r * (lds / 2) + 8 * c);
    const float sb = e4m3_value(scales[(size_t)r * ld_scale + c]);
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = __fmul_rn(__fmul_rn(e2m1_value<0>(i < 8 ? w.x : w.y, i % 8), sb), g);
      mx = fmaxf(mx, fabsf(v[i]));
    }
    uint8_t *p = dst + ((size_t)r * ldd + 16 * c) * Elem<T>::bytes;
    if (T == NVFP4A_FP32) {
#pragma unroll
      for (int q = 0; q < 4; ++q) ((float4 *)p)[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    } else {
      uint32_t h[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        if (T == NVFP4A_BF16) h[q] = __byte_perm(rne_hi(__float_as_uint(v[2 * q])), rne_hi(__float_as_uint(v[2 * q + 1])), 0x7632);
        else { const __half2 t = __floats2half2_rn(v[2 * q], v[2 * q + 1]); h[q] = *reinterpret_cast<const uint32_t *>(&t); }
      }
      ((uint4 *)p)[0] = make_uint4(h[0], h[1], h[2], h[3]);
      ((uint4 *)p)[1] = make_uint4(h[4], h[5], h[6], h[7]);
    }
  }
  if (amax) amax_commit(amax, mx);
}

}  // namespace kern

nvfp4a_status check_device() {
  int dev = -1;
  const cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return cuda_status(e, "cudaGetDevice");
  const kern::DeviceInfo &d = kern::device_info();
  if (d.supported) return NVFP4A_OK;
  if (d.cc >= 100)
    return fail(NVFP4A_NOT_SUPPORTED, "device %d is sm_%d, which has native FP4: use cuBLASLt / Transformer Engine "
                "(NVFP4EffMA targets sm_50..sm_90)", dev, d.cc);
  if (d.cc < 50) return fail(NVFP4A_NOT_SUPPORTED, "device %d is sm_%d; NVFP4EffMA needs sm_50 or newer", dev, d.cc);
  return fail(NVFP4A_NOT_SUPPORTED, "device %d is sm_%d, but this build of the library has no code for it: "
              "rebuild with ARCHS including %d", dev, d.cc, d.cc);
}

// Checks shared by nvfp4a_quantize / nvfp4a_dequantize: q is the NVFP4 side, x the dense one.
static nvfp4a_status check_convert(const char *fn, const nvfp4a_tensor *q, int ldq, const nvfp4a_tensor *x, int ldx,
                                   int rows, int cols) {
  if (!q || !x || !q->data || !x->data) return fail(NVFP4A_INVALID_VALUE, "%s: src / dst or their data is NULL", fn);
  if (q->dtype != NVFP4A_NVFP4) return fail(NVFP4A_INVALID_VALUE, "%s: the NVFP4 side has dtype %d", fn, (int)q->dtype);
  if (x->dtype != NVFP4A_BF16 && x->dtype != NVFP4A_FP16 && x->dtype != NVFP4A_FP32)
    return fail(NVFP4A_INVALID_VALUE, "%s: the dense side must be BF16, FP16 or FP32 (dtype %d)", fn, (int)x->dtype);
  if (!q->block_scale) return fail(NVFP4A_INVALID_VALUE, "%s: block_scale is NULL", fn);
  if (rows < 0 || cols < 0) return fail(NVFP4A_INVALID_VALUE, "%s: negative size (rows=%d cols=%d)", fn, rows, cols);
  if (cols % 16 || ldq % 16 || ldx % 16)
    return fail(NVFP4A_INVALID_VALUE, "%s: cols = %d and the leading dimensions (%d, %d) must be multiples of 16", fn, cols,
                ldq, ldx);
  if (ldq < cols || ldx < cols) return fail(NVFP4A_INVALID_VALUE, "%s: leading dimension < cols = %d", fn, cols);
  if (q->ld_scale && q->ld_scale < cols / 16)
    return fail(NVFP4A_INVALID_VALUE, "%s: ld_scale = %d < cols / 16 = %d", fn, q->ld_scale, cols / 16);
  if (((uintptr_t)q->data & 7) || ((uintptr_t)x->data & 15))
    return fail(NVFP4A_INVALID_VALUE, "%s: NVFP4 data must be 8-byte, dense data 16-byte aligned", fn);
  return NVFP4A_OK;
}

}  // namespace nvfp4a

extern "C" const char *nvfp4a_last_error(void) { return nvfp4a::g_error; }

extern "C" nvfp4a_status nvfp4a_quantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                                         int cols, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  if (nvfp4a_status st = check_convert("nvfp4a_quantize", dst, ldd, src, lds, rows, cols)) return st;
  if ((long long)rows * cols == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  const long long blocks = (long long)rows * cols / 16;
  const int grid = (int)((blocks + 255) / 256 < 8192 ? (blocks + 255) / 256 : 8192);
  const int lsc = dst->ld_scale ? dst->ld_scale : cols / 16;
  const uint8_t *s = (const uint8_t *)src->data;
  uint8_t *d = (uint8_t *)dst->data, *sc = (uint8_t *)dst->block_scale;
#define NVFP4A_Q(T) kern::quantize_kernel<T><<<grid, 256, 0, stream>>>(s, lds, d, ldd, sc, lsc, dst->global_scale, rows, cols, dst->amax)
  if (src->dtype == NVFP4A_BF16) NVFP4A_Q(NVFP4A_BF16);
  else if (src->dtype == NVFP4A_FP16) NVFP4A_Q(NVFP4A_FP16);
  else NVFP4A_Q(NVFP4A_FP32);
#undef NVFP4A_Q
  return cuda_status(cudaGetLastError(), "nvfp4a_quantize: kernel launch");
}

extern "C" nvfp4a_status nvfp4a_dequantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd,
                                           int rows, int cols, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  if (nvfp4a_status st = check_convert("nvfp4a_dequantize", src, lds, dst, ldd, rows, cols)) return st;
  if ((long long)rows * cols == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  const long long blocks = (long long)rows * cols / 16;
  const int grid = (int)((blocks + 255) / 256 < 8192 ? (blocks + 255) / 256 : 8192);
  const int lsc = src->ld_scale ? src->ld_scale : cols / 16;
  const uint8_t *s = (const uint8_t *)src->data, *sc = (const uint8_t *)src->block_scale;
  uint8_t *d = (uint8_t *)dst->data;
#define NVFP4A_D(T) kern::dequantize_kernel<T><<<grid, 256, 0, stream>>>(s, lds, sc, lsc, src->global_scale, d, ldd, rows, cols, dst->amax)
  if (dst->dtype == NVFP4A_BF16) NVFP4A_D(NVFP4A_BF16);
  else if (dst->dtype == NVFP4A_FP16) NVFP4A_D(NVFP4A_FP16);
  else NVFP4A_D(NVFP4A_FP32);
#undef NVFP4A_D
  return cuda_status(cudaGetLastError(), "nvfp4a_dequantize: kernel launch");
}

extern "C" const char *nvfp4a_status_string(nvfp4a_status s) {
  switch (s) {
    case NVFP4A_OK: return "NVFP4A_OK";
    case NVFP4A_INVALID_VALUE: return "NVFP4A_INVALID_VALUE";
    case NVFP4A_NOT_SUPPORTED: return "NVFP4A_NOT_SUPPORTED";
    case NVFP4A_CUDA_ERROR: return "NVFP4A_CUDA_ERROR";
  }
  return "unknown nvfp4a_status";
}
