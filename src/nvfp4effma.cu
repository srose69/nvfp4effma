// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA core (host side): nvfp4a_quantize / nvfp4a_dequantize, nvfp4a_status_string, errors and
// the choice of the kernel table for the current device.
#include <stdarg.h>
#include <stdio.h>
#include <type_traits>
#include "quant_nvfp4effma.cuh"

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
    d.ptx = loads ? fa.ptxVersion : 0;
    d.supported = loads && d.cc >= 50 && d.cc < 100;
    const char *dp4a = getenv("NVFP4A_DP4A");
    d.tc = loads && d.ptx >= 75 && d.ptx <= 89 && !(dp4a && dp4a[0] == '1');
    d.table = &kTables[loads && d.sass == d.cc ? table_index(d.cc) : 0];
  }
  return d;
}

// Global amax of src for the destinations (memset to 0 by the host before): per tensor (atomicMax into
// both pointers, which may alias) or per row (one warp per row).  Chunks of 8 values, 128-bit loads (rows
// start 16-byte aligned: 16-byte aligned data, lds and cols multiples of 16).
template <int T>
__device__ __forceinline__ float amax8(const uint8_t *p) {
  float x[8];
  load8<T>(p, x);
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(x[i]));
  return m;
}
template <int T, bool PER_ROW>
__global__ void amax_kernel(const uint8_t *__restrict__ src, int lds, int rows, int cols, float *a0, float *a1) {
  const int cpr = cols / 8;  // chunks per row
  if (PER_ROW) {
    const int lane = threadIdx.x & 31;
    for (int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32; r < rows; r += gridDim.x * blockDim.x / 32) {
      float m = 0.f;
      for (int c = lane; c < cpr; c += 32) m = fmaxf(m, amax8<T>(src + ((size_t)r * lds + 8 * c) * Elem<T>::bytes));
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
      if (lane == 0) { a0[r] = m; if (a1 && a1 != a0) a1[r] = m; }
    }
    return;
  }
  // two independent chunks per iteration (cpr is even: cols is a multiple of 16)
  float m = 0.f;
  const long long pairs = (long long)rows * cpr / 2;
  for (long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x; u < pairs; u += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(u / (cpr / 2)), c = 2 * (int)(u % (cpr / 2));
    const uint8_t *p = src + ((size_t)r * lds + 8 * c) * Elem<T>::bytes;
    m = fmaxf(m, fmaxf(amax8<T>(p), amax8<T>(p + 8 * Elem<T>::bytes)));
  }
  amax_commit(a0, m);
  if (a1 && a1 != a0) amax_commit(a1, m);
}

// INT8 destination: sb = amax_block / 127 (FP32), q = RNE(x / sb), one thread per block of 16.
template <int T>
__global__ void quantize_int8_kernel(const uint8_t *__restrict__ src, int lds, int8_t *__restrict__ dst, int ldd,
                                     float *__restrict__ scales, int ld_scale, int rows, int cols, float *amax) {
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
    const float sb = __fdiv_rn(bmax, 127.f);
    uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int q = sb > 0.f ? (int)fminf(fmaxf(rintf(__fdiv_rn(x[i], sb)), -127.f), 127.f) : 0;
      w[i / 4] |= (uint32_t)(uint8_t)(int8_t)q << (8 * (i % 4));
    }
    *(uint4 *)(dst + (size_t)r * ldd + 16 * c) = make_uint4(w[0], w[1], w[2], w[3]);
    scales[(size_t)r * ld_scale + c] = sb;
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
// (qcols: values per row of the NVFP4 side; cols of the dense side [rows][cols] otherwise)
static nvfp4a_status check_convert(const char *fn, const nvfp4a_tensor *q, int ldq, const nvfp4a_tensor *x, int ldx,
                                   int rows, int cols, int qcols = -1) {
  if (qcols < 0) qcols = cols;
  if (!q || !x || !q->data || !x->data) return fail(NVFP4A_INVALID_VALUE, "%s: src / dst or their data is NULL", fn);
  if (q->dtype != NVFP4A_NVFP4) return fail(NVFP4A_INVALID_VALUE, "%s: the NVFP4 side has dtype %d", fn, (int)q->dtype);
  if (x->dtype != NVFP4A_BF16 && x->dtype != NVFP4A_FP16 && x->dtype != NVFP4A_FP32)
    return fail(NVFP4A_INVALID_VALUE, "%s: the dense side must be BF16, FP16 or FP32 (dtype %d)", fn, (int)x->dtype);
  if (!q->block_scale) return fail(NVFP4A_INVALID_VALUE, "%s: block_scale is NULL", fn);
  if (rows < 0 || cols < 0) return fail(NVFP4A_INVALID_VALUE, "%s: negative size (rows=%d cols=%d)", fn, rows, cols);
  if (cols % 16 || ldq % 16 || ldx % 16)
    return fail(NVFP4A_INVALID_VALUE, "%s: cols = %d and the leading dimensions (%d, %d) must be multiples of 16", fn, cols,
                ldq, ldx);
  if (ldq < qcols || ldx < cols) return fail(NVFP4A_INVALID_VALUE, "%s: leading dimension < cols (%d, %d)", fn, qcols, cols);
  if (((uintptr_t)q->data & 7) || ((uintptr_t)x->data & 15))
    return fail(NVFP4A_INVALID_VALUE, "%s: NVFP4 data must be 8-byte, dense data 16-byte aligned", fn);
  return NVFP4A_OK;
}

// An NVFP4 tensor of [rows][cols] as the kernels see it (after check_scales), and its format (kQ_*).
static kern::QuantOut nvfp4_ref(const nvfp4a_tensor *t, int ld, int sld) {
  return {(uint8_t *)t->data, ld, (uint8_t *)t->block_scale, sld, t->global_scale, scale_max_of(t),
          (t->flags & NVFP4A_GLOBAL_AMAX) ? 1 : 0};
}
static int quant_format(const nvfp4a_tensor *t) {
  return (t->flags & NVFP4A_SCALES_SWIZZLED ? kern::kQ_SWZ : 0) | (t->flags & NVFP4A_SCALES_UE5M3 ? kern::kQ_UE : 0) |
         (t->flags & NVFP4A_GLOBAL_AMAX ? kern::kQ_AMAX : 0) | (t->flags & NVFP4A_GLOBAL_PER_ROW ? kern::kQ_ROW : 0);
}
// the quantization kernels' format: kQ_AMAX only with per-row globals (else QuantOut::amax)
static int quant_kernel_format(const nvfp4a_tensor *t) {
  const int qf = quant_format(t);
  return qf & kern::kQ_ROW ? qf : qf & ~kern::kQ_AMAX;
}
using QuantLaunch = cudaError_t (*)(int, int, const uint8_t *, int, int, int, const kern::QuantOut &, float *, cudaStream_t);
// the instance object's entry for dense dtype t, columnwise or not, tile rows 1 / 16
static QuantLaunch quant_entry(int t, bool col, int tile) {
  using namespace kern;
#define NVFP4A_Q(T) {{&quant_launch<T, false, 1>, &quant_launch<T, false, 16>}, {&quant_launch<T, true, 1>, &quant_launch<T, true, 16>}}
  static const QuantLaunch f[3][2][2] = {NVFP4A_Q(NVFP4A_BF16), NVFP4A_Q(NVFP4A_FP16), NVFP4A_Q(NVFP4A_FP32)};
#undef NVFP4A_Q
  return f[t == NVFP4A_BF16 ? 0 : t == NVFP4A_FP16 ? 1 : 2][col][tile == 16];
}

static int grid_for(long long threads) {
  const long long b = (threads + 255) / 256;
  return (int)(b < 16384 ? (b > 0 ? b : 1) : 16384);
}

template <class F> static void by_dense(int t, F f) {
  if (t == NVFP4A_BF16) f(std::integral_constant<int, NVFP4A_BF16>());
  else if (t == NVFP4A_FP16) f(std::integral_constant<int, NVFP4A_FP16>());
  else f(std::integral_constant<int, NVFP4A_FP32>());
}

}  // namespace nvfp4a

extern "C" const char *nvfp4a_last_error(void) { return nvfp4a::g_error; }

extern "C" nvfp4a_status nvfp4a_quantize_ex(const nvfp4a_tensor *src, int lds, int rows, int cols,
                                            const nvfp4a_tensor *rowwise, int ld_row, const nvfp4a_tensor *columnwise,
                                            int ld_col, const nvfp4a_quant_config *cfg, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  const char *fn = "nvfp4a_quantize_ex";
  const nvfp4a_quant_config c = cfg ? *cfg : nvfp4a_quant_config{1, NVFP4A_4OVER6_OFF, 0};
  if (!src || !src->data) return fail(NVFP4A_INVALID_VALUE, "%s: src or its data is NULL", fn);
  if (!rowwise && !columnwise) return fail(NVFP4A_INVALID_VALUE, "%s: no destination", fn);
  if (c.tile_rows != 1 && c.tile_rows != 16) return fail(NVFP4A_INVALID_VALUE, "%s: tile_rows = %d (1 or 16)", fn, c.tile_rows);
  if (c.four_over_six < NVFP4A_4OVER6_OFF || c.four_over_six > NVFP4A_4OVER6_MSE)
    return fail(NVFP4A_INVALID_VALUE, "%s: invalid four_over_six %d", fn, (int)c.four_over_six);
  const int rows16 = (rows + 15) / 16 * 16;
  int sld_row = 0, sld_col = 0;
  for (int col = 0; col < 2; ++col) {
    const nvfp4a_tensor *d = col ? columnwise : rowwise;
    if (!d) continue;
    const char *name = col ? "columnwise" : "rowwise";
    const int ld = col ? ld_col : ld_row, orows = col ? cols : rows, ocols = col ? rows16 : cols;
    // the dense side is src [rows][cols] in both cases; the NVFP4 side [orows][ocols]
    if (nvfp4a_status st = check_convert(fn, d, ld, src, lds, rows, cols, ocols)) return st;
    if (nvfp4a_status st = check_scales(fn, name, d, orows, ocols, col ? &sld_col : &sld_row)) return st;
    if (col && (d->flags & NVFP4A_GLOBAL_PER_ROW))
      return fail(NVFP4A_INVALID_VALUE, "%s: per-row globals are for the rowwise destination only (as in Transformer Engine)", fn);
    if ((d->flags & NVFP4A_GLOBAL_PER_ROW) && c.tile_rows != 1)
      return fail(NVFP4A_INVALID_VALUE, "%s: per-row globals take 1 x 16 blocks (as in Transformer Engine)", fn);
    if (c.four_over_six && (!(d->flags & NVFP4A_GLOBAL_AMAX) || (d->flags & NVFP4A_SCALES_UE5M3)))
      return fail(NVFP4A_INVALID_VALUE, "%s: 4over6 needs an amax global and E4M3 scales (%s)", fn, name);
    if (c.compute_amax && (!(d->flags & NVFP4A_GLOBAL_AMAX) || !d->global_scale))
      return fail(NVFP4A_INVALID_VALUE, "%s: compute_amax needs an amax global with storage (%s)", fn, name);
  }
  if (cols % 16) return fail(NVFP4A_INVALID_VALUE, "%s: cols = %d is not a multiple of 16", fn, cols);
  if ((long long)rows * cols == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;

  const uint8_t *s = (const uint8_t *)src->data;
  const int st_t = src->dtype;
  if (c.compute_amax) {  // TE current scaling: the global amax first
    float *ar = rowwise ? (float *)rowwise->global_scale : nullptr, *ac = columnwise ? (float *)columnwise->global_scale : nullptr;
    const bool row_per_row = rowwise && (rowwise->flags & NVFP4A_GLOBAL_PER_ROW);
    by_dense(st_t, [&](auto tc) {
      constexpr int T = decltype(tc)::value;
      if (row_per_row) {
        kern::amax_kernel<T, true><<<grid_for((long long)rows * 32), 256, 0, stream>>>(s, lds, rows, cols, ar, nullptr);
        ar = nullptr;
      }
      if (ar || ac) {
        float *p0 = ar ? ar : ac, *p1 = ar ? ac : nullptr;
        cudaMemsetAsync(p0, 0, 4, stream);
        if (p1 && p1 != p0) cudaMemsetAsync(p1, 0, 4, stream);
        kern::amax_kernel<T, false><<<grid_for((long long)rows * (cols / 16)), 256, 0, stream>>>(s, lds, rows, cols, p0, p1);
      }
    });
  }
  for (int col = 0; col < 2; ++col) {
    const nvfp4a_tensor *d = col ? columnwise : rowwise;
    if (!d) continue;
    const kern::QuantOut o = nvfp4_ref(d, col ? ld_col : ld_row, col ? sld_col : sld_row);
    const cudaError_t e = quant_entry(st_t, col, c.tile_rows)(c.four_over_six, quant_kernel_format(d), s, lds, rows, cols, o, d->amax, stream);
    if (e != cudaSuccess) return cuda_status(e, "nvfp4a_quantize_ex: kernel launch");
  }
  return cuda_status(cudaGetLastError(), "nvfp4a_quantize_ex: kernel launch");
}

extern "C" nvfp4a_status nvfp4a_quantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                                         int cols, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  if (dst && dst->dtype == NVFP4A_INT8) {  // W4A8 activations
    nvfp4a_tensor q = *dst; q.dtype = NVFP4A_NVFP4;  // same checks, then the INT8 alignment
    if (nvfp4a_status st = check_convert("nvfp4a_quantize", &q, ldd, src, lds, rows, cols)) return st;
    if (dst->flags) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_quantize: flags are for NVFP4 destinations");
    if (dst->ld_scale && dst->ld_scale < cols / 16)
      return fail(NVFP4A_INVALID_VALUE, "nvfp4a_quantize: ld_scale = %d < cols / 16 = %d", dst->ld_scale, cols / 16);
    if (((uintptr_t)dst->data & 15) || ((uintptr_t)dst->block_scale & 3))
      return fail(NVFP4A_INVALID_VALUE, "nvfp4a_quantize: INT8 data must be 16-byte, its scales 4-byte aligned");
    if ((long long)rows * cols == 0) return NVFP4A_OK;
    if (nvfp4a_status st = check_device()) return st;
    const long long blocks = (long long)rows * cols / 16;
    const int grid = (int)((blocks + 255) / 256 < 8192 ? (blocks + 255) / 256 : 8192);
    const int lsc = dst->ld_scale ? dst->ld_scale : cols / 16;
    const uint8_t *s = (const uint8_t *)src->data;
    int8_t *d = (int8_t *)dst->data;
    float *sc = (float *)dst->block_scale;
    by_dense(src->dtype, [&](auto tc) {
      kern::quantize_int8_kernel<decltype(tc)::value><<<grid, 256, 0, stream>>>(s, lds, d, ldd, sc, lsc, rows, cols, dst->amax);
    });
    return cuda_status(cudaGetLastError(), "nvfp4a_quantize: kernel launch");
  }
  const nvfp4a_status st = nvfp4a_quantize_ex(src, lds, rows, cols, dst, ldd, nullptr, 0, nullptr, stream);
  if (st != NVFP4A_OK) {  // report under this function's name
    char msg[512];
    snprintf(msg, sizeof msg, "%s", g_error);
    const char *p = strstr(msg, "nvfp4a_quantize_ex");
    if (p == msg) fail(st, "nvfp4a_quantize%s", msg + strlen("nvfp4a_quantize_ex"));
  }
  return st;
}

extern "C" nvfp4a_status nvfp4a_dequantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd,
                                           int rows, int cols, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  const char *fn = "nvfp4a_dequantize";
  if (nvfp4a_status st = check_convert(fn, src, lds, dst, ldd, rows, cols)) return st;
  int sld = 0;
  if (nvfp4a_status st = check_scales(fn, "src", src, rows, cols, &sld)) return st;
  if (dst->flags) return fail(NVFP4A_INVALID_VALUE, "%s: flags are for NVFP4 tensors (dst)", fn);
  if ((long long)rows * cols == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  const kern::QuantOut q = nvfp4_ref(src, lds, sld);
  uint8_t *d = (uint8_t *)dst->data;
  const int qf = quant_format(src);
  cudaError_t e = cudaSuccess;
  by_dense(dst->dtype, [&](auto tc) { e = kern::dequant_launch<decltype(tc)::value>(qf, q, d, ldd, rows, cols, dst->amax, stream); });
  return cuda_status(e, "nvfp4a_dequantize: kernel launch");
}

namespace nvfp4a {
namespace kern {

// one thread per scale of the padded matrix [roundup(rows, 128)][sc]
__global__ void swizzle_kernel(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int ld, int rows, int cols16,
                               int sc, bool to_swz) {
  const ScaleIdx si{-sc};
  const long long n = (long long)(rows + 127) / 128 * 128 * sc;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const int r = (int)(i / sc), c = (int)(i % sc);
    const bool in = r < rows && c < cols16;
    if (to_swz) dst[si.at(r, c)] = in ? src[(size_t)r * ld + c] : 0;
    else if (in) dst[(size_t)r * ld + c] = src[si.at(r, c)];
  }
}

// one thread per 16 x 16 tile: 16 rows of 8 bytes in, 16 rows of 8 bytes out (nibble transpose), and the
// tile's scale (taken from its first row) to the 16 rows of the transposed tile
__global__ void transpose_kernel(QuantOut s, QuantOut d, int rows, int cols) {
  const int tr = rows / 16, tcn = cols / 16;
  for (long long u = (long long)blockIdx.x * blockDim.x + threadIdx.x; u < (long long)tr * tcn; u += (long long)gridDim.x * blockDim.x) {
    const int t = (int)(u / tcn), c = (int)(u % tcn);
    uint2 in[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) in[i] = *(const uint2 *)(s.data + (size_t)(16 * t + i) * (s.ld / 2) + 8 * c);
    const uint8_t sc = s.sc[ScaleIdx{s.sld}.at(16 * t, c)];
#pragma unroll
    for (int j = 0; j < 16; ++j) {  // output row 16c + j = input column 16c + j
      uint32_t w[2] = {0, 0};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t nib = ((j < 8 ? in[i].x : in[i].y) >> (4 * (j % 8))) & 15u;
        w[i / 8] |= nib << (4 * (i % 8));
      }
      *(uint2 *)(d.data + (size_t)(16 * c + j) * (d.ld / 2) + 8 * t) = make_uint2(w[0], w[1]);
      d.sc[ScaleIdx{d.sld}.at(16 * c + j, t)] = sc;
    }
  }
}

}  // namespace kern
}  // namespace nvfp4a

extern "C" nvfp4a_status nvfp4a_swizzle_scales(const void *src, void *dst, int ld_linear, int rows, int cols,
                                               int to_swizzled, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  if (!src || !dst) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_swizzle_scales: src or dst is NULL");
  if (rows < 0 || cols < 0 || cols % 16) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_swizzle_scales: rows = %d, cols = %d (a multiple of 16)", rows, cols);
  if (ld_linear < cols / 16) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_swizzle_scales: ld_linear = %d < cols / 16 = %d", ld_linear, cols / 16);
  if ((long long)rows * cols == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  const int sc = (cols / 16 + 3) / 4 * 4;
  kern::swizzle_kernel<<<grid_for((long long)(rows + 127) / 128 * 128 * sc), 256, 0, stream>>>(
      (const uint8_t *)src, (uint8_t *)dst, ld_linear, rows, cols / 16, sc, to_swizzled != 0);
  return cuda_status(cudaGetLastError(), "nvfp4a_swizzle_scales: kernel launch");
}

extern "C" nvfp4a_status nvfp4a_transpose(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                                          int cols, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  const char *fn = "nvfp4a_transpose";
  if (!src || !dst || !src->data || !dst->data || !src->block_scale || !dst->block_scale)
    return fail(NVFP4A_INVALID_VALUE, "%s: src / dst, their data or their scales are NULL", fn);
  if (src->dtype != NVFP4A_NVFP4 || dst->dtype != NVFP4A_NVFP4) return fail(NVFP4A_INVALID_VALUE, "%s: src and dst must be NVFP4", fn);
  if (rows < 0 || cols < 0 || rows % 16 || cols % 16)
    return fail(NVFP4A_INVALID_VALUE, "%s: rows = %d and cols = %d must be multiples of 16", fn, rows, cols);
  if (lds % 16 || ldd % 16 || lds < cols || ldd < rows)
    return fail(NVFP4A_INVALID_VALUE, "%s: lds = %d (>= cols) / ldd = %d (>= rows) must be multiples of 16", fn, lds, ldd);
  if (((uintptr_t)src->data & 7) || ((uintptr_t)dst->data & 7)) return fail(NVFP4A_INVALID_VALUE, "%s: data must be 8-byte aligned", fn);
  if ((src->flags ^ dst->flags) & NVFP4A_SCALES_UE5M3) return fail(NVFP4A_INVALID_VALUE, "%s: src and dst scale types differ", fn);
  int sls = 0, sld = 0;
  if (nvfp4a_status st = check_scales(fn, "src", src, rows, cols, &sls)) return st;
  if (nvfp4a_status st = check_scales(fn, "dst", dst, cols, rows, &sld)) return st;
  if ((src->flags | dst->flags) & NVFP4A_GLOBAL_PER_ROW) return fail(NVFP4A_INVALID_VALUE, "%s: per-row globals do not transpose", fn);
  if ((long long)rows * cols == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  kern::transpose_kernel<<<grid_for((long long)rows / 16 * (cols / 16)), 256, 0, stream>>>(nvfp4_ref(src, lds, sls),
                                                                                           nvfp4_ref(dst, ldd, sld), rows, cols);
  return cuda_status(cudaGetLastError(), "nvfp4a_transpose: kernel launch");
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
