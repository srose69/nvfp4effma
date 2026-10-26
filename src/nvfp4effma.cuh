// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA core (device side): exact E2M1 / E4M3 decode, E2M1 / E4M3 / BF16 rounding, typed element
// access, the GEMM / GEMV epilogue and the per-architecture kernel tables.  Shared by gemm_*, gemv_*
// and the quantization kernels.
//
//   arithmetic : E2M1 x 2 is an exact int8 (0, +-1, 2, 3, 4, 6, 8, 12): NVFP4 x NVFP4 block sums are exact
//                integers (DP4A from sm_61, FFMA on integer-valued floats below); an E2M1 value times its
//                E4M3 block scale has at most 6 significant bits, so products with BF16 / FP16 values
//                are exact in FP32
//   rounding   : accumulate in FP32 over the whole reduction, round once when storing
//   contract   : block scales are finite (no E4M3 NaN code); not checked
#pragma once
#include <stdint.h>
#include <string.h>
#include <cuda_fp16.h>
#include "../include/nvfp4effma.h"

// Build policy: only for targets without native FP4 (sm_100+ has it: use cuBLASLt / TE there).
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 500 || __CUDA_ARCH__ >= 1000
#error "NVFP4EffMA targets sm_50..sm_90 (no native FP4); exclude it from sm_100+ builds."
#endif
#endif

namespace nvfp4a {

// ---- arithmetic primitives -------------------------------------------------------------------
// explicit .rn without .ftz: denormals must be honoured
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

// E4M3 block scale byte -> exact FP32.  sign | (b & 0x7F) << 20 is the FP32 encoding of value(b) * 2^-120
// for every finite E4M3 b (subnormals land on FP32 subnormals); one FMUL by 2^120 restores it.
__device__ __forceinline__ float e4m3_value(uint32_t b) {
  return mul_rn(__uint_as_float(((b & 0x80u) << 24) | ((b & 0x7Fu) << 20)), 0x1p120f);
}
// UE5M3 block scale byte (unsigned, 5 exponent bits, bias 15, 3 mantissa bits) -> exact FP32: b << 20 is
// value(b) * 2^-112 in FP32 (subnormals included), one FMUL by 2^112 restores it.
__device__ __forceinline__ float ue5m3_value(uint32_t b) { return mul_rn(__uint_as_float((b & 0xFFu) << 20), 0x1p112f); }
// a block scale byte of either type (run-time choice: quantization; the GEMM kernels use scale_val<UE>)
__device__ __forceinline__ float scale_value(uint32_t b, bool ue5m3) { return ue5m3 ? ue5m3_value(b) : e4m3_value(b); }
template <bool UE> __device__ __forceinline__ float scale_val(uint32_t b) { return UE ? ue5m3_value(b) : e4m3_value(b); }

// Position of block scale (r, c) of an NVFP4 matrix: row part + column part, so kernels compute the row part
// once and step the column.  ld > 0: linear, ld scales per row.  ld < 0: the swizzled GEMM layout of
// cuBLASLt / Transformer Engine with -ld = roundup(cols / 16, 4) scales per (padded) row: 128 x 4 tiles of
// 512 bytes, scale (r, c) at ((r/128) * (-ld/4) + c/4) * 512 + (r%32) * 16 + ((r/32)%4) * 4 + c%4.
struct ScaleIdx {
  int ld;
  __host__ __device__ __forceinline__ size_t row(int r) const {
    return ld > 0 ? (size_t)r * ld
                  : ((size_t)(r >> 7) * (size_t)(-ld >> 2) << 9) + (size_t)((r & 31) << 4) + (size_t)(((r >> 5) & 3) << 2);
  }
  __host__ __device__ __forceinline__ size_t col(int c) const { return ld > 0 ? (size_t)c : ((size_t)(c >> 2) << 9) + (c & 3); }
  __host__ __device__ __forceinline__ size_t at(int r, int c) const { return row(r) + col(c); }
};
// The swizzled layout's row and column parts (sc = padded scale columns), for the kernels whose layout is
// a template parameter.
__host__ __device__ __forceinline__ size_t swz_row(int r, int sc) {
  return ((size_t)(r >> 7) * (size_t)(sc >> 2) << 9) + (size_t)((r & 31) << 4) + (size_t)(((r >> 5) & 3) << 2);
}
__host__ __device__ __forceinline__ size_t swz_col(int c) { return ((size_t)(c >> 2) << 9) + (c & 3); }

// Block scale formats of a GEMM / GEMV kernel instance: template parameter F, one instance per combination,
// so no format test reaches a main loop (F = 0: linear E4M3 scales).  The globals (per tensor or per row,
// decode factors or amaxes) only enter the epilogue, which handles all of them with the same operations.
//   kF_ASWZ / kF_BSWZ  the NVFP4 block scales of A / B are in the swizzled layout (ld = -padded columns)
//   kF_AUE / kF_BUE    ... are UE5M3 (else E4M3)
// B bits apply to NVFP4 B only (INT8 and dense B have none): kFormats = all, kFormatsA = A bits only (the
// same values 0 .. 3).
constexpr int kF_ASWZ = 1, kF_AUE = 2, kF_BSWZ = 4, kF_BUE = 8;
constexpr int kFormats = 16, kFormatsA = 4;

// E2M1 nibble j of w -> exact FP32 times 2^E: sign | (n & 7) << 22 encodes value(n) * 2^-126 (the
// subnormal 0.5 lands on the FP32 subnormal 2^-127), one FMUL by 2^(126 + E) restores it.  E = 1 gives
// the integer 2 * value of the DP4A tiles.
__device__ __forceinline__ float e2m1_raw(uint32_t w, int j);
template <int E>
__device__ __forceinline__ float e2m1_value(uint32_t w, int j) { return mul_rn(e2m1_raw(w, j), E == 0 ? 0x1p126f : 0x1p127f); }

// The raw encoding alone: sign | (n & 7) << 22 = value(n) * 2^-126 as FP32 (the decode before the FMUL).
__device__ __forceinline__ float e2m1_raw(uint32_t w, int j) {
  const uint32_t s = j < 7 ? (w << (28 - 4 * j)) & 0x80000000u : w & 0x80000000u;
  const uint32_t m = 4 * j <= 22 ? (w << (22 - 4 * j)) & 0x01C00000u : (w >> (4 * j - 22)) & 0x01C00000u;
  return __uint_as_float(s | m);
}

// 4 nibbles (low 16 bits of w16) -> 4 int8 (2 * value) in one word: PRMT lookups in a + table and a
// - table (3-bit selectors), selected per byte by the sign mask, which a PRMT sign-replicate builds from
// the nibbles' sign bits (bits 7 / 15 of w16 and of w16 << 4).
__device__ __forceinline__ uint32_t dec4(uint32_t w16) {
  const uint32_t P0 = 0x03020100u, P1 = 0x0C080604u, N0 = 0xFDFEFF00u, N1 = 0xF4F8FAFCu;
  const uint32_t sel = w16 & 0x7777u;
  const uint32_t pos = prmt(P0, P1, sel), neg = prmt(N0, N1, sel);
  const uint32_t m = prmt(w16, w16 << 4, 0x9D8Cu);
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(r) : "r"(pos), "r"(neg), "r"(m));  // m ? neg : pos
  return r;
}
// 16 nibbles (one NVFP4 block) -> 16 int8
__device__ __forceinline__ uint4 dec16(uint32_t w0, uint32_t w1) {
  return make_uint4(dec4(w0 & 0xFFFFu), dec4(w0 >> 16), dec4(w1 & 0xFFFFu), dec4(w1 >> 16));
}

// FP32 bits -> BF16 bits in the high half, round-to-nearest-even; NaN -> canonical 0x7FC0.
// u + 0x7FFF + lsb(bit 16), with the LSB moved into the carry flag (t + 0xFFFF0000 carries iff
// t == 0x10000): LOP + IADD.CC + IADD.X, all full rate.
__device__ __forceinline__ uint32_t rne_hi(uint32_t u) {
  u = (u & 0x7FFFFFFFu) > 0x7F800000u ? 0x7FC00000u : u;  // NaN would otherwise wrap/round to Inf
  uint32_t t = u & 0x10000u, r;
  asm("add.cc.u32 %1, %1, 0xFFFF0000;\n\taddc.u32 %0, %2, 0x7FFF;" : "=r"(r), "+r"(t) : "r"(u));
  return r;
}

// FP32 -> E4M3 code, round-to-nearest-even, saturating to +-448 (cvt.rn.satfinite.e4m3 semantics);
// every NaN -> canonical 0x7F.
__host__ __device__ __forceinline__ uint32_t f32_to_e4m3(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t s = (u >> 24) & 0x80u, a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return 0x7Fu;                  // NaN
  if (a >= 0x43E00000u) return s | 0x7Eu;             // |x| >= 448 (incl. Inf): saturate
  if (a < 0x3C800000u) {                              // |x| < 2^-6: subnormal grid 2^-9
    float m; memcpy(&m, &a, 4);
    m = m * 512.0f + 8388608.0f;                      // RNE to an integer in [0, 8] (exact scaling)
    uint32_t q; memcpy(&q, &m, 4);
    return s | (q - 0x4B000000u);                     // 8 -> 0x08 = smallest normal
  }
  uint32_t r = a + 0x7FFFFu + ((a >> 20) & 1u);       // RNE to 3 mantissa bits
  return s | (((r >> 23) - 120u) << 3) | ((r >> 20) & 7u);
}

// FP32 -> E2M1 nibble, round-to-nearest-even, saturating to +-6 (cvt.rn.satfinite.e2m1x2 semantics);
// the sign is kept, also when the result is zero.  Grid 0.5 below 2, 1 in [2, 4), 2 in [4, 6]: |x| * 2
// (resp. / 2) + 2^23 rounds to that grid exactly in one FP32 addition.
__host__ __device__ __forceinline__ uint32_t f32_to_e2m1(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t s = (u >> 28) & 8u, a = u & 0x7FFFFFFFu;
  if (a >= 0x40C00000u) return s | 7u;                // |x| >= 6 (incl. Inf, NaN): saturate
  float f; memcpy(&f, &a, 4);
  uint32_t q;
  if (a < 0x40000000u) {                              // |x| < 2: integer of 2|x| in [0, 4] -> codes 0..4
    f = f * 2.0f + 8388608.0f; memcpy(&q, &f, 4);
    return s | (q - 0x4B000000u);
  }
  if (a < 0x40800000u) {                              // 2 <= |x| < 4: integer in [2, 4] -> codes 4..6
    f = f + 8388608.0f; memcpy(&q, &f, 4);
    return s | (q - 0x4B000000u + 2u);
  }
  f = f * 0.5f + 8388608.0f; memcpy(&q, &f, 4);       // 4 <= |x| < 6: integer of |x|/2 in [2, 3] -> 6, 7
  return s | (q - 0x4B000000u + 4u);
}

// x / d correctly rounded for a divisor known in advance, r = RN(1 / d): x * r, then one FMA correction with
// the exact remainder (Brisebarre, Muller, Raina).  Zero: x * r (the signed zero).  The correction does not
// hold for subnormal quotients and infinite x: those (and NaN) take IEEE division -- a branch real data does
// not take (zeros, frequent, must not: they would diverge into the slow path).  Valid for the divisors
// whose exhaustive test passes (tests/test_nvfp4effma.cu: the 4over6 denominators 6 * 256 and 6 * 448).
__device__ __forceinline__ float div_by(float x, float d, float r) {
  const float q = __fmul_rn(x, r), aq = fabsf(q);
  if (x != 0.f && !(aq >= 0x1p-125f && aq <= 3.40282347e38f)) return __fdiv_rn(x, d);
  return x == 0.f ? q : __fmaf_rn(__fmaf_rn(-q, d, x), r, q);
}

// The same conversion without branches (selects on the range, f * 2 / 1 / 0.5 is exact): for kernels with
// little occupancy, where divergent branches per element cost the instruction-level parallelism.
__device__ __forceinline__ uint32_t f32_to_e2m1_sel(float x) {
  const uint32_t u = __float_as_uint(x), s = (u >> 28) & 8u, a = u & 0x7FFFFFFFu;
  const bool lo = a < 0x40000000u, mid = a < 0x40800000u;  // |x| < 2, |x| < 4
  const float f = fmaf(__uint_as_float(a), lo ? 2.0f : mid ? 1.0f : 0.5f, 8388608.0f);
  const uint32_t q = __float_as_uint(f) - 0x4B000000u + (lo ? 0u : mid ? 2u : 4u);
  return s | (a >= 0x40C00000u ? 7u : q);
}

// FP32 -> UE5M3 code, round-to-nearest-even, saturating to 114688 (code 0xFE); negative values and NaN -> 0
// (as Transformer Engine's conversion of block scales, which are never negative).
__host__ __device__ __forceinline__ uint32_t f32_to_ue5m3(float x) {
  uint32_t a;
  memcpy(&a, &x, 4);
  if (a >= 0x80000000u || a > 0x7F800000u) return 0u;  // negative, -0, NaN
  if (a >= 0x47E00000u) return 0xFEu;                  // >= 114688 (incl. Inf)
  if (a < 0x38800000u) {                               // < 2^-14: subnormal grid 2^-17
    float m; memcpy(&m, &a, 4);
    m = m * 131072.0f + 8388608.0f;                    // RNE to an integer in [0, 8]
    uint32_t q; memcpy(&q, &m, 4);
    return q - 0x4B000000u;                            // 8 -> 0x08 = smallest normal
  }
  const uint32_t r = a + 0x7FFFFu + ((a >> 20) & 1u);  // RNE to 3 mantissa bits
  const uint32_t c = (((r >> 23) - 112u) << 3) | ((r >> 20) & 7u);
  return c > 0xFEu ? 0xFEu : c;
}

namespace kern {

// ---- typed element access (BF16 / FP16 / FP32) -----------------------------------------------
template <int T> struct Elem { static constexpr int bytes = T == NVFP4A_FP32 ? 4 : T == NVFP4A_INT8 ? 1 : 2; };

__device__ __forceinline__ float scale_or_one(const float *p) { return p ? *p : 1.0f; }

__device__ __forceinline__ float load_any(int t, const void *p, size_t i) {
  switch (t) {
    case NVFP4A_BF16: return __uint_as_float((uint32_t)((const uint16_t *)p)[i] << 16);
    case NVFP4A_FP16: return __half2float(((const __half *)p)[i]);
    default: return ((const float *)p)[i];
  }
}
__device__ __forceinline__ void store_any(int t, void *p, size_t i, float v) {
  switch (t) {
    case NVFP4A_BF16: ((uint16_t *)p)[i] = (uint16_t)(rne_hi(__float_as_uint(v)) >> 16); break;
    case NVFP4A_FP16: ((__half *)p)[i] = __float2half_rn(v); break;
    default: ((float *)p)[i] = v;
  }
}
// 2 consecutive values of type T from raw 32-bit words: word q of a BF16 / FP16 row holds values 2q, 2q + 1
template <int T>
__device__ __forceinline__ void unpack2(uint32_t w, float &v0, float &v1) {
  if (T == NVFP4A_BF16) { v0 = __uint_as_float(w << 16); v1 = __uint_as_float(w & 0xFFFF0000u); }
  else { const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&w)); v0 = f.x; v1 = f.y; }
}
// 8 consecutive dense values of type T (address aligned to 8 * bytes) -> 8 exact FP32 values
template <int T>
__device__ __forceinline__ void load8(const uint8_t *p, float v[8]) {
  if (T == NVFP4A_FP32) {
    const float4 a = *(const float4 *)p, b = *(const float4 *)(p + 16);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
    const uint4 w = *(const uint4 *)p;
    unpack2<T>(w.x, v[0], v[1]); unpack2<T>(w.y, v[2], v[3]); unpack2<T>(w.z, v[4], v[5]); unpack2<T>(w.w, v[6], v[7]);
  }
}

// Warp-reduce the amax statistic into *amax (non-negative floats order like ints).
__device__ __forceinline__ void amax_commit(float *amax, float m) {
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
  if ((threadIdx.x & 31) == 0 && m > 0.f) atomicMax((int *)amax, __float_as_int(m));
}

// ---- GEMM / GEMV epilogue --------------------------------------------------------------------
// Global value of an NVFP4 / INT8 operand: v = gX[row * sX] -- one value (stride 0) or one per row (stride 1)
// -- or 1 without one; amax globals contribute the factor 1 / (6 * scale_max) to gf (1 otherwise, exact).
// s(i, j) = ((alpha * vA(i)) * vB(j)) * gf: Transformer Engine's order, and for per-tensor decode factors
// ((alpha * gA) * gB) as before.  The same operations for every kind of global: no format branch.
struct Epilogue {
  const float *ga, *gb;  // device globals of A / B (NULL = 1)
  int sa, sb;            // their strides: 0 (per tensor) or 1 (per row of A = output row i, of B = column j)
  float gf;              // product of the amax factors (1 without amax globals)
  float alpha, beta;
  const void *c; int ldc; int c_dtype;
  void *d; int ldd; int d_dtype; float *amax;
  const void *bias; int bias_dtype;
};

struct EpiState { float amax; };

__device__ __forceinline__ EpiState epi_begin(const Epilogue &) { return {0.f}; }
// the row part a = alpha * vA(i) and the column part b = vB(j) of an output's scale (kernels with a register
// tile load them once per row / column), and the scale (a * b) * gf
__device__ __forceinline__ float epi_row_part(const Epilogue &e, int i) {
  return __fmul_rn(e.alpha, e.ga ? e.ga[(size_t)i * e.sa] : 1.f);
}
__device__ __forceinline__ float epi_col_part(const Epilogue &e, int j) { return e.gb ? e.gb[(size_t)j * e.sb] : 1.f; }
__device__ __forceinline__ float epi_scale_ab(const Epilogue &e, float a, float b) { return __fmul_rn(__fmul_rn(a, b), e.gf); }
__device__ __forceinline__ float epi_scale(const Epilogue &e, int i, int j) {
  return epi_scale_ab(e, epi_row_part(e, i), epi_col_part(e, j));
}
// D[i + j*ldd] = RNE(acc * s + bias[i] + beta * C[i + j*ldc]), max-reducing |value| into the amax
// statistic.  C may alias D: each element is read and then written by the same thread.
__device__ __forceinline__ void epi_store(const Epilogue &e, EpiState &st, int i, int j, float acc) {
  float v = __fmul_rn(acc, epi_scale(e, i, j));
  if (e.bias) v = __fadd_rn(v, load_any(e.bias_dtype, e.bias, i));
  if (e.beta != 0.f) v = __fmaf_rn(e.beta, load_any(e.c_dtype, e.c, (size_t)j * e.ldc + i), v);
  store_any(e.d_dtype, e.d, (size_t)j * e.ldd + i, v);
  st.amax = fmaxf(st.amax, fabsf(v));
}
__device__ __forceinline__ void epi_end(const Epilogue &e, EpiState &st) {
  if (e.amax) amax_commit(e.amax, st.amax);
}

// ---- per-architecture kernel tables ----------------------------------------------------------
// B / x dtypes NVFP4, BF16, FP16, FP32, INT8 -> 0, 1, 2, 3, 4 (the dtype codes)
constexpr int kOperandTypes = 5;

// Largest n that runs the GEMV kernel (in groups of 8 columns, each group reading A once) instead of the
// tiled kernel, per B dtype (NVFP4, BF16, FP16, FP32, INT8): where the tiled kernel has too few tiles to
// fill the GPU, re-reading A per group is faster.  Measured on a GTX 1080 at m = k = 8192.
#ifdef NVFP4A_GEMV_MAXN  // measurement builds: one limit for every dtype
constexpr int kGemvMaxN[kOperandTypes] = {NVFP4A_GEMV_MAXN, NVFP4A_GEMV_MAXN, NVFP4A_GEMV_MAXN, NVFP4A_GEMV_MAXN, NVFP4A_GEMV_MAXN};
#else
constexpr int kGemvMaxN[kOperandTypes] = {48, 56, 56, 32, 56};
#endif

struct ArchTable {
  int cc;                                // compute capability * 10 + minor (61 = sm_61); 0 = default
  int gemm_large, gemm_small;            // dense-B GEMM candidate ids (small: fewer than 2 x SMs large tiles)
  unsigned char gemv[kOperandTypes][8];  // GEMV candidate id per x dtype and n = 1..8
};

}  // namespace kern
}  // namespace nvfp4a

#include "nvfp4effma_tuned.h"  // generated by `make tune`
#ifndef NVFP4A_TUNED_TABLES
#define NVFP4A_TUNED_TABLES
#endif
#ifndef NVFP4A_GEMM_USED
#define NVFP4A_GEMM_USED(X)
#endif
#ifndef NVFP4A_GEMV_USED
#define NVFP4A_GEMV_USED(X)
#endif

namespace nvfp4a {
namespace kern {

// Defaults (GPUs without a tuned entry): dense GEMM 128x128 for large problems, 64x64 for small ones;
// GEMV 4 rows per warp.
constexpr ArchTable kDefaultTable = {0, 0, 1, {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}}};
constexpr ArchTable kTables[] = {kDefaultTable NVFP4A_TUNED_TABLES};
constexpr int kNumTables = sizeof(kTables) / sizeof(kTables[0]);

constexpr int table_index(int cc) {
  int r = 0;
  for (int i = 1; i < kNumTables; ++i) if (kTables[i].cc == cc) r = i;
  return r;
}
// Configurations a device of compute capability cc can be dispatched to: those of its table plus the
// defaults (a GPU running binary-compatible SASS of another minor version uses the defaults).
constexpr bool gemm_selected(int cc, int id) {
  return id == kDefaultTable.gemm_large || id == kDefaultTable.gemm_small ||
         id == kTables[table_index(cc)].gemm_large || id == kTables[table_index(cc)].gemm_small;
}
constexpr bool gemv_selected(int cc, int id, int xt, int nb) {
  return id == kDefaultTable.gemv[xt][nb - 1] || id == kTables[table_index(cc)].gemv[xt][nb - 1];
}

// compute capability of the device pass (0 in the host pass): kernel bodies of configurations not
// selected for it are compiled as trapping stubs
#if defined(__CUDA_ARCH__)
#define NVFP4A_ARCH_CC (__CUDA_ARCH__ / 10)
#else
#define NVFP4A_ARCH_CC 0
#endif

// Host: the current device's SM count and table (src/nvfp4effma.cu).  sass = the SASS version the device runs
// (DP4A from 61), ptx = the PTX version its code was built from: it decides the tensor-core paths, from 75 to 89 (tc;
// W4A16 from 80; NVFP4A_DP4A=1 in the environment turns them off) -- the same as sass for SASS builds, and right for
// PTX JIT-compiled for a newer GPU.  supported = false when the device has native FP4 (sm_100+), predates sm_50, or
// the library carries no SASS that runs on it.
struct DeviceInfo { int cc, sass, ptx, sms; bool supported, tc; const ArchTable *table; };
const DeviceInfo &device_info();

}  // namespace kern

// Host: error reporting of the C API.  Every entry point starts with clear_error(); failures record a
// message for nvfp4a_last_error() (thread-local) and return their status.
void clear_error();
nvfp4a_status fail(nvfp4a_status s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
nvfp4a_status cuda_status(cudaError_t e, const char *what);  // NVFP4A_OK, or NVFP4A_CUDA_ERROR with message
nvfp4a_status check_device();  // NVFP4A_OK, or NVFP4A_NOT_SUPPORTED / NVFP4A_CUDA_ERROR with message
// NVFP4 tensor metadata (src/gemm_nvfp4effma.cu): the bound of the global encoding (0 = invalid
// combination), the flag / layout checks (ld -> the ld of ScaleIdx), the epilogue's amax factor
float scale_max_of(const nvfp4a_tensor *t);
nvfp4a_status check_scales(const char *fn, const char *name, const nvfp4a_tensor *t, int rows, int cols, int *ld);
float amax_factor(const nvfp4a_tensor *A, const nvfp4a_tensor *B);

}  // namespace nvfp4a

#define NVFP4A_CAT6_(a, b, c, d, e, f) a##b##c##d##e##f
#define NVFP4A_CAT6(a, b, c, d, e, f) NVFP4A_CAT6_(a, b, c, d, e, f)
#define NVFP4A_CAT8_(a, b, c, d, e, f, g, h) a##b##c##d##e##f##g##h
#define NVFP4A_CAT8(a, b, c, d, e, f, g, h) NVFP4A_CAT8_(a, b, c, d, e, f, g, h)
