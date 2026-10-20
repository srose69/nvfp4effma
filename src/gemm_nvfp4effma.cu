// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMM host side: nvfp4a_gemm (validation, epilogue setup, routing of small n to the GEMV
// kernel), the NVFP4 x NVFP4 path (DP4A from sm_61, FFMA below) and the dense-B configuration choice.
#include <utility>
#include "gemm_nvfp4effma.cuh"
#include "gemv_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

using GemmFn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int,
                               int, int, const Epilogue &, cudaStream_t);
using Dp4aFn = cudaError_t (*)(bool, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int,
                               int, int, int, const Epilogue &, cudaStream_t);
// entries by format: NVFP4 B every format, the other B dtypes the formats without B bits (0 .. 3)
template <int ID, int TB, size_t... I>
static const GemmFn *gemm_table(std::index_sequence<I...>) {
  static const GemmFn t[] = {&gemm_entry<ID, TB, TB == NVFP4A_NVFP4 ? (int)I : (int)I>...};
  return t;
}
template <int TB, size_t... I>
static const Dp4aFn *dp4a_table(std::index_sequence<I...>) {
  static const Dp4aFn t[] = {&dp4a_entry<TB, TB == NVFP4A_NVFP4 ? (int)I : (int)I>...};
  return t;
}

// The 64-wide DP4A tile costs about 2/3 of the 128-wide one for half the work, so it only wins where the
// last 128-wide tile would be at most half full and there are few of them: n = 57..64 and 129..192
// (below, the GEMV kernel runs).  Measured on a GTX 1080 at m x k = 4096 x 14336 and 8192 x 8192.
static bool dp4a_narrow(int n) { return n % 128 != 0 && n % 128 <= 64 && n <= 192; }

static cudaError_t launch_gemm(int tb, int f, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                               const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const DeviceInfo &d = device_info();
  if (tb == NVFP4A_NVFP4 || tb == NVFP4A_INT8) {
    if (d.sass >= 61) {
      const Dp4aFn fn = tb == NVFP4A_NVFP4 ? dp4a_table<NVFP4A_NVFP4>(std::make_index_sequence<kFormats>())[f]
                                           : dp4a_table<NVFP4A_INT8>(std::make_index_sequence<kFormatsA>())[(f & 3)];
      return fn(dp4a_narrow(n), A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, s);
    }
    const GemmFn fn = tb == NVFP4A_NVFP4 ? gemm_table<0, NVFP4A_NVFP4>(std::make_index_sequence<kFormats>())[f]
                                         : gemm_table<0, NVFP4A_INT8>(std::make_index_sequence<kFormatsA>())[(f & 3)];
    return fn(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, s);
  }
  int id = d.table->gemm_large, bm = 128, bn = 128;
#define NVFP4A_TILE(cid, cbm, cbn, ctm, ctn) if (id == cid) { bm = cbm; bn = cbn; }
  NVFP4A_GEMM_CANDIDATES(NVFP4A_TILE)
#undef NVFP4A_TILE
  if ((long long)((m + bm - 1) / bm) * ((n + bn - 1) / bn) < 2LL * d.sms) id = d.table->gemm_small;
  const int fi = (f & 3);
#define NVFP4A_CASE(cid)                                                                                          \
  if (id == cid) {                                                                                                \
    const GemmFn fn = tb == NVFP4A_BF16   ? gemm_table<cid, NVFP4A_BF16>(std::make_index_sequence<kFormatsA>())[fi] \
                      : tb == NVFP4A_FP16 ? gemm_table<cid, NVFP4A_FP16>(std::make_index_sequence<kFormatsA>())[fi] \
                                          : gemm_table<cid, NVFP4A_FP32>(std::make_index_sequence<kFormatsA>())[fi]; \
    return fn(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, s);                                                 \
  }
  NVFP4A_GEMM_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
  return cudaErrorInvalidConfiguration;
}

// GEMV for n columns: groups of up to 8, each its own launch on the shifted B, its scales and globals, C and
// D.  The scales of rows j0 .. j0 + 7 (j0 a multiple of 8) start at row j0's row part in either layout.
static cudaError_t launch_gemv_groups(int tb, int f, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B,
                                      int ldb, const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e0,
                                      cudaStream_t s) {
  const size_t bbytes = tb == NVFP4A_NVFP4 ? 0 : tb == NVFP4A_INT8 ? 1 : tb == NVFP4A_FP32 ? 4 : 2;  // NVFP4: ld / 2
  const auto esize = [](int t) { return (size_t)(t == NVFP4A_FP32 ? 4 : 2); };
  for (int j0 = 0; j0 < n; j0 += 8) {
    Epilogue e = e0;
    e.d = (uint8_t *)e0.d + (size_t)j0 * e0.ldd * esize(e0.d_dtype);
    if (e0.c) e.c = (const uint8_t *)e0.c + (size_t)j0 * e0.ldc * esize(e0.c_dtype);
    const uint8_t *b = B + (bbytes ? (size_t)j0 * ldb * bbytes : (size_t)j0 * (ldb / 2));
    const uint8_t *sb = !SB ? nullptr : tb == NVFP4A_INT8 ? SB + (size_t)j0 * ldsb * 4
                        : f & kF_BSWZ ? SB + swz_row(j0, -ldsb) : SB + (size_t)j0 * ldsb;
    if (e0.sb) e.gb = e0.gb + j0;
    const cudaError_t err = launch_gemv(tb, f, A, lda, SA, ldsa, b, ldb, sb, ldsb, m, n - j0 < 8 ? n - j0 : 8, k, e, s);
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

}  // namespace kern

// The bound of an NVFP4 tensor's global encoding (scale_max, 0 = the scale type's maximum), or 0 when the
// combination is not one Transformer Engine defines.
float scale_max_of(const nvfp4a_tensor *t) {
  const bool ue = t->flags & NVFP4A_SCALES_UE5M3;
  if (t->scale_max == 0) return ue ? 114688.f : 448.f;
  if (ue) return t->scale_max == 114688 ? 114688.f : 0.f;
  return t->scale_max == 448 || t->scale_max == 256 ? (float)t->scale_max : 0.f;
}

// Checks of the NVFP4-specific fields of an operand with `cols` values per row; ld_scale -> the ld of
// ScaleIdx (negative: swizzled).
nvfp4a_status check_scales(const char *fn, const char *name, const nvfp4a_tensor *t, int rows, int cols, int *ld) {
  const unsigned known = NVFP4A_SCALES_SWIZZLED | NVFP4A_SCALES_UE5M3 | NVFP4A_GLOBAL_AMAX | NVFP4A_GLOBAL_PER_ROW;
  if (t->flags & ~known) return fail(NVFP4A_INVALID_VALUE, "%s: %s has unknown flags 0x%x", fn, name, t->flags & ~known);
  if (t->dtype != NVFP4A_NVFP4 && t->flags)
    return fail(NVFP4A_INVALID_VALUE, "%s: flags are for NVFP4 tensors (%s has dtype %d)", fn, name, (int)t->dtype);
  if (t->dtype == NVFP4A_NVFP4 && scale_max_of(t) == 0.f)
    return fail(NVFP4A_INVALID_VALUE, "%s: %s has scale_max = %d (E4M3: 0, 448 or 256; UE5M3: 0 or 114688)", fn, name,
                t->scale_max);
  if ((t->flags & NVFP4A_GLOBAL_PER_ROW) && !t->global_scale)
    return fail(NVFP4A_INVALID_VALUE, "%s: %s has per-row globals but no global_scale", fn, name);
  const int sc = (cols / 16 + 3) / 4 * 4;  // padded scale columns of the swizzled layout
  if (t->flags & NVFP4A_SCALES_SWIZZLED) {
    if (t->ld_scale != 0 && t->ld_scale != sc)
      return fail(NVFP4A_INVALID_VALUE, "%s: swizzled scales of %s have rows of %d scales (ld_scale must be 0 or %d)", fn,
                  name, sc, sc);
    *ld = -sc;
  } else {
    if (t->ld_scale && t->ld_scale < cols / 16)
      return fail(NVFP4A_INVALID_VALUE, "%s: %s has ld_scale = %d < %d", fn, name, t->ld_scale, cols / 16);
    *ld = t->ld_scale ? t->ld_scale : cols / 16;
  }
  (void)rows;
  return NVFP4A_OK;
}

// Product of the amax factors 1 / (6 * scale_max) of the operands whose global is an amax, as Transformer
// Engine forms it (1 / (6 * 6 * mA * mB) when both are); 1 when none is.
float amax_factor(const nvfp4a_tensor *A, const nvfp4a_tensor *B) {
  const bool aa = A && (A->flags & NVFP4A_GLOBAL_AMAX), ab = B && (B->flags & NVFP4A_GLOBAL_AMAX);
  if (aa && ab) return 1.0f / (6.f * 6.f * scale_max_of(A) * scale_max_of(B));
  if (aa) return 1.0f / (6.f * scale_max_of(A));
  if (ab) return 1.0f / (6.f * scale_max_of(B));
  return 1.0f;
}

}  // namespace nvfp4a

extern "C" nvfp4a_status nvfp4a_gemm(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda,
                                     const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                                     const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, cudaStream_t stream) {
  using namespace nvfp4a;
  using namespace nvfp4a::kern;
  nvfp4a::clear_error();
  const auto dense = [](const nvfp4a_tensor *t) { return t->dtype >= NVFP4A_BF16 && t->dtype <= NVFP4A_FP32; };
  const auto blocked = [](const nvfp4a_tensor *t) { return t->dtype == NVFP4A_NVFP4 || t->dtype == NVFP4A_INT8; };
  const auto aligned = [](const void *p, unsigned a) { return ((uintptr_t)p & (a - 1)) == 0; };
  if (!A || !B || !D || !A->data || !B->data || !D->data)
    return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: A, B, D or their data is NULL");
  if (m < 0 || n < 0 || k < 0) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: negative size (m=%d n=%d k=%d)", m, n, k);
  if (A->dtype != NVFP4A_NVFP4) return fail(NVFP4A_NOT_SUPPORTED, "nvfp4a_gemm: A must be NVFP4 (dtype %d)", (int)A->dtype);
  if (!blocked(B) && !dense(B))
    return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: invalid B dtype %d", (int)B->dtype);
  if (!dense(D)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: D must be BF16, FP16 or FP32 (dtype %d)", (int)D->dtype);
  if (m == 0 || n == 0) return NVFP4A_OK;
  if (k % 64) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: k = %d is not a multiple of 64", k);
  if (lda % 64 || ldb % 64) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: lda = %d / ldb = %d must be multiples of 64", lda, ldb);
  if (lda < k || ldb < k) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: lda = %d / ldb = %d < k = %d", lda, ldb, k);
  if (!aligned(A->data, 16) || !aligned(B->data, 16)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: A and B must be 16-byte aligned");
  int ldsa = 0, ldsb = 0;
  for (const nvfp4a_tensor *t : {A, B}) {
    if (!blocked(t)) {
      if (t->flags) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: flags are for NVFP4 tensors (dtype %d)", (int)t->dtype);
      continue;
    }
    const unsigned sa = t->dtype == NVFP4A_INT8 ? 16 : 4;
    if (!t->block_scale) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: NVFP4 / INT8 operand without block_scale");
    if (!aligned(t->block_scale, sa) || t->ld_scale % 4 || (t->ld_scale && t->ld_scale < k / 16))
      return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: block scales must be %u-byte aligned with ld_scale a multiple of 4 "
                  "and >= k / 16 (ld_scale = %d)", sa, t->ld_scale);
    if (nvfp4a_status st = nvfp4a::check_scales("nvfp4a_gemm", t == A ? "A" : "B", t, t == A ? m : n, k, t == A ? &ldsa : &ldsb))
      return st;
  }
  if (ldd < m) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: ldd = %d < m = %d", ldd, m);
  if (bias && bias->data && !dense(bias)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: invalid bias dtype %d", (int)bias->dtype);
  if (beta != 0.f && C && C->data && !dense(C)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: invalid C dtype %d", (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: ldc = %d < m = %d", ldc, m);
  if (nvfp4a_status st = nvfp4a::check_device()) return st;
  const bool c_is_d = beta != 0.f && (!C || !C->data);  // D += ...

  Epilogue e;
  e.ga = A->global_scale;
  e.gb = blocked(B) ? B->global_scale : nullptr;
  e.gf = nvfp4a::amax_factor(A, blocked(B) ? B : nullptr);
  e.sa = (A->flags & NVFP4A_GLOBAL_PER_ROW) ? 1 : 0;
  e.sb = (B->flags & NVFP4A_GLOBAL_PER_ROW) ? 1 : 0;
  // the kernel instance's block scale formats (B flags exist on NVFP4 B only: check_scales)
  const auto has = [](const nvfp4a_tensor *t, unsigned flag) { return (t->flags & flag) != 0; };
  const int f = (has(A, NVFP4A_SCALES_SWIZZLED) ? kF_ASWZ : 0) | (has(A, NVFP4A_SCALES_UE5M3) ? kF_AUE : 0) |
                (has(B, NVFP4A_SCALES_SWIZZLED) ? kF_BSWZ : 0) | (has(B, NVFP4A_SCALES_UE5M3) ? kF_BUE : 0);
  e.alpha = alpha; e.beta = beta;
  const nvfp4a_tensor *Cx = c_is_d ? D : C;
  e.c = beta != 0.f ? Cx->data : nullptr; e.ldc = beta != 0.f ? (c_is_d ? ldd : ldc) : 0;
  e.c_dtype = beta != 0.f ? Cx->dtype : NVFP4A_FP32;
  e.d = D->data; e.ldd = ldd; e.d_dtype = D->dtype; e.amax = D->amax;
  e.bias = bias && bias->data ? bias->data : nullptr; e.bias_dtype = bias ? bias->dtype : NVFP4A_FP32;

  const uint8_t *a = (const uint8_t *)A->data, *b = (const uint8_t *)B->data;
  const uint8_t *sa = (const uint8_t *)A->block_scale, *sb = (const uint8_t *)B->block_scale;
  const cudaError_t err = n <= kGemvMaxN[B->dtype]
                              ? launch_gemv_groups(B->dtype, f, a, lda, sa, ldsa, b, ldb, sb, ldsb, m, n, k, e, stream)
                              : launch_gemm(B->dtype, f, a, lda, sa, ldsa, b, ldb, sb, ldsb, m, n, k, e, stream);
  return nvfp4a::cuda_status(err, "nvfp4a_gemm: kernel launch");
}

extern "C" nvfp4a_status nvfp4a_gemm_dual(int m, int n, int k, float alpha, const nvfp4a_dual_tensor *A, int trans_a,
                                          const nvfp4a_dual_tensor *B, int trans_b, float beta, const nvfp4a_tensor *C,
                                          int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                          cudaStream_t stream) {
  using nvfp4a::fail;
  nvfp4a::clear_error();
  if (!A || !B) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm_dual: A or B is NULL");
  // the half whose blocks run along k: the row half of a [m][k] operand, the column half of a [k][m] one
  const auto half = [](const nvfp4a_dual_tensor *t, int trans, const char *name, const nvfp4a_tensor **q, int *ld) {
    *q = trans ? &t->col : &t->row;
    *ld = trans ? t->ld_col : t->ld_row;
    if (!(*q)->data)
      return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm_dual: %s%s has no %s half", name, trans ? " (trans = 1)" : "",
                  trans ? "column" : "row");
    return NVFP4A_OK;
  };
  const nvfp4a_tensor *a, *b;
  int lda, ldb;
  if (nvfp4a_status st = half(A, trans_a, "A", &a, &lda)) return st;
  if (nvfp4a_status st = half(B, trans_b, "B", &b, &ldb)) return st;
  return nvfp4a_gemm(m, n, k, alpha, a, lda, b, ldb, beta, C, ldc, D, ldd, bias, stream);
}
