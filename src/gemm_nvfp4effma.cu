// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMM host side: nvfp4a_gemm (validation, epilogue setup, routing of small n to the GEMV
// kernel), the NVFP4 x NVFP4 path (DP4A from sm_61, FFMA below) and the dense-B configuration choice; the
// tensor-core API (nvfp4a_tc_image, nvfp4a_gemm_tc, nvfp4a_gemv_tc; sm_75 .. sm_89).
#include <stdio.h>
#include <limits.h>
#include <utility>
#include "gemm_nvfp4effma.cuh"
#include "gemv_nvfp4effma.cuh"
#include "tensors/gemm_tc.cuh"
#include "tensors/gemv_tc.cuh"
#include "../include/nvfp4effma_items.h"

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
// tensor-core entries by scale formats (kF_AUE / kF_BUE: index F >> 1 & 1 | F >> 2 & 2)
using TcFn = cudaError_t (*)(const uint8_t *, const uint8_t *, int, int, int, const Epilogue &, uint8_t *, cudaStream_t);
static const TcFn kTc4[] = {&tc_entry<NVFP4A_NVFP4, 0>, &tc_entry<NVFP4A_NVFP4, kF_AUE>, &tc_entry<NVFP4A_NVFP4, kF_BUE>,
                            &tc_entry<NVFP4A_NVFP4, kF_AUE | kF_BUE>};
static const TcFn kTc8[] = {&tc_entry<NVFP4A_INT8, 0>, &tc_entry<NVFP4A_INT8, kF_AUE>};
static const TcFn kTcH[] = {&tc_entry<NVFP4A_BF16, 0>, &tc_entry<NVFP4A_BF16, kF_AUE>};
using TcExFn = cudaError_t (*)(const uint8_t *, const uint8_t *, int, int, int, const Epilogue &, const EpilogueEx &, uint8_t *, cudaStream_t);
static const TcExFn kTcEx8[] = {&tc_entry_ex<NVFP4A_INT8, 1>, &tc_entry_ex<NVFP4A_INT8, 2>, &tc_entry_ex<NVFP4A_INT8, 3>,
                                &tc_entry_ex<NVFP4A_INT8, 4>, &tc_entry_ex<NVFP4A_INT8, 5>};
static const TcExFn kTcExH[] = {&tc_entry_ex<NVFP4A_BF16, 1>, &tc_entry_ex<NVFP4A_BF16, 2>, &tc_entry_ex<NVFP4A_BF16, 3>,
                                &tc_entry_ex<NVFP4A_BF16, 4>, &tc_entry_ex<NVFP4A_BF16, 5>};
// tensor-core GEMV entries by formats F: NVFP4 x [x scales: linear, swizzled, linear by 16 bytes][kF_BUE][kF_AUE]; the
// others kF_AUE
using GvFn = cudaError_t (*)(const uint8_t *, const uint8_t *, int, const uint8_t *, int, int, int, int, const Epilogue &, cudaStream_t);
static const GvFn kGv4[] = {&gemv_tc_entry<NVFP4A_NVFP4, 0>,  &gemv_tc_entry<NVFP4A_NVFP4, 2>,  &gemv_tc_entry<NVFP4A_NVFP4, 8>,
                            &gemv_tc_entry<NVFP4A_NVFP4, 10>, &gemv_tc_entry<NVFP4A_NVFP4, 4>,  &gemv_tc_entry<NVFP4A_NVFP4, 6>,
                            &gemv_tc_entry<NVFP4A_NVFP4, 12>, &gemv_tc_entry<NVFP4A_NVFP4, 14>, &gemv_tc_entry<NVFP4A_NVFP4, 16>,
                            &gemv_tc_entry<NVFP4A_NVFP4, 18>, &gemv_tc_entry<NVFP4A_NVFP4, 24>, &gemv_tc_entry<NVFP4A_NVFP4, 26>};
static const GvFn kGv8[] = {&gemv_tc_entry<NVFP4A_INT8, 0>, &gemv_tc_entry<NVFP4A_INT8, kF_AUE>};
static const GvFn kGvH[] = {&gemv_tc_entry<NVFP4A_BF16, 0>, &gemv_tc_entry<NVFP4A_BF16, kF_AUE>};
using GvExFn = cudaError_t (*)(const uint8_t *, const uint8_t *, int, const uint8_t *, int, int, int, int, const Epilogue &,
                               const EpilogueEx &, cudaStream_t);
static const GvExFn kGvEx8[] = {&gemv_tc_entry_ex<NVFP4A_INT8, 1>, &gemv_tc_entry_ex<NVFP4A_INT8, 2>, &gemv_tc_entry_ex<NVFP4A_INT8, 3>,
                                &gemv_tc_entry_ex<NVFP4A_INT8, 4>, &gemv_tc_entry_ex<NVFP4A_INT8, 5>};
static const GvExFn kGvExH[] = {&gemv_tc_entry_ex<NVFP4A_BF16, 1>, &gemv_tc_entry_ex<NVFP4A_BF16, 2>, &gemv_tc_entry_ex<NVFP4A_BF16, 3>,
                                &gemv_tc_entry_ex<NVFP4A_BF16, 4>, &gemv_tc_entry_ex<NVFP4A_BF16, 5>};

// The 64-wide DP4A tile costs about 2/3 of the 128-wide one for half the work, so it only wins where the
// last 128-wide tile would be at most half full and there are few of them: n = 57..64 and 129..192
// (below, the GEMV kernel runs).  Measured on a GTX 1080 at m x k = 4096 x 14336 and 8192 x 8192.
static bool dp4a_narrow(int n) { return n % 128 != 0 && n % 128 <= 64 && n <= 192; }

static int gemm_config(int m, int n, int tb) {
  if (tb == NVFP4A_INT8 || tb == NVFP4A_NVFP4) return 0;
  const DeviceInfo &d = device_info();
  int id = d.table->gemm_large, bm = 128, bn = 128;
#define NVFP4A_TILE(cid, cbm, cbn, ctm, ctn) if (id == cid) { bm = cbm; bn = cbn; }
  NVFP4A_GEMM_CANDIDATES(NVFP4A_TILE)
#undef NVFP4A_TILE
  return (long long)((m + bm - 1) / bm) * ((n + bn - 1) / bn) < 2LL * d.sms ? d.table->gemm_small : id;
}

// the extended epilogue's tiled kernels: [EX - 1]
using GemmExFn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int, int, int,
                                 const Epilogue &, const EpilogueEx &, cudaStream_t);
using Dp4aExFn = cudaError_t (*)(bool, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int, int,
                                 int, const Epilogue &, const EpilogueEx &, cudaStream_t);
template <int ID, int TB>
static const GemmExFn *gemm_ex_table() {
  static const GemmExFn t[] = {&gemm_entry_ex<ID, TB, 1>, &gemm_entry_ex<ID, TB, 2>, &gemm_entry_ex<ID, TB, 3>,
                               &gemm_entry_ex<ID, TB, 4>, &gemm_entry_ex<ID, TB, 5>};
  return t;
}
static const Dp4aExFn kDp4aEx[] = {&dp4a_entry_ex<1>, &dp4a_entry_ex<2>, &dp4a_entry_ex<3>, &dp4a_entry_ex<4>, &dp4a_entry_ex<5>};

// B BF16 (the FFMA configuration launch_gemm picks) or INT8 (DP4A from sm_61, FFMA configuration 0 below)
static cudaError_t launch_gemm_ex(int tb, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                                  const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, const EpilogueEx &x, int EX,
                                  cudaStream_t s);

static cudaError_t launch_gemm(int tb, int f, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                               const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, const EpilogueEx &x, int EX,
                               cudaStream_t s) {
  const DeviceInfo &d = device_info();
  if (EX) return launch_gemm_ex(tb, A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, x, EX, s);
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
  const int id = gemm_config(m, n, tb);
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

static cudaError_t launch_gemm_ex(int tb, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                                  const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, const EpilogueEx &x, int EX,
                                  cudaStream_t s) {
  const DeviceInfo &d = device_info();
  if (EX < 1 || EX > kEX_MASKS || (tb != NVFP4A_BF16 && tb != NVFP4A_INT8)) return cudaErrorInvalidConfiguration;
  if (tb == NVFP4A_INT8 && d.sass >= 61) return kDp4aEx[EX - 1](dp4a_narrow(n), A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, x, s);
  if (tb == NVFP4A_INT8) return gemm_ex_table<0, NVFP4A_INT8>()[EX - 1](A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, x, s);
  const int id = gemm_config(m, n, tb);
#define NVFP4A_CASE(cid) \
  if (id == cid) return gemm_ex_table<cid, NVFP4A_BF16>()[EX - 1](A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, x, s);
  NVFP4A_GEMM_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
  return cudaErrorInvalidConfiguration;
}

// GEMV for n columns: groups of up to 8, each its own launch on the shifted B, its scales and globals, C and
// D.  The scales of rows j0 .. j0 + 7 (j0 a multiple of 8) start at row j0's row part in either layout.
static cudaError_t launch_gemv_groups(int tb, int f, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B,
                                      int ldb, const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e0,
                                      const EpilogueEx &x0, int EX, cudaStream_t s) {
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
    const int nb = n - j0 < 8 ? n - j0 : 8;
    if (!EX) {
      if (const cudaError_t err = launch_gemv(tb, f, A, lda, SA, ldsa, b, ldb, sb, ldsb, m, nb, k, e, s)) return err;
      continue;
    }
    EpilogueEx x = x0;  // the group's columns of the extended epilogue's arrays
    if (x0.norm_parts) x.norm_parts = x0.norm_parts + (size_t)j0 * x0.norm_ld;
    if (x0.d2) { x.d2 = (uint8_t *)x0.d2 + (size_t)j0 * x0.ldd2 * esize(x0.d2_dtype); x.sq = x0.sq + (size_t)j0 * x0.ld_sq; }
    const cudaError_t err = launch_gemv_ex(tb, EX, A, lda, SA, ldsa, b, ldb, sb, ldsb, m, nb, k, e, x, s);
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

// The epilogue of nvfp4a_gemm / nvfp4a_gemm_tc (arguments validated) and the kernel instance's block scale formats
// f (B flags exist on NVFP4 B only: check_scales)
static kern::Epilogue make_epilogue(float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta, const nvfp4a_tensor *C,
                                    int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, int *f) {
  using namespace kern;
  const bool blocked = B->dtype == NVFP4A_NVFP4 || B->dtype == NVFP4A_INT8;
  const bool c_is_d = beta != 0.f && (!C || !C->data);  // D += ...
  Epilogue e;
  e.ga = A->global_scale;
  e.gb = blocked ? B->global_scale : nullptr;
  e.gf = amax_factor(A, blocked ? B : nullptr);
  e.sa = (A->flags & NVFP4A_GLOBAL_PER_ROW) ? 1 : 0;
  e.sb = (B->flags & NVFP4A_GLOBAL_PER_ROW) ? 1 : 0;
  const auto has = [](const nvfp4a_tensor *t, unsigned flag) { return (t->flags & flag) != 0; };
  *f = (has(A, NVFP4A_SCALES_SWIZZLED) ? kF_ASWZ : 0) | (has(A, NVFP4A_SCALES_UE5M3) ? kF_AUE : 0) |
       (has(B, NVFP4A_SCALES_SWIZZLED) ? kF_BSWZ : 0) | (has(B, NVFP4A_SCALES_UE5M3) ? kF_BUE : 0);
  e.alpha = alpha; e.beta = beta;
  const nvfp4a_tensor *Cx = c_is_d ? D : C;
  e.c = beta != 0.f ? Cx->data : nullptr; e.ldc = beta != 0.f ? (c_is_d ? ldd : ldc) : 0;
  e.c_dtype = beta != 0.f ? Cx->dtype : NVFP4A_FP32;
  e.d = D->data; e.ldd = ldd; e.d_dtype = D->dtype; e.amax = D->amax;
  e.bias = bias && bias->data ? bias->data : nullptr; e.bias_dtype = bias ? bias->dtype : NVFP4A_FP32;
  return e;
}

}  // namespace nvfp4a

namespace nvfp4a {
// The geometry nvfp4a_item_access works from (nvfp4a_items::geom): per item `rows` output rows; byte sizes of the
// arguments' elements (0: the argument is not read / written) and their row strides in elements.
struct ItemGeom {
  long long kind, rows, cols, m, n, k, lda, ldsa, a_img, b_img, ga, b_eb, ldb, ldsb, gb, c_arg, c_eb, ldc, d_eb, ldd, swiglu,
      bias_eb, np, norm_ld, w2_eb, d2_eb, ldd2, ld_sq, amax, b_rows;
};
static_assert(sizeof(ItemGeom) <= sizeof(((nvfp4a_items *)0)->geom), "the geometry fits");

static int esize(int t) { return t == NVFP4A_FP32 ? 4 : t == NVFP4A_INT8 ? 1 : 2; }

template <class C, int ID, int TB>
static const void *plan_ffma(int EX, nvfp4a_items *p) {
  using namespace kern;
  static const void *const kernels[] = {gemm_kernel_ptr<ID, TB, 0>(), gemm_kernel_ex_ptr<ID, TB, 1>(),
      gemm_kernel_ex_ptr<ID, TB, 2>(), gemm_kernel_ex_ptr<ID, TB, 3>(), gemm_kernel_ex_ptr<ID, TB, 4>(), gemm_kernel_ex_ptr<ID, TB, 5>()};
  p->rows = C::BM; p->cols = C::BN; p->threads = C::NT; p->smem_bytes = sizeof(GemmShared<C, TB>);
  snprintf(p->entry, sizeof p->entry, "nvfp4a::kern::gemm_item<nvfp4a::kern::GemmCfg<%d, %d, %d, %d>, %d, %d>",
           C::BM, C::BN, C::TM, C::TN, TB, EX);
  return kernels[EX];
}
static const void *plan_dp4a(int EX, bool narrow, nvfp4a_items *p) {
  using namespace kern;
  const void *const kernels[] = {dp4a_kernel_ptr<NVFP4A_INT8, 0>(narrow), dp4a_kernel_ex_ptr<1>(narrow),
      dp4a_kernel_ex_ptr<2>(narrow), dp4a_kernel_ex_ptr<3>(narrow), dp4a_kernel_ex_ptr<4>(narrow), dp4a_kernel_ex_ptr<5>(narrow)};
  p->rows = 128; p->cols = narrow ? 64 : 128; p->threads = 256;
  p->smem_bytes = narrow ? sizeof(Dp4aShared<Dp4aCfgNarrow>) : sizeof(Dp4aShared<Dp4aCfg>);
  snprintf(p->entry, sizeof p->entry, "nvfp4a::kern::gemm_dp4a_item<nvfp4a::kern::Dp4aCfgT<%d, %d, %d>, %d>",
           p->cols, narrow ? 4 : 8, narrow ? 2 : 1, EX);
  return kernels[EX];
}

// The work items of a validated call (f: the formats; ldsa / ldsb: scale strides).
static nvfp4a_status plan_items(const char *fn, int kind, int m, int n, int k, const nvfp4a_tensor *A, int lda, int ldsa,
                                const nvfp4a_tensor *B, int ldb, int ldsb, float beta, const nvfp4a_tensor *C, const nvfp4a_tensor *D,
                                const nvfp4a_tensor *bias, const kern::Epilogue &e, const kern::EpilogueEx &x, int EX, int f,
                                nvfp4a_items *p) {
  using namespace nvfp4a::kern;
  if (f || !gemv_ex_dtype(B->dtype))
    return fail(NVFP4A_NOT_SUPPORTED, "%s: work items take A with linear E4M3 scales and B BF16 or INT8", fn);
  *p = nvfp4a_items{};
  const int xt = B->dtype;
  const void *kernel = nullptr;
  p->kind = kind;
  p->threads = 256;
  if (kind == NVFP4A_ITEM_GEMV) {
    int R, MX;
    const int nb = n < 8 ? n : 8;
    if (!gemv_item_config(xt, EX, nb, &R, &MX, &kernel)) return fail(NVFP4A_NOT_SUPPORTED, "%s: no GEMV configuration", fn);
    p->rows = 8 * R;
    p->cols = n > 8 ? 8 : 0;
    p->smem_bytes = (((EX & (kEX_SWIGLU | kEX_OUT2)) ? nb * p->rows : 0) + ((EX & kEX_NORM) ? nb : 0)) * 4;
    const GemvItemArgs a{(const uint8_t *)A->data, (const uint8_t *)A->block_scale, (const uint8_t *)B->data,
                         (const uint8_t *)B->block_scale, lda, ldsa, ldb, ldsb, m, k, e, x};
    static_assert(sizeof a <= NVFP4A_ITEM_ARGS_BYTES, "the arguments fit");
    if (n <= 8) {
      snprintf(p->entry, sizeof p->entry, "nvfp4a::kern::gemv_item<nvfp4a::kern::GemvCfg<%d, %d>, %d, %d, 0, %d>", R, MX, n, xt, EX);
      snprintf(p->args_type, sizeof p->args_type, "nvfp4a::kern::GemvItemArgs");
      memcpy(p->args.bytes, &a, sizeof a); p->args_bytes = (int)sizeof a;
    } else {
      snprintf(p->entry, sizeof p->entry, "nvfp4a::kern::gemv_group_item<nvfp4a::kern::GemvCfg<%d, %d>, %d, %d, %d>", R, MX, xt, EX, n);
      snprintf(p->args_type, sizeof p->args_type, "nvfp4a::kern::GemvItemArgs");
      memcpy(p->args.bytes, &a, sizeof a); p->args_bytes = sizeof a;
    }
  } else if (kind == NVFP4A_ITEM_GEMM || kind == NVFP4A_ITEM_GEMM_DP4A) {
    if (kind == NVFP4A_ITEM_GEMM_DP4A) kernel = plan_dp4a(EX, dp4a_narrow(n), p);
    else {
      const int id = gemm_config(m, n, xt);
      if (xt == NVFP4A_INT8) kernel = plan_ffma<GemmCfg0, 0, NVFP4A_INT8>(EX, p);
      else {
#define NVFP4A_PLAN(cid) if (id == cid) kernel = plan_ffma<GemmCfg##cid, cid, NVFP4A_BF16>(EX, p);
        NVFP4A_GEMM_LIB(NVFP4A_PLAN)
#undef NVFP4A_PLAN
      }
    }
    if (!kernel) return fail(NVFP4A_NOT_SUPPORTED, "%s: no GEMM configuration", fn);
    snprintf(p->args_type, sizeof p->args_type, "nvfp4a::kern::GemmItemArgs");
    const GemmItemArgs a{(const uint8_t *)A->data, (const uint8_t *)A->block_scale, (const uint8_t *)B->data,
                         (const uint8_t *)B->block_scale, lda, ldsa, ldb, ldsb, m, n, k, e, x};
    static_assert(sizeof a <= NVFP4A_ITEM_ARGS_BYTES, "the tiled arguments fit");
    memcpy(p->args.bytes, &a, sizeof a); p->args_bytes = sizeof a;
  } else if (kind == NVFP4A_ITEM_GEMM_TC) {
    static const void *const t8[] = {tc_kernel_ptr<NVFP4A_INT8, 0>(), tc_kernel_ex_ptr<NVFP4A_INT8, 1>(), tc_kernel_ex_ptr<NVFP4A_INT8, 2>(),
                                     tc_kernel_ex_ptr<NVFP4A_INT8, 3>(), tc_kernel_ex_ptr<NVFP4A_INT8, 4>(), tc_kernel_ex_ptr<NVFP4A_INT8, 5>()};
    static const void *const th[] = {tc_kernel_ptr<NVFP4A_BF16, 0>(), tc_kernel_ex_ptr<NVFP4A_BF16, 1>(), tc_kernel_ex_ptr<NVFP4A_BF16, 2>(),
                                     tc_kernel_ex_ptr<NVFP4A_BF16, 3>(), tc_kernel_ex_ptr<NVFP4A_BF16, 4>(), tc_kernel_ex_ptr<NVFP4A_BF16, 5>()};
    kernel = (xt == NVFP4A_INT8 ? t8 : th)[EX];
    cudaFuncAttributes fa{};
    if (cudaError_t err = cudaFuncGetAttributes(&fa, kernel)) return cuda_status(err, fn);
    const bool v75 = fa.ptxVersion < 80;  // the variant the device runs (tc_launch)
    p->rows = p->cols = 128;
    p->threads = 256 + 32 * (v75 ? kTcProducers75 : kTcProducers80);
    p->smem_bytes = xt == NVFP4A_INT8 ? tc_smem_bytes<NVFP4A_INT8>(v75) : tc_smem_bytes<NVFP4A_BF16>(v75);
    snprintf(p->entry, sizeof p->entry, "nvfp4a::kern::gemm_tc_item<%d, %d>", xt, EX);
    snprintf(p->args_type, sizeof p->args_type, "nvfp4a::kern::GemmTcItemArgs");
    const GemmTcItemArgs a{(const uint8_t *)A->data, (const uint8_t *)B->data, m, n, k / 64, e, x};
    static_assert(sizeof a <= NVFP4A_ITEM_ARGS_BYTES, "the arguments fit");
    memcpy(p->args.bytes, &a, sizeof a);
    p->args_bytes = (int)sizeof a;
  } else {
    static const void *const tc8[] = {gemv_tc_kernel_ptr<NVFP4A_INT8, 0>(), gemv_tc_kernel_ex_ptr<NVFP4A_INT8, 1>(),
                                      gemv_tc_kernel_ex_ptr<NVFP4A_INT8, 2>(), gemv_tc_kernel_ex_ptr<NVFP4A_INT8, 3>(),
                                      gemv_tc_kernel_ex_ptr<NVFP4A_INT8, 4>(), gemv_tc_kernel_ex_ptr<NVFP4A_INT8, 5>()};
    static const void *const tch[] = {gemv_tc_kernel_ptr<NVFP4A_BF16, 0>(), gemv_tc_kernel_ex_ptr<NVFP4A_BF16, 1>(),
                                      gemv_tc_kernel_ex_ptr<NVFP4A_BF16, 2>(), gemv_tc_kernel_ex_ptr<NVFP4A_BF16, 3>(),
                                      gemv_tc_kernel_ex_ptr<NVFP4A_BF16, 4>(), gemv_tc_kernel_ex_ptr<NVFP4A_BF16, 5>()};
    kernel = (xt == NVFP4A_INT8 ? tc8 : tch)[EX];
    p->rows = 16;
    p->smem_bytes = gemv_tc_smem_floats(EX) * 4;
    snprintf(p->entry, sizeof p->entry, "nvfp4a::kern::gemv_tc_item<%d, 0, %d>", xt, EX);
    snprintf(p->args_type, sizeof p->args_type, "nvfp4a::kern::GemvTcItemArgs");
    const GemvTcItemArgs a{(const uint8_t *)A->data, (const uint8_t *)B->data, (const uint8_t *)B->block_scale, ldb, ldsb, m, n, k / 64,
                           e, x};
    static_assert(sizeof a <= NVFP4A_ITEM_ARGS_BYTES, "the arguments fit");
    memcpy(p->args.bytes, &a, sizeof a);
    p->args_bytes = (int)sizeof a;
  }
  const long long count = (1LL + (m - 1) / p->rows) * (p->cols ? 1LL + (n - 1) / p->cols : 1);
  if (count > INT_MAX) return fail(NVFP4A_NOT_SUPPORTED, "%s: work-item count exceeds INT_MAX", fn);
  p->items = (int)count;
  cudaFuncAttributes fa{};
  if (cudaError_t err = cudaFuncGetAttributes(&fa, kernel)) return cuda_status(err, fn);
  p->regs = fa.numRegs;
  p->arch = fa.ptxVersion;

  ItemGeom g{};
  g.kind = kind; g.rows = p->rows; g.cols = p->cols; g.m = m; g.n = n; g.k = k; g.lda = lda; g.ldsa = ldsa;
  g.a_img = kind == NVFP4A_ITEM_GEMV_TC || kind == NVFP4A_ITEM_GEMM_TC ? (long long)((k / 64 + 3) / 4) * tc::GV : 0;
  g.b_rows = kind == NVFP4A_ITEM_GEMM_DP4A ? 128 : p->cols;
  g.b_img = kind == NVFP4A_ITEM_GEMM_TC ? (long long)(k / 64) * (xt == NVFP4A_INT8 ? tc::rb_bytes<NVFP4A_INT8>() : tc::rb_bytes<NVFP4A_BF16>())
                                        : 0;  // a 128-column block's records
  g.ga = !A->global_scale ? 0 : (A->flags & NVFP4A_GLOBAL_PER_ROW) ? 2 : 1;
  g.b_eb = esize(xt); g.ldb = ldb; g.ldsb = xt == NVFP4A_INT8 ? ldsb : 0;
  g.gb = !e.gb ? 0 : e.sb ? 2 : 1;
  g.c_arg = beta != 0.f && C && C->data ? NVFP4A_ARG_C : NVFP4A_ARG_D;
  g.c_eb = e.c ? esize(e.c_dtype) : 0; g.ldc = e.ldc;
  g.d_eb = esize(D->dtype); g.ldd = e.ldd; g.swiglu = (EX & kEX_SWIGLU) != 0;
  g.bias_eb = e.bias ? esize(bias->dtype) : 0;
  g.np = (EX & kEX_NORM) ? x.norm_nparts : 0; g.norm_ld = x.norm_ld;
  if (EX & kEX_OUT2) { g.w2_eb = esize(x.w2_dtype); g.d2_eb = esize(x.d2_dtype); g.ldd2 = x.ldd2; g.ld_sq = x.ld_sq; }
  g.amax = D->amax != nullptr;
  memcpy(p->geom, &g, sizeof g);
  return NVFP4A_OK;
}
}  // namespace nvfp4a

namespace nvfp4a {
// nvfp4a_gemm and nvfp4a_gemm_ex (fn: its name in the messages; ex: the extended epilogue, its additions in EX, 0: none)
static nvfp4a_status gemm_run(const char *fn, int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B,
                              int ldb, float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                              const nvfp4a_tensor *bias, const kern::EpilogueEx &ex, int EX, cudaStream_t stream,
                              nvfp4a_items *plan = nullptr) {
  using namespace nvfp4a::kern;
  const auto dense = [](const nvfp4a_tensor *t) { return t->dtype >= NVFP4A_BF16 && t->dtype <= NVFP4A_FP32; };
  const auto blocked = [](const nvfp4a_tensor *t) { return t->dtype == NVFP4A_NVFP4 || t->dtype == NVFP4A_INT8; };
  const auto aligned = [](const void *p, unsigned a) { return ((uintptr_t)p & (a - 1)) == 0; };
  if (!A || !B || !D || !A->data || !B->data || !D->data)
    return fail(NVFP4A_INVALID_VALUE, "%s: A, B, D or their data is NULL", fn);
  if (m < 0 || n < 0 || k < 0) return fail(NVFP4A_INVALID_VALUE, "%s: negative size (m=%d n=%d k=%d)", fn, m, n, k);
  if (A->dtype != NVFP4A_NVFP4) return fail(NVFP4A_NOT_SUPPORTED, "%s: A must be NVFP4 (dtype %d)", fn, (int)A->dtype);
  if (!blocked(B) && !dense(B))
    return fail(NVFP4A_INVALID_VALUE, "%s: invalid B dtype %d", fn, (int)B->dtype);
  if (!dense(D)) return fail(NVFP4A_INVALID_VALUE, "%s: D must be BF16, FP16 or FP32 (dtype %d)", fn, (int)D->dtype);
  if (m == 0 || n == 0) return NVFP4A_OK;
  if (k % 64) return fail(NVFP4A_INVALID_VALUE, "%s: k = %d is not a multiple of 64", fn, k);
  if (lda % 64 || ldb % 64) return fail(NVFP4A_INVALID_VALUE, "%s: lda = %d / ldb = %d must be multiples of 64", fn, lda, ldb);
  if (lda < k || ldb < k) return fail(NVFP4A_INVALID_VALUE, "%s: lda = %d / ldb = %d < k = %d", fn, lda, ldb, k);
  if (!aligned(A->data, 16) || !aligned(B->data, 16)) return fail(NVFP4A_INVALID_VALUE, "%s: A and B must be 16-byte aligned", fn);
  int ldsa = 0, ldsb = 0;
  for (const nvfp4a_tensor *t : {A, B}) {
    if (!blocked(t)) {
      if (t->flags) return fail(NVFP4A_INVALID_VALUE, "%s: flags are for NVFP4 tensors (dtype %d)", fn, (int)t->dtype);
      continue;
    }
    const unsigned sa = t->dtype == NVFP4A_INT8 ? 16 : 4;
    if (!t->block_scale) return fail(NVFP4A_INVALID_VALUE, "%s: NVFP4 / INT8 operand without block_scale", fn);
    if (!aligned(t->block_scale, sa) || t->ld_scale % 4 || (t->ld_scale && t->ld_scale < k / 16))
      return fail(NVFP4A_INVALID_VALUE, "%s: block scales must be %u-byte aligned with ld_scale a multiple of 4 "
                  "and >= k / 16 (ld_scale = %d)", fn, sa, t->ld_scale);
    if (nvfp4a_status st = nvfp4a::check_scales(fn, t == A ? "A" : "B", t, t == A ? m : n, k, t == A ? &ldsa : &ldsb))
      return st;
  }
  const int drows = (EX & kEX_SWIGLU) ? m / 2 : m;  // D's rows
  if (ldd < drows) return fail(NVFP4A_INVALID_VALUE, "%s: ldd = %d < %d", fn, ldd, drows);
  if (bias && bias->data && !dense(bias)) return fail(NVFP4A_INVALID_VALUE, "%s: invalid bias dtype %d", fn, (int)bias->dtype);
  if (beta != 0.f && C && C->data && !dense(C)) return fail(NVFP4A_INVALID_VALUE, "%s: invalid C dtype %d", fn, (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(NVFP4A_INVALID_VALUE, "%s: ldc = %d < m = %d", fn, ldc, m);
  if (nvfp4a_status st = nvfp4a::check_device()) return st;
  int f;
  const Epilogue e = nvfp4a::make_epilogue(alpha, A, B, beta, C, ldc, D, ldd, bias, &f);

  const uint8_t *a = (const uint8_t *)A->data, *b = (const uint8_t *)B->data;
  const uint8_t *sa = (const uint8_t *)A->block_scale, *sb = (const uint8_t *)B->block_scale;
  if (EX) {
    if (f & (kF_ASWZ | kF_AUE)) return fail(NVFP4A_NOT_SUPPORTED, "%s: the extended epilogue takes A with linear E4M3 scales", fn);
    if (!gemv_ex_dtype(B->dtype)) return fail(NVFP4A_NOT_SUPPORTED, "%s: the extended epilogue takes B BF16 or INT8", fn);
  }
  const bool gemv = k == 0 || n <= kGemvMaxN[B->dtype];
  const int kind = gemv ? NVFP4A_ITEM_GEMV : B->dtype == NVFP4A_INT8 && device_info().sass >= 61
                                               ? NVFP4A_ITEM_GEMM_DP4A : NVFP4A_ITEM_GEMM;
  if (plan) return plan_items(fn, kind, m, n, k, A, lda, ldsa, B, ldb, ldsb, beta, C, D, bias, e, ex, EX, f, plan);
  const cudaError_t err = gemv
                              ? launch_gemv_groups(B->dtype, f, a, lda, sa, ldsa, b, ldb, sb, ldsb, m, n, k, e, ex, EX, stream)
                              : launch_gemm(B->dtype, f, a, lda, sa, ldsa, b, ldb, sb, ldsb, m, n, k, e, ex, EX, stream);
  return nvfp4a::cuda_status(err, fn);
}

// The extended epilogue of a call (validated): its additions EX and the device descriptor
static nvfp4a_status make_epilogue_ex(const char *fn, const nvfp4a_epilogue *x, int m, int n, float beta, int ldd, kern::EpilogueEx *ex,
                                      int *EX) {
  using namespace nvfp4a::kern;
  *ex = EpilogueEx{};
  *EX = 0;
  if (!x) return NVFP4A_OK;
  const auto dense = [](nvfp4a_dtype t) { return t >= NVFP4A_BF16 && t <= NVFP4A_FP32; };
  if (x->norm_parts) {
    if (x->norm_nparts <= 0 || x->norm_ld < x->norm_nparts || !(x->norm_inv_n > 0.f) || !(x->norm_eps >= 0.f))
      return fail(NVFP4A_INVALID_VALUE, "%s: the column scale needs norm_nparts > 0, norm_ld >= norm_nparts, norm_inv_n > 0, "
                  "norm_eps >= 0", fn);
    ex->norm_parts = x->norm_parts; ex->norm_nparts = x->norm_nparts; ex->norm_ld = x->norm_ld;
    ex->norm_inv_n = x->norm_inv_n; ex->norm_eps = x->norm_eps;
    *EX |= kEX_NORM;
  }
  if (x->swiglu) {
    if (m % 16 || beta != 0.f || x->d2) return fail(NVFP4A_INVALID_VALUE, "%s: SwiGLU needs m a multiple of 16, no C, no second output", fn);
    if (ldd < m / 2) return fail(NVFP4A_INVALID_VALUE, "%s: SwiGLU: ldd = %d < m / 2 = %d", fn, ldd, m / 2);
    *EX |= kEX_SWIGLU;
  }
  if (x->d2) {
    if (m % 16 || !x->w2 || !x->sq || !dense(x->d2_dtype) || !dense(x->w2_dtype) || x->ldd2 < m || x->ld_sq < m / 16)
      return fail(NVFP4A_INVALID_VALUE, "%s: the second output needs m a multiple of 16, w2 and sq, BF16 / FP16 / FP32 d2 and w2, "
                  "ldd2 >= m, ld_sq >= m / 16", fn);
    ex->d2 = x->d2; ex->d2_dtype = x->d2_dtype; ex->ldd2 = x->ldd2; ex->w2 = x->w2; ex->w2_dtype = x->w2_dtype;
    ex->sq = x->sq; ex->ld_sq = x->ld_sq;
    *EX |= kEX_OUT2;
  }
  (void)n;
  return NVFP4A_OK;
}
}  // namespace nvfp4a

extern "C" nvfp4a_status nvfp4a_gemm(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda,
                                     const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                                     const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, cudaStream_t stream) {
  nvfp4a::clear_error();
  return nvfp4a::gemm_run("nvfp4a_gemm", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, bias, {}, 0, stream);
}

extern "C" nvfp4a_status nvfp4a_gemm_ex(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B, int ldb,
                                        float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                        const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, cudaStream_t stream) {
  nvfp4a::clear_error();
  nvfp4a::kern::EpilogueEx x;
  int EX;
  if (nvfp4a_status st = nvfp4a::make_epilogue_ex("nvfp4a_gemm_ex", ex, m, n, beta, ldd, &x, &EX)) return st;
  return nvfp4a::gemm_run("nvfp4a_gemm_ex", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, bias, x, EX, stream);
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

// ---- tensor-core GEMM (src/tensors) -----------------------------------------------------------------------------
// the record bytes of an image kind of dtype (0: no such image; A's images: GV, a group of 4 k-tiles of a slice)
static int tc_record(int operand, nvfp4a_dtype dtype) {
  using namespace nvfp4a::kern;
  if (operand == NVFP4A_TC_A16 || operand == NVFP4A_TC_A8) return dtype == NVFP4A_NVFP4 ? tc::GV : 0;
  if (operand != NVFP4A_TC_B) return 0;
  return dtype == NVFP4A_NVFP4 ? tc::RB4 : dtype == NVFP4A_INT8 ? tc::RB8 : dtype == NVFP4A_BF16 ? tc::RBH : 0;
}
// whether the current device runs the tensor-core GEMM with B of dtype tb
static nvfp4a_status tc_device(const char *fn, nvfp4a_dtype tb) {
  using namespace nvfp4a;
  if (nvfp4a_status st = check_device()) return st;
  const kern::DeviceInfo &d = kern::device_info();
  if (!d.tc || (tb == NVFP4A_BF16 && d.ptx < 80))
    return fail(NVFP4A_NOT_SUPPORTED, "%s: no tensor-core path on this device (sm_75 .. sm_89; BF16 B sm_80+; NVFP4A_DP4A unset)", fn);
  return NVFP4A_OK;
}

extern "C" size_t nvfp4a_tc_image_bytes(int operand, nvfp4a_dtype dtype, int rows, int k) {
  const int rec = tc_record(operand, dtype);
  if (!rec || rows < 0 || k < 0 || k % 64) return 0;
  return operand == NVFP4A_TC_B ? nvfp4a::kern::tc_image_bytes(rows, k, rec) : nvfp4a::kern::tc_a_image_bytes(rows, k);
}

extern "C" nvfp4a_status nvfp4a_tc_image(int operand, const nvfp4a_tensor *src, int ld, int rows, int k, void *image,
                                         cudaStream_t stream) {
  using namespace nvfp4a;
  using namespace nvfp4a::kern;
  clear_error();
  const auto aligned = [](const void *p, unsigned a) { return ((uintptr_t)p & (a - 1)) == 0; };
  if (!src || !src->data || !image) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: src, its data or image is NULL");
  if (!tc_record(operand, src->dtype))
    return fail(NVFP4A_NOT_SUPPORTED, "nvfp4a_tc_image: no image of kind %d with dtype %d (A16, A8: NVFP4; B: NVFP4, INT8, BF16)",
                operand, (int)src->dtype);
  if (rows < 0 || k < 0 || k % 64) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: rows = %d, k = %d (k a multiple of 64)", rows, k);
  if (rows == 0 || k == 0) return NVFP4A_OK;
  if (ld % 64 || ld < k) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: ld = %d (a multiple of 64, >= k = %d)", ld, k);
  if (!aligned(src->data, 16) || !aligned(image, 16)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: src and image must be 16-byte aligned");
  int ldsc = 0;
  if (src->dtype == NVFP4A_BF16) {
    if (src->flags) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: flags are for NVFP4 tensors");
  } else {
    const unsigned sa = src->dtype == NVFP4A_INT8 ? 16 : 4;
    if (!src->block_scale) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: NVFP4 / INT8 src without block_scale");
    if (!aligned(src->block_scale, sa) || src->ld_scale % 4 || (src->ld_scale && src->ld_scale < k / 16))
      return fail(NVFP4A_INVALID_VALUE, "nvfp4a_tc_image: block scales must be %u-byte aligned with ld_scale a multiple of 4 "
                  "and >= k / 16 (ld_scale = %d)", sa, src->ld_scale);
    if (nvfp4a_status st = check_scales("nvfp4a_tc_image", "src", src, rows, k, &ldsc)) return st;
  }
  // a weight's image is a rearrangement: any device builds it (e.g. to store it); B's serve the GEMM of this device
  if (nvfp4a_status st = operand == NVFP4A_TC_B ? tc_device("nvfp4a_tc_image", src->dtype) : check_device()) return st;
  const uint8_t *data = (const uint8_t *)src->data, *sc = (const uint8_t *)src->block_scale;
  uint8_t *img = (uint8_t *)image;
  const cudaError_t err = operand == NVFP4A_TC_B ? tc_image_b(src->dtype, data, ld, sc, ldsc, rows, k, img, stream)
                                                 : tc_image_a(operand == NVFP4A_TC_A16, data, ld, sc, ldsc, rows, k, img, stream);
  return cuda_status(err, "nvfp4a_tc_image: kernel launch");
}

// the checks of a weight image argument
static nvfp4a_status check_weight_image(const char *fn, int kind, const void *image, int rows, int k) {
  using namespace nvfp4a;
  if (kind != NVFP4A_TC_A16 && kind != NVFP4A_TC_A8) return fail(NVFP4A_INVALID_VALUE, "%s: image kind %d (NVFP4A_TC_A16 or NVFP4A_TC_A8)", fn, kind);
  if (!image) return fail(NVFP4A_INVALID_VALUE, "%s: image is NULL", fn);
  if (rows < 0 || k < 0 || k % 64) return fail(NVFP4A_INVALID_VALUE, "%s: rows = %d, k = %d (k a multiple of 64)", fn, rows, k);
  if ((uintptr_t)image & 15) return fail(NVFP4A_INVALID_VALUE, "%s: the image must be 16-byte aligned", fn);
  return NVFP4A_OK;
}

extern "C" nvfp4a_status nvfp4a_tc_image_convert(int from, int to, const void *image, int rows, int k, void *dst, cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  const char *fn = "nvfp4a_tc_image_convert";
  if (nvfp4a_status st = check_weight_image(fn, from, image, rows, k)) return st;
  if (nvfp4a_status st = check_weight_image(fn, to, dst, rows, k)) return st;
  if (from == to) return fail(NVFP4A_INVALID_VALUE, "%s: the same kind", fn);
  if (rows == 0 || k == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  return cuda_status(kern::tc_image_a_convert(from == NVFP4A_TC_A16, (const uint8_t *)image, rows, k, (uint8_t *)dst, stream),
                     "nvfp4a_tc_image_convert: kernel launch");
}

extern "C" nvfp4a_status nvfp4a_tc_image_to_raw(int from, const void *image, int rows, int k, const nvfp4a_tensor *dst, int ld,
                                                cudaStream_t stream) {
  using namespace nvfp4a;
  clear_error();
  const char *fn = "nvfp4a_tc_image_to_raw";
  if (nvfp4a_status st = check_weight_image(fn, from, image, rows, k)) return st;
  if (!dst || !dst->data || !dst->block_scale || dst->dtype != NVFP4A_NVFP4)
    return fail(NVFP4A_INVALID_VALUE, "%s: dst must be an NVFP4 tensor with data and block scales", fn);
  const int lds = dst->ld_scale ? dst->ld_scale : k / 16;
  if (ld % 64 || ld < k || lds % 4 || lds < k / 16 || ((uintptr_t)dst->data & 15) || ((uintptr_t)dst->block_scale & 3))
    return fail(NVFP4A_INVALID_VALUE, "%s: ld = %d (a multiple of 64, >= k), ld_scale = %d (a multiple of 4, >= k / 16), data 16-byte "
                "and scales 4-byte aligned", fn, ld, dst->ld_scale);
  if (dst->flags & NVFP4A_SCALES_SWIZZLED) return fail(NVFP4A_INVALID_VALUE, "%s: dst scales are linear", fn);
  if (rows == 0 || k == 0) return NVFP4A_OK;
  if (nvfp4a_status st = check_device()) return st;
  return cuda_status(kern::tc_image_a_raw(from == NVFP4A_TC_A16, (const uint8_t *)image, rows, k, (uint8_t *)dst->data, ld / 2,
                                          (uint8_t *)dst->block_scale, lds, stream),
                     "nvfp4a_tc_image_to_raw: kernel launch");
}

extern "C" size_t nvfp4a_tc_workspace_bytes(void) { return nvfp4a::kern::tc_ws_bytes(nvfp4a::kern::device_info().sms); }

namespace nvfp4a {
// Empty reductions use the existing slice epilogue: its zero-k path reads neither operand image.
static cudaError_t tc_zero(int tb, int m, int n, const uint8_t *ia, const uint8_t *ib, const kern::Epilogue &e0,
                            const kern::EpilogueEx &x0, int EX, cudaStream_t stream) {
  using namespace kern;
  for (int j0 = 0; j0 < n; j0 += 8) {
    const int nb = n - j0 < 8 ? n - j0 : 8;
    Epilogue e = e0;
    EpilogueEx x = x0;
    e.d = (uint8_t *)e0.d + (size_t)j0 * e.ldd * esize(e.d_dtype);
    if (e.c) e.c = (const uint8_t *)e0.c + (size_t)j0 * e.ldc * esize(e.c_dtype);
    if (e.sb) e.gb += j0;
    if (x.norm_parts) x.norm_parts += (size_t)j0 * x.norm_ld;
    if (x.d2) {
      x.d2 = (uint8_t *)x.d2 + (size_t)j0 * x.ldd2 * esize(x.d2_dtype);
      x.sq += (size_t)j0 * x.ld_sq;
    }
    cudaError_t err;
    if (EX) {
      const GvExFn g = (tb == NVFP4A_INT8 ? kGvEx8 : kGvExH)[EX - 1];
      err = g(ia, ib, 0, nullptr, 0, m, nb, 0, e, x, stream);
    } else {
      const GvFn g = tb == NVFP4A_INT8 ? kGv8[0] : tb == NVFP4A_NVFP4 ? kGv4[0] : kGvH[0];
      err = g(ia, ib, 0, nullptr, 0, m, nb, 0, e, stream);
    }
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

static nvfp4a_status gemm_tc_run(const char *fn, int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B,
                                 float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                 void *workspace, const kern::EpilogueEx &x, int EX, cudaStream_t stream,
                                 nvfp4a_items *plan = nullptr) {
  using namespace nvfp4a::kern;
  const auto dense = [](const nvfp4a_tensor *t) { return t->dtype >= NVFP4A_BF16 && t->dtype <= NVFP4A_FP32; };
  const auto aligned = [](const void *p, unsigned a) { return ((uintptr_t)p & (a - 1)) == 0; };
  if (!A || !B || !D || !A->data || !B->data || !D->data || (!workspace && !plan))
    return fail(NVFP4A_INVALID_VALUE, "%s: A, B, D, their data or the workspace is NULL", fn);
  if (m < 0 || n < 0 || k < 0) return fail(NVFP4A_INVALID_VALUE, "%s: negative size (m=%d n=%d k=%d)", fn, m, n, k);
  if (A->dtype != NVFP4A_NVFP4) return fail(NVFP4A_NOT_SUPPORTED, "%s: A must be NVFP4 (dtype %d)", fn, (int)A->dtype);
  if (!tc_record(NVFP4A_TC_B, B->dtype))
    return fail(NVFP4A_NOT_SUPPORTED, "%s: B must be NVFP4, INT8 or BF16 (dtype %d)", fn, (int)B->dtype);
  if (!dense(D)) return fail(NVFP4A_INVALID_VALUE, "%s: D must be BF16, FP16 or FP32 (dtype %d)", fn, (int)D->dtype);
  if (m == 0 || n == 0) return NVFP4A_OK;
  if (k % 64) return fail(NVFP4A_INVALID_VALUE, "%s: k = %d is not a multiple of 64", fn, k);
  if (!aligned(A->data, 16) || !aligned(B->data, 16) || !aligned(workspace, 16))
    return fail(NVFP4A_INVALID_VALUE, "%s: the images and the workspace must be 16-byte aligned", fn);
  for (const nvfp4a_tensor *t : {A, B}) {  // the flags and globals (the scales themselves are in the images)
    const unsigned known = NVFP4A_SCALES_SWIZZLED | NVFP4A_SCALES_UE5M3 | NVFP4A_GLOBAL_AMAX | NVFP4A_GLOBAL_PER_ROW;
    if (t->dtype != NVFP4A_NVFP4 && t->flags) return fail(NVFP4A_INVALID_VALUE, "%s: flags are for NVFP4 tensors", fn);
    if (t->flags & ~known) return fail(NVFP4A_INVALID_VALUE, "%s: unknown flags 0x%x", fn, t->flags & ~known);
    if (t->dtype == NVFP4A_NVFP4 && scale_max_of(t) == 0.f) return fail(NVFP4A_INVALID_VALUE, "%s: scale_max = %d", fn, t->scale_max);
    if ((t->flags & NVFP4A_GLOBAL_PER_ROW) && !t->global_scale)
      return fail(NVFP4A_INVALID_VALUE, "%s: per-row globals but no global_scale", fn);
  }
  const int drows = (EX & kEX_SWIGLU) ? m / 2 : m;
  if (ldd < drows) return fail(NVFP4A_INVALID_VALUE, "%s: ldd = %d < %d", fn, ldd, drows);
  if (bias && bias->data && !dense(bias)) return fail(NVFP4A_INVALID_VALUE, "%s: invalid bias dtype %d", fn, (int)bias->dtype);
  if (beta != 0.f && C && C->data && !dense(C)) return fail(NVFP4A_INVALID_VALUE, "%s: invalid C dtype %d", fn, (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(NVFP4A_INVALID_VALUE, "%s: ldc = %d < m = %d", fn, ldc, m);
  if (nvfp4a_status st = tc_device(fn, B->dtype)) return st;
  int f;
  const Epilogue e = make_epilogue(alpha, A, B, beta, C, ldc, D, ldd, bias, &f);
  const int fi = (f >> 1 & 1) | (f >> 2 & 2);  // kF_AUE, kF_BUE
  const uint8_t *ia = (const uint8_t *)A->data, *ib = (const uint8_t *)B->data;
  if (plan) return plan_items(fn, NVFP4A_ITEM_GEMM_TC, m, n, k, A, 0, 0, B, 0, 0, beta, C, D, bias, e, x, EX, f, plan);
  if (EX) {
    if (f & kF_AUE) return fail(NVFP4A_NOT_SUPPORTED, "%s: the extended epilogue takes A with E4M3 scales", fn);
    if (!gemv_ex_dtype(B->dtype)) return fail(NVFP4A_NOT_SUPPORTED, "%s: the extended epilogue takes B BF16 or INT8", fn);
  }
  if (k == 0) return cuda_status(tc_zero(B->dtype, m, n, ia, ib, e, x, EX, stream), fn);
  if (EX) {
    const TcExFn g = (B->dtype == NVFP4A_INT8 ? kTcEx8 : kTcExH)[EX - 1];
    return cuda_status(g(ia, ib, m, n, k, e, x, (uint8_t *)workspace, stream), "nvfp4a_gemm_tc_ex: kernel launch");
  }
  const TcFn g = B->dtype == NVFP4A_INT8 ? kTc8[fi & 1] : B->dtype == NVFP4A_NVFP4 ? kTc4[fi] : kTcH[fi & 1];
  return cuda_status(g(ia, ib, m, n, k, e, (uint8_t *)workspace, stream), "nvfp4a_gemm_tc: kernel launch");
}
}  // namespace nvfp4a

extern "C" nvfp4a_status nvfp4a_gemm_tc(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta,
                                        const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                        void *workspace, cudaStream_t stream) {
  nvfp4a::clear_error();
  return nvfp4a::gemm_tc_run("nvfp4a_gemm_tc", m, n, k, alpha, A, B, beta, C, ldc, D, ldd, bias, workspace, {}, 0, stream);
}

extern "C" nvfp4a_status nvfp4a_gemm_tc_ex(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta,
                                           const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                           const nvfp4a_epilogue *ex, void *workspace, cudaStream_t stream) {
  nvfp4a::clear_error();
  nvfp4a::kern::EpilogueEx x;
  int EX;
  if (nvfp4a_status st = nvfp4a::make_epilogue_ex("nvfp4a_gemm_tc_ex", ex, m, n, beta, ldd, &x, &EX)) return st;
  return nvfp4a::gemm_tc_run("nvfp4a_gemm_tc_ex", m, n, k, alpha, A, B, beta, C, ldc, D, ldd, bias, workspace, x, EX, stream);
}


namespace nvfp4a {
static nvfp4a_status gemv_tc_run(const char *fn, int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B,
                                 int ldb, float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                 const nvfp4a_tensor *bias, const kern::EpilogueEx &x, int EX, cudaStream_t stream,
                                 nvfp4a_items *plan = nullptr) {
  using namespace nvfp4a::kern;
  const auto dense = [](const nvfp4a_tensor *t) { return t->dtype >= NVFP4A_BF16 && t->dtype <= NVFP4A_FP32; };
  const auto aligned = [](const void *p, unsigned a) { return ((uintptr_t)p & (a - 1)) == 0; };
  if (!A || !B || !D || !A->data || !B->data || !D->data) return fail(NVFP4A_INVALID_VALUE, "%s: A, B, D or their data is NULL", fn);
  if (m < 0 || k < 0 || n < 1 || n > 8) return fail(NVFP4A_INVALID_VALUE, "%s: m = %d, n = %d, k = %d (1 <= n <= 8)", fn, m, n, k);
  if (A->dtype != NVFP4A_NVFP4) return fail(NVFP4A_NOT_SUPPORTED, "%s: A must be NVFP4 (dtype %d)", fn, (int)A->dtype);
  if (!tc_record(NVFP4A_TC_B, B->dtype))
    return fail(NVFP4A_NOT_SUPPORTED, "%s: B must be NVFP4, INT8 or BF16 (dtype %d)", fn, (int)B->dtype);
  if (!dense(D)) return fail(NVFP4A_INVALID_VALUE, "%s: D must be BF16, FP16 or FP32 (dtype %d)", fn, (int)D->dtype);
  if (m == 0) return NVFP4A_OK;
  if (k % 64) return fail(NVFP4A_INVALID_VALUE, "%s: k = %d is not a multiple of 64", fn, k);
  if (ldb % 64 || ldb < k) return fail(NVFP4A_INVALID_VALUE, "%s: ldb = %d (a multiple of 64, >= k = %d)", fn, ldb, k);
  if (!aligned(A->data, 16) || !aligned(B->data, 16)) return fail(NVFP4A_INVALID_VALUE, "%s: A's image and B must be 16-byte aligned", fn);
  const unsigned known = NVFP4A_SCALES_SWIZZLED | NVFP4A_SCALES_UE5M3 | NVFP4A_GLOBAL_AMAX | NVFP4A_GLOBAL_PER_ROW;
  if (A->flags & ~known) return fail(NVFP4A_INVALID_VALUE, "%s: unknown A flags 0x%x", fn, A->flags & ~known);
  if (scale_max_of(A) == 0.f) return fail(NVFP4A_INVALID_VALUE, "%s: A has scale_max = %d", fn, A->scale_max);
  if ((A->flags & NVFP4A_GLOBAL_PER_ROW) && !A->global_scale) return fail(NVFP4A_INVALID_VALUE, "%s: per-row globals but no global_scale", fn);
  int ldsb = 0;
  if (B->dtype == NVFP4A_BF16) {
    if (B->flags) return fail(NVFP4A_INVALID_VALUE, "%s: flags are for NVFP4 tensors", fn);
  } else {
    const unsigned sa = B->dtype == NVFP4A_INT8 ? 16 : 4;
    if (!B->block_scale) return fail(NVFP4A_INVALID_VALUE, "%s: NVFP4 / INT8 B without block_scale", fn);
    if (!aligned(B->block_scale, sa) || B->ld_scale % 4 || (B->ld_scale && B->ld_scale < k / 16))
      return fail(NVFP4A_INVALID_VALUE, "%s: B's block scales must be %u-byte aligned with ld_scale a multiple of 4 "
                  "and >= k / 16 (ld_scale = %d)", fn, sa, B->ld_scale);
    if (nvfp4a_status st = check_scales(fn, "B", B, n, k, &ldsb)) return st;
  }
  const int drows = (EX & kEX_SWIGLU) ? m / 2 : m;
  if (ldd < drows) return fail(NVFP4A_INVALID_VALUE, "%s: ldd = %d < %d", fn, ldd, drows);
  if (bias && bias->data && !dense(bias)) return fail(NVFP4A_INVALID_VALUE, "%s: invalid bias dtype %d", fn, (int)bias->dtype);
  if (beta != 0.f && C && C->data && !dense(C)) return fail(NVFP4A_INVALID_VALUE, "%s: invalid C dtype %d", fn, (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(NVFP4A_INVALID_VALUE, "%s: ldc = %d < m = %d", fn, ldc, m);
  if (nvfp4a_status st = tc_device(fn, B->dtype)) return st;
  int f;
  const Epilogue e = make_epilogue(alpha, A, B, beta, C, ldc, D, ldd, bias, &f);
  // NVFP4 x scales: swizzled, or linear -- a group's 4 k-tiles in one 16-byte load where every row's start is 16-byte
  // aligned and k a multiple of 256 (no partial group)
  const int xs = (f & kF_BSWZ) ? 1 : (ldsb % 16 == 0 && aligned(B->block_scale, 16) && k % 256 == 0) ? 2 : 0;
  const uint8_t *ia = (const uint8_t *)A->data, *xb = (const uint8_t *)B->data, *sxb = (const uint8_t *)B->block_scale;
  if (plan) return plan_items(fn, NVFP4A_ITEM_GEMV_TC, m, n, k, A, 0, 0, B, ldb, ldsb, beta, C, D, bias, e, x, EX, f, plan);
  if (EX) {
    if (f & kF_AUE) return fail(NVFP4A_NOT_SUPPORTED, "%s: the extended epilogue takes A with E4M3 scales", fn);
    if (!gemv_ex_dtype(B->dtype)) return fail(NVFP4A_NOT_SUPPORTED, "%s: the extended epilogue takes B BF16 or INT8", fn);
    const GvExFn g = (B->dtype == NVFP4A_INT8 ? kGvEx8 : kGvExH)[EX - 1];
    return cuda_status(g(ia, xb, ldb, sxb, ldsb, m, n, k, e, x, stream), "nvfp4a_gemv_tc_ex: kernel launch");
  }
  const GvFn g = B->dtype == NVFP4A_INT8    ? kGv8[(f >> 1) & 1]
                 : B->dtype == NVFP4A_NVFP4 ? kGv4[4 * xs + ((f >> 2) & 2) + ((f >> 1) & 1)]
                                            : kGvH[(f >> 1) & 1];
  return cuda_status(g(ia, xb, ldb, sxb, ldsb, m, n, k, e, stream), "nvfp4a_gemv_tc: kernel launch");
}
}  // namespace nvfp4a

extern "C" nvfp4a_status nvfp4a_gemv_tc(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb,
                                        float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                        const nvfp4a_tensor *bias, cudaStream_t stream) {
  nvfp4a::clear_error();
  return nvfp4a::gemv_tc_run("nvfp4a_gemv_tc", m, n, k, alpha, A, B, ldb, beta, C, ldc, D, ldd, bias, {}, 0, stream);
}

extern "C" nvfp4a_status nvfp4a_gemv_tc_ex(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb,
                                           float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                           const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, cudaStream_t stream) {
  nvfp4a::clear_error();
  nvfp4a::kern::EpilogueEx x;
  int EX;
  if (nvfp4a_status st = nvfp4a::make_epilogue_ex("nvfp4a_gemv_tc_ex", ex, m, n, beta, ldd, &x, &EX)) return st;
  return nvfp4a::gemv_tc_run("nvfp4a_gemv_tc_ex", m, n, k, alpha, A, B, ldb, beta, C, ldc, D, ldd, bias, x, EX, stream);
}


extern "C" nvfp4a_status nvfp4a_gemm_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B,
                                              int ldb, float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                              const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items) {
  nvfp4a::clear_error();
  if (!items) return nvfp4a::fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm_ex_items: items is NULL");
  if (m <= 0 || n <= 0) return nvfp4a::fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm_ex_items: m = %d, n = %d (no work)", m, n);
  nvfp4a::kern::EpilogueEx x;
  int EX;
  if (nvfp4a_status st = nvfp4a::make_epilogue_ex("nvfp4a_gemm_ex_items", ex, m, n, beta, ldd, &x, &EX)) return st;
  return nvfp4a::gemm_run("nvfp4a_gemm_ex_items", m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd, bias, x, EX, 0, items);
}

extern "C" nvfp4a_status nvfp4a_gemv_tc_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb,
                                                 float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                                 const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items) {
  nvfp4a::clear_error();
  if (!items) return nvfp4a::fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemv_tc_ex_items: items is NULL");
  if (m <= 0) return nvfp4a::fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemv_tc_ex_items: m = %d (no work)", m);
  nvfp4a::kern::EpilogueEx x;
  int EX;
  if (nvfp4a_status st = nvfp4a::make_epilogue_ex("nvfp4a_gemv_tc_ex_items", ex, m, n, beta, ldd, &x, &EX)) return st;
  return nvfp4a::gemv_tc_run("nvfp4a_gemv_tc_ex_items", m, n, k, alpha, A, B, ldb, beta, C, ldc, D, ldd, bias, x, EX, 0, items);
}

extern "C" int nvfp4a_item_access(const nvfp4a_items *items, int item, nvfp4a_item_range *ranges, int max) {
  if (!items || item < 0 || item >= items->items || max < 0 || (max && !ranges)) return -1;
  nvfp4a::ItemGeom g;
  memcpy(&g, items->geom, sizeof g);
  int cnt = 0;
  const auto add = [&](int arg, int write, long long offset, long long bytes, long long stride, long long rows) {
    if (!bytes || !rows) return;
    if (cnt < max) ranges[cnt] = nvfp4a_item_range{arg, write, offset, bytes, stride, (int)rows};
    ++cnt;
  };
  // the item's row block mb and column block nb (tiles: the kernel's raster, groups of 8 row blocks)
  long long mb = item, nb = 0;
  if (g.kind == NVFP4A_ITEM_GEMM_TC) {
    const long long nt = (g.n + g.cols - 1) / g.cols, mt = (g.m + g.rows - 1) / g.rows, grp = item / (8 * nt), first = grp * 8;
    const long long gsz = mt - first < 8 ? mt - first : 8, t = item - grp * 8 * nt;
    mb = first + t % gsz; nb = t / gsz;
  } else if (g.cols) {
    const long long mt = (g.m + g.rows - 1) / g.rows;
    mb = item % mt; nb = item / mt;
  }
  const long long r0 = mb * g.rows, rr = g.m - r0 < g.rows ? g.m - r0 : g.rows;
  const long long c0 = g.cols ? nb * g.cols : 0, cc = !g.cols ? g.n : g.n - c0 < g.cols ? g.n - c0 : g.cols;
  const bool raw = g.kind == NVFP4A_ITEM_GEMV || g.kind == NVFP4A_ITEM_GEMM || g.kind == NVFP4A_ITEM_GEMM_DP4A;
  if (raw) {
    add(NVFP4A_ARG_A, 0, r0 * (g.lda / 2), g.k / 2, g.lda / 2, rr);
    add(NVFP4A_ARG_A_SCALES, 0, r0 * g.ldsa, g.k / 16, g.ldsa, rr);
    if (r0 && rr < g.rows) {  // gemv_warp: inactive rows load row 0.
      add(NVFP4A_ARG_A, 0, 0, g.k / 2, 0, 1);
      add(NVFP4A_ARG_A_SCALES, 0, 0, g.k / 16, 0, 1);
    }
  } else {  // the image's slices of the rows (a tile: its 8)
    const long long sl = g.rows / 16;
    add(NVFP4A_ARG_A, 0, mb * sl * g.a_img, sl * g.a_img, 0, 1);
  }
  if (g.ga) add(NVFP4A_ARG_A_GLOBAL, 0, g.ga == 2 ? r0 * 4 : 0, g.ga == 2 ? rr * 4 : 4, 0, 1);
  if (g.kind != NVFP4A_ITEM_GEMV && g.kind != NVFP4A_ITEM_GEMV_TC && g.ga == 2 && r0 && rr < g.rows)
    add(NVFP4A_ARG_A_GLOBAL, 0, 0, 4, 0, 1);  // gemm_tc_body's padded epilogue rows.
  if (g.b_img) add(NVFP4A_ARG_B, 0, nb * g.b_img, g.b_img, 0, 1);
  else {
    const long long bc = g.b_rows ? (g.n - c0 < g.b_rows ? g.n - c0 : g.b_rows) : cc;
    add(NVFP4A_ARG_B, 0, c0 * g.ldb * g.b_eb, g.k * g.b_eb, g.ldb * g.b_eb, bc);
    if (g.ldsb) add(NVFP4A_ARG_B_SCALES, 0, c0 * g.ldsb * 4, g.k / 16 * 4, g.ldsb * 4, bc);
    if (raw && g.kind != NVFP4A_ITEM_GEMV && c0 && bc < g.b_rows) {
      add(NVFP4A_ARG_B, 0, 0, g.k * g.b_eb, 0, 1);
      if (g.ldsb) add(NVFP4A_ARG_B_SCALES, 0, 0, g.k / 16 * 4, 0, 1);
    }
  }
  if (g.gb) add(NVFP4A_ARG_B_GLOBAL, 0, g.gb == 2 ? c0 * 4 : 0, g.gb == 2 ? cc * 4 : 4, 0, 1);
  if (g.gb == 2 && c0 && cc < g.cols && g.kind != NVFP4A_ITEM_GEMV)
    add(NVFP4A_ARG_B_GLOBAL, 0, 0, 4, 0, 1);
  if (g.bias_eb) add(NVFP4A_ARG_BIAS, 0, r0 * g.bias_eb, rr * g.bias_eb, 0, 1);
  if (g.kind == NVFP4A_ITEM_GEMM_DP4A && g.bias_eb && !g.c_eb && !g.np && !g.swiglu && !g.d2_eb && r0 && rr < g.rows)
    add(NVFP4A_ARG_BIAS, 0, 0, g.bias_eb, 0, 1);
  if (g.c_eb) add((int)g.c_arg, 0, (c0 * g.ldc + r0) * g.c_eb, rr * g.c_eb, g.ldc * g.c_eb, cc);
  if (g.np) add(NVFP4A_ARG_NORM_PARTS, 0, c0 * g.norm_ld * 4, g.np * 4, g.norm_ld * 4, cc);
  if (g.np && c0 && cc < g.cols && g.kind != NVFP4A_ITEM_GEMV)
    add(NVFP4A_ARG_NORM_PARTS, 0, 0, g.np * 4, 0, 1);
  if (g.w2_eb) add(NVFP4A_ARG_W2, 0, r0 * g.w2_eb, rr * g.w2_eb, 0, 1);
  if (g.swiglu) add(NVFP4A_ARG_D, 1, (c0 * g.ldd + r0 / 2) * g.d_eb, rr / 2 * g.d_eb, g.ldd * g.d_eb, cc);
  else add(NVFP4A_ARG_D, 1, (c0 * g.ldd + r0) * g.d_eb, rr * g.d_eb, g.ldd * g.d_eb, cc);
  if (g.d2_eb) {
    add(NVFP4A_ARG_D2, 1, (c0 * g.ldd2 + r0) * g.d2_eb, rr * g.d2_eb, g.ldd2 * g.d2_eb, cc);
    add(NVFP4A_ARG_SQ, 1, (c0 * g.ld_sq + r0 / 16) * 4, rr / 16 * 4, g.ld_sq * 4, cc);
  }
  if (g.amax) add(NVFP4A_ARG_D_AMAX, 2, 0, 4, 0, 1);
  return cnt;
}

extern "C" nvfp4a_status nvfp4a_gemm_tc_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B,
                                                 float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                                 const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items) {
  nvfp4a::clear_error();
  if (!items) return nvfp4a::fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm_tc_ex_items: items is NULL");
  if (m <= 0 || n <= 0) return nvfp4a::fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm_tc_ex_items: m = %d, n = %d (no work)", m, n);
  nvfp4a::kern::EpilogueEx x;
  int EX;
  if (nvfp4a_status st = nvfp4a::make_epilogue_ex("nvfp4a_gemm_tc_ex_items", ex, m, n, beta, ldd, &x, &EX)) return st;
  return nvfp4a::gemm_tc_run("nvfp4a_gemm_tc_ex_items", m, n, k, alpha, A, B, beta, C, ldc, D, ldd, bias, nullptr, x, EX, 0, items);
}
