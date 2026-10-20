// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMM host side: nvfp4a_gemm (validation, epilogue setup, routing of n <= 8 to the GEMV
// kernel), the NVFP4 x NVFP4 path (DP4A from sm_61, FFMA below) and the dense-B configuration choice.
#include "gemm_nvfp4effma.cuh"
#include "gemv_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

#define NVFP4A_DECL_GEMM(id) NVFP4A_GEMM_PROTO(id, 1); NVFP4A_GEMM_PROTO(id, 2); NVFP4A_GEMM_PROTO(id, 3);
NVFP4A_GEMM_LIB(NVFP4A_DECL_GEMM)
#undef NVFP4A_DECL_GEMM
NVFP4A_GEMM_PROTO(0, 0);  // NVFP4 x NVFP4 without DP4A

static cudaError_t launch_gemm(int tb, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                               const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const DeviceInfo &d = device_info();
  if (tb == NVFP4A_NVFP4) {
    if (d.sass >= 61) {
      dim3 grid((n + Dp4aCfg::BN - 1) / Dp4aCfg::BN, (m + Dp4aCfg::BM - 1) / Dp4aCfg::BM);
      gemm_dp4a_kernel<Dp4aCfg><<<grid, Dp4aCfg::NT, 0, s>>>(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e);
      return cudaGetLastError();
    }
    return NVFP4A_GEMM_ENTRY(0, 0)(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, s);
  }
  int id = d.table->gemm_large, bm = 128, bn = 128;
#define NVFP4A_TILE(cid, cbm, cbn, ctm, ctn) if (id == cid) { bm = cbm; bn = cbn; }
  NVFP4A_GEMM_CANDIDATES(NVFP4A_TILE)
#undef NVFP4A_TILE
  if ((long long)((m + bm - 1) / bm) * ((n + bn - 1) / bn) < 2LL * d.sms) id = d.table->gemm_small;
  using Fn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int,
                             int, int, const Epilogue &, cudaStream_t);
#define NVFP4A_CASE(cid)                                                                                \
  if (id == cid) {                                                                                      \
    static const Fn f[3] = {NVFP4A_GEMM_ENTRY(cid, 1), NVFP4A_GEMM_ENTRY(cid, 2), NVFP4A_GEMM_ENTRY(cid, 3)}; \
    return f[tb - 1](A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, s);                                \
  }
  NVFP4A_GEMM_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace nvfp4a

extern "C" nvfp4a_status nvfp4a_gemm(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda,
                                     const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                                     const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, cudaStream_t stream) {
  using namespace nvfp4a::kern;
  using nvfp4a::fail;
  nvfp4a::clear_error();
  const auto dense = [](const nvfp4a_tensor *t) { return t->dtype >= NVFP4A_BF16 && t->dtype <= NVFP4A_FP32; };
  const auto aligned = [](const void *p, unsigned a) { return ((uintptr_t)p & (a - 1)) == 0; };
  if (!A || !B || !D || !A->data || !B->data || !D->data)
    return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: A, B, D or their data is NULL");
  if (m < 0 || n < 0 || k < 0) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: negative size (m=%d n=%d k=%d)", m, n, k);
  if (A->dtype != NVFP4A_NVFP4) return fail(NVFP4A_NOT_SUPPORTED, "nvfp4a_gemm: A must be NVFP4 (dtype %d)", (int)A->dtype);
  if (B->dtype != NVFP4A_NVFP4 && !dense(B))
    return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: invalid B dtype %d", (int)B->dtype);
  if (!dense(D)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: D must be BF16, FP16 or FP32 (dtype %d)", (int)D->dtype);
  if (m == 0 || n == 0) return NVFP4A_OK;
  if (k % 64) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: k = %d is not a multiple of 64", k);
  if (lda % 64 || ldb % 64) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: lda = %d / ldb = %d must be multiples of 64", lda, ldb);
  if (lda < k || ldb < k) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: lda = %d / ldb = %d < k = %d", lda, ldb, k);
  if (!aligned(A->data, 16) || !aligned(B->data, 16)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: A and B must be 16-byte aligned");
  for (const nvfp4a_tensor *t : {A, B}) {
    if (t->dtype != NVFP4A_NVFP4) continue;
    if (!t->block_scale) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: NVFP4 operand without block_scale");
    if (!aligned(t->block_scale, 4) || t->ld_scale % 4 || (t->ld_scale && t->ld_scale < k / 16))
      return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: block scales must be 4-byte aligned with ld_scale a multiple of 4 "
                  "and >= k / 16 (ld_scale = %d)", t->ld_scale);
  }
  if (ldd < m) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: ldd = %d < m = %d", ldd, m);
  if (bias && bias->data && !dense(bias)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: invalid bias dtype %d", (int)bias->dtype);
  if (beta != 0.f && C && C->data && !dense(C)) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: invalid C dtype %d", (int)C->dtype);
  if (beta != 0.f && C && C->data && ldc < m) return fail(NVFP4A_INVALID_VALUE, "nvfp4a_gemm: ldc = %d < m = %d", ldc, m);
  if (nvfp4a_status st = nvfp4a::check_device()) return st;
  const bool c_is_d = beta != 0.f && (!C || !C->data);  // D += ...

  Epilogue e;
  e.ga = A->global_scale;
  e.gb = B->dtype == NVFP4A_NVFP4 ? B->global_scale : nullptr;
  e.alpha = alpha; e.beta = beta;
  const nvfp4a_tensor *Cx = c_is_d ? D : C;
  e.c = beta != 0.f ? Cx->data : nullptr; e.ldc = beta != 0.f ? (c_is_d ? ldd : ldc) : 0;
  e.c_dtype = beta != 0.f ? Cx->dtype : NVFP4A_FP32;
  e.d = D->data; e.ldd = ldd; e.d_dtype = D->dtype; e.amax = D->amax;
  e.bias = bias && bias->data ? bias->data : nullptr; e.bias_dtype = bias ? bias->dtype : NVFP4A_FP32;

  const uint8_t *a = (const uint8_t *)A->data, *b = (const uint8_t *)B->data;
  const uint8_t *sa = (const uint8_t *)A->block_scale, *sb = (const uint8_t *)B->block_scale;
  const int ldsa = A->ld_scale ? A->ld_scale : k / 16;
  const int ldsb = B->dtype == NVFP4A_NVFP4 ? (B->ld_scale ? B->ld_scale : k / 16) : 0;
  const cudaError_t err = n <= 8 ? launch_gemv(B->dtype, a, lda, sa, ldsa, b, ldb, sb, ldsb, m, n, k, e, stream)
                                 : launch_gemm(B->dtype, a, lda, sa, ldsa, b, ldb, sb, ldsb, m, n, k, e, stream);
  return nvfp4a::cuda_status(err, "nvfp4a_gemm: kernel launch");
}
