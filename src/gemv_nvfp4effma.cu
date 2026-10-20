// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMV host side: the kernel for the x dtype and n, in the configuration the device's table
// selects per x dtype and n.
#include "gemv_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

#define NVFP4A_DECL_GEMV_X(id, t)                                                                       \
  NVFP4A_GEMV_PROTO(id, t, 1); NVFP4A_GEMV_PROTO(id, t, 2); NVFP4A_GEMV_PROTO(id, t, 3); NVFP4A_GEMV_PROTO(id, t, 4); \
  NVFP4A_GEMV_PROTO(id, t, 5); NVFP4A_GEMV_PROTO(id, t, 6); NVFP4A_GEMV_PROTO(id, t, 7); NVFP4A_GEMV_PROTO(id, t, 8);
#define NVFP4A_DECL_GEMV(id) NVFP4A_DECL_GEMV_X(id, 0) NVFP4A_DECL_GEMV_X(id, 1) NVFP4A_DECL_GEMV_X(id, 2) NVFP4A_DECL_GEMV_X(id, 3)
NVFP4A_GEMV_LIB(NVFP4A_DECL_GEMV)
#undef NVFP4A_DECL_GEMV
#undef NVFP4A_DECL_GEMV_X

cudaError_t launch_gemv(int xt, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                        const uint8_t *SX, int ldsx, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const int id = device_info().table->gemv[xt][n - 1];
  using Fn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int,
                             int, const Epilogue &, cudaStream_t);
#define NVFP4A_ROW(cid, t)                                                                                \
  {NVFP4A_GEMV_ENTRY(cid, t, 1), NVFP4A_GEMV_ENTRY(cid, t, 2), NVFP4A_GEMV_ENTRY(cid, t, 3), NVFP4A_GEMV_ENTRY(cid, t, 4), \
   NVFP4A_GEMV_ENTRY(cid, t, 5), NVFP4A_GEMV_ENTRY(cid, t, 6), NVFP4A_GEMV_ENTRY(cid, t, 7), NVFP4A_GEMV_ENTRY(cid, t, 8)}
#define NVFP4A_CASE(cid)                                                                                  \
  if (id == cid) {                                                                                        \
    static const Fn f[kOperandTypes][8] = {NVFP4A_ROW(cid, 0), NVFP4A_ROW(cid, 1), NVFP4A_ROW(cid, 2), NVFP4A_ROW(cid, 3)}; \
    return f[xt][n - 1](A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, s);                                  \
  }
  NVFP4A_GEMV_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
#undef NVFP4A_ROW
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace nvfp4a
