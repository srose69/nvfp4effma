// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One tensor-core GEMV instance object: -DNVFP4A_TB=<x dtype: NVFP4, INT8 or BF16> -DNVFP4A_F=<formats>, or with
// -DNVFP4A_EX=<additions> instead of the formats the extended epilogue's kernel.
// The kernel body is compiled for sm_75 .. sm_89 (BF16 x: sm_80+; a trapping stub elsewhere).
#include "gemv_tc.cuh"

namespace nvfp4a {
namespace kern {

#ifdef NVFP4A_EX
template <int TB, int EX>
cudaError_t gemv_tc_entry_ex(const uint8_t *ia, const uint8_t *x, int ldx, const uint8_t *sx, int ldsx, int m, int n, int k,
                             const Epilogue &e, const EpilogueEx &xd, cudaStream_t s) {
  gemv_tc_kernel<TB, 0, EX><<<(m + 15) / 16, kGvWarps * 32, 0, s>>>(ia, x, ldx, sx, ldsx, m, n, k / 64, e, xd);
  return cudaGetLastError();
}
template cudaError_t gemv_tc_entry_ex<NVFP4A_TB, NVFP4A_EX>(const uint8_t *, const uint8_t *, int, const uint8_t *, int, int, int, int,
                                                           const Epilogue &, const EpilogueEx &, cudaStream_t);
template <int TB, int EX>
const void *gemv_tc_kernel_ex_ptr() { return (const void *)gemv_tc_kernel<TB, 0, EX, EpilogueEx>; }
template const void *gemv_tc_kernel_ex_ptr<NVFP4A_TB, NVFP4A_EX>();
#else
template <int TB, int F>
cudaError_t gemv_tc_entry(const uint8_t *ia, const uint8_t *x, int ldx, const uint8_t *sx, int ldsx, int m, int n, int k,
                          const Epilogue &e, cudaStream_t s) {
  gemv_tc_kernel<TB, F><<<(m + 15) / 16, kGvWarps * 32, 0, s>>>(ia, x, ldx, sx, ldsx, m, n, k / 64, e);
  return cudaGetLastError();
}
template cudaError_t gemv_tc_entry<NVFP4A_TB, NVFP4A_F>(const uint8_t *, const uint8_t *, int, const uint8_t *, int, int, int, int,
                                                       const Epilogue &, cudaStream_t);
template <int TB, int F>
const void *gemv_tc_kernel_ptr() { return (const void *)gemv_tc_kernel<TB, F>; }
template const void *gemv_tc_kernel_ptr<NVFP4A_TB, NVFP4A_F>();
#endif

}  // namespace kern
}  // namespace nvfp4a
