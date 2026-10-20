// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One DP4A GEMM instance object: -DNVFP4A_TB=<B dtype: NVFP4 or INT8> -DNVFP4A_F=<formats>, both tiles.
#include "gemm_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

template <int TB, int F>
cudaError_t dp4a_entry(bool narrow, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                       const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const auto go = [&](auto cfg) {
    using C = decltype(cfg);
    dim3 grid((n + C::BN - 1) / C::BN, (m + C::BM - 1) / C::BM);
    gemm_dp4a_kernel<C, TB, F><<<grid, C::NT, 0, s>>>(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e);
    return cudaGetLastError();
  };
  return narrow ? go(Dp4aCfgNarrow()) : go(Dp4aCfg());
}
template cudaError_t dp4a_entry<NVFP4A_TB, NVFP4A_F>(bool, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int,
                                                    const uint8_t *, int, int, int, int, const Epilogue &, cudaStream_t);

}  // namespace kern
}  // namespace nvfp4a
