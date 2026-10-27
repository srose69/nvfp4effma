// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One FFMA GEMM instance object: -DNVFP4A_ID=<config> -DNVFP4A_TB=<B dtype> -DNVFP4A_F=<formats>, or with
// -DNVFP4A_EX=<additions> instead of the formats the extended epilogue's kernel (linear E4M3 scales).  Small
// objects keep the build parallel; the kernel bodies are compiled only for the architectures that dispatch
// to them (dense B: the configurations their table or the defaults select; NVFP4 / INT8 B: configuration 0
// below sm_61, where there is no DP4A), trapping stubs elsewhere.
#include "gemm_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

#define NVFP4A_CFG_(id) GemmCfg##id
#define NVFP4A_CFG(id) NVFP4A_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemm : NVFP4A_CFG(NVFP4A_ID) {
  template <int TB> struct On {
    static constexpr bool value = TB == NVFP4A_NVFP4 || TB == NVFP4A_INT8 ? NVFP4A_ARCH_CC < 61 : gemm_selected(NVFP4A_ARCH_CC, ID);
  };
};

#ifdef NVFP4A_EX
template <int ID, int TB, int EX>
const void *gemm_kernel_ex_ptr() { return (const void *)gemm_kernel<LibGemm<ID>, TB, 0, EX, EpilogueEx>; }
template const void *gemm_kernel_ex_ptr<NVFP4A_ID, NVFP4A_TB, NVFP4A_EX>();
template <int ID, int TB, int EX>
cudaError_t gemm_entry_ex(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb, const uint8_t *SB,
                          int ldsb, int m, int n, int k, const Epilogue &e, const EpilogueEx &x, cudaStream_t s) {
  return gemm_launch_ex<LibGemm<ID>, TB, EX>(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, x, s);
}
template cudaError_t gemm_entry_ex<NVFP4A_ID, NVFP4A_TB, NVFP4A_EX>(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int,
                                                                   const uint8_t *, int, int, int, int, const Epilogue &,
                                                                   const EpilogueEx &, cudaStream_t);
#else
template <int ID, int TB, int F>
const void *gemm_kernel_ptr() { return (const void *)gemm_kernel<LibGemm<ID>, TB, F>; }
template const void *gemm_kernel_ptr<NVFP4A_ID, NVFP4A_TB, NVFP4A_F>();
template <int ID, int TB, int F>
cudaError_t gemm_entry(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb, const uint8_t *SB,
                       int ldsb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  return gemm_launch<LibGemm<ID>, TB, F>(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e, s);
}
template cudaError_t gemm_entry<NVFP4A_ID, NVFP4A_TB, NVFP4A_F>(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int,
                                                               const uint8_t *, int, int, int, int, const Epilogue &, cudaStream_t);
#endif

}  // namespace kern
}  // namespace nvfp4a
