// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One GEMV instance object: -DNVFP4A_ID=<config> -DNVFP4A_TX=<x dtype> -DNVFP4A_F=<formats>, n = 1..8.  The
// kernel bodies are compiled only for the architectures whose table (or the defaults) select the
// configuration for that x dtype and n, trapping stubs elsewhere.
#include "gemv_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

#define NVFP4A_CFG_(id) GemvCfg##id
#define NVFP4A_CFG(id) NVFP4A_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemv : NVFP4A_CFG(NVFP4A_ID) {
  template <int NB, int XT> struct On { static constexpr bool value = gemv_selected(NVFP4A_ARCH_CC, ID, XT, NB); };
};

template <int ID, int XT, int NB, int F>
cudaError_t gemv_entry(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx, const uint8_t *SX,
                       int ldsx, int m, int k, const Epilogue &e, cudaStream_t s) {
  return gemv_launch<LibGemv<ID>, NB, XT, F>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, s);
}
#ifdef NVFP4A_EX  // an extended-epilogue object: -DNVFP4A_EX=<additions>, linear E4M3 scales
template <int ID, int XT, int NB, int EX>
cudaError_t gemv_entry_ex(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx, const uint8_t *SX,
                          int ldsx, int m, int k, const Epilogue &e, const EpilogueEx &x, cudaStream_t s) {
  return gemv_launch_ex<LibGemv<ID>, NB, XT, 0, EX>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, x, s);
}
template <int ID, int XT, int NB, int EX>
const void *gemv_kernel_ex_ptr() { return (const void *)gemv_kernel_ex<LibGemv<ID>, NB, XT, 0, EX>; }
#define NVFP4A_GEMV_INST(nb)                                                                                        \
  template cudaError_t gemv_entry_ex<NVFP4A_ID, NVFP4A_TX, nb, NVFP4A_EX>(const uint8_t *, int, const uint8_t *, int, \
                                                                         const uint8_t *, int, const uint8_t *, int, int, int, \
                                                                         const Epilogue &, const EpilogueEx &, cudaStream_t); \
  template const void *gemv_kernel_ex_ptr<NVFP4A_ID, NVFP4A_TX, nb, NVFP4A_EX>();
#else
template <int ID, int XT, int NB, int F>
const void *gemv_kernel_ptr() { return (const void *)gemv_kernel<LibGemv<ID>, NB, XT, F>; }
#define NVFP4A_GEMV_INST(nb)                                                                                     \
  template cudaError_t gemv_entry<NVFP4A_ID, NVFP4A_TX, nb, NVFP4A_F>(const uint8_t *, int, const uint8_t *, int, \
                                                                      const uint8_t *, int, const uint8_t *, int, int, int, \
                                                                      const Epilogue &, cudaStream_t);          \
  template const void *gemv_kernel_ptr<NVFP4A_ID, NVFP4A_TX, nb, NVFP4A_F>();
#endif
NVFP4A_GEMV_INST(1) NVFP4A_GEMV_INST(2) NVFP4A_GEMV_INST(3) NVFP4A_GEMV_INST(4)
NVFP4A_GEMV_INST(5) NVFP4A_GEMV_INST(6) NVFP4A_GEMV_INST(7) NVFP4A_GEMV_INST(8)

}  // namespace kern
}  // namespace nvfp4a
