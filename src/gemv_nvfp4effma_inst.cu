// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One GEMV instance object: -DNVFP4A_ID=<config> -DNVFP4A_TX=<x dtype>, n = 1..8.  The kernel bodies are
// compiled only for the architectures whose table (or the defaults) select the configuration for that
// x dtype and n, trapping stubs elsewhere.
#include "gemv_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

#define NVFP4A_CFG_(id) GemvCfg##id
#define NVFP4A_CFG(id) NVFP4A_CFG_(id)
// (a template on the id, not a plain struct: kernel names must differ between instance objects)
template <int ID> struct LibGemv : NVFP4A_CFG(NVFP4A_ID) {
  template <int NB, int XT> struct On { static constexpr bool value = gemv_selected(NVFP4A_ARCH_CC, ID, XT, NB); };
};
using C = LibGemv<NVFP4A_ID>;

#define NVFP4A_GEMV_DEF(nb)                                                     \
  NVFP4A_GEMV_PROTO(NVFP4A_ID, NVFP4A_TX, nb) {                                 \
    return gemv_launch<C, nb, NVFP4A_TX>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, s); \
  }
NVFP4A_GEMV_DEF(1) NVFP4A_GEMV_DEF(2) NVFP4A_GEMV_DEF(3) NVFP4A_GEMV_DEF(4)
NVFP4A_GEMV_DEF(5) NVFP4A_GEMV_DEF(6) NVFP4A_GEMV_DEF(7) NVFP4A_GEMV_DEF(8)

}  // namespace kern
}  // namespace nvfp4a
