// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMV host side: the kernel for the x dtype, formats and n, in the configuration the device's
// table selects per x dtype and n.
#include <utility>
#include "gemv_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

using GemvFn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int,
                               int, const Epilogue &, cudaStream_t);

// entries of config ID and x dtype XT, [format index][n - 1]: NVFP4 x every format, the others the formats
// without B bits (A bits: the same values 0 .. 3)
template <int ID, int XT, size_t... I>
static const GemvFn *gemv_table(std::index_sequence<I...>) {
  static const GemvFn t[] = {&gemv_entry<ID, XT, (int)(I % 8) + 1, XT == NVFP4A_NVFP4 ? (int)(I / 8) : (int)(I / 8)>...};
  return t;
}
template <int ID>
static const GemvFn *gemv_table(int xt) {
  switch (xt) {
    case NVFP4A_NVFP4: return gemv_table<ID, NVFP4A_NVFP4>(std::make_index_sequence<8 * kFormats>());
    case NVFP4A_BF16: return gemv_table<ID, NVFP4A_BF16>(std::make_index_sequence<8 * kFormatsA>());
    case NVFP4A_FP16: return gemv_table<ID, NVFP4A_FP16>(std::make_index_sequence<8 * kFormatsA>());
    case NVFP4A_FP32: return gemv_table<ID, NVFP4A_FP32>(std::make_index_sequence<8 * kFormatsA>());
    default: return gemv_table<ID, NVFP4A_INT8>(std::make_index_sequence<8 * kFormatsA>());
  }
}

cudaError_t launch_gemv(int xt, int f, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                        const uint8_t *SX, int ldsx, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  const int id = device_info().table->gemv[xt][n - 1], fi = xt == NVFP4A_NVFP4 ? f : (f & 3);
#define NVFP4A_CASE(cid) \
  if (id == cid) return gemv_table<cid>(xt)[8 * fi + n - 1](A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, s);
  NVFP4A_GEMV_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
  return cudaErrorInvalidConfiguration;
}

}  // namespace kern
}  // namespace nvfp4a
