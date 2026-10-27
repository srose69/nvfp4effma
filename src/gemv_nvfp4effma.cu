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

using GemvExFn = cudaError_t (*)(const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, const uint8_t *, int, int,
                                 int, const Epilogue &, const EpilogueEx &, cudaStream_t);

// entries of config ID and x dtype XT, [EX - 1][n - 1]
template <int ID, int XT, size_t... I>
static const GemvExFn *gemv_ex_table(std::index_sequence<I...>) {
  static const GemvExFn t[] = {&gemv_entry_ex<ID, XT, (int)(I % 8) + 1, (int)(I / 8) + 1>...};
  return t;
}

cudaError_t launch_gemv_ex(int xt, int ex, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                           const uint8_t *SX, int ldsx, int m, int n, int k, const Epilogue &e, const EpilogueEx &x, cudaStream_t s) {
  if (!gemv_ex_dtype(xt) || ex < 1 || ex > kEX_MASKS || n < 1 || n > 8) return cudaErrorInvalidConfiguration;
  const int id = device_info().table->gemv[xt][n - 1];
#define NVFP4A_CASE(cid)                                                                                             \
  if (id == cid)                                                                                                     \
    return (xt == NVFP4A_BF16 ? gemv_ex_table<cid, NVFP4A_BF16>(std::make_index_sequence<8 * kEX_MASKS>())          \
                              : gemv_ex_table<cid, NVFP4A_INT8>(std::make_index_sequence<8 * kEX_MASKS>()))[8 * (ex - 1) + n - 1]( \
        A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, x, s);
  NVFP4A_GEMV_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
  return cudaErrorInvalidConfiguration;
}

// [EX][n - 1], EX = 0 the plain kernel (formats 0)
template <int ID, int XT, int NB, int EX> struct GemvKernelOf { static const void *get() { return gemv_kernel_ex_ptr<ID, XT, NB, EX>(); } };
template <int ID, int XT, int NB> struct GemvKernelOf<ID, XT, NB, 0> { static const void *get() { return gemv_kernel_ptr<ID, XT, NB, 0>(); } };
template <int ID, int XT, size_t... I>
static const void *gemv_kernel_of(int ex, int n, std::index_sequence<I...>) {
  using Fn = const void *(*)();
  static const Fn t[] = {&GemvKernelOf<ID, XT, (int)(I % 8) + 1, (int)(I / 8)>::get...};
  return t[8 * ex + n - 1]();
}

bool gemv_item_config(int xt, int ex, int n, int *R, int *MX, const void **kernel) {
  if (!gemv_ex_dtype(xt) || ex < 0 || ex > kEX_MASKS || n < 1 || n > 8) return false;
  const int id = device_info().table->gemv[xt][n - 1];
#define NVFP4A_CAND(cid, r, mx) if (id == cid) { *R = r; *MX = mx; }
  NVFP4A_GEMV_CANDIDATES(NVFP4A_CAND)
#undef NVFP4A_CAND
#define NVFP4A_CASE(cid)                                                                                                    \
  if (id == cid) {                                                                                                          \
    *kernel = xt == NVFP4A_BF16 ? gemv_kernel_of<cid, NVFP4A_BF16>(ex, n, std::make_index_sequence<8 * (kEX_MASKS + 1)>()) \
                                : gemv_kernel_of<cid, NVFP4A_INT8>(ex, n, std::make_index_sequence<8 * (kEX_MASKS + 1)>()); \
    return true;                                                                                                            \
  }
  NVFP4A_GEMV_LIB(NVFP4A_CASE)
#undef NVFP4A_CASE
  return false;
}

}  // namespace kern
}  // namespace nvfp4a
