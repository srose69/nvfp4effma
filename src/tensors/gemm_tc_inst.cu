// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One tensor-core GEMM instance object: -DNVFP4A_TB=<B dtype: NVFP4, INT8 or BF16> -DNVFP4A_F=<scale formats: kF_AUE, kF_BUE>.
// The kernel body is compiled for sm_75 .. sm_89 (W4A16: sm_80+; a trapping stub elsewhere).
#include "gemm_tc.cuh"

namespace nvfp4a {
namespace kern {

template <int TB, int F>
cudaError_t tc_entry(const uint8_t *ia, const uint8_t *ib, int m, int n, int k, const Epilogue &e, uint8_t *ws, cudaStream_t s) {
  // the specialization the device runs (its PTX version: also right for compute_75 PTX JIT-compiled for a newer GPU):
  // sm_75 three slots, four producer warps
  static const bool v75 = [] { cudaFuncAttributes fa{}; cudaFuncGetAttributes(&fa, gemm_tc_kernel<TB, F>); return fa.ptxVersion < 80; }();
  const int smem = tc_smem_bytes<TB>(v75), threads = 256 + 32 * (v75 ? kTcProducers75 : kTcProducers80);
  static const cudaError_t attr = cudaFuncSetAttribute(gemm_tc_kernel<TB, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attr != cudaSuccess) return attr;
  const int sms = device_info().sms;  // persistent stream-K: one CTA an SM whatever the tile count
  gemm_tc_kernel<TB, F><<<sms, threads, smem, s>>>(ia, ib, m, n, k / 64, e, (float *)ws, (unsigned *)(ws + (size_t)sms * 256 * 64 * 4));
  return cudaGetLastError();
}
template cudaError_t tc_entry<NVFP4A_TB, NVFP4A_F>(const uint8_t *, const uint8_t *, int, int, int, const Epilogue &, uint8_t *,
                                                  cudaStream_t);

}  // namespace kern
}  // namespace nvfp4a
