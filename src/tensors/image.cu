// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The transcoders of the tensor-core images (image.cuh): launches.
#define NVFP4A_TC_IMAGES
#include "gemm_tc.cuh"

namespace nvfp4a {
namespace kern {

cudaError_t tc_image_a(bool h, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, int m, int k, uint8_t *img, cudaStream_t s) {
  const int tiles = k / 64;
  const dim3 grid((tiles + 3) / 4, (m + 127) / 128 * 8);  // (m16 slices of whole 128-row blocks)
  if (h) tc::image_a_kernel<true><<<grid, 128, 0, s>>>(A, lda, SA, ldsa, m, tiles, img);
  else tc::image_a_kernel<false><<<grid, 128, 0, s>>>(A, lda, SA, ldsa, m, tiles, img);
  return cudaGetLastError();
}
cudaError_t tc_image_b(int tb, const uint8_t *B, int ldb, const uint8_t *SB, int ldsb, int n, int k, uint8_t *img, cudaStream_t s) {
  const int tiles = k / 64;
  const dim3 grid((tiles + 1) / 2, (n + 127) / 128);
  if (tb == NVFP4A_INT8) tc::image_b8_kernel<<<grid, tc::kImgThreads, 0, s>>>(B, ldb, (const float *)SB, ldsb, n, tiles, img);
  else if (tb == NVFP4A_NVFP4) tc::image_b4_kernel<<<grid, tc::kImgThreads, 0, s>>>(B, ldb, SB, ldsb, n, tiles, img);
  else tc::image_bh_kernel<<<dim3(tiles, (n + 127) / 128), tc::kImgThreads, 0, s>>>((const uint16_t *)B, ldb, n, tiles, img);
  return cudaGetLastError();
}

}  // namespace kern
}  // namespace nvfp4a
