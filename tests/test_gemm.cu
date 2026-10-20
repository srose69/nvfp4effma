// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm, tiled kernels, bit-exact against the host emulation: NVFP4 A x B in {NVFP4, BF16, FP16,
// FP32}, m / n tails, ld and ld_scale padding, global scales, every output dtype, C of another dtype,
// accumulate (C = NULL, D += ...), alpha / beta / bias / amax, and larger problems (large-tile
// configurations and several k-tiles of the DP4A kernel).
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const nvfp4a_dtype outs[] = {NVFP4A_BF16, NVFP4A_FP32, NVFP4A_FP16};
  const nvfp4a_dtype ops[] = {NVFP4A_NVFP4, NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32};
  const int sizes[][3] = {{1, 9, 64}, {37, 45, 128}, {200, 300, 192}, {257, 129, 256}, {128, 128, 64}};
  for (nvfp4a_dtype bt : ops)
    for (const auto &sz : sizes) {
      Case c{sz[0], sz[1], sz[2], bt, outs[idx % 3],
             idx % 4 == 1 ? -1 : (idx % 4 == 2 ? (int)outs[(idx + 1) % 3] : (int)outs[idx % 3]),
             idx % 3 == 0 ? -1 : (idx % 3 == 1 ? (int)NVFP4A_BF16 : (int)NVFP4A_FP32),
             idx % 2 ? 1.25f : 1.f, idx % 4 == 1 ? 1.f : (idx % 4 == 3 ? 0.5f : 0.f)};
      fails += run_case(c); ++idx;
    }
  // larger problems
  for (nvfp4a_dtype bt : ops) {
    fails += run_case(Case{1100, 1300, 512, bt, NVFP4A_BF16, -1, -1, 1.f, 0.f});
    fails += run_case(Case{1029, 700, 1024, bt, NVFP4A_FP32, -1, NVFP4A_FP32, 0.5f, 1.f});
  }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
