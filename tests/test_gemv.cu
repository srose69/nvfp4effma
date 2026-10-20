// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm on the GEMV path (n <= 8), bit-exact against the host emulation of the kernel's lane
// order: x in {NVFP4, BF16, FP16, FP32} x n = 1..8, every output dtype, accumulate, bias, k not a
// multiple of the warp's 512-value pass.
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const nvfp4a_dtype outs[] = {NVFP4A_BF16, NVFP4A_FP32, NVFP4A_FP16};
  const nvfp4a_dtype ops[] = {NVFP4A_NVFP4, NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32};
  const int gs[][2] = {{1, 64}, {33, 576}, {300, 1024}, {1000, 4160}};
  for (nvfp4a_dtype bt : ops)
    for (int n = 1; n <= 8; ++n) {
      const int *g = gs[(n + bt) % 4];
      Case c{g[0], n, g[1], bt, outs[idx % 3], n == 3 ? -1 : (int)outs[idx % 3], idx % 3 == 0 ? -1 : (int)NVFP4A_BF16,
             idx % 2 ? 0.75f : 1.f, n == 3 ? 1.f : (idx % 4 == 1 ? -1.5f : 0.f)};
      fails += run_case(c);
      ++idx;
    }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
