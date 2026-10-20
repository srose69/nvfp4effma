// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm on the GEMV path (n <= 8), bit-exact against the host emulation of the kernel's lane
// order: x in {NVFP4, INT8, BF16, FP16, FP32} x n = 1..8, every output dtype, accumulate, bias, k not a
// multiple of the warp's 512-value pass; n beyond 8 (groups of 8) around the GEMV limit.
#include "test_util.h"

int main() {
  int fails = 0, idx = 0;
  const nvfp4a_dtype outs[] = {NVFP4A_BF16, NVFP4A_FP32, NVFP4A_FP16};
  const nvfp4a_dtype ops[] = {NVFP4A_NVFP4, NVFP4A_INT8, NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32};
  const int gs[][2] = {{1, 64}, {33, 576}, {300, 1024}, {1000, 4160}};
  for (nvfp4a_dtype bt : ops)
    for (int n = 1; n <= 8; ++n) {
      const int *g = gs[(n + bt) % 4];
      Case c{g[0], n, g[1], bt, outs[idx % 3], n == 3 ? -1 : (int)outs[idx % 3], idx % 3 == 0 ? -1 : (int)NVFP4A_BF16,
             idx % 2 ? 0.75f : 1.f, n == 3 ? 1.f : (idx % 4 == 1 ? -1.5f : 0.f)};
      fails += run_case(c);
      ++idx;
    }
  // more than 8 columns: groups of 8 on the GEMV path, around its limit (kGemvMaxN) on both sides
  for (nvfp4a_dtype bt : ops)
    for (int n : {12, 17, 32, 33, 48, 49}) {
      Case c{300, n, 1024, bt, outs[idx % 3], n == 17 ? -1 : (int)outs[idx % 3], idx % 2 ? (int)NVFP4A_BF16 : -1,
             1.f, n == 17 ? 1.f : 0.f};
      fails += run_case(c);
      ++idx;
    }
  // Transformer Engine / cuBLASLt tensors: every operand format (every kernel instance) -- swizzled
  // scales, UE5M3 scales, amax globals (scale_max 448 / 256 / 114688), per-row globals -- for every B dtype
  {
    const int sz[][3] = {{300, 3, 1024}, {1000, 8, 576}, {130, 20, 4160}, {77, 1, 128}, {64, 5, 64}};
    int v = 0;
    for (nvfp4a_dtype bt : {NVFP4A_NVFP4, NVFP4A_INT8, NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32})
      for (int i = 0; i < (bt == NVFP4A_NVFP4 ? kTestFormats : kTestFormatsA); ++i, ++v) {
        const int *z = sz[v % 5];
        Case c{z[0], z[1], z[2], bt, outs[v % 3], v % 4 == 1 ? -1 : (int)outs[v % 3], v % 3 ? (int)NVFP4A_BF16 : -1,
               v % 2 ? 1.25f : 1.f, v % 4 == 1 ? 1.f : 0.f};
        set_formats(c, bt == NVFP4A_NVFP4 ? i : test_format_a(i), v);
        fails += run_case(c);
      }
  }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
