// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The INT8 quantization of activations (nvfp4a_quantize to NVFP4A_INT8) as work items: a row of `cols` values becomes int8 and
// one FP32 scale per block of 16, sb = amax_block / 127, q = RNE(x / sb) -- the arithmetic of quantize_int8_kernel.
#pragma once
#include "nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

constexpr int kQuantItemThreads = 256;

// item = rows [item R, (item + 1) R) of the first min(rows, *n_dev) (R = rows_per_item); a block of 16 per thread.
struct QuantizeItemArgs {
  const uint8_t *src;
  int8_t *dst;
  float *scales;
  int lds, ldd, ld_scale, rows, cols, rows_per_item;
  const int *n_dev;
};

template <int T>
__device__ __forceinline__ void quantize_int8_item(const QuantizeItemArgs &a, int item, unsigned thread, void *) {
  const int rows = a.n_dev ? min(a.rows, __ldcg(a.n_dev)) : a.rows, bpr = a.cols / 16;
  const int last = min(rows, (item + 1) * a.rows_per_item);
  for (int r = item * a.rows_per_item; r < last; ++r)
    for (int c = static_cast<int>(thread); c < bpr; c += kQuantItemThreads) {
      float x[16];
      const uint8_t *p = a.src + ((size_t)r * a.lds + 16 * c) * Elem<T>::bytes;
      load8<T, true>(p, x);
      load8<T, true>(p + 8 * Elem<T>::bytes, x + 8);
      float bmax = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(x[i]));
      const float sb = __fdiv_rn(bmax, 127.f);
      uint32_t w[4] = {0, 0, 0, 0};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int q = sb > 0.f ? (int)fminf(fmaxf(rintf(__fdiv_rn(x[i], sb)), -127.f), 127.f) : 0;
        w[i / 4] |= (uint32_t)(uint8_t)(int8_t)q << (8 * (i % 4));
      }
      *(uint4 *)(a.dst + (size_t)r * a.ldd + 16 * c) = make_uint4(w[0], w[1], w[2], w[3]);
      a.scales[(size_t)r * a.ld_scale + c] = sb;
    }
}

}  // namespace kern
}  // namespace nvfp4a
