// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMV (n <= 8): bandwidth-bound kernel template, its configuration candidates and the entry
// points of the per-instance objects (gemv_nvfp4effma_inst.cu).  Dispatch: gemv_nvfp4effma.cu; the public
// entry is nvfp4a_gemm (gemm_nvfp4effma.cu), which routes here.
#pragma once
#include "nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

// ---------------------------------------------------------------------------------------------
// GEMV / small batch (A NVFP4 m x k, x n <= 8 rows of k values, NVFP4 / INT8 / BF16 / FP16 / FP32): every byte
// of A read once.  A warp owns R rows; per pass a lane reads one NVFP4 block (8 bytes + its scale) of
// each row; x values are converted once per pass and reused for all R rows.  Dense x: 16 x NB FFMAs per
// row and pass (x in two halves of 8, which bounds the registers), lane l accumulating k = 512c + 16l + j
// in (c, j) order.  NVFP4 / INT8 x: per row, column and block the exact block sum, lane l accumulating the
// blocks 32c + l.  Lanes are then combined with a butterfly (xor 16, 8, 4, 2, 1) -- tests/test_gemv.cu
// emulates exactly this order.
// ---------------------------------------------------------------------------------------------
// MAXNB: largest n the configuration is a candidate for (beyond it R x NB accumulators and x values
// exceed the register budget on every target -- all have 64K registers per SM).
template <int R_, int MAXNB_>
struct GemvCfg {
  static constexpr int R = R_, WARPS = 8, MAXNB = MAXNB_;
  template <int NB, int XT> struct On { static constexpr bool value = true; };
};

template <class C, int NB, int XT>
__global__ void __launch_bounds__(C::WARPS * 32)
gemv_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
            const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m, int k, Epilogue e) {
  if (!C::template On<NB, XT>::value) { __trap(); return; }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * C::WARPS + warp) * C::R;
  float acc[C::R][NB];
#pragma unroll
  for (int r = 0; r < C::R; ++r)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[r][b] = 0.f;

  for (int kb = lane * 16; kb < k; kb += 512) {
    uint2 aw[C::R];
    float sa[C::R];
#pragma unroll
    for (int r = 0; r < C::R; ++r) {  // rows past m read row 0 (not stored)
      const size_t row = row0 + r < m ? row0 + r : 0;
      aw[r] = *(const uint2 *)(A + row * (lda / 2) + kb / 2);
      sa[r] = e4m3_value(SA[row * ldsa + kb / 16]);
    }
    if (XT == NVFP4A_NVFP4 || XT == NVFP4A_INT8) {
      // NVFP4 / INT8 x: the exact block sum p of every (row, column), then acc = fma(p * sa, sx, acc) -- the
      // values of the tiled kernels.  DP4A on int8 from sm_61, FFMA on integer-valued floats below.
      float sx[NB];
#pragma unroll
      for (int b = 0; b < NB; ++b)
        sx[b] = XT == NVFP4A_INT8 ? ((const float *)SX)[(size_t)b * ldsx + kb / 16] : e4m3_value(SX[(size_t)b * ldsx + kb / 16]);
#if __CUDA_ARCH__ >= 610
      uint4 xi[NB];
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        if (XT == NVFP4A_INT8) xi[b] = *(const uint4 *)(X + (size_t)b * ldx + kb);
        else { const uint2 w = *(const uint2 *)(X + (size_t)b * (ldx / 2) + kb / 2); xi[b] = dec16(w.x, w.y); }
      }
#pragma unroll
      for (int r = 0; r < C::R; ++r) {
        const uint4 ai = dec16(aw[r].x, aw[r].y);
        const float nsa = -12582912.f * sa[r];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          int p = __dp4a((int)ai.x, (int)xi[b].x, 0x4B400000);  // starts at the bits of 1.5 * 2^23
          p = __dp4a((int)ai.y, (int)xi[b].y, p);
          p = __dp4a((int)ai.z, (int)xi[b].z, p);
          p = __dp4a((int)ai.w, (int)xi[b].w, p);
          acc[r][b] = fma_rn(fma_rn(__int_as_float(p), sa[r], nsa), sx[b], acc[r][b]);
        }
      }
#else
      uint4 xw[NB];  // NVFP4: 8 bytes of nibbles in .x / .y; INT8: 16 bytes
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        if (XT == NVFP4A_INT8) xw[b] = *(const uint4 *)(X + (size_t)b * ldx + kb);
        else { const uint2 w = *(const uint2 *)(X + (size_t)b * (ldx / 2) + kb / 2); xw[b] = make_uint4(w.x, w.y, 0, 0); }
      }
#pragma unroll
      for (int r = 0; r < C::R; ++r)
#pragma unroll
        for (int b = 0; b < NB; ++b) {
          const uint32_t xq[4] = {xw[b].x, xw[b].y, xw[b].z, xw[b].w};
          float p = 0.f;
#pragma unroll
          for (int q = 0; q < 16; ++q) {
            const float xv = XT == NVFP4A_INT8 ? (float)(int8_t)(xq[q / 4] >> (8 * (q % 4))) : e2m1_value<1>(xq[q / 8], q % 8);
            p = fma_rn(e2m1_value<1>(q < 8 ? aw[r].x : aw[r].y, q % 8), xv, p);
          }
          acc[r][b] = fma_rn(mul_rn(p, sa[r]), sx[b], acc[r][b]);
        }
#endif
    } else {
      // dense x: one FFMA per MAC with a = e2m1 * sa * 2^-9, decoded with a single FMUL (the raw E2M1
      // encoding value * 2^-126 times sa * 2^117, which cannot overflow); the 2^9 returns before the
      // epilogue, exactly
#pragma unroll
      for (int r = 0; r < C::R; ++r) sa[r] = mul_rn(sa[r], 0x1p117f);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float xv[NB][8];
#pragma unroll
        for (int b = 0; b < NB; ++b) load8<XT>(X + ((size_t)b * ldx + kb + 8 * h) * Elem<XT>::bytes, xv[b]);
#pragma unroll
        for (int r = 0; r < C::R; ++r)
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const float a = mul_rn(e2m1_raw(h ? aw[r].y : aw[r].x, q), sa[r]);
#pragma unroll
            for (int b = 0; b < NB; ++b) acc[r][b] = fma_rn(a, xv[b][q], acc[r][b]);
          }
      }
    }
  }

  // After the butterfly every lane holds every sum; output o = r * NB + b is stored by lane o % 32,
  // so the stores run in parallel and the epilogue is inlined once (twice for R * NB > 32).
  constexpr int NO = C::R * NB, PER = (NO + 31) / 32;
  float mine[PER];
#pragma unroll
  for (int t = 0; t < PER; ++t) mine[t] = 0.f;
#pragma unroll
  for (int r = 0; r < C::R; ++r)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      float v = acc[r][b];
#pragma unroll
      for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
      v = __fmul_rn(v, XT == NVFP4A_NVFP4 ? 0.25f : XT == NVFP4A_INT8 ? 0.5f : 512.f);  // exact
      if (((r * NB + b) & 31) == lane) mine[(r * NB + b) >> 5] = v;
    }
  EpiState st = epi_begin(e);
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int o = t * 32 + lane, r = o / NB;
    if (o < NO && row0 + r < m) epi_store(e, st, row0 + r, o - r * NB, mine[t]);
  }
  epi_end(e, st);
}

template <class C, int NB, int XT>
cudaError_t gemv_launch(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                        const uint8_t *SX, int ldsx, int m, int k, const Epilogue &e, cudaStream_t s) {
  const int rows = C::WARPS * C::R;
  gemv_kernel<C, NB, XT><<<(m + rows - 1) / rows, C::WARPS * 32, 0, s>>>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e);
  return cudaGetLastError();
}

// Candidates: id, rows per warp, largest n.
#define NVFP4A_GEMV_CANDIDATES(X) \
  X(0, 4, 8)                      \
  X(1, 8, 2)                      \
  X(2, 2, 8)
#define NVFP4A_DEF_GEMV(id, r, mx) using GemvCfg##id = GemvCfg<r, mx>;
NVFP4A_GEMV_CANDIDATES(NVFP4A_DEF_GEMV)
#undef NVFP4A_DEF_GEMV

// GEMV of NVFP4 A by n <= 8 rows of x (dtype xt) with the configuration the device's table selects.
cudaError_t launch_gemv(int xt, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                        const uint8_t *SX, int ldsx, int m, int n, int k, const Epilogue &e, cudaStream_t s);

}  // namespace kern
}  // namespace nvfp4a

// configurations the library builds: the default (0) and those the tuned tables use
#define NVFP4A_GEMV_LIB(X) X(0) NVFP4A_GEMV_USED(X)

// entry point of one kernel: config id x x dtype x n
#define NVFP4A_GEMV_ENTRY(id, t, nb) NVFP4A_CAT6(gemv_entry_, id, _, t, _, nb)
#define NVFP4A_GEMV_PROTO(id, t, nb)                                                                    \
  cudaError_t NVFP4A_GEMV_ENTRY(id, t, nb)(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, \
                                           int ldx, const uint8_t *SX, int ldsx, int m, int k,                      \
                                           const nvfp4a::kern::Epilogue &e, cudaStream_t s)
