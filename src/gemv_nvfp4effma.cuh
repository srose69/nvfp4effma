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

// The rows row0 .. row0 + R - 1 of a warp (`lane` its lane), over k in [k0, k1) (k0 a multiple of 512): every sum of a
// (row, column), reduced over the lanes and scaled back exactly (not by the epilogue's scale), goes to
// out.store(row, column, value) on the lane that owns it (between out.begin() and out.end()), rows past m skipped.  The
// R rows' loads of a pass are issued together.  The work of a warp of gemv_kernel and of a caller that gives its warps
// rows itself (a persistent program).
// F: operand formats (kF_*).  The scales of the x rows of a launch start at SX (the host offsets SX to the
// launch's first row, a multiple of 8, so in the swizzled layout row b of the launch is 16 b bytes on).
// CG: x and its block scales read through L2 for an intra-launch consumer.
template <int R, int NB, int XT, int F, class Out, bool CG = false>
__device__ __forceinline__ void gemv_warp(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
                                          const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m,
                                          int row0, int lane, int k0, int k1, Out &out) {
  float acc[R][NB];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[r][b] = 0.f;
  // swizzled scales: per row a pointer, lane l starting at scale column l and moving 32 columns (8 tiles of
  // 512 bytes) per iteration.  Rows of A past m read row 0.
  constexpr bool SWA = F & kF_ASWZ, SWX = XT == NVFP4A_NVFP4 && (F & kF_BSWZ);
  constexpr bool UEA = F & kF_AUE, UEX = XT == NVFP4A_NVFP4 && (F & kF_BUE);
  const uint8_t *psa[SWA ? R : 1], *psx = SX + (SWX ? swz_col(k0 / 16 + lane) : 0);
#pragma unroll
  for (int r = 0; r < (SWA ? R : 0); ++r) psa[r] = SA + swz_row(row0 + r < m ? row0 + r : 0, -ldsa) + swz_col(k0 / 16 + lane);

  for (int kb = k0 + lane * 16; kb < k1; kb += 512) {
    uint2 aw[R];
    float sa[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {  // rows past m read row 0 (not stored)
      const size_t row = row0 + r < m ? row0 + r : 0;
      aw[r] = *(const uint2 *)(A + row * (lda / 2) + kb / 2);
      if (SWA) { sa[r] = scale_val<UEA>(*psa[r]); psa[r] += 4096; }
      else sa[r] = scale_val<UEA>(SA[row * ldsa + kb / 16]);
    }
    if (XT == NVFP4A_NVFP4 || XT == NVFP4A_INT8) {
      // NVFP4 / INT8 x: the exact block sum p of every (row, column), then acc = fma(p * sa, sx, acc) -- the
      // values of the tiled kernels.  DP4A on int8 from sm_61, FFMA on integer-valued floats below.
      float sx[NB];
#pragma unroll
      for (int b = 0; b < NB; ++b)
        sx[b] = XT == NVFP4A_INT8 ? load_global<CG>((const float *)SX + (size_t)b * ldsx + kb / 16)
                : SWX ? scale_val<UEX>(load_global<CG>(psx + 16 * b))
                      : scale_val<UEX>(load_global<CG>(SX + (size_t)b * ldsx + kb / 16));
      if (SWX) psx += 4096;
#if __CUDA_ARCH__ >= 610
      uint4 xi[NB];
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        if (XT == NVFP4A_INT8) xi[b] = load_global<CG>((const uint4 *)(X + (size_t)b * ldx + kb));
        else { const uint2 w = load_global<CG>((const uint2 *)(X + (size_t)b * (ldx / 2) + kb / 2)); xi[b] = dec16(w.x, w.y); }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
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
        if (XT == NVFP4A_INT8) xw[b] = load_global<CG>((const uint4 *)(X + (size_t)b * ldx + kb));
        else { const uint2 w = load_global<CG>((const uint2 *)(X + (size_t)b * (ldx / 2) + kb / 2)); xw[b] = make_uint4(w.x, w.y, 0, 0); }
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
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
      // dense x: one FFMA per MAC with a = e2m1 * sa * 2^-9 (2^-15 for UE5M3 scales, up to 2^17), decoded
      // with a single FMUL (the raw E2M1 encoding value * 2^-126 times sa * 2^117 / 2^111, which cannot
      // overflow); the 2^9 / 2^15 returns before the epilogue, exactly
#pragma unroll
      for (int r = 0; r < R; ++r) sa[r] = mul_rn(sa[r], UEA ? 0x1p111f : 0x1p117f);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float xv[NB][8];
#pragma unroll
        for (int b = 0; b < NB; ++b) load8<XT, CG>(X + ((size_t)b * ldx + kb + 8 * h) * Elem<XT>::bytes, xv[b]);
#pragma unroll
        for (int r = 0; r < R; ++r)
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
  constexpr int NO = R * NB, PER = (NO + 31) / 32;
  float mine[PER];
#pragma unroll
  for (int t = 0; t < PER; ++t) mine[t] = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      float v = acc[r][b];
#pragma unroll
      for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
      v = __fmul_rn(v, XT == NVFP4A_NVFP4 ? 0.25f : XT == NVFP4A_INT8 ? 0.5f : UEA ? 32768.f : 512.f);  // exact
      if (((r * NB + b) & 31) == lane) mine[(r * NB + b) >> 5] = v;
    }
  out.begin();
#pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int o = t * 32 + lane, r = o / NB;
    if (o < NO && row0 + r < m) out.store(row0 + r, o - r * NB, mine[t]);
  }
  out.end();
}

// Block `block` of WARPS R rows (warp w: rows (block WARPS + w) R ..): gemv_warp of the block's warps.  `thread`: the
// thread in the CTA (threadIdx.x; a persistent caller passes it re-read, so nothing derived from it is kept live across
// its other work).  The work item of gemv_kernel (the whole k) and of a caller that splits k.
template <class C, int NB, int XT, int F, class Out, bool CG = false>
__device__ __forceinline__ void gemv_rows(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
                                          const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m,
                                          unsigned block, unsigned thread, int k0, int k1, Out &out) {
  const int lane = thread & 31, warp = thread >> 5;
  gemv_warp<C::R, NB, XT, F, Out, CG>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, (block * C::WARPS + warp) * C::R, lane, k0, k1, out);
}

// gemv_rows' output through the GEMM epilogue
template <bool CG = false>
struct EpiOut {
  const Epilogue &e;
  EpiState st;
  __device__ __forceinline__ void begin() { st = epi_begin(e); }
  __device__ __forceinline__ void store(int i, int j, float v) { epi_store<CG>(e, st, i, j, v); }
  __device__ __forceinline__ void end() { epi_end(e, st); }
};

template <class C, int NB, int XT, int F>
__global__ void __launch_bounds__(C::WARPS * 32)
gemv_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
            const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m, int k, Epilogue e) {
  if (!C::template On<NB, XT>::value) { __trap(); return; }
  EpiOut<> out{e, {}};
  gemv_rows<C, NB, XT, F>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, blockIdx.x, threadIdx.x, 0, k, out);
}

template <class C, int NB, int XT, int F = 0>
cudaError_t gemv_launch(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                        const uint8_t *SX, int ldsx, int m, int k, const Epilogue &e, cudaStream_t s) {
  const int rows = C::WARPS * C::R;
  gemv_kernel<C, NB, XT, F><<<(m + rows - 1) / rows, C::WARPS * 32, 0, s>>>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e);
  return cudaGetLastError();
}

// gemv_rows' output through the extended epilogue: the column scales of the block's NB columns (kEX_NORM), the values
// staged per block (WARPS R rows: whole m16 slices) for SwiGLU and the second output, finished by the block after its
// warps are done
template <class C, int NB, int EX, bool CG = false>
struct EpiOutEx {
  static constexpr int ROWS = C::WARPS * C::R;
  static_assert(ROWS % 16 == 0, "a block of whole m16 slices");
  const Epilogue &e;
  const EpilogueEx &x;
  float *stage;  // [NB][ROWS]
  int r0, m, lane, thread;
  EpiState st;
  __device__ __forceinline__ float *scales() { return stage + ((EX & (kEX_SWIGLU | kEX_OUT2)) ? NB * ROWS : 0); }
  __device__ __forceinline__ void begin() {
    st = epi_begin(e);
    if (EX & kEX_NORM) {
      // One warp forms each column scalar; every warp uses the same parts and butterfly order.
      if (thread < 32) {
        float cs[NB];
        ex_norm_scales<NB>(x, 0, NB, lane, cs);
#pragma unroll
        for (int b = 0; b < NB; ++b) if (lane == b) scales()[b] = cs[b];
      }
      __syncthreads();
    }
  }
  __device__ __forceinline__ void store(int i, int j, float acc) {
    const float c = (EX & kEX_NORM) ? scales()[j] : 1.f;
    const float v = ex_value<CG>(e, i, j, acc, c);
    if (!(EX & kEX_SWIGLU)) {
      store_any(e.d_dtype, e.d, (size_t)j * e.ldd + i, v);
      st.amax = fmaxf(st.amax, fabsf(v));
    }
    if (EX & (kEX_SWIGLU | kEX_OUT2)) stage[j * ROWS + i - r0] = v;
  }
  __device__ __forceinline__ void end() {
    if (EX & (kEX_SWIGLU | kEX_OUT2)) {
      __syncthreads();
      ex_finish<EX>(e, x, stage, ROWS, r0, ROWS / 16, 0, 1, NB, m, NB, thread, C::WARPS * 32, st);
    }
    epi_end(e, st);
  }
};

// The work item: block `item` of WARPS R rows (all of k) on the CTA's WARPS warps (thread: the thread in the CTA), its
// shared memory smem (gemv_smem_floats<C, NB, EX>() floats); EX = 0 the plain epilogue.  CG: mutable inputs through L2.
template <class C, int NB, int EX>
__host__ __device__ constexpr int gemv_smem_floats() {
  return ((EX & (kEX_SWIGLU | kEX_OUT2)) ? NB * C::WARPS * C::R : 0) + ((EX & kEX_NORM) ? NB : 0);
}
struct GemvItemArgs {
  const uint8_t *A, *SA, *X, *SX;
  int lda, ldsa, ldx, ldsx, m, k;
  Epilogue e;
  EpilogueEx x;
};
template <class C, int NB, int XT, int F, int EX, bool CG = true>
__device__ __forceinline__ void gemv_item(const GemvItemArgs &a, int item, unsigned thread, void *smem) {
  if (EX) {
    EpiOutEx<C, NB, EX, CG> out{a.e, a.x, (float *)smem, item * C::WARPS * C::R, a.m, (int)(thread & 31), (int)thread, {}};
    gemv_rows<C, NB, XT, F, decltype(out), CG>(a.A, a.lda, a.SA, a.ldsa, a.X, a.ldx, a.SX, a.ldsx, a.m, item, thread, 0, a.k, out);
  } else {
    EpiOut<CG> out{a.e, {}};
    gemv_rows<C, NB, XT, F, decltype(out), CG>(a.A, a.lda, a.SA, a.ldsa, a.X, a.ldx, a.SX, a.ldsx, a.m, item, thread, 0, a.k, out);
  }
}

template <class C, int XT, int EX, int N, bool CG = true>
__device__ __forceinline__ void gemv_group_item(const GemvItemArgs &base, int item, unsigned thread, void *smem) {
  const int mt = (base.m + C::WARPS * C::R - 1) / (C::WARPS * C::R), j0 = item / mt * 8;
  GemvItemArgs a = base;
  const auto bytes = [](int t) { return t == NVFP4A_FP32 ? 4 : 2; };
  a.X += (size_t)j0 * a.ldx * Elem<XT>::bytes;
  if (XT == NVFP4A_INT8) a.SX += (size_t)j0 * a.ldsx * 4;
  a.e.d = (uint8_t *)a.e.d + (size_t)j0 * a.e.ldd * bytes(a.e.d_dtype);
  if (a.e.c) a.e.c = (const uint8_t *)a.e.c + (size_t)j0 * a.e.ldc * bytes(a.e.c_dtype);
  if (a.e.sb) a.e.gb += j0;
  if (a.x.norm_parts) a.x.norm_parts += (size_t)j0 * a.x.norm_ld;
  if (a.x.d2) {
    a.x.d2 = (uint8_t *)a.x.d2 + (size_t)j0 * a.x.ldd2 * bytes(a.x.d2_dtype);
    a.x.sq += (size_t)j0 * a.x.ld_sq;
  }
  if (N % 8 && j0 + 8 > N)
    gemv_item<C, N % 8 ? N % 8 : 8, XT, 0, EX, CG>(a, item % mt, thread, smem);
  else gemv_item<C, 8, XT, 0, EX, CG>(a, item % mt, thread, smem);
}

// The EX tail lowers ptxas's register allocation of INT8 (127 -> 100) and with it the k-loop's
// memory-level parallelism: the loop runs at ~0.6x the plain kernel's bandwidth (the BF16 kernels'
// loops stay identical).  A 2-CTA floor restores the budget (124 registers, the plain kernel's load
// schedule; measured in handoff_fbla/tools/probe_gemv_ex.cu).  BF16 needs its full 133 registers
// (a 128 cap spills 8 bytes and slows gate_up by ~8%), so it keeps the plain kernel's bound.  The two
// variants are overloads on a hand-rolled SFINAE guard: <type_traits> does not compile under NVRTC.
template <bool B> struct GemvExOnly;  // (undefined when the guard fails)
template <> struct GemvExOnly<true> { using type = int; };

template <class C, int NB, int XT, int F, int EX>
__device__ __forceinline__ void gemv_ex_body(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
                                             const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m,
                                             int k, const Epilogue &e, const EpilogueEx &x) {
  if (!C::template On<NB, XT>::value) { __trap(); return; }
  __shared__ float stage[gemv_smem_floats<C, NB, EX>() ? gemv_smem_floats<C, NB, EX>() : 1];
  gemv_item<C, NB, XT, F, EX, false>(GemvItemArgs{A, SA, X, SX, lda, ldsa, ldx, ldsx, m, k, e, x}, blockIdx.x, threadIdx.x, stage);
}

template <class C, int NB, int XT, int F, int EX, typename GemvExOnly<XT == NVFP4A_INT8>::type = 0>
__global__ void __launch_bounds__(C::WARPS * 32, 2)
gemv_kernel_ex(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
               const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m, int k, Epilogue e, EpilogueEx x) {
  gemv_ex_body<C, NB, XT, F, EX>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, x);
}

template <class C, int NB, int XT, int F, int EX, typename GemvExOnly<XT != NVFP4A_INT8>::type = 0>
__global__ void __launch_bounds__(C::WARPS * 32)
gemv_kernel_ex(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
               const uint8_t *__restrict__ X, int ldx, const uint8_t *__restrict__ SX, int ldsx, int m, int k, Epilogue e, EpilogueEx x) {
  gemv_ex_body<C, NB, XT, F, EX>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, x);
}

template <class C, int NB, int XT, int F, int EX>
cudaError_t gemv_launch_ex(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx, const uint8_t *SX,
                           int ldsx, int m, int k, const Epilogue &e, const EpilogueEx &x, cudaStream_t s) {
  const int rows = C::WARPS * C::R;
  gemv_kernel_ex<C, NB, XT, F, EX><<<(m + rows - 1) / rows, C::WARPS * 32, 0, s>>>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, k, e, x);
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

// GEMV of NVFP4 A by n <= 8 rows of x (dtype xt, formats f) with the configuration the device's table selects.
cudaError_t launch_gemv(int xt, int f, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                        const uint8_t *SX, int ldsx, int m, int n, int k, const Epilogue &e, cudaStream_t s);

// Entry point of one kernel: config id x x dtype x n x formats, explicitly instantiated by the instance
// objects (gemv_nvfp4effma_inst.cu, one per id, x dtype and formats).
template <int ID, int XT, int NB, int F>
cudaError_t gemv_entry(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx, const uint8_t *SX,
                       int ldsx, int m, int k, const Epilogue &e, cudaStream_t s);

// The extended epilogue's GEMV (linear E4M3 scales, x BF16 or INT8): additions EX, 1 .. kEX_MASKS, n <= 8; other
// combinations return cudaErrorInvalidConfiguration.
constexpr bool gemv_ex_dtype(int xt) { return xt == NVFP4A_BF16 || xt == NVFP4A_INT8; }
cudaError_t launch_gemv_ex(int xt, int ex, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                           const uint8_t *SX, int ldsx, int m, int n, int k, const Epilogue &e, const EpilogueEx &x, cudaStream_t s);
template <int ID, int XT, int NB, int EX>
cudaError_t gemv_entry_ex(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx, const uint8_t *SX,
                          int ldsx, int m, int k, const Epilogue &e, const EpilogueEx &x, cudaStream_t s);
// the kernels (their attributes): plain of formats F, extended of additions EX
template <int ID, int XT, int NB, int F> const void *gemv_kernel_ptr();
template <int ID, int XT, int NB, int EX> const void *gemv_kernel_ex_ptr();
// the work item of a call on this device (x BF16 or INT8, A's scales linear E4M3, n <= 8, additions 0 .. kEX_MASKS): the
// configuration's rows per warp R and largest n MX, the library kernel of the same work; false: none
bool gemv_item_config(int xt, int ex, int n, int *R, int *MX, const void **kernel);

}  // namespace kern
}  // namespace nvfp4a

// configurations the library builds: the default (0) and those the tuned tables use
#define NVFP4A_GEMV_LIB(X) X(0) NVFP4A_GEMV_USED(X)
