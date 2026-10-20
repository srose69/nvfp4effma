// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMM: the tiled kernel templates, their configuration candidates and the entry points of
// the per-instance objects (gemm_nvfp4effma_inst.cu).  Host API and dispatch: gemm_nvfp4effma.cu.
//   gemm_kernel       FFMA, A NVFP4 x B BF16 / FP16 / FP32 (block scale applied when A is decoded into the
//                     FP32 tile), and NVFP4 x NVFP4 below sm_61 (integer-valued tiles, exact block sums)
//   gemm_dp4a_kernel  NVFP4 x NVFP4 / INT8 from sm_61: int8 tiles, 4 DP4A + 2 FFMA per output and block
// Both kernels compute the same NVFP4 x NVFP4 / INT8 values bit for bit.
#pragma once
#include <type_traits>
#include "nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

// ---------------------------------------------------------------------------------------------
// FFMA kernel: BM x BN block tile, BK = 16 (one NVFP4 block) k-slice, TM x TN FP32 register tile per
// thread (groups of 4 rows / columns spread over the block so that LDS.128 fragment reads are
// conflict-free).  Each operand element is converted to FP32 once per block into a single-buffered
// shared tile; the inner loop is LDS.128 + FFMA only.
// ---------------------------------------------------------------------------------------------
// On<..>::value: whether the kernel body is compiled for the architecture of the current device
// pass.  The library (gemm_nvfp4effma_inst.cu) compiles a configuration only where a table selects it
// and leaves a trapping stub elsewhere; the tuner keeps everything.
template <int BM_, int BN_, int TM_, int TN_>
struct GemmCfg {
  static constexpr int BM = BM_, BN = BN_, BK = 16, TM = TM_, TN = TN_;
  static constexpr int TX = BM / TM, TY = BN / TN, NT = TX * TY, PAD = 4;  // lanes run along m
  // resident blocks the register budget targets: 512 threads (<= 128 registers), e.g. 2 x 256 or 4 x 128;
  // 64-thread blocks (small problems, few blocks anyway) get 6 x 64, as 8 x 64 would make them spill
  static constexpr int MINB = NT >= 512 ? 1 : NT >= 128 ? 512 / NT : 6;
  template <int TB> struct On { static constexpr bool value = true; };
};

// R x BK tile of a K-contiguous operand of type T -> FP32 shared s[BK][R + PAD].  A chunk is E = 8
// consecutive k of one row (4 bytes of NVFP4), the BK / E chunks of a row go to adjacent lanes, so a warp
// reads whole row segments.  The transposing scalar stores would put the lanes of one row into one bank;
// instead the column of k-group kk / E is XOR-ed with swz(kk) < 32, which only moves it inside its aligned
// 32-column block: the lanes hit distinct banks, and fragment reads apply the same XOR (a compile-time
// constant per kk).  Rows past the end of the matrix read row 0: they only feed outputs that the
// epilogue does not store, so neither a branch nor zeroing is needed.
//   NVFP4, INT = false: values a * sa (exact)        NVFP4, INT = true: integers 2a, sa to ss[row]
//   INT8 (INT = true): the int8 values, their FP32 block scale to ss[row]
template <class C, int R, int T, bool INT>
struct TileLoader {
  static constexpr bool FP4 = T == NVFP4A_NVFP4, Q8 = T == NVFP4A_INT8;
  static constexpr int E = R * C::BK / C::NT < 8 ? R * C::BK / C::NT : 8;
  static constexpr int CH = R * C::BK / (E * C::NT), LPR = C::BK / E;  // chunks per thread, lanes per row
  static_assert(CH >= 1 && (R * C::BK) % (E * C::NT) == 0, "tile does not split into chunks");
  static_assert((E == 4 || E == 8) && (!FP4 || E == 8) && R % 32 == 0 && 32 % LPR == 0, "unsupported tile");
  static constexpr int RW = FP4 ? 1 : E * Elem<T>::bytes / 4;  // 32-bit words per chunk
  static __host__ __device__ constexpr int swz(int kk) { return (kk / E) * (32 / LPR); }
  const uint8_t *p[CH], *ps[CH];
  int dst[CH], row[CH];
  bool first[CH];  // the chunk at k = 0 of its row (stores the row's scale in INT mode)
  uint32_t raw[CH][RW], sc[CH];

  __device__ void init(const uint8_t *X, int ld, const uint8_t *S, int lds, int r0, int rmax) {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      const int w = threadIdx.x + C::NT * j, rl = w / LPR, kc = (w % LPR) * E;
      const size_t r = r0 + rl < rmax ? r0 + rl : 0;
      p[j] = FP4 ? X + r * (ld / 2) + kc / 2 : X + (r * ld + kc) * Elem<FP4 ? NVFP4A_FP32 : T>::bytes;
      ps[j] = FP4 ? S + r * lds : Q8 ? S + r * lds * 4 : nullptr;
      dst[j] = kc * (R + C::PAD) + (rl ^ swz(kc));
      row[j] = rl; first[j] = kc == 0;
    }
  }
  // Global data stays raw until the store, so the next tile can be in flight in a few registers while
  // the current one is computed.
  __device__ __forceinline__ void fetch() {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      if (FP4) { raw[j][0] = *(const uint32_t *)p[j]; sc[j] = *ps[j]; }
      else if (RW == 2) {
        const uint2 w = *(const uint2 *)p[j]; raw[j][0] = w.x; raw[j][1] = w.y;
        if (Q8) sc[j] = *(const uint32_t *)ps[j];
      }
      else {
#pragma unroll
        for (int q = 0; q < RW; q += 4) {
          const uint4 w = *(const uint4 *)(p[j] + 4 * q);
          raw[j][q] = w.x; raw[j][q + 1] = w.y; raw[j][q + 2] = w.z; raw[j][q + 3] = w.w;
        }
      }
    }
  }
  // next k-tile; on the last one the pointers stay, so the final (unused) fetch re-reads valid memory
  __device__ __forceinline__ void advance(bool more) {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      p[j] += more ? (FP4 ? C::BK / 2 : C::BK * Elem<FP4 ? NVFP4A_FP32 : T>::bytes) : 0;
      if (FP4 || Q8) ps[j] += more ? (FP4 ? 1 : 4) : 0;
    }
  }
  __device__ __forceinline__ void store(float *s, float *ss) const {
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      float v[E];
      if (FP4) {
        const float sa = e4m3_value(sc[j]);
#pragma unroll
        for (int q = 0; q < E; ++q) v[q] = INT ? e2m1_value<1>(raw[j][0], q) : mul_rn(e2m1_value<0>(raw[j][0], q), sa);
        if (INT && first[j]) ss[row[j]] = sa;
      } else if (Q8) {
#pragma unroll
        for (int q = 0; q < E; ++q) v[q] = (float)(int8_t)(raw[j][q / 4] >> (8 * (q % 4)));
        if (INT && first[j]) ss[row[j]] = __uint_as_float(sc[j]);
      } else {
#pragma unroll
        for (int q = 0; q < RW; ++q) {
          if (T == NVFP4A_FP32) v[q] = __uint_as_float(raw[j][q]);
          else unpack2<T>(raw[j][q], v[2 * q], v[2 * q + 1]);
        }
      }
#pragma unroll
      for (int q = 0; q < E; ++q) s[dst[j] + q * (R + C::PAD)] = v[q];
    }
  }
};

// Staged epilogue: pass j stages one column of every thread's register tile -- BM rows x NC columns --
// in shared memory s[c * BM + r], then each thread stores BM * NC / NT of them, consecutive threads on
// consecutive rows.  One inlined epilogue in a loop instead of a call per output: no call frame, no
// stack, and the stores coalesce along m.  col(j, c) maps a staged column to its column in D.
template <int BM, int NC, int NT, class Col>
__device__ __forceinline__ void epi_staged_pass(const Epilogue &e, EpiState &st, const float *s, int m0, int m, int n,
                                                int j, Col col) {
#pragma unroll 1
  for (int q = 0; q < BM * NC / NT; ++q) {
    const int el = q * NT + threadIdx.x, r = el % BM, c = el / BM;
    const int gi = m0 + r, gj = col(j, c);
    if (gi < m && gj < n) epi_store(e, st, gi, gj, s[c * BM + r]);
  }
}

template <class C, int TB>
__global__ void __launch_bounds__(C::NT, TB == NVFP4A_NVFP4 || TB == NVFP4A_INT8 ? 1 : C::MINB)
gemm_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
            const uint8_t *__restrict__ B, int ldb, const uint8_t *__restrict__ SB, int ldsb, int m, int n, int k,
            Epilogue e) {
  if (!C::template On<TB>::value) { __trap(); return; }
  // NVFP4 x NVFP4 / INT8: exact block sums, then the two scale FFMAs; the result times kOut
  constexpr bool INT = TB == NVFP4A_NVFP4 || TB == NVFP4A_INT8;
  constexpr float kOut = TB == NVFP4A_INT8 ? 0.5f : 0.25f;
  __shared__ __align__(16) float sA[C::BK * (C::BM + C::PAD)];
  __shared__ __align__(16) float sB[C::BK * (C::BN + C::PAD)];
  __shared__ __align__(16) float ssA[INT ? C::BM : 1], ssB[INT ? C::BN : 1];
  const int tx = threadIdx.x % C::TX, ty = threadIdx.x / C::TX;
  const int m0 = blockIdx.y * C::BM, n0 = blockIdx.x * C::BN;
  using LA = TileLoader<C, C::BM, NVFP4A_NVFP4, INT>;
  using LB = TileLoader<C, C::BN, TB, INT>;
  static_assert(C::TX * 4 >= 32 && C::TY * 4 >= 32, "fragment XOR must stay inside a 32-column block");
  LA la; la.init(A, lda, SA, ldsa, m0, m);
  LB lb; lb.init(B, ldb, SB, ldsb, n0, n);

  float acc[C::TM][C::TN];
#pragma unroll
  for (int i = 0; i < C::TM; ++i)
#pragma unroll
    for (int j = 0; j < C::TN; ++j) acc[i][j] = 0.f;

  // The next tile's global loads are issued before the current tile is computed and consumed only
  // after it: their latency hides behind the block's own FFMAs, not only behind other blocks.
  const int kts = k / C::BK;
  la.fetch(); lb.fetch();
  for (int kt = 0; kt < kts; ++kt) {
    if (kt) __syncthreads();  // every warp finished reading the previous tile
    la.store(sA, ssA); lb.store(sB, ssB);
    __syncthreads();           // tile visible to every warp
    la.advance(kt + 1 < kts); lb.advance(kt + 1 < kts);
    la.fetch(); lb.fetch();
    float p[INT ? C::TM : 1][INT ? C::TN : 1];  // INT: the block sums of this tile
    if (INT) {
#pragma unroll
      for (int i = 0; i < C::TM; ++i)
#pragma unroll
        for (int j = 0; j < C::TN; ++j) p[INT ? i : 0][INT ? j : 0] = 0.f;
    }
#pragma unroll
    for (int kk = 0; kk < C::BK; ++kk) {
      float a[C::TM], b[C::TN];
#pragma unroll
      for (int g = 0; g < C::TM / 4; ++g)
        *reinterpret_cast<float4 *>(&a[4 * g]) =
            *reinterpret_cast<const float4 *>(&sA[kk * (C::BM + C::PAD) + g * C::TX * 4 + ((tx * 4) ^ LA::swz(kk))]);
#pragma unroll
      for (int g = 0; g < C::TN / 4; ++g)
        *reinterpret_cast<float4 *>(&b[4 * g]) =
            *reinterpret_cast<const float4 *>(&sB[kk * (C::BN + C::PAD) + g * C::TY * 4 + ((ty * 4) ^ LB::swz(kk))]);
      // FFMA order: pairs of rows, zigzag over j (consecutive FFMAs share a or b); this halves the
      // register-bank conflicts ptxas leaves compared with row-by-row order
#pragma unroll
      for (int pp = 0; pp < C::TM / 2; ++pp) {
        const int i0 = 2 * pp, i1 = i0 + 1;
#pragma unroll
        for (int jj = 0; jj < C::TN; ++jj) {
          const int j = (pp & 1) ? C::TN - 1 - jj : jj;
          float &x0 = INT ? p[INT ? i0 : 0][INT ? j : 0] : acc[i0][j];
          float &x1 = INT ? p[INT ? i1 : 0][INT ? j : 0] : acc[i1][j];
          if (jj & 1) { x1 = fma_rn(a[i1], b[j], x1); x0 = fma_rn(a[i0], b[j], x0); }
          else { x0 = fma_rn(a[i0], b[j], x0); x1 = fma_rn(a[i1], b[j], x1); }
        }
      }
    }
    if (INT) {  // acc = fma(p * sa, sb, acc): p * sa is exact, one rounding per block
      float sa[C::TM], sb[C::TN];
#pragma unroll
      for (int g = 0; g < C::TM / 4; ++g)
        *reinterpret_cast<float4 *>(&sa[4 * g]) = *reinterpret_cast<const float4 *>(&ssA[g * C::TX * 4 + tx * 4]);
#pragma unroll
      for (int g = 0; g < C::TN / 4; ++g)
        *reinterpret_cast<float4 *>(&sb[4 * g]) = *reinterpret_cast<const float4 *>(&ssB[g * C::TY * 4 + ty * 4]);
#pragma unroll
      for (int i = 0; i < C::TM; ++i)
#pragma unroll
        for (int j = 0; j < C::TN; ++j) acc[i][j] = fma_rn(mul_rn(p[INT ? i : 0][INT ? j : 0], sa[i]), sb[j], acc[i][j]);
    }
  }

  // Epilogue, staged through sA one column of the thread tile at a time (BM rows x TY columns).
  static_assert(C::TY <= C::BK && C::BM % 4 == 0, "a staged column must fit in sA");
  EpiState st = epi_begin(e);
#pragma unroll 1
  for (int j = 0; j < C::TN; ++j) {  // run-time j: one copy of the epilogue
    __syncthreads();  // sA free: the k loop, or the previous pass, is done with it
#pragma unroll
    for (int jj = 0; jj < C::TN; ++jj)  // column j of acc: one branch per column, acc stays in registers
      if (jj == j) {
#pragma unroll
        for (int g = 0; g < C::TM / 4; ++g)
          *reinterpret_cast<float4 *>(&sA[ty * C::BM + g * C::TX * 4 + tx * 4]) =
              INT ? make_float4(acc[4 * g][jj] * kOut, acc[4 * g + 1][jj] * kOut, acc[4 * g + 2][jj] * kOut, acc[4 * g + 3][jj] * kOut)
                  : make_float4(acc[4 * g][jj], acc[4 * g + 1][jj], acc[4 * g + 2][jj], acc[4 * g + 3][jj]);
      }
    __syncthreads();
    epi_staged_pass<C::BM, C::TY, C::NT>(e, st, sA, m0, m, n, j,
                                          [&](int jp, int c) { return n0 + (jp / 4) * C::TY * 4 + c * 4 + (jp & 3); });
  }
  epi_end(e, st);
}

template <class C, int TB>
cudaError_t gemm_launch(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, int ldb,
                        const uint8_t *SB, int ldsb, int m, int n, int k, const Epilogue &e, cudaStream_t s) {
  dim3 grid((n + C::BN - 1) / C::BN, (m + C::BM - 1) / C::BM);
  gemm_kernel<C, TB><<<grid, C::NT, 0, s>>>(A, lda, SA, ldsa, B, ldb, SB, ldsb, m, n, k, e);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------
// DP4A kernel (sm_61+), NVFP4 x NVFP4: block 128 x 128, 256 threads, 8 x 8 outputs per thread (rows
// tx + 16i, columns ty + 16j), k-tile = 4 NVFP4 blocks.  Nibbles are decoded once per block into int8
// tiles (PRMT lookups); per output and block, 4 DP4A starting at the bits of 1.5 * 2^23 give the FP32
// value 1.5 * 2^23 + p of the exact block sum p, y = fma(x, sa, -1.5 * 2^23 * sa) = p * sa exactly, and
// acc = fma(y, sb, acc) rounds once.  Double-buffered tiles, one barrier per k-tile: tile t + 1 is
// decoded from registers at the start of iteration t and the loads of tile t + 2 are issued right after.
// One resident block per SM (153 registers): two blocks at 128 registers lose more to the DP4A chains'
// scheduling than they gain in occupancy.
// ---------------------------------------------------------------------------------------------
// 4 consecutive values i .. i + 3 of a BF16 / FP16 / FP32 array; vec: aligned for one vector access
template <int T>
__device__ __forceinline__ void load4_any(const void *p, size_t i, bool vec, float v[4]) {
  if (vec && T == NVFP4A_FP32) { const float4 a = *(const float4 *)((const float *)p + i); v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; }
  else if (vec) { const uint2 w = *(const uint2 *)((const uint16_t *)p + i); unpack2<T>(w.x, v[0], v[1]); unpack2<T>(w.y, v[2], v[3]); }
  else for (int q = 0; q < 4; ++q) v[q] = load_any(T, p, i + q);
}
template <int T>
__device__ __forceinline__ void store4_any(void *p, size_t i, bool vec, const float v[4]) {
  if (vec && T == NVFP4A_FP32) *(float4 *)((float *)p + i) = make_float4(v[0], v[1], v[2], v[3]);
  else if (vec && T == NVFP4A_BF16)
    *(uint2 *)((uint16_t *)p + i) = make_uint2(__byte_perm(rne_hi(__float_as_uint(v[0])), rne_hi(__float_as_uint(v[1])), 0x7632),
                                               __byte_perm(rne_hi(__float_as_uint(v[2])), rne_hi(__float_as_uint(v[3])), 0x7632));
  else if (vec) {
    const __half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    *(uint2 *)((uint16_t *)p + i) = make_uint2(*reinterpret_cast<const uint32_t *>(&a), *reinterpret_cast<const uint32_t *>(&b));
  } else for (int q = 0; q < 4; ++q) store_any(T, p, i + q, v[q]);
}

// BN_ x TN_: 128 x 8 (the large tile) or 64 x 4 (for n up to a few hundred, where 128-wide tiles are half
// empty and too few to fill the GPU); MINB_ resident blocks per SM.
template <int BN_, int TN_, int MINB_>
struct Dp4aCfgT {
  static constexpr int BM = 128, BN = BN_, TX = 16, TY = 16, TM = 8, TN = TN_, NT = 256, KBT = 4, MINB = MINB_;
  static_assert(BN == TY * TN && TN % 4 == 0 && BN <= BM, "unsupported DP4A tile");
};
using Dp4aCfg = Dp4aCfgT<128, 8, 1>;
using Dp4aCfgNarrow = Dp4aCfgT<64, 4, 2>;  // ~120 registers: two blocks per SM

template <class C, int TB>
__global__ void __launch_bounds__(C::NT, C::MINB)
gemm_dp4a_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
                 const uint8_t *__restrict__ B, int ldb, const uint8_t *__restrict__ SB, int ldsb, int m, int n,
                 int k, Epilogue e) {
#if __CUDA_ARCH__ < 610
  __trap();
#else
  constexpr int BM = C::BM, BN = C::BN, KBT = C::KBT;
  __shared__ __align__(16) uint4 sAB[2][2][KBT * BM];  // [A / B][buffer][block][row]: 16 int8 per block and row
  uint4 (*sA2)[KBT * BM] = sAB[0], (*sB2)[KBT * BM] = sAB[1];  // B buffers sized for BM >= BN
  __shared__ __align__(16) float2 ssa2[2][KBT][BM];  // (sa, -1.5 * 2^23 * sa)
  __shared__ __align__(16) float ssb2[2][KBT][BN];   // [block][ty * TN + j]
  const int tid = threadIdx.x, tx = tid % C::TX, ty = tid / C::TX;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  // loads: every thread takes two blocks of one row of A (16 bytes) and of B (16 bytes of NVFP4, 32 of
  // INT8) per k-tile; threads 0..127 the 4 block scales of a row of A (one 32-bit word), threads 128..255
  // those of a row of B (a word of 4 E4M3, or 4 floats for INT8).  Rows past the end of the matrix read
  // row 0 (outputs that are not stored).
  constexpr bool Q8 = TB == NVFP4A_INT8;
  constexpr float kOut = Q8 ? 0.5f : 0.25f;  // the A values are doubled; NVFP4 B values too
  constexpr int BW = Q8 ? 2 : 1;  // uint4 per thread and k-tile of B
  const int row = tid >> 1, half = tid & 1;
  const uint4 *pa = (const uint4 *)(A + (size_t)(m0 + row < m ? m0 + row : 0) * (lda / 2)) + half;
  const uint4 *pb = (const uint4 *)(B + (size_t)(n0 + row < n ? n0 + row : 0) * (Q8 ? ldb : ldb / 2)) + BW * half;
  const bool sa_thr = tid < BM;
  const int srow = tid & (BM - 1);
  const bool b_row = row < BN, b_srow = srow < BN;  // narrow tiles: only these threads carry B
  const size_t brow = n0 + srow < n ? n0 + srow : 0;
  const uint32_t *ps = (const uint32_t *)(sa_thr ? SA + (size_t)(m0 + srow < m ? m0 + srow : 0) * ldsa
                                                 : (Q8 ? SA : SB + brow * ldsb));
  const float4 *psb = Q8 ? (const float4 *)((const float *)SB + brow * ldsb) : nullptr;
  const int sbslot = srow % C::TY * C::TN + srow / C::TY;  // B row n0 + ty + TY * j -> slot ty * TN + j
  float acc[C::TM][C::TN];
#pragma unroll
  for (int i = 0; i < C::TM; ++i)
#pragma unroll
    for (int j = 0; j < C::TN; ++j) acc[i][j] = 0.f;
  uint4 ra = pa[0], rb = pb[0], rb2 = Q8 ? pb[1] : make_uint4(0, 0, 0, 0);
  uint32_t rs = sa_thr || !Q8 ? ps[0] : 0;
  float4 rsb = Q8 && !sa_thr ? psb[0] : make_float4(0.f, 0.f, 0.f, 0.f);
  const int tiles = k / (16 * KBT);
  auto decode = [&](int buf) {
    uint4 *sA = sA2[buf], *sB = sB2[buf];
    sA[(2 * half) * BM + row] = dec16(ra.x, ra.y);
    sA[(2 * half + 1) * BM + row] = dec16(ra.z, ra.w);
    if (b_row) {
      if (Q8) { sB[(2 * half) * BN + row] = rb; sB[(2 * half + 1) * BN + row] = rb2; }
      else { sB[(2 * half) * BN + row] = dec16(rb.x, rb.y); sB[(2 * half + 1) * BN + row] = dec16(rb.z, rb.w); }
    }
    if (sa_thr) {
#pragma unroll
      for (int b = 0; b < KBT; ++b) { const float v = e4m3_value((rs >> (8 * b)) & 0xFF); ssa2[buf][b][srow] = make_float2(v, -12582912.f * v); }
    } else if (!b_srow) {
    } else if (Q8) {
      ssb2[buf][0][sbslot] = rsb.x; ssb2[buf][1][sbslot] = rsb.y; ssb2[buf][2][sbslot] = rsb.z; ssb2[buf][3][sbslot] = rsb.w;
    } else {
#pragma unroll
      for (int b = 0; b < KBT; ++b) ssb2[buf][b][sbslot] = e4m3_value((rs >> (8 * b)) & 0xFF);
    }
  };
  auto load_next = [&] {
    pa += 2; ra = *pa;
    pb += 2 * BW; rb = pb[0];
    if (Q8) rb2 = pb[1];
    if (Q8 && !sa_thr) rsb = *++psb; else rs = *++ps;
  };
  decode(0);
  if (tiles > 1) load_next();
  __syncthreads();
  for (int t = 0; t < tiles; ++t) {
    if (t + 1 < tiles) {
      decode((t + 1) & 1);
      if (t + 2 < tiles) load_next();
    }
    const uint4 *sA = sA2[t & 1], *sB = sB2[t & 1];
    const float2 (*ssa)[BM] = ssa2[t & 1];
    const float (*ssb)[BN] = ssb2[t & 1];
#pragma unroll 1
    for (int kb = 0; kb < KBT; ++kb) {
#pragma unroll
      for (int jh = 0; jh < C::TN; jh += 4) {
        uint4 bv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) bv[j] = sB[kb * BN + ty + C::TY * (jh + j)];
        const float4 sb = *(const float4 *)&ssb[kb][ty * C::TN + jh];
        const float sbv[4] = {sb.x, sb.y, sb.z, sb.w};
#pragma unroll
        for (int i = 0; i < C::TM; ++i) {
          const uint4 av = sA[kb * BM + tx + C::TX * i];
          const float2 sn = ssa[kb][tx + C::TX * i];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            int p = __dp4a((int)av.x, (int)bv[j].x, 0x4B400000);
            p = __dp4a((int)av.y, (int)bv[j].y, p);
            p = __dp4a((int)av.z, (int)bv[j].z, p);
            p = __dp4a((int)av.w, (int)bv[j].w, p);
            const float y = fma_rn(__int_as_float(p), sn.x, sn.y);
            acc[i][jh + j] = fma_rn(y, sbv[j], acc[i][jh + j]);
          }
        }
      }
    }
    __syncthreads();
  }

  // Epilogue (the operations of epi_store; C is read by the thread that writes the same element of D, so C
  // may alias D).  With one resident block per SM nothing overlaps it.  Without C every output is stored
  // straight from the registers, an interior tile without bias or amax skipping the per-output checks.
  const auto direct = [&](auto tc, bool full) {
    constexpr int T = decltype(tc)::value;
    EpiState st = epi_begin(e);
    float bv[C::TM];
#pragma unroll
    for (int i = 0; i < C::TM; ++i) {
      const int gi = m0 + tx + C::TX * i;
      bv[i] = !full && e.bias ? load_any(e.bias_dtype, e.bias, gi < m ? gi : 0) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < C::TN; ++j)
#pragma unroll
      for (int i = 0; i < C::TM; ++i) {
        const int gi = m0 + tx + C::TX * i, gj = n0 + ty + C::TY * j;
        float v = __fmul_rn(acc[i][j] * kOut, st.s);
        if (full) { store_any(T, e.d, (size_t)gj * e.ldd + gi, v); continue; }
        if (gi < m && gj < n) {
          if (e.bias) v = __fadd_rn(v, bv[i]);
          store_any(T, e.d, (size_t)gj * e.ldd + gi, v);
          st.amax = fmaxf(st.amax, fabsf(v));
        }
      }
    if (!full) epi_end(e, st);
  };
  // C / accumulation: staged through the tile buffers (32 KB) in two halves of 4 columns per thread, so
  // that a warp reads C and writes D along 128 consecutive rows of one column with vector accesses.
  const auto staged = [&](auto dc, auto cc) {
    constexpr int DT = decltype(dc)::value, CT = decltype(cc)::value;
    float *stage = reinterpret_cast<float *>(sAB);  // [4 * TY columns][BM rows]
    EpiState st = epi_begin(e);
    const auto vec_ok = [](const void *p, int ld, int bytes) { return ((uintptr_t)p % (4 * bytes)) == 0 && ld % 4 == 0; };
    const bool vc = vec_ok(e.c, e.ldc, Elem<CT>::bytes), vd = vec_ok(e.d, e.ldd, Elem<DT>::bytes);
#pragma unroll
    for (int h = 0; h < C::TN / 4; ++h) {
      __syncthreads();  // the k loop, or the previous half, is done with the buffers
#pragma unroll
      for (int jl = 0; jl < 4; ++jl)
#pragma unroll
        for (int i = 0; i < C::TM; ++i) stage[(jl * C::TY + ty) * BM + tx + C::TX * i] = __fmul_rn(acc[i][4 * h + jl] * kOut, st.s);
      __syncthreads();
#pragma unroll 1
      for (int q = 0; q < 4 * C::TY * BM / 4 / C::NT; ++q) {
        const int g = q * C::NT + tid, c = g / (BM / 4), r = (g % (BM / 4)) * 4;
        const int gi = m0 + r, gj = n0 + c % C::TY + C::TY * (4 * h + c / C::TY);
        if (gi >= m || gj >= n) continue;
        const bool all4 = gi + 3 < m;
        float v[4], cv[4];
        *reinterpret_cast<float4 *>(v) = *reinterpret_cast<const float4 *>(&stage[c * BM + r]);
        load4_any<CT>(e.c, (size_t)gj * e.ldc + gi, all4 && vc, cv);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          if (e.bias && gi + u < m) v[u] = __fadd_rn(v[u], load_any(e.bias_dtype, e.bias, gi + u));
          v[u] = __fmaf_rn(e.beta, cv[u], v[u]);
          if (gi + u < m) st.amax = fmaxf(st.amax, fabsf(v[u]));
        }
        if (all4) store4_any<DT>(e.d, (size_t)gj * e.ldd + gi, vd, v);
        else for (int u = 0; u < 4 && gi + u < m; ++u) store_any(DT, e.d, (size_t)gj * e.ldd + gi + u, v[u]);
      }
    }
    epi_end(e, st);
  };
  if (e.beta != 0.f) {
    const auto by_c = [&](auto dc) {
      if (e.c_dtype == NVFP4A_BF16) staged(dc, std::integral_constant<int, NVFP4A_BF16>());
      else if (e.c_dtype == NVFP4A_FP16) staged(dc, std::integral_constant<int, NVFP4A_FP16>());
      else staged(dc, std::integral_constant<int, NVFP4A_FP32>());
    };
    if (e.d_dtype == NVFP4A_BF16) by_c(std::integral_constant<int, NVFP4A_BF16>());
    else if (e.d_dtype == NVFP4A_FP16) by_c(std::integral_constant<int, NVFP4A_FP16>());
    else by_c(std::integral_constant<int, NVFP4A_FP32>());
    return;
  }
  const bool full = !e.bias && !e.amax && m0 + BM <= m && n0 + BN <= n;
  if (e.d_dtype == NVFP4A_BF16) full ? direct(std::integral_constant<int, NVFP4A_BF16>(), true) : direct(std::integral_constant<int, NVFP4A_BF16>(), false);
  else if (e.d_dtype == NVFP4A_FP16) full ? direct(std::integral_constant<int, NVFP4A_FP16>(), true) : direct(std::integral_constant<int, NVFP4A_FP16>(), false);
  else full ? direct(std::integral_constant<int, NVFP4A_FP32>(), true) : direct(std::integral_constant<int, NVFP4A_FP32>(), false);
#endif
}

// Candidates of the dense-B FFMA kernel: id, BM, BN, TM, TN.  All share the 8 x 8 thread tile (same
// FFMA : LDS ratio); the block tile only trades parallelism on small grids, which depends on the SM
// count.  Ids are what src/nvfp4effma_tuned.h stores.
#define NVFP4A_GEMM_CANDIDATES(X) \
  X(0, 128, 128, 8, 8)            \
  X(1, 64, 64, 8, 8)              \
  X(2, 128, 64, 8, 8)             \
  X(3, 64, 128, 8, 8)
#define NVFP4A_DEF_GEMM(id, bm, bn, tm, tn) using GemmCfg##id = GemmCfg<bm, bn, tm, tn>;
NVFP4A_GEMM_CANDIDATES(NVFP4A_DEF_GEMM)
#undef NVFP4A_DEF_GEMM

}  // namespace kern
}  // namespace nvfp4a

// configurations the library builds: the defaults (0, 1) and those the tuned tables use
#define NVFP4A_GEMM_LIB(X) X(0) X(1) NVFP4A_GEMM_USED(X)

// entry point of one instance object: config id x B dtype
#define NVFP4A_GEMM_ENTRY(id, tb) NVFP4A_CAT6(gemm_entry_, id, _, tb, , )
#define NVFP4A_GEMM_PROTO(id, tb)                                                                       \
  cudaError_t NVFP4A_GEMM_ENTRY(id, tb)(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *B, \
                                        int ldb, const uint8_t *SB, int ldsb, int m, int n, int k,               \
                                        const nvfp4a::kern::Epilogue &e, cudaStream_t s)
