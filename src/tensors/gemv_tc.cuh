// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMV on tensor cores (sm_75 .. sm_89; BF16 x from sm_80): D = alpha * A x^T + ..., n <= 8 columns of x,
// A as its image (image.cuh, the GEMM's: A16 for BF16 x, A8 for INT8 x with FP32 block scales and for NVFP4 x), x as it
// is.  Bound by the DRAM stream of A's image, read once.
//
// CTA = one m16 slice of A, all of k; its KS warps split the slice's groups (4 k-tiles) into contiguous ranges.  Every
// load is at an immediate offset from one pointer per stream, stepped once per group: A's 4 k-tiles (a warp's 512
// contiguous bytes each, L1 bypassed, L2 evict-first) and their scales, x and its scales of the group.  A warp keeps P
// groups in flight in registers, no shared memory on the way.  Per k-tile:
//   BF16 x   lane (g, t) multiplies block t of rows g, g + 8 (decoded to BF16 a * sa exactly) with x's 16 values of
//            column g at the same k: 4 BF16 mma (their 16 k need not be one block: the scale is in A)
//   INT8 /   per block j an integer mma (its 16 k one block): 1.5 * 2^23 + p of the exact block sum, acc = fma(fma(D,
//   NVFP4 x  sa, -1.5 * 2^23 sa), sb, acc).  The lane loads block t of column g, a 4 x 4 transpose within the quad
//            (words of INT8 x, halfwords of NVFP4 x; SHFL) gives each lane its part of every block
// x's columns past n read column n - 1 (computed, not stored).  The k sums stay in registers to the end: the warps'
// partial sums of an output added in warp order through shared memory (deterministic), one store per output.
// The extended epilogue (EX, its descriptor the trailing parameter): warp w forms s_w after its k range; the slice's
// values go through shared memory to ex_finish.
#pragma once
#include "gemm_tc.cuh"  // dec4o, the images

namespace nvfp4a {
namespace kern {

constexpr int kGvWarps = 8;  // KS: warps of a CTA (the k split)
// formats of the GEMV beyond kF_*: NVFP4 x's scale rows 16-byte aligned with k a multiple of 256 -- its scales of a
// group (4 k-tiles) in one 16-byte load (else a word a k-tile; the swizzled layout: always)
constexpr int kF_XS16 = 16;

// shared memory of a slice (floats): the warps' partial sums; the extended epilogue's s_j and staged values
__host__ __device__ constexpr int gemv_tc_smem_floats(int ex) { return kGvWarps * 4 * 32 + (ex ? kGvWarps + 8 * 16 : 0); }

// The work item: m16 slice sl of A (all of k) on the CTA's kGvWarps warps (thread: the thread in the CTA), its shared
// memory smem (gemv_tc_smem_floats(EX) floats).  gemv_tc_kernel runs one a CTA; an executor may run it from its own kernel.
// CG: x, its block/global scales and C through L2; the immutable weight image retains its read-only loads.
template <int XT, int F, int EX, bool CG = false>
__device__ __forceinline__ void gemv_tc_slice(const uint8_t *__restrict__ ia, const uint8_t *__restrict__ x, int ldx,
                                              const uint8_t *__restrict__ sx, int ldsx, int m, int n, int kt, const Epilogue &e,
                                              const EpilogueEx &xd, int sl, unsigned thread, float *smem) {
#if !defined(NVFP4A_TC) || (!defined(NVFP4A_TC80) && XT != NVFP4A_NVFP4 && XT != NVFP4A_INT8)  // (W4A16: sm_80+)
  __trap();
#else
  using namespace tc;
  constexpr bool Q8 = XT == NVFP4A_INT8, N4 = XT == NVFP4A_NVFP4, H = !Q8 && !N4;
  // P: groups in flight per warp.  A group is already 2304 bytes of A in flight per warp, several times its share of the
  // DRAM's latency-bandwidth product; the integer paths take one (two CTAs an SM within 128 registers: the same bytes in
  // flight, twice the warps to hide the decode), W4A16 two (two CTAs an SM either way)
  constexpr int KS = kGvWarps, P = H ? 2 : 1;
  constexpr bool UEA = F & kF_AUE, UEB = N4 && (F & kF_BUE), SWB = N4 && (F & kF_BSWZ), S16 = N4 && !SWB && (F & kF_XS16);
  constexpr float kOut = H ? 1.f : Q8 ? 0.5f : 0.25f;  // integer paths: the A values are doubled, NVFP4 x values too
  // per group: x bytes of a lane's column, x scale bytes of a column; NVFP4 x scales per k-tile (no S16): a word ST apart
  constexpr int XG = H ? 512 : Q8 ? 256 : 128, SG = Q8 ? 64 : SWB ? 2048 : 16, ST = SWB ? 512 : 4;
  float (*red)[4][32] = reinterpret_cast<float (*)[4][32]>(smem);
  const int lane = thread & 31, w = thread >> 5, g = lane >> 2, t4 = lane & 3;
  const int ng = (kt + 3) >> 2, q = ng / KS, r = ng - q * KS;
  const int N = q + (w < r), g0 = w * q + (w < r ? w : r);  // the warp's groups g0 .. g0 + N - 1
  const int nlast = kt - 4 * (ng - 1);                     // k-tiles of the slice's last group (1 .. 4)
  const uint8_t *pa = ia + ((size_t)sl * ng + g0) * GV + 16 * lane;
  const uint8_t *ps = ia + ((size_t)sl * ng + g0) * GV + 2048 + (H ? 8 * lane : 32 * g);
  const int cg = g < n ? g : n - 1, c0 = 2 * t4 < n ? 2 * t4 : n - 1, c1 = 2 * t4 + 1 < n ? 2 * t4 + 1 : n - 1;
  const uint8_t *px = H ? x + ((size_t)cg * ldx + 256 * g0 + 16 * t4) * 2 : Q8 ? x + (size_t)cg * ldx + 256 * g0 + 16 * t4
                                                                         : x + (size_t)cg * (ldx / 2) + 128 * g0 + 8 * t4;
  const uint8_t *pb0 = nullptr, *pb1 = nullptr;
  if (Q8) {
    pb0 = sx + ((size_t)c0 * ldsx + 16 * g0) * 4; pb1 = sx + ((size_t)c1 * ldsx + 16 * g0) * 4;
  } else if (N4) {
    pb0 = sx + (SWB ? swz_row(c0, -ldsx) + swz_col(16 * g0) : (size_t)c0 * ldsx + 16 * g0);
    pb1 = sx + (SWB ? swz_row(c1, -ldsx) + swz_col(16 * g0) : (size_t)c1 * ldsx + 16 * g0);
  }
  const uint64_t pol = l2_evict_first();  // A's L2 policy (ldg_stream)
  struct Stage {
    uint4 a[4];  // A: k-tile T's words (word j: A16 the mma j, A8 block j)
    uint4 s[2];  // A scales: A16 s[0].x, .y (tile T: bytes 2 T, 2 T + 1 of 8); A8 tiles 0, 1 / 2, 3 (word row g, g + 8)
    uint4 x[8];  // x: BF16 x[2 T], x[2 T + 1] (the lane's 16 values); INT8 x[T] (block t of column g); NVFP4 x[T].x, .y
                 // (block t of column g)
    uint4 q[8];  // x scales: INT8 q[T] column c0, q[4 + T] c1 (FP32 per block); NVFP4 q[0], q[1] columns c0, c1 (word T: tile T)
  };
  // group loads; vt: the group's k-tiles (4 but in the slice's last group; 0 past the warp's groups: a zero stage, whose
  // consumption adds exact zeros), x and its scales only for those
  const auto load = [&](Stage &st, int vt) {
    const auto input = [](const auto *p) { return CG ? __ldcg(p) : __ldg(p); };
    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int T = 0; T < 4; ++T) st.a[T] = ldg_stream(pa + 512 * T, vt > 0, pol);
    if (H) {
      const uint2 v = vt > 0 ? __ldg(reinterpret_cast<const uint2 *>(ps)) : make_uint2(0, 0);
      st.s[0].x = v.x; st.s[0].y = v.y;
    } else {
      st.s[0] = ldg_stream(ps, vt > 0, pol);
      st.s[1] = ldg_stream(ps + 16, vt > 0, pol);
    }
#pragma unroll
    for (int T = 0; T < 4; ++T) {
      const bool in = T < vt;
      if (H) {
        st.x[2 * T] = in ? input(reinterpret_cast<const uint4 *>(px + 128 * T)) : z;
        st.x[2 * T + 1] = in ? input(reinterpret_cast<const uint4 *>(px + 128 * T + 16)) : z;
      } else if (Q8) {
        st.x[T] = in ? input(reinterpret_cast<const uint4 *>(px + 64 * T)) : z;
        st.q[T] = in ? input(reinterpret_cast<const uint4 *>(pb0 + 16 * T)) : z;
        st.q[4 + T] = in ? input(reinterpret_cast<const uint4 *>(pb1 + 16 * T)) : z;
      } else {
        const uint2 v = in ? input(reinterpret_cast<const uint2 *>(px + 32 * T)) : make_uint2(0, 0);
        st.x[T].x = v.x; st.x[T].y = v.y;
        if (!S16) {
          (&st.q[0].x)[T] = in ? input(reinterpret_cast<const uint32_t *>(pb0 + ST * T)) : 0u;
          (&st.q[1].x)[T] = in ? input(reinterpret_cast<const uint32_t *>(pb1 + ST * T)) : 0u;
        }
      }
    }
    if (S16) {  // (no partial groups: k a multiple of 256)
      st.q[0] = vt > 0 ? input(reinterpret_cast<const uint4 *>(pb0)) : z;
      st.q[1] = vt > 0 ? input(reinterpret_cast<const uint4 *>(pb1)) : z;
    }
    pa += GV; ps += GV; px += XG; pb0 += SG; pb1 += SG;
  };

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const uint32_t magic[4] = {kMagic + zero_of(lane), kMagic + zero_of(lane ^ 1), kMagic + zero_of(lane ^ 2), kMagic + zero_of(lane ^ 3)};
  const auto word = [](const uint4 &v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; };
  const auto consume = [&](const Stage &st) {
#pragma unroll
    for (int T = 0; T < 4; ++T) {
      if (H) {
#ifdef NVFP4A_TC80
        // the lane's block t of rows g, g + 8: 0x40 + 2a -> BF16 192 + 2a -> fma(., sa / 2, -96 sa) = a sa
        const uint32_t sw = (T < 2 ? st.s[0].x : st.s[0].y) >> (16 * (T & 1));
        const float sa0 = scale_val<UEA>(sw & 0xFF), sa1 = scale_val<UEA>((sw >> 8) & 0xFF);
        const auto pair = [](float v) { const uint32_t h = __float_as_uint(v) >> 16; return h | h << 16; };
        const uint32_t x0 = pair(0.5f * sa0), y0 = pair(-96.f * sa0), x1 = pair(0.5f * sa1), y1 = pair(-96.f * sa1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t aw = word(st.a[T], j), lo = dec4o(aw), hi = dec4o(aw >> 16);
          const uint32_t a0 = fma_bf16x2(prmt(lo, 0x43434343u, 0x4140u), x0, y0), a2 = fma_bf16x2(prmt(lo, 0x43434343u, 0x4342u), x0, y0);
          const uint32_t a1 = fma_bf16x2(prmt(hi, 0x43434343u, 0x4140u), x1, y1), a3 = fma_bf16x2(prmt(hi, 0x43434343u, 0x4342u), x1, y1);
          const uint4 &xv = st.x[2 * T + (j >> 1)];
          mma_bf16(acc, a0, a1, a2, a3, word(xv, 2 * (j & 1)), word(xv, 2 * (j & 1) + 1));
        }
#endif
      } else {
        uint32_t b[4];
        if (Q8) {  // the quad's 4 x 4 transpose: lane t has block t's words, gets word t of every block
          const uint4 v = st.x[T];
          const bool b1 = t4 & 1, b2 = t4 & 2;
          const uint32_t s0 = b1 ? v.x : v.y, s1 = b1 ? v.z : v.w, k0 = b1 ? v.y : v.x, k1 = b1 ? v.w : v.z;
          const uint32_t r0 = __shfl_xor_sync(0xFFFFFFFFu, s0, 1), r1 = __shfl_xor_sync(0xFFFFFFFFu, s1, 1);
          // u_ai: block (t & 2) | a, word 2 i + (t & 1)
          const uint32_t u00 = b1 ? r0 : k0, u10 = b1 ? k0 : r0, u01 = b1 ? r1 : k1, u11 = b1 ? k1 : r1;
          const uint32_t d0 = b2 ? u00 : u01, d1 = b2 ? u10 : u11, p0 = b2 ? u01 : u00, p1 = b2 ? u11 : u10;
          const uint32_t e0 = __shfl_xor_sync(0xFFFFFFFFu, d0, 2), e1 = __shfl_xor_sync(0xFFFFFFFFu, d1, 2);
          b[0] = b2 ? e0 : p0; b[1] = b2 ? e1 : p1; b[2] = b2 ? p0 : e0; b[3] = b2 ? p1 : e1;
        } else {  // the quad's 4 x 4 transpose of halfwords: lane t has block t's (2 words), gets halfword t of every block
          const uint32_t w0 = st.x[T].x, w1 = st.x[T].y;  // halfwords 0, 1 / 2, 3 (nibbles 4 i .. 4 i + 3)
          const bool b1 = t4 & 1, b2 = t4 & 2;
          // bit 1 (xor 2): keep word c = t / 2 of the block, get the partner block's (t ^ 2) word c
          const uint32_t kw = b2 ? w1 : w0, pw = __shfl_xor_sync(0xFFFFFFFFu, b2 ? w0 : w1, 2);
          // kw, pw: halfwords 2 c, 2 c + 1 of blocks t, t ^ 2; bit 0 (xor 1): keep halfword 2 c + b of both, send the other
          const uint32_t keep = prmt(kw, pw, b1 ? 0x7632u : 0x5410u), got = __shfl_xor_sync(0xFFFFFFFFu, prmt(kw, pw, b1 ? 0x5410u : 0x7632u), 1);
          // keep: halfword t of blocks t (low), t ^ 2 (high); got: of blocks t ^ 1 (low), t ^ 3 (high)
          const uint32_t lo_b2 = b2 ? keep >> 16 : keep, hi_b2 = b2 ? keep : keep >> 16;  // blocks (t & 1), (t & 1) | 2
          const uint32_t lo_g2 = b2 ? got >> 16 : got, hi_g2 = b2 ? got : got >> 16;      // blocks (t & 1) ^ 1, .. | 2
          b[0] = dec4(b1 ? lo_g2 : lo_b2); b[1] = dec4(b1 ? lo_b2 : lo_g2);
          b[2] = dec4(b1 ? hi_g2 : hi_b2); b[3] = dec4(b1 ? hi_b2 : hi_g2);
        }
        const uint4 &sv = st.s[T >> 1];
        const uint32_t sg = (T & 1) ? sv.z : sv.x, sg8 = (T & 1) ? sv.w : sv.y;  // rows g, g + 8: byte j = block j
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t aw = word(st.a[T], j);
          uint32_t d[4];
          imma_block(d, dec4(aw), dec4(aw >> 16), b[j], magic);
          const float sa0 = scale_val<UEA>((sg >> (8 * j)) & 0xFF), sa1 = scale_val<UEA>((sg8 >> (8 * j)) & 0xFF);
          float sb0, sb1;
          if (Q8) {
            sb0 = __uint_as_float(word(st.q[T], j)); sb1 = __uint_as_float(word(st.q[4 + T], j));
          } else {
            sb0 = scale_val<UEB>((word(st.q[0], T) >> (8 * j)) & 0xFF); sb1 = scale_val<UEB>((word(st.q[1], T) >> (8 * j)) & 0xFF);
          }
          const float n0 = -kMagicF * sa0, n1 = -kMagicF * sa1;
          acc[0] = fma_rn(fma_rn(__uint_as_float(d[0]), sa0, n0), sb0, acc[0]);
          acc[1] = fma_rn(fma_rn(__uint_as_float(d[1]), sa0, n0), sb1, acc[1]);
          acc[2] = fma_rn(fma_rn(__uint_as_float(d[2]), sa1, n1), sb0, acc[2]);
          acc[3] = fma_rn(fma_rn(__uint_as_float(d[3]), sa1, n1), sb1, acc[3]);
        }
      }
    }
  };

  // the ring: stage j holds group i + j.  Every warp runs the same Q groups, the busiest warp's count (uniform: no
  // branch around the mma and shuffles), past its own N zero stages; the slice's last group may hold fewer k-tiles
  const int Q = (ng + KS - 1) / KS;
  const bool lastw = g0 + N == ng;  // the warp holds the slice's last group
  const auto vt_of = [&](int gi) { return gi < N ? (lastw && gi == N - 1 ? nlast : 4) : 0; };
  Stage st[P];
#pragma unroll
  for (int j = 0; j < P; ++j) load(st[j], vt_of(j));
#pragma unroll 1
  for (int i = 0; i < Q; i += P) {
#pragma unroll
    for (int j = 0; j < P; ++j) { consume(st[j]); load(st[j], vt_of(i + j + P)); }
  }

  // the warps' partial sums -> one per output, added in warp order; lanes of warps 0 - 3 store output (lane, w)
#pragma unroll
  for (int u = 0; u < 4; ++u) red[w][u][lane] = acc[u];
  if (EX) {
    float *scs = smem + KS * 4 * 32, *stage = scs + KS;  // s_j; values [col][row in the slice]
    // s_w after the warp's own k range (before it, its dependent loads would hold back the stream of A)
    const float csw = (EX & kEX_NORM) ? ex_norm_scale(xd, w < n ? w : n - 1, lane) : 1.f;
    if (lane == 0) scs[w] = csw;
    __syncthreads();
    EpiState es = epi_begin(e);
    if (w < 4) {
      float v = red[0][w][lane];
#pragma unroll
      for (int u = 1; u < KS; ++u) v += red[u][w][lane];
      const int r = g + 8 * (w >> 1), row = sl * 16 + r, col = 2 * t4 + (w & 1);
      if (row < m && col < n) {
        v = ex_value<CG>(e, row, col, v * kOut, scs[col]);
        if (!(EX & kEX_SWIGLU)) {
          store_any(e.d_dtype, e.d, (size_t)col * e.ldd + row, v);
          es.amax = fmaxf(es.amax, fabsf(v));
        }
        if (EX & (kEX_SWIGLU | kEX_OUT2)) stage[col * 16 + r] = v;
      }
    }
    if (EX & (kEX_SWIGLU | kEX_OUT2)) {
      __syncthreads();
      ex_finish<EX>(e, xd, stage, 16, sl * 16, 1, 0, 1, 8, m, n, thread, KS * 32, es);
    }
    epi_end(e, es);
    return;
  }
  __syncthreads();
  if (w < 4) {
    float v = red[0][w][lane];
#pragma unroll
    for (int u = 1; u < KS; ++u) v += red[u][w][lane];
    const int row = sl * 16 + g + 8 * (w >> 1), col = 2 * t4 + (w & 1);
    EpiState es = epi_begin(e);
    if (row < m && col < n) epi_store<CG>(e, es, row, col, v * kOut);
    epi_end(e, es);
  }
#endif
}

template <int XT, int F, int EX = 0, class... X>
__global__ void __launch_bounds__(kGvWarps * 32)
gemv_tc_kernel(const uint8_t *__restrict__ ia, const uint8_t *__restrict__ x, int ldx, const uint8_t *__restrict__ sx, int ldsx,
               int m, int n, int kt, Epilogue e, X... xs) {
  __shared__ __align__(16) float smem[gemv_tc_smem_floats(EX)];
  gemv_tc_slice<XT, F, EX>(ia, x, ldx, sx, ldsx, m, n, kt, e, ex_arg(xs...), blockIdx.x, threadIdx.x, smem);
}

// the work item's arguments (one slice: item = sl)
struct GemvTcItemArgs {
  const uint8_t *ia, *x, *sx;
  int ldx, ldsx, m, n, kt;
  Epilogue e;
  EpilogueEx xd;
};
template <int XT, int F, int EX, bool CG = true>
__device__ __forceinline__ void gemv_tc_item(const GemvTcItemArgs &a, int item, unsigned thread, void *smem) {
  gemv_tc_slice<XT, F, EX, CG>(a.ia, a.x, a.ldx, a.sx, a.ldsx, a.m, a.n, a.kt, a.e, a.xd, item, thread, (float *)smem);
}

// Entry point (src/tensors/gemv_tc_inst.cu): x dtype XT (NVFP4, INT8, BF16), formats F (kF_AUE; NVFP4 x kF_BUE, kF_BSWZ,
// kF_XS16)
template <int XT, int F>
cudaError_t gemv_tc_entry(const uint8_t *ia, const uint8_t *x, int ldx, const uint8_t *sx, int ldsx, int m, int n, int k,
                          const Epilogue &e, cudaStream_t s);
// the extended epilogue's (src/tensors/gemv_tc_inst.cu with -DNVFP4A_EX): x INT8 or BF16, A's scales E4M3, additions EX
template <int XT, int F> const void *gemv_tc_kernel_ptr();
template <int XT, int EX> const void *gemv_tc_kernel_ex_ptr();
template <int XT, int EX>
cudaError_t gemv_tc_entry_ex(const uint8_t *ia, const uint8_t *x, int ldx, const uint8_t *sx, int ldsx, int m, int n, int k,
                             const Epilogue &e, const EpilogueEx &xd, cudaStream_t s);

}  // namespace kern
}  // namespace nvfp4a
