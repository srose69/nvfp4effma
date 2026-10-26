// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// NVFP4EffMA GEMM on tensor cores (sm_75 .. sm_89; BF16 B from sm_80), streaming the operands' images (image.cuh).
//   NVFP4 / INT8 B  per output and block of 16 k one integer mma gives 1.5 * 2^23 + p of the exact block sum p
//                   (mma.cuh); y = fma(1.5 * 2^23 + p, sa, -1.5 * 2^23 sa) = p sa exactly, acc = fma(y, sb, acc)
//   BF16 B          A decoded to BF16 a * sa (exact), one BF16 mma per 16 k with FP32 accumulation
// Then the DP4A kernel's epilogue (times 1/4 for NVFP4 B, 1/2 for INT8: the decoded values are doubled).
//
// CTA = 128 x 128 outputs, 8 compute warps (warp tile 64 x 32 for NVFP4 B, 32 x 64 otherwise) and the producers.  NS
// slots of shared memory, a slot = one k-tile: A's 8 slices, B's record, the scales converted in consumption order.
// K-tile u in slot u % NS:
//   copy     the copier waits on the slot's EMPTY named barrier (an mbarrier arrive is a MEMBAR on sm_8x, too dear for
//            every thread and k-tile), cp.async of A and B arriving on the slot's LANDED mbarrier
//   publish  the publisher loads A's scales from the image, waits for LANDED, writes the converted scales, arrives on FULL
//   consume  each compute warp waits for FULL on its own, runs the k-tile, arrives on EMPTY
// sm_75 (no cp.async, no mbarrier): four producer warps load through registers, a sequence number per slot.
#pragma once
#include "../gemm_nvfp4effma.cuh"
#include "image.cuh"

namespace nvfp4a {
namespace kern {

// 8 compute warps and the producers: sm_80+ a copier and a publisher; sm_75 four producer warps (no cp.async: their
// loads go through registers, eight 16-byte chunks a thread in flight)
constexpr int kTcProducers80 = 2, kTcProducers75 = 4;
// threads of the variant compiled for this device pass (its register bound: sm_80+ 320 threads, 204 registers each)
#ifdef NVFP4A_TC80
constexpr int kTcThreads = 256 + 32 * kTcProducers80;
#else
constexpr int kTcThreads = 256 + 32 * kTcProducers75;
#endif

// 4 nibbles (low 16 bits) -> 4 bytes 0x40 + 2 value: dec4 with its lookup tables offset by 0x40 (under a high byte
// 0x43 a byte b is the BF16 128 + b)
__device__ __forceinline__ uint32_t dec4o(uint32_t w16) {
  const uint32_t P0 = 0x43424140u, P1 = 0x4C484644u, N0 = 0x3D3E3F40u, N1 = 0x34383A3Cu;
  const uint32_t sel = w16 & 0x7777u;
  const uint32_t pos = prmt(P0, P1, sel), neg = prmt(N0, N1, sel);
  const uint32_t m = prmt(w16, w16 << 4, 0x9D8Cu);
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(r) : "r"(pos), "r"(neg), "r"(m));  // m ? neg : pos
  return r;
}
// dec4 (nvfp4effma.cuh) with its low lookup tables in registers the caller holds (p0 = 0x03020100, n0 = 0xFDFEFF00, made
// data-dependent with zero_of: as constants ptxas rematerializes them from uniform registers at every decode)
__device__ __forceinline__ uint32_t dec4r(uint32_t w16, uint32_t p0, uint32_t n0) {
  const uint32_t sel = w16 & 0x7777u;
  const uint32_t pos = prmt(p0, 0x0C080604u, sel), neg = prmt(n0, 0xF4F8FAFCu, sel);
  const uint32_t m = prmt(w16, w16 << 4, 0x9D8Cu);
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(r) : "r"(pos), "r"(neg), "r"(m));  // m ? neg : pos
  return r;
}
// slot layout: A's data (8 slices of the k-tile), B record, A's scales in consumption order, B's (NVFP4); NS slots in
// the shared memory of a CTA (sm_80+ 99 KB: five, BF16 four of 24 KB; sm_75 64 KB: three)
template <int TB, bool V75> struct TcSlot {
  static constexpr bool Q8 = TB == NVFP4A_INT8, N4 = TB == NVFP4A_NVFP4;
  static constexpr int A = 0, B = tc::RA, BS = B + (Q8 ? tc::RD8 : tc::RD4), CA = B + tc::rb_bytes<TB>(),
                       CB = CA + 4096, BYTES = CB + (N4 ? 2048 : 0), NS = V75 ? 3 : N4 || Q8 ? 5 : 4;
};
// the slots, then 16 bytes a slot of barriers (mbarriers FULL, LANDED; sm_75 the slot's sequence number)
template <int TB> __host__ __device__ constexpr int tc_smem_bytes(bool v75) {
  return v75 ? TcSlot<TB, true>::NS * (TcSlot<TB, true>::BYTES + 16) : TcSlot<TB, false>::NS * (TcSlot<TB, false>::BYTES + 16);
}

// F: the scale formats kF_AUE / kF_BUE (the images hold every layout linear)
template <int TB, int F>
__global__ void __launch_bounds__(kTcThreads, 1)
gemm_tc_kernel(const uint8_t *__restrict__ ia, const uint8_t *__restrict__ ib, int m, int n, int kt, Epilogue e, float *ws,
               unsigned *flags) {
#if !defined(NVFP4A_TC) || (!defined(NVFP4A_TC80) && TB != NVFP4A_NVFP4 && TB != NVFP4A_INT8)  // (W4A16: sm_80+)
  __trap();
#else
  using namespace tc;
#ifdef NVFP4A_TC80
  constexpr bool V75 = false;
#else
  constexpr bool V75 = true;
#endif
  using L = TcSlot<TB, V75>;
  constexpr bool Q8 = L::Q8, N4 = L::N4, H = !Q8 && !N4;  // H: BF16 B (W4A16)
  constexpr int RB = rb_bytes<TB>(), SLOT = L::BYTES, NS = L::NS, NC = 256;  // NC: compute threads
  constexpr int NP = V75 ? kTcProducers75 : kTcProducers80, EC = NC + (V75 ? 32 * NP : 32);  // EC: an EMPTY's count
  constexpr int BM = 128, BN = 128, WM = N4 ? 2 : 4, WN = 8 / WM, TM = BM / WM, TN = BN / WN, MT = TM / 16, NJ = TN / 8;
  constexpr float kOut = H ? 1.f : Q8 ? 0.5f : 0.25f;  // integer paths: the A values are doubled, NVFP4 B values too
  constexpr bool UEA = F & kF_AUE, UEB = F & kF_BUE;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int wm = warp / WN, wn = warp % WN;
  // Persistent CTAs over units = (output tile, k-tile), tiles rastered in groups of GM row blocks (tile_of: a wave's A
  // and B records stay in L2).  W full waves of whole tiles (CTA c: tiles c, c + G, ..); with a workspace the last,
  // partial wave split stream-K into even shares of its units: a tile cut between CTAs is finished by the one holding
  // its k-tile 0, last, from the FP32 partials of the others (ws; flags back to 0 once read).  The split is recomputed
  // from the parameters and re-read %ctaid where needed, so none of it stays live through the compute warps' hot loop.
  constexpr int GM = 8;
  struct Split { int c, G, W, beg; long long usk; };  // usk: stream-K units
  const auto split = [&] {
    Split s;
    s.c = (int)ctaid_x(); s.G = (int)nctaid_x();
    const int ntiles = ((n + BN - 1) / BN) * ((m + BM - 1) / BM);
    s.W = ws ? ntiles / s.G : (s.c < ntiles ? (ntiles - 1 - s.c) / s.G + 1 : 0);
    s.usk = ws ? (long long)(ntiles - s.W * s.G) * kt : 0;
    s.beg = (int)(s.usk * s.c / s.G);
    return s;
  };
  const auto locate_in = [&](const Split &s, int u, int &lin, int &kk) {  // unit u -> tile lin, k-tile kk
    if (u < s.W * kt) { const int i = u / kt; lin = s.c + i * s.G; kk = u - i * kt; }
    else { const int gs = s.beg + u - s.W * kt, q = gs / kt; lin = s.W * s.G + q; kk = gs - q * kt; }
  };
  const auto locate = [&](int u, int &lin, int &kk) { locate_in(split(), u, lin, kk); };
  const auto tile_of = [&](int lin, int &mb, int &nb) {
    const int nt = (n + BN - 1) / BN, mt = (m + BM - 1) / BM;
    const int grp = lin / (GM * nt), first = grp * GM, gsz = min(GM, mt - first), t = lin - grp * GM * nt;
    mb = first + t % gsz; nb = t / gsz;
  };
  int U;
  {
    const Split s = split();
    U = s.W * kt + (int)(s.usk * (s.c + 1) / s.G) - s.beg;
  }
  // A's image: m16 slice sl's group gr at ia + (sl ng + gr) GV; a k-tile kk of row block mb: slices 8 mb .. 8 mb + 7,
  // group kk / 4, its data at 512 (kk % 4), the group's scales at 2048
  const int ng = (kt + 3) >> 2;
  const size_t sstride = (size_t)ng * GV;  // slice to slice
  const auto a_tile = [&](int mb, int kk) { return ia + (size_t)mb * 8 * sstride + (size_t)(kk >> 2) * GV; };  // (group)
  const uint32_t full = smem_u32(smem) + NS * SLOT;  // + 8 slot (sm_75: + 4 slot, the sequence)
  constexpr uint32_t EMPTY = 1;  // EMPTY + slot (barrier 0: __syncthreads)
#ifdef NVFP4A_TC80
  const uint8_t *pb = ib + 16 * lane;
  const uint32_t s0 = smem_u32(smem) + 16 * lane, landed = full + 8 * NS;  // + 8 slot
  if (tid < NS) {
    mbar_init(full + 8 * tid, 32);
    mbar_init(landed + 8 * tid, 32);
  }
#else
  if (tid < NS) seq_set(full + 4 * tid, 0u);
#endif
  __syncthreads();

  // the scale bytes in consumption order (the publisher / the sm_75 producers), from the image (ga: the k-tile's group of
  // slice 0, T = kk % 4) into the slot:
  //   A: rows g and g + 8 of a slice -> [slice 8][block 4][g 8]: integer paths sa FP32 (8 bytes, A8: the words of tile T
  //      at 32 g + 8 T); BF16 (sa / 2, -96 sa) as BF16 pairs (16 bytes, exact: <= 5 significant bits; A16: lane (g, t)'s
  //      block t, bytes 8 (4 g + t) + 2 T, + 1)
  //   B (NVFP4): raw [pair 8][t 4][slice 2][c 2] words -> [block 4][slice quad 4][t 4][slice 4][c 2] floats
  // (load and conversion apart: the publisher issues the loads before its wait for the copy)
  struct RawA { uint4 u0, u1; };
  const auto load_a = [&](const uint8_t *ga, int T, int x) {  // pair x = 8 slice + g
    const uint8_t *sc = ga + (x >> 3) * sstride + 2048 + 32 * (x & 7);
    RawA r;
    if (H) {
      r.u0 = __ldg(reinterpret_cast<const uint4 *>(sc)); r.u1 = __ldg(reinterpret_cast<const uint4 *>(sc + 16));
    } else {
      const uint2 v = __ldg(reinterpret_cast<const uint2 *>(sc + 8 * T));
      r.u0.x = v.x; r.u0.y = v.y;
    }
    return r;
  };
  const auto conv_a = [&](uint8_t *p, const RawA &r, int T, int x) {
    uint32_t w0 = 0, w1 = 0;  // byte kb: block kb's scale of row g (w0), g + 8 (w1)
    if (H) {
      const uint4 &u0 = r.u0, &u1 = r.u1;
      // u16 4 t + T: (row g, row g + 8) of block t
      const uint32_t h[4] = {(T < 2 ? u0.x : u0.y) >> (16 * (T & 1)), (T < 2 ? u0.z : u0.w) >> (16 * (T & 1)),
                             (T < 2 ? u1.x : u1.y) >> (16 * (T & 1)), (T < 2 ? u1.z : u1.w) >> (16 * (T & 1))};
#pragma unroll
      for (int t = 0; t < 4; ++t) { w0 |= (h[t] & 0xFFu) << (8 * t); w1 |= ((h[t] >> 8) & 0xFFu) << (8 * t); }
    } else {
      w0 = r.u0.x; w1 = r.u0.y;
    }
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const float v0 = scale_val<UEA>((w0 >> (8 * kb)) & 0xFF), v1 = scale_val<UEA>((w1 >> (8 * kb)) & 0xFF);
      const int at = ((x >> 3) * 4 + kb) * 8 + (x & 7);
      if (H) {
        const auto pair = [](float v) { const uint32_t h = __float_as_uint(v) >> 16; return h | h << 16; };
        reinterpret_cast<uint4 *>(p + L::CA)[at] = make_uint4(pair(0.5f * v0), pair(-96.f * v0), pair(0.5f * v1), pair(-96.f * v1));
      } else {
        reinterpret_cast<float2 *>(p + L::CA)[at] = make_float2(v0, v1);  // (-1.5 * 2^23 sa: at the mma, keeps fragments small)
      }
    }
  };
  const auto conv_b = [&](uint8_t *p, int l) {  // l = 4 pair + t
    const int pr = l >> 2, t = l & 3;
    const uint4 w = reinterpret_cast<const uint4 *>(p + L::BS)[l];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
      reinterpret_cast<float4 *>(p + L::CB)[((kb * 4 + (pr >> 1)) * 4 + t) * 2 + (pr & 1)] =
          make_float4(scale_val<UEB>((w.x >> (8 * kb)) & 0xFF), scale_val<UEB>((w.y >> (8 * kb)) & 0xFF),
                      scale_val<UEB>((w.z >> (8 * kb)) & 0xFF), scale_val<UEB>((w.w >> (8 * kb)) & 0xFF));
  };
#ifdef NVFP4A_TC80
  // k-tile u into slot u % NS (the copier, lane l: A's 8 slices' 16-byte chunk l, B's chunks l + 32 i), completion
  // arriving on LANDED.  a: A's k-tile kk group of slice 0 (+ 16 lane), b: B's record (+ 16 lane)
  const auto copy = [&](int u, const uint8_t *a, int kk, const uint8_t *b) {
    const uint32_t d = s0 + (u % NS) * SLOT;
    a += 512 * (kk & 3);
#pragma unroll
    for (int i = 0; i < 8; ++i) cp_async16(d + 512 * i, a + i * sstride);
#pragma unroll
    for (int i = 0; i < RB / 512; ++i) cp_async16(d + L::B + 512 * i, b + 512 * i);
    cp_async_mbar_arrive(landed + 8 * (u % NS));
  };
  // the publisher: k-tile u's copy landed, its scales (from the image) converted, FULL
  const auto publish = [&](int u, const uint8_t *ga, int kk) {
    const RawA r0 = load_a(ga, kk & 3, lane), r1 = load_a(ga, kk & 3, lane + 32);
    mbar_wait(landed + 8 * (u % NS), (u / NS) & 1);
    uint8_t *p = smem + (u % NS) * SLOT;
    conv_a(p, r0, kk & 3, lane); conv_a(p, r1, kk & 3, lane + 32);
    if (N4) conv_b(p, lane);
    mbar_arrive(full + 8 * (u % NS));
  };
#else
  // sm_75: the producers (thread q = tid - NC) produce k-tile u: their share of its 16-byte chunks (LDG -> STS: A's
  // slices i / 32's chunk i % 32, then B's record), barrier, their share of the conversions, barrier, the slot's
  // sequence u + 1
  constexpr uint32_t PROD = 14;  // the producers' named barrier
  const auto produce = [&](const Split &s, int u) {
    const int q = tid - NC;
    int lin, kk, mb, nb;
    locate_in(s, u, lin, kk);
    tile_of(lin, mb, nb);
    uint8_t *p = smem + (u % NS) * SLOT;
    const uint8_t *ga = a_tile(mb, kk), *a = ga + 512 * (kk & 3), *b = ib + ((size_t)nb * kt + kk) * RB;
    constexpr int CH = (RA + RB) / 16, PER = (CH + 32 * NP - 1) / (32 * NP);
    static_assert(L::B == RA, "A and B contiguous in a slot");
    uint4 v[PER];
#pragma unroll
    for (int x = 0; x < PER; ++x) {
      const int i = q + 32 * NP * x;
      if (i < CH) v[x] = __ldg(reinterpret_cast<const uint4 *>(i < RA / 16 ? a + (i >> 5) * sstride + 16 * (i & 31) : b + 16 * (i - RA / 16)));
    }
#pragma unroll
    for (int x = 0; x < PER; ++x) {
      const int i = q + 32 * NP * x;
      if (i < CH) reinterpret_cast<uint4 *>(p)[i] = v[x];
    }
    bar_sync<32 * NP>(PROD);
    if (q < 64) conv_a(p, load_a(ga, kk & 3, q), kk & 3, q);
    else if (N4 && q < 96) conv_b(p, q - 64);
    bar_sync<32 * NP>(PROD);
    if (q == 0) seq_set(full + 4 * (u % NS), (uint32_t)u + 1);
  };
#endif

  float acc[MT][NJ][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int u = 0; u < 4; ++u) acc[i][j][u] = 0.f;
  const uint32_t magic[4] = {kMagic + zero_of(lane), kMagic + zero_of(lane ^ 1), kMagic + zero_of(lane ^ 2), kMagic + zero_of(lane ^ 3)};
  const uint32_t tp0 = 0x03020100u + zero_of(lane ^ 4), tn0 = 0xFDFEFF00u + zero_of(lane ^ 5);  // dec4r's tables

  // W4A16: ldmatrix lane base in the B record (lanes 8 q + r: column 8 (q / 2) + r of an n8 pair, k chunk q % 2) and the
  // chunk swizzle's lane part (chunk 2 kb + q % 2 at (2 kb + q % 2) ^ r = 2 kb ^ (q % 2 ^ r))
#ifdef NVFP4A_TC80
  const uint32_t bh_lane = (uint32_t)(wn * TN + ((lane >> 4) << 3) + (lane & 7)) * 128, bh_lo = ((lane >> 3) & 1) ^ (lane & 7);
#endif
  // W4A16: the fragments of one block (A decoded and scaled: a * sa exactly in BF16), loaded one block ahead of their
  // mma -- across k-tiles too, so the HMMA pipe runs on through the slot handover
  struct FragH { uint32_t a[MT][4], b[NJ][2]; };
  const auto load_h = [&](FragH &f, const uint8_t *p, int kb) {
#ifdef NVFP4A_TC80
    const uint32_t pb_s = smem_u32(p + L::B) + bh_lane;
#pragma unroll
    for (int x = 0; x < NJ / 2; ++x) {
      uint32_t r[4];
      ldsm_x4(r, pb_s + 2048 * x + ((bh_lo ^ (2 * kb)) << 4));
      f.b[2 * x][0] = r[0]; f.b[2 * x][1] = r[1]; f.b[2 * x + 1][0] = r[2]; f.b[2 * x + 1][1] = r[3];
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      // 0x40 + 2a of k 16 t + 4 kb .. + 3 of block t (A16; row g: lo, g + 8: hi) -> BF16 192 + 2a -> fma(., sa / 2, -96 sa)
      // = a sa, sa the lane's block t's
      const uint32_t w = reinterpret_cast<const uint32_t *>(p)[(32 * (MT * wm + i) + lane) * 4 + kb], lo = dec4o(w), hi = dec4o(w >> 16);
      const uint4 sn = reinterpret_cast<const uint4 *>(p + L::CA)[((MT * wm + i) * 4 + t4) * 8 + g];
      f.a[i][0] = fma_bf16x2(prmt(lo, 0x43434343u, 0x4140u), sn.x, sn.y); f.a[i][2] = fma_bf16x2(prmt(lo, 0x43434343u, 0x4342u), sn.x, sn.y);
      f.a[i][1] = fma_bf16x2(prmt(hi, 0x43434343u, 0x4140u), sn.z, sn.w); f.a[i][3] = fma_bf16x2(prmt(hi, 0x43434343u, 0x4342u), sn.z, sn.w);
    }
#endif
  };
  const auto mma_h = [&](const FragH &f, const uint8_t *, int) {
#ifdef NVFP4A_TC80
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j) mma_bf16(acc[i][j], f.a[i][0], f.a[i][1], f.a[i][2], f.a[i][3], f.b[j][0], f.b[j][1]);
#endif
  };
  // the integer paths: a k-tile in one straight block (ktile_i), per block its mma operands as loaded -- A's words, B's
  // words (INT8) or nibble halves (NVFP4) -- decoded (dec4) next to their use, the scales (A rows' sa, B columns' sb)
  // from the slot.  (Blocks loaded one ahead across k-tiles, as W4A16's, spilled the accumulators over the slot
  // handover: the FFMA chains leave no room for a second block of operands and scales.)  INT8 B [block][slice
  // quad][lane][slice 4] words, sb [block][octet][t][slice 8][c 2]; NVFP4 B [pair][lane][slice 2][word 2] (word w =
  // blocks 2 w, 2 w + 1), sb [block][quad][t][slice 4][c 2]
  struct FragI { uint32_t a[MT], b[NJ]; };
  const auto load_i = [&](FragI &f, const uint8_t *p, int kb) {
#pragma unroll
    for (int i = 0; i < MT; ++i) f.a[i] = reinterpret_cast<const uint32_t *>(p)[(32 * (MT * wm + i) + lane) * 4 + kb];
    if (Q8) {
#pragma unroll
      for (int jq = 0; jq < NJ / 4; ++jq) {  // 4 n8 slices' B words in one LDS.128
        const uint4 bw = reinterpret_cast<const uint4 *>(p + L::B)[(kb * 4 + NJ / 4 * wn + jq) * 32 + lane];
        f.b[4 * jq] = bw.x; f.b[4 * jq + 1] = bw.y; f.b[4 * jq + 2] = bw.z; f.b[4 * jq + 3] = bw.w;
      }
    } else {
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const uint32_t w = reinterpret_cast<const uint32_t *>(p + L::B)[(32 * (NJ / 2 * wn + j / 2) + lane) * 4 + 2 * (j & 1) + (kb >> 1)];
        f.b[j] = kb & 1 ? w >> 16 : w;
      }
    }
  };
  // one output block (i, j): the exact block sum (1.5 * 2^23 + p), p sa exactly, times sb into the accumulators
  const auto blk = [&](int i, int j, uint32_t a0, uint32_t a1, uint32_t b, float2 sa, float n0, float n1, float sb0, float sb1) {
    uint32_t d[4];
    imma_block(d, a0, a1, b, magic);
    float *o = acc[i][j];
    o[0] = fma_rn(fma_rn(__uint_as_float(d[0]), sa.x, n0), sb0, o[0]);
    o[1] = fma_rn(fma_rn(__uint_as_float(d[1]), sa.x, n0), sb1, o[1]);
    o[2] = fma_rn(fma_rn(__uint_as_float(d[2]), sa.y, n1), sb0, o[2]);
    o[3] = fma_rn(fma_rn(__uint_as_float(d[3]), sa.y, n1), sb1, o[3]);
  };
  // the block's mma and FFMA in the order that keeps the fewest scales live: INT8 B (MT 2, NJ 8) per n8 slice, its sb
  // two at a time; NVFP4 B (MT 4, NJ 4) per m16 slice, its sa one at a time
  const auto mma_i = [&](const FragI &f, const uint8_t *p, int kb) {
    if (Q8) {
      uint32_t a0[MT], a1[MT];
      float2 sa[MT];
      float n0[MT], n1[MT];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        // (few decodes here: their tables rematerialized beat two more live registers)
        a0[i] = dec4(f.a[i]); a1[i] = dec4(f.a[i] >> 16);
        sa[i] = reinterpret_cast<const float2 *>(p + L::CA)[((MT * wm + i) * 4 + kb) * 8 + g];  // rows g, g + 8
        n0[i] = -kMagicF * sa[i].x; n1[i] = -kMagicF * sa[i].y;  // (exact)
      }
#pragma unroll
      for (int jq = 0; jq < NJ / 4; ++jq) {  // 4 slices' sb in two LDS.128
        const float4 s01 = reinterpret_cast<const float4 *>(p + L::BS)[((kb * 2 + wn) * 4 + t4) * 4 + 2 * jq];
        const float4 s23 = reinterpret_cast<const float4 *>(p + L::BS)[((kb * 2 + wn) * 4 + t4) * 4 + 2 * jq + 1];
        const float sb[4][2] = {{s01.x, s01.y}, {s01.z, s01.w}, {s23.x, s23.y}, {s23.z, s23.w}};
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int i = 0; i < MT; ++i) blk(i, 4 * jq + jj, a0[i], a1[i], f.b[4 * jq + jj], sa[i], n0[i], n1[i], sb[jj][0], sb[jj][1]);
      }
    } else {
      uint32_t b[NJ];
      float sb[NJ][2];
#pragma unroll
      for (int j = 0; j < NJ; ++j) b[j] = dec4r(f.b[j], tp0, tn0);
#pragma unroll
      for (int x = 0; x < NJ / 2; ++x) {
        const float4 v = reinterpret_cast<const float4 *>(p + L::CB)[((kb * 4 + wn) * 4 + t4) * 2 + x];
        sb[2 * x][0] = v.x; sb[2 * x][1] = v.y; sb[2 * x + 1][0] = v.z; sb[2 * x + 1][1] = v.w;
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const float2 sa = reinterpret_cast<const float2 *>(p + L::CA)[((MT * wm + i) * 4 + kb) * 8 + g];  // rows g, g + 8
        const uint32_t a0 = dec4r(f.a[i], tp0, tn0), a1 = dec4r(f.a[i] >> 16, tp0, tn0);
#pragma unroll
        for (int j = 0; j < NJ; ++j) blk(i, j, a0, a1, b[j], sa, -kMagicF * sa.x, -kMagicF * sa.y, sb[j][0], sb[j][1]);
      }
    }
  };
  const auto ktile_i = [&](const uint8_t *p) {
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      FragI f;
      load_i(f, p, kb);
      mma_i(f, p, kb);
    }
  };

  // ---- epilogue of the tile at (m0, n0) (the operations of the DP4A kernel's): v = acc * kOut * s, + bias, beta * C + v
  // in one FMA, amax of the stored values.  Staged through slot p, the slot of the tile's last k-tile (its EMPTY
  // arrival waits for the epilogue; the stream fills the other slots with the next tile meanwhile), warp-local after
  // one barrier (every compute warp is done reading the slot): a warp stages its outputs in its own 8 SLD floats, one
  // n8 slice x two m16 slices (8 columns x 32 rows) a pass, then lane l writes rows 8 (l % 4) .. + 7 of column l / 4
  // (one 16-byte store for BF16 / FP16 output).  Cold code: lane and warp re-read (tid_x), nothing it derives is hoisted
  // into the hot loop; C (beta != 0) element by element.
  constexpr int EPI = 15, SLD = 36;  // EPI: the compute warps' named barrier; SLD: staged column, 32 rows + 4 (banks)
  static_assert(NC / 32 * 8 * SLD * 4 <= SLOT && MT % 2 == 0, "the warps' staging fits in a slot");
  const auto epilogue = [&](int m0, int n0, uint8_t *p, auto dc) {
    constexpr int DT = decltype(dc)::value;
    const int tx = (int)tid_x(), ln = tx & 31, wp = tx >> 5, gg = ln >> 2, tt = ln & 3;
    const int rb = m0 + wp / WN * TM, cb = n0 + wp % WN * TN;  // the warp's rows and columns
    float *stage = reinterpret_cast<float *>(p) + wp * 8 * SLD;
    const bool use_c = e.beta != 0.f, vd = ((uintptr_t)e.d % (8 * Elem<DT>::bytes)) == 0 && e.ldd % 8 == 0;
    EpiState st = epi_begin(e);
    bar_sync<NC>(EPI);  // the compute warps are done with the slot
#pragma unroll  // (constant indices into acc throughout)
    for (int i0 = 0; i0 < MT; i0 += 2) {
      float rowp[2][2];  // row parts of the output scales (rows past the end: row 0's)
#pragma unroll
      for (int ii = 0; ii < 2; ++ii)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int gi = rb + 16 * (i0 + ii) + gg + 8 * hh;
          rowp[ii][hh] = epi_row_part(e, gi < m ? gi : 0);
        }
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        float colp[2];
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int gj = cb + 8 * j + 2 * tt + x;
          colp[x] = epi_col_part(e, gj < n ? gj : 0);
        }
#pragma unroll
        for (int ii = 0; ii < 2; ++ii)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int x = 0; x < 2; ++x)
              stage[(2 * tt + x) * SLD + 16 * ii + gg + 8 * hh] =
                  __fmul_rn(acc[i0 + ii][j][2 * hh + x] * kOut, epi_scale_ab(e, rowp[ii][hh], colp[x]));
        warp_sync();
        // lane: rows gi .. gi + 7 (cnt of them in range) of column gj; bias and C (rare) applied in the stage
        const int gi = rb + 16 * i0 + 8 * tt, gj = cb + 8 * j + gg, cnt = gj < n ? min(8, m - gi) : 0;
        float *sv = &stage[gg * SLD + 8 * tt];
        if (e.bias || use_c) {
#pragma unroll 1
          for (int u = 0; u < cnt; ++u) {
            float x = sv[u];
            if (e.bias) x = __fadd_rn(x, load_any(e.bias_dtype, e.bias, gi + u));
            if (use_c) x = __fmaf_rn(e.beta, load_any(e.c_dtype, e.c, (size_t)gj * e.ldc + gi + u), x);
            sv[u] = x;
          }
        }
        float v[8];
        *reinterpret_cast<float4 *>(v) = *reinterpret_cast<const float4 *>(sv);
        *reinterpret_cast<float4 *>(v + 4) = *reinterpret_cast<const float4 *>(sv + 4);
#pragma unroll
        for (int u = 0; u < 8; ++u) st.amax = fmaxf(st.amax, u < cnt ? fabsf(v[u]) : 0.f);
        if (cnt == 8 && vd) {
          store4_any<DT>(e.d, (size_t)gj * e.ldd + gi, true, v);
          store4_any<DT>(e.d, (size_t)gj * e.ldd + gi + 4, true, v + 4);
        } else {
#pragma unroll 1
          for (int u = 0; u < cnt; ++u) store_any(DT, e.d, (size_t)gj * e.ldd + gi + u, sv[u]);
        }
        warp_sync();  // the lanes are done with the stage
      }
    }
    epi_end(e, st);
  };
  const auto finish = [&](int m0, int n0, uint8_t *p) {  // the epilogue instance of the output dtype
    if (e.d_dtype == NVFP4A_BF16) epilogue(m0, n0, p, std::integral_constant<int, NVFP4A_BF16>());
    else if (e.d_dtype == NVFP4A_FP16) epilogue(m0, n0, p, std::integral_constant<int, NVFP4A_FP16>());
    else epilogue(m0, n0, p, std::integral_constant<int, NVFP4A_FP32>());
  };

  // ---- the stream ---------------------------------------------------------------------------------------------
  if (warp >= NC / 32) {
    const Split s = split();  // (the producers hold theirs)
#ifdef NVFP4A_TC80
    // the copier and the publisher walk the units: k stepped, the tile located anew at its end (their divisions)
    int lin, kk, mb, nb;
    const uint8_t *b;
    const auto at = [&](int u) {
      locate_in(s, u, lin, kk);
      tile_of(lin, mb, nb);
      b = pb + ((size_t)nb * kt + kk) * RB;
    };
    const auto next = [&](int u) {
      if (kk == kt - 1 || u + 1 == s.W * kt) at(u + 1);
      else { ++kk; b += RB; }
    };
    at(0);
    if (warp == NC / 32) {  // the copier
#pragma unroll 1
      for (int u = 0; u < U; ++u) {
        if (u >= NS) bar_sync<EC>(EMPTY + u % NS);
        copy(u, a_tile(mb, kk) + 16 * lane, kk, b);
        next(u);
      }
    } else if (warp == NC / 32 + 1) {  // the publisher
#pragma unroll 1
      for (int u = 0; u < U; ++u) {
        publish(u, a_tile(mb, kk), kk);
        next(u);
      }
    }
#else
#pragma unroll 1
    for (int u = 0; u < U; ++u) {
      if (u >= NS) bar_sync<EC>(EMPTY + u % NS);
      produce(s, u);
    }
#endif
    return;
  }
  // a segment = the CTA's consecutive units of one tile, k-tiles k0 .. k1 - 1, the accumulators its sums; staged in
  // slot p.  Partials: [q = i NJ + j][compute thread] float4 at ws + c NC 64 (coalesced)
  const auto zero = [&] {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int x = 0; x < 4; ++x) acc[i][j][x] = 0.f;
  };
  const auto segment = [&](int lin, int k0, int k1, uint8_t *p) {
    const int tx = (int)tid_x(), cx = (int)ctaid_x();  // (re-read: the addresses below stay in the cold code)
    if (k0 > 0) {  // a later part of a tile (the CTA's first segment): its partial, then the flag
      float4 *w = reinterpret_cast<float4 *>(ws + (size_t)cx * NC * 64) + tx;
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) __stcg(w + (i * NJ + j) * NC, make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]));
      bar_sync<NC>(EPI);
      if (tx == 0) { __threadfence(); st_release(flags + cx, 1u); }
      return;
    }
    if (k1 < kt) {  // the head (the CTA's last segment, stream-K): the partials of CTAs c + 1, .. up to the tile's end
      const Split s = split();
      const long long qe = (long long)(lin - s.W * s.G + 1) * kt;  // the tile's end among the stream-K units
      for (int o = cx + 1, done = 0; !done; ++o) {
        const long long ob = s.usk * o / s.G, oe = s.usk * (o + 1) / s.G;  // CTA o's units (none: no partial, no flag)
        done = oe >= qe;
        if (oe == ob) continue;
        if (tx == 0) {
          while (ld_acquire(flags + o) != 1u) {}
          flags[o] = 0u;  // (read by the next launch only)
        }
        bar_sync<NC>(EPI);
        const float4 *w = reinterpret_cast<const float4 *>(ws + (size_t)o * NC * 64) + tx;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int j = 0; j < NJ; ++j) {
            const float4 v = __ldcg(w + (i * NJ + j) * NC);
            acc[i][j][0] += v.x; acc[i][j][1] += v.y; acc[i][j][2] += v.z; acc[i][j][3] += v.w;
          }
      }
    }
    int mb, nb;
    tile_of(lin, mb, nb);
    finish(mb * BM, nb * BN, p);
  };

  if (kt == 0) {  // no k: every tile its epilogue alone (no units)
    for (int lin = blockIdx.x; lin < ((n + BN - 1) / BN) * ((m + BM - 1) / BM); lin += gridDim.x) {
      int mb, nb;
      tile_of(lin, mb, nb);
      finish(mb * BM, nb * BN, smem);
    }
    return;
  }
  // the CTA's units: tile lin, k-tile kk of unit u; stepped within a tile, located anew after its segment
  int lin, kk;
  locate(0, lin, kk);
  const auto step = [&](int u, bool end) { if (end) locate(u + 1, lin, kk); else ++kk; };
  // the compute warps' stream: per k-tile blocks 1 .. 3 of its slot, then (the next slot full) the next k-tile's block
  // 0, then block 3's mma -- the fragments one block ahead throughout, the mma pipe busy through the slot handover
  uint32_t phase = 0;
  const auto wait_full = [&](int u, int slot) {
#ifdef NVFP4A_TC80
    mbar_wait(full + 8 * slot, phase);
#else
    seq_wait(full + 4 * slot, (uint32_t)u + 1);
#endif
  };
  const auto stream = [&](auto &f0, auto &f1, auto load, auto mma) {
    if (U > 0) { wait_full(0, 0); load(f0, smem, 0); }
#pragma unroll 1
    for (int u = 0, slot = 0, k0 = -1; u < U; ++u) {
      if (k0 < 0) k0 = kk;
      const uint8_t *p = smem + slot * SLOT;
      const int last = slot;
      const bool end = kk == kt - 1 || u == U - 1;
      if (++slot == NS) { slot = 0; phase ^= 1; }
      load(f1, p, 1); mma(f0, p, 0);
      load(f0, p, 2); mma(f1, p, 1);
      load(f1, p, 3); mma(f0, p, 2);
      if (u + 1 < U) { wait_full(u + 1, slot); load(f0, smem + slot * SLOT, 0); }  // the next k-tile's block 0
      mma(f1, p, 3);
      if (!end && u + NS < U) bar_arrive<EC>(EMPTY + last);  // (a segment's last slot: after the segment)
      if (end) {
        segment(lin, k0, kk + 1, smem + last * SLOT);
        if (u + NS < U) bar_arrive<EC>(EMPTY + last);
        zero();
        k0 = -1;
      }
      step(u, end);
    }
  };
  if (H) {
    FragH f0, f1;
    stream(f0, f1, load_h, mma_h);
  } else {  // the integer paths: a k-tile a straight block after its slot's wait
#pragma unroll 1
    for (int u = 0, slot = 0, k0 = -1; u < U; ++u) {
      if (k0 < 0) k0 = kk;
      wait_full(u, slot);
      ktile_i(smem + slot * SLOT);
      const int last = slot;
      const bool end = kk == kt - 1 || u == U - 1;
      if (!end && u + NS < U) bar_arrive<EC>(EMPTY + slot);  // (a segment's last slot: after the segment)
      if (++slot == NS) { slot = 0; phase ^= 1; }
      if (end) {
        segment(lin, k0, kk + 1, smem + last * SLOT);
        if (u + NS < U) bar_arrive<EC>(EMPTY + last);
        zero();
        k0 = -1;
      }
      step(u, end);
    }
  }
#endif
}

// Entry points (src/tensors/image.cu, gemm_tc_inst.cu): the images of A (h: A16, else A8) and B (image.cuh), and the
// GEMM over them per B dtype TB (NVFP4, INT8, BF16) and scale formats F (kF_AUE / kF_BUE)
cudaError_t tc_image_a(bool h, const uint8_t *A, int lda, const uint8_t *SA, int ldsa, int m, int k, uint8_t *img, cudaStream_t s);
cudaError_t tc_image_b(int tb, const uint8_t *B, int ldb, const uint8_t *SB, int ldsb, int n, int k, uint8_t *img, cudaStream_t s);
template <int TB, int F>
cudaError_t tc_entry(const uint8_t *ia, const uint8_t *ib, int m, int n, int k, const Epilogue &e, uint8_t *ws, cudaStream_t s);
// the workspace of a launch on `sms` SMs: FP32 partials (one per CTA), then the CTAs' flags
inline size_t tc_ws_bytes(int sms) { return (size_t)sms * 256 * 64 * 4 + (size_t)sms * 4; }
// bytes of A's image: m16 slices of whole 128-row blocks x groups of 4 k-tiles x GV
inline size_t tc_a_image_bytes(int rows, int k) { return (size_t)((rows + 127) / 128) * 8 * ((k / 64 + 3) / 4) * tc::GV; }
// bytes of B's image: 128-column blocks x k-tiles x record bytes (tc::RB4, tc::RB8, tc::RBH)
inline size_t tc_image_bytes(int rows, int k, int record) { return (size_t)((rows + 127) / 128) * (k / 64) * record; }

}  // namespace kern
}  // namespace nvfp4a
