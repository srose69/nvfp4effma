// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Tensor-core images of the operands: their bytes (NVFP4 nibbles, INT8, BF16 / FP16) and raw block scale bytes in the
// order the tensor-core kernels consume them, nothing added (rows padded with zeros).  The transcoders read every
// source byte once.
//
// A (the weight): one image serves the GEMM (gemm_tc.cuh) and the GEMV (gemv_tc.cuh) -- A16 for BF16 B (W4A16), A8 for
// INT8 / NVFP4 B (W4A8, W4A4).  [m16 slice][group of 4 k-tiles][GV = 2304 bytes], rows padded to a multiple of 128 (8
// slices): a slice's groups along k one after the other, each its lanes' 16-byte words of the four k-tiles (data [tile
// 4][lane 32], 512 bytes a tile) and its scales (256 bytes).  The GEMV streams a slice's groups (every load a warp's 512
// contiguous bytes, or the scales of 4 k-tiles); the GEMM gathers a k-tile's 8 slices (512 bytes each, and their
// groups' scales) into a slot.  Fragments (lane = 4 g + t; the k order within a k-tile is the images' choice, the same
// for A and B):
//   A16  the BF16 mma's 16 k need not be one NVFP4 block (the scale is in A): lane (g, t) holds block t of rows g and
//        g + 8, word j = (row g bytes 2 j, 2 j + 1) | (row g + 8's) << 16 = k 16 t + 4 j .. + 3, those of the mma j's
//        slots 2 t, 2 t + 1 (A0 / B0) and 2 t + 8, 2 t + 9 (A2 / B1).  Scales [lane 32][tile 4][row g, row g + 8]
//        bytes: the lane's block t
//   A8   the integer mma's 16 k are one block: word j = block j, lane (g, t) its k 4 t .. 4 t + 3 (bytes 2 t, 2 t + 1) of
//        rows g (low half) and g + 8, the mma's k 4 t .. 4 t + 3.  Scales [g 8][tile 4][word row g, word row g + 8]
//        (byte kb = block kb)
// B (the activations, per call): one record per 128 columns and k-tile, the records of a 128-column block consecutive
// along k, in the k order of A's image:
//   NVFP4 (RB4 = 4608): data [0, 4096) [n8 slice pair 8][lane 32][slice 2][word 2]: word w = block 2 w's 16 bits of
//        lane t (bytes 2 t, 2 t + 1: k 4 t .. 4 t + 3; low half), block 2 w + 1's (high); scales [4096, 4608)
//        [pair 8][t 4][slice 2][column 2 t + c 2] words (byte kb = block kb)
//   INT8 (RB8 = 10240): data [0, 8192) [block 4][n8 slice quad 4][lane 32][slice 4] words: column g's k 4 t .. 4 t + 3 of
//        the block; scales [8192, 10240) [block 4][n8 slice octet 2][t 4][slice 8][column 2 t + c 2] FP32
//   BF16 / FP16 (RBH = 16384): [column 128][16-byte chunk 8], chunk c = 2 kb + h of column r at c ^ (r % 8) (ldmatrix
//        reads 8 columns' chunk without bank conflicts): its words (BF16 pairs) 8 t + 2 kb + h of the column's 32, so
//        that ldmatrix gives lane t the k 16 t + 4 kb + 2 h, + 1 of the mma kb, A16's
#pragma once
#include "../nvfp4effma.cuh"
#include "mma.cuh"

namespace nvfp4a {
namespace kern {
namespace tc {

// RA: A's data in a GEMM slot (a k-tile's 8 slices, 512 bytes each; its scales go to the slot converted); B records
constexpr int RA = 4096, RB4 = 4608, RB8 = 10240, RBH = 16384;
template <int TB> __host__ __device__ constexpr int rb_bytes() {
  return TB == NVFP4A_INT8 ? RB8 : TB == NVFP4A_NVFP4 ? RB4 : RBH;
}
constexpr int RD4 = 4096, RD8 = 8192;  // data part of a B record
constexpr int GV = 2304;               // A's record: a group of 4 k-tiles of one m16 slice
constexpr int kImgThreads = 256;       // threads of a B transcoder's CTA

// The transcoders are compiled into one object only (image.cu, NVFP4A_TC_IMAGES).
#ifdef NVFP4A_TC_IMAGES
// 32 bytes of row r of an NVFP4 operand at k-tile kt (zeros past the rows)
__device__ __forceinline__ void row_tile(uint32_t (&w)[8], const uint8_t *p, int r, int rows, int ldb_bytes, int kt) {
  uint4 x = make_uint4(0, 0, 0, 0), y = x;
  if (r < rows) {
    const uint4 *q = reinterpret_cast<const uint4 *>(p + (size_t)r * ldb_bytes + 32 * (size_t)kt);
    x = __ldg(q); y = __ldg(q + 1);
  }
  w[0] = x.x; w[1] = x.y; w[2] = x.z; w[3] = x.w; w[4] = y.x; w[5] = y.y; w[6] = y.z; w[7] = y.w;
}
// the k-tile's scale word (4 blocks) of row r: linear (ld > 0) or swizzled (ld < 0, -ld padded scale columns)
__device__ __forceinline__ uint32_t scale_word(const uint8_t *s, int r, int rows, int ld, int kt) {
  if (r >= rows) return 0u;
  const size_t at = ld > 0 ? (size_t)r * ld + 4 * (size_t)kt : swz_row(r, -ld) + swz_col(4 * kt);
  return __ldg(reinterpret_cast<const uint32_t *>(s + at));
}
// A block's 16 bits of lane t in the integer fragments (bytes 2 t, 2 t + 1 of its 8 in words x, y: k 4 t .. 4 t + 3) in
// the low half
__device__ __forceinline__ uint32_t frag8(uint32_t x, uint32_t y, int t) { return prmt(x, y, (uint32_t)(2 * t | (2 * t + 1) << 4)); }

// A -> image (A16: H, else A8).  Grid (groups, m16 slices padded to a multiple of 8), thread = (tile T of the group,
// lane g, t): its words of the tile, its scale bytes (A16) / the slice's scale words of tile T for row g (A8, lanes
// t = 0).  Rows past m and k-tiles past the end: zeros.
template <bool H>
__global__ void __launch_bounds__(128) image_a_kernel(const uint8_t *__restrict__ A, int lda, const uint8_t *__restrict__ SA, int ldsa,
                                                       int m, int tiles, uint8_t *__restrict__ img) {
  const int tid = threadIdx.x, T = tid >> 5, l = tid & 31, g = l >> 2, t = l & 3;
  const int kt = blockIdx.x * 4 + T, r0 = blockIdx.y * 16 + g, r1 = r0 + 8;
  const bool in = kt < tiles;
  uint32_t w0[8], w1[8];
  row_tile(w0, A, in ? r0 : m, m, lda / 2, kt);  // (a row past m: zeros)
  row_tile(w1, A, in ? r1 : m, m, lda / 2, kt);
  const uint32_t s0 = in ? scale_word(SA, r0, m, ldsa, kt) : 0u, s1 = in ? scale_word(SA, r1, m, ldsa, kt) : 0u;
  uint8_t *rec = img + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * GV;
  uint32_t o[4];
  if (H) {  // block t: words 2 t, 2 t + 1 of the rows (selected, t per lane)
    const uint32_t h0 = t == 0 ? w0[0] : t == 1 ? w0[2] : t == 2 ? w0[4] : w0[6], h1 = t == 0 ? w0[1] : t == 1 ? w0[3] : t == 2 ? w0[5] : w0[7];
    const uint32_t q0 = t == 0 ? w1[0] : t == 1 ? w1[2] : t == 2 ? w1[4] : w1[6], q1 = t == 0 ? w1[1] : t == 1 ? w1[3] : t == 2 ? w1[5] : w1[7];
    o[0] = prmt(h0, q0, 0x5410u); o[1] = prmt(h0, q0, 0x7632u); o[2] = prmt(h1, q1, 0x5410u); o[3] = prmt(h1, q1, 0x7632u);
    reinterpret_cast<uint16_t *>(rec + 2048)[4 * l + T] = (uint16_t)(((s0 >> (8 * t)) & 0xFFu) | ((s1 >> (8 * t)) & 0xFFu) << 8);
  } else {  // block j: bytes 2 t, 2 t + 1 of its 8 (words 2 j, 2 j + 1)
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = prmt(frag8(w0[2 * j], w0[2 * j + 1], t), frag8(w1[2 * j], w1[2 * j + 1], t), 0x5410u);
    if (t == 0) reinterpret_cast<uint2 *>(rec + 2048)[4 * g + T] = make_uint2(s0, s1);
  }
  reinterpret_cast<uint4 *>(rec)[32 * T + l] = make_uint4(o[0], o[1], o[2], o[3]);
}

// B, NVFP4 [n][ldb / 2 bytes] -> image.  Thread = (record, n8 slice j, g): column g of slice j.
__global__ void __launch_bounds__(kImgThreads) image_b4_kernel(const uint8_t *__restrict__ B, int ldb, const uint8_t *__restrict__ SB, int ldsb,
                                                      int n, int tiles, uint8_t *__restrict__ img) {
  const int tid = threadIdx.x, kt = blockIdx.x * 2 + (tid >> 7), j = (tid >> 3) & 15, g = tid & 7;
  if (kt >= tiles) return;
  const int c = blockIdx.y * 128 + 8 * j + g;
  uint32_t w[8];
  row_tile(w, B, c, n, ldb / 2, kt);
  uint8_t *rec = img + ((size_t)blockIdx.y * tiles + kt) * RB4;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    // word 0: blocks 0 (low), 1 (high); word 1: blocks 2, 3
    const uint32_t x0 = prmt(frag8(w[0], w[1], t), frag8(w[2], w[3], t), 0x5410u);
    const uint32_t x1 = prmt(frag8(w[4], w[5], t), frag8(w[6], w[7], t), 0x5410u);
    reinterpret_cast<uint2 *>(rec)[(32 * (j >> 1) + 4 * g + t) * 2 + (j & 1)] = make_uint2(x0, x1);
  }
  reinterpret_cast<uint32_t *>(rec + RD4)[((j >> 1) * 4 + (g >> 1)) * 4 + (j & 1) * 2 + (g & 1)] = scale_word(SB, c, n, ldsb, kt);
}

// B, INT8 [n][ldb bytes] with FP32 scales [n][ldsb] -> image.  Thread = (record, n8 slice j, g): column g of slice j.
__global__ void __launch_bounds__(kImgThreads) image_b8_kernel(const uint8_t *__restrict__ B, int ldb, const float *__restrict__ SB, int ldsb,
                                                      int n, int tiles, uint8_t *__restrict__ img) {
  const int tid = threadIdx.x, kt = blockIdx.x * 2 + (tid >> 7), j = (tid >> 3) & 15, g = tid & 7;
  if (kt >= tiles) return;
  const int c = blockIdx.y * 128 + 8 * j + g;
  uint4 v[4] = {};
  float4 sc = make_float4(0.f, 0.f, 0.f, 0.f);
  if (c < n) {
    const uint4 *q = reinterpret_cast<const uint4 *>(B + (size_t)c * ldb + 64 * (size_t)kt);
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) v[kb] = __ldg(q + kb);
    sc = __ldg(reinterpret_cast<const float4 *>(SB + (size_t)c * ldsb + 4 * (size_t)kt));
  }
  uint8_t *rec = img + ((size_t)blockIdx.y * tiles + kt) * RB8;
  uint32_t *d = reinterpret_cast<uint32_t *>(rec) + (j >> 2) * 128 + 16 * g + (j & 3);  // block 0, lane 4 g
  float *ds = reinterpret_cast<float *>(rec + RD8) + ((j >> 3) * 4 + (g >> 1)) * 16 + (j & 7) * 2 + (g & 1);
  const float sv[4] = {sc.x, sc.y, sc.z, sc.w};
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    const uint32_t x[4] = {v[kb].x, v[kb].y, v[kb].z, v[kb].w};
#pragma unroll
    for (int t = 0; t < 4; ++t) d[512 * kb + 4 * t] = x[t];  // k 4 t .. 4 t + 3
    ds[128 * kb] = sv[kb];
  }
}

// B, BF16 / FP16 [n][ldb] -> image.  Thread = (column r, h): the column's 32 words of the k-tile, its chunks 2 kb + h =
// words 8 t + 2 kb + h, t = 0 .. 3.
__global__ void __launch_bounds__(kImgThreads) image_bh_kernel(const uint16_t *__restrict__ B, int ldb, int n, int tiles, uint8_t *__restrict__ img) {
  const int kt = blockIdx.x, r = threadIdx.x >> 1, h = threadIdx.x & 1, col = blockIdx.y * 128 + r;
  uint4 *rec = reinterpret_cast<uint4 *>(img + ((size_t)blockIdx.y * tiles + kt) * RBH);
  uint32_t wd[32];
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint4 v = col < n ? __ldg(reinterpret_cast<const uint4 *>(B + (size_t)col * ldb + 64 * (size_t)kt) + c) : make_uint4(0, 0, 0, 0);
    wd[4 * c] = v.x; wd[4 * c + 1] = v.y; wd[4 * c + 2] = v.z; wd[4 * c + 3] = v.w;
  }
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    const int c = 2 * kb + h;
    const uint32_t w0 = h ? wd[2 * kb + 1] : wd[2 * kb], w1 = h ? wd[9 + 2 * kb] : wd[8 + 2 * kb];
    const uint32_t w2 = h ? wd[17 + 2 * kb] : wd[16 + 2 * kb], w3 = h ? wd[25 + 2 * kb] : wd[24 + 2 * kb];
    rec[r * 8 + (c ^ (r & 7))] = make_uint4(w0, w1, w2, w3);
  }
}
#endif  // NVFP4A_TC_IMAGES

}  // namespace tc
}  // namespace kern
}  // namespace nvfp4a
