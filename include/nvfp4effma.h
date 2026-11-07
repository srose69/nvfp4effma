// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* NVFP4EffMA: NVFP4 GEMM / GEMV / quantization for NVIDIA GPUs without native FP4 (sm_50 ... sm_90).
 * NVFP4: E2M1 values (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}), one 8-bit block scale per 16 consecutive values
 * along a row, a global FP32 factor: value = e2m1 * block_scale * global.  Format details beside nvfp4a_tensor. */
#ifndef NVFP4EFFMA_H
#define NVFP4EFFMA_H

#define NVFP4EFFMA_VERSION "0.7.0"

#ifdef __CUDACC_RTC__  /* NVRTC (the work items' device entries): the types only */
typedef struct CUstream_st *cudaStream_t;
#else
#include <stddef.h>
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NVFP4A_NVFP4 = 0, NVFP4A_BF16 = 1, NVFP4A_FP16 = 2, NVFP4A_FP32 = 3, NVFP4A_INT8 = 4 } nvfp4a_dtype;
typedef enum { NVFP4A_OK = 0, NVFP4A_INVALID_VALUE = 1, NVFP4A_NOT_SUPPORTED = 2, NVFP4A_CUDA_ERROR = 3 } nvfp4a_status;

/* NVFP4 data: [rows][cols] row-major, two values per byte (column 2c in the low nibble of byte c, 2c + 1 in the
 * high one), ld (in values) apart.  Block scales: E4M3 (default) or UE5M3; the scale of value (r, c) is block
 * (r, c / 16).  Linear layout: byte block_scale[r * ld_scale + c / 16]. */

/* Swizzled scales (NVFP4A_SCALES_SWIZZLED, the GEMM layout of cuBLASLt and Transformer Engine): the scale matrix, padded to
 * 128 rows and 4 columns (sc = roundup(cols / 16, 4)), is cut into 128 x 4 tiles of 512 bytes stored one after the other along
 * a row of tiles; scale (r, c) is byte ((r / 128) * (sc / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + c % 4. */

/* nvfp4a_tensor.flags (NVFP4 tensors) */
#define NVFP4A_SCALES_SWIZZLED 0x1u /* block scales in the cuBLASLt / Transformer Engine GEMM layout */
#define NVFP4A_SCALES_UE5M3    0x2u /* block scales are UE5M3 (unsigned, 5 exponent bits, bias 15) */
#define NVFP4A_GLOBAL_AMAX     0x4u /* global_scale holds the amax: g = amax / (6 * scale_max) */
#define NVFP4A_GLOBAL_PER_ROW  0x8u /* one global value per row (row-scaled NVFP4) */

/* global_scale: one float (one per row with NVFP4A_GLOBAL_PER_ROW, Transformer Engine's row-scaled NVFP4), NULL = 1.
 * Default: the decode factor g itself; NVFP4A_GLOBAL_AMAX: the amax of the tensor (row), g = amax / (6 * scale_max),
 * scale_max = 448 for E4M3 scales (256 with 4over6 headroom), 114688 for UE5M3. */

/* INT8 (activations for W4A8): int8 values, one FP32 scale per 16 values along a row:
 * value = q * block_scale (* global_scale); block_scale is a float array, ld_scale counts floats. */

typedef struct {
  void *data;                /* device pointer */
  nvfp4a_dtype dtype;
  void *block_scale;         /* NVFP4: E4M3 / UE5M3 block scales, INT8: FP32 ones (device), one per 16 values */
  int ld_scale;              /* linear scales: scales between consecutive rows; 0 = cols / 16 */
  const float *global_scale; /* NVFP4 / INT8: global decode factor or amax (device), see flags; NULL = 1 */
  float *amax;               /* optional output statistic: max |v| (device, 1 float, max-reduced into the
                                existing value); NULL = off */
  unsigned flags;            /* NVFP4A_SCALES_* / NVFP4A_GLOBAL_*; 0 = linear E4M3 scales, global = g */
  int scale_max;             /* bound of the global encoding: 0 = the scale type's maximum (448 / 114688);
                                256 = the 4over6 headroom of Transformer Engine */
} nvfp4a_tensor;

/* A tensor with its rowwise and columnwise NVFP4 representations (Transformer Engine's kNVTERowwise* /
 * kNVTEColumnwise*): row is the [rows][cols] matrix, col the quantization of its transpose [cols][rows].
 * The 16 x 16 block scaling of Transformer Engine weights is the same format (one scale per tile). */
typedef struct {
  nvfp4a_tensor row, col;
  int ld_row, ld_col;
} nvfp4a_dual_tensor;

/* GEMM, column-major D (a linear layer y = x W^T: A = W, B = x, D = y):
 *     D[i + j*ldd] = alpha * sum_k A(i, k) * B(j, k) + bias[i] + beta * C[i + j*ldc] */

/* A: NVFP4, m rows of k values (lda).  B: NVFP4, INT8, BF16, FP16 or FP32, n rows of k values (ldb), K-contiguous
 * ("TN", as cuBLASLt's FP4 GEMM).  C, D, bias: BF16, FP16 or FP32.  C may alias D; C == NULL with beta != 0 means
 * C = D, so beta = 1 adds the product into D in place. */

/* Requirements: k, lda, ldb multiples of 64; A and B 16-byte aligned; ld_scale a multiple of 4 (0 for swizzled
 * scales, whose row length is roundup(k / 16, 4)); NVFP4 block scales 4-byte, INT8 ones 16-byte aligned; m, n, ldc,
 * ldd unrestricted (ldc, ldd >= m).  Block scales must be finite (no E4M3 NaN 0x7F / 0xFF, no UE5M3 0xFF; unchecked). */

/* Arithmetic (FP32 throughout, one rounding per step, bit-reproducible per kernel path).  NVFP4 x NVFP4 / INT8:
 * per output and 16-block the sum p = sum (2a) b' (b' = 2b for NVFP4, the int8 value for INT8) is an exact integer;
 * acc = fma(p * sa, sb, acc) (p * sa exact); finally acc *= 1/4 (NVFP4) or 1/2 (INT8).  Tiled kernels: blocks in ascending k. */

/* NVFP4 x BF16 / FP16 / FP32: acc = fma(a * sa, b, acc) per MAC (a * sa exact, so is the product with a BF16 or FP16 b);
 * tiled kernels: k ascending.  One FP32 FFMA per MAC: beyond a few columns, quantizing B to INT8 (nvfp4a_quantize)
 * and running W4A8 is faster. */

/* GEMV kernel (small n, every B type; columns in groups of 8, each output computed alone): lane l takes the blocks 32c + l,
 * k = 512c + 16l + j in (c, j) order, then the 32 lanes combine by an xor butterfly (16, 8, 4, 2, 1).  Runs for n <= 8 and
 * up to the n where it stays faster than the tiled kernel (kGemvMaxN in src/nvfp4effma.cuh). */

/* Epilogue: v = acc * s, s = ((alpha * vA) * vB) * f, vX = the global value of NVFP4 / INT8 operand X for output row i (A) /
 * column j (B), else 1; f = 1 / (6 * scale_max) per operand whose global is an amax, multiplied in FP32 (1 / (36 * mA * mB)
 * for both), 1 for none; v += bias[i]; v = beta * C + v (one FMA); amax = max(amax, |v|); D = RNE(v). */
nvfp4a_status nvfp4a_gemm(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda,
                          const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                          const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, cudaStream_t stream);

/* nvfp4a_gemm on dual tensors: D[i + j*ldd] = alpha * sum_k a(i, k) * b(j, k) + ..., a the [m][k] matrix of A (trans_a = 0: A.row;
 * 1: A is [k][m], A.col), b likewise the [n][k] matrix of B (k contiguous in both).  Y[t][o] = X[t][i] W[o][i]^T: forward A = W,
 * B = X (trans 0, 0); dgrad dX = dY W: A = W, trans_a = 1, B = dY; wgrad dW = dY^T X: A = X, B = dY, both trans 1. */
nvfp4a_status nvfp4a_gemm_dual(int m, int n, int k, float alpha, const nvfp4a_dual_tensor *A, int trans_a,
                               const nvfp4a_dual_tensor *B, int trans_b, float beta, const nvfp4a_tensor *C,
                               int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                               cudaStream_t stream);

/* Grouped GEMM (mixture of experts): group g computes D_g = alpha * A_g B_g^T + bias + beta * C_g of nvfp4a_gemm, A_g = rows [g m, (g + 1) m)
 * of the stacked weight A ([groups * m][k], NVFP4), B_g = rows [seg[g], seg[g + 1]) of B ([rows][k]), D_g (C_g) likewise the rows [seg[g], seg[g + 1])
 * of D (C) (D[i + j * ldd], i in [0, m) the output of the group, j the row of B: row-major [rows][ldd] for ldd >= m).  Every operand of nvfp4a_gemm
 * is taken: A's scales E4M3 / UE5M3, linear or swizzled (over all groups * m rows), the globals of every kind (per-row ones stacked: groups * m
 * for A, rows for B), B NVFP4 / INT8 / BF16 / FP16 / FP32, bias [groups * m], C, the amax of D; it runs on the kernels nvfp4a_gemm picks for
 * the B dtype (DP4A from sm_61, FFMA below and for dense B).  seg is a device array of groups + 1 non-decreasing int32, seg[0] = 0,
 * seg[groups] <= rows: the row counts stay on the device (no host synchronization, graph capturable); `rows` bounds seg[groups] and sizes the launch.
 * Every output is what nvfp4a_gemm computes for that group with its tiled kernel (never the GEMV kernel: the same bytes as nvfp4a_gemm for a group
 * with more rows than the GEMV takes), whatever the other groups hold; rows past seg[groups] are left alone. */
nvfp4a_status nvfp4a_gemm_grouped(int groups, int m, int k, int rows, float alpha, const nvfp4a_tensor *A, int lda,
                                  const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                                  const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, const int *seg, cudaStream_t stream);

/* Quantization (nvfp4a_quantize, nvfp4a_quantize_ex), the arithmetic of Transformer Engine, per block (16 values of a row,
 * or a 16 x 16 tile), M = scale_max.  Global amax: S_enc = min(M * 6 / amax, FLT_MAX) (1 for amax = 0), S_dec = 1 / S_enc;
 * given decode factor g: S_enc = min(1 / g, FLT_MAX), S_dec = g. */

/* sb = scale_RNE_satfinite(block_amax * (S_enc * (1/6)));  q = E2M1_RNE_satfinite(x * min(1 / (sb * S_dec), FLT_MAX)) */

/* BF16 / FP16 / FP32 src [rows][cols] (ld lds) -> NVFP4 dst (ld ldd, 1 x 16 blocks, the destination's global) or INT8 dst (ld ldd,
 * FP32 block scales: sb = amax_block / 127, q = RNE(x / sb), 0 where sb = 0; global_scale ignored); dst->amax gets max |x| over src.
 * cols, lds, ldd multiples of 16; src 16-byte, dst data 8 (NVFP4) / 16 (INT8), block scales 1 / 4 byte aligned; src finite. */
nvfp4a_status nvfp4a_quantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                              int cols, cudaStream_t stream);

/* 4over6 (Transformer Engine's NVFP4 4over6, 1 x 16 or 16 x 16 blocks): two candidates, the scale of base = (block_amax / 6) * S_enc
 * (map-to-6) and of 1.5 * base (map-to-4), each quantized as above; the smaller error sum (|d| or d^2 over the block, d = ((e2m1 * sb) *
 * amax) / (6 * M) - x, rows of a 16 x 16 tile summed as a 16-way tree) wins, ties go to map-to-6. */

/* Quantization recipe of nvfp4a_quantize_ex. */
typedef enum { NVFP4A_4OVER6_OFF = 0, NVFP4A_4OVER6_MAE = 1, NVFP4A_4OVER6_MSE = 2 } nvfp4a_4over6;
typedef struct {
  int tile_rows;             /* 1: 1 x 16 blocks (activations, gradients); 16: 16 x 16 tiles (TE weights) */
  nvfp4a_4over6 four_over_six;
  int compute_amax;          /* 1: first write the global amax (max |x| of the tensor, or of each row with
                                NVFP4A_GLOBAL_PER_ROW) into global_scale of the NVFP4A_GLOBAL_AMAX
                                destinations; 0: use the global value they hold */
} nvfp4a_quant_config;

/* src [rows][cols] -> rowwise NVFP4 [rows][cols] (ld ld_row) and / or columnwise NVFP4 [cols][rows] (the quantization of src^T,
 * ld ld_col); either may be NULL.  Each destination's flags select its scale layout, type and global (an amax for 4over6 and
 * per-row globals; per-row only for the rowwise one).  cfg NULL = 1 x 16, no 4over6, compute_amax = 0. */

/* cols a multiple of 16; with a columnwise destination or 16 x 16 tiles the last block may be partial (zero-filled;
 * ld_col >= roundup(rows, 16)).  Scale buffers hold the padded matrix (swizzled: roundup(rows, 128) x roundup(cols / 16, 4));
 * padding is not written. */
nvfp4a_status nvfp4a_quantize_ex(const nvfp4a_tensor *src, int lds, int rows, int cols, const nvfp4a_tensor *rowwise,
                                 int ld_row, const nvfp4a_tensor *columnwise, int ld_col,
                                 const nvfp4a_quant_config *cfg, cudaStream_t stream);

/* NVFP4 src -> BF16 / FP16 / FP32 dst [rows][cols]: v = (e2m1 * sb) * g, rounded once (RNE); with an amax
 * global (Transformer Engine) v = e2m1 * ((sb * amax) * (1 / (6 * scale_max))).  dst->amax gets max |v|.
 * Same alignment rules as nvfp4a_quantize. */
nvfp4a_status nvfp4a_dequantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                                int cols, cudaStream_t stream);

/* Converts the block scales of a [rows][cols] NVFP4 matrix between the linear layout (ld_linear bytes per
 * row) and the swizzled GEMM layout (to_swizzled = 1: linear -> swizzled, 0: back).  The swizzled buffer
 * holds roundup(rows, 128) x roundup(cols / 16, 4) bytes; its padding is written as 0. */
nvfp4a_status nvfp4a_swizzle_scales(const void *src, void *dst, int ld_linear, int rows, int cols, int to_swizzled,
                                    cudaStream_t stream);

/* Transpose of an NVFP4 matrix quantized with 16 x 16 tiles: src [rows][cols] -> dst [cols][rows], data and scales
 * (dst->global_scale may point at the same value).  Exact only for 16 x 16 tiles; other matrices must be quantized
 * columnwise from the source instead.  rows and cols multiples of 16. */
nvfp4a_status nvfp4a_transpose(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                               int cols, cudaStream_t stream);

/* ---- Tensor-core GEMM (sm_75 .. sm_89; BF16 B from sm_80) --------------------------------------------------------
 * nvfp4a_gemm_tc computes D = alpha * A * B^T + bias + beta * C of nvfp4a_gemm on tensor cores, from images of A and B: the
 * operand's bytes and block scales in streaming order (rows padded to 128); built once per weight, per call for activations. */

/* B is NVFP4 (W4A4), INT8 with FP32 block scales (W4A8) or BF16 (W4A16).  Sums are FP32 (block sums exact integers, BF16
 * products a * scale exact) in the tensor cores' order, work split along k between CTAs: nvfp4a_gemm's values within the
 * rounding of an FP32 accumulation, not bit for bit. */
/* Image kinds.  A (the weight, NVFP4): one image serves nvfp4a_gemm_tc and nvfp4a_gemv_tc -- A16 for BF16 B (W4A16,
 * sm_80+), A8 for INT8 and NVFP4 B (W4A8, W4A4).  B (the activations of nvfp4a_gemm_tc): NVFP4, INT8, BF16. */
#define NVFP4A_TC_A16 0
#define NVFP4A_TC_B 1
#define NVFP4A_TC_A8 2

/* Bytes of the image of kind `operand` (NVFP4A_TC_A16, NVFP4A_TC_A8: NVFP4; NVFP4A_TC_B: NVFP4, INT8, BF16) of `rows`
 * rows of k values, k a multiple of 64. */
size_t nvfp4a_tc_image_bytes(int operand, nvfp4a_dtype dtype, int rows, int k);

/* src [rows][k] (data and block scales as for nvfp4a_gemm, ld values between rows, a multiple of 64) -> image
 * (16-byte aligned, nvfp4a_tc_image_bytes).  A16 / A8 images are built on any supported device (e.g. to be stored and
 * loaded elsewhere); B images only where nvfp4a_gemm_tc runs. */
nvfp4a_status nvfp4a_tc_image(int operand, const nvfp4a_tensor *src, int ld, int rows, int k, void *image,
                              cudaStream_t stream);

/* A weight image (NVFP4A_TC_A16 / NVFP4A_TC_A8, from) of [rows][k] -> the image of the other kind (to), or back to the NVFP4
 * tensor: dst data [rows][k] (ld values between rows, a multiple of 64; 16-byte aligned), linear block scales (dst->ld_scale,
 * 0 = k / 16, a multiple of 4; 4-byte aligned); dst's flags and global are untouched (the image keeps the scale bytes). Any device. */
nvfp4a_status nvfp4a_tc_image_convert(int from, int to, const void *image, int rows, int k, void *dst, cudaStream_t stream);
nvfp4a_status nvfp4a_tc_image_to_raw(int from, const void *image, int rows, int k, const nvfp4a_tensor *dst, int ld, cudaStream_t stream);

/* Bytes of nvfp4a_gemm_tc's workspace on the current device.  The caller zeroes it once (cudaMemset) after
 * allocating it; every call leaves it zeroed.  A workspace serves one call at a time. */
size_t nvfp4a_tc_workspace_bytes(void);

/* A, B: their images (nvfp4a_tc_image; A's of the kind for B's dtype: A16 for BF16, A8 for INT8 / NVFP4) in data;
 * dtype, flags (scale type, globals), global_scale and scale_max as for nvfp4a_gemm (block_scale and ld_scale unused:
 * the images hold the scales).  C, D, bias as for nvfp4a_gemm. */
nvfp4a_status nvfp4a_gemm_tc(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta,
                             const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                             void *workspace, cudaStream_t stream);

/* GEMV on tensor cores: nvfp4a_gemm_tc for n <= 8 (1 <= n <= 8), A its image (the same as for nvfp4a_gemm_tc), B as
 * for nvfp4a_gemm (NVFP4, INT8 with FP32 block scales, BF16 from sm_80; ldb values between its rows): every byte of
 * A's image read once, no workspace.  Sums as nvfp4a_gemm_tc's. */
nvfp4a_status nvfp4a_gemv_tc(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb, float beta,
                             const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                             cudaStream_t stream);

/* ---- Extended epilogue (the GEMM / GEMV of a transformer layer in one call) ------------------------------------------
 * nvfp4a_gemm_ex, _gemv_tc_ex, _gemm_tc_ex do what nvfp4a_gemm, _gemv_tc, _gemm_tc do, plus these epilogue additions (each off when its
 * pointer is NULL / flag 0); with any: A's scales E4M3 (linear for _gemm_ex), B BF16 or INT8.  Items: include/nvfp4effma_items.h. */
typedef struct {
  /* Column scale of an RMSNorm (deferred scalar: W (x w s) = s (W (x w))): column j of B (a token) is scaled by
   * s_j = RN(1 / sqrt(sum_j * norm_inv_n + norm_eps)), sum_j = the norm_nparts values at norm_parts + j * norm_ld,
   * summed by 32 lanes (lane l: values l, l + 32, ... ascending) then an xor butterfly (16, 8, 4, 2, 1). */
  /* s_j enters the output's scale with B's global: ((alpha * vA(i)) * (vB(j) * s_j)) * f.  The values are read
   * through L2 (they may come from another kernel of the same program). */
  const float *norm_parts;       /* column scale: the parts of each column's sum of squares (device), NULL = none */
  int norm_nparts, norm_ld;
  float norm_inv_n, norm_eps;
  /* SwiGLU: rows of A come in slices of 16, the gate of outputs 8q .. 8q + 7 then their up; D has m / 2 rows,
   * D[8q + r] = silu(g) * u (FP32, silu(g) = g / (1 + exp(-g)), the GPU's fast exp and division), g and u the plain
   * epilogue's values of rows 16q + r and 16q + 8 + r (scale and bias); amax over D.  m a multiple of 16; no C, no d2. */
  int swiglu;                    /* 1: SwiGLU of interleaved gate / up rows */
  /* Second output and the parts of its sum of squares (input of the next RMSNorm): v a value of D as stored (before
   * rounding), d2[i + j * ldd2] = RN(v * w2[i]); per m16 slice s of column j, sq[j * ld_sq + s] = sum of RN(v * v) over the
   * slice's 16 rows ascending (one thread per part: no atomics).  m a multiple of 16; sq = the next call's norm_parts. */
  void *d2;                      /* second output (device), NULL = none */
  nvfp4a_dtype d2_dtype;         /* BF16, FP16 or FP32 */
  int ldd2;
  const void *w2;                /* its row weights (device, m values) */
  nvfp4a_dtype w2_dtype;         /* BF16, FP16 or FP32 */
  float *sq;                     /* the parts of its sum of squares (device, m / 16 per column) */
  int ld_sq;
} nvfp4a_epilogue;

nvfp4a_status nvfp4a_gemm_ex(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B, int ldb,
                             float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                             const nvfp4a_epilogue *ex, cudaStream_t stream);
nvfp4a_status nvfp4a_gemv_tc_ex(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb, float beta,
                                const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                const nvfp4a_epilogue *ex, cudaStream_t stream);
nvfp4a_status nvfp4a_gemm_tc_ex(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta,
                                const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                const nvfp4a_epilogue *ex, void *workspace, cudaStream_t stream);

const char *nvfp4a_status_string(nvfp4a_status status);

/* Description of the most recent failure of an nvfp4a_* call on the calling thread (which argument
 * was wrong, or the CUDA error); "" after a successful call.  Valid until the next nvfp4a_* call. */
const char *nvfp4a_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* NVFP4EFFMA_H */
