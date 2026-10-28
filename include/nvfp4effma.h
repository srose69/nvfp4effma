// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* NVFP4EffMA: NVFP4 GEMM / GEMV / quantization for NVIDIA GPUs without native FP4 (sm_50 ... sm_90).
 *
 * NVFP4: E2M1 values (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}), one 8-bit block scale per 16 consecutive values
 * along a row, and a global FP32 factor:  value = e2m1 * block_scale * global.
 *   Data: a matrix [rows][cols] is stored row-major, two values per byte (column 2c in the low nibble of
 *   byte c, 2c + 1 in the high one), ld (in values) apart.
 *   Block scales: E4M3 (default) or UE5M3 (NVFP4A_SCALES_UE5M3); the scale of value (r, c) belongs to
 *   block (r, c / 16).  Linear layout: byte block_scale[r * ld_scale + c / 16].  Swizzled layout
 *   (NVFP4A_SCALES_SWIZZLED, the GEMM layout of cuBLASLt and Transformer Engine): the scale matrix,
 *   padded to a multiple of 128 rows and 4 columns (sc = roundup(cols / 16, 4)), is cut into 128 x 4
 *   tiles of 512 bytes stored one after the other along a row of tiles; scale (r, c) is the byte
 *       ((r / 128) * (sc / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + c % 4.
 *   Global: global_scale points at one float (or one per row with NVFP4A_GLOBAL_PER_ROW, the row-scaled
 *   NVFP4 of Transformer Engine).  By default it is the decode factor g itself; with NVFP4A_GLOBAL_AMAX
 *   it is the amax of the tensor (row), as Transformer Engine stores it: g = amax / (6 * scale_max),
 *   scale_max = 448 for E4M3 scales (256 with 4over6 headroom), 114688 for UE5M3.  NULL = 1.
 *   The 16 x 16 block scaling of Transformer Engine weights is the same format (each scale is shared by
 *   the 16 rows of its tile); a tensor and its columnwise version (the quantization of its transpose)
 *   are two NVFP4 matrices.
 * INT8 (activations for W4A8): int8 values, one FP32 scale per block of 16 values along a row:
 * value = q * block_scale (* global_scale); block_scale is a float array, ld_scale counts floats.
 *
 * GEMM (column-major D, the layout of a linear layer y = x W^T with A = W, B = x, D = y):
 *
 *     D[i + j*ldd] = alpha * sum_k A(i, k) * B(j, k) + bias[i] + beta * C[i + j*ldc]
 *
 *   A: NVFP4, m rows of k values (lda).  B: NVFP4, INT8, BF16, FP16 or FP32, n rows of k values (ldb).
 *   C, D, bias: BF16, FP16 or FP32.  C may alias D, and C == NULL with beta != 0 means C = D, so
 *   beta = 1 adds the product into D in place.  Both operands are K-contiguous ("TN", as cuBLASLt's FP4
 *   GEMM); nvfp4a_gemm_dual picks the rowwise or columnwise half of a Transformer Engine style tensor.
 *
 * Arithmetic (one rounding per step, FP32 throughout, bit-reproducible per kernel path):
 *   NVFP4 x NVFP4 / INT8: per output and per block of 16 the block sum p = sum (2a) b' (b' = 2b for
 *     NVFP4, the int8 value for INT8) is an exact integer; acc = fma(p * sa, sb, acc) (p * sa exact);
 *     finally acc *= 1/4 (NVFP4) or 1/2 (INT8).  Tiled kernels: blocks in ascending k.
 *   NVFP4 x BF16 / FP16 / FP32: acc = fma(a * sa, b, acc) per MAC (a * sa exact, so is the product
 *     with a BF16 or FP16 b).  Tiled kernels: k ascending.  One FP32 FFMA per MAC: beyond a few columns,
 *     quantizing B to INT8 (nvfp4a_quantize) and running W4A8 is about twice as fast.
 *   GEMV kernel (small n, every B type; columns in groups of 8, each output computed alone): lane l
 *     takes the blocks 32c + l, k = 512c + 16l + j in (c, j) order, then the 32 lanes are combined by
 *     an xor butterfly (16, 8, 4, 2, 1).  It runs for n <= 8, and up to the n where it stays faster
 *     than the tiled kernel (kGemvMaxN in src/nvfp4effma.cuh).
 *   Epilogue: v = acc * s, s = ((alpha * vA) * vB) * f, vX = the global value of NVFP4 / INT8 operand X
 *     for output row i (A) / column j (B), 1 otherwise; f = 1 / (6 * scale_max) for each operand whose
 *     global is an amax, multiplied together in FP32 (1 / (36 * mA * mB) when both are), 1 when none;
 *     v = v + bias[i]; v = beta * C + v (one FMA); amax = max(amax, |v|); D = RNE(v).
 *
 * Requirements: k, lda, ldb multiples of 64; A and B 16-byte aligned; ld_scale a multiple of 4 (0 for
 * swizzled scales, whose row length is roundup(k / 16, 4)), NVFP4 block scales 4-byte and INT8 ones
 * 16-byte aligned; m, n, ldc, ldd unrestricted (ldc, ldd >= m).  Block scales must be finite (no E4M3
 * NaN code 0x7F / 0xFF, no UE5M3 code 0xFF; not checked).
 *
 * Quantization (nvfp4a_quantize, nvfp4a_quantize_ex), the arithmetic of Transformer Engine, per block
 * (16 values of a row, or a 16 x 16 tile), M = scale_max:
 *     global amax: S_enc = min(M * 6 / amax, FLT_MAX) (1 for amax = 0), S_dec = 1 / S_enc;
 *     given decode factor g: S_enc = min(1 / g, FLT_MAX), S_dec = g
 *     sb = scale_RNE_satfinite(block_amax * (S_enc * (1/6)))
 *     q  = E2M1_RNE_satfinite(x * min(1 / (sb * S_dec), FLT_MAX))
 *   4over6 (Transformer Engine's NVFP4 4over6, 1 x 16 or 16 x 16 blocks): two candidates, the scale of
 *   base = (block_amax / 6) * S_enc (map-to-6) and of 1.5 * base (map-to-4), each quantized as above; the
 *   one with the smaller error sum (|d| or d^2 over the block, d = ((e2m1 * sb) * amax) / (6 * M) - x,
 *   rows of a 16 x 16 tile summed as a 16-way tree) wins, ties go to map-to-6.
 *   To INT8: sb = amax_block / 127, q = RNE(x / sb) (q = 0 where sb = 0); dst->global_scale is ignored.
 */
#ifndef NVFP4EFFMA_H
#define NVFP4EFFMA_H

#define NVFP4EFFMA_VERSION "0.6.0"

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

/* nvfp4a_tensor.flags (NVFP4 tensors) */
#define NVFP4A_SCALES_SWIZZLED 0x1u /* block scales in the cuBLASLt / Transformer Engine GEMM layout */
#define NVFP4A_SCALES_UE5M3    0x2u /* block scales are UE5M3 (unsigned, 5 exponent bits, bias 15) */
#define NVFP4A_GLOBAL_AMAX     0x4u /* global_scale holds the amax: g = amax / (6 * scale_max) */
#define NVFP4A_GLOBAL_PER_ROW  0x8u /* one global value per row (row-scaled NVFP4) */

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
 * kNVTEColumnwise*): row is the [rows][cols] matrix, col the quantization of its transpose [cols][rows]. */
typedef struct {
  nvfp4a_tensor row, col;
  int ld_row, ld_col;
} nvfp4a_dual_tensor;

/* D = alpha * A * B^T + bias + beta * C; see the header comment. */
nvfp4a_status nvfp4a_gemm(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda,
                          const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                          const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, cudaStream_t stream);

/* nvfp4a_gemm on dual tensors: D[i + j*ldd] = alpha * sum_k a(i, k) * b(j, k) + ..., where a is the
 * [m][k] matrix of A -- A's own matrix (trans_a = 0: A.row) or the transpose of it (trans_a = 1, A is
 * [k][m]: A.col, the quantization of the transpose) -- and b likewise the [n][k] matrix of B.  The kernels
 * need k contiguous in both operands, which is what the two halves provide.  For a linear layer
 * Y[t][o] = X[t][i] W[o][i]^T, with D the row-major result: forward Y: A = W, B = X (trans 0, 0); dgrad
 * dX = dY W: A = W with trans_a = 1, B = dY (trans_b = 0); wgrad dW = dY^T X: A = X, B = dY, both trans 1. */
nvfp4a_status nvfp4a_gemm_dual(int m, int n, int k, float alpha, const nvfp4a_dual_tensor *A, int trans_a,
                               const nvfp4a_dual_tensor *B, int trans_b, float beta, const nvfp4a_tensor *C,
                               int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                               cudaStream_t stream);

/* BF16 / FP16 / FP32 src [rows][cols] (ld lds) -> NVFP4 dst (ld ldd, 1 x 16 blocks, the destination's
 * global) or INT8 dst (ld ldd, FP32 block scales); dst->amax gets max |x| over src (e.g. to choose the
 * next global scale as amax / (6 * 448)).  cols, lds and ldd multiples of 16; src 16-byte, dst data
 * 8-byte (NVFP4) / 16-byte (INT8), block scales 1-byte (NVFP4) / 4-byte (INT8) aligned.  src must be
 * finite. */
nvfp4a_status nvfp4a_quantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                              int cols, cudaStream_t stream);

/* Quantization recipe of nvfp4a_quantize_ex. */
typedef enum { NVFP4A_4OVER6_OFF = 0, NVFP4A_4OVER6_MAE = 1, NVFP4A_4OVER6_MSE = 2 } nvfp4a_4over6;
typedef struct {
  int tile_rows;             /* 1: 1 x 16 blocks (activations, gradients); 16: 16 x 16 tiles (TE weights) */
  nvfp4a_4over6 four_over_six;
  int compute_amax;          /* 1: first write the global amax (max |x| of the tensor, or of each row with
                                NVFP4A_GLOBAL_PER_ROW) into global_scale of the NVFP4A_GLOBAL_AMAX
                                destinations; 0: use the global value they hold */
} nvfp4a_quant_config;

/* src [rows][cols] -> rowwise NVFP4 [rows][cols] (ld ld_row) and / or columnwise NVFP4 [cols][rows] (the
 * quantization of src^T, ld ld_col); either destination may be NULL.  Each destination's flags select
 * its scale layout and type and its global (an amax for 4over6 and per-row globals; per-row globals for
 * the rowwise destination only).  cols a multiple of 16; with a columnwise destination or 16 x 16 tiles
 * the last block may be partial (zeros fill it; ld_col >= roundup(rows, 16)).  Scale buffers hold the
 * padded matrix (swizzled: roundup(rows, 128) x roundup(cols / 16, 4)); padding is not written.
 * cfg NULL = 1 x 16, no 4over6, compute_amax = 0. */
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

/* Transpose of an NVFP4 matrix quantized with 16 x 16 tiles: src [rows][cols] -> dst [cols][rows], data
 * and scales (the global is shared: dst->global_scale may point at the same value).  Exact for 16 x 16
 * tiles (every scale is shared by its tile); other NVFP4 matrices must be quantized columnwise from the
 * source instead.  rows and cols multiples of 16. */
nvfp4a_status nvfp4a_transpose(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                               int cols, cudaStream_t stream);

/* ---- Tensor-core GEMM (sm_75 .. sm_89; BF16 B from sm_80) --------------------------------------------------------
 * nvfp4a_gemm_tc computes what nvfp4a_gemm does, D = alpha * A * B^T + bias + beta * C, on tensor cores, from images
 * of A and B: the operand's bytes and block scales rearranged in the order the kernel streams them, nothing added
 * (rows padded to 128): built once for a weight, per call for activations.  B is NVFP4 (W4A4), INT8 with FP32
 * block scales (W4A8) or BF16 (W4A16).  Sums are FP32 throughout (block sums exact integers, BF16 products of
 * a * scale exact) in the tensor cores' order and with the work split along k between CTAs: the values of
 * nvfp4a_gemm within the rounding of an FP32 accumulation, not bit for bit. */
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

/* A weight image (NVFP4A_TC_A16 / NVFP4A_TC_A8, from) of [rows][k] -> the image of the other kind (to), or back to the
 * NVFP4 tensor: dst data [rows][k] (ld values between rows, a multiple of 64; 16-byte aligned) with linear block
 * scales (dst->ld_scale, 0 = k / 16, a multiple of 4; 4-byte aligned); dst's flags and global are not touched (the image
 * holds the scale bytes as they were).  Any supported device. */
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
 * nvfp4a_gemm_ex, nvfp4a_gemv_tc_ex and nvfp4a_gemm_tc_ex compute what nvfp4a_gemm, nvfp4a_gemv_tc and nvfp4a_gemm_tc do,
 * with these additions to the epilogue (each off when its pointer is NULL / its flag 0):
 *
 * Column scale of an RMSNorm (the deferred scalar: W (x w s) = s (W (x w))): column j of B (a token) is scaled by
 *     s_j = RN(1 / sqrt(sum_j * norm_inv_n + norm_eps)),  sum_j = the norm_nparts values at norm_parts + j * norm_ld,
 * summed by 32 lanes (lane l: values l, l + 32, ... in ascending order) and then an xor butterfly (16, 8, 4, 2, 1);
 * it enters the output's scale with B's global: ((alpha * vA(i)) * (vB(j) * s_j)) * f.  The values are read through L2
 * (they may come from another kernel of the same program).
 *
 * SwiGLU: the rows of A come in slices of 16, the gate of outputs 8q .. 8q + 7 and then their up; D has m / 2 rows,
 * D[8q + r] = silu(g) * u (FP32, silu(g) = g / (1 + exp(-g)) with the GPU's fast exp and division), g and u the
 * values the plain epilogue gives rows 16q + r and 16q + 8 + r (scale and bias); the amax statistic over D.  m a
 * multiple of 16; not with C (beta 0) or a second output.
 *
 * Second output and the parts of its sum of squares (the input of the next RMSNorm): with v a value of D as stored
 * (before its rounding), d2[i + j * ldd2] = RN(v * w2[i]) and, per m16 slice s of the rows and column j, sq[j * ld_sq
 * + s] = sum of RN(v * v) over the slice's 16 rows in ascending order (each part written by one thread: no atomics).
 * m a multiple of 16.  The parts are the norm_parts of the next call (norm_nparts = m / 16 there).
 *
 * With any addition: A's block scales linear E4M3 (nvfp4a_gemm_ex) or E4M3 (the tensor-core entries), B BF16 or INT8.
 * The work of these calls as items for an executor of one's own: include/nvfp4effma_items.h. */
typedef struct {
  const float *norm_parts;       /* column scale: the parts of each column's sum of squares (device), NULL = none */
  int norm_nparts, norm_ld;
  float norm_inv_n, norm_eps;
  int swiglu;                    /* 1: SwiGLU of interleaved gate / up rows */
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
