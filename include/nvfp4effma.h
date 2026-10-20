// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* NVFP4EffMA: NVFP4 GEMM / GEMV / quantization for NVIDIA GPUs without native FP4 (sm_50 ... sm_90).
 *
 * NVFP4: E2M1 values (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}), one E4M3 scale per block of 16 consecutive
 * values along a row, and one FP32 scale per tensor:  value = e2m1 * block_scale * global_scale.
 * A matrix [rows][cols] is stored row-major, two values per byte (column 2c in the low nibble of byte
 * c, 2c + 1 in the high one), ld (in values) apart; block scale (r, c / 16) is the byte
 * block_scale[r * ld_scale + c / 16].
 * INT8 (activations for W4A8): int8 values, one FP32 scale per block of 16 values along a row:
 * value = q * block_scale (* global_scale); block_scale is a float array, ld_scale counts floats.
 *
 * GEMM (column-major D, the layout of a linear layer y = x W^T with A = W, B = x, D = y):
 *
 *     D[i + j*ldd] = alpha * sum_k A(i, k) * B(j, k) + bias[i] + beta * C[i + j*ldc]
 *
 *   A: NVFP4, m rows of k values (lda).  B: NVFP4, INT8, BF16, FP16 or FP32, n rows of k values (ldb).
 *   C, D, bias: BF16, FP16 or FP32.  C may alias D, and C == NULL with beta != 0 means C = D, so
 *   beta = 1 adds the product into D in place.
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
 *     than the tiled kernel (src/gemm_nvfp4effma.cu, kGemvMaxN).
 *   Epilogue: v = acc * s (s = (alpha * gA) * gB, the global scales of NVFP4 operands, 1 otherwise);
 *     v = v + bias[i]; v = beta * C + v (one FMA); amax = max(amax, |v|); D = RNE(v).
 *
 * Requirements: k, lda, ldb multiples of 64; A and B 16-byte aligned; ld_scale a multiple of 4, NVFP4
 * block scales 4-byte and INT8 ones 16-byte aligned; m, n, ldc, ldd unrestricted (ldc, ldd >= m).
 * Block scales must be finite (no E4M3 NaN code 0x7F / 0xFF; not checked).
 *
 * Quantization (nvfp4a_quantize), per block of 16 values x of a row, g = global scale of the destination:
 *     sb = E4M3_RNE_satfinite(amax_block / (6 * g)),  q = E2M1_RNE_satfinite(x / (sb * g))
 * (divisions correctly rounded in FP32; q = 0 where sb = 0; the sign of x is kept, also on zero).
 * To INT8: sb = amax_block / 127, q = RNE(x / sb) (q = 0 where sb = 0); dst->global_scale is ignored.
 */
#ifndef NVFP4EFFMA_H
#define NVFP4EFFMA_H

#define NVFP4EFFMA_VERSION "0.2.0"

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NVFP4A_NVFP4 = 0, NVFP4A_BF16 = 1, NVFP4A_FP16 = 2, NVFP4A_FP32 = 3, NVFP4A_INT8 = 4 } nvfp4a_dtype;
typedef enum { NVFP4A_OK = 0, NVFP4A_INVALID_VALUE = 1, NVFP4A_NOT_SUPPORTED = 2, NVFP4A_CUDA_ERROR = 3 } nvfp4a_status;

typedef struct {
  void *data;                /* device pointer */
  nvfp4a_dtype dtype;
  void *block_scale;         /* NVFP4: E4M3 block scales, INT8: FP32 ones (device), one per 16 values of a row */
  int ld_scale;              /* NVFP4 / INT8: scales between the block scales of consecutive rows; 0 = cols / 16 */
  const float *global_scale; /* NVFP4 / INT8: per-tensor factor (device, 1 float), NULL = 1 */
  float *amax;               /* optional output statistic: max |v| (device, 1 float, max-reduced into the
                                existing value); NULL = off */
} nvfp4a_tensor;

/* D = alpha * A * B^T + bias + beta * C; see the header comment. */
nvfp4a_status nvfp4a_gemm(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda,
                          const nvfp4a_tensor *B, int ldb, float beta, const nvfp4a_tensor *C, int ldc,
                          const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias, cudaStream_t stream);

/* BF16 / FP16 / FP32 src [rows][cols] (ld lds) -> NVFP4 dst (ld ldd, block scales, dst->global_scale
 * as given) or INT8 dst (ld ldd, FP32 block scales); dst->amax gets max |x| over src (e.g. to choose the
 * next global scale as amax / (6 * 448)).  cols, lds and ldd multiples of 16; src 16-byte, dst data
 * 8-byte (NVFP4) / 16-byte (INT8), block scales 1-byte (NVFP4) / 4-byte (INT8) aligned.  src must be
 * finite. */
nvfp4a_status nvfp4a_quantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                              int cols, cudaStream_t stream);

/* NVFP4 src -> BF16 / FP16 / FP32 dst [rows][cols]: v = (e2m1 * sb) * g, rounded once (RNE);
 * dst->amax gets max |v|.  Same alignment rules as nvfp4a_quantize. */
nvfp4a_status nvfp4a_dequantize(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows,
                                int cols, cudaStream_t stream);

const char *nvfp4a_status_string(nvfp4a_status status);

/* Description of the most recent failure of an nvfp4a_* call on the calling thread (which argument
 * was wrong, or the CUDA error); "" after a successful call.  Valid until the next nvfp4a_* call. */
const char *nvfp4a_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* NVFP4EFFMA_H */
