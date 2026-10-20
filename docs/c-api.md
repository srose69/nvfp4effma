# C API

Everything is declared in [`include/nvfp4effma.h`](../include/nvfp4effma.h), which also documents the
exact order of operations ([numerics.md](numerics.md)).

```c
typedef struct {
  void *data;                /* device pointer */
  nvfp4a_dtype dtype;        /* NVFP4A_NVFP4, NVFP4A_INT8, NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32 */
  void *block_scale;         /* NVFP4: E4M3 / UE5M3 block scales, INT8: FP32 ones; one per 16 values of a row */
  int ld_scale;              /* linear scales: scales between consecutive rows; 0 = cols / 16 */
  const float *global_scale; /* NVFP4 / INT8: decode factor or amax (device), see flags; NULL = 1 */
  float *amax;               /* optional output: max |value|, max-reduced into it */
  unsigned flags;            /* NVFP4 only: NVFP4A_SCALES_* / NVFP4A_GLOBAL_* (below); 0 = ModelOpt layout */
  int scale_max;             /* bound of an amax global: 0 = the scale type's maximum, or 256 (4over6) */
} nvfp4a_tensor;
```

An NVFP4 matrix `[rows][cols]` is stored row-major with two values per byte — column `2c` in the low
nibble of byte `c`, `2c + 1` in the high one — `ld` values apart; with `flags = 0` the scale of value
`(r, c)` is `block_scale[r · ld_scale + c / 16]`, and `value = e2m1 · block_scale · global_scale`. This is
the layout of ModelOpt checkpoints (`weight` uint8, `weight_scale` E4M3, `weight_scale_2` FP32); the
Transformer Engine / cuBLASLt variants are selected with `flags`
([NVIDIA NVFP4 formats](#nvidia-nvfp4-formats)). The two fields at the end may be left out of an
initializer (`{data, dtype, scales, 0, global, NULL}` is the ModelOpt format).

INT8 is the activation format of the W4A8 path: int8 values, `ld` per row, and one FP32 scale per 16
values of a row (`value = q · block_scale`); `nvfp4a_quantize` produces it from BF16 / FP16 / FP32 with
`sb = amax_block / 127`, `q = RNE(x / sb)`.

| Function | Description |
|---|---|
| `nvfp4a_gemm` | `D = α·A·Bᵀ + bias + β·C`, column-major D (`D[i + j·ldd]`). A: NVFP4, m rows of k values. B: NVFP4 (W4A4), INT8 (W4A8), BF16, FP16 or FP32 (W4A16), n rows of k values. C, D, bias: BF16, FP16 or FP32; C may alias D, and `C = NULL` with β ≠ 0 accumulates into D. For a linear layer `y = x·Wᵀ`: A = W (`m` = out features), B = x (`n` = tokens), D = y (`ldd` = out features). For small `n` the bandwidth-bound GEMV kernel runs, in groups of 8 columns. |
| `nvfp4a_gemm_dual` | `nvfp4a_gemm` on tensors with a rowwise and a columnwise representation (`nvfp4a_dual_tensor`): `trans_a` / `trans_b` pick the half whose blocks run along k — the backward GEMMs of a linear layer (example below). |
| `nvfp4a_quantize` | BF16 / FP16 / FP32 `[rows][cols]` → NVFP4 (1 × 16 blocks, the destination's flags and global), or → INT8 (W4A8 activations); `dst->amax` gets `max |x|`. |
| `nvfp4a_quantize_ex` | Transformer Engine's quantization: rowwise and / or columnwise destination, 1 × 16 or 16 × 16 blocks, 4over6 (MAE / MSE), the global amax computed first (`nvfp4a_quant_config`). |
| `nvfp4a_dequantize` | NVFP4 → BF16 / FP16 / FP32: `(e2m1 · sb) · g` (amax global: `e2m1 · ((sb · amax) · 1 / (6 M))`), rounded once; `dst->amax` gets `max |value|`. |
| `nvfp4a_swizzle_scales` | Block scales linear ↔ GEMM-swizzled. |
| `nvfp4a_transpose` | Transpose of a 16 × 16-tiled NVFP4 matrix (data and scales; exact for tiles). |
| `nvfp4a_status_string` | Name of a status code. |
| `nvfp4a_last_error` | Description of the last failure on the calling thread: the invalid argument, the CUDA error, or a device the build does not cover (sm_100+ with native FP4, or an architecture missing from `ARCHS`). |

**Constraints of `nvfp4a_gemm`:** `k`, `lda`, `ldb` multiples of 64; A and B 16-byte aligned; `ld_scale` a
multiple of 4, NVFP4 block scales 4-byte and INT8 ones 16-byte aligned; `m`, `n`, `ldc`, `ldd` arbitrary (`ldc, ldd ≥ m`); block
scales finite (no E4M3 NaN code; not checked).
**Constraints of the conversions:** `cols`, `lds`, `ldd` multiples of 16; dense data 16-byte aligned, NVFP4
data 8-byte aligned; quantization input finite.

## NVIDIA NVFP4 formats

`flags` of an NVFP4 tensor (any combination, A and B independently, in every function):

| Flag | Meaning |
|---|---|
| `NVFP4A_SCALES_SWIZZLED` | Block scales in the GEMM layout of cuBLASLt / Transformer Engine: the scale matrix padded to `roundup(rows, 128)` × `sc = roundup(cols / 16, 4)`, in 512-byte tiles of 128 × 4 scales; scale `(r, c)` at `((r / 128) · (sc / 4) + c / 4) · 512 + (r % 32) · 16 + ((r / 32) % 4) · 4 + c % 4`. `ld_scale` must be 0 or `sc`. |
| `NVFP4A_SCALES_UE5M3` | UE5M3 block scales (unsigned, 5 exponent bits, bias 15, maximum 114688) instead of E4M3. |
| `NVFP4A_GLOBAL_AMAX` | `global_scale` holds the amax (Transformer Engine): the decode factor is `amax / (6 · M)`, `M = scale_max` (0 → 448 for E4M3, 114688 for UE5M3; 256 for 4over6). |
| `NVFP4A_GLOBAL_PER_ROW` | One global value per row (row-scaled NVFP4): `global_scale[rows]`. |

Without `NVFP4A_GLOBAL_AMAX` the global is a decode factor (ModelOpt's `weight_scale_2`). The GEMM output
scale is `((α · vA(i)) · vB(j)) · gf`, Transformer Engine's order, with `gf` the product of the amax
factors `1 / (6 · M)`. Transformer Engine's tensors map as follows: `_rowwise_data` / `_rowwise_scale_inv`
/ `_amax_rowwise` (and the columnwise ones) are `data` / `block_scale` / `global_scale` with
`NVFP4A_GLOBAL_AMAX`, `_with_gemm_swizzled_scales` is `NVFP4A_SCALES_SWIZZLED`, `_row_scaled_nvfp4` is
`NVFP4A_GLOBAL_PER_ROW`, `_nvfp4_e4m3_max` is `scale_max`; its scale buffers are always padded
(`roundup(rows, 128)` × `roundup(cols / 16, 4)`), so a linear one has `ld_scale = roundup(cols / 16, 4)`.
16 × 16-tiled weights need no flag: their scales are stored per row like any other (replicated over the 16
rows of a tile).

```c
// Transformer Engine's recipe for activations X [tokens][in] (BF16): amax computed, 1 x 16 blocks,
// rowwise and columnwise halves, swizzled scales
nvfp4a_dual_tensor Xq = {
    .row = {d_xr, NVFP4A_NVFP4, d_xr_sc, 0, d_amax_r, NULL, NVFP4A_SCALES_SWIZZLED | NVFP4A_GLOBAL_AMAX, 0},
    .col = {d_xc, NVFP4A_NVFP4, d_xc_sc, 0, d_amax_c, NULL, NVFP4A_SCALES_SWIZZLED | NVFP4A_GLOBAL_AMAX, 0},
    .ld_row = in, .ld_col = tokens};
nvfp4a_quant_config qc = {1, NVFP4A_4OVER6_OFF, /*compute_amax=*/1};
nvfp4a_quantize_ex(&X, in, tokens, in, &Xq.row, in, &Xq.col, tokens, &qc, stream);

// weights: 16 x 16 tiles, 4over6 (MAE) with the E4M3 bound 256
nvfp4a_tensor Wq = {d_w, NVFP4A_NVFP4, d_w_sc, 0, d_w_amax, NULL, NVFP4A_SCALES_SWIZZLED | NVFP4A_GLOBAL_AMAX, 256};
nvfp4a_quant_config qw = {16, NVFP4A_4OVER6_MAE, 1};
nvfp4a_quantize_ex(&Wf, in, out, in, &Wq, in, NULL, 0, &qw, stream);

// forward Y = X W^T, and the backward GEMMs from the dual tensors (D row-major [rows of the result]):
// Wd = {Wq, its transpose (nvfp4a_transpose: exact for 16 x 16 tiles)}, dYq = the gradient quantized like Xq
nvfp4a_gemm(out, tokens, in, 1.0f, &Wq, in, &Xq.row, in, 0.0f, NULL, 0, &Y, out, NULL, stream);
nvfp4a_gemm_dual(in, tokens, out, 1.0f, &Wd, 1, &dYq, 0, 0.0f, NULL, 0, &dX, in, NULL, stream);      // dX = dY W
nvfp4a_gemm_dual(in, out, tokens, 1.0f, &Xq, 1, &dYq, 1, 0.0f, NULL, 0, &dW, in, NULL, stream);      // dW = dY^T X
```

Not supported: the random Hadamard transform and stochastic rounding of Transformer Engine's gradient
recipe, and 4over6's fast-math error mode (the default, exact division, is what NVFP4EffMA computes).
Transformer Engine maps `-0.0` to code 0 in its Python reference; NVFP4EffMA, like the hardware
conversion, keeps the sign (code 8) — both decode to 0.

## Errors

Every function returns an `nvfp4a_status` (`NVFP4A_OK`, `NVFP4A_INVALID_VALUE`, `NVFP4A_NOT_SUPPORTED`,
`NVFP4A_CUDA_ERROR`) and validates its arguments before launching anything. On failure
`nvfp4a_last_error()` says what was wrong:

```
nvfp4a_gemm: k = 96 is not a multiple of 64
nvfp4a_gemm: block scales must be 4-byte aligned with ld_scale a multiple of 4 and >= k / 16 (ld_scale = 6)
device 0 is sm_61, but this build of the library has no code for it: rebuild with ARCHS including 61
```

## Example

```c
#include "nvfp4effma.h"

// A linear layer y [tokens, out] = x [tokens, in] · Wᵀ, W NVFP4 [out, in] (e.g. from a checkpoint).
nvfp4a_tensor W = {d_w, NVFP4A_NVFP4, d_w_scale, 0, d_w_scale_2, NULL};  // data, dtype, scales, ld_scale, global, amax
nvfp4a_tensor X = {d_x, NVFP4A_BF16, NULL, 0, NULL, NULL};
nvfp4a_tensor Y = {d_y, NVFP4A_BF16, NULL, 0, NULL, NULL};
nvfp4a_tensor B = {d_bias, NVFP4A_BF16, NULL, 0, NULL, NULL};

// W4A16: weights NVFP4, activations BF16 (GEMV kernel for small token counts)
nvfp4a_status st = nvfp4a_gemm(out, tokens, in, 1.0f, &W, in, &X, in, 0.0f, NULL, 0, &Y, out, &B, stream);
if (st != NVFP4A_OK) fprintf(stderr, "%s: %s\n", nvfp4a_status_string(st), nvfp4a_last_error());

// W4A8: int8 activations with an FP32 scale per 16 values, exact integer block sums on DP4A from sm_61
nvfp4a_tensor X8 = {d_x8, NVFP4A_INT8, d_x8_scale, 0, NULL, NULL};   // int8 [tokens][in], float [tokens][in / 16]
nvfp4a_quantize(&X, in, &X8, in, tokens, in, stream);
nvfp4a_gemm(out, tokens, in, 1.0f, &W, in, &X8, in, 0.0f, NULL, 0, &Y, out, &B, stream);

// W4A4: quantize the activations to NVFP4 instead (global scale chosen by the caller)
nvfp4a_tensor Xq = {d_xq, NVFP4A_NVFP4, d_xq_scale, 0, d_x_global, NULL};
nvfp4a_quantize(&X, in, &Xq, in, tokens, in, stream);
nvfp4a_gemm(out, tokens, in, 1.0f, &W, in, &Xq, in, 0.0f, NULL, 0, &Y, out, &B, stream);
```

The library is a plain C ABI, so it can also be called from any language with a C FFI.
