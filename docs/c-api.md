# C API

Everything is declared in [`include/nvfp4effma.h`](../include/nvfp4effma.h), which also documents the
exact order of operations ([numerics.md](numerics.md)).

```c
typedef struct {
  void *data;                /* device pointer */
  nvfp4a_dtype dtype;        /* NVFP4A_NVFP4, NVFP4A_INT8, NVFP4A_BF16, NVFP4A_FP16, NVFP4A_FP32 */
  void *block_scale;         /* NVFP4: E4M3 block scales, INT8: FP32 ones; one per 16 values of a row */
  int ld_scale;              /* NVFP4 / INT8: scales between consecutive rows; 0 = cols / 16 */
  const float *global_scale; /* NVFP4 / INT8: per-tensor factor (device, 1 float), NULL = 1 */
  float *amax;               /* optional output: max |value|, max-reduced into it */
} nvfp4a_tensor;
```

An NVFP4 matrix `[rows][cols]` is stored row-major with two values per byte — column `2c` in the low
nibble of byte `c`, `2c + 1` in the high one — `ld` values apart; the scale of value `(r, c)` is
`block_scale[r · ld_scale + c / 16]`, and `value = e2m1 · block_scale · global_scale`. This is the layout
of NVFP4 checkpoints (ModelOpt: `weight` uint8, `weight_scale` E4M3, `weight_scale_2` FP32).

INT8 is the activation format of the W4A8 path: int8 values, `ld` per row, and one FP32 scale per 16
values of a row (`value = q · block_scale`); `nvfp4a_quantize` produces it from BF16 / FP16 / FP32 with
`sb = amax_block / 127`, `q = RNE(x / sb)`.

| Function | Description |
|---|---|
| `nvfp4a_gemm` | `D = α·A·Bᵀ + bias + β·C`, column-major D (`D[i + j·ldd]`). A: NVFP4, m rows of k values. B: NVFP4 (W4A4), INT8 (W4A8), BF16, FP16 or FP32 (W4A16), n rows of k values. C, D, bias: BF16, FP16 or FP32; C may alias D, and `C = NULL` with β ≠ 0 accumulates into D. For a linear layer `y = x·Wᵀ`: A = W (`m` = out features), B = x (`n` = tokens), D = y (`ldd` = out features). For small `n` the bandwidth-bound GEMV kernel runs, in groups of 8 columns. |
| `nvfp4a_quantize` | BF16 / FP16 / FP32 `[rows][cols]` → NVFP4 with the destination's `global_scale`, or → INT8 (W4A8 activations); `dst->amax` gets `max |x|`. |
| `nvfp4a_dequantize` | NVFP4 → BF16 / FP16 / FP32: `(e2m1 · sb) · g`, rounded once; `dst->amax` gets `max |value|`. |
| `nvfp4a_status_string` | Name of a status code. |
| `nvfp4a_last_error` | Description of the last failure on the calling thread: the invalid argument, the CUDA error, or a device the build does not cover (sm_100+ with native FP4, or an architecture missing from `ARCHS`). |

**Constraints of `nvfp4a_gemm`:** `k`, `lda`, `ldb` multiples of 64; A and B 16-byte aligned; `ld_scale` a
multiple of 4, NVFP4 block scales 4-byte and INT8 ones 16-byte aligned; `m`, `n`, `ldc`, `ldd` arbitrary (`ldc, ldd ≥ m`); block
scales finite (no E4M3 NaN code; not checked).
**Constraints of the conversions:** `cols`, `lds`, `ldd` multiples of 16; dense data 16-byte aligned, NVFP4
data 8-byte aligned; quantization input finite.

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
