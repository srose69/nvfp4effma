# Testing

`make test` runs the C tests below on GPU `$(GPU)` (default 0); the exhaustive checks take a few seconds
on a GTX 1080. `make test-sm60` builds the library for sm_60 only and runs the GEMM / GEMV tests against
it: on an sm_61+ GPU that SASS runs (binary compatible), so it exercises the code of GPUs without DP4A —
the FFMA NVFP4 × NVFP4 kernels — and the default kernel tables.

| Test | What it checks |
|---|---|
| `tests/test_gemm.cu` | `nvfp4a_gemm`, tiled kernels, bit-exact against a host emulation of the documented order: NVFP4 A × B in {NVFP4, INT8, BF16, FP16, FP32}, m / n tails, padded `ld` and `ld_scale`, global scales, every output dtype, C of another dtype, accumulation into D, α / β / bias / amax, zero block scales, larger problems. |
| `tests/test_gemv.cu` | The GEMV path, bit-exact against an emulation of its lane order: x in {NVFP4, INT8, BF16, FP16, FP32} × n = 1 … 8, every output dtype, accumulation, bias, k not a multiple of the 512-value pass; n = 12 … 49 (groups of 8) on both sides of the GEMV limit. |
| `tests/test_nvfp4effma.cu` | The DP4A scale step (`fma(1.5·2²³ + p, sa, −1.5·2²³·sa) = p·sa` for every block sum of W4A4 and W4A8, \|p\| ≤ 24384, and every finite scale); the PRMT int8 decode on all 2¹⁶ inputs and the FP32 decode of every nibble at every position; exactness of the dense-B operands (`a·sa` times every BF16 / FP16 value, 2²⁸ cases each); FP32 → BF16, E2M1 and E4M3 rounding on all 2³² FP32 bit patterns against independent midpoint searches; `nvfp4a_quantize` (to NVFP4 and to INT8) / `nvfp4a_dequantize` bit-exact against a host emulation, with zero, underflowing and saturating blocks. |
| `test_nvfp4effma -DNVFP4A_NEGATIVE_CONTROLS` | Five deliberately broken variants (the magic removed after rounding, decode without signs, RNE without NaN handling, E2M1 and E4M3 ties away from zero) must be rejected. |
| `binds/python/tests` | The PyTorch bindings (`pytest`): quantization bit-exact against an independent PyTorch reference of the formulas, dequantization, `linear` with dense, INT8 (`act="int8"`) and NVFP4 activations against a float64 reference (GEMV and tiled), batched inputs and accumulation, strided inputs, `NVFP4Linear` / `quantize_linear_`, and rejection of bad input. |
