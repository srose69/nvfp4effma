# Numerics

## The format

An NVFP4 value is an E2M1 code (sign, 2 exponent bits, 1 mantissa bit: ±{0, ½, 1, 1½, 2, 3, 4, 6}) times
the E4M3 scale of its block of 16 consecutive values along a row, times one FP32 scale for the whole
tensor:

    value = e2m1 · block_scale · global_scale

Twice an E2M1 value is an integer in {0, ±1, 2, 3, 4, 6, 8, 12}, i.e. an exact int8. That is what the
NVFP4 × NVFP4 kernels are built on.

## Guarantees

- **NVFP4 × NVFP4 / INT8, tiled kernels.** Per output and per block of 16 (k ascending) the block sum
  `p = Σ (2a)·b'` (`b' = 2b` for NVFP4, the int8 value for INT8) is an exact integer (|p| ≤ 2304, resp.
  ≤ 24384); `y = p · sa` is exact (at most 15 × 4 significant bits); `acc = fma(y, sb, acc)` rounds once
  per block, not once per MAC; finally `acc · ¼` resp. `acc · ½` (exact). The DP4A kernel (sm_61+) and the
  FFMA kernel used below sm_61 compute the same values bit for bit (`make test` on the GPU,
  `make test-sm60` on the sm_60 path).
- **NVFP4 × BF16 / FP16 / FP32, tiled kernel.** `acc = fma(a · sa, b, acc)` per MAC, k ascending. `a · sa`
  is exact (at most 6 significant bits, ≥ 2⁻¹⁰), so the product with a BF16 or FP16 `b` is exact whenever
  it is a normal FP32 number; FP32 `b` products round as in SGEMM.
- **GEMV kernel** (small n, every B type; columns in groups of 8, each output computed on its own), in
  the kernel's own fixed order: dense B — lane `l` accumulates `k = 512c + 16l + j` in `(c, j)` order, one
  FFMA per MAC as above; NVFP4 / INT8 B — lane `l` accumulates the blocks `32c + l` with the exact block
  sums of the tiled kernels; then the 32 lanes are combined by an xor butterfly (16, 8, 4, 2, 1). It runs
  for n up to a limit per B dtype (`kGemvMaxN` in `src/nvfp4effma.cuh`: 48 for NVFP4, 56 for INT8 / BF16 /
  FP16, 32 for FP32), where it is faster than the tiled kernel. Internally the dense path works with `a · sa · 2⁻⁹` and restores the
  factor before the epilogue (exactly); this only matters if a single product falls below 2⁻¹¹⁷ in
  magnitude, where it underflows earlier than documented.
- **Epilogue**, in FP32, in this order, with a single rounding to the output type: `v = acc · s`
  (`s = (α · gA) · gB`, the global scales of NVFP4 operands); `v = v + bias[i]`; `v = β · C + v` (one
  FMA); `amax = max(amax, |v|)`; `D = RNE(v)` (BF16 / FP16 / FP32).
- **Reproducibility.** Results are deterministic and bit-reproducible for a fixed kernel path. Which path
  runs depends on n (GEMV up to `kGemvMaxN`) and, for NVFP4 × NVFP4 / INT8, not on the architecture. Cross-architecture
  execution has been validated on sm_61 (and the sm_60 code on it) only.
- **Contract.** Block scales must be finite (no E4M3 NaN code `0x7F` / `0xFF`); they are not checked.
  Quantization inputs must be finite.

## Quantization

Per block of 16 values `x` of a row, with `g` the global scale of the destination:

    sb = E4M3_RNE_satfinite(amax_block / (6 · g))
    q  = E2M1_RNE_satfinite(x / (sb · g))          (q = ±0 where sb = 0)

Both divisions are correctly rounded FP32 divisions; E2M1 rounding is to nearest, ties to the even code,
saturating at ±6, and keeps the sign of zero. `nvfp4a_dequantize` computes `(e2m1 · sb) · g`, rounded once.

- **Global scale.** The usual choice is `g = amax(|x|) / (6 · 448)`: the block with the largest value gets
  the top E4M3 scale. The Python `quantize` uses it by default; the C API takes `g` as given and reports
  `max |x|` in `dst->amax` for the next call.
- **Rounding the block scale to nearest** (as ModelOpt / Transformer Engine do, so quantized weights match
  theirs) can round `amax_block / 6` down; the block's largest value then saturates at 6 with a relative
  error of at most one E4M3 step (1/16). Rounding the scale up instead would avoid saturation at the cost
  of a coarser grid for the whole block; the library follows the common convention.

## W4A8 activations

`nvfp4a_quantize` to INT8: per block of 16 values `sb = amax_block / 127` (FP32, correctly rounded) and
`q = RNE(x / sb)`, `q = 0` where `sb = 0`. That is a relative error of at most 1/254 of the block's largest
value per activation — the accuracy class of the int8 activations llama.cpp uses for its quantized
matmuls (blocks of 32 there), against 4-bit activations for W4A4.

## Why FP32 accumulation and one rounding at the end

The same argument as for the FP8 libraries ([E4M3EffMA](https://github.com/srose69/e4m3effma),
[E5M2EffMA](https://github.com/srose69/e5m2effma)): rounding a BF16 accumulator after every MAC makes
same-sign sums stagnate and zero-mean sums random-walk; FP32 accumulation with one rounding when the
result is stored keeps the error at the output rounding for any practical K. NVFP4 × NVFP4 is even one
step better: within a block of 16 there is no rounding at all.
