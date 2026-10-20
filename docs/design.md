# Design notes

Everything here was implemented and measured on a GTX 1080 (GP104, sm_61) during development. MAC/clk/SM
figures are 4096³ MACs per measured time, per SM clock actually observed, per SM; comparisons are
interleaved A/B runs, because small source changes move ptxas's schedule by several percent.

## NVFP4 × NVFP4: four exact products per instruction

On Pascal an exact FP8 MAC costs one FP32 FFMA ([E4M3EffMA](https://github.com/srose69/e4m3effma) design
notes): FP8 in int8 needs several digit planes, so DP4A loses. E2M1 is different: **2 × E2M1 is an exact
int8**, so one DP4A does four exact products and adds them to an int32.

Per output and block of 16:

    p = dp4a(a0, b0, 0x4B400000) … dp4a(a3, b3, p)   the bits of the FP32 value 1.5·2²³ + p
    y = fma(as_float(p), sa, −1.5·2²³·sa)             = p · sa exactly
    acc = fma(y, sb, acc)                             one rounding per block

The magic start turns the integer sum into an FP32 value without the quarter-rate I2F, and the FFMA with
the row scale removes the magic in the same instruction. That is **6 issue slots per 16 MACs instead of
16** — a ceiling of 2.67× the FFMA kernel. Measured with this instruction mix in isolation: about 268
MAC/clk/SM with 16 warps per SM, about 220 with 8.

The kernel: 128 × 128 block, 256 threads, 8 × 8 outputs per thread, k-tile of 4 blocks, one 16-byte load
per thread and operand per k-tile, double-buffered int8 tiles in shared memory with one barrier per
k-tile (tile t + 1 is decoded at the start of iteration t, the loads of tile t + 2 are issued right after).

- **One block per SM.** The kernel needs ~155–175 registers. Squeezed to 128 (two blocks, 16 warps)
  ptxas schedules the dependent DP4A chains with stalls of 5–6 cycles; every variant tried that way
  (leaner addressing, smaller tiles, register-only double buffering) ran at 146–186 MAC/clk/SM.
- **Decode.** Nibbles become int8 once per block through PRMT lookups: a + table and a − table indexed by
  the 3-bit magnitudes, and a sign mask that a PRMT sign-replicate builds from the nibbles' sign bits; one
  LOP3 selects. Two PRMTs are not enough: the sign-replicate mode gives the same inputs for a positive
  value and a negative non-zero one in bit 7, so no function of two lookups (and a constant) can produce
  both. Feeding pre-decoded int8 tiles instead — decode for free — gives 209–210 MAC/clk/SM: decode costs
  about 3.5 %.
- **Epilogue.** With one block per SM nothing overlaps the epilogue, so a generic shared-memory staged
  epilogue (as in the FFMA kernel) cost 8–10 %. Without C every output is stored straight from the
  registers; an interior tile without bias or amax skips the per-output checks. With C (accumulation into
  D) the results go through the tile buffers in two halves, so that a warp reads C and writes D along 128
  consecutive rows with vector accesses and the C / D types are compile-time: 10.1 instead of 11.3 ms at
  4096³ with an FP32 D, close to the cost of the extra DRAM traffic. Overlapping it with the next tile
  (a persistent kernel) would need the finished tile somewhere: 64 KB of FP32 is more than the 48 KB of
  shared memory a block can have on sm_61, and a second set of accumulators does not fit the registers.
  Prefetching C into L2 during the k loop was tried and was slower.
- **Result:** 197–202 MAC/clk/SM, 9.7–9.9 ms at 4096³ (about 7 TMAC/s at 1.8 GHz), 1.96× cuBLAS SGEMM.
  Accumulating into an FP32 D (`β = 1`) adds the read of D to the unoverlapped epilogue: 189 MAC/clk/SM.
- **A 64-wide tile** (8 × 4 outputs per thread, ~120 registers, two blocks per SM) for n where the last
  128-wide tile would be at most half full: n = 57 … 64 and 129 … 192. It costs about 2/3 of the
  128-wide tile for half the work, so beyond that the large tile wins; at n = 64 it takes 0.76–0.80 ms
  instead of 1.2–1.4 ms (m × k = 4096 × 14336 and 8192²).
- **Below sm_61** (no DP4A: sm_50 … sm_53, sm_60) the FFMA kernel runs on integer-valued FP32 tiles: 16
  FFMAs give the exact block sum, then the same `fma(p · sa, sb, acc)`, so both paths produce the same
  bits. Packing two products into one FFMA (`a = x0 + W·x1`, the middle field of the product carries two
  MACs) works exactly for W = 2¹² up to about 14 k per extraction, but a whole block of 16 at the extremes
  exceeds it (0.03 % of cases): not used.

## NVFP4 × INT8 (W4A8)

The same kernel with int8 activations: B tiles are stored as they arrive (no decode), the block scales of B
are FP32 (one float4 per row and k-tile), and the block sums reach 16 · 12 · 127 = 24384 — still far below
the 2²² the magic-constant conversion allows, and `p · sa` stays exact. 9.4 ms at 4096³ (7.3 TMAC/s), as
fast as W4A4 or slightly faster: B needs no decode. With llama.cpp's int8 activations as the reference
point this is the like-for-like comparison (below).

## NVFP4 × BF16 / FP16 / FP32: the scale goes into the decode

The tiled FFMA kernel of the FP8 libraries, with an NVFP4 tile loader: each value is decoded once per
block into the FP32 shared tile as `a · sa` — `sign | (n & 7) << 22` is the FP32 encoding of
`value · 2⁻¹²⁶` (the subnormal ½ lands on an FP32 subnormal), one FMUL by 2¹²⁶ and one by the block scale
restore it exactly. The inner loop is LDS.128 + one FFMA per MAC and does not know the format; it runs at
cuBLAS SGEMM speed (74–75 % of the FP32 peak) while reading 7× fewer weight bytes.

## GEMV (small n)

Bandwidth-bound, every byte of A read once: a lane takes one block (8 bytes + its scale) of each of R
rows per pass. At 4.5 bits per value the decode is the critical part — at the DRAM peak of a GTX 1080 there
are about 8 issue slots per weight value.

- **Dense x:** the two FMULs of the decode become one: the raw encoding times `sa · 2¹¹⁷` (cannot overflow:
  sa ≤ 448 < 2⁹) is `a · sa · 2⁻⁹`, and the exact factor 2⁹ is applied once per output before the
  epilogue.
- **NVFP4 / INT8 x:** int8 decode of both sides (INT8 x as it is) and DP4A per block, as in the tiled
  kernel (FFMA on integers below sm_61). This made n = 4 … 8 2–3× faster than decoding x to FP32.
- **More than 8 columns** run as groups of 8, each reading A once. The tiled kernels have a single row of
  128-wide tiles there and leave most of the GPU idle: at m = k = 8192 the DP4A kernel takes ~1.34 ms for
  any n ≤ 64. Grouped GEMV wins up to n = 48 (NVFP4), 56 (INT8, BF16, FP16) and 32 (FP32) on a GTX 1080
  (`kGemvMaxN`).

## Operand formats: one kernel instance per scale format

The Transformer Engine / cuBLASLt variants of NVFP4 differ in the block scale layout (linear or
GEMM-swizzled) and type (E4M3 or UE5M3) of each operand, and in the globals (decode factor or amax, per
tensor or per row). The scale formats change the main loops, so they are a template parameter `F`
(`kF_ASWZ`, `kF_AUE`, `kF_BSWZ`, `kF_BUE`): 16 instances for NVFP4 B, 4 for the other B dtypes, one object
file each, dispatched on the host. Run-time format tests in the kernels were measured first and rejected:
one per-element branch in the FFMA epilogue cost the dense-B GEMM 7 % (ptxas allocated the main loop's
registers differently), and recomputing the swizzled scale address per iteration made the GEMV with NVFP4 /
INT8 x up to 2× slower. In the instances the swizzled scales are stepped pointers like the linear ones
(+1 per k-tile, 512 bytes every 4 k-tiles; the GEMV's x rows of a launch lie in one 32-row group, so their
row parts are a base plus 16 bytes per row).

The globals only enter the epilogue, and one code path handles all of them: per row of A and of B a
pointer with stride 0 or 1 (NULL = 1), loaded once per output row / column of the thread, and
`((α · vA) · vB) · gf` per output — the same bits as `(α · gA) · gB` for per-tensor decode factors
(`gf = 1`). Keeping them in `F` as well (128 instances) made the library 8× larger for differences of
under 1 % on the GTX 1080.

## Quantization kernels

- **Rowwise 1 × 16**: one thread per block, consecutive threads on consecutive blocks of a row — coalesced,
  89 % of the DRAM peak.
- **Rowwise 16 × 16**: the warp keeps a 16 × 16 tile per 4 lanes in registers (4 values per lane and row: a
  warp reads 8 consecutive blocks of each of the 16 rows), the tile amax comes from registers and two
  shuffles, the nibbles are stored coalesced. 4 values per lane, not a 16-byte word: with 8 BF16 values
  per lane the per-lane work doubles at the same occupancy (2.9× slower). The E2M1 conversion there is the
  branchless form (`f32_to_e2m1_sel`, exhaustively equal), since divergent branches per element cost the
  instruction-level parallelism of a low-occupancy kernel.
- **4over6 and columnwise**: a unit of 16 lanes owns 16 rows of one block column, a lane per row; 16 × 16
  4over6 sums the rows' errors over the unit. 4over6 is bound by the division by `6 · M` in its error,
  which runs as `x · r` plus one FMA correction (`div_by`, exact for these divisors, with a branch to IEEE
  division only for subnormal quotients and infinities — zeros, frequent, take the exact fast path).
- **The global amax** (current scaling): a separate pass with 128-bit loads, two independent loads per
  thread; one atomic max per warp.

## Kernel configurations

Kernels are templates; the candidates are those that can win on some target, chosen per architecture by
`make tune` ([building.md](building.md#tuning)):

- **Dense-B GEMM**: 128 × 128, 64 × 64, 128 × 64 and 64 × 128 blocks, all with an 8 × 8 thread tile and a
  16-deep k-slice (one NVFP4 block); register budget 512 threads per SM at ≤ 128 registers.
- **NVFP4 × NVFP4 / INT8**: the DP4A kernel has a 128- and a 64-wide tile (above); the FFMA kernel below sm_61 uses the
  128 × 128 configuration with one block per SM (64 accumulators plus 64 block sums per thread).
- **GEMV**: 2, 4 or 8 rows per warp (8 only for n ≤ 2).
