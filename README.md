<div align="center">

# NVFP4EffMA

**NVFP4 GEMM, GEMV, linear layers and quantization for NVIDIA GPUs without native FP4**

![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia)
![Targets](https://img.shields.io/badge/targets-sm__50%E2%80%93sm__90-blue)
![Tested](https://img.shields.io/badge/tested%20on-GTX%201080-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)
![API](https://img.shields.io/badge/API-C%20%2F%20cuBLAS--style-lightgrey)
![License](https://img.shields.io/badge/license-AGPL--3.0%20%2B%20linking%20exception-blue)

</div>

NVIDIA's native FP4 compute stack (NVFP4 in cuBLASLt, Transformer Engine and ModelOpt) starts at
Blackwell-class GPUs (sm_100+). **NVFP4EffMA** brings the same data format to the GPUs most people still
have — Maxwell, Pascal, Volta, Turing, Ampere, Jetson Orin, Ada and Hopper (sm_50 … sm_90). Weights stay
in NVFP4 (E2M1 values, one 8-bit scale per 16 values, an FP32 global: 4.5 bits per value, 7× smaller than
FP32), products are exact, accumulation is FP32, and outputs come out in BF16 / FP16 / FP32. Every NVFP4
variant NVIDIA uses is read and produced as it is, bit for bit: ModelOpt checkpoints, Transformer Engine's
1 × 16 and 16 × 16 recipes (amax globals, row-scaled activations, 4over6, UE5M3 scales, columnwise data) and
the GEMM-swizzled scale layout of cuBLASLt ([below](#nvidia-nvfp4-variants)).
Activations can be BF16 / FP16 / FP32 (W4A16, exact), int8 with a scale per 16 values (W4A8) or NVFP4
(W4A4); the PyTorch `linear` quantizes BF16 activations to int8 on the fly beyond 2 rows by default, where
that is twice as fast as the exact path. Twice an E2M1 value is an exact int8, so from sm_61 on one DP4A does four exact products: W4A8 and
W4A4 run at 2× cuBLAS SGEMM on a GTX 1080, and faster than llama.cpp's own NVFP4 kernels on every shape
measured (1.0–4.5×; 1.5× at 4096³, 3.9–4.5× for 8-token decoding).

For FP8 on the same GPUs, see the sibling libraries [E4M3EffMA](https://github.com/srose69/e4m3effma)
and [E5M2EffMA](https://github.com/srose69/e5m2effma); they can be used together with this one.

> [!IMPORTANT]
> **Tested and tuned by the author on a GTX 1080 only.** The author has no GPU newer than Pascal;
> everything else is up to the community, or to time. The code path of GPUs without DP4A (Maxwell, Tesla
> P100) is tested on the GTX 1080 by running SASS built for sm_60. The library compiles for every
> architecture from sm_50 to sm_90, but it does **not** use Tensor Cores (Turing's INT8 / INT4 `mma` would
> suit this format very well), `cp.async`, `ldmatrix`, or any other Volta/Turing/Ampere-specific feature,
> although it could and should on those GPUs. Contributions are very welcome — and so is hardware, which
> is honestly the better way to get those kernels written :)  See [Contributing](#contributing).

## Performance at a glance

GTX 1080. The baseline is cuBLAS on **FP32 data** of the same shape (SGEMM / SGEMV) — the fastest way to
run the same model on this GPU without 4-bit weights. Median times; the FP32 peak of each GEMM row is
taken at the SM clock measured while that row ran:

| Workload | NVFP4EffMA | Utilization | Baseline: cuBLAS FP32 | Speed vs baseline |
|---|---:|---|---:|---:|
| GEMM 4096³, NVFP4 × NVFP4 → BF16 | 9.42 ms | **156 %** of FP32 peak (7.3 TMAC/s) | SGEMM 19.1 ms (76 % of FP32 peak) | **2.03×** |
| GEMM 4096³, NVFP4 × INT8 (W4A8) → BF16 | 9.33 ms | **158 %** of FP32 peak (7.4 TMAC/s) | SGEMM 19.1 ms (76 % of FP32 peak) | **2.05×** |
| GEMM 4096³, NVFP4 × NVFP4, FP32 D += (accumulate) | 9.77 ms | 150 % of FP32 peak | SGEMM 19.1 ms (76 % of FP32 peak) | **1.96×** |
| GEMM 4096³, NVFP4 weights × BF16 activations, exact | 19.6 ms | 75 % of FP32 peak | SGEMM 19.1 ms (76 % of FP32 peak) | 0.98× |
| `linear` 4096³, BF16 activations, default `act="auto"` (int8 quantization included) | 10.0 ms | — | SGEMM 19.1 ms | **1.9×** |
| GEMV 8192 × 8192, batch 1 (BF16 x) | 0.139 ms | 272 GB/s, **85 %** of DRAM peak | SGEMV 1.042 ms | **7.5×** |
| GEMV 8192 × 8192, batch 8 (INT8 x) | 0.158 ms | 238 GB/s, 74 % of DRAM peak | SGEMM 1.538 ms | **9.7×** |
| GEMV 8192 × 8192, batch 8 (BF16 x) | 0.346 ms | 109 GB/s, 34 % of DRAM peak | SGEMM 1.538 ms | **4.4×** |
| Quantization, 8192 × 8192 BF16 → NVFP4 | 0.64 ms | 270 GB/s, **84 %** of DRAM peak | — | — |

Most Transformer Engine representations (swizzled or UE5M3 scales, amax and per-row globals) compute
within a few percent of plain NVFP4 — within 1.5 % in GEMM, −6.5 … +3 % in GEMV, where swizzled x scales
are even about 5 % faster. Creating them costs more than using them: quantize once, not per call
([docs/performance.md](docs/performance.md#transformer-engine-formats-using-them-is-cheap-creating-them-is-not);
full tables: [benches/1080.md](benches/1080.md)).

FP32 peak: 20 SMs × 128 FFMA × SM clock (1818–1837 MHz in the GEMM rows); DRAM peak: 320 GB/s; GEMV
bandwidth counts the values and the block scales. With both operands in NVFP4 the DP4A kernel does four
exact products per instruction and exceeds the FP32 FFMA peak; with BF16 activations a MAC costs one FP32
FFMA, so compute-bound GEMM tops out at SGEMM speed while reading 7× fewer weight bytes, and
bandwidth-bound kernels (decoding) get the full benefit of the smaller weights. **On Pascal, use W4A8 for
prefill speed** (`act="auto"` / `"int8"`): exact W4A16 is bound by the FP32 FFMA rate there (no fast FP16, no
int path for BF16 values) and runs at SGEMM speed. Against llama.cpp's NVFP4
kernels on the same GPU (int8 activations, like W4A8 here), interleaved runs: 1.0–4.5× faster on every
shape measured — 1.47× at 4096³, 3.9–4.5× for 8-token decoding, 1.45–1.66× at 512 tokens
([docs/performance.md](docs/performance.md#compared-with-llamacpp)). CUTLASS has not been measured.

Results by GPU: [benches/table.md](benches/table.md) (full GTX 1080 run: [benches/1080.md](benches/1080.md));
methodology and how to add your GPU: [docs/performance.md](docs/performance.md).

## Features

- **Exact products, FP32 accumulation.** NVFP4 × NVFP4 / INT8: exact integer sums per block of 16, one
  rounding per block. NVFP4 × BF16 / FP16: every product is exact, one FFMA per MAC. One rounding to the output
  type at the end — deterministic and bit-reproducible for a fixed kernel path, and the same bits with or
  without DP4A. Details: [docs/numerics.md](docs/numerics.md).
- **cuBLAS-style semantics.** `D = α·A·Bᵀ + bias + β·C`, column-major D, accumulation into D (e.g. an FP32
  `main_grad`), `amax` of the result, padded leading dimensions for the data and the block scales.
- **Mixed operands.** NVFP4 weights × BF16, FP16, FP32 (W4A16), INT8 (W4A8) or NVFP4 (W4A4) activations.
- **GEMV path for decoding and small batches**, picked automatically, reading each weight byte once per 8
  columns.
- **Every NVIDIA NVFP4 variant** in GEMM, GEMV, quantization and dequantization, with Transformer Engine's
  arithmetic bit for bit (see [NVIDIA NVFP4 variants](#nvidia-nvfp4-variants)); scale layout conversion
  (linear ↔ GEMM-swizzled) and the transpose of 16 × 16-tiled tensors.
- **PyTorch bindings**: functional API (`quantize`, `dequantize`, `linear` with `act="auto" / "exact" /
  "int8" / "nvfp4"`), `NVFP4Tensor`, and `NVFP4Linear` with `quantize_linear_`, which swaps the
  `nn.Linear` layers of a model.
- **Strict input checking** with readable errors in C (`nvfp4a_last_error`) and Python (`TypeError` /
  `ValueError` / `NotImplementedError`).
- **Per-architecture tuning** (`make tune`) and a parallel build (about 3 minutes for all 14 architectures
  on 20 cores).
- **Heavily tested**: bit-exact against host emulation on both NVFP4 × NVFP4 paths and for every operand
  format, the quantization against an emulation of Transformer Engine's arithmetic, exhaustive 2³² checks
  of the rounding and the decode.

## NVIDIA NVFP4 variants

| Variant | Where it comes from | Support |
|---|---|---|
| 1 × 16 blocks, E4M3 block scales, FP32 decode factor | ModelOpt checkpoints (`weight`, `weight_scale`, `weight_scale_2`) | ✅ |
| 1 × 16 blocks, FP32 amax global (current scaling) | Transformer Engine activations and gradients | ✅ |
| 16 × 16 tiles (one scale per tile) | Transformer Engine **weights (its default)** | ✅ |
| rowwise and columnwise data (`x` and `xᵀ` quantized) | Transformer Engine, for the backward GEMMs | ✅ (`nvfp4a_gemm_dual`) |
| GEMM-swizzled scales (128 × 4 tiles, padded) | cuBLASLt FP4 GEMM, Transformer Engine | ✅ |
| row-scaled: one amax per row (rowwise only) | Transformer Engine activations | ✅ |
| 4over6: map-to-4 / map-to-6 per block, MAE or MSE, E4M3 bound 448 or 256 | Transformer Engine | ✅ |
| UE5M3 block scales | Transformer Engine | ✅ |
| random Hadamard transform, stochastic rounding, 4over6 fast-math error | Transformer Engine (gradients) | ❌ |

A ModelOpt checkpoint (`weight` + `weight_scale` + `weight_scale_2`, one scale per 16 values of a row) is
the 1 × 16 representation; Transformer Engine quantizes weights with 16 × 16 tiles by default. Both — and
every combination above, A and B independently — go straight into `nvfp4a_gemm` / `nf.linear` (C flags:
[docs/c-api.md](docs/c-api.md#nvidia-nvfp4-formats); Python: `nf.quantize(..., block="16x16",
swizzled=True, ...)`, `NVFP4Tensor.from_te`).

## Supported GPUs

| Architecture | Compute capability | Status |
|---|---|---|
| Pascal (GTX 10xx, Tesla P4/P40) | sm_61, sm_62 | ✅ tested and tuned by the author (GTX 1080, sm_61) |
| Pascal (Tesla P100) | sm_60 | ✅ its code path (no DP4A) tested on the GTX 1080 (`make test-sm60`) |
| Maxwell | sm_50, sm_52, sm_53 | ⚙️ compiles, untested (same code path as sm_60) |
| Volta / Turing | sm_70, sm_72, sm_75 | ⚙️ compiles, untested, no Tensor Core path yet |
| Ampere / Orin, Ada, Hopper | sm_80, sm_86, sm_87, sm_89, sm_90 | ⚙️ compiles, untested, no Tensor Core / `cp.async` path yet |
| Blackwell | sm_100+ | ❌ excluded on purpose — native FP4, use cuBLASLt / TE |

## Installation

**Python (PyTorch ≥ 2.1, CUDA toolkit of the same major version as PyTorch's):**

```sh
git clone https://github.com/srose69/nvfp4effma && cd nvfp4effma
pip install ./binds/python --no-build-isolation          # builds for the GPUs visible right now
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ./binds/python --no-build-isolation   # or chosen ones
```

**C / C++ (CUDA 12.0+, GNU make):**

```sh
make lib                                   # out/libnvfp4effma.a for all of sm_50 … sm_90
make install PREFIX=$HOME/.local           # header + library + pkg-config (ARCHS="61 75" to narrow)
cc app.c $(pkg-config --cflags --libs nvfp4effma)
```

Build options, the parallel build and tuning: [docs/building.md](docs/building.md).

## Usage

**PyTorch**

```python
import torch, nvfp4effma as nf

w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
q = nf.quantize(w)                                  # NVFP4Tensor: data uint8 [4096, 2048], scales E4M3
                                                    # [4096, 256], global: amax (TE current scaling)
qt = nf.quantize(w, block="16x16", swizzled=True)   # Transformer Engine's weight format
x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)

y = nf.linear(x, q)                                 # act="auto": exact for <= 2 rows, int8 (W4A8) beyond
y = nf.linear(x, q, act="exact")                    # W4A16, every product exact
y = nf.linear(x, q, act="int8")                     # W4A8: x to int8 per 16 values, DP4A from sm_61
y = nf.linear(nf.quantize(x), q)                    # W4A4
w2 = nf.dequantize(q, torch.float32)
# t = nf.NVFP4Tensor.from_te(te_tensor)             # a Transformer Engine NVFP4 tensor, as it is

mlp = torch.nn.Sequential(torch.nn.Linear(4096, 11008), torch.nn.SiLU(),
                          torch.nn.Linear(11008, 4096)).cuda().bfloat16()
nf.quantize_linear_(mlp)                            # nn.Linear -> NVFP4Linear (act="auto")
y = mlp(x)                                          # NVFP4 x BF16 matmuls on NVFP4EffMA
```

More (`out` / accumulation, `amax`, `NVFP4Linear`, error behaviour):
[binds/python/README.md](binds/python/README.md).

**C**

```c
#include "nvfp4effma.h"

// Y[out, batch] (BF16) = W[out, in] (NVFP4) · X[in, batch] (BF16), column-major, both K-contiguous
nvfp4a_tensor W = {d_w, NVFP4A_NVFP4, d_w_scale, 0, d_w_scale_2, NULL};  // data, dtype, scales, ld_scale, global, amax
// (Transformer Engine tensors: .flags = NVFP4A_SCALES_SWIZZLED | NVFP4A_GLOBAL_AMAX | ..., see docs/c-api.md)
nvfp4a_tensor X = {d_x, NVFP4A_BF16, NULL, 0, NULL, NULL};
nvfp4a_tensor Y = {d_y, NVFP4A_BF16, NULL, 0, NULL, NULL};
nvfp4a_status st = nvfp4a_gemm(out, batch, in, 1.0f, &W, in, &X, in,
                               0.0f, NULL, 0, &Y, out, /*bias=*/NULL, stream);
if (st != NVFP4A_OK) fprintf(stderr, "%s\n", nvfp4a_last_error());   // e.g. "k = 96 is not a multiple of 64"
```

Full API, constraints and more examples: [docs/c-api.md](docs/c-api.md).

**Together with E4M3EffMA / E5M2EffMA**

The FP8 libraries [E4M3EffMA](https://github.com/srose69/e4m3effma) and
[E5M2EffMA](https://github.com/srose69/e5m2effma) cover E4M3 / E5M2 tensors, FP8 training (TE-style
scales, `torch._scaled_mm`, AdamW). The three have separate names everywhere — C prefixes `nvfp4a_` /
`effma_` / `e5ma_`, Python packages `nvfp4effma` / `e4m3effma` / `e5m2effma` — so they link into one binary
and import into one process. A typical split: the large MLP weights in NVFP4, the more sensitive layers in
E4M3:

```python
import torch
import nvfp4effma as nf, e4m3effma as e4             # import order does not matter

mlp  = torch.nn.Sequential(torch.nn.Linear(4096, 11008), torch.nn.SiLU(),
                           torch.nn.Linear(11008, 4096)).cuda().bfloat16()
attn = torch.nn.Linear(4096, 4096).cuda().bfloat16()

nf.quantize_linear_(mlp)                              # NVFP4 weights        (NVFP4EffMA)
e4.quantize_linear_(attn)                             # E4M3 weights, per row (E4M3EffMA)

x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
y = mlp(attn(x))
```

```c
#include "nvfp4effma.h"   /* nvfp4a_*, NVFP4A_* */
#include "e4m3effma.h"    /* effma_*,  EFFMA_*  */

// MLP on NVFP4EffMA: H [hidden, batch] = W1 (NVFP4, [hidden, in]) · X (BF16, [in, batch])
nvfp4a_tensor W1 = {d_w1, NVFP4A_NVFP4, d_w1_scale, 0, d_w1_scale_2, NULL};
nvfp4a_tensor X = {d_x, NVFP4A_BF16, NULL, 0, NULL, NULL}, H = {d_h, NVFP4A_BF16, NULL, 0, NULL, NULL};
nvfp4a_gemm(hidden, batch, in, 1.0f, &W1, in, &X, in, 0.0f, NULL, 0, &H, hidden, NULL, stream);

// attention projection on E4M3EffMA: Q [out, batch] = Wq (E4M3, [out, in]) · X (same BF16 buffer)
effma_tensor Wq = {d_wq, EFFMA_E4M3, d_wq_sinv, NULL, NULL}, X8 = {d_x, EFFMA_BF16, NULL, NULL, NULL};
effma_tensor Q = {d_q, EFFMA_BF16, NULL, NULL, NULL};
effma_gemm(EFFMA_OP_T, EFFMA_OP_N, out, batch, in, 1.0f, &Wq, in, &X8, in, 0.0f, NULL, 0, &Q, out, NULL, stream);
```

```sh
cc app.c $(pkg-config --cflags --libs nvfp4effma e4m3effma)    # both static libraries in one binary
```

## Documentation

| | |
|---|---|
| [docs/c-api.md](docs/c-api.md) | C API, data layout, constraints, error handling, examples |
| [binds/python/README.md](binds/python/README.md) | Python / PyTorch bindings, `NVFP4Tensor`, `NVFP4Linear` |
| [docs/building.md](docs/building.md) | build options, parallel build, install, tuning, project layout |
| [benches/table.md](benches/table.md) | benchmark results by GPU (one file per GPU in `benches/`) |
| [docs/performance.md](docs/performance.md) | how the benchmarks are run and measured, how to add a GPU, GTX 1080 analysis |
| [docs/numerics.md](docs/numerics.md) | exactness, accumulation order, quantization formulas, why FP32 accumulation |
| [docs/design.md](docs/design.md) | DP4A on E2M1, the int8 decode, why one block per SM, the FFMA paths, GEMV, kernel configurations |
| [docs/testing.md](docs/testing.md) | what the tests check |
| [NOTICE](NOTICE), [LICENSE](LICENSE), [LICENSE-EXCEPTION](LICENSE-EXCEPTION) | license: AGPL v3 + linking and attribution exception |

## Roadmap

Things that would make the library much faster on newer GPUs, none of which can be tested here, and the
next steps:

- [ ] **Tensor Core path for sm_75+.** Twice an E2M1 value is an exact int8, so INT8 `mma.sync` with int32
      accumulation computes the same exact block sums as DP4A (W4A4, W4A8), on Tensor Cores.
- [ ] **`cp.async` + multi-stage pipelines for sm_80+**, and `ldmatrix` for operand fetch.
- [ ] **A direct Hugging Face / ModelOpt checkpoint loader into `NVFP4Linear`.** The formats are supported
      (`NVFP4Tensor` takes `weight`, `weight_scale`, `weight_scale_2` as they are, `from_te` wraps
      Transformer Engine tensors); what is missing is the loader plumbing.
- [ ] **Faster GEMV with BF16 activations for 4 … 8 columns**, where the per-value decode, not the memory,
      is the limit ([docs/design.md](docs/design.md)).
- [ ] **128-thread variants of the heaviest kernels for Jetson TX1 / TX2** (sm_53 / sm_62 allow 32 K
      registers per block, so the DP4A kernel and the GEMV for 5 … 8 columns spill there).
- [ ] **Per-architecture tuning entries** for Maxwell, Volta, Turing, Ampere, Ada and Hopper GPUs.

## Contributing

Issues and pull requests are welcome, especially from anyone who can run the code on something newer
than a GTX 1080:

1. Run `make test` (and `pytest binds/python/tests`) on your GPU and report the result with the GPU model.
2. Run `make tune` and open a pull request with your entry in `src/nvfp4effma_tuned.h`.
3. Run `bench/run_bench.sh` and open a pull request with the `benches/<gpu>.md` file and the
   `benches/table.md` row it writes (see [docs/performance.md](docs/performance.md)).
4. Tensor Core / `cp.async` kernels for sm_75+ / sm_80+ are the most valuable contribution. They must
   pass `make test` bit-exactly, or document why a different accumulation order is needed.

If you would rather donate or lend a newer GPU (Turing, Ampere or Jetson Orin), please open an issue —
that is the fastest way for those kernels to get written and tested.

Contributions are accepted under the project's license (below).

## License

NVFP4EffMA is licensed under the [GNU AGPL v3](LICENSE) with the
[Linking and Attribution Exception](LICENSE-EXCEPTION); details and the exact attribution text are in
[NOTICE](NOTICE).

- **Use it in anything.** Programs and services that use NVFP4EffMA — link it, call it, load its Python
  bindings — can be under any license, including proprietary ones.
- **Changes to NVFP4EffMA itself stay open**: modified versions remain under the AGPL with the exception,
  with their source available, and must be marked as modified.
- **Credit is required**: every work that uses NVFP4EffMA shows the attribution from NOTICE (for example
  in its third-party notices, credits or README).
- **A request, not a condition**: if you run or publish a model with NVFP4EffMA, please mention it in the
  model card or paper. [CITATION.cff](CITATION.cff) has the citation.
