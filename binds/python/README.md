# nvfp4effma (PyTorch bindings)

NVFP4 linear layers, quantization and dequantization for NVIDIA GPUs **without** native FP4
(sm_50 … sm_90: Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper), on ordinary `torch` tensors.

## Install

Requires PyTorch ≥ 2.1 with CUDA, a CUDA toolkit (`nvcc`) of the same major version, and GNU make.

```sh
git clone https://github.com/srose69/nvfp4effma && cd nvfp4effma
pip install ./binds/python --no-build-isolation
```

The library is compiled for the GPUs visible at build time. To build for other GPUs, for example a wheel
for several machines:

```sh
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" pip install ./binds/python --no-build-isolation
NVFP4EFFMA_ARCHS="50 52 53 60 61 62 70 72 75 80 86 87 89 90" pip wheel ./binds/python --no-build-isolation
```

Other build variables: `CUDA_HOME`, `NVFP4EFFMA_CCBIN` (host compiler for nvcc, e.g. `g++-12` with
CUDA 12.0–12.3), `MAX_JOBS`.

## Use

```python
import torch, nvfp4effma as nf

w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
q = nf.quantize(w)                                   # NVFP4Tensor, Transformer Engine's current scaling
x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)

y = nf.linear(x, q)                                  # x @ W.T, BF16 [8, 4096]; act="auto": int8 activations here
y = nf.linear(x, q, act="exact")                     # W4A16: every product exact
y = nf.linear(x, q, bias, out_dtype=torch.float32)
y = nf.linear(x, q, act="int8")                      # W4A8: x to int8 per 16 values, exact block sums (DP4A)
xq = nf.quantize_int8(x)                             # the same int8 activations, kept: Int8Tensor
y = nf.linear(xq, q)                                 # == nf.linear(x, q, act="int8"), bit for bit
y = nf.linear(nf.quantize(x), q)                     # W4A4
tw = nf.TCWeight(q, "exact")                         # tensor cores (sm_75 .. sm_89): the weight's image, once
y = nf.linear(x, tw)                                 # tensor-core GEMV (<= 8 rows) or GEMM, W4A16
acc = torch.zeros(8, 4096, device="cuda")
nf.linear(x, q, out=acc, beta=1.0, act="exact")      # acc += x @ W.T

model = torch.nn.Linear(4096, 4096).cuda().bfloat16()
nf.quantize_linear_(model)                           # or nf.NVFP4Linear.from_linear(layer, act=...)

# Transformer Engine's formats, as TE produces them (bit for bit) or straight from a TE tensor
qw = nf.quantize(w, block="16x16", swizzled=True)    # TE weights: 16 x 16 tiles, GEMM-swizzled scales
qx = nf.quantize(x, row_scaled=True, swizzled=True)  # row-scaled activations
q6 = nf.quantize(w, four_over_six="mse")             # 4over6 (E4M3 bound 256)
qc = nf.quantize(x, columnwise=True)                 # + qc.columnwise = the quantization of x^T
y = nf.linear(qx, qw)                                # any formats on either side
t = nf.NVFP4Tensor.from_te(te_tensor)                # wraps a TE NVFP4Tensor / NVFP4TensorStorage, no copy
```

On Pascal, use W4A8 (`act="auto"` / `"int8"`) for prefill speed: the exact W4A16 path is bound by the FP32
FFMA rate there and runs at SGEMM speed.

| | |
|---|---|
| `quantize(x, global_scale=None, *, amax=None, block="1x16", four_over_six=None, scale_max=None, row_scaled=False, columnwise=False, swizzled=False, scale_dtype="e4m3")` | BF16 / FP16 / FP32 `[..., K]` (K a multiple of 16) → `NVFP4Tensor`, Transformer Engine's arithmetic. Global: by default the amax of `x` (of each row with `row_scaled`) is computed and stored (TE current scaling); a given `global_scale` is a decode factor (ModelOpt's `weight_scale_2`). `block` `"1x16"` / `"16x16"`; `four_over_six` `"mae"` / `"mse"`; `columnwise` (rows a multiple of 16) also quantizes `x^T` into `.columnwise`; `swizzled` scales in the GEMM layout; `scale_dtype` `"e4m3"` / `"ue5m3"`; `amax` (float32 `[1]`) receives `max |x|` |
| `quantize_int8(x)` | BF16 / FP16 / FP32 `[..., K]` (K a multiple of 16) → `Int8Tensor`: `scale = max \|x\| / 127` per 16 values, `data = round(x / scale)` — the activations of `act="int8"` |
| `dequantize(t, dtype=torch.bfloat16)` | `(e2m1 · scale) · global`, or TE's `e2m1 · ((scale · amax) / (6 · scale_max))`, rounded once |
| `linear(x, weight, bias=None, *, act, out_dtype, out, alpha, beta, amax_out)` | `alpha·x @ weight.T + bias + beta·out`; `weight` an `NVFP4Tensor` `[N, K]` (K a multiple of 64); `x` a dense `[..., K]` tensor, a 2-D `NVFP4Tensor` or an `Int8Tensor`; `act`: `"exact"` (W4A16), `"int8"` (W4A8, x quantized to int8 per 16 values first), `"nvfp4"` (W4A4) or `"auto"` (default: `"exact"` up to `AUTO_EXACT_MAX_ROWS` = 2 rows, where both run at memory speed, `"int8"` beyond, where it is twice as fast); output BF16 / FP16 / FP32 (default: the dtype of a dense `x` or the source dtype of an `Int8Tensor`, else BF16) |
| `NVFP4Tensor(data, scales, global_scale=None, *, amax=None, scale_max=0, swizzled=False, ue5m3=False, cols=None, columnwise=None, block=None)` | `data` uint8 `[..., K/2]` (the even column in the low nibble), `scales` float8_e4m3fn `[..., K/16]` (linear; padded 2-D buffers as TE allocates them are fine) or the swizzled `[roundup(rows, 128), roundup(K/16, 4)]` buffer, UE5M3 as uint8; `global_scale` (decode factor) or `amax` (TE), float32 `[1]` or one per row; `.shape`, `.dequantize()`, `.with_scale_layout(swizzled)`, `.transpose_2d()` (16 × 16 tiles), `NVFP4Tensor.from_float(x, **recipe)`, `NVFP4Tensor.from_te(t)` |
| `Int8Tensor(data, scales, dtype=torch.bfloat16)` | `data` int8 `[..., K]`, `scales` float32 `[..., K/16]`, `dtype` the source dtype (default output dtype of `linear`); `.shape`, `.dequantize(dtype=None)`, `Int8Tensor.from_float(x)` |
| `get_include()` | `[include, src]` directories of `nvfp4effma.h` and the device-side codecs `nvfp4effma.cuh`, for CUDA extensions built on this package; code compiled from them matches only this version |
| `TCWeight(weight, act="exact")` | An `NVFP4Tensor` weight `[N, K]` with its tensor-core image (sm_75 … sm_89; as large as the weight, built once): `linear(x, tw)` runs the tensor-core GEMV up to 8 rows of `x` and the GEMM beyond. `act` fixes the activations: `"exact"` bfloat16 `x` (W4A16, sm_80+), `"int8"` / `"nvfp4"` (W4A8 / W4A4: a dense `x` quantized first, or an `Int8Tensor` / `NVFP4Tensor`). The sums are FP32 in the tensor cores' order: the values of the other paths within the FP32 accumulation bound, not bit for bit |
| `NVFP4Linear(in_features, out_features, bias=True, act="auto")` | `nn.Module` with buffers `weight` / `weight_scale` / `weight_scale_2` (the names of NVFP4 checkpoints); `NVFP4Linear.from_linear(layer, act="auto")` |
| `quantize_linear_(module, *, act="auto")` | replaces every `nn.Linear` with `in_features % 64 == 0` by an `NVFP4Linear`, in place |

Few rows of `x` run a GEMV kernel that reads each weight byte once per 8 rows. NVFP4 × NVFP4 / INT8 sums
every block of 16 exactly and rounds once per block; with a dense `x` every product is exact (BF16 / FP16)
and accumulated in FP32 — [docs/numerics.md](../../docs/numerics.md) has the exact order.

Wrong dtypes, devices, shapes or an output that overlaps an input raise `TypeError` / `ValueError` before
anything runs; operands in a layout the kernels cannot read directly (strided views, unaligned slices) are
copied once; GPUs with native FP4 (sm_100+) raise `NotImplementedError`; library or CUDA failures raise
`RuntimeError` with the library's description. Inference only: there is no autograd support.

## Tests

```sh
pytest binds/python/tests
```

Quantization bit for bit against an independent PyTorch reference, `linear` against float64, the module
and the error behaviour. The kernels themselves are checked bit for bit by the C tests (`make test`).
