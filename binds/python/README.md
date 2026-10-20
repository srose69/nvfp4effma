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
q = nf.quantize(w)                                   # NVFP4Tensor
x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)

y = nf.linear(x, q)                                  # x @ W.T, BF16 [8, 4096]; act="auto": int8 activations here
y = nf.linear(x, q, act="exact")                     # W4A16: every product exact
y = nf.linear(x, q, bias, out_dtype=torch.float32)
y = nf.linear(x, q, act="int8")                      # W4A8: x to int8 per 16 values, exact block sums (DP4A)
y = nf.linear(nf.quantize(x), q)                     # W4A4
acc = torch.zeros(8, 4096, device="cuda")
nf.linear(x, q, out=acc, beta=1.0, act="exact")      # acc += x @ W.T

model = torch.nn.Linear(4096, 4096).cuda().bfloat16()
nf.quantize_linear_(model)                           # or nf.NVFP4Linear.from_linear(layer, act=...)
```

| | |
|---|---|
| `quantize(x, global_scale=None, *, amax=None)` | BF16 / FP16 / FP32 `[..., K]` (K a multiple of 16) → `NVFP4Tensor`; `global_scale` defaults to `amax(|x|) / (6·448)`; `amax` (float32 `[1]`) receives `max |x|` |
| `dequantize(t, dtype=torch.bfloat16)` | `(e2m1 · scale) · global_scale`, rounded once |
| `linear(x, weight, bias=None, *, act, out_dtype, out, alpha, beta, amax_out)` | `alpha·x @ weight.T + bias + beta·out`; `weight` an `NVFP4Tensor` `[N, K]` (K a multiple of 64); `x` a dense `[..., K]` tensor or a 2-D `NVFP4Tensor`; `act`: `"exact"` (W4A16), `"int8"` (W4A8, x quantized to int8 per 16 values first), `"nvfp4"` (W4A4) or `"auto"` (default: `"exact"` up to `AUTO_EXACT_MAX_ROWS` = 2 rows, where both run at memory speed, `"int8"` beyond, where it is twice as fast); output BF16 / FP16 / FP32 (default: the dtype of a dense `x`, else BF16) |
| `NVFP4Tensor(data, scales, global_scale)` | `data` uint8 `[..., K/2]` (the even column in the low nibble), `scales` float8_e4m3fn `[..., K/16]`, `global_scale` float32 `[1]`; `.shape`, `.dequantize()`, `NVFP4Tensor.from_float(x)` |
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
