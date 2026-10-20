# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Tests of the PyTorch bindings: quantization bit for bit against an independent reference of the
documented formulas, linear layers against a float64 reference, the module, and rejection of bad input.

Bit-exactness of the kernels themselves is covered by the C tests (make test); here the bound for
matmuls is the output rounding plus the FP32 accumulation error.
"""
import pytest
import torch

import nvfp4effma as nf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def rand(shape, dtype=torch.float32, scale=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g) * scale
    return x.to(DEV).to(dtype)


def e2m1_rne(y):
    """FP32 -> E2M1 code (sign in bit 3), nearest with ties to the even code, saturating; float64 search."""
    a = y.double().abs().clamp(max=6.0)
    d = (a[..., None] - GRID.to(a.device)).abs()
    best = d.min(dim=-1, keepdim=True).values
    cand = (d == best)
    # among ties prefer the even code
    code = torch.where(cand[..., ::2].any(-1), cand[..., ::2].float().argmax(-1) * 2, cand.float().argmax(-1))
    return code.to(torch.uint8) | (torch.signbit(y).to(torch.uint8) << 3)


def ref_quantize(x, g):
    """The documented formulas: sb = E4M3(amax / (6 g)), q = E2M1(x / (sb g)); q = signed 0 where sb = 0."""
    xb = x.float().reshape(*x.shape[:-1], -1, 16)
    bmax = xb.abs().amax(-1)
    sb = (bmax / (6.0 * g)).clamp(max=448.0).to(torch.float8_e4m3fn)
    d = sb.float() * g
    y = xb / d[..., None]
    q = torch.where(sb.float()[..., None] > 0, e2m1_rne(y), torch.signbit(xb).to(torch.uint8) << 3)
    q = q.reshape(*x.shape)
    data = q[..., 0::2] | (q[..., 1::2] << 4)
    return data, sb


def ref_value(t: nf.NVFP4Tensor):
    lo, hi = t.data & 15, t.data >> 4
    codes = torch.stack([lo, hi], -1).reshape(*t.data.shape[:-1], -1).long()
    v = GRID.to(codes.device)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    s = t.scales.float().double().repeat_interleave(16, -1)
    return v * s * float(t.global_scale)


def check_close(y, ref, mag, k, out_dtype):
    """|y - ref| <= half an ulp of the output type + FP32 accumulation error (k * 2^-23 * sum |a b|)."""
    eps = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 2.0 ** -24}[out_dtype]
    err = (y.double() - ref).abs()
    bound = eps * ref.abs() + k * 2.0 ** -23 * mag + 1e-30
    assert bool((err <= bound * 1.0001).all()), f"max excess {float((err - bound).max())}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantize_bit_exact(dtype):
    x = rand((67, 320), dtype, 3.0)
    x[3, :16] = 0          # zero block
    x[5, 16:32] *= 1e-30   # scale underflows to 0
    g = torch.tensor([0.01], device=DEV)
    amax = torch.zeros(1, device=DEV)
    q = nf.quantize(x, g, amax=amax)
    data, sb = ref_quantize(x, g)
    assert torch.equal(q.scales.view(torch.uint8), sb.view(torch.uint8))
    assert torch.equal(q.data, data)
    assert float(amax) == float(x.float().abs().max())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dequantize_bit_exact(dtype):
    q = nf.quantize(rand((5, 7, 64), torch.float32, 2.0))  # 3-D: flattened to rows
    y = nf.dequantize(q, dtype)
    assert y.shape == (5, 7, 64) and y.dtype == dtype
    ref = ((ref_value(q) / float(q.global_scale)).float() * q.global_scale).to(dtype)
    assert torch.equal(y, ref)


def test_default_global_scale_uses_the_e4m3_range():
    x = rand((32, 256), torch.float32, 5.0)
    q = nf.quantize(x)
    assert torch.allclose(q.global_scale, x.abs().max() / (6 * 448))
    assert float(q.scales.float().max()) == 448.0


@pytest.mark.parametrize("T", [1, 3, 8, 9, 200])
@pytest.mark.parametrize("xdtype", [torch.bfloat16, torch.float16, torch.float32])
def test_linear_dense_x(T, xdtype):
    N, K = 192, 320
    w = nf.quantize(rand((N, K), seed=1))
    x = rand((T, K), xdtype, seed=2)
    bias = rand((N,), xdtype, seed=3)
    y = nf.linear(x, w, bias, act="exact")
    assert y.shape == (T, N) and y.dtype == xdtype
    wv = ref_value(w)
    ref = x.double() @ wv.t() + bias.double()
    check_close(y, ref, x.double().abs() @ wv.abs().t(), K, xdtype)


@pytest.mark.parametrize("T", [1, 5, 8, 33, 260])
def test_linear_nvfp4_x(T):
    N, K = 130, 256
    w = nf.quantize(rand((N, K), seed=4))
    x = nf.quantize(rand((T, K), seed=5))
    y = nf.linear(x, w, out_dtype=torch.float32)
    assert y.shape == (T, N) and y.dtype == torch.float32
    xv, wv = ref_value(x), ref_value(w)
    check_close(y, xv @ wv.t(), xv.abs() @ wv.abs().t(), K // 16, torch.float32)


@pytest.mark.parametrize("T", [1, 8, 20, 200])
def test_linear_int8_activations(T):
    """W4A8: x quantized to int8 with an FP32 scale per 16 values, exact integer block sums."""
    N, K = 130, 256
    w = nf.quantize(rand((N, K), seed=10))
    x = rand((T, K), torch.bfloat16, seed=11)
    q, s = nf._C.quantize_int8(x)
    assert q.dtype == torch.int8 and s.shape == (T, K // 16)
    xb = x.float().reshape(T, -1, 16)
    bmax = xb.abs().amax(-1)
    assert torch.equal(s, bmax / torch.full_like(bmax, 127.0))  # tensor / tensor: a correctly rounded division
    assert torch.equal(q.float(), torch.where(s[..., None] > 0, torch.round(xb / s[..., None]), torch.zeros_like(xb)).reshape(T, K))
    xv, wv = q.double() * s.double().repeat_interleave(16, -1), ref_value(w)
    y = nf.linear(x, w, act="int8", out_dtype=torch.float32)
    check_close(y, xv @ wv.t(), xv.abs() @ wv.abs().t(), K // 16, torch.float32)
    assert nf.linear(x, w, act="int8").dtype == torch.bfloat16
    m = nf.NVFP4Linear.from_linear(torch.nn.Linear(K, N).cuda().bfloat16(), act="int8")
    assert m(x).shape == (T, N)


def test_linear_auto_activations():
    """act="auto": the exact path for up to 2 rows, int8 activations beyond -- bit for bit."""
    w = nf.quantize(rand((96, 256), seed=12))
    for T in (1, 2, 3, 40):
        x = rand((T, 256), torch.bfloat16, seed=13 + T)
        want = "exact" if T <= nf.AUTO_EXACT_MAX_ROWS else "int8"
        assert torch.equal(nf.linear(x, w), nf.linear(x, w, act=want))
    with pytest.raises(ValueError):
        nf.linear(x, w, act="fp8")


def test_linear_batched_input_and_accumulate():
    w = nf.quantize(rand((64, 128), seed=6))
    x = rand((2, 3, 128), torch.bfloat16, seed=7)
    y = nf.linear(x, w)
    assert y.shape == (2, 3, 64)
    acc = torch.ones(6, 64, device=DEV)
    nf.linear(x.reshape(6, 128), w, out=acc, beta=1.0, act="exact")
    ref = x.reshape(6, 128).double() @ ref_value(w).t() + 1
    check_close(acc, ref, x.reshape(6, 128).double().abs() @ ref_value(w).abs().t() + 1, 128, torch.float32)


def test_strided_inputs_are_copied():
    w = nf.quantize(rand((64, 256), seed=8))
    big = rand((10, 320), torch.bfloat16, seed=9)
    x = big[:, 32:288]  # row stride 320, offset: not usable as is
    assert torch.equal(nf.linear(x, w, act="exact"), nf.linear(x.contiguous(), w, act="exact"))
    assert torch.equal(nf.linear(x, w), nf.linear(x.contiguous(), w))


def test_nvfp4_linear_module():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10),
                                torch.nn.Linear(10, 64)).to(DEV).to(torch.bfloat16)
    x = torch.randn(4, 128, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = model(x).float()
        nf.quantize_linear_(model)
        y = model(x).float()
    assert isinstance(model[0], nf.NVFP4Linear) and isinstance(model[2], nf.NVFP4Linear)
    assert isinstance(model[3], torch.nn.Linear)  # in_features 10: not a multiple of 64, kept
    assert set(model[0].state_dict()) == {"weight", "weight_scale", "weight_scale_2", "bias"}
    assert float((y - ref).abs().max()) < 0.25 * float(ref.abs().max()) + 0.05  # 4-bit weights: coarse


def test_rejects_bad_input():
    w = nf.quantize(rand((64, 128)))
    with pytest.raises(TypeError):
        nf.linear(rand((2, 128), torch.float64), w)
    with pytest.raises(ValueError):
        nf.linear(rand((2, 96)), w)  # K mismatch
    with pytest.raises(ValueError):
        nf.linear(rand((2, 128)).cpu(), w)
    with pytest.raises(ValueError):
        nf.linear(rand((2, 96)), nf.quantize(rand((64, 96))))  # K not a multiple of 64
    with pytest.raises(ValueError):
        nf.quantize(rand((4, 24)))  # not a multiple of 16
    with pytest.raises(TypeError):
        nf.NVFP4Tensor(w.data.float(), w.scales, w.global_scale)
    with pytest.raises(ValueError):
        nf.NVFP4Tensor(w.data, w.scales[:, :4], w.global_scale)
    with pytest.raises(ValueError):
        x = rand((3, 128))
        nf.linear(x, w, out=x.new_empty(3, 64), beta=1.0, out_dtype=torch.bfloat16)  # dtype mismatch
