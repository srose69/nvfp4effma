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


FLT_MAX = torch.finfo(torch.float32).max


def unswizzle(sc, rows, cols):
    """GEMM-swizzled scales [roundup(rows, 128), roundup(cols / 16, 4)] -> linear [rows, cols / 16] (the
    layout's definition: 128 x 4 tiles stored as [32][4][4])."""
    pr, pc = sc.shape
    t = sc.reshape(pr // 128, pc // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(pr, pc)
    return t[:rows, : cols // 16]


def ue5m3_value(b):
    b = b.to(torch.int64)
    e, m = (b >> 3) & 31, (b & 7).double()
    return torch.where(e > 0, torch.ldexp(1 + m / 8, e - 15), torch.ldexp(m / 8, torch.full_like(e, -14)))


def scale_values(t: nf.NVFP4Tensor):
    """The block scale of every block of t (float64 [rows, cols / 16])."""
    d, s, *_ = t._flat()
    s = unswizzle(s.view(torch.uint8), d.shape[0], t.cols) if t.swizzled else s.view(torch.uint8)[: d.shape[0], : t.cols // 16]
    return ue5m3_value(s) if t.ue5m3 else s.view(torch.float8_e4m3fn).double()


def ref_value(t: nf.NVFP4Tensor):
    """float64 values of t: e2m1 * scale * decode factor (amax / (6 M) for an amax global)."""
    d = t._flat()[0]
    lo, hi = d & 15, d >> 4
    codes = torch.stack([lo, hi], -1).reshape(d.shape[0], -1)[:, : t.cols].long()
    v = GRID.to(codes.device)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    g = t.global_scale.double() if t.global_scale is not None else torch.ones(1, dtype=torch.float64, device=DEV)
    if t.global_is_amax:
        g = g / (6.0 * (t.scale_max or (114688 if t.ue5m3 else 448)))
    g = g.reshape(-1, 1) if g.numel() > 1 else g
    return (v * scale_values(t).repeat_interleave(16, -1) * g).reshape(*t.shape)


def e4m3_rn(v):  # FP32 -> E4M3 code (RNE, saturating at 448), as stored
    return v.clamp(max=448.0).to(torch.float8_e4m3fn)


def ref_quantize(x, *, g=None, tile=1, fos=0, smax=448, row_scaled=False):
    """Transformer Engine's NVFP4 quantization in float32 torch ops (independent of the extension): x [rows,
    cols] -> (data, scale codes, global used).  g: a given decode factor, else the amax global (per row with
    row_scaled)."""
    xf = x.float()
    rows, cols = xf.shape
    xb = xf.reshape(rows, cols // 16, 16)
    bmax = xb.abs().amax(-1)
    if tile == 16:
        pad = (-rows) % 16
        bm = torch.nn.functional.pad(bmax, (0, 0, 0, pad)).reshape(-1, 16, cols // 16).amax(1)
        bmax = bm.repeat_interleave(16, 0)[:rows]
    one = torch.ones(1, device=x.device)
    fmax = torch.tensor(FLT_MAX, device=x.device)
    if g is None:
        amax = xf.abs().amax(-1, keepdim=True) if row_scaled else xf.abs().amax().reshape(1, 1)
        enc = torch.minimum(torch.tensor(smax * 6.0, device=x.device) / amax, fmax)
        enc = torch.where((amax == 0) | (enc == 0), one, enc)
        dec = 1.0 / enc
        gv = amax
    else:
        dec = g.float().reshape(-1, 1)
        enc = torch.minimum(1.0 / dec, fmax)
        gv = dec
    sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)

    def candidate(sf_code):
        sf = sf_code.float()
        inv = torch.minimum(1.0 / (sf * dec), fmax)
        return e2m1_rne(xb * inv[..., None]), sf

    if not fos:
        code = e4m3_rn(torch.minimum(bmax * (enc * sixth), fmax))
        q, _ = candidate(code)
    else:
        base = (bmax / torch.full_like(bmax, 6.0)) * enc  # IEEE division (torch turns "/ scalar" into "* (1 / scalar)")
        c4, c6 = e4m3_rn(base * 1.5), e4m3_rn(base)
        q4, sf4 = candidate(c4)
        q6, sf6 = candidate(c6)
        denom = 6.0 * smax

        def err(q, sf):
            v = GRID.to(x.device).float()[(q & 7).long()] * torch.where(q & 8 > 0, -1.0, 1.0)
            val = ((v * sf[..., None]) * gv.reshape(-1, 1, 1)) / torch.tensor(denom, device=x.device)
            d = val - xb
            e = d * d if fos == 2 else d.abs()
            acc = torch.zeros_like(e[..., 0])
            for k in range(16):  # sequential, as the kernels
                acc = acc + e[..., k]
            return acc

        e4, e6 = err(q4, sf4), err(q6, sf6)
        if tile == 16:
            def tree(e):
                pad = (-rows) % 16
                r = torch.nn.functional.pad(e, (0, 0, 0, pad)).reshape(-1, 16, cols // 16)
                r = r[:, 0:8] + r[:, 8:16]
                r = r[:, 0:4] + r[:, 4:8]
                r = r[:, 0:2] + r[:, 2:4]
                return (r[:, 0] + r[:, 1]).repeat_interleave(16, 0)[:rows]
            e4, e6 = tree(e4), tree(e6)
        pick = e4 < e6
        code = torch.where(pick, c4.float(), c6.float()).to(torch.float8_e4m3fn)
        q = torch.where(pick[..., None], q4, q6)
    q = q.reshape(rows, cols)
    return q[:, 0::2] | (q[:, 1::2] << 4), code, gv


def check_close(y, ref, mag, k, out_dtype):
    """|y - ref| <= half an ulp of the output type + FP32 accumulation error (k * 2^-23 * sum |a b|)."""
    eps = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 2.0 ** -24}[out_dtype]
    err = (y.double() - ref).abs()
    bound = eps * ref.abs() + k * 2.0 ** -23 * mag + 1e-30
    assert bool((err <= bound * 1.0001).all()), f"max excess {float((err - bound).max())}"


RECIPES = [
    dict(),                                          # TE 1 x 16, amax global (current scaling)
    dict(block="16x16"),                             # TE weights
    dict(four_over_six="mae"),
    dict(four_over_six="mse", block="16x16"),
    dict(four_over_six="mae", scale_max=448),
    dict(row_scaled=True),
    dict(row_scaled=True, four_over_six="mse"),
    dict(swizzled=True),
    dict(block="16x16", swizzled=True, four_over_six="mae"),
]


def recipe_id(r):
    return ",".join(f"{k}={v}" for k, v in r.items()) or "default"


@pytest.mark.parametrize("recipe", RECIPES, ids=recipe_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantize_matches_transformer_engine(recipe, dtype):
    x = rand((67, 320), dtype, 3.0)
    x[3, :16] = 0          # zero block
    x[5, 16:32] *= 1e-30   # scale underflows to 0
    x[7] = 0               # a zero row (amax 0 with row scaling)
    q = nf.quantize(x, **recipe)
    fos = {"mae": 1, "mse": 2}.get(recipe.get("four_over_six"), 0)
    smax = recipe.get("scale_max", 256 if fos else 448)
    data, code, gv = ref_quantize(x, tile=16 if recipe.get("block") == "16x16" else 1, fos=fos, smax=smax,
                                  row_scaled=recipe.get("row_scaled", False))
    assert torch.equal(q._flat()[0], data)
    assert torch.equal(scale_values(q), code.double()), "block scales differ"
    assert q.global_is_amax and torch.equal(q.global_scale, gv.reshape(-1))


def test_quantize_given_global_and_ue5m3():
    x = rand((40, 256), torch.bfloat16, 2.0)
    g = torch.tensor([0.004], device=DEV)
    amax = torch.zeros(1, device=DEV)
    q = nf.quantize(x, g, amax=amax)
    data, code, _ = ref_quantize(x, g=g)
    assert torch.equal(q.data, data) and torch.equal(scale_values(q), code.double()) and not q.global_is_amax
    assert float(amax) == float(x.float().abs().max())
    u = nf.quantize(x, scale_dtype="ue5m3")
    assert u.ue5m3 and u.scales.dtype == torch.uint8
    err = (ref_value(u) - x.double()).abs().max()
    assert float(err) <= 0.5 * float(x.abs().max())  # 4-bit values: coarse, but the right ones


@pytest.mark.parametrize("block", ["1x16", "16x16"])
def test_columnwise_is_the_quantized_transpose(block):
    x = rand((96, 160), torch.bfloat16, 1.5, seed=3)
    q = nf.quantize(x, block=block, columnwise=True)
    c = q.columnwise
    assert c.shape == (160, 96) and c.global_is_amax and torch.equal(c.global_scale, q.global_scale)
    ref = nf.quantize(x.t().contiguous(), block=block)
    assert torch.equal(c._flat()[0][:, :48], ref.data) and torch.equal(scale_values(c), scale_values(ref))
    if block == "16x16":  # the transpose of the 2-D quantization is the columnwise one, bit for bit
        t = q.transpose_2d()
        assert torch.equal(t.data, c._flat()[0][:, :48]) and torch.equal(scale_values(t), scale_values(c))


def test_scale_layouts_and_dequantize():
    x = rand((200, 192), torch.float32, 4.0, seed=4)
    for recipe in RECIPES + [dict(scale_dtype="ue5m3", swizzled=True)]:
        q = nf.quantize(x, **recipe)
        o = q.with_scale_layout(not q.swizzled)
        assert o.swizzled != q.swizzled and torch.equal(scale_values(q), scale_values(o))
        for t in (q, o):
            y = nf.dequantize(t, torch.float32)
            assert torch.allclose(y.double(), ref_value(t), rtol=2 ** -22, atol=0), recipe_id(recipe)


@pytest.mark.parametrize("recipe", RECIPES[:6] + [dict(scale_dtype="ue5m3")], ids=recipe_id)
@pytest.mark.parametrize("T", [1, 5, 40, 260])
def test_linear_every_weight_format(recipe, T):
    N, K = 130, 256
    w = nf.quantize(rand((N, K), seed=14), **recipe)
    x = rand((T, K), torch.bfloat16, seed=15)
    wv = ref_value(w)
    y = nf.linear(x, w, act="exact", out_dtype=torch.float32)
    check_close(y, x.double() @ wv.t(), x.double().abs() @ wv.abs().t(), K, torch.float32)
    xq = nf.quantize(x, row_scaled=T > 1, swizzled=True)  # W4A4 with a row-scaled, swizzled activation
    xv = ref_value(xq)
    y = nf.linear(xq, w, out_dtype=torch.float32)
    check_close(y, xv @ wv.t(), xv.abs() @ wv.abs().t(), K // 16, torch.float32)


def test_backward_gemms_with_columnwise_halves():
    """dgrad dX = dY W and wgrad dW = dY^T X of a linear layer from the columnwise representations."""
    Tk, I, O = 128, 192, 64
    w = nf.quantize(rand((O, I), seed=16), block="16x16", columnwise=True)
    x = nf.quantize(rand((Tk, I), seed=17), columnwise=True)
    dy = nf.quantize(rand((Tk, O), seed=18), columnwise=True)
    wv, dyv = ref_value(w), ref_value(dy)
    dx = nf.linear(dy, w.columnwise, out_dtype=torch.float32)
    wcv = ref_value(w.columnwise)
    assert torch.equal(wcv, wv.t())  # 16 x 16 tiles: the columnwise weight is exactly W^T
    check_close(dx, dyv @ wcv.t(), dyv.abs() @ wcv.abs().t(), O // 16, torch.float32)
    dw = nf.linear(dy.columnwise, x.columnwise, out_dtype=torch.float32)
    a, b = ref_value(dy.columnwise), ref_value(x.columnwise)
    check_close(dw, a @ b.t(), a.abs() @ b.abs().t(), Tk // 16, torch.float32)


def test_from_transformer_engine_tensor():
    """A Transformer Engine NVFP4 tensor (its attribute names; TE itself is not needed) is used as it is."""
    from types import SimpleNamespace
    x = rand((144, 128), torch.bfloat16, seed=19)
    q = nf.quantize(x, swizzled=True, columnwise=True, four_over_six="mae")
    te = SimpleNamespace(_rowwise_data=q.data, _rowwise_scale_inv=q.scales.view(torch.uint8), _amax_rowwise=q.global_scale,
                         _columnwise_data=q.columnwise.data, _columnwise_scale_inv=q.columnwise.scales.view(torch.uint8),
                         _amax_columnwise=q.columnwise.global_scale, _with_gemm_swizzled_scales=True,
                         _row_scaled_nvfp4=False, _nvfp4_e4m3_max=256, _scale_dtype="DType.kFloat8E4M3")
    t = nf.NVFP4Tensor.from_te(te)
    assert t.swizzled and t.global_is_amax and t.scale_max == 256 and t.columnwise is not None
    assert torch.equal(nf.dequantize(t, torch.float32), nf.dequantize(q, torch.float32))
    w = nf.quantize(rand((64, 128), seed=20))
    assert torch.equal(nf.linear(t, w, out_dtype=torch.float32), nf.linear(q, w, out_dtype=torch.float32))


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
    with pytest.raises(ValueError):
        nf.quantize(rand((40, 32)), columnwise=True)  # columnwise blocks need rows % 16 == 0
    with pytest.raises(ValueError):
        nf.quantize(rand((32, 32)), row_scaled=True, columnwise=True)
    with pytest.raises(ValueError):
        nf.quantize(rand((32, 32)), block="4x4")
    with pytest.raises(TypeError):
        nf.NVFP4Tensor(w.data.float(), w.scales, w.global_scale)
    with pytest.raises(ValueError):
        nf.NVFP4Tensor(w.data, w.scales[:, :4], w.global_scale)
    with pytest.raises(ValueError):
        x = rand((3, 128))
        nf.linear(x, w, out=x.new_empty(3, 64), beta=1.0, out_dtype=torch.bfloat16)  # dtype mismatch
