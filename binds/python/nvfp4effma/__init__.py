# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""NVFP4EffMA: NVFP4 linear layers, GEMM / GEMV and quantization for NVIDIA GPUs without native FP4 (sm_50..sm_90).
E2M1 values (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}), one 8-bit block scale per 16 values, a float32 global; ``NVFP4Tensor.from_te`` wraps a TE tensor.
Errors: ``TypeError`` (dtype), ``ValueError`` (device, shape, K % 64, overlapping output), ``NotImplementedError`` (native-FP4 GPU, < sm_50), ``RuntimeError``."""
import math
import os
from typing import List, Optional, Union

import torch

from . import _C, _ops
from .tensor import Int8Tensor, NVFP4Linear, NVFP4Tensor, TCWeight, quantize_linear_

__version__ = _C.__version__
__all__ = ["quantize", "quantize_int8", "dequantize", "linear", "grouped_linear", "NVFP4Tensor", "Int8Tensor", "NVFP4Linear", "TCWeight",
           "quantize_linear_", "get_include", "__version__"]

# act="auto": up to this many rows of x the exact W4A16 path runs; beyond it the activations are quantized to int8 first (W4A8, DP4A)
AUTO_EXACT_MAX_ROWS = 2


def quantize(x: torch.Tensor, global_scale: Optional[torch.Tensor] = None, *, amax: Optional[torch.Tensor] = None,
             block: str = "1x16", four_over_six: Optional[str] = None, scale_max: Optional[int] = None,
             row_scaled: bool = False, columnwise: bool = False, swizzled: bool = False,
             scale_dtype: str = "e4m3") -> NVFP4Tensor:
    """Quantize ``x`` (bfloat16 / float16 / float32 CUDA tensor ``[..., K]``, K a multiple of 16, finite) to NVFP4 with
    the arithmetic of Transformer Engine (bit for bit); ``columnwise`` also quantizes ``x^T`` (``result.columnwise``);
    ``swizzled``: GEMM-layout scales; ``scale_dtype``: ``"e4m3"`` or ``"ue5m3"``; ``amax`` (float32 ``[1]``) gets ``max |x|``."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    # block: "1x16" (activations, gradients, ModelOpt) or "16x16" (TE weights: one scale per 16 x 16 tile, so the rowwise
    # and columnwise quantizations are transposes of each other)
    if block not in ("1x16", "16x16"):
        raise ValueError(f"block must be '1x16' or '16x16', got {block!r}")
    # four_over_six: None, "mae" or "mse", TE's 4over6 block scale selection (scale_max 256 unless given)
    fos = {None: 0, "mae": 1, "mse": 2}.get(four_over_six.lower() if isinstance(four_over_six, str) else four_over_six)
    if fos is None:
        raise ValueError(f"four_over_six must be None, 'mae' or 'mse', got {four_over_six!r}")
    if scale_dtype not in ("e4m3", "ue5m3"):
        raise ValueError(f"scale_dtype must be 'e4m3' or 'ue5m3', got {scale_dtype!r}")
    smax = scale_max if scale_max is not None else (256 if fos else 0)
    x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
    rows = x2.shape[0]
    if columnwise and row_scaled:
        raise ValueError("row-scaled tensors have no columnwise representation")
    if columnwise and rows % 16:
        raise ValueError(f"a columnwise representation needs rows % 16 == 0 (got {rows}), as in Transformer Engine")
    # global: by default TE's current scaling -- the amax of x (of each row with row_scaled) is computed and stored, decode factor
    # amax / (6 * scale_max); a given global_scale (float32, one element or one per row) is the decode factor instead
    te = global_scale is None
    flags = ((_C.SCALES_SWIZZLED if swizzled else 0) | (_C.SCALES_UE5M3 if scale_dtype == "ue5m3" else 0) |
             (_C.GLOBAL_AMAX if te else 0))
    if te:
        g_row = torch.zeros(rows if row_scaled else 1, dtype=torch.float32, device=x.device)
        g_col = torch.zeros(1, dtype=torch.float32, device=x.device) if columnwise else None
        if row_scaled and rows > 1:
            flags |= _C.GLOBAL_PER_ROW
    else:
        g_row = global_scale.reshape(-1).float().contiguous()
        if g_row.numel() > 1:
            if g_row.numel() != rows:
                raise ValueError(f"global_scale has {g_row.numel()} elements: 1 or one per row ({rows})")
            flags |= _C.GLOBAL_PER_ROW
            if columnwise:
                raise ValueError("per-row globals have no columnwise representation")
        g_col = g_row if columnwise else None
    dr, sr, dc, sc, g_row, g_col, amax_x = _ops.quantize_ex(x2, True, columnwise, g_row, g_col, flags, smax, 16 if block == "16x16" else 1,
                                                            fos, te, amax is not None)
    if amax is not None:  # max-reduced into the given amax, as the kernels do
        torch.maximum(amax, amax_x, out=amax)
    shape = tuple(x.shape)
    ue = scale_dtype == "ue5m3"
    blk = (16, 16) if block == "16x16" else (1, 16)

    def wrap(d, s, g, per_row_shape, cols):
        kw = dict(scale_max=smax, swizzled=swizzled, ue5m3=ue, cols=cols, block=blk)
        if te:
            return NVFP4Tensor(d, s, amax=g, **kw)
        return NVFP4Tensor(d, s, g, **kw)

    if swizzled:
        out = wrap(dr.reshape(*shape[:-1], shape[-1] // 2), sr, g_row, None, shape[-1])
    else:
        out = wrap(dr.reshape(*shape[:-1], shape[-1] // 2), sr.reshape(*shape[:-1], shape[-1] // 16), g_row, None, shape[-1])
    if columnwise:
        out.columnwise = wrap(dc, sc, g_col, None, rows)
    return out


def quantize_int8(x: torch.Tensor) -> Int8Tensor:
    """Quantize ``x`` (bfloat16 / float16 / float32 CUDA tensor ``[..., K]``, K a multiple of 16) to int8 with an
    FP32 scale per 16 values: ``scale = max |x| / 127`` over the block, ``data = round(x / scale)`` -- the
    activations of ``linear(..., act="int8")``, which takes the result as ``x`` directly."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    q, s = _ops.quantize_int8(x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x)
    return Int8Tensor(q.reshape(x.shape), s.reshape(*x.shape[:-1], x.shape[-1] // 16), x.dtype)


def get_include() -> List[str]:
    """Directories with ``nvfp4effma.h`` and the device codecs ``nvfp4effma.cuh`` (E2M1 / E4M3 decode, RNE encode, scale addressing)
    for CUDA extensions: ``-I`` both, ``#include <nvfp4effma.cuh>``; the compiled code is only consistent with this package version."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, "include"), os.path.join(here, "src")]


def dequantize(x: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The values of ``x`` rounded once to ``dtype`` (bfloat16, float16 or float32): ``(e2m1 * scale) *
    global_scale``, or Transformer Engine's ``e2m1 * ((scale * amax) / (6 * scale_max))``."""
    if not isinstance(x, NVFP4Tensor):
        raise TypeError(f"expected an NVFP4Tensor, got {type(x).__name__}")
    d, s, g, flags, smax, cols = x._flat()
    return _ops.dequantize(d, s.view(torch.uint8), g, flags, smax, cols, dtype).reshape(x.shape)


def linear(x: Union[torch.Tensor, NVFP4Tensor, Int8Tensor], weight: NVFP4Tensor, bias: Optional[torch.Tensor] = None, *,
           act: str = "auto", out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None,
           alpha: float = 1.0, beta: float = 0.0, amax_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``alpha * x @ weight.T + bias + beta * out`` like ``torch.nn.functional.linear``; ``weight``: ``NVFP4Tensor`` ``[N, K]``
    (K a multiple of 64) or ``TCWeight``; ``x``: dense ``[..., K]``, a 2-D ``NVFP4Tensor`` ``[T, K]`` or an ``Int8Tensor``;
    ``amax_out`` (float32 ``[1]``) gets max |result|; ``act`` and ``out_dtype`` are described beside their use."""
    # out_dtype: bfloat16, float16 or float32 (default: dtype of a dense x, source dtype of an Int8Tensor, else bfloat16);
    # beta != 0 accumulates into out, which must then be given [T, N]
    if isinstance(weight, TCWeight):
        return _linear_tc(x, weight, bias, act, out_dtype, out, alpha, beta, amax_out)
    # the backward GEMMs of a TE linear layer use the columnwise halves: dX = linear(dY, W.columnwise),
    # dW = linear(dY_q.columnwise, X_q.columnwise) for NVFP4-quantized dY_q / X_q
    if not isinstance(weight, NVFP4Tensor):
        raise TypeError(f"weight must be an NVFP4Tensor, got {type(weight).__name__}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [out_features, in_features], got shape {weight.shape}")
    N, K = weight.shape
    wd, ws, wg, wf, wm, _ = weight._flat()
    ws = ws.view(torch.uint8)
    if isinstance(x, NVFP4Tensor):  # W4A4: exact integer sums per block of 16 (DP4A from sm_61)
        if x.dim() != 2 or x.shape[1] != K:
            raise ValueError(f"an NVFP4 x must have shape [T, {K}], got {x.shape}")
        xd, xs, xg, xf, xm, _ = x._flat()
        return _ops.linear(xd, xs.view(torch.uint8), xg, xf, xm, wd, ws, wg, wf, wm, K, bias, out_dtype, out, alpha, beta,
                         amax_out)
    if isinstance(x, Int8Tensor):
        if x.shape[-1] != K:
            raise ValueError(f"x must have shape [..., {K}], got {x.shape}")
        o2 = out.reshape(-1, N) if out is not None else None
        od = out_dtype if out_dtype is not None or out is not None else x.dtype
        y = _ops.linear(x.data.reshape(-1, K), x.scales.reshape(-1, K // 16), None, 0, 0, wd, ws, wg, wf, wm, K, bias, od,
                      o2, alpha, beta, amax_out)
        return out if out is not None else y.reshape(*x.shape[:-1], N)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor, an NVFP4Tensor or an Int8Tensor")
    if x.dim() < 1 or x.shape[-1] != K:
        raise ValueError(f"x must have shape [..., {K}], got {tuple(x.shape)}")
    # act: "exact" keeps x (W4A16, every product exact, one FFMA per MAC); "int8" quantizes it to int8 with an FP32 scale per 16
    # values (W4A8, exact integer block sums on DP4A); "nvfp4" to NVFP4 (W4A4); "auto": "exact" up to AUTO_EXACT_MAX_ROWS rows, else "int8"
    if act not in ("auto", "exact", "int8", "nvfp4"):
        raise ValueError(f"act must be 'auto', 'exact', 'int8' or 'nvfp4', got {act!r}")
    x2 = x.reshape(-1, K)
    if act == "auto":
        act = "exact" if x2.shape[0] <= AUTO_EXACT_MAX_ROWS else "int8"
    if act == "int8":
        y = linear(quantize_int8(x2), weight, bias, out_dtype=out_dtype, out=out, alpha=alpha, beta=beta,
                   amax_out=amax_out)
        return out if out is not None else y.reshape(*x.shape[:-1], N)
    o2 = out.reshape(-1, N) if out is not None else None
    od = out_dtype if out_dtype is not None or out is not None else x.dtype  # quantized x: output in x's dtype
    if act == "nvfp4":
        qd, qs, qg, qf, qm, _ = quantize(x2)._flat()
        y = _ops.linear(qd, qs.view(torch.uint8), qg, qf, qm, wd, ws, wg, wf, wm, K, bias, od, o2, alpha, beta, amax_out)
    else:
        y = _ops.linear(x2, None, None, 0, 0, wd, ws, wg, wf, wm, K, bias, out_dtype, o2, alpha, beta, amax_out)
    return out if out is not None else y.reshape(*x.shape[:-1], N)


def grouped_linear(x: Union[torch.Tensor, NVFP4Tensor, Int8Tensor], weight: NVFP4Tensor, seg: torch.Tensor, groups: int,
                   bias: Optional[torch.Tensor] = None, *, act: str = "int8", out_dtype: Optional[torch.dtype] = None,
                   out: Optional[torch.Tensor] = None, alpha: float = 1.0, beta: float = 0.0,
                   amax_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mixture-of-experts linear layer in one launch, ``linear`` for every group: ``alpha * x_g @ weight_g.T + bias_g + beta * out`` for the rows
    ``seg[g] .. seg[g + 1]`` of ``x`` ``[R, K]`` (grouped by expert). ``weight``: the ``NVFP4Tensor`` ``[groups * N, K]`` stacking the groups' ``[N, K]``
    weights (K a multiple of 64; every scale layout, type and global ``linear`` takes); ``bias``: ``[groups * N]``; ``seg``: int32 ``[groups + 1]`` on the
    device, ``seg[0] = 0`` (the row counts never reach the host: no synchronization, CUDA-graph capturable). ``x``: dense, a 2-D ``NVFP4Tensor`` or an
    ``Int8Tensor``; for a dense ``x``, ``act``: ``"int8"`` (default; W4A8, DP4A), ``"exact"`` (W4A16), ``"nvfp4"`` (W4A4) or ``"auto"`` (= ``"int8"``: the
    row counts of the groups are not known here). Each group's rows are ``linear``'s on the tiled kernel (never the GEMV kernel), bit for bit. Result
    ``[R, N]`` (rows past ``seg[groups]`` zero); with ``out`` and ``beta`` the products accumulate into it."""
    if not isinstance(weight, NVFP4Tensor):
        raise TypeError(f"weight must be an NVFP4Tensor, got {type(weight).__name__}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [groups * out_features, in_features], got shape {weight.shape}")
    if not isinstance(seg, torch.Tensor):
        raise TypeError("seg must be an int32 tensor on the device")
    if act not in ("auto", "exact", "int8", "nvfp4"):
        raise ValueError(f"act must be 'auto', 'exact', 'int8' or 'nvfp4', got {act!r}")
    K = weight.shape[1]
    wd, ws, wg, wf, wm, _ = weight._flat()
    ws = ws.view(torch.uint8)
    if isinstance(x, NVFP4Tensor):
        if x.dim() != 2 or x.shape[1] != K:
            raise ValueError(f"an NVFP4 x must have shape [R, {K}], got {x.shape}")
        xd, xs, xg, xf, xm, _ = x._flat()
        return _ops.grouped_linear(xd, xs.view(torch.uint8), xg, xf, xm, seg, wd, ws, wg, wf, wm, K, groups, bias, out_dtype, out, alpha, beta,
                                   amax_out)
    if isinstance(x, torch.Tensor) and act in ("auto", "int8"):
        x = quantize_int8(x.reshape(-1, K))
    if isinstance(x, Int8Tensor):
        if x.shape[-1] != K:
            raise ValueError(f"x must have shape [R, {K}], got {tuple(x.shape)}")
        return _ops.grouped_linear(x.data.reshape(-1, K), x.scales.reshape(-1, K // 16), None, 0, 0, seg, wd, ws, wg, wf, wm, K, groups, bias,
                                   out_dtype or (out.dtype if out is not None else x.dtype), out, alpha, beta, amax_out)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor, an NVFP4Tensor or an Int8Tensor")
    if x.dim() < 1 or x.shape[-1] != K:
        raise ValueError(f"x must have shape [R, {K}], got {tuple(x.shape)}")
    x2 = x.reshape(-1, K)
    if act == "nvfp4":
        qd, qs, qg, qf, qm, _ = quantize(x2)._flat()
        return _ops.grouped_linear(qd, qs.view(torch.uint8), qg, qf, qm, seg, wd, ws, wg, wf, wm, K, groups, bias,
                                   out_dtype or (out.dtype if out is not None else x.dtype), out, alpha, beta, amax_out)
    return _ops.grouped_linear(x2, None, None, 0, 0, seg, wd, ws, wg, wf, wm, K, groups, bias, out_dtype, out, alpha, beta, amax_out)


def _linear_tc(x, weight, bias, act, out_dtype, out, alpha, beta, amax_out):
    """``linear`` with a ``TCWeight``: tensor-core GEMV up to 8 rows of ``x``, GEMM beyond.  The image fixes the activations:
    bfloat16 (``weight.act == "exact"``, W4A16) or int8 / NVFP4 (a dense ``x`` quantized by ``act``, ``"auto"`` taking the
    weight's; an ``Int8Tensor`` / ``NVFP4Tensor`` as it is)."""
    N, K = weight.shape
    if act == "auto":
        act = weight.act
    if act not in ("exact", "int8", "nvfp4"):
        raise ValueError(f"act must be 'auto', 'exact', 'int8' or 'nvfp4', got {act!r}")
    if isinstance(x, torch.Tensor):
        if (act == "exact") != (weight.act == "exact"):
            raise ValueError(f"act {act!r} does not match the weight's image ({weight.act!r})")
        if act == "exact" and x.dtype != torch.bfloat16:
            raise TypeError(f"the exact tensor-core path (W4A16) takes a bfloat16 x, got {x.dtype}")
        od = out_dtype or (out.dtype if out is not None else x.dtype)
        lead = tuple(x.shape[:-1])
        x2 = x.reshape(-1, K)
        x = quantize_int8(x2) if act == "int8" else quantize(x2) if act == "nvfp4" else x2
    else:
        if weight.act == "exact":
            raise ValueError("an Int8Tensor / NVFP4Tensor x needs a weight image for int8 / nvfp4")
        od = out_dtype or (out.dtype if out is not None else x.dtype if isinstance(x, Int8Tensor) else torch.bfloat16)
        lead = tuple(x.shape[:-1])
    rows = x.rows if isinstance(x, NVFP4Tensor) else math.prod(x.shape[:-1])
    wd, ws, wg, wf, wm, _ = weight.weight._flat()
    o2 = out.reshape(-1, N) if out is not None else None
    if rows <= 8:  # the GEMV: x as it is
        if isinstance(x, NVFP4Tensor):
            xd, xs, xg, xf, xm, _ = x._flat()
            y = _ops.linear(xd, xs.view(torch.uint8), xg, xf, xm, wd, ws, wg, wf, wm, K, bias, od, o2, alpha, beta, amax_out,
                          weight.image)
        elif isinstance(x, Int8Tensor):
            y = _ops.linear(x.data.reshape(-1, K), x.scales.reshape(-1, K // 16), None, 0, 0, wd, ws, wg, wf, wm, K, bias, od,
                          o2, alpha, beta, amax_out, weight.image)
        else:
            y = _ops.linear(x, None, None, 0, 0, wd, ws, wg, wf, wm, K, bias, od, o2, alpha, beta, amax_out, weight.image)
    elif isinstance(x, NVFP4Tensor):
        xd, xs, xg, xf, xm, _ = x._flat()
        img = _ops.tc_image(1, xd, xs.view(torch.uint8), xf, K)
        y = _ops.linear_tc(img, 0, xg, xf, xm, rows, weight.image, wg, wf, wm, N, K, bias, od, o2, alpha, beta, amax_out)
    elif isinstance(x, Int8Tensor):
        img = _ops.tc_image(1, x.data.reshape(-1, K), x.scales.reshape(-1, K // 16), 0, K)
        y = _ops.linear_tc(img, 1, None, 0, 0, rows, weight.image, wg, wf, wm, N, K, bias, od, o2, alpha, beta, amax_out)
    else:
        img = _ops.tc_image(1, x, None, 0, K)
        y = _ops.linear_tc(img, 2, None, 0, 0, rows, weight.image, wg, wf, wm, N, K, bias, od, o2, alpha, beta, amax_out)
    return out if out is not None else y.reshape(*lead, N)
