# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""NVFP4EffMA: NVFP4 linear layers, GEMM / GEMV and quantization for NVIDIA GPUs without native FP4
(sm_50..sm_90).

NVFP4 values are E2M1 (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}) with one 8-bit block scale per 16 values of a row
and a float32 global: ``value = e2m1 * scale * global``.  An ``NVFP4Tensor`` holds the parts as ordinary
CUDA tensors in any of the layouts in use: ModelOpt checkpoints (``weight`` uint8 ``[rows, cols / 2]``,
``weight_scale`` E4M3 ``[rows, cols / 16]``, ``weight_scale_2`` the global decode factor), Transformer
Engine (padded or GEMM-swizzled scales, an amax global per tensor or per row, 1 x 16 or 16 x 16 blocks,
4over6, UE5M3 scales, a columnwise representation) and the scale layout of cuBLASLt's FP4 GEMM.
``NVFP4Tensor.from_te`` wraps a Transformer Engine NVFP4 tensor, ``quantize`` produces any of them with
Transformer Engine's arithmetic, bit for bit.

Every call checks its arguments and raises ``TypeError`` (dtype), ``ValueError`` (device, shape, K not a
multiple of 64, overlapping output), ``NotImplementedError`` (GPU with native FP4, or older than sm_50)
or ``RuntimeError`` (library / CUDA failure) instead of running on bad input.
"""
from typing import Optional, Union

import torch

from . import _C
from .tensor import NVFP4Linear, NVFP4Tensor, quantize_linear_

__version__ = _C.__version__
__all__ = ["quantize", "dequantize", "linear", "NVFP4Tensor", "NVFP4Linear", "quantize_linear_", "__version__"]

# act="auto": up to this many rows of x the exact W4A16 path runs (bandwidth-bound: as fast as W4A8 there);
# beyond it the activations are quantized to int8 first (W4A8, DP4A), which is about twice as fast.
AUTO_EXACT_MAX_ROWS = 2


def quantize(x: torch.Tensor, global_scale: Optional[torch.Tensor] = None, *, amax: Optional[torch.Tensor] = None,
             block: str = "1x16", four_over_six: Optional[str] = None, scale_max: Optional[int] = None,
             row_scaled: bool = False, columnwise: bool = False, swizzled: bool = False,
             scale_dtype: str = "e4m3") -> NVFP4Tensor:
    """Quantize ``x`` (bfloat16 / float16 / float32 CUDA tensor ``[..., K]``, K a multiple of 16) to NVFP4 with
    the arithmetic of Transformer Engine (bit for bit).

    * global: by default Transformer Engine's current scaling -- the amax of ``x`` (of each row with
      ``row_scaled=True``) is computed and stored, decode factor amax / (6 * scale_max).  A given
      ``global_scale`` (float32, one element or one per row) is used as the decode factor instead (ModelOpt
      checkpoints store this one as ``weight_scale_2``).
    * ``block``: ``"1x16"`` (activations, gradients, ModelOpt) or ``"16x16"`` (Transformer Engine's weights:
      one scale per 16 x 16 tile, so the rowwise and columnwise quantizations are transposes of each other).
    * ``four_over_six``: ``None``, ``"mae"`` or ``"mse"`` -- Transformer Engine's 4over6 block scale
      selection (with ``scale_max`` 256 unless given).
    * ``columnwise``: also quantize ``x^T`` (stored as ``result.columnwise``), as Transformer Engine does for
      the backward GEMMs.
    * ``swizzled``: scales in the GEMM layout of cuBLASLt / Transformer Engine; ``scale_dtype``: ``"e4m3"``
      or ``"ue5m3"``.
    * ``amax`` (float32 ``[1]``) optionally receives ``max |x|``.  ``x`` must be finite.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if block not in ("1x16", "16x16"):
        raise ValueError(f"block must be '1x16' or '16x16', got {block!r}")
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
    dr, sr, dc, sc = _C.quantize_ex(x2, True, columnwise, g_row, g_col, flags, smax, 16 if block == "16x16" else 1, fos,
                                    te, amax)
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


def dequantize(x: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The values of ``x`` rounded once to ``dtype`` (bfloat16, float16 or float32): ``(e2m1 * scale) *
    global_scale``, or Transformer Engine's ``e2m1 * ((scale * amax) / (6 * scale_max))``."""
    if not isinstance(x, NVFP4Tensor):
        raise TypeError(f"expected an NVFP4Tensor, got {type(x).__name__}")
    d, s, g, flags, smax, cols = x._flat()
    return _C.dequantize(d, s.view(torch.uint8), g, flags, smax, cols, dtype).reshape(x.shape)


def linear(x: Union[torch.Tensor, NVFP4Tensor], weight: NVFP4Tensor, bias: Optional[torch.Tensor] = None, *,
           act: str = "auto", out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None,
           alpha: float = 1.0, beta: float = 0.0, amax_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``alpha * x @ weight.T + bias + beta * out`` like ``torch.nn.functional.linear``.

    * ``weight``: ``NVFP4Tensor`` ``[N, K]``, K a multiple of 64, in any of its layouts.  The backward GEMMs of
      a Transformer Engine linear layer use the columnwise halves: ``dX = linear(dY, W.columnwise)``,
      ``dW = linear(dY_q.columnwise, X_q.columnwise)`` for NVFP4-quantized ``dY_q`` / ``X_q``.
    * ``x``: bfloat16 / float16 / float32 tensor ``[..., K]`` (W4A16: one FFMA per MAC on the exactly
      decoded weights), or a 2-D ``NVFP4Tensor`` ``[T, K]`` (W4A4: exact integer sums per block of 16, DP4A
      from sm_61).  For few rows a bandwidth-bound GEMV kernel reads every weight byte once per 8 rows.
    * ``act`` (dense ``x``): ``"exact"`` keeps it as it is (W4A16, every product exact; on Pascal bound by
      the FP32 FFMA rate); ``"int8"`` quantizes it first to int8 with an FP32 scale per 16 values (W4A8:
      exact integer block sums on DP4A, about twice as fast, the accuracy class of llama.cpp's int8
      activations); ``"nvfp4"`` to NVFP4 (W4A4); ``"auto"`` (default) is ``"exact"`` up to
      ``AUTO_EXACT_MAX_ROWS`` (2) rows of ``x``, where both run at memory speed, and ``"int8"`` beyond.
    * ``out_dtype``: bfloat16, float16 or float32 (default: the dtype of a dense ``x``, else bfloat16);
      ``beta != 0`` accumulates into ``out``, which must then be given ``[T, N]``; ``amax_out`` (float32
      ``[1]``) gets max |result|.
    """
    if not isinstance(weight, NVFP4Tensor):
        raise TypeError(f"weight must be an NVFP4Tensor, got {type(weight).__name__}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [out_features, in_features], got shape {weight.shape}")
    N, K = weight.shape
    wd, ws, wg, wf, wm, _ = weight._flat()
    ws = ws.view(torch.uint8)
    if isinstance(x, NVFP4Tensor):
        if x.dim() != 2 or x.shape[1] != K:
            raise ValueError(f"an NVFP4 x must have shape [T, {K}], got {x.shape}")
        xd, xs, xg, xf, xm, _ = x._flat()
        return _C.linear(xd, xs.view(torch.uint8), xg, xf, xm, wd, ws, wg, wf, wm, K, bias, out_dtype, out, alpha, beta,
                         amax_out)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor or an NVFP4Tensor")
    if x.dim() < 1 or x.shape[-1] != K:
        raise ValueError(f"x must have shape [..., {K}], got {tuple(x.shape)}")
    if act not in ("auto", "exact", "int8", "nvfp4"):
        raise ValueError(f"act must be 'auto', 'exact', 'int8' or 'nvfp4', got {act!r}")
    x2 = x.reshape(-1, K)
    if act == "auto":
        act = "exact" if x2.shape[0] <= AUTO_EXACT_MAX_ROWS else "int8"
    o2 = out.reshape(-1, N) if out is not None else None
    od = out_dtype if out_dtype is not None or out is not None else x.dtype  # quantized x: output in x's dtype
    if act == "int8":
        q, s = _C.quantize_int8(x2)
        y = _C.linear(q, s, None, 0, 0, wd, ws, wg, wf, wm, K, bias, od, o2, alpha, beta, amax_out)
    elif act == "nvfp4":
        qd, qs, qg, qf, qm, _ = quantize(x2)._flat()
        y = _C.linear(qd, qs.view(torch.uint8), qg, qf, qm, wd, ws, wg, wf, wm, K, bias, od, o2, alpha, beta, amax_out)
    else:
        y = _C.linear(x2, None, None, 0, 0, wd, ws, wg, wf, wm, K, bias, out_dtype, o2, alpha, beta, amax_out)
    return out if out is not None else y.reshape(*x.shape[:-1], N)
