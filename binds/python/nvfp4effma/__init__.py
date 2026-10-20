# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""NVFP4EffMA: NVFP4 linear layers, GEMM / GEMV and quantization for NVIDIA GPUs without native FP4
(sm_50..sm_90).

NVFP4 values are E2M1 (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}) with one E4M3 scale per block of 16 consecutive
values of a row and one float32 scale per tensor: ``value = e2m1 * scale * global_scale``.  An
``NVFP4Tensor`` holds the three parts as ordinary CUDA tensors -- ``data`` uint8 ``[rows, cols / 2]`` (two
values per byte, the even column in the low nibble), ``scales`` float8_e4m3fn ``[rows, cols / 16]`` and
``global_scale`` float32 ``[1]`` -- the layout of NVFP4 checkpoints (ModelOpt: ``weight``,
``weight_scale``, ``weight_scale_2``).

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


def quantize(x: torch.Tensor, global_scale: Optional[torch.Tensor] = None, *,
             amax: Optional[torch.Tensor] = None) -> NVFP4Tensor:
    """Quantize ``x`` (bfloat16 / float16 / float32 CUDA tensor ``[..., K]``, K a multiple of 16) to NVFP4.

    Per block of 16 values: ``scale = E4M3_RNE(amax_block / (6 * global_scale))``, ``q = E2M1_RNE(x / (scale
    * global_scale))`` (saturating).  ``global_scale`` (float32, one element) defaults to
    ``amax(|x|) / (6 * 448)``, so the largest block scale lands on the top of the E4M3 range; ``amax``
    (float32 ``[1]``) optionally receives ``max |x|``.  ``x`` must be finite.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    if global_scale is None:
        a = x.detach().abs().amax().float().reshape(1)
        global_scale = torch.where(a > 0, a / (6.0 * 448.0), torch.ones_like(a))
    x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
    data, scales = _C.quantize(x2, global_scale, amax)
    shape = tuple(x.shape)
    return NVFP4Tensor(data.reshape(*shape[:-1], shape[-1] // 2), scales.reshape(*shape[:-1], shape[-1] // 16), global_scale)


def dequantize(x: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``(e2m1 * scale) * global_scale`` rounded once to ``dtype`` (bfloat16, float16 or float32)."""
    if not isinstance(x, NVFP4Tensor):
        raise TypeError(f"expected an NVFP4Tensor, got {type(x).__name__}")
    shape = x.shape
    d = x.data.reshape(-1, x.data.shape[-1]) if x.data.dim() != 2 else x.data
    s = x.scales.reshape(-1, x.scales.shape[-1]) if x.scales.dim() != 2 else x.scales
    return _C.dequantize(d, s, x.global_scale, dtype).reshape(shape)


def linear(x: Union[torch.Tensor, NVFP4Tensor], weight: NVFP4Tensor, bias: Optional[torch.Tensor] = None, *,
           out_dtype: Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None, alpha: float = 1.0,
           beta: float = 0.0, amax_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``alpha * x @ weight.T + bias + beta * out`` like ``torch.nn.functional.linear``.

    * ``weight``: ``NVFP4Tensor`` ``[N, K]``, K a multiple of 64.
    * ``x``: bfloat16 / float16 / float32 tensor ``[..., K]`` (W4A16: one FFMA per MAC on the exactly
      decoded weights), or a 2-D ``NVFP4Tensor`` ``[T, K]`` (W4A4: exact integer sums per block of 16, DP4A
      from sm_61).  Up to 8 rows run a bandwidth-bound GEMV kernel that reads every weight byte once.
    * ``out_dtype``: bfloat16, float16 or float32 (default: the dtype of a dense ``x``, else bfloat16);
      ``beta != 0`` accumulates into ``out``, which must then be given ``[T, N]``; ``amax_out`` (float32
      ``[1]``) gets max |result|.
    """
    if not isinstance(weight, NVFP4Tensor):
        raise TypeError(f"weight must be an NVFP4Tensor, got {type(weight).__name__}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D [out_features, in_features], got shape {weight.shape}")
    N, K = weight.shape
    if isinstance(x, NVFP4Tensor):
        if x.dim() != 2 or x.shape[1] != K:
            raise ValueError(f"an NVFP4 x must have shape [T, {K}], got {x.shape}")
        return _C.linear(x.data, x.scales, x.global_scale, weight.data, weight.scales, weight.global_scale, bias,
                         out_dtype, out, alpha, beta, amax_out)
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor or an NVFP4Tensor")
    if x.dim() < 1 or x.shape[-1] != K:
        raise ValueError(f"x must have shape [..., {K}], got {tuple(x.shape)}")
    x2 = x.reshape(-1, K)
    o2 = out.reshape(-1, N) if out is not None else None
    y = _C.linear(x2, None, None, weight.data, weight.scales, weight.global_scale, bias, out_dtype, o2, alpha, beta,
                  amax_out)
    return out if out is not None else y.reshape(*x.shape[:-1], N)
