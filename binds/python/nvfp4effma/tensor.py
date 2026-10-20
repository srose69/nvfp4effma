# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""``NVFP4Tensor`` (the three tensors of an NVFP4 matrix) and ``NVFP4Linear``, a drop-in replacement for
``nn.Linear`` with NVFP4 weights.  Inference only: there is no autograd support."""
from typing import Optional, Tuple

import torch


class NVFP4Tensor:
    """An NVFP4 tensor of logical shape ``[..., K]``: ``data`` uint8 ``[..., K / 2]`` (column 2c in the low nibble
    of byte c), ``scales`` float8_e4m3fn ``[..., K / 16]`` (one per block of 16 values along the last
    dimension), ``global_scale`` float32 ``[1]``; ``value = e2m1 * scale * global_scale``."""

    def __init__(self, data: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor):
        if data.dtype != torch.uint8:
            raise TypeError(f"data must be torch.uint8 (two E2M1 values per byte), got {data.dtype}")
        if scales.dtype == torch.uint8:
            scales = scales.view(torch.float8_e4m3fn)
        if scales.dtype != torch.float8_e4m3fn:
            raise TypeError(f"scales must be torch.float8_e4m3fn, got {scales.dtype}")
        if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
            raise TypeError("global_scale must be a float32 tensor with one element")
        if data.dim() < 1 or scales.shape[:-1] != data.shape[:-1] or data.shape[-1] != 8 * scales.shape[-1]:
            raise ValueError(f"data {tuple(data.shape)} and scales {tuple(scales.shape)} do not describe the same "
                             "matrix (data [..., K / 2], scales [..., K / 16])")
        if not (data.device == scales.device == global_scale.device):
            raise ValueError("data, scales and global_scale must be on one device")
        self.data, self.scales, self.global_scale = data, scales, global_scale.reshape(1)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (*self.data.shape[:-1], 2 * self.data.shape[-1])

    def dim(self) -> int:
        return self.data.dim()

    @property
    def device(self) -> torch.device:
        return self.data.device

    @classmethod
    def from_float(cls, x: torch.Tensor, global_scale: Optional[torch.Tensor] = None) -> "NVFP4Tensor":
        from . import quantize
        return quantize(x, global_scale)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        from . import dequantize
        return dequantize(self, dtype)

    def __repr__(self):
        return f"NVFP4Tensor(shape={self.shape}, device={self.device}, global_scale={float(self.global_scale):.6g})"


class NVFP4Linear(torch.nn.Module):
    """``y = x @ W.T + bias`` with an NVFP4 weight ``W`` ``[out_features, in_features]`` (in_features a
    multiple of 64).  The buffers carry the names of NVFP4 checkpoints: ``weight`` (uint8, packed),
    ``weight_scale`` (float8_e4m3fn), ``weight_scale_2`` (float32 global scale).  ``x`` stays in its own
    dtype (bfloat16 / float16 / float32); the output has the dtype of ``x``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if in_features % 64:
            raise ValueError(f"in_features ({in_features}) must be a multiple of 64")
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("weight_scale", torch.zeros(out_features, in_features // 16, dtype=torch.float8_e4m3fn,
                                                         device=device))
        self.register_buffer("weight_scale_2", torch.ones(1, dtype=torch.float32, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, global_scale: Optional[torch.Tensor] = None) -> "NVFP4Linear":
        """Quantize the weight of ``linear`` (a CUDA module) to NVFP4; the bias keeps its dtype."""
        w = linear.weight.detach()
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, device=w.device,
                dtype=linear.bias.dtype if linear.bias is not None else w.dtype)
        q = NVFP4Tensor.from_float(w, global_scale)
        m.weight.copy_(q.data); m.weight_scale.copy_(q.scales); m.weight_scale_2.copy_(q.global_scale)
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.detach())
        return m

    def nvfp4_weight(self) -> NVFP4Tensor:
        return NVFP4Tensor(self.weight, self.weight_scale, self.weight_scale_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from . import linear
        bias = self.bias if self.bias is None or self.bias.dtype == x.dtype else self.bias.to(x.dtype)
        return linear(x, self.nvfp4_weight(), bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, NVFP4"


def quantize_linear_(module: torch.nn.Module) -> torch.nn.Module:
    """Replace every ``nn.Linear`` inside ``module`` (in place) whose in_features is a multiple of 64 by an
    ``NVFP4Linear`` with the quantized weight; other layers stay as they are.  Returns ``module``."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and child.in_features % 64 == 0:
            setattr(module, name, NVFP4Linear.from_linear(child))
        else:
            quantize_linear_(child)
    return module
