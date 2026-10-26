# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""``NVFP4Tensor`` (the parts of an NVFP4 matrix, in any of the layouts of ModelOpt, Transformer Engine and
cuBLASLt), ``Int8Tensor`` (int8 activations with a scale per 16 values) and ``NVFP4Linear``, a drop-in replacement for ``nn.Linear`` with NVFP4 weights.  Inference only:
there is no autograd support."""
import math
from typing import Optional, Tuple

import torch

from . import _C


class NVFP4Tensor:
    """An NVFP4 tensor of logical shape ``[..., cols]``.

    * ``data``: uint8 ``[..., >= cols / 2]``, column 2c in the low nibble of byte c.
    * ``scales``: one block scale per 16 values along the last dimension -- float8_e4m3fn (or its bytes as
      uint8), or UE5M3 bytes (uint8, ``ue5m3=True``).  Linear layout ``[..., >= cols / 16]`` (ModelOpt's
      ``weight_scale``; Transformer Engine pads it to ``[roundup(rows, 128), roundup(cols / 16, 4)]``), or
      the swizzled GEMM layout of cuBLASLt / Transformer Engine (``swizzled=True``): a 2-D
      ``[roundup(rows, 128), roundup(cols / 16, 4)]`` buffer.
    * the global: ``global_scale`` (the decode factor, ModelOpt's ``weight_scale_2``) or ``amax``
      (Transformer Engine stores the amax: decode factor = amax / (6 * scale_max)); float32 with one element,
      or one per row (row-scaled NVFP4).  Neither: 1.
    * ``scale_max``: the bound of the global encoding, 0 = the scale type's maximum (448 / 114688), 256 for
      Transformer Engine's 4over6 tensors.
    * ``columnwise``: optionally the columnwise representation, an ``NVFP4Tensor`` of the transpose
      (Transformer Engine's columnwise data), used by the backward GEMMs.

    ``value = e2m1 * scale * global``.
    """

    def __init__(self, data: torch.Tensor, scales: torch.Tensor, global_scale: Optional[torch.Tensor] = None, *,
                 amax: Optional[torch.Tensor] = None, scale_max: int = 0, swizzled: bool = False, ue5m3: bool = False,
                 cols: Optional[int] = None, columnwise: Optional["NVFP4Tensor"] = None, block: Optional[Tuple[int, int]] = None):
        if data.dtype != torch.uint8:
            raise TypeError(f"data must be torch.uint8 (two E2M1 values per byte), got {data.dtype}")
        if ue5m3:
            if scales.dtype != torch.uint8:
                raise TypeError(f"UE5M3 scales are stored as torch.uint8, got {scales.dtype}")
        elif scales.dtype == torch.uint8:
            scales = scales.view(torch.float8_e4m3fn)
        elif scales.dtype != torch.float8_e4m3fn:
            raise TypeError(f"scales must be torch.float8_e4m3fn (or uint8), got {scales.dtype}")
        if global_scale is not None and amax is not None:
            raise ValueError("give either global_scale (a decode factor) or amax, not both")
        g = global_scale if global_scale is not None else amax
        if g is not None and (g.dtype != torch.float32 or g.numel() < 1):
            raise TypeError("global_scale / amax must be a float32 tensor")
        if data.dim() < 1:
            raise ValueError("data must have at least one dimension")
        self.cols = int(cols) if cols is not None else 2 * data.shape[-1]
        if self.cols > 2 * data.shape[-1] or self.cols % 16:
            raise ValueError(f"cols = {self.cols}: a multiple of 16 with at most 2 * data.shape[-1] = {2 * data.shape[-1]} values")
        rows = math.prod(data.shape[:-1])
        if swizzled:
            pr, pc = (rows + 127) // 128 * 128, (self.cols // 16 + 3) // 4 * 4
            if tuple(scales.shape) != (pr, pc):
                raise ValueError(f"swizzled scales of a [{rows}, {self.cols}] matrix are [{pr}, {pc}], got {tuple(scales.shape)}")
        elif scales.dim() == data.dim() and scales.shape[:-1] == data.shape[:-1]:
            if scales.shape[-1] < self.cols // 16:
                raise ValueError(f"scales {tuple(scales.shape)} have fewer than cols / 16 = {self.cols // 16} columns")
        elif scales.dim() != 2 or scales.shape[0] < rows or scales.shape[1] < self.cols // 16:
            raise ValueError(f"data {tuple(data.shape)} and scales {tuple(scales.shape)} do not describe the same matrix "
                             f"(scales [..., cols / 16], or a padded 2-D [>= rows, >= cols / 16])")
        if g is not None and g.numel() not in (1, rows):
            raise ValueError(f"the global has {g.numel()} elements: 1 (per tensor) or {rows} (one per row)")
        if not (data.device == scales.device and (g is None or g.device == data.device)):
            raise ValueError("data, scales and the global must be on one device")
        if scale_max not in (0, 256, 448, 114688):
            raise ValueError(f"scale_max must be 0, 256, 448 or 114688, got {scale_max}")
        self.data, self.scales = data, scales
        self.global_scale = g.reshape(-1) if g is not None else None
        self.global_is_amax = amax is not None
        self.scale_max, self.swizzled, self.ue5m3 = scale_max, swizzled, ue5m3
        self.columnwise, self.block = columnwise, block

    # ---- metadata ---------------------------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return (*self.data.shape[:-1], self.cols)

    def dim(self) -> int:
        return self.data.dim()

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def rows(self) -> int:
        return math.prod(self.data.shape[:-1])

    @property
    def per_row(self) -> bool:
        return self.global_scale is not None and self.global_scale.numel() > 1

    @property
    def amax(self) -> Optional[torch.Tensor]:
        return self.global_scale if self.global_is_amax else None

    @property
    def flags(self) -> int:
        return ((_C.SCALES_SWIZZLED if self.swizzled else 0) | (_C.SCALES_UE5M3 if self.ue5m3 else 0) |
                (_C.GLOBAL_AMAX if self.global_is_amax else 0) | (_C.GLOBAL_PER_ROW if self.per_row else 0))

    def _flat(self):
        """(data 2-D, scales 2-D, global, flags, scale_max, cols) as the extension takes them."""
        d = self.data.reshape(-1, self.data.shape[-1]) if self.data.dim() != 2 else self.data
        s = self.scales if self.scales.dim() == 2 else self.scales.reshape(-1, self.scales.shape[-1])
        return d, s, self.global_scale, self.flags, self.scale_max, self.cols

    # ---- conversions ------------------------------------------------------------------------------------
    @classmethod
    def from_float(cls, x: torch.Tensor, global_scale: Optional[torch.Tensor] = None, **recipe) -> "NVFP4Tensor":
        from . import quantize
        return quantize(x, global_scale, **recipe)

    @classmethod
    def from_te(cls, t) -> "NVFP4Tensor":
        """Wrap a Transformer Engine NVFP4 tensor (``NVFP4Tensor`` / ``NVFP4TensorStorage``) without copying: its
        rowwise data, scales (padded, linear or GEMM-swizzled), amax (per tensor or per row), 4over6 bound and
        scale type, and its columnwise representation when present."""
        swz = bool(getattr(t, "_with_gemm_swizzled_scales", False))
        ue = "UE5M3" in str(getattr(t, "_scale_dtype", "")).upper()
        smax = int(getattr(t, "_nvfp4_e4m3_max", 0) or 0)

        def half(data, scales, amax):
            if data is None:
                return None
            sc = scales if not swz else scales.reshape(scales.shape[0], -1)
            return cls(data, sc.view(torch.uint8) if ue else sc, amax=None if amax is None else amax.float(),
                       scale_max=smax, swizzled=swz, ue5m3=ue)

        col = half(getattr(t, "_columnwise_data", None), getattr(t, "_columnwise_scale_inv", None),
                   getattr(t, "_amax_columnwise", None))
        row = half(getattr(t, "_rowwise_data", None), getattr(t, "_rowwise_scale_inv", None),
                   getattr(t, "_amax_rowwise", None))
        if row is None:
            if col is None:
                raise ValueError("the tensor holds neither rowwise nor columnwise NVFP4 data")
            return col  # only the columnwise half: an NVFP4 tensor of the transpose
        row.columnwise = col
        return row

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        from . import dequantize
        return dequantize(self, dtype)

    def with_scale_layout(self, swizzled: bool) -> "NVFP4Tensor":
        """The same tensor with its block scales in the linear (``swizzled=False``) or the cuBLASLt / TE GEMM
        layout (both halves, if there is a columnwise one)."""
        if swizzled == self.swizzled:
            out = self
        else:
            d, s, g, _, _, cols = self._flat()
            sc = _C.swizzle_scales(s.view(torch.uint8), d.shape[0], cols, swizzled)
            if not self.ue5m3:
                sc = sc.view(torch.float8_e4m3fn)
            out = NVFP4Tensor(self.data, sc, None if self.global_is_amax else g, amax=g if self.global_is_amax else None,
                              scale_max=self.scale_max, swizzled=swizzled, ue5m3=self.ue5m3, cols=cols, block=self.block)
        if self.columnwise is not None and self.columnwise.swizzled != swizzled:
            if out is self:
                out = NVFP4Tensor(self.data, self.scales, None if self.global_is_amax else self.global_scale,
                                  amax=self.amax, scale_max=self.scale_max, swizzled=self.swizzled, ue5m3=self.ue5m3,
                                  cols=self.cols, block=self.block)
            out.columnwise = self.columnwise.with_scale_layout(swizzled)
        return out

    def transpose_2d(self) -> "NVFP4Tensor":
        """The transpose of a 16 x 16-tile tensor (Transformer Engine's weight recipe), data and scales: exactly
        the columnwise quantization of the original.  Not valid for 1 x 16 tensors (quantize them with
        ``columnwise=True`` instead)."""
        if self.block is not None and tuple(self.block) != (16, 16):
            raise ValueError("transpose_2d needs a tensor quantized with 16 x 16 tiles")
        if self.per_row:
            raise ValueError("per-row globals do not transpose")
        d, s, g, flags, smax, cols = self._flat()
        dt, st = _C.transpose(d, s.view(torch.uint8), flags, smax, cols)
        if not self.ue5m3:
            st = st.view(torch.float8_e4m3fn)
        return NVFP4Tensor(dt, st, None if self.global_is_amax else g, amax=g if self.global_is_amax else None,
                           scale_max=smax, swizzled=self.swizzled, ue5m3=self.ue5m3, cols=d.shape[0], block=(16, 16))

    def __repr__(self):
        parts = [f"shape={self.shape}", f"device={self.device}"]
        if self.global_scale is not None:
            kind = "amax" if self.global_is_amax else "global_scale"
            parts.append(f"{kind}={'per row' if self.per_row else f'{float(self.global_scale):.6g}'}")
        if self.swizzled:
            parts.append("swizzled scales")
        if self.ue5m3:
            parts.append("UE5M3 scales")
        if self.scale_max:
            parts.append(f"scale_max={self.scale_max}")
        if self.block is not None:
            parts.append(f"block={self.block[0]}x{self.block[1]}")
        if self.columnwise is not None:
            parts.append("+ columnwise")
        return f"NVFP4Tensor({', '.join(parts)})"


class Int8Tensor:
    """Activations quantized to int8 with one FP32 scale per 16 values along the last dimension, as
    ``linear(..., act="int8")`` does internally (W4A8): ``value = data * scale``.  Produced by
    ``nvfp4effma.quantize_int8``; ``linear`` takes it as ``x`` directly, so the quantized activations can be
    kept (e.g. for a backward pass) and reused without quantizing twice.

    * ``data``: int8 ``[..., cols]``, cols a multiple of 16.
    * ``scales``: float32 ``[..., cols / 16]``.
    * ``dtype``: the dtype of the values it was quantized from; ``linear`` returns this dtype by default.
    """

    def __init__(self, data: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16):
        if data.dtype != torch.int8:
            raise TypeError(f"data must be torch.int8, got {data.dtype}")
        if scales.dtype != torch.float32:
            raise TypeError(f"scales must be torch.float32, got {scales.dtype}")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"dtype must be bfloat16, float16 or float32, got {dtype}")
        if data.dim() < 1 or data.shape[-1] % 16:
            raise ValueError(f"data must be [..., cols] with cols a multiple of 16, got {tuple(data.shape)}")
        if tuple(scales.shape) != (*data.shape[:-1], data.shape[-1] // 16):
            raise ValueError(f"scales of a {tuple(data.shape)} tensor are {(*data.shape[:-1], data.shape[-1] // 16)}, "
                             f"got {tuple(scales.shape)}")
        if data.device != scales.device:
            raise ValueError("data and scales must be on one device")
        self.data, self.scales, self.dtype = data, scales, dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    def dim(self) -> int:
        return self.data.dim()

    @property
    def device(self) -> torch.device:
        return self.data.device

    @classmethod
    def from_float(cls, x: torch.Tensor) -> "Int8Tensor":
        from . import quantize_int8
        return quantize_int8(x)

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """``data * scale`` in float32, rounded once to ``dtype`` (default: the source dtype)."""
        k = self.data.shape[-1]
        v = self.data.float().reshape(-1, k // 16, 16) * self.scales.reshape(-1, k // 16, 1)
        return v.reshape(self.data.shape).to(dtype or self.dtype)

    def __repr__(self):
        return f"Int8Tensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"


class TCWeight:
    """An NVFP4 weight ``[N, K]`` with its tensor-core image (sm_75 .. sm_89), which serves both the GEMM and, up to 8
    rows of ``x``, the GEMV.  ``act`` picks the image and so the activations it multiplies: ``"exact"`` bfloat16
    (W4A16, sm_80+), ``"int8"`` / ``"nvfp4"`` (W4A8 / W4A4: one image for both).  The image is as large as the
    weight; it is built once, here."""

    def __init__(self, weight: NVFP4Tensor, act: str = "exact"):
        if not isinstance(weight, NVFP4Tensor) or weight.dim() != 2:
            raise TypeError("weight must be a 2-D NVFP4Tensor")
        if act not in ("exact", "int8", "nvfp4"):
            raise ValueError(f"act must be 'exact', 'int8' or 'nvfp4', got {act!r}")
        d, s, _, f, _, k = weight._flat()
        self.weight, self.act = weight, act
        self.image = _C.tc_image(0 if act == "exact" else 2, d, s.view(torch.uint8), f, k)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.weight.shape

    def dim(self) -> int:
        return 2


class NVFP4Linear(torch.nn.Module):
    """``y = x @ W.T + bias`` with an NVFP4 weight ``W`` ``[out_features, in_features]`` (in_features a
    multiple of 64).  The buffers carry the names of NVFP4 checkpoints: ``weight`` (uint8, packed),
    ``weight_scale`` (float8_e4m3fn, linear ``[out, in / 16]``), ``weight_scale_2`` (float32 global decode
    factor).  ``x`` stays in its own dtype (bfloat16 / float16 / float32); the output has the dtype of ``x``.
    ``act`` chooses how the activations enter the matmul (default ``"auto"``: exact for up to 2 rows, int8
    beyond; see ``nvfp4effma.linear``)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None,
                 dtype: torch.dtype = torch.bfloat16, act: str = "auto"):
        super().__init__()
        self.act = act  # "auto" / "exact" (W4A16) / "int8" (W4A8) / "nvfp4" (W4A4), see nvfp4effma.linear
        if in_features % 64:
            raise ValueError(f"in_features ({in_features}) must be a multiple of 64")
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("weight_scale", torch.zeros(out_features, in_features // 16, dtype=torch.float8_e4m3fn,
                                                         device=device))
        self.register_buffer("weight_scale_2", torch.ones(1, dtype=torch.float32, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, global_scale: Optional[torch.Tensor] = None,
                    act: str = "auto") -> "NVFP4Linear":
        """Quantize the weight of ``linear`` (a CUDA module) to NVFP4 with ModelOpt's checkpoint layout (1 x 16
        blocks, linear scales, a global decode factor; amax / (6 * 448) unless given); the bias keeps its dtype."""
        w = linear.weight.detach()
        m = cls(linear.in_features, linear.out_features, linear.bias is not None, device=w.device,
                dtype=linear.bias.dtype if linear.bias is not None else w.dtype, act=act)
        if global_scale is None:
            a = w.abs().amax().float().reshape(1)
            global_scale = torch.where(a > 0, a / (6.0 * 448.0), torch.ones_like(a))
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
        return linear(x, self.nvfp4_weight(), bias, act=self.act)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"NVFP4, act={self.act}")


def quantize_linear_(module: torch.nn.Module, *, act: str = "auto") -> torch.nn.Module:
    """Replace every ``nn.Linear`` inside ``module`` (in place) whose in_features is a multiple of 64 by an
    ``NVFP4Linear`` with the quantized weight (``act``: activation quantization, see ``NVFP4Linear``); other
    layers stay as they are.  Returns ``module``."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and child.in_features % 64 == 0:
            setattr(module, name, NVFP4Linear.from_linear(child, act=act))
        else:
            quantize_linear_(child, act=act)
    return module
