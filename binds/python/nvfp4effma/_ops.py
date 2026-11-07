# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""The extension's operations as PyTorch operators (``torch.ops.nvfp4effma.*``, registered by ``_C``), under the call signatures of the extension's own
functions.  Every call is one opaque node for ``torch.compile`` and CUDA graph capture: the operators have schemas (the ``_out`` variants' ``out`` is
mutated, nothing else: an amax is a result), Meta kernels that compute only the results' shapes and dtypes, and CUDA kernels that enqueue on the current
stream and read nothing back.  An operator may not return its input, so a given ``out`` goes to the ``*_out`` variant and the wrapper returns it."""
import torch

from . import _C  # noqa: F401  (loads the extension, which registers the operators)

_o = torch.ops.nvfp4effma

quantize_ex = _o.quantize_ex
quantize_int8 = _o.quantize_int8
dequantize = _o.dequantize
transpose = _o.transpose
swizzle_scales = _o.swizzle_scales
tc_image = _o.tc_image


def _accumulate_needs_out(beta, out):
    if beta != 0.0 and out is None:
        raise ValueError("beta != 0 accumulates into out, so out must be given")


def _out_dtype_matches(out, out_dtype):
    if out is not None and out_dtype is not None and out_dtype != out.dtype:
        raise ValueError(f"out_dtype {out_dtype} does not match out.dtype {out.dtype}")


def _finish(result, out, amax_out):
    """The operators return the amax of the result (when asked) as a result; the kernels max-reduce into a caller's buffer, so does this."""
    y, amax = result if out is None else (out, result)
    if amax_out is not None:
        torch.maximum(amax_out, amax, out=amax_out)
    return y


def linear(x, x_scales, x_global, x_flags, x_smax, w, w_scales, w_global, w_flags, w_smax, k, bias=None, out_dtype=None, out=None, alpha=1.0,
           beta=0.0, amax_out=None, w_img=None):
    _accumulate_needs_out(beta, out)
    _out_dtype_matches(out, out_dtype)
    args = (x, x_scales, x_global, x_flags, x_smax, w, w_scales, w_global, w_flags, w_smax, k, bias)
    if out is None:
        return _finish(_o.linear(*args, out_dtype, alpha, amax_out is not None, w_img), out, amax_out)
    return _finish(_o.linear_out(*args, out, alpha, beta, amax_out is not None, w_img), out, amax_out)


def grouped_linear(x, x_scales, x_global, x_flags, x_smax, seg, w, w_scales, w_global, w_flags, w_smax, k, groups, bias=None, out_dtype=None,
                   out=None, alpha=1.0, beta=0.0, amax_out=None):
    _accumulate_needs_out(beta, out)
    _out_dtype_matches(out, out_dtype)
    args = (x, x_scales, x_global, x_flags, x_smax, seg, w, w_scales, w_global, w_flags, w_smax, k, groups, bias)
    if out is None:
        return _finish(_o.grouped_linear(*args, out_dtype, alpha, amax_out is not None), out, amax_out)
    return _finish(_o.grouped_linear_out(*args, out, alpha, beta, amax_out is not None), out, amax_out)


def linear_tc(x_img, x_dtype, x_global, x_flags, x_smax, T, w_img, w_global, w_flags, w_smax, N, K, bias=None, out_dtype=torch.bfloat16, out=None,
              alpha=1.0, beta=0.0, amax_out=None):
    _accumulate_needs_out(beta, out)
    args = (x_img, x_dtype, x_global, x_flags, x_smax, T, w_img, w_global, w_flags, w_smax, N, K, bias)
    if out is None:
        return _finish(_o.linear_tc(*args, out_dtype, alpha, amax_out is not None), out, amax_out)
    return _finish(_o.linear_tc_out(*args, out, alpha, beta, amax_out is not None), out, amax_out)
