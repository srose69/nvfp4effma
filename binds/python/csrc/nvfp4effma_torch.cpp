// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of NVFP4EffMA (include/nvfp4effma.h).  Arguments are checked here before any pointer reaches the library:
// wrong dtype -> TypeError, wrong device / shape / layout / alignment -> ValueError, unsupported GPU -> NotImplementedError,
// library or CUDA failure -> RuntimeError (nvfp4a_last_error).  Calls run on the current CUDA stream of the tensors' device.
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include "nvfp4effma.h"

using at::Tensor;
using OptT = std::optional<Tensor>;

namespace {

nvfp4a_dtype dense_dtype(at::ScalarType t, const char *what) {
  switch (t) {
    case at::kBFloat16: return NVFP4A_BF16;
    case at::kHalf: return NVFP4A_FP16;
    case at::kFloat: return NVFP4A_FP32;
    default:
      TORCH_CHECK_TYPE(false, what, ": unsupported dtype ", t, " (expected torch.bfloat16, torch.float16 or torch.float32)");
  }
  return NVFP4A_FP32;
}

// The device every tensor of a call must live on; also rejects GPUs the library does not cover.
void check_device(const Tensor &t, const at::Device &dev, const char *what) {
  TORCH_CHECK_VALUE(t.is_cuda(), what, " must be a CUDA tensor, got a tensor on ", t.device());
  TORCH_CHECK_VALUE(t.device() == dev, what, " is on ", t.device(), " but the operation runs on ", dev);
}
void check_gpu(const at::Device &dev) {
  const cudaDeviceProp *p = at::cuda::getDeviceProperties(dev.index());
  const int cc = p->major * 10 + p->minor;
  TORCH_CHECK_NOT_IMPLEMENTED(cc < 100, "NVFP4EffMA targets GPUs without native FP4 (sm_50..sm_90); ", p->name,
                              " is sm_", cc, " and has native FP4: use cuBLASLt / Transformer Engine");
  TORCH_CHECK_NOT_IMPLEMENTED(cc >= 50, p->name, " is sm_", cc, "; NVFP4EffMA needs sm_50 or newer");
}

// optional per-tensor scale / amax: float32, one element, on dev
float *scalar_ptr(const OptT &t, const at::Device &dev, const char *what) {
  if (!t) return nullptr;
  check_device(*t, dev, what);
  TORCH_CHECK_TYPE(t->scalar_type() == at::kFloat, what, " must be torch.float32, got ", t->scalar_type());
  TORCH_CHECK_VALUE(t->numel() == 1, what, " must have exactly one element, got shape ", t->sizes());
  return t->data_ptr<float>();
}

void check_status(nvfp4a_status s, const char *what) {
  if (s == NVFP4A_OK) return;
  const std::string msg = std::string(what) + " failed (" + nvfp4a_status_string(s) + "): " + nvfp4a_last_error();
  TORCH_CHECK_NOT_IMPLEMENTED(s != NVFP4A_NOT_SUPPORTED, msg);
  TORCH_CHECK_VALUE(s != NVFP4A_INVALID_VALUE, msg);
  TORCH_CHECK(false, msg);
}

int as_int(int64_t v, const char *what) {
  TORCH_CHECK_VALUE(v <= INT32_MAX, what, " = ", v, " exceeds the 32-bit range of the library");
  return (int)v;
}

// An NVFP4 matrix [rows, cols] = data uint8 [rows, >= cols / 2] (two values per byte, column 2c in the low nibble; row stride
// in values a multiple of `align`), scales E4M3 / UE5M3 (or their bytes) and a float32 global, one value or one per row.
// Scales: linear [>= rows, >= ceil(cols / 16)] with row stride a multiple of `salign`, or swizzled (padded GEMM shape); linear misfits are copied.
struct Fp4 { Tensor data, scales; int ld, ld_scale; const float *g; unsigned flags; int smax; };
Fp4 fp4_operand(const Tensor &data, const Tensor &scales, const OptT &global, int64_t flags, int64_t smax, int64_t rows,
                int64_t cols, const at::Device &dev, const char *what, int align, int salign) {
  const std::string w(what);
  check_device(data, dev, (w + ".data").c_str());
  check_device(scales, dev, (w + ".scales").c_str());
  TORCH_CHECK_TYPE(data.scalar_type() == at::kByte, what, ".data must be torch.uint8 (two E2M1 values per byte), got ",
                   data.scalar_type());
  TORCH_CHECK_TYPE(scales.scalar_type() == at::kFloat8_e4m3fn || scales.scalar_type() == at::kByte, what,
                   ".scales must be torch.float8_e4m3fn or torch.uint8 (UE5M3 / raw bytes), got ", scales.scalar_type());
  const int64_t sc = (cols + 15) / 16;
  TORCH_CHECK_VALUE(data.dim() == 2 && data.size(0) == rows && data.size(1) * 2 >= cols && data.stride(1) == 1, what,
                    ".data must be [", rows, ", >= ", (cols + 1) / 2, "] with contiguous rows, got ", data.sizes());
  const bool swz = flags & NVFP4A_SCALES_SWIZZLED;
  Fp4 r{data, scales.view(at::kByte), 0, 0, nullptr, (unsigned)flags, (int)smax};
  if (swz) {
    const int64_t pr = (rows + 127) / 128 * 128, pc = (sc + 3) / 4 * 4;
    TORCH_CHECK_VALUE(scales.dim() == 2 && scales.size(0) == pr && scales.size(1) == pc && scales.is_contiguous(), what,
                      ".scales (swizzled) must be a contiguous [", pr, ", ", pc, "], got ", scales.sizes());
  } else {
    TORCH_CHECK_VALUE(scales.dim() == 2 && scales.size(0) >= rows && scales.size(1) >= sc, what, ".scales must be [>= ",
                      rows, ", >= ", sc, "], got ", scales.sizes());
    const auto ok = [](const Tensor &t, int64_t step, uintptr_t palign) {
      return t.stride(1) == 1 && (t.size(0) <= 1 || t.stride(0) % step == 0) && reinterpret_cast<uintptr_t>(t.data_ptr()) % palign == 0;
    };
    if (!ok(r.scales, salign, salign)) r.scales = r.scales.narrow(1, 0, sc).contiguous();
    r.ld_scale = (int)(rows <= 1 ? (sc + salign - 1) / salign * salign : r.scales.stride(0));
    if (r.ld_scale % salign || r.ld_scale < sc) { r.scales = r.scales.narrow(1, 0, sc).contiguous(); r.ld_scale = (int)sc; }
  }
  const int64_t cols_al = (cols + align - 1) / align * align;
  const bool data_ok = reinterpret_cast<uintptr_t>(data.data_ptr()) % 16 == 0 &&
                       (rows <= 1 || (data.stride(0) * 2) % align == 0) && data.size(1) * 2 >= cols_al;
  if (!data_ok) {  // a copy with rows padded to `align` values
    Tensor c = at::zeros({rows, cols_al / 2}, data.options());
    c.narrow(1, 0, (cols + 1) / 2).copy_(data.narrow(1, 0, (cols + 1) / 2));
    r.data = c;
  }
  r.ld = (int)(rows <= 1 ? cols_al : r.data.stride(0) * 2);
  if (global) {
    check_device(*global, dev, (w + ".global").c_str());
    TORCH_CHECK_TYPE(global->scalar_type() == at::kFloat, what, ".global must be torch.float32, got ", global->scalar_type());
    TORCH_CHECK_VALUE(global->is_contiguous(), what, ".global must be contiguous");
    const bool per_row = flags & NVFP4A_GLOBAL_PER_ROW;
    TORCH_CHECK_VALUE(global->numel() == (per_row ? rows : 1), what, ".global must have ", per_row ? rows : 1,
                      " element(s), got ", global->numel());
    r.g = global->data_ptr<float>();
  } else {
    TORCH_CHECK_VALUE(!(flags & NVFP4A_GLOBAL_PER_ROW), what, ": per-row globals need a global tensor");
  }
  return r;
}
nvfp4a_tensor as_tensor(const Fp4 &f) {
  return nvfp4a_tensor{f.data.data_ptr(), NVFP4A_NVFP4, f.scales.data_ptr(), f.ld_scale, f.g, nullptr, f.flags, f.smax};
}

Tensor dense_2d(const Tensor &x, int64_t align) {  // row stride a multiple of `align` values, 16-byte aligned
  const int64_t rows = x.size(0);
  if (x.stride(1) != 1 || (rows > 1 && x.stride(0) % align) || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) return x.contiguous();
  return x;
}
int ld_of(const Tensor &x, int64_t cols) { return (int)(x.size(0) <= 1 ? (cols + 15) / 16 * 16 : x.stride(0)); }

}  // namespace

// x [rows, cols] (BF16 / FP16 / FP32, cols a multiple of 16) -> NVFP4: (rowwise data, scales, columnwise data, scales), the
// columnwise half (quantization of x^T, [cols, rows]) when requested.  Linear scales [rows, cols / 16] (columnwise [cols,
// ceil(rows / 16)]), swizzled the padded GEMM shape.  Globals (float32): inputs (decode factors or amaxes), outputs with compute_amax.
std::tuple<OptT, OptT, OptT, OptT> quantize_ex(const Tensor &x, bool rowwise, bool columnwise, const OptT &g_row,
                                               const OptT &g_col, int64_t flags, int64_t scale_max, int64_t tile_rows,
                                               int64_t four_over_six, bool compute_amax, const OptT &amax) {
  TORCH_CHECK_VALUE(x.is_cuda(), "x must be a CUDA tensor, got a tensor on ", x.device());
  const at::Device dev = x.device();
  check_gpu(dev);
  const nvfp4a_dtype t = dense_dtype(x.scalar_type(), "x");
  TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2-D, got shape ", x.sizes());
  const int64_t rows = x.size(0), cols = x.size(1);
  TORCH_CHECK_VALUE(cols % 16 == 0, "the last dimension (", cols, ") must be a multiple of 16 (the NVFP4 block)");
  TORCH_CHECK_VALUE(rowwise || columnwise, "nothing to quantize: rowwise and columnwise are both off");
  const Tensor xs = dense_2d(x, 16);
  const bool swz = flags & NVFP4A_SCALES_SWIZZLED;
  const auto sdt = (flags & NVFP4A_SCALES_UE5M3) ? at::kByte : at::kFloat8_e4m3fn;
  const auto scales_for = [&](int64_t r, int64_t c) {
    const int64_t sc = (c + 15) / 16;
    return swz ? at::zeros({(r + 127) / 128 * 128, (sc + 3) / 4 * 4}, x.options().dtype(sdt))
               : at::zeros({r, sc}, x.options().dtype(sdt));
  };
  const int64_t rows16 = (rows + 15) / 16 * 16;
  OptT dr, sr, dc, scl;
  if (rowwise) { dr = at::zeros({rows, cols / 2}, x.options().dtype(at::kByte)); sr = scales_for(rows, cols); }
  if (columnwise) { dc = at::zeros({cols, rows16 / 2}, x.options().dtype(at::kByte)); scl = scales_for(cols, rows); }
  if (rows == 0) return {dr, sr, dc, scl};
  const auto gptr = [&](const OptT &g, int64_t n, const char *what) -> const float * {
    if (!g) return nullptr;
    check_device(*g, dev, what);
    TORCH_CHECK_TYPE(g->scalar_type() == at::kFloat && g->is_contiguous(), what, " must be a contiguous torch.float32");
    TORCH_CHECK_VALUE(g->numel() == n, what, " must have ", n, " element(s), got ", g->numel());
    return g->data_ptr<float>();
  };
  const bool per_row = flags & NVFP4A_GLOBAL_PER_ROW;
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor S = {xs.data_ptr(), t, nullptr, 0, nullptr, nullptr};
  nvfp4a_tensor R{}, C{};
  if (rowwise) R = {dr->data_ptr(), NVFP4A_NVFP4, sr->data_ptr(), swz ? 0 : (int)(cols / 16), gptr(g_row, per_row ? rows : 1, "global"),
                    scalar_ptr(amax, dev, "amax"), (unsigned)flags, (int)scale_max};
  if (columnwise) C = {dc->data_ptr(), NVFP4A_NVFP4, scl->data_ptr(), swz ? 0 : (int)((rows + 15) / 16),
                       gptr(g_col, 1, "columnwise global"), rowwise ? nullptr : scalar_ptr(amax, dev, "amax"),
                       (unsigned)(flags & ~NVFP4A_GLOBAL_PER_ROW), (int)scale_max};
  const nvfp4a_quant_config cfg = {(int)tile_rows, (nvfp4a_4over6)four_over_six, compute_amax ? 1 : 0};
  check_status(nvfp4a_quantize_ex(&S, ld_of(xs, cols), as_int(rows, "rows"), as_int(cols, "cols"), rowwise ? &R : nullptr,
                                  as_int(cols, "cols"), columnwise ? &C : nullptr, as_int(rows16, "rows"), &cfg,
                                  at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_quantize_ex");
  return {dr, sr, dc, scl};
}

// x [rows, cols] -> (int8 [rows, cols], float32 scales [rows, cols / 16]): sb = amax_block / 127,
// q = RNE(x / sb) -- the activations of the W4A8 path.
std::tuple<Tensor, Tensor> quantize_int8(const Tensor &x) {
  TORCH_CHECK_VALUE(x.is_cuda(), "x must be a CUDA tensor, got a tensor on ", x.device());
  const at::Device dev = x.device();
  check_gpu(dev);
  const nvfp4a_dtype t = dense_dtype(x.scalar_type(), "x");
  TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2-D, got shape ", x.sizes());
  const int64_t rows = x.size(0), cols = x.size(1);
  TORCH_CHECK_VALUE(cols % 16 == 0, "the last dimension (", cols, ") must be a multiple of 16");
  const Tensor xs = dense_2d(x, 16);
  Tensor q = at::empty({rows, cols}, x.options().dtype(at::kChar));
  Tensor scales = at::empty({rows, cols / 16}, x.options().dtype(at::kFloat));
  if (rows == 0 || cols == 0) return {q, scales};
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor S = {xs.data_ptr(), t, nullptr, 0, nullptr, nullptr};
  const nvfp4a_tensor Q = {q.data_ptr(), NVFP4A_INT8, scales.data_ptr(), (int)(cols / 16), nullptr, nullptr};
  check_status(nvfp4a_quantize(&S, ld_of(xs, cols), &Q, as_int(cols, "cols"), as_int(rows, "rows"), as_int(cols, "cols"),
                               at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_quantize");
  return {q, scales};
}

// NVFP4 [rows, cols] -> dense tensor of dtype
Tensor dequantize(const Tensor &data, const Tensor &scales, const OptT &global, int64_t flags, int64_t scale_max,
                  int64_t cols, at::ScalarType dtype) {
  TORCH_CHECK_VALUE(data.is_cuda(), "data must be a CUDA tensor, got a tensor on ", data.device());
  const at::Device dev = data.device();
  check_gpu(dev);
  TORCH_CHECK_VALUE(data.dim() == 2, "data must be 2-D [rows, cols / 2], got shape ", data.sizes());
  const int64_t rows = data.size(0);
  TORCH_CHECK_VALUE(cols % 16 == 0, "cols = ", cols, " must be a multiple of 16");
  const nvfp4a_dtype t = dense_dtype(dtype, "dtype");
  const Fp4 q = fp4_operand(data, scales, global, flags, scale_max, rows, cols, dev, "x", 16, 1);
  Tensor y = at::empty({rows, cols}, data.options().dtype(dtype));
  if (rows == 0 || cols == 0) return y;
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor Q = as_tensor(q);
  const nvfp4a_tensor D = {y.data_ptr(), t, nullptr, 0, nullptr, nullptr};
  check_status(nvfp4a_dequantize(&Q, q.ld, &D, as_int(cols, "cols"), as_int(rows, "rows"), as_int(cols, "cols"),
                                 at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_dequantize");
  return y;
}

// transpose of a 16 x 16-tile NVFP4 [rows, cols] -> (data [cols, rows / 2], scales in the same layout)
std::tuple<Tensor, Tensor> transpose(const Tensor &data, const Tensor &scales, int64_t flags, int64_t scale_max,
                                     int64_t cols) {
  TORCH_CHECK_VALUE(data.is_cuda(), "data must be a CUDA tensor");
  const at::Device dev = data.device();
  check_gpu(dev);
  TORCH_CHECK_VALUE(data.dim() == 2, "data must be 2-D");
  const int64_t rows = data.size(0);
  TORCH_CHECK_VALUE(rows % 16 == 0 && cols % 16 == 0, "rows (", rows, ") and cols (", cols, ") must be multiples of 16");
  const unsigned fl = (unsigned)flags & ~NVFP4A_GLOBAL_PER_ROW;
  const Fp4 s = fp4_operand(data, scales, std::nullopt, fl & ~NVFP4A_GLOBAL_AMAX, scale_max, rows, cols, dev, "x", 16, 1);
  const bool swz = fl & NVFP4A_SCALES_SWIZZLED;
  Tensor dt = at::empty({cols, rows / 2}, data.options());
  Tensor st = swz ? at::zeros({(cols + 127) / 128 * 128, (rows / 16 + 3) / 4 * 4}, scales.options())
                  : at::empty({cols, rows / 16}, scales.options());
  if (rows == 0 || cols == 0) return {dt, st};
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor S = as_tensor(s);
  const nvfp4a_tensor T = {dt.data_ptr(), NVFP4A_NVFP4, st.data_ptr(), swz ? 0 : (int)(rows / 16), nullptr, nullptr,
                           fl & ~NVFP4A_GLOBAL_AMAX, (int)scale_max};
  check_status(nvfp4a_transpose(&S, s.ld, &T, as_int(rows, "rows"), as_int(rows, "rows"), as_int(cols, "cols"),
                                at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_transpose");
  return {dt, st};
}

// linear scales [rows, >= cols / 16] <-> swizzled [roundup(rows, 128), roundup(cols / 16, 4)]
Tensor swizzle_scales(const Tensor &scales, int64_t rows, int64_t cols, bool to_swizzled) {
  TORCH_CHECK_VALUE(scales.is_cuda() && scales.dim() == 2, "scales must be a 2-D CUDA tensor");
  TORCH_CHECK_VALUE(cols % 16 == 0, "cols must be a multiple of 16");
  const c10::cuda::CUDAGuard guard(scales.device());
  const int64_t sc = cols / 16, pr = (rows + 127) / 128 * 128, pc = (sc + 3) / 4 * 4;
  Tensor src = scales.contiguous(), out;
  int ld;
  if (to_swizzled) {
    TORCH_CHECK_VALUE(scales.size(0) >= rows && scales.size(1) >= sc, "scales must be [>= ", rows, ", >= ", sc, "]");
    out = at::empty({pr, pc}, scales.options());
    ld = (int)src.size(1);
  } else {
    TORCH_CHECK_VALUE(scales.size(0) == pr && scales.size(1) == pc, "swizzled scales must be [", pr, ", ", pc, "]");
    out = at::empty({rows, sc}, scales.options());
    ld = (int)sc;
  }
  check_status(nvfp4a_swizzle_scales(src.data_ptr(), out.data_ptr(), ld, as_int(rows, "rows"), as_int(cols, "cols"),
                                     to_swizzled ? 1 : 0, at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_swizzle_scales");
  return out;
}

// out [T, N] = alpha * x @ W^T + bias + beta * out, W NVFP4 [N, K]; x [T, K] dense, NVFP4 (uint8 data with
// its scales / global / flags) or INT8 (int8 [T, K] with float32 x_scales [T, K / 16]).  K a multiple of 64.
Tensor linear(const Tensor &x, const OptT &x_scales, const OptT &x_global, int64_t x_flags, int64_t x_smax,
              const Tensor &w, const Tensor &w_scales, const OptT &w_global, int64_t w_flags, int64_t w_smax, int64_t k,
              const OptT &bias, std::optional<at::ScalarType> out_dtype, OptT out, double alpha, double beta,
              const OptT &amax_out, const OptT &w_img) {
  TORCH_CHECK_VALUE(w.is_cuda(), "weight must be a CUDA tensor, got a tensor on ", w.device());
  const at::Device dev = w.device();
  check_device(x, dev, "x");
  check_gpu(dev);
  TORCH_CHECK_VALUE(w.dim() == 2 && x.dim() == 2, "x and weight.data must be 2-D, got shapes ", x.sizes(), " and ", w.sizes());
  const bool x_q8 = x_scales.has_value() && x.scalar_type() == at::kChar, x_fp4 = x_scales.has_value() && !x_q8;
  const int64_t N = w.size(0), K = k, T = x.size(0);
  if (!x_fp4) TORCH_CHECK_VALUE(x.size(1) == K, "x has ", x.size(1), " input features, the weight ", K);
  TORCH_CHECK_VALUE(K % 64 == 0, "the number of input features (", K, ") must be a multiple of 64 (pad with zeros)");
  const Fp4 W = fp4_operand(w, w_scales, w_global, w_flags, w_smax, N, K, dev, "weight", 64, 4);
  Fp4 X{};
  Tensor xd;
  int ldx = 0;
  if (x_fp4) {
    X = fp4_operand(x, *x_scales, x_global, x_flags, x_smax, T, K, dev, "x", 64, 4);
    ldx = X.ld;
  } else if (x_q8) {
    TORCH_CHECK_TYPE(x_scales->scalar_type() == at::kFloat, "the scales of an int8 x must be torch.float32, got ",
                     x_scales->scalar_type());
    check_device(*x_scales, dev, "x scales");
    TORCH_CHECK_VALUE(x_scales->dim() == 2 && x_scales->size(0) == T && x_scales->size(1) == K / 16,
                      "the scales of an int8 x must have shape [", T, ", ", K / 16, "], got ", x_scales->sizes());
    xd = dense_2d(x, 64);
    X.scales = x_scales->stride(1) == 1 && (T <= 1 || x_scales->stride(0) % 4 == 0) &&
                       reinterpret_cast<uintptr_t>(x_scales->data_ptr()) % 16 == 0
                   ? *x_scales : x_scales->contiguous();
    ldx = (int)(T <= 1 ? K : xd.stride(0));
    X.ld_scale = (int)(T <= 1 ? K / 16 : X.scales.stride(0));
  } else {
    dense_dtype(x.scalar_type(), "x");
    xd = dense_2d(x, 64);
    ldx = (int)(T <= 1 ? K : xd.stride(0));
  }
  if (beta != 0.0) TORCH_CHECK_VALUE(out.has_value(), "beta != 0 accumulates into out, so out must be given");
  if (out) {
    check_device(*out, dev, "out");
    dense_dtype(out->scalar_type(), "out");
    TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out->scalar_type(), "out_dtype ", *out_dtype, " does not match out.dtype ",
                      out->scalar_type());
    TORCH_CHECK_VALUE(out->dim() == 2 && out->size(0) == T && out->size(1) == N, "out must have shape [", T, ", ", N,
                      "], got ", out->sizes());
    TORCH_CHECK_VALUE(out->stride(1) == 1 && (T <= 1 || out->stride(0) >= N),
                      "out must be row-contiguous (stride(1) == 1), got strides ", out->strides());
    TORCH_CHECK_VALUE(at::has_internal_overlap(*out) != at::MemOverlap::Yes, "out must not have internally overlapping memory");
    TORCH_CHECK_VALUE(at::get_overlap_status(*out, x) == at::MemOverlapStatus::No, "out must not share memory with x");
  } else {
    out = at::empty({T, N}, x.options().dtype(out_dtype.value_or(x_fp4 || x_q8 ? at::kBFloat16 : x.scalar_type())));
  }
  if (bias) {
    check_device(*bias, dev, "bias");
    dense_dtype(bias->scalar_type(), "bias");
    TORCH_CHECK_VALUE(bias->dim() == 1 && bias->size(0) == N && bias->stride(0) == 1,
                      "bias must be a contiguous vector of length ", N, ", got shape ", bias->sizes());
  }
  if (T == 0 || N == 0) return *out;
  const c10::cuda::CUDAGuard guard(dev);
  nvfp4a_tensor A = as_tensor(W);
  const nvfp4a_tensor B = x_fp4 ? as_tensor(X)
                          : x_q8 ? nvfp4a_tensor{xd.data_ptr(), NVFP4A_INT8, X.scales.data_ptr(), X.ld_scale, nullptr, nullptr}
                                 : nvfp4a_tensor{xd.data_ptr(), dense_dtype(x.scalar_type(), "x"), nullptr, 0, nullptr, nullptr};
  const nvfp4a_tensor D = {out->data_ptr(), dense_dtype(out->scalar_type(), "out"), nullptr, 0, nullptr,
                           scalar_ptr(amax_out, dev, "amax_out")};
  nvfp4a_tensor BIAS = {};
  if (bias) BIAS = {bias->data_ptr(), dense_dtype(bias->scalar_type(), "bias"), nullptr, 0, nullptr, nullptr};
  const int64_t ldd = T <= 1 ? N : out->stride(0);
  if (w_img) {  // the tensor-core GEMV on the weight's image (nvfp4a_tc_image A16 for a BF16 x, A8 for INT8 / NVFP4)
    check_device(*w_img, dev, "the weight image");
    TORCH_CHECK_VALUE(T <= 8 && w_img->scalar_type() == at::kByte &&
                      (size_t)w_img->numel() == nvfp4a_tc_image_bytes(NVFP4A_TC_A16, NVFP4A_NVFP4, as_int(N, "N"), (int)K),
                      "the weight image does not match N = ", N, ", K = ", K, " (or x has more than 8 rows)");
    A.data = w_img->data_ptr();
    check_status(nvfp4a_gemv_tc(as_int(N, "N"), as_int(T, "T"), as_int(K, "K"), (float)alpha, &A, &B, ldx, (float)beta, nullptr, 0,
                                &D, as_int(ldd, "ld(out)"), bias ? &BIAS : nullptr, at::cuda::getCurrentCUDAStream().stream()),
                 "nvfp4a_gemv_tc");
    return *out;
  }
  check_status(nvfp4a_gemm(as_int(N, "N"), as_int(T, "T"), as_int(K, "K"), (float)alpha, &A, W.ld, &B, ldx, (float)beta,
                           nullptr, 0, &D, as_int(ldd, "ld(out)"), bias ? &BIAS : nullptr,
                           at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_gemm");
  return *out;
}

// Tensor-core image (nvfp4a_tc_image) of kind `operand` [rows, K]: 0 / 2 (A16 / A8: the weight, NVFP4 data with its
// scales, for BF16 / INT8 and NVFP4 activations); 1 (B, the activations) NVFP4, INT8 (int8 with float32 scales
// [rows, K / 16]) or BF16.  uint8 [bytes].
Tensor tc_image(int64_t operand, const Tensor &x, const OptT &scales, int64_t flags, int64_t k) {
  TORCH_CHECK_VALUE(x.is_cuda(), "x must be a CUDA tensor, got a tensor on ", x.device());
  const at::Device dev = x.device();
  check_gpu(dev);
  TORCH_CHECK_VALUE(operand == NVFP4A_TC_A16 || operand == NVFP4A_TC_B || operand == NVFP4A_TC_A8,
                    "operand must be 0 (A16), 1 (B) or 2 (A8), got ", operand);
  TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2-D, got shape ", x.sizes());
  TORCH_CHECK_VALUE(k % 64 == 0, "the number of input features (", k, ") must be a multiple of 64 (pad with zeros)");
  const int64_t rows = x.size(0);
  nvfp4a_tensor src{};
  Fp4 f{};
  Tensor xd, xs;
  int ld = 0;
  if (x.scalar_type() == at::kByte) {
    TORCH_CHECK_VALUE(scales.has_value(), "an NVFP4 x needs its scales");
    f = fp4_operand(x, *scales, std::nullopt, flags & (NVFP4A_SCALES_SWIZZLED | NVFP4A_SCALES_UE5M3), 0, rows, k, dev, "x", 64, 4);
    src = as_tensor(f);
    ld = f.ld;
  } else if (x.scalar_type() == at::kChar) {
    TORCH_CHECK_VALUE(operand == NVFP4A_TC_B, "an INT8 operand is B");
    TORCH_CHECK_VALUE(scales.has_value() && scales->scalar_type() == at::kFloat && scales->dim() == 2 && scales->size(0) == rows &&
                      scales->size(1) == k / 16, "the scales of an int8 x must be float32 [", rows, ", ", k / 16, "]");
    check_device(*scales, dev, "x scales");
    xd = dense_2d(x, 64);
    xs = scales->contiguous();
    ld = (int)(rows <= 1 ? k : xd.stride(0));
    src = nvfp4a_tensor{xd.data_ptr(), NVFP4A_INT8, xs.data_ptr(), (int)(k / 16), nullptr, nullptr};
  } else {
    TORCH_CHECK_TYPE(x.scalar_type() == at::kBFloat16, "a dense x must be torch.bfloat16 (W4A16 image), got ", x.scalar_type());
    TORCH_CHECK_VALUE(operand == NVFP4A_TC_B, "a BF16 operand is B");
    xd = dense_2d(x, 64);
    ld = (int)(rows <= 1 ? k : xd.stride(0));
    src = nvfp4a_tensor{xd.data_ptr(), NVFP4A_BF16, nullptr, 0, nullptr, nullptr};
  }
  TORCH_CHECK_VALUE(x.scalar_type() == at::kByte || x.size(1) == k, "x has ", x.size(1), " columns, k = ", k);
  const c10::cuda::CUDAGuard guard(dev);
  Tensor img = at::empty({(int64_t)nvfp4a_tc_image_bytes((int)operand, src.dtype, as_int(rows, "rows"), as_int(k, "K"))},
                         x.options().dtype(at::kByte));
  check_status(nvfp4a_tc_image((int)operand, &src, ld, (int)rows, (int)k, img.data_ptr(), at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_tc_image");
  return img;
}

// nvfp4a_gemm_tc's workspace of the device and stream (zeroed once; every call leaves it zeroed)
const Tensor &tc_workspace(const at::Device &dev, cudaStream_t s) {
  static std::map<std::pair<int, cudaStream_t>, Tensor> ws;
  Tensor &w = ws[{dev.index(), s}];
  if (!w.defined()) w = at::zeros({(int64_t)nvfp4a_tc_workspace_bytes()}, at::TensorOptions().dtype(at::kByte).device(dev));
  return w;
}

// out [T, N] = alpha * x @ W^T + bias + beta * out from the images of W ([N, K], NVFP4) and x ([T, K]; x_dtype 0
// NVFP4, 1 INT8, 2 BF16), on tensor cores.
Tensor linear_tc(const Tensor &x_img, int64_t x_dtype, const OptT &x_global, int64_t x_flags, int64_t x_smax, int64_t T,
                 const Tensor &w_img, const OptT &w_global, int64_t w_flags, int64_t w_smax, int64_t N, int64_t K,
                 const OptT &bias, at::ScalarType out_dtype, OptT out, double alpha, double beta, const OptT &amax_out) {
  TORCH_CHECK_VALUE(w_img.is_cuda(), "the weight image must be a CUDA tensor");
  const at::Device dev = w_img.device();
  check_device(x_img, dev, "the x image");
  TORCH_CHECK_VALUE(x_dtype >= 0 && x_dtype <= 2, "x_dtype must be 0 (NVFP4), 1 (INT8) or 2 (BF16)");
  const nvfp4a_dtype xt = x_dtype == 0 ? NVFP4A_NVFP4 : x_dtype == 1 ? NVFP4A_INT8 : NVFP4A_BF16;
  TORCH_CHECK_VALUE(x_img.scalar_type() == at::kByte && w_img.scalar_type() == at::kByte &&
                    (size_t)x_img.numel() == nvfp4a_tc_image_bytes(NVFP4A_TC_B, xt, as_int(T, "T"), as_int(K, "K")) &&
                    (size_t)w_img.numel() == nvfp4a_tc_image_bytes(NVFP4A_TC_A16, NVFP4A_NVFP4, as_int(N, "N"), (int)K),
                    "the images do not match T = ", T, ", N = ", N, ", K = ", K);
  const auto global = [&](const OptT &g, int64_t flags, int64_t rows, const char *what) -> const float * {
    if (!g) return nullptr;
    check_device(*g, dev, what);
    TORCH_CHECK_TYPE(g->scalar_type() == at::kFloat && g->is_contiguous(), what, " must be a contiguous torch.float32");
    TORCH_CHECK_VALUE(g->numel() == ((flags & NVFP4A_GLOBAL_PER_ROW) ? rows : 1), what, " has ", g->numel(), " elements");
    return g->data_ptr<float>();
  };
  if (beta != 0.0) TORCH_CHECK_VALUE(out.has_value(), "beta != 0 accumulates into out, so out must be given");
  if (out) {
    check_device(*out, dev, "out");
    TORCH_CHECK_VALUE(out->dim() == 2 && out->size(0) == T && out->size(1) == N && out->stride(1) == 1 && (T <= 1 || out->stride(0) >= N),
                      "out must be a row-contiguous [", T, ", ", N, "], got ", out->sizes());
  } else {
    out = at::empty({T, N}, w_img.options().dtype(out_dtype));
  }
  if (bias) {
    check_device(*bias, dev, "bias");
    TORCH_CHECK_VALUE(bias->dim() == 1 && bias->size(0) == N && bias->stride(0) == 1, "bias must be a contiguous vector of length ", N);
  }
  if (T == 0 || N == 0) return *out;
  const c10::cuda::CUDAGuard guard(dev);
  const cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
  const nvfp4a_tensor A{w_img.data_ptr(), NVFP4A_NVFP4, nullptr, 0, global(w_global, w_flags, N, "weight.global"), nullptr,
                        (unsigned)w_flags, (int)w_smax};
  const nvfp4a_tensor B{x_img.data_ptr(), xt, nullptr, 0, global(x_global, x_flags, T, "x.global"), nullptr,
                        (unsigned)(x_dtype == 0 ? x_flags : 0), (int)x_smax};
  const nvfp4a_tensor D{out->data_ptr(), dense_dtype(out->scalar_type(), "out"), nullptr, 0, nullptr, scalar_ptr(amax_out, dev, "amax_out")};
  nvfp4a_tensor BIAS = {};
  if (bias) BIAS = {bias->data_ptr(), dense_dtype(bias->scalar_type(), "bias"), nullptr, 0, nullptr, nullptr};
  check_status(nvfp4a_gemm_tc((int)N, (int)T, (int)K, (float)alpha, &A, &B, (float)beta, nullptr, 0, &D, (int)(T <= 1 ? N : out->stride(0)),
                              bias ? &BIAS : nullptr, tc_workspace(dev, s).data_ptr(), s),
               "nvfp4a_gemm_tc");
  return *out;
}

// out [R, N] = alpha * x_g @ W_g^T + bias_g + beta * out for every group g of seg (int32 [groups + 1] on the device, seg[0] = 0; x_g the rows
// seg[g] .. seg[g + 1] of x), W the stacked weight [groups * N, K] (NVFP4; W_g = rows [g N, (g + 1) N); linear or swizzled scales), bias [groups * N];
// x [R, K] dense, NVFP4 (uint8 data with its scales / global / flags) or INT8 (int8 with float32 x_scales [R, K / 16]).  Rows of out past
// seg[groups] are zero (a fresh out) or as given.  K a multiple of 64.  The row counts stay on the device: no synchronization.
Tensor grouped_linear(const Tensor &x, const OptT &x_scales, const OptT &x_global, int64_t x_flags, int64_t x_smax, const Tensor &seg,
                      const Tensor &w, const Tensor &w_scales, const OptT &w_global, int64_t w_flags, int64_t w_smax, int64_t k, int64_t groups,
                      const OptT &bias, std::optional<at::ScalarType> out_dtype, OptT out, double alpha, double beta, const OptT &amax_out) {
  TORCH_CHECK_VALUE(w.is_cuda(), "weight must be a CUDA tensor, got a tensor on ", w.device());
  const at::Device dev = w.device();
  check_device(x, dev, "x");
  check_device(seg, dev, "seg");
  check_gpu(dev);
  TORCH_CHECK_TYPE(seg.scalar_type() == at::kInt, "seg must be torch.int32, got ", seg.scalar_type());
  TORCH_CHECK_VALUE(w.dim() == 2 && x.dim() == 2, "x and weight.data must be 2-D, got shapes ", x.sizes(), " and ", w.sizes());
  const bool x_q8 = x_scales.has_value() && x.scalar_type() == at::kChar, x_fp4 = x_scales.has_value() && !x_q8;
  const int64_t K = k, R = x.size(0);
  TORCH_CHECK_VALUE(groups >= 1 && w.size(0) % groups == 0, "the weight has ", w.size(0), " rows, not a multiple of groups = ", groups);
  const int64_t N = w.size(0) / groups;
  TORCH_CHECK_VALUE(K % 64 == 0, "the number of input features (", K, ") must be a multiple of 64");
  if (!x_fp4) TORCH_CHECK_VALUE(x.size(1) == K, "x has ", x.size(1), " input features, the weight ", K);
  TORCH_CHECK_VALUE(seg.dim() == 1 && seg.size(0) == groups + 1 && seg.is_contiguous(), "seg must be a contiguous int32 [", groups + 1, "]");
  const Fp4 W = fp4_operand(w, w_scales, w_global, w_flags, w_smax, w.size(0), K, dev, "weight", 64, 4);
  Fp4 X{};
  Tensor xd;
  int ldx = 0;
  if (x_fp4) {
    X = fp4_operand(x, *x_scales, x_global, x_flags, x_smax, R, K, dev, "x", 64, 4);
    ldx = X.ld;
  } else if (x_q8) {
    TORCH_CHECK_TYPE(x_scales->scalar_type() == at::kFloat, "the scales of an int8 x must be torch.float32, got ", x_scales->scalar_type());
    check_device(*x_scales, dev, "x scales");
    TORCH_CHECK_VALUE(x_scales->dim() == 2 && x_scales->size(0) == R && x_scales->size(1) == K / 16,
                      "the scales of an int8 x must have shape [", R, ", ", K / 16, "], got ", x_scales->sizes());
    xd = dense_2d(x, 64);
    X.scales = x_scales->stride(1) == 1 && (R <= 1 || x_scales->stride(0) % 4 == 0) &&
                       reinterpret_cast<uintptr_t>(x_scales->data_ptr()) % 16 == 0
                   ? *x_scales : x_scales->contiguous();
    ldx = (int)(R <= 1 ? K : xd.stride(0));
    X.ld_scale = (int)(R <= 1 ? K / 16 : X.scales.stride(0));
  } else {
    dense_dtype(x.scalar_type(), "x");
    xd = dense_2d(x, 64);
    ldx = (int)(R <= 1 ? K : xd.stride(0));
  }
  if (beta != 0.0) TORCH_CHECK_VALUE(out.has_value(), "beta != 0 accumulates into out, so out must be given");
  if (out) {
    check_device(*out, dev, "out");
    dense_dtype(out->scalar_type(), "out");
    TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out->scalar_type(), "out_dtype ", *out_dtype, " does not match out.dtype ", out->scalar_type());
    TORCH_CHECK_VALUE(out->dim() == 2 && out->size(0) == R && out->size(1) == N && out->stride(1) == 1 && (R <= 1 || out->stride(0) >= N),
                      "out must be a row-contiguous [", R, ", ", N, "], got ", out->sizes());
    TORCH_CHECK_VALUE(at::has_internal_overlap(*out) != at::MemOverlap::Yes, "out must not have internally overlapping memory");
  } else {
    out = at::zeros({R, N}, x.options().dtype(out_dtype.value_or(x_fp4 || x_q8 ? at::kBFloat16 : x.scalar_type())));
  }
  if (bias) {
    check_device(*bias, dev, "bias");
    dense_dtype(bias->scalar_type(), "bias");
    TORCH_CHECK_VALUE(bias->dim() == 1 && bias->size(0) == groups * N && bias->stride(0) == 1, "bias must be a contiguous vector of length ",
                      groups * N, ", got shape ", bias->sizes());
  }
  if (R == 0 || N == 0) return *out;
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor A = as_tensor(W);
  const nvfp4a_tensor B = x_fp4 ? as_tensor(X)
                          : x_q8 ? nvfp4a_tensor{xd.data_ptr(), NVFP4A_INT8, X.scales.data_ptr(), X.ld_scale, nullptr, nullptr}
                                 : nvfp4a_tensor{xd.data_ptr(), dense_dtype(x.scalar_type(), "x"), nullptr, 0, nullptr, nullptr};
  const nvfp4a_tensor D = {out->data_ptr(), dense_dtype(out->scalar_type(), "out"), nullptr, 0, nullptr, scalar_ptr(amax_out, dev, "amax_out")};
  nvfp4a_tensor BIAS = {};
  if (bias) BIAS = {bias->data_ptr(), dense_dtype(bias->scalar_type(), "bias"), nullptr, 0, nullptr, nullptr};
  check_status(nvfp4a_gemm_grouped((int)groups, as_int(N, "N"), (int)K, as_int(R, "rows"), (float)alpha, &A, W.ld, &B, ldx, (float)beta, nullptr, 0,
                                   &D, (int)(R <= 1 ? N : out->stride(0)), bias ? &BIAS : nullptr, seg.data_ptr<int>(),
                                   at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_gemm_grouped");
  return *out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVFP4EffMA: NVFP4 GEMM / GEMV / quantization for GPUs without native FP4 (sm_50..sm_90)";
  m.attr("__version__") = NVFP4EFFMA_VERSION;
  m.attr("SCALES_SWIZZLED") = (int)NVFP4A_SCALES_SWIZZLED;
  m.attr("SCALES_UE5M3") = (int)NVFP4A_SCALES_UE5M3;
  m.attr("GLOBAL_AMAX") = (int)NVFP4A_GLOBAL_AMAX;
  m.attr("GLOBAL_PER_ROW") = (int)NVFP4A_GLOBAL_PER_ROW;
  m.def("quantize_ex", &quantize_ex, py::arg("x"), py::arg("rowwise"), py::arg("columnwise"), py::arg("g_row"),
        py::arg("g_col"), py::arg("flags"), py::arg("scale_max"), py::arg("tile_rows"), py::arg("four_over_six"),
        py::arg("compute_amax"), py::arg("amax") = py::none());
  m.def("quantize_int8", &quantize_int8, py::arg("x"));
  m.def("dequantize", &dequantize, py::arg("data"), py::arg("scales"), py::arg("global_"), py::arg("flags"),
        py::arg("scale_max"), py::arg("cols"), py::arg("dtype"));
  m.def("transpose", &transpose, py::arg("data"), py::arg("scales"), py::arg("flags"), py::arg("scale_max"), py::arg("cols"));
  m.def("swizzle_scales", &swizzle_scales, py::arg("scales"), py::arg("rows"), py::arg("cols"), py::arg("to_swizzled"));
  m.def("linear", &linear, py::arg("x"), py::arg("x_scales"), py::arg("x_global"), py::arg("x_flags"), py::arg("x_smax"),
        py::arg("w"), py::arg("w_scales"), py::arg("w_global"), py::arg("w_flags"), py::arg("w_smax"), py::arg("k"),
        py::arg("bias") = py::none(), py::arg("out_dtype") = py::none(), py::arg("out") = py::none(),
        py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("amax_out") = py::none(), py::arg("w_img") = py::none());
  m.def("grouped_linear", &grouped_linear, py::arg("x"), py::arg("x_scales"), py::arg("x_global"), py::arg("x_flags"), py::arg("x_smax"),
        py::arg("seg"), py::arg("w"), py::arg("w_scales"), py::arg("w_global"), py::arg("w_flags"), py::arg("w_smax"), py::arg("k"),
        py::arg("groups"), py::arg("bias") = py::none(), py::arg("out_dtype") = py::none(), py::arg("out") = py::none(),
        py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("amax_out") = py::none());
  m.def("tc_image", &tc_image, py::arg("operand"), py::arg("x"), py::arg("scales"), py::arg("flags"), py::arg("k"));
  m.def("linear_tc", &linear_tc, py::arg("x_img"), py::arg("x_dtype"), py::arg("x_global"), py::arg("x_flags"), py::arg("x_smax"),
        py::arg("T"), py::arg("w_img"), py::arg("w_global"), py::arg("w_flags"), py::arg("w_smax"), py::arg("N"), py::arg("K"),
        py::arg("bias") = py::none(), py::arg("out_dtype") = at::kBFloat16, py::arg("out") = py::none(), py::arg("alpha") = 1.0,
        py::arg("beta") = 0.0, py::arg("amax_out") = py::none());
}

// ---- the operations as PyTorch operators (torch.ops.nvfp4effma.*) ---------------------------------------------------------------------------
// Each call is one opaque node for torch.compile and CUDA graphs: a schema, a CUDA kernel (the functions above) and a Meta kernel that only computes
// the results' shapes and dtypes.  The calls enqueue on the current stream, allocate through PyTorch's allocator and read nothing back to the host, so
// they can be captured in a CUDA graph.  Nothing is an optional mutated argument (functionalization cannot lower those): the amax of a result is a
// result (`want_amax`: a fresh float32 [1], else None; the Python layer max-reduces it into a caller's buffer), and the `_out` variants, which write into
// a given `out` (beta accumulates into it), return only that amax: an operator may not return its input.
namespace {

at::ScalarType linear_out_dtype(const Tensor &x, bool blocked, const std::optional<at::ScalarType> &t) {
  return t.value_or(blocked ? at::kBFloat16 : x.scalar_type());
}
OptT amax_buffer(bool want, const at::Device &dev) {
  return want ? OptT(at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(dev))) : std::nullopt;
}
OptT amax_meta(bool want, const Tensor &like) { return want ? OptT(at::empty({1}, like.options().dtype(at::kFloat))) : std::nullopt; }

std::tuple<Tensor, OptT> linear_new(const Tensor &x, const OptT &xs, const OptT &xg, int64_t xf, int64_t xm, const Tensor &w, const Tensor &ws,
                                    const OptT &wg, int64_t wf, int64_t wm, int64_t k, const OptT &bias, std::optional<at::ScalarType> out_dtype,
                                    double alpha, bool want_amax, const OptT &w_img) {
  OptT amax = amax_buffer(want_amax, x.device());
  Tensor y = linear(x, xs, xg, xf, xm, w, ws, wg, wf, wm, k, bias, out_dtype, std::nullopt, alpha, 0.0, amax, w_img);
  return {y, amax};
}
OptT linear_into(const Tensor &x, const OptT &xs, const OptT &xg, int64_t xf, int64_t xm, const Tensor &w, const Tensor &ws, const OptT &wg,
                 int64_t wf, int64_t wm, int64_t k, const OptT &bias, const Tensor &out, double alpha, double beta, bool want_amax,
                 const OptT &w_img) {
  OptT amax = amax_buffer(want_amax, x.device());
  linear(x, xs, xg, xf, xm, w, ws, wg, wf, wm, k, bias, std::nullopt, out, alpha, beta, amax, w_img);
  return amax;
}
std::tuple<Tensor, OptT> linear_meta(const Tensor &x, const OptT &xs, const OptT &, int64_t, int64_t, const Tensor &w, const Tensor &, const OptT &,
                                     int64_t, int64_t, int64_t, const OptT &, std::optional<at::ScalarType> out_dtype, double, bool want_amax,
                                     const OptT &) {
  return {at::empty({x.size(0), w.size(0)}, x.options().dtype(linear_out_dtype(x, xs.has_value(), out_dtype))), amax_meta(want_amax, x)};
}
OptT linear_into_meta(const Tensor &x, const OptT &, const OptT &, int64_t, int64_t, const Tensor &, const Tensor &, const OptT &, int64_t, int64_t,
                      int64_t, const OptT &, const Tensor &, double, double, bool want_amax, const OptT &) {
  return amax_meta(want_amax, x);
}

std::tuple<Tensor, OptT> grouped_new(const Tensor &x, const OptT &xs, const OptT &xg, int64_t xf, int64_t xm, const Tensor &seg, const Tensor &w,
                                     const Tensor &ws, const OptT &wg, int64_t wf, int64_t wm, int64_t k, int64_t groups, const OptT &bias,
                                     std::optional<at::ScalarType> out_dtype, double alpha, bool want_amax) {
  OptT amax = amax_buffer(want_amax, x.device());
  Tensor y = grouped_linear(x, xs, xg, xf, xm, seg, w, ws, wg, wf, wm, k, groups, bias, out_dtype, std::nullopt, alpha, 0.0, amax);
  return {y, amax};
}
OptT grouped_into(const Tensor &x, const OptT &xs, const OptT &xg, int64_t xf, int64_t xm, const Tensor &seg, const Tensor &w, const Tensor &ws,
                  const OptT &wg, int64_t wf, int64_t wm, int64_t k, int64_t groups, const OptT &bias, const Tensor &out, double alpha, double beta,
                  bool want_amax) {
  OptT amax = amax_buffer(want_amax, x.device());
  grouped_linear(x, xs, xg, xf, xm, seg, w, ws, wg, wf, wm, k, groups, bias, std::nullopt, out, alpha, beta, amax);
  return amax;
}
std::tuple<Tensor, OptT> grouped_meta(const Tensor &x, const OptT &xs, const OptT &, int64_t, int64_t, const Tensor &, const Tensor &w, const Tensor &,
                                      const OptT &, int64_t, int64_t, int64_t, int64_t groups, const OptT &, std::optional<at::ScalarType> out_dtype,
                                      double, bool want_amax) {
  return {at::empty({x.size(0), w.size(0) / groups}, x.options().dtype(linear_out_dtype(x, xs.has_value(), out_dtype))), amax_meta(want_amax, x)};
}
OptT grouped_into_meta(const Tensor &x, const OptT &, const OptT &, int64_t, int64_t, const Tensor &, const Tensor &, const Tensor &, const OptT &,
                       int64_t, int64_t, int64_t, int64_t, const OptT &, const Tensor &, double, double, bool want_amax) {
  return amax_meta(want_amax, x);
}

std::tuple<Tensor, OptT> linear_tc_new(const Tensor &xi, int64_t xt, const OptT &xg, int64_t xf, int64_t xm, int64_t T, const Tensor &wi,
                                       const OptT &wg, int64_t wf, int64_t wm, int64_t N, int64_t K, const OptT &bias, at::ScalarType out_dtype,
                                       double alpha, bool want_amax) {
  OptT amax = amax_buffer(want_amax, xi.device());
  Tensor y = linear_tc(xi, xt, xg, xf, xm, T, wi, wg, wf, wm, N, K, bias, out_dtype, std::nullopt, alpha, 0.0, amax);
  return {y, amax};
}
OptT linear_tc_into(const Tensor &xi, int64_t xt, const OptT &xg, int64_t xf, int64_t xm, int64_t T, const Tensor &wi, const OptT &wg, int64_t wf,
                    int64_t wm, int64_t N, int64_t K, const OptT &bias, const Tensor &out, double alpha, double beta, bool want_amax) {
  OptT amax = amax_buffer(want_amax, xi.device());
  linear_tc(xi, xt, xg, xf, xm, T, wi, wg, wf, wm, N, K, bias, out.scalar_type(), out, alpha, beta, amax);
  return amax;
}
std::tuple<Tensor, OptT> linear_tc_meta(const Tensor &xi, int64_t, const OptT &, int64_t, int64_t, int64_t T, const Tensor &, const OptT &, int64_t,
                                        int64_t, int64_t N, int64_t, const OptT &, at::ScalarType out_dtype, double, bool want_amax) {
  return {at::empty({T, N}, xi.options().dtype(out_dtype)), amax_meta(want_amax, xi)};
}
OptT linear_tc_into_meta(const Tensor &xi, int64_t, const OptT &, int64_t, int64_t, int64_t, const Tensor &, const OptT &, int64_t, int64_t, int64_t,
                         int64_t, const OptT &, const Tensor &, double, double, bool want_amax) {
  return amax_meta(want_amax, xi);
}

using Quantized = std::tuple<OptT, OptT, OptT, OptT, OptT, OptT, OptT>;

// quantize_ex with its globals and amax as results instead of mutated arguments (what functionalization wants): the input globals are copied, the
// copies receive the amaxes (compute_amax), and want_amax gives a fresh float32 [1] holding max |x|.
Quantized quantize_ex_f(const Tensor &x, bool rowwise, bool columnwise, const OptT &g_row, const OptT &g_col, int64_t flags, int64_t scale_max,
                        int64_t tile_rows, int64_t four_over_six, bool compute_amax, bool want_amax) {
  OptT gr, gc, am;
  if (g_row) gr = g_row->clone();
  if (g_col) gc = g_col->clone();
  if (want_amax) am = at::zeros({1}, x.options().dtype(at::kFloat));
  const auto [dr, sr, dc, scl] = quantize_ex(x, rowwise, columnwise, gr, gc, flags, scale_max, tile_rows, four_over_six, compute_amax, am);
  return {dr, sr, dc, scl, gr, gc, am};
}

Quantized quantize_ex_meta(const Tensor &x, bool rowwise, bool columnwise, const OptT &g_row, const OptT &g_col, int64_t flags, int64_t, int64_t,
                           int64_t, bool, bool want_amax) {
  TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2-D, got shape ", x.sizes());
  const int64_t rows = x.size(0), cols = x.size(1), rows16 = (rows + 15) / 16 * 16;
  const bool swz = flags & NVFP4A_SCALES_SWIZZLED;
  const auto sdt = (flags & NVFP4A_SCALES_UE5M3) ? at::kByte : at::kFloat8_e4m3fn;
  const auto scales_for = [&](int64_t r, int64_t c) {
    const int64_t sc = (c + 15) / 16;
    return swz ? at::empty({(r + 127) / 128 * 128, (sc + 3) / 4 * 4}, x.options().dtype(sdt)) : at::empty({r, sc}, x.options().dtype(sdt));
  };
  OptT dr, sr, dc, scl;
  if (rowwise) { dr = at::empty({rows, cols / 2}, x.options().dtype(at::kByte)); sr = scales_for(rows, cols); }
  if (columnwise) { dc = at::empty({cols, rows16 / 2}, x.options().dtype(at::kByte)); scl = scales_for(cols, rows); }
  OptT gr, gc, am;
  if (g_row) gr = at::empty_like(*g_row);
  if (g_col) gc = at::empty_like(*g_col);
  if (want_amax) am = at::empty({1}, x.options().dtype(at::kFloat));
  return {dr, sr, dc, scl, gr, gc, am};
}
std::tuple<Tensor, Tensor> quantize_int8_meta(const Tensor &x) {
  TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2-D, got shape ", x.sizes());
  return {at::empty({x.size(0), x.size(1)}, x.options().dtype(at::kChar)), at::empty({x.size(0), x.size(1) / 16}, x.options().dtype(at::kFloat))};
}
Tensor dequantize_meta(const Tensor &data, const Tensor &, const OptT &, int64_t, int64_t, int64_t cols, at::ScalarType dtype) {
  return at::empty({data.size(0), cols}, data.options().dtype(dtype));
}
std::tuple<Tensor, Tensor> transpose_meta(const Tensor &data, const Tensor &scales, int64_t flags, int64_t, int64_t cols) {
  const int64_t rows = data.size(0);
  const bool swz = (unsigned)flags & NVFP4A_SCALES_SWIZZLED;
  return {at::empty({cols, rows / 2}, data.options()),
          swz ? at::empty({(cols + 127) / 128 * 128, (rows / 16 + 3) / 4 * 4}, scales.options()) : at::empty({cols, rows / 16}, scales.options())};
}
Tensor swizzle_scales_meta(const Tensor &scales, int64_t rows, int64_t cols, bool to_swizzled) {
  const int64_t sc = cols / 16, pr = (rows + 127) / 128 * 128, pc = (sc + 3) / 4 * 4;
  return to_swizzled ? at::empty({pr, pc}, scales.options()) : at::empty({rows, sc}, scales.options());
}
Tensor tc_image_meta(int64_t operand, const Tensor &x, const OptT &, int64_t, int64_t k) {
  const nvfp4a_dtype t = x.scalar_type() == at::kByte ? NVFP4A_NVFP4 : x.scalar_type() == at::kChar ? NVFP4A_INT8 : NVFP4A_BF16;
  return at::empty({(int64_t)nvfp4a_tc_image_bytes((int)operand, t, (int)x.size(0), (int)k)}, x.options().dtype(at::kByte));
}

}  // namespace

TORCH_LIBRARY(nvfp4effma, m) {
  m.def("quantize_ex(Tensor x, bool rowwise, bool columnwise, Tensor? g_row, Tensor? g_col, int flags, int scale_max, int tile_rows, "
        "int four_over_six, bool compute_amax, bool want_amax) -> (Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)");
  m.def("quantize_int8(Tensor x) -> (Tensor, Tensor)");
  m.def("dequantize(Tensor data, Tensor scales, Tensor? global_, int flags, int scale_max, int cols, ScalarType dtype) -> Tensor");
  m.def("transpose(Tensor data, Tensor scales, int flags, int scale_max, int cols) -> (Tensor, Tensor)");
  m.def("swizzle_scales(Tensor scales, int rows, int cols, bool to_swizzled) -> Tensor");
  m.def("linear(Tensor x, Tensor? x_scales, Tensor? x_global, int x_flags, int x_smax, Tensor w, Tensor w_scales, Tensor? w_global, int w_flags, "
        "int w_smax, int k, Tensor? bias, ScalarType? out_dtype, float alpha, bool want_amax, Tensor? w_img) -> (Tensor, Tensor?)");
  m.def("linear_out(Tensor x, Tensor? x_scales, Tensor? x_global, int x_flags, int x_smax, Tensor w, Tensor w_scales, Tensor? w_global, int w_flags, "
        "int w_smax, int k, Tensor? bias, Tensor(a!) out, float alpha, float beta, bool want_amax, Tensor? w_img) -> Tensor?");
  m.def("grouped_linear(Tensor x, Tensor? x_scales, Tensor? x_global, int x_flags, int x_smax, Tensor seg, Tensor w, Tensor w_scales, "
        "Tensor? w_global, int w_flags, int w_smax, int k, int groups, Tensor? bias, ScalarType? out_dtype, float alpha, bool want_amax) "
        "-> (Tensor, Tensor?)");
  m.def("grouped_linear_out(Tensor x, Tensor? x_scales, Tensor? x_global, int x_flags, int x_smax, Tensor seg, Tensor w, Tensor w_scales, "
        "Tensor? w_global, int w_flags, int w_smax, int k, int groups, Tensor? bias, Tensor(a!) out, float alpha, float beta, bool want_amax) "
        "-> Tensor?");
  m.def("tc_image(int operand, Tensor x, Tensor? scales, int flags, int k) -> Tensor");
  m.def("linear_tc(Tensor x_img, int x_dtype, Tensor? x_global, int x_flags, int x_smax, int T, Tensor w_img, Tensor? w_global, int w_flags, "
        "int w_smax, int N, int K, Tensor? bias, ScalarType out_dtype, float alpha, bool want_amax) -> (Tensor, Tensor?)");
  m.def("linear_tc_out(Tensor x_img, int x_dtype, Tensor? x_global, int x_flags, int x_smax, int T, Tensor w_img, Tensor? w_global, int w_flags, "
        "int w_smax, int N, int K, Tensor? bias, Tensor(a!) out, float alpha, float beta, bool want_amax) -> Tensor?");
}

TORCH_LIBRARY_IMPL(nvfp4effma, CUDA, m) {
  m.impl("quantize_ex", &quantize_ex_f);
  m.impl("quantize_int8", &quantize_int8);
  m.impl("dequantize", &dequantize);
  m.impl("transpose", &transpose);
  m.impl("swizzle_scales", &swizzle_scales);
  m.impl("linear", &linear_new);
  m.impl("linear_out", &linear_into);
  m.impl("grouped_linear", &grouped_new);
  m.impl("grouped_linear_out", &grouped_into);
  m.impl("tc_image", &tc_image);
  m.impl("linear_tc", &linear_tc_new);
  m.impl("linear_tc_out", &linear_tc_into);
}

TORCH_LIBRARY_IMPL(nvfp4effma, Meta, m) {
  m.impl("quantize_ex", &quantize_ex_meta);
  m.impl("quantize_int8", &quantize_int8_meta);
  m.impl("dequantize", &dequantize_meta);
  m.impl("transpose", &transpose_meta);
  m.impl("swizzle_scales", &swizzle_scales_meta);
  m.impl("linear", &linear_meta);
  m.impl("linear_out", &linear_into_meta);
  m.impl("grouped_linear", &grouped_meta);
  m.impl("grouped_linear_out", &grouped_into_meta);
  m.impl("tc_image", &tc_image_meta);
  m.impl("linear_tc", &linear_tc_meta);
  m.impl("linear_tc_out", &linear_tc_into_meta);
}
