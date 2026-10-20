// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// PyTorch bindings of NVFP4EffMA (include/nvfp4effma.h).  Every argument is checked here, before any
// pointer reaches the library: wrong dtype -> TypeError, wrong device / shape / layout / alignment ->
// ValueError, unsupported GPU -> NotImplementedError, library or CUDA failure -> RuntimeError with the
// library's description (nvfp4a_last_error).  Calls run on the current CUDA stream of the tensors' device.
//
// An NVFP4 matrix [rows, cols] is three tensors: data uint8 [rows, cols / 2] (two values per byte, column
// 2c in the low nibble), scales float8_e4m3fn (or uint8) [rows, cols / 16], and a float32 global scale with
// one element (value = e2m1 * scale * global_scale).
#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
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

// A 2-D NVFP4 operand as the library takes it: data rows 16-byte aligned with a row stride (in values) that
// is a multiple of `align`, scale rows with a stride that is a multiple of `salign` bytes.  Anything else
// is copied once into a contiguous layout.
struct Fp4 { Tensor data, scales; int ld, ld_scale; };
Fp4 fp4_operand(const Tensor &data, const Tensor &scales, int64_t rows, int64_t cols, const at::Device &dev,
                const char *what, int align, int salign) {
  const std::string w(what);
  check_device(data, dev, (w + ".data").c_str());
  check_device(scales, dev, (w + ".scales").c_str());
  TORCH_CHECK_TYPE(data.scalar_type() == at::kByte, what, ".data must be torch.uint8 (two E2M1 values per byte), got ",
                   data.scalar_type());
  TORCH_CHECK_TYPE(scales.scalar_type() == at::kFloat8_e4m3fn || scales.scalar_type() == at::kByte, what,
                   ".scales must be torch.float8_e4m3fn (or its bytes as torch.uint8), got ", scales.scalar_type());
  TORCH_CHECK_VALUE(data.dim() == 2 && data.size(0) == rows && data.size(1) == cols / 2, what, ".data must have shape [",
                    rows, ", ", cols / 2, "], got ", data.sizes());
  TORCH_CHECK_VALUE(scales.dim() == 2 && scales.size(0) == rows && scales.size(1) == cols / 16, what,
                    ".scales must have shape [", rows, ", ", cols / 16, "], got ", scales.sizes());
  Fp4 r{data, scales.view(at::kByte), 0, 0};
  // row stride (bytes) a multiple of step, base address aligned to palign
  const auto ok = [](const Tensor &t, int64_t step, uintptr_t palign) {
    return t.stride(1) == 1 && (t.size(0) <= 1 || t.stride(0) % step == 0) && reinterpret_cast<uintptr_t>(t.data_ptr()) % palign == 0;
  };
  if (!ok(r.data, align / 2, 16)) r.data = r.data.contiguous();
  if (!ok(r.scales, salign, salign)) r.scales = r.scales.contiguous();
  r.ld = (int)(rows <= 1 ? cols : r.data.stride(0) * 2);
  r.ld_scale = (int)(rows <= 1 ? cols / 16 : r.scales.stride(0));
  if (r.ld % align) { r.data = r.data.contiguous(); r.ld = (int)cols; }
  if (r.ld_scale % salign) { r.scales = r.scales.contiguous(); r.ld_scale = (int)(cols / 16); }
  return r;
}

}  // namespace

// x [rows, cols] (BF16 / FP16 / FP32, cols a multiple of 16) -> (data, scales) with the given global scale;
// amax (optional, float32 [1]) gets max |x|.
std::tuple<Tensor, Tensor> quantize(const Tensor &x, const Tensor &global_scale, const OptT &amax) {
  TORCH_CHECK_VALUE(x.is_cuda(), "x must be a CUDA tensor, got a tensor on ", x.device());
  const at::Device dev = x.device();
  check_gpu(dev);
  const nvfp4a_dtype t = dense_dtype(x.scalar_type(), "x");
  TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2-D, got shape ", x.sizes());
  const int64_t rows = x.size(0), cols = x.size(1);
  TORCH_CHECK_VALUE(cols % 16 == 0, "the last dimension (", cols, ") must be a multiple of 16 (the NVFP4 block)");
  const float *g = scalar_ptr(global_scale, dev, "global_scale");
  float *am = scalar_ptr(amax, dev, "amax");
  Tensor xs = x;
  if (x.stride(1) != 1 || (rows > 1 && x.stride(0) % 16) || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) xs = x.contiguous();
  Tensor data = at::empty({rows, cols / 2}, x.options().dtype(at::kByte));
  Tensor scales = at::empty({rows, cols / 16}, x.options().dtype(at::kFloat8_e4m3fn));
  if (rows == 0 || cols == 0) return {data, scales};
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor S = {xs.data_ptr(), t, nullptr, 0, nullptr, nullptr};
  const nvfp4a_tensor Q = {data.data_ptr(), NVFP4A_NVFP4, scales.data_ptr(), (int)(cols / 16), g, am};
  check_status(nvfp4a_quantize(&S, as_int(rows <= 1 ? cols : xs.stride(0), "ld(x)"), &Q, as_int(cols, "cols"),
                               as_int(rows, "rows"), as_int(cols, "cols"), at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_quantize");
  return {data, scales};
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
  Tensor xs = x;
  if (x.stride(1) != 1 || (rows > 1 && x.stride(0) % 16) || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) xs = x.contiguous();
  Tensor q = at::empty({rows, cols}, x.options().dtype(at::kChar));
  Tensor scales = at::empty({rows, cols / 16}, x.options().dtype(at::kFloat));
  if (rows == 0 || cols == 0) return {q, scales};
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor S = {xs.data_ptr(), t, nullptr, 0, nullptr, nullptr};
  const nvfp4a_tensor Q = {q.data_ptr(), NVFP4A_INT8, scales.data_ptr(), (int)(cols / 16), nullptr, nullptr};
  check_status(nvfp4a_quantize(&S, as_int(rows <= 1 ? cols : xs.stride(0), "ld(x)"), &Q, as_int(cols, "cols"),
                               as_int(rows, "rows"), as_int(cols, "cols"), at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_quantize");
  return {q, scales};
}

// (data, scales, global_scale) of an NVFP4 [rows, cols] -> dense tensor of dtype
Tensor dequantize(const Tensor &data, const Tensor &scales, const Tensor &global_scale, at::ScalarType dtype) {
  TORCH_CHECK_VALUE(data.is_cuda(), "data must be a CUDA tensor, got a tensor on ", data.device());
  const at::Device dev = data.device();
  check_gpu(dev);
  TORCH_CHECK_VALUE(data.dim() == 2, "data must be 2-D [rows, cols / 2], got shape ", data.sizes());
  const int64_t rows = data.size(0), cols = data.size(1) * 2;
  TORCH_CHECK_VALUE(cols % 16 == 0, "cols = ", cols, " must be a multiple of 16");
  const nvfp4a_dtype t = dense_dtype(dtype, "dtype");
  const Fp4 q = fp4_operand(data, scales, rows, cols, dev, "x", 16, 1);
  Tensor y = at::empty({rows, cols}, data.options().dtype(dtype));
  if (rows == 0 || cols == 0) return y;
  const c10::cuda::CUDAGuard guard(dev);
  const nvfp4a_tensor Q = {q.data.data_ptr(), NVFP4A_NVFP4, q.scales.data_ptr(), q.ld_scale,
                           scalar_ptr(global_scale, dev, "global_scale"), nullptr};
  const nvfp4a_tensor D = {y.data_ptr(), t, nullptr, 0, nullptr, nullptr};
  check_status(nvfp4a_dequantize(&Q, q.ld, &D, as_int(cols, "cols"), as_int(rows, "rows"), as_int(cols, "cols"),
                                 at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_dequantize");
  return y;
}

// out [T, N] = alpha * x @ W^T + bias + beta * out, W NVFP4 [N, K]; x [T, K] dense, NVFP4 (uint8
// [T, K / 2] with E4M3 x_scales) or INT8 (int8 [T, K] with float32 x_scales [T, K / 16]).  K a multiple
// of 64.
Tensor linear(const Tensor &x, const OptT &x_scales, const OptT &x_global_scale, const Tensor &w, const Tensor &w_scales,
              const Tensor &w_global_scale, const OptT &bias, std::optional<at::ScalarType> out_dtype, OptT out,
              double alpha, double beta, const OptT &amax_out) {
  TORCH_CHECK_VALUE(w.is_cuda(), "weight must be a CUDA tensor, got a tensor on ", w.device());
  const at::Device dev = w.device();
  check_device(x, dev, "x");
  check_gpu(dev);
  TORCH_CHECK_VALUE(w.dim() == 2 && x.dim() == 2, "x and weight.data must be 2-D, got shapes ", x.sizes(), " and ", w.sizes());
  const bool x_q8 = x_scales.has_value() && x.scalar_type() == at::kChar, x_fp4 = x_scales.has_value() && !x_q8;
  const int64_t N = w.size(0), K = w.size(1) * 2, T = x.size(0), xk = x_fp4 ? x.size(1) * 2 : x.size(1);
  TORCH_CHECK_VALUE(xk == K, "x has ", xk, " input features, the weight ", K);
  TORCH_CHECK_VALUE(K % 64 == 0, "the number of input features (", K, ") must be a multiple of 64 (pad with zeros)");
  const Fp4 W = fp4_operand(w, w_scales, N, K, dev, "weight", 64, 4);
  Fp4 X{};
  Tensor xd;
  int ldx = 0;
  if (x_fp4) {
    X = fp4_operand(x, *x_scales, T, K, dev, "x", 64, 4);
    ldx = X.ld;
  } else if (x_q8) {
    TORCH_CHECK_TYPE(x_scales->scalar_type() == at::kFloat, "the scales of an int8 x must be torch.float32, got ",
                     x_scales->scalar_type());
    check_device(*x_scales, dev, "x scales");
    TORCH_CHECK_VALUE(x_scales->dim() == 2 && x_scales->size(0) == T && x_scales->size(1) == K / 16,
                      "the scales of an int8 x must have shape [", T, ", ", K / 16, "], got ", x_scales->sizes());
    xd = x.stride(1) == 1 && (T <= 1 || x.stride(0) % 64 == 0) && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0
             ? x : x.contiguous();
    X.scales = x_scales->stride(1) == 1 && (T <= 1 || x_scales->stride(0) % 4 == 0) &&
                       reinterpret_cast<uintptr_t>(x_scales->data_ptr()) % 16 == 0
                   ? *x_scales : x_scales->contiguous();
    ldx = (int)(T <= 1 ? K : xd.stride(0));
    X.ld_scale = (int)(T <= 1 ? K / 16 : X.scales.stride(0));
  } else {
    dense_dtype(x.scalar_type(), "x");
    xd = x;
    if (x.stride(1) != 1 || (T > 1 && x.stride(0) % 64) || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) xd = x.contiguous();
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
  const nvfp4a_tensor A = {W.data.data_ptr(), NVFP4A_NVFP4, W.scales.data_ptr(), W.ld_scale,
                           scalar_ptr(w_global_scale, dev, "weight.global_scale"), nullptr};
  const nvfp4a_tensor B = x_fp4 ? nvfp4a_tensor{X.data.data_ptr(), NVFP4A_NVFP4, X.scales.data_ptr(), X.ld_scale,
                                                scalar_ptr(x_global_scale, dev, "x.global_scale"), nullptr}
                          : x_q8 ? nvfp4a_tensor{xd.data_ptr(), NVFP4A_INT8, X.scales.data_ptr(), X.ld_scale, nullptr, nullptr}
                                 : nvfp4a_tensor{xd.data_ptr(), dense_dtype(x.scalar_type(), "x"), nullptr, 0, nullptr, nullptr};
  const nvfp4a_tensor D = {out->data_ptr(), dense_dtype(out->scalar_type(), "out"), nullptr, 0, nullptr,
                           scalar_ptr(amax_out, dev, "amax_out")};
  nvfp4a_tensor BIAS = {};
  if (bias) BIAS = {bias->data_ptr(), dense_dtype(bias->scalar_type(), "bias"), nullptr, 0, nullptr, nullptr};
  const int64_t ldd = T <= 1 ? N : out->stride(0);
  check_status(nvfp4a_gemm(as_int(N, "N"), as_int(T, "T"), as_int(K, "K"), (float)alpha, &A, W.ld, &B, ldx, (float)beta,
                           nullptr, 0, &D, as_int(ldd, "ld(out)"), bias ? &BIAS : nullptr,
                           at::cuda::getCurrentCUDAStream().stream()),
               "nvfp4a_gemm");
  return *out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVFP4EffMA: NVFP4 GEMM / GEMV / quantization for GPUs without native FP4 (sm_50..sm_90)";
  m.attr("__version__") = NVFP4EFFMA_VERSION;
  m.def("quantize", &quantize, py::arg("x"), py::arg("global_scale"), py::arg("amax") = py::none());
  m.def("quantize_int8", &quantize_int8, py::arg("x"));
  m.def("dequantize", &dequantize, py::arg("data"), py::arg("scales"), py::arg("global_scale"), py::arg("dtype"));
  m.def("linear", &linear, py::arg("x"), py::arg("x_scales"), py::arg("x_global_scale"), py::arg("w"),
        py::arg("w_scales"), py::arg("w_global_scale"), py::arg("bias") = py::none(), py::arg("out_dtype") = py::none(),
        py::arg("out") = py::none(), py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("amax_out") = py::none());
}
