// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// One quantization instance object: -DNVFP4A_QT=<dense dtype> -DNVFP4A_QCOL=<0 rowwise, 1 columnwise>
// -DNVFP4A_QTILE=<1 or 16>: the kernels of every valid 4over6 mode and format; the rowwise 1 x 16 object
// also has the dequantization kernels of its dtype.
#include <utility>
#include "quant_nvfp4effma.cuh"

namespace nvfp4a {
namespace kern {

using QuantFn = cudaError_t (*)(const uint8_t *, int, int, int, const QuantOut &, float *, cudaStream_t);

template <int T, bool COL, int TILE, int FOS, int QF, bool OK = quant_valid(COL, TILE, FOS, QF)>
struct QuantOne {
  static cudaError_t run(const uint8_t *src, int lds, int rows, int cols, const QuantOut &o, float *stat, cudaStream_t s) {
    constexpr int BPW = 8;  // 16 x 16 rowwise: blocks per warp
    const long long threads = !COL && TILE == 1 ? (long long)rows * (cols / 16)
                            : !COL && FOS == NVFP4A_4OVER6_OFF ? (long long)((rows + 15) / 16) * ((cols / 16 + BPW - 1) / BPW) * 32
                                   : (long long)((cols + 15) / 16) * ((rows + 15) / 16) * 16;
    quantize_kernel<T, COL, TILE, FOS, QF><<<quant_grid(threads), 256, 0, s>>>(src, lds, rows, cols, o, stat);
    return cudaGetLastError();
  }
};
template <int T, bool COL, int TILE, int FOS, int QF>
struct QuantOne<T, COL, TILE, FOS, QF, false> {
  static cudaError_t run(const uint8_t *, int, int, int, const QuantOut &, float *, cudaStream_t) { return cudaErrorInvalidValue; }
};

template <int T, bool COL, int TILE, size_t... I>
static const QuantFn *quant_table(std::index_sequence<I...>) {
  static const QuantFn t[] = {&QuantOne<T, COL, TILE, (int)(I / kQFormats), (int)(I % kQFormats)>::run...};
  return t;
}

template <int T, bool COL, int TILE>
cudaError_t quant_launch(int fos, int qf, const uint8_t *src, int lds, int rows, int cols, const QuantOut &o, float *stat,
                         cudaStream_t s) {
  return quant_table<T, COL, TILE>(std::make_index_sequence<3 * kQFormats>())[fos * kQFormats + qf](src, lds, rows, cols, o, stat, s);
}
template cudaError_t quant_launch<NVFP4A_QT, NVFP4A_QCOL != 0, NVFP4A_QTILE>(int, int, const uint8_t *, int, int, int,
                                                                           const QuantOut &, float *, cudaStream_t);

#if !NVFP4A_QCOL && NVFP4A_QTILE == 1
using DequantFn = cudaError_t (*)(const QuantOut &, uint8_t *, int, int, int, float *, cudaStream_t);
template <int T, int QF>
static cudaError_t dequant_one(const QuantOut &q, uint8_t *dst, int ldd, int rows, int cols, float *amax, cudaStream_t s) {
  dequantize_kernel<T, QF><<<quant_grid((long long)rows * (cols / 16)), 256, 0, s>>>(q, dst, ldd, rows, cols, amax);
  return cudaGetLastError();
}
template <int T, size_t... I>
static const DequantFn *dequant_table(std::index_sequence<I...>) {
  static const DequantFn t[] = {&dequant_one<T, (int)I>...};
  return t;
}
template <int T>
cudaError_t dequant_launch(int qf, const QuantOut &q, uint8_t *dst, int ldd, int rows, int cols, float *amax, cudaStream_t s) {
  return dequant_table<T>(std::make_index_sequence<kQFormats>())[qf](q, dst, ldd, rows, cols, amax, s);
}
template cudaError_t dequant_launch<NVFP4A_QT>(int, const QuantOut &, uint8_t *, int, int, int, float *, cudaStream_t);
#endif

}  // namespace kern
}  // namespace nvfp4a
