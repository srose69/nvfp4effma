// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The GEMV work item (gemv_rows) on parts of k, as a caller that splits k runs it: every part's sums of every
// (row, column), bit-exact against the host emulation of the lane order over that part -- x NVFP4 / INT8 / BF16, linear
// and swizzled scales of A and x, parts from k0 > 0, a short last part, a last row block past m.
#include "test_util.h"
#include "../src/gemv_nvfp4effma.cuh"

template <int NB, int XT, int F>
__global__ void __launch_bounds__(256) parts_kernel(const uint8_t *A, int lda, const uint8_t *SA, int ldsa, const uint8_t *X, int ldx,
                                                    const uint8_t *SX, int ldsx, int m, int k, int kpart, float *out) {
  struct Out {
    float *o;
    int m, part;
    __device__ void begin() {}
    __device__ void store(int i, int j, float v) { o[((size_t)part * m + i) * NB + j] = v; }
    __device__ void end() {}
  } w{out, m, (int)blockIdx.y};
  const int k0 = blockIdx.y * kpart, k1 = min(k, k0 + kpart);
  nvfp4a::kern::gemv_rows<nvfp4a::kern::GemvCfg0, NB, XT, F>(A, lda, SA, ldsa, X, ldx, SX, ldsx, m, blockIdx.x, threadIdx.x, k0, k1, w);
}

template <int NB, int XT, int F>
static int run(int m, int k, int kpart) {
  using namespace nvfp4a;
  constexpr bool fp4b = XT == NVFP4A_NVFP4, q8b = XT == NVFP4A_INT8, blk = fp4b || q8b;
  const unsigned fa = F & kF_ASWZ ? NVFP4A_SCALES_SWIZZLED : 0u, fb = fp4b && (F & kF_BSWZ) ? NVFP4A_SCALES_SWIZZLED : 0u;
  const int lda = k + 64, ldb = k + (fp4b ? 128 : 64);
  HostFp4 hA(m, k, lda, k / 16 + 4, 1.f, fa);
  HostFp4 hBq(fp4b ? NB : 0, k, ldb, k / 16 + 8, 1.f, fb);
  HostQ8 hB8(q8b ? NB : 0, k, ldb, k / 16 + 8, 1.f);
  std::vector<uint8_t> hB(blk ? 0 : (size_t)ldb * NB * 2);
  if (!blk) fill(NVFP4A_BF16, hB, 2.f);
  const int pad = (k / 16 + 3) / 4 * 4;  // a swizzled scale row's padded columns (the kernels' ld: its negative)
  const int ldsa = fa ? -pad : hA.ldsc, ldsx = fp4b ? (fb ? -pad : hBq.ldsc) : q8b ? hB8.ldsc : 0;
  uint8_t *dA = dev(hA.data), *dSA = dev(hA.sc);
  uint8_t *dX = fp4b ? dev(hBq.data) : q8b ? (uint8_t *)dev(hB8.data) : dev(hB);
  uint8_t *dSX = fp4b ? dev(hBq.sc) : q8b ? (uint8_t *)dev(hB8.sc) : nullptr;
  const int nparts = (k + kpart - 1) / kpart, rows = 8 * kern::GemvCfg0::R;
  float *dout;
  cudaMalloc(&dout, (size_t)nparts * m * NB * 4);
  parts_kernel<NB, XT, F><<<dim3((m + rows - 1) / rows, nparts), 256>>>(dA, lda, dSA, ldsa, dX, ldb, dSX, ldsx, m, k, kpart, dout);
  const cudaError_t ce = cudaDeviceSynchronize();
  std::vector<float> got((size_t)nparts * m * NB);
  cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);

  auto Bval = [&](int j, int kk) { return fp4b ? hBq.val(j, kk) : bf(((const uint16_t *)hB.data())[(size_t)j * ldb + kk]); };
  auto block = [&](int i, int j, int kb, float acc) {
    int p = 0;
    for (int kk = kb; kk < kb + 16; ++kk)
      p += (int)(2 * e2(hA.nib(i, kk))) * (fp4b ? (int)(2 * e2(hBq.nib(j, kk))) : hB8.q(j, kk));
    return fmaf((float)p * hA.scale(i, kb), fp4b ? hBq.scale(j, kb) : hB8.scale(j, kb), acc);
  };
  size_t bad = 0;
  for (int part = 0; part < nparts; ++part)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < NB; ++j) {
        const int k0 = part * kpart, k1 = std::min(k, k0 + kpart);
        float lanes[32];
        for (int l = 0; l < 32; ++l) {
          float p = 0.f;
          for (int kb = k0 + 16 * l; kb < k1; kb += 512) {
            if (blk) p = block(i, j, kb, p);
            else for (int jj = 0; jj < 16; ++jj) p = fmaf(hA.val(i, kb + jj), Bval(j, kb + jj), p);
          }
          lanes[l] = p;
        }
        for (int o = 16; o; o >>= 1) {
          float nx[32];
          for (int l = 0; l < 32; ++l) nx[l] = lanes[l] + lanes[l ^ o];
          memcpy(lanes, nx, sizeof nx);
        }
        const float want = blk ? lanes[0] * (fp4b ? 0.25f : 0.5f) : lanes[0];
        const float g = got[((size_t)part * m + i) * NB + j];
        bad += memcmp(&want, &g, 4) != 0;
      }
  const int fail = ce != cudaSuccess || bad;
  printf("gemv_rows m=%4d k=%5d part %4d (%d parts) x=%-5s NB=%d scales A %s x %s: %s", m, k, kpart, nparts, DN[XT], NB,
         fa ? "swizzled" : "linear", fp4b ? (fb ? "swizzled" : "linear") : "-", fail ? "FAIL" : "ok");
  if (bad) printf(" (%zu sums differ)", bad);
  if (ce) printf(" (cuda %s)", cudaGetErrorString(ce));
  printf("\n");
  for (void *p : {(void *)dA, (void *)dSA, (void *)dX, (void *)dSX, (void *)dout}) cudaFree(p);
  return fail;
}

int main() {
  using namespace nvfp4a;
  int f = 0;
  f += run<1, NVFP4A_BF16, 0>(300, 4160, 1024);
  f += run<4, NVFP4A_BF16, kF_ASWZ>(300, 4160, 1024);
  f += run<1, NVFP4A_INT8, 0>(130, 4096, 512);
  f += run<4, NVFP4A_INT8, kF_ASWZ>(1000, 12288, 3072);
  f += run<1, NVFP4A_NVFP4, 0>(77, 4160, 2048);
  f += run<4, NVFP4A_NVFP4, kF_ASWZ | kF_BSWZ>(300, 4160, 1024);
  f += run<2, NVFP4A_NVFP4, kF_BSWZ>(64, 2048, 512);
  printf(f ? "\n%d FAILED\n" : "\nALL PASS\n", f);
  return f != 0;
}
