// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm_ex (and the tensor-core _ex entries) against the host: the accumulators come from the same entry without the
// descriptor (same kernel, same order; unit scales, FP32 D), the epilogue's operations are emulated:
//   D, second output, sums of squares   bit for bit (s_j: the lanes' order, the butterfly, 1 / sqrt rounded to nearest)
//   SwiGLU                              against silu(g) u, within the GPU's fast exp (its error grows with |g|) and division
// Every addition mask, B BF16 and INT8, D BF16 and FP32, with and without bias / C, n on the GEMV path and beyond.
#include "test_util.h"

namespace {

// s_j as the kernels form it: lane l adds parts l, l + 32, ... in order, then the xor butterfly 16 .. 1
float host_norm_scale(const float *p, int np, float inv_n, float eps) {
  float s[32];
  for (int l = 0; l < 32; ++l) {
    s[l] = 0.f;
    for (int i = l; i < np; i += 32) s[l] = s[l] + p[i];
  }
  for (int o = 16; o; o >>= 1) {
    float t[32];
    for (int l = 0; l < 32; ++l) t[l] = s[l] + s[l ^ o];
    memcpy(s, t, sizeof s);
  }
  return (float)(1.0L / sqrtl((long double)(s[0] * inv_n + eps)));
}

struct ExCase {
  int m, n, k;
  nvfp4a_dtype bt, dt;
  int mask;      // kEX_*
  int bias, c;   // bias (BF16) and C (D += ...) on / off
  bool tc;       // the tensor-core entries (GEMV for n <= 8, GEMM beyond)
};

template <class T> T *todev(const std::vector<T> &v) {
  T *d;
  cudaMalloc(&d, v.size() * sizeof(T) + 16);
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

int run(const ExCase &c) {
  using namespace nvfp4a::kern;
  const bool q8 = c.bt == NVFP4A_INT8, norm = c.mask & kEX_NORM, glu = c.mask & kEX_SWIGLU, out2 = c.mask & kEX_OUT2;
  const int lda = c.k, ldb = c.k, mo = glu ? c.m / 2 : c.m, ldd = mo + 3, np = 37, ldp = 40, lsq = c.m / 16 + 1;
  HostFp4 hA(c.m, c.k, lda, c.k / 16, 0.f);
  HostQ8 hB8(q8 ? c.n : 0, c.k, ldb, c.k / 16, 1.75f);
  std::vector<uint8_t> hB(q8 ? 0 : (size_t)ldb * c.n * 2), hbias((size_t)c.m * 2), hw2((size_t)c.m * 2);
  if (!q8) fill(NVFP4A_BF16, hB, 2.f);
  fill(NVFP4A_BF16, hbias, 4.f);
  fill(NVFP4A_BF16, hw2, 2.f);
  std::vector<float> parts((size_t)ldp * c.n), cs(c.n, 1.f);
  for (auto &x : parts) x = (float)(rnd() % 4096) / 64.f;
  for (int j = 0; j < c.n && norm; ++j) cs[j] = host_norm_scale(&parts[(size_t)j * ldp], np, 1.f / 1024.f, 1e-6f);
  std::vector<uint8_t> hD((size_t)ldd * c.n * esize(c.dt));
  fill(c.dt, hD, 4.f);

  uint8_t *dA = todev(hA.data), *dSA = todev(hA.sc), *dbias = todev(hbias), *dw2 = todev(hw2);
  void *dB = q8 ? (void *)todev(hB8.data) : (void *)todev(hB);
  float *dSB = q8 ? todev(hB8.sc) : nullptr, *dparts = todev(parts);
  const float gb = 1.75f;
  float *dgb = devf(gb);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, 0, nullptr, nullptr, 0, 0};
  nvfp4a_tensor B = {dB, c.bt, dSB, q8 ? c.k / 16 : 0, q8 ? dgb : nullptr, nullptr, 0, 0};
  nvfp4a_tensor bias = {dbias, NVFP4A_BF16, nullptr, 0, nullptr, nullptr, 0, 0};
  const float alpha = 0.75f;
  float beta = c.c ? 1.f : 0.f;

  // the extended call
  uint8_t *dD = todev(hD), *dD2 = nullptr;
  float *dsq = nullptr;
  std::vector<uint8_t> hd2((size_t)c.m * c.n * 4, 0);
  if (out2) { dD2 = todev(hd2); std::vector<float> z((size_t)lsq * c.n, -1.f); dsq = todev(z); }
  nvfp4a_epilogue ex{};
  if (norm) { ex.norm_parts = dparts; ex.norm_nparts = np; ex.norm_ld = ldp; ex.norm_inv_n = 1.f / 1024.f; ex.norm_eps = 1e-6f; }
  ex.swiglu = glu;
  if (out2) { ex.d2 = dD2; ex.d2_dtype = NVFP4A_BF16; ex.ldd2 = c.m; ex.w2 = dw2; ex.w2_dtype = NVFP4A_BF16; ex.sq = dsq; ex.ld_sq = lsq; }
  nvfp4a_tensor D = {dD, c.dt, nullptr, 0, nullptr, nullptr, 0, 0};
  nvfp4a_status st;
  void *ia = nullptr;
  static void *ws = nullptr;
  if (c.tc) {
    if (!ws) { cudaMalloc(&ws, nvfp4a_tc_workspace_bytes()); cudaMemset(ws, 0, nvfp4a_tc_workspace_bytes()); }
    const int kind = q8 ? NVFP4A_TC_A8 : NVFP4A_TC_A16;
    cudaMalloc(&ia, nvfp4a_tc_image_bytes(kind, NVFP4A_NVFP4, c.m, c.k) + 16);
    st = nvfp4a_tc_image(kind, &A, lda, c.m, c.k, ia, 0);
  }
  nvfp4a_tensor Ai = A;
  if (c.tc) Ai.data = ia;
  void *ib = nullptr;
  const nvfp4a_tensor *bp = c.bias ? &bias : nullptr;
  auto call = [&](const nvfp4a_tensor &Bx, int n, float al, const nvfp4a_tensor *C, const nvfp4a_tensor &Dx, int ld, const nvfp4a_epilogue *e) {
    if (!c.tc) return nvfp4a_gemm_ex(c.m, n, c.k, al, &A, lda, &Bx, ldb, beta, C, ld, &Dx, ld, bp, e, 0);
    if (n <= 8) return nvfp4a_gemv_tc_ex(c.m, n, c.k, al, &Ai, &Bx, ldb, beta, C, ld, &Dx, ld, bp, e, 0);
    cudaFree(ib);
    cudaMalloc(&ib, nvfp4a_tc_image_bytes(NVFP4A_TC_B, c.bt, n, c.k) + 16);
    nvfp4a_status s2 = nvfp4a_tc_image(NVFP4A_TC_B, &Bx, ldb, n, c.k, ib, 0);
    nvfp4a_tensor Bi = Bx;
    Bi.data = ib;
    return s2 ? s2 : nvfp4a_gemm_tc_ex(c.m, n, c.k, al, &Ai, &Bi, beta, C, ld, &Dx, ld, bp, e, ws, 0);
  };
  st = call(B, c.n, alpha, nullptr, D, ldd, &ex);
  cudaError_t ce = cudaDeviceSynchronize();
  if (st == NVFP4A_NOT_SUPPORTED) {
    printf("   (%s)\n", nvfp4a_last_error());
    return 0;
  }
  std::vector<uint8_t> got(hD.size());
  cudaMemcpy(got.data(), dD, got.size(), cudaMemcpyDeviceToHost);
  std::vector<uint8_t> got2(hd2.size());
  std::vector<float> gsq((size_t)lsq * c.n);
  if (out2) { cudaMemcpy(got2.data(), dD2, got2.size(), cudaMemcpyDeviceToHost); cudaMemcpy(gsq.data(), dsq, gsq.size() * 4, cudaMemcpyDeviceToHost); }

  // the reference: the same call without the descriptor, unit scales and FP32 D (the accumulators acc, the same kernel and
  // order), then the epilogue's operations on the host
  std::vector<float> acc((size_t)c.m * c.n);
  {
    float *dR;
    cudaMalloc(&dR, acc.size() * 4);
    nvfp4a_tensor Bu = B, R = {dR, NVFP4A_FP32, nullptr, 0, nullptr, nullptr, 0, 0};
    Bu.global_scale = nullptr;
    const float b0 = beta;
    beta = 0.f;
    bp = nullptr;
    const nvfp4a_status s1 = call(Bu, c.n, 1.f, nullptr, R, c.m, nullptr);
    beta = b0;
    cudaDeviceSynchronize();
    cudaMemcpy(acc.data(), dR, acc.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dR);
    if (s1) { printf("   reference call: %s\n", nvfp4a_last_error()); return 1; }
  }
  int bad = 0;
  std::vector<float> v(c.m);
  for (int j = 0; j < c.n && !bad; ++j) {
    const float sc = alpha * ((q8 ? gb : 1.f) * cs[j]);
    for (int i = 0; i < c.m; ++i) {
      float x = acc[(size_t)j * c.m + i] * sc;
      if (c.bias) x = x + bf(((const uint16_t *)hbias.data())[i]);
      if (c.c) x = fmaf(beta, hload(c.dt, hD.data(), (size_t)j * ldd + i), x);
      v[i] = x;
    }
    for (int i = 0; i < mo; ++i) {
      if (glu) {
        const int g = (i / 8) * 16 + i % 8;
        const double h = (double)v[g] / (1. + exp(-(double)v[g])) * v[g + 8];
        const double y = hload(c.dt, got.data(), (size_t)j * ldd + i);
        // the fast exp's error grows with |g| (its argument g log2 e rounded to FP32), then the output's rounding
        const double tol = fabs(h) * (4e-7 * (2. + fabs(v[g])) + out_rel(c.dt)) + 1e-30;
        if (fabs(y - h) > tol) { printf("   D[%d, %d] = %.9g, silu(g) u = %.9g\n", i, j, y, h); ++bad; break; }
      } else {
        std::vector<uint8_t> one(esize(c.dt));
        hstore(c.dt, one.data(), 0, v[i]);
        if (memcmp(one.data(), got.data() + ((size_t)j * ldd + i) * esize(c.dt), esize(c.dt))) {
          printf("   D[%d, %d] = %.9g, expected %.9g\n", i, j, hload(c.dt, got.data(), (size_t)j * ldd + i), v[i]);
          ++bad;
          break;
        }
      }
    }
    for (int i = 0; out2 && !bad && i < c.m; ++i) {
      const uint16_t e2 = to_bf(v[i] * bf(((const uint16_t *)hw2.data())[i]));
      const uint16_t g2 = ((const uint16_t *)got2.data())[(size_t)j * c.m + i];
      if (e2 != g2) { printf("   d2[%d, %d] = %04x, expected %04x\n", i, j, g2, e2); ++bad; }
    }
    for (int s = 0; out2 && !bad && s < c.m / 16; ++s) {
      float q = 0.f;
      for (int r = 0; r < 16; ++r) q = q + v[16 * s + r] * v[16 * s + r];
      if (memcmp(&q, &gsq[(size_t)j * lsq + s], 4)) { printf("   sq[%d, %d] = %.9g, expected %.9g\n", s, j, gsq[(size_t)j * lsq + s], q); ++bad; }
    }
  }
  printf("%-8s m %5d n %3d k %5d B %-4s D %-4s ex %d%s%s: %s\n", c.tc ? "tc_ex" : "gemm_ex", c.m, c.n, c.k, DN[c.bt], DN[c.dt], c.mask,
         c.bias ? " bias" : "", c.c ? " +=" : "", ce ? cudaGetErrorString(ce) : st ? nvfp4a_last_error() : bad ? "FAIL" : "ok");
  for (void *p : {(void *)dA, (void *)dSA, (void *)dbias, (void *)dw2, dB, (void *)dSB, (void *)dparts, (void *)dgb, (void *)dD,
                  (void *)dD2, (void *)dsq, ia, ib})
    cudaFree(p);
  return bad || st || ce ? 1 : 0;
}

}  // namespace

int main() {
  using namespace nvfp4a::kern;
  int fails = 0, v = 0;
  const int shapes[][3] = {{64, 1, 64}, {1024, 1, 1024}, {4096, 1, 1024}, {528, 3, 576}, {6144, 8, 1024}, {2048, 5, 3072},
                           {1024, 12, 2048}, {512, 17, 1024}, {512, 64, 1024}, {1024, 200, 512}};
  for (bool tc : {false, true})
    for (const auto &s : shapes)
      for (nvfp4a_dtype bt : {NVFP4A_BF16, NVFP4A_INT8})
        for (int mask = 1; mask <= kEX_MASKS; ++mask, ++v) {
          const bool glu = mask & kEX_SWIGLU;
          ExCase c{s[0], s[1], s[2], bt, v % 2 ? NVFP4A_FP32 : NVFP4A_BF16, mask, !glu && v % 3 == 1, !glu && v % 4 == 2, tc};
          fails += run(c);
        }
  printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
