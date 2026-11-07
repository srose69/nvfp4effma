// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// nvfp4a_gemm_grouped: every group's output equals nvfp4a_gemm of that group, byte for byte, for every operand format of nvfp4a_gemm: A's scales
// E4M3 / UE5M3, linear / swizzled, per-tensor / per-row / amax globals; B NVFP4 (linear or swizzled scales, globals), INT8, BF16, FP16, FP32; bias,
// C (given, or accumulating into D), BF16 / FP16 / FP32 outputs; empty and tiny groups, ragged tiles; rows past seg[groups] and the padding of D stay
// untouched; the amax of D.  The reference runs the group's A rows against all rows of B (columns are independent, so any layout of B works) with the
// tiled kernel (more columns than the GEMV takes), and keeps the group's columns.
#include "test_util.h"

namespace {

struct GroupCase {
  int groups, m, k;
  std::vector<int> counts;
  nvfp4a_dtype bt, dt;
  unsigned fa, fb;
  int bias, ct;  // bias dtype or -1; ct: C dtype, -1 none, -2 accumulate into D
  float alpha, beta;
  int tail;
};

int run(const GroupCase &c) {
  using namespace nvfp4a::kern;
  const bool fp4b = c.bt == NVFP4A_NVFP4, q8b = c.bt == NVFP4A_INT8, blk = fp4b || q8b;
  std::vector<int> seg(c.groups + 1, 0);
  for (int g = 0; g < c.groups; ++g) seg[g + 1] = seg[g] + c.counts[g];
  const int used = seg[c.groups], rows = used + c.tail, total = std::max(rows, kGemvMaxN[c.bt] + 1);
  const int lda = c.k + 64, ldb = c.k + 64, ldd = c.m + 4, ldc = c.m + 8, M = c.groups * c.m;
  const nvfp4a_dtype ct = c.ct >= 0 ? (nvfp4a_dtype)c.ct : c.dt;
  HostFp4 hA(M, c.k, lda, c.k / 16 + 4, (c.fa & NVFP4A_GLOBAL_AMAX) ? 37.5f : 0.375f, c.fa, 0);
  HostFp4 hBq(fp4b ? total : 0, c.k, ldb, c.k / 16 + 8, (c.fb & NVFP4A_GLOBAL_AMAX) ? 175.f : 1.75f, c.fb, 0);
  HostQ8 hB8(q8b ? total : 0, c.k, ldb, c.k / 16 + 8, 1.75f);
  std::vector<uint8_t> hB(blk ? 0 : (size_t)ldb * total * esize(c.bt));
  if (!blk) fill(c.bt, hB, 2.f);
  const size_t dn = (size_t)total * ldd * esize(c.dt), cn = (size_t)total * ldc * esize(ct);
  std::vector<uint8_t> hD(dn, 0xA5), hC(cn), hbias((size_t)M * 4);
  if (c.ct == -2) fill(c.dt, hD, 4.f);
  fill(ct, hC, 4.f);
  if (c.bias >= 0) fill((nvfp4a_dtype)c.bias, hbias, 8.f);
  const std::vector<uint8_t> hD0 = hD;

  uint8_t *dA = dev(hA.data), *dSA = dev(hA.sc), *dD = dev(hD), *dRef = dev(hD0), *dC = dev(hC), *dbias = dev(hbias);
  void *dB = fp4b ? (void *)dev(hBq.data) : q8b ? (void *)dev(hB8.data) : (void *)dev(hB);
  void *dSB = fp4b ? (void *)dev(hBq.sc) : q8b ? (void *)dev(hB8.sc) : nullptr;
  float *dga = dev(hA.gv), *dgb = fp4b ? dev(hBq.gv) : devf(hB8.g), *damax = devf(0.f);
  int *dseg = dev(seg);
  nvfp4a_tensor A = {dA, NVFP4A_NVFP4, dSA, hA.ld_api(), dga, nullptr, c.fa, 0};
  nvfp4a_tensor B = {dB, c.bt, dSB, fp4b ? hBq.ld_api() : q8b ? hB8.ldsc : 0, blk ? dgb : nullptr, nullptr, fp4b ? c.fb : 0u, 0};
  nvfp4a_tensor C = {dC, ct, nullptr, 0, nullptr, nullptr}, D = {dD, c.dt, nullptr, 0, nullptr, damax};
  nvfp4a_tensor bias = {dbias, c.bias >= 0 ? (nvfp4a_dtype)c.bias : NVFP4A_FP32, nullptr, 0, nullptr, nullptr};
  nvfp4a_status st = nvfp4a_gemm_grouped(c.groups, c.m, c.k, rows, c.alpha, &A, lda, &B, ldb, c.ct == -1 ? 0.f : c.beta,
                                         c.ct >= 0 ? &C : nullptr, ldc, &D, ldd, c.bias >= 0 ? &bias : nullptr, dseg, 0);
  if (st != NVFP4A_OK) { printf("  grouped: %s: %s\n", nvfp4a_status_string(st), nvfp4a_last_error()); return 1; }
  std::vector<uint8_t> got(dn), ref(dn);
  cudaMemcpy(got.data(), dD, dn, cudaMemcpyDeviceToHost);
  int bad = 0;
  const size_t eb = esize(c.dt);
  for (int g = 0; g < c.groups; ++g) {
    if (!c.counts[g]) continue;
    nvfp4a_tensor Ag = A, Bn = B, bg = bias;
    if (!(c.fa & NVFP4A_SCALES_SWIZZLED)) {
      Ag.data = dA + (size_t)g * c.m * lda / 2;
      Ag.block_scale = dSA + (size_t)g * c.m * hA.ldsc;
    } else {  // swizzled scales: a group is a sub-tensor when its rows start at a multiple of 128
      Ag.data = dA + (size_t)g * c.m * lda / 2;
      Ag.block_scale = dSA + (size_t)(g * c.m / 128) * (((c.k / 16 + 3) / 4 * 4) / 4) * 512;
    }
    if (c.fa & NVFP4A_GLOBAL_PER_ROW) Ag.global_scale = dga + (size_t)g * c.m;
    if (c.bias >= 0) bg.data = dbias + (size_t)g * c.m * esize((nvfp4a_dtype)c.bias);
    cudaMemcpy(dRef, hD0.data(), dn, cudaMemcpyHostToDevice);
    nvfp4a_tensor Dg = {dRef, c.dt, nullptr, 0, nullptr, nullptr};
    st = nvfp4a_gemm(c.m, total, c.k, c.alpha, &Ag, lda, &Bn, ldb, c.ct == -1 ? 0.f : c.beta, c.ct >= 0 ? &C : nullptr, ldc, &Dg, ldd,
                     c.bias >= 0 ? &bg : nullptr, 0);
    if (st != NVFP4A_OK) { printf("  reference: %s: %s\n", nvfp4a_status_string(st), nvfp4a_last_error()); return 1; }
    cudaMemcpy(ref.data(), dRef, dn, cudaMemcpyDeviceToHost);
    for (int j = seg[g]; j < seg[g + 1]; ++j)
      for (int i = 0; i < c.m; ++i) {
        const size_t o = ((size_t)j * ldd + i) * eb;
        if (memcmp(&got[o], &ref[o], eb)) ++bad;
      }
  }
  // untouched: whole rows past seg[groups] and the columns m .. ldd of the used rows, as they were
  for (int j = 0; j < total; ++j)
    for (int i = (j < used ? c.m : 0); i < ldd; ++i)
      if (memcmp(&got[((size_t)j * ldd + i) * eb], &hD0[((size_t)j * ldd + i) * eb], eb)) ++bad;
  // the amax of D: the largest |value| before rounding, so the largest stored output rounded back by at most half an ulp (an FP32 D: exactly it);
  // an output that overflowed to inf has no amax to compare
  float amax = 0.f, top = 0.f;
  cudaMemcpy(&amax, damax, 4, cudaMemcpyDeviceToHost);
  for (int j = 0; j < used; ++j)
    for (int i = 0; i < c.m; ++i) top = std::max(top, fabsf(hload(c.dt, got.data(), (size_t)j * ldd + i)));
  if (std::isfinite(top) && !(amax >= top * (1.f - 1.01f * out_rel(c.dt)) && amax <= top * (1.f + 1.01f * out_rel(c.dt)))) {
    printf("  amax %g vs %g\n", amax, top);
    ++bad;
  }
  for (void *p : std::initializer_list<void *>{dA, dSA, dD, dRef, dC, dbias, dB, dSB, dga, dgb, damax, dseg}) cudaFree(p);
  return bad;
}

}  // namespace

int main() {
  cudaFree(nullptr);
  const unsigned UE = NVFP4A_SCALES_UE5M3, SW = NVFP4A_SCALES_SWIZZLED, AM = NVFP4A_GLOBAL_AMAX, ROW = NVFP4A_GLOBAL_PER_ROW;
  const nvfp4a_dtype I8 = NVFP4A_INT8, F4 = NVFP4A_NVFP4, BF = NVFP4A_BF16, H = NVFP4A_FP16, F = NVFP4A_FP32;
  const std::vector<GroupCase> cases = {
      // INT8 B, A linear: tiles, empty / tiny groups, tails, scale types, globals
      {1, 128, 64, {200}, I8, BF, 0, 0, -1, -1, 1.f, 0.f, 0},
      {4, 128, 128, {0, 1, 57, 130}, I8, BF, 0, 0, -1, -1, 1.f, 0.f, 3},
      {4, 192, 192, {300, 0, 0, 65}, I8, F, UE, 0, -1, -1, 0.5f, 0.f, 0},
      {8, 160, 512, {5, 64, 128, 129, 192, 193, 7, 1}, I8, BF, UE | AM, 0, -1, -1, 1.f, 0.f, 11},
      {3, 128, 256, {90, 91, 92}, I8, BF, ROW, 0, -1, -1, 1.f, 0.f, 5},
      {3, 130, 128, {10, 700, 20}, I8, F, UE | ROW, 0, -1, -1, 2.f, 0.f, 0},
      {8, 512, 128, {1, 1, 1, 1, 1, 1, 1, 1}, I8, BF, UE, 0, -1, -1, 1.f, 0.f, 0},
      // A's scales swizzled (groups of 128 rows: sub-tensors for the reference)
      {3, 128, 256, {70, 0, 150}, I8, BF, SW, 0, -1, -1, 1.f, 0.f, 2},
      {4, 256, 128, {60, 61, 62, 63}, I8, H, SW | UE | AM, 0, -1, -1, 1.f, 0.f, 0},
      // B NVFP4: linear and swizzled scales, E4M3 / UE5M3, amax / per-row globals
      {4, 128, 128, {70, 0, 61, 140}, F4, BF, 0, 0, -1, -1, 1.f, 0.f, 4},
      {3, 130, 256, {90, 91, 92}, F4, F, UE, UE | AM, -1, -1, 1.f, 0.f, 0},
      {3, 128, 192, {100, 7, 90}, F4, BF, SW, SW, -1, -1, 1.f, 0.f, 6},
      {4, 256, 128, {65, 66, 0, 130}, F4, H, SW | UE | ROW, SW | UE | AM, -1, -1, 0.5f, 0.f, 0},
      {3, 128, 256, {80, 81, 82}, F4, BF, AM, ROW, -1, -1, 1.f, 0.f, 0},
      // dense B (W4A16, FFMA)
      {4, 128, 128, {70, 0, 61, 140}, BF, BF, 0, 0, -1, -1, 1.f, 0.f, 3},
      {3, 192, 256, {90, 91, 92}, H, H, UE, 0, -1, -1, 1.f, 0.f, 0},
      {3, 130, 192, {100, 7, 90}, F, F, UE | AM, 0, -1, -1, 1.f, 0.f, 5},
      {3, 128, 128, {70, 71, 72}, BF, BF, SW, 0, -1, -1, 1.f, 0.f, 0},
      // epilogue: bias, C, accumulating into D
      {4, 128, 128, {70, 0, 61, 140}, I8, BF, 0, 0, BF, -1, 1.f, 0.f, 2},
      {3, 130, 128, {90, 91, 92}, I8, F, UE, 0, F, F, 1.f, 0.5f, 0},
      {3, 128, 256, {80, 70, 90}, F4, BF, 0, 0, H, BF, 1.f, 1.f, 1},
      {3, 130, 128, {70, 80, 90}, BF, F, 0, 0, F, -2, 1.f, 2.f, 0},
      {3, 256, 128, {70, 80, 90}, I8, BF, SW, 0, BF, -2, 0.5f, 1.f, 0},
  };
  int bad = 0, i = 0;
  for (const GroupCase &c : cases) {
    rng = 0x2545F491u + 0x9E3779B9u * (uint32_t)i;
    const int b = run(c);
    printf("case %2d (groups %d, m %3d, k %3d, B %-5s -> %-5s, fa 0x%x fb 0x%x%s%s): %s\n", i++, c.groups, c.m, c.k, DN[c.bt], DN[c.dt], c.fa,
           c.fb, c.bias >= 0 ? ", bias" : "", c.ct != -1 ? (c.ct == -2 ? ", D += " : ", C") : "", b ? "FAILED" : "ok");
    bad += b != 0;
  }
  printf("%s\n", bad ? "FAILED" : "all passed");
  return bad ? 1 : 0;
}
