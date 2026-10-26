// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Weight images: nvfp4a_tc_image_convert (A16 <-> A8) equals building the other kind from the tensor, and
// nvfp4a_tc_image_to_raw gives the tensor back byte for byte (padded pitches untouched), for row counts and k that
// leave partial slices, records and groups.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../include/nvfp4effma.h"

static int check(nvfp4a_status st, const char *what) {
  if (st == NVFP4A_OK) return 0;
  std::printf("   %s: %s\n", what, nvfp4a_last_error());
  return 1;
}

static int run(int rows, int k) {
  const int ld = k + 64, lds = k / 16 + 4;  // padded pitches
  std::vector<unsigned char> data((size_t)rows * ld / 2), sc((size_t)rows * lds);
  unsigned s = 12345u + rows * 7 + k;
  for (auto &b : data) b = (unsigned char)((s = s * 1664525u + 1013904223u) >> 24);
  for (auto &b : sc) b = (unsigned char)((s = s * 1664525u + 1013904223u) >> 24);
  const size_t ib = nvfp4a_tc_image_bytes(NVFP4A_TC_A16, NVFP4A_NVFP4, rows, k);
  unsigned char *d_data, *d_sc, *i16, *i8, *c16, *c8, *r_data, *r_sc;
  cudaMalloc(&d_data, data.size()); cudaMalloc(&d_sc, sc.size());
  cudaMalloc(&i16, ib); cudaMalloc(&i8, ib); cudaMalloc(&c16, ib); cudaMalloc(&c8, ib);
  cudaMalloc(&r_data, data.size()); cudaMalloc(&r_sc, sc.size());
  cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sc, sc.data(), sc.size(), cudaMemcpyHostToDevice);
  int fails = 0;
  const nvfp4a_tensor src = {d_data, NVFP4A_NVFP4, d_sc, lds, nullptr, nullptr, 0, 0};
  fails += check(nvfp4a_tc_image(NVFP4A_TC_A16, &src, ld, rows, k, i16, 0), "image A16");
  fails += check(nvfp4a_tc_image(NVFP4A_TC_A8, &src, ld, rows, k, i8, 0), "image A8");
  fails += check(nvfp4a_tc_image_convert(NVFP4A_TC_A16, NVFP4A_TC_A8, i16, rows, k, c8, 0), "A16 -> A8");
  fails += check(nvfp4a_tc_image_convert(NVFP4A_TC_A8, NVFP4A_TC_A16, i8, rows, k, c16, 0), "A8 -> A16");
  std::vector<unsigned char> h16(ib), h8(ib), hc16(ib), hc8(ib);
  cudaMemcpy(h16.data(), i16, ib, cudaMemcpyDeviceToHost); cudaMemcpy(h8.data(), i8, ib, cudaMemcpyDeviceToHost);
  cudaMemcpy(hc16.data(), c16, ib, cudaMemcpyDeviceToHost); cudaMemcpy(hc8.data(), c8, ib, cudaMemcpyDeviceToHost);
  if (h8 != hc8) { std::printf("   A16 -> A8 differs from the A8 image\n"); ++fails; }
  if (h16 != hc16) { std::printf("   A8 -> A16 differs from the A16 image\n"); ++fails; }
  for (int from : {NVFP4A_TC_A16, NVFP4A_TC_A8}) {
    cudaMemset(r_data, 0xA5, data.size()); cudaMemset(r_sc, 0xA5, sc.size());
    const nvfp4a_tensor dst = {r_data, NVFP4A_NVFP4, r_sc, lds, nullptr, nullptr, 0, 0};
    fails += check(nvfp4a_tc_image_to_raw(from, from == NVFP4A_TC_A16 ? i16 : i8, rows, k, &dst, ld, 0), "to raw");
    std::vector<unsigned char> gd(data.size()), gs(sc.size());
    cudaMemcpy(gd.data(), r_data, gd.size(), cudaMemcpyDeviceToHost); cudaMemcpy(gs.data(), r_sc, gs.size(), cudaMemcpyDeviceToHost);
    size_t bad = 0;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < ld / 2; ++c) bad += gd[(size_t)r * ld / 2 + c] != (c < k / 2 ? data[(size_t)r * ld / 2 + c] : 0xA5);
      for (int c = 0; c < lds; ++c) bad += gs[(size_t)r * lds + c] != (c < k / 16 ? sc[(size_t)r * lds + c] : 0xA5);
    }
    if (bad) { std::printf("   %s -> raw: %zu bytes differ\n", from == NVFP4A_TC_A16 ? "A16" : "A8", bad); ++fails; }
  }
  if (cudaError_t e = cudaDeviceSynchronize()) { std::printf("   cuda %s\n", cudaGetErrorString(e)); ++fails; }
  for (void *p : {(void *)d_data, (void *)d_sc, (void *)i16, (void *)i8, (void *)c16, (void *)c8, (void *)r_data, (void *)r_sc}) cudaFree(p);
  std::printf("images rows=%5d k=%5d : %s\n", rows, k, fails ? "FAIL" : "ok");
  return fails != 0;
}

int main() {
  int fails = 0;
  const int shapes[][2] = {{1, 64}, {16, 256}, {17, 320}, {128, 576}, {300, 1024}, {1000, 4160}, {4096, 4096}, {130, 12288}};
  for (const auto &s : shapes) fails += run(s[0], s[1]);
  std::printf(fails ? "\n%d FAILED\n" : "\nALL PASS\n", fails);
  return fails != 0;
}
