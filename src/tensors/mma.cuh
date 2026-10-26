// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Warp-level instructions of the tensor-core GEMM (sm_75 .. sm_89): the integer mma of one NVFP4 block, ldmatrix, named
// barriers, shared-memory flags; from sm_80 the BF16 mma, cp.async, mbarriers.  A wrapper an architecture lacks is not
// declared in its device pass.
//
// One mma covers exactly one block of 16 k (PTX ISA, Matrix Fragments for mma.m16n8k16 with integer type;
// g = lane / 4, t = lane % 4):
//   A [16 m][16 k] s8   a0 = row g, k 4t .. 4t + 3;  a1 = row g + 8
//   B [16 k][8 n] s8    b  = k 4t .. 4t + 3 of column g
//   D [16 m][8 n] s32   d0, d1 = row g, columns 2t, 2t + 1;  d2, d3 = row g + 8
// so the block sum p of every output is an exact integer (|p| <= 16 * 12 * 128 < 2^15) and the DP4A kernel's scale
// steps apply to it unchanged: the accumulator starts at the bits of 1.5 * 2^23 (MAGIC), D read as FP32 is
// 1.5 * 2^23 + p exactly.  m16n8k16 runs at the full int8 rate (an m16n8k32 sums two blocks and cannot be split);
// sm_75: two m8n8k16 (rows 0 - 7 with a0, 8 - 15 with a1) on the same registers.
#pragma once
#include <stdint.h>

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900)
#define NVFP4A_TC 1
#endif
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900)
#define NVFP4A_TC80 1
#endif

namespace nvfp4a {
namespace kern {
namespace tc {

constexpr uint32_t kMagic = 0x4B400000u;  // FP32 bits of 1.5 * 2^23
constexpr float kMagicF = 12582912.f;     // 1.5 * 2^23

#ifdef NVFP4A_TC
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
// %tid.x / %ctaid.x re-read where used: cold code (segments, epilogues) derives its addresses from these, so ptxas
// cannot hoist them into the hot loop and keep them live there
__device__ __forceinline__ uint32_t tid_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t ctaid_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t nctaid_x() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(r));
  return r;
}

// 0, data-dependent on b: min(|b|, +0) is +0 for every bit pattern (one FMNMX).  MAGIC + zero_of(x) is a value
// ptxas cannot rematerialize: the C quad of the mma stays one live quad instead of being rebuilt (four MOV) whenever
// an accumulator took its registers.
__device__ __forceinline__ uint32_t zero_of(uint32_t b) {
  uint32_t z;
  asm("{ .reg .f32 t; abs.f32 t, %1; min.f32 t, t, 0f00000000; mov.b32 %0, t; }" : "=r"(z) : "f"(__uint_as_float(b)));
  return z;
}

// d = c + A B for one block (A rows 0 - 7: a0, 8 - 15: a1); c = the MAGIC quad
__device__ __forceinline__ void imma_block(uint32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b, const uint32_t (&c)[4]) {
#ifdef NVFP4A_TC80
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
      : "=r"(d[0]), "=r"(d[1]) : "r"(a0), "r"(b), "r"(c[0]), "r"(c[1]));
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
      : "=r"(d[2]), "=r"(d[3]) : "r"(a1), "r"(b), "r"(c[2]), "r"(c[3]));
#endif
}

// four 8 x 8 matrices of 16-bit elements; lanes 8 q .. 8 q + 7 give the row addresses of matrix q (register q), lane l
// receives elements 2 (l % 4), + 1 of row l / 4.  A memory clobber: the slots are written by cp.async
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

#ifdef NVFP4A_TC80
// d += A B, BF16 m16n8k16 with FP32 accumulation: A0 (row g, k 2t, 2t + 1), A1 (row g + 8), A2 / A3 at k 8 + 2t;
// B0 (k 2t, 2t + 1 of column g), B1 (k 8 + 2t ..)
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// x * y + z on BF16 pairs, one rounding
__device__ __forceinline__ uint32_t fma_bf16x2(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(x), "r"(y), "r"(z));
  return d;
}
// 16 bytes global -> shared (L2 only); visible to the issuing thread after its wait
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
// arrive on mbarrier bar when the thread's prior cp.async are complete (.noinc: bar's count includes this arrival)
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(bar) : "memory");
}

// mbarriers (shared memory addresses): arrive releases the thread's prior accesses, a completed wait acquires them
// (sm_80: test_wait polled; try_wait is sm_90).  On sm_8x an arrive is MEMBAR.ALL.CTA + ATOMS: for rare arrivals.
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.shared.b64 st, [%0]; }" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  } while (!done);
}
#endif  // NVFP4A_TC80

// A stream read once (the GEMV's weights): 16 bytes if `on` (else zeros), not allocated in L1 (x and the outputs keep
// it), with the L2 policy `pol` (sm_80+: evict first, the weights must not push x, the outputs or other kernels' data
// out of L2) and an L2 prefetch of the 256 bytes around it (sm_75: 128)
#ifdef NVFP4A_TC80
__device__ __forceinline__ uint64_t l2_evict_first() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 ldg_stream(const void *a, bool on, uint64_t pol) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm("{ .reg .pred p; setp.ne.u32 p, %5, 0;\n\t"
      "@p ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4], %6; }"
      : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "l"(a), "r"((uint32_t)on), "l"(pol));
  return v;
}
#else
__device__ __forceinline__ uint64_t l2_evict_first() { return 0; }
__device__ __forceinline__ uint4 ldg_stream(const void *a, bool on, uint64_t) {
  uint4 v = make_uint4(0, 0, 0, 0);
  asm("{ .reg .pred p; setp.ne.u32 p, %5, 0;\n\t"
      "@p ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4]; }"
      : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "l"(a), "r"((uint32_t)on));
  return v;
}
#endif

// Named barriers (barrier.cta, not .aligned: the warps that release a slot arrive, the one refilling it syncs)
template <int N> __device__ __forceinline__ void bar_sync(uint32_t id) {
  asm volatile("barrier.cta.sync %0, %1;" :: "r"(id), "n"(N) : "memory");
}
template <int N> __device__ __forceinline__ void bar_arrive(uint32_t id) {
  asm volatile("barrier.cta.arrive %0, %1;" :: "r"(id), "n"(N) : "memory");
}
__device__ __forceinline__ void warp_sync() { asm volatile("bar.warp.sync -1;" ::: "memory"); }
// sm_75 slot flags (no mbarrier): a sequence number in shared memory, set after a CTA fence, waited for before one
__device__ __forceinline__ void seq_set(uint32_t addr, uint32_t v) {
  asm volatile("{ membar.cta; st.volatile.shared.u32 [%0], %1; }" :: "r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void seq_wait(uint32_t addr, uint32_t v) {
  uint32_t x;
  do { asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(x) : "r"(addr) : "memory"); } while (x != v);
  asm volatile("membar.cta;" ::: "memory");
}
// GPU-scope flag publication / observation (stream-K partials)
__device__ __forceinline__ void st_release(unsigned *p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned *p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
#endif  // NVFP4A_TC

}  // namespace tc
}  // namespace kern
}  // namespace nvfp4a
