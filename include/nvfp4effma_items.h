// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* NVFP4EffMA work items: the work of a call as items that an executor runs from kernels of its own, together with what
 * each item reads and writes.  The items of a call compute what the call does, bit for bit -- but a tile (GEMM_TC) sums
 * all of k in one CTA where nvfp4a_gemm_tc_ex may split a tile's k between CTAs (stream-K): the same products, the sum
 * grouped differently.
 *
 * nvfp4a_gemm_ex_items / nvfp4a_gemv_tc_ex_items / nvfp4a_gemm_tc_ex_items take the arguments of nvfp4a_gemm_ex /
 * nvfp4a_gemv_tc_ex / nvfp4a_gemm_tc_ex without the workspace (validated the same way; ex may be NULL: the plain
 * epilogue) and describe the call on the current device:
 *   kind        NVFP4A_ITEM_GEMV     a block of `rows` output rows, every column, over all of k on CUDA cores (n <= 8)
 *               NVFP4A_ITEM_GEMV_TC  an m16 slice (rows = 16), every column, over all of k on tensor cores (sm_75 .. sm_89)
 *               NVFP4A_ITEM_GEMM_TC  a `rows` x `cols` tile over all of k on tensor cores; the CTA's warps specialize
 *   items       their number (nvfp4a_item_access: which outputs each one computes)
 *   rows, cols  the outputs of an item (cols 0: every column)
 *   threads     the CTA one item runs on, exactly (all of its threads call the entry; it synchronizes them)
 *   smem_bytes  shared memory the executor hands over (16-byte aligned); synchronize the CTA after an entry before
 *               reusing its shared memory or beginning the next item
 *   regs        registers a thread of the library's own kernel for this work uses (the executor's may differ)
 *   arch        virtual architecture of that instance (e.g. 75); compile the device entries for compute_<arch>,
 *               then JIT the PTX on the current device, so their CTA shape and shared layout match the plan
 *   entry       the device entry (include/nvfp4effma_items.cuh), called as
 *                 entry(*(const args_type *)p, item, thread, smem)
 *               with p a 16-byte aligned device copy of args.bytes[0 .. args_bytes - 1], thread the thread's index in
 *               the CTA.  The data the arguments point to must stay where they were at the time of the description.
 * A blocked operand needs linear E4M3 block scales (A, and NVFP4 B is not taken); B BF16 or INT8.
 *
 * The weight A (data/image, block scales and global), bias and w2 are immutable for the executor launch.  B
 * (data/image, block and global scales), C (including C = D) and norm_parts are read through coherent L2; another
 * CTA may produce them in the same launch.  The executor must establish GPU-scope release/acquire ordering before
 * calling the entry and finish all consuming threads before reusing inputs.  It also publishes outputs after all
 * writing threads finish: on sm_6x, a device fence in every writer, a CTA barrier and then an atomic signal.
 * The entries do not implement the executor's inter-CTA waits or signals.
 *
 * nvfp4a_item_access lists what an item reads and writes as ranges of bytes relative to a call argument's pointer:
 * `rows` runs of `bytes` bytes, `stride` apart, from `offset`.  This includes fallback reads of row / column 0 by
 * inactive lanes at the boundaries.  write: 0 read, 1 store, 2 atomic max (the amax). */
#ifndef NVFP4EFFMA_ITEMS_H
#define NVFP4EFFMA_ITEMS_H

#include "nvfp4effma.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { NVFP4A_ITEM_GEMV = 1, NVFP4A_ITEM_GEMV_TC = 2, NVFP4A_ITEM_GEMM_TC = 3,
       NVFP4A_ITEM_GEMM = 4, NVFP4A_ITEM_GEMM_DP4A = 5 };

/* the argument a range is relative to: a tensor's data, block_scale, global_scale or amax; bias->data; the epilogue
 * descriptor's arrays */
enum {
  NVFP4A_ARG_A = 0, NVFP4A_ARG_A_SCALES, NVFP4A_ARG_A_GLOBAL, NVFP4A_ARG_B, NVFP4A_ARG_B_SCALES, NVFP4A_ARG_B_GLOBAL,
  NVFP4A_ARG_C, NVFP4A_ARG_D, NVFP4A_ARG_D_AMAX, NVFP4A_ARG_BIAS, NVFP4A_ARG_NORM_PARTS, NVFP4A_ARG_D2, NVFP4A_ARG_W2,
  NVFP4A_ARG_SQ
};

typedef struct {
  int arg, write;
  long long offset, bytes, stride;
  int rows;
} nvfp4a_item_range;

#define NVFP4A_ITEM_ARGS_BYTES 256
typedef struct {
  int kind, items, rows, cols, threads, smem_bytes, regs, arch;
  char entry[160], args_type[48];
  int args_bytes;
  union { long long align; unsigned char bytes[NVFP4A_ITEM_ARGS_BYTES]; } args;
  long long geom[32];  /* (nvfp4a_item_access) */
} nvfp4a_items;

nvfp4a_status nvfp4a_gemm_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B, int ldb,
                                   float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                   const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items);
nvfp4a_status nvfp4a_gemv_tc_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb,
                                      float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                      const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items);
nvfp4a_status nvfp4a_gemm_tc_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta,
                                      const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                      const nvfp4a_epilogue *ex, nvfp4a_items *items);
/* The ranges item `item` reads and writes, at most max of them to ranges; returns their total number.
 * max = 0, ranges = NULL queries the count.  NULL items, an out-of-range item, max < 0 or NULL ranges with max > 0
 * returns -1. */
int nvfp4a_item_access(const nvfp4a_items *items, int item, nvfp4a_item_range *ranges, int max);

#ifdef __cplusplus
}
#endif
#endif /* NVFP4EFFMA_ITEMS_H */
