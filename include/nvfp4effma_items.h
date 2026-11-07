// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
/* NVFP4EffMA work items: the work of a call as items that an executor runs from kernels of its own, with what each reads and
 * writes.  Items compute what the call does bit for bit, except that a GEMM_TC tile sums all of k in one CTA where
 * nvfp4a_gemm_tc_ex may split a tile's k between CTAs (stream-K): the same products, the sum grouped differently. */
#ifndef NVFP4EFFMA_ITEMS_H
#define NVFP4EFFMA_ITEMS_H

#include <stddef.h>

#include "nvfp4effma.h"

#ifdef __cplusplus
extern "C" {
#endif

/* kind: NVFP4A_ITEM_GEMV a block of `rows` output rows, every column, over all of k on CUDA cores (n <= 8); NVFP4A_ITEM_GEMV_TC
 * an m16 slice (rows = 16), every column, over all of k on tensor cores (sm_75 .. sm_89); NVFP4A_ITEM_GEMM_TC a `rows` x `cols`
 * tile over all of k on tensor cores, the CTA's warps specialize. */
enum { NVFP4A_ITEM_GEMV = 1, NVFP4A_ITEM_GEMV_TC = 2, NVFP4A_ITEM_GEMM_TC = 3,
       NVFP4A_ITEM_GEMM = 4, NVFP4A_ITEM_GEMM_DP4A = 5, NVFP4A_ITEM_GEMV_PART = 6, NVFP4A_ITEM_GEMV_COMBINE = 7,
       NVFP4A_ITEM_GEMV_TC_PART = 8, NVFP4A_ITEM_GEMV_TC_COMBINE = 9, NVFP4A_ITEM_IMAGE_B = 10, NVFP4A_ITEM_QUANT_INT8 = 11 };

/* the argument a range is relative to: a tensor's data, block_scale, global_scale or amax; bias->data; the epilogue
 * descriptor's arrays */
enum {
  NVFP4A_ARG_A = 0, NVFP4A_ARG_A_SCALES, NVFP4A_ARG_A_GLOBAL, NVFP4A_ARG_B, NVFP4A_ARG_B_SCALES, NVFP4A_ARG_B_GLOBAL,
  NVFP4A_ARG_C, NVFP4A_ARG_D, NVFP4A_ARG_D_AMAX, NVFP4A_ARG_BIAS, NVFP4A_ARG_NORM_PARTS, NVFP4A_ARG_D2, NVFP4A_ARG_W2,
  NVFP4A_ARG_SQ, NVFP4A_ARG_PARTIALS, NVFP4A_ARG_B_IMAGE,
  NVFP4A_ARG_Q_SRC, NVFP4A_ARG_Q_DATA, NVFP4A_ARG_Q_SCALES  /* nvfp4a_quantize_int8_items */
};

/* A range: `rows` runs of `bytes` bytes, `stride` apart, from `offset` relative to the argument's pointer (nvfp4a_item_access);
 * includes fallback reads of row / column 0 by inactive lanes at the boundaries.  write: 0 read, 1 store, 2 atomic max (the amax). */
typedef struct {
  int arg, write;
  long long offset, bytes, stride;
  int rows;
} nvfp4a_item_range;

/* The weight A (data/image, block scales, global), bias and w2 are immutable for the executor launch.  B (data/image, block and
 * global scales), C (including C = D) and norm_parts are read through coherent L2: another CTA may produce them in the same
 * launch.  The executor orders GPU-scope release/acquire before calling the entry and finishes all consuming threads before reusing inputs. */
/* It also publishes outputs after all writing threads finish: on sm_6x, a device fence in every writer, a CTA barrier and then an
 * atomic signal.  The entries do not implement the executor's inter-CTA waits or signals. */

/* items: their number (nvfp4a_item_access: which outputs each computes).  rows, cols: the outputs of an item (cols 0: every column).
 * threads: the CTA one item runs on, exactly (all its threads call the entry, which synchronizes them).  smem_bytes: shared memory the
 * executor hands over (16-byte aligned); synchronize the CTA after an entry before reusing it or beginning the next item. */
/* regs: registers a thread of the library's own kernel for this work uses (the executor's may differ).  arch: virtual architecture of
 * that instance (e.g. 75); compile the device entries for compute_<arch> (the .cuh: nvcc or NVRTC), so their CTA shape and
 * shared layout match the plan. */
/* entry: the device entry (include/nvfp4effma_items.cuh), called entry(*(const args_type *)p, item, thread, smem), p a 16-byte aligned
 * device copy of args.bytes[0 .. args_bytes - 1], thread the thread's index in the CTA.  The data the arguments point to must stay
 * where it was at the time of the description. */
#define NVFP4A_ITEM_ARGS_BYTES 256
typedef struct {
  int kind, items, rows, cols, threads, smem_bytes, regs, arch;
  char entry[160], args_type[48];
  int args_bytes;
  union { long long align; unsigned char bytes[NVFP4A_ITEM_ARGS_BYTES]; } args;
  long long geom[32];  /* (nvfp4a_item_access) */
} nvfp4a_items;

/* nvfp4a_gemm_ex_items / nvfp4a_gemv_tc_ex_items / nvfp4a_gemm_tc_ex_items take the arguments of nvfp4a_gemm_ex /
 * nvfp4a_gemv_tc_ex / nvfp4a_gemm_tc_ex without the workspace (validated the same way; ex may be NULL: the plain epilogue)
 * and describe the call on the current device.  A blocked operand needs linear E4M3 block scales (A; NVFP4 B is not taken); B BF16 or INT8. */
nvfp4a_status nvfp4a_gemm_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B, int ldb,
                                   float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                   const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items);
nvfp4a_status nvfp4a_gemv_tc_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb,
                                      float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                      const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, nvfp4a_items *items);
nvfp4a_status nvfp4a_gemm_tc_ex_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, float beta,
                                      const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd, const nvfp4a_tensor *bias,
                                      const nvfp4a_epilogue *ex, nvfp4a_items *items);
/* The GEMV on CUDA cores cut along k (n <= 8, A with linear E4M3 scales, B BF16 or INT8): the call as two passes of items.
 * NVFP4A_ITEM_GEMV_PART: item = part * blocks + block, a block of `rows` output rows over part p of k, the runs of 512-value
 * units [p U / k_parts, (p + 1) U / k_parts), U = ceil(k / 512); its lane-reduced sums go to the FP32 partials. */
/* NVFP4A_ITEM_GEMV_COMBINE: item = block, the block's partials added in part order (the first part, then each next one, one FP32
 * addition each), then the call's epilogue -- everything the whole-k item does besides the sums. */
/* A (row, column) sum is the whole-k item's with its additions regrouped by part: equal within the FP32 rounding of the regrouping,
 * the same for every executor and launch shape; k_parts = 1 is the whole-k item's, bit for bit.  k_parts is 1 ..
 * nvfp4a_gemv_split_max_parts(k); arguments as nvfp4a_gemm_ex_items. */
/* The partials are `nvfp4a_gemv_split_workspace_bytes` bytes (executor-owned, 16-byte aligned), written by the part items and
 * read by the combine items; nvfp4a_item_access names them NVFP4A_ARG_PARTIALS. */
int nvfp4a_gemv_split_max_parts(int k);
size_t nvfp4a_gemv_split_workspace_bytes(int m, int n, int k_parts, nvfp4a_dtype b_dtype, const nvfp4a_epilogue *ex);
nvfp4a_status nvfp4a_gemv_split_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, int lda, const nvfp4a_tensor *B, int ldb,
                                      float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                      const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, int k_parts, float *partials,
                                      nvfp4a_items *part, nvfp4a_items *combine);

/* The GEMV on tensor cores cut along k the same way (n <= 8, B BF16 or INT8; sm_75 .. sm_89): NVFP4A_ITEM_GEMV_TC_PART (item =
 * part * slices + slice) runs the slice's warps over part p of its groups (a group = 4 k-tiles of 64 values; groups [p G / k_parts,
 * (p + 1) G / k_parts), G = ceil(k / 256)) and stores the warps' sums (warp order, scaled back exactly) of the 16 rows to the partials. */
/* NVFP4A_ITEM_GEMV_TC_COMBINE (item = slice) adds a slice's partials in part order and runs the epilogue.  k_parts = 1 is the
 * whole-k item's, bit for bit; more parts regroup the additions (FP32 rounding), the same bits for every executor and launch shape. */
/* k_parts is 1 .. nvfp4a_gemv_tc_split_max_parts(k); the FP32 partials are nvfp4a_gemv_tc_split_workspace_bytes bytes, 16-byte
 * aligned (NVFP4A_ARG_PARTIALS).  The arguments are those of nvfp4a_gemv_tc_ex_items. */
int nvfp4a_gemv_tc_split_max_parts(int k);
size_t nvfp4a_gemv_tc_split_workspace_bytes(int m, int n, int k_parts);
nvfp4a_status nvfp4a_gemv_tc_split_items(int m, int n, int k, float alpha, const nvfp4a_tensor *A, const nvfp4a_tensor *B, int ldb,
                                         float beta, const nvfp4a_tensor *C, int ldc, const nvfp4a_tensor *D, int ldd,
                                         const nvfp4a_tensor *bias, const nvfp4a_epilogue *ex, int k_parts, float *partials,
                                         nvfp4a_items *part, nvfp4a_items *combine);

/* The image of a BF16 B (sm_80 .. sm_89) or INT8 B (data and FP32 block scales as for nvfp4a_gemm; sm_75 .. sm_89), [n][ldb] values,
 * the activations of nvfp4a_gemm_tc_ex_items, as items (nvfp4a_tc_image's NVFP4A_TC_B): 256 threads, no shared memory; item = k-tile +
 * (k / 64) * (128-column block) for BF16, k-tile pair + ((k / 64 + 1) / 2) * block for INT8. */
/* image is nvfp4a_tc_image_bytes(NVFP4A_TC_B, dtype, n, k) bytes, 16-byte aligned; the items read NVFP4A_ARG_B (and NVFP4A_ARG_B_SCALES)
 * through L2 (the activations may come from another item of the launch) and write NVFP4A_ARG_B_IMAGE; the columns past n in the last
 * block are zeros.  nvfp4a_items_n_device applies: a block of columns wholly past the count is left alone. */
nvfp4a_status nvfp4a_tc_image_b_items(int n, int k, const nvfp4a_tensor *B, int ldb, void *image, nvfp4a_items *items);

/* nvfp4a_quantize to INT8 as items (bit for bit): src [rows][lds] BF16, FP16 or FP32 -> dst int8 [rows][ldd], FP32 scales [rows][ld_scale] (0: cols / 16).
 * item = a run of items->rows rows, 256 threads, no shared memory; reads NVFP4A_ARG_Q_SRC through L2, writes NVFP4A_ARG_Q_DATA and
 * NVFP4A_ARG_Q_SCALES (dst->amax is not computed).  nvfp4a_items_n_device applies: rows past the count are left alone. */
nvfp4a_status nvfp4a_quantize_int8_items(const nvfp4a_tensor *src, int lds, const nvfp4a_tensor *dst, int ldd, int rows, int cols,
                                         nvfp4a_items *items);

/* Columns of a GEMM's items (GEMV with more than 8 columns, GEMM, GEMM_DP4A, GEMM_TC) in device memory (1 .. n, read when an item runs):
 * items compute min(n, *n_device) columns, tiles wholly past them return at once (a prefill chunk whose token count the device
 * knows).  Item numbers, tiles and ranges stay those of n columns; within a tile or group of 8 columns the columns past it may hold values. */
nvfp4a_status nvfp4a_items_n_device(nvfp4a_items *items, const int *n_device);

/* The ranges item `item` reads and writes, at most max of them to ranges; returns their total number.
 * max = 0, ranges = NULL queries the count.  NULL items, an out-of-range item, max < 0 or NULL ranges with max > 0
 * returns -1. */
int nvfp4a_item_access(const nvfp4a_items *items, int item, nvfp4a_item_range *ranges, int max);

#ifdef __cplusplus
}
#endif
#endif /* NVFP4EFFMA_ITEMS_H */
