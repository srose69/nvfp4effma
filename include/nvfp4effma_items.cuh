// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// Device entries of the work items (named in include/nvfp4effma_items.h), for an executor's kernels, compiled ahead of time with nvcc
// (-std=c++17 -gencode arch=compute_<arch>,code=sm_<the device's>) or at run time with NVRTC (--std=c++17
// --device-as-default-execution-space, compute_<arch>, the PTX JIT-compiled on the device); include paths: library include/ and src/,
// CUDA include/ and (CUDA 13+) include/cccl. Every thread of the CTA calls the entry.
#pragma once
#include "../src/gemv_nvfp4effma.cuh"
#include "../src/quantize_item.cuh"
#include "../src/tensors/gemv_tc.cuh"  // (and gemm_tc.cuh)
