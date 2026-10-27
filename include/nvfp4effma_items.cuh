// SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
// Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
// The device entries of the work items (include/nvfp4effma_items.h names them and their argument types), for an
// executor's kernels.  NVRTC: --std=c++17 --device-as-default-execution-space, include paths: the library's include/ and
// src/, CUDA's include/ and (CUDA 13+) include/cccl.  Compile for the plan's compute_<arch> and JIT on the current
// device.  Every thread of the described CTA calls the entry; synchronize the CTA before reusing its shared memory.
#pragma once
#include "../src/gemv_nvfp4effma.cuh"
#include "../src/tensors/gemv_tc.cuh"  // (and gemm_tc.cuh)
