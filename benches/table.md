# Results by GPU

One row per GPU, produced by `bench/run_bench.sh` (see [docs/performance.md](../docs/performance.md)); the
linked file has the full tables and the conditions of the run. Percentages of the FP32 peak use the SM
clock measured during each row; the baseline is cuBLAS on FP32 data of the same shape.

| GPU | CC | GEMM NVFP4 × NVFP4, GMAC/s | NVFP4 × NVFP4 vs SGEMM | NVFP4 × INT8 vs SGEMM | NVFP4 × BF16 vs SGEMM | GEMV n = 1, BF16 x (% of DRAM peak) | GEMV n = 1 vs SGEMV | Quantize BF16 (% of DRAM peak) | Date |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| [NVIDIA GeForce GTX 1080](1080.md) | 6.1 | 6846 | 1.93× | 2.04× | 0.97× | 271 GB/s (85 %) | 7.6× | 270 GB/s (84 %) | 2026-10-28 |
| [NVIDIA L40S](l40s.md) | 8.9 | 48012 | 1.82× | 1.76× | 1.06× | 1462 GB/s (190 %) | 14.2× | 721 GB/s (94 %) | 2026-10-28 |
