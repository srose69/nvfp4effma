# Performance

Results per GPU are in [benches/table.md](../benches/table.md) (one row per GPU) and in
`benches/<gpu>.md` (full tables and the conditions of each run). This page describes how they are
measured, how to add a GPU, and what the GTX 1080 numbers say.

## Running the benchmarks

```sh
bench/run_bench.sh                                          # GPU 0
GPU=1 COOLING="open bench, 100 % fan" NOTES="..." bench/run_bench.sh
```

The script builds the benchmarks, sets the board up (below), runs `bench_gemm`, `bench_gemv` and
`bench_quant`, and writes `benches/<gpu>.md` plus that GPU's row in `benches/table.md`
(`bench/report.py`). Raw output, including clock / temperature / power logs, stays in
`out/bench_<gpu>/`. It needs `sudo` for `nvidia-smi` and restores every setting on exit.
`make bench` runs the same benchmarks without touching the board settings.

**Adding a GPU:** run the script on it and open a pull request with `benches/<gpu>.md` and the
updated `benches/table.md`. Results from any GPU without native FP4 are welcome (sm_50 … sm_90); please
describe the cooling in `COOLING`.

## Methodology

- **Median per row.** Every row (one kernel, one shape) gets one untimed call, then 2 s of idle GPU
  (`NVFP4A_BENCH_COOL`), then 50–2000 timed calls (`NVFP4A_BENCH_REPS`); the median is reported. The
  benchmarks are separated by 10 s of idle GPU and there is no warm-up, so every row starts from a
  similar temperature.
- **FP32 peak at the row's own clock.** % of FP32 peak = GMAC/s / (SMs × 128 × SM clock), one FFMA = one
  MAC (the DP4A kernel does more than one MAC per issue slot and exceeds 100 %), with the SM clock sampled through NVML every 5 ms while the row runs (`bench/bench_util.h`).
  GPU Boost moves the clock with temperature and power, and kernels draw different power, so one clock
  for the whole run would credit some rows with a clock they never had.
- **Board settings.** The power limit is raised to its maximum and the SM clock is allowed up to its
  top bin; the memory clock is locked at its maximum. Without a display, GeForce drivers run CUDA work
  in the P2 state with a reduced memory clock (4513 instead of 5005 MHz on a GTX 1080, about 10 % of
  bandwidth); the lock avoids that. The per-GPU file records what was actually set and the clock,
  temperature, power and throttling seen during each benchmark.
- **Bandwidth.** DRAM peak = memory clock × bus width; a device-to-device copy (read + write) is
  measured as the practical ceiling for streaming kernels.
- **Bandwidth of NVFP4 data** counts the values and the block scales (4.5 bits per value).
- **Baselines.** cuBLAS on FP32 data of the same shape: SGEMM for GEMM, SGEMV and SGEMM n = 8 for the
  GEMV path — the fastest way to run the same model on these GPUs without the 4-bit weights. CUTLASS
  and llama.cpp are not measured.

## What the GTX 1080 numbers say

Full tables: [benches/1080.md](../benches/1080.md).

- **NVFP4 × NVFP4 and NVFP4 × INT8 GEMM are 2× cuBLAS SGEMM**: 9.5 ms at 4096³ (7.2 TMAC/s,
  197–202 MAC/clk/SM) vs 19.4–19.5 ms. On Pascal this is the only way past the FP32 FFMA rate: 4 exact
  products per DP4A ([design.md](design.md)). Accumulating into an FP32 D costs 4 % more (the epilogue reads
  D and nothing overlaps it with one block per SM).
- **NVFP4 × BF16 / FP16 / FP32 GEMM runs at SGEMM speed** (19.8–20.0 ms, 75 % of the FP32 peak, one
  FFMA per MAC) while reading 7× fewer weight bytes than SGEMM's FP32 weights.
- **GEMV is bandwidth-bound for small n**: 0.139–0.142 ms for an 8192 × 8192 NVFP4 matrix, 266–271 GB/s
  or 83–85 % of the DRAM peak (94–96 % of a device-to-device copy) — 7.3× cuBLAS SGEMV on FP32 weights. With
  NVFP4 or INT8 activations (DP4A per block) n = 2 runs as fast as n = 1 and n = 8 takes 0.16–0.20 ms,
  7.7–9.6× SGEMM n = 8; with BF16 activations the per-value decode makes n = 4 … 8 compute-bound
  (178 → 109 GB/s), still 4.4× SGEMM at n = 8. Beyond 8 columns the GEMV kernel runs in groups of 8:
  0.32 ms for n = 16 with INT8 activations.
- **Quantization** streams at 87–89 % of the DRAM peak (BF16 → NVFP4: 0.60 ms for 64 M values).

## Compared with llama.cpp

llama.cpp (ggml) has an NVFP4 type with CUDA kernels, including a DP4A path for Pascal; it quantizes the
activations to int8 (q8_1, blocks of 32) for its matmuls, so its accuracy class is W4A8. Same GTX 1080,
same shapes (ggml's own perf shapes m × k = 4096 × 14336 plus 8192 × 8192 and 4096³), FP32 activations
whose int8 quantization is included in both timings, FP32 output. ggml measured with `test-backend-ops
perf` (MUL_MAT), NVFP4EffMA with `nvfp4a_quantize` + `nvfp4a_gemm`; three interleaved rounds (ggml, then
NVFP4EffMA, 20 s idle before each run), median per shape. Stock board settings; the GPU ran at
~1800 MHz during ggml's runs and ~1775 MHz during NVFP4EffMA's (which draw more power).

| m, n, k | ggml NVFP4 | NVFP4EffMA W4A8 | vs ggml | NVFP4EffMA W4A16, exact |
|---|---:|---:|---:|---:|
| 4096, 4096, 4096 | 14.49 ms | 9.89 ms | **1.47×** | 20.86 ms |
| 4096, 1, 14336 | 158 µs | 153 µs | 1.04× | 154 µs |
| 4096, 2, 14336 | 193 µs | 152 µs | 1.27× | 160 µs |
| 4096, 4, 14336 | 357 µs | 182 µs | 1.96× | 188 µs |
| 4096, 8, 14336 | 704 µs | 181 µs | **3.89×** | 328 µs |
| 4096, 16, 14336 | 451 µs | 357 µs | 1.26× | 657 µs |
| 4096, 32, 14336 | 717 µs | 708 µs | 1.01× | 1.33 ms |
| 4096, 64, 14336 | 958 µs | 799 µs | 1.20× | 2.49 ms |
| 4096, 128, 14336 | 2.02 ms | 1.21 ms | 1.67× | 2.54 ms |
| 4096, 512, 14336 | 6.14 ms | 4.25 ms | **1.45×** | 10.53 ms |
| 8192, 1, 8192 | 175 µs | 143 µs | 1.23× | 144 µs |
| 8192, 8, 8192 | 744 µs | 165 µs | **4.51×** | 353 µs |
| 8192, 64, 8192 | 1.21 ms | 755 µs | 1.60× | 2.85 ms |
| 8192, 512, 8192 | 8.36 ms | 5.04 ms | **1.66×** | 11.13 ms |

The W4A16 path is the exact one (no activation quantization at all) and is compute-bound at SGEMM speed
(on Pascal no exact path can beat the FP32 FFMA rate: ≥ ~15 ms at 4096³). The PyTorch `linear` therefore
quantizes BF16 activations to int8 by default beyond 2 rows (`act="auto"`): 10.0 ms at 4096³ including
the quantization, with `act="exact"` for the exact path.
