#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
# Benchmark run behind benches/<gpu>.md (methodology: docs/performance.md).
#   bench/run_bench.sh                      GPU 0
#   GPU=1 COOLING="open bench, 100 % fan" NOTES="..." bench/run_bench.sh
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
PACK_COOL=${PACK_COOL:-10}
# per benchmark: NVFP4A_BENCH_WARMUP FP32 products (2048 x 2048, 0.5 s apart) raise the clocks; each row starts after
# NVFP4A_BENCH_COOL s of idle GPU; benchmarks are separated by PACK_COOL s; a row reports its median time and mean SM clock
export NVFP4A_BENCH_COOL=${NVFP4A_BENCH_COOL:-2}
export NVFP4A_BENCH_WARMUP=${NVFP4A_BENCH_WARMUP:-4}
GEMM_REPS=${GEMM_REPS:-50}       # timed calls per row
GEMV_REPS=${GEMV_REPS:-2000}
QUANT_REPS=${QUANT_REPS:-500}

q() { nvidia-smi -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }
NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)
# GPU_NAME overrides the result file name (default: from the GPU name); raw results go to out/bench_<file>/,
# bench/report.py turns them into benches/<file>.md and the GPU's row in benches/table.md
FILE=${GPU_NAME:-$(python3 -c "import sys; sys.path.insert(0, 'bench'); from report import slug; print(slug('''$NAME'''))")}
OUT=out/bench_$FILE
rm -rf "$OUT" && mkdir -p "$OUT"

echo "== $NAME (GPU $GPU) -> benches/$FILE.md; processes on the GPU (should be none):"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
make -s bench-bin || exit 1

# needs sudo for nvidia-smi: power limit to its maximum, SM clock allowed up to its top, memory clock locked at its
# maximum; all restored on exit, also on Ctrl+C
PL_OLD=$(q power.limit); PL_MAX=$(q power.max_limit); SM_MAX=$(q clocks.max.sm); MEM_MAX=$(q clocks.max.mem)
restore() {
  sudo nvidia-smi -i "$GPU" -rgc > /dev/null
  sudo nvidia-smi -i "$GPU" -rmc > /dev/null 2>&1
  sudo nvidia-smi -i "$GPU" -pl "$PL_OLD" > /dev/null
  echo "restored: power limit $PL_OLD W, SM and memory clocks unlocked"
}
trap restore EXIT
sudo -v || exit 1
sudo nvidia-smi -i "$GPU" -pm 1 > /dev/null
sudo nvidia-smi -i "$GPU" -pl "$PL_MAX" > /dev/null && PL="$PL_MAX W (maximum; default $PL_OLD W)" || PL="$PL_OLD W"
LOCKS=""
sudo nvidia-smi -i "$GPU" -lgc "$SM_MAX,$SM_MAX" > /dev/null && LOCKS="SM clock allowed up to $SM_MAX MHz" || LOCKS="SM clock: GPU Boost"
if sudo nvidia-smi -i "$GPU" -lmc "$MEM_MAX,$MEM_MAX" > /dev/null 2>&1; then LOCKS="$LOCKS, memory clock locked at $MEM_MAX MHz"
else LOCKS="$LOCKS, memory clock not lockable"; echo "!! memory clock lock not supported: check the memory clock in the logs"; fi

{
  echo "name=$NAME"
  echo "file=$FILE"
  echo "compute_cap=$(q compute_cap)"
  echo "memory=$(q memory.total) MiB"  # SM count and bus width: from the benchmarks' headers (report.py)
  echo "driver=$(q driver_version)"
  echo "cuda=$(${CUDA_HOME:-/usr/local/cuda}/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  echo "library=$(sed -n 's/^#define NVFP4EFFMA_VERSION "\(.*\)"/\1/p' include/nvfp4effma.h)"
  echo "date=$(date +%Y-%m-%d)"
  echo "power_limit=$PL"
  echo "clock_lock=$LOCKS"
  echo "cooling=${COOLING:-}"
  echo "notes=${NOTES:-}"
} > "$OUT/env.txt"

run() {  # name reps
  echo "== $1 (after ${PACK_COOL} s idle)"
  sleep "$PACK_COOL"
  nvidia-smi -i "$GPU" --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active \
    --format=csv,noheader -lms 250 > "$OUT/clocks_$1.csv" &
  local log=$!
  CUDA_VISIBLE_DEVICES=$GPU NVFP4A_BENCH_REPS=$2 ./out/$1 | tee "$OUT/$1.txt"
  kill $log; wait $log 2> /dev/null
}
run bench_gemm "$GEMM_REPS"
run bench_gemv "$GEMV_REPS"
run bench_quant "$QUANT_REPS"
python3 bench/report.py "$OUT" benches
