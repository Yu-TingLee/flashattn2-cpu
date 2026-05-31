#!/usr/bin/env bash
# Runs every benchmark sweep, then plots. Build the C++ benchmarks first:
#   cmake -S . -B build && cmake --build build --parallel
set -euo pipefail

NUM_TESTSETS=50
T_VALUES=(256 512 1024 2048 4096 8192)
D_VALUES=(64 128)
NUM_THREADS=(8 16)
SCHEDULES=(static dynamic work_stealing)

# Cache budgets M from 32 KiB to 1 MiB.
M_VALUES=()
for k in 1 2 4 5 6 8 9 10 12 13 14 16 17 18 20 21 22 24 25 26 28 29 30 32; do
  M_VALUES+=($((k * 32768)))
done
PROFILE_T_VALUES=(512 2048 8192)
PROFILE_M_VALUES=(16384 262144 1048576)

BIN=./build/bin

# Calls "$@" once for every (T, d) pair, with T and d set.
for_each_t_d() {
  for T in "${T_VALUES[@]}"; do
    for d in "${D_VALUES[@]}"; do
      "$@"
    done
  done
}

run_naive() {
  echo "Naive attention: T=$T, d=$d"
  python -m benchmarks.bench_naive --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
  "$BIN/bench_naive" --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
}

run_fa2() {
  for M in "${M_VALUES[@]}"; do
    echo "FA-2: T=$T, d=$d, M=$((M / 1024))KiB"
    python -m benchmarks.bench_fa2 --T "$T" --d "$d" --M_bytes "$M" --num_testsets "$NUM_TESTSETS"
    "$BIN/bench_fa2" --T "$T" --d "$d" --M_bytes "$M" --num_testsets "$NUM_TESTSETS"
    for nt in "${NUM_THREADS[@]}"; do
      python -m benchmarks.bench_fa2_numba --T "$T" --d "$d" --M_bytes "$M" \
        --num_testsets "$NUM_TESTSETS" --num_threads "$nt"
      "$BIN/bench_fa2_mt" --T "$T" --d "$d" --M_bytes "$M" \
        --num_testsets "$NUM_TESTSETS" --num_threads "$nt"
    done
  done
}

run_profile() {
  for T in "${PROFILE_T_VALUES[@]}"; do
    for d in "${D_VALUES[@]}"; do
      for M in "${PROFILE_M_VALUES[@]}"; do
        echo "FA-2 profile: T=$T, d=$d, M=$((M / 1024))KiB"
        python -m benchmarks.bench_fa2_profile --T "$T" --d "$d" --M_bytes "$M" \
          --num_testsets "$NUM_TESTSETS"
        "$BIN/bench_fa2_profile" --T "$T" --d "$d" --M_bytes "$M" \
          --num_testsets "$NUM_TESTSETS"
      done
    done
  done
}

run_naive_causal() {
  echo "Masked naive attention: T=$T, d=$d"
  python -m benchmarks.bench_naive_causal --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
  "$BIN/bench_naive_causal" --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
}

run_fa2_causal() {
  for M in "${M_VALUES[@]}"; do
    echo "Masked FA-2: T=$T, d=$d, M=$((M / 1024))KiB"
    "$BIN/bench_fa2_causal" --T "$T" --d "$d" --M_bytes "$M" --num_testsets "$NUM_TESTSETS"
    for sched in "${SCHEDULES[@]}"; do
      for nt in "${NUM_THREADS[@]}"; do
        "$BIN/bench_fa2_causal_mt" --T "$T" --d "$d" --M_bytes "$M" \
          --num_testsets "$NUM_TESTSETS" --schedule "$sched" --num_threads "$nt"
      done
    done
  done
}

python data_generation.py --T 8192 --d 128 --num_testsets "$NUM_TESTSETS"

for_each_t_d run_naive
for_each_t_d run_fa2
run_profile
for_each_t_d run_naive_causal
for_each_t_d run_fa2_causal

python plot.py
