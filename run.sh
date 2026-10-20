#!/usr/bin/env bash
# Runs every benchmark the paper uses, then plots. Build the C++ benchmarks first:
#   cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-11 && cmake --build build --parallel
set -euo pipefail

NUM_TESTSETS=50
T_VALUES=(512 1024 2048 4096 8192)
D_VALUES=(64 128)
NUM_THREADS=(8 16)
SDPA_THREADS=(8 16)
SCHEDULES=(static dynamic work_stealing)

# Cache budgets M from 32 KiB to 1 MiB.
M_VALUES=()
for k in 1 2 4 5 6 8 9 10 12 13 14 16 17 18 20 21 22 24 25 26 28 29 30 32; do
  M_VALUES+=($((k * 32768)))
done
PROFILE_T_VALUES=(512 8192)
PROFILE_M_VALUES=(32768 1048576)

BIN=./build/bin

for_each_t_d() {
  for T in "${T_VALUES[@]}"; do
    for d in "${D_VALUES[@]}"; do
      "$@"
    done
  done
}

# The NumPy runs also write the reference outputs.
run_naive() {
  echo "Naive attention: T=$T, d=$d"
  python -m benchmarks.bench_naive --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
  "$BIN/bench_naive" --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
  echo "Masked naive attention: T=$T, d=$d"
  python -m benchmarks.bench_naive_causal --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
  "$BIN/bench_naive_causal" --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS"
}

run_sdpa() {
  for nt in "${SDPA_THREADS[@]}"; do
    echo "PyTorch SDPA: T=$T, d=$d, threads=$nt"
    python -m benchmarks.bench_sdpa --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS" \
      --num_threads "$nt"
    python -m benchmarks.bench_sdpa --T "$T" --d "$d" --num_testsets "$NUM_TESTSETS" \
      --num_threads "$nt" --causal
  done
}

run_fa2() {
  for M in "${M_VALUES[@]}"; do
    echo "FA-2: T=$T, d=$d, M=$((M / 1024))KiB"
    for nt in "${NUM_THREADS[@]}"; do
      "$BIN/bench_fa2_mt" --T "$T" --d "$d" --M_bytes "$M" \
        --num_testsets "$NUM_TESTSETS" --num_threads "$nt"
    done
  done
}

run_fa2_causal() {
  for M in "${M_VALUES[@]}"; do
    echo "Masked FA-2: T=$T, d=$d, M=$((M / 1024))KiB"
    for sched in "${SCHEDULES[@]}"; do
      for nt in "${NUM_THREADS[@]}"; do
        "$BIN/bench_fa2_causal_mt" --T "$T" --d "$d" --M_bytes "$M" \
          --num_testsets "$NUM_TESTSETS" --schedule "$sched" --num_threads "$nt"
      done
    done
  done
}

run_profile_mt() {
  for T in "${PROFILE_T_VALUES[@]}"; do
    for d in "${D_VALUES[@]}"; do
      for M in "${PROFILE_M_VALUES[@]}"; do
        for nt in "${NUM_THREADS[@]}"; do
          echo "FA-2 MT profile: T=$T, d=$d, M=$((M / 1024))KiB, threads=$nt"
          "$BIN/bench_fa2_mt_profile" --T "$T" --d "$d" --M_bytes "$M" \
            --num_testsets "$NUM_TESTSETS" --num_threads "$nt"
          for sched in "${SCHEDULES[@]}"; do
            "$BIN/bench_fa2_mt_profile" --T "$T" --d "$d" --M_bytes "$M" \
              --num_testsets "$NUM_TESTSETS" --num_threads "$nt" --causal 1 --schedule "$sched"
          done
        done
      done
    done
  done
}

python data_generation.py --T 8192 --d 128 --num_testsets "$NUM_TESTSETS"

for_each_t_d run_naive
for_each_t_d run_sdpa
for_each_t_d run_fa2
for_each_t_d run_fa2_causal
run_profile_mt

python plot.py --sdpa
