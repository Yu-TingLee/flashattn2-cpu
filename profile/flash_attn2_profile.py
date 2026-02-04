import numpy as np
import os
import time
import argparse
import sys

_PROFILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_PROFILE_DIR, "..")) 
sys.path.insert(0, _PROJECT_ROOT)
from utils import read_QKV

def fa2_forward_profile(Q, Kt, V, T, d, M_bytes, scale):
    I = M_bytes // (4 * d * 4)
    J = min(I, d)
    if I < 1: I = 1
    if J < 1: J = 1

    Tr = (T + I - 1) // I
    Tc = (T + J - 1) // J

    O = np.zeros((T, d), dtype=np.float32)

    perf = time.perf_counter

    t7 = 0.0   # Line 7: view/slice construction (Kt_j, V_j)
    t8 = 0.0   # Line 8: S = Q_i @ Kt_j
    t10 = 0.0  # Line 10: softmax/P + l,m updates
    t11 = 0.0  # Line 11: O update (includes P @ V_j)

    t0 = perf()

    for i in range(Tr):
        row_start = i * I
        row_end = min(row_start + I, T)
        cur_I = row_end - row_start

        l = np.zeros(cur_I, dtype=np.float32)
        m = np.full(cur_I, -np.inf, dtype=np.float32)

        Q_i = Q[row_start:row_end, :]

        for j in range(Tc):
            col_start = j * J
            col_end = min(col_start + J, T)

            # -------- Line 7  --------
            tt = perf()
            Kt_j = np.asfortranarray(Kt[:, col_start:col_end])
            V_j  = np.ascontiguousarray(V[col_start:col_end, :])
            t7 += perf() - tt

            # -------- Line 8 --------
            tt = perf()
            S = (Q_i @ Kt_j) * scale
            t8 += perf() - tt

            # -------- Line 10 --------
            tt = perf()
            m_block = S.max(axis=1)
            m_new = np.maximum(m, m_block)

            alpha = np.exp(m - m_new)
            P = np.exp(S - m_new[:, None])

            l_new = alpha * l + P.sum(axis=1)
            t10 += perf() - tt

            # -------- Line 11 --------
            tt = perf()
            O_block = O[row_start:row_end, :]
            O[row_start:row_end, :] = (O_block * alpha[:, None]) + (P @ V_j)
            t11 += perf() - tt

            m, l = m_new, l_new

        O[row_start:row_end, :] /= l[:, None]

    runtime = perf() - t0
    others = runtime - (t7 + t8 + t10 + t11)
    if others < 0.0:
        others = 0.0

    return O, runtime, t7, t8, t10, t11, others


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--M_bytes", type=int, default=131072)
    parser.add_argument("--num_testsets", type=int, default=1)
    args = parser.parse_args()

    output_dir = os.path.join("outputs", "fa2_profile")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "runtime.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("testset,T,d,M_bytes,runtime,line7,line8,line10,line11,others\n")

    completed_runs = set()
    with open(log_path, "r") as f:
        _ = next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                completed_runs.add((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))

    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    for i in range(args.num_testsets):
        key = (i, args.T, args.d, args.M_bytes)
        if key in completed_runs:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = read_QKV(data_dir, args.T, args.d)

        _, runtime, line7, line8, line10, line11, others = fa2_forward_profile(
            Q, Kt, V, args.T, args.d, args.M_bytes, scale
        )

        with open(log_path, "a") as f:
            f.write(
                f"{i},{args.T},{args.d},{args.M_bytes},"
                f"{runtime:.6f},{line7:.6f},{line8:.6f},{line10:.6f},{line11:.6f},{others:.6f}\n"
            )
        print("-" * 30)
        print(
            f"Runtime {i}: {runtime:.6f} seconds.\n"
            f"line7 {line7:.6f}s, line8 {line8:.6f}s, line10 {line10:.6f}s, line11 {line11:.6f}s, others {others:.6f}s"
        )
        print("-" * 30)
