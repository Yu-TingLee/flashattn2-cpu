import numpy as np
import os
import time
import argparse
from utils import read_QKV


def fa2_forward(Q, Kt, V, T, d, M_bytes, scale):
    I = M_bytes // (4 * d * 4)
    J = min(I, d)
    if I < 1: I = 1
    if J < 1: J = 1

    Tr = (T + I - 1) // I
    Tc = (T + J - 1) // J

    O = np.zeros((T, d), dtype=np.float32)

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

            Kt_j = np.asfortranarray(Kt[:, col_start:col_end])
            V_j  = np.ascontiguousarray(V[col_start:col_end, :])

            S = (Q_i @ Kt_j) * scale

            m_block = S.max(axis=1)
            m_new = np.maximum(m, m_block)

            alpha = np.exp(m - m_new)
            P = np.exp(S - m_new[:, None])

            l_new = alpha * l + P.sum(axis=1)

            O_block = O[row_start:row_end, :]
            O[row_start:row_end, :] = (O_block * alpha[:, None]) + (P @ V_j)

            m, l = m_new, l_new

        O[row_start:row_end, :] /= l[:, None]

    return O


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--M_bytes", type=int, default=131072)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()

    output_dir = os.path.join("outputs", "fa2")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "runtime.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("testset,T,d,M_bytes,runtime,is_correct\n")

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

        t0 = time.perf_counter()
        O_np = fa2_forward(Q, Kt, V, args.T, args.d, args.M_bytes, scale)
        t1 = time.perf_counter()

        runtime = t1 - t0
        print("-" * 30)
        print(f"Runtime {i}: {runtime:.4f} seconds.")
        print("-" * 30)

        O_ref = np.load(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"))
        is_correct = np.allclose(O_np, O_ref, atol=1e-3, rtol=1e-3)

        with open(log_path, "a") as f:
            f.write(f"{i},{args.T},{args.d},{args.M_bytes},{runtime:.6f},{is_correct}\n")
