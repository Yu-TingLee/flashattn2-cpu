import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import argparse
from flash_attn.attn_numba import fa2_forward
from flash_attn.io import read_QKV


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--M_bytes", type=int, default=131072)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()

    output_dir = os.path.join("outputs", "fa2_jit")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "runtime.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("testset,T,d,M_bytes,runtime,is_correct\n")

    completed_runs = set()
    with open(log_path, "r") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                completed_runs.add((int(parts[0]), int(parts[1]),
                                    int(parts[2]), int(parts[3])))

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    # Warm-up: trigger Numba JIT compilation before timed runs
    Q0, Kt0, V0 = read_QKV(os.path.join("data", "testset0"), args.T, args.d)
    Q0  = np.ascontiguousarray(Q0)
    Kt0 = np.asfortranarray(Kt0)
    V0  = np.ascontiguousarray(V0)
    _ = fa2_forward(Q0, Kt0, V0, args.T, args.d, args.M_bytes, scale)

    for i in range(args.num_testsets):
        key = (i, args.T, args.d, args.M_bytes)
        if key in completed_runs:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = read_QKV(data_dir, args.T, args.d)
        Q  = np.ascontiguousarray(Q)
        Kt = np.asfortranarray(Kt)
        V  = np.ascontiguousarray(V)

        t0 = time.perf_counter()
        O_numba = fa2_forward(Q, Kt, V, args.T, args.d, args.M_bytes, scale)
        runtime = time.perf_counter() - t0

        print("-" * 30)
        print(f"Runtime {i}: {runtime:.4f} seconds.")
        print("-" * 30)

        O_ref = np.load(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"))
        is_correct = np.allclose(O_numba, O_ref, atol=1e-3, rtol=1e-3)

        with open(log_path, "a") as f:
            f.write(f"{i},{args.T},{args.d},{args.M_bytes},{runtime:.6f},{is_correct}\n")
