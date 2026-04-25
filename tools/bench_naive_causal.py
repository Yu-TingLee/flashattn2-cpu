import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import argparse
from flash_attn.attn import naive_attention_causal
from flash_attn.io import read_QKV


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()

    output_dir = os.path.join("outputs", "naive_causal_attn")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "runtime.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("testset,T,d,runtime\n")

    completed_runs = set()
    with open(log_path, "r") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                completed_runs.add((int(parts[0]), int(parts[1]), int(parts[2])))

    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    for i in range(args.num_testsets):
        if (i, args.T, args.d) in completed_runs:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = read_QKV(data_dir, args.T, args.d)

        t0 = time.perf_counter()
        O_causal = naive_attention_causal(Q, Kt, V, scale)
        runtime = time.perf_counter() - t0

        print("-" * 30)
        print(f"Runtime {i}: {runtime:.4f} seconds.")
        print("-" * 30)

        with open(log_path, "a") as f:
            f.write(f"{i},{args.T},{args.d},{runtime:.6f}\n")

        np.save(os.path.join(data_dir, f"O_causal_T{args.T}_d{args.d}.npy"), O_causal)
