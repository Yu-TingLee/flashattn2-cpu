import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from flash_attn.attn_profile import fa2_forward_profile
from flash_attn.io import read_QKV


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
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                completed_runs.add((int(parts[0]), int(parts[1]),
                                    int(parts[2]), int(parts[3])))

    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    for i in range(args.num_testsets):
        key = (i, args.T, args.d, args.M_bytes)
        if key in completed_runs:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = read_QKV(data_dir, args.T, args.d)

        _, runtime, t7, t8, t10, t11, others = fa2_forward_profile(
            Q, Kt, V, args.T, args.d, args.M_bytes, scale
        )

        with open(log_path, "a") as f:
            f.write(
                f"{i},{args.T},{args.d},{args.M_bytes},"
                f"{runtime:.6f},{t7:.6f},{t8:.6f},{t10:.6f},{t11:.6f},{others:.6f}\n"
            )

        print("-" * 30)
        print(
            f"Runtime {i}: {runtime:.6f} s\n"
            f"line7 {t7:.6f}s  line8 {t8:.6f}s  "
            f"line10 {t10:.6f}s  line11 {t11:.6f}s  others {others:.6f}s"
        )
        print("-" * 30)
