import argparse
import os
import time

import numpy as np

from benchmarks.bench_utils import RuntimeLog, print_runtime
from flash_attn.flash_attention import fa2_forward
from flash_attn.testset import load_testset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--M_bytes", type=int, default=131072)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()

    log = RuntimeLog("fa2", "testset,T,d,M_bytes,runtime,is_correct", key_len=4)
    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    for i in range(args.num_testsets):
        if (i, args.T, args.d, args.M_bytes) in log.done:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = load_testset(data_dir, args.T, args.d)

        t0 = time.perf_counter()
        O = fa2_forward(Q, Kt, V, args.T, args.d, args.M_bytes, scale)
        runtime = time.perf_counter() - t0
        print_runtime(i, runtime)

        O_ref = np.load(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"))
        is_correct = np.allclose(O, O_ref, atol=1e-3, rtol=1e-3)
        log.append(i, args.T, args.d, args.M_bytes, f"{runtime:.6f}", is_correct)


if __name__ == "__main__":
    main()
