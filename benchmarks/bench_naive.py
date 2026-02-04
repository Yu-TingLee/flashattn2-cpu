import argparse
import os
import time

import numpy as np

from benchmarks.bench_utils import RuntimeLog, print_runtime
from flash_attn.naive_attention import naive_attention
from flash_attn.testset import load_testset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()

    log = RuntimeLog("naive_attn", "testset,T,d,runtime", key_len=3)
    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    for i in range(args.num_testsets):
        if (i, args.T, args.d) in log.done:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = load_testset(data_dir, args.T, args.d)

        t0 = time.perf_counter()
        O = naive_attention(Q, Kt, V, scale)
        runtime = time.perf_counter() - t0
        print_runtime(i, runtime)

        log.append(i, args.T, args.d, f"{runtime:.6f}")
        # Reference output for every other benchmark.
        np.save(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"), O)


if __name__ == "__main__":
    main()
