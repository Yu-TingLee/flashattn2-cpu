import argparse
import os

import numpy as np


def generate_data(T, d, num_testsets):
    """Write T*d uniform(-1, 1) values for each of Q, K^T and V per test set.

    Benchmarks with a smaller T or d read a prefix of these files.
    """
    for i in range(num_testsets):
        rng = np.random.RandomState(i)
        out_dir = os.path.join("data", f"testset{i}")
        os.makedirs(out_dir, exist_ok=True)
        for name in ["Q", "Kt", "V"]:
            data = rng.uniform(-1, 1, size=T * d)
            np.savetxt(os.path.join(out_dir, f"{name}.txt"), data)
        print(f"Saved testset{i}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()

    generate_data(args.T, args.d, args.num_testsets)
