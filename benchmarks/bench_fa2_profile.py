import argparse
import os

import numpy as np

from benchmarks.bench_utils import RuntimeLog
from flash_attn.flash_attention_profile import fa2_forward_profile
from flash_attn.testset import load_testset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--M_bytes", type=int, default=131072)
    parser.add_argument("--num_testsets", type=int, default=1)
    args = parser.parse_args()

    # line7..line11 refer to the steps of Algorithm 1 in the report.
    log = RuntimeLog(
        "fa2_profile",
        "testset,T,d,M_bytes,runtime,line7,line8,line10,line11,others",
        key_len=4,
    )
    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    for i in range(args.num_testsets):
        if (i, args.T, args.d, args.M_bytes) in log.done:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = load_testset(data_dir, args.T, args.d)

        _, *times = fa2_forward_profile(Q, Kt, V, args.T, args.d, args.M_bytes, scale)
        runtime, load_kv, compute_s, online_softmax, update_o, others = times

        log.append(i, args.T, args.d, args.M_bytes, *(f"{t:.6f}" for t in times))

        print("-" * 30)
        print(
            f"Runtime {i}: {runtime:.6f} s\n"
            f"line7 {load_kv:.6f}s  line8 {compute_s:.6f}s  "
            f"line10 {online_softmax:.6f}s  line11 {update_o:.6f}s  others {others:.6f}s"
        )
        print("-" * 30)


if __name__ == "__main__":
    main()
