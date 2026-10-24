import argparse
import os
import time

import numpy as np

from benchmarks.bench_utils import RuntimeLog, print_runtime
from flash_attn.testset import load_testset


def load_contiguous(data_dir, T, d):
    Q, Kt, V = load_testset(data_dir, T, d)
    return np.ascontiguousarray(Q), np.asfortranarray(Kt), np.ascontiguousarray(V)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--M_bytes", type=int, default=131072)
    parser.add_argument("--num_testsets", type=int, default=100)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count())),
    )
    args = parser.parse_args()

    # Numba reads its thread count at import time.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = str(args.num_threads)
    import numba

    numba.set_num_threads(args.num_threads)
    from flash_attn.flash_attention_numba import fa2_forward

    log = RuntimeLog(
        f"fa2_numba_t{args.num_threads}",
        "testset,T,d,M_bytes,runtime,is_correct",
        key_len=4,
    )
    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))

    # Trigger JIT compilation before the timed runs.
    Q, Kt, V = load_contiguous(os.path.join("data", "testset0"), args.T, args.d)
    fa2_forward(Q, Kt, V, args.T, args.d, args.M_bytes, scale)

    for i in range(args.num_testsets):
        if (i, args.T, args.d, args.M_bytes) in log.done:
            print(f"Skipping testset{i}.")
            continue

        data_dir = os.path.join("data", f"testset{i}")
        Q, Kt, V = load_contiguous(data_dir, args.T, args.d)

        t0 = time.perf_counter()
        O = fa2_forward(Q, Kt, V, args.T, args.d, args.M_bytes, scale)
        runtime = time.perf_counter() - t0
        print_runtime(i, runtime)

        O_ref = np.load(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"))
        is_correct = np.allclose(O, O_ref, atol=1e-3, rtol=1e-3)
        log.append(i, args.T, args.d, args.M_bytes, f"{runtime:.6f}", is_correct)


if __name__ == "__main__":
    main()
