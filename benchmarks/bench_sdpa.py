import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.bench_utils import RuntimeLog, print_runtime
from flash_attn.testset import load_testset


def load_tensors(data_dir, T, d):
    """Q, K, V as (1, 1, T, d) contiguous tensors: batch 1, one head."""
    Q, Kt, V = load_testset(data_dir, T, d)
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).view(1, 1, T, d)
                 for x in (Q, Kt.T, V))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--num_testsets", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    prefix = "sdpa_causal" if args.causal else "sdpa"
    log = RuntimeLog(f"{prefix}_t{args.num_threads}",
                     "testset,T,d,runtime,is_correct", key_len=3)
    ref_name = f"O_{'causal_' if args.causal else ''}T{args.T}_d{args.d}.npy"
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, causal={args.causal}")

    # Fail rather than fall back to the math backend.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.inference_mode():
        # Warm up before the timed runs.
        Q, K, V = load_tensors(os.path.join("data", "testset0"), args.T, args.d)
        F.scaled_dot_product_attention(Q, K, V, is_causal=args.causal)

        for i in range(args.num_testsets):
            if (i, args.T, args.d) in log.done:
                print(f"Skipping testset{i}.")
                continue

            data_dir = os.path.join("data", f"testset{i}")
            Q, K, V = load_tensors(data_dir, args.T, args.d)

            t0 = time.perf_counter()
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=args.causal)
            runtime = time.perf_counter() - t0
            print_runtime(i, runtime)

            O_ref = np.load(os.path.join(data_dir, ref_name))
            is_correct = np.allclose(O.view(args.T, args.d).numpy(), O_ref,
                                     atol=1e-3, rtol=1e-3)
            log.append(i, args.T, args.d, f"{runtime:.6f}", is_correct)


if __name__ == "__main__":
    main()
