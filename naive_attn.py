import numpy as np
import os
import argparse
import time
from utils import read_QKV

def naive_attention(Q, Kt, V, scale):
    # 1. Compute S = scale*(QK^T)
    # 2. Apply safe softmax row-wise on S
    # 3. Compute O = P*V
    S = Q @ Kt
    S *= scale
    S_max = np.max(S, axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = np.sum(S_exp, axis=1, keepdims=True)
    P = S_exp / S_sum
    O = P @ V

    return O

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default = 1024)
    parser.add_argument("--d", type=int, default = 64)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()
    
    output_dir = os.path.join("outputs", "naive_attn")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "runtime.csv")
    
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("testset,T,d,runtime\n")

    completed_runs = set()
    with open(log_path, "r") as f:
        header = next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                key = (int(parts[0]), int(parts[1]), int(parts[2]))
                completed_runs.add(key)
    
    for i in range(args.num_testsets):
        current_key = (i, args.T, args.d)
        if current_key in completed_runs:
            print(f"Skipping testset{i}.")
            continue
        
        data_dir = os.path.join("data",f"testset{i}")
        
        Q, Kt, V = read_QKV(data_dir, args.T, args.d)
        scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))
        
        start_time_i = time.perf_counter()
        O_naive = naive_attention(Q, Kt, V, scale)
        end_time_i = time.perf_counter()
        
        runtime = end_time_i - start_time_i
        print("-" * 30)
        print(f"Runtime {i}: {runtime:.4f} seconds.")
        print("-" * 30)
        with open(log_path, "a") as f:
            f.write(f"{i},{args.T},{args.d},{runtime:.6f}\n")
        np.save(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"), O_naive)