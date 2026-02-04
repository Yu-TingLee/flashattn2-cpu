import os
import numpy as np
import argparse

def generate_data(T, d, num_testsets):  
    total_elements = T * d
    for i in range(num_testsets):
        rng = np.random.RandomState(i)
        out_dir = os.path.join("data", f"testset{i}")
        os.makedirs(out_dir, exist_ok=True)
        for fname in ["Q","Kt","V"]:
            data = rng.uniform(-1, 1, size=total_elements)
            np.savetxt(os.path.join(out_dir, f"{fname}.txt"), data)
        print(f"Saved testset{i}.")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()
    
    generate_data(args.T, args.d, args.num_testsets)