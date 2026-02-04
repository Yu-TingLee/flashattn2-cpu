import numpy as np
from numba import jit, prange
import os
import time
import argparse
from utils import read_QKV

@jit(nopython=True, parallel=True, fastmath=True)
def fa2_forward(Q, Kt, V, T, d, M_bytes, scale):
    # Tr = number of row blocks, Tc = number of column blocks
    I = M_bytes // (4 * d * 4)
    J = min(I, d)
    
    if I < 1: I = 1
    if J < 1: J = 1
    
    Tr, Tc = (T + I - 1) // I, (T + J - 1) // J
    O = np.zeros((T, d), dtype=np.float32)
    
    # prange for parallelism
    for i in prange(Tr):
        row_start = i * I
        row_end = min(row_start + I, T)
        cur_I = row_end - row_start
        
        # Initialize l and m
        l = np.zeros(cur_I, dtype=np.float32)
        m = np.full(cur_I, -np.inf, dtype=np.float32)
        m_block = np.empty(cur_I, dtype=np.float32)
        # Load block Q_{I_i,:}
        Q_i = np.ascontiguousarray(Q[row_start:row_end, :])
        
        for j in range(Tc):
            col_start = j * J
            col_end = min(col_start + J, T)
            cur_J = col_end - col_start
            
            # Load block K_{J_j, :}^\top and V_{J_j, :}
            Kt_j = np.asfortranarray(Kt[:, col_start:col_end])
            V_j  = np.ascontiguousarray(V[col_start:col_end, :])
            
            # Compute S \gets Q_{I_i,:} K_{J_j,:}^\top
            S = Q_i @ Kt_j
            S *= scale
            
            # Compute m_new
            for r in range(cur_I):
                row_max = -np.inf
                for c in range(cur_J):
                    val = S[r, c]
                    if val > row_max:
                        row_max = val
                m_block[r] = row_max
            m_new = np.maximum(m, m_block)
            
            # Compute alpha and l_new
            alpha = np.exp(m - m_new).reshape((cur_I, 1))
            P = np.exp(S - m_new.reshape((cur_I, 1)))
            l_new = (alpha.ravel() * l) + np.sum(P, axis=1)
            
            # Update O_{i_I, :} \gets (alpha \cdot O_{i_I, :}) + (P * V_{j_J, :})
            O[row_start:row_end, :] = (O[row_start:row_end, :] * alpha) + (P @ V_j)
            m, l = m_new, l_new
            
        # Final update O_{i_I, :} \gets (1\oslash l) * O_{i_I, :}
        O[row_start:row_end, :] /= l.reshape((cur_I, 1))
        
    return O

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default = 8192)
    parser.add_argument("--d", type=int, default = 64)
    parser.add_argument("--M_bytes", type=int, default = 131072)
    parser.add_argument("--num_testsets", type=int, default=100)
    args = parser.parse_args()
    
    output_dir = os.path.join("outputs", "fa2_jit")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "runtime.csv")
    
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("testset,T,d,M_bytes,runtime,is_correct\n")
            
    completed_runs = set()
    with open(log_path, "r") as f:
        header = next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                key = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
                completed_runs.add(key)
            
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # Catch OpenBLAS
    
    scale = np.float32(1.0 / np.sqrt(np.float32(args.d)))
    
    # Warm up
    Q0, Kt0, V0 = read_QKV(os.path.join("data",f"testset0"), args.T, args.d)
    Q0 = np.ascontiguousarray(Q0)
    Kt0 = np.asfortranarray(Kt0)
    V0 = np.ascontiguousarray(V0)
    _ = fa2_forward(Q0, Kt0, V0, args.T, args.d, args.M_bytes, scale)
    
    for i in range(args.num_testsets):
        current_key = (i, args.T, args.d, args.M_bytes)
        if current_key in completed_runs:
            print(f"Skipping testset{i}.")
            continue
        
        data_dir=os.path.join("data",f"testset{i}")
        
        Q, Kt, V = read_QKV(data_dir, args.T, args.d)
        Q = np.ascontiguousarray(Q)
        Kt = np.asfortranarray(Kt)
        V = np.ascontiguousarray(V)
        
        start_time_i = time.perf_counter()
        O_jit = fa2_forward(Q, Kt, V, args.T, args.d, args.M_bytes, scale)
        end_time_i = time.perf_counter()
        
        runtime = end_time_i - start_time_i
        print("-" * 30)
        print(f"Runtime {i}: {runtime:.4f} seconds.")
        print("-" * 30)
        
        O_ref = np.load(os.path.join(data_dir, f"O_T{args.T}_d{args.d}.npy"))
        is_correct = np.allclose(O_jit, O_ref, atol=1e-3, rtol=1e-3)
        with open(os.path.join(output_dir, "runtime.csv"), "a") as f:
            f.write(f"{i},{args.T},{args.d},{args.M_bytes},{runtime:.6f},{is_correct}\n")