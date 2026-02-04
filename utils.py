import os
import numpy as np

def read_QKV(data_dir, T, d):
    total_elements = T * d
    for fname in ["Q", "Kt", "V"]:
        path = os.path.join(data_dir, f"{fname}.txt")
        data = np.array(np.loadtxt(path, max_rows=total_elements), dtype=np.float32)
        if fname == "Q":
            Q = data.reshape(T, d)
        elif fname == "Kt":
            Kt = data.reshape(d, T)
        elif fname == "V":
            V = data.reshape(T, d)
    return Q, Kt, V