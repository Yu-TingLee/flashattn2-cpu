import os

import numpy as np


def load_testset(data_dir, T, d):
    """Load the first T x d entries of Q, K^T and V from data/testset<i>/."""
    def load(name, shape):
        path = os.path.join(data_dir, f"{name}.txt")
        return np.array(np.loadtxt(path, max_rows=T * d), dtype=np.float32).reshape(shape)

    return load("Q", (T, d)), load("Kt", (d, T)), load("V", (T, d))
