import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def fa2_forward(Q, Kt, V, T, d, M_bytes, scale):
    I = M_bytes // (4 * d * 4)
    J = min(I, d)
    if I < 1: I = 1
    if J < 1: J = 1

    Tr, Tc = (T + I - 1) // I, (T + J - 1) // J
    O = np.zeros((T, d), dtype=np.float32)

    for i in prange(Tr):
        row_start = i * I
        row_end   = min(row_start + I, T)
        cur_I     = row_end - row_start

        l       = np.zeros(cur_I, dtype=np.float32)
        m       = np.full(cur_I, -np.inf, dtype=np.float32)
        m_block = np.empty(cur_I, dtype=np.float32)
        Q_i     = np.ascontiguousarray(Q[row_start:row_end, :])

        for j in range(Tc):
            col_start = j * J
            col_end   = min(col_start + J, T)
            cur_J     = col_end - col_start

            Kt_j = np.asfortranarray(Kt[:, col_start:col_end])
            V_j  = np.ascontiguousarray(V[col_start:col_end, :])

            S  = Q_i @ Kt_j
            S *= scale

            for r in range(cur_I):
                row_max = -np.inf
                for c in range(cur_J):
                    val = S[r, c]
                    if val > row_max:
                        row_max = val
                m_block[r] = row_max
            m_new = np.maximum(m, m_block)

            alpha = np.exp(m - m_new).reshape((cur_I, 1))
            P     = np.exp(S - m_new.reshape((cur_I, 1)))
            l_new = (alpha.ravel() * l) + np.sum(P, axis=1)

            O[row_start:row_end, :] = (O[row_start:row_end, :] * alpha) + (P @ V_j)
            m, l = m_new, l_new

        O[row_start:row_end, :] /= l.reshape((cur_I, 1))

    return O
