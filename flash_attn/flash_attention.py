import numpy as np


def block_sizes(d, M_bytes):
    """Query and key/value block sizes for an M_bytes on-chip memory budget."""
    I = M_bytes // (4 * d * 4)
    J = min(I, d)
    return max(I, 1), max(J, 1)


def fa2_forward(Q, Kt, V, T, d, M_bytes, scale):
    I, J = block_sizes(d, M_bytes)
    Tr = (T + I - 1) // I
    Tc = (T + J - 1) // J

    O = np.zeros((T, d), dtype=np.float32)

    for i in range(Tr):
        row_start = i * I
        row_end   = min(row_start + I, T)
        cur_I     = row_end - row_start

        l = np.zeros(cur_I, dtype=np.float32)
        m = np.full(cur_I, -np.inf, dtype=np.float32)
        Q_i = Q[row_start:row_end, :]

        for j in range(Tc):
            col_start = j * J
            col_end   = min(col_start + J, T)

            Kt_j = np.asfortranarray(Kt[:, col_start:col_end])
            V_j  = np.ascontiguousarray(V[col_start:col_end, :])

            S = (Q_i @ Kt_j) * scale

            m_block = S.max(axis=1)
            m_new   = np.maximum(m, m_block)
            alpha   = np.exp(m - m_new)
            P       = np.exp(S - m_new[:, None])
            l_new   = alpha * l + P.sum(axis=1)

            O[row_start:row_end, :] = (O[row_start:row_end, :] * alpha[:, None]) + (P @ V_j)
            m, l = m_new, l_new

        O[row_start:row_end, :] /= l[:, None]

    return O
