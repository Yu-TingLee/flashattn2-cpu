import numpy as np
import time


def fa2_forward_profile(Q, Kt, V, T, d, M_bytes, scale):
    I = M_bytes // (4 * d * 4)
    J = min(I, d)
    if I < 1: I = 1
    if J < 1: J = 1

    Tr = (T + I - 1) // I
    Tc = (T + J - 1) // J

    O    = np.zeros((T, d), dtype=np.float32)
    perf = time.perf_counter
    t7 = t8 = t10 = t11 = 0.0

    t0 = perf()

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

            # Line 7: tile loads
            tt   = perf()
            Kt_j = np.asfortranarray(Kt[:, col_start:col_end])
            V_j  = np.ascontiguousarray(V[col_start:col_end, :])
            t7  += perf() - tt

            # Line 8: S = Q_i @ Kt_j
            tt   = perf()
            S    = (Q_i @ Kt_j) * scale
            t8  += perf() - tt

            # Line 10: softmax statistics
            tt     = perf()
            m_block = S.max(axis=1)
            m_new   = np.maximum(m, m_block)
            alpha   = np.exp(m - m_new)
            P       = np.exp(S - m_new[:, None])
            l_new   = alpha * l + P.sum(axis=1)
            t10    += perf() - tt

            # Line 11: output accumulation
            tt  = perf()
            O[row_start:row_end, :] = (O[row_start:row_end, :] * alpha[:, None]) + (P @ V_j)
            t11 += perf() - tt

            m, l = m_new, l_new

        O[row_start:row_end, :] /= l[:, None]

    runtime = perf() - t0
    others  = max(0.0, runtime - (t7 + t8 + t10 + t11))
    return O, runtime, t7, t8, t10, t11, others
