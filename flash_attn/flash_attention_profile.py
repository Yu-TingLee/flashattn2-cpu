import time

import numpy as np

from flash_attn.flash_attention import block_sizes


def fa2_forward_profile(Q, Kt, V, T, d, M_bytes, scale):
    """Same computation as fa2_forward, with each step of the inner loop timed.

    Returns O, the total runtime, and the time spent loading K/V tiles,
    computing S, updating the softmax statistics, updating O, and elsewhere.
    """
    I, J = block_sizes(d, M_bytes)
    Tr = (T + I - 1) // I
    Tc = (T + J - 1) // J

    O    = np.zeros((T, d), dtype=np.float32)
    perf = time.perf_counter
    load_kv = compute_s = online_softmax = update_o = 0.0

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

            tt       = perf()
            Kt_j     = np.asfortranarray(Kt[:, col_start:col_end])
            V_j      = np.ascontiguousarray(V[col_start:col_end, :])
            load_kv += perf() - tt

            tt         = perf()
            S          = (Q_i @ Kt_j) * scale
            compute_s += perf() - tt

            tt              = perf()
            m_block         = S.max(axis=1)
            m_new           = np.maximum(m, m_block)
            alpha           = np.exp(m - m_new)
            P               = np.exp(S - m_new[:, None])
            l_new           = alpha * l + P.sum(axis=1)
            online_softmax += perf() - tt

            tt        = perf()
            O[row_start:row_end, :] = (O[row_start:row_end, :] * alpha[:, None]) + (P @ V_j)
            update_o += perf() - tt

            m, l = m_new, l_new

        O[row_start:row_end, :] /= l[:, None]

    runtime = perf() - t0
    others  = max(0.0, runtime - (load_kv + compute_s + online_softmax + update_o))
    return O, runtime, load_kv, compute_s, online_softmax, update_o, others
