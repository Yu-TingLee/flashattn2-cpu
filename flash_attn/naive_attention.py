import numpy as np


def naive_attention(Q, Kt, V, scale):
    S = Q @ Kt
    S *= scale
    S_max = np.max(S, axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = np.sum(S_exp, axis=1, keepdims=True)
    P = S_exp / S_sum
    return P @ V


def naive_attention_causal(Q, Kt, V, scale):
    S = Q @ Kt
    S *= scale
    T = S.shape[0]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    S[mask] = -np.inf
    S_max = np.max(S, axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_sum = np.sum(S_exp, axis=1, keepdims=True)
    P = S_exp / S_sum
    return P @ V
