#pragma once
#include "types.hpp"

// Naive scaled dot-product attention
MatrixXfRM naive_attention(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                            const MatrixXfRM& V, float scale);

// FlashAttention-2 forward pass
MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                        const MatrixXfRM& V, int T, int d, int M_bytes, float scale);

// Per-line profiling variant — same math as fa2_forward, returns timing breakdown
struct ProfileData {
    double runtime, t7, t8, t10, t11, others;
};

struct ProfileResult {
    MatrixXfRM  O;
    ProfileData profile;
};

ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                   const MatrixXfRM& V, int T, int d,
                                   int M_bytes, float scale);
