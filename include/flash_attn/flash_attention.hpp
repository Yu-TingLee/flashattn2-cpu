#pragma once
#include "flash_attn/types.hpp"

namespace flash_attn {

// FlashAttention-2 forward pass. Q is T x d, Kt is K transposed (d x T),
// V is T x d. M_bytes is the on-chip memory budget that sets the block sizes.

MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                       const MatrixXfRM& V, int T, int d, int M_bytes, float scale);

// Query blocks are distributed over OpenMP threads.
MatrixXfRM fa2_forward_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                          const MatrixXfRM& V, int T, int d, int M_bytes, float scale);

// Wall time per step of the inner loop, in seconds.
struct ProfileData {
    double runtime        = 0.0;
    double load_kv        = 0.0;
    double compute_s      = 0.0;
    double online_softmax = 0.0;
    double update_o       = 0.0;
    double others         = 0.0;
};

struct ProfileResult {
    MatrixXfRM  O;
    ProfileData profile;
};

ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                  const MatrixXfRM& V, int T, int d,
                                  int M_bytes, float scale);

}  // namespace flash_attn
