#pragma once
#include "flash_attn/types.hpp"

namespace flash_attn {

// FlashAttention-2 forward pass. Q is T x d, Kt is K transposed (d x T),
// V is T x d. M_bytes is the on-chip memory budget that sets the block sizes.

MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                       const MatrixXfRM& V, int T, int d, int M_bytes, float scale);

}  // namespace flash_attn
