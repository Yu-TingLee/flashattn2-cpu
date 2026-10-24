#pragma once
#include "flash_attn/types.hpp"

namespace flash_attn {

// Standard attention that materializes the full T x T score matrix.
// Q is T x d, Kt is K transposed (d x T), V is T x d.

MatrixXfRM naive_attention(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                           float scale);

MatrixXfRM naive_attention_causal(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                                  float scale);

}  // namespace flash_attn
