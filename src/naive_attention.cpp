#include "flash_attn/naive_attention.hpp"
#include <limits>

namespace flash_attn {

MatrixXfRM naive_attention(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                           const MatrixXfRM& V, float scale) {
    MatrixXfRM S = Q * Kt;
    S *= scale;
    Eigen::VectorXf row_max = S.rowwise().maxCoeff();
    S.colwise() -= row_max;
    S = S.array().exp();
    Eigen::VectorXf row_sum = S.rowwise().sum();
    S.array().colwise() /= row_sum.array();
    return S * V;
}

MatrixXfRM naive_attention_causal(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                  const MatrixXfRM& V, float scale) {
    MatrixXfRM S = Q * Kt;
    S *= scale;
    const int T = S.rows();
    for (int r = 0; r < T; ++r)
        for (int c = r + 1; c < T; ++c)
            S(r, c) = -std::numeric_limits<float>::infinity();
    Eigen::VectorXf row_max = S.rowwise().maxCoeff();
    S.colwise() -= row_max;
    S = S.array().exp();
    Eigen::VectorXf row_sum = S.rowwise().sum();
    S.array().colwise() /= row_sum.array();
    return S * V;
}

}  // namespace flash_attn
