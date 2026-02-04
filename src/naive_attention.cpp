#include "flash_attn/naive_attention.hpp"

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

}  // namespace flash_attn
