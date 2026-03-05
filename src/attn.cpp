#include "flash_attn/attn.hpp"
#include <algorithm>
#include <cmath>

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

MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                        const MatrixXfRM& V, int T, int d, int M_bytes, float scale) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;

    const int Tr = (T + I - 1) / I;
    const int Tc = (T + J - 1) / J;

    MatrixXfRM O = MatrixXfRM::Zero(T, d);

    for (int i = 0; i < Tr; ++i) {
        const int row_start = i * I;
        const int cur_I     = std::min(row_start + I, T) - row_start;

        Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
        Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
        MatrixXfRM Q_i = Q.middleRows(row_start, cur_I);

        for (int j = 0; j < Tc; ++j) {
            const int col_start = j * J;
            const int cur_J     = std::min(col_start + J, T) - col_start;

            MatrixXfRM Kt_j = Kt.middleCols(col_start, cur_J);
            MatrixXfRM V_j  = V.middleRows(col_start, cur_J);

            MatrixXfRM S = Q_i * Kt_j;
            S *= scale;

            Eigen::VectorXf m_new = m.cwiseMax(S.rowwise().maxCoeff());
            Eigen::VectorXf alpha = (m - m_new).array().exp();
            Eigen::MatrixXf P     = (S.colwise() - m_new).array().exp().matrix();
            Eigen::VectorXf l_new = alpha.cwiseProduct(l) + P.rowwise().sum();

            auto O_block = O.middleRows(row_start, cur_I);
            O_block.array().colwise() *= alpha.array();
            O.middleRows(row_start, cur_I) += P * V_j;

            m = m_new;
            l = l_new;
        }
        O.middleRows(row_start, cur_I).array().colwise() /= l.array();
    }
    return O;
}
