#include "flash_attn/attn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>

ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                   const MatrixXfRM& V, int T, int d,
                                   int M_bytes, float scale) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;

    const int Tr = (T + I - 1) / I;
    const int Tc = (T + J - 1) / J;

    MatrixXfRM O = MatrixXfRM::Zero(T, d);
    ProfileData p{};

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    for (int i = 0; i < Tr; ++i) {
        const int row_start = i * I;
        const int cur_I     = std::min(row_start + I, T) - row_start;

        Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
        Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
        MatrixXfRM Q_i = Q.middleRows(row_start, cur_I);

        for (int j = 0; j < Tc; ++j) {
            const int col_start = j * J;
            const int cur_J     = std::min(col_start + J, T) - col_start;

            // Line 7: tile loads
            auto tt = Clock::now();
            MatrixXfRM Kt_j = Kt.middleCols(col_start, cur_J);
            MatrixXfRM V_j  = V.middleRows(col_start, cur_J);
            p.t7 += std::chrono::duration<double>(Clock::now() - tt).count();

            // Line 8: S = Q_i * Kt_j
            tt = Clock::now();
            MatrixXfRM S = Q_i * Kt_j;
            S *= scale;
            p.t8 += std::chrono::duration<double>(Clock::now() - tt).count();

            // Line 10: softmax statistics
            tt = Clock::now();
            Eigen::VectorXf m_new = m.cwiseMax(S.rowwise().maxCoeff());
            Eigen::VectorXf alpha = (m - m_new).array().exp();
            Eigen::MatrixXf P     = (S.colwise() - m_new).array().exp().matrix();
            Eigen::VectorXf l_new = alpha.cwiseProduct(l) + P.rowwise().sum();
            p.t10 += std::chrono::duration<double>(Clock::now() - tt).count();

            // Line 11: output accumulation
            tt = Clock::now();
            auto O_block = O.middleRows(row_start, cur_I);
            O_block.array().colwise() *= alpha.array();
            O.middleRows(row_start, cur_I) += P * V_j;
            p.t11 += std::chrono::duration<double>(Clock::now() - tt).count();

            m = m_new;
            l = l_new;
        }
        O.middleRows(row_start, cur_I).array().colwise() /= l.array();
    }

    p.runtime = std::chrono::duration<double>(Clock::now() - t0).count();
    p.others  = std::max(0.0, p.runtime - (p.t7 + p.t8 + p.t10 + p.t11));
    return {std::move(O), p};
}
