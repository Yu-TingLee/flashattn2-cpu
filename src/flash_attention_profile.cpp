#include "flash_attn/flash_attention.hpp"
#include <algorithm>
#include <chrono>
#include "flash_attention_block.hpp"

namespace flash_attn {

// Same computation as fa2_forward, with each step of the inner loop timed.
ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                  const MatrixXfRM& V, int T, int d,
                                  int M_bytes, float scale) {
    using Clock = std::chrono::steady_clock;
    const auto since = [](Clock::time_point t0) {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    };

    const Tiling t = make_tiling(T, d, M_bytes);
    MatrixXfRM O = MatrixXfRM::Zero(T, d);
    ProfileData p;

    const auto t0 = Clock::now();

    for (int i = 0; i < t.Tr; ++i) {
        const int row_start = i * t.I;
        const int cur_I     = std::min(row_start + t.I, T) - row_start;

        Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
        Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
        MatrixXfRM Q_i = Q.middleRows(row_start, cur_I);

        for (int j = 0; j < t.Tc; ++j) {
            const int col_start = j * t.J;
            const int cur_J     = std::min(col_start + t.J, T) - col_start;

            auto tt = Clock::now();
            MatrixXfRM Kt_j = Kt.middleCols(col_start, cur_J);
            MatrixXfRM V_j  = V.middleRows(col_start, cur_J);
            p.load_kv += since(tt);

            tt = Clock::now();
            MatrixXfRM S = Q_i * Kt_j;
            S *= scale;
            p.compute_s += since(tt);

            tt = Clock::now();
            Eigen::VectorXf m_new = m.cwiseMax(S.rowwise().maxCoeff());
            Eigen::VectorXf alpha = (m - m_new).array().exp();
            Eigen::MatrixXf P     = (S.colwise() - m_new).array().exp().matrix();
            Eigen::VectorXf l_new = alpha.cwiseProduct(l) + P.rowwise().sum();
            p.online_softmax += since(tt);

            tt = Clock::now();
            auto O_block = O.middleRows(row_start, cur_I);
            O_block.array().colwise() *= alpha.array();
            O.middleRows(row_start, cur_I) += P * V_j;
            p.update_o += since(tt);

            m = m_new;
            l = l_new;
        }
        O.middleRows(row_start, cur_I).array().colwise() /= l.array();
    }

    p.runtime = since(t0);
    const double timed = p.load_kv + p.compute_s + p.online_softmax + p.update_o;
    p.others  = std::max(0.0, p.runtime - timed);
    return {std::move(O), p};
}

}  // namespace flash_attn
