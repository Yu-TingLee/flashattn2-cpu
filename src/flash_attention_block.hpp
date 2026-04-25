#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include "flash_attn/types.hpp"

namespace flash_attn {

// Internal linkage on purpose: this header is compiled both with and without
// OpenMP, and the serial kernels must not pick up an OpenMP-built copy.
namespace {

struct Tiling {
    int I;   // query rows per block
    int J;   // key/value rows per block
    int Tr;  // number of query blocks
    int Tc;  // number of key/value blocks
};

Tiling make_tiling(int T, int d, int M_bytes) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;
    return {I, J, (T + I - 1) / I, (T + J - 1) / J};
}

// Runs the FA-2 inner loop for query block i and writes its rows of O.
template <bool Causal>
void process_query_block(int i, const Tiling& t, int T,
                         const MatrixXfRM& Q, const MatrixXfRM& Kt,
                         const MatrixXfRM& V, float scale, MatrixXfRM& O) {
    const int row_start = i * t.I;
    const int cur_I     = std::min(row_start + t.I, T) - row_start;

    Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
    Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
    MatrixXfRM Q_i = Q.middleRows(row_start, cur_I);

    // Under a causal mask, key blocks that start past the last query row are skipped.
    const int j_end = Causal ? std::min(t.Tc, ((i + 1) * t.I + t.J - 1) / t.J) : t.Tc;

    for (int j = 0; j < j_end; ++j) {
        const int col_start = j * t.J;
        const int cur_J     = std::min(col_start + t.J, T) - col_start;

        MatrixXfRM Kt_j = Kt.middleCols(col_start, cur_J);
        MatrixXfRM V_j  = V.middleRows(col_start, cur_J);

        MatrixXfRM S = Q_i * Kt_j;
        S *= scale;

        if constexpr (Causal) {
            const int diag_offset = row_start - col_start;
            if (diag_offset < cur_J) {
                for (int r = 0; r < cur_I; ++r) {
                    const int first_masked = std::max(0, r + diag_offset + 1);
                    if (first_masked < cur_J)
                        S.row(r).tail(cur_J - first_masked)
                            .fill(-std::numeric_limits<float>::infinity());
                }
            }
        }

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

}  // namespace
}  // namespace flash_attn
