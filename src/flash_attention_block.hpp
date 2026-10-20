#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include "flash_attn/flash_attention.hpp"

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

// Empty unless Profile is set, so the unprofiled kernels carry no timers.
template <bool Profile>
struct StepTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void lap(double* acc) {
        const auto now = std::chrono::steady_clock::now();
        *acc += std::chrono::duration<double>(now - t0).count();
        t0 = now;
    }
};

template <>
struct StepTimer<false> {
    void lap(double*) {}
};

template <bool Causal, bool Profile = false>
void process_query_block(int i, const Tiling& t, int T,
                         const MatrixXfRM& Q, const MatrixXfRM& Kt,
                         const MatrixXfRM& V, float scale, MatrixXfRM& O,
                         ProfileData* prof = nullptr) {
    const int row_start = i * t.I;
    const int cur_I     = std::min(row_start + t.I, T) - row_start;

    Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
    Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
    Eigen::VectorXf m_new(cur_I), alpha(cur_I);
    MatrixXfRM S(cur_I, t.J), P(cur_I, t.J);
    const auto Q_i = Q.middleRows(row_start, cur_I);
    auto O_i       = O.middleRows(row_start, cur_I);

    // Under a causal mask, key blocks that start past the last query row are skipped.
    const int j_end = Causal ? std::min(t.Tc, ((i + 1) * t.I + t.J - 1) / t.J) : t.Tc;

    for (int j = 0; j < j_end; ++j) {
        StepTimer<Profile> timer;
        const int col_start = j * t.J;
        const int cur_J     = std::min(col_start + t.J, T) - col_start;

        // Views, not copies: Eigen packs the operands inside the product anyway.
        const auto Kt_j = Kt.middleCols(col_start, cur_J);
        const auto V_j  = V.middleRows(col_start, cur_J);

        S.resize(cur_I, cur_J);
        P.resize(cur_I, cur_J);
        // K and V are read through views, so their load is timed inside the products.
        timer.lap(Profile ? &prof->load_kv : nullptr);

        S.noalias() = Q_i * Kt_j;
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

        timer.lap(Profile ? &prof->compute_s : nullptr);

        m_new = m.cwiseMax(S.rowwise().maxCoeff());
        alpha = (m - m_new).array().exp();
        P     = (S.colwise() - m_new).array().exp().matrix();
        l     = alpha.cwiseProduct(l) + P.rowwise().sum();
        timer.lap(Profile ? &prof->online_softmax : nullptr);

        O_i.array().colwise() *= alpha.array();
        O_i.noalias() += P * V_j;
        timer.lap(Profile ? &prof->update_o : nullptr);

        m = m_new;
    }
    O_i.array().colwise() /= l.array();
}

}  // namespace
}  // namespace flash_attn
