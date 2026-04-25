#include "flash_attn/attn.hpp"
#include "flash_attn/wsq.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <memory>
#include <thread>
#include <omp.h>

// process one query block i
static void process_block(int i, int I, int J, int Tc, int T,
                           const MatrixXfRM& Q, const MatrixXfRM& Kt,
                           const MatrixXfRM& V, float scale, MatrixXfRM& O) {
    const int row_start = i * I;
    const int cur_I     = std::min(row_start + I, T) - row_start;

    Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
    Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
    MatrixXfRM Q_i = Q.middleRows(row_start, cur_I);

    const int j_max = std::min(Tc, (int)std::ceil((float)((i + 1) * I) / J));

    for (int j = 0; j < j_max; ++j) {
        const int col_start = j * J;
        const int cur_J     = std::min(col_start + J, T) - col_start;

        MatrixXfRM Kt_j = Kt.middleCols(col_start, cur_J);
        MatrixXfRM V_j  = V.middleRows(col_start, cur_J);

        MatrixXfRM S = Q_i * Kt_j;
        S *= scale;

        // Apply causal mask within any block that crosses the diagonal.
        const int diag_offset = row_start - col_start;
        if (diag_offset < cur_J) {
            for (int r = 0; r < cur_I; ++r) {
                const int first_masked = std::max(0, r + diag_offset + 1);
                if (first_masked < cur_J)
                    S.row(r).tail(cur_J - first_masked)
                        .fill(-std::numeric_limits<float>::infinity());
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

// Strategy 1: static schedule
static void run_static(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                        int T, int d, int M_bytes, float scale, int num_threads,
                        MatrixXfRM& O, std::vector<ThreadMetrics>& metrics) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;
    const int Tr = (T + I - 1) / I;
    const int Tc = (T + J - 1) / J;

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const auto t0 = std::chrono::steady_clock::now();

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < Tr; ++i) {
            process_block(i, I, J, Tc, T, Q, Kt, V, scale, O);
            metrics[tid].blocks_processed++;
        }

        metrics[tid].wall_time_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
}

// Strategy 2: dynamic schedule
static void run_dynamic(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                         int T, int d, int M_bytes, float scale, int num_threads,
                         MatrixXfRM& O, std::vector<ThreadMetrics>& metrics) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;
    const int Tr = (T + I - 1) / I;
    const int Tc = (T + J - 1) / J;

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const auto t0 = std::chrono::steady_clock::now();

        #pragma omp for schedule(dynamic) nowait
        for (int i = 0; i < Tr; ++i) {
            process_block(i, I, J, Tc, T, Q, Kt, V, scale, O);
            metrics[tid].blocks_processed++;
        }

        metrics[tid].wall_time_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
}

// Strategy 3: work-stealing
static void run_work_stealing(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                               int T, int d, int M_bytes, float scale, int num_threads,
                               MatrixXfRM& O, std::vector<ThreadMetrics>& metrics) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;
    const int Tr = (T + I - 1) / I;
    const int Tc = (T + J - 1) / J;

    // Initial capacity: smallest power of 2 >= Tr
    int64_t cap = 1;
    while (cap < Tr) cap <<= 1;

    std::vector<std::unique_ptr<WorkStealingQueue<int>>> queues(num_threads);
    for (int t = 0; t < num_threads; ++t)
        queues[t] = std::make_unique<WorkStealingQueue<int>>(cap);

    // Round-robin
    // for (int i = 0; i < Tr; ++i)
    //     queues[i % num_threads]->push(i);

    const int base_blocks = Tr / num_threads;
    const int rem_blocks  = Tr % num_threads;
    for (int t = 0; t < num_threads; ++t) {
        const int start = t * base_blocks + std::min(t, rem_blocks);
        const int count = base_blocks + (t < rem_blocks ? 1 : 0);
        for (int i = start; i < start + count; ++i) {
            queues[t]->push(i);
        }
    }

    std::vector<std::thread> threads(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads[t] = std::thread([&, t]() {
            ThreadMetrics& m = metrics[t];
            const auto t0 = std::chrono::steady_clock::now();

            while (true) {
                auto item = queues[t]->pop();
                if (!item) {
                    bool found = false;
                    for (int k = 1; k < num_threads; ++k) {
                        const int victim = (t + k) % num_threads;
                        m.steal_attempts++;
                        item = queues[victim]->steal();
                        if (item) {
                            m.steal_successes++;
                            found = true;
                            break;
                        }
                    }
                    if (!found) break;
                }
                process_block(*item, I, J, Tc, T, Q, Kt, V, scale, O);
                m.blocks_processed++;
            }

            m.wall_time_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        });
    }
    for (auto& th : threads) th.join();
}

// Public entry point
MatrixXfRM fa2_forward_causal_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                  const MatrixXfRM& V, int T, int d, int M_bytes,
                                  float scale, Schedule sched, int num_threads,
                                  std::vector<ThreadMetrics>& metrics_out) {
    metrics_out.assign(num_threads, ThreadMetrics{});
    MatrixXfRM O = MatrixXfRM::Zero(T, d);

    switch (sched) {
        case Schedule::STATIC:
            run_static(Q, Kt, V, T, d, M_bytes, scale, num_threads, O, metrics_out);
            break;
        case Schedule::DYNAMIC:
            run_dynamic(Q, Kt, V, T, d, M_bytes, scale, num_threads, O, metrics_out);
            break;
        case Schedule::WORK_STEALING:
            run_work_stealing(Q, Kt, V, T, d, M_bytes, scale, num_threads, O, metrics_out);
            break;
    }
    return O;
}
