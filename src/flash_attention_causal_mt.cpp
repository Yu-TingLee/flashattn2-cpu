#include "flash_attn/flash_attention.hpp"
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <omp.h>
#include "flash_attention_block.hpp"
#include "work_stealing_queue.hpp"

namespace flash_attn {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

template <Schedule S>
void run_omp(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
             const Tiling& t, int T, float scale, int num_threads,
             MatrixXfRM& O, std::vector<ThreadMetrics>& metrics) {
    #pragma omp parallel num_threads(num_threads)
    {
        ThreadMetrics& tm = metrics[omp_get_thread_num()];
        const auto t0 = Clock::now();

        if constexpr (S == Schedule::Static) {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < t.Tr; ++i) {
                process_query_block<true>(i, t, T, Q, Kt, V, scale, O);
                tm.blocks_processed++;
            }
        } else {
            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < t.Tr; ++i) {
                process_query_block<true>(i, t, T, Q, Kt, V, scale, O);
                tm.blocks_processed++;
            }
        }

        tm.wall_time_sec = seconds_since(t0);
    }
}

// Each thread starts with a contiguous range of query blocks and steals from
// the others once its own queue is empty.
void run_work_stealing(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                       const Tiling& t, int T, float scale, int num_threads,
                       MatrixXfRM& O, std::vector<ThreadMetrics>& metrics) {
    int64_t capacity = 1;
    while (capacity < t.Tr) capacity <<= 1;

    std::vector<std::unique_ptr<WorkStealingQueue<int>>> queues(num_threads);
    for (auto& q : queues)
        q = std::make_unique<WorkStealingQueue<int>>(capacity);

    const int base = t.Tr / num_threads;
    const int rem  = t.Tr % num_threads;
    for (int tid = 0; tid < num_threads; ++tid) {
        const int start = tid * base + std::min(tid, rem);
        const int count = base + (tid < rem ? 1 : 0);
        for (int i = start; i < start + count; ++i)
            queues[tid]->push(i);
    }

    std::vector<std::thread> threads(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        threads[tid] = std::thread([&, tid]() {
            // Outside an OpenMP region Eigen would start its own team for every
            // product, oversubscribing the cores num_threads times over.
            omp_set_num_threads(1);
            ThreadMetrics& tm = metrics[tid];
            const auto t0 = Clock::now();

            while (true) {
                auto item = queues[tid]->pop();
                for (int k = 1; !item && k < num_threads; ++k) {
                    tm.steal_attempts++;
                    item = queues[(tid + k) % num_threads]->steal();
                    if (item) tm.steal_successes++;
                }
                if (!item) break;
                process_query_block<true>(*item, t, T, Q, Kt, V, scale, O);
                tm.blocks_processed++;
            }

            tm.wall_time_sec = seconds_since(t0);
        });
    }
    for (auto& th : threads) th.join();
}

}  // namespace

MatrixXfRM fa2_forward_causal_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                 const MatrixXfRM& V, int T, int d, int M_bytes,
                                 float scale, Schedule sched, int num_threads,
                                 std::vector<ThreadMetrics>& metrics) {
    const Tiling t = make_tiling(T, d, M_bytes);
    metrics.assign(num_threads, ThreadMetrics{});
    MatrixXfRM O = MatrixXfRM::Zero(T, d);

    switch (sched) {
        case Schedule::Static:
            run_omp<Schedule::Static>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
            break;
        case Schedule::Dynamic:
            run_omp<Schedule::Dynamic>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
            break;
        case Schedule::WorkStealing:
            run_work_stealing(Q, Kt, V, t, T, scale, num_threads, O, metrics);
            break;
    }
    return O;
}

}  // namespace flash_attn
