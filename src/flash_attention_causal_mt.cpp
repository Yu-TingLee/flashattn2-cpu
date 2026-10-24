#include <omp.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>

#include "flash_attention_block.hpp"
#include "flash_attn/flash_attention.hpp"
#include "work_stealing_queue.hpp"

namespace flash_attn {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

template <bool Causal, Schedule S, bool Profile>
void run_omp(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V, const Tiling& t, int T,
             float scale, int num_threads, MatrixXfRM& O, std::vector<ThreadMetrics>& metrics) {
#pragma omp parallel num_threads(num_threads)
  {
    ThreadMetrics& tm = metrics[omp_get_thread_num()];
    const auto t0 = Clock::now();

    if constexpr (S == Schedule::Static) {
#pragma omp for schedule(static) nowait
      for (int i = 0; i < t.Tr; ++i) {
        process_query_block<Causal, Profile>(i, t, T, Q, Kt, V, scale, O, &tm.steps);
        tm.blocks_processed++;
      }
    } else {
#pragma omp for schedule(dynamic) nowait
      for (int i = 0; i < t.Tr; ++i) {
        process_query_block<Causal, Profile>(i, t, T, Q, Kt, V, scale, O, &tm.steps);
        tm.blocks_processed++;
      }
    }

    tm.wall_time_sec = seconds_since(t0);
  }
}

template <bool Profile>
void run_work_stealing(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                       const Tiling& t, int T, float scale, int num_threads, MatrixXfRM& O,
                       std::vector<ThreadMetrics>& metrics) {
  int64_t capacity = 1;
  while (capacity < t.Tr) {
    capacity <<= 1;
  }

  std::vector<std::unique_ptr<WorkStealingQueue<int>>> queues(num_threads);
  for (auto& q : queues) {
    q = std::make_unique<WorkStealingQueue<int>>(capacity);
  }

  const int base = t.Tr / num_threads;
  const int rem = t.Tr % num_threads;
  for (int tid = 0; tid < num_threads; ++tid) {
    const int start = tid * base + std::min(tid, rem);
    const int count = base + (tid < rem ? 1 : 0);
    for (int i = start; i < start + count; ++i) {
      queues[tid]->push(i);
    }
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
          if (item) {
            tm.steal_successes++;
          }
        }
        if (!item) {
          break;
        }
        process_query_block<true, Profile>(*item, t, T, Q, Kt, V, scale, O, &tm.steps);
        tm.blocks_processed++;
      }

      tm.wall_time_sec = seconds_since(t0);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}

}  // namespace

MatrixXfRM fa2_forward_causal_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                                 int T, int d, int M_bytes, float scale, Schedule sched,
                                 int num_threads, std::vector<ThreadMetrics>& metrics) {
  const Tiling t = make_tiling(T, d, M_bytes);
  metrics.assign(num_threads, ThreadMetrics{});
  MatrixXfRM O = MatrixXfRM::Zero(T, d);

  switch (sched) {
    case Schedule::Static:
      run_omp<true, Schedule::Static, false>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
      break;
    case Schedule::Dynamic:
      run_omp<true, Schedule::Dynamic, false>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
      break;
    case Schedule::WorkStealing:
      run_work_stealing<false>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
      break;
  }
  return O;
}

ProfileResult fa2_forward_mt_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                                     int T, int d, int M_bytes, float scale, bool causal,
                                     Schedule sched, int num_threads) {
  const Tiling t = make_tiling(T, d, M_bytes);
  std::vector<ThreadMetrics> metrics(num_threads);
  MatrixXfRM O = MatrixXfRM::Zero(T, d);

  const auto t0 = Clock::now();
  if (!causal) {
    run_omp<false, Schedule::Dynamic, true>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
  } else {
    switch (sched) {
      case Schedule::Static:
        run_omp<true, Schedule::Static, true>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
        break;
      case Schedule::Dynamic:
        run_omp<true, Schedule::Dynamic, true>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
        break;
      case Schedule::WorkStealing:
        run_work_stealing<true>(Q, Kt, V, t, T, scale, num_threads, O, metrics);
        break;
    }
  }

  ProfileData p;
  p.runtime = seconds_since(t0);
  double busy = 0.0;
  for (const ThreadMetrics& tm : metrics) {
    const ProfileData& s = tm.steps;
    p.load_kv += s.load_kv;
    p.compute_s += s.compute_s;
    p.online_softmax += s.online_softmax;
    p.update_o += s.update_o;
    p.others +=
        std::max(0.0, tm.wall_time_sec - s.load_kv - s.compute_s - s.online_softmax - s.update_o);
    busy += tm.wall_time_sec;
  }
  p.idle = std::max(0.0, num_threads * p.runtime - busy);
  return {std::move(O), p};
}

}  // namespace flash_attn
