#include <omp.h>

#include <algorithm>

#include "bench_utils.hpp"
#include "flash_attn/flash_attention.hpp"

int main(int argc, char** argv) {
  const int T = arg_int(argc, argv, "--T", 1024);
  const int d = arg_int(argc, argv, "--d", 64);
  const int M_bytes = arg_int(argc, argv, "--M_bytes", 131072);
  const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
  const std::string sched_str = arg_str(argc, argv, "--schedule", "dynamic");
  const int num_threads = arg_int(argc, argv, "--num_threads", omp_get_max_threads());
  const flash_attn::Schedule sched = parse_schedule(sched_str);
  const float scale = 1.0f / std::sqrt(float(d));

  const std::string name = "fa2_causal_cpp_mt_" + sched_str + "_t" + std::to_string(num_threads);
  RuntimeLog log(name,
                 "testset,T,d,M_bytes,schedule,num_threads,runtime,"
                 "load_imbalance,steal_overhead,steal_success_rate,is_correct",
                 4, /*only_correct=*/true);

  std::cout << "Running " << name << ": T=" << T << ", d=" << d << ", M=" << M_bytes / 1024
            << " KiB\n";

  for (int i = 0; i < num_testsets; ++i) {
    if (log.done({i, T, d, M_bytes})) {
      std::cout << "Skipping testset " << i << "\n";
      continue;
    }
    const TestSet ts = load_testset(i, T, d);

    std::vector<flash_attn::ThreadMetrics> metrics;
    flash_attn::MatrixXfRM O;
    const double runtime = time_sec([&] {
      O = flash_attn::fa2_forward_causal_mt(ts.Q, ts.Kt, ts.V, T, d, M_bytes, scale, sched,
                                            num_threads, metrics);
    });

    double max_t = 0.0, sum_t = 0.0;
    double total_steals = 0.0, total_successes = 0.0, total_blocks = 0.0;
    for (const auto& tm : metrics) {
      max_t = std::max(max_t, tm.wall_time_sec);
      sum_t += tm.wall_time_sec;
      total_steals += tm.steal_attempts;
      total_successes += tm.steal_successes;
      total_blocks += tm.blocks_processed;
    }
    const int n = static_cast<int>(metrics.size());
    const double mean_t = (n > 0) ? sum_t / n : 1.0;
    const double load_imbalance = (mean_t > 0) ? max_t / mean_t : 1.0;
    const double steal_overhead = (total_blocks > 0) ? total_steals / total_blocks : 0.0;
    const double steal_success_rate = (total_steals > 0) ? total_successes / total_steals : 0.0;

    const bool ok = allclose(O, load_reference(ts, T, d, true));

    std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4) << runtime
              << " s  |  imbalance=" << std::setprecision(3) << load_imbalance
              << "  |  steal_ovhd=" << steal_overhead << "  |  steal_ok=" << steal_success_rate
              << "  |  ok=" << ok << "\n";

    log.append() << i << "," << T << "," << d << "," << M_bytes << "," << sched_str << ","
                 << num_threads << "," << std::fixed << std::setprecision(6) << runtime << ","
                 << load_imbalance << "," << steal_overhead << "," << steal_success_rate << ","
                 << ok << "\n";
  }
  return 0;
}
