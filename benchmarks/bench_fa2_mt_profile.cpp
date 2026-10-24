#include <omp.h>

#include "bench_utils.hpp"
#include "flash_attn/flash_attention.hpp"

int main(int argc, char** argv) {
  const int T = arg_int(argc, argv, "--T", 1024);
  const int d = arg_int(argc, argv, "--d", 64);
  const int M_bytes = arg_int(argc, argv, "--M_bytes", 131072);
  const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
  const bool causal = arg_int(argc, argv, "--causal", 0) != 0;
  const std::string sched_str = arg_str(argc, argv, "--schedule", "dynamic");
  const int num_threads = arg_int(argc, argv, "--num_threads", omp_get_max_threads());
  const flash_attn::Schedule sched = parse_schedule(sched_str);
  const float scale = 1.0f / std::sqrt(float(d));

  const std::string name =
      causal ? "fa2_causal_cpp_mt_profile_" + sched_str + "_t" + std::to_string(num_threads)
             : "fa2_cpp_mt_profile_t" + std::to_string(num_threads);
  // line7..line11 refer to the steps of Algorithm 1, summed over threads.
  RuntimeLog log(name,
                 "testset,T,d,M_bytes,num_threads,runtime,"
                 "line7,line8,line10,line11,others,idle,is_correct",
                 4, /*only_correct=*/true);

  std::cout << "Running " << name << ": T=" << T << ", d=" << d << ", M=" << M_bytes / 1024
            << " KiB\n";

  for (int i = 0; i < num_testsets; ++i) {
    if (log.done({i, T, d, M_bytes})) {
      std::cout << "Skipping testset " << i << "\n";
      continue;
    }
    const TestSet ts = load_testset(i, T, d);

    const flash_attn::ProfileResult r = flash_attn::fa2_forward_mt_profile(
        ts.Q, ts.Kt, ts.V, T, d, M_bytes, scale, causal, sched, num_threads);
    const flash_attn::ProfileData& p = r.profile;
    const bool ok = allclose(r.O, load_reference(ts, T, d, causal));

    std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4) << p.runtime
              << " s  |  thread-s: L7 " << p.load_kv << "  L8 " << p.compute_s << "  L10 "
              << p.online_softmax << "  L11 " << p.update_o << "  others " << p.others << "  idle "
              << p.idle << "  |  ok=" << ok << "\n";

    log.append() << i << "," << T << "," << d << "," << M_bytes << "," << num_threads << ","
                 << std::fixed << std::setprecision(6) << p.runtime << "," << p.load_kv << ","
                 << p.compute_s << "," << p.online_softmax << "," << p.update_o << "," << p.others
                 << "," << p.idle << "," << ok << "\n";
  }
  return 0;
}
