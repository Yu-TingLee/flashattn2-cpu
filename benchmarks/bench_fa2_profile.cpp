#include "bench_utils.hpp"
#include "flash_attn/flash_attention.hpp"

int main(int argc, char** argv) {
  const int T = arg_int(argc, argv, "--T", 1024);
  const int d = arg_int(argc, argv, "--d", 64);
  const int M_bytes = arg_int(argc, argv, "--M_bytes", 131072);
  const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
  const float scale = 1.0f / std::sqrt(float(d));

  // line7..line11 refer to the steps of Algorithm 1 in the report.
  RuntimeLog log("fa2_cpp_profile", "testset,T,d,M_bytes,runtime,line7,line8,line10,line11,others",
                 4);

  for (int i = 0; i < num_testsets; ++i) {
    if (log.done({i, T, d, M_bytes})) {
      std::cout << "Skipping testset " << i << "\n";
      continue;
    }
    const TestSet ts = load_testset(i, T, d);

    const flash_attn::ProfileData p =
        flash_attn::fa2_forward_profile(ts.Q, ts.Kt, ts.V, T, d, M_bytes, scale).profile;

    std::cout << "---------------------------------------\n"
              << "Runtime " << i << ": " << std::fixed << std::setprecision(4) << p.runtime
              << " s\n"
              << "Line 7: " << p.load_kv << " s  "
              << "Line 8: " << p.compute_s << " s  "
              << "Line 10: " << p.online_softmax << " s  "
              << "Line 11: " << p.update_o << " s  "
              << "Others: " << p.others << " s\n";

    log.append() << i << "," << T << "," << d << "," << M_bytes << "," << std::fixed
                 << std::setprecision(6) << p.runtime << "," << p.load_kv << "," << p.compute_s
                 << "," << p.online_softmax << "," << p.update_o << "," << p.others << "\n";
  }
  return 0;
}
