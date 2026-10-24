#include <omp.h>

#include "bench_utils.hpp"
#include "flash_attn/flash_attention.hpp"

int main(int argc, char** argv) {
  const int T = arg_int(argc, argv, "--T", 1024);
  const int d = arg_int(argc, argv, "--d", 64);
  const int M_bytes = arg_int(argc, argv, "--M_bytes", 131072);
  const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
  const int num_threads = arg_int(argc, argv, "--num_threads", omp_get_max_threads());
  const float scale = 1.0f / std::sqrt(float(d));

  omp_set_num_threads(num_threads);
  const std::string name = "fa2_cpp_mt_t" + std::to_string(num_threads);
  RuntimeLog log(name, "testset,T,d,M_bytes,runtime,is_correct", 4);

#pragma omp parallel
  {
#pragma omp single
    std::cout << "Running " << name << ": T=" << T << ", d=" << d << ", M=" << M_bytes / 1024
              << " KiB, " << omp_get_num_threads() << " OpenMP threads\n";
  }

  for (int i = 0; i < num_testsets; ++i) {
    if (log.done({i, T, d, M_bytes})) {
      std::cout << "Skipping testset " << i << "\n";
      continue;
    }
    const TestSet ts = load_testset(i, T, d);

    flash_attn::MatrixXfRM O;
    const double runtime =
        time_sec([&] { O = flash_attn::fa2_forward_mt(ts.Q, ts.Kt, ts.V, T, d, M_bytes, scale); });
    std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4) << runtime << " s\n";

    const bool ok = allclose(O, load_reference(ts, T, d, false));
    log.append() << i << "," << T << "," << d << "," << M_bytes << "," << std::fixed
                 << std::setprecision(6) << runtime << "," << ok << "\n";
  }
  return 0;
}
