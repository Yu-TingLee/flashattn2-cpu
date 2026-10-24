#include "bench_utils.hpp"
#include "flash_attn/naive_attention.hpp"

int main(int argc, char** argv) {
  const int T = arg_int(argc, argv, "--T", 1024);
  const int d = arg_int(argc, argv, "--d", 64);
  const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
  const float scale = 1.0f / std::sqrt(float(d));

  RuntimeLog log("naive_attn_cpp", "testset,T,d,runtime,is_correct", 3);

  for (int i = 0; i < num_testsets; ++i) {
    if (log.done({i, T, d})) {
      std::cout << "Skipping testset " << i << "\n";
      continue;
    }
    const TestSet ts = load_testset(i, T, d);

    flash_attn::MatrixXfRM O;
    const double runtime =
        time_sec([&] { O = flash_attn::naive_attention(ts.Q, ts.Kt, ts.V, scale); });
    std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4) << runtime << " s\n";

    const bool ok = allclose(O, load_reference(ts, T, d, false));
    log.append() << i << "," << T << "," << d << "," << std::fixed << std::setprecision(6)
                 << runtime << "," << ok << "\n";
  }
  return 0;
}
