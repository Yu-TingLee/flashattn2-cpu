#include "flash_attn/flash_attention.hpp"
#include <algorithm>
#include <chrono>
#include "flash_attention_block.hpp"

namespace flash_attn {

ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                  const MatrixXfRM& V, int T, int d,
                                  int M_bytes, float scale) {
    using Clock = std::chrono::steady_clock;

    const Tiling t = make_tiling(T, d, M_bytes);
    MatrixXfRM O = MatrixXfRM::Zero(T, d);
    ProfileData p;

    const auto t0 = Clock::now();
    for (int i = 0; i < t.Tr; ++i)
        process_query_block<false, true>(i, t, T, Q, Kt, V, scale, O, &p);
    p.runtime = std::chrono::duration<double>(Clock::now() - t0).count();

    const double timed = p.load_kv + p.compute_s + p.online_softmax + p.update_o;
    p.others  = std::max(0.0, p.runtime - timed);
    return {std::move(O), p};
}

}  // namespace flash_attn
