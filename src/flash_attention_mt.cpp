#include "flash_attn/flash_attention.hpp"
#include "flash_attention_block.hpp"

namespace flash_attn {

MatrixXfRM fa2_forward_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                          const MatrixXfRM& V, int T, int d, int M_bytes, float scale) {
    const Tiling t = make_tiling(T, d, M_bytes);
    MatrixXfRM O = MatrixXfRM::Zero(T, d);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < t.Tr; ++i)
        process_query_block(i, t, T, Q, Kt, V, scale, O);
    return O;
}

}  // namespace flash_attn
