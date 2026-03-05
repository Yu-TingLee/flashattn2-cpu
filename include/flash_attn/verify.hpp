#pragma once
#include <cmath>
#include "types.hpp"

inline bool verify_output(const MatrixXfRM& O, const MatrixXfRM& O_ref,
                           float atol = 1e-3f, float rtol = 1e-3f) {
    if (O.rows() != O_ref.rows() || O.cols() != O_ref.cols()) return false;
    const float* p     = O.data();
    const float* p_ref = O_ref.data();
    int total = static_cast<int>(O.size());
    for (int k = 0; k < total; ++k)
        if (std::abs(p[k] - p_ref[k]) > atol + rtol * std::abs(p_ref[k]))
            return false;
    return true;
}
