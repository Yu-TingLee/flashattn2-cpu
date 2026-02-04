#pragma once
#include <Eigen/Dense>

namespace flash_attn {

using MatrixXfRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace flash_attn
