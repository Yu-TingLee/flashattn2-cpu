#pragma once
#include <string>

#include "flash_attn/types.hpp"

namespace flash_attn {

MatrixXfRM load_npy(const std::string& path, int rows, int cols);
MatrixXfRM load_matrix(const std::string& path, int rows, int cols);

}  // namespace flash_attn
