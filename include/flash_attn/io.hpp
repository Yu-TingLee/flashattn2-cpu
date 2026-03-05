#pragma once
#include <string>
#include "types.hpp"

MatrixXfRM load_npy(const std::string& path, int rows, int cols);
MatrixXfRM load_matrix(const std::string& path, int rows, int cols);
