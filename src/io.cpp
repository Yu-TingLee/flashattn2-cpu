#include "flash_attn/io.hpp"
#include <fstream>
#include <iostream>
#include <cstdint>

MatrixXfRM load_npy(const std::string& path, int rows, int cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << path << "\n";
        return MatrixXfRM(0, 0);
    }
    file.ignore(8);                                          // magic (6) + version (2)
    uint16_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(uint16_t));
    file.ignore(header_len);
    MatrixXfRM mat(rows, cols);
    file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(float));
    return mat;
}

MatrixXfRM load_matrix(const std::string& path, int rows, int cols) {
    MatrixXfRM mat(rows, cols);
    std::ifstream fin(path);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            fin >> mat(i, j);
    return mat;
}
