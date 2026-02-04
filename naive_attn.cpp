#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <set>
#include <tuple>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <Eigen/Dense>
#include <cstdint>

using MatrixXfRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

MatrixXfRM load_npy(const std::string& path, int rows, int cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << path << "\n";
        return MatrixXfRM(0, 0);
    }

    // 1. Skip Magic String (6 bytes) + Version (2 bytes) = 8 bytes
    file.ignore(8);

    // 2. Read Header Length (2 bytes, little-endian)
    uint16_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(uint16_t));

    // 3. Skip the Header Text
    file.ignore(header_len);

    // 4. Read Data directly into Eigen Matrix
    MatrixXfRM mat(rows, cols);
    file.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(float));

    return mat;
}

MatrixXfRM load_matrix(const std::string& path, int rows, int cols) {
    MatrixXfRM mat(rows, cols);
    std::ifstream fin(path);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fin >> mat(i, j);
        }
    }
    return mat;
}

MatrixXfRM naive_attention(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V, float scale) {
    // 1. Compute S = scale*(QK^T)
    // 2. Apply safe softmax row-wise on S
    // 3. Compute O = P*V
    MatrixXfRM S = Q * Kt;
    S *= scale;
    Eigen::VectorXf row_max = S.rowwise().maxCoeff();
    S.colwise() -= row_max;
    S = S.array().exp();
    Eigen::VectorXf row_sum = S.rowwise().sum();
    S.array().colwise() /= row_sum.array();
    MatrixXfRM O = S * V;
    return O;
}

int arg_int(int argc, char** argv, const std::string& key, int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) return std::stoi(argv[i + 1]);
    }
    return def;
}

int main(int argc, char** argv) {
    int T = arg_int(argc, argv, "--T", 1024);
    int d = arg_int(argc, argv, "--d", 64);
    int num_testsets = arg_int(argc, argv, "--num_testsets", 100);

    std::string opt_flag = "O3";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--opt_flag" && i + 1 < argc) opt_flag = argv[i + 1];
    }

    std::string out_dir = "outputs/naive_attn_cpp_" + opt_flag;
    std::filesystem::create_directories(out_dir);
    std::string log_path = out_dir + "/runtime.csv";

    // Resume logic
    std::set<std::tuple<int, int, int>> completed_runs;

    if (std::filesystem::exists(log_path)) {
        std::ifstream fin(log_path);
        std::string line;
        std::getline(fin, line); // skip header
        
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> parts;
            
            // Parse CSV line
            while(std::getline(ss, segment, ',')) {
                parts.push_back(segment);
            }

            // Ensure line has at least id, T, d (3 parts)
            if (parts.size() >= 3) {
                try {
                    int r_id = std::stoi(parts[0]);
                    int r_T  = std::stoi(parts[1]);
                    int r_d  = std::stoi(parts[2]);
                    completed_runs.insert({r_id, r_T, r_d});
                } catch (...) { continue; } // Skip malformed lines
            }
        }
    } else {
        std::ofstream f(log_path);
        f << "testset,T,d,runtime,is_correct\n";
    }
    // --- Main loop ---
    for (int i = 0; i < num_testsets; ++i) {
        if (completed_runs.count({i, T, d})) {
            std::cout << "Skipping testset " << i << "\n";
            continue;
        }

        std::string data_dir = "data/testset" + std::to_string(i);

        // Load Inputs
        auto Q = load_matrix(data_dir + "/Q.txt", T, d);
        auto Kt = load_matrix(data_dir + "/Kt.txt", d, T);
        auto V = load_matrix(data_dir + "/V.txt", T, d);
        float scale = 1.0f / std::sqrt(float(d));

        // Execute & Time
        auto t1 = std::chrono::steady_clock::now();
        auto O = naive_attention(Q, Kt, V, scale);
        auto t2 = std::chrono::steady_clock::now();
        double runtime = std::chrono::duration<double>(t2 - t1).count();

        std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4) << runtime << " seconds." << std::endl;

        // Verify
        std::string ref_path = data_dir + "/O_T" + std::to_string(T) + "_d" + std::to_string(d) + ".npy";
        auto O_ref = load_npy(ref_path, T, d);

        bool is_correct = true;
        float atol = 1e-3f, rtol = 1e-3f;

        // Validation Loop
        if (O.rows() != O_ref.rows() || O.cols() != O_ref.cols()) {
            is_correct = false;
        } else {
            const float* p_calc = O.data();
            const float* p_ref = O_ref.data();
            int total = T * d;

            for (int k = 0; k < total; ++k) {
                float diff = std::abs(p_calc[k] - p_ref[k]);
                float tol = atol + rtol * std::abs(p_ref[k]);
                if (diff > tol) {
                    is_correct = false;
                    break;
                }
            }
        }

        std::ofstream f(log_path, std::ios::app);
        f << i << "," << T << "," << d << "," << std::fixed << std::setprecision(6) << runtime << "," << is_correct << "\n";
    }
    return 0;
}