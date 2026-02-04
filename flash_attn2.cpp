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

MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V, 
    const int T, const int d,const int M_bytes, float scale) {
    int I = M_bytes / (4 * d * 4);
    int J = std::min(I, d);
    if (I < 1) I = 1;
    if (J < 1) J = 1;

    int Tr = (T + I - 1) / I;
    int Tc = (T + J - 1) / J;

    MatrixXfRM O = MatrixXfRM::Zero(T, d);
    
    for (int i =0; i < Tr; i++){
        int row_start = i * I;
        int row_end = std::min(row_start + I, T);
        int cur_I = row_end - row_start;

        Eigen::VectorXf l = Eigen::VectorXf::Zero(cur_I);
        Eigen::VectorXf m = Eigen::VectorXf::Constant(cur_I, -INFINITY);
        MatrixXfRM Q_i = Q.middleRows(row_start, row_end - row_start);
        
        for (int j = 0; j < Tc; j++){
            int col_start = j * J;
            int col_end = std::min(col_start + J, T);

            MatrixXfRM Kt_j = Kt.middleCols(col_start, col_end - col_start);
            MatrixXfRM V_j = V.middleRows(col_start, col_end - col_start);

            MatrixXfRM S = Q_i * Kt_j;
            S *= scale;

            Eigen::VectorXf m_block = S.rowwise().maxCoeff();
            Eigen::VectorXf m_new = m.cwiseMax(m_block);
            
            Eigen::VectorXf alpha = (m - m_new).array().exp();
            Eigen::MatrixXf P = (S.colwise() - m_new).array().exp().matrix();
            
            Eigen::VectorXf l_new = (alpha.cwiseProduct(l)) + P.rowwise().sum();
            auto O_block = O.middleRows(row_start, cur_I);
            O_block.array().colwise() *= alpha.array();
            O.middleRows(row_start, cur_I) += (P * V_j);
            m = m_new;
            l = l_new;
        }

        O.middleRows(row_start, cur_I).array().colwise() /= l.array();
    }
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
    int M_bytes = arg_int(argc, argv, "--M_bytes", 131072);
    int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
    std::string opt_flag = "O3";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--opt_flag" && i + 1 < argc) opt_flag = argv[i + 1];
    }

    std::string out_dir = "outputs/fa2_cpp_" + opt_flag;
    std::filesystem::create_directories(out_dir);
    std::string log_path = out_dir + "/runtime.csv";

    // Resume logic
    std::set<std::tuple<int, int, int, int>> completed_runs;

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

            // Ensure line has at least id, T, d, M_bytes (4 parts)
            if (parts.size() >= 4) {
                try {
                    int r_id = std::stoi(parts[0]);
                    int r_T  = std::stoi(parts[1]);
                    int r_d  = std::stoi(parts[2]);
                    int r_M_bytes = std::stoi(parts[3]);
                    completed_runs.insert({r_id, r_T, r_d, r_M_bytes});
                } catch (...) { continue; } // Skip malformed lines
            }
        }
    } else {
        std::ofstream f(log_path);
        f << "testset,T,d,M_bytes,runtime,is_correct\n";
    }
    // --- Main loop ---
    for (int i = 0; i < num_testsets; ++i) {
        if (completed_runs.count({i, T, d, M_bytes})) {
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
        auto O = fa2_forward(Q, Kt, V, T, d, M_bytes, scale);
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
        f << i << "," << T << "," << d << "," << M_bytes << "," << std::fixed << std::setprecision(6) << runtime << "," << is_correct << "\n";
    }
    return 0;
}