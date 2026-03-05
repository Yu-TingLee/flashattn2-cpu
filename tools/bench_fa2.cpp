#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <tuple>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "flash_attn/attn.hpp"
#include "flash_attn/io.hpp"
#include "flash_attn/args.hpp"
#include "flash_attn/verify.hpp"

int main(int argc, char** argv) {
    const int T            = arg_int(argc, argv, "--T", 1024);
    const int d            = arg_int(argc, argv, "--d", 64);
    const int M_bytes      = arg_int(argc, argv, "--M_bytes", 131072);
    const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
    const std::string opt  = arg_str(argc, argv, "--opt_flag", "O3");

    const std::string out_dir  = "outputs/fa2_cpp_" + opt;
    const std::string log_path = out_dir + "/runtime.csv";
    std::filesystem::create_directories(out_dir);

    std::set<std::tuple<int,int,int,int>> done;
    if (std::filesystem::exists(log_path)) {
        std::ifstream fin(log_path);
        std::string line;
        std::getline(fin, line);
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string seg;
            std::vector<std::string> p;
            while (std::getline(ss, seg, ',')) p.push_back(seg);
            if (p.size() >= 4) {
                try { done.insert({std::stoi(p[0]), std::stoi(p[1]),
                                   std::stoi(p[2]), std::stoi(p[3])}); }
                catch (...) {}
            }
        }
    } else {
        std::ofstream f(log_path);
        f << "testset,T,d,M_bytes,runtime,is_correct\n";
    }

    const float scale = 1.0f / std::sqrt(float(d));

    for (int i = 0; i < num_testsets; ++i) {
        if (done.count({i, T, d, M_bytes})) {
            std::cout << "Skipping testset " << i << "\n";
            continue;
        }

        const std::string data_dir = "data/testset" + std::to_string(i);
        auto Q  = load_matrix(data_dir + "/Q.txt",  T, d);
        auto Kt = load_matrix(data_dir + "/Kt.txt", d, T);
        auto V  = load_matrix(data_dir + "/V.txt",  T, d);

        const auto t1 = std::chrono::steady_clock::now();
        auto O = fa2_forward(Q, Kt, V, T, d, M_bytes, scale);
        const double runtime =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

        std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4)
                  << runtime << " s\n";

        const std::string ref =
            data_dir + "/O_T" + std::to_string(T) + "_d" + std::to_string(d) + ".npy";
        const bool ok = verify_output(O, load_npy(ref, T, d));

        std::ofstream f(log_path, std::ios::app);
        f << i << "," << T << "," << d << "," << M_bytes << ","
          << std::fixed << std::setprecision(6) << runtime << "," << ok << "\n";
    }
    return 0;
}
