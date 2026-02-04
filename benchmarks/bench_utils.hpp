#pragma once
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "flash_attn/matrix_io.hpp"

inline int arg_int(int argc, char** argv, const std::string& key, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return std::stoi(argv[i + 1]);
    return def;
}

inline std::string arg_str(int argc, char** argv, const std::string& key,
                           const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return argv[i + 1];
    return def;
}

// CSV log at outputs/<name>/runtime.csv. Rows already present are keyed by
// their first key_cols integer fields so an interrupted sweep can resume.
class RuntimeLog {
public:
    RuntimeLog(const std::string& name, const std::string& header,
               size_t key_cols, bool only_correct = false)
        : path_("outputs/" + name + "/runtime.csv") {
        std::filesystem::create_directories("outputs/" + name);
        if (!std::filesystem::exists(path_)) {
            std::ofstream(path_) << header << "\n";
            return;
        }
        std::ifstream fin(path_);
        std::string line;
        std::getline(fin, line);
        while (std::getline(fin, line)) {
            std::vector<std::string> fields;
            std::stringstream ss(line);
            for (std::string f; std::getline(ss, f, ',');) fields.push_back(f);
            if (fields.size() < key_cols) continue;
            if (only_correct && fields.back() != "1") continue;
            try {
                std::vector<int> key;
                for (size_t k = 0; k < key_cols; ++k) key.push_back(std::stoi(fields[k]));
                done_.insert(std::move(key));
            } catch (...) {}
        }
    }

    bool done(const std::vector<int>& key) const { return done_.count(key) > 0; }

    std::ofstream append() const { return std::ofstream(path_, std::ios::app); }

private:
    std::string path_;
    std::set<std::vector<int>> done_;
};

struct TestSet {
    std::string dir;
    flash_attn::MatrixXfRM Q, Kt, V;
};

inline TestSet load_testset(int i, int T, int d) {
    TestSet ts;
    ts.dir = "data/testset" + std::to_string(i);
    ts.Q  = flash_attn::load_matrix(ts.dir + "/Q.txt",  T, d);
    ts.Kt = flash_attn::load_matrix(ts.dir + "/Kt.txt", d, T);
    ts.V  = flash_attn::load_matrix(ts.dir + "/V.txt",  T, d);
    return ts;
}

// Reference outputs are written by the Python naive-attention benchmarks.
inline flash_attn::MatrixXfRM load_reference(const TestSet& ts, int T, int d, bool causal) {
    const std::string name = std::string(causal ? "O_causal_T" : "O_T") +
                             std::to_string(T) + "_d" + std::to_string(d) + ".npy";
    return flash_attn::load_npy(ts.dir + "/" + name, T, d);
}

inline bool allclose(const flash_attn::MatrixXfRM& O, const flash_attn::MatrixXfRM& O_ref,
                     float atol = 1e-3f, float rtol = 1e-3f) {
    if (O.rows() != O_ref.rows() || O.cols() != O_ref.cols()) return false;
    for (Eigen::Index k = 0; k < O.size(); ++k)
        if (std::abs(O.data()[k] - O_ref.data()[k]) > atol + rtol * std::abs(O_ref.data()[k]))
            return false;
    return true;
}

template <typename F>
double time_sec(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
