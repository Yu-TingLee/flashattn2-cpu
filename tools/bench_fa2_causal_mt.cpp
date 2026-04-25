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
#include <numeric>
#include <omp.h>
#include "flash_attn/attn.hpp"
#include "flash_attn/io.hpp"
#include "flash_attn/args.hpp"
#include "flash_attn/verify.hpp"

static Schedule parse_schedule(const std::string& s) {
    if (s == "static")        return Schedule::STATIC;
    if (s == "dynamic")       return Schedule::DYNAMIC;
    if (s == "work_stealing") return Schedule::WORK_STEALING;
    std::cerr << "Unknown schedule '" << s << "'. Use: static | dynamic | work_stealing\n";
    std::exit(1);
}

int main(int argc, char** argv) {
    const int T            = arg_int(argc, argv, "--T", 1024);
    const int d            = arg_int(argc, argv, "--d", 64);
    const int M_bytes      = arg_int(argc, argv, "--M_bytes", 131072);
    const int num_testsets = arg_int(argc, argv, "--num_testsets", 100);
    const std::string opt  = arg_str(argc, argv, "--opt_flag", "O3");
    const std::string sched_str = arg_str(argc, argv, "--schedule", "dynamic");
    const int num_threads  = arg_int(argc, argv, "--num_threads", omp_get_max_threads());

    const Schedule sched = parse_schedule(sched_str);

    const std::string out_dir  = "outputs/fa2_causal_" + opt + "_mt_" + sched_str
                                 + "_t" + std::to_string(num_threads);
    const std::string log_path = out_dir + "/runtime.csv";
    std::filesystem::create_directories(out_dir);

    std::cout << "Running fa2_causal_" << opt << "_mt (" << sched_str << ")"
              << ": T=" << T << ", d=" << d << ", M=" << M_bytes / 1024
              << " KiB, threads=" << num_threads << "\n";

    static const std::string CSV_HEADER =
        "testset,T,d,M_bytes,schedule,num_threads,runtime,"
        "load_imbalance,steal_overhead,steal_success_rate,is_correct\n";

    std::set<std::tuple<int,int,int,int>> done;

    if (std::filesystem::exists(log_path)) {
        std::ifstream fin(log_path);
        std::string header, line;
        std::getline(fin, header);
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string seg;
            std::vector<std::string> p;
            while (std::getline(ss, seg, ',')) p.push_back(seg);
            if (p.size() >= 10 && p.back() == "1") {
                try { done.insert({std::stoi(p[0]), std::stoi(p[1]),
                                   std::stoi(p[2]), std::stoi(p[3])}); }
                catch (...) {}
            }
        }
    } else {
        std::ofstream f(log_path);
        f << CSV_HEADER;
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

        std::vector<ThreadMetrics> metrics;
        const auto t1 = std::chrono::steady_clock::now();
        auto O = fa2_forward_causal_mt(Q, Kt, V, T, d, M_bytes, scale,
                                       sched, num_threads, metrics);
        const double runtime =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

        // Load imbalance: max thread time / mean thread time
        double max_t = 0.0, sum_t = 0.0;
        double total_steals = 0.0, total_successes = 0.0, total_blocks = 0.0;
        for (const auto& tm : metrics) {
            max_t             = std::max(max_t, tm.wall_time_sec);
            sum_t            += tm.wall_time_sec;
            total_steals     += tm.steal_attempts;
            total_successes  += tm.steal_successes;
            total_blocks     += tm.blocks_processed;
        }
        const int n = static_cast<int>(metrics.size());
        const double mean_t            = (n > 0) ? sum_t / n : 1.0;
        const double load_imbalance    = (mean_t > 0) ? max_t / mean_t : 1.0;
        const double steal_overhead    = (total_blocks  > 0) ? total_steals    / total_blocks  : 0.0;
        const double steal_success_rate = (total_steals > 0) ? total_successes / total_steals  : 0.0;

        const std::string ref_path =
            data_dir + "/O_causal_T" + std::to_string(T) + "_d" + std::to_string(d) + ".npy";
        const bool ok = verify_output(O, load_npy(ref_path, T, d));

        std::cout << "Runtime " << i << ": " << std::fixed << std::setprecision(4)
                  << runtime << " s  |  imbalance=" << std::setprecision(3)
                  << load_imbalance << "  |  steal_ovhd=" << steal_overhead
                  << "  |  steal_ok=" << steal_success_rate
                  << "  |  ok=" << ok << "\n";

        std::ofstream f(log_path, std::ios::app);
        f << i << "," << T << "," << d << "," << M_bytes << ","
          << sched_str << "," << num_threads << ","
          << std::fixed << std::setprecision(6) << runtime << ","
          << std::setprecision(6) << load_imbalance << ","
          << steal_overhead << "," << steal_success_rate << "," << ok << "\n";
    }
    return 0;
}
