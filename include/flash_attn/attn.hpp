#pragma once
#include "types.hpp"
#include <vector>

// Naive scaled dot-product attention
MatrixXfRM naive_attention(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                            const MatrixXfRM& V, float scale);

// Naive attention with causal mask
MatrixXfRM naive_attention_causal(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                   const MatrixXfRM& V, float scale);

// FlashAttention-2 forward pass
MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                        const MatrixXfRM& V, int T, int d, int M_bytes, float scale);

// FlashAttention-2 forward pass with causal mask
MatrixXfRM fa2_forward_causal(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                               const MatrixXfRM& V, int T, int d, int M_bytes, float scale);

// Scheduling strategy for multi-threaded causal FA-2
enum class Schedule { STATIC, DYNAMIC, WORK_STEALING };

// Per-thread metrics collected during causal MT forward pass
struct ThreadMetrics {
    int    blocks_processed = 0;
    double wall_time_sec    = 0.0;
    double steal_attempts   = 0.0;  // work-stealing only
    double steal_successes  = 0.0;  // work-stealing only
};

// Multi-threaded causal FA-2 with selectable scheduling strategy
MatrixXfRM fa2_forward_causal_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                  const MatrixXfRM& V, int T, int d, int M_bytes,
                                  float scale, Schedule sched, int num_threads,
                                  std::vector<ThreadMetrics>& metrics_out);

// Per-line profiling variant — same math as fa2_forward, returns timing breakdown
struct ProfileData {
    double runtime, t7, t8, t10, t11, others;
};

struct ProfileResult {
    MatrixXfRM  O;
    ProfileData profile;
};

ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt,
                                   const MatrixXfRM& V, int T, int d,
                                   int M_bytes, float scale);
