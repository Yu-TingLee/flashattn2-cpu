#pragma once
#include <cstdint>
#include <vector>

#include "flash_attn/types.hpp"

namespace flash_attn {

// FlashAttention-2 forward pass. Q is T x d, Kt is K transposed (d x T),
// V is T x d. M_bytes is the on-chip memory budget that sets the block sizes.

MatrixXfRM fa2_forward(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V, int T, int d,
                       int M_bytes, float scale);

MatrixXfRM fa2_forward_causal(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V, int T,
                              int d, int M_bytes, float scale);

MatrixXfRM fa2_forward_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V, int T,
                          int d, int M_bytes, float scale);

enum class Schedule { Static, Dynamic, WorkStealing };

// Seconds per inner-loop step, summed over threads.
struct ProfileData {
  double runtime = 0.0;
  double load_kv = 0.0;
  double compute_s = 0.0;
  double online_softmax = 0.0;
  double update_o = 0.0;
  double others = 0.0;
  double idle = 0.0;
};

struct ThreadMetrics {
  int64_t blocks_processed = 0;
  double wall_time_sec = 0.0;
  int64_t steal_attempts = 0;
  int64_t steal_successes = 0;
  ProfileData steps;  // filled only by the profiled kernels
};

MatrixXfRM fa2_forward_causal_mt(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                                 int T, int d, int M_bytes, float scale, Schedule sched,
                                 int num_threads, std::vector<ThreadMetrics>& metrics);

struct ProfileResult {
  MatrixXfRM O;
  ProfileData profile;
};

ProfileResult fa2_forward_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                                  int T, int d, int M_bytes, float scale);

// sched is ignored when causal is false: self-attention always schedules
// dynamically, as fa2_forward_mt does.
ProfileResult fa2_forward_mt_profile(const MatrixXfRM& Q, const MatrixXfRM& Kt, const MatrixXfRM& V,
                                     int T, int d, int M_bytes, float scale, bool causal,
                                     Schedule sched, int num_threads);

}  // namespace flash_attn
