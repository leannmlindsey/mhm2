#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#define NSTREAMS 2

namespace adept_sw {

// for storing the alignment results
struct AlignmentResults {
  short *ref_begin = nullptr;
  short *query_begin = nullptr;
  short *ref_end = nullptr;
  short *query_end = nullptr;
  short *top_scores = nullptr;
};

size_t get_tot_gpu_mem();
size_t get_avail_gpu_mem_per_rank(int totRanks);
int get_num_node_gpus();

struct DriverState;

class GPUDriver {
  DriverState *driver_state = nullptr;
  AlignmentResults alignments;

 public:
  ~GPUDriver();

  void init(int upcxx_rank_me, int upcxx_rank_n, short match_score, short mismatch_score, short gap_opening_score,
            short gap_extending_score, int rlen_limit);
  void run_kernel_forwards(std::vector<std::string> &reads, std::vector<std::string> &contigs, unsigned maxReadSize,
                           unsigned maxContigSize);
  void run_kernel_backwards(std::vector<std::string> &reads, std::vector<std::string> &contigs, unsigned maxReadSize,
                            unsigned maxContigSize);
  bool kernel_is_done();

  AlignmentResults &get_aln_results() { return alignments; }
};

}  // namespace adept_sw
