/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/

#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#define NSTREAMS 2

#ifndef KLIGN_GPU_BLOCK_SIZE
#define KLIGN_GPU_BLOCK_SIZE 20000
#endif

namespace adept_sw {

// for storing the alignment results
struct AlignmentResults {
  short *ref_begin = nullptr;
  short *query_begin = nullptr;
  short *ref_end = nullptr;
  short *query_end = nullptr;
  short *top_scores = nullptr;
  char  *cigar = nullptr;
};

struct DriverState;

class GPUDriver {
  DriverState *driver_state = nullptr;
  AlignmentResults alignments;

 public:
  GPUDriver(int upcxx_rank_me, int upcxx_rank_n, short match_score, short mismatch_score, short gap_opening_score,
            short gap_extending_score, int rlen_limit, bool compute_cigar, double &init_time);
  ~GPUDriver();

  void run_kernel_forwards(std::vector<std::string> &reads, std::vector<std::string> &contigs, unsigned maxReadSize,
                           unsigned maxContigSize);
  void run_kernel_backwards(std::vector<std::string> &reads, std::vector<std::string> &contigs, unsigned maxReadSize,
                            unsigned maxContigSize);
  void run_kernel_traceback(std::vector<std::string> &reads, std::vector<std::string> &contigs, unsigned maxReadSize,
                            unsigned maxContigSize);
  bool kernel_is_done();
  void kernel_block();

  AlignmentResults &get_aln_results() { return alignments; }
};

}  // namespace adept_sw
