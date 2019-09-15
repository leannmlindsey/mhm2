#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <queue>
#include "zstr.hpp"
#include <upcxx/upcxx.hpp>

#include "utils.hpp"
#include "progressbar.hpp"
#include "walk_graph.hpp"
#include "colors.h"

using namespace std;
using namespace upcxx;

static const int MIN_SCAFFOLD_LEN = 500;

static Options *_options = nullptr;
static shared_ptr<CtgGraph> _graph(nullptr);

// used for building up scaffolds
struct Walk {
  int64_t len;
  double depth;
  vector<pair<cid_t, Orient> > vertices;
};


struct ScaffVertex {
  cid_t cid;
  Orient orient;
  int depth;
  int len;
};


struct Scaffold {
  int64_t id;
  string seq;
  vector<ScaffVertex> vertices;
  vector<int> gaps;
  double depth;
};


struct WalkStats {
  int64_t num_steps, dead_ends, term_visited, term_no_candidate, term_multi_candidates;
  
  void print() {
    int64_t tot_dead_ends = reduce_one(dead_ends, op_fast_add, 0).wait();
    int64_t tot_steps = reduce_one(num_steps, op_fast_add, 0).wait();

    int64_t tot_visited = reduce_one(term_visited, op_fast_add, 0).wait();
    int64_t tot_no_candidate = reduce_one(term_no_candidate, op_fast_add, 0).wait();
    int64_t tot_multi_candidates = reduce_one(term_multi_candidates, op_fast_add, 0).wait();
    int64_t tot_terms = tot_dead_ends + tot_visited + tot_no_candidate + tot_multi_candidates;

    cout << setprecision(2) << fixed;
    SOUT("Walks statistics:\n");
    SOUT("  total walk steps:           ", tot_steps, "\n");
    SOUT("  walk terminations:          ", perc_str(tot_terms, tot_steps), "\n");
    SOUT("    dead ends:                ", perc_str(tot_dead_ends, tot_terms), "\n");
    SOUT("    no viable candidates:     ", perc_str(tot_no_candidate, tot_terms), "\n");
    SOUT("    multiple candidates:      ", perc_str(tot_multi_candidates, tot_terms), "\n");
    SOUT("    already visited:          ", perc_str(tot_visited, tot_terms), "\n");
  }
};

  
struct GapStats {
  int64_t mismatched_splints, mismatched_spans, gaps, positive, unclosed, corrected_splints, corrected_spans;
  
  void print() {
    int64_t tot_mismatched_splints = reduce_one(mismatched_splints, op_fast_add, 0).wait();
    int64_t tot_mismatched_spans = reduce_one(mismatched_spans, op_fast_add, 0).wait();
    int64_t tot_gaps = reduce_one(gaps, op_fast_add, 0).wait();
    int64_t tot_positive = reduce_one(positive, op_fast_add, 0).wait();
    int64_t tot_unclosed = reduce_one(unclosed, op_fast_add, 0).wait();
    int64_t tot_corrected_splints = reduce_one(corrected_splints, op_fast_add, 0).wait();
    int64_t tot_corrected_spans = reduce_one(corrected_spans, op_fast_add, 0).wait();

    cout << setprecision(2) << fixed;
    SOUT("Gaps statistics:\n");
    SOUT("  total:                    ", tot_gaps, "\n");
    SOUT("  positive:                 ", perc_str(tot_positive, tot_gaps), "\n");
    SOUT("  mismatched splints:       ", perc_str(tot_mismatched_splints, tot_gaps), "\n");
    SOUT("  mismatched spans:         ", perc_str(tot_mismatched_spans, tot_gaps), "\n");
    SOUT("  unclosed:                 ", perc_str(tot_unclosed, tot_gaps), "\n");
    SOUT("  corrected splints:        ", perc_str(tot_corrected_splints, tot_gaps), "\n");
    SOUT("  corrected spans:          ", perc_str(tot_corrected_spans, tot_gaps), "\n");
  }
};

  
void print_assembly_stats(vector<Scaffold> &scaffs)
{
  int64_t tot_len = 0, max_len = 0;
  double tot_depth = 0;
  int64_t scaff_len_1kbp = 0, scaff_len_5kbp = 0, scaff_len_25kbp = 0, scaff_len_50kbp = 0;
  int64_t num_scaffs = 0;
  int64_t num_ns = 0;
  for (auto scaff : scaffs) {
    if (scaff.seq.size() < MIN_SCAFFOLD_LEN) continue;
    num_scaffs++;
    tot_len += scaff.seq.size();
    tot_depth += scaff.depth;
    max_len = max(max_len, static_cast<int64_t>(scaff.seq.size()));
    if (scaff.seq.size() >= 1000) scaff_len_1kbp += scaff.seq.size();
    if (scaff.seq.size() >= 5000) scaff_len_5kbp += scaff.seq.size();
    if (scaff.seq.size() >= 25000) scaff_len_25kbp += scaff.seq.size();
    if (scaff.seq.size() >= 50000) scaff_len_50kbp += scaff.seq.size();
    num_ns += count(scaff.seq.begin(), scaff.seq.end(), 'N');
  }
  barrier();
  int64_t num_scaffolds = reduce_one(num_scaffs, op_fast_add, 0).wait();
  int64_t all_tot_len = reduce_one(tot_len, op_fast_add, 0).wait();
  int64_t all_max_len = reduce_one(max_len, op_fast_max, 0).wait();
  int64_t all_scaff_len_1kbp = reduce_one(scaff_len_1kbp, op_fast_add, 0).wait();
  int64_t all_scaff_len_5kbp = reduce_one(scaff_len_5kbp, op_fast_add, 0).wait();
  int64_t all_scaff_len_25kbp = reduce_one(scaff_len_25kbp, op_fast_add, 0).wait();
  int64_t all_scaff_len_50kbp = reduce_one(scaff_len_50kbp, op_fast_add, 0).wait();
  double all_tot_depth = reduce_one(tot_depth, op_fast_add, 0).wait();
  int64_t all_num_ns = reduce_one(num_ns, op_fast_add, 0).wait();
  
  SOUT("Assembly statistics (ctg lens >= 500):\n");
  SOUT("    Number of scaffolds:     ", num_scaffolds, "\n");
  SOUT("    Total assembled length:  ", all_tot_len, "\n");
  SOUT("    Max. scaffold length:    ", all_max_len, "\n");
  SOUT("    Scaffold lengths:\n");
  SOUT("        > 1kbp:              ", perc_str(all_scaff_len_1kbp, all_tot_len), "\n");
  SOUT("        > 5kbp:              ", perc_str(all_scaff_len_5kbp, all_tot_len), "\n");
  SOUT("        > 25kbp:             ", perc_str(all_scaff_len_25kbp, all_tot_len), "\n");
  SOUT("        > 50kbp:             ", perc_str(all_scaff_len_50kbp, all_tot_len), "\n");
  SOUT("    Average scaffold depth:  ", all_tot_depth / num_scaffolds, "\n");
  SOUT("    Number of Ns/100kbp:     ", (double)all_num_ns * 100000.0 / all_tot_len, " (", all_num_ns, ")", KNORM, "\n");
}


void write_assembly(const vector<Scaffold> &scaffs, const string &fname, int max_clen, int min_clen)
{
  Timer timer(__func__);
  string tmpfname = fname + ".tmp"; // make a .tmp file and rename on success
  string fasta = "";
  for (auto &scaff : scaffs) {
    if (scaff.seq.size() > max_clen || scaff.seq.size() < min_clen) continue;
    fasta += ">Scaffold" + to_string(scaff.id) + "-" + to_string(scaff.seq.length()) + "\n";
    for (int64_t i = 0; i < scaff.seq.length(); i += 50) fasta += scaff.seq.substr(i, 50) + "\n";
  }
  auto sz = fasta.size();
  atomic_domain<size_t> ad({atomic_op::fetch_add, atomic_op::load});
  global_ptr<size_t> fpos = nullptr;
  if (!rank_me()) fpos = new_<size_t>(0);
  fpos = broadcast(fpos, 0).wait();
  size_t my_fpos = ad.fetch_add(fpos, sz, memory_order_relaxed).wait();
  // wait until all ranks have updated the global counter
  barrier();
  int bytes_written = 0;
  int fileno = -1;
  size_t fsize = 0;
  if (!rank_me()) {
    fsize = ad.load(fpos, memory_order_relaxed).wait();
    // rank 0 creates the file and truncates it to the correct length
    fileno = open(tmpfname.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH );
    if (fileno == -1) WARN("Error trying to create file ", tmpfname, ": ", strerror(errno), "\n");
    if (ftruncate(fileno, fsize) == -1) WARN("Could not truncate ", tmpfname, " to ", fsize, " bytes\n");
  }
  barrier();
  ad.destroy();
  // wait until rank 0 has finished setting up the file
  if (rank_me()) fileno = open(tmpfname.c_str(), O_WRONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
  if (fileno == -1) WARN("Error trying to open file ", tmpfname, ": ", strerror(errno), "\n");
  bytes_written = pwrite(fileno, fasta.c_str(), sz, my_fpos);
  close(fileno);

  if (bytes_written != sz)
    DIE("Could not write all ", sz, " bytes; only wrote ", bytes_written, "\n");
  barrier();
  if (rank_me() == 0) {
    if (rename(tmpfname.c_str(), fname.c_str()) != 0)
       SDIE("Could not rename ", tmpfname, " to ", fname);
    SOUT("Successfully wrote ", fsize, " bytes to ", fname, "\n");
  }
}


void write_assembly_per_rank(const vector<Scaffold> &scaffs, const string &fname)
{
  Timer timer(__func__);
  string fasta = "";
  string depths = "";
  for (auto &scaff : scaffs) {
    fasta += ">Contig_" + to_string(scaff.id) + "\n";
    fasta += scaff.seq + "\n";
    depths += "Contig" + to_string(scaff.id) + "\t" + to_string(scaff.seq.length()) + "\t" + to_string(scaff.depth) + "\n";
  }
  string rank_path_fname = string(_options->cached_io ? "/dev/shm/" : "") + fname + ".fasta.gz";
  get_rank_path(rank_path_fname, upcxx::rank_me());
  zstr::ofstream f(rank_path_fname);
  f.write(fasta.c_str(), fasta.size());

  string rank_path_depth_fname = string(_options->cached_io ? "/dev/shm/" : "") + "merDepth_" + fname + ".txt.gz";
  get_rank_path(rank_path_depth_fname, upcxx::rank_me());
  zstr::ofstream df(rank_path_depth_fname);
  df.write(depths.c_str(), depths.size());
}


// SRF format (all tab separated)
// Scaffold1  CONTIG1  +Contig626559  1  125  26.000000
// Scaffold1  GAP1  -611
// Scaffold1  CONTIG2  -Contig42483  65  247  35.000000
// Format of file:
// scaffold_id CONTIGi <strand>contig_id start end depth
// scaffold_id GAPi gap_size gap_uncertainty
void write_srf(vector<Scaffold> &scaffs, const string &fname)
{
  Timer timer(__func__);
  string rank_path_fname = string(_options->cached_io ? "/dev/shm/" : "") + fname;
  get_rank_path(rank_path_fname, upcxx::rank_me());
  zstr::ofstream f(rank_path_fname);
  for (auto &scaff : scaffs) {
    if (scaff.vertices.size() - 1 != scaff.gaps.size()) 
      DIE("srf error: #vertices - 1 != #gaps, #vertices ", scaff.vertices.size(), " #gaps ", scaff.gaps.size(), "\n");
    /*
    int64_t start = 0;
    for (int i = 0; i < scaff.vertices.size(); i++) {
      ScaffVertex *v = &scaff.vertices[i];
      f << "Scaffold" << scaff.id << "\tCONTIG" << i << "\t" << (v->orient == Orient::REVCOMP ? "-" : "+") << "Contig" << v->cid
        << "\t" << start << "\t" << (start + v->len) << "\t" << v->depth << endl;
      start += v->len;
      if (i < scaff.gaps.size()) {
        f << "Scaffold" << scaff.id << "\tGAP" << i << "\t" << scaff.gaps[i] << endl;
        start += scaff.gaps[i];
      }
    }
    */
    ScaffVertex *prev_v = nullptr;
    for (int i = 1; i < scaff.vertices.size(); i++) {
      ScaffVertex *v = &scaff.vertices[i];
      if (prev_v) {
        f << (prev_v->orient == Orient::REVCOMP ? "-" : "+") << prev_v->cid << ","
          << (v->orient == Orient::REVCOMP ? "-" : "+") << v->cid << endl;
      }
      prev_v = v;
    }
  }
}


vector<Scaffold> get_scaffolds(vector<Walk> &walks)
{
  Timer timer(__func__);

  int num_break_scaffs = 0;
  GapStats gap_stats = {0};
  vector<Scaffold> scaffs;
  for (auto &walk : walks) {
    shared_ptr<Vertex> prev_v(nullptr);
    Scaffold scaff = {0};
    scaff.depth = walk.depth;
    for (int i = 0; i < walk.vertices.size(); i++) {
      bool break_scaffold = false;
      auto v = _graph->get_vertex(walk.vertices[i].first);
      auto orient = walk.vertices[i].second;
      auto seq = _graph->get_vertex_seq(v->seq_gptr, v->clen);
      if (orient == Orient::REVCOMP) seq = revcomp(seq);
      ScaffVertex scaff_vertex = { .cid = v->cid, .orient = orient, .depth = (int)v->depth, .len = v->clen };
      if (!prev_v) {
        // no previous vertex - the start of the scaffold
        scaff.seq = seq;
        scaff.vertices.push_back(scaff_vertex);
      } else {
        gap_stats.gaps++;
        auto edge = _graph->get_edge(v->cid, prev_v->cid);      
        if (edge->gap > 0) {
          gap_stats.positive++;
          string gap_seq;
          auto gap_edge_seq = edge->seq;
          if (gap_edge_seq.size()) {
            // can only close when the gap fill sequence exists
            // check that gap filling seq matches properly, kmer_len - 1 on each side
            int len = _options->kmer_len - 1;
            // first check the left
            int dist = hamming_dist(tail(scaff.seq, len), head(gap_edge_seq, len));
            if (is_overlap_mismatch(dist, len)) {
              auto orig_gap_seq = gap_edge_seq;
              // norm is bad, the revcomp may be good
              gap_edge_seq = revcomp(gap_edge_seq);
              dist = hamming_dist(tail(scaff.seq, len), head(gap_edge_seq, len));
              if (is_overlap_mismatch(dist, len)) {
                DBG_WALK("breaking pos gap after revcomp, orig:\n", orig_gap_seq, "\n");
                // the revcomp is also bad, break the scaffold
                break_scaffold = true;
              }
            }
            if (!break_scaffold) {
              // now check the right
              dist = hamming_dist(tail(gap_edge_seq, len), head(seq, len));
              if (is_overlap_mismatch(dist, len)) {
                // for debugging, check non-revcomp
                auto rc_gap_edge_seq = revcomp(gap_edge_seq);
                DBG_WALK("dist to revcomp ", dist, " and to non-revcomp ", hamming_dist(tail(rc_gap_edge_seq, len), head(seq, len)),
                         ":\n  ", tail(rc_gap_edge_seq, len), "\n  ", tail(gap_edge_seq, len), "\n  ", head(seq, len), "\n");
                // the left was fine, but the right is not, so break
                break_scaffold = true;
              } else {
                gap_seq = gap_edge_seq.substr(len, edge->gap);
              }
              // now break if too many Ns
              if (_options->break_scaffolds > 0 && !break_scaffold && gap_seq.find(string(_options->break_scaffolds, 'N')) != string::npos)
                break_scaffold = true;
            }
          } else {  // gap sequence does not exist
            gap_stats.unclosed++;
            // fill with Ns - we could also break the scaffold here
            DBG_WALK("SPAN (", edge->cids.cid1, ", ", edge->cids.cid2, ") gap size ", edge->gap, "\n");
            //gap_seq = string(edge->gap, 'N');
            break_scaffold = true;
          }
          if (!break_scaffold) {
            scaff.gaps.push_back(gap_seq.length());
            scaff.seq += gap_seq + seq;
          }
        } else if (edge->gap < 0) {
          int gap_excess = -edge->gap;
          if (gap_excess > scaff.seq.size() || gap_excess > seq.size()) gap_excess = min(scaff.seq.size(), seq.size()) - 5;
          // now check min hamming distance between overlaps 
          auto min_dist = min_hamming_dist(scaff.seq, seq, _options->max_kmer_len - 1, gap_excess);
          if (is_overlap_mismatch(min_dist.first, min_dist.second)) {
            break_scaffold = true;
          } else {
            gap_excess = min_dist.second;
            if (gap_excess != -edge->gap) {
              if (edge->edge_type == EdgeType::SPLINT) gap_stats.corrected_splints++;
              else gap_stats.corrected_spans++;
              DBG_WALK("corrected neg ", edge_type_str(edge->edge_type),  " gap from ", edge->gap, " to ", -gap_excess, "\n",  
                       tail(scaff.seq, gap_excess), "\n", head(seq, gap_excess), "\n");
            }
            scaff.gaps.push_back(-gap_excess);
            scaff.seq += tail(seq, seq.size() - gap_excess);
          }
        } else {
          // gap is exactly 0
          scaff.gaps.push_back(0);
          scaff.seq += seq;
        }
        if (!break_scaffold) {
          scaff.vertices.push_back(scaff_vertex);
        } else {
          num_break_scaffs++;
          DBG_WALK("break scaffold from ", prev_v->cid, " to ", v->cid, " gap ", edge->gap, 
                   " type ", edge_type_str(edge->edge_type), " prev_v clen ", prev_v->clen, " curr_v clen ", v->clen, "\n");
          if (edge->edge_type == EdgeType::SPLINT) gap_stats.mismatched_splints++;
          else gap_stats.mismatched_spans++;
          gap_stats.gaps++;
          // save current scaffold
          scaffs.push_back(scaff);
          // start new scaffold
          scaff.depth = v->depth;
          scaff.seq = seq;
          scaff.vertices.clear();
          scaff.gaps.clear();
          scaff_vertex = { .cid = v->cid, .orient = orient, .depth = (int)v->depth, .len = v->clen };
          scaff.vertices.push_back(scaff_vertex);
        }
      }
      prev_v = v;
    }
    // done with all walk vertices
    if (scaff.seq != "") scaffs.push_back(scaff);
  }
  barrier();
  SOUT("Number of scaffold breaks ", reduce_one(num_break_scaffs, op_fast_add, 0).wait(), "\n");
  gap_stats.print();
  print_assembly_stats(scaffs);
  // now get unique ids for all the scaffolds
  size_t num_scaffs = scaffs.size();
  atomic_domain<size_t> ad({atomic_op::fetch_add, atomic_op::load});
  global_ptr<size_t> counter = nullptr;
  if (!rank_me()) counter = new_<size_t>(0);
  counter = broadcast(counter, 0).wait();
  size_t my_counter = ad.fetch_add(counter, num_scaffs, memory_order_relaxed).wait();
  // wait until all ranks have updated the global counter
  barrier();
  ad.destroy();
  for (auto &scaff : scaffs) scaff.id = my_counter++;

  //string rank_path_srf = string(_options->cached_io ? "/dev/shm/" : "") + "cgraph.srf";
  //get_rank_path(rank_path_srf, upcxx::rank_me());
  //zstr::ofstream srf(rank_path_srf);

  return scaffs;
}


bool depth_match(double depth, double walk_depth)
{
//TESTING
  const int MAX_DEPTH_DIFF = 15;
  const int MIN_DEPTH_DIFF = 5;
  double depth_diff = fabs(depth - walk_depth);
  double allowable_diff = _options->depth_diff_thres * walk_depth;
  if (allowable_diff > MAX_DEPTH_DIFF) allowable_diff = MAX_DEPTH_DIFF;
  if (allowable_diff < MIN_DEPTH_DIFF) allowable_diff = MIN_DEPTH_DIFF;
  return (depth_diff <= allowable_diff);
}


vector<shared_ptr<Vertex> > get_vertex_list(vector<cid_t> &cids) 
{
  vector<shared_ptr<Vertex> > vertices;
  for (auto &cid : cids) vertices.push_back(_graph->get_vertex_cached(cid));
  return vertices;
}


string vertex_list_to_cid_string(vector<shared_ptr<Vertex> > &vertices) 
{
  string s;
  for (auto &v : vertices) s += to_string(v->cid) + " ";
  return s;
}

cid_t bfs_branch(shared_ptr<Vertex> curr_v, int end, double walk_depth)
{
  const int MAX_SEARCH_LEVEL = 5;
  const int MAX_QUEUE_SIZE = 100;
  
  queue<pair<shared_ptr<Vertex>, int> > q;
  unordered_map<cid_t, bool> visited;

  vector<shared_ptr<Vertex> > frontier = {};
  
  q.push({curr_v, end});
  // nullptr is a level marker
  q.push({nullptr, 0});
  int search_level = 0;
  cid_t candidate = -1;
  while (!q.empty()) {
    progress();
    tie(curr_v, end) = q.front();
    q.pop();
    if (!curr_v) {
      if (q.empty()) return -1;
      search_level++;
      string offset(6 + search_level * 2, ' ');
      // break if the search level is too high, or if the queue size is too big
      if (search_level >= MAX_SEARCH_LEVEL) {
        DBG_WALK(offset, "Reached max search level ", MAX_SEARCH_LEVEL, ", stopping...\n");
        break;
      }
      if (q.size() >= MAX_QUEUE_SIZE) {
        DBG_WALK(offset, "Reached max queue size ", q.size(), " > ", MAX_QUEUE_SIZE, " stopping...\n");
        break;
      }
      q.push({nullptr, 0});
      DBG_WALK(offset, "* level ", search_level, "\n");
      continue;
    }
    string offset(6 + search_level * 2, ' ');
    auto nb_cids = (end == 5 ? curr_v->end5 : curr_v->end3);
    DBG_WALK(offset, curr_v->cid, " depth ", curr_v->depth, " length ", curr_v->clen, " num nbs ", nb_cids.size(), "\n");
    if (!depth_match(curr_v->depth, walk_depth) && curr_v->depth < walk_depth / 2) {
      DBG_WALK(offset, "-> vertex ", curr_v->cid, " depth is too low ", curr_v->depth, "\n");
      continue;
    }
    // terminate if visited before
    if (visited.find(curr_v->cid) != visited.end()) {
      DBG_WALK(offset, "-> vertex ", curr_v->cid, " is already visited\n");
      continue;
    }
    visited[curr_v->cid] = true;
    // terminate if a suitable vertex is found
    //if (depth_match(curr_v->depth, walk_depth) && curr_v->clen > 200) {
    if (depth_match(curr_v->depth, walk_depth)) {
      DBG_WALK(offset, "-> found candidate vertex ", curr_v->cid, "\n");
      frontier.push_back(curr_v);
      candidate = curr_v->cid;
      // short circuit
      break;
    }
    DBG_WALK(offset, "adding ", nb_cids.size(), " vertices to the queue\n");
    for (auto &nb_cid : nb_cids) {
      auto nb = _graph->get_vertex_cached(nb_cid);
      auto edge = _graph->get_edge_cached(curr_v->cid, nb->cid);
      DBG_WALK(offset, "added ", nb->cid, " to the queue\n");
      auto nb_end = (_graph->get_other_end(curr_v, nb, edge) == 3 ? 5 : 3);
      q.push({nb, nb_end});
    }
  }
  if (!frontier.empty() || !q.empty()) {
    while (!q.empty()) {
      auto elem = q.front();
      q.pop();
      if (elem.first) frontier.push_back(elem.first);
    }
    DBG_WALK("      frontier consists of: ");
    for (auto v : frontier) DBG_WALK(v->cid, " ");
    DBG_WALK("\n");
  }
  return candidate;
}


vector<shared_ptr<Vertex> > search_for_next_nbs(shared_ptr<Vertex> curr_v, int end, double walk_depth, WalkStats &stats,
                                                cid_t fwd_cid=-1)
{
  auto in_cid_list = [](vector<cid_t> cids, cid_t query_cid) -> bool {
    for (auto cid : cids) if (cid == query_cid) return true;
    return false;
  };

  stats.num_steps++;
  // get the nbs from the correct end
  auto nbs_cids = (end == 5 ? curr_v->end5_merged : curr_v->end3_merged);
  DBG_WALK("curr_v ", curr_v->cid, " depth ", curr_v->depth, " length ", curr_v->clen, " nbs ", nbs_cids.size(), 
           " walk_depth ", walk_depth);
  DBG_WALK("\n");
  if (nbs_cids.empty()) {
    stats.dead_ends++;
    DBG_WALK("    -> terminate: dead end\n");
    return {};
  }
  vector<shared_ptr<Vertex> > nb_vertices;
  for (auto nb_cids : nbs_cids) nb_vertices.push_back(_graph->get_vertex_cached(nb_cids.back()));
  vector<shared_ptr<Edge> > nb_edges;
  for (auto nb_cids : nbs_cids) nb_edges.push_back(_graph->get_edge_cached(curr_v->cid, nb_cids.back()));

  vector<pair<int, int> > candidate_branches;
  unordered_map<cid_t, int> candidates;
  bool bulge = false;
  // candidate first search from each of the neighbors (branches)
  for (int i = 0; i < nb_vertices.size(); i++) {
    auto nb = nb_vertices[i];
    auto edge = nb_edges[i];
    cid_t candidate = -1;
    if (fwd_cid == nb->cid) {
      candidate = fwd_cid;
      DBG_WALK("      ", i, ". branch ", nb->cid, " edge support ", edge->support, ", fwd cid, accept candidate\n");
    } else {
      auto nb_end = _graph->get_other_end(curr_v, nb, edge);
      DBG_WALK("      ", i, ". branch ", nb->cid, " edge support ", edge->support, ", searching...\n");
      candidate = bfs_branch(nb, nb_end == 5 ? 3 : 5, walk_depth);
    }
    progress();
    if (candidate != -1) {
      auto it = candidates.find(candidate);
      if (it != candidates.end()) {
        bulge = true;
        DBG_WALK("      -> ", nb->cid, " (", i, ") is a bulge with branch ", it->second, "\n");
      }
      candidates[candidate] = i;
      candidate_branches.push_back({i, nb->clen});
      DBG_WALK("      -> ", nb->cid, " (", i, ") found candidate ", candidate, "\n");
    }
  }

  int branch_chosen = -1;
  
  if (candidate_branches.size() == 1) {
    branch_chosen = candidate_branches[0].first;
    DBG_WALK("      -> viable candidate found on branch ", candidate_branches[0].first, "\n");
  } else if (candidate_branches.size() > 1) {
    stats.term_multi_candidates++;
    DBG_WALK("      -> found ", candidate_branches.size(), " viable candidates\n");
    if (_options->quality_tuning != QualityTuning::MIN_ERROR || walk_depth <= 10) {
      if (_options->max_kmer_len > _options->kmer_len) {
        // if one branch has much better aln len than the others, choose it
        int num_max_kmer_alns = 0;
        vector<pair<int, int> > nbs_aln_lens;
        for (auto candidate : candidate_branches) {
          auto edge = nb_edges[candidate.first];
          if (edge->aln_len >= _options->max_kmer_len) {
            num_max_kmer_alns++;
            if (num_max_kmer_alns > 1) break;
          }
          nbs_aln_lens.push_back({edge->aln_len, candidate.first});
        }
        if (num_max_kmer_alns < 2) {
          sort(nbs_aln_lens.begin(), nbs_aln_lens.end(),
               [](auto &a, auto &b) {
                 return a.first > b.first;
               });
          DBG_WALK("    -> best aln len branch is ", nbs_aln_lens[0].second, " (", nbs_aln_lens[0].first, 
                   "), next best is ", nbs_aln_lens[1].second, " (", nbs_aln_lens[1].first, ")\n");
          if (num_max_kmer_alns == 1) {
            branch_chosen = nbs_aln_lens[0].second;
            DBG_WALK("    -> resolve only max aln len ", branch_chosen, "\n");
          } else {
            if (nbs_aln_lens[0].first >= 2 * nbs_aln_lens[1].first) {
              branch_chosen = nbs_aln_lens[0].second;
              DBG_WALK("    -> resolve best aln len ", branch_chosen, "\n");
            }
          }
        }
      }
      if (branch_chosen == -1) {
        // if one branch has much higher edge support than the others, choose it
        double support_thres = 3;

        if (fwd_cid != -1 && _options->quality_tuning == QualityTuning::MAX_CONTIGUITY) support_thres = 1.5;

        /*
        if (fwd_cid != -1) {
          if (walk_depth <= 10) support_thres = 2.0;
          else if (walk_depth <= 5) support_thres = 1.5;
        }
        */
        vector<pair<int, int> > nbs_support;
        for (auto candidate : candidate_branches) {
          auto edge = nb_edges[candidate.first];
          nbs_support.push_back({edge->support, candidate.first});
        }
        sort(nbs_support.begin(), nbs_support.end(),
             [](auto &a, auto &b) {
               return a.first > b.first;
             });
        DBG_WALK("    -> most supported branch is ", nbs_support[0].second, " (", nbs_support[0].first, 
                 "), next best is ", nbs_support[1].second, " (", nbs_support[1].first, ")\n");
        if (nbs_support[0].first >= support_thres * nbs_support[1].first && nbs_support[0].first > 2) {
          branch_chosen = nbs_support[0].second;
          DBG_WALK("    -> resolve most supported ", branch_chosen, "\n");
        }
      }
//TESTING
      /*
      if (branch_chosen == -1 && walk_depth <= 5) {
        // if one candidate has much greater length, choose it
        sort(candidate_branches.begin(), candidate_branches.end(),
             [](auto &a, auto &b) {
               return a.second > b.second;
             });
        DBG_WALK("    -> longest branch is ", candidate_branches[0].first, " (", candidate_branches[0].second,
                 "), next best is ", candidate_branches[1].first, " (", candidate_branches[1].second, ")\n");
        if (candidate_branches[0].second > 300 && candidate_branches[1].second <= 300 &&
            candidate_branches[0].second >= 2.0 * candidate_branches[1].second) {
          branch_chosen = candidate_branches[0].first;
          DBG_WALK("    -> resolve longest ", branch_chosen, "\n");
        }
      }
      */
    }
  }
  
  vector<shared_ptr<Vertex> > next_nbs = {};
  if (branch_chosen != -1) {
    next_nbs = get_vertex_list(nbs_cids[branch_chosen]);
    // make sure that this list contains the fwd cid, if it is specified
    if (fwd_cid != -1) {
      bool found = false;
      for (auto &next_nb : next_nbs) {
        // check to see if the merged list for the back candidate contains the current vertex
        if (next_nb->cid == fwd_cid) {
          found = true;
          break;
        }
      }
      if (!found) {
        DBG_WALK("    -> chosen bwd path does not include fwd vertex ", fwd_cid, "\n");
        next_nbs = {};
      }
    }
  }
  if (next_nbs.empty()) stats.term_no_candidate++;
  else DBG_WALK("    -> resolved: ", vertex_list_to_cid_string(next_nbs), "\n");
  return next_nbs;
}


vector<Walk> do_walks(vector<pair<cid_t, int32_t> > &sorted_ctgs, WalkStats &walk_stats, IntermittentTimer &next_nbs_timer)
{
  auto has_depth_remaining = [](unordered_map<cid_t, double> &depth_remaining, shared_ptr<Vertex> v, double walk_depth) {
    if (v->visited) return false;
    auto it = depth_remaining.find(v->cid);
    if (it == depth_remaining.end()) return true;
    if (it->second < v->depth && it->second < walk_depth) return false;
    return true;
  };
  
  auto get_start_vertex = [&](vector<pair<cid_t, int32_t> > &sorted_ctgs, int64_t *ctg_pos, 
                              unordered_map<cid_t, double> &depth_remaining) -> shared_ptr<Vertex> {
    while (*ctg_pos < sorted_ctgs.size()) {
      auto v = _graph->get_local_vertex(sorted_ctgs[*ctg_pos].first);
      (*ctg_pos)++;
      // don't use if used in a walk in a previous round (possibly by another rank's walk)
      if (v->visited) continue;
      // don't use if already in walk in this round by local rank
      auto it = depth_remaining.find(v->cid);
      if (it != depth_remaining.end() && it->second < v->depth) continue;
      return v;
    }
    return nullptr;
  };

  // keep track of vertex depths used up by walks so we don't visit them too many times
  unordered_map<cid_t, double> depth_remaining;
  int64_t sum_scaff_lens = 0;
  int64_t ctg_pos = 0;
  shared_ptr<Vertex> start_v;
  // temporarily store the scaffolds from this rank - some may get discarded due to conflicts
  vector<Walk> tmp_walks;
  _graph->clear_caches();
  // each rank does an independent set of walks over the graph until it has run out of start vertices
  while ((start_v = get_start_vertex(sorted_ctgs, &ctg_pos, depth_remaining)) != nullptr) {
    DBG_WALK("start ", start_v->cid,  " len ",  start_v->clen,  " depth ",  start_v->depth,  "\n");
    // store the walk in a double ended queue because we could start walking in the middle and so need to add to
    // either the front or back
    deque<pair<shared_ptr<Vertex>, Orient> > walk_vertices;
    // start walk backwards, going from 5 to 3 ends
    Dirn dirn = Dirn::BACKWARD;
    int end = 5;
    Orient orient = Orient::NORMAL;

    double walk_depth = start_v->depth;
    depth_remaining[start_v->cid] = 0;
                
    walk_vertices.push_front({start_v, orient});
    auto curr_v = start_v;
    int64_t scaff_len = curr_v->clen;
    // do walk
    while (curr_v) {
      DBG_WALK("    search fwd: ");
      next_nbs_timer.start();
      auto next_nbs = search_for_next_nbs(curr_v, end, walk_depth, walk_stats);
      next_nbs_timer.stop();
      bool already_visited = false;
      if (!next_nbs.empty()) {
        // we have possibly multiple next nbs in sequence
        // update the last one and reject if it has insufficient depth remaining
        if (!has_depth_remaining(depth_remaining, next_nbs.back(), 0.9 * walk_depth)) {
          walk_stats.term_visited++;
          DBG_WALK("    -> terminate: ", next_nbs.back()->cid, " is already visited\n");
          curr_v = nullptr;
        }
      } else {
        curr_v = nullptr;
      }
      if (curr_v) {
        bool join_resolved = false;
        auto next_nb = next_nbs.back();
        // resolve joins (backward forks) from next_nb
        int next_nb_end = _graph->get_other_end(curr_v, next_nb);
        if ((next_nb_end == 5 && next_nb->end5_merged.size() > 1) || (next_nb_end == 3 && next_nb->end3_merged.size() > 1)) {
          DBG_WALK("    search bwd: ");
          next_nbs_timer.start();
          auto back_nbs = search_for_next_nbs(next_nb, next_nb_end, walk_depth, walk_stats, curr_v->cid);
          next_nbs_timer.stop();
          if (!back_nbs.empty()) join_resolved = true;
        } else {
          DBG_WALK("    accept single bwd path\n");
          join_resolved = true;
        }
        if (join_resolved) {
          if (depth_match(next_nbs.back()->depth, walk_depth)) walk_depth = next_nbs.back()->depth;
          // add all vertices in merged path to the walk
          for (auto next_nb : next_nbs) {
            // update depth remaining for the newly added vertex
            auto it = depth_remaining.find(next_nb->cid);
            if (it == depth_remaining.end()) depth_remaining[next_nb->cid] = next_nb->depth;
            depth_remaining[next_nb->cid] -= walk_depth;
            //depth_remaining[next_nb->cid] = 0;
            auto edge = _graph->get_edge_cached(curr_v->cid, next_nb->cid);
            if (!edge) DIE("edge not found\n");
            scaff_len += next_nb->clen + edge->gap;
            auto next_nb_end = _graph->get_other_end(curr_v, next_nb, edge);
            curr_v = next_nb;
            // if the ends are the same, change the orientation of the next vertex
            if (end == next_nb_end) orient = flip_orient(orient);
            end = (next_nb_end == 3 ? 5 : 3);
            if (dirn == Dirn::BACKWARD) walk_vertices.push_front({curr_v, orient});
            else walk_vertices.push_back({curr_v, orient});
          }
        } else {
          DBG_WALK("    -> terminate: join not resolved\n");
          curr_v = nullptr;
        }
      }      
      if (!curr_v && dirn == Dirn::BACKWARD) {
        // backward walk terminated, change walk direction
        dirn = Dirn::FORWARD;
        end = 3;
        curr_v = start_v;
        walk_depth = curr_v->depth;
        orient = Orient::NORMAL;
        DBG_WALK("  switch to dirn FORWARD\n");
      }
    } // walk loop
    sum_scaff_lens += scaff_len;
    // unique ids are generated later
    Walk walk = { .len = scaff_len, .depth = walk_depth, .vertices = {} };
    for (auto &w : walk_vertices) walk.vertices.push_back({w.first->cid, w.second});
    for (auto &w : walk.vertices) depth_remaining[w.first] = 0;
    tmp_walks.push_back(walk);
  }
  return tmp_walks;
}


void walk_graph(Options *options, shared_ptr<CtgGraph> graph)
{
  auto sort_ctgs = []() -> vector<pair<cid_t, int32_t> > {
    Timer timer("sort_ctgs");
    vector<pair<cid_t, int32_t> > sorted_ctgs;
    for (auto v = _graph->get_first_local_vertex(); v != nullptr; v = _graph->get_next_local_vertex()) {
      // don't start on a contig that has already been used
      if (v->visited) continue;
      // don't start walks on vertices without depth information - this can happen in alignment-based depth calcs
      if (v->depth == 0) continue;
      // don't start walks on short contigs
      if (v->clen < _options->min_ctg_len) continue;
      // don't start walks on very low depth
      //if (v->depth < 3 && v->clen < 1000) continue;
      // don't start walks on a high depth contig, which is a contig that has at least 2x depth higher than its nb average
      // and doesn't have any nbs of higher depth
      if (v->depth > 10) {
        auto all_nb_cids = v->end5;
        all_nb_cids.insert(all_nb_cids.end(), v->end3.begin(), v->end3.end());
        double sum_nb_depths = 0;
        bool found_higher_depth = false;
        for (auto nb_cid : all_nb_cids) {
          auto nb = _graph->get_vertex_cached(nb_cid);
          if (nb->depth > v->depth) {
            found_higher_depth = true;
            break;
          }
          sum_nb_depths += nb->depth;
        }
        if (!found_higher_depth) {
          if (v->depth > 2.0 * sum_nb_depths / all_nb_cids.size()) continue;
        }
      }
      sorted_ctgs.push_back({v->cid, v->clen});
    }
    sort(sorted_ctgs.begin(), sorted_ctgs.end(),
         [](auto &a, auto &b) {
           return a.second > b.second;
         });
    return sorted_ctgs;
  };

  _options = options;
  _graph = graph;
  
  // The general approach is to have each rank do walks starting from its local vertices only.
  // First, to prevent loops within a walk, the vertices visited locally are kept track of using a 'depth_remaining' hash 
  // table, which allows for repeat use of vertices of high depth, e.g. if depth is 2x walk depth, then the 
  // vertex could be used twice.
  // Once walks starting from all local vertices have been completed, any conflicts (overlaps) between walks from different 
  // ranks are resolved. These are resolved in favor of the longest walks (and for ties, the highest numbered rank). The walks 
  // that lose are discarded, and the whole process is repeated, since there are potentially left-over vertices freed 
  // up when walks are dropped. 
  // The vertices in winning walks are marked as visited.
  // This is repeated until there are no more walks found.

  Timer timer(__func__);
  vector<Walk> walks;
  WalkStats walk_stats = {0};
  int64_t num_rounds = 0;
  auto sorted_ctgs = sort_ctgs();
  barrier();
  int64_t num_start_ctgs = reduce_one(sorted_ctgs.size(), op_fast_add, 0).wait();
  SOUT("Number of starting contigs: ", perc_str(num_start_ctgs, _graph->get_num_vertices()), "\n");
  // need to repeat the sets of walks because there may be a conflicts between walks across ranks, which results
  // in one of the walks being dropped. So this loop will repeat until no more scaffolds can be built.
  {
    IntermittentTimer next_nbs_timer("next_nbs");
    IntermittentTimer walks_timer("walks");
    while (true) {
      walks_timer.start();
      ProgressBar progbar(sorted_ctgs.size(), "Walking graph round " + to_string(num_rounds));
      auto tmp_walks = do_walks(sorted_ctgs, walk_stats, next_nbs_timer);
      progbar.done();
      walks_timer.stop();
      barrier();
      // now eliminate duplicate walks. Each vertex will get labeled with the rank that has the longest walk, 
      // first set all the vertex fields to empty
      for (auto v = _graph->get_first_local_vertex(); v != nullptr; v = _graph->get_next_local_vertex()) {
        v->walk_rank = -1;
        v->walk_len = 0;
      }
      barrier();
      for (auto &walk : tmp_walks) {
        // resolve conflict in favor of longest walk - this marks the walk the vertex belongs to 
        for (auto &w : walk.vertices) _graph->update_vertex_walk(w.first, walk.len);
      }
      barrier();
      int num_walks_added = 0;
      // now drop all walks where any vertex's rank does not match this rank
      for (auto walk : tmp_walks) {
        bool add_walk = true;
        for (auto &w : walk.vertices) {
          auto v = _graph->get_vertex(w.first);
          if (v->walk_rank != rank_me()) {
            add_walk = false;
            break;
          }
        }
        if (add_walk) {
          num_walks_added++;
          // update depth remaining
          for (auto &w : walk.vertices) _graph->set_vertex_visited(w.first);
          walks.push_back(walk);
        }
      }
      barrier();
      auto tot_walks_added = reduce_all(num_walks_added, op_fast_add).wait();
      if (tot_walks_added == 0) break;
      //SOUT("Walk round ", num_rounds, " found ", tot_walks_added, " new walks\n");
      num_rounds++;
      if (num_rounds > rank_n() * 5) {
        SWARN("breaking on high count\n");
        break;
      }
    } // loop until no more walks are found
  }
  barrier();
  // now add any unvisited to the walks
  int64_t num_unvisited = 0;
  int64_t max_unvisited_len = 0;
  int64_t unvisited_len = 0;
  for (auto v = _graph->get_first_local_vertex(); v != nullptr; v = _graph->get_next_local_vertex()) {
    if (!v->visited) {
      num_unvisited++;
      if (v->clen > max_unvisited_len) max_unvisited_len = v->clen;
      unvisited_len += v->clen;
      Walk walk = { .len = v->clen, .depth = v->depth, .vertices = { {v->cid, Orient::NORMAL} } };
      walks.push_back(walk);
      //DBG_WALK("unvisited ", v->cid, "\n");
    }
  }
  barrier();
  auto tot_unvisited = reduce_all(num_unvisited, op_fast_add).wait();
  auto tot_max_unvisited_len = reduce_all(max_unvisited_len, op_fast_max).wait();
  auto tot_unvisited_len = reduce_all(unvisited_len, op_fast_add).wait();
  if (tot_unvisited) 
    SOUT("Didn't visit ", tot_unvisited, " vertices, max len ", tot_max_unvisited_len, " total length ", 
         tot_unvisited_len, "\n");
  walk_stats.print();

  auto scaffs = get_scaffolds(walks);
  //write_srf(scaffs, "cgraph-" + to_string(_options->kmer_len) + ".srf.gz");
  if (_options->out_dirname != "") {
    write_assembly(scaffs, _options->out_dirname + "/final_assembly.fa", INT_MAX, 200);
    write_assembly(scaffs, _options->out_dirname + "/final_assembly-200.fa", 200, 0);
  } else {
    string contigs_fname = "cgraph-" + to_string(_options->max_kmer_len);
    write_assembly_per_rank(scaffs, contigs_fname);
    int64_t num_scaffolds = reduce_one(scaffs.size(), op_fast_add, 0).wait();
    if (!rank_me()) {
      ofstream f("per_thread/n" + contigs_fname + ".txt");
      f << num_scaffolds << endl;
      f.close();
    }
  }
  barrier();
}



