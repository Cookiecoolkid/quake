//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef COMMON_H
#define COMMON_H

#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>
#include <mutex>
#include <atomic>
#include <utility>
#include <unordered_map>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <faiss/MetricType.h>
#include <filesystem>
#include <unordered_set>
#include <sstream>
#include <thread>
#include <pthread.h>
#include <ctime>
#include <omp.h>

#ifdef QUAKE_USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

using torch::Tensor;
using std::vector;
using std::unordered_map;
using std::shared_ptr;
using std::tuple;
using std::make_shared;
using std::size_t;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using faiss::idx_t;
using faiss::MetricType;

struct _EnsureSingleOmp {
    _EnsureSingleOmp() {
        // Disable OpenMP’s dynamic adjustment and nested teams:
        omp_set_dynamic(0);
        omp_set_max_active_levels(0);
        // Force exactly one thread:
        omp_set_num_threads(1);
    }
};

static _EnsureSingleOmp _ensure_single_omp;

// constants
static const uint32_t SerializationMagicNumber = 0x44494E4C;
static const uint32_t SerializationVersion = 3;

// Default constants for index build parameters
constexpr int DEFAULT_NLIST = 0;                   ///< Default number of clusters (lists); if not specified, a flat index is assumed.
constexpr int DEFAULT_NITER = 5;                   ///< Default number of k-means iterations used during clustering.
constexpr const char* DEFAULT_METRIC = "l2";       ///< Default distance metric (either "l2" for Euclidean or "ip" for inner product).
constexpr int DEFAULT_NUM_WORKERS = 0;             ///< Default number of workers (0 means single-threaded).
constexpr int DEFAULT_NUM_MERGE_WORKERS = 1;      ///< Default number of merge workers (for worker_scan)
constexpr int DEFAULT_GPU_BATCH_SIZE = 100000;             ///< Default batch size for GPU index building.
constexpr int DEFAULT_GPU_SAMPLE_SIZE = 1000000;           ///< Default sample size for GPU index building.

// Default constants for search parameters
constexpr int DEFAULT_K = 1;                             ///< Default number of neighbors to return.
constexpr int DEFAULT_NPROBE = 1;                        ///< Default number of partitions to probe during search.
constexpr float DEFAULT_RECALL_TARGET = -1.0f;           ///< Default recall target (a negative value means no adaptive search).
constexpr bool DEFAULT_BATCHED_SCAN = false;             ///< Default flag for batched scanning.
constexpr bool DEFAULT_PRECOMPUTED = true;               ///< Default flag to use precomputed incomplete beta fn for APS.
constexpr float DEFAULT_INITIAL_SEARCH_FRACTION = 0.1f; ///< Default initial fraction of partitions to search.
constexpr float DEFAULT_RECOMPUTE_THRESHOLD = 0.001f;    ///< Default threshold to trigger recomputation of search parameters.
constexpr int DEFAULT_APS_FLUSH_PERIOD_US = 5;         ///< Default period (in microseconds) for flushing the APS buffer.
constexpr int MAX_SUBBATCH = 128;
constexpr int MIN_BATCH_SCAN_SIZE = 4; ///< Minimum batch size for scanning partitions.
constexpr int BLAS_DB_BS = 256;
constexpr int DEFAULT_BLAS_Q_BS = 256;

// Default constants for maintenance policy parameters
constexpr const char* DEFAULT_MAINTENANCE_POLICY = "query_cost"; ///< Default maintenance policy type.
constexpr int DEFAULT_WINDOW_SIZE = 1000;              ///< Default window size for measuring hit rates.
constexpr int DEFAULT_REFINEMENT_RADIUS = 25;         ///< Default radius for local partition refinement.
constexpr int DEFAULT_REFINEMENT_ITERATIONS = 3;       ///< Default number of iterations for refinement.
constexpr int DEFAULT_MIN_PARTITION_SIZE = 32;         ///< Default minimum allowed partition size.
constexpr float DEFAULT_ALPHA = 0.9f;                  ///< Default alpha parameter for maintenance.
constexpr bool DEFAULT_ENABLE_SPLIT_REJECTION = true;  ///< Default flag to enable rejection of splits.
constexpr bool DEFAULT_ENABLE_DELETE_REJECTION = true; ///< Default flag to enable rejection of deletions.
constexpr float DEFAULT_DELETE_THRESHOLD_NS = 100.0f;   ///< Default threshold in nanoseconds for deletion decisions.
constexpr float DEFAULT_SPLIT_THRESHOLD_NS = 100.0f;    ///< Default threshold in nanoseconds for split decisions.
constexpr float DEFAULT_PARTITION_REDUCTION_THRESHOLD = 0.3;
constexpr float DEFAULT_CHURN_RECLUSTER_THRESHOLD = 0.4;
constexpr bool DEFAULT_ENABLE_CXL_COST_MODEL = false;
constexpr int DEFAULT_CXL_NUM_MCS = 4;
constexpr int DEFAULT_CXL_LINE_BYTES = 64;
constexpr int DEFAULT_CXL_ENTRY_BYTES = 517;
constexpr int DEFAULT_CXL_METADATA_BYTES = 8;
constexpr float DEFAULT_CXL_MC_BW_BYTES_PER_NS = 16.0f;
constexpr float DEFAULT_CXL_LINK_BW_BYTES_PER_NS = 32.0f;
constexpr float DEFAULT_CXL_MAINTENANCE_BANDWIDTH_FRACTION = 0.10f;
constexpr float DEFAULT_CXL_SPLIT_SCORE_WEIGHT = 1.0f;
constexpr float DEFAULT_CXL_MAINTENANCE_PENALTY_WEIGHT = 1.0f;
constexpr float DEFAULT_CXL_FANOUT_PENALTY_NS = 1024.0f;
constexpr float DEFAULT_CXL_FANOUT_PENALTY_WEIGHT = 0.0f;
constexpr bool DEFAULT_CXL_WORKLOAD_ADAPTIVE = false;
constexpr float DEFAULT_CXL_ADAPTIVE_EWMA_ALPHA = 0.2f;
constexpr float DEFAULT_CXL_ADAPTIVE_GROWTH_SPLIT_GAIN = 1.5f;
constexpr float DEFAULT_CXL_ADAPTIVE_CHURN_SPLIT_GAIN = 1.0f;
constexpr float DEFAULT_CXL_ADAPTIVE_SCAN_SPLIT_GAIN = 0.5f;
constexpr float DEFAULT_CXL_ADAPTIVE_MAINTENANCE_PENALTY_GAIN = 1.0f;
constexpr float DEFAULT_CXL_ADAPTIVE_DELETE_RELIEF_GAIN = 1.0f;
constexpr float DEFAULT_CXL_ADAPTIVE_FANOUT_GROWTH_GAIN = 1.0f;
constexpr bool DEFAULT_CXL_WORKLOAD_ADAPTIVE_V2 = false;
constexpr float DEFAULT_CXL_ADAPTIVE_ROI_THRESHOLD = 1.0f;
constexpr float DEFAULT_CXL_ADAPTIVE_MIN_GAIN_NS = 50.0f;
constexpr float DEFAULT_CXL_ADAPTIVE_ACTION_OVERHEAD_NS = 100000.0f;
constexpr float DEFAULT_CXL_ADAPTIVE_BASE_SPLIT_BUDGET_FRACTION = 0.025f;
constexpr float DEFAULT_CXL_ADAPTIVE_MAX_SPLIT_BUDGET_FRACTION = 0.10f;
constexpr int DEFAULT_CXL_ADAPTIVE_MIN_SPLIT_BUDGET = 8;
constexpr int DEFAULT_CXL_ADAPTIVE_MAX_SPLIT_BUDGET = 512;
constexpr int DEFAULT_CXL_ADAPTIVE_TARGET_PARTITION_SIZE = 0;
constexpr float DEFAULT_CXL_ADAPTIVE_GROWTH_DEBT_REPAY_FRACTION = 0.15f;
constexpr float DEFAULT_CXL_ADAPTIVE_STRUCTURAL_SIZE_RATIO = 1.10f;
constexpr float DEFAULT_CXL_ADAPTIVE_PARTITION_COUNT_SLACK = 0.05f;
constexpr int DEFAULT_CXL_ADAPTIVE_MAX_REASSIGN_BUDGET = 128;
constexpr bool DEFAULT_CXL_ADAPTIVE_OBSERVED_COST_ENABLED = true;
constexpr float DEFAULT_CXL_ADAPTIVE_OBSERVED_COST_ALPHA = 0.4f;
constexpr float DEFAULT_CXL_ADAPTIVE_MAINTENANCE_TIME_BUDGET_FRACTION = 0.10f;
constexpr int DEFAULT_CXL_ADAPTIVE_WARMUP_SPLIT_BUDGET = 16;
constexpr const char* DEFAULT_CXL_ADAPTIVE_REFINEMENT_MODE = "full";
constexpr int DEFAULT_CXL_ADAPTIVE_MAX_REFINE_SPLITS = 16;
constexpr float DEFAULT_CXL_ADAPTIVE_MAX_PAYBACK_WINDOWS = 8.0f;
constexpr bool DEFAULT_CXL_ADAPTIVE_APS_FEEDBACK_ENABLED = false;
constexpr bool DEFAULT_CXL_ADAPTIVE_REASSIGN_TIME_BUDGET_ENABLED = false;
constexpr int DEFAULT_CXL_ADAPTIVE_WARMUP_REASSIGN_BUDGET = 16;
constexpr bool DEFAULT_CXL_STREAMING_RENT_BUY = false;
constexpr bool DEFAULT_CXL_STREAMING_STAGED = false;
constexpr bool DEFAULT_CXL_RESOURCE_RENT_BUY = false;
constexpr bool DEFAULT_CXL_SEARCH_FIRST = false;
constexpr float DEFAULT_CXL_SEARCH_FIRST_GAIN_TARGET = 0.90f;
constexpr int DEFAULT_CXL_SEARCH_FIRST_MAX_COHORT = 24;
constexpr bool DEFAULT_CXL_DEFAULT_PLUS = false;
constexpr bool DEFAULT_CXL_SEARCH_GUARDED_PLUS = false;

const vector<int> DEFAULT_LATENCY_ESTIMATOR_RANGE_N = {1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536};   ///< Default range of n values for latency estimator.
const vector<int> DEFAULT_LATENCY_ESTIMATOR_RANGE_K = {1, 4, 16, 64, 256};                                ///< Default range of k values for latency estimator.
constexpr int DEFAULT_LATENCY_ESTIMATOR_NTRIALS = 5;                                                          ///< Default number of trials for latency estimator.

// macros
#define DEBUG_PRINT(x) std::cout << #x << " = " << x << std::endl;

struct CxlResourcePrice {
    string resource_id;
    float byte_price_ns = 0.0f;
    float op_price_ns = 0.0f;
    float utilization = 0.0f;
    // Prior-window byte demand lets online admission project an added logical
    // split without observing a future query window.
    int64_t demand_bytes = 0;
    // Explicit capacity is required for cold resources whose prior-window
    // demand and utilization are both zero.
    int64_t capacity_bytes = 0;
};

struct CxlResourcePriceSnapshot {
    int64_t window_id = -1;
    int64_t window_duration_ns = 0;
    vector<CxlResourcePrice> resources;
    vector<float> home_read_byte_price_ns;
    vector<float> home_read_op_price_ns;
    // Map each MC home to its enclosing device-link identifier.
    vector<int> home_device_ids;
    unordered_map<int64_t, int> list_home_ids;
    float maintenance_read_byte_price_ns = 0.0f;
    float maintenance_write_byte_price_ns = 0.0f;
    float routing_metadata_shadow_price_ns_per_byte = 0.0f;
    float dram_shadow_price_ns_per_byte = 0.0f;
    bool valid = false;
};

struct CxlPolicyDecision {
    string action_kind;
    int64_t partition_id = -1;
    int home_id = -1;
    int64_t records = 0;
    float cost_before_ns = 0.0f;
    float cost_after_ns = 0.0f;
    float rent_ns = 0.0f;
    float buy_ns = 0.0f;
    float credit_ns = 0.0f;
    float rent_buy_ratio = 0.0f;
    float native_rent_ns = 0.0f;
    float resource_rent_ns = 0.0f;
    bool native_legal = false;
    bool cxl_search_profitable = false;
    float read_line_bytes = 0.0f;
    float write_line_bytes = 0.0f;
    int64_t cohort_id = -1;
    int64_t cohort_size = 0;
    float cohort_credit_ns = 0.0f;
    float cohort_variable_buy_ns = 0.0f;
    float cohort_shared_buy_ns = 0.0f;
    float cohort_buy_ns = 0.0f;
    float cohort_rent_buy_ratio = 0.0f;
    float cohort_search_gain_ns = 0.0f;
    float cohort_search_gain_fraction = 0.0f;
    bool selected = false;
    string rejection_reason;
};

// A data-free source fragment for reconstructing a refined split child.  Full
// local refinement may move records between the split children and neighboring
// partitions, so a child is generally a view over more than its retired
// parent.  Positions are summarized as exact record counts, cache-line
// footprints, and source-order run counts without exporting vector IDs.
struct CxlLineageSourceFragment {
    int64_t source_id = -1;
    int64_t source_size = 0;
    vector<int64_t> child_record_counts;
    vector<int64_t> child_gather_run_counts;
    vector<int64_t> child_gather_line_bytes;
    int64_t membership_bitmap_bytes = 0;
};

// Exact semantic lineage for a physical Quake split.  This is deliberately
// data-free: the replay needs child identity, final membership shape, and
// source-order gather properties, not vector payloads.
struct CxlSplitLineage {
    int64_t parent_id = -1;
    vector<int64_t> child_ids;
    vector<int64_t> final_child_sizes;
    vector<int64_t> source_order_gather_run_counts;
    vector<int64_t> child_logical_line_footprint;
    vector<int64_t> child_gather_line_footprint;
    int64_t membership_bitmap_bytes = 0;
    vector<CxlLineageSourceFragment> source_fragments;
};

struct MaintenancePolicyParams {
    std::string maintenance_policy = DEFAULT_MAINTENANCE_POLICY;
    int window_size = DEFAULT_WINDOW_SIZE;
    int refinement_radius = DEFAULT_REFINEMENT_RADIUS;
    int refinement_iterations = DEFAULT_REFINEMENT_ITERATIONS;
    int min_partition_size = DEFAULT_MIN_PARTITION_SIZE;
    float alpha = DEFAULT_ALPHA;
    bool enable_split_rejection = DEFAULT_ENABLE_SPLIT_REJECTION;
    bool enable_delete_rejection = DEFAULT_ENABLE_DELETE_REJECTION;
    int split_knn_iterations = DEFAULT_NITER;
    float partition_reduction_threshold = DEFAULT_PARTITION_REDUCTION_THRESHOLD;
    float delete_threshold_ns = DEFAULT_DELETE_THRESHOLD_NS;
    float split_threshold_ns = DEFAULT_SPLIT_THRESHOLD_NS;
    // Optional shared CPU scan profile. Experiments should reuse one profile
    // so concurrent process load cannot change the native Quake baseline.
    std::string latency_profile_path;

    // SPFresh Param
    int max_partition_size = -1; // -1 means default to standard cost-based maintenance, if set then we use size-based thresholding

    // CXL-aware maintenance extension. Disabled by default to preserve Quake behavior.
    bool enable_cxl_cost_model = DEFAULT_ENABLE_CXL_COST_MODEL;
    int cxl_num_mcs = DEFAULT_CXL_NUM_MCS;
    int cxl_line_bytes = DEFAULT_CXL_LINE_BYTES;
    int cxl_entry_bytes = DEFAULT_CXL_ENTRY_BYTES;
    int cxl_metadata_bytes = DEFAULT_CXL_METADATA_BYTES;
    std::string cxl_scan_mode = "fpga_scan";
    float cxl_mc_bw_bytes_per_ns = DEFAULT_CXL_MC_BW_BYTES_PER_NS;
    float cxl_link_bw_bytes_per_ns = DEFAULT_CXL_LINK_BW_BYTES_PER_NS;
    float cxl_maintenance_bandwidth_fraction = DEFAULT_CXL_MAINTENANCE_BANDWIDTH_FRACTION;
    float cxl_split_score_weight = DEFAULT_CXL_SPLIT_SCORE_WEIGHT;
    float cxl_maintenance_penalty_weight = DEFAULT_CXL_MAINTENANCE_PENALTY_WEIGHT;
    float cxl_fanout_penalty_ns = DEFAULT_CXL_FANOUT_PENALTY_NS;
    float cxl_fanout_penalty_weight = DEFAULT_CXL_FANOUT_PENALTY_WEIGHT;
    bool cxl_workload_adaptive = DEFAULT_CXL_WORKLOAD_ADAPTIVE;
    float cxl_adaptive_ewma_alpha = DEFAULT_CXL_ADAPTIVE_EWMA_ALPHA;
    float cxl_adaptive_growth_split_gain = DEFAULT_CXL_ADAPTIVE_GROWTH_SPLIT_GAIN;
    float cxl_adaptive_churn_split_gain = DEFAULT_CXL_ADAPTIVE_CHURN_SPLIT_GAIN;
    float cxl_adaptive_scan_split_gain = DEFAULT_CXL_ADAPTIVE_SCAN_SPLIT_GAIN;
    float cxl_adaptive_maintenance_penalty_gain = DEFAULT_CXL_ADAPTIVE_MAINTENANCE_PENALTY_GAIN;
    float cxl_adaptive_delete_relief_gain = DEFAULT_CXL_ADAPTIVE_DELETE_RELIEF_GAIN;
    float cxl_adaptive_fanout_growth_gain = DEFAULT_CXL_ADAPTIVE_FANOUT_GROWTH_GAIN;
    bool cxl_workload_adaptive_v2 = DEFAULT_CXL_WORKLOAD_ADAPTIVE_V2;
    float cxl_adaptive_roi_threshold = DEFAULT_CXL_ADAPTIVE_ROI_THRESHOLD;
    float cxl_adaptive_min_gain_ns = DEFAULT_CXL_ADAPTIVE_MIN_GAIN_NS;
    float cxl_adaptive_action_overhead_ns = DEFAULT_CXL_ADAPTIVE_ACTION_OVERHEAD_NS;
    float cxl_adaptive_base_split_budget_fraction = DEFAULT_CXL_ADAPTIVE_BASE_SPLIT_BUDGET_FRACTION;
    float cxl_adaptive_max_split_budget_fraction = DEFAULT_CXL_ADAPTIVE_MAX_SPLIT_BUDGET_FRACTION;
    int cxl_adaptive_min_split_budget = DEFAULT_CXL_ADAPTIVE_MIN_SPLIT_BUDGET;
    int cxl_adaptive_max_split_budget = DEFAULT_CXL_ADAPTIVE_MAX_SPLIT_BUDGET;
    int cxl_adaptive_target_partition_size = DEFAULT_CXL_ADAPTIVE_TARGET_PARTITION_SIZE;
    float cxl_adaptive_growth_debt_repay_fraction = DEFAULT_CXL_ADAPTIVE_GROWTH_DEBT_REPAY_FRACTION;
    float cxl_adaptive_structural_size_ratio = DEFAULT_CXL_ADAPTIVE_STRUCTURAL_SIZE_RATIO;
    float cxl_adaptive_partition_count_slack = DEFAULT_CXL_ADAPTIVE_PARTITION_COUNT_SLACK;
    int cxl_adaptive_max_reassign_budget = DEFAULT_CXL_ADAPTIVE_MAX_REASSIGN_BUDGET;
    bool cxl_adaptive_observed_cost_enabled = DEFAULT_CXL_ADAPTIVE_OBSERVED_COST_ENABLED;
    float cxl_adaptive_observed_cost_alpha = DEFAULT_CXL_ADAPTIVE_OBSERVED_COST_ALPHA;
    float cxl_adaptive_maintenance_time_budget_fraction =
        DEFAULT_CXL_ADAPTIVE_MAINTENANCE_TIME_BUDGET_FRACTION;
    int cxl_adaptive_warmup_split_budget = DEFAULT_CXL_ADAPTIVE_WARMUP_SPLIT_BUDGET;
    std::string cxl_adaptive_refinement_mode = DEFAULT_CXL_ADAPTIVE_REFINEMENT_MODE;
    int cxl_adaptive_max_refine_splits = DEFAULT_CXL_ADAPTIVE_MAX_REFINE_SPLITS;
    float cxl_adaptive_max_payback_windows = DEFAULT_CXL_ADAPTIVE_MAX_PAYBACK_WINDOWS;
    bool cxl_adaptive_aps_feedback_enabled = DEFAULT_CXL_ADAPTIVE_APS_FEEDBACK_ENABLED;
    bool cxl_adaptive_reassign_time_budget_enabled =
        DEFAULT_CXL_ADAPTIVE_REASSIGN_TIME_BUDGET_ENABLED;
    int cxl_adaptive_warmup_reassign_budget = DEFAULT_CXL_ADAPTIVE_WARMUP_REASSIGN_BUDGET;
    // Online policy: recurrent query savings accumulate until they pay for an
    // observed (or mechanistically estimated) maintenance action. It uses the
    // existing window size and maintenance bandwidth fraction, without the v2
    // pressure weights and per-workload budget knobs.
    bool cxl_streaming_rent_buy = DEFAULT_CXL_STREAMING_RENT_BUY;
    // Coherent descriptor/view policy.  It has no fixed action-count or
    // search-gain target; resource-priced admission is evaluated online.
    bool cxl_streaming_staged = DEFAULT_CXL_STREAMING_STAGED;
    // Resource-priced variant. Prices are supplied causally from the previous
    // completed workload window through CxlResourcePriceSnapshot.
    bool cxl_resource_rent_buy = DEFAULT_CXL_RESOURCE_RENT_BUY;
    // Search-first resource policy. Prior-window CXL prices and observed APS
    // probes define the split candidates and their search benefit. Native
    // Quake cost is telemetry only; CXL maintenance cost orders and truncates
    // the cohort but is not a mandatory one-window payback gate.
    bool cxl_search_first = DEFAULT_CXL_SEARCH_FIRST;
    float cxl_search_first_gain_target = DEFAULT_CXL_SEARCH_FIRST_GAIN_TARGET;
    int cxl_search_first_max_cohort = DEFAULT_CXL_SEARCH_FIRST_MAX_COHORT;
    // Preserve native Quake decisions, then repay a causal fraction of the
    // current hot/oversized partition growth debt under CXL headroom.
    bool cxl_default_plus = DEFAULT_CXL_DEFAULT_PLUS;
    // Native Quake actions plus causally priced, search-positive CXL-only
    // splits.  Physical materialization cost is telemetry, not a veto.
    bool cxl_search_guarded_plus = DEFAULT_CXL_SEARCH_GUARDED_PLUS;
    // Audit-only override for replaying a candidate-set prefix from the same
    // checkpoint. Empty vectors with this flag set mean no optional actions.
    bool cxl_resource_force_action_set = false;
    // Restrict the audit override to one resource-price window. A negative
    // value preserves the legacy behavior of applying it to every window.
    int64_t cxl_resource_force_window_id = -1;
    vector<int64_t> cxl_resource_forced_split_ids;
    vector<int64_t> cxl_resource_forced_reassign_ids;

    MaintenancePolicyParams() = default;
};

/**
 * @brief Parameters that govern how the DynamicIVF index should be built.
 */
struct IndexBuildParams {
    // Basic configuration
    int dimension = 0;
    int nlist = DEFAULT_NLIST;
    int num_workers = DEFAULT_NUM_WORKERS;
    int num_merge_workers = DEFAULT_NUM_MERGE_WORKERS;
    int code_size = -1;         // for PQ
    int num_codebooks = -1;     // for PQ
    string metric = DEFAULT_METRIC;
    int niter = DEFAULT_NITER;

    bool use_adaptive_nprobe = false;
    bool use_numa = false;
    bool verify_numa = false;
    bool same_core = true;
    bool verbose = false;

    // gpu index build params
    bool use_gpu = false;
    int gpu_batch_size = DEFAULT_GPU_BATCH_SIZE;
    int gpu_sample_size = DEFAULT_GPU_SAMPLE_SIZE;

    shared_ptr<IndexBuildParams> parent_params = nullptr;

    IndexBuildParams() = default;
};

inline faiss::MetricType str_to_metric_type(string metric) {
    // convert the string to lowercase
    std::transform(metric.begin(), metric.end(), metric.begin(), ::tolower);

    if (metric == "l2") {
        return faiss::METRIC_L2;
    } else if (metric == "ip") {
        return faiss::METRIC_INNER_PRODUCT;
    } else {
        throw std::invalid_argument("Invalid metric type: " + metric);
    }
}

inline string metric_type_to_str(faiss::MetricType metric) {
    if (metric == faiss::METRIC_L2) {
        return "l2";
    } else if (metric == faiss::METRIC_INNER_PRODUCT) {
        return "ip";
    } else {
        throw std::invalid_argument("Invalid metric type");
    }
}

/**
* @brief Parameters for the search operation
*/
struct SearchParams {
    int nprobe = DEFAULT_NPROBE;
    int k = DEFAULT_K;
    float recall_target = DEFAULT_RECALL_TARGET;
    int num_threads = 1; // number of threads to use for search within a single worker
    float k_factor = 1.0f;
    bool batched_scan = DEFAULT_BATCHED_SCAN;
    int batch_size = MAX_SUBBATCH;

    bool track_hits = true;
    bool scan_all = false;
    bool deterministic_serial_scan = false;

    // APS params
    bool use_precomputed = DEFAULT_PRECOMPUTED;
    float recompute_threshold = DEFAULT_RECOMPUTE_THRESHOLD;
    float initial_search_fraction = DEFAULT_INITIAL_SEARCH_FRACTION;
    int aps_flush_period_us = DEFAULT_APS_FLUSH_PERIOD_US;
    int sample_prefix = 0;
    int sample_stride = 10;

    // Auncel params
    bool use_auncel = false;
    float auncel_a = 1.0f;
    float auncel_b = 1.0f;

    // Spann params
    bool use_spann = false;
    float spann_eps = 1.25;

    shared_ptr<SearchParams> parent_params = nullptr; ///< Search parameters for the parent index, if any.

    SearchParams() = default;
};

/**
 * @brief Structure to hold timing information for building the index.
 */
struct BuildTimingInfo {
    int64_t n_vectors; ///< Number of vectors.
    int64_t n_clusters; ///< Number of clusters.
    int d; ///< Dimensionality of the vectors.
    int num_codebooks; ///< Number of codebooks used in PQ.
    int code_size; ///< Code size for PQ.
    int train_time_us; ///< Training time in microseconds.
    int assign_time_us; ///< Assignment time in microseconds.
    int total_time_us; ///< Total time in microseconds.
};

/**
 * @brief Structure to hold timing information for modify (add/remove) operations.
 */
struct ModifyTimingInfo {
    int64_t n_vectors; ///< Number of vectors.
    int input_validation_time_us; ///< Time spent on input validation in microseconds.
    int find_partition_time_us; ///< Time spent on finding the partition for each vector in microseconds.
    int modify_time_us; ///< Time spent on modify operations in microseconds.
    int maintenance_time_us; ///< Time spent on maintenance operations in microseconds.
};

/**
 * @brief Structure to hold timing information for search operations.
 */
struct SearchTimingInfo {
    int64_t n_queries; ///< Number of queries.
    int64_t n_clusters; ///< Number of clusters (nlist).
    int partitions_scanned; ///< Number of partitions scanned.
    std::vector<std::vector<int64_t>> scanned_partition_ids; ///< Actual partition ids scanned per query.
    shared_ptr<SearchParams> search_params = nullptr; ///< Search parameters.
    shared_ptr<SearchTimingInfo> parent_info = nullptr; ///< Timing info for the parent index, if any.

    // main thread counters for worker scan
    int64_t buffer_init_time_ns; ///< Time spent on initializing buffers in nanoseconds.
    int64_t copy_query_time_ns;
    int64_t job_enqueue_time_ns; ///< Time spent on creating jobs in nanoseconds.
    int64_t boundary_distance_time_ns; ///< Time spent on computing boundary distances in nanoseconds.
    int64_t aps_time_ns; ///< Time spent on APS in nanoseconds.
    int64_t scan_time_ns; ///< Time spent on scanning in nanoseconds.
    int64_t job_wait_time_ns; ///< Time spent waiting for jobs to complete in nanoseconds.
    int64_t result_aggregate_time_ns; ///< Time spent on aggregating results in nanoseconds.
    int64_t total_time_ns; ///< Total time spent in nanoseconds.
    double worker_wait_time_ns = 0; ///< Average worker wait time in nanoseconds.
    double worker_process_time_ns = 0; ///< Average worker process time in nanoseconds.
    double worker_process_preamble_time_ns = 0; ///< Average worker process preamble time in nanoseconds.
    double worker_enqueue_time_ns = 0; ///< Average worker enqueue time in nanoseconds.
    double worker_job_time_ns = 0; ///< Average worker job time in nanoseconds.
    double worker_scan_time_ns = 0; ///< Average worker scan time in nanoseconds.

    int64_t total_worker_jobs = 0; ///< The number of worker jobs
    double worker_partition_size_bytes = 0; ///< Average partition size scanned by worker (in bytes)
    double worker_scan_throughput = 0; ///< Average worker scan throughput (bytes/ns = GB/s).
    double local_scan_throughput = 0; ///< Scan throughput per job rather than averaged across all workers
    double worker_partition_size = 0; ///< Average worker partition size

    double worker_batch_scan_ipc = 0;
    double worker_batch_scan_miss_rate = 0;

    double single_scan_job_time_ns = 0;
    double faiss_norms_x_time_ns = 0;
    double faiss_norms_y_time_ns = 0;
    double sgemm_time_ns = 0;
    double ip_to_l2_time_ns = 0;
    double top_k_buffer_add_ns = 0;
};

/**
 * @brief Structure to hold timing information for maintenance operations.
 */
struct MaintenanceTimingInfo {
    int64_t n_splits = 0; ///< Number of splits.
    int64_t n_deletes = 0; ///< Number of merge-like reassignments.
    int64_t n_recluster = 0; ///< Number of reclusters.

    int64_t decision_time_us = 0; ///< Time spent selecting maintenance actions.
    int64_t action_time_us = 0; ///< Time spent executing committed actions.
    int64_t delete_time_us = 0; ///< Time spent on deletions in microseconds.
    int64_t split_time_us = 0; ///< Time spent on splits in microseconds.
    int64_t refinement_time_us = 0; ///< Time spent on refinement of split output.
    int64_t recluster_time_us = 0; ///< Time spent on reclustering.
    int64_t total_time_us = 0; ///< Total time spent in microseconds.

    int64_t decision_partition_count = 0;
    int64_t centroid_update_count = 0;
    int64_t delete_candidate_count = 0;
    int64_t delete_candidate_records = 0;
    int64_t split_candidate_count = 0;
    int64_t split_candidate_selected_count = 0;
    int64_t split_candidate_roi_rejected_count = 0;
    int64_t split_candidate_budget_rejected_count = 0;
    int64_t split_records_read = 0;
    int64_t split_records_written = 0;
    int64_t reassign_records_read = 0;
    int64_t reassign_records_written = 0;
    int64_t refinement_partition_count = 0;
    int64_t refinement_records_per_iteration = 0;
    int64_t refinement_records_read = 0;
    int64_t refinement_records_written = 0;
    int64_t refinement_iterations = 0;
    int64_t refinement_source_split_count = 0;
    bool refinement_skipped = false;
    int64_t observed_split_ns_per_record = 0;
    int64_t observed_refine_ns_per_split = 0;
    int64_t observed_refine_ns_per_parent_record = 0;
    float observed_refine_read_records_per_parent = 0.0f;
    float observed_refine_write_records_per_parent = 0.0f;
    int64_t observed_reassign_ns_per_record = 0;
    int64_t estimated_query_window_ns = 0;
    int64_t maintenance_time_budget_ns = 0;
    int64_t split_budget_by_time = 0;
    int64_t reassign_budget_by_time = 0;
    float aps_average_fanout = 0.0f;
    float aps_average_scanned_records = 0.0f;
    float aps_average_scanned_list_size = 0.0f;
    float aps_fanout_growth = 0.0f;
    float aps_scanned_list_size_pressure = 0.0f;
    float payback_windows = 1.0f;
    int64_t partition_count_before = 0;
    int64_t partition_count_after = 0;
    bool streaming_rent_buy_enabled = false;
    int64_t streaming_candidate_count = 0;
    int64_t streaming_selected_count = 0;
    int64_t streaming_window_rent_ns = 0;
    int64_t streaming_structural_rent_ns = 0;
    int64_t streaming_structural_credit_ns = 0;
    int64_t streaming_selected_buy_ns = 0;
    int64_t streaming_budget_credit_ns = 0;
    float streaming_scan_growth_pressure = 0.0f;
    bool resource_rent_buy_enabled = false;
    int64_t resource_price_window_id = -1;
    int64_t resource_price_window_duration_ns = 0;
    float resource_child_probe_factor = 0.0f;
    int64_t resource_candidate_count = 0;
    int64_t resource_selected_count = 0;
    int64_t resource_window_rent_ns = 0;
    int64_t resource_selected_buy_ns = 0;
    int64_t resource_split_candidate_count = 0;
    int64_t resource_best_cohort_size = 0;
    int64_t resource_best_cohort_credit_ns = 0;
    int64_t resource_best_cohort_buy_ns = 0;
    float resource_best_cohort_ratio = 0.0f;
    int64_t resource_selected_cohort_size = 0;
    int64_t resource_selected_cohort_buy_ns = 0;
    bool search_first_enabled = false;
    float search_first_gain_target = 0.0f;
    int64_t search_first_max_cohort = 0;
    int64_t search_first_available_gain_ns = 0;
    int64_t search_first_selected_gain_ns = 0;
    float search_first_selected_gain_fraction = 0.0f;
    int64_t search_first_cxl_split_candidate_count = 0;
    int64_t search_first_cxl_only_split_candidate_count = 0;
    int64_t search_first_selected_cxl_only_split_count = 0;
    bool resource_forced_action_set = false;
    vector<CxlPolicyDecision> resource_policy_decisions;
    vector<CxlSplitLineage> split_lineage;
    vector<int64_t> lineage_source_ids;
    vector<int64_t> lineage_source_sizes;
    vector<int64_t> refinement_lineage_source_ids;
    vector<int64_t> refinement_lineage_source_sizes;
};

struct SearchResult {
    Tensor ids;
    Tensor distances;
    shared_ptr<SearchTimingInfo> timing_info;
};

struct Clustering {
    Tensor centroids;
    Tensor partition_ids;
    vector<Tensor> vectors;
    vector<Tensor> vector_ids;

    int64_t ntotal() const {
        int64_t n = 0;
        for (const auto &v : vectors) {
            if (v.defined() && v.numel() > 0) {
                n += v.size(0);
            }
        }
        return n;
    }

    int64_t nlist() const {
        return vectors.size();
    }

    int64_t dim() const {
        return centroids.size(1);
    }

    int64_t cluster_size(int64_t i) const {
        return vectors[i].size(0);
    }
};

#endif //COMMON_H
