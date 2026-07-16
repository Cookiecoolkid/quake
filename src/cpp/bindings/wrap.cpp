//
// Created by Jason on 7/25/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef QUAKE_WRAP_H
#define QUAKE_WRAP_H

#include "common.h"
#include <quake_index.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include "torch/extension.h"
#include <pybind11/stl/filesystem.h>
#include <sstream>  // For JSON-style string formatting

// Helper macros for converting constants to strings
#define TOSTRING(x) + std::to_string(x)

namespace py = pybind11;

using py::arg;
using py::class_;
using py::enum_;
using py::init;
using py::module;

using std::shared_ptr;

using faiss::idx_t;

/**
 * @brief Pybind11 module definition for the Quake bindings.
 *
 * This module exposes the following classes:
 *  - QuakeIndex: The central class for building, searching, and updating the index.
 *  - MaintenanceTimingInfo: Contains timing details for maintenance operations.
 *  - BuildTimingInfo: Contains timing information for the build phase.
 *  - ModifyTimingInfo: Contains timing info for add/remove operations.
 *  - SearchTimingInfo: Contains detailed timing statistics for search.
 *  - MaintenancePolicyParams: Parameters to control the maintenance policy.
 *  - IndexBuildParams: Parameters used during the index build.
 *  - SearchParams: Parameters used during index search.
 *  - SearchResult: The result structure returned from a search.
 */
PYBIND11_MODULE(_bindings, m) {
    m.doc() = R"pbdoc(
        Quake Python Bindings
        ----------------------
        This module provides Python access to the Quake index. Use it to build, search, update,
        and maintain your index.
    )pbdoc";

    /*********** Index Partition Bindings ***********/
    class_<IndexPartition>(m, "IndexPartition")
        .def_readwrite_static("delete_resize_threshold", &IndexPartition::delete_resize_threshold_)
        .def_readwrite_static("capacity_resize_threshold", &IndexPartition::capacity_resize_threshold_);

     class_<QueryCoordinator>(m, "QueryCoordinator")
        .def_readwrite_static("partition_chunk_size", &QueryCoordinator::batch_scan_partition_chunk_size_)
        .def_readwrite_static("query_chunk_size", &QueryCoordinator::batch_scan_query_chunk_size_);

    /*********** QuakeIndex Binding ***********/
    class_<QuakeIndex, shared_ptr<QuakeIndex>>(m, "QuakeIndex")
        .def(init<int>(), arg("current_level") = 0,
             "Create a new QuakeIndex (default current_level = 0).")
        .def("build", &QuakeIndex::build,
             ([]() -> const char* {
                 static const std::string doc = std::string("Build the index from a tensor of vectors and a corresponding tensor of IDs.\n\n"
                     "Args:\n"
                     "    x (Tensor): Tensor of shape [num_vectors, dimension].\n"
                     "    ids (Tensor): Tensor of shape [num_vectors].\n"
                     "    build_params (IndexBuildParams): Parameters for building the index. Defaults include:\n"
                     "         - niter: default = ") + std::to_string(DEFAULT_NITER) + "\n"
                     "         - metric: default = " + std::string(DEFAULT_METRIC) + "\n"
                     "         - num_workers: default = " + std::to_string(DEFAULT_NUM_WORKERS);
                 return doc.c_str();
             })())
        .def("add_level", &QuakeIndex::add_level,
             "Add a new level to the index.\n\n"
             "Args:\n"
             "    build_params (IndexBuildParams): Parameters for building the new level.")
        .def("search", &QuakeIndex::search,
             ([]() -> const char* {
                 static const std::string doc = std::string("Search the index for nearest neighbors.\n\n"
                     "Args:\n"
                     "    x (Tensor): Query tensor of shape [num_queries, dimension].\n"
                     "    search_params (SearchParams): Parameters for the search operation. Defaults include:\n"
                     "         - k: default = ") + std::to_string(DEFAULT_K) + "\n"
                     "         - nprobe: default = " + std::to_string(DEFAULT_NPROBE) + "\n"
                     "         - recall_target: default = " + std::to_string(DEFAULT_RECALL_TARGET);
                 return doc.c_str();
             })())
        .def("get", &QuakeIndex::get,
             "Retrieve vectors from the index by ID.\n\n"
             "Args:\n"
             "    ids (Tensor): Tensor of IDs to retrieve.")
        .def("get_ids", &QuakeIndex::get_ids, "Return all vector IDs stored in the index.")
        .def("add", &QuakeIndex::add,
             "Add new vectors to the index.\n\n"
             "Args:\n"
             "    x (Tensor): Tensor of vectors to add.\n"
             "    ids (Tensor): Tensor of corresponding IDs.")
        .def("remove", &QuakeIndex::remove,
             "Remove vectors from the index.\n\n"
             "Args:\n"
             "    ids (Tensor): Tensor of IDs to remove.")
        .def("maintenance", &QuakeIndex::maintenance,
             "Perform maintenance operations on the index (e.g., splits and merges).\n"
             "Returns timing information for the maintenance operation.")
        .def("record_cxl_update_counts", &QuakeIndex::record_cxl_update_counts,
             "Record recent add/delete counts for streaming CXL-aware maintenance.")
        .def("set_cxl_resource_price_snapshot", &QuakeIndex::set_cxl_resource_price_snapshot,
             "Install the causal CXL resource prices used by resource rent-or-buy maintenance.")
        .def("initialize_maintenance_policy", &QuakeIndex::initialize_maintenance_policy,
             "Initialize the maintenance policy for the index.\n\n"
             "Args:\n"
             "    maintenance_policy_params (MaintenancePolicyParams): Parameters for the maintenance policy.")
        .def("save", &QuakeIndex::save,
             "Save the index to a specified path.\n\n"
             "Args:\n"
             "    path (str): The path to save the index.")
        .def("load", &QuakeIndex::load,
             "Load an index from a specified path.\n\n"
             "Args:\n"
             "    path (str): The path from which to load the index.\n"
             "    n_workers (int, optional): Number of workers for query processing (default = 0).")
        .def("ntotal", &QuakeIndex::ntotal,
             "Return the total number of vectors stored in the index.")
        .def("nlist", &QuakeIndex::nlist,
             "Return the number of partitions (lists) in the index.")
        .def("partition_ids", &QuakeIndex::partition_ids,
             "Return active partition IDs in partition-store order.")
        .def("partition_sizes", &QuakeIndex::partition_sizes,
             "Return physical partition sizes aligned with partition_ids().")
        .def_readonly("parent", &QuakeIndex::parent_,
            "Return the parent index over the centroids.")
        .def_readonly("current_level", &QuakeIndex::current_level_,
             "The current level of the index.")
        // __repr__ produces a JSON-style string summary.
        .def("__repr__", [](const QuakeIndex &q) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"current_level\": " << q.current_level_ << ", ";
            oss << "}";
            return oss.str();
        });

    // bool use_gpu = false;
    // int gpu_batch_size = DEFAULT_GPU_BATCH_SIZE;
    // int gpu_sample_size = DEFAULT_GPU_SAMPLE_SIZE;

    /*********** IndexBuildParams Binding ***********/
    class_<IndexBuildParams, shared_ptr<IndexBuildParams>>(m, "IndexBuildParams")
        .def(init<>())
        .def_readwrite("nlist", &IndexBuildParams::nlist,
             (std::string("Number of clusters (lists). default = ") + std::to_string(DEFAULT_NLIST)).c_str())
        .def_readwrite("niter", &IndexBuildParams::niter,
             (std::string("Number of k-means iterations. default = ") + std::to_string(DEFAULT_NITER)).c_str())
        .def_readwrite("metric", &IndexBuildParams::metric,
             (std::string("Distance metric. default = ") + DEFAULT_METRIC).c_str())
        .def_readwrite("num_workers", &IndexBuildParams::num_workers,
             (std::string("Number of workers. default = ") + std::to_string(DEFAULT_NUM_WORKERS)).c_str())
        .def_readwrite("parent_params", &IndexBuildParams::parent_params,
             "Parameters for the parent index, if any.")
        .def_readwrite("num_merge_workers", &IndexBuildParams::num_merge_workers,
             (std::string("Number of workers for merging. default = ") + std::to_string(DEFAULT_NUM_MERGE_WORKERS)).c_str())
        .def_readwrite("use_numa", &IndexBuildParams::use_numa,
         (std::string("Flag to use NUMA for index building. default = ") + std::to_string(false)).c_str())
        .def_readwrite("use_gpu", &IndexBuildParams::use_gpu,
             (std::string("Flag to use GPU for index building. default = ") + std::to_string(false)).c_str())
        .def_readwrite("gpu_batch_size", &IndexBuildParams::gpu_batch_size,
             (std::string("Batch size for GPU index building. default = ") + std::to_string(DEFAULT_GPU_BATCH_SIZE)).c_str())
        .def_readwrite("gpu_sample_size", &IndexBuildParams::gpu_sample_size,
             (std::string("Sample size for GPU index building. default = ") + std::to_string(DEFAULT_GPU_SAMPLE_SIZE)).c_str())

        .def("__repr__", [](const IndexBuildParams &p) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"nlist\": " << p.nlist << ", ";
            oss << "\"niter\": " << p.niter << ", ";
            oss << "\"metric\": \"" << p.metric << "\", ";
            oss << "\"num_workers\": " << p.num_workers;
            oss << "}";
            return oss.str();
        });

    /*********** SearchParams Binding ***********/
    class_<SearchParams, shared_ptr<SearchParams>>(m, "SearchParams")
        .def(init<>())
        .def_readwrite("k", &SearchParams::k,
             (std::string("Number of neighbors to return. default = ") + std::to_string(DEFAULT_K)).c_str())
        .def_readwrite("nprobe", &SearchParams::nprobe,
             (std::string("Number of partitions to probe. default = ") + std::to_string(DEFAULT_NPROBE)).c_str())
        .def_readwrite("recall_target", &SearchParams::recall_target,
             (std::string("Recall target. default = ") + std::to_string(DEFAULT_RECALL_TARGET)).c_str())
        .def_readwrite("num_threads", &SearchParams::num_threads,
             "Number of threads to use for search within a single worker.")
        .def_readwrite("batched_scan", &SearchParams::batched_scan,
             (std::string("Flag for batched scanning. default = ") + std::to_string(DEFAULT_BATCHED_SCAN)).c_str())
        .def_readwrite("use_precomputed", &SearchParams::use_precomputed,
             (std::string("Flag to use precomputed inc beta fn for APS. default = ") + std::to_string(DEFAULT_PRECOMPUTED)).c_str())
        .def_readwrite("initial_search_fraction", &SearchParams::initial_search_fraction,
             (std::string("Initial fraction of partitions to search. default = ") + std::to_string(DEFAULT_INITIAL_SEARCH_FRACTION)).c_str())
        .def_readwrite("recompute_threshold", &SearchParams::recompute_threshold,
             (std::string("Threshold to trigger recomputation of APS. default = ") + std::to_string(DEFAULT_RECOMPUTE_THRESHOLD)).c_str())
        .def_readwrite("aps_flush_period_us", &SearchParams::aps_flush_period_us,
             (std::string("APS flush period in microseconds. default = ") + std::to_string(DEFAULT_APS_FLUSH_PERIOD_US)).c_str())
        .def_readwrite("batch_size", &SearchParams::batch_size,
             (std::string("Batch size for batched scan. default = ") + std::to_string(MAX_SUBBATCH)).c_str())
        .def_readwrite("k_factor", &SearchParams::k_factor,
             "Factor to adjust the number of neighbors to return.")
        .def_readwrite("track_hits", &SearchParams::track_hits,
             "Flag to track hits for maintenance policy.")
        .def_readwrite("deterministic_serial_scan", &SearchParams::deterministic_serial_scan,
             "Use Quake's serial scan path for deterministic APS trace collection.")
        .def_readwrite("use_auncel", &SearchParams::use_auncel,
                "Flag to use Auncel recall estimation for search.")
        .def_readwrite("auncel_a", &SearchParams::auncel_a,
                "Auncel parameter a for recall estimation.")
        .def_readwrite("auncel_b", &SearchParams::auncel_b,
                "Auncel parameter b for recall estimation.")
        .def_readwrite("use_spann", &SearchParams::use_spann,
                "Flag to use SPANN for search.")
        .def_readwrite("spann_eps", &SearchParams::spann_eps,
                "SPANN parameter epsilon for search.")
        .def_readwrite("sample_prefix", &SearchParams::sample_prefix,
                "Prefix length for APS sampling.")
        .def_readwrite("sample_stride", &SearchParams::sample_stride,
                "Stride length for APS sampling.")

        .def_readwrite("parent_params", &SearchParams::parent_params,
             "Search parameters for the parent index, if any.")
        .def("__repr__", [](const SearchParams &s) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"k\": " << s.k << ", ";
            oss << "\"nprobe\": " << s.nprobe << ", ";
            oss << "\"recall_target\": " << s.recall_target << ", ";
            oss << "\"batched_scan\": " << (s.batched_scan ? "true" : "false") << ", ";
            oss << "\"deterministic_serial_scan\": "
                << (s.deterministic_serial_scan ? "true" : "false") << ", ";
            oss << "\"use_precomputed\": " << (s.use_precomputed ? "true" : "false") << ", ";
            oss << "\"initial_search_fraction\": " << s.initial_search_fraction << ", ";
            oss << "\"recompute_threshold\": " << s.recompute_threshold << ", ";
            oss << "\"aps_flush_period_us\": " << s.aps_flush_period_us;
            oss << "}";
            return oss.str();
        });

    class_<CxlResourcePrice>(m, "CxlResourcePrice")
        .def(init<>())
        .def_readwrite("resource_id", &CxlResourcePrice::resource_id)
        .def_readwrite("byte_price_ns", &CxlResourcePrice::byte_price_ns)
        .def_readwrite("op_price_ns", &CxlResourcePrice::op_price_ns)
        .def_readwrite("utilization", &CxlResourcePrice::utilization)
        .def_readwrite("demand_bytes", &CxlResourcePrice::demand_bytes);

    class_<CxlResourcePriceSnapshot, shared_ptr<CxlResourcePriceSnapshot>>(
        m, "CxlResourcePriceSnapshot")
        .def(init<>())
        .def_readwrite("window_id", &CxlResourcePriceSnapshot::window_id)
        .def_readwrite("window_duration_ns", &CxlResourcePriceSnapshot::window_duration_ns)
        .def_readwrite("resources", &CxlResourcePriceSnapshot::resources)
        .def_readwrite("home_read_byte_price_ns", &CxlResourcePriceSnapshot::home_read_byte_price_ns)
        .def_readwrite("home_read_op_price_ns", &CxlResourcePriceSnapshot::home_read_op_price_ns)
        .def_readwrite("list_home_ids", &CxlResourcePriceSnapshot::list_home_ids)
        .def_readwrite("maintenance_read_byte_price_ns", &CxlResourcePriceSnapshot::maintenance_read_byte_price_ns)
        .def_readwrite("maintenance_write_byte_price_ns", &CxlResourcePriceSnapshot::maintenance_write_byte_price_ns)
        .def_readwrite("routing_metadata_shadow_price_ns_per_byte", &CxlResourcePriceSnapshot::routing_metadata_shadow_price_ns_per_byte)
        .def_readwrite("dram_shadow_price_ns_per_byte", &CxlResourcePriceSnapshot::dram_shadow_price_ns_per_byte)
        .def_readwrite("valid", &CxlResourcePriceSnapshot::valid);

    class_<CxlPolicyDecision>(m, "CxlPolicyDecision")
        .def(init<>())
        .def_readonly("action_kind", &CxlPolicyDecision::action_kind)
        .def_readonly("partition_id", &CxlPolicyDecision::partition_id)
        .def_readonly("home_id", &CxlPolicyDecision::home_id)
        .def_readonly("records", &CxlPolicyDecision::records)
        .def_readonly("cost_before_ns", &CxlPolicyDecision::cost_before_ns)
        .def_readonly("cost_after_ns", &CxlPolicyDecision::cost_after_ns)
        .def_readonly("rent_ns", &CxlPolicyDecision::rent_ns)
        .def_readonly("buy_ns", &CxlPolicyDecision::buy_ns)
        .def_readonly("credit_ns", &CxlPolicyDecision::credit_ns)
        .def_readonly("rent_buy_ratio", &CxlPolicyDecision::rent_buy_ratio)
        .def_readonly("native_rent_ns", &CxlPolicyDecision::native_rent_ns)
        .def_readonly("resource_rent_ns", &CxlPolicyDecision::resource_rent_ns)
        .def_readonly("native_legal", &CxlPolicyDecision::native_legal)
        .def_readonly("cxl_search_profitable", &CxlPolicyDecision::cxl_search_profitable)
        .def_readonly("read_line_bytes", &CxlPolicyDecision::read_line_bytes)
        .def_readonly("write_line_bytes", &CxlPolicyDecision::write_line_bytes)
        .def_readonly("cohort_id", &CxlPolicyDecision::cohort_id)
        .def_readonly("cohort_size", &CxlPolicyDecision::cohort_size)
        .def_readonly("cohort_credit_ns", &CxlPolicyDecision::cohort_credit_ns)
        .def_readonly("cohort_variable_buy_ns", &CxlPolicyDecision::cohort_variable_buy_ns)
        .def_readonly("cohort_shared_buy_ns", &CxlPolicyDecision::cohort_shared_buy_ns)
        .def_readonly("cohort_buy_ns", &CxlPolicyDecision::cohort_buy_ns)
        .def_readonly("cohort_rent_buy_ratio", &CxlPolicyDecision::cohort_rent_buy_ratio)
        .def_readonly("cohort_search_gain_ns", &CxlPolicyDecision::cohort_search_gain_ns)
        .def_readonly("cohort_search_gain_fraction", &CxlPolicyDecision::cohort_search_gain_fraction)
        .def_readonly("selected", &CxlPolicyDecision::selected)
        .def_readonly("rejection_reason", &CxlPolicyDecision::rejection_reason);

    class_<CxlSplitLineage>(m, "CxlSplitLineage")
        .def_readonly("parent_id", &CxlSplitLineage::parent_id)
        .def_readonly("child_ids", &CxlSplitLineage::child_ids)
        .def_readonly("final_child_sizes", &CxlSplitLineage::final_child_sizes)
        .def_readonly("source_order_gather_run_counts", &CxlSplitLineage::source_order_gather_run_counts)
        .def_readonly("child_logical_line_footprint", &CxlSplitLineage::child_logical_line_footprint)
        .def_readonly("membership_bitmap_bytes", &CxlSplitLineage::membership_bitmap_bytes);

    /*********** MaintenancePolicyParams Binding ***********/
    class_<MaintenancePolicyParams, shared_ptr<MaintenancePolicyParams>>(m, "MaintenancePolicyParams")
        .def(init<>())
        .def_readwrite("maintenance_policy", &MaintenancePolicyParams::maintenance_policy,
             (std::string("Maintenance policy type. default = ") + DEFAULT_MAINTENANCE_POLICY).c_str())
        .def_readwrite("window_size", &MaintenancePolicyParams::window_size,
             (std::string("Window size for measuring hit rates. default = ") + std::to_string(DEFAULT_WINDOW_SIZE)).c_str())
        .def_readwrite("refinement_radius", &MaintenancePolicyParams::refinement_radius,
             (std::string("Radius for local partition refinement. default = ") + std::to_string(DEFAULT_REFINEMENT_RADIUS)).c_str())
        .def_readwrite("refinement_iterations", &MaintenancePolicyParams::refinement_iterations,
             (std::string("Number of refinement iterations. default = ") + std::to_string(DEFAULT_REFINEMENT_ITERATIONS)).c_str())
        .def_readwrite("min_partition_size", &MaintenancePolicyParams::min_partition_size,
             (std::string("Minimum allowed partition size. default = ") + std::to_string(DEFAULT_MIN_PARTITION_SIZE)).c_str())
        .def_readwrite("max_partition_size", &MaintenancePolicyParams::max_partition_size,
            (std::string("Maximum allowed partition size. default = ") + std::to_string(-1)).c_str())
        .def_readwrite("alpha", &MaintenancePolicyParams::alpha,
             (std::string("Alpha parameter. default = ") + std::to_string(DEFAULT_ALPHA)).c_str())
        .def_readwrite("enable_split_rejection", &MaintenancePolicyParams::enable_split_rejection,
             (std::string("Enable split rejection. default = ") + std::to_string(DEFAULT_ENABLE_SPLIT_REJECTION)).c_str())
        .def_readwrite("enable_delete_rejection", &MaintenancePolicyParams::enable_delete_rejection,
             (std::string("Enable delete rejection. default = ") + std::to_string(DEFAULT_ENABLE_DELETE_REJECTION)).c_str())
        .def_readwrite("split_knn_iterations", &MaintenancePolicyParams::split_knn_iterations,
             (std::string("Number of clustering iterations to perform during a clustering. default = ") + std::to_string(DEFAULT_NITER)).c_str())
        .def_readwrite("partition_reduction_threshold", &MaintenancePolicyParams::partition_reduction_threshold,
             (std::string("Threshold for deleting a partition based on the number of vectors it has lost. default = ") + std::to_string(DEFAULT_PARTITION_REDUCTION_THRESHOLD)).c_str())   
        .def_readwrite("delete_threshold_ns", &MaintenancePolicyParams::delete_threshold_ns,
             (std::string("Delete threshold (ns). default = ") + std::to_string(DEFAULT_DELETE_THRESHOLD_NS)).c_str())
        .def_readwrite("split_threshold_ns", &MaintenancePolicyParams::split_threshold_ns,
             (std::string("Split threshold (ns). default = ") + std::to_string(DEFAULT_SPLIT_THRESHOLD_NS)).c_str())
        .def_readwrite("latency_profile_path", &MaintenancePolicyParams::latency_profile_path,
             "Optional shared CSV profile for the native Quake list-scan cost estimator.")
        .def_readwrite("enable_cxl_cost_model", &MaintenancePolicyParams::enable_cxl_cost_model,
             "Enable the lightweight CXL-aware maintenance score extension.")
        .def_readwrite("cxl_num_mcs", &MaintenancePolicyParams::cxl_num_mcs,
             "Number of CXL memory controllers used by the lightweight CXL cost model.")
        .def_readwrite("cxl_line_bytes", &MaintenancePolicyParams::cxl_line_bytes,
             "CXL cache-line transfer size used by the lightweight CXL cost model.")
        .def_readwrite("cxl_entry_bytes", &MaintenancePolicyParams::cxl_entry_bytes,
             "Posting entry bytes used by the lightweight CXL cost model.")
        .def_readwrite("cxl_metadata_bytes", &MaintenancePolicyParams::cxl_metadata_bytes,
             "Metadata bytes per record reserved for future CXL layout-aware policies.")
        .def_readwrite("cxl_scan_mode", &MaintenancePolicyParams::cxl_scan_mode,
             "CXL scan mode used by the lightweight CXL cost model: host_scan or fpga_scan.")
        .def_readwrite("cxl_mc_bw_bytes_per_ns", &MaintenancePolicyParams::cxl_mc_bw_bytes_per_ns,
             "Per-MC bandwidth in bytes/ns used by the lightweight CXL cost model.")
        .def_readwrite("cxl_link_bw_bytes_per_ns", &MaintenancePolicyParams::cxl_link_bw_bytes_per_ns,
             "CXL link bandwidth in bytes/ns used by the lightweight CXL cost model.")
        .def_readwrite("cxl_maintenance_bandwidth_fraction", &MaintenancePolicyParams::cxl_maintenance_bandwidth_fraction,
             "Fraction of bottleneck bandwidth budgeted for background maintenance.")
        .def_readwrite("cxl_split_score_weight", &MaintenancePolicyParams::cxl_split_score_weight,
             "Weight for CXL scan benefit in split scoring.")
        .def_readwrite("cxl_maintenance_penalty_weight", &MaintenancePolicyParams::cxl_maintenance_penalty_weight,
             "Weight for CXL maintenance traffic penalty.")
        .def_readwrite("cxl_fanout_penalty_ns", &MaintenancePolicyParams::cxl_fanout_penalty_ns,
             "Per-hit fanout/control penalty used by the lightweight CXL cost model.")
        .def_readwrite("cxl_fanout_penalty_weight", &MaintenancePolicyParams::cxl_fanout_penalty_weight,
             "Weight for CXL fanout/control penalty in split scoring.")
        .def_readwrite("cxl_workload_adaptive", &MaintenancePolicyParams::cxl_workload_adaptive,
             "Enable streaming workload-adaptive CXL cost weights.")
        .def_readwrite("cxl_adaptive_ewma_alpha", &MaintenancePolicyParams::cxl_adaptive_ewma_alpha,
             "EWMA alpha for streaming CXL workload pressure signals.")
        .def_readwrite("cxl_adaptive_growth_split_gain", &MaintenancePolicyParams::cxl_adaptive_growth_split_gain,
             "Split-score gain applied when recent vector growth is high.")
        .def_readwrite("cxl_adaptive_churn_split_gain", &MaintenancePolicyParams::cxl_adaptive_churn_split_gain,
             "Split-score gain applied when recent insert/delete churn is high.")
        .def_readwrite("cxl_adaptive_scan_split_gain", &MaintenancePolicyParams::cxl_adaptive_scan_split_gain,
             "Split-score gain applied when recent scan fraction is high.")
        .def_readwrite("cxl_adaptive_maintenance_penalty_gain", &MaintenancePolicyParams::cxl_adaptive_maintenance_penalty_gain,
             "Penalty gain applied when previous CXL maintenance traffic was high.")
        .def_readwrite("cxl_adaptive_delete_relief_gain", &MaintenancePolicyParams::cxl_adaptive_delete_relief_gain,
             "Delete maintenance penalty relief applied under delete-heavy windows.")
        .def_readwrite("cxl_adaptive_fanout_growth_gain", &MaintenancePolicyParams::cxl_adaptive_fanout_growth_gain,
             "Fanout penalty gain applied under growth-heavy windows.")
        .def_readwrite("cxl_workload_adaptive_v2", &MaintenancePolicyParams::cxl_workload_adaptive_v2,
             "Enable ROI- and budget-gated streaming CXL workload-adaptive policy.")
        .def_readwrite("cxl_adaptive_roi_threshold", &MaintenancePolicyParams::cxl_adaptive_roi_threshold,
             "Minimum split benefit/cost ratio for the ROI-gated adaptive policy.")
        .def_readwrite("cxl_adaptive_min_gain_ns", &MaintenancePolicyParams::cxl_adaptive_min_gain_ns,
             "Minimum amortized split gain in ns for the ROI-gated adaptive policy.")
        .def_readwrite("cxl_adaptive_action_overhead_ns", &MaintenancePolicyParams::cxl_adaptive_action_overhead_ns,
             "Fixed split action cost in ns before window amortization.")
        .def_readwrite("cxl_adaptive_base_split_budget_fraction", &MaintenancePolicyParams::cxl_adaptive_base_split_budget_fraction,
             "Base fraction of partitions that the ROI-gated adaptive policy may split per maintenance window.")
        .def_readwrite("cxl_adaptive_max_split_budget_fraction", &MaintenancePolicyParams::cxl_adaptive_max_split_budget_fraction,
             "Maximum fraction of partitions that the ROI-gated adaptive policy may split per maintenance window.")
        .def_readwrite("cxl_adaptive_min_split_budget", &MaintenancePolicyParams::cxl_adaptive_min_split_budget,
             "Minimum non-zero split budget for active ROI-gated adaptive windows.")
        .def_readwrite("cxl_adaptive_max_split_budget", &MaintenancePolicyParams::cxl_adaptive_max_split_budget,
             "Maximum split budget for one ROI-gated adaptive maintenance window.")
        .def_readwrite("cxl_adaptive_target_partition_size", &MaintenancePolicyParams::cxl_adaptive_target_partition_size,
             "Target live records per partition; zero derives it from the initial index.")
        .def_readwrite("cxl_adaptive_growth_debt_repay_fraction", &MaintenancePolicyParams::cxl_adaptive_growth_debt_repay_fraction,
             "Fraction of missing structural partitions repaid per maintenance window.")
        .def_readwrite("cxl_adaptive_structural_size_ratio", &MaintenancePolicyParams::cxl_adaptive_structural_size_ratio,
             "Minimum size relative to the initial target for a structural split candidate.")
        .def_readwrite("cxl_adaptive_partition_count_slack", &MaintenancePolicyParams::cxl_adaptive_partition_count_slack,
             "Allowed partition-count slack before cold-list reassignment is selected.")
        .def_readwrite("cxl_adaptive_max_reassign_budget", &MaintenancePolicyParams::cxl_adaptive_max_reassign_budget,
             "Maximum cold-list reassignments in one maintenance window.")
        .def_readwrite("cxl_adaptive_observed_cost_enabled", &MaintenancePolicyParams::cxl_adaptive_observed_cost_enabled,
             "Use observed split/refine/reassign time in subsequent maintenance decisions.")
        .def_readwrite("cxl_adaptive_observed_cost_alpha", &MaintenancePolicyParams::cxl_adaptive_observed_cost_alpha,
             "EWMA alpha for observed maintenance action costs.")
        .def_readwrite("cxl_adaptive_maintenance_time_budget_fraction", &MaintenancePolicyParams::cxl_adaptive_maintenance_time_budget_fraction,
             "Maintenance-time budget relative to the estimated query-window scan time.")
        .def_readwrite("cxl_adaptive_warmup_split_budget", &MaintenancePolicyParams::cxl_adaptive_warmup_split_budget,
             "Split budget before the first observed maintenance cost sample.")
        .def_readwrite("cxl_adaptive_refinement_mode", &MaintenancePolicyParams::cxl_adaptive_refinement_mode,
             "Refinement mode for v2: full, bounded, or lazy.")
        .def_readwrite("cxl_adaptive_max_refine_splits", &MaintenancePolicyParams::cxl_adaptive_max_refine_splits,
             "Maximum parent splits whose children receive local refinement in bounded mode.")
        .def_readwrite("cxl_adaptive_max_payback_windows", &MaintenancePolicyParams::cxl_adaptive_max_payback_windows,
             "Maximum online payback horizon inferred from the observed update rate.")
        .def_readwrite("cxl_adaptive_aps_feedback_enabled", &MaintenancePolicyParams::cxl_adaptive_aps_feedback_enabled,
             "Use APS fanout and average scanned-list size as online split signals.")
        .def_readwrite("cxl_adaptive_reassign_time_budget_enabled", &MaintenancePolicyParams::cxl_adaptive_reassign_time_budget_enabled,
             "Apply the observed maintenance-time budget to merge-like reassignments.")
        .def_readwrite("cxl_adaptive_warmup_reassign_budget", &MaintenancePolicyParams::cxl_adaptive_warmup_reassign_budget,
             "Reassign action cap before an observed reassign cost is available.")
        .def_readwrite("cxl_streaming_rent_buy", &MaintenancePolicyParams::cxl_streaming_rent_buy,
             "Enable online rent-or-buy maintenance using only past query windows and observed action costs.")
        .def_readwrite("cxl_streaming_staged", &MaintenancePolicyParams::cxl_streaming_staged,
             "Enable coherent descriptor/view staged maintenance with causal resource pricing.")
        .def_readwrite("cxl_resource_rent_buy", &MaintenancePolicyParams::cxl_resource_rent_buy,
             "Enable causal resource-priced rent-or-buy maintenance.")
        .def_readwrite("cxl_search_first", &MaintenancePolicyParams::cxl_search_first,
             "Select CXL-profitable splits covering the target current search gain; native Quake cost is telemetry only.")
        .def_readwrite("cxl_search_first_gain_target", &MaintenancePolicyParams::cxl_search_first_gain_target,
             "Fraction of available current search gain targeted by search-first maintenance.")
        .def_readwrite("cxl_search_first_max_cohort", &MaintenancePolicyParams::cxl_search_first_max_cohort,
             "Maximum split cohort selected by search-first maintenance.")
        .def_readwrite("cxl_resource_force_action_set", &MaintenancePolicyParams::cxl_resource_force_action_set,
             "Audit-only: execute exactly the requested optional resource-policy candidates.")
        .def_readwrite("cxl_resource_force_window_id", &MaintenancePolicyParams::cxl_resource_force_window_id,
             "Audit-only resource-price window for the forced action set; negative means every window.")
        .def_readwrite("cxl_resource_forced_split_ids", &MaintenancePolicyParams::cxl_resource_forced_split_ids,
             "Split candidate IDs requested by the audit override.")
        .def_readwrite("cxl_resource_forced_reassign_ids", &MaintenancePolicyParams::cxl_resource_forced_reassign_ids,
             "Reassign candidate IDs requested by the audit override.")
        .def("__repr__", [](const MaintenancePolicyParams &m) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"maintenance_policy\": \"" << m.maintenance_policy << "\", ";
            oss << "\"window_size\": " << m.window_size << ", ";
            oss << "\"refinement_radius\": " << m.refinement_radius << ", ";
            oss << "\"refinement_iterations\": " << m.refinement_iterations << ", ";
            oss << "\"min_partition_size\": " << m.min_partition_size << ", ";
            oss << "\"alpha\": " << m.alpha << ", ";
            oss << "\"enable_split_rejection\": " << (m.enable_split_rejection ? "true" : "false") << ", ";
            oss << "\"enable_delete_rejection\": " << (m.enable_delete_rejection ? "true" : "false") << ", ";
            oss << "\"delete_threshold_ns\": " << m.delete_threshold_ns << ", ";
            oss << "\"split_threshold_ns\": " << m.split_threshold_ns << ", ";
            oss << "\"enable_cxl_cost_model\": " << (m.enable_cxl_cost_model ? "true" : "false") << ", ";
            oss << "\"cxl_num_mcs\": " << m.cxl_num_mcs << ", ";
            oss << "\"cxl_line_bytes\": " << m.cxl_line_bytes << ", ";
            oss << "\"cxl_entry_bytes\": " << m.cxl_entry_bytes << ", ";
            oss << "\"cxl_metadata_bytes\": " << m.cxl_metadata_bytes << ", ";
            oss << "\"cxl_scan_mode\": \"" << m.cxl_scan_mode << "\", ";
            oss << "\"cxl_mc_bw_bytes_per_ns\": " << m.cxl_mc_bw_bytes_per_ns << ", ";
            oss << "\"cxl_link_bw_bytes_per_ns\": " << m.cxl_link_bw_bytes_per_ns << ", ";
            oss << "\"cxl_maintenance_bandwidth_fraction\": " << m.cxl_maintenance_bandwidth_fraction << ", ";
            oss << "\"cxl_split_score_weight\": " << m.cxl_split_score_weight << ", ";
            oss << "\"cxl_maintenance_penalty_weight\": " << m.cxl_maintenance_penalty_weight << ", ";
            oss << "\"cxl_fanout_penalty_ns\": " << m.cxl_fanout_penalty_ns << ", ";
            oss << "\"cxl_fanout_penalty_weight\": " << m.cxl_fanout_penalty_weight << ", ";
            oss << "\"cxl_workload_adaptive\": " << (m.cxl_workload_adaptive ? "true" : "false") << ", ";
            oss << "\"cxl_adaptive_ewma_alpha\": " << m.cxl_adaptive_ewma_alpha << ", ";
            oss << "\"cxl_adaptive_growth_split_gain\": " << m.cxl_adaptive_growth_split_gain << ", ";
            oss << "\"cxl_adaptive_churn_split_gain\": " << m.cxl_adaptive_churn_split_gain << ", ";
            oss << "\"cxl_adaptive_scan_split_gain\": " << m.cxl_adaptive_scan_split_gain << ", ";
            oss << "\"cxl_adaptive_maintenance_penalty_gain\": " << m.cxl_adaptive_maintenance_penalty_gain << ", ";
            oss << "\"cxl_adaptive_delete_relief_gain\": " << m.cxl_adaptive_delete_relief_gain << ", ";
            oss << "\"cxl_adaptive_fanout_growth_gain\": " << m.cxl_adaptive_fanout_growth_gain << ", ";
            oss << "\"cxl_workload_adaptive_v2\": " << (m.cxl_workload_adaptive_v2 ? "true" : "false") << ", ";
            oss << "\"cxl_adaptive_roi_threshold\": " << m.cxl_adaptive_roi_threshold << ", ";
            oss << "\"cxl_adaptive_min_gain_ns\": " << m.cxl_adaptive_min_gain_ns << ", ";
            oss << "\"cxl_adaptive_action_overhead_ns\": " << m.cxl_adaptive_action_overhead_ns << ", ";
            oss << "\"cxl_adaptive_base_split_budget_fraction\": " << m.cxl_adaptive_base_split_budget_fraction << ", ";
            oss << "\"cxl_adaptive_max_split_budget_fraction\": " << m.cxl_adaptive_max_split_budget_fraction << ", ";
            oss << "\"cxl_adaptive_min_split_budget\": " << m.cxl_adaptive_min_split_budget << ", ";
            oss << "\"cxl_adaptive_max_split_budget\": " << m.cxl_adaptive_max_split_budget << ", ";
            oss << "\"cxl_adaptive_target_partition_size\": " << m.cxl_adaptive_target_partition_size << ", ";
            oss << "\"cxl_adaptive_growth_debt_repay_fraction\": " << m.cxl_adaptive_growth_debt_repay_fraction << ", ";
            oss << "\"cxl_adaptive_structural_size_ratio\": " << m.cxl_adaptive_structural_size_ratio << ", ";
            oss << "\"cxl_adaptive_partition_count_slack\": " << m.cxl_adaptive_partition_count_slack << ", ";
            oss << "\"cxl_adaptive_max_reassign_budget\": " << m.cxl_adaptive_max_reassign_budget << ", ";
            oss << "\"cxl_adaptive_observed_cost_enabled\": " << (m.cxl_adaptive_observed_cost_enabled ? "true" : "false") << ", ";
            oss << "\"cxl_adaptive_observed_cost_alpha\": " << m.cxl_adaptive_observed_cost_alpha << ", ";
            oss << "\"cxl_adaptive_maintenance_time_budget_fraction\": " << m.cxl_adaptive_maintenance_time_budget_fraction << ", ";
            oss << "\"cxl_adaptive_warmup_split_budget\": " << m.cxl_adaptive_warmup_split_budget << ", ";
            oss << "\"cxl_adaptive_refinement_mode\": \"" << m.cxl_adaptive_refinement_mode << "\", ";
            oss << "\"cxl_adaptive_max_refine_splits\": " << m.cxl_adaptive_max_refine_splits << ", ";
            oss << "\"cxl_adaptive_max_payback_windows\": " << m.cxl_adaptive_max_payback_windows << ", ";
            oss << "\"cxl_adaptive_aps_feedback_enabled\": " << (m.cxl_adaptive_aps_feedback_enabled ? "true" : "false") << ", ";
            oss << "\"cxl_adaptive_reassign_time_budget_enabled\": " << (m.cxl_adaptive_reassign_time_budget_enabled ? "true" : "false") << ", ";
            oss << "\"cxl_adaptive_warmup_reassign_budget\": " << m.cxl_adaptive_warmup_reassign_budget << ", ";
            oss << "\"cxl_streaming_rent_buy\": " << (m.cxl_streaming_rent_buy ? "true" : "false") << ", ";
            oss << "\"cxl_resource_rent_buy\": " << (m.cxl_resource_rent_buy ? "true" : "false") << ", ";
            oss << "\"cxl_search_first\": " << (m.cxl_search_first ? "true" : "false") << ", ";
            oss << "\"cxl_search_first_gain_target\": " << m.cxl_search_first_gain_target << ", ";
            oss << "\"cxl_search_first_max_cohort\": " << m.cxl_search_first_max_cohort << ", ";
            oss << "\"cxl_resource_force_action_set\": " << (m.cxl_resource_force_action_set ? "true" : "false") << ", ";
            oss << "\"cxl_resource_force_window_id\": " << m.cxl_resource_force_window_id;
            oss << "}";
            return oss.str();
        });

    /*********** MaintenanceTimingInfo Binding ***********/
    class_<MaintenanceTimingInfo, shared_ptr<MaintenanceTimingInfo>>(m, "MaintenanceTimingInfo")
         .def_readonly("total_time_us", &MaintenanceTimingInfo::total_time_us,
             "Total time taken for maintenance in microseconds.")
         .def_readonly("split_time_us", &MaintenanceTimingInfo::split_time_us,
             "Time taken for split operations in microseconds.")
         .def_readonly("delete_time_us", &MaintenanceTimingInfo::delete_time_us,
             "Time taken for delete operations in microseconds.")
         .def_readonly("refinement_time_us", &MaintenanceTimingInfo::refinement_time_us,
             "Time taken for refinement of split operations in microseconds.")
         .def_readonly("decision_time_us", &MaintenanceTimingInfo::decision_time_us,
             "Time spent selecting maintenance actions in microseconds.")
         .def_readonly("action_time_us", &MaintenanceTimingInfo::action_time_us,
             "Time spent executing committed maintenance actions in microseconds.")
         .def_readonly("n_splits", &MaintenanceTimingInfo::n_splits,
             "Number of partition split operations performed.")
         .def_readonly("n_deletes", &MaintenanceTimingInfo::n_deletes,
             "Number of partition delete operations performed.")
         .def_readonly("decision_partition_count", &MaintenanceTimingInfo::decision_partition_count)
         .def_readonly("centroid_update_count", &MaintenanceTimingInfo::centroid_update_count)
         .def_readonly("delete_candidate_count", &MaintenanceTimingInfo::delete_candidate_count)
         .def_readonly("delete_candidate_records", &MaintenanceTimingInfo::delete_candidate_records)
         .def_readonly("split_candidate_count", &MaintenanceTimingInfo::split_candidate_count)
         .def_readonly("split_candidate_selected_count", &MaintenanceTimingInfo::split_candidate_selected_count)
         .def_readonly("split_candidate_roi_rejected_count", &MaintenanceTimingInfo::split_candidate_roi_rejected_count)
         .def_readonly("split_candidate_budget_rejected_count", &MaintenanceTimingInfo::split_candidate_budget_rejected_count)
         .def_readonly("split_records_read", &MaintenanceTimingInfo::split_records_read)
         .def_readonly("split_records_written", &MaintenanceTimingInfo::split_records_written)
         .def_readonly("reassign_records_read", &MaintenanceTimingInfo::reassign_records_read)
         .def_readonly("reassign_records_written", &MaintenanceTimingInfo::reassign_records_written)
         .def_readonly("refinement_partition_count", &MaintenanceTimingInfo::refinement_partition_count)
         .def_readonly("refinement_records_per_iteration", &MaintenanceTimingInfo::refinement_records_per_iteration)
         .def_readonly("refinement_records_read", &MaintenanceTimingInfo::refinement_records_read)
         .def_readonly("refinement_records_written", &MaintenanceTimingInfo::refinement_records_written)
         .def_readonly("refinement_iterations", &MaintenanceTimingInfo::refinement_iterations)
         .def_readonly("refinement_source_split_count", &MaintenanceTimingInfo::refinement_source_split_count)
         .def_readonly("refinement_skipped", &MaintenanceTimingInfo::refinement_skipped)
         .def_readonly("observed_split_ns_per_record", &MaintenanceTimingInfo::observed_split_ns_per_record)
         .def_readonly("observed_refine_ns_per_split", &MaintenanceTimingInfo::observed_refine_ns_per_split)
         .def_readonly("observed_refine_ns_per_parent_record", &MaintenanceTimingInfo::observed_refine_ns_per_parent_record)
         .def_readonly("observed_refine_read_records_per_parent", &MaintenanceTimingInfo::observed_refine_read_records_per_parent)
         .def_readonly("observed_refine_write_records_per_parent", &MaintenanceTimingInfo::observed_refine_write_records_per_parent)
         .def_readonly("observed_reassign_ns_per_record", &MaintenanceTimingInfo::observed_reassign_ns_per_record)
         .def_readonly("estimated_query_window_ns", &MaintenanceTimingInfo::estimated_query_window_ns)
         .def_readonly("maintenance_time_budget_ns", &MaintenanceTimingInfo::maintenance_time_budget_ns)
         .def_readonly("split_budget_by_time", &MaintenanceTimingInfo::split_budget_by_time)
         .def_readonly("reassign_budget_by_time", &MaintenanceTimingInfo::reassign_budget_by_time)
         .def_readonly("aps_average_fanout", &MaintenanceTimingInfo::aps_average_fanout)
         .def_readonly("aps_average_scanned_records", &MaintenanceTimingInfo::aps_average_scanned_records)
         .def_readonly("aps_average_scanned_list_size", &MaintenanceTimingInfo::aps_average_scanned_list_size)
         .def_readonly("aps_fanout_growth", &MaintenanceTimingInfo::aps_fanout_growth)
         .def_readonly("aps_scanned_list_size_pressure", &MaintenanceTimingInfo::aps_scanned_list_size_pressure)
         .def_readonly("payback_windows", &MaintenanceTimingInfo::payback_windows)
         .def_readonly("partition_count_before", &MaintenanceTimingInfo::partition_count_before)
         .def_readonly("partition_count_after", &MaintenanceTimingInfo::partition_count_after)
         .def_readonly("streaming_rent_buy_enabled", &MaintenanceTimingInfo::streaming_rent_buy_enabled)
         .def_readonly("streaming_candidate_count", &MaintenanceTimingInfo::streaming_candidate_count)
         .def_readonly("streaming_selected_count", &MaintenanceTimingInfo::streaming_selected_count)
         .def_readonly("streaming_window_rent_ns", &MaintenanceTimingInfo::streaming_window_rent_ns)
         .def_readonly("streaming_structural_rent_ns", &MaintenanceTimingInfo::streaming_structural_rent_ns)
         .def_readonly("streaming_structural_credit_ns", &MaintenanceTimingInfo::streaming_structural_credit_ns)
         .def_readonly("streaming_selected_buy_ns", &MaintenanceTimingInfo::streaming_selected_buy_ns)
         .def_readonly("streaming_budget_credit_ns", &MaintenanceTimingInfo::streaming_budget_credit_ns)
         .def_readonly("streaming_scan_growth_pressure", &MaintenanceTimingInfo::streaming_scan_growth_pressure)
         .def_readonly("resource_rent_buy_enabled", &MaintenanceTimingInfo::resource_rent_buy_enabled)
         .def_readonly("resource_price_window_id", &MaintenanceTimingInfo::resource_price_window_id)
         .def_readonly("resource_price_window_duration_ns", &MaintenanceTimingInfo::resource_price_window_duration_ns)
         .def_readonly("resource_child_probe_factor", &MaintenanceTimingInfo::resource_child_probe_factor)
         .def_readonly("resource_candidate_count", &MaintenanceTimingInfo::resource_candidate_count)
         .def_readonly("resource_selected_count", &MaintenanceTimingInfo::resource_selected_count)
         .def_readonly("resource_window_rent_ns", &MaintenanceTimingInfo::resource_window_rent_ns)
         .def_readonly("resource_selected_buy_ns", &MaintenanceTimingInfo::resource_selected_buy_ns)
         .def_readonly("resource_split_candidate_count", &MaintenanceTimingInfo::resource_split_candidate_count)
         .def_readonly("resource_best_cohort_size", &MaintenanceTimingInfo::resource_best_cohort_size)
         .def_readonly("resource_best_cohort_credit_ns", &MaintenanceTimingInfo::resource_best_cohort_credit_ns)
         .def_readonly("resource_best_cohort_buy_ns", &MaintenanceTimingInfo::resource_best_cohort_buy_ns)
         .def_readonly("resource_best_cohort_ratio", &MaintenanceTimingInfo::resource_best_cohort_ratio)
         .def_readonly("resource_selected_cohort_size", &MaintenanceTimingInfo::resource_selected_cohort_size)
         .def_readonly("resource_selected_cohort_buy_ns", &MaintenanceTimingInfo::resource_selected_cohort_buy_ns)
         .def_readonly("search_first_enabled", &MaintenanceTimingInfo::search_first_enabled)
         .def_readonly("search_first_gain_target", &MaintenanceTimingInfo::search_first_gain_target)
         .def_readonly("search_first_max_cohort", &MaintenanceTimingInfo::search_first_max_cohort)
         .def_readonly("search_first_available_gain_ns", &MaintenanceTimingInfo::search_first_available_gain_ns)
         .def_readonly("search_first_selected_gain_ns", &MaintenanceTimingInfo::search_first_selected_gain_ns)
         .def_readonly("search_first_selected_gain_fraction", &MaintenanceTimingInfo::search_first_selected_gain_fraction)
         .def_readonly("search_first_cxl_split_candidate_count", &MaintenanceTimingInfo::search_first_cxl_split_candidate_count)
         .def_readonly("search_first_cxl_only_split_candidate_count", &MaintenanceTimingInfo::search_first_cxl_only_split_candidate_count)
         .def_readonly("search_first_selected_cxl_only_split_count", &MaintenanceTimingInfo::search_first_selected_cxl_only_split_count)
         .def_readonly("resource_forced_action_set", &MaintenanceTimingInfo::resource_forced_action_set)
         .def_readonly("resource_policy_decisions", &MaintenanceTimingInfo::resource_policy_decisions)
         .def_readonly("split_lineage", &MaintenanceTimingInfo::split_lineage)
         .def("__repr__", [](const MaintenanceTimingInfo &t) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"total_time_us\": " << t.total_time_us << ", ";
             oss << "\"split_time_us\": " << t.split_time_us << ", ";
             oss << "\"delete_time_us\": " << t.delete_time_us << ", ";
             oss << "\"refinement_time_us\": " << t.refinement_time_us << ", ";
             oss << "\"n_splits\": " << t.n_splits << ", ";
             oss << "\"n_deletes\": " << t.n_deletes << ", ";
             oss << "\"decision_time_us\": " << t.decision_time_us << ", ";
             oss << "\"action_time_us\": " << t.action_time_us << ", ";
             oss << "\"decision_partition_count\": " << t.decision_partition_count << ", ";
             oss << "\"split_records_read\": " << t.split_records_read << ", ";
             oss << "\"reassign_records_read\": " << t.reassign_records_read << ", ";
             oss << "\"partition_count_before\": " << t.partition_count_before << ", ";
             oss << "\"partition_count_after\": " << t.partition_count_after;
             oss << "}";
             return oss.str();
         });

    /*********** ModifyTimingInfo Binding ***********/
    class_<ModifyTimingInfo, shared_ptr<ModifyTimingInfo>>(m, "ModifyTimingInfo")
         .def_readonly("modify_time_us", &ModifyTimingInfo::modify_time_us,
             "Total time taken for the modify operation in microseconds.")
        .def_readonly("input_validation_time_us", &ModifyTimingInfo::input_validation_time_us,
             "Time taken for validation in microseconds.")
         .def_readonly("modify_count", &ModifyTimingInfo::n_vectors)
         .def_readonly("find_partition_time_us", &ModifyTimingInfo::find_partition_time_us,
             "Time taken to find the partition for the modify operation in microseconds.")
         .def("__repr__", [](const ModifyTimingInfo &m) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"modify_count\": " << m.n_vectors << ", ";
             oss << "\"input_validation_time_us\": " << m.input_validation_time_us << ", ";
             oss << "\"modify_time_us\": " << m.modify_time_us << ", ";
             oss << "\"find_partition_time_us\": " << m.find_partition_time_us;
             oss << "}";
             return oss.str();
         });


    // double worker_wait_time_ns = 0; ///< Average worker wait time in nanoseconds.
    // double worker_process_time_ns = 0; ///< Average worker process time in nanoseconds.
    // double worker_process_preamble_time_ns = 0; ///< Average worker process preamble time in nanoseconds.
    // double worker_enqueue_time_ns = 0; ///< Average worker enqueue time in nanoseconds.
    // double worker_job_time_ns = 0; ///< Average worker job time in nanoseconds.
    // double worker_scan_time_ns = 0; ///< Average worker scan time in nanoseconds.

    /*********** SearchTimingInfo Binding ***********/
    class_<SearchTimingInfo, shared_ptr<SearchTimingInfo>>(m, "SearchTimingInfo")
         .def(init<>())
         .def_readwrite("total_time_ns", &SearchTimingInfo::total_time_ns,
             "Total time taken for the search operation in nanoseconds.")
        .def_readwrite("buffer_init_time_ns", &SearchTimingInfo::buffer_init_time_ns,
             "Time spent on initializing buffers in nanoseconds.")
         .def_readwrite("copy_query_time_ns", &SearchTimingInfo::copy_query_time_ns,
                "Time spent on copying query vectors to NUMA buffers in nanoseconds.")
        .def_readwrite("job_enqueue_time_ns", &SearchTimingInfo::job_enqueue_time_ns,
             "Time spent on creating jobs in nanoseconds.")
        .def_readwrite("boundary_distance_time_ns", &SearchTimingInfo::boundary_distance_time_ns,
             "Time spent on computing boundary distances in nanoseconds.")
        .def_readwrite("job_wait_time_ns", &SearchTimingInfo::job_wait_time_ns,
             "Time spent waiting for jobs to complete in nanoseconds.")
        .def_readwrite("result_aggregate_time_ns", &SearchTimingInfo::result_aggregate_time_ns,
             "Time spent on aggregating results in nanoseconds.")
         .def_readwrite("n_queries", &SearchTimingInfo::n_queries,
             "Number of queries performed.")
         .def_readwrite("n_clusters", &SearchTimingInfo::n_clusters,
             "Number of clusters searched.")
         .def_readwrite("partitions_scanned", &SearchTimingInfo::partitions_scanned,
             "Number of partitions scanned.")
         .def_readwrite("scanned_partition_ids", &SearchTimingInfo::scanned_partition_ids,
             "Actual partition ids scanned per query.")
         .def_readwrite("search_params", &SearchTimingInfo::search_params,
             "Parameters used for the search operation.")
         .def_readwrite("parent_info", &SearchTimingInfo::parent_info,
             "Search info for the parent index.")
        .def_readwrite("aps_time_ns", &SearchTimingInfo::aps_time_ns,
            "Time spent on APS in nanoseconds.")
        .def_readwrite("scan_time_ns", &SearchTimingInfo::scan_time_ns,
            "Time spent on scanning in nanoseconds.")
            .def_readwrite("worker_wait_time_ns", &SearchTimingInfo::worker_wait_time_ns,
                "Average worker wait time in nanoseconds.")
        .def_readwrite("worker_process_time_ns", &SearchTimingInfo::worker_process_time_ns,
                 "Average worker process time in nanoseconds.")
        .def_readwrite("worker_process_preamble_time_ns", &SearchTimingInfo::worker_process_preamble_time_ns,
                 "Average worker process preamble time in nanoseconds.")
        .def_readwrite("worker_enqueue_time_ns", &SearchTimingInfo::worker_enqueue_time_ns,
                 "Average worker enqueue time in nanoseconds.")
        .def_readwrite("worker_job_time_ns", &SearchTimingInfo::worker_job_time_ns,
                 "Average worker job time in nanoseconds.")
        .def_readwrite("worker_scan_time_ns", &SearchTimingInfo::worker_scan_time_ns,
                 "Average worker scan time in nanoseconds.")
         .def_readwrite("local_scan_throughput", &SearchTimingInfo::local_scan_throughput,
                 "Average batch scan throughput")
        .def_readwrite("worker_partition_size", &SearchTimingInfo::worker_partition_size,
                 "Average worker partition size")
         .def("__repr__", [](const SearchTimingInfo &s) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"total_time_ns\": " << s.total_time_ns << ", ";
             oss << "\"buffer_init_time_ns\": " << s.buffer_init_time_ns << ", ";
             oss << "\"job_enqueue_time_ns\": " << s.job_enqueue_time_ns << ", ";
             oss << "\"boundary_distance_time_ns\": " << s.boundary_distance_time_ns << ", ";
             oss << "\"job_wait_time_ns\": " << s.job_wait_time_ns << ", ";
             oss << "\"result_aggregate_time_ns\": " << s.result_aggregate_time_ns << ", ";
             if (s.parent_info != nullptr) {
                 oss << "\"parent_scan_time_ns\": " << s.parent_info->total_time_ns << ", ";
             }
             oss << "\"n_queries\": " << s.n_queries << ", ";
             oss << "\"n_clusters\": " << s.n_clusters << ", ";
             oss << "\"partitions_scanned\": " << s.partitions_scanned << ", ";
             oss << "\"scanned_partition_id_rows\": " << s.scanned_partition_ids.size();
             oss << "}";
             return oss.str();
         });

    /**************** BuildTimingInfo Binding ***********/
    class_<BuildTimingInfo, shared_ptr<BuildTimingInfo>>(m, "BuildTimingInfo")
         .def_readonly("total_time_us", &BuildTimingInfo::total_time_us,
            "Total time taken for the build operation in microseconds.")
         .def_readonly("assign_time_us", &BuildTimingInfo::assign_time_us,
            "Time taken for assignment in microseconds.")
         .def_readonly("train_time_us", &BuildTimingInfo::train_time_us,
            "Time taken for training in microseconds.")
         .def_readonly("d", &BuildTimingInfo::d,
            "Dimension of the vectors.")
         .def_readonly("code_size", &BuildTimingInfo::code_size,
            "Size of PQ codes in bytes.")
         .def_readonly("n_codebooks", &BuildTimingInfo::num_codebooks,
            "Number of codebooks in the index.")
         .def_readonly("n_vectors", &BuildTimingInfo::n_vectors,
            "Number of vectors in the index.")
         .def("__repr__", [](const BuildTimingInfo &b) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"total_time_us\": " << b.total_time_us << ", ";
             oss << "\"assign_time_us\": " << b.assign_time_us << ", ";
             oss << "\"train_time_us\": " << b.train_time_us << ", ";
             oss << "\"d\": " << b.d << ", ";
             oss << "\"code_size\": " << b.code_size << ", ";
             oss << "\"n_codebooks\": " << b.num_codebooks << ", ";
             oss << "\"n_vectors\": " << b.n_vectors;
             oss << "}";
             return oss.str();
         });

    /************* SearchResult Binding ***********/
    class_<SearchResult, shared_ptr<SearchResult>>(m, "SearchResult")
         .def(init<>())
         .def_readwrite("distances", &SearchResult::distances,
             "Distances to the nearest neighbors.")
         .def_readwrite("ids", &SearchResult::ids,
             "Indices of the nearest neighbors.")
         .def_readwrite("timing_info", &SearchResult::timing_info,
             "Timing information for the search operation.")
         .def("__repr__", [](const SearchResult &r) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"num_ids\": " << r.ids.numel() << ", ";
             oss << "\"num_distances\": " << r.distances.numel();
             oss << "}";
             return oss.str();
         });
}

#endif //QUAKE_WRAP_H
