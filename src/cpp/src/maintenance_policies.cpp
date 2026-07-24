#include "maintenance_policies.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <torch/torch.h>

#include "quake_index.h"

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::vector;
using std::unordered_map;
using std::shared_ptr;

namespace {

float round_up_line_bytes(int64_t records, int entry_bytes, int line_bytes) {
    int64_t safe_records = std::max<int64_t>(0, records);
    int safe_entry_bytes = std::max(1, entry_bytes);
    int safe_line_bytes = std::max(1, line_bytes);
    double bytes = static_cast<double>(safe_records) * static_cast<double>(safe_entry_bytes);
    return static_cast<float>(std::ceil(bytes / static_cast<double>(safe_line_bytes)) *
                              static_cast<double>(safe_line_bytes));
}

float cxl_bottleneck_bw_bytes_per_ns(const MaintenancePolicyParams& params) {
    float mc_bw = std::max(1.0e-6f, params.cxl_mc_bw_bytes_per_ns);
    float link_bw = std::max(1.0e-6f, params.cxl_link_bw_bytes_per_ns);
    float aggregate_mc_bw = std::max(1, params.cxl_num_mcs) * mc_bw;
    if (params.cxl_scan_mode == "host_scan") {
        return std::min(link_bw, aggregate_mc_bw);
    }
    return aggregate_mc_bw;
}

float cxl_scan_benefit_ns(const MaintenancePolicyParams& params,
                          int partition_size,
                          float hit_rate) {
    float line_bytes = round_up_line_bytes(
        partition_size, params.cxl_entry_bytes, params.cxl_line_bytes);
    return std::max(0.0f, hit_rate) * line_bytes / cxl_bottleneck_bw_bytes_per_ns(params);
}

float cxl_maintenance_penalty_ns(const MaintenancePolicyParams& params,
                                 int64_t action_records) {
    float background_fraction = std::max(1.0e-6f, params.cxl_maintenance_bandwidth_fraction);
    float background_bw = cxl_bottleneck_bw_bytes_per_ns(params) * background_fraction;
    float line_bytes = round_up_line_bytes(
        action_records, params.cxl_entry_bytes, params.cxl_line_bytes);
    return line_bytes / background_bw;
}

float cxl_window_amortized_penalty_ns(const MaintenancePolicyParams& params,
                                      int64_t action_records) {
    int window = std::max(1, params.window_size);
    return cxl_maintenance_penalty_ns(params, action_records) / static_cast<float>(window);
}

float cxl_fanout_penalty_ns(const MaintenancePolicyParams& params,
                            float hit_rate) {
    return std::max(0.0f, hit_rate) * std::max(0.0f, params.cxl_fanout_penalty_ns);
}

float clamp01(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

int64_t source_order_run_count(vector<int64_t> positions) {
    if (positions.empty()) {
        return 0;
    }
    std::sort(positions.begin(), positions.end());
    int64_t runs = 1;
    for (size_t index = 1; index < positions.size(); ++index) {
        if (positions[index] != positions[index - 1] + 1) {
            runs++;
        }
    }
    return runs;
}

int64_t source_order_gather_line_bytes(vector<int64_t> positions,
                                       int entry_bytes,
                                       int line_bytes) {
    if (positions.empty()) {
        return 0;
    }
    std::sort(positions.begin(), positions.end());
    const int64_t safe_entry_bytes = std::max(1, entry_bytes);
    const int64_t safe_line_bytes = std::max(1, line_bytes);
    int64_t touched_lines = 0;
    int64_t current_first = -1;
    int64_t current_last = -1;
    for (int64_t position : positions) {
        const int64_t byte_first = std::max<int64_t>(0, position) * safe_entry_bytes;
        const int64_t byte_last = byte_first + safe_entry_bytes - 1;
        const int64_t first_line = byte_first / safe_line_bytes;
        const int64_t last_line = byte_last / safe_line_bytes;
        if (current_first < 0) {
            current_first = first_line;
            current_last = last_line;
        } else if (first_line <= current_last + 1) {
            current_last = std::max(current_last, last_line);
        } else {
            touched_lines += current_last - current_first + 1;
            current_first = first_line;
            current_last = last_line;
        }
    }
    touched_lines += current_last - current_first + 1;
    return touched_lines * safe_line_bytes;
}

}  // namespace


MaintenancePolicy::MaintenancePolicy(
    shared_ptr<PartitionManager> partition_manager,
    shared_ptr<MaintenancePolicyParams> params)
    : partition_manager_(partition_manager),
      params_(params) {
    // Initialize the cost estimator.
    cost_estimator_ = std::make_shared<MaintenanceCostEstimator>(
        partition_manager_->d(), // Assumes PartitionManager::get_dimension() exists.
        params_->alpha,
        10,
        params_->latency_profile_path);
    // Initialize the hit count tracker using the window size and total vector count.
    hit_count_tracker_ = std::make_shared<HitCountTracker>(
        params_->window_size, partition_manager_->ntotal());
    int64_t initial_partitions = std::max<int64_t>(1, partition_manager_->nlist());
    int64_t configured_target = std::max<int64_t>(0, params_->cxl_adaptive_target_partition_size);
    cxl_target_partition_size_ = static_cast<int>(
        configured_target > 0
            ? configured_target
            : std::max<int64_t>(
                  1,
                  (partition_manager_->ntotal() + initial_partitions - 1) / initial_partitions));
}

void MaintenancePolicy::update_cxl_adaptive_state(int64_t current_ntotal,
                                                  int64_t current_partition_count,
                                                  float current_scan_fraction,
                                                  float current_aps_fanout,
                                                  float current_avg_scanned_list_size) {
    if (!params_->enable_cxl_cost_model ||
        (!params_->cxl_workload_adaptive && !cxl_use_resource_rent_buy())) {
        cxl_prev_ntotal_ = current_ntotal;
        cxl_prev_partition_count_ = current_partition_count;
        return;
    }
    float alpha = clamp01(params_->cxl_adaptive_ewma_alpha);
    if (alpha <= 0.0f) {
        alpha = DEFAULT_CXL_ADAPTIVE_EWMA_ALPHA;
    }
    if (cxl_prev_ntotal_ < 0 || cxl_prev_partition_count_ < 0) {
        cxl_prev_ntotal_ = current_ntotal;
        cxl_prev_partition_count_ = current_partition_count;
    }

    float previous_total = static_cast<float>(std::max<int64_t>(1, cxl_prev_ntotal_));
    float delta_vectors = static_cast<float>(current_ntotal - cxl_prev_ntotal_);
    int64_t pending_adds = std::max<int64_t>(0, cxl_pending_add_count_);
    int64_t pending_deletes = std::max<int64_t>(0, cxl_pending_delete_count_);
    if (pending_adds == 0 && pending_deletes == 0) {
        pending_adds = static_cast<int64_t>(std::max(0.0f, delta_vectors));
        pending_deletes = static_cast<int64_t>(std::max(0.0f, -delta_vectors));
    }
    float growth_sample = clamp01(static_cast<float>(pending_adds) / previous_total);
    float delete_sample = clamp01(static_cast<float>(pending_deletes) / previous_total);
    float churn_sample = clamp01(static_cast<float>(pending_adds + pending_deletes) / previous_total);
    float partition_delta = static_cast<float>(current_partition_count - cxl_prev_partition_count_);
    float fanout_sample = clamp01(std::max(0.0f, partition_delta) /
                                  static_cast<float>(std::max<int64_t>(1, cxl_prev_partition_count_)));
    float maintenance_sample = clamp01(
        static_cast<float>(std::max<int64_t>(0, cxl_prev_maintenance_records_)) /
        static_cast<float>(std::max<int64_t>(1, current_ntotal)));
    float fanout_growth_sample = 0.0f;
    float scanned_list_size_growth_sample = 0.0f;
    float current_avg_scanned_records =
        clamp01(current_scan_fraction) * static_cast<float>(std::max<int64_t>(0, current_ntotal));
    float scan_growth_sample = 0.0f;
    if (cxl_prev_aps_fanout_ > 0.0f) {
        fanout_growth_sample = clamp01(
            std::max(0.0f, current_aps_fanout - cxl_prev_aps_fanout_) /
            cxl_prev_aps_fanout_);
    }
    if (cxl_prev_avg_scanned_list_size_ > 0.0f) {
        scanned_list_size_growth_sample = clamp01(
            std::max(0.0f, current_avg_scanned_list_size - cxl_prev_avg_scanned_list_size_) /
            cxl_prev_avg_scanned_list_size_);
    }
    if (cxl_prev_avg_scanned_records_ > 0.0f) {
        scan_growth_sample = clamp01(
            std::max(0.0f, current_avg_scanned_records - cxl_prev_avg_scanned_records_) /
            cxl_prev_avg_scanned_records_);
    }
    float scanned_list_size_level = clamp01(
        std::max(0.0f,
                 current_avg_scanned_list_size /
                         static_cast<float>(std::max(1, cxl_target_partition_size_)) -
                     1.0f));
    float scanned_list_size_sample = std::max(
        scanned_list_size_level,
        scanned_list_size_growth_sample);

    cxl_growth_score_ = (1.0f - alpha) * cxl_growth_score_ + alpha * growth_sample;
    cxl_delete_score_ = (1.0f - alpha) * cxl_delete_score_ + alpha * delete_sample;
    cxl_churn_score_ = (1.0f - alpha) * cxl_churn_score_ + alpha * churn_sample;
    cxl_scan_pressure_ = (1.0f - alpha) * cxl_scan_pressure_ + alpha * clamp01(current_scan_fraction);
    cxl_maintenance_pressure_ = (1.0f - alpha) * cxl_maintenance_pressure_ + alpha * maintenance_sample;
    float selected_fanout_sample = cxl_use_aps_feedback() ? fanout_growth_sample : fanout_sample;
    cxl_fanout_pressure_ = (1.0f - alpha) * cxl_fanout_pressure_ + alpha * selected_fanout_sample;
    cxl_scanned_list_size_pressure_ =
        (1.0f - alpha) * cxl_scanned_list_size_pressure_ + alpha * scanned_list_size_sample;
    cxl_scan_growth_score_ =
        (1.0f - alpha) * cxl_scan_growth_score_ + alpha * scan_growth_sample;
    cxl_last_growth_sample_ = growth_sample;
    cxl_last_delete_sample_ = delete_sample;
    cxl_last_churn_sample_ = churn_sample;
    cxl_last_aps_fanout_ = std::max(0.0f, current_aps_fanout);
    cxl_last_avg_scanned_list_size_ = std::max(0.0f, current_avg_scanned_list_size);
    cxl_last_fanout_growth_sample_ = fanout_growth_sample;
    cxl_last_scan_growth_sample_ = scan_growth_sample;

    float query_scan_bytes = round_up_line_bytes(
        static_cast<int64_t>(std::ceil(
            static_cast<double>(std::max<int64_t>(0, current_ntotal)) *
            static_cast<double>(clamp01(current_scan_fraction)))),
        params_->cxl_entry_bytes,
        params_->cxl_line_bytes);
    float query_scan_ns = query_scan_bytes / cxl_bottleneck_bw_bytes_per_ns(*params_);
    cxl_estimated_query_window_ns_ = std::max(
        1.0f,
        query_scan_ns * static_cast<float>(std::max(1, params_->window_size)));
    if (cxl_use_resource_rent_buy() && cxl_resource_price_snapshot_ != nullptr &&
        cxl_resource_price_snapshot_->window_duration_ns > 0) {
        cxl_estimated_query_window_ns_ = static_cast<float>(
            cxl_resource_price_snapshot_->window_duration_ns);
    }
    if (cxl_use_streaming_rent_buy() || cxl_use_resource_rent_buy()) {
        cxl_payback_windows_ = 1.0f;
        cxl_maintenance_time_budget_ns_ = cxl_estimated_query_window_ns_ * std::max(
            0.0f, params_->cxl_adaptive_maintenance_time_budget_fraction);
        cxl_maintenance_budget_credit_ns_ += cxl_maintenance_time_budget_ns_;
    } else {
        float max_payback_windows = std::max(1.0f, params_->cxl_adaptive_max_payback_windows);
        float update_pressure = std::max(cxl_last_churn_sample_, cxl_churn_score_);
        cxl_payback_windows_ = update_pressure > 1.0e-6f
                                   ? std::min(max_payback_windows, std::max(1.0f, 1.0f / update_pressure))
                                   : max_payback_windows;
        cxl_maintenance_time_budget_ns_ = cxl_estimated_query_window_ns_ * std::max(
            0.0f,
            params_->cxl_adaptive_maintenance_time_budget_fraction) * cxl_payback_windows_;
    }

    cxl_prev_ntotal_ = current_ntotal;
    cxl_prev_partition_count_ = current_partition_count;
    cxl_prev_aps_fanout_ = std::max(0.0f, current_aps_fanout);
    cxl_prev_avg_scanned_list_size_ = std::max(0.0f, current_avg_scanned_list_size);
    cxl_prev_avg_scanned_records_ = std::max(0.0f, current_avg_scanned_records);
    cxl_pending_add_count_ = 0;
    cxl_pending_delete_count_ = 0;
}

float MaintenancePolicy::cxl_adaptive_split_score_weight() const {
    float weight = params_->cxl_split_score_weight;
    if (!params_->cxl_workload_adaptive) {
        return weight;
    }
    float growth_pressure = std::max(cxl_growth_score_, cxl_last_growth_sample_);
    float delete_pressure = std::max(cxl_delete_score_, cxl_last_delete_sample_);
    float churn_pressure = std::max(cxl_churn_score_, cxl_last_churn_sample_);
    if (cxl_use_aps_feedback()) {
        return std::max(
            0.0f,
            weight * (1.0f
                      + params_->cxl_adaptive_growth_split_gain * growth_pressure
                      + params_->cxl_adaptive_scan_split_gain * cxl_scanned_list_size_pressure_
                      - params_->cxl_adaptive_delete_relief_gain * delete_pressure));
    }
    return std::max(
        0.0f,
        weight * (1.0f
                  + params_->cxl_adaptive_growth_split_gain * growth_pressure
                  + params_->cxl_adaptive_churn_split_gain * churn_pressure
                  + params_->cxl_adaptive_scan_split_gain * cxl_scan_pressure_
                  + params_->cxl_adaptive_delete_relief_gain * delete_pressure));
}

float MaintenancePolicy::cxl_adaptive_maintenance_penalty_weight() const {
    float weight = params_->cxl_maintenance_penalty_weight;
    if (!params_->cxl_workload_adaptive) {
        return weight;
    }
    float growth_pressure = std::max(cxl_growth_score_, cxl_last_growth_sample_);
    float delete_pressure = std::max(cxl_delete_score_, cxl_last_delete_sample_);
    float churn_pressure = std::max(cxl_churn_score_, cxl_last_churn_sample_);
    return std::max(
        0.05f,
        weight * (1.0f
                  + params_->cxl_adaptive_maintenance_penalty_gain * cxl_maintenance_pressure_
                  + 0.25f * growth_pressure
                  - 0.75f * churn_pressure
                  - params_->cxl_adaptive_delete_relief_gain * delete_pressure));
}

float MaintenancePolicy::cxl_adaptive_delete_penalty_weight() const {
    float weight = cxl_adaptive_maintenance_penalty_weight();
    if (!params_->cxl_workload_adaptive) {
        return weight;
    }
    float delete_pressure = std::max(cxl_delete_score_, cxl_last_delete_sample_);
    return std::max(
        0.05f,
        weight * (1.0f - params_->cxl_adaptive_delete_relief_gain * delete_pressure));
}

float MaintenancePolicy::cxl_adaptive_fanout_penalty_weight() const {
    float weight = params_->cxl_fanout_penalty_weight;
    if (!params_->cxl_workload_adaptive) {
        return weight;
    }
    float growth_pressure = std::max(cxl_growth_score_, cxl_last_growth_sample_);
    float delete_pressure = std::max(cxl_delete_score_, cxl_last_delete_sample_);
    float churn_pressure = std::max(cxl_churn_score_, cxl_last_churn_sample_);
    if (cxl_use_aps_feedback()) {
        return std::max(
            0.0f,
            weight * (1.0f
                      + params_->cxl_adaptive_fanout_growth_gain *
                            (cxl_fanout_pressure_ + cxl_last_fanout_growth_sample_)));
    }
    return std::max(
        0.0f,
        weight * (1.0f
                  + params_->cxl_adaptive_fanout_growth_gain * growth_pressure
                  + cxl_fanout_pressure_
                  - churn_pressure
                  - params_->cxl_adaptive_delete_relief_gain * delete_pressure));
}

bool MaintenancePolicy::cxl_use_roi_budget() const {
    return params_->enable_cxl_cost_model &&
           params_->cxl_workload_adaptive &&
           params_->cxl_workload_adaptive_v2;
}

bool MaintenancePolicy::cxl_use_structural_budget() const {
    return cxl_use_roi_budget();
}

bool MaintenancePolicy::cxl_use_aps_feedback() const {
    return cxl_use_roi_budget() && params_->cxl_adaptive_aps_feedback_enabled;
}

bool MaintenancePolicy::cxl_use_streaming_rent_buy() const {
    return params_->enable_cxl_cost_model && params_->cxl_streaming_rent_buy;
}

bool MaintenancePolicy::cxl_use_resource_rent_buy() const {
    return params_->enable_cxl_cost_model &&
           (params_->cxl_resource_rent_buy || params_->cxl_search_first ||
            params_->cxl_streaming_staged || params_->cxl_default_plus ||
            params_->cxl_search_guarded_plus);
}

bool MaintenancePolicy::cxl_use_streaming_staged() const {
    return params_->enable_cxl_cost_model && params_->cxl_streaming_staged;
}

bool MaintenancePolicy::cxl_use_search_first() const {
    return params_->enable_cxl_cost_model && params_->cxl_search_first;
}

bool MaintenancePolicy::cxl_use_default_plus() const {
    return params_->enable_cxl_cost_model && params_->cxl_default_plus;
}

bool MaintenancePolicy::cxl_use_search_guarded_plus() const {
    return params_->enable_cxl_cost_model &&
           params_->cxl_search_guarded_plus;
}

bool MaintenancePolicy::cxl_use_observed_action_cost() const {
    return (cxl_use_roi_budget() || cxl_use_streaming_rent_buy() ||
            cxl_use_resource_rent_buy()) &&
           params_->cxl_adaptive_observed_cost_enabled;
}

void MaintenancePolicy::set_cxl_resource_price_snapshot(
    shared_ptr<CxlResourcePriceSnapshot> snapshot) {
    if (snapshot == nullptr || !snapshot->valid) {
        throw std::invalid_argument(
            "CxlResourcePriceSnapshot must be non-null and valid.");
    }
    if (snapshot->home_read_byte_price_ns.empty() ||
        snapshot->home_read_byte_price_ns.size() !=
            snapshot->home_read_op_price_ns.size()) {
        throw std::invalid_argument(
            "CxlResourcePriceSnapshot requires equal non-empty home price vectors.");
    }
    cxl_resource_price_snapshot_ = std::move(snapshot);
}

int MaintenancePolicy::cxl_resource_home_id(int64_t partition_id) const {
    if (cxl_resource_price_snapshot_ == nullptr) {
        return -1;
    }
    auto explicit_home =
        cxl_resource_price_snapshot_->list_home_ids.find(partition_id);
    if (explicit_home != cxl_resource_price_snapshot_->list_home_ids.end()) {
        int home_count = static_cast<int>(
            cxl_resource_price_snapshot_->home_read_byte_price_ns.size());
        if (explicit_home->second >= 0 && explicit_home->second < home_count) {
            return explicit_home->second;
        }
    }
    int home_count = static_cast<int>(
        cxl_resource_price_snapshot_->home_read_byte_price_ns.size());
    int64_t positive_id = partition_id >= 0 ? partition_id : -partition_id;
    return static_cast<int>(positive_id % std::max(1, home_count));
}

float MaintenancePolicy::cxl_resource_scan_cost_ns(
    int64_t partition_id,
    int64_t records) const {
    if (cxl_resource_price_snapshot_ == nullptr) {
        return 0.0f;
    }
    int home_id = cxl_resource_home_id(partition_id);
    if (home_id < 0 ||
        home_id >= static_cast<int>(
                       cxl_resource_price_snapshot_->home_read_byte_price_ns.size())) {
        return 0.0f;
    }
    float line_bytes = round_up_line_bytes(
        records, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    return line_bytes *
               std::max(0.0f,
                        cxl_resource_price_snapshot_
                            ->home_read_byte_price_ns[home_id]) +
           std::max(0.0f,
                    cxl_resource_price_snapshot_->home_read_op_price_ns[home_id]);
}

float MaintenancePolicy::cxl_resource_balanced_birth_scan_cost_ns(
    int64_t records) const {
    if (cxl_resource_price_snapshot_ == nullptr ||
        cxl_resource_price_snapshot_->home_read_byte_price_ns.empty()) {
        return 0.0f;
    }
    float line_bytes = round_up_line_bytes(
        records, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float total_ns = 0.0f;
    size_t home_count =
        cxl_resource_price_snapshot_->home_read_byte_price_ns.size();
    for (size_t home_id = 0; home_id < home_count; home_id++) {
        total_ns += line_bytes * std::max(
                                     0.0f,
                                     cxl_resource_price_snapshot_
                                         ->home_read_byte_price_ns[home_id]) +
                    std::max(
                        0.0f,
                        cxl_resource_price_snapshot_
                            ->home_read_op_price_ns[home_id]);
    }
    return total_ns / static_cast<float>(home_count);
}

float MaintenancePolicy::cxl_resource_split_variable_buy_ns(
    int64_t partition_id,
    int partition_size) const {
    float line_bytes = round_up_line_bytes(
        partition_size, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float traffic_ns = line_bytes *
        (std::max(0.0f, cxl_resource_price_snapshot_->maintenance_read_byte_price_ns) +
         std::max(0.0f, cxl_resource_price_snapshot_->maintenance_write_byte_price_ns));
    float metadata_bytes = static_cast<float>(
        std::max(0, params_->cxl_metadata_bytes) * 2);
    traffic_ns += metadata_bytes * std::max(
        0.0f,
        cxl_resource_price_snapshot_->routing_metadata_shadow_price_ns_per_byte);

    float compute_ns = 0.0f;
    if (cxl_use_observed_action_cost() && cxl_has_observed_split_cost_) {
        float observed_ns =
            cxl_observed_split_ns_per_record_ *
            static_cast<float>(std::max(1, partition_size));
        float old_transfer_ns = 2.0f * line_bytes /
            (cxl_bottleneck_bw_bytes_per_ns(*params_) *
             std::max(1.0e-6f, params_->cxl_maintenance_bandwidth_fraction));
        compute_ns = std::max(0.0f, observed_ns - old_transfer_ns);
    }
    (void)partition_id;
    return std::max(1.0f, compute_ns + traffic_ns);
}

float MaintenancePolicy::cxl_resource_split_cohort_shared_buy_ns(
    int64_t parent_records) const {
    float fixed_ns = std::max(0.0f, params_->cxl_adaptive_action_overhead_ns);
    if (!cxl_use_observed_action_cost() || !cxl_has_observed_split_cost_ ||
        parent_records <= 0 || cxl_observed_refine_ns_per_parent_record_ <= 0.0f) {
        return std::max(1.0f, fixed_ns);
    }

    float parent_count = static_cast<float>(std::max<int64_t>(1, parent_records));
    int64_t refine_read_records = static_cast<int64_t>(std::ceil(
        parent_count * std::max(0.0f, cxl_observed_refine_read_records_per_parent_)));
    int64_t refine_write_records = static_cast<int64_t>(std::ceil(
        parent_count * std::max(0.0f, cxl_observed_refine_write_records_per_parent_)));
    float read_line_bytes = round_up_line_bytes(
        refine_read_records, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float write_line_bytes = round_up_line_bytes(
        refine_write_records, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float traffic_ns =
        read_line_bytes * std::max(
            0.0f, cxl_resource_price_snapshot_->maintenance_read_byte_price_ns) +
        write_line_bytes * std::max(
            0.0f, cxl_resource_price_snapshot_->maintenance_write_byte_price_ns);
    float observed_ns = cxl_observed_refine_ns_per_parent_record_ * parent_count;
    float old_transfer_ns = (read_line_bytes + write_line_bytes) /
        (cxl_bottleneck_bw_bytes_per_ns(*params_) *
         std::max(1.0e-6f, params_->cxl_maintenance_bandwidth_fraction));
    float residual_compute_ns = std::max(0.0f, observed_ns - old_transfer_ns);
    return std::max(1.0f, fixed_ns + residual_compute_ns + traffic_ns);
}

float MaintenancePolicy::cxl_resource_reassign_buy_ns(
    int64_t partition_id,
    int partition_size) const {
    float line_bytes = round_up_line_bytes(
        partition_size, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float traffic_ns = line_bytes *
        (std::max(0.0f, cxl_resource_price_snapshot_->maintenance_read_byte_price_ns) +
         std::max(0.0f, cxl_resource_price_snapshot_->maintenance_write_byte_price_ns));
    float compute_ns = std::max(0.0f, params_->cxl_adaptive_action_overhead_ns);
    if (cxl_use_observed_action_cost() &&
        cxl_observed_reassign_ns_per_record_ > 0.0f) {
        float observed_ns = cxl_observed_reassign_ns_per_record_ *
                            static_cast<float>(std::max(1, partition_size));
        float old_transfer_ns = 2.0f * line_bytes /
            (cxl_bottleneck_bw_bytes_per_ns(*params_) *
             std::max(1.0e-6f, params_->cxl_maintenance_bandwidth_fraction));
        compute_ns = std::max(compute_ns, observed_ns - old_transfer_ns);
    }
    (void)partition_id;
    return std::max(1.0f, compute_ns + traffic_ns);
}

float MaintenancePolicy::cxl_resource_reassign_rent_ns(
    int64_t partition_id,
    int partition_size,
    int source_hit_count,
    const vector<int64_t>& reassign_ids,
    const vector<int64_t>& reassign_counts,
    const vector<int64_t>& reassign_sizes,
    const vector<float>& hit_rates,
    int64_t recorded_queries,
    float* cost_before_ns,
    float* cost_after_ns) const {
    float before_ns = static_cast<float>(std::max(0, source_hit_count)) *
                      cxl_resource_scan_cost_ns(partition_id, partition_size);
    float after_ns = 0.0f;
    int64_t assigned_total = std::accumulate(
        reassign_counts.begin(), reassign_counts.end(), int64_t{0});
    for (size_t index = 0; index < reassign_ids.size(); index++) {
        int64_t target_id = reassign_ids[index];
        int64_t target_size = index < reassign_sizes.size()
                                  ? std::max<int64_t>(0, reassign_sizes[index])
                                  : 0;
        int64_t moved = index < reassign_counts.size()
                            ? std::max<int64_t>(0, reassign_counts[index])
                            : 0;
        float target_hits = index < hit_rates.size()
                                ? std::max(0.0f, hit_rates[index]) *
                                      static_cast<float>(recorded_queries)
                                : 0.0f;
        before_ns += target_hits *
                     cxl_resource_scan_cost_ns(target_id, target_size);
        float source_fraction = assigned_total > 0
                                    ? static_cast<float>(moved) /
                                          static_cast<float>(assigned_total)
                                    : 0.0f;
        float after_hits = target_hits +
                           static_cast<float>(std::max(0, source_hit_count)) *
                               source_fraction;
        after_ns += after_hits * cxl_resource_scan_cost_ns(
                                     target_id, target_size + moved);
    }
    if (cost_before_ns != nullptr) {
        *cost_before_ns = before_ns;
    }
    if (cost_after_ns != nullptr) {
        *cost_after_ns = after_ns;
    }
    return before_ns - after_ns;
}

void MaintenancePolicy::update_cxl_child_probe_factor(
    const vector<vector<int64_t>>& per_query_hits,
    const std::unordered_set<int64_t>& active_partition_ids) {
    if (cxl_resource_child_probe_factor_ < 0.0f) {
        cxl_resource_child_probe_factor_ =
            std::min(2.0f, std::max(1.0f, 2.0f * params_->alpha));
    }
    vector<std::pair<int64_t, int64_t>> active_pairs;
    double child_hits = 0.0;
    double parent_equivalent_hits = 0.0;
    for (const auto& pair : cxl_split_sibling_pairs_) {
        if (active_partition_ids.count(pair.first) == 0 ||
            active_partition_ids.count(pair.second) == 0) {
            continue;
        }
        active_pairs.push_back(pair);
        for (const auto& query_hits : per_query_hits) {
            bool first = std::find(query_hits.begin(), query_hits.end(), pair.first) !=
                         query_hits.end();
            bool second = std::find(query_hits.begin(), query_hits.end(), pair.second) !=
                          query_hits.end();
            if (first || second) {
                parent_equivalent_hits += 1.0;
                child_hits += static_cast<double>(first) +
                              static_cast<double>(second);
            }
        }
    }
    cxl_split_sibling_pairs_.swap(active_pairs);
    if (parent_equivalent_hits > 0.0) {
        float observed = static_cast<float>(child_hits / parent_equivalent_hits);
        observed = std::min(2.0f, std::max(1.0f, observed));
        cxl_resource_child_probe_factor_ =
            0.8f * cxl_resource_child_probe_factor_ + 0.2f * observed;
    }
}

MaintenancePolicy::StreamingSplitRent MaintenancePolicy::cxl_streaming_split_rent_ns(
    int partition_size,
    float hit_rate,
    float native_split_delta,
    int64_t recorded_queries) {
    float safe_hit_rate = std::max(0.0f, hit_rate);
    float bandwidth = cxl_bottleneck_bw_bytes_per_ns(*params_);
    float before_ns = safe_hit_rate *
                      (round_up_line_bytes(partition_size, params_->cxl_entry_bytes,
                                           params_->cxl_line_bytes) /
                           bandwidth +
                       std::max(0.0f, params_->cxl_fanout_penalty_ns));
    float child_probe_factor = std::max(0.0f, 2.0f * params_->alpha);
    int child_size = std::max(1, (partition_size + 1) / 2);
    float after_ns = safe_hit_rate * child_probe_factor *
                     (round_up_line_bytes(child_size, params_->cxl_entry_bytes,
                                          params_->cxl_line_bytes) /
                          bandwidth +
                      std::max(0.0f, params_->cxl_fanout_penalty_ns));
    float cxl_gain_per_query = std::max(0.0f, before_ns - after_ns);
    float native_gain_per_query = std::max(0.0f, -native_split_delta);
    // Native Quake and the CXL line model overlap in their scan component. Use
    // the stronger measured signal rather than summing and double-counting it.
    float rent_per_query = std::max(native_gain_per_query, cxl_gain_per_query);

    // A local split estimator cannot see that APS is scanning progressively
    // more records as a growing index drifts away from its initial partition
    // size. Attribute that observed global rent only to hot, oversized lists.
    // Net growth catches the first update window; scanned-record growth takes
    // over once two completed query windows are available. Delete-dominant
    // streams therefore do not manufacture structural split pressure.
    float net_growth_pressure = std::max(
        0.0f,
        std::max(cxl_growth_score_, cxl_last_growth_sample_) -
            std::max(cxl_delete_score_, cxl_last_delete_sample_));
    float scan_growth_pressure = std::max(cxl_scan_growth_score_, cxl_last_scan_growth_sample_);
    float structural_pressure = clamp01(std::max(net_growth_pressure, scan_growth_pressure));
    int excess_records = std::max(0, partition_size - cxl_target_partition_size_);
    float structural_gain_per_query = safe_hit_rate * structural_pressure *
                                      (round_up_line_bytes(excess_records, params_->cxl_entry_bytes,
                                                           params_->cxl_line_bytes) /
                                       bandwidth);
    float structural_rent_ns = structural_gain_per_query *
                               static_cast<float>(std::max<int64_t>(1, recorded_queries));
    cxl_streaming_structural_rent_ns_ += structural_rent_ns;
    return {
        rent_per_query * static_cast<float>(std::max<int64_t>(1, recorded_queries)),
        structural_rent_ns,
    };
}

float MaintenancePolicy::cxl_streaming_split_buy_ns(int partition_size) const {
    float bytes = round_up_line_bytes(
        partition_size, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float maintenance_bw = cxl_bottleneck_bw_bytes_per_ns(*params_) *
                           std::max(1.0e-6f, params_->cxl_maintenance_bandwidth_fraction);
    float transfer_ns = 2.0f * bytes / maintenance_bw;
    float action_ns = transfer_ns;
    if (cxl_use_observed_action_cost() && cxl_has_observed_split_cost_) {
        action_ns = std::max(
            action_ns,
            cxl_observed_split_ns_per_record_ * static_cast<float>(std::max(1, partition_size)) +
                cxl_observed_refine_ns_per_split_);
    }
    return action_ns + std::max(0.0f, params_->cxl_adaptive_action_overhead_ns);
}

float MaintenancePolicy::cxl_streaming_reassign_buy_ns(int partition_size) const {
    float bytes = round_up_line_bytes(
        partition_size, params_->cxl_entry_bytes, params_->cxl_line_bytes);
    float maintenance_bw = cxl_bottleneck_bw_bytes_per_ns(*params_) *
                           std::max(1.0e-6f, params_->cxl_maintenance_bandwidth_fraction);
    float transfer_ns = 2.0f * bytes / maintenance_bw;
    float action_ns = transfer_ns;
    if (cxl_use_observed_action_cost() && cxl_observed_reassign_ns_per_record_ > 0.0f) {
        action_ns = std::max(
            action_ns,
            cxl_observed_reassign_ns_per_record_ * static_cast<float>(std::max(1, partition_size)));
    }
    return action_ns + std::max(0.0f, params_->cxl_adaptive_action_overhead_ns);
}

int MaintenancePolicy::cxl_adaptive_split_budget(int total_partitions) const {
    if (!cxl_use_roi_budget()) {
        return total_partitions;
    }
    int safe_total = std::max(0, total_partitions);
    if (safe_total == 0) {
        return 0;
    }
    float growth_pressure = std::max(cxl_growth_score_, cxl_last_growth_sample_);
    float delete_pressure = std::max(cxl_delete_score_, cxl_last_delete_sample_);
    float churn_pressure = std::max(cxl_churn_score_, cxl_last_churn_sample_);
    float budget_fraction = std::max(0.0f, params_->cxl_adaptive_base_split_budget_fraction);
    if (cxl_use_aps_feedback()) {
        budget_fraction *=
            (1.0f
             + 1.25f * growth_pressure
             + 1.5f * cxl_scanned_list_size_pressure_
             - params_->cxl_adaptive_delete_relief_gain * delete_pressure
             - params_->cxl_adaptive_fanout_growth_gain * cxl_fanout_pressure_
             - params_->cxl_adaptive_maintenance_penalty_gain * cxl_maintenance_pressure_);
    } else {
        budget_fraction *= (1.0f
                            + 1.25f * growth_pressure
                            + 2.0f * churn_pressure
                            + 0.75f * cxl_scan_pressure_
                            + 1.25f * delete_pressure
                            - params_->cxl_adaptive_maintenance_penalty_gain * cxl_maintenance_pressure_);
    }
    budget_fraction = std::min(
        std::max(0.0f, params_->cxl_adaptive_max_split_budget_fraction),
        std::max(0.0f, budget_fraction));

    int budget = static_cast<int>(std::ceil(static_cast<float>(safe_total) * budget_fraction));
    int max_budget = std::max(0, params_->cxl_adaptive_max_split_budget);
    int min_budget = std::max(0, std::min(params_->cxl_adaptive_min_split_budget, max_budget));
    float active_pressure = cxl_use_aps_feedback()
                                ? growth_pressure + cxl_scanned_list_size_pressure_
                                : growth_pressure + delete_pressure + churn_pressure + cxl_scan_pressure_;
    if (budget == 0 && active_pressure > 0.01f) {
        budget = min_budget;
    }
    budget = std::min(max_budget, std::max(0, budget));
    return std::min(safe_total, budget);
}

int MaintenancePolicy::cxl_time_budget_split_limit() const {
    if ((!cxl_use_roi_budget() && !cxl_use_streaming_rent_buy()) ||
        !params_->cxl_adaptive_observed_cost_enabled) {
        return std::max(0, params_->cxl_adaptive_max_split_budget);
    }
    if (!cxl_has_observed_split_cost_) {
        return std::max(0, params_->cxl_adaptive_warmup_split_budget);
    }
    float estimated_split_ns =
        cxl_observed_split_ns_per_record_ * static_cast<float>(std::max(1, cxl_target_partition_size_)) +
        cxl_observed_refine_ns_per_split_;
    if (estimated_split_ns <= 0.0f) {
        return std::max(0, params_->cxl_adaptive_max_split_budget);
    }
    return std::max(
        0,
        static_cast<int>(std::floor(cxl_maintenance_time_budget_ns_ / estimated_split_ns)));
}

void MaintenancePolicy::update_cxl_observed_action_costs(int64_t split_records,
                                                         int64_t split_time_us,
                                                         int64_t split_count,
                                                         int64_t refinement_time_us,
                                                         int64_t refinement_records_read,
                                                         int64_t refinement_records_written,
                                                         int64_t reassign_records,
                                                         int64_t reassign_time_us) {
    if (!cxl_use_observed_action_cost()) {
        return;
    }
    float alpha = clamp01(params_->cxl_adaptive_observed_cost_alpha);
    if (alpha <= 0.0f) {
        alpha = DEFAULT_CXL_ADAPTIVE_OBSERVED_COST_ALPHA;
    }
    auto update = [alpha](float previous, float sample, bool initialized) {
        return initialized ? (1.0f - alpha) * previous + alpha * sample : sample;
    };
    if (split_count > 0) {
        bool initialized = cxl_has_observed_split_cost_;
        if (split_records > 0 && split_time_us >= 0) {
            float sample = static_cast<float>(split_time_us) * 1000.0f /
                           static_cast<float>(split_records);
            cxl_observed_split_ns_per_record_ = update(
                cxl_observed_split_ns_per_record_, sample, initialized);
        }
        float refine_sample = static_cast<float>(std::max<int64_t>(0, refinement_time_us)) * 1000.0f /
                              static_cast<float>(split_count);
        cxl_observed_refine_ns_per_split_ = update(
            cxl_observed_refine_ns_per_split_, refine_sample, initialized);
        if (split_records > 0) {
            float refine_parent_sample =
                static_cast<float>(std::max<int64_t>(0, refinement_time_us)) * 1000.0f /
                static_cast<float>(split_records);
            float refine_read_sample =
                static_cast<float>(std::max<int64_t>(0, refinement_records_read)) /
                static_cast<float>(split_records);
            float refine_write_sample =
                static_cast<float>(std::max<int64_t>(0, refinement_records_written)) /
                static_cast<float>(split_records);
            cxl_observed_refine_ns_per_parent_record_ = update(
                cxl_observed_refine_ns_per_parent_record_, refine_parent_sample, initialized);
            cxl_observed_refine_read_records_per_parent_ = update(
                cxl_observed_refine_read_records_per_parent_, refine_read_sample, initialized);
            cxl_observed_refine_write_records_per_parent_ = update(
                cxl_observed_refine_write_records_per_parent_, refine_write_sample, initialized);
        }
        cxl_has_observed_split_cost_ = true;
    }
    if (reassign_records > 0 && reassign_time_us >= 0) {
        float sample = static_cast<float>(reassign_time_us) * 1000.0f /
                       static_cast<float>(reassign_records);
        bool initialized = cxl_observed_reassign_ns_per_record_ > 0.0f;
        cxl_observed_reassign_ns_per_record_ = update(
            cxl_observed_reassign_ns_per_record_, sample, initialized);
    }
}

constexpr bool LOG_WMA = true;
shared_ptr<MaintenanceTimingInfo> MaintenancePolicy::perform_maintenance() {
    // only consider split/deletion once the window is full

    auto start_total = steady_clock::now();
    if (partition_manager_->parent_ == nullptr) {
        return std::make_shared<MaintenanceTimingInfo>();
    }

    vector<int64_t> partitions_to_delete;
    vector<int64_t> partitions_to_split;
    vector<int64_t> partitions_to_recluster;

    Tensor all_partition_ids_tens = partition_manager_->get_partition_ids();
    vector<int64_t> all_partition_ids = vector<int64_t>(all_partition_ids_tens.data_ptr<int64_t>(),
                                                        all_partition_ids_tens.data_ptr<int64_t>() +
                                                        all_partition_ids_tens.size(0));
    int64_t decision_partition_count = static_cast<int64_t>(all_partition_ids.size());
    int64_t centroid_update_count = 0;
    int64_t delete_candidate_count = 0;
    int64_t delete_candidate_records = 0;
    int64_t split_candidate_count = 0;
    int64_t split_candidate_selected_count = 0;
    int64_t split_candidate_roi_rejected_count = 0;
    int64_t split_candidate_budget_rejected_count = 0;
    int64_t native_split_candidate_count = 0;
    int64_t split_budget_by_time = 0;
    int64_t reassign_budget_by_time = 0;
    int64_t partition_count_before = std::max<int64_t>(0, partition_manager_->nlist());
    int64_t streaming_candidate_count = 0;
    int64_t streaming_selected_count = 0;
    float streaming_window_rent_ns = 0.0f;
    float streaming_selected_buy_ns = 0.0f;
    float resource_window_rent_ns = 0.0f;
    float resource_selected_buy_ns = 0.0f;
    int64_t resource_split_candidate_count = 0;
    int64_t resource_best_cohort_size = 0;
    float resource_best_cohort_credit_ns = 0.0f;
    float resource_best_cohort_buy_ns = 0.0f;
    float resource_best_cohort_ratio = 0.0f;
    int64_t resource_selected_cohort_size = 0;
    float resource_selected_cohort_buy_ns = 0.0f;
    float search_first_available_gain_ns = 0.0f;
    float search_first_selected_gain_ns = 0.0f;
    int64_t search_first_cxl_split_candidate_count = 0;
    int64_t search_first_cxl_only_split_candidate_count = 0;
    int64_t search_first_selected_cxl_only_split_count = 0;
    vector<CxlPolicyDecision> resource_policy_decisions;
    cxl_streaming_structural_rent_ns_ = 0.0f;
    float current_aps_fanout = 0.0f;
    float current_avg_scanned_records = 0.0f;
    float current_avg_scanned_list_size = 0.0f;

    if (params_->max_partition_size != -1) {
        if constexpr(debug_) std::cout << "Mainteance bounding partition sizes to [" << params_->min_partition_size << "," << params_->max_partition_size << "]" << std::endl;
        for (const auto &partition_id: all_partition_ids) {
            int partition_size = partition_manager_->get_partition_size(partition_id);

            if (partition_size > params_->max_partition_size) {
                partitions_to_split.emplace_back(partition_id);
                split_candidate_count++;
            } else if (partition_size < params_->min_partition_size) {
                partitions_to_delete.emplace_back(partition_id);
            }
        }

    } else {
        if constexpr(debug_) std::cout << "Using the cost model to determine delete/split" << std::endl;

        int64_t num_queries = hit_count_tracker_->get_num_queries_recorded();
        if (hit_count_tracker_->get_num_queries_recorded() < params_->window_size) {
            std::cout << "Window not full yet. " << num_queries << " queries recorded and " << params_->window_size
                      << " queries required." << std::endl;
            return std::make_shared<MaintenanceTimingInfo>();
        }
        const bool search_guarded_has_prior_price =
            cxl_use_search_guarded_plus() &&
            cxl_resource_price_snapshot_ != nullptr &&
            cxl_resource_price_snapshot_->valid &&
            cxl_resource_price_snapshot_->window_duration_ns > 0;
        if (cxl_use_resource_rent_buy() &&
            (cxl_resource_price_snapshot_ == nullptr ||
             !cxl_resource_price_snapshot_->valid) &&
            !cxl_use_search_guarded_plus()) {
            throw std::runtime_error(
                "cxl_resource_rent_buy requires a valid resource price snapshot.");
        }

        // STEP 1: Aggregate hit counts from the HitCountTracker.
        vector<vector<int64_t> > per_query_hits = hit_count_tracker_->get_per_query_hits();
        unordered_map<int64_t, int> aggregated_hits;
        int64_t recorded_queries = std::min<int64_t>(
            hit_count_tracker_->get_num_queries_recorded(),
            static_cast<int64_t>(per_query_hits.size()));
        int64_t total_aps_fanout = 0;
        for (int64_t query_index = 0; query_index < recorded_queries; query_index++) {
            total_aps_fanout += static_cast<int64_t>(per_query_hits[query_index].size());
        }
        current_aps_fanout = recorded_queries > 0
                                 ? static_cast<float>(total_aps_fanout) /
                                       static_cast<float>(recorded_queries)
                                 : 0.0f;
        for (const auto &query_hits: per_query_hits) {
            for (int64_t pid: query_hits) {
                aggregated_hits[pid]++;
            }
        }

        // STEP 2: Use cost estimation to decide which partitions to delete or split.
        int total_partitions = partition_manager_->nlist();
        int desired_partition_count = static_cast<int>(std::ceil(
            static_cast<double>(std::max<int64_t>(0, partition_manager_->ntotal())) /
            static_cast<double>(std::max(1, cxl_target_partition_size_))));
        int structural_split_debt = std::max(0, desired_partition_count - total_partitions);
        float structural_size_ratio = std::max(1.0f, params_->cxl_adaptive_structural_size_ratio);
        int structural_size_threshold = static_cast<int>(std::ceil(
            static_cast<float>(std::max(1, cxl_target_partition_size_)) * structural_size_ratio));
        auto device_for_home = [&](int home_id) {
            if (cxl_resource_price_snapshot_ != nullptr &&
                home_id >= 0 &&
                home_id < static_cast<int>(
                    cxl_resource_price_snapshot_->home_device_ids.size())) {
                return cxl_resource_price_snapshot_->home_device_ids[home_id];
            }
            // Backward-compatible snapshots used one MC per device.
            return home_id;
        };
        auto resource_capacity_bytes = [](const CxlResourcePrice& resource) {
            if (resource.capacity_bytes > 0) {
                return static_cast<float>(resource.capacity_bytes);
            }
            if (resource.utilization > 0.0f &&
                resource.demand_bytes > 0) {
                return static_cast<float>(resource.demand_bytes) /
                    resource.utilization;
            }
            return 0.0f;
        };
        float current_scan_fraction = hit_count_tracker_->get_current_scan_fraction();
        current_avg_scanned_records =
            current_scan_fraction * static_cast<float>(std::max<int64_t>(0, partition_manager_->ntotal()));
        current_avg_scanned_list_size = current_aps_fanout > 0.0f
                                            ? current_avg_scanned_records / current_aps_fanout
                                            : 0.0f;
        update_cxl_adaptive_state(
            partition_manager_->ntotal(),
            total_partitions,
            current_scan_fraction,
            current_aps_fanout,
            current_avg_scanned_list_size);
        float cxl_split_weight = cxl_adaptive_split_score_weight();
        float cxl_maintenance_weight = cxl_adaptive_maintenance_penalty_weight();
        float cxl_delete_maintenance_weight = cxl_adaptive_delete_penalty_weight();
        float cxl_fanout_weight = cxl_adaptive_fanout_penalty_weight();
        struct CxlSplitCandidate {
            int64_t partition_id;
            float gain_ns;
            float roi;
            float benefit_ns;
            float cost_ns;
        };
        vector<CxlSplitCandidate> cxl_split_candidates;
        struct StreamingActionCandidate {
            bool split;
            int64_t partition_id;
            float rank_ns;
            float buy_ns;
            float local_credit_ns;
        };
        vector<StreamingActionCandidate> streaming_action_candidates;
        struct ResourceActionCandidate {
            bool split;
            int64_t partition_id;
            float ratio;
            float buy_ns;
            float credit_ns;
            float search_gain_ns;
            int64_t records;
            size_t decision_index;
        };
        vector<ResourceActionCandidate> resource_action_candidates;
        const std::unordered_set<int64_t> forced_split_ids(
            params_->cxl_resource_forced_split_ids.begin(),
            params_->cxl_resource_forced_split_ids.end());
        const std::unordered_set<int64_t> forced_reassign_ids(
            params_->cxl_resource_forced_reassign_ids.begin(),
            params_->cxl_resource_forced_reassign_ids.end());
        const bool resource_force_mode =
            cxl_use_resource_rent_buy() && params_->cxl_resource_force_action_set;
        const bool resource_force_active =
            resource_force_mode &&
            (params_->cxl_resource_force_window_id < 0 ||
             (cxl_resource_price_snapshot_ != nullptr &&
              cxl_resource_price_snapshot_->window_id ==
                  params_->cxl_resource_force_window_id));

        std::unordered_set<int64_t> active_partition_ids(
            all_partition_ids.begin(), all_partition_ids.end());
        auto drop_retired_credits = [&active_partition_ids](auto& credits) {
            for (auto it = credits.begin(); it != credits.end();) {
                if (active_partition_ids.count(it->first) == 0) {
                    it = credits.erase(it);
                } else {
                    ++it;
                }
            }
        };
        drop_retired_credits(cxl_split_credit_ns_);
        drop_retired_credits(cxl_reassign_credit_ns_);
        if (cxl_use_resource_rent_buy()) {
            update_cxl_child_probe_factor(per_query_hits, active_partition_ids);
            if (cxl_use_streaming_staged()) {
                // A mature sibling observation tells us how much of the
                // prior-window split benefit survives in the next window.
                // Keep the estimator causal and bounded.
                float observed = cxl_resource_child_probe_factor_ > 0.0f
                    ? std::min(1.0f, 1.0f / cxl_resource_child_probe_factor_)
                    : 1.0f;
                cxl_staged_calibration_r_ = std::min(
                    1.0f,
                    std::max(0.0f, 0.75f * cxl_staged_calibration_r_ +
                                      0.25f * observed));
            }
        }

        auto accumulate_streaming_reassign = [&](int64_t partition_id,
                                                 int partition_size,
                                                 float recurrent_delta_ns) {
            float rent_ns = std::max(0.0f, -recurrent_delta_ns) *
                            static_cast<float>(std::max<int64_t>(1, recorded_queries));
            float buy_ns = cxl_streaming_reassign_buy_ns(partition_size);
            float& credit_ns = cxl_reassign_credit_ns_[partition_id];
            credit_ns += rent_ns;
            streaming_candidate_count++;
            streaming_window_rent_ns += rent_ns;
            if (rent_ns > 0.0f && credit_ns >= buy_ns) {
                streaming_action_candidates.push_back(
                    {false, partition_id, credit_ns - buy_ns, buy_ns, credit_ns});
                return true;
            }
            return false;
        };

        auto accumulate_resource_reassign = [&] (
            int64_t partition_id,
            int partition_size,
            int source_hit_count,
            const vector<int64_t>& reassign_ids,
            const vector<int64_t>& reassign_counts,
            const vector<int64_t>& reassign_sizes,
            const vector<float>& hit_rates,
            float native_reassign_delta_ns) {
            float before_ns = 0.0f;
            float after_ns = 0.0f;
            float raw_rent_ns = cxl_resource_reassign_rent_ns(
                partition_id,
                partition_size,
                source_hit_count,
                reassign_ids,
                reassign_counts,
                reassign_sizes,
                hit_rates,
                recorded_queries,
                &before_ns,
                &after_ns);
            float resource_rent_ns = std::max(0.0f, raw_rent_ns);
            float native_rent_ns =
                std::max(0.0f, -native_reassign_delta_ns) *
                static_cast<float>(std::max<int64_t>(1, recorded_queries));
            float rent_ns = (cxl_use_search_first() || cxl_use_default_plus() ||
                             cxl_use_search_guarded_plus())
                                ? resource_rent_ns
                                : std::max(resource_rent_ns, native_rent_ns);
            bool native_legal_candidate =
                native_reassign_delta_ns < -params_->delete_threshold_ns &&
                (rent_ns > 0.0f || cxl_use_default_plus() ||
                 cxl_use_search_guarded_plus());
            float buy_ns = cxl_resource_reassign_buy_ns(
                partition_id, partition_size);
            float& credit_ns = cxl_reassign_credit_ns_[partition_id];
            credit_ns = 0.5f * credit_ns + rent_ns;
            float ratio = credit_ns / std::max(1.0f, buy_ns);
            CxlPolicyDecision decision;
            decision.action_kind = "reassign";
            decision.partition_id = partition_id;
            decision.home_id = cxl_resource_home_id(partition_id);
            decision.records = partition_size;
            decision.cost_before_ns = before_ns;
            decision.cost_after_ns = after_ns;
            decision.rent_ns = rent_ns;
            decision.buy_ns = buy_ns;
            decision.credit_ns = credit_ns;
            decision.rent_buy_ratio = ratio;
            decision.native_rent_ns = native_rent_ns;
            decision.resource_rent_ns = resource_rent_ns;
            decision.native_legal = native_legal_candidate;
            decision.cxl_search_profitable = resource_rent_ns > 0.0f;
            decision.read_line_bytes = round_up_line_bytes(
                partition_size, params_->cxl_entry_bytes, params_->cxl_line_bytes);
            decision.write_line_bytes = decision.read_line_bytes;
            // Reassign target discovery is still supplied by Quake. Keep its
            // final semantic gate, but rank an eligible action only by its CXL
            // search gain under search-first.
            bool search_first_candidate =
                cxl_use_search_first() && native_legal_candidate &&
                resource_rent_ns > 0.0f;
            decision.rejection_reason =
                !native_legal_candidate
                    ? "native_policy_rejected"
                    : (search_first_candidate
                           ? "pending_search_gain_selection"
                           : (ratio >= 1.0f
                                  ? "pending_budget_selection"
                                  : "credit_below_buy"));
            size_t decision_index = resource_policy_decisions.size();
            resource_policy_decisions.push_back(decision);
            resource_window_rent_ns += rent_ns;
            if ((!resource_force_mode && native_legal_candidate &&
                 (ratio >= 1.0f || search_first_candidate ||
                  cxl_use_default_plus() ||
                  cxl_use_search_guarded_plus())) ||
                resource_force_active) {
                resource_action_candidates.push_back(
                    {false,
                     partition_id,
                     ratio,
                     buy_ns,
                     credit_ns,
                     rent_ns,
                     partition_size,
                     decision_index});
            }
        };
        
        float* new_centroids_buffer = reinterpret_cast<float*>(quake_alloc(partition_manager_->d() * sizeof(float), 0));
        int avg_partition_size = partition_manager_->ntotal() / total_partitions;

        for (const auto &partition_id: all_partition_ids) {
            // Update the centroid for this vector if we have a delta
            bool choose_partition = false;
            float delete_factor = partition_manager_->get_delete_factor(partition_id);
            if (partition_manager_->update_centroid(partition_id, new_centroids_buffer) != 0) {
                centroid_update_count++;
            }

            // Get hit count and hit rate for the partition.
            int hit_count = aggregated_hits[partition_id];
            float hit_rate = static_cast<float>(hit_count) / static_cast<float>(params_->window_size);
            int partition_size = partition_manager_->get_partition_size(partition_id);

            // Deletion decision.
            float delete_delta = cost_estimator_->compute_delete_delta(
                partition_size, hit_rate, total_partitions, current_scan_fraction, avg_partition_size);
            if (params_->enable_cxl_cost_model && !cxl_use_streaming_rent_buy() &&
                !cxl_use_resource_rent_buy()) {
                delete_delta += cxl_delete_maintenance_weight *
                                cxl_window_amortized_penalty_ns(*params_, partition_size);
            }
            bool allow_cost_based_delete =
                !cxl_use_structural_budget() || total_partitions >= desired_partition_count;
            bool consider_partition_for_delete =
                allow_cost_based_delete && delete_delta < -params_->delete_threshold_ns;
            // Forced actions are an oracle-audit mechanism. Keep an explicitly
            // requested split independent of the native CPU model's candidate
            // classification, which can otherwise vary with profile noise.
            if (resource_force_active &&
                forced_split_ids.count(partition_id) > 0) {
                consider_partition_for_delete = false;
            }
            // if constexpr(debug_) std::cout << "For partition " << partition_id << " of size " << partition_size << " got delete delta " << delete_delta << " leading to delete decision of " << consider_partition_for_delete << std::endl;

            if (consider_partition_for_delete) {

                if (params_->enable_delete_rejection && partition_size > params_->min_partition_size) {
                    delete_candidate_count++;
                    delete_candidate_records += std::max(0, partition_size);
                    // check the assignments of the partitions to be deleted.
                    auto search_params = make_shared<SearchParams>();
                    search_params->k = 2; // get the top 2 partitions, ignore the first one as it is the partition itself
                    search_params->batched_scan = true;
                    search_params->track_hits = false;
                    float *partition_vectors = (float *) partition_manager_->partition_store_->partitions_[partition_id]->codes_;
                    Tensor part_vecs = torch::from_blob(partition_vectors, {(int64_t) partition_manager_->partition_store_->list_size(partition_id),
                                                                           partition_manager_->d()}, torch::kFloat32);
                    auto res = partition_manager_->parent_->search(part_vecs, search_params);

                    Tensor reassign_ids = res->ids.flatten();

                    // remove the partition itself
                    reassign_ids = reassign_ids.masked_select(reassign_ids != partition_id);

                    // Get A) the unique partitions, B) the number reassigned, C) the size of the partitions, D) hit rates of the partitions
                    Tensor uniques;
                    Tensor counts;
                    std::tie(uniques, std::ignore, counts) = torch::_unique2(reassign_ids, true, false, true);
                    Tensor part_sizes = partition_manager_->get_partition_sizes(uniques);

                    // convert to vectors
                    vector<int64_t> reassign_id_vec = vector<int64_t>(uniques.data_ptr<int64_t>(), uniques.data_ptr<int64_t>() + uniques.size(0));

                    vector<int64_t> reassign_sizes = vector<int64_t>(part_sizes.data_ptr<int64_t>(),
                                                                     part_sizes.data_ptr<int64_t>() + part_sizes.size(0));
                    vector<int64_t> reassign_counts = vector<int64_t>(counts.data_ptr<int64_t>(),
                                                                      counts.data_ptr<int64_t>() + counts.size(0));
                    vector<float> hit_rates;
                    for (int64_t reassign_id: reassign_id_vec) {
                        hit_rates.push_back(static_cast<float>(aggregated_hits[reassign_id]) / static_cast<float>(params_->window_size));
                    }

                    float delta = cost_estimator_->compute_delete_delta_w_reassign(partition_manager_->get_partition_size(partition_id),
                                                                                  static_cast<float>(aggregated_hits[partition_id]) / static_cast<float>(params_->window_size),
                                                                                  total_partitions,
                                                                                  reassign_counts,
                                                                                  reassign_sizes,
                                                                                  hit_rates);
                    if (params_->enable_cxl_cost_model && !cxl_use_streaming_rent_buy() &&
                        !cxl_use_resource_rent_buy()) {
                        delta += cxl_delete_maintenance_weight *
                                 cxl_window_amortized_penalty_ns(*params_, partition_size);
                    }

                    if (cxl_use_default_plus() ||
                        cxl_use_search_guarded_plus()) {
                        // Preserve the native delete-rejection decision.  A
                        // partition reaching this branch is only a candidate;
                        // default deletes it iff reassignment remains a net
                        // win after the exact target-set calculation above.
                        if (delta < -params_->delete_threshold_ns) {
                            partitions_to_delete.push_back(partition_id);
                            choose_partition = true;
                        }
                    } else if (cxl_use_resource_rent_buy()) {
                        accumulate_resource_reassign(
                            partition_id,
                            partition_size,
                            hit_count,
                            reassign_id_vec,
                            reassign_counts,
                            reassign_sizes,
                            hit_rates,
                            delta);
                    } else if (cxl_use_streaming_rent_buy()) {
                        accumulate_streaming_reassign(partition_id, partition_size, delta);
                    } else if (delta < -params_->delete_threshold_ns) {
                        choose_partition = true;
                        partitions_to_delete.push_back(partition_id);
                    }
                } else {
                    if (cxl_use_default_plus() ||
                        cxl_use_search_guarded_plus()) {
                        // Match default's no-rejection / too-small fallback.
                        partitions_to_delete.push_back(partition_id);
                        choose_partition = true;
                    } else if (cxl_use_resource_rent_buy()) {
                        CxlPolicyDecision decision;
                        decision.action_kind = "reassign";
                        decision.partition_id = partition_id;
                        decision.home_id = cxl_resource_home_id(partition_id);
                        decision.records = partition_size;
                        decision.buy_ns = cxl_resource_reassign_buy_ns(
                            partition_id, partition_size);
                        decision.rejection_reason = "missing_reassign_target_state";
                        resource_policy_decisions.push_back(decision);
                    } else if (cxl_use_streaming_rent_buy()) {
                        accumulate_streaming_reassign(
                            partition_id, partition_size, delete_delta);
                    } else {
                        partitions_to_delete.push_back(partition_id);
                        choose_partition = true;
                    }
                }
            } else {
                bool partition_large_enough = partition_size > params_->min_partition_size;
                if (partition_size > params_->min_partition_size) {
                    split_candidate_count++;
                    float split_delta = cost_estimator_->compute_split_delta(
                        partition_size, hit_rate, total_partitions);
                    float native_split_delta = split_delta;
                    bool native_should_split = native_split_delta < -params_->split_threshold_ns;
                    if (native_should_split) {
                        native_split_candidate_count++;
                    }
                    bool split_candidate_deferred = false;
                    if (cxl_use_search_guarded_plus() &&
                        !search_guarded_has_prior_price) {
                        // A missing completed price window disables only the
                        // CXL increment. Preserve Quake's native split exactly.
                        if (native_should_split) {
                            partitions_to_split.push_back(partition_id);
                            choose_partition = true;
                        }
                        split_candidate_deferred = true;
                    } else if (cxl_use_resource_rent_buy()) {
                        int child_size = std::max(1, (partition_size + 1) / 2);
                        float child_probe_factor = std::min(
                            2.0f,
                            std::max(1.0f, cxl_resource_child_probe_factor_));
                        float before_ns = static_cast<float>(std::max(0, hit_count)) *
                                          cxl_resource_scan_cost_ns(
                                              partition_id, partition_size);
                        float child_scan_cost_ns =
                            (cxl_use_search_first() || cxl_use_default_plus() ||
                             cxl_use_search_guarded_plus())
                                                       ? cxl_resource_balanced_birth_scan_cost_ns(
                                                             child_size)
                                                       : cxl_resource_scan_cost_ns(
                                                             partition_id, child_size);
                        float after_ns = static_cast<float>(std::max(0, hit_count)) *
                                         child_probe_factor * child_scan_cost_ns;
                        float resource_rent_ns = std::max(0.0f, before_ns - after_ns);
                        float native_rent_ns =
                            std::max(0.0f, -native_split_delta) *
                            static_cast<float>(std::max<int64_t>(1, recorded_queries));
                        // Strict resource rent/buy preserves Quake's native
                        // candidate semantics. Search-first is independent:
                        // its candidate and value are the causal CXL search
                        // saving from the previous resource-price window.
                        float rent_ns = (cxl_use_search_first() || cxl_use_streaming_staged() ||
                                         cxl_use_default_plus() ||
                                         cxl_use_search_guarded_plus())
                                            ? resource_rent_ns
                                            : std::max(resource_rent_ns, native_rent_ns);
                        float variable_buy_ns = cxl_resource_split_variable_buy_ns(
                            partition_id, partition_size);
                        float single_shared_buy_ns =
                            cxl_resource_split_cohort_shared_buy_ns(partition_size);
                        float single_buy_ns = variable_buy_ns + single_shared_buy_ns;
                        float& credit_ns = cxl_split_credit_ns_[partition_id];
                        credit_ns = cxl_use_streaming_staged()
                            ? 0.75f * credit_ns + 0.25f * rent_ns
                            : (cxl_use_default_plus() ||
                               cxl_use_search_guarded_plus())
                            ? rent_ns
                            : 0.5f * credit_ns + rent_ns;
                        if (cxl_use_streaming_staged() || cxl_use_default_plus() ||
                            cxl_use_search_guarded_plus()) {
                            // Logical work reads once and writes only
                            // descriptors/centroids/bitmap metadata.  Payload
                            // materialization is deferred to the simulator.
                            const float metadata_price_ns_per_byte =
                                cxl_resource_price_snapshot_ != nullptr
                                    ? std::max(
                                          0.0f,
                                          cxl_resource_price_snapshot_->
                                              routing_metadata_shadow_price_ns_per_byte)
                                    : 0.0f;
                            single_buy_ns = std::max(
                                1.0f,
                                0.25f * variable_buy_ns +
                                    2.0f * std::max(0, params_->cxl_metadata_bytes) *
                                        metadata_price_ns_per_byte);
                        }
                        float ratio = credit_ns / std::max(1.0f, single_buy_ns);
                        CxlPolicyDecision decision;
                        decision.action_kind = "split";
                        decision.partition_id = partition_id;
                        decision.home_id = cxl_resource_home_id(partition_id);
                        decision.records = partition_size;
                        decision.cost_before_ns = before_ns;
                        decision.cost_after_ns = after_ns;
                        decision.rent_ns = rent_ns;
                        decision.buy_ns = single_buy_ns;
                        decision.credit_ns = credit_ns;
                        decision.rent_buy_ratio = ratio;
                        decision.native_rent_ns = native_rent_ns;
                        decision.resource_rent_ns = resource_rent_ns;
                        decision.native_legal = native_should_split;
                        decision.cxl_search_profitable =
                            hit_count > 0 && resource_rent_ns > 0.0f;
                        decision.read_line_bytes = round_up_line_bytes(
                            partition_size,
                            params_->cxl_entry_bytes,
                            params_->cxl_line_bytes);
                        decision.write_line_bytes = decision.read_line_bytes;
                        bool native_legal_candidate =
                            native_should_split &&
                            (rent_ns > 0.0f || cxl_use_default_plus() ||
                             cxl_use_search_guarded_plus());
                        bool search_first_candidate =
                            cxl_use_search_first() &&
                            decision.cxl_search_profitable;
                        bool default_plus_growth_candidate =
                            cxl_use_default_plus() &&
                            structural_split_debt > 0 &&
                            hit_count > 0 &&
                            partition_size >= structural_size_threshold;
                        bool default_plus_candidate =
                            cxl_use_default_plus() &&
                            (native_should_split ||
                             (decision.cxl_search_profitable && resource_rent_ns > single_buy_ns) ||
                             default_plus_growth_candidate);
                        const bool guarded_has_prior_price =
                            cxl_use_search_guarded_plus() &&
                            search_guarded_has_prior_price;
                        const bool search_guarded_candidate =
                            cxl_use_search_guarded_plus() &&
                            (native_should_split ||
                             (guarded_has_prior_price &&
                              structural_split_debt > 0 &&
                              hit_count > 0 &&
                              partition_size >= structural_size_threshold &&
                              resource_rent_ns > 0.0f));
                        bool eligible_candidate = native_legal_candidate;
                        if (cxl_use_streaming_staged()) {
                            eligible_candidate =
                                cxl_staged_calibration_r_ *
                                    resource_rent_ns >
                                single_buy_ns;
                        } else if (cxl_use_search_guarded_plus()) {
                            eligible_candidate = search_guarded_candidate;
                        } else if (cxl_use_default_plus()) {
                            eligible_candidate = default_plus_candidate;
                        } else if (cxl_use_search_first()) {
                            eligible_candidate = search_first_candidate;
                        }
                        if (cxl_use_search_first()) {
                            decision.rejection_reason = search_first_candidate
                                                            ? "pending_cxl_search_selection"
                                                            : "cxl_search_gain_nonpositive";
                            if (search_first_candidate) {
                                search_first_cxl_split_candidate_count++;
                                if (!native_should_split) {
                                    search_first_cxl_only_split_candidate_count++;
                                }
                            }
                        } else if (cxl_use_streaming_staged()) {
                            decision.rejection_reason = eligible_candidate
                                ? "pending_staged_utilization_selection"
                                : "staged_benefit_below_logical_buy";
                        } else if (cxl_use_default_plus() ||
                                   cxl_use_search_guarded_plus()) {
                            if (native_should_split) {
                                decision.rejection_reason =
                                    "pending_native_default_selection";
                            } else if (eligible_candidate) {
                                decision.rejection_reason =
                                    cxl_use_search_guarded_plus()
                                        ? "pending_search_guarded_selection"
                                        : (default_plus_growth_candidate
                                               ? "pending_default_plus_growth_selection"
                                               : "pending_default_plus_selection");
                            } else {
                                decision.rejection_reason =
                                    guarded_has_prior_price
                                        ? "cxl_search_benefit_nonpositive"
                                        : "missing_prior_resource_price";
                            }
                        } else {
                            decision.rejection_reason = native_legal_candidate
                                                            ? "pending_cohort_selection"
                                                            : "native_policy_rejected";
                        }
                        size_t decision_index = resource_policy_decisions.size();
                        resource_policy_decisions.push_back(decision);
                        resource_window_rent_ns += rent_ns;
                        if ((!resource_force_mode && eligible_candidate) ||
                            resource_force_active) {
                            resource_split_candidate_count++;
                            resource_action_candidates.push_back(
                                {true,
                                 partition_id,
                                 ratio,
                                 cxl_use_streaming_staged() ? single_buy_ns : variable_buy_ns,
                                 credit_ns,
                                 rent_ns,
                                 partition_size,
                                 decision_index});
                            if ((cxl_use_default_plus() ||
                                 cxl_use_search_guarded_plus()) &&
                                native_should_split) {
                                // Native Quake would commit this choice before
                                // the later delete-factor fallback.  Mark it
                                // chosen now so default-plus preserves that
                                // control-flow priority while deferring only
                                // the physical cohort commit.
                                choose_partition = true;
                            }
                        } else if (!resource_force_mode) {
                            split_candidate_roi_rejected_count++;
                        }
                        split_candidate_deferred = true;
                    } else if (cxl_use_streaming_rent_buy()) {
                        StreamingSplitRent rent = cxl_streaming_split_rent_ns(
                            partition_size,
                            hit_rate,
                            native_split_delta,
                            recorded_queries);
                        float buy_ns = cxl_streaming_split_buy_ns(partition_size);
                        float& credit_ns = cxl_split_credit_ns_[partition_id];
                        credit_ns += rent.local_ns;
                        cxl_streaming_structural_credit_ns_ += rent.structural_ns;
                        streaming_candidate_count++;
                        streaming_window_rent_ns += rent.local_ns + rent.structural_ns;
                        if (rent.local_ns + rent.structural_ns > 0.0f) {
                            streaming_action_candidates.push_back(
                                {true,
                                 partition_id,
                                 credit_ns + rent.structural_ns,
                                 buy_ns,
                                 credit_ns});
                        } else {
                            split_candidate_roi_rejected_count++;
                        }
                        split_candidate_deferred = true;
                    } else if (params_->enable_cxl_cost_model) {
                        float scan_benefit = cxl_scan_benefit_ns(*params_, partition_size, hit_rate);
                        float maintenance_penalty = cxl_window_amortized_penalty_ns(*params_, partition_size);
                        if (cxl_use_roi_budget()) {
                            maintenance_penalty /= std::max(1.0f, cxl_payback_windows_);
                        }
                        float fanout_penalty = cxl_fanout_penalty_ns(*params_, hit_rate);
                        float weighted_benefit = cxl_split_weight * scan_benefit;
                        float weighted_cost = cxl_maintenance_weight * maintenance_penalty
                                              + cxl_fanout_weight * fanout_penalty;
                        if (cxl_use_roi_budget()) {
                            if (params_->cxl_adaptive_observed_cost_enabled && cxl_has_observed_split_cost_) {
                                float observed_action_ns =
                                    cxl_observed_split_ns_per_record_ * static_cast<float>(partition_size) +
                                    cxl_observed_refine_ns_per_split_;
                                weighted_cost += observed_action_ns /
                                                 (static_cast<float>(std::max(1, params_->window_size)) *
                                                  std::max(1.0f, cxl_payback_windows_));
                            }
                            weighted_cost += std::max(0.0f, params_->cxl_adaptive_action_overhead_ns) /
                                             (static_cast<float>(std::max(1, params_->window_size)) *
                                              std::max(1.0f, cxl_payback_windows_));
                        }
                        split_delta = split_delta - weighted_benefit + weighted_cost;
                        if (cxl_use_roi_budget()) {
                            float native_benefit = std::max(0.0f, -native_split_delta);
                            float total_benefit = native_benefit + weighted_benefit;
                            float total_cost = std::max(1.0f, weighted_cost);
                            float roi = total_benefit / total_cost;
                            bool should_split = split_delta < -params_->split_threshold_ns &&
                                                roi >= std::max(0.0f, params_->cxl_adaptive_roi_threshold) &&
                                                (total_benefit - total_cost) >= params_->cxl_adaptive_min_gain_ns;
                            bool structural_candidate =
                                cxl_use_structural_budget() &&
                                structural_split_debt > 0 &&
                                hit_count > 0 &&
                                partition_size >= structural_size_threshold;
                            bool selected_candidate = should_split;
                            if (cxl_use_structural_budget()) {
                                selected_candidate = cxl_use_aps_feedback()
                                                         ? should_split || structural_candidate
                                                         : should_split || native_should_split || structural_candidate;
                            }
                            if constexpr(debug_) std::cout << "For partition " << partition_id << " of size " << partition_size << " got split delta " << split_delta << " roi " << roi << " leading to split decision of " << should_split << std::endl;
                            if (selected_candidate) {
                                cxl_split_candidates.push_back(
                                    {partition_id, total_benefit - total_cost, roi, total_benefit, total_cost});
                            } else {
                                split_candidate_roi_rejected_count++;
                            }
                            split_candidate_deferred = true;
                        }
                    }
                    if (!split_candidate_deferred) {
                        bool should_split = split_delta < -params_->split_threshold_ns;
                        if constexpr(debug_) std::cout << "For partition " << partition_id << " of size " << partition_size << " got split delta " << split_delta << " leading to split decision of " << should_split << std::endl;
                        if (should_split) {
                            partitions_to_split.push_back(partition_id);
                            choose_partition = true;
                        }
                    }
                }
            }

            // If it was not chosen then consider it for some tracking based optimizations
            if(choose_partition) { 
                continue;
            }

            // If a large chunk of the partition was deleted then mark it for deletion
            bool perform_delete = delete_factor != -1.0 && delete_factor > params_->partition_reduction_threshold;
            if(perform_delete) { 
                partitions_to_delete.push_back(partition_id);
                cxl_reassign_credit_ns_.erase(partition_id);
                continue;
            } 
        } 

        if (cxl_use_resource_rent_buy()) {
            float available_budget_ns = std::max(
                0.0f, cxl_maintenance_time_budget_ns_);
            std::unordered_set<int64_t> selected_ids(
                partitions_to_delete.begin(), partitions_to_delete.end());
            vector<ResourceActionCandidate> split_actions;
            vector<ResourceActionCandidate> reassign_actions;
            for (const auto& candidate : resource_action_candidates) {
                (candidate.split ? split_actions : reassign_actions).push_back(candidate);
            }
            auto candidate_order = [&](const ResourceActionCandidate& lhs,
                                       const ResourceActionCandidate& rhs) {
                float lhs_value = cxl_use_search_first()
                                      ? lhs.search_gain_ns
                                      : lhs.credit_ns;
                float rhs_value = cxl_use_search_first()
                                      ? rhs.search_gain_ns
                                      : rhs.credit_ns;
                float lhs_density = lhs_value / std::max(1.0f, lhs.buy_ns);
                float rhs_density = rhs_value / std::max(1.0f, rhs.buy_ns);
                if (lhs_density == rhs_density) {
                    if (lhs_value == rhs_value) {
                        return lhs.partition_id < rhs.partition_id;
                    }
                    return lhs_value > rhs_value;
                }
                return lhs_density > rhs_density;
            };
            // default-plus is intentionally "native Quake + at most one CXL
            // increment".  Preserve the native candidates' original
            // all_partition_ids order because split order affects the joint
            // refinement neighborhood and therefore child membership.  The
            // one CXL-only increment is ranked separately below.
            if (!cxl_use_default_plus() &&
                !cxl_use_search_guarded_plus()) {
                std::sort(split_actions.begin(), split_actions.end(), candidate_order);
            }
            std::sort(reassign_actions.begin(), reassign_actions.end(), candidate_order);

            const int64_t cohort_id = cxl_resource_price_snapshot_ != nullptr
                                          ? cxl_resource_price_snapshot_->window_id
                                          : -1;
            auto cohort_search_gain = [](const vector<ResourceActionCandidate>& cohort) {
                float gain_ns = 0.0f;
                for (const auto& candidate : cohort) {
                    gain_ns += candidate.search_gain_ns;
                }
                return gain_ns;
            };
            auto set_cohort_telemetry = [&](const vector<ResourceActionCandidate>& cohort,
                                            float credit_ns,
                                            float variable_buy_ns,
                                            float shared_buy_ns,
                                            float total_buy_ns,
                                            float search_gain_ns,
                                            float available_gain_ns) {
                float ratio = credit_ns / std::max(1.0f, total_buy_ns);
                float gain_fraction = available_gain_ns > 0.0f
                                          ? search_gain_ns / available_gain_ns
                                          : 0.0f;
                for (const auto& candidate : cohort) {
                    CxlPolicyDecision& decision =
                        resource_policy_decisions[candidate.decision_index];
                    decision.cohort_id = cohort_id;
                    decision.cohort_size = static_cast<int64_t>(cohort.size());
                    decision.cohort_credit_ns = credit_ns;
                    decision.cohort_variable_buy_ns = variable_buy_ns;
                    decision.cohort_shared_buy_ns = shared_buy_ns;
                    decision.cohort_buy_ns = total_buy_ns;
                    decision.cohort_rent_buy_ratio = ratio;
                    decision.cohort_search_gain_ns = search_gain_ns;
                    decision.cohort_search_gain_fraction = gain_fraction;
                }
            };
            auto commit_split_cohort = [&](const vector<ResourceActionCandidate>& cohort,
                                           float credit_ns,
                                           float variable_buy_ns,
                                           float shared_buy_ns,
                                           float total_buy_ns,
                                           float search_gain_ns,
                                           float available_gain_ns) {
                set_cohort_telemetry(
                    cohort,
                    credit_ns,
                    variable_buy_ns,
                    shared_buy_ns,
                    total_buy_ns,
                    search_gain_ns,
                    available_gain_ns);
                for (const auto& candidate : cohort) {
                    CxlPolicyDecision& decision =
                        resource_policy_decisions[candidate.decision_index];
                    partitions_to_split.push_back(candidate.partition_id);
                    cxl_split_credit_ns_.erase(candidate.partition_id);
                    selected_ids.insert(candidate.partition_id);
                    decision.selected = true;
                    decision.rejection_reason.clear();
                    if (cxl_use_search_first() &&
                        decision.cxl_search_profitable &&
                        !decision.native_legal) {
                        search_first_selected_cxl_only_split_count++;
                    }
                }
                available_budget_ns = std::max(0.0f, available_budget_ns - total_buy_ns);
                resource_selected_buy_ns += total_buy_ns;
                resource_selected_cohort_size = static_cast<int64_t>(cohort.size());
                resource_selected_cohort_buy_ns = total_buy_ns;
                if (cxl_use_search_first()) {
                    search_first_selected_gain_ns += search_gain_ns;
                }
            };

            if (resource_force_mode && !resource_force_active) {
                for (auto& decision : resource_policy_decisions) {
                    decision.rejection_reason = "audit_window_not_active";
                }
            } else {
                vector<ResourceActionCandidate> eligible_splits;
                for (const auto& candidate : split_actions) {
                    CxlPolicyDecision& decision =
                        resource_policy_decisions[candidate.decision_index];
                    bool forced = resource_force_active &&
                                  forced_split_ids.count(candidate.partition_id) > 0;
                    if (selected_ids.count(candidate.partition_id) > 0) {
                        decision.rejection_reason = "conflicting_or_mandatory_action";
                    } else if (resource_force_active && !forced) {
                        decision.rejection_reason = "audit_not_forced";
                    } else {
                        eligible_splits.push_back(candidate);
                    }
                }

                if (resource_force_active) {
                    if (!eligible_splits.empty()) {
                        float credit_ns = 0.0f;
                        float variable_buy_ns = 0.0f;
                        int64_t parent_records = 0;
                        for (const auto& candidate : eligible_splits) {
                            credit_ns += candidate.credit_ns;
                            variable_buy_ns += candidate.buy_ns;
                            parent_records += candidate.records;
                        }
                        float shared_buy_ns =
                            cxl_resource_split_cohort_shared_buy_ns(parent_records);
                        float total_buy_ns = variable_buy_ns + shared_buy_ns;
                        float search_gain_ns = cohort_search_gain(eligible_splits);
                        if (cxl_use_search_first()) {
                            search_first_available_gain_ns += search_gain_ns;
                        }
                        resource_best_cohort_size = static_cast<int64_t>(eligible_splits.size());
                        resource_best_cohort_credit_ns = credit_ns;
                        resource_best_cohort_buy_ns = total_buy_ns;
                        resource_best_cohort_ratio =
                            credit_ns / std::max(1.0f, total_buy_ns);
                        commit_split_cohort(
                            eligible_splits,
                            credit_ns,
                            variable_buy_ns,
                            shared_buy_ns,
                            total_buy_ns,
                            search_gain_ns,
                            search_gain_ns);
                    }
                } else if (cxl_use_search_guarded_plus() &&
                           !eligible_splits.empty()) {
                    // Native Quake is the base action set.  CXL-only additions
                    // require a completed prior price window and strictly
                    // positive predicted search gain; their physical buy is
                    // reported but never used as a search veto.
                    vector<ResourceActionCandidate> native_splits;
                    vector<ResourceActionCandidate> cxl_only_splits;
                    for (const auto& candidate : eligible_splits) {
                        const CxlPolicyDecision& decision =
                            resource_policy_decisions[
                                candidate.decision_index];
                        (decision.native_legal
                             ? native_splits
                             : cxl_only_splits)
                            .push_back(candidate);
                    }
                    if (!native_splits.empty()) {
                        float native_credit_ns = 0.0f;
                        float native_buy_ns = 0.0f;
                        float native_gain_ns = 0.0f;
                        for (const auto& candidate : native_splits) {
                            native_credit_ns += candidate.credit_ns;
                            native_buy_ns += candidate.buy_ns;
                            native_gain_ns += candidate.search_gain_ns;
                        }
                        commit_split_cohort(
                            native_splits,
                            native_credit_ns,
                            native_buy_ns,
                            0.0f,
                            native_buy_ns,
                            native_gain_ns,
                            native_gain_ns);
                    }
                    std::sort(
                        cxl_only_splits.begin(),
                        cxl_only_splits.end(),
                        [&](const ResourceActionCandidate& lhs,
                            const ResourceActionCandidate& rhs) {
                            const CxlPolicyDecision& lhs_decision =
                                resource_policy_decisions[
                                    lhs.decision_index];
                            const CxlPolicyDecision& rhs_decision =
                                resource_policy_decisions[
                                    rhs.decision_index];
                            const float lhs_relative =
                                lhs.search_gain_ns /
                                std::max(
                                    1.0f,
                                    lhs_decision.cost_before_ns);
                            const float rhs_relative =
                                rhs.search_gain_ns /
                                std::max(
                                    1.0f,
                                    rhs_decision.cost_before_ns);
                            if (lhs_relative != rhs_relative) {
                                return lhs_relative > rhs_relative;
                            }
                            if (lhs.search_gain_ns !=
                                rhs.search_gain_ns) {
                                return lhs.search_gain_ns >
                                       rhs.search_gain_ns;
                            }
                            return lhs.partition_id <
                                   rhs.partition_id;
                        });
                    const int remaining_growth_debt = std::max(
                        0,
                        structural_split_debt -
                            static_cast<int>(native_splits.size()));
                    const int extra_split_budget =
                        static_cast<int>(std::ceil(
                            static_cast<float>(
                                remaining_growth_debt) *
                            std::max(
                                0.0f,
                                params_->
                                    cxl_adaptive_growth_debt_repay_fraction)));
                    int added_cxl_splits = 0;
                    std::unordered_map<string, float>
                        added_bytes_by_resource;
                    for (const auto& candidate : cxl_only_splits) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[
                                candidate.decision_index];
                        if (added_cxl_splits >= extra_split_budget) {
                            decision.rejection_reason =
                                "search_guarded_growth_debt_budget";
                            continue;
                        }
                        bool utilization_ok = true;
                        const float logical_bytes =
                            static_cast<float>(
                                round_up_line_bytes(
                                    candidate.records,
                                    params_->cxl_entry_bytes,
                                    params_->cxl_line_bytes) +
                                2 * std::max(
                                        0,
                                        params_->cxl_metadata_bytes) +
                                std::max<int64_t>(
                                    0,
                                    (candidate.records + 7) / 8));
                        const string parent_home =
                            "mc:" +
                            std::to_string(
                                cxl_resource_home_id(
                                    candidate.partition_id));
                        const string parent_link =
                            "link:" +
                            std::to_string(
                                device_for_home(
                                    cxl_resource_home_id(
                                        candidate.partition_id)));
                        if (cxl_resource_price_snapshot_ != nullptr) {
                            for (const auto& resource :
                                 cxl_resource_price_snapshot_->
                                     resources) {
                                const bool affected =
                                    resource.resource_id ==
                                        parent_home ||
                                    resource.resource_id ==
                                        parent_link;
                                if (!affected) {
                                    continue;
                                }
                                const float capacity_bytes =
                                    resource_capacity_bytes(resource);
                                if (capacity_bytes <= 0.0f) {
                                    continue;
                                }
                                const float projected =
                                    resource.utilization +
                                    (added_bytes_by_resource[
                                         resource.resource_id] +
                                     logical_bytes) /
                                        std::max(
                                            1.0f,
                                            capacity_bytes);
                                if (projected >= 0.90f) {
                                    utilization_ok = false;
                                    break;
                                }
                            }
                        }
                        if (!utilization_ok) {
                            decision.rejection_reason =
                                "projected_resource_utilization";
                            split_candidate_budget_rejected_count++;
                            continue;
                        }
                        vector<ResourceActionCandidate> singleton = {
                            candidate};
                        commit_split_cohort(
                            singleton,
                            candidate.credit_ns,
                            candidate.buy_ns,
                            0.0f,
                            candidate.buy_ns,
                            candidate.search_gain_ns,
                            candidate.search_gain_ns);
                        added_cxl_splits++;
                        added_bytes_by_resource[parent_home] +=
                            logical_bytes;
                        added_bytes_by_resource[parent_link] +=
                            logical_bytes;
                    }
                } else if (cxl_use_default_plus() && !eligible_splits.empty()) {
                    // Keep Quake's native split set intact, then repay a
                    // causal fraction of the current structural growth debt.
                    // Only lists observed in this query window and already
                    // above the initial-size guard enter the extra cohort.
                    vector<ResourceActionCandidate> native_splits;
                    vector<ResourceActionCandidate> cxl_only_splits;
                    for (const auto& candidate : eligible_splits) {
                        const CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        (decision.native_legal ? native_splits : cxl_only_splits)
                            .push_back(candidate);
                    }
                    std::sort(
                        cxl_only_splits.begin(),
                        cxl_only_splits.end(),
                        [&](const ResourceActionCandidate& lhs,
                            const ResourceActionCandidate& rhs) {
                            const int64_t lhs_excess = std::max<int64_t>(
                                0, lhs.records - cxl_target_partition_size_);
                            const int64_t rhs_excess = std::max<int64_t>(
                                0, rhs.records - cxl_target_partition_size_);
                            const int64_t lhs_pressure = lhs_excess *
                                static_cast<int64_t>(aggregated_hits.at(lhs.partition_id));
                            const int64_t rhs_pressure = rhs_excess *
                                static_cast<int64_t>(aggregated_hits.at(rhs.partition_id));
                            if (lhs_pressure != rhs_pressure) {
                                return lhs_pressure > rhs_pressure;
                            }
                            return candidate_order(lhs, rhs);
                        });
                    if (!native_splits.empty()) {
                        float native_credit_ns = 0.0f;
                        float native_buy_ns = 0.0f;
                        float native_gain_ns = 0.0f;
                        for (const auto& candidate : native_splits) {
                            native_credit_ns += candidate.credit_ns;
                            native_buy_ns += candidate.buy_ns;
                            native_gain_ns += candidate.search_gain_ns;
                        }
                        commit_split_cohort(
                            native_splits,
                            native_credit_ns,
                            native_buy_ns,
                            0.0f,
                            native_buy_ns,
                            native_gain_ns,
                            native_gain_ns);
                    }

                    const int remaining_growth_debt = std::max(
                        0,
                        structural_split_debt -
                            static_cast<int>(native_splits.size()));
                    const int extra_split_budget = static_cast<int>(std::ceil(
                        static_cast<float>(remaining_growth_debt) *
                        std::max(
                            0.0f,
                            params_->cxl_adaptive_growth_debt_repay_fraction)));
                    int added_cxl_splits = 0;
                    std::unordered_map<string, float> added_bytes_by_resource;
                    for (const auto& candidate : cxl_only_splits) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        if (added_cxl_splits >= extra_split_budget) {
                            decision.rejection_reason = "default_plus_growth_debt_budget";
                            continue;
                        }
                        bool utilization_ok = true;
                        const float logical_bytes = static_cast<float>(
                            round_up_line_bytes(
                                candidate.records,
                                params_->cxl_entry_bytes,
                                params_->cxl_line_bytes) +
                            2 * std::max(0, params_->cxl_metadata_bytes) +
                            std::max<int64_t>(0, (candidate.records + 7) / 8));
                        if (cxl_resource_price_snapshot_ != nullptr) {
                            const string parent_home = "mc:" +
                                std::to_string(cxl_resource_home_id(candidate.partition_id));
                            const string parent_link = "link:" +
                                std::to_string(device_for_home(
                                    cxl_resource_home_id(candidate.partition_id)));
                            for (const auto& resource : cxl_resource_price_snapshot_->resources) {
                                const bool affected = resource.resource_id == parent_home ||
                                    resource.resource_id == parent_link;
                                if (!affected) {
                                    continue;
                                }
                                const float capacity_bytes =
                                    resource_capacity_bytes(resource);
                                if (capacity_bytes <= 0.0f) {
                                    continue;
                                }
                                const float projected = resource.utilization +
                                    (added_bytes_by_resource[resource.resource_id] + logical_bytes) /
                                        std::max(1.0f, capacity_bytes);
                                if (projected >= 0.90f) {
                                    utilization_ok = false;
                                    break;
                                }
                            }
                        }
                        if (!utilization_ok) {
                            decision.rejection_reason = "projected_resource_utilization";
                            split_candidate_budget_rejected_count++;
                            continue;
                        }
                        vector<ResourceActionCandidate> singleton = {candidate};
                        commit_split_cohort(
                            singleton,
                            candidate.credit_ns,
                            candidate.buy_ns,
                            0.0f,
                            candidate.buy_ns,
                            candidate.search_gain_ns,
                            candidate.search_gain_ns);
                        added_cxl_splits++;
                        const string parent_home = "mc:" +
                            std::to_string(cxl_resource_home_id(candidate.partition_id));
                        const string parent_link = "link:" +
                            std::to_string(device_for_home(
                                cxl_resource_home_id(candidate.partition_id)));
                        added_bytes_by_resource[parent_home] += logical_bytes;
                        added_bytes_by_resource[parent_link] += logical_bytes;
                    }
                } else if (cxl_use_streaming_staged() && !eligible_splits.empty()) {
                    // No cohort cap, gain-coverage target, or workload label:
                    // rank each positive-net logical buy and admit it only if
                    // every prior-window priced resource remains below the
                    // 0.90 projected-utilization guard.  The simulator later
                    // accounts the added descriptor/view traffic exactly.
                    std::sort(
                        eligible_splits.begin(), eligible_splits.end(),
                        [this](const ResourceActionCandidate& lhs,
                               const ResourceActionCandidate& rhs) {
                            float lhs_score =
                                (cxl_staged_calibration_r_ * lhs.search_gain_ns - lhs.buy_ns) /
                                std::max(1.0f, lhs.buy_ns);
                            float rhs_score =
                                (cxl_staged_calibration_r_ * rhs.search_gain_ns - rhs.buy_ns) /
                                std::max(1.0f, rhs.buy_ns);
                            if (lhs_score != rhs_score) {
                                return lhs_score > rhs_score;
                            }
                            return lhs.partition_id < rhs.partition_id;
                        });
                    std::unordered_map<string, float> projected_utilization;
                    if (cxl_resource_price_snapshot_ != nullptr) {
                        for (const auto& resource : cxl_resource_price_snapshot_->resources) {
                            projected_utilization[resource.resource_id] = resource.utilization;
                        }
                    }
                    for (const auto& candidate : eligible_splits) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        bool utilization_ok = true;
                        std::unordered_map<string, float> next_utilization =
                            projected_utilization;
                        const float logical_bytes = static_cast<float>(
                            round_up_line_bytes(
                                candidate.records,
                                params_->cxl_entry_bytes,
                                params_->cxl_line_bytes) +
                            2 * std::max(0, params_->cxl_metadata_bytes) +
                            std::max<int64_t>(0, (candidate.records + 7) / 8));
                        if (cxl_resource_price_snapshot_ != nullptr) {
                            const string parent_home = "mc:" +
                                std::to_string(cxl_resource_home_id(candidate.partition_id));
                            for (const auto& resource : cxl_resource_price_snapshot_->resources) {
                                const bool is_parent_mc = resource.resource_id == parent_home;
                                // Quake's semantic oracle has no PBR topology;
                                // conservatively charge every observed link.
                                const bool is_link =
                                    resource.resource_id.rfind("link:", 0) == 0;
                                if (!is_parent_mc && !is_link) {
                                    continue;
                                }
                                float projected = resource.utilization;
                                float capacity_bytes =
                                    resource_capacity_bytes(resource);
                                if (capacity_bytes > 0.0f) {
                                    projected = next_utilization[resource.resource_id] +
                                        logical_bytes / std::max(1.0f, capacity_bytes);
                                }
                                next_utilization[resource.resource_id] = projected;
                                if (projected >= 0.90f) {
                                    utilization_ok = false;
                                }
                            }
                        }
                        if (!utilization_ok) {
                            decision.rejection_reason = "projected_resource_utilization";
                            split_candidate_budget_rejected_count++;
                            continue;
                        }
                        projected_utilization = std::move(next_utilization);
                        vector<ResourceActionCandidate> singleton = {candidate};
                        commit_split_cohort(
                            singleton,
                            candidate.credit_ns,
                            candidate.buy_ns,
                            0.0f,
                            candidate.buy_ns,
                            candidate.search_gain_ns,
                            candidate.search_gain_ns);
                    }
                } else if (!eligible_splits.empty()) {
                    int max_splits = std::max(0, params_->cxl_adaptive_max_split_budget);
                    if (cxl_use_search_first() &&
                        params_->cxl_search_first_max_cohort > 0) {
                        max_splits = max_splits > 0
                                         ? std::min(
                                               max_splits,
                                               params_->cxl_search_first_max_cohort)
                                         : params_->cxl_search_first_max_cohort;
                    }
                    size_t prefix_limit = eligible_splits.size();
                    if (max_splits > 0) {
                        prefix_limit = std::min(prefix_limit, static_cast<size_t>(max_splits));
                    }
                    float available_search_gain_ns = cohort_search_gain(eligible_splits);
                    if (cxl_use_search_first()) {
                        search_first_available_gain_ns += available_search_gain_ns;
                        float gain_target = std::min(
                            1.0f,
                            std::max(0.0f, params_->cxl_search_first_gain_target));
                        float target_gain_ns = gain_target * available_search_gain_ns;
                        float prefix_credit_ns = 0.0f;
                        float prefix_search_gain_ns = 0.0f;
                        float prefix_variable_buy_ns = 0.0f;
                        int64_t prefix_records = 0;
                        size_t selected_size = 0;
                        float selected_credit_ns = 0.0f;
                        float selected_search_gain_ns = 0.0f;
                        float selected_variable_buy_ns = 0.0f;
                        float selected_shared_buy_ns = 0.0f;
                        float selected_total_buy_ns = 0.0f;
                        if (target_gain_ns > 0.0f) {
                            for (size_t index = 0; index < prefix_limit; index++) {
                                const auto& candidate = eligible_splits[index];
                                prefix_credit_ns += candidate.credit_ns;
                                prefix_search_gain_ns += candidate.search_gain_ns;
                                prefix_variable_buy_ns += candidate.buy_ns;
                                prefix_records += candidate.records;
                                float shared_buy_ns =
                                    cxl_resource_split_cohort_shared_buy_ns(prefix_records);
                                selected_size = index + 1;
                                selected_credit_ns = prefix_credit_ns;
                                selected_search_gain_ns = prefix_search_gain_ns;
                                selected_variable_buy_ns = prefix_variable_buy_ns;
                                selected_shared_buy_ns = shared_buy_ns;
                                selected_total_buy_ns =
                                    prefix_variable_buy_ns + shared_buy_ns;
                                if (prefix_search_gain_ns >= target_gain_ns) {
                                    break;
                                }
                            }
                        }
                        if (selected_size > 0) {
                            vector<ResourceActionCandidate> selected_cohort(
                                eligible_splits.begin(),
                                eligible_splits.begin() + selected_size);
                            resource_best_cohort_size =
                                static_cast<int64_t>(selected_size);
                            resource_best_cohort_credit_ns = selected_credit_ns;
                            resource_best_cohort_buy_ns = selected_total_buy_ns;
                            resource_best_cohort_ratio = selected_credit_ns /
                                std::max(1.0f, selected_total_buy_ns);
                            commit_split_cohort(
                                selected_cohort,
                                selected_credit_ns,
                                selected_variable_buy_ns,
                                selected_shared_buy_ns,
                                selected_total_buy_ns,
                                selected_search_gain_ns,
                                available_search_gain_ns);
                        }
                        float selected_gain_fraction = available_search_gain_ns > 0.0f
                            ? selected_search_gain_ns / available_search_gain_ns
                            : 0.0f;
                        for (const auto& candidate : eligible_splits) {
                            CxlPolicyDecision& decision =
                                resource_policy_decisions[candidate.decision_index];
                            if (decision.selected) {
                                continue;
                            }
                            if (target_gain_ns <= 0.0f) {
                                decision.rejection_reason = "search_gain_target_zero";
                            } else if (selected_gain_fraction >= gain_target) {
                                decision.rejection_reason = "search_gain_target_satisfied";
                            } else {
                                decision.rejection_reason = "search_first_max_cohort";
                            }
                        }
                    } else {
                        float prefix_credit_ns = 0.0f;
                        float prefix_variable_buy_ns = 0.0f;
                        int64_t prefix_records = 0;
                        size_t best_ratio_size = 0;
                        float best_ratio_variable_buy_ns = 0.0f;
                        float best_ratio_shared_buy_ns = 0.0f;
                        float best_ratio_total_buy_ns = 0.0f;
                        float best_surplus_ns = -1.0f;
                        size_t selected_size = 0;
                        float selected_credit_ns = 0.0f;
                        float selected_variable_buy_ns = 0.0f;
                        float selected_shared_buy_ns = 0.0f;
                        float selected_total_buy_ns = 0.0f;
                        bool profitable_prefix_exists = false;
                        for (size_t index = 0; index < prefix_limit; index++) {
                            const auto& candidate = eligible_splits[index];
                            prefix_credit_ns += candidate.credit_ns;
                            prefix_variable_buy_ns += candidate.buy_ns;
                            prefix_records += candidate.records;
                            float shared_buy_ns =
                                cxl_resource_split_cohort_shared_buy_ns(prefix_records);
                            float total_buy_ns = prefix_variable_buy_ns + shared_buy_ns;
                            float ratio = prefix_credit_ns / std::max(1.0f, total_buy_ns);
                            if (best_ratio_size == 0 || ratio > resource_best_cohort_ratio) {
                                best_ratio_size = index + 1;
                                best_ratio_variable_buy_ns = prefix_variable_buy_ns;
                                best_ratio_shared_buy_ns = shared_buy_ns;
                                best_ratio_total_buy_ns = total_buy_ns;
                                resource_best_cohort_size = static_cast<int64_t>(index + 1);
                                resource_best_cohort_credit_ns = prefix_credit_ns;
                                resource_best_cohort_buy_ns = total_buy_ns;
                                resource_best_cohort_ratio = ratio;
                            }
                            float surplus_ns = prefix_credit_ns - total_buy_ns;
                            profitable_prefix_exists = profitable_prefix_exists || surplus_ns >= 0.0f;
                            if (surplus_ns >= 0.0f && total_buy_ns <= available_budget_ns &&
                                surplus_ns > best_surplus_ns) {
                                best_surplus_ns = surplus_ns;
                                selected_size = index + 1;
                                selected_credit_ns = prefix_credit_ns;
                                selected_variable_buy_ns = prefix_variable_buy_ns;
                                selected_shared_buy_ns = shared_buy_ns;
                                selected_total_buy_ns = total_buy_ns;
                            }
                        }
                        if (best_ratio_size > 0) {
                            vector<ResourceActionCandidate> best_ratio_cohort(
                                eligible_splits.begin(),
                                eligible_splits.begin() + best_ratio_size);
                            set_cohort_telemetry(
                                best_ratio_cohort,
                                resource_best_cohort_credit_ns,
                                best_ratio_variable_buy_ns,
                                best_ratio_shared_buy_ns,
                                best_ratio_total_buy_ns,
                                cohort_search_gain(best_ratio_cohort),
                                available_search_gain_ns);
                        }
                        if (selected_size > 0) {
                            vector<ResourceActionCandidate> selected_cohort(
                                eligible_splits.begin(),
                                eligible_splits.begin() + selected_size);
                            commit_split_cohort(
                                selected_cohort,
                                selected_credit_ns,
                                selected_variable_buy_ns,
                                selected_shared_buy_ns,
                                selected_total_buy_ns,
                                cohort_search_gain(selected_cohort),
                                available_search_gain_ns);
                        }
                        for (const auto& candidate : eligible_splits) {
                            CxlPolicyDecision& decision =
                                resource_policy_decisions[candidate.decision_index];
                            if (decision.selected) {
                                continue;
                            }
                            if (selected_size > 0) {
                                decision.rejection_reason = "outside_selected_cohort";
                            } else if (profitable_prefix_exists) {
                                decision.rejection_reason = "cohort_maintenance_time_budget";
                                split_candidate_budget_rejected_count++;
                            } else {
                                decision.rejection_reason = "cohort_credit_below_buy";
                            }
                        }
                    }
                }

                if (cxl_use_streaming_staged() && !resource_force_active) {
                    for (const auto& candidate : reassign_actions) {
                        resource_policy_decisions[candidate.decision_index].rejection_reason =
                            "staged_policy_split_only";
                    }
                } else if ((cxl_use_default_plus() ||
                            cxl_use_search_guarded_plus()) &&
                           !resource_force_active) {
                    // Default-plus changes only split admission.  Preserve all
                    // native Quake reassignments exactly.
                    for (const auto& candidate : reassign_actions) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        if (selected_ids.count(candidate.partition_id) > 0) {
                            decision.rejection_reason = "conflicting_or_mandatory_action";
                            continue;
                        }
                        partitions_to_delete.push_back(candidate.partition_id);
                        cxl_reassign_credit_ns_.erase(candidate.partition_id);
                        selected_ids.insert(candidate.partition_id);
                        decision.selected = true;
                        decision.rejection_reason.clear();
                    }
                } else if (cxl_use_search_first() && !resource_force_active) {
                    vector<ResourceActionCandidate> eligible_reassigns;
                    for (const auto& candidate : reassign_actions) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        if (selected_ids.count(candidate.partition_id) > 0) {
                            decision.rejection_reason = "conflicting_or_mandatory_action";
                        } else {
                            eligible_reassigns.push_back(candidate);
                        }
                    }
                    float available_search_gain_ns =
                        cohort_search_gain(eligible_reassigns);
                    search_first_available_gain_ns += available_search_gain_ns;
                    float gain_target = std::min(
                        1.0f,
                        std::max(0.0f, params_->cxl_search_first_gain_target));
                    float target_gain_ns = gain_target * available_search_gain_ns;
                    size_t prefix_limit = eligible_reassigns.size();
                    int max_reassign = std::max(
                        0, params_->cxl_adaptive_max_reassign_budget);
                    if (max_reassign > 0) {
                        prefix_limit = std::min(
                            prefix_limit, static_cast<size_t>(max_reassign));
                    }
                    vector<ResourceActionCandidate> selected_reassigns;
                    float selected_credit_ns = 0.0f;
                    float selected_search_gain_ns = 0.0f;
                    float selected_buy_ns = 0.0f;
                    if (target_gain_ns > 0.0f) {
                        for (size_t index = 0; index < prefix_limit; index++) {
                            const auto& candidate = eligible_reassigns[index];
                            selected_reassigns.push_back(candidate);
                            selected_credit_ns += candidate.credit_ns;
                            selected_search_gain_ns += candidate.search_gain_ns;
                            selected_buy_ns += candidate.buy_ns;
                            if (selected_search_gain_ns >= target_gain_ns) {
                                break;
                            }
                        }
                    }
                    if (!selected_reassigns.empty()) {
                        set_cohort_telemetry(
                            selected_reassigns,
                            selected_credit_ns,
                            selected_buy_ns,
                            0.0f,
                            selected_buy_ns,
                            selected_search_gain_ns,
                            available_search_gain_ns);
                        for (const auto& candidate : selected_reassigns) {
                            CxlPolicyDecision& decision =
                                resource_policy_decisions[candidate.decision_index];
                            partitions_to_delete.push_back(candidate.partition_id);
                            cxl_reassign_credit_ns_.erase(candidate.partition_id);
                            selected_ids.insert(candidate.partition_id);
                            decision.selected = true;
                            decision.rejection_reason.clear();
                        }
                        available_budget_ns = std::max(
                            0.0f, available_budget_ns - selected_buy_ns);
                        resource_selected_buy_ns += selected_buy_ns;
                        search_first_selected_gain_ns += selected_search_gain_ns;
                    }
                    float selected_gain_fraction = available_search_gain_ns > 0.0f
                        ? selected_search_gain_ns / available_search_gain_ns
                        : 0.0f;
                    for (const auto& candidate : eligible_reassigns) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        if (decision.selected) {
                            continue;
                        }
                        if (target_gain_ns <= 0.0f) {
                            decision.rejection_reason = "search_gain_target_zero";
                        } else if (selected_gain_fraction >= gain_target) {
                            decision.rejection_reason = "search_gain_target_satisfied";
                        } else {
                            decision.rejection_reason = "search_first_reassign_limit";
                        }
                    }
                } else {
                    for (const auto& candidate : reassign_actions) {
                        CxlPolicyDecision& decision =
                            resource_policy_decisions[candidate.decision_index];
                        bool forced = resource_force_active &&
                                      forced_reassign_ids.count(candidate.partition_id) > 0;
                        if (resource_force_active && !forced) {
                            decision.rejection_reason = "audit_not_forced";
                            continue;
                        }
                        if (selected_ids.count(candidate.partition_id) > 0) {
                            decision.rejection_reason = "conflicting_or_mandatory_action";
                            continue;
                        }
                        if (!resource_force_active && candidate.buy_ns > available_budget_ns) {
                            decision.rejection_reason = "maintenance_time_budget";
                            continue;
                        }
                        partitions_to_delete.push_back(candidate.partition_id);
                        cxl_reassign_credit_ns_.erase(candidate.partition_id);
                        selected_ids.insert(candidate.partition_id);
                        available_budget_ns = std::max(
                            0.0f, available_budget_ns - candidate.buy_ns);
                        resource_selected_buy_ns += candidate.buy_ns;
                        decision.selected = true;
                        decision.rejection_reason.clear();
                    }
                }
            }
            if (resource_force_active) {
                auto append_missing_forced = [&](const std::unordered_set<int64_t>& requested,
                                                 const char* action_kind) {
                    for (int64_t partition_id : requested) {
                        bool found = std::any_of(
                            resource_policy_decisions.begin(),
                            resource_policy_decisions.end(),
                            [&](const CxlPolicyDecision& decision) {
                                return decision.partition_id == partition_id &&
                                       decision.action_kind == action_kind;
                            });
                        if (!found) {
                            CxlPolicyDecision decision;
                            decision.action_kind = action_kind;
                            decision.partition_id = partition_id;
                            decision.home_id = cxl_resource_home_id(partition_id);
                            decision.rejection_reason = "forced_candidate_unavailable";
                            resource_policy_decisions.push_back(decision);
                        }
                    }
                };
                append_missing_forced(forced_split_ids, "split");
                append_missing_forced(forced_reassign_ids, "reassign");
            }
            split_candidate_selected_count = static_cast<int64_t>(
                partitions_to_split.size());
            split_budget_by_time = split_candidate_selected_count;
            reassign_budget_by_time = static_cast<int64_t>(
                partitions_to_delete.size());
        } else if (cxl_use_streaming_rent_buy()) {
            std::sort(
                streaming_action_candidates.begin(),
                streaming_action_candidates.end(),
                [](const StreamingActionCandidate& lhs, const StreamingActionCandidate& rhs) {
                    if (lhs.rank_ns == rhs.rank_ns) {
                        return lhs.buy_ns < rhs.buy_ns;
                    }
                    return lhs.rank_ns > rhs.rank_ns;
                });
            float available_budget_ns = std::max(0.0f, cxl_maintenance_budget_credit_ns_);
            std::unordered_set<int64_t> mandatory_deletes(
                partitions_to_delete.begin(), partitions_to_delete.end());
            bool selected_unknown_split = false;
            bool selected_unknown_reassign = false;
            for (const auto& candidate : streaming_action_candidates) {
                float structural_payment_ns = candidate.split
                                                  ? std::max(0.0f, candidate.buy_ns - candidate.local_credit_ns)
                                                  : 0.0f;
                // Global APS drift can match a list's own measured rent, but it
                // cannot fund an otherwise cold split by itself. This keeps the
                // online rule parameter-free and prevents system-wide growth
                // from being charged repeatedly to unrelated partitions.
                float structural_match_limit_ns = candidate.split
                                                      ? candidate.local_credit_ns
                                                      : 0.0f;
                bool rent_paid = candidate.local_credit_ns >= candidate.buy_ns ||
                                 (candidate.split &&
                                  structural_payment_ns <= structural_match_limit_ns &&
                                  cxl_streaming_structural_credit_ns_ >= structural_payment_ns);
                if (!rent_paid) {
                    split_candidate_roi_rejected_count += candidate.split ? 1 : 0;
                    continue;
                }
                if (candidate.buy_ns > available_budget_ns) {
                    split_candidate_budget_rejected_count += candidate.split ? 1 : 0;
                    continue;
                }
                if (candidate.split && !cxl_has_observed_split_cost_ && selected_unknown_split) {
                    continue;
                }
                if (!candidate.split && cxl_observed_reassign_ns_per_record_ <= 0.0f &&
                    selected_unknown_reassign) {
                    continue;
                }
                if (candidate.split && mandatory_deletes.count(candidate.partition_id) > 0) {
                    continue;
                }
                if (candidate.split) {
                    partitions_to_split.push_back(candidate.partition_id);
                    cxl_split_credit_ns_.erase(candidate.partition_id);
                    cxl_streaming_structural_credit_ns_ = std::max(
                        0.0f,
                        cxl_streaming_structural_credit_ns_ - structural_payment_ns);
                    selected_unknown_split = selected_unknown_split || !cxl_has_observed_split_cost_;
                } else {
                    partitions_to_delete.push_back(candidate.partition_id);
                    cxl_reassign_credit_ns_.erase(candidate.partition_id);
                    selected_unknown_reassign =
                        selected_unknown_reassign || cxl_observed_reassign_ns_per_record_ <= 0.0f;
                }
                available_budget_ns -= candidate.buy_ns;
                streaming_selected_buy_ns += candidate.buy_ns;
                streaming_selected_count++;
            }
            cxl_maintenance_budget_credit_ns_ = available_budget_ns;
            split_candidate_selected_count = static_cast<int64_t>(partitions_to_split.size());
            split_budget_by_time = split_candidate_selected_count;
            reassign_budget_by_time = static_cast<int64_t>(partitions_to_delete.size());
        }

        if (!cxl_use_streaming_rent_buy() && cxl_use_roi_budget() && !cxl_split_candidates.empty()) {
            std::sort(
                cxl_split_candidates.begin(),
                cxl_split_candidates.end(),
                [](const CxlSplitCandidate& lhs, const CxlSplitCandidate& rhs) {
                    if (lhs.gain_ns == rhs.gain_ns) {
                        return lhs.roi > rhs.roi;
                    }
                    return lhs.gain_ns > rhs.gain_ns;
                });
            int split_budget = cxl_adaptive_split_budget(total_partitions);
            if (cxl_use_structural_budget()) {
                int max_budget = std::max(0, params_->cxl_adaptive_max_split_budget);
                int debt_budget = static_cast<int>(std::ceil(
                    static_cast<float>(structural_split_debt) *
                    std::max(0.0f, params_->cxl_adaptive_growth_debt_repay_fraction)));
                split_budget = std::max(split_budget, debt_budget);
                split_budget = std::max(
                    split_budget,
                    static_cast<int>(std::min<int64_t>(native_split_candidate_count, max_budget)));
                split_budget = std::min(max_budget, split_budget);
            }
            split_budget_by_time = cxl_time_budget_split_limit();
            split_budget = std::min(split_budget, static_cast<int>(split_budget_by_time));
            for (const auto& candidate : cxl_split_candidates) {
                if (static_cast<int>(partitions_to_split.size()) >= split_budget) {
                    break;
                }
                bool deleted = std::find(
                    partitions_to_delete.begin(),
                    partitions_to_delete.end(),
                    candidate.partition_id) != partitions_to_delete.end();
                if (!deleted) {
                    partitions_to_split.push_back(candidate.partition_id);
                }
            }
            split_candidate_selected_count = static_cast<int64_t>(partitions_to_split.size());
            split_candidate_budget_rejected_count = std::max<int64_t>(
                0,
                static_cast<int64_t>(cxl_split_candidates.size()) - split_candidate_selected_count);
        } else if (!cxl_use_streaming_rent_buy()) {
            split_candidate_selected_count = static_cast<int64_t>(partitions_to_split.size());
        }

        if (!cxl_use_streaming_rent_buy() && cxl_use_structural_budget()) {
            float count_slack = std::max(0.0f, params_->cxl_adaptive_partition_count_slack);
            int upper_partition_count = static_cast<int>(std::ceil(
                static_cast<float>(std::max(1, desired_partition_count)) * (1.0f + count_slack)));
            int projected_partition_count = total_partitions
                                            + static_cast<int>(partitions_to_split.size())
                                            - static_cast<int>(partitions_to_delete.size());
            int reassign_needed = std::max(0, projected_partition_count - upper_partition_count);
            int reassign_budget = std::min(
                reassign_needed,
                std::max(0, params_->cxl_adaptive_max_reassign_budget));
            if (reassign_budget > 0) {
                struct ColdPartition {
                    int64_t partition_id;
                    int hit_count;
                    int partition_size;
                };
                std::unordered_set<int64_t> selected_ids(
                    partitions_to_split.begin(), partitions_to_split.end());
                selected_ids.insert(partitions_to_delete.begin(), partitions_to_delete.end());
                vector<ColdPartition> cold_candidates;
                for (const auto& partition_id : all_partition_ids) {
                    if (selected_ids.count(partition_id) > 0) {
                        continue;
                    }
                    int partition_size = partition_manager_->get_partition_size(partition_id);
                    if (partition_size <= 0 || partition_size > cxl_target_partition_size_) {
                        continue;
                    }
                    cold_candidates.push_back(
                        {partition_id, aggregated_hits[partition_id], partition_size});
                }
                std::sort(
                    cold_candidates.begin(),
                    cold_candidates.end(),
                    [](const ColdPartition& lhs, const ColdPartition& rhs) {
                        if (lhs.hit_count == rhs.hit_count) {
                            return lhs.partition_size < rhs.partition_size;
                        }
                        return lhs.hit_count < rhs.hit_count;
                    });
                for (const auto& candidate : cold_candidates) {
                    if (reassign_budget <= 0) {
                        break;
                    }
                    partitions_to_delete.push_back(candidate.partition_id);
                    reassign_budget--;
                }
            }
        }

        if (!cxl_use_streaming_rent_buy() && cxl_use_aps_feedback() && params_->cxl_adaptive_reassign_time_budget_enabled &&
            !partitions_to_delete.empty()) {
            int count_cap = std::min(
                static_cast<int>(partitions_to_delete.size()),
                std::max(0, params_->cxl_adaptive_max_reassign_budget));
            if (cxl_observed_reassign_ns_per_record_ <= 0.0f) {
                count_cap = std::min(
                    count_cap,
                    std::max(0, params_->cxl_adaptive_warmup_reassign_budget));
            }

            float estimated_split_ns = 0.0f;
            for (const auto& partition_id : partitions_to_split) {
                int64_t partition_size = std::max<int64_t>(
                    0, partition_manager_->get_partition_size(partition_id));
                if (cxl_has_observed_split_cost_) {
                    estimated_split_ns +=
                        cxl_observed_split_ns_per_record_ * static_cast<float>(partition_size) +
                        cxl_observed_refine_ns_per_split_;
                } else {
                    estimated_split_ns += std::max(0.0f, params_->cxl_adaptive_action_overhead_ns);
                }
            }
            float remaining_budget_ns = std::max(
                0.0f,
                cxl_maintenance_time_budget_ns_ - estimated_split_ns);
            float selected_reassign_ns = 0.0f;
            vector<int64_t> budgeted_partitions_to_delete;
            budgeted_partitions_to_delete.reserve(static_cast<size_t>(count_cap));
            for (const auto& partition_id : partitions_to_delete) {
                if (static_cast<int>(budgeted_partitions_to_delete.size()) >= count_cap) {
                    break;
                }
                int64_t partition_size = std::max<int64_t>(
                    0, partition_manager_->get_partition_size(partition_id));
                float estimated_action_ns =
                    cxl_observed_reassign_ns_per_record_ > 0.0f
                        ? cxl_observed_reassign_ns_per_record_ * static_cast<float>(partition_size) +
                              std::max(0.0f, params_->cxl_adaptive_action_overhead_ns)
                        : 0.0f;
                if (cxl_observed_reassign_ns_per_record_ > 0.0f &&
                    selected_reassign_ns + estimated_action_ns > remaining_budget_ns) {
                    continue;
                }
                budgeted_partitions_to_delete.push_back(partition_id);
                selected_reassign_ns += estimated_action_ns;
            }
            partitions_to_delete.swap(budgeted_partitions_to_delete);
            reassign_budget_by_time = static_cast<int64_t>(partitions_to_delete.size());
        } else if (!cxl_use_streaming_rent_buy()) {
            reassign_budget_by_time = static_cast<int64_t>(partitions_to_delete.size());
        }

        quake_free(new_centroids_buffer, partition_manager_->d() * sizeof(float));
    }


    // Convert partition ID vectors to Torch tensors.
    Tensor partitions_to_delete_tens = torch::from_blob(
        partitions_to_delete.data(), {static_cast<int64_t>(partitions_to_delete.size())},
        torch::kInt64).clone();
    Tensor partitions_to_split_tens = torch::from_blob(
        partitions_to_split.data(), {static_cast<int64_t>(partitions_to_split.size())},
        torch::kInt64).clone();

    int64_t cxl_selected_maintenance_records = 0;
    int64_t split_records_read = 0;
    int64_t reassign_records_read = 0;
    for (const auto& partition_id : partitions_to_delete) {
        reassign_records_read += std::max<int64_t>(0, partition_manager_->get_partition_size(partition_id));
    }
    for (const auto& partition_id : partitions_to_split) {
        split_records_read += std::max<int64_t>(0, partition_manager_->get_partition_size(partition_id));
    }
    if (params_->enable_cxl_cost_model &&
        (params_->cxl_workload_adaptive || cxl_use_streaming_rent_buy() ||
         cxl_use_resource_rent_buy())) {
        cxl_selected_maintenance_records = reassign_records_read + split_records_read;
    }

    // STEP 3: Process deletions.
    auto start_delete = steady_clock::now();
    if (partitions_to_delete_tens.numel() > 0) {
        partition_manager_->delete_partitions(partitions_to_delete_tens, true);
    }
    auto end_delete = steady_clock::now();

    // STEP 4: Process splits.
    auto start_split = steady_clock::now();
    shared_ptr<Clustering> split_partitions;
    torch::Tensor refine_partitions;
    bool refinement_skipped = false;
    int64_t refinement_source_split_count = 0;
    RefinementWorkInfo refinement_work;
    vector<CxlSplitLineage> split_lineage;
    vector<int64_t> lineage_source_ids;
    vector<int64_t> lineage_source_sizes;
    vector<int64_t> refinement_lineage_source_ids;
    vector<int64_t> refinement_lineage_source_sizes;
    std::unordered_map<int64_t, vector<int64_t>> split_source_order;
    std::unordered_map<int64_t, vector<int64_t>> split_child_ids;
    if (partitions_to_split_tens.numel() > 0) {
        for (const auto& parent_id : partitions_to_split) {
            auto it = partition_manager_->partition_store_->partitions_.find(parent_id);
            if (it == partition_manager_->partition_store_->partitions_.end() ||
                it->second == nullptr) {
                continue;
            }
            const int64_t count = std::max<int64_t>(0, it->second->num_vectors_);
            const int64_t* ids = it->second->ids_;
            if (ids != nullptr) {
                split_source_order[parent_id] = vector<int64_t>(ids, ids + count);
            }
        }
        // split the partitions into two
        split_partitions = partition_manager_->split_partitions(partitions_to_split_tens, params_->split_knn_iterations);

        // remove old partitions
        partition_manager_->delete_partitions(partitions_to_split_tens, false);

        // add new partitions
        partition_manager_->add_partitions(split_partitions);
        refine_partitions = split_partitions->partition_ids;
        if (split_partitions->partition_ids.defined()) {
            Tensor output_ids = split_partitions->partition_ids.to(torch::kCPU).contiguous();
            const int64_t* ids = output_ids.data_ptr<int64_t>();
            for (size_t parent_index = 0;
                 parent_index < partitions_to_split.size() &&
                 static_cast<int64_t>(2 * parent_index + 1) < output_ids.numel();
                 ++parent_index) {
                split_child_ids[partitions_to_split[parent_index]] = {
                    ids[2 * parent_index], ids[2 * parent_index + 1]};
            }
        }
        if (cxl_use_resource_rent_buy() &&
            split_partitions->partition_ids.defined()) {
            Tensor output_ids = split_partitions->partition_ids.to(torch::kCPU).contiguous();
            const int64_t* ids = output_ids.data_ptr<int64_t>();
            for (int64_t index = 0; index + 1 < output_ids.numel(); index += 2) {
                cxl_split_sibling_pairs_.emplace_back(ids[index], ids[index + 1]);
            }
        }
    } else { 
        refine_partitions = torch::empty({0}, torch::kInt64);
    }
    auto end_split = steady_clock::now();

    // STEP 5: Perform local refinement on newly split partitions.
    if (refine_partitions.numel() > 0) {
        std::string refinement_mode = params_->cxl_adaptive_refinement_mode;
        std::transform(
            refinement_mode.begin(), refinement_mode.end(), refinement_mode.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        bool lazy_refinement = (cxl_use_roi_budget() || cxl_use_streaming_rent_buy() ||
                                cxl_use_resource_rent_buy()) &&
                               (refinement_mode == "lazy" || refinement_mode == "none" ||
                                refinement_mode == "deferred");
        if (lazy_refinement) {
            refinement_skipped = true;
        } else {
            torch::Tensor selected_refine_partitions = refine_partitions;
            bool bounded_refinement = (cxl_use_roi_budget() || cxl_use_streaming_rent_buy() ||
                                       cxl_use_resource_rent_buy()) &&
                                      (refinement_mode == "bounded" || refinement_mode == "budgeted");
            if (bounded_refinement) {
                int max_parent_splits = std::max(0, params_->cxl_adaptive_max_refine_splits);
                int64_t max_children = static_cast<int64_t>(max_parent_splits) * 2;
                if (max_children <= 0) {
                    selected_refine_partitions = torch::empty({0}, torch::kInt64);
                    refinement_skipped = true;
                } else if (selected_refine_partitions.numel() > max_children) {
                    selected_refine_partitions = selected_refine_partitions.slice(0, 0, max_children);
                }
            }
            refinement_source_split_count = std::min<int64_t>(
                static_cast<int64_t>(partitions_to_split.size()),
                (selected_refine_partitions.numel() + 1) / 2);
            if (selected_refine_partitions.numel() > 0) {
                refinement_work = local_refinement(selected_refine_partitions);
            }
        }
    }
    auto end_refinement = steady_clock::now();
    int64_t refinement_time_us = static_cast<int64_t>(duration_cast<microseconds>(end_refinement - end_split).count());

    // Step 6: Recluster any partitions

    // STEP 7: Clean up any empty partitions
    vector<int64_t> empty_ids = {};
    for (auto pair : partition_manager_->partition_store_->partitions_) {
        if (pair.second->num_vectors_ <= 0) {
            empty_ids.emplace_back(pair.first);
        }
    }
    if (empty_ids.size() > 0) {
        partition_manager_->delete_partitions(torch::from_blob(empty_ids.data(), {static_cast<int64_t>(empty_ids.size())}, torch::kInt64));
    }

    // Build a stable record-origin map from the state immediately before local
    // refinement.  Provisional split children are folded back into their
    // retired parent source order; neighboring refinement partitions retain
    // their own source identities.  This makes the refined child views
    // reconstructible across every source partition/home.
    std::unordered_map<int64_t, int64_t> split_child_to_parent;
    for (const auto& pair : split_child_ids) {
        for (int64_t child_id : pair.second) {
            split_child_to_parent[child_id] = pair.first;
        }
    }
    std::unordered_map<int64_t, vector<int64_t>> lineage_source_orders =
        split_source_order;
    for (const auto& pair : refinement_work.source_orders) {
        if (split_child_to_parent.count(pair.first) == 0) {
            lineage_source_orders[pair.first] = pair.second;
        }
    }
    std::unordered_set<int64_t> refinement_logical_sources;
    for (const auto& pair : refinement_work.source_orders) {
        auto parent_it = split_child_to_parent.find(pair.first);
        refinement_logical_sources.insert(
            parent_it == split_child_to_parent.end() ? pair.first : parent_it->second);
    }
    refinement_lineage_source_ids.assign(
        refinement_logical_sources.begin(), refinement_logical_sources.end());
    std::sort(
        refinement_lineage_source_ids.begin(),
        refinement_lineage_source_ids.end());
    for (int64_t source_id : refinement_lineage_source_ids) {
        refinement_lineage_source_sizes.push_back(static_cast<int64_t>(
            lineage_source_orders.at(source_id).size()));
    }
    lineage_source_ids.reserve(lineage_source_orders.size());
    for (const auto& pair : lineage_source_orders) {
        lineage_source_ids.push_back(pair.first);
    }
    std::sort(lineage_source_ids.begin(), lineage_source_ids.end());
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> record_origins;
    for (int64_t source_id : lineage_source_ids) {
        const auto& source = lineage_source_orders.at(source_id);
        lineage_source_sizes.push_back(static_cast<int64_t>(source.size()));
        for (size_t position = 0; position < source.size(); ++position) {
            auto inserted = record_origins.emplace(
                source[position],
                std::make_pair(source_id, static_cast<int64_t>(position)));
            if (!inserted.second) {
                throw std::runtime_error(
                    "duplicate record origin while constructing CXL split lineage");
            }
        }
    }

    // Read child memberships after local refinement and empty-child cleanup.
    // The trace must describe the semantic state Quake leaves behind, rather
    // than the provisional split assignment used before refinement.
    for (const auto& parent_id : partitions_to_split) {
        auto children_it = split_child_ids.find(parent_id);
        if (children_it == split_child_ids.end()) {
            continue;
        }
        CxlSplitLineage lineage;
        lineage.parent_id = parent_id;
        lineage.child_ids = children_it->second;
        lineage.source_order_gather_run_counts.assign(lineage.child_ids.size(), 0);
        lineage.child_gather_line_footprint.assign(lineage.child_ids.size(), 0);
        std::unordered_map<int64_t, vector<vector<int64_t>>> fragment_positions;
        for (size_t child_index = 0; child_index < lineage.child_ids.size(); ++child_index) {
            int64_t child_id = lineage.child_ids[child_index];
            auto child_it = partition_manager_->partition_store_->partitions_.find(child_id);
            if (child_it != partition_manager_->partition_store_->partitions_.end() &&
                child_it->second != nullptr) {
                const int64_t child_count =
                    std::max<int64_t>(0, child_it->second->num_vectors_);
                lineage.final_child_sizes.push_back(child_count);
                lineage.child_logical_line_footprint.push_back(
                    static_cast<int64_t>(round_up_line_bytes(
                        child_count,
                        params_->cxl_entry_bytes,
                        params_->cxl_line_bytes)));
                const int64_t* child_ptr = child_it->second->ids_;
                if (child_ptr != nullptr) {
                    for (int64_t item = 0; item < child_count; ++item) {
                        auto origin_it = record_origins.find(child_ptr[item]);
                        if (origin_it == record_origins.end()) {
                            throw std::runtime_error(
                                "refined child record has no pre-refinement CXL source");
                        }
                        auto& source_positions =
                            fragment_positions[origin_it->second.first];
                        if (source_positions.empty()) {
                            source_positions.resize(lineage.child_ids.size());
                        }
                        source_positions[child_index].push_back(
                            origin_it->second.second);
                    }
                }
            } else {
                lineage.final_child_sizes.push_back(0);
                lineage.child_logical_line_footprint.push_back(0);
            }
        }
        vector<int64_t> fragment_source_ids;
        fragment_source_ids.reserve(fragment_positions.size());
        for (const auto& pair : fragment_positions) {
            fragment_source_ids.push_back(pair.first);
        }
        std::sort(fragment_source_ids.begin(), fragment_source_ids.end());
        vector<int64_t> reconstructed_child_sizes(lineage.child_ids.size(), 0);
        for (int64_t source_id : fragment_source_ids) {
            CxlLineageSourceFragment fragment;
            fragment.source_id = source_id;
            fragment.source_size = static_cast<int64_t>(
                lineage_source_orders.at(source_id).size());
            fragment.membership_bitmap_bytes =
                static_cast<int64_t>(lineage.child_ids.size()) *
                ((fragment.source_size + 7) / 8);
            lineage.membership_bitmap_bytes += fragment.membership_bitmap_bytes;
            for (size_t child_index = 0; child_index < lineage.child_ids.size(); ++child_index) {
                const auto& positions = fragment_positions[source_id][child_index];
                const int64_t records = static_cast<int64_t>(positions.size());
                const int64_t runs = source_order_run_count(positions);
                const int64_t gather_bytes = source_order_gather_line_bytes(
                    positions,
                    params_->cxl_entry_bytes,
                    params_->cxl_line_bytes);
                fragment.child_record_counts.push_back(records);
                fragment.child_gather_run_counts.push_back(runs);
                fragment.child_gather_line_bytes.push_back(gather_bytes);
                reconstructed_child_sizes[child_index] += records;
                lineage.source_order_gather_run_counts[child_index] += runs;
                lineage.child_gather_line_footprint[child_index] += gather_bytes;
            }
            lineage.source_fragments.push_back(std::move(fragment));
        }
        if (reconstructed_child_sizes != lineage.final_child_sizes) {
            throw std::runtime_error(
                "CXL split lineage does not reconstruct refined child sizes");
        }
        split_lineage.push_back(std::move(lineage));
    }
    auto end_total = steady_clock::now();

    // STEP 7: Fill in timing details.
    shared_ptr<MaintenanceTimingInfo> timing_info = std::make_shared<MaintenanceTimingInfo>();
    timing_info->delete_time_us = duration_cast<microseconds>(end_delete - start_delete).count();
    timing_info->split_time_us = duration_cast<microseconds>(end_split - start_split).count();
    timing_info->refinement_time_us = refinement_time_us;
    timing_info->decision_time_us = duration_cast<microseconds>(start_delete - start_total).count();
    timing_info->action_time_us = duration_cast<microseconds>(end_total - start_delete).count();
    timing_info->total_time_us = duration_cast<microseconds>(end_total - start_total).count();

    timing_info->n_splits      = static_cast<int64_t>(partitions_to_split.size());
    timing_info->n_deletes     = static_cast<int64_t>(partitions_to_delete.size());
    timing_info->decision_partition_count = decision_partition_count;
    timing_info->centroid_update_count = centroid_update_count;
    timing_info->delete_candidate_count = delete_candidate_count;
    timing_info->delete_candidate_records = delete_candidate_records;
    timing_info->split_candidate_count = split_candidate_count;
    timing_info->split_candidate_selected_count = split_candidate_selected_count;
    timing_info->split_candidate_roi_rejected_count = split_candidate_roi_rejected_count;
    timing_info->split_candidate_budget_rejected_count = split_candidate_budget_rejected_count;
    timing_info->split_records_read = split_records_read;
    timing_info->split_records_written = split_records_read;
    timing_info->reassign_records_read = reassign_records_read;
    timing_info->reassign_records_written = reassign_records_read;
    timing_info->refinement_partition_count = refinement_work.partition_count;
    timing_info->refinement_records_per_iteration = refinement_work.records_per_iteration;
    timing_info->refinement_records_read = refinement_work.records_read;
    timing_info->refinement_records_written = refinement_work.records_written;
    timing_info->refinement_iterations = refinement_work.iterations;
    timing_info->refinement_source_split_count = refinement_source_split_count;
    timing_info->refinement_skipped = refinement_skipped;
    timing_info->observed_split_ns_per_record = static_cast<int64_t>(cxl_observed_split_ns_per_record_);
    timing_info->observed_refine_ns_per_split = static_cast<int64_t>(cxl_observed_refine_ns_per_split_);
    timing_info->observed_refine_ns_per_parent_record =
        static_cast<int64_t>(cxl_observed_refine_ns_per_parent_record_);
    timing_info->observed_refine_read_records_per_parent =
        cxl_observed_refine_read_records_per_parent_;
    timing_info->observed_refine_write_records_per_parent =
        cxl_observed_refine_write_records_per_parent_;
    timing_info->observed_reassign_ns_per_record = static_cast<int64_t>(cxl_observed_reassign_ns_per_record_);
    timing_info->estimated_query_window_ns = static_cast<int64_t>(cxl_estimated_query_window_ns_);
    timing_info->maintenance_time_budget_ns = static_cast<int64_t>(cxl_maintenance_time_budget_ns_);
    timing_info->split_budget_by_time = split_budget_by_time;
    timing_info->reassign_budget_by_time = reassign_budget_by_time;
    timing_info->aps_average_fanout = current_aps_fanout;
    timing_info->aps_average_scanned_records = current_avg_scanned_records;
    timing_info->aps_average_scanned_list_size = current_avg_scanned_list_size;
    timing_info->aps_fanout_growth = cxl_last_fanout_growth_sample_;
    timing_info->aps_scanned_list_size_pressure = cxl_scanned_list_size_pressure_;
    timing_info->payback_windows = cxl_payback_windows_;
    timing_info->partition_count_before = partition_count_before;
    timing_info->partition_count_after = std::max<int64_t>(0, partition_manager_->nlist());
    timing_info->streaming_rent_buy_enabled = cxl_use_streaming_rent_buy();
    timing_info->streaming_candidate_count = streaming_candidate_count;
    timing_info->streaming_selected_count = streaming_selected_count;
    timing_info->streaming_window_rent_ns = static_cast<int64_t>(streaming_window_rent_ns);
    timing_info->streaming_structural_rent_ns =
        static_cast<int64_t>(cxl_streaming_structural_rent_ns_);
    timing_info->streaming_structural_credit_ns =
        static_cast<int64_t>(cxl_streaming_structural_credit_ns_);
    timing_info->streaming_selected_buy_ns = static_cast<int64_t>(streaming_selected_buy_ns);
    timing_info->streaming_budget_credit_ns = static_cast<int64_t>(cxl_maintenance_budget_credit_ns_);
    timing_info->streaming_scan_growth_pressure =
        std::max(cxl_scan_growth_score_, cxl_last_scan_growth_sample_);
    timing_info->resource_rent_buy_enabled = cxl_use_resource_rent_buy();
    timing_info->resource_price_window_id =
        cxl_resource_price_snapshot_ != nullptr
            ? cxl_resource_price_snapshot_->window_id
            : -1;
    timing_info->resource_price_window_duration_ns =
        cxl_resource_price_snapshot_ != nullptr
            ? cxl_resource_price_snapshot_->window_duration_ns
            : 0;
    timing_info->resource_child_probe_factor = std::max(
        0.0f, cxl_resource_child_probe_factor_);
    timing_info->resource_candidate_count = static_cast<int64_t>(
        resource_policy_decisions.size());
    timing_info->resource_selected_count = static_cast<int64_t>(
        std::count_if(
            resource_policy_decisions.begin(),
            resource_policy_decisions.end(),
            [](const CxlPolicyDecision& decision) { return decision.selected; }));
    timing_info->resource_window_rent_ns = static_cast<int64_t>(
        resource_window_rent_ns);
    timing_info->resource_selected_buy_ns = static_cast<int64_t>(
        resource_selected_buy_ns);
    timing_info->resource_split_candidate_count = resource_split_candidate_count;
    timing_info->resource_best_cohort_size = resource_best_cohort_size;
    timing_info->resource_best_cohort_credit_ns = static_cast<int64_t>(
        resource_best_cohort_credit_ns);
    timing_info->resource_best_cohort_buy_ns = static_cast<int64_t>(
        resource_best_cohort_buy_ns);
    timing_info->resource_best_cohort_ratio = resource_best_cohort_ratio;
    timing_info->resource_selected_cohort_size = resource_selected_cohort_size;
    timing_info->resource_selected_cohort_buy_ns = static_cast<int64_t>(
        resource_selected_cohort_buy_ns);
    timing_info->search_first_enabled = cxl_use_search_first();
    timing_info->search_first_gain_target = std::min(
        1.0f,
        std::max(0.0f, params_->cxl_search_first_gain_target));
    timing_info->search_first_max_cohort = std::max(
        0, params_->cxl_search_first_max_cohort);
    timing_info->search_first_available_gain_ns = static_cast<int64_t>(
        search_first_available_gain_ns);
    timing_info->search_first_selected_gain_ns = static_cast<int64_t>(
        search_first_selected_gain_ns);
    timing_info->search_first_selected_gain_fraction =
        search_first_available_gain_ns > 0.0f
            ? search_first_selected_gain_ns / search_first_available_gain_ns
            : 0.0f;
    timing_info->search_first_cxl_split_candidate_count =
        search_first_cxl_split_candidate_count;
    timing_info->search_first_cxl_only_split_candidate_count =
        search_first_cxl_only_split_candidate_count;
    timing_info->search_first_selected_cxl_only_split_count =
        search_first_selected_cxl_only_split_count;
    timing_info->resource_forced_action_set =
        params_->cxl_resource_force_action_set;
    timing_info->resource_policy_decisions = std::move(
        resource_policy_decisions);
    timing_info->split_lineage = std::move(split_lineage);
    timing_info->lineage_source_ids = std::move(lineage_source_ids);
    timing_info->lineage_source_sizes = std::move(lineage_source_sizes);
    timing_info->refinement_lineage_source_ids = std::move(
        refinement_lineage_source_ids);
    timing_info->refinement_lineage_source_sizes = std::move(
        refinement_lineage_source_sizes);

    if (params_->enable_cxl_cost_model &&
        (params_->cxl_workload_adaptive || cxl_use_streaming_rent_buy() ||
         cxl_use_resource_rent_buy())) {
        update_cxl_observed_action_costs(
            split_records_read,
            timing_info->split_time_us,
            timing_info->n_splits,
            timing_info->refinement_time_us,
            timing_info->refinement_records_read,
            timing_info->refinement_records_written,
            reassign_records_read,
            timing_info->delete_time_us);
        cxl_prev_maintenance_records_ = cxl_selected_maintenance_records;
    }

    return timing_info;
}

void MaintenancePolicy::record_query_hits(vector<int64_t> partition_ids) {
    // Keep scan-fraction accounting aligned with the live dynamic index size.
    // The tracker is constructed at build time, but ntotal changes after every
    // streaming add/delete batch.
    hit_count_tracker_->set_total_vectors(
        static_cast<int>(std::max<int64_t>(1, partition_manager_->ntotal())));
    vector<int64_t> scanned_sizes = partition_manager_->get_partition_sizes(partition_ids);
    hit_count_tracker_->add_query_data(partition_ids, scanned_sizes);
}

void MaintenancePolicy::record_cxl_update_counts(int64_t add_count, int64_t delete_count) {
    if (!params_->enable_cxl_cost_model ||
        (!params_->cxl_workload_adaptive && !cxl_use_resource_rent_buy())) {
        return;
    }
    cxl_pending_add_count_ += std::max<int64_t>(0, add_count);
    cxl_pending_delete_count_ += std::max<int64_t>(0, delete_count);
}

void MaintenancePolicy::reset() {
    hit_count_tracker_->reset();
}

MaintenancePolicy::RefinementWorkInfo MaintenancePolicy::local_refinement(
    const torch::Tensor &partition_ids) {
    RefinementWorkInfo work;
    Tensor split_centroids = partition_manager_->parent_->get(partition_ids);
    auto search_params = std::make_shared<SearchParams>();
    search_params->nprobe = 1000;
    search_params->k = params_->refinement_radius;
    search_params->batched_scan = true;
    search_params->track_hits = false;

    if (params_->refinement_radius == 0) {
        return work;
    }

    auto result = partition_manager_->parent_->search(split_centroids, search_params);
    Tensor refine_ids = std::get<0>(torch::_unique(result->ids));
    refine_ids = refine_ids.masked_select(refine_ids != -1);
    work.partition_count = refine_ids.numel();
    if (work.partition_count <= 0) {
        return work;
    }
    Tensor refine_sizes = partition_manager_->get_partition_sizes(refine_ids);
    work.records_per_iteration = refine_sizes.sum().item<int64_t>();
    work.iterations = std::max(1, params_->refinement_iterations);
    work.records_read = work.records_per_iteration * work.iterations;
    work.records_written = work.records_per_iteration;
    Tensor refine_ids_cpu = refine_ids.to(torch::kCPU).contiguous();
    const int64_t* refine_id_ptr = refine_ids_cpu.data_ptr<int64_t>();
    for (int64_t index = 0; index < refine_ids_cpu.numel(); ++index) {
        const int64_t source_id = refine_id_ptr[index];
        auto source_it = partition_manager_->partition_store_->partitions_.find(source_id);
        if (source_it == partition_manager_->partition_store_->partitions_.end() ||
            source_it->second == nullptr) {
            continue;
        }
        const int64_t count = std::max<int64_t>(0, source_it->second->num_vectors_);
        const int64_t* ids = source_it->second->ids_;
        if (ids != nullptr) {
            work.source_orders[source_id] = vector<int64_t>(ids, ids + count);
        } else {
            work.source_orders[source_id] = {};
        }
    }
    partition_manager_->refine_partitions(refine_ids, params_->refinement_iterations);
    return work;
}
