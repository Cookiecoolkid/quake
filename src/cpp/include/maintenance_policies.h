#ifndef MAINTENANCE_POLICY_REFACTORED_H
#define MAINTENANCE_POLICY_REFACTORED_H

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <memory>

#include "partition_manager.h"
#include "hit_count_tracker.h"
#include "maintenance_cost_estimator.h"

/**
 * @brief Maintenance policy that manages partition hit counts and
 * performs maintenance operations (such as deletion and splitting) in a single pass.
 *
 */
class MaintenancePolicy {
 public:
  /**
   * @brief Construct a new MaintenancePolicy object.
   *
   * @param partition_manager Shared pointer to the partition manager.
   * @param params Configuration parameters for the maintenance policy.
   */
  MaintenancePolicy(
      shared_ptr<PartitionManager> partition_manager,
      shared_ptr<MaintenancePolicyParams> params);

  /**
   * @brief Perform maintenance operations including deletion and splitting.
   *
   * @return MaintenanceTimingInfo with timing details.
   */
  shared_ptr<MaintenanceTimingInfo> perform_maintenance();

  /**
   * @brief Record a hit event for a given partition.
   *
   * @param partition_id Identifier of the partition.
   */
  void record_query_hits(vector<int64_t> partition_ids);

  void record_cxl_update_counts(int64_t add_count, int64_t delete_count);

  void set_cxl_resource_price_snapshot(
      shared_ptr<CxlResourcePriceSnapshot> snapshot);

  /**
   * @brief Reset the internal maintenance state.
   */
  void reset();

  shared_ptr<PartitionManager> partition_manager_;  ///< Manages partition state.
  shared_ptr<MaintenancePolicyParams> params_;        ///< Maintenance parameters.
  shared_ptr<MaintenanceCostEstimator> cost_estimator_; ///< Cost estimator for maintenance actions.
  shared_ptr<HitCountTracker> hit_count_tracker_;       ///< Hit count tracker for partition hit rates.
  std::unordered_map<int64_t, int64_t> snapshot_sizes_; /// Map to keep track of index sizes

  /**
   * @brief Perform local refinement on a set of partition IDs.
   *
   * @param partition_ids Tensor of partition IDs.
   */
  struct RefinementWorkInfo {
    int64_t partition_count = 0;
    int64_t records_per_iteration = 0;
    int64_t records_read = 0;
    int64_t records_written = 0;
    int64_t iterations = 0;
  };

  RefinementWorkInfo local_refinement(const Tensor& partition_ids);
private:
  void update_cxl_adaptive_state(int64_t current_ntotal,
                                 int64_t current_partition_count,
                                 float current_scan_fraction,
                                 float current_aps_fanout,
                                 float current_avg_scanned_list_size);
  float cxl_adaptive_split_score_weight() const;
  float cxl_adaptive_maintenance_penalty_weight() const;
  float cxl_adaptive_delete_penalty_weight() const;
  float cxl_adaptive_fanout_penalty_weight() const;
  bool cxl_use_roi_budget() const;
  bool cxl_use_structural_budget() const;
  bool cxl_use_aps_feedback() const;
  bool cxl_use_streaming_rent_buy() const;
  bool cxl_use_resource_rent_buy() const;
  bool cxl_use_streaming_staged() const;
  bool cxl_use_search_first() const;
  bool cxl_use_observed_action_cost() const;
  struct StreamingSplitRent {
    float local_ns = 0.0f;
    float structural_ns = 0.0f;
  };
  StreamingSplitRent cxl_streaming_split_rent_ns(int partition_size,
                                                  float hit_rate,
                                                  float native_split_delta,
                                                  int64_t recorded_queries);
  float cxl_streaming_split_buy_ns(int partition_size) const;
  float cxl_streaming_reassign_buy_ns(int partition_size) const;
  int cxl_resource_home_id(int64_t partition_id) const;
  float cxl_resource_scan_cost_ns(int64_t partition_id,
                                  int64_t records) const;
  float cxl_resource_balanced_birth_scan_cost_ns(int64_t records) const;
  float cxl_resource_split_variable_buy_ns(int64_t partition_id,
                                           int partition_size) const;
  float cxl_resource_split_cohort_shared_buy_ns(
      int64_t parent_records) const;
  float cxl_resource_reassign_buy_ns(int64_t partition_id,
                                     int partition_size) const;
  float cxl_resource_reassign_rent_ns(
      int64_t partition_id,
      int partition_size,
      int source_hit_count,
      const vector<int64_t>& reassign_ids,
      const vector<int64_t>& reassign_counts,
      const vector<int64_t>& reassign_sizes,
      const vector<float>& hit_rates,
      int64_t recorded_queries,
      float* cost_before_ns,
      float* cost_after_ns) const;
  void update_cxl_child_probe_factor(
      const vector<vector<int64_t>>& per_query_hits,
      const std::unordered_set<int64_t>& active_partition_ids);
  int cxl_adaptive_split_budget(int total_partitions) const;
  int cxl_time_budget_split_limit() const;
  void update_cxl_observed_action_costs(int64_t split_records,
                                        int64_t split_time_us,
                                        int64_t split_count,
                                        int64_t refinement_time_us,
                                        int64_t refinement_records_read,
                                        int64_t refinement_records_written,
                                        int64_t reassign_records,
                                        int64_t reassign_time_us);

  int64_t cxl_prev_ntotal_ = -1;
  int64_t cxl_prev_partition_count_ = -1;
  int64_t cxl_prev_maintenance_records_ = 0;
  int64_t cxl_pending_add_count_ = 0;
  int64_t cxl_pending_delete_count_ = 0;
  float cxl_last_growth_sample_ = 0.0f;
  float cxl_last_delete_sample_ = 0.0f;
  float cxl_last_churn_sample_ = 0.0f;
  float cxl_growth_score_ = 0.0f;
  float cxl_delete_score_ = 0.0f;
  float cxl_churn_score_ = 0.0f;
  float cxl_scan_pressure_ = 0.0f;
  float cxl_maintenance_pressure_ = 0.0f;
  float cxl_fanout_pressure_ = 0.0f;
  float cxl_scanned_list_size_pressure_ = 0.0f;
  float cxl_last_aps_fanout_ = 0.0f;
  float cxl_last_avg_scanned_list_size_ = 0.0f;
  float cxl_prev_avg_scanned_records_ = -1.0f;
  float cxl_scan_growth_score_ = 0.0f;
  float cxl_last_scan_growth_sample_ = 0.0f;
  float cxl_last_fanout_growth_sample_ = 0.0f;
  float cxl_prev_aps_fanout_ = -1.0f;
  float cxl_prev_avg_scanned_list_size_ = -1.0f;
  float cxl_observed_split_ns_per_record_ = 0.0f;
  float cxl_observed_refine_ns_per_split_ = 0.0f;
  float cxl_observed_refine_ns_per_parent_record_ = 0.0f;
  float cxl_observed_refine_read_records_per_parent_ = 0.0f;
  float cxl_observed_refine_write_records_per_parent_ = 0.0f;
  float cxl_observed_reassign_ns_per_record_ = 0.0f;
  float cxl_estimated_query_window_ns_ = 0.0f;
  float cxl_maintenance_time_budget_ns_ = 0.0f;
  float cxl_payback_windows_ = 1.0f;
  float cxl_maintenance_budget_credit_ns_ = 0.0f;
  float cxl_streaming_structural_rent_ns_ = 0.0f;
  float cxl_streaming_structural_credit_ns_ = 0.0f;
  std::unordered_map<int64_t, float> cxl_split_credit_ns_;
  std::unordered_map<int64_t, float> cxl_reassign_credit_ns_;
  shared_ptr<CxlResourcePriceSnapshot> cxl_resource_price_snapshot_;
  vector<std::pair<int64_t, int64_t>> cxl_split_sibling_pairs_;
  float cxl_resource_child_probe_factor_ = -1.0f;
  float cxl_staged_calibration_r_ = 1.0f;
  bool cxl_has_observed_split_cost_ = false;
  int cxl_target_partition_size_ = 1;
  static constexpr bool debug_ = false;   
};

#endif  // MAINTENANCE_POLICY_REFACTORED_H
