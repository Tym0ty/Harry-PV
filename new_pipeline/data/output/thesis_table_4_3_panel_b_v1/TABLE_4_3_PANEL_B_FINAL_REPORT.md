# Final Status

FINAL_STATUS: SUCCESS

# Formal Output Sources

- R1 output root: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R1_kmedoids`
- R3 output root: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R3_lowpv_safe`
- Raw N=500 scenario pool: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/raw/G1_temporal_gaussian/id_h24_raw_netload_M500.parquet`
- R1 retained K=5 trajectories: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R1_kmedoids/id_h24_K5_scenarios.parquet`
- R3 retained K=5 trajectories: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R3_lowpv_safe/id_h24_K5_scenarios.parquet`
- R1 metrics: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R1_kmedoids/reduction_metrics_by_issue.csv`
- R3 metrics: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R3_lowpv_safe/reduction_metrics_by_issue.csv`
- Generation metrics: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/validation/scenario_generation_metrics.csv`
- Validation code: `new_pipeline/scripts/experiments/pvfocus_scenario_bridge_ablation_utils.py`

# Comparison Universe

- Issue times evaluated: 8776
- Raw scenarios per issue_time: 500
- Retained scenarios per method: K=5
- Same raw pool for R1 and R3: True
- R1 retained raw IDs subset of raw pool: True
- R3 retained raw IDs subset of raw pool: True

# Metric Definitions

See `metric_definition_audit.md`.

# Low-PV Inclusion Results

- R1_LOW_PV_CLASS_INCLUSION_RATIO: 0.053099361896
- R3_LOW_PV_CLASS_INCLUSION_RATIO: 1.000000000000

# High-Net-Load Inclusion Results

- R1_HIGH_NET_LOAD_CLASS_INCLUSION_RATIO: 0.225387420237
- R3_HIGH_NET_LOAD_CLASS_INCLUSION_RATIO: 1.000000000000

# Raw-Pool Peak Preservation Results

- R1_RAW_POOL_PEAK_PRESERVATION_RATIO: 0.974058789837
- R3_RAW_POOL_PEAK_PRESERVATION_RATIO: 1.000000000000

# Average Representation Error

- Definition: mean nearest-representative distance in standardized H24 net-load trajectory feature space.
- Unit: dimensionless standardized net-load distance.
- R1_AVERAGE_REPRESENTATION_ERROR: 2.838817381812
- R3_AVERAGE_REPRESENTATION_ERROR: 3.206321794797
- R3_RELATIVE_ERROR_CHANGE_VS_R1: 12.945687%
- R1 median / std / P25 / P75: 2.829047699007 / 0.319443972474 / 2.617577666635 / 3.131297694622
- R3 median / std / P25 / P75: 3.198793328944 / 0.389845604039 / 2.944836381051 / 3.507921978418

# R1 versus R3 Trade-Off

- R3 minus R1 paired mean difference: 0.367504412985
- R3 minus R1 paired median difference: 0.349186790417
- 95% bootstrap CI for paired mean difference: [0.364047027408, 0.370872786995]
- Issue times where R1 has lower representation error: 8755
- Issue times where R3 has lower representation error: 14
- Tied issue times: 7

# Final Table 4-3 Panel B

| Property                                                      |   Ordinary K-medoids (R1) |   Contract-risk-preserving reduction (R3) | Desired property   |
|:--------------------------------------------------------------|--------------------------:|------------------------------------------:|:-------------------|
| Low-PV class inclusion ratio                                  |                 0.0530994 |                                   1       | 1.00               |
| High-net-load class inclusion ratio                           |                 0.225387  |                                   1       | 1.00               |
| Raw-pool peak preservation ratio                              |                 0.974059  |                                   1       | 1.00               |
| Average representation error (standardized net-load distance) |                 2.83882   |                                   3.20632 | Lower is better    |

# Statements Supported by the Results

- R3 achieves event-class inclusion ratios of 1.0 for low-PV and high-net-load classes under the formal binary inclusion definitions.
- R3 achieves raw-pool peak preservation ratio of 1.0 under the formal extremum ratio definition.
- R1 has a lower average standardized representation error than R3 in the formal validation metric.

# Statements Not Supported by the Results

- The inclusion ratios do not mean probability-mass retention.
- R3 should not be described as having lower representation error than R1.
- The values are not based on the three figure days; they use the full formal issue-time universe.

# Output Index

- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/metric_definition_audit.md`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/comparison_universe_audit.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/low_pv_inclusion_by_issue_time.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/high_net_load_inclusion_by_issue_time.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/raw_pool_peak_preservation_by_issue_time.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/representation_error_by_issue_time.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/table_4_3_panel_b_scenario_reduction_validation.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/table_4_3_panel_b_scenario_reduction_validation_full_precision.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/table_4_3_generation_and_reduction_validation.csv`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/table_4_3_generation_and_reduction_validation.md`
