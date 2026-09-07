# Metric Definition Audit

## Formal code source
- File: `new_pipeline/scripts/experiments/pvfocus_scenario_bridge_ablation_utils.py`
- `RED_SPECS` defines R1 as `R1_ordinary_kmedoids_K5_netload_distance` and R3 as `R3_lowpv_safe_contract_risk_preserving_K5`.
- `reduce_one()` writes `reduction_metrics_by_issue.csv` and computes the four Panel B metrics.

## Low-PV class inclusion ratio
- Code function: `reduce_one()`.
- Input columns: raw `pv_s_kw`, retained `pv_s_kw`.
- Threshold: raw M=500 24h PV energy 5th percentile per issue_time.
- Numerator: number of issue_times where the retained K=5 set contains at least one representative with 24h PV energy <= raw 5th percentile.
- Denominator: number of evaluated issue_times.
- Aggregation: binary per issue_time, arithmetic mean across issue_times.
- Probability weights: not used in the metric. Raw probability mass is reported only for audit.
- Interpretation: event-class inclusion, not probability-mass retention.

## High-net-load class inclusion ratio
- Code function: `reduce_one()`.
- Input columns: raw `netload_s_kw`, retained `netload_s_kw`.
- Threshold: raw M=500 trajectory peak net-load 95th percentile per issue_time.
- Numerator: number of issue_times where the retained K=5 set contains at least one representative with peak net-load >= raw 95th percentile.
- Denominator: number of evaluated issue_times.
- Aggregation: binary per issue_time, arithmetic mean across issue_times.
- Probability weights: not used in the metric.
- Net load definition: `netload_s_kw = load_kw - pv_s_kw` from the formal raw G1 net-load scenario pool.

## Raw-pool peak preservation ratio
- Code function: `reduce_one()`.
- Input columns: raw `netload_s_kw`, retained `netload_s_kw`.
- Formula per issue_time: `max_peak_reduced / max_peak_raw`, where peak is maximum hourly net-load over H=24.
- Aggregation: arithmetic mean across issue_times.
- Interpretation: extremum preservation ratio, not probability-mass retention.

## Average representation error
- Code function: `reduce_one()`.
- Input columns: raw and retained `netload_s_kw` H=24 trajectories.
- Distance definition: `_features(mat)` standardizes each lead-hour by the raw pool mean and standard deviation; assignment uses nearest medoid in squared Euclidean distance over the standardized net-load trajectory.
- Per issue_time metric: mean Euclidean distance from each raw trajectory to its nearest retained representative in this standardized feature space.
- Unit: dimensionless standardized net-load trajectory distance.
- Aggregation: arithmetic mean across issue_times.
