# Scenario Generation Figure With Q19 Quantile Bands Audit

- QUANTILE_SOURCE: `new_pipeline/data/output/experiments/id_h24_ghi_prob_q19_independent_agaci_v4/q19_agaci_calibrated_noncrossing_forecasts.parquet`
- Quantile source type: Q19 independent-model AgACI v4 calibrated GHI product, converted to PV using `clip(0.80 * GHI / 1000 * 2687, 0, 2687)`.
- RAW_SCENARIO_SOURCE: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/raw/G1_temporal_gaussian/id_h24_pv_raw_scenarios_M500.parquet`
- MEDOID_SOURCE: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R3_lowpv_safe/id_h24_K5_scenarios.parquet`
- P25/P75 source: direct `q25_cal` and `q75_cal`, not interpolation.
- FORECAST_MODEL_RERUN: FALSE
- AGACI_RERUN: FALSE
- SCENARIO_GENERATION_RERUN: FALSE
- KMEDOIDS_RERUN: FALSE
- OPTIMIZATION_RERUN: FALSE
- LOWER_PANELS_MODIFIED: FALSE, except only titles were expanded to repeat target date/context; K=5 curves, legend style, realized style, and line widths are unchanged.

## Panel QA
| panel           | target_day   | issue_time          |   raw_rows |   raw_scenarios |   medoid_rows |   medoids |   q19_rows |   q19_hour_min |   q19_hour_max |   q19_quantile_crossings |
|:----------------|:-------------|:--------------------|-----------:|----------------:|--------------:|----------:|-----------:|---------------:|---------------:|-------------------------:|
| Sunny Autumn    | 2025-09-30   | 2025-09-29 23:00:00 |      12000 |             500 |           120 |         5 |         11 |              7 |             17 |                        0 |
| Variable Spring | 2025-04-08   | 2025-04-07 23:00:00 |      12000 |             500 |           120 |         5 |         12 |              6 |             17 |                        0 |
| Overcast Winter | 2025-01-11   | 2025-01-10 23:00:00 |      12000 |             500 |           120 |         5 |         11 |              7 |             17 |                        0 |
