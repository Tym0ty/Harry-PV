# Scenario Reduction Method Comparison Audit

## Layout Fixes
- Cleaned the original single-method 2x3 figure by shortening panel titles.
- Moved raw-scenario legend to a shared figure-level legend above the panels.
- Replaced repeated method names in panel titles with row labels.
- Increased vertical spacing to avoid top-row tick labels colliding with lower-row titles.
- Removed `line width proportional to pi` from panel titles; medoid probabilities remain in legend labels.

## Formal Reduction Methods
- METHOD_A_NAME: R1 ordinary K-medoids
- METHOD_A_FORMAL: `R1_ordinary_kmedoids_K5_netload_distance`
- METHOD_B_NAME: R3 low-PV-safe / contract-risk-preserving
- METHOD_B_FORMAL: `R3_lowpv_safe_contract_risk_preserving_K5`

## Sources
- RAW_SCENARIO_SOURCE: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/raw/G1_temporal_gaussian/id_h24_pv_raw_scenarios_M500.parquet`
- METHOD_A_SOURCE: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R1_kmedoids/id_h24_K5_scenarios.parquet`
- METHOD_B_SOURCE: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R3_lowpv_safe/id_h24_K5_scenarios.parquet`
- QUANTILE_SOURCE: `new_pipeline/data/output/experiments/id_h24_ghi_prob_q19_independent_agaci_v4/q19_agaci_calibrated_noncrossing_forecasts.parquet`
- REALIZED_PV_SOURCE: `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_calibrated_quantiles.parquet`

No scenario generation, scenario reduction, k-medoids, MPC, CCOR-MPC, MILP, or replay was rerun.

## Panel QA
| date_label      | target_day   | issue_time          |   raw_scenarios |   raw_horizon_hours |   r1_representatives |   r3_representatives |   q19_rows |   q19_quantile_crossings |
|:----------------|:-------------|:--------------------|----------------:|--------------------:|---------------------:|---------------------:|-----------:|-------------------------:|
| Sunny Autumn    | 2025-09-30   | 2025-09-29 23:00:00 |             500 |                  24 |                    5 |                    5 |         11 |                        0 |
| Variable Spring | 2025-04-08   | 2025-04-07 23:00:00 |             500 |                  24 |                    5 |                    5 |         12 |                        0 |
| Overcast Winter | 2025-01-11   | 2025-01-10 23:00:00 |             500 |                  24 |                    5 |                    5 |         11 |                        0 |

## Captions

**Part A caption draft:** Figure X. Raw daily PV scenario distributions and retained representative trajectories for three characteristic sky conditions. The upper panels show the 500 generated scenarios, empirical prediction bands, the scenario median, and the realized PV trajectory. The lower panels show the retained K=5 representative trajectories and their associated probabilities.

**Part B caption draft:** Figure Y. Comparison of two scenario-reduction methods for three representative days. The first row shows the raw N=500 daily PV scenario distributions, while the second and third rows show the K=5 retained representative trajectories produced by R1 ordinary K-medoids and R3 low-PV-safe / contract-risk-preserving, respectively. The comparison highlights how the risk-preserving reduction retains low-PV or stress-relevant trajectories relative to the ordinary K-medoids baseline.

## Data Gaps
- None found for the requested dates and methods.
