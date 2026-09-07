# Scenario and Replay Usage Audit

## PV Truth Source

`PV truth source = CWA_GHI_PV_REBUILT`; replay/KPI uses `new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet`.

## Scenario Source

| case                           | scenario_path                                                                                                                       | bridge_type       |   k_min |   k_max |   weight_sum_min |   weight_sum_max |   effective_scenarios_min |   effective_scenarios_mean |   effective_scenarios_max |   min_weight |   max_weight | has_realized_or_truth_columns   | truth_like_columns   |   max_netload_kw |   min_pv_kw |   max_pv_kw |
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:------------------|--------:|--------:|-----------------:|-----------------:|--------------------------:|---------------------------:|--------------------------:|-------------:|-------------:|:--------------------------------|:---------------------|-----------------:|------------:|------------:|
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  | new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet     | PV_TKM_LOWPV_SAFE |       5 |       5 |                1 |                1 |                   1.35034 |                    3.16535 |                   4.76863 |        0.002 |        0.854 | False                           |                      |          4910.17 |           0 |     2258.86 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 | new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet | PV_TKM_LOWPV_SAFE |       5 |       5 |                1 |                1 |                   1       |                    2.98929 |                   4.92184 |        0     |        1     | False                           |                      |          4798.11 |           0 |     2546.62 |

## Replay Truth

| component                              | truth_path                                                                                                      | pv_truth_source    | uses_old_invalid_ntust_solar_kwh   |   row_count |   pv_max_kw |   pv_mean_kw |   load_max_kw |   load_mean_kw | source_marker_values   |
|:---------------------------------------|:----------------------------------------------------------------------------------------------------------------|:-------------------|:-----------------------------------|------------:|------------:|-------------:|--------------:|---------------:|:-----------------------|
| PV-focused scheduling replay/KPI truth | new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet | CWA_GHI_PV_REBUILT | False                              |        8760 |     2298.88 |      328.374 |          5179 |         2427.9 | CWA_GHI_REALIZED_TO_PV |

## Optimization Input Safety

The probabilistic scenario files do not include realized/actual/truth columns. LOAD-QN and random load error are not used in this batch.
