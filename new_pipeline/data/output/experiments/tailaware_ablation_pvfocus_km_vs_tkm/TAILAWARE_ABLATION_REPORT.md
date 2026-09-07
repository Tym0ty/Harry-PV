# PV-Focused Tail-Aware Reduction Ablation

## Final Status

`TAILAWARE_ABLATION_SUPPORTS_TKM_MAIN`

## Scope

Ran exactly two ordinary PV-KM K=5 probabilistic cases under base load: standard MPC-PROB and CCOR-v2 PROB m150. No DA, DET, K10, structured load bias, LitErr, LOAD-QN, old S1-S6, or old invalid PV truth were used.

## Annual Cost

| case                                   | scenario_reduction   |   full_total_m_ntd |   capex_m_ntd |   basic_m_ntd |   TOU_m_ntd |   OC_DCT_m_ntd |   Deg_m_ntd |   TREC_RE20_m_ntd |   monthly_peak_max_kw |   oc_hours |   risk0_hours |   shield_slack_positive_hours |   infeasible_steps |   fallback_steps | solver_status_summary   |   solve_time_total_sec |
|:---------------------------------------|:---------------------|-------------------:|--------------:|--------------:|------------:|---------------:|------------:|------------------:|----------------------:|-----------:|--------------:|------------------------------:|-------------------:|-----------------:|:------------------------|-----------------------:|
| MPC_PROB_PVFOCUS_KM_K5                 | PV_KM_K5             |            105.202 |       7.34889 |       7.74701 |     75.7917 |        5.79552 |     2.07906 |            6.4395 |               4958.2  |       1024 |          1551 |                             0 |                  0 |                0 | {"2": 8760}             |                349.783 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 | PV_KM_K5             |            104.204 |       7.34889 |       7.74701 |     76.12   |        4.56769 |     1.98109 |            6.4389 |               4810.12 |        812 |           206 |                             5 |                  0 |                0 | {"2": 8760}             |                359.244 |

## KM vs Tail-Aware TKM Comparison

| comparison                         | ordinary_km_case                       | tailaware_tkm_case                             |   delta_full_total_m_ntd_tkm_minus_km |   delta_TOU_m_ntd_tkm_minus_km |   delta_OC_DCT_m_ntd_tkm_minus_km |   delta_Deg_m_ntd_tkm_minus_km |   delta_monthly_peak_kw_tkm_minus_km |   delta_oc_hours_tkm_minus_km | tailaware_improves_total   | tailaware_reduces_OC_DCT   | tailaware_increases_TOU   |
|:-----------------------------------|:---------------------------------------|:-----------------------------------------------|--------------------------------------:|-------------------------------:|----------------------------------:|-------------------------------:|-------------------------------------:|------------------------------:|:---------------------------|:---------------------------|:--------------------------|
| MPC tail-aware TKM vs ordinary KM  | MPC_PROB_PVFOCUS_KM_K5                 | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5                 |                             -0.759444 |                      0.0848908 |                         -0.821794 |                    -0.03124    |                             -148.079 |                           -17 | True                       | True                       | True                      |
| CCOR tail-aware TKM vs ordinary KM | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 |                             -0.700605 |                      0.0968294 |                         -0.794863 |                    -0.00977141 |                             -114.701 |                            38 | True                       | True                       | True                      |

## Interpretation

| comparison                         | tailaware_improves_total   | tailaware_reduces_OC_DCT   | tailaware_increases_TOU   | final_status                         |
|:-----------------------------------|:---------------------------|:---------------------------|:--------------------------|:-------------------------------------|
| MPC tail-aware TKM vs ordinary KM  | True                       | True                       | True                      | TAILAWARE_ABLATION_SUPPORTS_TKM_MAIN |
| CCOR tail-aware TKM vs ordinary KM | True                       | True                       | True                      | TAILAWARE_ABLATION_SUPPORTS_TKM_MAIN |

## Scenario Audit

| case           | scenario_file                                                                                                           | bridge_type   | tail_aware   | low_pv_safe   |   K_min |   K_max | weights_type        |   weight_sum_min |   weight_sum_max |   effective_scenarios_min |   effective_scenarios_mean |   effective_scenarios_max | load_profile_name   | has_realized_or_truth_columns   | truth_like_columns   | LOAD_QN_used   | random_load_error_used   | old_S1S6_used   | structured_load_bias_used   |
|:---------------|:------------------------------------------------------------------------------------------------------------------------|:--------------|:-------------|:--------------|--------:|--------:|:--------------------|-----------------:|-----------------:|--------------------------:|---------------------------:|--------------------------:|:--------------------|:--------------------------------|:---------------------|:---------------|:-------------------------|:----------------|:----------------------------|
| KM_K5_ABLATION | new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_km_K5_scenarios.parquet | PV_KM         | False        | False         |       5 |       5 | cluster_probability |                1 |                1 |                         1 |                    3.24812 |                   4.98644 | base                | False                           |                      | False          | False                    | False           | False                       |

## Replay Truth Audit

| component                            | truth_path                                                                                                      | pv_truth_source    | uses_old_invalid_ntust_solar_kwh   | uses_LOAD_QN   | uses_random_load_error   | uses_old_S1S6   |   row_count |   pv_max_kw |   pv_mean_kw |
|:-------------------------------------|:----------------------------------------------------------------------------------------------------------------|:-------------------|:-----------------------------------|:---------------|:-------------------------|:----------------|------------:|------------:|-------------:|
| tail-aware ablation replay/KPI truth | new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet | CWA_GHI_PV_REBUILT | False                              | False          | False                    | False           |        8760 |     2298.88 |      328.374 |

## Monthly Cost

| case                                   |   month_id |   TOU_m_ntd |   OC_DCT_m_ntd |   Deg_m_ntd |   monthly_peak_kw |   oc_hours |
|:---------------------------------------|-----------:|------------:|---------------:|------------:|------------------:|-----------:|
| MPC_PROB_PVFOCUS_KM_K5                 |          1 |     3.24789 |     0.0816891  |    0.21709  |           3551.17 |         11 |
| MPC_PROB_PVFOCUS_KM_K5                 |          2 |     3.49185 |     0.173694   |    0.190216 |           3763.57 |          9 |
| MPC_PROB_PVFOCUS_KM_K5                 |          3 |     4.72048 |     0.259621   |    0.191187 |           3935.18 |         45 |
| MPC_PROB_PVFOCUS_KM_K5                 |          4 |     5.06219 |     0.287835   |    0.193318 |           3991.53 |         57 |
| MPC_PROB_PVFOCUS_KM_K5                 |          5 |     7.34731 |     0.682577   |    0.157351 |           4434.22 |        115 |
| MPC_PROB_PVFOCUS_KM_K5                 |          6 |     8.39786 |     0.641806   |    0.132696 |           4373.44 |        121 |
| MPC_PROB_PVFOCUS_KM_K5                 |          7 |     8.40732 |     0.471243   |    0.145501 |           4119.17 |         87 |
| MPC_PROB_PVFOCUS_KM_K5                 |          8 |     7.53018 |     0.542211   |    0.138472 |           4224.97 |         87 |
| MPC_PROB_PVFOCUS_KM_K5                 |          9 |     9.67842 |     0.886642   |    0.139416 |           4738.43 |        195 |
| MPC_PROB_PVFOCUS_KM_K5                 |         10 |     7.8795  |     1.03406    |    0.181454 |           4958.2  |        161 |
| MPC_PROB_PVFOCUS_KM_K5                 |         11 |     5.38849 |     0.402396   |    0.179267 |           4220.33 |         85 |
| MPC_PROB_PVFOCUS_KM_K5                 |         12 |     4.64023 |     0.331739   |    0.213089 |           4079.21 |         51 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          1 |     3.26154 |     0.00330603 |    0.211849 |           3316.35 |          1 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          2 |     3.50385 |     0.0741206  |    0.185622 |           3528.5  |          5 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          3 |     4.76504 |     0.0723916  |    0.174405 |           3523.32 |         17 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          4 |     5.1244  |     0.0459544  |    0.173859 |           3444.12 |          9 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          5 |     7.37641 |     0.684717   |    0.148698 |           4437.41 |        114 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          6 |     8.4045  |     0.697341   |    0.133744 |           4456.23 |        120 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          7 |     8.42117 |     0.471243   |    0.144781 |           4119.17 |         83 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          8 |     7.53247 |     0.584977   |    0.138325 |           4288.72 |         89 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |          9 |     9.68791 |     0.825682   |    0.140621 |           4647.56 |        172 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |         10 |     7.89316 |     0.934731   |    0.178841 |           4810.12 |        155 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |         11 |     5.44234 |     0.148352   |    0.15959  |           3712.95 |         41 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 |         12 |     4.70725 |     0.0248789  |    0.190751 |           3380.98 |          6 |
