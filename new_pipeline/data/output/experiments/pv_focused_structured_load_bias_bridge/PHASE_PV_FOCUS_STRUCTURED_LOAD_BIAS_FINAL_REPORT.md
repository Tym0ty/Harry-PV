# PV-Focused Structured Load-Bias Final Report

## Executive Summary

Status: `NEEDS_FIX_BEFORE_SCHEDULING`.

The PV-focused bridge was implemented through Phase 5:

1. deterministic given load profiles were created;
2. calibrated PV quantiles were sampled into M=500 raw PV scenarios;
3. net-load packages were built as `load_profile - pv_s`;
4. PV-KM, PV-TKM, and PV-TKM-safe K=5 reductions were generated;
5. validation was completed against old S1-S6 and realized ex-post truth.

Scheduling was not run because the base ID_H24 PV-TKM-safe bridge did not preserve enough OC-risk / high-tail coverage. This follows the requested safety rule: if validation fails, stop before scheduling.

## Methodological Rationale

The main stochastic source is PV uncertainty. The load side is treated as a deterministic given demand profile in the mainline. Structured load-bias profiles are outer robustness cases only, not probabilistic load forecasts and not assigned scenario probabilities.

Previous LitErr-TKM-safe joint PV + literature load-error results should remain appendix/diagnostic. They are not overwritten and are not used as the new mainline.

## Load-Bias Matrix And Sanity

| profile         |   annual_kwh |   max_load_kw |   min_load_kw |   negative_load_count |   oc_risk_hours_gt_CC |   near_oc_hours_gt_0p95CC |
|:----------------|-------------:|--------------:|--------------:|----------------------:|----------------------:|--------------------------:|
| base            |  2.12684e+07 |       5179    |             0 |                     0 |                  1503 |                      1741 |
| structured_low  |  2.03357e+07 |       4920.05 |             0 |                     0 |                  1246 |                      1495 |
| structured_high |  2.22012e+07 |       5437.95 |             0 |                     0 |                  1733 |                      2057 |
| uniform_low_5   |  2.0205e+07  |       4920.05 |             0 |                     0 |                  1255 |                      1503 |
| uniform_high_5  |  2.23319e+07 |       5437.95 |             0 |                     0 |                  1727 |                      2039 |

## PV Scenario Generation Summary

| mode   |   M |   issue_count |    rows | path                                                                                                                                                             |   elapsed_sec |
|:-------|----:|--------------:|--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| DA     | 500 |           365 | 4380000 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pv_raw_scenarios_M500.parquet |       116.116 |

## Scenario Reduction Summary

| mode   | profile         | bridge_type   |   issues | path                                                                                                                                                                                         |   elapsed_sec |
|:-------|:----------------|:--------------|---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| DA     | base            | PV_KM         |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_base_km_K5_scenarios.parquet                      |       8.0587  |
| DA     | base            | PV_TKM        |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_base_tkm_K5_scenarios.parquet                     |       8.02219 |
| DA     | base            | PV_TKM_SAFE   |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_base_tkm_safe_K5_scenarios.parquet                |       8.15537 |
| ID_H24 | base            | PV_KM         |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_km_K5_scenarios.parquet                  |     194.136   |
| ID_H24 | base            | PV_TKM        |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_tkm_K5_scenarios.parquet                 |     194.143   |
| ID_H24 | base            | PV_TKM_SAFE   |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_tkm_safe_K5_scenarios.parquet            |     199.682   |
| DA     | structured_low  | PV_KM         |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_low_km_K5_scenarios.parquet            |       8.1256  |
| DA     | structured_low  | PV_TKM        |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_low_tkm_K5_scenarios.parquet           |       8.08042 |
| DA     | structured_low  | PV_TKM_SAFE   |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_low_tkm_safe_K5_scenarios.parquet      |       8.24938 |
| ID_H24 | structured_low  | PV_KM         |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_low_km_K5_scenarios.parquet        |     196.637   |
| ID_H24 | structured_low  | PV_TKM        |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_low_tkm_K5_scenarios.parquet       |     200.692   |
| ID_H24 | structured_low  | PV_TKM_SAFE   |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_low_tkm_safe_K5_scenarios.parquet  |     202.068   |
| DA     | structured_high | PV_KM         |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_high_km_K5_scenarios.parquet           |       8.25166 |
| DA     | structured_high | PV_TKM        |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_high_tkm_K5_scenarios.parquet          |       8.08772 |
| DA     | structured_high | PV_TKM_SAFE   |      365 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_high_tkm_safe_K5_scenarios.parquet     |       8.35291 |
| ID_H24 | structured_high | PV_KM         |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_high_km_K5_scenarios.parquet       |     199.101   |
| ID_H24 | structured_high | PV_TKM        |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_high_tkm_K5_scenarios.parquet      |     197.561   |
| ID_H24 | structured_high | PV_TKM_SAFE   |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_high_tkm_safe_K5_scenarios.parquet |     202.794   |

## Base Scenario Validation Results

| mode   | bridge_type                |   realized_netload_upper_coverage |   top5_netload_upper_coverage |   top1_netload_upper_coverage |   oc_risk_hour_upper_coverage |   near_oc_hour_upper_coverage |   scenario_max_netload_kw |
|:-------|:---------------------------|----------------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------------:|--------------------------:|
| DA     | PV_KM                      |                          0.895548 |                      0.817352 |                      0.795455 |                      0.838577 |                      0.85253  |                   4860.67 |
| DA     | PV_TKM                     |                          0.898288 |                      0.840183 |                      0.852273 |                      0.868673 |                      0.877287 |                   4910.17 |
| DA     | PV_TKM_SAFE                |                          0.893265 |                      0.815068 |                      0.795455 |                      0.835841 |                      0.853606 |                   4910.17 |
| DA     | old_S1S6_LOAD_QN_REFERENCE |                          0.942237 |                      0.76484  |                      0.897727 |                      0.768274 |                      0.792517 |                   5693    |
| ID_H24 | PV_KM                      |                          0.877562 |                      0.48906  |                      0.293087 |                      0.542807 |                      0.574364 |                   4793.28 |
| ID_H24 | PV_TKM                     |                          0.881713 |                      0.524639 |                      0.330019 |                      0.572275 |                      0.600565 |                   4798.11 |
| ID_H24 | PV_TKM_SAFE                |                          0.878833 |                      0.501617 |                      0.314867 |                      0.552269 |                      0.58244  |                   4798.11 |
| ID_H24 | old_S1S6_LOAD_QN_REFERENCE |                          0.940302 |                      0.685608 |                      0.807765 |                      0.689489 |                      0.721441 |                   5845.09 |

## Validation Interpretation

- DA base PV-TKM/PV-TKM-safe coverage is reasonable for OC-risk hours: DA PV-TKM-safe OC-risk upper coverage is about 0.836.
- ID_H24 base PV-TKM-safe coverage is not sufficient: OC-risk upper coverage is about 0.552, and top-1% net-load upper coverage is about 0.315.
- This indicates that PV-only uncertainty with the current ID H24 calibrated PV quantiles/reduction is not yet sufficient as an MPC probabilistic scheduling bridge for contract-capacity tail risk.
- No pathological 16 MW-style medoid appears in the PV-focused bridge; the issue is under-coverage, not extreme over-conservatism.

## Basic Four-Case Scheduling Results

NOT_RUN_VALIDATION_FAILED

Basic DA/MPC scheduling was not run because validation ended with `NEEDS_FIX_BEFORE_SCHEDULING`.

## Comparison Against LitErr-TKM-safe Diagnostic

LitErr-TKM-safe previously passed Phase 1b after tail-safe repair and was schedulable. However, it uses literature/proxy random load error, which the revised method decision demotes to appendix/diagnostic. The PV-focused bridge better matches the thesis focus but currently needs repair before it can replace LitErr/S1-S6 in scheduling.

## Recommendation For Next Step

Do not run CCOR-MPC yet.

Recommended fixes before scheduling:

1. Repair ID H24 PV tail coverage at peak-forming first-step / short-lead hours.
2. Add an explicit low-PV stress medoid rule for PV-TKM-safe, rather than only high net-load tail-band selection.
3. Validate raw PV scenario coverage before reduction to separate PV quantile under-coverage from reduction loss.
4. Consider K=10 sensitivity for ID H24 only if K=5 cannot preserve low-PV tail without degrading typical behavior.

Final recommendation: `NEEDS_FIX_BEFORE_SCHEDULING`.
