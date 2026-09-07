# PV-Focused Low-PV Reduction Fix Report

## Status

Generated `PV_TKM_LOWPV_SAFE` reduced scenario packages without overwriting existing PV-KM/PV-TKM/PV-TKM-safe outputs.

## Medoid Roles

K=5 uses explicit roles: high-net-load tail, low-PV/PV-shortfall tail, DCT-risk tail, and two typical medoids. K=10 keeps the same tail roles and adds typical/diversity medoids.

## Weights

Scenario weights are cluster probability masses from nearest-medoid assignment in a peak- and low-PV-aware feature space.

## Generated Packages

| mode   | profile         | bridge_type       |   k |   issues | path                                                                                                                                                                                               |   elapsed_sec |
|:-------|:----------------|:------------------|----:|---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| ID_H24 | structured_high | PV_TKM_LOWPV_SAFE |   5 |     8776 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_structured_high_tkm_lowpv_safe_K5_scenarios.parquet |       267.036 |

## Weight Sanity

| mode   | bridge_type       |   k |   min |   max |   mean |
|:-------|:------------------|----:|------:|------:|-------:|
| ID_H24 | PV_TKM_LOWPV_SAFE |   5 |     1 |     1 |      1 |

## No Scheduling

This script only creates reduced scenario packages for validation. It does not run DA/MPC/CCOR scheduling.
