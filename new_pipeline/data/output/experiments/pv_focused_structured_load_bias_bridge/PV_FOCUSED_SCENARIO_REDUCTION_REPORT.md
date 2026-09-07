# PV-Focused Scenario Reduction Report

## Status

Generated PV-KM, PV-TKM, and PV-TKM-safe reduced K=5 packages for each requested load profile.

## Weights

Scenario weights are empirical cluster masses, not equal weights.

## Tail-Safe Logic

PV-TKM-safe uses tail-band medoid selection to preserve high net-load / DCT-risk behavior without selecting an unexplained pathological extreme.

## Summary

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
