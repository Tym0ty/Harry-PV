# PV-Focused Net-Load Build Report

## Status

Built raw net-load scenario packages for each deterministic load profile separately.

## Method

`netload_s_kw = load_profile_kw - pv_s_kw`. PV is the stochastic source. Load profile is deterministic within each outer sensitivity case.

## No Mixing

Base, structured-low, and structured-high load profiles are not assigned probabilities and are not mixed into one probabilistic scenario set.

## Summary

| mode   | profile         |    rows | path                                                                                                                                                                                |   elapsed_sec |
|:-------|:----------------|--------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| DA     | base            | 4380000 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_base_raw_netload_M500.parquet            |       15.1749 |
| DA     | structured_low  | 4380000 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_low_raw_netload_M500.parquet  |       15.1986 |
| DA     | structured_high | 4380000 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_structured_high_raw_netload_M500.parquet |       15.2575 |
