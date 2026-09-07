# PV Raw Scenario Generation Report

## Status

Generated DA and ID H24 raw PV scenarios from calibrated PV quantiles using inverse-CDF uniform sampling.

## Safety

Realized PV was not used to generate scenarios. Quantile artifacts may contain truth columns, but this script reads only forecast quantile columns.

## Summary

| mode   |   M |   issue_count |    rows | path                                                                                                                                                             |   elapsed_sec |
|:-------|----:|--------------:|--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|
| DA     | 500 |           365 | 4380000 | C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pv_raw_scenarios_M500.parquet |       116.116 |
