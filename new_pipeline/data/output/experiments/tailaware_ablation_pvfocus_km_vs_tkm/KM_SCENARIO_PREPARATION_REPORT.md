# KM Scenario Preparation Report

## Status

Ordinary PV-KM K=5 scenario package exists and was audited.

## Audit

| case           | scenario_file                                                                                                           | bridge_type   | tail_aware   | low_pv_safe   |   K_min |   K_max | weights_type        |   weight_sum_min |   weight_sum_max |   effective_scenarios_min |   effective_scenarios_mean |   effective_scenarios_max | load_profile_name   | has_realized_or_truth_columns   | truth_like_columns   | LOAD_QN_used   | random_load_error_used   | old_S1S6_used   | structured_load_bias_used   |
|:---------------|:------------------------------------------------------------------------------------------------------------------------|:--------------|:-------------|:--------------|--------:|--------:|:--------------------|-----------------:|-----------------:|--------------------------:|---------------------------:|--------------------------:|:--------------------|:--------------------------------|:---------------------|:---------------|:-------------------------|:----------------|:----------------------------|
| KM_K5_ABLATION | new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_km_K5_scenarios.parquet | PV_KM         | False        | False         |       5 |       5 | cluster_probability |                1 |                1 |                         1 |                    3.24812 |                   4.98644 | base                | False                           |                      | False          | False                    | False           | False                       |
