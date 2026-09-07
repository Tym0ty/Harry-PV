# DA/MPC PV-Focused LowPV-Safe Scheduling Report

## Final Decision

`BASIC_FOUR_CASES_READY_FOR_CCOR_NEXT`

## Case Definitions

- `DA_DET_PVFOCUS_BASE`: deterministic given load + PV q50, DA MILP.
- `DA_PROB_PVFOCUS_LOWPV_SAFE_K5`: deterministic given load + DA PV-focused K5 low-PV-safe scenarios, stochastic DA MILP.
- `MPC_DET_PVFOCUS_BASE`: deterministic given load + issue-time PV q50/DA tail, hourly rolling MPC.
- `MPC_PROB_PVFOCUS_LOWPV_SAFE_K5`: deterministic given load + ID H24 PV-focused K5 low-PV-safe scenarios, hourly stochastic MPC.

## PV Truth and Scenario Inputs

Replay/KPI uses `new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet` (`CWA_GHI_PV_REBUILT`). DA-PROB uses `new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\da_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`. MPC-PROB uses `new_pipeline\data\output\experiments\pv_focused_structured_load_bias_bridge\id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`. Old S1-S6, LOAD-QN, random load error, and old invalid NTUST `Solar_kWh` PV truth are not used.

## Annual Realized-Basis Cost

| case                           |   full_total_m_ntd |   capex_m_ntd |   basic_m_ntd |   TOU_m_ntd |   OC_DCT_m_ntd |   Deg_m_ntd |   TREC_RE20_m_ntd |   monthly_peak_max_kw |   oc_hours |   total_charge_kwh |   total_discharge_kwh |   soc_min_kwh |   soc_max_kwh |   infeasible_steps |   fallback_steps | solver_status_summary   |   solve_time_total_sec |
|:-------------------------------|-------------------:|--------------:|--------------:|------------:|---------------:|------------:|------------------:|----------------------:|-----------:|-------------------:|----------------------:|--------------:|--------------:|-------------------:|-----------------:|:------------------------|-----------------------:|
| DA_DET_PVFOCUS_BASE            |            106.438 |       7.34889 |       7.74701 |     75.5739 |        7.19779 |     2.13876 |            6.432  |               5253.01 |        992 |        2.3171e+06  |           2.09154e+06 |       752.583 |       6773.25 |                  0 |                0 | {"2": 8760}             |                5.56957 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |            103.622 |       7.34889 |       7.74701 |     76.5207 |        3.69317 |     1.89186 |            6.4205 |               4790.03 |        850 |        2.09455e+06 |           1.89068e+06 |       752.583 |       6773.25 |                  0 |                0 | {"2": 8760}             |               14.827   |
| MPC_DET_PVFOCUS_BASE           |            106.486 |       7.34889 |       7.74701 |     75.557  |        7.23912 |     2.14717 |            6.4464 |               5133.7  |       1021 |        2.31982e+06 |           2.0965e+06  |       752.583 |       6773.25 |                  0 |                0 | {"2": 8760}             |              130.839   |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |            104.442 |       7.34889 |       7.74701 |     75.8766 |        4.97372 |     2.04782 |            6.4482 |               4810.12 |       1007 |        2.21417e+06 |           2.00115e+06 |       752.583 |       6773.25 |                  0 |                0 | {"2": 8760}             |              355.097   |

## Monthly Cost Summary

| case                           |   month_id |   TOU_m_ntd |   OC_DCT_m_ntd |   monthly_peak_kw |   oc_hours |
|:-------------------------------|-----------:|------------:|---------------:|------------------:|-----------:|
| DA_DET_PVFOCUS_BASE            |          1 |     3.23256 |     0.100958   |           3608.9  |         12 |
| DA_DET_PVFOCUS_BASE            |          2 |     3.4819  |     0.355843   |           4127.36 |         13 |
| DA_DET_PVFOCUS_BASE            |          3 |     4.6904  |     0.338365   |           4092.45 |         42 |
| DA_DET_PVFOCUS_BASE            |          4 |     5.00133 |     0.741512   |           4897.61 |         74 |
| DA_DET_PVFOCUS_BASE            |          5 |     7.29294 |     1.12484    |           5093.52 |        123 |
| DA_DET_PVFOCUS_BASE            |          6 |     8.39967 |     0.458516   |           4100.2  |        111 |
| DA_DET_PVFOCUS_BASE            |          7 |     8.4059  |     0.504193   |           4168.29 |         87 |
| DA_DET_PVFOCUS_BASE            |          8 |     7.51628 |     0.33707    |           3919.15 |         51 |
| DA_DET_PVFOCUS_BASE            |          9 |     9.69094 |     0.921254   |           4790.03 |        202 |
| DA_DET_PVFOCUS_BASE            |         10 |     7.88384 |     1.23182    |           5253.01 |        157 |
| DA_DET_PVFOCUS_BASE            |         11 |     5.33745 |     0.718265   |           4851.19 |         78 |
| DA_DET_PVFOCUS_BASE            |         12 |     4.64071 |     0.365157   |           4145.96 |         42 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          1 |     3.25221 |     0.0114847  |           3340.86 |          2 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          2 |     3.50527 |     0.0870854  |           3567.34 |          6 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          3 |     4.72574 |     0.278391   |           3972.67 |         36 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          4 |     5.145   |     0.052776   |           3464.56 |         56 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          5 |     7.47919 |     0.16136    |           3657.21 |         99 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          6 |     8.46036 |     0.286555   |           3843.85 |        107 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          7 |     8.46879 |     0.508941   |           4175.37 |         82 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          8 |     7.56913 |     0.425322   |           4050.72 |         56 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |          9 |     9.75139 |     0.921254   |           4790.03 |        191 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |         10 |     7.93932 |     0.921108   |           4789.81 |        160 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |         11 |     5.49236 |     0.0190024  |           3363.38 |         46 |
| DA_PROB_PVFOCUS_LOWPV_SAFE_K5  |         12 |     4.73198 |     0.0198959  |           3366.05 |          9 |
| MPC_DET_PVFOCUS_BASE           |          1 |     3.24927 |     0.0870928  |           3567.36 |         11 |
| MPC_DET_PVFOCUS_BASE           |          2 |     3.48492 |     0.260949   |           3937.83 |         14 |
| MPC_DET_PVFOCUS_BASE           |          3 |     4.68862 |     0.298268   |           4012.37 |         53 |
| MPC_DET_PVFOCUS_BASE           |          4 |     5.011   |     0.615427   |           4645.8  |         86 |
| MPC_DET_PVFOCUS_BASE           |          5 |     7.30494 |     0.973589   |           4868.05 |        122 |
| MPC_DET_PVFOCUS_BASE           |          6 |     8.38946 |     0.717301   |           4485.99 |        107 |
| MPC_DET_PVFOCUS_BASE           |          7 |     8.40527 |     0.481581   |           4134.59 |         91 |
| MPC_DET_PVFOCUS_BASE           |          8 |     7.52397 |     0.469493   |           4116.57 |         56 |
| MPC_DET_PVFOCUS_BASE           |          9 |     9.67167 |     1.10021    |           5056.82 |        187 |
| MPC_DET_PVFOCUS_BASE           |         10 |     7.86355 |     1.15179    |           5133.7  |        153 |
| MPC_DET_PVFOCUS_BASE           |         11 |     5.32932 |     0.718265   |           4851.19 |         89 |
| MPC_DET_PVFOCUS_BASE           |         12 |     4.63498 |     0.365157   |           4145.96 |         52 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          1 |     3.25466 |     0.00476936 |           3320.74 |          2 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          2 |     3.50324 |     0.0848477  |           3560.64 |          7 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          3 |     4.72858 |     0.209022   |           3834.12 |         46 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          4 |     5.069   |     0.241658   |           3899.31 |         61 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          5 |     7.35896 |     0.510759   |           4178.08 |        119 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          6 |     8.39437 |     0.500397   |           4162.64 |        123 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          7 |     8.41004 |     0.471243   |           4119.17 |         88 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          8 |     7.53367 |     0.584977   |           4288.72 |         89 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |          9 |     9.67101 |     0.921254   |           4790.03 |        190 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |         10 |     7.88737 |     0.934731   |           4810.12 |        167 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |         11 |     5.37663 |     0.426864   |           4269.2  |         78 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 |         12 |     4.68906 |     0.0832004  |           3555.7  |         37 |

## Pairwise Comparisons

- PV probabilistic value under DA: DA_PROB_PVFOCUS_LOWPV_SAFE_K5 minus DA_DET_PVFOCUS_BASE = -2.8162 M NTD total, TOU +0.9468, OC/DCT -3.5046, Deg -0.2469, monthly peak -462.98 kW.
- PV probabilistic value under MPC: MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 minus MPC_DET_PVFOCUS_BASE = -2.0433 M NTD total, TOU +0.3196, OC/DCT -2.2654, Deg -0.0994, monthly peak -323.58 kW.
- Rolling MPC value under point forecast: MPC_DET_PVFOCUS_BASE minus DA_DET_PVFOCUS_BASE = +0.0472 M NTD total, TOU -0.0169, OC/DCT +0.0413, Deg +0.0084, monthly peak -119.31 kW.
- Rolling MPC value under PV probabilistic forecast: MPC_PROB_PVFOCUS_LOWPV_SAFE_K5 minus DA_PROB_PVFOCUS_LOWPV_SAFE_K5 = +0.8200 M NTD total, TOU -0.6442, OC/DCT +1.2805, Deg +0.1560, monthly peak +20.09 kW.

## Interpretation

Use the pairwise deltas to judge whether PV probabilistic scenarios reduce OC/DCT at the expense of TOU, whether rolling MPC improves over DA, and whether standard MPC still motivates CCOR-MPC.

## Diagnostics

Solver status, infeasible step counts, scenario weights, and replay truth source are reported in the required CSV audit files.

## Next Batch Recommendation

If the final decision is `BASIC_FOUR_CASES_READY_FOR_CCOR_NEXT`, the next batch should run `CCOR_MPC_DET_PVFOCUS_BASE`, `CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5`, optional structured-high robustness, and optional ID K10 sensitivity. These were not run in this batch.
