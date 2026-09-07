# Final Thesis Method and Results Report: PV-Focused Rolling MPC and CCOR-MPC

## 0. Executive Summary

This report consolidates the final converged thesis method line for the BH PV-BESS scheduling project. The final study focuses on PV uncertainty and PV-BESS scheduling under a deterministic given campus load profile. Load uncertainty is not modeled as a probabilistic load forecast because only one year of site load data is available; instead, the load assumption is defended through season x time-of-day structured load-bias robustness.

PV uncertainty is represented through calibrated probabilistic GHI/PV forecasts. The final optimization-facing net-load scenario bridge is:

```text
netload_s(t) = load_given(t) - PV_s(t)
```

Scenario reduction uses a low-PV-safe, tail-aware K-medoids reduction with K=5. The final main cases are only `MPC_DET`, `MPC_PROB`, `CCOR_DET`, and `CCOR_PROB`; DA is excluded from the final main case set and retained only as a removed/reference-only benchmark if needed.

The final results support three thesis conclusions. First, PV probabilistic scenarios improve both standard MPC and CCOR-v2. Second, CCOR-v2 improves standard MPC mainly by reducing OC/DCT exposure. Third, the conclusion remains stable under structured load-bias robustness. Tail-aware low-PV-safe scenario reduction also improves PROB scheduling results compared with ordinary non-tail-aware KM.

Final report status: `FINAL_REPORT_READY_WITH_MINOR_UNCERTAIN_ITEMS`.

Minor uncertain items:

- A final literature citation for the simple PR-based GHI-to-PV conversion is not stored in the repository artifact; mark as CITATION_NEEDED.
- Some detailed feature-level descriptions for the selected forecast models are summarized from reports, but exact feature lists are not fully reproduced in this final report.

## 1. Final Research Scope and Case Definition

The final scope is the NTUST/BH campus PV-BESS contract-capacity scheduling problem. The confirmed system parameters used in the final PV-focused scripts include PV capacity `PV_CAP = 2687 kWp`, performance ratio `PR = 0.80`, contract capacity `CC = 3232 kW`, and the test period `2024-11-01` to `2025-10-31`. The corrected replay truth package is `new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet`.

The final main case set is:

| Final Case   | Case ID                                        | Scheduling Method                                | Forecast Type    | Scenario Type                 | Load Setting                  | Execution                       |
|:-------------|:-----------------------------------------------|:-------------------------------------------------|:-----------------|:------------------------------|:------------------------------|:--------------------------------|
| MPC_DET      | MPC_DET_PVFOCUS_BASE                           | Standard hourly rolling MPC                      | PV point/q50     | none                          | base deterministic given load | H24 rolling, execute first step |
| MPC_PROB     | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5                 | Standard stochastic hourly rolling MPC           | PV probabilistic | PV-focused low-PV-safe TKM K5 | base deterministic given load | H24 rolling, execute first step |
| CCOR_DET     | CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE           | CCOR-v2 first-step peak-certified MPC            | PV point/q50     | none                          | base deterministic given load | H24 rolling, execute first step |
| CCOR_PROB    | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 | CCOR-v2 stochastic first-step peak-certified MPC | PV probabilistic | PV-focused low-PV-safe TKM K5 | base deterministic given load | H24 rolling, execute first step |

Excluded from the final mainline: DA main cases, LOAD-QN, random load-error mainline, old S1-S6 scenario inputs, old invalid PV truth, and LitErr random load-error cases. DA is removed from the final main case matrix because the final research question is whether PV probabilistic information and CCOR control improve rolling MPC. DA can remain a reference-only or appendix benchmark, but it is not part of the final method matrix.

## 2. GHI Forecasting: Point Forecast Comparison

The repository contains a point forecast comparison for GHI/PV-related models. The main H1 comparison shows XGBoost, weather-classified CatBoost, and CNN-LSTM baselines. A separate ID-H24 literature-guided benchmark compares persistence and rolling multi-step ML variants over H1-H24.

Table: GHI point forecasting model comparison.

| Model Type   | Model                               |    RMSE |     MAE |       MSE |     R2 |    n | Remark                                                        |
|:-------------|:------------------------------------|--------:|--------:|----------:|-------:|-----:|:--------------------------------------------------------------|
| XGBoost      | Main XGBoost H1                     | 95.6715 | 63.3208 | 9153.0409 | 0.8926 | 3846 | BH-safe H1 run; chronological split; NWP h1 features used     |
| CatBoost     | Weather-Classified CatBoost (WC-CB) | 96.6837 | 63.2140 | 9347.7468 | 0.8903 | 3846 | BH-safe literature-style baseline                             |
| CNN-LSTM     | CNN-LSTM hybrid                     | 98.0615 | 68.3478 | 9616.0673 | 0.8872 | 3844 | BH-safe deterministic deep-learning baseline; PyTorch backend |

ID-H24 overall model comparison rows:

| label                 |      n |   rmse_ghi |   mae_ghi |   mbe_ghi |   r2_ghi |
|:----------------------|-------:|-----------:|----------:|----------:|---------:|
| PERS|Overall          | 101688 |   341.9913 |  234.0886 | -162.5795 |  -0.4146 |
| P1_XGB_LG|Overall     | 101688 |   136.9490 |   92.1591 |   -1.3901 |   0.7732 |
| P2_CB_LG|Overall      | 101688 |   136.7832 |   93.0359 |   -0.2365 |   0.7737 |
| G1_LGBM_LG|Overall    | 101688 |   137.5366 |   92.6443 |   -0.8972 |   0.7712 |
| G2_LGBM_D24|Overall   | 101688 |   136.7828 |   92.1286 |   -0.7019 |   0.7737 |
| G3_LGBM_Pool|Overall  | 101688 |   139.2081 |   94.6843 |   -1.1161 |   0.7656 |
| W1_XGB_D24|Overall    | 101688 |   136.1785 |   91.6789 |   -1.1442 |   0.7757 |
| W2_XGB_Pool|Overall   | 101688 |   138.4390 |   93.8674 |   -1.8418 |   0.7682 |
| W3_CB_D24|Overall     | 101688 |   137.0508 |   93.1664 |    0.0743 |   0.7728 |
| W4_CB_Pool|Overall    | 101688 |   139.6045 |   95.6334 |   -0.7313 |   0.7643 |
| W5_XGB_LG_Reg|Overall | 101688 |   136.5833 |   91.9270 |   -1.4023 |   0.7744 |
| W6_CB_LG_Reg|Overall  | 101688 |   136.8967 |   93.0258 |   -0.2514 |   0.7733 |

Evidence files: `new_pipeline/data/output/experiments/main_forecasting_model_comparison/main_model_comparison.csv` and `new_pipeline/data/output/experiments/lit_guided_id_h24_ghi/literature_guided_id_h24_ghi_model_comparison.csv`.

The final probabilistic path is XGBoost quantile-style forecasting with AgACI calibration because it provides usable quantile outputs for scenario construction and conformal calibration, while remaining compatible with the rolling H24 setting.

## 3. XGBQ Probabilistic GHI Forecast and Dynamic AgACI Calibration

The final probabilistic GHI/PV pipeline is based on XGBoost quantile outputs and AgACI-style adaptive conformal calibration. The ID-H24 probabilistic report defines approaches Q1/Q2/Q3 and calibrated variants. AgACI adjusts interval bounds online by lead-group/issue-date style updates to target empirical coverage and reduce interval miscoverage.

Overall ID-H24 probabilistic GHI metrics:

| approach        | subset   |      n |   raw_picp80 |   raw_picp90 |   cal_picp80 |   cal_picp90 |   cal_mpiw80 |   cal_mpiw90 |   q50_rmse |   mean_pinball |    crps |
|:----------------|:---------|-------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-----------:|---------------:|--------:|
| Q1_D24_raw      | Overall  | 101688 |      74.4500 |      92.4400 |      74.4500 |      92.4400 |     278.4900 |     521.2600 |   139.6600 |        30.8600 | 61.7200 |
| Q2_LG_raw       | Overall  | 101688 |      75.0500 |      92.6700 |      75.0500 |      92.6700 |     281.7400 |     519.6200 |   139.8300 |        30.8790 | 61.7580 |
| Q3_Residual_raw | Overall  | 101688 |      83.7800 |      91.6900 |      83.7800 |      91.6900 |     316.9600 |     421.3500 |   136.1800 |        33.5860 | 67.1720 |
| Q1_D24_cal      | Overall  | 101688 |      74.4500 |      92.4400 |      79.8100 |      89.7500 |     305.7900 |     544.4400 |   139.6600 |        30.8600 | 61.7200 |
| Q2_LG_cal       | Overall  | 101688 |      75.0500 |      92.6700 |      79.8100 |      89.7400 |     306.8500 |     545.3800 |   139.8300 |        30.8790 | 61.7580 |
| Q3_Residual_cal | Overall  | 101688 |      83.7800 |      91.6900 |      82.5200 |      90.7100 |     312.4400 |     419.1500 |   136.1800 |        33.5860 | 67.1720 |

Publication-style AgACI summary table:

| Method             | PICP80 All Daytime   | PICP90 All Daytime   |   ACE80 All Daytime |   MPIW80 (W/m2) |   CRPS All Daytime | PICP80 Peak (10-15h)   |
|:-------------------|:---------------------|:---------------------|--------------------:|----------------:|-------------------:|:-----------------------|
| Raw (uncalibrated) | 73.9%                | 84.4%                |             -0.0610 |        252.4000 |            59.9100 | 73.2%                  |
| AgACI (calibrated) | 80.3%                | 90.1%                |              0.0030 |        275.1000 |            59.8700 | 80.3%                  |

Implementation evidence:

- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_GHI_PROBABILISTIC_MODEL_SELECTION_REPORT.md`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_ghi_prob_metrics.csv`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_prob_metrics.csv`
- `new_pipeline/data/output/experiments/figures/table1_agaci_metrics.csv`

The final scheduling bridge uses PV-focused scenarios produced from calibrated GHI/PV probabilistic information. Realized PV/GHI is used for calibration/evaluation and final replay diagnostics according to the corresponding reports, not as future optimization input.

## 4. GHI-to-PV Conversion and PV Truth Rebuild

The final GHI-to-PV conversion is implemented as:

```text
PV_kW = clip(PR * GHI_Wm2 / 1000 * PV_CAP, 0, PV_CAP)
```

with `PR = 0.80` and `PV_CAP = 2687 kWp`. Code evidence includes `new_pipeline/scripts/experiments/exp_rebuild_pv_truth_from_cwa_ghi.py` and `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py`.

PV truth summary:

| series                         |   min_kw |    max_kw |   mean_kw |    p95_kw |    p99_kw |   annual_energy_kwh |   capacity_factor |   zero_fraction |
|:-------------------------------|---------:|----------:|----------:|----------:|----------:|--------------------:|------------------:|----------------:|
| CWA_GHI_PV_TRUTH               |   0.0000 | 2298.8778 |  328.3744 | 1671.9111 | 2054.0622 |        2876559.4738 |            0.1222 |          0.5059 |
| OLD_INVALID_NTUST_SOLAR_SCALED |   0.0000 | 2531.0264 |  357.5124 | 1843.6790 | 2240.3483 |        3131808.3905 |            0.1331 |          0.4997 |

The old `NTUST_Load_PV.csv::Solar_kWh` PV field is invalid and is not used for final replay. The corrected CWA-GHI-derived truth package is used for all final validation, settlement, KPI, and scheduling reports. An annual average of about 328 kW is plausible for 2687 kWp because the mean includes night hours; the rebuilt truth has max PV about 2298.88 kW and capacity factor about 12.22%.

Citation note: `CITATION_NEEDED` for the final thesis text if a formal reference is desired for the performance-ratio style PV conversion. A PVWatts/performance-ratio style source would be appropriate, but no final citation file was found in the repository.

## 5. PV Scenario Generation and Net-Load Scenario Construction

The final scenario bridge samples PV scenarios from calibrated probabilistic PV/GHI forecasts, applies physical clipping (`0 <= PV <= PV_CAP`), and combines those PV trajectories with a deterministic given campus load profile:

```text
netload_s_kw = load_given_kw - pv_s_kw
```

The load profile is deterministic within each outer case. It is not a probabilistic load forecast. Structured low/high load-bias profiles are used only for robustness and are never mixed into a single probability distribution.

Evidence:

- `new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py`
- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/PV_FOCUSED_NETLOAD_BUILD_REPORT.md`

Scenario input safety is audited in final scheduling folders. The optimization-facing scenario files do not contain realized/actual/truth columns.

## 6. Low-PV-Safe Tail-Aware Scenario Reduction

Raw PV scenario sets are too large for repeated hourly rolling MILP solves. The final method therefore uses K=5 representative scenarios. Ordinary KM is distribution-oriented and may drop low-probability, high-impact low-PV/high-net-load trajectories. The final low-PV-safe TKM explicitly preserves typical medoids, high-net-load medoids, low-PV/PV-shortfall medoids, and DCT-risk medoids; scenario weights are cluster probabilities.

The low-PV-safe validation status is:

```text
PASS_PV_TKM_LOWPV_SAFE_K5_FOR_DA_MPC_SCHEDULING
```

Selected validation metrics:

| mode   | bridge_type           |   pv_envelope_coverage |   pv_low_tail_coverage |   pv_active_netload_upper_coverage |   pv_sensitive_oc_upper_coverage |   daylight_top5_netload_upper_coverage |   scenario_max_netload_kw |
|:-------|:----------------------|-----------------------:|-----------------------:|-----------------------------------:|---------------------------------:|---------------------------------------:|--------------------------:|
| DA     | PV_TKM_LOWPV_SAFE_K5  |                 0.8414 |                 0.8324 |                             0.9424 |                           0.9033 |                                 0.8329 |                 4910.1709 |
| DA     | PV_TKM_SAFE_K5_OLD    |                 0.8414 |                 0.7441 |                             0.9045 |                           0.8533 |                                 0.7702 |                 4910.1709 |
| DA     | RAW_M500              |                 0.9556 |                 0.9471 |                             0.9842 |                           0.9598 |                                 0.9034 |                 4910.1709 |
| ID_H24 | PV_TKM_LOWPV_SAFE_K10 |                 0.7633 |                 0.7750 |                             0.9244 |                           0.6488 |                                 0.5015 |                 4798.1133 |
| ID_H24 | PV_TKM_LOWPV_SAFE_K5  |                 0.7144 |                 0.7221 |                             0.9022 |                           0.6237 |                                 0.4704 |                 4798.1133 |
| ID_H24 | PV_TKM_SAFE_K5_OLD    |                 0.7172 |                 0.6452 |                             0.8630 |                           0.5583 |                                 0.3945 |                 4798.1133 |
| ID_H24 | RAW_M500              |                 0.8235 |                 0.8301 |                             0.9470 |                           0.6773 |                                 0.5407 |                 4798.1133 |

Weight sanity:

| mode   | bridge_type           |   issue_count |   scenario_count_min |   scenario_count_max |   weight_sum_min |   weight_sum_max |   min_weight |   max_weight |   effective_scenarios_mean |   max_netload_kw |   min_pv_kw | medoid_type_counts                                                                 |
|:-------|:----------------------|--------------:|---------------------:|---------------------:|-----------------:|-----------------:|-------------:|-------------:|---------------------------:|-----------------:|------------:|:-----------------------------------------------------------------------------------|
| DA     | PV_TKM_SAFE_K5_OLD    |           365 |                    5 |                    5 |           1.0000 |           1.0000 |       0.0080 |       0.8280 |                     3.8099 |        4910.1709 |      0.0000 | dct_risk_tail:826;high_netload_tail:365;typical:634                                |
| DA     | PV_TKM_LOWPV_SAFE_K5  |           365 |                    5 |                    5 |           1.0000 |           1.0000 |       0.0020 |       0.8540 |                     3.1654 |        4910.1709 |      0.0000 | dct_risk_tail:365;high_netload_tail:365;low_pv_shortfall_tail:365;typical:730      |
| ID_H24 | PV_TKM_SAFE_K5_OLD    |          8776 |                    5 |                    5 |           1.0000 |           1.0000 |       0.0000 |       1.0000 |                     3.2763 |        4798.1133 |      0.0000 | dct_risk_tail:19048;high_netload_tail:8776;typical:16056                           |
| ID_H24 | PV_TKM_LOWPV_SAFE_K5  |          8776 |                    5 |                    5 |           1.0000 |           1.0000 |       0.0000 |       1.0000 |                     2.9893 |        4798.1133 |      0.0000 | dct_risk_tail:8776;high_netload_tail:8776;low_pv_shortfall_tail:8776;typical:17552 |
| ID_H24 | PV_TKM_LOWPV_SAFE_K10 |          8776 |                   10 |                   10 |           1.0000 |           1.0000 |       0.0000 |       1.0000 |                     6.4053 |        4798.1133 |      0.0000 | dct_risk_tail:8776;high_netload_tail:8776;low_pv_shortfall_tail:8776;typical:61432 |

Tail-aware ablation annual results:

| case                                   | scenario_reduction   |   full_total_m_ntd |   capex_m_ntd |   basic_m_ntd |   TOU_m_ntd |   OC_DCT_m_ntd |   Deg_m_ntd |   TREC_RE20_m_ntd |   monthly_peak_max_kw |   oc_hours |   risk0_hours |   shield_slack_positive_hours |   infeasible_steps |   fallback_steps | solver_status_summary   |   solve_time_total_sec |
|:---------------------------------------|:---------------------|-------------------:|--------------:|--------------:|------------:|---------------:|------------:|------------------:|----------------------:|-----------:|--------------:|------------------------------:|-------------------:|-----------------:|:------------------------|-----------------------:|
| MPC_PROB_PVFOCUS_KM_K5                 | PV_KM_K5             |           105.2017 |        7.3489 |        7.7470 |     75.7917 |         5.7955 |      2.0791 |            6.4395 |             4958.2013 |       1024 |          1551 |                             0 |                  0 |                0 | {"2": 8760}             |               349.7835 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 | PV_KM_K5             |           104.2036 |        7.3489 |        7.7470 |     76.1200 |         4.5677 |      1.9811 |            6.4389 |             4810.1219 |        812 |           206 |                             5 |                  0 |                0 | {"2": 8760}             |               359.2444 |

Ordinary KM K5 vs low-PV-safe TKM K5:

| comparison                         | ordinary_km_case                       | tailaware_tkm_case                             |   delta_full_total_m_ntd_tkm_minus_km |   delta_TOU_m_ntd_tkm_minus_km |   delta_OC_DCT_m_ntd_tkm_minus_km |   delta_Deg_m_ntd_tkm_minus_km |   delta_monthly_peak_kw_tkm_minus_km |   delta_oc_hours_tkm_minus_km | tailaware_improves_total   | tailaware_reduces_OC_DCT   | tailaware_increases_TOU   |
|:-----------------------------------|:---------------------------------------|:-----------------------------------------------|--------------------------------------:|-------------------------------:|----------------------------------:|-------------------------------:|-------------------------------------:|------------------------------:|:---------------------------|:---------------------------|:--------------------------|
| MPC tail-aware TKM vs ordinary KM  | MPC_PROB_PVFOCUS_KM_K5                 | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5                 |                               -0.7594 |                         0.0849 |                           -0.8218 |                        -0.0312 |                            -148.0793 |                           -17 | True                       | True                       | True                      |
| CCOR tail-aware TKM vs ordinary KM | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5 | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 |                               -0.7006 |                         0.0968 |                           -0.7949 |                        -0.0098 |                            -114.7010 |                            38 | True                       | True                       | True                      |

Interpretation: low-PV-safe TKM improves total cost mainly by reducing OC/DCT while slightly increasing TOU. This supports keeping low-PV-safe TKM K5 as the final PROB scenario reduction.

## 7. Shared MILP Framework for MPC and CCOR-MPC

The final standard MPC and CCOR-v2 cases use the same physical and settlement-oriented Layer-B MILP structure, with CCOR-v2 adding only a first-step soft peak shield. The core builder evidence is `milp_v2/layer_b/milp_cvar.py::solve_day_ahead()`. Final rolling scripts call this builder or reproduce its same equations with the added CCOR shield.

### Sets and Indices

- `h = 0,...,H-1`: rolling horizon step, with `H=24` in final cases.
- `omega in Omega`: scenario index; DET has one scenario, PROB has K=5.
- `k`: degradation segment index.
- Monthly state is carried through `D_init`/month-to-date peak rather than modeled as a separate annual horizon.

### Parameters

Key parameters include net-load or separate load/PV scenario trajectories, scenario weights `pi_omega`, TOU price, BESS power limit `PB`, BESS energy limit `EB`, SOC lower/upper bounds, charge/discharge efficiencies, initial SOC, initial Green SOC, contract capacity `CC=3232 kW`, month-to-date peak state `D_init`, OC/DCT segment multipliers, degradation segment slopes, one-hour time step, and for CCOR-v2 the shield margin `150 kW` and shield penalty coefficient.

### Decision Variables

The common MILP includes battery charge `P_ch_h`, discharge `P_dis_h`, binary charge/discharge mode `u_h`, SOC `E_h`, degradation segment energy `e_seg_hk`, scenario-specific grid-to-load `P_gl`, grid-to-charge `P_gc`, PV-to-load `P_pvl`, PV-to-charge `P_pvc`, PV curtailment `P_pvcu`, Green SOC variables `E_g`, green charge/discharge variables, scenario-specific demand proxy `D_cand_omega`, over-contract variables `Over_full_omega`, `O1_full_omega`, `O2_full_omega`, and CVaR variables. In final cases CVaR is disabled with `lam=0.0`.

CCOR-v2 adds `z_shield_omega` only for the first-step peak shield.

### Objective Function

The common objective minimizes:

```text
expected TOU energy cost
+ battery degradation cost
+ expected incremental OC/DCT cost
+ lam * CVaR_OC
```

with `lam=0.0` in final cases. CCOR-v2 adds:

```text
sum_omega pi_omega * rho_shield * z_shield_omega
```

Basic charge, CAPEX, and TREC/RE20 settlement components are included in final annual accounting, not as dispatch-changing terms in the final MILP objective. Load uncertainty is not an objective term.

### Constraints

The implemented constraints include power balance, PV allocation, grid import through grid-to-load/grid-to-charge, charge/discharge exclusivity, SOC transition, SOC bounds, optional year-end terminal SOC condition, degradation segment decomposition, DCT candidate peak constraints, segmented over-contract constraints, Green SOC transition and bounds, and CVaR linearization constraints. The final CCOR-v2 shield-only run does not rely on terminal reserve.

Equation-to-code traceability:

| Equation / Block                      | Mathematical Form                                                                                   | Code Evidence                                                                          | Notes                                                                                          |
|:--------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
| Horizon and scenarios                 | h in {0,...,H-1}; omega in Omega; pi_omega scenario weights                                         | milp_v2/layer_b/milp_cvar.py:27-56; scenario dictionaries passed by scheduling scripts | Final H24 rolling MPC executes first step only.                                                |
| Power balance                         | P_gl_{omega,h} + P_pvl_{omega,h} + P_dis_h = L_{omega,h}                                            | milp_v2/layer_b/milp_cvar.py:174                                                       | Load balance is scenario-specific; discharge is common across scenarios.                       |
| PV allocation                         | P_pvl_{omega,h}+P_pvc_{omega,h}+P_pvcu_{omega,h}=PV_{omega,h}                                       | milp_v2/layer_b/milp_cvar.py:178-182                                                   | PV can serve load, charge BESS, or be curtailed.                                               |
| Grid import                           | G_{omega,h}=kappa*(P_gl_{omega,h}+P_gc_{omega,h})                                                   | milp_v2/layer_b/milp_cvar.py:190                                                       | Used for DCT candidate peak; replay also computes realized grid import.                        |
| Charge/discharge exclusivity          | P_ch_h <= u_h PB; P_dis_h <= (1-u_h) PB                                                             | milp_v2/layer_b/milp_cvar.py:140-141                                                   | Binary u prevents simultaneous charge and discharge.                                           |
| SOC transition                        | E_h = E_{h-1} + eta_ch P_ch_h - P_dis_h/eta_dis                                                     | milp_v2/layer_b/milp_cvar.py:145-147                                                   | Delta t is one hour in the final experiments.                                                  |
| SOC bounds                            | SOC_min <= E_h <= SOC_max                                                                           | milp_v2/layer_b/milp_cvar.py:113                                                       | Bounds are applied through variable lower/upper bounds.                                        |
| Terminal SOC                          | Only final-day/year-end terminal SOC bound if is_final_day                                          | milp_v2/layer_b/milp_cvar.py:151-156                                                   | CCOR-v2 final m150 does not rely on terminal reserve.                                          |
| DCT candidate peak                    | D_cand_omega >= D_init; D_cand_omega >= G_{omega,h} for all h                                       | milp_v2/layer_b/milp_cvar.py:127,190                                                   | D_init is the month-to-date realized peak state.                                               |
| Over-contract segmentation            | Over_omega >= D_cand_omega-CC; Over_omega=O1_omega+O2_omega; 0<=O1<=0.1CC                           | milp_v2/layer_b/milp_cvar.py:127-130,193-195                                           | Segmented Taiwan DCT proxy.                                                                    |
| Expected incremental OC/DCT objective | sum_omega pi_omega c_basic(m1 O1_omega+m2 O2_omega) - PeakCost_prev                                 | milp_v2/layer_b/milp_cvar.py:230-235                                                   | Constant previous peak cost is subtracted in optimization objective.                           |
| Energy cost objective                 | sum_omega pi_omega sum_h tou_h * (P_gl_{omega,h}+P_gc_{omega,h})                                    | milp_v2/layer_b/milp_cvar.py:222-245                                                   | TOU price is included in MILP objective.                                                       |
| Degradation objective                 | sum_h sum_k deg_slope_k e_seg_{h,k}                                                                 | milp_v2/layer_b/milp_cvar.py:149-161,222-245                                           | Segmented discharge degradation is optimized.                                                  |
| Green SOC constraints                 | E_g transition; E_g<=E; P_chg<=P_pvc; P_disg<=P_dis                                                 | milp_v2/layer_b/milp_cvar.py:197-209                                                   | Green SOC accounting constraints are present; TREC/RE20 final amount is settlement/accounting. |
| CVaR                                  | eta + (1/(1-alpha)) sum pi xi; xi>=C_oc-eta                                                         | milp_v2/layer_b/milp_cvar.py:132-133,211-245                                           | Final cases pass lam=0.0, so CVaR is disabled.                                                 |
| CCOR-v2 shield                        | z_shield_omega >= G_{omega,0} - (max(CC,D_init)-150); objective += sum pi rho_shield z_shield_omega | run_scheduling_pv_focus_ccor_v2_smoke.py:189,202,236,263,286; full_year_m150.py:52     | Soft first-step shield only; no terminal reserve in final v2.                                  |
| Rolling replay/update                 | Execute h=0 action; update SOC and D_mtd using realized CWA-GHI PV truth and load profile           | run_scheduling_pv_focus_da_mpc.py:149-190; pv_focus_bridge_utils.py:62-139             | Truth enters replay/KPI only.                                                                  |

## 8. CCOR-MPC v2 Shield m150 Design

Standard rolling MPC already includes an OC/DCT term and month-to-date peak state, but the rolling first-step architecture can still create irreversible monthly peaks, especially by charging during peak-sensitive first steps. CCOR-v1 terminal reserve did not solve this robustly because the reserve did not bind when the critical risk was at the current executable step.

CCOR-v2 therefore adds first-step peak certification. At issue time `t`:

```text
D_mtd = current month-to-date realized peak before executing step t
D_guard = max(CC, D_mtd)
D_cert = D_guard - 150 kW
```

The shield is:

```text
G_0 <= D_cert + z_shield
z_shield >= 0
objective += rho_shield * z_shield
```

In the implementation this is scenario-specific for PROB:

```text
z_shield_omega >= kappa * (P_gl_omega0 + P_gc_omega0) - D_cert
```

DET and PROB both use the same shield principle. PROB retains the scenario-weighted stochastic structure.

Full-year mechanism evidence:

| final_label   | case                                           |   first_step_charging_near_risk_hours |   shield_slack_positive_hours |   risk0_hours |   infeasible_steps |   fallback_steps |
|:--------------|:-----------------------------------------------|--------------------------------------:|------------------------------:|--------------:|-------------------:|-----------------:|
| CCOR_DET      | CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE           |                                0.0000 |                        0.0000 |       39.0000 |                  0 |                0 |
| CCOR_PROB     | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 |                                3.0000 |                        7.0000 |      295.0000 |                  0 |                0 |
| MPC_DET       | MPC_DET_PVFOCUS_BASE                           |                              nan      |                      nan      |      nan      |                  0 |                0 |
| MPC_PROB      | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5                 |                              nan      |                      nan      |      nan      |                  0 |                0 |

June event caveat: At 2025-06-12 08:00, CCOR-v2 demand was 4485.99 kW, with p_ch=1510.74 kW, D_guard=4444.51 kW, D_cert=4294.51 kW, risk0=False, z_shield=0.00, and first-step stress net-load max=2755.87 kW. This indicates that m150 works annually but is not perfect; future work may explore m200, multi-step shielding, or realized-charge-aware first-step guards.

## 9. Main Four-Case Results

Final main annual cost table:

| final_label   | case                                           |   full_total_m_ntd |   TOU_m_ntd |   OC_DCT_m_ntd |   Deg_m_ntd |   TREC_RE20_m_ntd |   basic_m_ntd |   capex_m_ntd |   monthly_peak_max_kw |   oc_hours |   infeasible_steps |   fallback_steps | solver_status_summary   |
|:--------------|:-----------------------------------------------|-------------------:|------------:|---------------:|------------:|------------------:|--------------:|--------------:|----------------------:|-----------:|-------------------:|-----------------:|:------------------------|
| CCOR_DET      | CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE           |           105.7682 |     75.6416 |         6.4520 |      2.1258 |            6.4529 |        7.7470 |        7.3489 |             4977.6858 |        970 |                  0 |                0 | {"2": 8760}             |
| CCOR_PROB     | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 |           103.5030 |     76.2169 |         3.7728 |      1.9713 |            6.4461 |        7.7470 |        7.3489 |             4695.4209 |        850 |                  0 |                0 | {"2": 8760}             |
| MPC_DET       | MPC_DET_PVFOCUS_BASE                           |           106.4856 |     75.5570 |         7.2391 |      2.1472 |            6.4464 |        7.7470 |        7.3489 |             5133.6971 |       1021 |                  0 |                0 | {"2": 8760}             |
| MPC_PROB      | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5                 |           104.4422 |     75.8766 |         4.9737 |      2.0478 |            6.4482 |        7.7470 |        7.3489 |             4810.1219 |       1007 |                  0 |                0 | {"2": 8760}             |

Delta comparison table:

| comparison            | case_a                               | case_b                                         |   delta_full_total_m_ntd_b_minus_a |   delta_TOU_m_ntd_b_minus_a |   delta_OC_DCT_m_ntd_b_minus_a |   delta_Deg_m_ntd_b_minus_a |   delta_TREC_RE20_m_ntd_b_minus_a |   delta_monthly_peak_max_kw_b_minus_a |   delta_oc_hours_b_minus_a |
|:----------------------|:-------------------------------------|:-----------------------------------------------|-----------------------------------:|----------------------------:|-------------------------------:|----------------------------:|----------------------------------:|--------------------------------------:|---------------------------:|
| MPC_PROB vs MPC_DET   | MPC_DET_PVFOCUS_BASE                 | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5                 |                            -2.0433 |                      0.3196 |                        -2.2654 |                     -0.0994 |                            0.0018 |                             -323.5752 |                   -14.0000 |
| CCOR_PROB vs CCOR_DET | CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 |                            -2.2652 |                      0.5753 |                        -2.6792 |                     -0.1544 |                           -0.0068 |                             -282.2648 |                  -120.0000 |
| CCOR_DET vs MPC_DET   | MPC_DET_PVFOCUS_BASE                 | CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE           |                            -0.7174 |                      0.0846 |                        -0.7871 |                     -0.0214 |                            0.0065 |                             -156.0114 |                   -51.0000 |
| CCOR_PROB vs MPC_PROB | MPC_PROB_PVFOCUS_LOWPV_SAFE_K5       | CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5 |                            -0.9392 |                      0.3403 |                        -1.2009 |                     -0.0765 |                           -0.0021 |                             -114.7010 |                  -157.0000 |

Interpretation:

- `MPC_PROB` improves over `MPC_DET` by reducing OC/DCT, with a small TOU increase.
- `CCOR_PROB` improves over `CCOR_DET` by reducing OC/DCT more strongly, again with TOU increasing because dispatch becomes more risk-aware.
- `CCOR_DET` improves over `MPC_DET` mainly through OC/DCT reduction.
- `CCOR_PROB` is the best final case and improves over `MPC_PROB` mainly through lower OC/DCT and monthly peak.

## 10. Structured Load-Bias Robustness

The mainline uses deterministic given load. To address the critique that load is assumed perfect, the final robustness batch applies structured load-bias profiles as outer deterministic sensitivity cases, not probabilistic load forecasts. The profiles are season x time-of-day dependent and are not assigned probabilities.

Load profile audit:

| load_profile_name   | column                  | exists   |    annual_kwh |   annual_pct_vs_base |   max_load_kw |   min_load_kw |   negative_load_count |   oc_risk_hours_gt_CC |   near_oc_hours_gt_095CC | load_qn_used   | random_load_error_used   |
|:--------------------|:------------------------|:---------|--------------:|---------------------:|--------------:|--------------:|----------------------:|----------------------:|-------------------------:|:---------------|:-------------------------|
| base                | load_base_kw            | True     | 21268438.0000 |               0.0000 |     5179.0000 |        0.0000 |                     0 |                  1412 |                     1665 | False          | False                    |
| structured_low      | load_structured_low_kw  | True     | 20335670.9300 |              -4.3857 |     4920.0500 |        0.0000 |                     0 |                  1175 |                     1403 | False          | False                    |
| structured_high     | load_structured_high_kw | True     | 22201205.0700 |               4.3857 |     5437.9500 |        0.0000 |                     0 |                  1664 |                     1944 | False          | False                    |

Structured load-bias annual results:

| case                                                           | load_profile_name   |   full_total_m_ntd |   TOU_m_ntd |   OC_DCT_m_ntd |   Deg_m_ntd |   TREC_RE20_m_ntd |   monthly_peak_max_kw |   oc_hours |   infeasible_steps |   fallback_steps |
|:---------------------------------------------------------------|:--------------------|-------------------:|------------:|---------------:|------------:|------------------:|----------------------:|-----------:|-------------------:|-----------------:|
| MPC_DET_PVFOCUS_STRUCTURED_LOW                                 | structured_low      |            99.8732 |     70.9639 |         6.0580 |      2.1495 |            5.6058 |             4951.6514 |        798 |                  0 |                0 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_LOW                  | structured_low      |            98.2157 |     71.2264 |         4.2289 |      2.0651 |            5.5994 |             4524.3360 |        820 |                  0 |                0 |
| CCOR_V2_DET_SHIELD_m150_STRUCTURED_LOW                         | structured_low      |            99.3242 |     71.0370 |         5.4519 |      2.1322 |            5.6072 |             4843.3561 |        755 |                  0 |                0 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_LOW  | structured_low      |            97.2039 |     71.4242 |         3.0672 |      2.0169 |            5.5997 |             4459.0543 |        696 |                  0 |                0 |
| MPC_DET_PVFOCUS_STRUCTURED_HIGH                                | structured_high     |           113.1580 |     80.1590 |         8.4602 |      2.1444 |            7.2985 |             5467.4915 |       1278 |                  0 |                0 |
| MPC_PROB_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_HIGH                 | structured_high     |           111.2462 |     80.4840 |         6.3253 |      2.0413 |            7.2997 |             4961.0510 |       1227 |                  0 |                0 |
| CCOR_V2_DET_SHIELD_m150_STRUCTURED_HIGH                        | structured_high     |           112.4990 |     80.2579 |         7.7234 |      2.1201 |            7.3017 |             5207.4476 |       1226 |                  0 |                0 |
| CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_HIGH | structured_high     |           110.3747 |     80.9132 |         5.1227 |      1.9428 |            7.3001 |             4970.9985 |       1062 |                  0 |                0 |

The robustness folder reports final status:

```text
LOAD_ROBUSTNESS_PASS_MAIN_CONCLUSION_STABLE
```

Under both `structured_low` and `structured_high`, PROB improves DET and CCOR_PROB improves MPC_PROB. The main conclusion is therefore stable under the structured load-bias robustness design.

## 11. Implementation Audit and Traceability

| Method Component                                              | Thesis Description                                                                                            | Script / Artifact Evidence                                                                                                                    | Status   | Notes                                                                                                           |
|:--------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|:---------|:----------------------------------------------------------------------------------------------------------------|
| Corrected CWA-GHI PV truth used                               | Replay/KPI settlement uses CWA-GHI-derived PV truth.                                                          | pv_focus_bridge_utils.py:23, run_scheduling_pv_focus_da_mpc.py:254-271, run_scheduling_pv_focus_ccor_v2_full_year_m150.py: truth audit output | MATCH    | new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet |
| Old invalid PV truth excluded                                 | NTUST_Load_PV.csv::Solar_kWh is not used for final replay.                                                    | replay_truth_usage_audit.csv in main, CCOR, robustness, and tail-aware ablation folders                                                       | MATCH    | All final audit files mark CWA_GHI_PV_REBUILT and old invalid NTUST Solar_kWh false.                            |
| LOAD-QN excluded                                              | LOAD-QN is not used in final mainline.                                                                        | scenario/replay audit fields; run_scheduling_pv_focus_da_mpc.py:286, run_tailaware_ablation_pvfocus_km_vs_tkm.py:275                          | MATCH    | Final load is deterministic given profile; no probabilistic load forecast in mainline.                          |
| Random load error excluded                                    | Literature/random load error is appendix/diagnostic only, not final mainline.                                 | PV-focused scripts and reports; tail-aware/robustness audit flags                                                                             | MATCH    | Structured load bias is outer robustness, not random scenario mixing.                                           |
| Old S1-S6 excluded from mainline                              | Old S1-S6 is diagnostic-only and not a final scenario input.                                                  | PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md; scenario audit flags                                                                     | MATCH    | Old S1-S6 all-hour OC coverage is not used as pass/fail criterion.                                              |
| PV-focused scenario generation used                           | PV scenarios are sampled from calibrated PV/GHI probabilistic forecasts and combined with deterministic load. | exp_pv_focus_generate_pv_scenarios.py; exp_pv_focus_build_netload_scenarios.py; pv_focus_bridge_utils.py:207-300                              | MATCH    | netload_s_kw = load_kw - pv_s_kw.                                                                               |
| Low-PV-safe TKM K5 used for PROB mainline                     | PROB cases use low-PV-safe tail-aware K-medoids K=5.                                                          | exp_pv_focus_lowpv_reduction_fix.py; scenario_usage_audit.csv                                                                                 | MATCH    | Scenario files: id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet.                                        |
| Cluster probability weights used                              | Reduced scenario weights are cluster probabilities, not equal weights.                                        | scenario_usage_audit.csv; PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md weight table                                                    | MATCH    | Weight sums equal 1 per issue_time in final audits.                                                             |
| Optimization scenario files contain no truth columns          | No realized/actual/truth columns are passed into optimization-facing scenario files.                          | scenario_usage_audit.csv; run_scheduling_pv_focus_da_mpc.py:227-243                                                                           | MATCH    | Truth only enters replay/KPI settlement.                                                                        |
| Standard MPC uses base deterministic load                     | MPC_DET and MPC_PROB use the base deterministic load profile.                                                 | run_scheduling_pv_focus_da_mpc.py:154; load_profile_lookup('base')                                                                            | MATCH    | Structured profiles appear only in robustness folder.                                                           |
| CCOR-v2 uses shield m150                                      | CCOR-v2 adds a first-step soft peak shield with 150 kW margin.                                                | run_scheduling_pv_focus_ccor_v2_full_year_m150.py:52; run_scheduling_pv_focus_ccor_v2_smoke.py:263,286                                        | MATCH    | Terminal reserve is disabled in final v2 shield-only run.                                                       |
| Structured load-bias robustness uses structured_low/high only | Load bias is an outer deterministic robustness setting.                                                       | run_final_four_load_bias_robustness_pvfocus.py:608; load_profile_usage_audit.csv                                                              | MATCH    | No uniform or random error profiles used in final robustness batch.                                             |
| All final cases solved optimally 8760/8760                    | No infeasible or fallback steps in final main and robustness cases.                                           | annual_cost_summary_realized_basis.csv in main, CCOR, robustness, tail-aware folders                                                          | MATCH    | Gurobi status summary {'2': 8760}; infeasible_steps=0; fallback_steps=0.                                        |

Source manifest:

| source_path                                                                                                                         | exists   |   size_bytes |
|:------------------------------------------------------------------------------------------------------------------------------------|:---------|-------------:|
| new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/annual_cost_summary_realized_basis.csv                   | True     |         1333 |
| new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150/annual_cost_summary_realized_basis.csv              | True     |          816 |
| new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus/annual_cost_summary_realized_basis.csv                 | True     |         2526 |
| new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/annual_cost_summary_realized_basis.csv                    | True     |          712 |
| new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/km_vs_tkm_delta_comparison.csv                            | True     |          766 |
| new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/cwa_pv_truth_summary.csv                                              | True     |          416 |
| new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet                     | True     |       109789 |
| new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet | True     |     36026041 |
| new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md     | True     |        16525 |
| new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_ghi_prob_metrics.csv                                                    | True     |         8382 |
| new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_prob_metrics.csv                                                     | True     |         1560 |
| new_pipeline/data/output/experiments/main_forecasting_model_comparison/main_model_comparison.csv                                    | True     |          720 |
| new_pipeline/data/output/experiments/lit_guided_id_h24_ghi/literature_guided_id_h24_ghi_model_comparison.csv                        | True     |         6616 |
| new_pipeline/data/output/experiments/figures/table1_agaci_metrics.csv                                                               | True     |          230 |

## 12. Final Thesis Interpretation

PV probabilistic forecasting improves scheduling because it preserves low-PV/high-net-load risk information that directly affects Taiwan contract-capacity demand charges. Standard MPC can reduce local TOU cost but may still expose the system to month-maximum DCT risk. CCOR-v2 adds a first-step peak-certification mechanism that directly addresses this rolling-execution weakness.

Low-PV-safe TKM is beneficial because DCT risk is tail-driven. Ordinary scenario reduction can discard decision-relevant low-PV scenarios; the tail-aware reduction keeps these scenarios while preserving cluster probability weights. Structured load-bias robustness shows that the main conclusion remains stable when the deterministic load profile is perturbed.

The remaining caveat is the June 2025 event described above. CCOR-v2 shield m150 is a successful final method for this thesis stage, but not a perfect controller. Future work can test m200, multi-step shields, or realized-charge-aware guards. Load probabilistic forecasting is not claimed as a contribution because the site has only one year of load data.

## 13. Files Generated

This report generated the following supporting files in `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus`:

- `final_four_main_case_summary.csv`
- `final_four_delta_comparison.csv`
- `load_bias_robustness_summary.csv`
- `tailaware_ablation_summary.csv`
- `tailaware_ablation_delta_summary.csv`
- `method_component_traceability.csv`
- `milp_equation_traceability.csv`
- `final_case_definition_table.csv`
- `final_report_source_manifest.csv`
- `ghi_point_model_comparison_summary.csv`
- `ghi_probabilistic_agaci_summary.csv`
- `agaci_publication_table_summary.csv`

## Quality Checks

- DA cases are not included in the final main result table.
- Old invalid PV truth is not used in final replay/KPI artifacts.
- LOAD-QN is not used.
- Random load error is not used in the mainline.
- Old S1-S6 is not used as final scenario input.
- The main four-case results use base deterministic load.
- Structured load-bias robustness uses only structured_low and structured_high.
- Tail-aware ablation is separated from load-bias robustness.
- MILP equations are mapped to implemented code paths.

Final status: `FINAL_REPORT_READY_WITH_MINOR_UNCERTAIN_ITEMS`.
