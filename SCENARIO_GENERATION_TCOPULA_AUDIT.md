# Scenario Generation / t-Copula Audit

Audit date: 2026-06-12  
Scope: forecast-to-optimization **raw scenario generation** for current PV-focused PROB scheduling lines. Scenario reduction is mentioned only where needed to identify the formal PROB input file.

## 1. Executive Summary

- **Final verdict: the current formal PV-focused PROB experiment line does not use seasonal Student-t copula for raw scenario generation.**
- Seasonal t-copula code exists in legacy / archived DA scenario pipelines, especially `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py` and `THESIS_MAINLINE_ARCHIVE_2026_06/01_forecast_pipeline/stage2_t_copula_generation.py`.
- The current PV-focused bridge instead uses **calibrated PV quantiles + independent hour-by-hour inverse-CDF uniform sampling**, with `M=500` raw PV scenarios per issue time.
- The implemented raw scenario method is explicitly recorded in the raw PV parquet column `scenario_generation_method = quantile_inverse_uniform_sampling`.
- The formal net-load bridge is `netload_s_kw = load_kw - pv_s_kw`.
- Load is deterministic given the base campus load profile in the mainline. Structured low/high load profiles are outer robustness cases, not probabilistic load scenarios.
- I found no code evidence that the PV-focused mainline estimates a temporal covariance matrix, empirical copula, Gaussian copula, Student-t copula, or season-specific dependence structure for raw scenario generation.
- Some older text artifacts still describe t-copula or joint GHI+Load scenario generation. Those should be treated as legacy/stale for the final PV-focused thesis line unless explicitly scoped as historical DA/reference material.

## 2. Files and Scripts Inspected

Primary authority / final report:

- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md`
- `new_pipeline/scripts/experiments/write_final_thesis_method_result_report_pvfocus.py`

Current PV-focused bridge implementation:

- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_reduce_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_validate_scenarios.py`

Current formal scheduling consumers:

- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_da_mpc.py`
- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_v2_full_year_m150.py`
- `new_pipeline/scripts/experiments/run_pvfocus_milp_revised_conformant_v1.py`

Current PV-focused bridge outputs:

- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/PV_RAW_SCENARIO_GENERATION_REPORT.md`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/pv_raw_scenario_generation_summary.csv`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pv_raw_scenarios_M500.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/da_pv_raw_scenarios_M500.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pvfocus_base_raw_netload_M500.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`

Forecast / calibration evidence:

- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_GHI_PROBABILISTIC_MODEL_SELECTION_REPORT.md`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_PV_FORECAST_READINESS_REPORT.md`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_calibrated_quantiles.parquet`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/run_prob.log`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/run_pv_audit.log`

Formal scheduling output audits:

- `new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/scenario_usage_audit.csv`
- `new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/SCENARIO_AND_REPLAY_USAGE_AUDIT.md`
- `new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150/scenario_usage_audit.csv`
- `new_pipeline/data/output/experiments/pvfocus_milp_revised_conformant_v1/INPUT_FILE_MANIFEST.csv`

Legacy / archived t-copula evidence:

- `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py`
- `pipeline_standalone/README.md`
- `pipeline_standalone/AUDIT_INDEX.md`
- `THESIS_MAINLINE_ARCHIVE_2026_06/01_forecast_pipeline/stage2_t_copula_generation.py`
- `new_pipeline/diagnostics/pipeline_mapping.md`
- `MODULE_1B_PIPELINE_AUDIT_2026-05-22.md`
- `milp_v2/audit_outputs/PROBABILISTIC_FORECAST_TO_SCENARIO_AUDIT.md`
- `forecasting_handover_for_thesis_text/FORECASTING_FINAL_STATUS.md`
- `MILP_V2_PIPELINE_DOCUMENTATION.md`

Search terms used included `t-copula`, `copula`, `scenario generation`, `raw scenarios`, `M500`, `M=500`, `PV scenarios`, `GHI scenarios`, `probabilistic forecast`, `quantile`, `AgACI`, `calibration`, `net-load`, `representative scenarios`, `low-PV-safe`, `tail-aware`, `K-medoids`, `DA_PROB`, `MPC_PROB`, `CCOR_PROB`, `bridge`, `forecast-to-optimization`, `covariance`, `correlation`, `empirical copula`, `Gaussian copula`, `Student`, `load uncertainty`, and `load error`.

## 3. Current Implemented Scenario Generation Pipeline

### Step 1: Probabilistic forecast and calibration

The final report defines the mainline as PV uncertainty under deterministic load:

- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:5`: final study focuses on PV uncertainty and deterministic given campus load; load uncertainty is not probabilistic.
- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:72-76`: final probabilistic path is XGBoost quantile-style forecasting with AgACI calibration.
- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:80-87`: calibrated metrics table identifies `Q2_LG_cal`.

Forecast report evidence:

- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_GHI_PROBABILISTIC_MODEL_SELECTION_REPORT.md:12-19`: Q1/Q2/Q3 and calibrated variants; Q2 is LEADGROUP direct quantile regression.
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_GHI_PROBABILISTIC_MODEL_SELECTION_REPORT.md:21-23`: AgACI update rule is per issue_date and lead group.
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_GHI_PROBABILISTIC_MODEL_SELECTION_REPORT.md:34`: `Q2_LG_cal` calibrated row.
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/run_prob.log:86-95`: Q2_LG AgACI calibration applied and `q2_lg_calibrated.parquet` saved.
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/run_prob.log:119`: `Q2_LG_cal` reported with calibrated PICP values.
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/run_pv_audit.log:16-18`: Q2_LG_cal GHI quantiles converted to PV quantiles, producing `id_h24_pv_calibrated_quantiles.parquet`.

Implementation evidence:

- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py:26-27`: DA and ID calibrated PV quantile input paths.
- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py:142-143`: GHI-to-PV formula uses `clip(PV_PR * GHI / 1000 * PV_CAP_KW, 0, PV_CAP_KW)`.
- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py:162-184`: `full_id_pv_quantiles()` reads `id_h24_pv_calibrated_quantiles.parquet`, selects `pv_q05`, `pv_q10`, `pv_q50`, `pv_q90`, `pv_q95`, clips to `[0, PV_CAP]`, and returns the PV quantile package.

Observed quantile file:

- `id_h24_pv_calibrated_quantiles.parquet`: exists, 101,688 rows, includes `model_id = Q2_LG_cal`, `lead_group`, GHI quantiles, calibrated interval columns, and PV quantile columns.

### Step 2: Raw PV scenario generation

The raw PV scenario generator is:

- `new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py`

Key code evidence:

- `exp_pv_focus_generate_pv_scenarios.py:20-22`: input is `full_da_pv_quantiles()` or `full_id_pv_quantiles()`, output is `*_pv_raw_scenarios_M{m}.parquet`.
- `exp_pv_focus_generate_pv_scenarios.py:29-31`: groups by `issue_time`, calls `simulate_pv_for_issue(g, m, mode)`.
- `exp_pv_focus_generate_pv_scenarios.py:42-44`: CLI default `--M 500`, `--mode both`.
- `exp_pv_focus_generate_pv_scenarios.py:57`: report states raw PV scenarios are generated from calibrated PV quantiles using inverse-CDF uniform sampling.

Core generation function:

- `pv_focus_bridge_utils.py:187-197`: `inv_from_quantiles()` linearly interpolates between quantile levels `[0.05, 0.10, 0.50, 0.90, 0.95]`.
- `pv_focus_bridge_utils.py:200-202`: deterministic hash seed by `mode`, `issue_time`, and `m`.
- `pv_focus_bridge_utils.py:205-210`: for each issue, random matrix `u = rng.uniform(..., size=(m, h))`.
- `pv_focus_bridge_utils.py:212-213`: each raw scenario row is generated by inverse-CDF interpolation.
- `pv_focus_bridge_utils.py:221-223`: output columns include `source_quantile_package` and `scenario_generation_method = quantile_inverse_uniform_sampling`.

Output evidence:

- `id_h24_pv_raw_scenarios_M500.parquet`: exists, 105,032,500 rows, 8,776 issue times, 500 raw scenario IDs for the first issue.
- `da_pv_raw_scenarios_M500.parquet`: exists, 4,380,000 rows, 365 issue times, 500 raw scenario IDs for the first issue.
- `scenario_generation_method` unique value in raw PV files: `quantile_inverse_uniform_sampling`.
- `source_quantile_package` unique value for ID raw file: `ID_H24_calibrated`.
- `source_quantile_package` unique value for DA raw file: `DA_ACI`.

Important nuance:

- `pv_raw_scenario_generation_summary.csv` and `PV_RAW_SCENARIO_GENERATION_REPORT.md` currently list only the DA row. This appears to be stale or overwritten metadata because the ID raw parquet exists and is consumed downstream. The raw ID parquet itself and formal scenario usage audits are stronger evidence for the formal MPC/CCOR line.

### Step 3: Net-load construction

The raw net-load builder is:

- `new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py`

Key code evidence:

- `exp_pv_focus_build_netload_scenarios.py:12-15`: reads `*_pv_raw_scenarios_M{m}.parquet` and writes `*_raw_netload_M{m}.parquet`.
- `exp_pv_focus_build_netload_scenarios.py:18`: loads deterministic load profile lookup.
- `exp_pv_focus_build_netload_scenarios.py:29-31`: assigns `load_kw`, then `netload_s_kw = load_kw - pv_s_kw`.
- `exp_pv_focus_build_netload_scenarios.py:59-61`: report text says PV is stochastic, load profile deterministic, and structured profiles are not mixed into one probability distribution.

Output evidence:

- `id_h24_pvfocus_base_raw_netload_M500.parquet`: exists, 105,032,500 rows, 8,776 issue times, 500 raw scenarios for the first issue.
- Sample identity check: max absolute residual of `load_kw - pv_s_kw - netload_s_kw` over sampled rows was `0.0`.

### Step 4: Formal PROB scheduling input

Although reduction is out of scope for this audit, it identifies the formal PROB scenario file consumed by the scheduling line:

- `run_scheduling_pv_focus_da_mpc.py:31-32`: `MPC_PROB_SCENARIO = BRIDGE_OUT / "id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet"`.
- `run_scheduling_pv_focus_da_mpc.py:154-161`: MPC loads base deterministic load and reads `MPC_PROB_SCENARIO`.
- `run_scheduling_pv_focus_da_mpc.py:75-92`: `day_dict_prob()` constructs stochastic MILP scenarios from `pv_s_kw`, `load_kw`, and scenario weights.
- `run_scheduling_pv_focus_ccor_v2_full_year_m150.py:93-104`: CCOR loads base deterministic load and reads `PROB_SCENARIO`.
- `run_pvfocus_milp_revised_conformant_v1.py:50`: revised fullspec runner also reads `id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`.
- `run_pvfocus_milp_revised_conformant_v1.py:147-154`: scenario loader rejects truth-leakage columns and groups by `issue_time`.

Formal output audits:

- `scheduling_pv_focus_lowpv_safe_da_mpc/scenario_usage_audit.csv`: `MPC_PROB_PVFOCUS_LOWPV_SAFE_K5` uses `id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`, `k_min=5`, `k_max=5`, weight sums equal 1, and no realized/actual/truth columns.
- `scheduling_pv_focus_ccor_v2_full_year_m150/scenario_usage_audit.csv`: `CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5` uses the same ID_H24 PV-focused K5 scenario file.
- `pvfocus_milp_revised_conformant_v1/INPUT_FILE_MANIFEST.csv`: revised-fullspec line records the same `prob_scenarios` path.

## 4. Is Seasonal t-Copula Implemented?

Answer for the current formal PV-focused experiment line: **NO**.

What is implemented in current formal PV-focused generation:

- Method: **independent hour-by-hour quantile inverse-CDF uniform sampling** from calibrated PV quantiles.
- Input: calibrated PV quantile forecast file, not precomputed t-copula GHI scenarios.
- Raw sample count: `M=500`.
- Scenario seed: issue-time hash seed.
- PV conversion: performed upstream for quantile package; PV values clipped to `[0, PV_CAP]`.
- Net-load: deterministic load minus PV scenario.

Positive evidence against t-copula in the current formal line:

- `pv_focus_bridge_utils.py:205-210` creates independent uniform samples with `rng.uniform(size=(m, h))`.
- `pv_focus_bridge_utils.py:187-197` maps each uniform draw through marginal quantile interpolation.
- `pv_focus_bridge_utils.py:221-223` writes `scenario_generation_method = quantile_inverse_uniform_sampling`.
- Actual raw scenario parquet contains only `quantile_inverse_uniform_sampling`.
- No current formal PV-focused scripts (`exp_pv_focus_generate_pv_scenarios.py`, `exp_pv_focus_build_netload_scenarios.py`, `pv_focus_bridge_utils.py`, `run_scheduling_pv_focus_da_mpc.py`, `run_scheduling_pv_focus_ccor_v2_full_year_m150.py`, `run_pvfocus_milp_revised_conformant_v1.py`) contain `copula`, `covariance`, `correlation`, `student`, or equivalent t-copula implementation terms.

What t-copula implementation exists, but is not the current PV-focused mainline:

- `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py:3-24`: declares Stage 2 seasonal t-copula GHI scenario generation, N=500, four seasons.
- `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py:119-153`: implements `t_copula_loglik()`.
- `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py:157-205`: implements `fit_seasonal_copula()`, including seasonal PIT matrix, correlation matrix, and nu grid search.
- `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py:209-263`: implements `generate_scenarios_one_day()` using multivariate Student-t sampling.
- `THESIS_MAINLINE_ARCHIVE_2026_06/01_forecast_pipeline/stage2_t_copula_generation.py:279-284`: archived version uses `da_v2_quantiles.parquet` for calibration PITs.

Interpretation:

- Seasonal t-copula is a **legacy / archived DA forecast pipeline component**, not the implemented raw scenario generation method for the current PV-focused formal MPC/CCOR PROB mainline.

## 5. Temporal Dependence and Seasonal Dependence

Current formal PV-focused raw generation does **not** model temporal dependence.

Evidence:

- `pv_focus_bridge_utils.py:209`: `u = rng.uniform(EPS, 1 - EPS, size=(m, h))` generates an independent uniform draw for every raw scenario and every lead hour.
- `pv_focus_bridge_utils.py:212-213`: each scenario path is built by applying the inverse marginal quantile function independently at each lead hour.
- There is no current formal PV-focused code that estimates or loads an `R` matrix, covariance matrix, rank correlation matrix, empirical copula table, Gaussian copula, or Student-t copula.
- There is no season-specific dependence estimation in the current PV-focused raw generator.

What is preserved:

- The marginal uncertainty per lead time is represented through calibrated PV quantiles.
- Scenario IDs create 24-hour trajectories, but the cross-hour joint dependence comes only from sharing a row index across independently sampled lead-hour values. It is not a modeled autocorrelation or copula dependence structure.
- Tail preservation occurs in the later scenario reduction step, not in raw generation. Therefore it is safer to say **tail-aware reduction**, not **tail-aware generation**.

## 6. Load Treatment in Scenario Generation

Current formal mainline load treatment: **deterministic given load**.

Evidence:

- `pv_focus_bridge_utils.py:2-6`: docstring states mainline is PV probabilistic forecast plus deterministic given load; structured load-bias profiles are outer robustness cases, not probabilistic load scenarios.
- `pv_focus_bridge_utils.py:115-139`: load profiles are built from the CWA-GHI truth package load and deterministic bias matrices.
- `exp_pv_focus_build_netload_scenarios.py:18-31`: load is loaded from deterministic lookup; net-load equals `load_kw - pv_s_kw`.
- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:5`: load uncertainty is not modeled as a probabilistic load forecast because only one year of site load data is available.
- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:128-134`: final scenario bridge combines PV trajectories with deterministic given campus load; structured low/high load bias is robustness only.

Not used in final mainline:

- No probabilistic load forecast.
- No simulated random load error.
- No joint PV-load copula.
- No old S1-S6 load/PV scenario mainline.

## 7. Consistency Between Code and Thesis Textbase

### Consistent final text

The revised authority report is consistent with the implemented PV-focused raw generation at the high level:

- It does **not** claim seasonal t-copula for the final PV-focused bridge.
- It says PV scenarios are sampled from calibrated probabilistic PV/GHI forecasts.
- It says load is deterministic within each outer case.
- It defines net-load as `load_given - PV_s`.

Relevant lines:

- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:72-76`: XGBoost quantile + AgACI.
- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:126-134`: PV scenario generation and deterministic net-load construction.
- `../FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS_MILP_REVISED.md:136-143`: scenario evidence points to PV-focused scripts, not t-copula scripts.

### Text artifacts that can mislead if reused

These are not the current final PV-focused authority line, but they contain wording that can be mistaken for the final method:

| File | Problematic wording | Actual current PV-focused implementation | Recommendation |
|---|---|---|---|
| `forecasting_handover_for_thesis_text/FORECASTING_FINAL_STATUS.md:135-138` | "T-Copula" and "GHI + Load joint distribution" | Current PV-focused mainline uses PV-only quantile inverse sampling plus deterministic load | Mark as legacy DA text or remove from thesis mainline |
| `forecasting_handover_for_thesis_text/FORECASTING_FINAL_STATUS.md:223` | "T-Copula 建立 GHI+Load 聯合分布" | No joint load uncertainty in final PV-focused mainline | Replace with PV-focused wording |
| `MILP_V2_PIPELINE_DOCUMENTATION.md:126-137` | T-copula joint PV + Load scenario generation | Not current final method | Keep as historical documentation only |
| `milp_v2/audit_outputs/PROBABILISTIC_FORECAST_TO_SCENARIO_AUDIT.md:82-85` | Seasonal t-copula N=500 path | Legacy DA/M2 audit, not current PV-focused mainline | Label as superseded by PV-focused revised method |
| `new_pipeline/diagnostics/pipeline_mapping.md:13,75-76` | Mainline seasonal t-copula | Diagnostic mapping for older pipeline | Do not cite as final PV-focused method |
| `new_pipeline/scripts/experiments/write_ghi_audit_reports.py:187-210` | Calls Stage 2 t-copula outputs "Production" | GHI audit generator describes old DA pipeline; final PV-focused report does not use this as method authority | Avoid using this generated DA pipeline language in final thesis |

Recommended correction pattern:

- Replace "seasonal t-copula scenario generation" with "calibrated marginal PV quantile sampling".
- Replace "joint PV-load probabilistic scenario generation" with "PV-only probabilistic scenario generation with deterministic load bridge".
- Replace "temporal copula / autocorrelated scenarios" with "hour-wise marginal sampling; temporal dependence is not explicitly modeled".
- Replace "tail-aware generation" with "tail-aware scenario reduction".

## 8. Why Seasonal t-Copula Is / Is Not Used

The decision not to use seasonal t-copula in the current final PV-focused line is defensible, provided the thesis text does not overclaim.

Engineering and thesis-defense rationale:

1. **Data sufficiency**
   - A season-specific 24-dimensional dependence structure requires enough calibration days per season and stable PIT behavior across lead hours.
   - The old t-copula implementation itself contains a fallback when season sample size is too small: `pipeline_standalone/scenario/scripts/02_stage2_t_copula_generation.py:166-169` uses identity correlation if `n < d + 5`.
   - Only one year of site data makes joint PV-load and seasonal dependence estimation difficult to defend.

2. **PV uncertainty is already represented by calibrated quantiles**
   - The final forecasting layer produces calibrated marginal PV/GHI quantiles through Q2_LG + AgACI.
   - For a scheduling bridge whose key uncertainty is low PV / high net-load risk, calibrated lower-tail PV quantiles are directly useful.

3. **Load uncertainty is not mature enough for a true joint copula**
   - Final report explicitly excludes probabilistic load forecasting.
   - A "joint PV-load t-copula" claim would be false for the current mainline because load is deterministic.

4. **Method complexity and defense burden**
   - Seasonal t-copula would require defending PIT calibration quality, season-specific parameter stability, correlation matrix conditioning, nu selection, temporal dependence validation, and downstream sensitivity.
   - This complexity is not necessary if the thesis contribution is the scheduling bridge and CCOR-MPC control under calibrated PV uncertainty.

5. **Main contribution alignment**
   - Current final line is better described as a pragmatic forecast-to-optimization bridge: calibrated PV quantiles -> raw PV scenarios -> deterministic net-load bridge -> low-PV-safe reduction -> MPC/CCOR.
   - A full joint probabilistic forecasting model is a separate contribution.

Recommended position:

- Do **not** implement t-copula now unless the thesis needs a separate robustness extension.
- Do **not** claim t-copula in the final main method.
- Put seasonal t-copula in future work or appendix as a legacy/alternative scenario generator, if discussed at all.

## 9. Risk Assessment

| Risk | Level | Why it matters | Mitigation |
|---|---:|---|---|
| Hour-by-hour independent sampling may underrepresent temporal persistence of cloudy periods | MEDIUM | MPC decisions depend on 24h PV trajectory shape | State explicitly; defend calibrated marginal uncertainty and low-PV-safe reduction; future work: temporal copula / block bootstrap |
| No seasonal dependence structure | LOW-MEDIUM | Seasonal weather regimes may alter PV forecast dependence | Avoid seasonal t-copula claims; structured robustness and calibrated lead groups partially address nonstationarity |
| No joint PV-load probabilistic scenario generation | MEDIUM | Net-load risk can include load uncertainty | Final mainline uses deterministic load; structured load-bias robustness addresses sensitivity; do not overclaim joint distribution |
| Stale text says t-copula or GHI+Load joint distribution | HIGH if included in thesis | It would be a method overclaim relative to current code | Replace wording before submission |
| Raw generation report summary lists only DA although ID raw file exists | LOW | Metadata confusion, not method failure | Prefer parquet evidence and formal scenario usage audits; regenerate report only if documentation cleanup is needed |
| Tail-aware wording can be attached to generation instead of reduction | MEDIUM | Misstates where tail logic is implemented | Use "tail-aware reduction" only |
| Scenario paths include reduced K5 files, so raw generation is one step removed from optimizer | LOW | Common in stochastic optimization pipeline | Document raw M=500 generation separately from K5 reduction |

## 10. Recommended Thesis Wording

### 中文版

本研究最終主線採用 PV-focused 的 forecast-to-optimization 橋接方式。機率預測端先由 XGBoost 多分位數模型產生 GHI/PV 分位數，並經 AgACI 動態校準以改善邊際覆蓋率。針對每一個 MPC issue time，本研究從校準後的 PV 分位數邊際分布進行 inverse-CDF 均勻抽樣，產生 \(M=500\) 條 raw PV scenario；抽樣種子由 issue time 雜湊決定，以確保可再現性。PV scenario 經物理限制裁切於 \([0, PV_{cap}]\)。

排程最佳化端使用的 net-load scenario 定義為

```text
netload_s(t) = load_given(t) - PV_s(t)
```

其中 \(load_given(t)\) 為 deterministic campus load profile。最終主線不建立 probabilistic load scenario，也不宣稱 PV-load joint distribution。低 PV / 高 net-load 風險是在後續的 low-PV-safe tail-aware K-medoids scenario reduction 中保留，而不是在 raw scenario generation 階段透過 copula 建模。時間相關性與季節性 joint dependence modelling 留作 future work 或 robustness extension。

### English version

The final experiment line uses a PV-focused forecast-to-optimization bridge. The forecasting layer produces GHI/PV quantile forecasts using an XGBoost multi-quantile model, followed by AgACI calibration to improve marginal coverage. For each MPC issue time, \(M=500\) raw PV scenarios are generated by inverse-CDF sampling from the calibrated marginal PV quantiles, using an issue-time hash seed for reproducibility. The resulting PV values are physically clipped to \([0, PV_{cap}]\).

The optimization-facing net-load scenario is constructed as

```text
netload_s(t) = load_given(t) - PV_s(t)
```

where \(load_given(t)\) is a deterministic campus load profile. The final mainline does not construct probabilistic load scenarios and does not claim a joint PV-load distribution. Low-PV and high-net-load risk is preserved in the subsequent low-PV-safe tail-aware K-medoids scenario reduction stage, not through copula-based raw scenario generation. Explicit temporal or seasonal dependence modelling is left for future work or robustness extensions.

## 11. Final Verdict

- **是否使用 seasonal t-copula？** No, not in the current formal PV-focused mainline.
- **目前正式方法名稱應該叫什麼？** Calibrated marginal PV quantile inverse-CDF sampling with deterministic-load net-load bridge. If naming needs to be shorter: **PV-focused calibrated quantile sampling bridge**.
- **raw scenario 數量是否為 M=500？** Yes. Current raw PV files contain 500 raw scenarios per issue time.
- **raw scenarios 是否從 calibrated marginal quantiles 抽樣？** Yes. ID mainline uses `ID_H24_calibrated` PV quantile package; DA file uses `DA_ACI`.
- **是否有 temporal correlation / covariance / empirical copula / Gaussian copula / Student-t copula？** No evidence in the current formal PV-focused generation scripts or output metadata.
- **是否依 season 建立 dependence structure？** No evidence in the current formal PV-focused generation scripts.
- **是否使用 load uncertainty？** No. Load is deterministic given profile in the final mainline.
- **net-load 是否為 `given load - PV scenario`？** Yes.
- **是否需要修改論文文字？** Yes, if any final thesis text still says seasonal t-copula, joint PV-load probabilistic scenario generation, probabilistic load scenarios, autocorrelated scenarios, temporal copula scenarios, or tail-aware generation.
- **是否建議補跑 t-copula？** Not for the main thesis line. It would add complexity and require new validation. If desired, treat it as a future-work or appendix robustness extension.
- **若不補跑，如何防守？** State that the contribution is a calibrated PV uncertainty bridge for rolling MPC/CCOR under deterministic load, with load sensitivity handled by structured robustness, and with tail preservation handled in scenario reduction.

