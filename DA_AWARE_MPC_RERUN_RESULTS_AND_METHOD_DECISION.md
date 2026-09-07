# DA_AWARE_MPC_RERUN_RESULTS_AND_METHOD_DECISION

本報告整理新增的 **DA-aware MPC / reference-guided MPC baseline** full-year rerun。目的不是取代既有正式結果，而是用實驗回答：

> 既然 DA master plan 能維持 monthly peak discipline 與 SOC discipline，為什麼不直接把 DA reference 放進 intraday MPC，而要設計 M8 arbitration？

本次所有新輸出均放在獨立資料夾，未覆蓋正式 artifacts。所有成本使用 corrected annual cost formula：

```text
CAPEX + Basic + TOU + OC + Deg + TREC_purchase
```

主要 output folders：

- H24 main DA-aware MPC run: `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/`
- H48 optional diagnostic run: `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/`

## 1. Executive Summary

### 1.1 最終判斷

**DA-aware MPC 沒有打敗 M8。**  
最佳 H24 DA-aware MPC case 是：

```text
MPC_Prob_H24_DARefPeakSOC_w1p0_band15 = 101.3114 M NTD
```

比較基準：

- `DA-Prob-NoCVaR` = `100.4633 M NTD`
- `MPC-Prob-H24 current` = `101.3338 M NTD`
- `M8_H24_FY2_M2_lam0_master` = `99.5205 M NTD`
- `M8_H24_FY2_filter_id_lead1` = `99.6348 M NTD`

因此：

- DA-aware MPC 比 standalone `MPC-Prob-H24` 略好：`-0.0224 M NTD`。
- DA-aware MPC 比 `DA-Prob-NoCVaR` 差：`+0.8481 M NTD`。
- DA-aware MPC 比 `M8_H24_FY2_M2_lam0_master` 差：`+1.7908 M NTD`。
- DA-aware MPC 比 forecast-only M8 robustness case 差：`+1.6766 M NTD`。

**方法決策：保留 M8 為主方法。**  
DA-aware MPC 可作為 strong comparison / robustness baseline，結論是：直接把 DA reference embedding 進 MPC objective 只帶來很小改善，沒有取代 M8 execution-stage arbitration。

### 1.2 對教授問題的直接回答

這次實驗已經讓 MPC 直接看見 DA monthly peak reference，並加入 DA terminal SOC soft reference。結果顯示：

- peak reference penalty 可以小幅降低 OC；
- SOC terminal reference band15 可以再小幅降低 OC；
- 但 TOU 增加抵銷大部分 OC 改善；
- H24/H48 DA-aware MPC 仍明顯高於 M8。

這支持論文主張：

> Direct DA reference embedding can over-constrain or blunt intraday MPC flexibility, while M8 preserves MPC as a candidate generator and applies DA discipline only at the first-action execution stage through safety arbitration.

## 2. What Was Implemented

### 2.1 Main H24 cases

| Case | Design |
|---|---|
| `MPC_Prob_H24_DARefPeak_w0p5` | H24 risk-neutral probabilistic MPC + DA monthly peak soft penalty, weight `0.5 * basic_charge_rate` |
| `MPC_Prob_H24_DARefPeak_w1p0` | H24 risk-neutral probabilistic MPC + DA monthly peak soft penalty, weight `1.0 * basic_charge_rate` |
| `MPC_Prob_H24_DARefPeak_w2p0` | H24 risk-neutral probabilistic MPC + DA monthly peak soft penalty, weight `2.0 * basic_charge_rate` |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band15` | H24 peak penalty + terminal SOC soft band, `15% EB` |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band10` | H24 peak penalty + terminal SOC soft band, `10% EB` |

### 2.2 Optional H48 cases

| Case | Design |
|---|---|
| `MPC_Prob_H48_DARefPeak_w1p0` | H48 risk-neutral probabilistic MPC + DA monthly peak soft penalty |
| `MPC_Prob_H48_DARefPeakSOC_w1p0_band15` | H48 peak penalty + terminal SOC soft band, `15% EB` |

H48 was completed because H24 ran without infeasibility, but H48 remains diagnostic because it relies on the existing DA-tail / persistence forecast construction beyond the intraday lead range.

## 3. Which Existing Hooks Were Reused

Task 0 result:

1. The solver already had DA reference hooks:
   - `m6b_grid_ref`
   - `m6b_weight`
   - `m6c_d_ref`
   - `m6c_weight`
   - `m7a_*`
   - `m7d_*`
   - `m71b_*`
2. The most directly relevant hook is `m6c_d_ref/m6c_weight`, which implements a soft monthly peak reference penalty:

```text
s_peak_w >= D_cand_w - D_ref_month
s_peak_w >= 0
penalty = m6c_weight * sum_w pi_w * s_peak_w
```

3. These hooks are solver-level and can be reused in H24/H48 full-year runs.
4. Existing hooks did not include terminal SOC reference to DA trajectory, so a small default-off solver extension was added.

## 4. Solver Modifications

Modified file:

- `milp_v2/layer_b/milp_mpc.py`

Added optional arguments to `solve_mpc_milp()`:

- `da_soc_terminal_ref`
- `da_soc_terminal_band`
- `da_soc_terminal_weight`

Implementation:

```text
E_H <= E_ref + band + s_soc_pos
E_H >= E_ref - band - s_soc_neg
penalty = da_soc_terminal_weight * (s_soc_pos + s_soc_neg)
```

Line references:

- New args: `milp_v2/layer_b/milp_mpc.py:75-77`
- Soft SOC reference constraints: `milp_v2/layer_b/milp_mpc.py:193-211`
- Objective term: `milp_v2/layer_b/milp_mpc.py:377`
- Returned slack diagnostic: `milp_v2/layer_b/milp_mpc.py:421`, `:445`
- Fallback default: `milp_v2/layer_b/milp_mpc.py:471`

Compatibility:

- All new arguments default to off.
- Existing official H24/M8/Phase3B calls are unaffected unless these arguments are explicitly passed.
- No existing official outputs were overwritten.

## 5. Exact Scripts and Config Paths

Created script:

- `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py`

Important functions:

- `DAAwareCase`: `run_da_aware_mpc.py:67`
- `load_no_cvar_refs()`: `run_da_aware_mpc.py:89`
- `run_case()`: `run_da_aware_mpc.py:119`
- `write_report()`: `run_da_aware_mpc.py:335`
- `run_all()`: `run_da_aware_mpc.py:459`

Config / manifests:

- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/config/manifest.json`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/config/manifest.json`

## 6. Input Artifacts

Primary DA reference source uses **no-CVaR DA master**:

- `milp_v2/experiments/robustness_m8_cvar/results/run_20260531_013140/da_prob_no_cvar/M2_lam0_ablation_da_daily.parquet`
- `milp_v2/experiments/robustness_m8_cvar/results/run_20260531_013140/da_prob_no_cvar/M2_lam0_ablation_replay_hourly.parquet`
- `milp_v2/experiments/robustness_m8_cvar/results/run_20260531_013140/da_prob_no_cvar/M2_lam0_ablation_replay_daily.parquet`

Reference construction:

| Reference | Source | Calculation |
|---|---|---|
| `D_ref_month` | `M2_lam0_ablation_replay_daily.parquet` | monthly max of `D_mth_end` |
| `SOC_ref[t]` | `M2_lam0_ablation_replay_hourly.parquet` | hourly `E_soc` after DA action replay |
| `grid_ref[t]` | `M2_lam0_ablation_replay_hourly.parquet` | hourly `p_grid_realized`; loaded but not used in final cases |

Other inputs:

- H24/H48 forecast construction from `mpc_fixed_horizon_utils.py`
- ID forecast source via `new_pipeline/data/final_forecast/id_forecast_v4_final.parquet`
- DA tail via existing `stage4_pv_scenarios_K5.parquet`
- load uses the existing deterministic/perfect load assumption in `new_pipeline/data/input/load_truth.parquet`

## 7. Output Artifacts

H24 run:

- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cost_component_comparison.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/runtime_infeasibility_diagnostics.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/all_kpis.json`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/DA_AWARE_MPC_RERUN_RESULTS_AND_METHOD_DECISION.md`
- Per-case hourly/daily/kpi outputs under `run_20260531_223624/cases/`

H48 diagnostic run:

- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cost_component_comparison.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/runtime_infeasibility_diagnostics.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/all_kpis.json`
- Per-case hourly/daily/kpi outputs under `run_20260531_225727/cases/`

Root summary report:

- `DA_AWARE_MPC_RERUN_RESULTS_AND_METHOD_DECISION.md`

## 8. Cost Comparison Table

### 8.1 H24 main comparison

| Case | Full total | TOU | OC | Deg | TREC | CAPEX | Basic | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `DA-Prob-NoCVaR` | 100.4633 | 76.0157 | 2.2769 | 1.8059 | 5.2688 | 7.3489 | 7.7470 | DA no-CVaR baseline |
| `MPC-Prob-H24 current` | 101.3338 | 75.2665 | 3.7720 | 1.9583 | 5.2410 | 7.3489 | 7.7470 | standalone H24 MPC |
| `M8_H24_FY2_M2_lam0_master` | 99.5205 | 75.1854 | 2.1020 | 1.8785 | 5.2587 | 7.3489 | 7.7470 | current best no-CVaR M8 |
| `M8_H24_FY2_filter_id_lead1` | 99.6348 | 75.2420 | 2.1532 | 1.8805 | 5.2632 | 7.3489 | 7.7470 | forecast-filter M8 robustness |
| `MPC_Prob_H24_DARefPeak_w0p5` | 101.3419 | 75.2971 | 3.7539 | 1.9525 | 5.2425 | 7.3489 | 7.7470 | DA-aware peak penalty |
| `MPC_Prob_H24_DARefPeak_w1p0` | 101.3262 | 75.2987 | 3.7385 | 1.9514 | 5.2417 | 7.3489 | 7.7470 | DA-aware peak penalty |
| `MPC_Prob_H24_DARefPeak_w2p0` | 101.3653 | 75.2967 | 3.7778 | 1.9529 | 5.2420 | 7.3489 | 7.7470 | DA-aware peak penalty |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band15` | 101.3114 | 75.3118 | 3.7144 | 1.9481 | 5.2412 | 7.3489 | 7.7470 | DA-aware peak + SOC |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band10` | 101.4659 | 75.3053 | 3.8711 | 1.9514 | 5.2422 | 7.3489 | 7.7470 | DA-aware peak + tighter SOC |

### 8.2 H48 diagnostic comparison

| Case | Full total | TOU | OC | Deg | TREC | CAPEX | Basic | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `MPC_Prob_H48_DARefPeak_w1p0` | 101.3636 | 75.2983 | 3.7878 | 1.9456 | 5.2360 | 7.3489 | 7.7470 | H48 DA-aware peak |
| `MPC_Prob_H48_DARefPeakSOC_w1p0_band15` | 101.3616 | 75.2954 | 3.7878 | 1.9479 | 5.2345 | 7.3489 | 7.7470 | H48 DA-aware peak + SOC |

## 9. Runtime / Infeasibility Issues

| Case | Runtime s | MILP fails | Full total | OC | TOU |
|---|---:|---:|---:|---:|---:|
| `MPC_Prob_H24_DARefPeak_w0p5` | 244.1 | 0 | 101.3419 | 3.7539 | 75.2971 |
| `MPC_Prob_H24_DARefPeak_w1p0` | 241.9 | 0 | 101.3262 | 3.7385 | 75.2987 |
| `MPC_Prob_H24_DARefPeak_w2p0` | 241.0 | 0 | 101.3653 | 3.7778 | 75.2967 |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band15` | 242.7 | 0 | 101.3114 | 3.7144 | 75.3118 |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band10` | 242.7 | 0 | 101.4659 | 3.8711 | 75.3053 |
| `MPC_Prob_H48_DARefPeak_w1p0` | 446.6 | 0 | 101.3636 | 3.7878 | 75.2983 |
| `MPC_Prob_H48_DARefPeakSOC_w1p0_band15` | 447.4 | 0 | 101.3616 | 3.7878 | 75.2954 |

No infeasibility occurred. Runtime was acceptable for H24 and H48 diagnostics, but H48 was roughly 1.8x slower and did not improve the result.

## 10. Interpretation Versus DA, MPC, and M8

### 10.1 Answers to required questions

1. **DA-aware MPC 是否低於 standalone MPC-Prob-H24？**  
   Partially yes. Best H24 DA-aware MPC (`PeakSOC_w1p0_band15`) is `0.0224 M NTD` lower than standalone MPC.

2. **DA-aware MPC 是否低於 DA-Prob-NoCVaR？**  
   No. Best H24 DA-aware MPC is `0.8481 M NTD` higher than DA-Prob-NoCVaR.

3. **DA-aware MPC 是否低於 M8_H24_FY2_M2_lam0_master？**  
   No. Best H24 DA-aware MPC is `1.7908 M NTD` higher than M8 no-CVaR master.

4. **DA-aware MPC 的改善主要來自 TOU 還是 OC？**  
   Relative to standalone H24 MPC, improvement comes from small OC reduction. TOU increases.  
   Best H24 case: OC decreases from `3.7720` to `3.7144 M`, while TOU increases from `75.2665` to `75.3118 M`.

5. **DA peak reference penalty 是否成功降低 OC？**  
   Slightly. `w1p0` reduces OC by about `0.0335 M`; `PeakSOC_w1p0_band15` reduces OC by about `0.0576 M`.

6. **加入 SOC terminal reference 是否改善或惡化結果？**  
   `band15` improves slightly versus peak-only w1.0. `band10` worsens substantially, indicating tighter SOC anchoring over-constrains the rolling MPC.

7. **是否出現 infeasibility？**  
   No. All H24 and H48 runs had zero MILP failures.

8. **是否出現過度保守導致 TOU 增加？**  
   Yes. DA reference penalty reduces OC slightly but raises TOU by about `0.03-0.05 M NTD`.

9. **若 DA-aware MPC 表現不如 M8，原因可能是什麼？**  
   Direct reference embedding penalizes the entire rolling horizon but still executes only the first action. It cannot use the simple fallback logic of M8. It also trades energy cost for modest peak reduction and may be affected by 24/48h forecast tail quality. M8 is less intrusive: it keeps MPC flexible and only filters the first action against DA peak/SOC discipline.

10. **若 DA-aware MPC 表現優於 M8，是否建議改主方法？**  
    Not applicable. DA-aware MPC did not beat M8.

### 10.2 Why H48 did not help

H48 increases the horizon but relies more heavily on DA-tail and persistence approximations beyond the reliable intraday lead range. It did not lower OC and did not improve total cost. Therefore H48 should remain diagnostic/future-work, not main text.

## 11. Whether the Thesis Main Method Should Change

**No. Keep M8 as the thesis proposed method.**

DA-aware MPC is a useful strong comparison, but the result supports M8 rather than replacing it:

- It is feasible.
- It gives MPC access to DA peak/SOC discipline.
- It slightly improves standalone MPC.
- It remains far worse than M8 under the no-CVaR master comparison.

Recommended classification:

| Case group | Thesis status |
|---|---|
| H24 DA-aware MPC | citable robustness / strong comparison |
| H48 DA-aware MPC | diagnostic appendix |
| M8 no-CVaR master | strong robustness result |
| M8 forecast-filter | strong information-boundary robustness result |

## 12. Recommended Chapter 3 Wording

> To address whether the DA peak and SOC references could be embedded directly into the intraday MPC, an additional DA-aware MPC baseline was implemented. The baseline uses the same H24 probabilistic intraday forecast construction and risk-neutral MPC objective as the standalone MPC, but adds a soft monthly peak-reference penalty based on the no-CVaR DA master. A second variant also adds a terminal SOC soft band around the DA SOC trajectory at the end of each rolling horizon. These variants are solved as standard MILPs and do not use the M8 arbitration filters.

## 13. Recommended Chapter 4 Table / Figure

Include a compact robustness table:

| Method | Full total | OC | TOU | Interpretation |
|---|---:|---:|---:|---|
| DA-Prob-NoCVaR | 100.4633 | 2.2769 | 76.0157 | DA master baseline |
| MPC-Prob-H24 | 101.3338 | 3.7720 | 75.2665 | flexible but weak peak discipline |
| DA-aware MPC best H24 | 101.3114 | 3.7144 | 75.3118 | slightly improves MPC but remains worse |
| M8 no-CVaR master | 99.5205 | 2.1020 | 75.1854 | best no-CVaR comparison |

Recommended figure:

- stacked cost component bars for DA, MPC, DA-aware MPC, and M8;
- highlight OC and TOU trade-off.

## 14. Oral Defense Answers

### Why not full-month MPC?

Full-month MPC requires a long-horizon probabilistic PV/load forecast and a much larger scenario tree. It would become a separate long-horizon stochastic MPC study. This thesis focuses on forecast-to-scheduling value under day-ahead and intraday information boundaries. The H24/H48 DA-aware MPC experiments are a reasonable strong comparison without changing the research scope.

### Why not directly put DA reference into MPC?

We tested that directly. The H24 DA-aware MPC includes the DA monthly peak reference in the MPC objective, and the PeakSOC variant also includes DA SOC discipline. It slightly lowers OC relative to standalone MPC, but total cost remains much higher than M8. This suggests direct embedding is less effective than using DA references as execution-stage safety filters.

### Is M8 unfair compared to MPC?

The new DA-aware MPC baseline gives MPC access to the same DA reference concepts: monthly peak reference and SOC trajectory. Since it still does not match M8, the comparison supports M8's design rather than relying on an unfair standalone MPC baseline.

### Does DA-aware MPC replace M8?

No. DA-aware MPC is feasible and useful as a strong baseline, but it does not beat M8. M8 should remain the proposed method, while DA-aware MPC should be reported as robustness / ablation evidence.

## 15. Remaining Limitations

- Load remains deterministic/perfect, consistent with the existing H24 MPC pipeline.
- SOC reference is implemented as a terminal soft band, not full trajectory tracking.
- H48 relies on DA-tail and persistence approximations; it is diagnostic.
- Peak-reference weights were limited to `0.5x`, `1.0x`, and `2.0x` basic charge rate to avoid uncontrolled tuning.
- Full-month MPC was intentionally not attempted.

## 16. Exact Commands to Reproduce

H24 main run:

```powershell
python milp_v2\experiments\da_aware_mpc\run_da_aware_mpc.py
```

H48 diagnostic run:

```powershell
python milp_v2\experiments\da_aware_mpc\run_da_aware_mpc.py --h48-only
```

Script validation:

```powershell
python -m py_compile milp_v2\layer_b\milp_mpc.py milp_v2\experiments\da_aware_mpc\run_da_aware_mpc.py
```

## 17. Appendix: File Paths, Functions, Changed Lines, Generated Outputs

### 17.1 Code changes

Modified:

- `milp_v2/layer_b/milp_mpc.py`
  - Added optional terminal SOC soft reference arguments and penalty.
  - Backward compatible; default off.

Added:

- `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py`

### 17.2 Function mapping

| File | Function / lines | Role |
|---|---:|---|
| `milp_v2/layer_b/milp_mpc.py` | `solve_mpc_milp()` args `:75-77` | optional DA SOC terminal reference inputs |
| `milp_v2/layer_b/milp_mpc.py` | `:193-211` | terminal SOC soft-band constraints |
| `milp_v2/layer_b/milp_mpc.py` | `:377` | objective includes SOC reference penalty |
| `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py` | `DAAwareCase :67` | case configuration |
| `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py` | `load_no_cvar_refs() :89` | builds `D_ref_month`, `SOC_ref`, optional `grid_ref` |
| `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py` | `run_case() :119` | full-year H24/H48 DA-aware MPC runner |
| `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py` | `run_case() :204-207` | passes `m6c_d_ref`, `m6c_weight`, and SOC reference into solver |
| `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py` | `run_all() :459` | orchestrates cases and writes comparison outputs |

### 17.3 Generated outputs

H24:

- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/config/manifest.json`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cost_component_comparison.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/runtime_infeasibility_diagnostics.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/all_kpis.json`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cases/*/*_hourly.parquet`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cases/*/*_daily.parquet`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cases/*/*_kpi.json`

H48:

- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/config/manifest.json`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cost_component_comparison.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/runtime_infeasibility_diagnostics.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/all_kpis.json`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cases/*/*_hourly.parquet`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cases/*/*_daily.parquet`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cases/*/*_kpi.json`

### 17.4 Citation status

| Case | Status |
|---|---|
| `MPC_Prob_H24_DARefPeak_w0p5/w1p0/w2p0` | thesis-safe as robustness / strong comparison |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band15` | thesis-safe as best H24 DA-aware comparison |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band10` | diagnostic; shows tighter SOC band can over-constrain |
| H48 cases | diagnostic appendix |

No official old outputs were overwritten.
