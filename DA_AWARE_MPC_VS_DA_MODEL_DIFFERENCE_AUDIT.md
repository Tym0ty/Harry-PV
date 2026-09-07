# DA_AWARE_MPC_VS_DA_MODEL_DIFFERENCE_AUDIT

本報告釐清 BH 太陽能 / PV-BESS scheduling thesis 專案中，`DA / DA-Prob` 與最新 `DA-aware MPC / reference-guided MPC` 的數學模型與實作差異。結論先講清楚：

> `DA-aware MPC` 和 `DA-Prob` **不是同一個 MILP 只差在 rolling execution**。  
> 它們共享一部分電池與 power-balance 物理邏輯，但使用不同 solver、不同 forecast package、不同 scenario structure、不同 objective terms、不同 reference treatment，且 `DA-Prob` 有 Green SOC variables 與 CVaR peak-risk term，而 best H24 DA-aware MPC 是 risk-neutral MPC 加 DA peak/SOC reference penalties。

Scope clarification:

> 本報告的主要目的不是證明 DA-aware MPC 是否優於或劣於 M8，而是嚴格回答「DA-aware MPC 是否只是 DA MILP 改成每小時 rolling、只執行第一小時」。成本結果只作為輔助 context；模型等價性判斷主要依據 solver、變數、constraints、objective、scenario package、forecast information set 與 execution code path。

## 1. Executive Summary

核心問題：

> DA-aware MPC 和 DA 的差別是否只是：DA-aware MPC 每小時求解未來 24/48 小時、只執行第一小時；DA 前一日求解隔日 24 小時、整天照日前計畫執行；其餘 MILP 完全一樣？

答案：**No**。

兩者確實有「rolling first-stage execution vs frozen full-day execution」這個差異，但這不是唯一差異。主要差異包括：

- Solver 不同：DA 使用 `milp_cvar.py::solve_day_ahead()`；DA-aware MPC 使用 `milp_mpc.py::solve_mpc_milp()`。
- Forecast package 不同：DA 使用 frozen DA package `layerB_prob_package_x.parquet`；DA-aware MPC 使用 H24/H48 rolling forecast construction，包含 ID lead 1-6、same-day persistence、next-day DA tail。
- Scenario structure 不同：DA-Prob 使用 6 scenarios/day: `s0`-`s4` + `synth_miss`；DA-aware MPC 使用 K=5 probabilistic scenarios with `SC_WEIGHTS=[0.15,0.20,0.30,0.20,0.15]`，不含 `synth_miss`。
- Objective 不同：current `DA-Prob / M2` 包含 CVaR peak-risk term；best H24 DA-aware MPC 是 risk-neutral, `lam=0.0`，但新增 DA monthly peak reference soft penalty 與 terminal SOC reference soft penalty。
- Green SOC 不同：DA solver 有 Green SOC / renewable attribution variables；MPC solver 沒有 Green SOC variables。
- Terminal SOC 不同：DA 只有 final-year terminal SOC hard band；best DA-aware MPC 有每次 rolling horizon end 的 DA SOC soft reference band。
- Execution 不同：DA replay 整天執行 frozen `P_ch_plan/P_dis_plan`；DA-aware MPC 每小時重解，只執行 first-stage action。

因此最精確的分類是：

> **大部分 physical constraints 相同或相似，但 objective / forecast / scenario / reference terms / execution mode 不同。**

## 2. Direct Answer to Q1 / Q2

### Q1. 差別是否只是 rolling 24/48h + first-hour execution vs daily 24h frozen execution？

**No.**

這是重要差異之一，但不是唯一差異。還有以下主要差異：

1. different solver and code path；
2. different forecast information set；
3. different scenario count and scenario source；
4. different objective function；
5. current DA-Prob has CVaR, DA-aware MPC best H24 is risk-neutral；
6. DA-aware MPC has DA reference penalties that DA solver itself does not have；
7. DA solver includes Green SOC variables, MPC solver does not；
8. DA-aware MPC uses current executed SOC and month-to-date peak at every issue time；
9. DA-aware MPC uses no-CVaR DA master only as reference source in the latest rerun。

### Q2. DA-aware MPC 和 DA 的 MILP 設定是否完全一模一樣？

答案是選項 2：

> **大部分 physical constraints 相同，但 objective / forecast / scenario / reference terms 不同。**

不能稱為完全一樣，也不建議稱為同一個 MILP formulation。比較安全的說法是：

> DA and DA-aware MPC share the same core battery and settlement physics, but they are different MILP formulations implemented through different solvers.

## 3. Confirmed Comparison Objects

### 3.1 DA / DA-Prob

| Item | Verified |
|---|---|
| Solver | `milp_v2/layer_b/milp_cvar.py::solve_day_ahead()` |
| Runner | `milp_v2/run_phase3b.py` |
| DA-Det package | `milp_v2/bridge/packages/layerB_det_package.parquet` |
| DA-Prob package | `milp_v2/bridge/packages/layerB_prob_package_x.parquet` |
| Current M2 CVaR | yes, `lam=1.0`, `alpha=0.90` |
| Current M2 scenarios | 6 scenarios/day: `s0`-`s4` + `synth_miss` |
| DA replay execution | frozen full-day DA plan against realized PV/load |

Evidence:

- `milp_v2/run_phase3b.py:56` sets `CASE_LAM = {"M1":0.0, "M2":1.0, ...}`.
- `milp_v2/run_phase3b.py:250` defines `run_da_case()`.
- `milp_v2/run_phase3b.py:324` defines `replay_da_case()`.
- `milp_v2/run_phase3b.py:885-894` runs `M1` with `layerB_det_package.parquet` and `M2` with `layerB_prob_package_x.parquet`.
- `milp_v2/layer_b/milp_cvar.py:27` defines `solve_day_ahead()`.
- `milp_v2/layer_b/milp_cvar.py:241-245` defines CVaR objective term and objective.
- `layerB_prob_package_x.parquet` was verified to contain `s0`, `s1`, `s2`, `s3`, `s4`, `synth_miss`.

### 3.2 DA-aware MPC / reference-guided MPC

Best H24 case:

```text
MPC_Prob_H24_DARefPeakSOC_w1p0_band15
Full total = 101.3114 M NTD
```

| Item | Verified setting |
|---|---|
| Runner | `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py` |
| Solver | `milp_v2/layer_b/milp_mpc.py::solve_mpc_milp()` |
| Horizon | H=24 |
| Forecast source | fixed-horizon rolling MPC builder: ID lead 1-6 + persistence + DA tail |
| Scenario count | K=5 |
| CVaR | no, `lam=0.0` |
| Risk treatment | risk-neutral expected-cost MPC plus DA reference penalties |
| DA peak reference | no-CVaR DA master monthly max `D_mth_end` |
| DA SOC reference | no-CVaR DA replay hourly `E_soc` |
| Peak penalty weight | `w1p0 = 1.0 * basic_charge_rate` |
| SOC terminal band | `band15 = 0.15 * EB` |
| SOC penalty location | horizon end only |
| Full trajectory SOC tracking | no |

Evidence:

- `run_da_aware_mpc.py:67` defines `DAAwareCase`.
- `run_da_aware_mpc.py:89` defines `load_no_cvar_refs()`.
- `run_da_aware_mpc.py:119` defines `run_case()`.
- `run_da_aware_mpc.py:132-143` creates `FixedHorizonConfig`.
- `run_da_aware_mpc.py:165` calls `build_pv_scenarios()`.
- `run_da_aware_mpc.py:204-206` passes `m6c_d_ref`, `m6c_weight`, and `da_soc_terminal_ref`.
- `run_da_aware_mpc.py:469` defines `MPC_Prob_H24_DARefPeakSOC_w1p0_band15`.
- `milp_v2/layer_b/milp_mpc.py:55-56` defines `m6c_d_ref/m6c_weight`.
- `milp_v2/layer_b/milp_mpc.py:75-77` defines terminal SOC reference arguments.
- `milp_v2/layer_b/milp_mpc.py:193-211` implements terminal SOC soft band.
- Cost output: `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cost_component_comparison.csv`.

## 4. DA vs DA-aware MPC Comparison Table

| Item | DA-Prob / M2 | DA-aware MPC best H24 | Same / Different / Partly same | Evidence |
|---|---|---|---|---|
| Solver | `milp_cvar.py::solve_day_ahead()` | `milp_mpc.py::solve_mpc_milp()` | Different | `milp_cvar.py:27`; `milp_mpc.py:33`; `run_da_aware_mpc.py:204-207` |
| Runner | `run_phase3b.py` | `run_da_aware_mpc.py` | Different | `run_phase3b.py:250`; `run_da_aware_mpc.py:119` |
| Code path type | Day-ahead MILP | MPC MILP with DA reference penalties | Different | `run_phase3b.py:885-894`; `run_da_aware_mpc.py:469` |
| Solve frequency | once per target day | hourly rolling solve | Different | DA `run_da_case()` daily loop; DA-aware `run_case()` hourly loop |
| Execution | full-day frozen plan replay | first-stage action only | Different | `run_phase3b.py:324`; `milp_mpc.py:393-394` |
| Horizon | target-day 24h | future 24h from current hour | Partly same | both length 24 in best H24, but window meaning differs |
| Cross-day horizon | no, target day only | yes, H24 may cross midnight | Different | `mpc_fixed_horizon_utils.py:207-219`, `:222-292` |
| Forecast gate | frozen DA package under DA gate | rolling intraday information set | Different | `layerB_prob_package_x.parquet`; `build_pv_scenarios()` |
| PV forecast source | `layerB_prob_package_x.parquet` | ID lead 1-6 + persistence + DA tail | Different | `run_phase3b.py:891`; `mpc_fixed_horizon_utils.py:222-292` |
| Load forecast | DA package `load_input_kw` | deterministic/perfect load lookup | Partly same assumption, different source | `milp_cvar.py` scenario input; `mpc_fixed_horizon_utils.py:151`, `:324-331` |
| Scenario count | 6 | 5 | Different | package check; `mpc_fixed_horizon_utils.py:54`, `:244` |
| Scenario IDs | `s0-s4 + synth_miss` | K=5 quantile/scenario array, no `synth_miss` | Different | `layerB_prob_package_x.parquet`; `SC_WEIGHTS` |
| Scenario weights | daily package probabilities, including synth_miss | fixed `SC_WEIGHTS=[0.15,0.20,0.30,0.20,0.15]` | Different | package check; `mpc_fixed_horizon_utils.py:54` |
| Current SOC | DA carries state day-to-day | MPC uses executed SOC each hour | Different | `run_phase3b.py:250`; `run_da_aware_mpc.py:119` |
| Current month peak | DA state has `D_mth` | MPC state has `D_mth` / `peak_demand_running` | Partly same | `milp_cvar.py:46`; `milp_mpc.py:41` |
| Objective energy | expected energy cost | expected energy cost | Mathematically same idea | `milp_cvar.py:220`; `milp_mpc.py:347` |
| Objective degradation | PWL degradation | PWL degradation | Mathematically same idea | `milp_cvar.py:225`; `milp_mpc.py:352` |
| Objective OC | expected over-contract cost | expected over-contract cost | Similar | `milp_cvar.py:228-239`; `milp_mpc.py:355-367` |
| CVaR | yes, current M2 `lam=1.0` | no, `lam=0.0` | Different | `run_phase3b.py:56`, `:891`; `run_da_aware_mpc.py:191-192` |
| DA peak reference penalty | no separate reference penalty; DA optimizes own peak | yes, `m6c_d_ref/m6c_weight` | Different | `milp_mpc.py:269-274`, `:372-377` |
| DA SOC reference penalty | no | yes, terminal soft band | Different | `milp_mpc.py:193-211` |
| SOC tracking | DA own SOC trajectory | terminal reference only at rolling horizon end | Different | `run_da_aware_mpc.py:174-180`; no full trajectory tracking |
| Charge/discharge exclusivity | yes | yes | Mathematically same | `milp_cvar.py:140-141`; `milp_mpc.py:174-175` |
| SOC dynamics | yes | yes | Mathematically same | `milp_cvar.py:143-148`; `milp_mpc.py:177-182` |
| SOC bounds | yes | yes | Mathematically same | variable `E` bounds in both solvers |
| Power balance | yes | yes | Mathematically same | `milp_cvar.py:174-183`; `milp_mpc.py:223-238` |
| PV allocation | yes | yes | Mathematically same | `P_pvl/P_pvc/P_pvcu` in both solvers |
| Over-contract linearization | yes | yes | Similar | `milp_cvar.py:127-194`; `milp_mpc.py:162-252` |
| Degradation PWL | yes | yes | Mathematically same | `milp_cvar.py:161-166`; `milp_mpc.py:216-221` |
| Green SOC | yes | no | Different | `milp_cvar.py:119-122`, `:197-209`; no equivalent in `milp_mpc.py` |
| TREC in MILP objective | no | no | Same | final TREC computed in corrected KPI, not MILP |
| Final settlement formula | corrected annual formula | corrected annual formula | Same | `mpc_fixed_horizon_utils.py:595-603`; cost CSVs |
| Fixed design/year/replay truth | same final design/year/replay basis | same final design/year/replay basis | Same for final comparison | cost output artifacts |

## 5. Solver and Code Path Interpretation

### Are they the same MILP with different execution timing?

**No.**

They are different MILP formulations sharing some physical constraints.

More precise statement:

> DA-Prob and DA-aware MPC share the same battery power, SOC, PV allocation, grid import, degradation, and over-contract accounting logic at a high level. However, DA-Prob is a day-ahead scenario MILP with Green SOC and CVaR peak-risk treatment, while DA-aware MPC is a rolling intraday MPC MILP with DA peak/SOC reference penalties and no CVaR. Therefore, they are not the same MILP merely run at different times.

## 6. Forecast / Information Set Difference

### DA-Prob / M2

DA uses frozen day-ahead package:

```text
milp_v2/bridge/packages/layerB_prob_package_x.parquet
```

This package contains:

```text
s0, s1, s2, s3, s4, synth_miss
```

It is a day-ahead information set. The DA solver sees the target day as one 24h problem.

### DA-aware MPC

DA-aware MPC uses `build_pv_scenarios()` from `mpc_fixed_horizon_utils.py`:

- lead 1-6: intraday forecast;
- same-day lead > 6: persistence fallback;
- next day: DA tail if `forecast_tail_mode="da_forecast_tail"`;
- H48 D+2: persistence fallback.

Evidence:

- `mpc_fixed_horizon_utils.py:222` defines `build_pv_scenarios()`.
- `mpc_fixed_horizon_utils.py:260` uses lead logic.
- `mpc_fixed_horizon_utils.py:283` uses persistence for same-day lead > 6.
- `mpc_fixed_horizon_utils.py:292` uses DA tail.
- `mpc_fixed_horizon_utils.py:324-331` builds load scenarios using existing perfect-load assumption.

Therefore, the forecast package is not the same.

## 7. Scenario Count and Scenario Structure

| Item | DA-Prob / M2 | DA-aware MPC |
|---|---|---|
| K | 6 | 5 |
| IDs | `s0-s4 + synth_miss` | array scenarios, no `synth_miss` |
| Weights | daily probabilities from package | fixed `SC_WEIGHTS` |
| ID conformal | no, DA package already frozen | yes, H24 uses RC conformal mode |

This alone is enough to conclude they are not exactly the same MILP or the same stochastic program.

## 8. Objective Function Difference

### DA-Prob / M2 objective

From `milp_cvar.py`, the DA objective is:

```text
C_ene + C_deg + C_peak_exp + lam * C_cvar
```

For current `M2`, `lam=1.0`, so CVaR is active.

Evidence:

- `milp_cvar.py:220` energy cost.
- `milp_cvar.py:225` degradation cost.
- `milp_cvar.py:228-239` expected peak / over-contract cost.
- `milp_cvar.py:241-245` CVaR and objective.
- `run_phase3b.py:56` sets `M2=1.0`.
- `run_phase3b.py:891` runs M2 with `lam=1.0`.

### Best H24 DA-aware MPC objective

Best case:

```text
MPC_Prob_H24_DARefPeakSOC_w1p0_band15
```

Objective includes:

```text
C_ene + C_deg + C_peak_exp
+ C_m6c
+ C_da_soc_ref
```

It does **not** use CVaR because `lam=0.0`.

Peak reference term:

```text
s_peak_w >= D_cand_w - D_ref_month
C_m6c = m6c_weight * sum_w pi_w * s_peak_w
```

In best case, `w1p0` means:

```text
m6c_weight = 1.0 * basic_charge_rate
```

SOC terminal reference:

```text
E_H <= E_ref + 0.15*EB + s_soc_pos
E_H >= E_ref - 0.15*EB - s_soc_neg
C_da_soc_ref = 1.0 * (s_soc_pos + s_soc_neg)
```

Evidence:

- `run_da_aware_mpc.py:204-206` passes peak and SOC reference terms.
- `milp_mpc.py:269-274` implements `m6c`.
- `milp_mpc.py:372-377` adds `C_m6c` and `C_da_soc_ref` to objective.
- `milp_mpc.py:193-211` implements SOC terminal soft band.
- `run_da_aware_mpc.py:469` sets `band15`.

Important distinction:

> SOC reference is terminal-only at the rolling horizon end. It is not full-trajectory SOC tracking.

## 9. Green SOC / Renewable / TREC Handling

DA solver includes Green SOC variables:

- `E_g`
- `P_chg`
- `P_disg`

Evidence:

- `milp_cvar.py:119-122`
- `milp_cvar.py:197-209`

DA-aware MPC / `milp_mpc.py` does not include equivalent Green SOC variables.

TREC:

- TREC is not part of either MILP objective.
- TREC purchase is computed in final corrected annual KPI.
- Final settlement remains comparable because both are evaluated through corrected cost accounting.

Evidence:

- `mpc_fixed_horizon_utils.py:595-603` defines corrected KPI formula.
- DA-aware cost table and DA no-CVaR cost table use CAPEX, Basic, TOU, OC, Deg, TREC purchase.

## 10. Peak Reference and Month-to-Date Peak

### DA

DA uses state:

```text
{E_soc, E_g, D_mth}
```

It optimizes its own `D_cand` and carries `D_mth` day-to-day.

Evidence:

- `milp_cvar.py:46`
- `milp_cvar.py:127-194`
- `run_phase3b.py:250`

### DA-aware MPC

DA-aware MPC uses:

1. current executed month-to-date peak as MPC state;
2. DA `D_ref_month` as a soft objective reference.

Reference source is no-CVaR DA master:

```text
milp_v2/experiments/robustness_m8_cvar/results/run_20260531_013140/da_prob_no_cvar/M2_lam0_ablation_replay_daily.parquet
```

Calculation:

```text
D_ref_month = monthly max of D_mth_end
```

Evidence:

- `run_da_aware_mpc.py:89-108` builds `D_ref`, `SOC_ref`, and `grid_ref`.
- `run_da_aware_mpc.py:204-205` passes `m6c_d_ref` and `m6c_weight`.

It is a soft penalty, not a hard constraint.

## 11. Terminal SOC

### DA

DA has a terminal SOC hard band only on final simulation day / final-year condition:

- `milp_cvar.py:151-156`

It does not include the DA-aware MPC terminal SOC reference penalty.

### Original H24 MPC

Original H24 MPC has no active terminal SOC anchor in main fixed-H24 runs unless `is_last_hour=True`; for fixed H24 this is not the formal main behavior.

### DA-aware MPC

Best H24 DA-aware MPC adds:

- DA SOC reference from no-CVaR DA replay hourly `E_soc`;
- terminal soft band at horizon end;
- `band15 = 0.15 * EB`;
- penalty only on slack outside band;
- no full trajectory SOC tracking.

Evidence:

- `run_da_aware_mpc.py:174-180`
- `run_da_aware_mpc.py:206`
- `milp_mpc.py:193-211`

This is another reason DA-aware MPC and DA are not the same formulation.

## 12. Replay / Execution / Final Cost Accounting

Final settlement is comparable even if MILP formulation differs.

Why:

- same fixed design;
- same test year;
- same realized PV/load replay basis;
- same corrected annual cost formula:

```text
CAPEX + Basic + TOU + OC + Deg + TREC_purchase
```

Evidence:

- `mpc_fixed_horizon_utils.py:595-603`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cost_component_comparison.csv`
- `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cost_component_comparison.csv`
- `milp_v2/experiments/robustness_m8_cvar/results/run_20260531_013140/cost_component_comparison.csv`

## 13. Interpretation of DA-aware MPC Result

Key results:

| Case | Full total M NTD | TOU | OC |
|---|---:|---:|---:|
| `DA-Prob-NoCVaR` | 100.4633 | 76.0157 | 2.2769 |
| `MPC-Prob-H24 current` | 101.3338 | 75.2665 | 3.7720 |
| `MPC_Prob_H24_DARefPeakSOC_w1p0_band15` | 101.3114 | 75.3118 | 3.7144 |
| `M8_H24_FY2_M2_lam0_master` | 99.5205 | 75.1854 | 2.1020 |
| `MPC_Prob_H48_DARefPeak_w1p0` | 101.3636 | 75.2983 | 3.7878 |
| `MPC_Prob_H48_DARefPeakSOC_w1p0_band15` | 101.3616 | 75.2954 | 3.7878 |

Interpretation:

- DA-aware MPC only slightly improves standalone H24 MPC: `101.3338 -> 101.3114 M NTD`.
- The improvement mainly comes from OC reduction: `3.7720 -> 3.7144 M NTD`.
- TOU increases: `75.2665 -> 75.3118 M NTD`.
- Therefore, DA reference embedding trades energy flexibility for modest peak discipline.
- It still performs much worse than M8 no-CVaR master: `101.3114` vs `99.5205 M NTD`.
- H48 does not help and remains diagnostic.

Mechanistic explanation:

> DA-aware MPC puts DA reference into the optimization objective over the rolling horizon, but it still solves an expected-cost rolling MPC and executes only the first action. The reference penalty is soft and can be traded against TOU and degradation. In contrast, M8 uses a first-action accept/fallback rule: candidate actions are accepted only when they pass DA peak/SOC safety filters; otherwise the controller falls back to the DA action. This execution-stage arbitration can preserve MPC flexibility without allowing it to erode DA peak discipline.

## 14. Recommendation for Thesis Main Method

Recommendation: **keep M8 as the proposed method**.

DA-aware MPC should be included as:

- strong comparison baseline;
- robustness / ablation evidence;
- appendix or Chapter 4 robustness subsection.

H48 should be:

- diagnostic appendix only.

No need to replace M8 with DA-aware MPC, because:

- DA-aware MPC is feasible but not competitive;
- it only marginally improves standalone MPC;
- it remains worse than DA-Prob-NoCVaR and much worse than M8;
- it gives a fairer comparison to MPC by granting DA reference information, yet still supports M8.

## 15. Thesis Wording

### 15.1 中文口語說明版

DA-aware MPC 不是把 DA 原本的 MILP 拿來每小時滾動而已。它是用 intraday MPC 的 solver，再把 DA 的月尖峰參考值和 SOC 參考軌跡用 soft penalty 放進去。物理限制像電池充放電、SOC、PV 分配、grid import、over-contract 線性化大致相同，但 forecast、scenario、objective、CVaR、Green SOC 和 reference terms 都不同。實驗結果顯示，這種直接把 DA reference 放進 MPC 的方式只讓 standalone MPC 從 101.3338 M 降到 101.3114 M，改善很小，而且仍然比 M8 的 99.5205 M 高很多。所以這個強對照支持 M8 的設計：不要把 DA discipline 硬塞進整個 MPC objective，而是在執行第一個 action 時用 DA reference 做安全仲裁。

### 15.2 English thesis version

To test whether the DA reference discipline could be embedded directly into the intraday optimizer, a DA-aware MPC baseline was implemented. This baseline reuses the H24 probabilistic MPC forecast construction and adds a soft monthly peak-reference penalty based on the no-CVaR DA master. A second variant also adds a terminal SOC soft band around the DA SOC trajectory at the end of the rolling horizon. Although these variants share the core battery and settlement physics with the DA MILP, they are not the same formulation: they use a different rolling forecast information set, K=5 intraday scenarios without the synthetic miss scenario, no CVaR term, and additional DA-reference penalties. The best H24 DA-aware MPC reduced the full-year cost only slightly relative to standalone H24 MPC, from 101.3338 to 101.3114 M NTD, and remained substantially above the M8 no-CVaR-master result of 99.5205 M NTD. This indicates that directly embedding DA references into the MPC objective provides only limited peak-discipline improvement, while the M8 execution-stage arbitration better preserves intraday flexibility and DA safety discipline.

### 15.3 Oral defense Q&A version

**Q: Why not directly use a DA-aware MPC instead of M8?**  
A: We implemented that comparison. The DA-aware MPC uses the same H24 intraday forecast construction as MPC, but adds the DA monthly peak reference and a terminal DA SOC reference soft band into the MPC objective. It is therefore a stronger MPC baseline than standalone MPC. The best H24 DA-aware MPC was 101.3114 M NTD, only slightly better than standalone MPC at 101.3338 M NTD, but still much worse than M8 with the no-CVaR master at 99.5205 M NTD. This suggests that direct reference embedding over-constrains or weakens MPC flexibility, whereas M8 applies DA discipline only to the first executed action through accept/fallback arbitration.

## 16. Oral Defense Q&A

### Q: DA-aware MPC and DA are the same model, just different timing, right?

No. They share physical constraints, but they are different formulations. DA uses `solve_day_ahead()` with a frozen DA scenario package, Green SOC variables, and CVaR. DA-aware MPC uses `solve_mpc_milp()` with rolling H24/H48 forecasts, K=5 MPC scenarios, no CVaR, and DA reference penalties.

### Q: Is DA-aware MPC a fairer comparison than standalone MPC?

Yes. It is a strong comparison because it gives MPC direct access to DA peak and SOC reference concepts. Since it still underperforms M8, it strengthens the argument for M8 rather than weakening it.

### Q: Why does DA-aware MPC not beat M8?

Because the DA reference penalty is inside the rolling optimization objective and can be traded off against TOU, degradation, and expected cost. It modestly reduces OC but increases TOU. M8 instead applies DA discipline at execution time: if a candidate action violates DA peak/SOC safety logic, it falls back to DA. This preserves flexibility while preventing damaging first actions.

### Q: Should the thesis main method change?

No. The data support retaining M8 as the proposed method. DA-aware MPC should be reported as robustness / strong comparison. H48 should be placed in diagnostic appendix.

### Q: Do we need an even stronger DA-aware MPC?

Not before defense. The H24 and H48 versions already answer the methodological challenge at a reasonable scope. Full-month MPC would require long-horizon probabilistic forecasts and a much larger scenario tree, which is a separate research problem.

## 17. Suggested Slide Design

### Slide 1

**Title:** DA-aware MPC: Stronger MPC Baseline

**Key message:** We tested the obvious alternative: put DA peak/SOC references directly into MPC.

**Table columns:**

| Method | DA reference used? | Horizon | Full total | TOU | OC | Interpretation |
|---|---|---:|---:|---:|---:|---|
| DA-Prob-NoCVaR | native DA | 24h day | 100.4633 | 76.0157 | 2.2769 | safe DA master |
| MPC-Prob-H24 | no | H24 rolling | 101.3338 | 75.2665 | 3.7720 | flexible but weak peak discipline |
| DA-aware MPC best | peak + SOC | H24 rolling | 101.3114 | 75.3118 | 3.7144 | slight MPC improvement |
| M8 no-CVaR master | filter uses DA refs | first-action arbitration | 99.5205 | 75.1854 | 2.1020 | best result |

**Speaker note:** DA-aware MPC is not a weak baseline. It directly receives DA reference information, yet it only slightly improves MPC and still underperforms M8.

### Slide 2

**Title:** Why M8 Instead of DA-aware MPC?

**Key message:** Direct reference embedding trades TOU for small OC reduction; M8 filters the first action and preserves flexibility.

**Suggested visual:**

- Left: objective-level DA reference embedding.
- Right: M8 accept/fallback first-action arbitration.

**Speaker note:** DA-aware MPC puts the reference into every rolling optimization, so the optimizer can still trade off reference violations against expected cost. M8 uses DA references as safety filters at the execution stage. This explains why M8 is more effective in the full-year settlement.

## 18. Evidence Table

| Topic | Evidence |
|---|---|
| DA solver | `milp_v2/layer_b/milp_cvar.py:27` |
| DA runner | `milp_v2/run_phase3b.py:250`, `:885-894` |
| DA M2 CVaR active | `milp_v2/run_phase3b.py:56`, `milp_v2/layer_b/milp_cvar.py:241-245` |
| DA replay frozen plan | `milp_v2/run_phase3b.py:324` |
| DA Green SOC | `milp_v2/layer_b/milp_cvar.py:119-122`, `:197-209` |
| DA-aware runner | `milp_v2/experiments/da_aware_mpc/run_da_aware_mpc.py:119` |
| DA-aware best H24 case config | `run_da_aware_mpc.py:469` |
| DA-aware no-CVaR reference loader | `run_da_aware_mpc.py:89-108` |
| DA-aware H24 forecast builder | `run_da_aware_mpc.py:165`; `mpc_fixed_horizon_utils.py:222-292` |
| DA-aware peak penalty | `milp_v2/layer_b/milp_mpc.py:269-274`, `:372-377` |
| DA-aware SOC terminal penalty | `milp_v2/layer_b/milp_mpc.py:193-211` |
| DA-aware cost output | `milp_v2/experiments/da_aware_mpc/results/run_20260531_223624/cost_component_comparison.csv` |
| H48 diagnostic output | `milp_v2/experiments/da_aware_mpc/results/run_20260531_225727/cost_component_comparison.csv` |
| Corrected cost formula | `milp_v2/experiments/mpc_fixed_rolling_horizon/mpc_fixed_horizon_utils.py:595-603` |

## 19. Final Conclusion

DA-aware MPC is a valid and useful strong comparison, but it is **not** mathematically identical to DA and does **not** replace M8.

Recommended thesis positioning:

- `DA-Prob / DA-Det`: day-ahead baselines / master planning.
- `MPC-Prob-H24`: standalone intraday MPC baseline.
- `DA-aware MPC`: strong robustness baseline showing direct DA reference embedding.
- `M8`: proposed method, because it outperforms DA-aware MPC and better balances DA discipline with intraday flexibility.
