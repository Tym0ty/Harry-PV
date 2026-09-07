# FINAL_METHOD_LOGIC_AUDIT_AND_RERUN_RECOMMENDATION

本報告針對 BH 太陽能 / PV-BESS scheduling thesis 專案進行最終方法邏輯與實作一致性稽核。稽核原則是：以實際執行並產生正式結果的 code path、artifact、CSV/parquet/json outputs 為最高可信來源；handover、README、thesis draft 若與程式或正式 artifact 不一致，視為需修正文字，而非反推程式。

## 1. Executive Summary

### 1.1 最終可確認的方法真相

目前正式 thesis pipeline 可以定義為下列六個主 case，但其中 `DA-Prob = M2` 必須在方法文字中明確說明其 day-ahead MILP objective 包含 scenario-based CVaR peak-risk term：

| 論文 case name | Internal ID | 實作判定 | 角色 |
|---|---:|---|---|
| `DA-Det` | `M1` | confirmed | deterministic DA baseline |
| `DA-Prob` | `M2` | confirmed, but includes CVaR | probabilistic DA / master plan |
| `MPC-Det-H24` | `M3_H24_det` | confirmed | deterministic intraday MPC baseline |
| `MPC-Prob-H24` | `M5_H24_prob_lam0` | confirmed | probabilistic intraday MPC baseline |
| `M8-Det-H24` | `M8_Det_H24_FY2` | confirmed | deterministic M8 arbitration |
| `M8-Prob-H24` | `M8_H24_FY2` | confirmed | final proposed method |

正式成本數字以 `milp_v2/experiments/mpc_fixed_rolling_horizon/results/final_thesis_package/FINAL_MAIN_COMPARISON_H24_ALIGNED.csv` 與 `thesis_tables/Table_4-4-1_main_cost_comparison.csv` 為準。`94.x M NTD`、TREC revenue、舊版 `compute_kpis()` 產生的總成本均應作廢。

### 1.2 高風險不一致

| Issue | 稽核結論 | 風險 |
|---|---|---|
| `DA-Prob` 是否 `No CVaR` | 程式與 artifact 顯示 `M2` 使用 `lam=1.0, alpha=0.90` 的 CVaR peak-risk term；handover 中 `No CVaR` 是錯誤或至少不精確。 | high |
| DA gate time | 正式 artifact 支援 `D-1 20:00 local / D-1 12:00 UTC`。`D-1 16:00` 是後期文字帶入，不是正式實作；未找到正式 `D-1 22:00` code/artifact 證據。 | high |
| DA scenario count | 正式 `M2` 使用 6 scenarios/day：`s0`-`s4` plus `synth_miss`，不是單純 five scenarios。 | medium |
| ID H24 terminal SOC anchor | `MPC-Det-H24` 與 `MPC-Prob-H24` 正式主結果沒有 terminal SOC hard band、soft penalty、也沒有 anchor to DA SOC trajectory。 | medium |
| ID MPC 是否使用 DA `D_ref` | 主 H24 MPC 沒有直接使用 DA `D_ref`；`D_ref` 主要在 M8 F1 filter 使用。 | medium |
| M8 information boundary | M8 filter replay 使用 current-hour realized PV/load 進行安全檢查，應揭露為 replay/near-real-time approximation，不宜宣稱完全 deployment-clean online controller。 | high |

### 1.3 是否需要補跑實驗

若論文願意如實描述上述設計，沒有任何實驗屬於「必須補跑才能答辯」。但若論文或簡報堅持宣稱 `DA-Prob` 是 `No CVaR`，則必須重跑 `DA-Prob lam=0` 並同步更新 downstream M8 master plan；否則應改文字，不改實驗。

最有價值但非必要的補強是：`M8` 使用 forecasted current-hour PV/load 取代 realized current-hour PV/load 的 robustness replay，以及 `DA-Prob without CVaR` ablation。

## 2. Confirmed Method Truth

### 2.1 正式 case mapping 與成本

正式主表來源：

- `milp_v2/experiments/mpc_fixed_rolling_horizon/results/final_thesis_package/FINAL_MAIN_COMPARISON_H24_ALIGNED.csv`
- `BH_THESIS_HANDOVER_2026_05_FINAL/04_FINAL_RESULT_TABLES/FINAL_MAIN_COMPARISON_H24_ALIGNED.csv`
- `thesis_tables/Table_4-4-1_main_cost_comparison.csv`

| Case | Internal ID | Script / solver | Forecast package / input | Output evidence | Full total M NTD | TOU | OC | Deg | TREC | CAPEX | Basic | Role |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `DA-Det` | `M1` | `milp_v2/run_phase3b.py` -> `layer_b.milp_cvar.solve_day_ahead()` | `milp_v2/bridge/packages/layerB_det_package.parquet` | final aligned table | `102.47 (est)` | 75.1357 | 5.0032 | 1.9900 | 5.2650 | 7.3489 | 7.7470 | baseline |
| `DA-Prob` | `M2` | `milp_v2/run_phase3b.py` -> `layer_b.milp_cvar.solve_day_ahead()` | `milp_v2/bridge/packages/layerB_prob_package_x.parquet` | `phase3b_M2_da_daily.parquet`, corrected final table | 100.4855 | 76.0648 | 2.2498 | 1.8084 | 5.2666 | 7.3489 | 7.7470 | DA probabilistic master |
| `MPC-Det-H24` | `M3_H24_det` | `run_mpc_fixed_horizon_fullyear.py` -> `mpc_fixed_horizon_utils.run_fullyear()` -> `layer_b.milp_mpc.solve_mpc_milp()` | H24 deterministic ID forecast package | `H24_cost_component_raw.csv` | 104.5324 | 74.6310 | 7.4222 | 2.1155 | 5.2679 | 7.3489 | 7.7470 | ID deterministic baseline |
| `MPC-Prob-H24` | `M5_H24_prob_lam0` | same H24 MPC path, `forecast_mode="prob"`, `lam=0` | ID probabilistic lead 1-6 plus DA tail | `H24_cost_component_raw.csv` | 101.3338 | 75.2665 | 3.7720 | 1.9583 | 5.2410 | 7.3489 | 7.7470 | ID probabilistic baseline |
| `M8-Det-H24` | `M8_Det_H24_FY2` | `run_m8_det_h24.py` + `m8_utils.py` | `M1` master + `M3_H24_det` candidate | `FINAL_MAIN_COMPARISON_H24_ALIGNED.csv` | 101.6968 | 74.2915 | 4.9520 | 2.0706 | 5.2868 | 7.3489 | 7.7470 | deterministic arbitration |
| `M8-Prob-H24` | `M8_H24_FY2` | `run_m8_h24_standalone.py` + `m8_utils.py` | `M2` master + `M5_H24_prob_lam0` candidate | `H24_cost_component_raw.csv`, final aligned table | 99.6088 | 75.2394 | 2.1368 | 1.8797 | 5.2570 | 7.3489 | 7.7470 | final proposed method |

`DA-Det` 的 `Deg/TREC/full total` 在正式表中標為 estimated，因為其 full-year corrected replay components 與其他 H24 cases 的最終 replay pipeline 不完全同級；論文中可作 baseline，但應保留 `estimated` 註記。

### 2.2 舊結果與作廢規則

以下不得作為正式 thesis result：

- `94.x M NTD` 的舊 `M8_H24_FY2` 或 `MPC-Prob-H24` 數字。
- 任何把 TREC 當作 revenue/subtraction 的舊 total。
- `milp_v2/run_phase3b.py::compute_kpis()` 舊式 KPI，因其不等於 final corrected full cost accounting。
- `m8_h24_results.csv` 舊版未修正 total，例如舊 `M8_H24_FY2 = 94.3518 M NTD`。

作廢證據：

- `BH_THESIS_HANDOVER_2026_05_FINAL/ARCHIVE_NOTES_AND_OBSOLETE_NUMBERS.md:28-30`
- `BH_THESIS_HANDOVER_2026_05_FINAL/ARCHIVE_NOTES_AND_OBSOLETE_NUMBERS.md:64-69`
- `BH_THESIS_HANDOVER_2026_05_FINAL/ARCHIVE_NOTES_AND_OBSOLETE_NUMBERS.md:90-106`
- `BH_THESIS_HANDOVER_2026_05_FINAL/07_SCRIPT_OUTPUT_INDEX.md:112-118`

## 3. Unresolved Inconsistencies

| 不一致 | 較可信來源 | 較低可信來源 | 修正建議 |
|---|---|---|---|
| `DA-Prob` 被稱為 `No CVaR` | `run_phase3b.py` 傳 `lam=1.0`；`milp_cvar.py` objective；`phase3b_M2_da_daily.parquet` 有非零 `cvar_val` | handover case naming | 保留 `DA-Prob` 名稱可以，但方法章須寫明包含 CVaR peak-risk treatment；或改稱 `DA-Prob-Risk-Aware` |
| DA gate `D-1 16:00` | bridge package / issue-target dataset / gate audit 指向 `D-1 20:00 local` | late handover / thesis text | Chapter 3 改成 `D-1 20:00 local / D-1 12:00 UTC`，Abstract 避免寫死 |
| `five reduced scenarios` | package 實際 6 scenarios/day | 簡化文字 | 寫成 `five reduced scenarios plus one synthetic miss scenario` |
| ID MPC anchor to DA SOC | H24 runner 與 solver 沒有啟用 terminal SOC band/anchor | thesis draft 若有 anchor wording | 改成 M8 F2 在 MILP 外檢查 SOC corridor，不是 ID MPC constraint |
| ID MPC 使用 DA `D_ref` | H24 MPC objective/constraints 沒有 DA `D_ref` | 直覺式描述 | 改成 DA `D_ref` only used by M8 F1 filter in the final method |

## 4. DA Gate Time and Forecast Information Boundary

### 4.1 最終 gate time 判定

正式實作應寫成：

> The day-ahead forecast package is frozen at `D-1 20:00` local time (`D-1 12:00 UTC`) under the repository's current GFS availability assumption.

證據：

- `DA_GATE_TIME_AUDIT.md:3-15` 結論指出正式 artifacts 支援 `D-1 20:00 local / D-1 12:00 UTC`。
- `DA_GATE_TIME_AUDIT.md:21-38` 列出 bridge outputs、issue-target dataset、README、strict builder 的 gate evidence。
- `DA_GATE_TIME_AUDIT.md:71-84` 指出若採 16:00，GFS 12Z D-1 會有 NWP availability risk。
- `DA_GATE_TIME_AUDIT.md:90-112` 已提供論文建議文字。

### 4.2 對 user 七個問題的結論

1. 目前正式 DA gate 應寫成 `D-1 20:00 local / D-1 12:00 UTC`。
2. 未找到 repository code/artifact 支持 `D-1 22:00` 作為正式 gate。
3. `D-1 16:00` 是後期 thesis/handover text 帶入，非正式執行結果。
4. 若採 repo assumption：`initial_time <= deadline_utc`，`da_v2 RMSE = 140.88 W/m²` 在目前 artifact assumption 下成立。
5. 需要在 limitation 揭露 GFS 12Z D-1 dissemination latency caveat。
6. 若嚴格採 6-hour NWP availability latency，必須重建 DA features、重訓 DA forecasting、重建 scenario package、重跑 DA scheduling / MPC / M8。
7. 現在不建議重跑 06Z-only / latency-aware DA model，除非 advisor 要求把論文 claim 改成嚴格 operational NWP latency benchmark。

### 4.3 建議文字

**Chapter 3 methodology**

> In the implemented day-ahead setting, the forecast package is frozen at `D-1 20:00` local time (`D-1 12:00 UTC`). The day-ahead optimizer uses only the forecast features and scenario package available under this repository-level gate-time convention; same-day measured GHI and target-day realized PV information are excluded from the day-ahead forecast construction.

**Abstract 避免寫死時間**

> The proposed framework combines day-ahead probabilistic planning with intraday receding-horizon adjustment under a fixed pre-dispatch information boundary.

**Limitation caveat**

> The present implementation treats the `D-1 12Z` NWP initialization as available at the repository's `D-1 20:00` local day-ahead gate. A stricter operational dissemination-latency model could require using earlier NWP cycles such as 06Z, which would necessitate rebuilding the day-ahead forecast, scenario package, and downstream scheduling results.

## 5. DA CVaR Investigation

### 5.1 結論

`DA-Prob / M2` 實際上使用 CVaR。此 CVaR 是針對 scenario-dependent over-contract / peak-risk cost 的 tail-risk term，不是單純 expected-cost probabilistic DA MILP。

因此，handover 或 case mapping 若寫 `DA-Prob = No CVaR`，應視為文字錯誤或命名不精確。

### 5.2 證據

程式證據：

- `milp_v2/run_phase3b.py:37` import `layer_b.milp_cvar.solve_day_ahead`。
- `milp_v2/run_phase3b.py:56-57` 設定 `CASE_LAM = {"M1": 0, "M2": 1, ...}` 與 `ALPHA = 0.90`。
- `milp_v2/run_phase3b.py:271-282` 在 `run_da_case()` 將 `lam=lam, alpha=alpha` 傳入 solver。
- `milp_v2/run_phase3b.py:890-894` 正式 `M2` 使用 `layerB_prob_package_x.parquet` 並傳 `lam=CASE_LAM["M2"]`。
- `milp_v2/layer_b/milp_cvar.py:1-10` 檔頭說明 CVaR extension，`lam=0` reduces to expected-cost model。
- `milp_v2/layer_b/milp_cvar.py:132-136` 建立 CVaR variables `eta`、`xi`。
- `milp_v2/layer_b/milp_cvar.py:211-216` 建立 `xi[s] >= C_oc_s - eta`。
- `milp_v2/layer_b/milp_cvar.py:228-243` 計算 expected over-contract cost 與 CVaR。
- `milp_v2/layer_b/milp_cvar.py:245` objective 為 `C_ene + C_deg + C_peak_exp + lam * C_cvar`。

Artifact 證據：

- `milp_v2/layer_b/results/phase3b_M2_da_daily.parquet` 包含 `cvar_val` 欄位。
- 該 artifact 中 `cvar_val` 有非零值；365 天中 116 天非零，平均約 `64,494 NTD/day`，最大約 `324,413 NTD/day`。

反證：

- 未找到正式 `phase3b_M2_lam0` 或 final aligned no-CVaR `DA-Prob` artifact。
- 舊 `PP` / `K5_base` artifacts 使用不同 case/sizing/cost convention，不可直接當正式 `M2 no-CVaR` ablation。

### 5.3 回答核心問題

| 問題 | 回答 |
|---|---|
| 正式 M2 是否真的使用 CVaR peak-risk term？ | 是。 |
| 是否存在另一份正式 no-CVaR DA-Prob run artifact？ | not confirmed；未找到與 final M2 同級的 no-CVaR artifact。 |
| `phase3b_M2_da_daily.parquet` 是否有非零 `cvar_val`？ | 是。 |
| `lam=1.0` 是否真的進入 objective？ | 是，見 `milp_cvar.py:245`。 |
| `lam=1.0` 是否可能只是保留欄位？ | 否；程式將 `lam * C_cvar` 加入 objective。 |
| case name 是否應改？ | 可保留 `DA-Prob`，但方法文字必須寫 `with CVaR peak-risk treatment`；若要更精確，可稱 `DA-Prob-Risk-Aware`。 |

### 5.4 設計邏輯判斷

DA layer 使用 CVaR 的合理解釋是：

- DA plan 是隔日前固定的 master plan，對 monthly peak 與 over-contract penalty 的長期風險很敏感。
- PV scenario uncertainty 會造成 tail scenarios 下的 grid import peak 上升。
- CVaR term 懲罰的是 scenario-dependent over-contract cost 的 tail risk，而不是總電費或 energy cost 的 tail risk。
- ID MPC main case 不使用 CVaR，是因為 H24 MPC 已每小時重算，且 `MPC-Prob-H24` 在 formal main table 中採 risk-neutral expected-cost formulation；CVaR MPC 可作 diagnostic，而不是主方法。

這樣設計不會讓 M8 novelty 消失。DA CVaR 使 master plan 較有 peak discipline，但 M8 的 novelty 在於 online arbitration：用 DA master 的 peak/SOC discipline 約束 MPC candidate 的 intraday flexibility。M8 的改善不是單純 DA conservative planning，而是 DA master 與 ID candidate 的安全篩選與 blending。

### 5.5 是否需要重跑 no-CVaR

建議：

- 若論文願意承認 `DA-Prob` 含 CVaR：不必重跑。
- 若論文必須保留 `DA-Prob = No CVaR` 敘述：必須重跑 `M2 lam=0`，並重新產生 DA replay、M8 master plan、M8 arbitration result。

最小 rerun scope：

1. 以 `layerB_prob_package_x.parquet` 跑 `solve_day_ahead(lam=0, alpha=0.90)`。
2. 輸出 no-CVaR DA daily/hourly/replay files。
3. 以 no-CVaR DA master 重跑 M8 H24 arbitration。
4. 用 corrected cost accounting 產生 full-year total。

## 6. DA Scenario Investigation

### 6.1 結論

正式 `DA-Prob / M2` 使用的是 `five reduced scenarios plus one synthetic miss scenario`，不是單純 5 scenarios。

正式 package：

- `milp_v2/bridge/packages/layerB_prob_package_x.parquet`

實際內容：

- 365 days。
- 每日 6 scenarios。
- `scenario_id = s0, s1, s2, s3, s4, synth_miss`。
- `synth_miss` 固定 weight 約 `0.0368`。
- 所有 scenario probabilities 每日總和為 1。

### 6.2 synth_miss 的意義

`synth_miss` 是低 PV / forecast miss synthetic scenario，用來補足 calibrated DA quantile/scenario package 在低 PV tail 的 coverage deficit。它不應被描述為第六個自然 cluster，而應描述為 one synthetic low-PV miss scenario added to the five reduced scenarios。

證據：

- `milp_v2/config.yaml:72-90` 設定 `method_a`、`pi_synth=0.0368`、`pv_synth_ratio=0.6059`，並說明與 PIT left-tail coverage deficit / da_v2 recalibration 有關。
- `milp_v2/config.yaml:89-90` 也註記 `pv_synth_ratio` 使用 test-period K5 scenarios as proxy 的 information-boundary caveat。

### 6.3 權重是否納入 objective

有。`run_phase3b.py:240-242` 會檢查並 normalize probabilities；`milp_cvar.py:218-245` 使用 `pi_s[s]` 計算 expected cost 與 CVaR。因此 `synth_miss` 會進入 expected cost 與 CVaR tail-risk calculation。

### 6.4 論文建議文字

短版：

> The DA probabilistic optimizer uses five reduced PV scenarios plus one synthetic low-PV miss scenario. Scenario probabilities are normalized daily and are used in both the expected over-contract cost and the CVaR peak-risk term.

正式版：

> The day-ahead probabilistic scenario package consists of five reduced PV trajectories (`s0`-`s4`) and one synthetic low-PV miss scenario (`synth_miss`). The synthetic scenario is introduced to represent residual low-generation forecast miss risk identified during calibration. All six scenario probabilities are normalized at the daily level and enter both the expected peak-related cost and the CVaR tail-risk term in the day-ahead MILP.

## 7. ID Terminal SOC and DA D_ref Investigation

### 7.1 H24 main case 是否有 terminal SOC anchor

結論：沒有。

正式 `MPC-Det-H24` 與 `MPC-Prob-H24`：

- 沒有 terminal SOC hard band。
- 沒有 terminal SOC soft penalty。
- 沒有 anchor 到 DA SOC trajectory。
- `M8` 的 F2 SOC corridor 是 MILP 外的 arbitration filter，不是 ID MPC constraint。

證據：

- `milp_v2/layer_b/milp_mpc.py:181-186` terminal SOC band 只在 `is_last_hour` 為 true 時加上。
- `milp_v2/experiments/mpc_fixed_rolling_horizon/mpc_fixed_horizon_utils.py:395` H 由 `fixed_24h` config 決定。
- `milp_v2/experiments/mpc_fixed_rolling_horizon/mpc_fixed_horizon_utils.py:403-404` `is_last = (cfg_h.horizon_mode == "day_bounded") and (h == 23) and is_final_day`。
- `run_mpc_fixed_horizon_fullyear.py:69-84` 正式 H24 variants 使用 `fixed_24h`，不是 `day_bounded`。
- `milp_v2/layer_b/milp_mpc.py:320-354` objective 無 terminal SOC soft penalty；只有 expected cost、degradation、CVaR、optional M6/M7 penalties。

因此，main H24 case 中 `is_last_hour=True` 基本不會發生；terminal SOC band 不會啟用。

### 7.2 沒有 terminal anchor 的可能影響

沒有 terminal SOC anchor 時，MPC 每小時只看未來 24 小時與當前 month-to-date peak state，可能為了短期 energy / peak trade-off 在 horizon end 留下偏低或偏高 SOC。這會使 pure MPC 在某些月份缺乏 DA master 的長期 SOC discipline。

這與結果方向一致：`MPC-Prob-H24` 的 TOU 較低，但 OC cost 高於 `DA-Prob`：

- `DA-Prob`: TOU 76.0648, OC 2.2498 M NTD。
- `MPC-Prob-H24`: TOU 75.2665, OC 3.7720 M NTD。

但不能把 OC 較高全部歸因於 terminal SOC；也包含 pure MPC 未直接使用 DA monthly reference、scenario horizon treatment、rolling peak proxy 等因素。

### 7.3 M8 F2 如何補這件事

M8 在 MILP 外執行。它先形成 blended candidate：

> `B_cand = B_M2 + alpha * (B_M5 - B_M2)`

然後用 F2 檢查執行後 SOC 是否落在 DA SOC corridor 內。這不是 MPC constraint，而是 acceptance filter。若 F2 fail，M8 fallback to DA action。

此設計合理處在於：

- DA master 提供 full-day / monthly peak discipline。
- MPC candidate 提供 intraday flexibility。
- M8 不重解 MILP，而是在執行層做安全篩選。

限制是：它不是一個 fully integrated DA-aware MPC；若口委問為什麼不直接把 DA SOC 放入 MPC，應回答該版本屬於 reference-guided MPC，已在 M6/M7 diagnostics 顯示可能犧牲 intraday flexibility，故未作主方法。

### 7.4 若論文寫「ID MPC anchor to DA SOC」是否錯誤

是。正確寫法應為：

> The formal H24 MPC baselines do not impose a terminal SOC anchor to the DA trajectory. Instead, DA SOC discipline is introduced only in the M8 arbitration layer through an external SOC corridor filter around the DA master trajectory.

## 8. ID MPC and DA D_ref Investigation

### 8.1 H24 MPC 是否直接使用 DA D_ref

結論：沒有。

`MPC-Prob-H24` objective / constraints 不使用 DA `D_ref`。它使用的是 current month-to-date peak state `D_init` / `peak_demand_running`，用來計算 rolling peak proxy 與 incremental over-contract cost。

證據：

- `milp_v2/layer_b/milp_mpc.py:41` solver argument 包含 `peak_demand_running`。
- `milp_v2/layer_b/milp_mpc.py:107` 設定 `D_init`。
- `milp_v2/layer_b/milp_mpc.py:157` `D_cand` lower bound 與 contract capacity / running peak 有關。
- `milp_v2/layer_b/milp_mpc.py:330` over-contract objective 扣除 previous peak cost。
- `milp_v2/layer_b/milp_mpc.py:52-73` 的 M6/M7 reference arguments 是 optional，不是 main H24 case。

### 8.2 DA D_ref 來源與 M8 使用方式

DA `D_ref` 由 M2 replay daily monthly peak reference 產生。

證據：

- `milp_v2/experiments/m8_m2_safe_no_regret_recourse/m8_utils.py:116-120` `load_m2_d_ref_month()` 從 `phase3b_M2_replay_daily.parquet` 取得 monthly max `D_mth_end`。
- `m8_utils.py:210-216` F1 使用 `D_ref_month + eps_peak` 作為 monthly peak cap。

因此，DA `D_ref` 在 final main method 中是 M8 F1 filter 的 reference，不是 ID MPC MILP 的 constraint/objective term。

### 8.3 為什麼不直接放進 MPC

已存在 reference-guided MPC diagnostics，但未成為主方法：

- `M6_REPORT_2026-05-25.md:61-80` 說明 M6 optional `m6b_grid_ref`、`m6c_d_ref`。
- `M6_REPORT_2026-05-25.md:158-168` 顯示 M6 reference tracking 降低 OC 但提高 TOU，總成本變差。
- `BH_THESIS_HANDOVER_2026_05_FINAL/06_ROBUSTNESS_AND_DIAGNOSTIC_SUMMARIES/M6_M7_DIAGNOSTIC_SUMMARY.md:18-27` 說明 M6 reference tracking over-constrains intraday flexibility。
- `BH_THESIS_HANDOVER_2026_05_FINAL/06_ROBUSTNESS_AND_DIAGNOSTIC_SUMMARIES/M6_M7_DIAGNOSTIC_SUMMARY.md:35-44` 說明 M7 local monthly diagnostics 不是 final full-year result。

設計判斷：

- 把 DA `D_ref` 直接放入 MPC 會把 reference discipline 硬塞進每小時 optimizer，可能犧牲 intraday correction capability。
- M8 把 DA `D_ref` 放在 filter 層，保留 MPC candidate 的候選彈性，但只接受不破壞 DA peak discipline 的行動。
- 這是合理設計，但必須承認不是 mathematically integrated MPC。

口試安全答法：

> We tested reference-guided MPC variants, but direct reference tracking tended to reduce the flexibility that intraday MPC was meant to provide. The final M8 design therefore keeps the MPC optimizer risk-neutral and flexible, while applying the DA peak reference as an external safety filter. This separates candidate generation from safety arbitration.

## 9. M8 Logic and Approximation Investigation

### 9.1 正式 M8 實作

M8 不是 MILP / MPC。它是 MILP 外的 online arbitration / replay layer。

Formal `M8-Prob-H24`：

- Master: `DA-Prob / M2`。
- Candidate: `MPC-Prob-H24 / M5_H24_prob_lam0`。
- Candidate action 不是直接使用 M5 action，而是 blended candidate。
- Blend formula:

```text
B_cand = B_M2 + alpha * (B_M5 - B_M2)
```

主設定：

- `alpha = 0.25`
- SOC band = 15%
- cost tolerance = 100 NTD
- charge headroom = 300 kW
- FY2 configuration

證據：

- `milp_v2/experiments/mpc_fixed_rolling_horizon/run_m8_h24_standalone.py:3-11` 說明 H24 M8 使用 M5_H24 candidate + M2 master + filters。
- `run_m8_h24_standalone.py:50-55` 設定 FY1/FY2 configs；FY2 使用 SOC band 15%、alpha 0.25、cost tolerance 100、headroom 300。
- `milp_v2/experiments/m8_m2_safe_no_regret_recourse/m8_utils.py:49-68` 定義 M8Config defaults。
- `m8_utils.py:140-169` 定義 `blend_actions()`。
- `m8_utils.py:174-297` 定義 F1-F4 safety filter 與 fallback logic。
- `run_m8_det_h24.py:38-47` deterministic FY2 setting 同樣使用 alpha 0.25、SOC band 15%、tolerance 100、headroom 300。

### 9.2 F1-F4 filter

| Filter | 實作意義 | 是否 MILP constraint |
|---|---|---|
| F1 | monthly peak cap relative to DA `D_ref` | no |
| F2 | SOC corridor around DA SOC trajectory | no |
| F3 | immediate TOU cost consistency / relaxed no-regret cost check | no |
| F4 | charge headroom check | no |

任一 filter fail，fallback to DA action；全部 pass，execute blended candidate。

### 9.3 為什麼使用 blended candidate

直接接受 MPC candidate 可能使 action 大幅偏離 DA master，破壞 DA 的 peak/SOC discipline。Blending 讓 candidate correction 有漸進性：

- `alpha=0` 等同 DA master。
- `alpha=1` 等同直接 MPC candidate。
- `alpha=0.25` 是 partial recourse，降低接受候選時的 SOC / peak shock。

這不是 pure optimal control，而是安全 recourse heuristic。論文中應稱為 arbitration / safety-filtered recourse，不宜稱為新的 MILP formulation。

### 9.4 FY1 / FY2 / FY3 與 post-hoc tuning risk

現有 evidence 顯示：

- FY1 使用較窄 SOC band 10%，較接近 pre-specified conservative setting。
- FY2 使用 SOC band 15%，為正式主結果，改善較大。
- Core_A / Core_B / FY1 / FY2 ablation 顯示 F1/F2 已有貢獻，F3/F4 提供額外安全與成本一致性檢查。

證據：

- `BH_THESIS_HANDOVER_2026_05_FINAL/06_ROBUSTNESS_AND_DIAGNOSTIC_SUMMARIES/M8_CORE_ABLATION_SUMMARY.md:34-37`
- `BH_THESIS_HANDOVER_2026_05_FINAL/06_ROBUSTNESS_AND_DIAGNOSTIC_SUMMARIES/M8_CORE_ABLATION_REVIEWER_DEFENSE_TABLE.md:8-16`
- `BH_THESIS_HANDOVER_2026_05_FINAL/06_ROBUSTNESS_AND_DIAGNOSTIC_SUMMARIES/M8_CORE_ABLATION_REVIEWER_DEFENSE_TABLE.md:32-40`
- `M8_POSTHOC_AND_PARAMETER_AUDIT.md:116-120`

風險判斷：

- FY2 有 parameter selection / post-hoc tuning risk，應在 robustness section 中揭露。
- 但 Core_A / FY1 也有正向改善，可降低「只靠硬調參數」的疑慮。

### 9.5 Realized PV/load approximation

M8 filter replay 使用 current-hour realized PV/load 進行 safety checks。這是最大 information-boundary caveat。

證據：

- `milp_v2/experiments/m8_m2_safe_no_regret_recourse/m8_utils.py:1-14` 檔頭明示 M8 pure re-simulation，使用 realized PV/load in safety filter as disclosed simplification。
- `M8_INFORMATION_BOUNDARY_AUDIT.md:67-83` 指出 F1/F3/F4 使用 current-hour realized PV/load 是最重要問題。
- `M8_INFORMATION_BOUNDARY_AUDIT.md:152-193` 回答此為 mild leakage / near-perfect one-hour forecast assumption，需揭露。
- `M8_INFORMATION_BOUNDARY_AUDIT.md:196-213` 提供 disclosure 與替代方案。

論文應寫：

> In the replay implementation, M8 uses current-hour realized PV/load in the safety filter to evaluate the immediate feasibility and cost consistency of a candidate action. This should be interpreted as a near-real-time or perfect one-step measurement approximation. A deployment-clean implementation would replace this with nowcasted current-hour PV/load or re-run the filter using only information available at the decision timestamp.

### 9.6 M8 能否稱為 no-regret

不應使用無限定的 `no-regret`。可使用：

- `safety-filtered recourse`
- `DA-anchored arbitration`
- `replay no-regret against the DA action under the implemented filter`
- `limited no-regret acceptance rule`

安全說法：

> M8 is no-regret only in the limited replay/filter sense: when a candidate fails the predefined safety filters, the controller falls back to the DA action. It is not a formal online-learning no-regret guarantee and should not be described as a globally optimal no-regret controller.

## 10. Overall Method Self-Consistency Review

### 10.1 審稿風險表

| Risk | Evidence | Severity | Recommended fix | Rerun required |
|---|---|---:|---|---|
| `DA-Prob` 被稱為 `No CVaR` 但 code 使用 CVaR | `run_phase3b.py:56-57`, `milp_cvar.py:245`, `phase3b_M2_da_daily.parquet` | high | 修改命名/方法描述，承認 CVaR peak-risk treatment | no, unless insisting no-CVaR claim |
| DA gate 寫成 16:00 | `DA_GATE_TIME_AUDIT.md:3-15` | high | 改成 20:00 local / 12:00 UTC；Abstract 避免寫死 | no |
| NWP latency caveat | `DA_GATE_TIME_AUDIT.md:71-84` | medium-high | limitation 揭露 12Z availability assumption | no, unless strict latency benchmark |
| DA scenarios 寫成 K5 only | scenario package actual 6 IDs | medium | 改成 K5 plus synthetic miss | no |
| H24 MPC 被描述為 anchor to DA SOC | `mpc_fixed_horizon_utils.py:403-404`, `milp_mpc.py:181-186` | medium | 改為 M8 F2 external SOC corridor | no |
| H24 MPC 被描述為 using DA `D_ref` | `milp_mpc.py` main args/objective；M8 F1 uses D_ref | medium | 改為 DA `D_ref` used in M8 F1 only | no |
| M8 realized current-hour PV/load | `M8_INFORMATION_BOUNDARY_AUDIT.md:67-83` | high | limitation + optional robustness replay | nice-to-have |
| M8 FY2 parameter tuning | M8 parameter audit / core ablation | medium | 報告 FY1/Core_A robustness；避免過度宣稱 | no |
| Deterministic vs probabilistic fairness | Different forecast uncertainty input | medium | 表格分 deterministic/probabilistic tracks；同一 settlement formula | no |
| Cost accounting consistency | final aligned table and corrected KPI | low if corrected table used | 只引用 corrected full total | no |
| Load uncertainty not modeled | MPC utils use realized/perfect load lookup | medium | limitation：load treated deterministic/perfect | no |

### 10.2 對主結論的支撐

目前結果可支撐 thesis contribution，但主線必須調整為：

- Forecasting / scheduling contribution 不是 forecasting SOTA。
- DA probabilistic planning 提供 risk-aware peak discipline。
- ID MPC 單獨使用時可降低 TOU，但可能增加 OC risk。
- M8 透過 DA-anchored arbitration 接受部分 ID correction，同時避免破壞 DA peak/SOC discipline。
- 最終改善來自 probabilistic forecast、DA risk-aware master、ID recourse candidate 與 safety arbitration 的組合，而不是單一因素。

## 11. Rerun Recommendation Matrix

| Experiment | Classification | Reason | Runtime / implementation risk | Main conclusion risk |
|---|---|---|---|---|
| `DA-Prob without CVaR / lam=0` | Nice-to-have; conditional must if claiming `No CVaR` | 可分離 CVaR 對 M2/M8 的貢獻 | medium；需 365 daily DA MILPs + replay；若更新 M8 還需 arbitration rerun | medium-high |
| `DA-Prob K5 only without synth_miss` | Nice-to-have / appendix | 可檢查 synthetic miss 對 peak risk 的影響 | medium | medium |
| `MPC-Prob-H24 with DA D_ref hard/soft constraint` | Future work / not needed | M6/M7 已顯示 reference-guided MPC 可能變差 | high | low-medium |
| `MPC-Prob-H24 with terminal SOC anchor to DA` | Future work / nice-to-have | 可測 terminal discipline，但屬新 controller | high | medium |
| `M8 using forecasted current-hour PV/load` | Nice-to-have, strongest robustness if time permits | 回應 M8 information-boundary caveat | low-medium if replay only；需構造 nowcast/forecast lookup | medium |
| `M8 alpha sensitivity` | Nice-to-have | 回應 post-hoc tuning risk | low if pure replay | low-medium |
| `M8 FY1/FY2/FY3 sensitivity` | Not needed; already partly available | existing ablation supports FY variants | low | low |
| `M8 direct MPC candidate instead of blended` | Nice-to-have | 檢查 blending necessity | low-medium | low-medium |
| `M8 F1/F2 only vs F1/F2/F3/F4` | Not needed; already available in core ablation | Core_A/B/FY results exist | no | low |
| `06Z-only / latency-aware DA forecasting and downstream scheduling` | Future work / limitation only; must only if strict latency claim | 會重建 forecasting-to-scheduling 全鏈 | very high | high if thesis claims strict operational NWP latency |

## 12. Thesis Wording Recommendations

### 12.1 One-paragraph pipeline summary

> This study evaluates a PV-BESS scheduling pipeline that combines day-ahead probabilistic planning, intraday receding-horizon control, and a DA-anchored arbitration layer. The day-ahead layer produces a risk-aware master schedule from deterministic or probabilistic GHI/PV forecasts. The intraday MPC layer updates battery actions using short-horizon forecast revisions. The final M8 controller does not solve an additional MILP; instead, it blends the DA master action with the H24 MPC candidate and accepts the candidate only when predefined peak, SOC, cost-consistency, and charging-headroom filters are satisfied.

### 12.2 DA-Prob model description

> The formal `DA-Prob` case uses a probabilistic day-ahead MILP with five reduced PV scenarios plus one synthetic low-PV miss scenario. Scenario probabilities are normalized daily. The objective includes expected energy cost, battery degradation cost, expected over-contract cost, and a CVaR peak-risk term applied to the scenario-dependent over-contract cost. Therefore, `DA-Prob` should be interpreted as a probabilistic risk-aware DA master plan rather than a no-CVaR expected-cost-only model.

### 12.3 ID MPC model description

> The formal H24 MPC cases solve a rolling 24-hour MILP every hour. The first six lead hours use intraday forecast information, while later horizon slots are completed with the day-ahead tail or persistence logic implemented in the forecast package builder. The MPC state is initialized from the executed SOC and the current month-to-date peak. The main H24 MPC baselines do not impose a terminal SOC anchor to the DA trajectory and do not directly constrain the solution by the DA monthly peak reference.

### 12.4 M8 arbitration description

> M8 is a safety-filtered arbitration layer between the DA master plan and the H24 MPC candidate. At each hour, it forms a blended battery action from the DA action and the MPC action, then evaluates four filters: a DA monthly peak reference filter, a DA SOC corridor filter, an immediate cost-consistency filter, and a charging-headroom filter. If any filter fails, M8 executes the DA action; otherwise it executes the blended candidate.

### 12.5 Forecast information boundary

> The implemented DA forecast package is frozen at `D-1 20:00` local time (`D-1 12:00 UTC`) under the repository's current NWP availability convention. The thesis does not claim a strict 16:00 operational bidding cutoff implementation. A stricter dissemination-latency treatment of NWP products is left as future work.

### 12.6 Cost accounting

> All final scheduling cases are compared using the same corrected replay settlement accounting, including TOU energy cost, over-contract charge, battery degradation cost, TREC purchase cost, CAPEX, and basic contract cost. Earlier outputs that treated TREC as revenue or used the deprecated `compute_kpis()` totals are not used as formal results.

### 12.7 Limitation paragraph

> Two implementation limitations should be noted. First, the DA forecast package assumes availability of the `D-1 12Z` NWP cycle at the repository's `D-1 20:00` local gate; stricter NWP dissemination latency would require rebuilding the DA forecast and downstream scheduling chain. Second, the M8 replay filter uses current-hour realized PV/load for immediate safety checks, which approximates a near-real-time measurement or perfect one-step nowcast. A deployment-clean controller should replace this with decision-time available measurements or nowcasts.

## 13. Oral Defense Q&A

**Q: Why does DA use CVaR?**  
A: DA decisions are fixed before the operating day and are exposed to PV scenario tail risk. The CVaR term penalizes tail over-contract cost under low-PV or high-grid-import scenarios, giving the DA master stronger peak discipline.

**Q: Why does ID MPC not directly use DA `D_ref`?**  
A: The main H24 MPC is kept as a flexible receding-horizon optimizer using current SOC and month-to-date peak state. Direct DA reference tracking was tested in M6/M7 diagnostics and tended to reduce intraday flexibility or worsen total cost. The final method therefore applies DA `D_ref` in M8 as an external safety filter instead of a hard MPC constraint.

**Q: Why no terminal SOC anchor in ID MPC?**  
A: The formal H24 MPC baseline intentionally uses a rolling 24-hour horizon without DA terminal anchoring. DA SOC discipline is introduced only in M8 through the F2 SOC corridor. Adding a terminal anchor would define a different DA-aware MPC variant and is appropriate as future work or ablation, not as the current main result.

**Q: Is M8 just cherry-picking the better action?**  
A: No. M8 uses a fixed rule: it forms a pre-specified blended candidate and accepts it only if all filters pass; otherwise it falls back to DA. It does not compare realized annual costs hour-by-hour or choose the ex-post best case. However, the replay uses current-hour realized PV/load in safety checks, so this approximation must be disclosed.

**Q: Can M8 be called no-regret?**  
A: Only in a limited filter/replay sense: the candidate is accepted only when it passes predefined safety checks relative to the DA action/reference. It is not a formal online-learning no-regret guarantee. A safer term is `safety-filtered DA-anchored recourse`.

**Q: Why not directly design DA-aware MPC?**  
A: That is a valid future-work direction. The current project tested reference-guided variants, but direct reference tracking increased rigidity and did not become the best full-year method. M8 separates candidate generation from safety arbitration, preserving MPC flexibility while retaining DA peak/SOC discipline.

**Q: Does realized PV/load in M8 cause unfairness?**  
A: It is a limitation. It should be interpreted as a near-real-time or perfect one-step measurement approximation. The main result is useful for method evaluation, but a deployment-clean version should use only timestamp-available measurements or forecasts.

## 14. Appendix: Evidence Table

| Topic | Evidence path | Line / artifact | Finding |
|---|---|---|---|
| M1/M2 solver import | `milp_v2/run_phase3b.py` | `:37` | imports `layer_b.milp_cvar.solve_day_ahead` |
| M1/M2 lambda | `milp_v2/run_phase3b.py` | `:56-57` | `M1=0`, `M2=1`, `ALPHA=0.90` |
| M2 dispatch | `milp_v2/run_phase3b.py` | `:890-894` | uses `layerB_prob_package_x.parquet`, `lam=CASE_LAM["M2"]` |
| DA CVaR variables | `milp_v2/layer_b/milp_cvar.py` | `:132-136` | defines `eta`, `xi` |
| DA CVaR constraints | `milp_v2/layer_b/milp_cvar.py` | `:211-216` | `xi >= C_oc_s - eta` |
| DA objective | `milp_v2/layer_b/milp_cvar.py` | `:218-245` | `C_ene + C_deg + C_peak_exp + lam * C_cvar` |
| M2 CVaR artifact | `milp_v2/layer_b/results/phase3b_M2_da_daily.parquet` | columns `cvar_val`, `E_coc_val` | `cvar_val` nonzero in formal M2 |
| Formal M2 scenario package | `milp_v2/bridge/packages/layerB_prob_package_x.parquet` | parquet content | 6 scenarios/day: `s0`-`s4`, `synth_miss` |
| synth_miss config | `milp_v2/config.yaml` | `:72-90` | `pi_synth=0.0368`, low-PV miss construction |
| DA gate conclusion | `DA_GATE_TIME_AUDIT.md` | `:3-15` | formal gate `D-1 20:00 local / D-1 12:00 UTC` |
| NWP latency caveat | `DA_GATE_TIME_AUDIT.md` | `:71-84` | strict latency may require rebuild |
| H24 variants | `milp_v2/experiments/mpc_fixed_rolling_horizon/run_mpc_fixed_horizon_fullyear.py` | `:69-84` | defines `M3_H24_det`, `M5_H24_prob_lam0` |
| MPC terminal band | `milp_v2/layer_b/milp_mpc.py` | `:181-186` | only if `is_last_hour` |
| H24 `is_last` logic | `milp_v2/experiments/mpc_fixed_rolling_horizon/mpc_fixed_horizon_utils.py` | `:403-404` | true only for `day_bounded`, not `fixed_24h` |
| MPC objective | `milp_v2/layer_b/milp_mpc.py` | `:320-354` | no terminal SOC soft penalty in main objective |
| MPC current peak state | `milp_v2/layer_b/milp_mpc.py` | `:41`, `:107`, `:157`, `:330` | uses month-to-date peak state, not DA `D_ref` |
| M6 reference optional | `milp_v2/layer_b/milp_mpc.py` | `:52-73` | reference-guided args optional, not main H24 |
| M8 H24 runner | `milp_v2/experiments/mpc_fixed_rolling_horizon/run_m8_h24_standalone.py` | `:3-11` | M8 H24 uses M2 master + M5 H24 candidate |
| M8 FY2 config | `run_m8_h24_standalone.py` | `:50-55` | alpha 0.25, SOC band 15%, tolerance 100, headroom 300 |
| M8 blend | `milp_v2/experiments/m8_m2_safe_no_regret_recourse/m8_utils.py` | `:140-169` | `B_M2 + alpha*(B_M5-B_M2)` |
| M8 filters | `m8_utils.py` | `:174-297` | F1-F4 and fallback/pass logic |
| M8 D_ref | `m8_utils.py` | `:116-120`, `:210-216` | loads M2 monthly `D_ref`, applies F1 |
| M8 realized PV/load caveat | `M8_INFORMATION_BOUNDARY_AUDIT.md` | `:67-83`, `:152-193` | current-hour realized PV/load used in filter replay |
| Final H24 costs | `milp_v2/experiments/mpc_fixed_rolling_horizon/results/H24_cost_component_raw.csv` | CSV rows | corrected full totals |
| Final thesis table | `milp_v2/experiments/mpc_fixed_rolling_horizon/results/final_thesis_package/FINAL_MAIN_COMPARISON_H24_ALIGNED.csv` | CSV rows | official main comparison |
| Obsolete costs | `BH_THESIS_HANDOVER_2026_05_FINAL/ARCHIVE_NOTES_AND_OBSOLETE_NUMBERS.md` | `:28-30`, `:64-69`, `:90-106` | old 94.x, TREC revenue, old `compute_kpis()` obsolete |

## 15. Manual Confirmation Needed

目前不需要 user 提供額外檔案才能定稿主要方法線。不過若要做 deployment-clean extension，需確認：

1. GFS / NWP 實際 dissemination latency policy 是否要以 0-hour、3-hour 或 6-hour latency 寫入論文。
2. M8 current-hour filter 在實際部署時可取得的 measurement/nowcast timing。
3. 是否要把 `DA-Prob` case name 改成 `DA-Prob-Risk-Aware`，或保留 `DA-Prob` 並在方法文字中說明 CVaR。
4. 是否要在口試前補一個 `DA-Prob lam=0` ablation 或 `M8 forecasted-current-hour` robustness replay。
