# Day-ahead MILP 與 Intraday MPC MILP 實作模型稽核

產出日期：2026-05-30  
範圍：BH 太陽能 / PV-BESS scheduling thesis 專案目前可追溯的正式 `milp_v2` 排程主線。  
寫作原則：以下模型只根據程式、保存 artifact、handover 與結果表交叉確認；無法由程式確認者標示為 `not confirmed`。

---

## 1. Executive Summary

目前正式 scheduling 主線不是舊版 `pipeline_standalone` 或 `notebooks_milp`，而是 `milp_v2`：

- Day-ahead cases `DA-Det` / `DA-Prob` 由 `milp_v2/run_phase3b.py` 呼叫 `milp_v2/layer_b/milp_cvar.py::solve_day_ahead()` 產生，而非單獨以 `milp_v2/layer_b/milp_daily.py` 為準。證據：`run_phase3b.py:37`, `run_phase3b.py:250-282`, `run_phase3b.py:884-894`。
- Intraday H24 cases `MPC-Det-H24` / `MPC-Prob-H24` 由 `milp_v2/experiments/mpc_fixed_rolling_horizon/run_mpc_fixed_horizon_fullyear.py` 與 `mpc_fixed_horizon_utils.py` 呼叫 `milp_v2/layer_b/milp_mpc.py::solve_mpc_milp()`。證據：`run_mpc_fixed_horizon_fullyear.py:69-84`, `mpc_fixed_horizon_utils.py:397-424`。
- `M8-Prob-H24` 不是 MILP / MPC；它是 MILP 外的 no-regret arbitration / safety-filtered recourse，使用 `DA-Prob (M2)` 作 master plan，使用 `MPC-Prob-H24 (M5_H24_prob_lam0)` 的 first-stage action 作 candidate。證據：`run_m8_h24_standalone.py:3-11`, `run_m8_h24_standalone.py:85-112`, `m8_utils.py:1-14`。
- 最終成本表應使用修正後公式 `CAPEX + Basic + TOU + OC + Deg + TREC_purchase`，不可使用 `run_phase3b.py::compute_kpis()` 內仍存在的舊 TREC revenue 公式。權威數字來源是 `milp_v2/experiments/mpc_fixed_rolling_horizon/results/H24_cost_component_raw.csv` 與 `.../final_thesis_package/FINAL_MAIN_COMPARISON_H24_ALIGNED.csv`。證據：`mpc_fixed_horizon_utils.py:595-655`, `BH_THESIS_HANDOVER_2026_05_FINAL/07_SCRIPT_OUTPUT_INDEX.md:11-22`, `H24_cost_component_raw.csv`。

高風險不一致：

- `DA-Prob (M2)` 的實際程式路徑使用 `milp_cvar.py` 並以 `lam=1.0, alpha=0.90` 求解，且保存的 `phase3b_M2_da_daily.parquet` 有非零 `cvar_val`。這與 `BH_THESIS_HANDOVER_2026_05_FINAL/03_FINAL_CASE_DEFINITIONS_AND_NAMING.md:17-18` 將 `DA-Prob` 標示為 `No CVaR` 不一致。若論文要寫「DA-Prob 無 CVaR」，必須人工確認是否曾以不同腳本重跑；以目前 repo 程式與 artifact 看，正式 M2 應描述為「probabilistic DA MILP with CVaR peak-risk term」或至少加 caveat。
- `DA-Prob` 不是純 5 scenario。正式輸入 `layerB_prob_package_x.parquet` 每日有 6 個 scenario：`s0`-`s4` 加 `synth_miss`。因此「5 reduced scenarios」應改寫為「five reduced DA PV scenarios plus one synthetic miss scenario」。證據：`run_phase3b.py:890-894` 使用 `layerB_prob_package_x.parquet`；實際 parquet 檢查 rows=52,560 = 365 days × 24 h × 6 scenarios。
- DA gate time 依先前稽核應寫成 `D-1 20:00 local / D-1 12:00 UTC`，不是 `D-1 16:00` 或 `D-1 22:00`。證據：`DA_GATE_TIME_AUDIT.md:3-15`, `DA_GATE_TIME_AUDIT.md:21-38`。

PDF caveat：本 repo 未找到 `0528thesis.pdf`；只找到 `607482528173129951_HARRY_thesis_wirhour_Rep_0331.pdf`。本機缺少 `pypdf`、`PyPDF2` 與 `pdftotext`，因此 Section 3.9 PDF 文字未能直接抽取確認。以下報告以程式與 handover/result artifacts 為主。

---

## 2. 正式 Case 命名、腳本、輸出與成本

### 2.1 主要 case 對照

| Thesis case | Internal ID | 正式角色 | Script / solver | 主要輸出 | Full total (M NTD) | 成本來源 |
|---|---|---|---|---|---:|---|
| DA-Det | `M1` | Day-ahead deterministic MILP | `milp_v2/run_phase3b.py:884-889` → `milp_cvar.solve_day_ahead()` with `lam=0` | `milp_v2/layer_b/results/phase3b_M1_da_daily.parquet`, `phase3b_M1_replay_hourly.parquet` | `~102.47 est.` | `FINAL_MAIN_COMPARISON_H24_ALIGNED.csv`; handover marks Deg/TREC estimated |
| DA-Prob | `M2` | Day-ahead probabilistic MILP | `milp_v2/run_phase3b.py:890-894` → `milp_cvar.solve_day_ahead()` with `lam=1` | `phase3b_M2_da_daily.parquet`, `phase3b_M2_replay_hourly.parquet` | `100.4855` | `H24_cost_component_raw.csv`, `FINAL_MAIN_COMPARISON_H24_ALIGNED.csv` |
| MPC-Det-H24 | `M3_H24_det` | Fixed 24h rolling deterministic MPC | `run_mpc_fixed_horizon_fullyear.py:69-74` → `solve_mpc_milp()` | `results/fullyear/M3_H24_det_hourly.parquet`, `...daily.parquet` | `104.5324` | `H24_cost_component_raw.csv` |
| MPC-Prob-H24 | `M5_H24_prob_lam0` | Fixed 24h rolling probabilistic MPC, risk-neutral | `run_mpc_fixed_horizon_fullyear.py:75-79` → `solve_mpc_milp()` with `lam=0`, RC conformal | `results/fullyear/M5_H24_prob_lam0_hourly.parquet`, `...daily.parquet` | `101.3338` | `H24_cost_component_raw.csv` |
| M8-Det-H24 | `M8_Det_H24_FY2` | Deterministic M8 ablation | `run_m8_det_h24.py:3-8`, `run_m8_det_h24.py:220-237` | `results/m8_det_h24/M8_Det_H24_kpis.json`, hourly/daily parquet | `101.6968` | `M8_Det_H24_kpis.json`, final table |
| M8-Prob-H24 | `M8_H24_FY2` | Proposed no-regret arbitration | `run_m8_h24_standalone.py:3-11`, `run_m8_h24_standalone.py:85-120` | `results/m8_h24/M8_H24_FY2_15pct/hourly.parquet`, daily parquet | `99.6088` | `H24_cost_component_raw.csv`, final table |

### 2.2 成本分解（修正後）

來源：`BH_THESIS_HANDOVER_2026_05_FINAL/04_FINAL_RESULT_TABLES/FINAL_MAIN_COMPARISON_H24_ALIGNED.csv` 與 `thesis_tables/Table_4-4-1_main_cost_comparison.csv`。

| Case | TOU | OC | Deg | TREC purchase | CAPEX | Basic | Full total |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA-Det | 75.1357 | 5.0032 | 1.9900* | 5.2650* | 7.3489 | 7.7470 | ~102.47* |
| DA-Prob | 76.0648 | 2.2498 | 1.8084 | 5.2666 | 7.3489 | 7.7470 | 100.4855 |
| MPC-Det-H24 | 74.6310 | 7.4222 | 2.1155 | 5.2679 | 7.3489 | 7.7470 | 104.5324 |
| MPC-Prob-H24 | 75.2665 | 3.7720 | 1.9583 | 5.2410 | 7.3489 | 7.7470 | 101.3338 |
| M8-Det-H24 | 74.2915 | 4.9520 | 2.0706 | 5.2868 | 7.3489 | 7.7470 | 101.6968 |
| M8-Prob-H24 | 75.2394 | 2.1368 | 1.8797 | 5.2570 | 7.3489 | 7.7470 | 99.6088 |

`*` 表示 final handover 標為估計值。  
注意：`milp_v2/experiments/mpc_fixed_rolling_horizon/results/m8_h24_results.csv` 仍含舊版未修正 TREC 的 `94.x M NTD` 數字，不可作正式成本引用。

---

## 3. Exact Implemented DA MILP

正式 DA solver：`milp_v2/layer_b/milp_cvar.py::solve_day_ahead()`。  
Formal runner：`milp_v2/run_phase3b.py::run_da_case()`。

### 3.1 Sets and indices

- $t \in T=\{0,\ldots,23\}$：24 小時日前時段。程式：`milp_cvar.py:91-94`。
- $w \in W=\{1,\ldots,K\}$：DA forecast scenarios。程式：`milp_cvar.py:87-94`。
- $k \in S$：degradation PWL segments。程式：`milp_cvar.py:78-80`。

正式 scenario 設定：

- `DA-Det (M1)` 使用 `layerB_det_package.parquet`，每日至少由 artifact 確認為 `K=1`。
- `DA-Prob (M2)` runner 使用 `layerB_prob_package_x.parquet`。實際 artifact 每日 `K=6`，scenario IDs 為 `s0, s1, s2, s3, s4, synth_miss`。這是「K=5 reduced scenarios + synthetic miss」，不是純 K=5。
- `config.yaml:68-79` 與 `run_phase3b.py:59-61` 仍寫 `K=5` / 5 quantile weights；這是文件/程式註解與正式 package 的不一致，需在論文避免寫成「exactly 5 scenarios」。

### 3.2 Parameters

固定設計參數來自 `milp_v2/layer_a/designs/design_BASE.json`：

- $CC=3306.45$ kW。
- $PB=1510.74$ kW。
- $EB=7525.83$ kWh。

電池與費率參數：

- $\eta_{ch}=0.95$, $\eta_{dis}=0.95$，SOC bounds $[0.10EB, 0.90EB]$，initial SOC fraction $0.50$。程式：`config.yaml:35-40`, `milp_cvar.py:72-77`。
- Terminal SOC tolerance $\epsilon=0.05$。程式：`config.yaml:40`, `milp_cvar.py:151-156`。
- Degradation breakpoints $b=[0,0.1,0.3,0.6,0.8]$ and marginal costs $\lambda=[0.97,1.61,2.58,3.87]$ NTD/kWh。程式：`config.yaml:42-47`, `milp_cvar.py:158-166`。
- Demand amplification $\kappa=1.006$。程式：`config.yaml:49-54`, `milp_cvar.py:82-85`。
- Over-contract tier: $O_1\le 0.10CC$, multipliers $m_1=2.0$, $m_2=3.0$。程式：`config.yaml:52-54`。
- Basic charge by month from `get_basic_charge()`：summer months 5-10 use 223.6 NTD/kW/month, otherwise 166.9。程式：`config.yaml:56-60`, `common.py:85-89`。
- TOU hourly price from `get_tou_price()`。程式：`common.py:43-82`。

Forecast / scenario inputs：

- 每個 day dictionary 包含 `scenarios: list of {id, pi, pv[24], load[24]}`。程式：`milp_cvar.py:43-47`, `run_phase3b.py:220-245`。
- `M1` package：`layerB_det_package.parquet`。程式：`run_phase3b.py:884-889`。
- `M2` package：`layerB_prob_package_x.parquet`。程式：`run_phase3b.py:890-894`。

### 3.3 Decision variables

Here-and-now battery variables, common across scenarios：

- Binary charge/discharge mode $u_t$：`u[t]`。程式：`milp_cvar.py:105-110`。
- Charging power $P^{ch}_t\ge0$：`P_ch[t]`。程式：`milp_cvar.py:105-110`。
- Discharging power $P^{dis}_t\ge0$：`P_dis[t]`。程式：`milp_cvar.py:105-110`。
- SOC $E_t$ with bounds $[SOC_{min}EB,SOC_{max}EB]$：`E[t]`。程式：`milp_cvar.py:105-110`。
- Degradation segment variables $e_{t,k}\ge0$：`e_seg[t,k]`。程式：`milp_cvar.py:105-110`。

Scenario-dependent settlement variables：

- Grid to load $P^{gl}_{w,t}$：`P_gl[w,t]`。
- Grid to charge $P^{gc}_{w,t}$：`P_gc[w,t]`。
- PV to load $P^{pvl}_{w,t}$：`P_pvl[w,t]`。
- PV to charge $P^{pvc}_{w,t}$：`P_pvc[w,t]`。
- PV curtailed $P^{pvcu}_{w,t}$：`P_pvcu[w,t]`。
- Evidence：`milp_cvar.py:112-117`。

Green SOC variables：

- Green SOC $E^g_{w,t}$：`E_g[w,t]`。
- Green charge $P^{chg}_{w,t}$：`P_chg[w,t]`。
- Green discharge $P^{disg}_{w,t}$：`P_disg[w,t]`。
- Evidence：`milp_cvar.py:119-122`。

Peak / over-contract / CVaR variables：

- Per-scenario monthly peak proxy $D^{cand}_w$：`D_cand[w]`。
- Over-contract split $Over_w = O1_w+O2_w$：`Over_full[w]`, `O1_full[w]`, `O2_full[w]`。
- CVaR variables $\eta,\xi_w$：`eta`, `xi[w]`。
- Evidence：`milp_cvar.py:124-136`。

### 3.4 Objective function

Implemented objective:

$$
\min \; C_{ene}+C_{deg}+\Delta C_{peak}^{exp}+\lambda_{CVaR} CVaR_{\alpha}(C_{oc})
$$

where:

$$
C_{ene}=\sum_w \pi_w \sum_t c^{TOU}_t(P^{gl}_{w,t}+P^{gc}_{w,t})
$$

Evidence：`milp_cvar.py:218-222`。

$$
C_{deg}=\sum_t\sum_k \lambda_k e_{t,k}
$$

Evidence：`milp_cvar.py:224-226`。

Previous peak cost is computed from carry-in $D_{init}$:

$$
C_{peak,prev}=c^{basic}_m(m_1O1_{prev}+m_2O2_{prev})
$$

Scenario expected candidate peak cost:

$$
E[C_{oc}]=\sum_w \pi_w c^{basic}_m(m_1O1_w+m_2O2_w)
$$

Incremental proxy:

$$
\Delta C_{peak}^{exp}=E[C_{oc}]-C_{peak,prev}
$$

Evidence：`milp_cvar.py:228-239`。

CVaR term:

$$
CVaR_\alpha(C_{oc})=\eta+\frac{1}{1-\alpha}\sum_w \pi_w\xi_w
$$

Evidence：`milp_cvar.py:240-245`。

Case-specific CVaR weight:

- `M1`: `lam=0.0`。Evidence：`run_phase3b.py:884-889`。
- `M2`: `lam=1.0`, `alpha=0.90`。Evidence：`run_phase3b.py:56-57`, `run_phase3b.py:890-894`。

Costs not included in DA MILP objective:

- CAPEX：not included in daily solve; only annual KPI.
- Basic charge base component：not included except over-contract proxy uses basic charge rate.
- TREC purchase：not included in DA objective; final replay/KPI only.
- O&M/Other：not modeled.

### 3.5 Constraints

Battery charge/discharge mutual exclusivity and power limits:

$$
P^{ch}_t \le u_t PB,\quad P^{dis}_t \le (1-u_t)PB
$$

Evidence：`milp_cvar.py:138-141`。

SOC dynamics:

$$
E_t=E_{t-1}+\eta_{ch}P^{ch}_t-\frac{1}{\eta_{dis}}P^{dis}_t
$$

with $E_{-1}=E_{init}$ for the first slot. Evidence：`milp_cvar.py:143-149`。

Terminal SOC:

Only on final day of the test year:

$$
(SOC_{init}-\epsilon)EB \le E_{23}\le (SOC_{init}+\epsilon)EB
$$

Evidence：`milp_cvar.py:151-156`, `run_phase3b.py:266-278`。

There is no confirmed daily terminal SOC anchor and no soft terminal penalty in formal DA code.

Degradation PWL:

$$
P^{dis}_t=\sum_k e_{t,k},\quad 0\le e_{t,k}\le (b_{k+1}-b_k)EB
$$

Evidence：`milp_cvar.py:158-166`。

Scenario settlement:

Load balance:

$$
P^{gl}_{w,t}+P^{pvl}_{w,t}+P^{dis}_t=L_{w,t}
$$

Charge balance:

$$
P^{ch}_t=P^{gc}_{w,t}+P^{pvc}_{w,t}
$$

PV allocation:

$$
P^{pvl}_{w,t}+P^{pvc}_{w,t}+P^{pvcu}_{w,t}=PV_{w,t}
$$

Evidence：`milp_cvar.py:168-184`。

Per-scenario demand proxy:

$$
D^{cand}_w \ge \kappa(P^{gl}_{w,t}+P^{gc}_{w,t}),\quad \forall w,t
$$

$$
Over_w \ge D^{cand}_w-CC,\quad Over_w=O1_w+O2_w,\quad 0\le O1_w\le0.10CC
$$

Evidence：`milp_cvar.py:186-195`。

Important: formal `milp_cvar.py` uses per-scenario $D^{cand}_w$ and expected/CVaR cost. The alternative `milp_daily.py` uses one robust `D_cand` covering all scenarios (`milp_daily.py:189-202`), but `run_phase3b.py` does not use that solver for M1/M2.

Green SOC:

$$
E^g_{w,t}=E^g_{w,t-1}+\eta_{ch}P^{chg}_{w,t}-\frac{1}{\eta_{dis}}P^{disg}_{w,t}
$$

$$
E^g_{w,t}\le E_t,\quad P^{chg}_{w,t}\le P^{pvc}_{w,t},\quad P^{disg}_{w,t}\le P^{dis}_t
$$

Evidence：`milp_cvar.py:197-209`。

Green SOC terminal constraint: not confirmed in `milp_cvar.py`; config comment says terminal band is for both SOC and Green SOC (`config.yaml:40`), but the formal solver only constrains `E[H-1]` on final day.

CVaR linearization:

$$
\xi_w\ge C_{oc,w}-\eta,\quad \xi_w\ge0
$$

Evidence：`milp_cvar.py:211-216`。

### 3.6 DA output variables and files

DA solver returns:

- `u_plan`, `P_ch_plan`, `P_dis_plan`, `E_plan`, `E_end`, `E_g_end`
- `D_cand` as probability-weighted expected candidate peak
- `D_cand_by_scenario`, `C_oc_by_scenario`, `eta_val`, `cvar_val`, `E_coc_val`
- `scenario_dispatch`

Evidence：`milp_cvar.py:255-320`。

`run_phase3b.py` saves:

- `phase3b_M1_da_daily.parquet`, `phase3b_M2_da_daily.parquet`：`run_phase3b.py:315-318`
- `phase3b_M1_replay_hourly.parquet`, `phase3b_M2_replay_hourly.parquet`：`run_phase3b.py:400-404`

Replay settlement executes frozen DA `P_ch_plan` / `P_dis_plan` against realized PV/load. Evidence：`run_phase3b.py:324-405`。

---

## 4. Exact Implemented Intraday MPC MILP

Formal ID solver：`milp_v2/layer_b/milp_mpc.py::solve_mpc_milp()`。  
Formal H24 runner：`milp_v2/experiments/mpc_fixed_rolling_horizon/run_mpc_fixed_horizon_fullyear.py` and `mpc_fixed_horizon_utils.py`。

### 4.1 Issue time and horizon

- H24 main case solves once per hour, 24 solves/day, executes only the first-stage action. Evidence：`mpc_fixed_horizon_utils.py:365-384`, `mpc_fixed_horizon_utils.py:394-424`, `milp_mpc.py:367-370`。
- For `fixed_24h`, horizon length is always $H=24$ regardless of current hour. Evidence：`mpc_fixed_horizon_utils.py:61-78`, `run_mpc_fixed_horizon_fullyear.py:69-84`。
- The old `milp_mpc.py` docstring still says shrinking horizon $H=24-current\_hour$ (`milp_mpc.py:1-10`, `milp_mpc.py:76-79`), but H24 formal runner overrides this via `FixedHorizonConfig.horizon_h()`.

### 4.2 Forecast input construction

H24 PV forecast matrix $PV_{w,j}$ is built by `build_pv_scenarios()`:

- lead 1-6 for same-day daylight slots: use v4 intraday quantile forecasts. Evidence：`mpc_fixed_horizon_utils.py:258-281`。
- lead > 6 for same-day daylight slots: use persistence based on lead-6 v4 forecast. Evidence：`mpc_fixed_horizon_utils.py:282-288`。
- tomorrow `D+1` slots: use DA tail forecast when `forecast_tail_mode="da_forecast_tail"`。Evidence：`mpc_fixed_horizon_utils.py:290-304`。
- D+2 slots, H48 only: persistence fallback. Evidence：`mpc_fixed_horizon_utils.py:313-320`。

Case-specific forecast source:

- `MPC-Det-H24`: `scenario_type="det"`, q50 / K=1, no conformal adjustment. Evidence：`run_mpc_fixed_horizon_fullyear.py:69-74`, `mpc_fixed_horizon_utils.py:242-244`, `mpc_fixed_horizon_utils.py:265-268`。
- `MPC-Prob-H24`: `scenario_type="prob"`, K=5, RC-conformal adjustment for q10/q90, `lam=0.0` risk-neutral. Evidence：`run_mpc_fixed_horizon_fullyear.py:75-79`, `mpc_fixed_horizon_utils.py:269-281`。
- DA deterministic tail lookup uses `layerB_det_package.parquet`。Evidence：`mpc_fixed_horizon_utils.py:161-169`。
- DA probabilistic tail lookup uses `new_pipeline/data/output/stage4_pv_scenarios_K5.parquet` and requires exactly K=5 values. Evidence：`mpc_fixed_horizon_utils.py:172-184`。

Load forecast:

- All scenarios use the same realized load lookup. The code explicitly labels this as perfect load assumption. Evidence：`mpc_fixed_horizon_utils.py:150-158`, `mpc_fixed_horizon_utils.py:324-338`。

### 4.3 Decision variables

Common candidate battery variables:

- $u_j$, $P^{ch}_j$, $P^{dis}_j$, $E_j$, $e_{j,k}$ are scenario-invariant. Evidence：`milp_mpc.py:142-147`。

Scenario-dependent settlement variables:

- $P^{gl}_{w,j}$, $P^{gc}_{w,j}$, $P^{pvl}_{w,j}$, $P^{pvc}_{w,j}$, $P^{pvcu}_{w,j}$。Evidence：`milp_mpc.py:149-154`。

Peak / CVaR variables:

- $D^{cand}_w$, $Over_w$, $O1_w$, $O2_w$ per scenario。Evidence：`milp_mpc.py:156-160`。
- $\eta$, $\xi_w$ CVaR variables exist even when `lam=0`。Evidence：`milp_mpc.py:162-164`。

Optional M6/M7 variables:

- `m6b`, `m6c`, `m7a`, `m7d`, `m71b` are optional arguments and not used in the main H24 cases. Evidence：`milp_mpc.py:52-73`, `milp_mpc.py:231-318`。

### 4.4 Objective function

Implemented objective:

$$
\min C_{ene}^H+C_{deg}^H+\Delta C_{peak}^{exp,H}+\lambda CVaR_\alpha(C_{oc}^H)+C_{M6/M7}
$$

For main H24 cases:

- `MPC-Det-H24`: $K=1$, $\lambda=0$。
- `MPC-Prob-H24`: $K=5$, $\lambda=0$ risk-neutral。
- Optional M6/M7 terms are zero because no optional references/masks/penalties are passed by the formal H24 runner.

Energy:

$$
C_{ene}^H=\sum_w\pi_w\sum_j c^{TOU}_j(P^{gl}_{w,j}+P^{gc}_{w,j})
$$

Evidence：`milp_mpc.py:320-324`。

Degradation:

$$
C_{deg}^H=\sum_j\sum_k\lambda_k e_{j,k}
$$

Evidence：`milp_mpc.py:326-328`。

Expected over-contract proxy:

$$
\Delta C_{peak}^{exp,H}=\sum_w\pi_w c^{basic}_m(m_1O1_w+m_2O2_w)-C_{peak,prev}
$$

Evidence：`milp_mpc.py:330-339`。

CVaR:

$$
CVaR_\alpha=\eta+\frac{1}{1-\alpha}\sum_w\pi_w\xi_w
$$

Evidence：`milp_mpc.py:341-343`。

Final objective line：`milp_mpc.py:350-354`。

### 4.5 Constraints

Battery mutual exclusivity and limits:

$$
P^{ch}_j\le u_jPB,\quad P^{dis}_j\le(1-u_j)PB
$$

Evidence：`milp_mpc.py:168-171`。

SOC dynamics:

$$
E_j=E_{j-1}+\eta_{ch}P^{ch}_j-\frac{1}{\eta_{dis}}P^{dis}_j
$$

with $E_{-1}=E_{soc,current}$. Evidence：`milp_mpc.py:173-179`。

Terminal SOC:

- Solver supports a hard terminal band only if `is_last_hour=True`: `milp_mpc.py:181-186`。
- Formal H24 runner sets `is_last=True` only for `horizon_mode=="day_bounded"` and final day hour 23. Since main H24 uses `horizon_mode="fixed_24h"`, no terminal SOC hard band is active in `MPC-Det-H24` or `MPC-Prob-H24`。Evidence：`mpc_fixed_horizon_utils.py:403-404`。
- There is no confirmed soft terminal penalty and no terminal anchor to DA SOC in main H24 ID MILP.

Degradation:

$$
P^{dis}_j=\sum_k e_{j,k},\quad e_{j,k}\le(b_{k+1}-b_k)EB
$$

Evidence：`milp_mpc.py:188-196`。

Scenario settlement:

$$
P^{gl}_{w,j}+P^{pvl}_{w,j}+P^{dis}_j=L_{w,j}
$$

$$
P^{ch}_j=P^{gc}_{w,j}+P^{pvc}_{w,j}
$$

$$
P^{pvl}_{w,j}+P^{pvc}_{w,j}+P^{pvcu}_{w,j}=PV_{w,j}
$$

Evidence：`milp_mpc.py:198-214`。

Per-scenario demand proxy:

$$
D^{cand}_w\ge\kappa(P^{gl}_{w,j}+P^{gc}_{w,j})
$$

$$
Over_w\ge D^{cand}_w-CC,\quad Over_w=O1_w+O2_w,\quad O1_w\le0.10CC
$$

Evidence：`milp_mpc.py:216-224`。

CVaR linearization:

$$
\xi_w\ge C_{oc,w}-\eta
$$

Evidence：`milp_mpc.py:226-229`。

### 4.6 Output and execution

The MILP returns full candidate trajectories, but only first-stage action is executed:

- `P_ch_plan`, `P_dis_plan`, `E_plan`：`milp_mpc.py:363-365`。
- `p_ch_exec=P_ch_plan[0]`, `p_dis_exec=P_dis_plan[0]`, `soc_next=E_plan[0]`：`milp_mpc.py:367-370`。

The H24 runner then updates realized replay state:

$$
p^{grid}_{realized}=\max(L_{realized}-PV_{realized}-P^{dis}_{exec}+P^{ch}_{exec},0)
$$

$$
D_{mth}\leftarrow \max(D_{mth},\kappa p^{grid}_{realized})
$$

$$
E_{soc}\leftarrow E_{soc}+\eta_{ch}P^{ch}_{exec}-\frac{1}{\eta_{dis}}P^{dis}_{exec}
$$

Evidence：`mpc_fixed_horizon_utils.py:426-435`。

Outputs:

- `M3_H24_det_hourly.parquet`, `M3_H24_det_daily.parquet`
- `M5_H24_prob_lam0_hourly.parquet`, `M5_H24_prob_lam0_daily.parquet`
- Evidence：`mpc_fixed_horizon_utils.py:563-567`, `run_mpc_fixed_horizon_fullyear.py:115-132`。

---

## 5. M8 Arbitration：MILP 外的執行模型

M8 is not an optimization model. It does not solve a MILP. Evidence：`m8_utils.py:1-14`, `BH_THESIS_HANDOVER_2026_05_FINAL/03_FINAL_CASE_DEFINITIONS_AND_NAMING.md:4-9`。

### 5.1 M8-Prob-H24 formal setting

- Master：`DA-Prob (M2)` from `phase3b_M2_replay_hourly.parquet`。Evidence：`m8_utils.py:73-95`, `run_m8_h24_standalone.py:90-96`。
- Candidate：`MPC-Prob-H24 (M5_H24_prob_lam0)` from `results/fullyear/M5_H24_prob_lam0_hourly.parquet`。Evidence：`run_m8_h24_standalone.py:85-91`。
- Parameters：`alpha=0.25`, `soc_band_frac=0.15`, `cost_filter="relaxed"`, `cost_tolerance_ntd=100`, `charge_headroom_kw=300`。Evidence：`run_m8_h24_standalone.py:50-55`。

### 5.2 Blended candidate

Net battery power:

$$
B=P^{dis}-P^{ch}
$$

Candidate blend:

$$
B_{cand}=B_{M2}+\alpha(B_{M5}-B_{M2})
$$

Then convert back to charge/discharge and clip by PB and SOC limits. Evidence：`m8_utils.py:140-169`。

### 5.3 F1-F4 safety filters

M8 uses current-hour realized PV/load in replay to estimate grid import. This is explicitly documented as a research simplification, not a deployable information boundary. Evidence：`m8_utils.py:11-14`, `m8_utils.py:187-194`。

F1 Peak cap:

$$
D_{safe}=D_{ref,month}+\epsilon_{peak}
$$

Accept if candidate is grid-reducing or

$$
\max(D_{mth,current},\kappa p^{grid}_{cand})\le D_{safe}
$$

Evidence：`m8_utils.py:198-216`。

F2 SOC corridor:

$$
E^{M2,ref}_{after}-band \le E^{cand}_{after}\le E^{M2,ref}_{after}+band
$$

where $band=soc\_band\_frac\cdot EB$。Evidence：`m8_utils.py:218-222`。

F3 immediate TOU cost:

$$
c^{TOU}_t p^{grid}_{cand}\le c^{TOU}_t p^{grid}_{M2}+tol
$$

Evidence：`m8_utils.py:224-233`。

F4 charge headroom:

If candidate is extra charging,

$$
\kappa p^{grid}_{cand}\le D_{ref,month}-charge\_headroom
$$

Evidence：`m8_utils.py:235-241`。

All filters pass → execute candidate. Otherwise fallback to master action. Evidence：`m8_utils.py:253-263`。

---

## 6. DA vs ID Comparison Table

| Aspect | Day-ahead MILP | Intraday H24 MPC MILP |
|---|---|---|
| Solve frequency | Once per target day | Once per hour, 24 solves/day |
| Horizon | 24h daily | Fixed H=24 rolling horizon |
| Information set | DA package frozen before target day; current repo gate is D-1 20:00 local / D-1 12:00 UTC | Latest intraday v4 for lead 1-6, persistence for same-day lead >6, DA tail for D+1 |
| Forecast source | `layerB_det_package.parquet` or `layerB_prob_package_x.parquet` | `id_forecast_v4_final.parquet`, `stage1_rc_conformal.parquet`, DA tail lookups |
| Scenario construction | M1 K=1; M2 formal package has K=6 = K5 + synthetic miss | M3 K=1; M5_H24 K=5 RC conformal |
| Load input | `load_input_kw` from package | realized load lookup, explicitly perfect load assumption |
| Initial SOC | Carry from previous day; initial $0.5EB$ | Current executed SOC from previous hour/day |
| Peak state | `D_mth` carry from previous day; monthly reset in runner | `D_mth` carry from previous executed hour; monthly reset in runner |
| Peak proxy | Formal `milp_cvar`: per-scenario $D_cand_w$, expected OC plus optional CVaR | per-scenario $D_cand_w$, expected OC; main M5_H24 has `lam=0` |
| CVaR | Code evidence: M2 uses `lam=1`; handover naming says No CVaR, inconsistent | Main MPC-Prob-H24 is risk-neutral `lam=0`; CVaR H24 is appendix/diagnostic |
| Battery action | Common across scenarios for all 24h | Common across scenarios for H slots |
| Terminal SOC | Hard band only on final day of year | Solver supports final-step band, but main fixed H24 does not activate it |
| Green SOC | Present in DA solver constraints | Not present in `milp_mpc.py` |
| TREC | Not in MILP objective; annual/replay KPI only | Not in MPC objective; annual/replay KPI only |
| Output role | Frozen master day plan and replay schedule | Candidate first-stage action stream; pure MPC cases execute directly, M8 uses as candidate |
| Execution role | DA-Prob M2 is M8 master and safety baseline | MPC-Prob-H24 is M8 candidate, not final controller by itself |

---

## 7. Easily Misstated Points

1. **DA gate time**：目前 implemented gate 應寫 `D-1 20:00 local / D-1 12:00 UTC`，不是 16:00 或 22:00。若論文採 16:00 market cutoff，現有 `da_v2`/DA package 的 NWP availability 需重建或 caveat。Evidence：`DA_GATE_TIME_AUDIT.md:3-15`, `DA_GATE_TIME_AUDIT.md:71-84`。

2. **DA 是否真的使用 5 reduced scenarios**：正式 `DA-Prob` 使用 `layerB_prob_package_x.parquet`，artifact 顯示每日至少 6 scenarios (`s0`-`s4` + `synth_miss`)。應寫「five reduced scenarios plus one synthetic miss scenario」。純 K5 檔案 `layerB_prob_K5.parquet` 存在，但不是 `run_phase3b.py` 的 formal M2 input。

3. **DA common action 是否跨 scenario 共用**：是。`u`, `P_ch`, `P_dis`, `E`, `e_seg` 沒有 scenario index。Evidence：`milp_cvar.py:105-110`。

4. **ID MPC common action 是否跨 scenario 共用**：是。Evidence：`milp_mpc.py:142-147`。

5. **Peak proxy 寫法**：formal DA 和 ID 都是 per-scenario $D_cand_w$，objective 用 expected over-contract plus optional CVaR。`milp_daily.py` 的 robust single `D_cand` 不是 formal M1/M2 runner 使用的 solver。

6. **Over-contract penalty two-segment linearization**：實作為 $O1\le0.10CC$, $O2$ 為剩餘超約，成本 $c^{basic}(2O1+3O2)$。Evidence：`config.yaml:52-54`, `milp_cvar.py:193-195`, `milp_mpc.py:223-224`, `mpc_fixed_horizon_utils.py:621-625`。這是小時解析度需量 proxy；台電實務 15-min demand charge 未完整重建。

7. **TREC 是 MILP objective 還是 replay settlement**：在 formal DA/ID objective 中沒有 TREC。最終成本表用 replay/KPI 的正向 TREC purchase。Evidence：`milp_cvar.py:218-245`, `milp_mpc.py:320-354`, `mpc_fixed_horizon_utils.py:631-645`。

8. **Battery degradation**：實作是 discharge-throughput PWL，每小時獨立分段，不是 full cycle-life rainflow 或 SOC-depth dynamic model。Evidence：`milp_cvar.py:158-166`, `milp_mpc.py:188-196`, `mpc_fixed_horizon_utils.py:577-592`。

9. **Terminal SOC anchor**：DA 只有 final day hard band；ID H24 main 沒有 terminal SOC hard band、soft penalty、或 DA SOC anchor。M8 的 F2 是 MILP 外 SOC corridor。Evidence：`milp_cvar.py:151-156`, `milp_mpc.py:181-186`, `mpc_fixed_horizon_utils.py:403-404`, `m8_utils.py:218-222`。

10. **ID MPC 是否直接考慮 DA D_ref**：main H24 `MPC-Prob-H24` 沒有直接使用 DA `D_ref`；只有 M8 F1 使用 master 的 monthly `D_ref`。Optional M6/M7 有 reference-guided arguments，但不是 main case。Evidence：`milp_mpc.py:52-73`, `m8_utils.py:116-120`, `m8_utils.py:210-216`。

11. **M8 F1-F4 是否在 MILP 外執行**：是。M8 純 replay arbitration，非 MILP constraints。Evidence：`m8_utils.py:1-14`, `m8_utils.py:174-297`。

---

## 8. Mapping from Equations to Code

| Model element | Code file / function / line |
|---|---|
| DA formal solver | `milp_v2/layer_b/milp_cvar.py::solve_day_ahead`, lines `27-320` |
| DA runner and package routing | `milp_v2/run_phase3b.py::run_da_case`, lines `250-319`; case dispatch lines `884-894` |
| DA replay settlement | `milp_v2/run_phase3b.py::replay_da_case`, lines `324-405` |
| DA variables | `milp_cvar.py:105-136` |
| DA battery/SOC/degradation constraints | `milp_cvar.py:138-166` |
| DA settlement constraints | `milp_cvar.py:168-184` |
| DA peak/CVaR constraints | `milp_cvar.py:186-216` |
| DA objective | `milp_cvar.py:218-245` |
| ID formal solver | `milp_v2/layer_b/milp_mpc.py::solve_mpc_milp`, lines `33-418` |
| ID H24 variant definitions | `run_mpc_fixed_horizon_fullyear.py:69-84` |
| ID forecast construction | `mpc_fixed_horizon_utils.py:222-321` |
| ID load assumption | `mpc_fixed_horizon_utils.py:324-338` |
| ID hourly solve/execution | `mpc_fixed_horizon_utils.py:365-453` |
| ID variables | `milp_mpc.py:142-164` |
| ID constraints | `milp_mpc.py:168-229` |
| ID objective | `milp_mpc.py:320-354` |
| ID first-stage output | `milp_mpc.py:363-370` |
| Corrected annual KPI | `mpc_fixed_horizon_utils.py::corrected_kpis`, lines `595-655` |
| M8 master/candidate setup | `run_m8_h24_standalone.py:79-120` |
| M8 action blending | `m8_utils.py:140-169` |
| M8 F1-F4 filters | `m8_utils.py:174-297` |
| M8 full-year replay | `m8_utils.py:475-620` |

---

## 9. Inconsistencies Between Thesis/Handover Text and Code

| Issue | Code/artifact evidence | Conflicting text | Recommendation |
|---|---|---|---|
| DA-Prob CVaR status | `run_phase3b.py:890-894` passes `lam=1.0`; `milp_cvar.py:245` objective includes `lam*C_cvar`; saved M2 `cvar_val` nonzero | `03_FINAL_CASE_DEFINITIONS_AND_NAMING.md:17-18` says `No CVaR` | Do not write DA-Prob as definitely no-CVaR unless manually confirmed by rerun/log. Thesis-safe wording: "probabilistic DA MILP with scenario-based peak-risk treatment; code path includes a CVaR term." |
| DA scenario count | formal M2 input `layerB_prob_package_x.parquet` has 6 scenarios/day | config and comments say K=5 | Write "five reduced scenarios plus a synthetic miss scenario"; avoid "exactly five scenarios" for M2 |
| ID H24 vs shrinking horizon | H24 runner fixed `H=24`; old `milp_mpc.py` docstring says shrinking horizon | older docs/model audit mention `H=24-h` for M3/M5 | For final main cases write fixed 24h rolling; place EOD/day-bounded as sensitivity |
| TREC sign/accounting | final table and `corrected_kpis()` use positive TREC purchase | old audits and `run_phase3b.py::compute_kpis()` still use TREC revenue/subtraction | Use final H24 cost table only; mark old TREC revenue documents deprecated |
| DA gate time | artifacts support D-1 20:00 | later forecasting handover text mentions D-1 16:00 | Use D-1 20:00 for implemented pipeline, or caveat/rebuild for 16:00 market cutoff |
| M8 no-regret wording | M8 filters use realized PV/load in replay (`m8_utils.py:11-14`) | "no-regret guarantee" can sound operationally exact | Add caveat: replay-based no-regret under realized-value filter approximation |

---

## 10. Recommended Corrected Wording for Chapter 3

### 10.1 Scheduling case definitions

建議寫法：

> The scheduling layer is evaluated using a fixed PV-BESS design, \(CC=3306.45\) kW, \(PB=1510.74\) kW, and \(EB=7525.83\) kWh. The day-ahead layer solves a 24-hour MILP once per target day. The deterministic DA case uses a single q50 PV trajectory, while the probabilistic DA case uses a reduced scenario package consisting of five DA PV scenarios plus a synthetic miss scenario. Battery charge/discharge decisions are here-and-now variables shared across scenarios, while grid/PV settlement, over-contract proxy, and Green-SOC accounting are scenario-dependent.

If keeping the code-verified CVaR term:

> In the implemented DA probabilistic solver, the over-contract proxy is scenario-dependent and the objective includes expected over-contract cost plus a CVaR peak-risk term. The final reported cost is not the MILP objective value; it is recomputed by replay settlement using realized PV/load and the corrected annual cost formula.

If advisor decides to follow handover `No CVaR` wording:

> Manual confirmation required: current saved code path imports `milp_cvar.py` and passes `lam=1.0` for M2. A no-CVaR DA-Prob statement should only be used if a separate no-CVaR run/log can be provided.

### 10.2 Intraday MPC

建議寫法：

> The intraday candidate is generated by a fixed 24-hour rolling-horizon MILP solved at each hour. Only the first-stage charge/discharge action is executed before the horizon is rolled forward. For same-day PV, lead times 1-6 use the v4 intraday forecast; same-day lead times beyond 6 use a lead-6 persistence tail; next-day slots use the day-ahead forecast tail. The probabilistic H24 MPC uses five RC-conformal PV scenarios and is risk-neutral in the main comparison; the CVaR variant is retained only as a diagnostic appendix case.

### 10.3 M8

建議寫法：

> M8 is not an MPC solver. It is a no-regret arbitration layer applied after the DA and ID schedules have been generated. At each hour, M8 blends the DA-Prob master action and the MPC-Prob-H24 candidate action, then applies four safety filters: a monthly peak cap relative to the DA reference, an SOC corridor around the DA SOC trajectory, an immediate TOU-cost consistency check, and an extra-charge headroom check. If any filter fails, the controller falls back to the DA-Prob action. In the replay implementation, these filters use realized current-hour PV/load to estimate grid import; this is disclosed as a replay approximation rather than a deployable real-time information set.

### 10.4 Cost formula

建議寫法：

> Annual cost is reported using the corrected replay settlement formula \(C_{total}=C_{CAPEX}+C_{basic}+C_{TOU}+C_{OC}+C_{deg}+C_{TREC}\). The MILP objective includes TOU energy cost, a degradation proxy, and an incremental over-contract proxy, but it does not include CAPEX, the basic charge base component, or TREC purchase cost. TREC is accounted for only in the annual replay settlement.

---

## 11. Issues Requiring Manual Confirmation

1. `0528thesis.pdf` was not found in the repository. If Section 3.9 must be audited against the latest draft, provide the exact PDF or source `.docx/.tex/.md`.
2. Confirm whether thesis should describe `DA-Prob (M2)` as CVaR-based. Current code/artifact evidence says yes; final case mapping says No CVaR.
3. Confirm whether formal DA scenario statement should be updated to "K5 + synthetic miss". Current formal input `layerB_prob_package_x.parquet` has 6 scenarios/day.
4. Confirm whether the thesis will use implemented `D-1 20:00` forecast-freeze gate, or market `D-1 16:00` as an operational limitation/future refinement.
5. If strict operational deployability is required, M8 F1-F4 should be rewritten as using forecasted PV/load rather than realized PV/load; current replay result is thesis-usable only with the documented approximation.

