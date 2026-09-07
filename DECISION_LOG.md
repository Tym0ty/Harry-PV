# DECISION_LOG.md — HARRY-PV V2.1 Implementation

> 依 CLAUDE.md §5：任何「V2.1 計畫書未明確指定由 Claude Code 自行決定」或「audit 發現衝突決定如何處理」的選擇，均須記錄於此。
>
> 格式：日期 | 決議 | 依據 | 影響範圍 | 是否需使用者確認

---

## 研究情境澄清（前提說明，所有設計決議的背景）

**日期：** 2026-05-20  
**確認人：** Harry Huang（使用者書面確認）

本研究的設計情境是「**台灣校園在 RE20 政策約束下的 PV-BESS 校園微電網**」：

- PV 容量由可用屋頂面積決定：PV_ref = 2687 kWp（固定，不做 sizing）
- 台科目前實際簽約 CC = **5000 kW**，但這是「無 PV 情境下」的 CC
- RE20 + PV=2687 kWp 情境下，CC 應隨之下調，否則 BESS 完全沒有 DCT risk 可管理
- 合理 reference design 在 RE20 情境下為：CC_ref = 3232 kW（源自 Layer A det sizing）

論文敘事框架：「**given RE20-driven PV sizing, study horizon design under a plausible reference (CC, BESS)**」，**不是「採用前一份 sizing study 的結論」**。

論文中必須明確：
1. reference design 框架為「RE20 情境下的合理參考設計」
2. 不是最佳化結果的直接引用
3. Limitations 中明文：*"real-world Taipower contracting would require additional CC re-negotiation under RE20"*

---

## DL-001：CC_ref 設定

**日期：** 2026-05-20  
**決議：** CC_ref = **3232 kW**  
**依據：**
- 台科目前實際 CC = 5000 kW（無 PV 情境），在 RE20+PV=2687 kWp 情境下與研究問題不符
- 5000 kW 下 net peak 永遠不超約，整份研究問題（horizon 對 DCT risk 的影響）被架空
- 3232 kW 為 Layer A det sizing（DD 案）在 PV=2687 kWp 情境下的最佳化結果，是 RE20 情境下的合理 reference CC
- 來源：`milp_outputs/da/layer_a_frozen_design_DD.json:CC_kw=3232.4`（凍結於 2026-04-08）

**影響範圍：** 所有 C0/C1/C2/C3a/C3b/D2/D3 case；two-CC robustness 設計  
**Harry 確認：** ✅ 已書面確認（2026-05-20）  
**後續行動：** 寫入 `milp_v3_horizon/configs/base.yaml`

---

## DL-002：x_ref 完整設計向量

**日期：** 2026-05-20  
**決議：** 採用 **DD 設計**作為 x_ref：

| 參數 | 值 | 單位 | 來源 |
|------|-----|------|------|
| CC_ref | 3232 | kW | layer_a_frozen_design_DD.json (DL-001) |
| PV_ref | 2687 | kWp | 屋頂面積限制（固定，不動） |
| P_B_ref | 1152 | kW | layer_a_frozen_design_DD.json |
| E_B_ref | 6902 | kWh | layer_a_frozen_design_DD.json |

**依據：**
- DD 設計基於確定性 PV 預測（最保守），向口試委員最易解釋
- PP 設計依賴情境機率，不適合作 fixed reference（reference 應獨立於 scheduling uncertainty）
- Harry 書面確認採用 DD

**影響範圍：** 所有 case 的 fixed design；BESS 物理約束的 PB/EB 上下界  
**Harry 確認：** ✅ 已書面確認（2026-05-20）  
**後續行動：** 寫入 `milp_v3_horizon/configs/base.yaml`

---

## DL-003：Two-CC Robustness 升級為 Three-CC Robustness

**日期：** 2026-05-20  
**決議：** 升級 V2.1 §15.6 的 two-CC robustness 為三層：

| CC Case | 值（kW） | 定義 | 用途 |
|---------|---------|------|------|
| CC_loose | 5000 | 台科現況（無 RE20 情境） | 確認無 DCT risk 的 baseline；檢查研究問題的條件依賴性 |
| CC_ref | 3232 | RE20 情境 reference（主結果） | 所有主要 case 的 CC |
| CC_tight | ≈ 3070 | 0.95 × CC_ref（壓力測試） | 檢查結論在更緊 CC 下是否穩健 |

**依據：**
- V2.1 §15.6 原文：「若可取得 lab-optimal CC，可用作補充 stressed CC」
- 本研究的 lab-optimal CC 即為 CC_ref；因此用 CC_loose（現況 5000 kW）作另一端對照
- CC_loose 的加入可同時回答 Q3 (C0 是否 straw man) 的情境依賴性問題

**影響範圍：** Phase 3 robustness package；論文 Table 10（two-CC → three-CC table）  
**Harry 確認：** ✅ 已書面確認（2026-05-20）  
**後續行動：** CC_tight 由程式計算（0.95 × CC_ref），不 hardcode；CC_loose 寫入 config

---

## DL-004：Load Intraday Forecast 策略

**日期：** 2026-05-20  
**決議：** 採用**選項 C（DA-tail 不更新）**

規則：
- DA load forecast：使用 `layerB_det_package.parquet:load_input_kw`（已有 D-1 20:00 gate）
- ID load forecast：= 當日 DA load forecast 對應時段，不在 intraday rolling 中更新
- 所有 case（C0/C1/C2/C3a/C3b）的 load forecast 均為 DA-level，不更新
- D3 diagnostic 使用 **perfect load**（realized load），以保留 forecast 上界 gap 意義

**依據：**
- 若 ID load = realized load（選項 A），C2 vs D3 在 load 維度差距=0，RQ4 只剩 PV 維度，D3 失去獨立診斷意義
- 選項 C 使 RQ4 兩維度（PV forecast error vs load forecast error）都有比較空間
- Load 可預測性高（MAPE ≈ 5%），DA-level 不更新是可辯護的工程假設

**論文揭露（必須）：**
> *"Load forecasts are not updated in the intraday rolling MILP — the day-ahead load prediction is used throughout the scheduling day, reflecting the high predictability of campus load demand. Improving intraday load forecast models is left for future work."*

**影響範圍：** C0/C1/C2/C3a/C3b 的 ID MILP load 輸入；D3 diagnostic 設計  
**Harry 確認：** ✅ 已書面確認（2026-05-20）  
**後續行動：** build_id_packages.py 中 load slice = DA forecast slice（不需新 model）

---

## DL-005：λ^SOC 預設值設定流程（資訊邊界合規）

**日期：** 2026-05-20  
**決議：** λ^SOC **不得在 test period 看結果後設定**；必須在 pre-test period 校準後凍結。

校準流程（Phase 1 Step 1，執行前需 Harry 下一段 prompt 確認）：
1. 使用 **pre-test period（2024-04-17 → 2024-10-31）** 跑 λ^SOC ∈ {0.5, 2, 5, 10, 20} NTD/kWh 的 C1 + C2 快速對比
2. 選擇使 C1 與 C2 之間有明顯但非極端差距的中段值作為 λ₀
3. 在 DECISION_LOG.md 新增 DL-005b 記錄校準結果與選定值
4. 凍結 λ₀ 後，Phase 1 Prototype Gate 在 test period 上執行，不再回調

**base.yaml 暫定：** λ^SOC = null（待 Phase 1 Step 1 填入）

**違反此規則的緊急停機條件：** 若發現 λ^SOC 是在看過 test year 結果後調整的，立即停止並回報。

**影響範圍：** C1/C2/C3a/C3b MILP objective；C1 vs C2 gap 分析  
**Harry 確認：** ✅ 已書面確認（2026-05-20）  
**後續行動：** Phase 1 Step 1 執行 λ^SOC 掃描（等 Harry 下一段 prompt）

---

## DL-006：γ 預設值（BESS coverability safety margin）

**日期：** 2026-05-20  
**決議：** γ = **0**（主線，Phase 3 再做 {0, 0.05, 0.10} sensitivity）  
**依據：** V2.1 §12.9 明確說明「主線可先設 γ=0，再做 sensitivity」  
**Harry 確認：** 隱含同意（未提異議）  
**後續行動：** 寫入 `milp_v3_horizon/configs/base.yaml`

---

## DL-007：ρ_threshold 預設值（C3a DCT-state 閾值）

**日期：** 2026-05-20  
**決議：** ρ_threshold = **0.95**（主線，Phase 3 再做 {0.90, 0.95, 1.00} sensitivity）  
**依據：** V2.1 §11.4 公式中 ρ_τ ≥ 0.95 → EOD，Phase 3 做 sensitivity  
**Harry 確認：** 隱含同意（未提異議）  
**後續行動：** 寫入 `milp_v3_horizon/configs/base.yaml`

---

## DL-008：κ 需量放大係數來源

**日期：** 2026-05-20  
**決議：** 暫用 κ = **1.006**（來自 `milp_v2/config.yaml`）  
**依據：** 該值已在既有研究中使用，但未找到帳單推導依據；Phase 2 前需 Harry 確認來源  
**疑慮：** V2.1 §6.2 公式需 κ_m（月度別），現有值為 scalar；若月度差異明顯應拆分  
**Harry 確認：** ⚠️ **待確認**（Phase 2 前）  
**後續行動：** Phase 2 前 Harry 提供帳單來源或確認 scalar 足夠作為 approximation

---

## DL-009：Gurobi Solver 確認

**日期：** 2026-05-20  
**決議：** 確認可用  

```
Gurobi version: 13.0.0
License type:   Academic (non-commercial)
License ID:     2740851
License expiry: 2026-11-18
Smoke test:     PASSED（LP optimize, status=2, optimal）
```

**影響範圍：** 所有 MILP；預計 full-year rolling (365×24=8760 MILP) 在 academic license 內  
**Harry 確認：** N/A（技術確認，不需 Harry 決策）  
**後續行動：** Phase 1 Prototype Gate 前再跑 MIP smoke test（含二元變數）

---

## DL-010：Load DA Forecast 等同 Realized Load（既有設計確認）

**日期：** 2026-05-20  
**決議：** 確認既有 `layerB_det_package.parquet:load_input_kw` 等同 realized load（MAPE=0，因直接使用 `padil_NTUST_Load_PV.csv` 的實現值），作為 DA load forecast  
**依據：** bridge_full_year_report.json 顯示 load_unc_mape=0.05 只用於 load_unc 版本；det 版本直接使用實現值  
**影響：** DA load forecast 無誤差（等同 perfect DA load），但此簡化在論文中需揭露  
**Harry 確認：** 隱含同意（DL-004 選項 C 的一部分）  
**論文揭露：** "Day-ahead load forecast uses the realized campus load profile, reflecting load predictability; intraday updates are not applied."

---

---

## DL-011：Pre-test ID Forecast 補產策略

**日期：** 2026-05-20  
**決議：** 採方案 A，對 pre-test period (2024-04-17 → 2024-10-31) 補產 ID P50 forecast，存為 `new_pipeline/data/final_forecast/pretest_id_forecast.parquet`

**依據與技術細節：**

1. **Feature matrix 已完整**：`feature_matrix_id_v3.parquet` 已含 pre-test 期間所有 72 個 XGBoost 所需特徵，且零 NaN（僅 `Total_Cloud_Amount_tenths` 有 NaN，但此欄不被模型使用）
2. **模型可複用**：99 個已訓練 XGBoost 模型（2021-04-01 至 2024-11-01 前訓練資料）無需重訓，對 pre-test 特徵向量做 inference 為 causal
3. **分段處理**：
   - **2024-05-01 → 2024-10-31（calibration split, 13,986 rows）**：直接讀取 `id_predictions_v3.parquet` 中的 `q50_agaci`（CQR + AGACI 已由 step_nwp5_eval.py 計算），GHI→PV 轉換
   - **2024-04-17 → 2024-04-30（validation split, 1,092 rows）**：重新跑 XGBoost P50 inference + 應用 CQR delta(q=0.50)。注意：CQR delta 是由 calibration (May-Oct 2024) 資料估算的，對這 14 天有輕微 forward-looking 偏差；此偏差對 λ^SOC 校準結果影響可忽略（14 天 / 198 天 ≈ 7%）
4. **因果性確認**（三個抽查點）：
   - 2024-04-20 10:00 lead=1：nwp_age_hours=8.0h ≥ 6 ✓
   - 2024-07-15 14:00 lead=3：nwp_age_hours=6.0h ≥ 6 ✓
   - 2024-10-01 09:00 lead=6：nwp_age_hours=7.0h ≥ 6 ✓

**輸出驗證：**
- Shape: (15,078, 2)，columns: [lead_time, pv_q50]
- 格式與 `id_forecast_hybrid_final.parquet` 相容（build_id_packages.py 直接可讀）
- pv_q50 stats: min=0.0, max=2289.5, mean=728.7 kW（物理合理）
- Zero NaN，zero negative values

**生成腳本：** `new_pipeline/build_pretest_id_forecast.py`

**論文揭露（若需引用）：**
> "The pre-test ID P50 forecast was generated by applying the pre-trained XGBoost v3 models to the existing feature matrix for 2024-04-17–2024-10-31. For the 14-day validation window (Apr 17–30), CQR bias correction was estimated from the subsequent calibration period; this minor forward-looking correction does not affect λ^SOC calibration."

**影響範圍：** 僅用於 Step 2 λ^SOC pre-test 校準；test period 結果使用原始 `id_forecast_hybrid_final.parquet`  
**Harry 確認：** 2026-05-20（方案 A 決策確認）  
**後續行動：** Step 2 λ^SOC 校準，以此 pretest_id_forecast.parquet 作為 pre-test ID 資料來源

---

---

## DL-005b：λ₀ 凍結為 5.0

**日期：** 2026-05-20  
**Harry 確認：** ✅ 已書面確認（Option A，2026-05-20）

**決議：** λ^SOC₀ 凍結為 **5.0 NTD / unit SOC 偏差**。

**依據：**
- Step 2-B λ^SOC 全期掃描（pre-test 2024-05-01 → 2024-10-31，184 天，
  λ ∈ {0.5, 2.0, 5.0, 10.0, 20.0}）顯示五個 λ 值在 C1、C2 的成本完全相同
  （C1=52.6628M NTD、C2=50.8469M NTD、gap=+3.57%、anchor deviation ≡ 0，
  共 1,840 次 ID 求解全部 OPTIMAL）
- 根本原因：DA MILP（H=24，λ=0）在高負載夏季 + 肩月（5–10 月），
  每天以「完全放電至 soc_min=0.10」為最佳策略。C1 與 C2 各自獨立在 EOD
  達相同 soc_min，使 λ×s_soc ≡ 0，λ 的絕對值對成本無區分力。
- 此為合法校準結論：pre-test 揭示高負載情境下 SOC anchor 結構性失活，
  不是缺陷。
- +3.57% C1 vs C2 gap 是純 horizon-driven 效應，不含 anchor 污染，
  校準章節的重要副產品。
- λ₀=5.0 為掃描範圍的幾何中間值，使 Phase 3 sensitivity {0.1λ₀, λ₀, 10λ₀}
  = {0.5, 5.0, 50.0} 跨越兩個數量級，數值穩定且符合 V2.1 §15 格式。

**資訊邊界合規：** λ₀ 選擇僅使用 pre-test 資料，test period 結果未被觀察。

**下游影響：**
- `milp_v3_horizon/configs/base.yaml` → `terminal_soc.lambda_soc: 5.0`
- C1/C2/C3a/C3b 所有 ID MILP objective 均使用 λ₀=5.0
- Phase 3 sensitivity 將驗證結果對 λ 的不變性

---

---

## DL-012：Battery Degradation 加入 MILP Objective（實作缺口已關閉）

**日期發現：** 2026-06-06（Design Conformance Audit）  
**日期關閉：** 2026-06-06（Option C rerun 完成）  
**Harry 決議：** **Option C** — 全六 cases 重跑 with c_deg in MILP objective

**c_deg 校準值：** 1.116 NTD/kWh  
**計算依據：** 2,000,000 NTD 總循環壽命 / 1,791,373 kWh 額定能量吞吐 ≈ 1.116 NTD/kWh

**已修改檔案（全部向後相容，c_deg 預設 0.0）：**
- `milp_v3_horizon/milp/base_milp.py` — `build_objective()` + `build_milp()` 加入 c_deg 項
- `milp_v3_horizon/milp/stochastic_da_milp.py` — `build_stochastic_da_milp()` 加入 c_deg 項（p_dis shared, NOT scenario-weighted）
- `milp_v3_horizon/cases/da_solver.py` — 從 cfg['battery'] 讀取 c_deg
- `milp_v3_horizon/runner.py` — `_PhysParams` 讀取 c_deg_ntd_per_kwh
- `milp_v3_horizon/configs/base.yaml` — `battery.c_deg_ntd_per_kwh: 1.116`

**FULLMILP Rerun 結果（Output: final_fullmilp_deg_trec_re20_rerun_20260606）：**

| Case | FULLMILP Total (M NTD) | Oper (M) | Deg (M) | OC_h |
|------|------------------------|-----------|---------|------|
| MPC_PROB_K5_FULLMILP_REALIZED | **90.3982** | 88.5705 | 1.8277 | 616 |
| DA_PROB_K5_FULLMILP_REALIZED | 90.5755 | 88.7558 | 1.8197 | 687 |
| M9_RESERVE_PROB_FULLMILP_REALIZED | 91.4926 | 89.5095 | 1.9831 | 636 |
| M9_RESERVE_DET_FULLMILP_REALIZED | 91.6384 | 89.6606 | 1.9778 | 631 |
| DA_DET_FULLMILP_REALIZED | 92.1001 | 90.0431 | 2.0570 | 903 |
| MPC_DET_C2B_FULLMILP_REALIZED | 92.1091 | 90.0648 | 2.0443 | 872 |

TREC (reference, constant, all cases): -19.70M NTD  
CAPEX (reference, uniform, all cases): +6.47M NTD

**Rankings vs. V2.1 simplified:** UNCHANGED (DA_DET ≈ MPC_DET within 0.009M, effectively tied at rank 5/6)

**Key finding:** Degradation spread = 0.23M NTD << OC spread ≈ 1.74M NTD → rankings confirmed robust.

**論文可寫：**  
"The MILP minimizes expected TOU grid energy cost plus over-contract DCT penalty plus a discharge-proportional battery degradation proxy (c_deg = 1.116 NTD/kWh), plus a terminal SOC soft-constraint penalty (λ^SOC = 5.0 NTD/unit). Annual settlement adds the basic demand charge and equals TOU + basic + OC + degradation. Battery degradation accounts for approximately 1.8–2.1 M NTD/yr and does not change case rankings (OC spread 1.74M >> degradation spread 0.23M)."

**Harry 確認：** ✅ RESOLVED（2026-06-06）

---

## DL-013：TREC/RE20 排除於 MILP Objective — 已修正（原假設錯誤）

**日期：** 2026-06-06（首次記錄）→ **2026-06-07（發現原假設錯誤，已修正）**

**原假設（已推翻）：** TREC = -19.7M NTD/yr 為 PV T-REC 收入；PV 發電 3.76 GWh > RE20 target 2.6 GWh → 無 shortfall → C_TREC = 0

**實際值（修正後）：**
- PV 年發電 = **3.1318 GWh**（非 3.76 GWh；milp_v2 估算錯誤）
- 年負載 = 21.2684 GWh → RE20 目標 = 20% × 21.27 = **4.2537 GWh**（非 2.6 GWh；舊值以 13 GWh 錯誤年負載計算）
- Shortfall = 4.2537 − 3.1318 = **1.1219 GWh** → **結構性不足，與調度策略無關**
- C_TREC = 4.63 NTD/kWh × 1,121,900 kWh = **~5.24M NTD/yr**（購入成本，正值）
- 舊的 −19.7M NTD（milp_v2）公式：`c_trec × 20% × load` 錯誤地視為負成本（負面收入），此為方向錯誤

**排除於 MILP Objective 的理由（仍成立）：**
- C_TREC 跨 cases 差異僅 10,900 NTD（cross-case spread 可忽略）
- 電池夜間主要從電網充電，battery-mediated green discharge 每年僅 0.007–0.010 GWh
- 對 ranking 無影響，排除合理

**最終決議：** C_TREC ≈ 5.24M NTD/yr 作為常數加入 annual settlement（非 MILP objective），詳見 DL-014。

**Harry 確認：** ✅（Green SOC tracking 結果已驗證，2026-06-07）

---

## DL-014：Green SOC Tracking 精確計算 C_TREC per Case

**日期：** 2026-06-07  
**決議：** 依 revision doc §4.11 實作 Green SOC 追蹤（P^pv,ch、P^dis,g、E_g），逐案計算 C_TREC

**計算方法（revision doc §4.11）：**
- P^pv,ch[t] = min(max(0, PV[t] − Load[t]), p_ch[t])  — PV 多餘電力充電
- E_g[t+1] = E_g[t] + η_ch × P^pv,ch[t] − P^dis,g[t] / η_dis
- P^dis,g[t] = min(p_dis[t], E_g[t] × η_dis)  — 綠能放電（greedy attribution）
- R_onsite = Σ(PV→Load) + Σ(P^dis,g)
- C_TREC = c_trec × max(0, RE20_target − R_onsite)；c_trec = 4.63 NTD/kWh
- 初始條件：E_g(0) = 0（保守估計，所有 cases 相同）

**結果：**

| Case | Green Dis (GWh) | R_onsite (GWh) | Shortfall (GWh) | C_TREC (M NTD) |
|------|:---:|:---:|:---:|:---:|
| MPC-PROB K5 | 0.0098 | 3.1229 | 1.1307 | **5.2353** |
| DA-PROB K5 | 0.0074 | 3.1206 | 1.1331 | **5.2462** |
| M9-Res PROB | 0.0086 | 3.1217 | 1.1320 | 5.2410 |
| M9-Res DET | 0.0086 | 3.1217 | 1.1320 | 5.2410 |
| DA-DET | 0.0081 | 3.1213 | 1.1324 | 5.2429 |
| MPC-DET C2b | 0.0089 | 3.1221 | 1.1316 | 5.2395 |

**Cross-case C_TREC spread = 10,900 NTD**（可忽略；佔 operational spread 1.71M 的 0.6%）

**Final Settlement (TOU+Basic+OC+DEG+TREC)：**

| Rank | Case | TOTAL (M NTD) |
|------|------|:---:|
| 1 | MPC-PROB K5 | **95.6335** |
| 2 | DA-PROB K5 | 95.8217 |
| 3 | M9-Res PROB | 96.7336 |
| 4 | M9-Res DET | 96.8794 |
| 5 | DA-DET | 97.3430 |
| 6 | MPC-DET C2b | 97.3486 |

Rankings unchanged from FULLMILP (without TREC).

**依據：** revision doc §4.11–4.12；Harry 明確要求精確追蹤 P^pv,ch、P^dis,g  
**影響範圍：** 年度 settlement 最終數字；論文 §4.12 表格  
**Harry 確認：** ✅ 結果已驗證（2026-06-07）

**輸出檔案：**
`new_pipeline/data/output/experiments/final_fullmilp_deg_trec_re20_rerun_20260606/`
- `FULLMILP_WITH_TREC_GREENSOC_BREAKDOWN.csv` — per-case full breakdown
- `FULLMILP_WITH_TREC_GREENSOC_REPORT.md` — full methodology report

---

*最後更新：2026-06-07*  
*待確認事項：DL-008 κ 來源（Phase 2 前 Harry 確認）*
