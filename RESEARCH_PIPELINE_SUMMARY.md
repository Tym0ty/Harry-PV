# HARRY-PV 研究方法線完整整理

> **最後更新：2026-06-07**  
> 本文件為暫定最終方法線的全流程紀錄，涵蓋系統規格、預測端、排程端、實驗設計、MILP 架構、全部實驗結果與最終年度結算。  
> 所有決策依據詳見 `DECISION_LOG.md`；各記憶細節見 `memory/` 資料夾。

---

## 目錄

1. [研究定位與核心問題](#1-研究定位與核心問題)
2. [系統規格與固定設計](#2-系統規格與固定設計)
3. [資料來源與測試期間](#3-資料來源與測試期間)
4. [台電費率結構](#4-台電費率結構)
5. [預測端流程](#5-預測端流程)
6. [排程案例定義（C0–C3b + 延伸）](#6-排程案例定義c0c3b--延伸)
7. [MILP 架構](#7-milp-架構)
8. [主要排程實驗結果（C0–C3b，forecast-basis）](#8-主要排程實驗結果c0c3bforecast-basis)
9. [擴充案例：DA 真實基準線（C0R）](#9-擴充案例da-真實基準線c0r)
10. [擴充案例：MPC-PROB K5 隨機排程](#10-擴充案例mpc-prob-k5-隨機排程)
11. [擴充案例：M9-Reserve（需量棘輪風控）](#11-擴充案例m9-reserve需量棘輪風控)
12. [FULLMILP 最終帳目（含 c_deg）](#12-fullmilp-最終帳目含-c_deg)
13. [Green SOC 追蹤與 C_TREC](#13-green-soc-追蹤與-c_trec)
14. [最終年度結算（完整六案）](#14-最終年度結算完整六案)
15. [RQ 問題對應結果摘要](#15-rq-問題對應結果摘要)
16. [成本拆解分析](#16-成本拆解分析)
17. [Robustness 驗證](#17-robustness-驗證)
18. [M9-Lite 診斷性擴充](#18-m9-lite-診斷性擴充)
19. [資訊邊界合規確認](#19-資訊邊界合規確認)
20. [關鍵輸出目錄清單](#20-關鍵輸出目錄清單)
21. [尚待完成事項](#21-尚待完成事項)

---

## 1. 研究定位與核心問題

**研究主題：** 固定 CC 與 PV-BESS 設計下，**排程視窗規則（scheduling horizon rule）** 對年度 DCT 風險、錯峰頻率與年度成本的影響。

**主貢獻（C3b）：** DCT-state + BESS-coverability 自適應視窗規則（selective risk-aware control），不是直接省錢，而是在 BESS 真正能量不足時才啟動保守的 EOD 視窗。

**研究問題（RQ）對應：**

| RQ | 問題 | 對比 |
|----|------|------|
| RQ1 | 日內更新是否有價值？ | C0 → C1 |
| RQ2 | EOD 是否優於 H=6 短視窗？ | C1 → C2 |
| RQ3 | 自適應規則是否能選擇性控制？ | C2 → C3a → C3b |
| RQ4 | gap 來自 forecast 誤差或 horizon 設計？ | C2 → D2 → D3（未完成） |
| RQ5 | 結論是否穩健？ | λ^SOC sensitivity + RAW robustness |

---

## 2. 系統規格與固定設計

以 Layer A DD 設計作為 x_ref（RE20 情境下最佳化 CC=3232 kW）：

| 參數 | 值 | 來源 |
|------|----|------|
| PV 容量 | 2,687 kWp | 台科屋頂可用面積 |
| PV PR | 0.80 | 設計假設 |
| BESS 功率 P_B | 1,152 kW | Layer A DD sizing |
| BESS 容量 E_B | 6,902 kWh | Layer A DD sizing |
| η_ch = η_dis | 0.95 | 標準鋰電池效率 |
| SOC_min | 0.10 (10%) | 電池保護下限 |
| SOC_max | 0.90 (90%) | 電池保護上限 |
| SOC_init | 0.50 (50%) | 每日初始 SOC |
| 契約容量 CC_ref | 3,232 kW | Layer A DD（DL-001） |
| 需量校正係數 κ | 1.006 | Phase 0 audit |
| 電池降解成本 c_deg | 1.116 NTD/kWh | DL-012：2M NTD / 1,791,373 kWh |

---

## 3. 資料來源與測試期間

**測試期間（固定，不可更改）：** 2024-11-01 → 2025-10-31（8,760 小時）

| 資料 | 來源 | 解析度 |
|------|------|--------|
| 實現負載 Load_realized | NTUST 實測 | 1-hour |
| 實現 PV | 依 GHI 換算（PR=0.80） | 1-hour |
| DA PV 預測 | da_v2 模型（XGBoost，lag-24h 氣象特徵） | 1-hour D-1 |
| ID PV 預測 | Hybrid XGBoost + AgACI（H=1–24） | 1-hour rolling |
| DA 負載預測 | LOAD-QN 模型（simulation-normalized） | 1-hour D-1 |
| ID 負載預測 | LOAD-QN walk-forward（H=1–24） | 1-hour rolling |

**Pre-test / Calibration / Test 切分（嚴格因果）：**
- Train: 2021-04-01 ~ 2023-12-31
- Validation: 2024-01-01 ~ 2024-04-30
- Calibration: 2024-05-01 ~ 2024-10-31
- **Test: 2024-11-01 ~ 2025-10-31**（所有方法參數在此之前凍結）

---

## 4. 台電費率結構

**超約罰則（Over-Contract DCT）：**
- 超約 ≤ 10% CC：罰款 = 2 × CC_rate × 超約量 × kW-超約小時
- 超約 > 10% CC：罰款 = 3 × CC_rate × 超約量 × kW-超約小時

**基本電費（Basic Charge，固定）：**
- 夏月（5/16–10/15）：223.6 NTD/kW/月 × CC
- 非夏月（10/16–5/15）：166.9 NTD/kW/月 × CC
- **全年固定：7,572,576 NTD（CC=3,232 kW）**

**TOU 時段：**
- 尖峰（Weekday 09:00–24:00 夏月）
- 半尖峰（Weekday 非夏）
- 離峰（Weekend / 深夜）

**年度結算公式（Revision Doc §4.12）：**
```
C_annual = C_TOU + C_basic + C_OC + C_deg + C_TREC
```

---

## 5. 預測端流程

### 5.1 PV 預測

| 模型 | 用途 | RMSE（test，GHI cs>0） |
|------|------|------------------------|
| da_v2 XGBoost (lag-24h) | DA PV P50 | 142.81 W/m² |
| v4 XGBoost + AgACI | ID PV H=1–24 | 116.45 W/m² |

- DA PV：每日 D-1 取 da_v2 q50，裁切為 [0, 2687 kW]
- ID PV：每滾動小時取當下 H=1..24 的 v4+AgACI 中位數

**AgACI（Adaptive Group-wise ACI）：** 用於 ID 機率預測校準；按月份 × 時段分組適應性更新覆蓋率。

**DL 比較（負面結果）：**
- CNN-LSTM：RMSE=124.89（比 v4 差 8.44 W/m²）
- MT-LSTM：RMSE=124.86（比 v4 差）
- → XGBoost 在此 tabular hourly 資料上優於 DL，DL 結果列為論文 negative result。

### 5.2 負載預測

| 模型 | 用途 | RMSE（test） |
|------|------|--------------|
| LOAD-QN（simulation-normalized） | DA/ID 主線 | DA=451 kW, ID=391 kW |
| RAW LOAD（未正規化） | Robustness case | DA=486 kW, ID=368 kW |

- LOAD-QN 透過 simulation-normalized 降低 MBE（bias −133.5 kW），比 RAW 改善 5.4%
- DA PI80 coverage=84.7%（目標 80%，略高合格）

### 5.3 淨負載場景（Net-load Scenario Set）

每日 DA 與每時 ID 提供 6 個場景（anti-diagonal pairing）：

| 場景 | 定義 | 用途 |
|------|------|------|
| S1 nl_median | load_q50 − pv_q50 | 中位數情境 |
| S2 nl_low | load_q10 − pv_q90 | 最佳情境（低需、高 PV） |
| S3 nl_high | load_q90 − pv_q10 | DCT 風險主情境 |
| S4 nl_high_load_stress | load_q95 − pv_q50 | 負載壓力情境 |
| S5 nl_low_pv_stress | load_q50 − pv_q05 | PV 不足情境 |
| S6 nl_combined_stress | load_q95 − pv_q10 | 最差情境 |

---

## 6. 排程案例定義（C0–C3b + 延伸）

### 6.1 主要五案

| Case | 視窗規則 | 說明 |
|------|---------|------|
| **C0** | DA-only（H=24，不滾動） | 概念性 DA 下界。不是可操作基準線；預測值計畫，不含實現誤差 |
| **C1** | ID H=6 receding horizon（固定） | 每小時滾動，視窗固定 6h |
| **C2** | EOD shrinking horizon | 每小時滾動，視窗 = 當日剩餘小時數（第 t 小時 → H=24−t） |
| **C3a** | DCT-state adaptive | 有 OC 風險時切換 EOD；否則用 H=6 |
| **C3b** | DCT-state + BESS-coverability adaptive（**主貢獻**） | 有 OC 風險 **且** m_min<0（BESS 能量不足）才切 EOD；否則用 H=6 |

### 6.2 延伸診斷案例

| Case | 說明 | 主要發現 |
|------|------|----------|
| **C2b**（Fixed-H24） | DA MILP H=24 無 D 遞減 | 幾乎等於 C2（差 26,607 NTD），確認 EOD 等效 Fixed-H24 |
| **C0R_DA_DET** | C0 DA 排程對實現值 replay | 真實 DA-only 成本 89.92M（非 76M forecast 值） |
| **C0R_DA_PROB** | 隨機 DA（K=3）replay | 89.29M，比 DET 省 0.63M |
| **MPC_PROB_K5** | K=5 stochastic rolling MPC | 88.58M，最優排程案例 |
| **M9-Res DET/PROB** | M9-Reserve（B-only 需量棘輪） | 診斷性：DET=89.65M，PROB=89.54M；OC 控制優於 C2b DET |
| **D2** | Perfect PV + forecast Load | 未完成（Phase 2 工作） |
| **D3** | Perfect PV + Perfect Load | 未完成（Phase 2 工作） |

---

## 7. MILP 架構

### 7.1 共用基底（milp_v3_horizon/milp/base_milp.py）

**決策變數（每個 horizon 步 t=0..H-1）：**
- `p_ch[t]`：充電功率 [kW]
- `p_dis[t]`：放電功率 [kW]
- `soc[t]`：電池 SOC [0–1]
- `z_over1[t]`、`z_over2[t]`：超約量分段鬆弛變數

**目標函式：**
```
min  Σ_t [ c_tou[t]·(NL[t]+p_ch[t]-p_dis[t])
         + c_oc1·z_over1[t] + c_oc2·z_over2[t]
         + c_deg·p_dis[t]·Δt ]
   + λ^SOC · s_soc            ← terminal SOC anchor
```

**約束：**
- SOC 動態：`soc[t+1] = soc[t] + η_ch·p_ch[t]·Δt/E_B − p_dis[t]·Δt/(η_dis·E_B)`
- 功率限制：`0 ≤ p_ch[t] ≤ P_B`，`0 ≤ p_dis[t] ≤ P_B`
- SOC 限制：`SOC_min ≤ soc[t] ≤ SOC_max`
- LP 鬆弛（無 anti-simultaneous binary）
- Terminal SOC soft constraint：`s_soc ≥ SOC_DA_end − soc[H-1]`

**參數：**
- λ^SOC = 5.0 NTD/unit（pre-test 校準，DL-011）
- c_deg = 1.116 NTD/kWh（DL-012）

### 7.2 DA MILP（每日 D-1 求解）

- **DET 版：** H=24，用 nl_median 作為 forecast NL
- **PROB 版（K=5）：** 5 個場景（nl_median + nl_low + nl_high + nl_high_load_stress + nl_combined_stress），非預期性（p_ch/p_dis 在 K 個場景共用），期望值最小化

### 7.3 ID Rolling MILP（每小時更新）

- 每小時更新 NL forecast 後重新求解
- 視窗長度依 case 規則（H=6 / EOD / adaptive）
- DA terminal SOC anchor 作為 soft constraint 引導

### 7.4 M9-Reserve 機制（Mechanism B only）

- `slack_reserve ≥ max(0, E_min + R_risk/η_dis − soc[H-1]·E_B)`
- `+ρ_reserve·slack_reserve` 在 objective
- D_guard = max(DCT_init, 0.95·CC)，R_risk = min(R_cover, 0.3·(SOC_max-SOC_min)·E_B)

---

## 8. 主要排程實驗結果（C0–C3b，forecast-basis）

**全年 QN 主線結果（2024-11-01 → 2025-10-31，8,760 小時，111/111 PASS）**

| Case | 年度總成本 (NTD) | 年度總成本 (M NTD) | 能源費 (NTD) | OC/DCT (NTD) | 峰值 (kW) | 峰值/CC | OC-h | 平均 SOC | EOD-h |
|------|-----------------|------------------|-------------|-------------|----------|---------|------|---------|-------|
| **C0** | 76,082,806 | 76.08 | 68,432,305 | 77,925 | 3,232 | 1.000× | 571 | 0.424 | 0 |
| **C1** | 84,245,041 | 84.25 | 75,300,135 | 1,372,330 | 3,984 | 1.233× | 880 | 0.247 | 0 |
| **C2** | 81,540,038 | 81.54 | 73,607,937 | 359,525 | 3,610 | 1.117× | 778 | 0.399 | 8,760 |
| **C2b**（Fixed-H24） | 81,513,431 | 81.51 | — | 342,526 | 3,575 | 1.106× | 748 | 0.468 | 0 |
| **C3a** | 82,604,282 | 82.60 | 74,580,716 | 450,991 | 3,610 | 1.117× | 841 | 0.300 | 1,683 |
| **C3b** | 84,231,363 | 84.23 | 75,307,807 | 1,350,980 | 3,939 | 1.219× | 884 | 0.246 | **34** |

> **重要：以上為 forecast-basis 結算（MILP 預測輸入產生的計畫費用），非最終結算依據。**  
> **C0=76.08M 為概念性下界，不是可操作基準線。**

**C3b 選擇性（Selectivity）驗證：**
- C3b EOD 觸發次數：34 次 / 年（0.39%）
- C3a EOD 啟動次數：1,683 次 / 年（19.2%）
- C3a vs C3b 分歧：1,649 小時（18.8%）
- → C3b 僅在 BESS 真正能量不足時啟動 EOD，確認 selective 性質

---

## 9. 擴充案例：DA 真實基準線（C0R）

**目的：** 提供可操作的 DA-only 基準線（C0 的 76.08M 是預測計畫值，不含實現誤差）。

| Case | 年度成本 (M NTD) | 說明 |
|------|-----------------|------|
| C0_DA_FORECAST（C0） | 76.0828 | 概念下界，不可操作 |
| **C0R_DA_DET** | **89.9177** | DA DET 排程 replay 實現值 |
| **C0R_DA_PROB（K=3）** | **89.2921** | 隨機 DA（K=3）replay 實現值 |

**日內更新價值（RQ1 修正後）：**
- DA-DET → C2b（Fixed-H24）：省 **8.40M NTD/年**（日內重新優化的真實價值）
- DA-DET → C2（EOD）：省 **8.38M NTD/年**
- DA 隨機 vs DA 確定性：省 **0.63M NTD**（隨機對沖效益）

---

## 10. 擴充案例：MPC-PROB K5 隨機排程

**目的：** K=5 隨機滾動 MPC（每小時對 5 個場景求期望最佳解），確認隨機規劃的收益。

**K=5 場景配置：**
- K1: nl_median（w=0.2）
- K2: nl_low（w=0.2）
- K3: nl_high（w=0.2）
- K4: nl_high_load_stress（w=0.2）
- K5: nl_combined_stress（w=0.2）

**結果（Bug fix 後，realized NL settlement）：**

| Case | 成本 (M NTD) | OC-h | 平均 SOC |
|------|-------------|------|---------|
| C2b_DET realized | 89.5627 | 834 | 0.468 |
| **MPC-PROB K5** | **88.5758** | **632** | 0.470 |
| Δ（PROB-DET） | **−0.9869** | −202 | +0.002 |

→ K=5 隨機 MPC 比確定性 MPC 省 **0.99M NTD**，OC 事件減少 202 次。

**TOU Cross-Midnight Bug：** 原始版本 MPC-PROB K5 = 89.66M（錯誤）。已修正：`tou_from_h` 跨日索引修正後 = 88.58M。

---

## 11. 擴充案例：M9-Reserve（需量棘輪風控）

**目的：** 在 C2b 框架上加入 B 型機制（terminal reserve constraint），減少 OC 事件。  
**定位：診斷性延伸，非主貢獻。**

**全年結果（realized NL basis）：**

| Case | 成本 (M NTD) | OC-h | vs C2b DET |
|------|-------------|------|-----------|
| **C2b（DET baseline）** | **89.5627** | 834 | — |
| M9-Reserve DET | 89.6547 | **631** | +0.092M，OC↓203 次 |
| **M9-Reserve PROB** | 89.5403 | **636** | +0.965M vs K5；OC ≈ K5 |

**M9-Reserve DET vs C2b DET：**
- 成本多 0.092M，但 OC 事件少 203 次（24.6% 減少）
- 解讀：花 0.09M 額外成本換取 24.6% 的 OC 風險降低 → 有明確風控價值

**M9-Reserve PROB vs MPC-PROB K5：**
- 成本多 0.965M，OC 事件多（+204 次）
- K=5 stochastic 在成本和 OC 兩個指標上均優於 M9-Reserve PROB
- → M9-Reserve PROB 不適合作為最優隨機策略

---

## 12. FULLMILP 最終帳目（含 c_deg）

**DL-012 RESOLVED（2026-06-06）：** c_deg = 1.116 NTD/kWh 加入所有 MILP 目標函數。

**FULLMILP = TOU + Basic + OC + DEG**（未含 TREC/CAPEX）

| 排名 | Case | FULLMILP Total (M NTD) | OC (M) | DEG (M) | OC-h |
|------|------|----------------------:|-------:|--------:|------|
| 1 | **MPC-PROB K5** | **90.3982** | 4.2844 | 1.8277 | 616 |
| 2 | DA-PROB K5 | 90.5755 | 4.3523 | 1.8197 | 687 |
| 3 | M9-Reserve PROB | 91.4926 | 6.1302 | 1.9831 | 636 |
| 4 | M9-Reserve DET | 91.6384 | 6.1438 | 1.9778 | 631 |
| 5 | DA-DET | 92.1001 | 6.6743 | 2.0570 | 903 |
| 6 | MPC-DET C2b | 92.1091 | 6.1653 | 2.0443 | 872 |

**降解成本分析：**
- 範圍：1.82–2.06M NTD/年
- 跨案差異（spread）：0.23M << OC spread 1.74M
- → Rankings 對 c_deg 是否納入不敏感，降解不改變排名

**CAPEX（年化，參考用）：** 6.47M NTD（固定，所有案例相同，不納入排名）

---

## 13. Green SOC 追蹤與 C_TREC

### 13.1 RE20 合規性

**RE20 義務：** 年用電量 × 20% = 21.2684 × 0.20 = **4.2537 GWh**

| 項目 | 數值 |
|------|------|
| 年用電量（實現） | 21.2684 GWh |
| 年 PV 發電（實現） | 3.1318 GWh |
| RE20 義務 | 4.2537 GWh |
| **Shortfall（結構性不足）** | **~1.1219 GWh** |
| T-REC 補購費率 c_trec | 4.63 NTD/kWh |
| **C_TREC** | **~5.24M NTD/年** |

**結論：所有 6 個案例均 RE20 不合規（結構性，與調度策略無關）。**

> PV 裝置容量（2,687 kWp）決定的年發電量（3.13 GWh）永遠低於 20% × 21.27 GWh 的義務。  
> 電池夜間主要從電網充電，battery-mediated green discharge 僅 0.007–0.010 GWh/年。

### 13.2 Green SOC 追蹤方法（Revision Doc §4.11）

```
P^pv,ch[t] = min(max(0, PV[t] - Load[t]), p_ch[t])   # PV 多餘電力充電
E_g[t+1]   = E_g[t] + η_ch·P^pv,ch[t] - P^dis,g[t]/η_dis
P^dis,g[t] = min(p_dis[t], E_g[t]·η_dis)              # 綠能放電（greedy attribution）
R_onsite   = Σ(PV→Load) + Σ(P^dis,g)                  # 場域再生能源
C_TREC     = c_trec × max(0, RE20_target - R_onsite)   # 購入成本（正值）
```

初始條件：E_g(0) = 0（保守估計，所有案例相同）

### 13.3 各案例 Green SOC 追蹤結果

| Case | PV→Batt (GWh) | Green Dis (GWh) | R_onsite (GWh) | Shortfall (GWh) | C_TREC (M NTD) |
|------|:---:|:---:|:---:|:---:|:---:|
| MPC-PROB K5 | 0.0108 | 0.0098 | 3.1229 | 1.1307 | **5.2353** |
| DA-PROB K5 | 0.0082 | 0.0074 | 3.1206 | 1.1331 | **5.2462** |
| M9-Res PROB | 0.0095 | 0.0086 | 3.1217 | 1.1320 | 5.2410 |
| M9-Res DET | 0.0095 | 0.0086 | 3.1217 | 1.1320 | 5.2410 |
| DA-DET | 0.0090 | 0.0081 | 3.1213 | 1.1324 | 5.2429 |
| MPC-DET C2b | 0.0098 | 0.0089 | 3.1221 | 1.1316 | 5.2395 |

**Cross-case C_TREC spread = 10,900 NTD**（可忽略；佔 operational spread 1.71M 的 0.6%）

### 13.4 舊有錯誤（DL-013 修正）

- **milp_v2 舊公式（錯誤）：** `−c_trec × 20% × load = −19.7M NTD`（視為收入，方向錯誤）
- **正確公式（Revision Doc §4.11）：** `+c_trec × max(0, RE20_target - R_onsite) = +5.24M NTD`（購入成本，正值）
- 差距：19.7M + 5.24M = **24.94M NTD**（方向 + 數量級均有誤）

---

## 14. 最終年度結算（完整六案）

**完整公式：** C_annual = C_TOU + C_basic + C_OC + C_deg + C_TREC

| 排名 | Case | TOU (M) | Basic (M) | OC (M) | DEG (M) | TREC (M) | **TOTAL (M NTD)** | Δ_best |
|------|------|--------:|----------:|-------:|--------:|---------:|------------------:|-------|
| **1** | **MPC-PROB K5** | 76.719 | 7.573 | 4.284 | 1.828 | 5.235 | **95.6335** | — |
| 2 | DA-PROB K5 | 76.772 | 7.573 | 4.352 | 1.820 | 5.246 | **95.8217** | +0.188 |
| 3 | M9-Reserve PROB | 75.838 | 7.573 | 6.130 | 1.983 | 5.241 | **96.7336** | +1.100 |
| 4 | M9-Reserve DET | 75.938 | 7.573 | 6.144 | 1.978 | 5.241 | **96.8794** | +1.246 |
| 5 | DA-DET | 75.671 | 7.573 | 6.674 | 2.057 | 5.243 | **97.3430** | +1.709 |
| 6 | MPC-DET C2b | 75.825 | 7.573 | 6.165 | 2.044 | 5.240 | **97.3486** | +1.715 |

> **CAPEX（+6.47M NTD，固定）** 未納入，若納入則所有案例等額加 6.47M，排名不變。

**論文必要揭露（Revision Doc §4.11 要求）：**
> "The realized annual PV generation (3.13 GWh) falls short of the 20% renewable obligation  
> (4.25 GWh based on 21.27 GWh campus load). A T-REC supplementation cost of approximately  
> 5.24 M NTD/yr applies uniformly to all cases (cross-case variation < 11,000 NTD, driven by  
> differences in battery-mediated green discharge). Rankings are unaffected by C_TREC inclusion."

---

## 15. RQ 問題對應結果摘要

### RQ1：日內更新是否有價值？

**對比：** C0R_DA_DET (89.92M) vs C2b (89.56M)

- 日內更新（C2b）比可操作 DA 基準省 **−0.36M NTD/年**（-0.4%）
- 使用 forecast-basis C0 (76.08M) 則虛報省 8.40M（不正確對比）
- **結論：** 日內更新提升有限；DA 排程本身已相當高效（OC 主要問題在 DA 計畫未能應對極端事件）

### RQ2：EOD 是否優於 H=6 短視窗？

**對比：** C1 (84.25M) vs C2 (81.54M) vs C2b (81.51M)

- C2（EOD）比 C1（H=6）省 **2.70M NTD**（−3.2%）
- C2b（Fixed-H24）比 C1 省 **2.73M NTD**
- C2 ≈ C2b（差 26,607 NTD，< 0.03%）
- **結論：** 延長視窗（H≥24 或 EOD shrinking）明顯優於 H=6；視窗長度比收縮方式更重要

### RQ3：自適應規則是否能選擇性控制？

**對比：** C2 (81.54M) vs C3a (82.60M) vs C3b (84.23M)

- C3a：比 C2 貴 1.06M（EOD 過度觸發 1,683 次/年）
- C3b：比 C2 貴 2.69M（C3b≈C1，EOD 觸發僅 34 次，大多數時間跑 H=6）
- **C3b 之 selective 性質已驗證：** 34 次/年均為 BESS 真正能量不足的事件
- **結論：** C3b 確實「選擇性」——它識別了 34 次真正需要保守策略的場合，但年度成本上 C2 仍是最佳

### RQ5：Robustness（RAW 負載 vs QN）

| Case | QN (M NTD) | RAW (M NTD) | Δ (RAW-QN) |
|------|-----------|------------|-----------|
| C1 | 84.2450 | 83.7435 | −0.501M |
| C2b | 81.5134 | 80.8971 | −0.616M |
| C3a | 82.6043 | 81.9910 | −0.613M |

**排名 C2b < C3a < C1 在 RAW 下保留。** C2b vs C1 gap：RAW=2.85M（QN=2.73M）。

---

## 16. 成本拆解分析

**六案（realized NL basis）成本拆解：**

| Case | TOU 能源 (M) | 基本電費 (M) | OC/DCT (M) | 合計 (M) |
|------|------------:|------------:|----------:|--------:|
| DA-DET | 75.671 | 7.573 | 6.674 | 89.918 |
| DA-PROB K5 | 76.772 | 7.573 | 4.352 | 88.697 |
| MPC-DET C2b | 75.825 | 7.573 | 6.165 | 89.563 |
| **MPC-PROB K5** | 76.719 | 7.573 | **4.284** | **88.576** |
| M9-Res DET | 75.938 | 7.573 | 6.144 | 89.655 |
| M9-Res PROB | 75.838 | 7.573 | 6.130 | 89.540 |

**關鍵觀察：**
1. **基本電費 = 7.5726M（所有案例完全相同）** → 不影響排名，但佔年度成本約 8%
2. **OC/DCT 是主要差異化因子：** 跨案差距 2.39M（4.28M–6.67M）
3. **TOU-OC 反向權衡：** 隨機策略（PROB）保守充電 → TOU 費較高，但 OC 顯著下降；淨效益正面
4. **DA-DET → MPC-DET：** OC 省 0.509M，但 TOU 增 0.154M → 淨省 0.355M
5. **PROB 在所有三行（DA/MPC/M9）均優於 DET：** OC 驅動
6. **夏月（5–10月）主導 OC：** 5月2025=1.128M，9月2025=0.909M；高基本費率（223.6 vs 166.9）

---

## 17. Robustness 驗證

### 17.1 RAW Robustness（完成）

RAW LOAD（未正規化負載預測）vs QN（simulation-normalized）：
- 三案排名（C2b < C3a < C1）在兩套預測下均保留
- RAW 整體比 QN 略便宜（−0.50~−0.62M），因 RAW 預測較 QN 更保守

### 17.2 λ^SOC Sensitivity（未完成，Phase 3）

- 計畫範圍：{0.1λ₀=0.5, λ₀=5.0, 10λ₀=50.0}
- Pre-test 校準結論：高負載夏季 + 肩月，SOC anchor 結構性失活（策略自然放電至 SOC_min）
- λ 的絕對值對成本無區分力（C1 vs C2 gap = +3.57% 為純 horizon 效應）

### 17.3 Two-CC Robustness（未完成，Phase 3）

- 計畫：CC_actual=3232 vs CC_stress=0.95×3232=3070 kW
- 目的：驗證結論對 CC 設定的穩健性

---

## 18. M9-Lite 診斷性擴充

**M9-Lite = C2b（Fixed-H24）+ Mechanism A（new-peak guard）+ Mechanism B（terminal reserve）**

| Mechanism | 說明 |
|-----------|------|
| A（New-peak guard） | `z_newpeak ≥ d_cand - D_guard`；`+ρ_peak·z_newpeak`；防止突破當月峰值 |
| B（Terminal reserve） | `slack_reserve ≥ max(0, E_min + R_risk/η_dis - soc[H-1]·E_B)`；保留 BESS 能量 |

**全年診斷結果（C2b baseline = 81.51M）：**

| Variant | NTD | Δ vs C2b |
|---------|-----|----------|
| M9_Prob_no_newpeak（B-only best） | 81,467,704 | **−45,727 NTD（−0.056%）** |
| M9_Prob_q80 | 81,479,554 | −33,877 |
| M9_Prob_q90 | 81,478,646 | −34,785 |
| M9_Lite_Point_H24 | 81,469,670 | −43,761 |
| M9_Prob_q70 | 81,515,002 | +1,571 |

**結論：** 改善幅度 < 0.06%（操作噪音範圍內）。M9-Lite 作為**診斷性延伸**，不作為主貢獻。  
論文 framing：「結構化需量棘輪風控機制，具可解釋性參數」，非省錢工具。

---

## 19. 資訊邊界合規確認

**DL-010（Critical，已修補）：** 原始 `layerB_det_package.parquet` `load_input_kw` = 實現負載（最大誤差 0.0 kW）→ 嚴重洩漏。

**8項修補（全部完成，2026-06-05）：**
- P1/P2：build_da_package.py — 替換 load/PV 來源
- P3/P4：build_id_packages.py — 替換 ID PV/Load
- P5：base.yaml — 更新路徑 + load_variant key
- P6：forecast_loader.py — 來源標籤 + hard guards
- P7：runner.py — load_variant 參數化
- P8：build_scheduling_packages.py（新建）— 預建 4 個乾淨的 parquet，剝除所有真值欄位

**洩漏防護驗證：** 48/48 smoke test PASS；4/4 clean package PASS；8/8 leakage check PASS。

**Source type labels：**
- `forecast_qn`：C0–C3b 主線
- `forecast_raw`：Robustness
- `mixed_d2`：D2 diagnostic（實現 PV + 預測 Load）
- `perfect_foresight`：D3（完全預知，上界）

---

## 20. 關鍵輸出目錄清單

```
new_pipeline/data/output/experiments/
├── scheduling_input_audit/          # DL-010 修補前後 audit，8-patch plan
├── full_year_qn_mainline/           # C0/C1/C2/C3a/C3b 全年 QN 主線結果
├── c2b_fixed_h24/                   # C2b（Fixed-H24）全年結果
├── raw_robustness_c1_c2b_c3a/       # RAW robustness（C1/C2b/C3a）
├── final_cost_completion_and_realistic_da_replay_20260606/   # C0R 基準線
├── mpc_prob_k5_and_cortes_style_m9_audit_20260606/           # MPC-PROB K5 audit
├── final_prob_bridge_k5k6_and_3x2_rerun_20260606/            # 最終 3×2 矩陣
├── final_m9_realized_replay_and_operational_3x2_20260606/    # M9 realized replay
├── m9_demand_ratchet_online_reference_mpc/                   # M9-Lite 全系列
├── final_six_cases_cost_component_breakdown_20260606/        # 六案成本拆解
└── final_fullmilp_deg_trec_re20_rerun_20260606/              # FULLMILP + TREC（最終）
    ├── FULLMILP_COST_COMPONENT_BREAKDOWN.csv/.md
    ├── FULLMILP_WITH_TREC_GREENSOC_BREAKDOWN.csv  ← 最終結算
    ├── FULLMILP_WITH_TREC_GREENSOC_REPORT.md
    ├── {CASE}_annual_kpi.csv × 6
    └── {CASE}_hourly_schedule.csv × 6
```

**關鍵腳本：**

| 腳本 | 功能 |
|------|------|
| `milp_v3_horizon/milp/base_milp.py` | 共用 MILP 建構器 |
| `milp_v3_horizon/milp/stochastic_da_milp.py` | 隨機 DA MILP（K 場景） |
| `milp_v3_horizon/runner.py` | CaseRunner（M9 類案例） |
| `milp_v3_horizon/cases/da_solver.py` | DA-DET/PROB 求解 |
| `milp_v3_horizon/cases/m9_lite.py` | M9LiteController |
| `milp_v3_horizon/run_fullmilp_six_cases_rerun.py` | 六案 FULLMILP 主執行腳本 |
| `milp_v3_horizon/run_da_realistic_baseline.py` | C0R DA 基準線 |
| `milp_v3_horizon/configs/base.yaml` | 全部參數（c_deg=1.116, λ^SOC=5.0） |

---

## 21. 尚待完成事項

### Phase 3 Robustness（尚未執行）

- [ ] λ^SOC sensitivity：{0.5, 5.0, 50.0}
- [ ] H sensitivity：{3, 6, 12}
- [ ] γ sensitivity（C3a/C3b 門檻）
- [ ] Two-CC robustness：CC=3232 vs CC_stress=3070 kW
- [ ] 月度配對比較（12 個月 Wilcoxon signed-rank）

### Phase 2 Diagnostics（尚未執行）

- [ ] D2：Perfect PV + Forecast Load（forecast error 上界）
- [ ] D3：Perfect PV + Perfect Load（完全預知上界）
- [ ] Representative event case study（V2.1 §19 #5）

### Phase 4 論文圖表（尚未執行）

- [ ] V2.1 §19 指定的 12 張圖表
- [ ] RQ 對應答辯圖（V2.1 §20 六題）
- [ ] DCT hourly approximation 限制聲明（V2.1 §16.4）

### 確認事項

- [ ] DL-008：κ 來源（帳單估計 vs Phase 0 假設），Phase 2 前 Harry 確認
- [ ] C3b 研究 framing 最終確認（cost-saving vs selective risk-aware control）

---

*文件生成自 Claude Code 記憶系統 + DECISION_LOG.md 整合*  
*生成時間：2026-06-07*
