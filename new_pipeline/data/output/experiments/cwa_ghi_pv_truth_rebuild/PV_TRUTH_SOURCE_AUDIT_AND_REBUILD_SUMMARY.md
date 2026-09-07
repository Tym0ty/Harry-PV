# PV Truth Source Audit and CWA-GHI Rebuild Summary

## Executive Summary

本次 PV 稽核結論是：

1. **PV/GHI 預測端的 GHI-to-PV scaling 公式是正常的。**
2. **錯誤在 replay / settlement 使用的 realized PV truth 來源。**
3. 原本 `bridge_outputs_fullyear/full_year_replay_truth_package.parquet::pv_realized_kw` 來自 `NTUST_Load_PV.csv::Solar_kWh` 經比例放大；但 `NTUST_Load_PV.csv` 的 PV 欄位已確認錯誤，因此該 replay PV truth 不可再作正式 KPI / settlement 依據。
4. 已建立新的 replacement truth package，將 realized CWA GHI 轉成 PV：

```text
pv_realized_kw = clip(0.80 * ghi_realized_Wm2 / 1000 * 2687, 0, 2687)
```

新的 PV truth package：

```text
new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet
```

目前尚未覆蓋舊 truth，也尚未重跑 scheduling。

---

## 1. Original Concern

使用者發現 PV installed capacity 約為 2000+ kWp，但部分資料中 PV output 似乎只有幾百 kW。這可能是年平均 / 陰天 / 夜間平均造成，也可能是 PV scaling 或 replay truth 使用錯誤。

因此進行兩層檢查：

1. 預測端 GHI-to-PV conversion 是否正確。
2. Replay / settlement 的 realized PV truth 是否抓對資料。

---

## 2. Forecast-Side GHI-to-PV Conversion Audit

目前正式 forecast / scenario pipeline 中使用的主要 PV conversion 公式一致：

```text
PV_kW = clip(PR * GHI_Wm2 / 1000 * PV_CAP, 0, PV_CAP)
PR = 0.80
PV_CAP = 2687 kWp
```

主要 code evidence：

```text
milp_v2/config.yaml
  pv_system.pv_cap_kwp = 2687
  pv_system.pr = 0.80

new_pipeline/step_v4_eval.py
new_pipeline/step_nwp6_final.py
new_pipeline/scripts/experiments/pv_focus_bridge_utils.py
```

Forecast-side PV conversion 後量級如下：

| Forecast artifact | Mean kW | Max kW | P95 kW | Notes |
|---|---:|---:|---:|---|
| DA q50 PV, full 8760h | 329.42 | 2147.45 | 1519.61 | includes night hours |
| DA q95 PV, full 8760h | 524.56 | 2275.05 | 1990.26 | upper forecast |
| ID H1 q50 PV | 681.79 | 2252.61 | 1818.78 | daylight / available ID rows |
| ID H1 q95 PV | 1403.95 | 2546.62 | 2452.65 | upper forecast |
| DA PV-focused TKM-safe K5 | 316.78 | 2254.47 | 1556.38 | reduced scenarios |
| ID H24 PV-focused TKM-safe K5 | 347.30 | 2546.62 | 1701.04 | reduced scenarios |

判斷：

```text
Forecast-side PV scaling is code-consistent and physically plausible.
```

全年平均只有 300–350 kW 是合理的，因為包含夜間與低日照時段；晴天或高分位 PV 可達 2.1–2.55 MW。

---

## 3. Replay PV Truth Source Error

稽核發現原 replay truth package 的 PV 來源為：

```text
notebooks_bridge/bridge_full_year.py
  DATA_CSV = ROOT / "NTUST_Load_PV.csv"
  solar_val = row.get("Solar_kWh", 0.0)
  pv_realized = Solar_kWh * NTUST_PV_SCALE

bridge_outputs_fullyear/bridge_run_metadata.json
  solar_source = "NTUST_Load_PV.csv Solar_kWh (for replay truth only)"
```

也就是：

```text
NTUST_Load_PV.csv::Solar_kWh
    -> scaled by NTUST_PV_SCALE = 2687 / 379
    -> bridge_outputs_fullyear/full_year_replay_truth_package.parquet::pv_realized_kw
```

但使用者已確認：

```text
NTUST_Load_PV.csv 裡的 PV / Solar_kWh 欄位是錯誤的，應整欄忽略。
```

因此：

```text
bridge_outputs_fullyear/full_year_replay_truth_package.parquet::pv_realized_kw
is tainted and should not be used as formal realized PV truth.
```

---

## 4. Affected Scripts and Outputs

### Directly affected source builder

```text
notebooks_bridge/bridge_full_year.py
```

This script directly reads `NTUST_Load_PV.csv::Solar_kWh` and generates `pv_realized_kw`.

### Indirectly affected pipelines

Any replay / KPI / scheduling script using:

```text
bridge_outputs_fullyear/full_year_replay_truth_package.parquet
```

is affected if it uses `pv_realized_kw`.

Examples include:

```text
milp_v2/config.yaml
milp_v2/bridge/build_packages.py
milp_v2/replay/replay_settlement.py
milp_v2/layer_b/run_layer_b_mpc.py
milp_v2/experiments/mpc_fixed_rolling_horizon/mpc_fixed_horizon_utils.py
new_pipeline/scripts/experiments/run_scheduling_literr_tkm_safe_da_mpc.py
new_pipeline/scripts/experiments/run_scheduling_pv_focus_da_mpc.py
many milp_v2/experiments/load_forecast_rerun/*.py replay/audit scripts
```

Implication:

```text
Previous realized-basis annual cost results should be treated as not final until replay truth is rebuilt and relevant cases are rerun.
```

---

## 5. Replacement PV Truth: Realized CWA GHI to PV

New methodological decision:

```text
Valid PV truth should be derived from realized CWA GHI.
```

Implemented conversion:

```text
pv_realized_kw = clip(0.80 * ghi_realized_Wm2 / 1000 * 2687, 0, 2687)
```

Input GHI source:

```text
new_pipeline/data/output/stage1_aci_calibrated_da_v2.parquet
  column: ghi_realized
```

Generated replacement truth package:

```text
new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet
```

This package preserves the existing load/calendar fields and replaces PV truth with CWA-GHI-derived PV.

---

## 6. New PV Truth Summary

| Series | Min kW | Max kW | Mean kW | P95 kW | P99 kW | Annual Energy kWh | Capacity Factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| CWA_GHI_PV_TRUTH | 0.00 | 2298.88 | 328.37 | 1671.91 | 2054.06 | 2,876,559 | 12.22% |
| Old invalid NTUST solar scaled | 0.00 | 2531.03 | 357.51 | 1843.68 | 2240.35 | 3,131,808 | 13.31% |

Difference:

```text
New CWA-GHI PV truth annual energy is about 255 MWh lower than old invalid scaled NTUST PV.
```

This comparison is only a magnitude check. The old PV is invalid and should not be cited as ground truth.

---

## 7. Current Artifact Status

| Artifact | Status |
|---|---|
| `NTUST_Load_PV.csv::Solar_kWh` | INVALID; ignore completely |
| `bridge_outputs_fullyear/full_year_replay_truth_package.parquet::pv_realized_kw` | TAINTED; generated from invalid PV source |
| `new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet` | Replacement PV truth package |
| Forecast-side GHI-to-PV quantiles/scenarios | Formula is code-consistent; not invalidated by this replay truth issue |

---

## 8. What Can Still Be Used

The following remain usable:

1. PV / GHI forecast model outputs, as forecast artifacts.
2. GHI-to-PV conversion formula.
3. PV scenario generation logic, subject to using the correct realized PV only for validation/replay.
4. Load data and load forecast artifacts, unless separately invalidated.

The following should not be used as final evidence until rebuilt:

1. Any realized-basis cost result using old `pv_realized_kw`.
2. Any PV forecast validation metric using old `pv_realized_kw` as actual PV.
3. Any net-load realized KPI using old `load_realized_kw - pv_realized_kw`.

---

## 9. Recommended Next Steps

1. Patch replay / scheduling runners to accept:

```text
new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet
```

2. Rerun necessary replay / scheduling cases using the corrected PV truth.

3. Recompute:

```text
annual_cost_summary
monthly_cost_summary
OC / DCT summary
SOC behavior
forecast validation against realized PV
net-load scenario validation
```

4. Do not update thesis final tables until reruns with CWA-GHI PV truth are complete.

---

## 10. Generated Files

```text
new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/
  full_year_replay_truth_package_CWA_GHI_PV.parquet
  CWA_GHI_PV_TRUTH_REBUILD_REPORT.md
  cwa_pv_truth_summary.csv
  cwa_pv_truth_monthly_summary.csv
  cwa_pv_truth_vs_old_invalid_pv_comparison.csv
  pv_truth_replacement_status.csv
  PV_TRUTH_SOURCE_AUDIT_AND_REBUILD_SUMMARY.md
```

Related invalidation report:

```text
new_pipeline/data/output/experiments/pv_scaling_ghi_to_pv_audit/PV_SOURCE_INVALIDATION_ADDENDUM.md
```

---

## Final Status

```text
PV_FORECAST_SCALING_OK_BUT_REPLAY_TRUTH_REBUILT_REQUIRED
```

