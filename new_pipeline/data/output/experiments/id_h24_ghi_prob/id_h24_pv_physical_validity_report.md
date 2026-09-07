# ID-H24 PV Physical Validity Report

**Generated**: 2026-06-05  
**Source**: Q2_LG_cal GHI quantiles → PV via clip(0.80 × q_ghi/1000 × 2687, 0, 2687)

---

## 1. Pre-Clip Negatives (Raw GHI Quantile Level)

| Quantile | n_negative GHI rows | Notes |
|----------|--------------------|---------|
| (none) | 0 | No negative GHI predictions |

---

## 2. Post-Clip Summary

| Check | Count | Notes |
|-------|-------|-------|
| PV q50 clipped to 0 (zero PV) | 55 (0.05%) | Night hours and low-light periods |
| PV q95 clipped to PV_cap (2687.0 kW) | 0 (0.00%) | Upper capacity constraint |
| PV q50 at PV_cap | 0 (0.00%) | Very high irradiance cases |
| Max PV q95 | 2546.62 kW | Should be ≤ 2687.0 kW: OK |
| Max PV q50 | 2252.61 kW | Should be ≤ 2687.0 kW: OK |

---

## 3. Nighttime False Positives

Rows with ghi_clear_target < 5 W/m² and pv_q50 > 1 kW: 0

Result: **PASS** — no nighttime false positive PV predictions.

---

## 4. Quantile Crossing After Conversion

Rows with crossing among pv_q05 through pv_q95: 0

Result: **PASS** — no quantile crossings after PV conversion.

---

## 5. Interval Width Check

| Check | PI80 | PI90 |
|-------|------|------|
| Calibrated MPIW (kW) | 659.61 | 1170.50 |
| Mean true PV | 678.68 kW | — |
| MPIW / mean PV | 0.97 | — |
| Assessment | ACCEPTABLE (<2×) | — |

---

## 6. Summary

| Check | Result |
|-------|--------|
| Pre-clip negatives | NONE |
| Max PV ≤ 2687.0 kW | PASS |
| No nighttime false PV | PASS |
| No quantile crossings | PASS |
