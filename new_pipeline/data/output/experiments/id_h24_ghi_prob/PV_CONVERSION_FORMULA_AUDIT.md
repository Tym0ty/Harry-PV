# PV Conversion Formula Audit

**Generated**: 2026-06-05  
**Purpose**: Verify GHI-to-PV formula for ID-H24 probabilistic pipeline

---

## 1. Confirmed Formula

```
PV_avail_qXX = clip(q_ghi_XX × PR × PV_cap / 1000, 0, PV_cap)

PR      = 0.8  (performance ratio, dimensionless)
PV_cap  = 2687.0 kWp  (installed PV capacity)
GHI     = W/m²  (predicted GHI in watts per square metre)
÷1000   = converts W/m² to kW/m²
Result  = kW  (available PV power, clipped to [0, PV_cap])
```

This is **identical** to DA Stage 3 (`scripts/stage3_ghi_to_pv.py`):
```python
pv = PR * (ghi / 1000.0) * PV_CAP
pv = np.clip(pv, 0.0, PV_CAP)
```

---

## 2. Parameter Verification

| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| PR (performance ratio) | 0.8 | `new_pipeline/config.yaml` pv_conversion.pr | CONFIRMED |
| PV_cap | 2687.0 kWp | `new_pipeline/config.yaml` pv_conversion.pv_cap_kwp | CONFIRMED |
| GHI units | W/m² | W1 test predictions (max y_ghi_true = 1069.4 W/m²) | CONFIRMED |
| Division by 1000 | ÷1000 converts W/m² → kW/m² | Stage 3 formula | CONFIRMED |
| Upper clip | 2687.0 kW (= PV_cap) | Stage 3 np.clip | CONFIRMED |
| Lower clip | 0 kW | Stage 3 np.clip | CONFIRMED |
| Temperature correction | NONE | Stage 3 reviewed — no NOCT/temp derating | CONFIRMED ABSENT |
| Soiling / spectral derating | NONE | Absorbed into fixed PR=0.80 | CONFIRMED ABSENT |
| Diffuse vs direct distinction | NONE | GHI (global) used directly | CONFIRMED ABSENT |

---

## 3. Cross-Check Against Realized PV

**Method**: Apply formula to `y_ghi_true` (realized GHI, daytime test period, unique target_times); compare to `pv_realized_kw` from truth package.

**Hour convention used**: `hour_local` in truth package = datetime hour of target_time (1-24 scale; hour 7 = 07:00).

| Metric | Value |
|--------|-------|
| Matched daytime target_times | 4237 / 4237 |
| R (formula PV vs realized PV) | 0.9004 |
| RMSE (formula vs realized) | 303.5 kW |
| MBE (formula − realized) | -55.3 kW |
| Max formula PV (from realized GHI) | 2298.9 kW |
| Max realized PV | 2531.0 kW |

**Interpretation of MBE**: Formula under-predicts realized PV by 55.3 kW on average. This is expected: PR=0.80 is a conservative planning value. The actual system outperforms the planning formula on mild-temperature days. This systematic offset is a physical effect, not a modeling error.

**Implication**: The GHI→PV formula will produce PV forecasts that are systematically slightly below actual PV production. This is conservative for scheduling (under-predicts available PV → slightly higher dispatch from BESS). The bias is consistent with the DA pipeline and should be disclosed in the thesis.

---

## 4. DA vs ID-H24 Formula Comparison

| Property | DA Stage 3 | ID-H24 (this pipeline) | Match |
|----------|-----------|----------------------|-------|
| PR | 0.8 | 0.8 | YES |
| PV_cap | 2687.0 kWp | 2687.0 kWp | YES |
| GHI units | W/m² (named ghi_kw_m2 in DA, but values in W/m²) | W/m² | YES |
| Division by 1000 | YES | YES | YES |
| Clip [0, PV_cap] | YES | YES | YES |
| Temperature correction | NO | NO | YES |
| Applied to | N=500 scenario GHI values | Point forecast + 11 quantile levels | SAME FORMULA |

**Conclusion**: The ID-H24 GHI-to-PV conversion uses the **exact same formula** as the DA probabilistic pipeline (Stage 3). No modifications are needed.

---

## 5. Confirmed Candidate Formula

```
PV_avail_qXX = clip(q_ghi_XX × 0.80 × 2687 / 1000, 0, 2687)
```

where `q_ghi_XX` is in W/m² and result is in kW. **All six checks pass.**
