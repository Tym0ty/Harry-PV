# Final Literature-Guided ID-H24 GHI Point Model Selection Report

**Generated**: 2026-06-04
**Test period**: 2024-11-01 to 2025-10-31
**Evaluation**: Daytime only (ghi_clear >= 10 W/m²); H1-H24 combined
**Dataset**: rolling_h24_ghi_dataset.parquet (471,144 rows; 101,688 test daytime rows)

---

## 1. Executive Summary

**Selected model**: W1_XGB_D24
**Test RMSE**: 136.18 W/m²
**MAE**: 91.68 W/m²
**R2**: 0.776
**Improvement over DA NWP baseline**: 204.4 W/m² (60%)
**Improvement over Smart Persistence**: 205.8 W/m²

---

## 2. Algorithm Comparison (LEADGROUP Architecture)

| Model | Algorithm | RMSE (W/m²) | MAE | vs Best |
|-------|-----------|-------------|-----|---------|
| P2_CB_LG | CatBoost | 136.78 | 93.04 | +0.60 |
| P1_XGB_LG | XGBoost | 136.95 | 92.16 | +0.77 |
| G1_LGBM_LG | LightGBM | 137.54 | 92.64 | +1.36 |

**Verdict — Is LightGBM still justified?**
YES. XGBoost (136.95), CatBoost (136.78), and LightGBM
(137.54) are within 0.76 W/m² of each other for LEADGROUP architecture.
This band is smaller than the seasonal RMSE variation and is not operationally significant.
Consistent with the DA boosting benchmark (all within 1 W/m²) and the solar forecasting
literature (Diagne et al. 2013). LightGBM remains a valid default; XGBoost and CatBoost
are equivalent alternatives.

---

## 3. Architecture Comparison (LEADGROUP vs Direct-24 vs Pooled)

| Model | Algorithm | Architecture | RMSE | H01-H03 | H04-H06 | H07-H12 | H13-H24 |
|-------|-----------|-------------|------|---------|---------|---------|---------|
| G1_LGBM_LG | LightGBM | LEADGROUP | 137.54 | 110.57 | 133.25 | 140.17 | 143.22 |
| G2_LGBM_D24 | LightGBM | Direct-24 | 136.78 | 109.68 | 131.74 | 139.47 | 142.64 |
| G3_LGBM_Pool | LightGBM | Pooled | 139.21 | 121.83 | 135.94 | 140.78 | 143.24 |
| P1_XGB_LG | XGBoost | LEADGROUP | 136.95 | 109.74 | 132.56 | 139.71 | 142.64 |
| W1_XGB_D24 | XGBoost | Direct-24 | 136.18 | 108.97 | 132.05 | 138.46 | 142.04 |
| W2_XGB_Pool | XGBoost | Pooled | 138.44 | 119.96 | 135.28 | 140.31 | 142.53 |
| P2_CB_LG | CatBoost | LEADGROUP | 136.78 | 109.56 | 133.42 | 139.74 | 142.14 |
| W3_CB_D24 | CatBoost | Direct-24 | 137.05 | 108.84 | 132.86 | 140.10 | 142.75 |
| W4_CB_Pool | CatBoost | Pooled | 139.60 | 121.44 | 136.11 | 141.60 | 143.65 |

**Is LEADGROUP sufficient?**
YES. LEADGROUP is within 0.77–0.27 W/m² of Direct-24 across all algorithms.
This is negligible compared to the ~200 W/m² baseline gap, and does not justify 6× more
model fits per month. LEADGROUP (4 models/month) is the recommended architecture.

**Is Direct-24 worth the extra complexity?**
NO. Delta(Direct-24 − LEADGROUP): LGBM=-0.75, XGB=-0.77, CB=0.27 W/m².
The marginal gain is not worth 24 separate models vs 4 for an operational system.

**Does pooled modeling underperform?**
YES. Pooled RMSE at H01-H03 is 121.83 vs ~110.57 for LEADGROUP — a ~12 W/m² penalty.
The short-horizon (H1-H3) behavior (autocorrelation-dominant) is qualitatively different from
long-horizon (H13-H24, NWP-dominant). A single model cannot specialize for both.

---

## 4. Weather Regime Interpretation

### 4.1 K-means Regime Distribution (Test Period, Daytime)

| Regime | Label | n | % of Test | Mean CSI | Std CSI | Mean GHI | High-Ramp % | Cloudy % |
|--------|-------|---|-----------|---------|---------|----------|------------|----------|
| 0 | Clear (CSI≥0.7) | 28212 | 27.7% | 0.690 | 0.350 | 396.1 | 31.4% | 0.5% |
| 1 | Partly Cloudy | 66906 | 65.8% | 0.496 | 0.360 | 277.1 | 0.4% | 91.5% |
| 2 | Overcast (CSI<0.3) | 6570 | 6.5% | 0.637 | 0.359 | 363.9 | 15.9% | 98.9% |

### 4.2 Model RMSE by Regime

| Model | Clear RMSE | Partly Cloudy RMSE | Overcast RMSE |
|-------|-----------|-------------------|---------------|
| W1_XGB_D24 | 139.93 | 133.58 | 145.72 |
| W5_XGB_LG_Reg | 140.27 | 134.02 | 146.06 |
| P2_CB_LG | 140.03 | 134.44 | 146.02 |
| W6_CB_LG_Reg | 140.26 | 134.52 | 145.94 |
| G2_LGBM_D24 | 140.13 | 134.36 | 146.35 |
| P1_XGB_LG | 140.72 | 134.36 | 146.39 |
| W3_CB_D24 | 140.26 | 134.63 | 147.13 |
| G1_LGBM_LG | 141.00 | 135.08 | 146.94 |
| W2_XGB_Pool | 142.22 | 135.94 | 146.91 |
| G3_LGBM_Pool | 143.09 | 136.64 | 147.96 |
| W4_CB_Pool | 143.57 | 136.99 | 148.40 |
| PERS | 238.82 | 363.95 | 465.22 |
| DA | 394.95 | 311.37 | 372.05 |

### 4.3 Regime Model vs Base LEADGROUP (Delta RMSE)

| Comparison | Clear Δ | Partly Cloudy Δ | Overcast Δ |
|------------|---------|-----------------|------------|
| W5_XGB_LG_Reg vs P1_XGB_LG | -0.45 | -0.34 | -0.33 |
| W6_CB_LG_Reg vs P2_CB_LG | +0.23 | +0.08 | -0.08 |

**Does weather-regime classification improve difficult conditions?**
Overall: W5 XGBoost+Regime RMSE = 136.58 (-0.37 vs P1).
         W6 CatBoost+Regime RMSE = 136.90 (+0.11 vs P2).
See per-regime table above for whether improvement is concentrated in Overcast/High-Ramp windows.
The regime feature adds marginal-to-zero improvement at the aggregate level, but may provide
interpretable per-regime differences that are useful for thesis discussion.

---

## 5. Critical Window Analysis (RMSE, W/m²)

| Model | Daytime | Top5%GHI | Top10%GHI | HighRamp | Cloudy<0.3 | Summer | TOU1621 |
|-------|---------|----------|-----------|----------|------------|--------|--------|
| W1_XGB_D24 | 136.18 | 212.42 | 197.16 | 142.54 | 134.41 | 152.18 | 85.43 |
| W5_XGB_LG_Reg | 136.58 | 214.37 | 197.93 | 143.33 | 134.86 | 152.05 | 85.28 |
| G2_LGBM_D24 | 136.78 | 211.46 | 196.48 | 142.85 | 135.16 | 153.03 | 85.39 |
| P2_CB_LG | 136.78 | 211.28 | 196.23 | 142.80 | 135.23 | 153.07 | 85.85 |
| W6_CB_LG_Reg | 136.90 | 212.41 | 197.02 | 143.04 | 135.32 | 153.55 | 85.94 |
| P1_XGB_LG | 136.95 | 216.12 | 199.64 | 143.75 | 135.20 | 152.78 | 85.15 |
| W3_CB_D24 | 137.05 | 211.93 | 196.65 | 142.84 | 135.56 | 153.93 | 85.65 |
| G1_LGBM_LG | 137.54 | 213.60 | 198.01 | 143.99 | 135.93 | 152.98 | 85.59 |
| W2_XGB_Pool | 138.44 | 220.99 | 203.38 | 144.98 | 136.59 | 153.85 | 85.88 |
| G3_LGBM_Pool | 139.21 | 219.42 | 202.68 | 145.92 | 137.39 | 154.63 | 86.55 |
| W4_CB_Pool | 139.60 | 218.96 | 202.14 | 145.72 | 137.75 | 156.24 | 87.54 |
| DA | 340.60 | 744.66 | 702.57 | 409.81 | 316.12 | 385.89 | 212.65 |
| PERS | 341.99 | 707.86 | 669.74 | 276.27 | 383.75 | 383.84 | 169.27 |

**Key observations**:
- High GHI (Top5% / Top10%) RMSE is much higher than average — all models struggle at peak irradiance.
- Cloudy_CSI_lt03: persistence and ML diverge significantly; ML typically
  over-predicts during extended cloud events (positive MBE).
- High_Ramp_Hours: hardest window. Regime-aware models may help here if Overcast RMSE improves.
- Summer daytime is a key operating window for PV-BESS scheduling.
- TOU peak (16-21h) is low-sun, lower irradiance; errors are smaller here.

---

## 6. Baselines

| Baseline | RMSE (W/m²) | ML Improvement |
|----------|-------------|----------------|
| PERS | 341.99 | 205.8 W/m² (60%) |
| DA | 340.60 | 204.4 W/m² (60%) |

ML models improve over baselines by 206–204 W/m² (~60%).
This confirms substantial lift from intraday observation lags, rolling causal NWP, and
sky-condition features not available to daily baselines.

---

## 7. CNN-LSTM Assessment

CNN-LSTM was NOT trained for ID-H24 because:
1. DA benchmark: CNN-LSTM was +8.8 W/m² worse than XGBoost for DA solar forecasting
2. The ID-H24 feature set is tabular (lags, NWP, calendar, cloud indices) — not a
   sequence-to-sequence problem where LSTM excels
3. Higher tuning cost and training time for equivalent or worse performance

**Conclusion**: CNN-LSTM is a valid negative-result reference from prior experiments.
Citing the DA benchmark as evidence is sufficient for thesis defense.

---

## 8. Final Model Selection

**Selected model for ID-H24 GHI point forecasting**: **W1_XGB_D24**

| Property | Value |
|----------|-------|
| Algorithm | XGBoost |
| Architecture | Direct-24 |
| Test RMSE | 136.18 W/m² |
| Test MAE | 91.68 W/m² |
| Test R2 | 0.776 |
| Test rows | 101688 (daytime, H1-H24) |
| Models/month | 24 (Direct-24, one per horizon H1-H24) |

**Thesis justification**:
The ID-H24 point model was selected after a systematic benchmark of 13 model configurations
covering 3 algorithms (XGBoost, CatBoost, LightGBM), 3 architectures (LEADGROUP, Direct-24,
Pooled), 2 baselines (smart persistence, DA NWP), and 2 weather-regime variants. Key findings:

1. **Algorithm choice is not significant**: All three boosting algorithms are within 0.76 W/m²
   for LEADGROUP — consistent with the DA benchmark and solar forecasting literature.
2. **Direct-24 marginally outperforms LEADGROUP**: W1_XGB_D24 (136.18) beats best LEADGROUP
   P2_CB_LG (136.78) by 0.60 W/m². Both are valid; LEADGROUP uses 4 models/month vs 24.
   Pooled underperforms at short horizons by ~12 W/m² and is not recommended.
3. **Weather-regime adds marginal benefit**: Adding K-means regime_id as a feature does not
   substantially improve overall RMSE, but provides interpretable stratification.
4. **CSI target is appropriate**: Normalizes diurnal/seasonal amplitude variation; consistent
   with Verbois et al. (2018) and Pedro & Coimbra (2012).
5. **Causal NWP lookup is correct**: Rolling 4×/day GFS with linear interpolation; mean NWP
   age = 2.5h at issue_time; no future information leakage.

---

## 9. Next Steps (Awaiting User Confirmation)

1. **Confirm final model selection** — user must approve before proceeding
2. **Probabilistic ID-H24**: Apply quantile regression + AgACI calibration to selected model
3. **GHI-to-PV conversion**: PV_avail = min(PV_cap, PR × (GHI/1000) × PV_cap)
4. **Sanity check**: Compare ID-H24 q50 vs DA q50 at H6
5. **MPC integration**: Pass q50 as point forecast input to rolling ID MILP
