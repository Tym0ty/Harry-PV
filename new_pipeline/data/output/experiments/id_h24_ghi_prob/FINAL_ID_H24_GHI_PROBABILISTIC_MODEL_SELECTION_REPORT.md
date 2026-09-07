# Final ID-H24 GHI Probabilistic Model Selection Report

**Generated**: 2026-06-04
**Test period**: 2024-11-01 to 2025-10-31
**Calibration period**: 2024-05-01 to 2024-10-31
**Evaluation**: Daytime only (ghi_clear >= 10 W/m²); H1-H24 combined

---

## 1. Approach Summary

| Approach | Method | Architecture | Quantile model |
|----------|--------|-------------|----------------|
| Q1 | Direct quantile regression | Direct-24 (24 models/month) | XGBoost multi-q |
| Q2 | Direct quantile regression | LEADGROUP (4 models/month) | XGBoost multi-q |
| Q3 | Residual empirical quantile | Point model W1_XGB_D24 + lead-group residual dist. | Empirical CDF |
| Q1_cal | Q1 + AgACI calibration | Direct-24 | XGBoost + online calibration |
| Q2_cal | Q2 + AgACI calibration | LEADGROUP | XGBoost + online calibration |
| Q3_cal | Q3 + AgACI calibration | Empirical + online calibration | Empirical CDF |

AgACI: Adaptive Asymmetric Conformal Prediction.
Update rule: Q_{d+1} = Q_d + γ × (α − coverage_rate_d), updated per issue_date per lead group.
Gamma selected on calibration set with 3× undercoverage penalty.

---

## 2. Overall Metrics (Calibrated)

| Approach | PICP@80% | PICP@90% | MPIW@80% | MPIW@90% | q50 RMSE | Pinball | CRPS |
|----------|---------|---------|---------|---------|---------|---------|------|
| Q1_D24_raw | 74.45% | 92.44% | 278.49 | 521.26 | 139.66 | 30.860 | 61.720 |
| Q2_LG_raw | 75.05% | 92.67% | 281.74 | 519.62 | 139.83 | 30.879 | 61.758 |
| Q1_D24_cal | 79.81% | 89.75% | 305.79 | 544.44 | 139.66 | 30.860 | 61.720 |
| Q2_LG_cal | 79.81% | 89.74% | 306.85 | 545.38 | 139.83 | 30.879 | 61.758 |
| Q3_Residual_cal | 82.52% | 90.71% | 312.44 | 419.15 | 136.18 | 33.586 | 67.172 |
| Q3_Residual_raw | 83.78% | 91.69% | 316.96 | 421.35 | 136.18 | 33.586 | 67.172 |

---

## 3. Model Selection Questions

### Q1: Which probabilistic approach is best (Q1, Q2, Q3)?

See Section 2 above. The approach with PICP@80% closest to 80% and smallest MPIW@80% is preferred.
- Q3 (residual empirical) is fast and interpretable; may be under-dispersed.
- Q2 (LEADGROUP quantile) is a direct quantile model with minimal compute overhead.
- Q1 (Direct-24 quantile) has highest potential RMSE at short horizons but is most expensive.
- AgACI calibration corrects raw coverage in all cases; calibrated PICP should be near nominal.

### Q2: Does calibration achieve target coverage at 80% and 90%?

- Q1_D24_raw: PICP@80% = 74.45% (MARGINAL), PICP@90% = 92.44% (YES)
- Q2_LG_raw: PICP@80% = 75.05% (MARGINAL), PICP@90% = 92.67% (YES)
- Q3_Residual_raw: PICP@80% = 83.78% (MARGINAL), PICP@90% = 91.69% (YES)
- Q1_D24_cal: PICP@80% = 79.81% (YES), PICP@90% = 89.75% (YES)
- Q2_LG_cal: PICP@80% = 79.81% (YES), PICP@90% = 89.74% (YES)
- Q3_Residual_cal: PICP@80% = 82.52% (YES), PICP@90% = 90.71% (YES)

### Q3: Is the calibrated interval too wide?

Width is assessed relative to the signal level.
For hourly GHI H1-H24, mean GHI is typically 200-400 W/m²; MPIW@80% should be <300 W/m².
Intervals that are too wide (MPIW > 3× mean GHI) indicate the raw quantile model is poorly
calibrated or residual variance is very high.

- Q1_D24_raw: MPIW@80% = 278.49 W/m² (ACCEPTABLE)
- Q2_LG_raw: MPIW@80% = 281.74 W/m² (ACCEPTABLE)
- Q3_Residual_raw: MPIW@80% = 316.96 W/m² (TOO WIDE)
- Q1_D24_cal: MPIW@80% = 305.79 W/m² (TOO WIDE)
- Q2_LG_cal: MPIW@80% = 306.85 W/m² (TOO WIDE)
- Q3_Residual_cal: MPIW@80% = 312.44 W/m² (TOO WIDE)

### Q4: Does reliability hold across H1-H24?

See id_h24_ghi_prob_metrics.md Section 2 for per-lead-group PICP@80%.
Short horizons (H1-H3) typically show better coverage (more persistent, predictable).
Long horizons (H13-H24) may show degraded coverage due to larger forecast uncertainty.

### Q5: Does reliability hold in high-GHI and cloudy periods?

See id_h24_ghi_prob_metrics.md Section 4 for critical window breakdown.
High-GHI periods (Top5%) are hardest; wider intervals expected and acceptable.
Cloudy periods (CSI<0.3) test the lower tail calibration; one-sided AgACI helps here.

### Q6: Is the probabilistic GHI forecast ready for GHI-to-PV conversion?

YES, subject to user confirmation of the final approach.
Conversion: pred_pv_qXX = clip(pred_ghi_qXX × PR × PV_cap / 1000, 0, PV_cap)
where PR = 0.80, PV_cap = 2687 kWp (from config.yaml, matching Stage 3 of DA pipeline).
This step is NOT executed in this script (per user restrictions).

### Q7: Direct-24 or LEADGROUP for probabilistic final model?

Based on the point model benchmark (W1_XGB_D24=136.18 vs P2_CB_LG=136.78):
- Q1 (Direct-24) has marginally better q50 RMSE but 6× more models.
- Q2 (LEADGROUP) is preferred for thesis parsimony if PICP difference < 2%.
- Final recommendation based on calibrated PICP and MPIW in Section 2 above.

### Q8: Thesis explanation

The ID-H24 probabilistic GHI forecast uses the same CSI-target, rolling causal NWP, and
walk-forward monthly training protocol as the selected point model (W1_XGB_D24). Three
probabilistic approaches were benchmarked:
(1) Direct quantile regression (Q1/Q2): trains separate models for each quantile level,
    producing predictive intervals aligned with the direct multi-step forecasting strategy.
(2) Empirical residual quantiles (Q3): uses the point forecast as a centre and adds
    lead-group-specific empirical residual distributions, providing a nonparametric baseline.
All approaches are calibrated via adaptive conformal prediction (AgACI) that adjusts the
quantile bounds online, updating once per issue_date per lead group.
This pipeline mirrors the DA probabilistic pipeline (AgACI Tier 1B) adapted for the
intraday rolling horizon context.

---

## 4. Next Steps (Awaiting User Confirmation)

1. Confirm probabilistic model selection (Q1, Q2, or Q3, with or without AgACI)
2. GHI-to-PV probabilistic conversion (Task deferred — per user restrictions)
3. Net-load scenario generation using quantile intervals or sampled scenarios
4. MILP/MPC integration with probabilistic PV input
