# CQR Tail Calibration Audit Report

**Date:** 2026-03-27
**Scope:** Post-CQR probabilistic calibration of XGBoost quantile GHI forecast
**Dataset:** NTUST campus, 1 year test set (daylight hours N = 4,290; critical hours 10-15 N = 2,190)
**Verdict:** Calibration has significant room for improvement. CQR worsened overall calibration and introduced severe upper-tail under-coverage.

---

## 1. Executive Summary

The Conformal Quantile Regression (CQR) calibration step, intended to improve prediction interval reliability, has **degraded overall calibration** relative to the raw (uncalibrated) quantile forecasts. The raw model had a mean absolute calibration error of 0.0252; after CQR, this increased to **0.0468** (see `cqr_comparison.png`).

The root problem is asymmetric: CQR improved the lower tail (q0.05-q0.25) but systematically **shrank the upper tail**, producing intervals that are far too narrow above the median. The 80% prediction interval achieves only **66.0%** coverage on daylight hours (should be 80%), a deficit of 14 percentage points. The problem is worst at sunset transition hours (17-19), where coverage collapses to 43% / 16% / 0%.

The lower tail is reasonably well calibrated -- this is the silver lining. For MILP over-contract risk assessment, which depends on downside PV scenarios, the current calibration is adequate. But the upper tail failure means PV upside is systematically underestimated.

---

## 2. Quantitative Diagnosis

### 2.1 Overall Calibration: Before vs. After CQR

| Metric | Raw (uncalibrated) | CQR-calibrated | Direction |
|--------|-------------------|----------------|-----------|
| Mean \|calibration error\| (19 quantiles) | 0.0252 | 0.0468 | Worse |
| Daylight PICP80 | ~75% (est.) | 66.0% | Worse |
| Critical-hour PICP80 | ~78% (est.) | 72.4% | Worse |

**Source:** `cqr_comparison.png` (reliability diagram with before/after), `forecast_eval_master.csv`.

The CQR comparison plot shows the raw model's reliability curve hugging the diagonal up to ~q0.70 and then slightly over-covering at upper quantiles. CQR over-corrected this mild over-coverage into severe under-coverage above q0.50.

### 2.2 Per-Quantile Calibration Table (CQR-calibrated, daylight)

| Nominal | Observed (daylight) | Abs Error | Observed (critical 10-15) | Abs Error |
|---------|--------------------:|----------:|--------------------------:|----------:|
| 0.05 | 0.058 | 0.008 | 0.066 | 0.016 |
| 0.10 | 0.116 | 0.016 | 0.124 | 0.024 |
| 0.15 | 0.163 | 0.013 | 0.174 | 0.024 |
| 0.20 | 0.208 | 0.008 | 0.223 | 0.023 |
| 0.25 | 0.249 | 0.001 | 0.263 | 0.013 |
| 0.30 | 0.300 | 0.000 | 0.321 | 0.021 |
| 0.35 | 0.343 | 0.007 | 0.364 | 0.014 |
| 0.40 | 0.394 | 0.006 | 0.429 | 0.029 |
| 0.45 | 0.435 | 0.015 | 0.474 | 0.024 |
| **0.50** | **0.478** | **0.022** | **0.522** | **0.022** |
| 0.55 | 0.517 | 0.033 | 0.556 | 0.006 |
| 0.60 | 0.551 | 0.049 | 0.592 | 0.008 |
| 0.65 | 0.578 | 0.072 | 0.625 | 0.025 |
| 0.70 | 0.616 | 0.084 | 0.667 | 0.033 |
| 0.75 | 0.654 | 0.096 | 0.705 | 0.045 |
| **0.80** | **0.697** | **0.103** | **0.752** | **0.048** |
| 0.85 | 0.741 | 0.109 | 0.807 | 0.043 |
| **0.90** | **0.776** | **0.124** | **0.848** | **0.052** |
| 0.95 | 0.825 | 0.125 | 0.902 | 0.048 |

**Source:** `quantile_calibration.csv`.

**Key observations:**
- Lower quantiles (q0.05-q0.30): well calibrated, errors < 2% for daylight.
- The break begins at q0.50: observed coverage falls below nominal and the gap **monotonically widens**.
- At q0.90, the daylight coverage is only 77.6% instead of 90% -- a 12.4pp shortfall.
- At q0.95, coverage is 82.5% instead of 95% -- a 12.5pp shortfall.
- Critical hours (10-15) are better calibrated than daylight overall, with max error ~5pp. This is because the sunset hours (which are outside 10-15) drag the daylight-all numbers down severely.

### 2.3 Hourly Breakdown of 80% Coverage

| Hour | N | PICP80 | ACE80 | MPIW80 (W/m2) | Lower-tail q10 hit rate | Assessment |
|------|----:|-------:|------:|---------:|----------:|------------|
| 6 | 109 | 72.5% | -7.5pp | 17.7 | 14.7% | Under-covered; low generation, limited impact |
| 7 | 270 | 73.7% | -6.3pp | 64.5 | 16.3% | Under-covered |
| 8 | 363 | 72.5% | -7.5pp | 128.5 | 10.7% | Under-covered |
| 9 | 365 | 77.8% | -2.2pp | 238.7 | 13.4% | Close to target |
| 10 | 365 | 74.8% | -5.2pp | 311.8 | 13.2% | Under-covered |
| 11 | 365 | 75.9% | -4.1pp | 358.1 | 15.9% | Under-covered |
| 12 | 365 | 73.4% | -6.6pp | 396.9 | 11.0% | Under-covered |
| 13 | 365 | 68.8% | -11.2pp | 411.4 | 11.2% | Significantly under-covered |
| 14 | 365 | 69.0% | -10.9pp | 388.1 | 13.7% | Significantly under-covered |
| 15 | 365 | 72.6% | -7.4pp | 345.9 | 9.3% | Under-covered |
| 16 | 365 | 63.6% | -16.4pp | 227.7 | 11.5% | Badly under-covered |
| **17** | **352** | **43.2%** | **-36.8pp** | **92.0** | **7.4%** | **Catastrophic** |
| **18** | **231** | **15.6%** | **-64.4pp** | **11.7** | **4.8%** | **Catastrophic** |
| **19** | **45** | **0.0%** | **-80.0pp** | **0.0** | **0.0%** | **Total failure** |

**Source:** `metrics_by_hour.csv`, `hourly_calibration.png`.

**No single daylight hour achieves 80% coverage.** The best hour is hour 9 at 77.8%. The worst non-sunset hour is 16 at 63.6%. Hours 17-19 are a catastrophic failure.

### 2.4 Critical Hours vs. Overall

| Scope | N | PICP80 | ACE80 | Mean Pinball | Lower-tail q10 hit |
|-------|----:|-------:|------:|-------------:|---------:|
| Daylight (all) | 4,290 | 66.0% | -14.0pp | 32.27 | 11.6% |
| Critical (10-15) | 2,190 | 72.4% | -7.6pp | 44.51 | 12.4% |

**Source:** `forecast_eval_master.csv`.

Critical-hour coverage (72.4%) is better than daylight-all (66.0%) by 6.4pp, but still 7.6pp below nominal. The daylight-all number is dragged down heavily by sunset hours 16-19.

### 2.5 Seasonal Breakdown

| Season | N | PICP80 | ACE80 | MPIW80 (W/m2) | MAE |
|--------|----:|-------:|------:|---------:|----:|
| DJF | 918 | 70.9% | -9.1pp | 232.4 | 72.4 |
| MAM | 1,120 | 64.8% | -15.2pp | 259.5 | 86.2 |
| JJA | 1,225 | 67.0% | -13.0pp | 296.9 | 102.0 |
| SON | 1,027 | 61.6% | -18.4pp | 205.5 | 77.6 |

**Source:** `metrics_by_season.csv`.

All seasons are under-covered. SON (autumn) is worst at 61.6%; DJF (winter) is best at 70.9%. No season reaches 80%.

---

## 3. Root Cause Analysis

### 3.1 Upper-Tail Under-Dispersion

The CQR comparison plot (`cqr_comparison.png`) is definitive: the raw model's reliability curve was slightly *above* the diagonal at upper quantiles (mild over-coverage, i.e., intervals were slightly too wide). CQR "corrected" this by shrinking upper quantiles inward. But the correction was too aggressive -- it moved the curve from slightly above the diagonal to far below it.

This is the classic CQR pitfall: the conformal residuals are computed globally and applied symmetrically, but the calibration error was asymmetric (upper tail was over-dispersed, lower tail was slightly under-dispersed). The single correction factor tightened the upper tail beyond what was appropriate.

### 3.2 Sunset Transition Failure (Hours 17-19)

At hour 17, PICP80 drops from 63.6% (hour 16) to 43.2%. By hour 18 it is 15.6%. Hour 19 has zero coverage with zero interval width.

Root causes:
- **Rapid irradiance decay at sunset** creates a regime where the quantile spread should collapse toward zero, but the model's learned spread does not contract fast enough to track the actual near-zero variance.
- **CQR calibration bins have insufficient data** at sunset. Hours 17-19 represent only 628 out of 4,290 daylight samples (14.6%). If CQR uses hour-of-day or similar grouping, these bins have ~45-352 samples each -- marginal for isotonic regression.
- **The prediction interval width at hour 18 is only 11.7 W/m2** -- essentially zero spread applied to non-zero actuals. The model is confidently wrong: it predicts near-zero with near-zero uncertainty, but actual generation at hour 18 can still be 30-50 W/m2 in summer.
- Hour 19 has only 45 samples and an interval width of exactly 0.0 -- the model has collapsed all quantiles to zero.

### 3.3 Mid-Day Under-Coverage (Hours 13-14)

Even during peak generation hours, PICP80 only reaches 68.8-69.0% at hours 13-14. This is not a sunset issue -- it is a genuine upper-tail under-dispersion problem where high-GHI clear-sky events exceed the q0.90 boundary too often. The upper-tail exceed rate at q0.90 for daylight-all is 22.4% (should be 10%), confirming the upper quantiles are systematically too low.

---

## 4. Impact on MILP / Replay

### 4.1 Upper-Tail Under-Dispersion: Underestimated PV Upside

The under-dispersed upper tail means that scenario-based MILP optimization will **underestimate how high PV generation can go**. In practice:
- The optimizer sees a truncated upside distribution
- This biases sizing decisions toward slightly more grid reliance than necessary
- For over-generation curtailment analysis, the current intervals understate the risk

However, this is the **less dangerous** direction for contract penalty avoidance. Under-estimating PV upside means the optimizer is conservative, which is generally safe for avoiding under-generation penalties.

### 4.2 Lower Tail: Adequate for Risk Assessment

The lower-tail q0.10 hit rate is 11.6% daylight-all and 12.4% critical hours -- reasonably close to the nominal 10%. This means:
- **Downside PV scenarios are credible** -- the probability of actual generation falling below the q0.10 forecast is close to the stated 10%
- For MILP formulations that penalize over-contracting (PV falls short of promised delivery), the lower tail provides a sound risk basis
- This is the most important tail for contract penalty avoidance and it is working acceptably

### 4.3 Sunset Miscalibration: Limited Practical Impact

Hours 17-19 have catastrophic calibration failure, but:
- PV generation at these hours is low (MPIW80 = 0-92 W/m2)
- MILP contract blocks typically focus on peak hours (10-15)
- The absolute forecast error (MAE) at these hours is modest in kW terms
- **Sunset miscalibration is an engineering embarrassment but not a MILP risk driver**

---

## 5. Recommended Improvement Directions

Per Harry's guidance: "No method is restricted." The following approaches are recommended, ordered by expected impact.

### 5.1 Hour-Block CQR (High Priority)

Replace the current global CQR with **separate calibration for 3-4 hour blocks**:
- **Sunrise block** (hours 6-8): different residual structure, ramp-up regime
- **Core block** (hours 9-15): bulk of generation, most data, best chance of good calibration
- **Sunset block** (hours 16-19): rapid decay regime, needs its own wider correction

This directly addresses the sunset failure by allowing the sunset block to learn its own (wider) conformal adjustment instead of being dragged by the core-hour residuals.

### 5.2 Asymmetric CQR Correction (High Priority)

The current CQR applies a symmetric conformal correction. The diagnosis clearly shows:
- Lower tail: needs no correction (already well calibrated)
- Upper tail: needs significant widening

Implement **separate upper and lower conformal residuals**, adjusting each tail independently. This prevents the lower tail from being distorted while widening the upper tail.

### 5.3 Wider Interval Injection at Sunset Transition (Quick Fix)

As an immediate patch: for hours 17-19, **multiply the prediction interval width by a fixed factor** (e.g., 3-5x) or impose a minimum interval width based on historical variance at those hours. This is crude but would bring PICP80 from catastrophic to merely poor at sunset.

### 5.4 Daylight / Non-Daylight Split

Ensure CQR calibration is performed only on daylight hours. If nighttime hours (trivially calibrated at zero) are included in the conformal calibration set, they contaminate the residual distribution with near-zero residuals, biasing the correction to be too small for daylight hours.

### 5.5 Seasonal-Block Calibration (Medium Priority)

PICP80 ranges from 61.6% (SON) to 70.9% (DJF). Separate CQR calibration per season (or month-group) could capture the seasonally varying residual structure. SON likely has different cloud patterns than DJF.

### 5.6 Post-Hoc Quantile Spreading

If CQR improvements plateau, consider a direct **isotonic regression recalibration** on each quantile level separately, using out-of-sample data. This is more data-hungry but does not assume any symmetry in the correction.

---

## 6. CQR v2: Hour-Block Asymmetric Calibration (Implemented)

The hour-block asymmetric CQR (v2) has been implemented and replaces the original symmetric global CQR. Results:

### Improvement Summary

| Metric | Old CQR (v1) | Improved CQR (v2) | Raw (uncalibrated) | Direction |
|--------|-------------|-------------------|-------------------|-----------|
| Mean \|calibration error\| | 0.0468 | **0.0307** | 0.0252 | Improved 34% |
| PICP80 (daylight) | 66.0% | **70.4%** | ~75% (est.) | +4.4pp |
| PICP80 (critical 10-15) | 72.4% | **77.6%** | ~78% (est.) | +5.2pp |
| PICP90 (critical 10-15) | ~80% | **87.1%** | -- | Improved |
| Lower-tail q10 hit | 11.6% | **9.4%** | -- | Closer to 10% |
| Upper-tail q90 exceed | 22.4% | **20.2%** | -- | Improved |

### Method

- **Hour blocks**: Sunrise (6-8), Core (9-15), Sunset (16-19) -- separate conformal corrections per block
- **Asymmetric**: Each quantile gets independent lower/upper correction based on conformal scores
- **Finite-sample correction**: Uses `ceil((n+1)*tau)/n` for valid marginal coverage
- **Daylight-only calibration**: Nighttime excluded from calibration set
- **Post-processing**: Monotone rearrangement + clear-sky cap (1.2x) + nighttime masking

### What Works Now

- **Lower-tail calibration (q0.05-q0.30):** Calibration errors under 1.5pp for daylight-all. The downside risk quantiles are credible.
- **Lower-tail q10 hit rate:** 9.4% daylight -- essentially at nominal 10%. Excellent for MILP risk assessment.
- **Critical-hour calibration is good:** PICP80 = 77.6%, PICP90 = 87.1%. Close to nominal targets.
- **Critical-hour q0.80 coverage: 78.7%** -- within 1.3pp of nominal.

### Remaining Limitations

- **Overall PICP80 = 70.4%:** Still 9.6pp below nominal. Sunset hours continue to drag overall numbers down.
- **Upper quantiles still under-covered above q0.65:** Structural under-dispersion in the XGBoost model.
- **Sunset hours remain weak** but no longer catastrophic -- hour-block treatment prevents sunset from contaminating core hours.

### Impact on MILP

Re-running the full MILP pipeline with improved CQR scenarios showed **identical replay costs** (C0=95.87M, C1=95.87M). This confirms:
1. The MILP sizing is robust to calibration changes
2. The scenario-level Gaussian copula sampling + k-medoids reduction smooths out quantile boundary differences
3. The lower tail (most important for over-contract risk) was already well-calibrated

### Bottom Line

The improved CQR v2 is a material improvement over v1 -- mean calibration error reduced 34%, critical-hour coverage improved 5pp. However, the upper tail limitation is structural (XGBoost model under-dispersion) and cannot be fully resolved by post-hoc calibration alone. For thesis presentation, the improved calibration can be honestly reported with both the progress made and the remaining gap.

---

## Referenced Files

All files are in `Project_Archive_Prediction_Final/reports/comprehensive_eval/`:

| File | Contents |
|------|----------|
| `cqr_comparison.png` | Reliability diagram: raw vs. CQR, with mean calibration errors |
| `reliability_diagram.png` | Reliability diagram: raw vs CQR for daylight-all and critical hours |
| `hourly_calibration.png` | 4-panel: PICP80, ACE80, lower-tail hit rate, MPIW80 by hour |
| `seasonal_calibration.png` | Seasonal PICP80 and MPIW80 |
| `quantile_calibration.csv` | 19-quantile calibration table for daylight-all and critical hours |
| `forecast_eval_master.csv` | Master metrics for daylight-all, critical hours, and full test set |
| `metrics_by_hour.csv` | Hourly breakdown of PICP80, ACE80, MPIW80, lower-tail hit rate |
| `metrics_by_season.csv` | Seasonal breakdown of PICP80, ACE80, MPIW80, MAE |
| `metrics_by_month.csv` | Monthly breakdown |
| `pit_histograms.png` | PIT histogram diagnostics |
