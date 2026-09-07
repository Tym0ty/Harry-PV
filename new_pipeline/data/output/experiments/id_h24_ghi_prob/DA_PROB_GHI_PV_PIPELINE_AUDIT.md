# DA Probabilistic GHI / PV Pipeline Audit

**Generated**: 2026-06-04  
**Purpose**: Reference for designing the ID-H24 GHI probabilistic forecast (Task 1 of 5)  
**Working directory**: `new_pipeline/`

---

## 1. Pipeline Overview

The DA probabilistic pipeline is a 4-stage system:

| Stage | Script | Purpose | Output |
|-------|--------|---------|--------|
| 0 (base) | `experiments/exp_da_baseline_verification.py` | Point forecast baseline | `da_v2_quantiles.parquet` |
| 1A/1B/2C/2D | `scripts/stage1_agaci_calibration.py` | AgACI dynamic calibration | `stage1_aci_calibrated.parquet` |
| 1-RC | `scripts/stage1_rc_conformal.py` | Risk-conditional conformal | `stage1_rc_conformal.parquet` |
| 2 | `scripts/stage2_t_copula_generation.py` | t-Copula scenario generation | `stage2_ghi_scenarios_N500.parquet` |
| 3 | `scripts/stage3_ghi_to_pv.py` | GHI → PV conversion | `stage3_pv_scenarios_N500.parquet` |
| 4 | `scripts/stage4_kmedoids_reduction.py` | Scenario reduction | `stage4_pv_scenarios_K5.parquet` |

---

## 2. Base Quantile Model (Stage 0 Input)

**Script**: `new_pipeline/scripts/experiments/exp_da_baseline_verification.py`  
**Model type**: XGBoost quantile regression (multi-quantile, pre-trained)  
**Input parquet**: `new_pipeline/data/input/da_v2_quantiles.parquet`

| Property | Value |
|----------|-------|
| Target variable | Raw GHI (W/m²) — `ghi_realized` |
| Quantile levels | **19 levels**: q05, q10, q15, q20, q25, q30, q35, q40, q45, q50, q55, q60, q65, q70, q75, q80, q85, q90, q95 |
| Feature set | 55 DA-valid features (no intraday lags, no future info) |
| Issue-time definition | D-1 ~20:00 local for target day D |
| Calibration period | 2024-04-30 to 2024-10-31 (rows where issue_day in this range) |
| Test period | 2024-11-01 to 2025-10-31 |
| q50 RMSE (test) | **140.88 W/m²** (XGBoost, walk-forward monthly) |

**Parquet schema**: `issue_day`, `target_day`, `hour_local`, `ghi_realized`, `ghi_clear_sky`, `q05`..`q95` (19 cols)

---

## 3. Stage 1 — AgACI Calibration

**Script**: `new_pipeline/scripts/stage1_agaci_calibration.py`  
**Input**: `da_v2_quantiles.parquet`

### 3.1 Coverage Targets

| Coverage type | Alpha | Lower quantile | Upper quantile |
|--------------|-------|---------------|---------------|
| pi80 | 0.80 | q10 | q90 |
| pi90 | 0.90 | q05 | q95 |
| lower (one-sided) | 0.90 | q10 | — |
| upper_q90 (one-sided) | 0.90 | — | q90 |

### 3.2 AgACI Tiers

| Tier | Name | Update rule | Gamma selection |
|------|------|-------------|----------------|
| 1A | Static CQR | Frozen Q_0 from calibration set (no online update) | N/A |
| 1B | Conservative online | Q_{d+1} = Q_d + γ × (α − cov_d) | Biased loss: 3× undercoverage penalty |
| 2C | Hour-group AgACI | Separate Q per daytime hour group (morning/peak/afternoon) | Biased loss per group |
| 2D | Asymmetric | Separate Q_lower and Q_upper (one-sided NCQR scores) | Combined biased loss |

### 3.3 NCQR Score Functions

**Symmetric**: `max(q_lo − y, y − q_hi) / (q_hi − q_lo + ε)` — positive when y outside interval  
**One-sided lower**: `(q_lo − y) / (|q_lo| + ε)` — positive when y < q_lo  
**One-sided upper**: `(y − q_hi) / (|q_hi| + ε)` — positive when y > q_hi

### 3.4 Interval Adjustment Formula

For symmetric PI: adjusted_lower = q_lo − Q_d × (q_hi − q_lo), adjusted_upper = q_hi + Q_d × (q_hi − q_lo)  
For one-sided lower: adjusted_lower = q_lo − Q_d × (|q_lo| + ε)  
For one-sided upper: adjusted_upper = q_hi + Q_d × (|q_hi| + ε)

### 3.5 Gamma Grid

From `config.yaml`: `gamma_grid` (typically 0.001 to 0.500, ~20 grid points)  
Gamma selected by **biased validation loss** on calibration set: 3× penalty for undercoverage, 1× for overcoverage.

### 3.6 Outputs

| File | Size | Key columns |
|------|------|------------|
| `data/output/stage1_aci_calibrated.parquet` | 821 KB | q10_aci, q90_aci, q05_aci, q95_aci, q10_aci_lower, q90_aci_upper, q10_static80, q90_static80, q10_aci_hg, q90_aci_hg |
| `data/output/stage1_aci_threshold_daily.parquet` | 108 KB | Daily Q_d traces per method |

---

## 4. Stage 1-RC — Risk-Conditional Conformal

**Script**: `new_pipeline/scripts/stage1_rc_conformal.py`

4-class risk stratification: HIGH (summer weekday peak), MID-WORKING, MID-OTHER, LOW  
Per-class online AgACI with class-specific alpha targets (HIGH: 95/90% lower/upper).  
Output: `stage1_rc_conformal.parquet` — q10_rc, q90_rc, q05_rc, q95_rc

---

## 5. Stage 2 — t-Copula Scenario Generation

**Script**: `new_pipeline/scripts/stage2_t_copula_generation.py`  
Input: `stage1_aci_calibrated.parquet` (calibrated marginal CDFs)  
Method: Seasonal t-copula (4 seasons), MLE for correlation matrix R and degrees-of-freedom ν  
Scenarios: **N = 500** per target day  
Post-processing: clip GHI < 0 → 0; cap at 1.2 × ghi_clear_sky  
Output: `stage2_ghi_scenarios_N500.parquet` — schema: (issue_day, target_day, scenario_id, hour_local, ghi_kw_m2)

---

## 6. Stage 3 — GHI-to-PV Conversion

**Script**: `new_pipeline/scripts/stage3_ghi_to_pv.py`  
Formula: `PV_avail(t) = min(PV_cap, PR × GHI(t)/1000 × PV_cap)`  
Parameters: PV_cap = 2687 kWp, PR = 0.80  
Output: `stage3_pv_scenarios_N500.parquet`

---

## 7. Stage 4 — k-Medoids Scenario Reduction

**Script**: `new_pipeline/scripts/stage4_kmedoids_reduction.py`  
Method: PAM k-medoids, N=500 → K=5 representative daily PV profiles  
Output: `stage4_pv_scenarios_K5.parquet` — each medoid with probability weight

---

## 8. Downstream Usage

YES — the PV scenario outputs feed directly into the rolling ID MILP scheduling pipeline.  
The reduction to K=5 scenarios is used in stochastic scheduling formulations.  
PV-to-Net-Load conversion is done inside the MILP/MPC scripts.

---

## 9. Key Design Choices Relevant to ID-H24 Adaptation

| DA choice | ID-H24 adaptation |
|-----------|------------------|
| Target: raw GHI W/m² | Target: **CSI**, back-convert to GHI using clear-sky |
| Issue: once per day (D-1) | Issue: **hourly** (rolling) |
| Quantile levels: 19 (every 5%) | Quantile levels: **11** (q05, q10, q20, q30, q40, q50, q60, q70, q80, q90, q95) |
| AgACI update: once per day | AgACI update: **once per issue_date** (group hourly issues by calendar day) |
| Hour groups: morning/peak/afternoon | **Lead groups: H1-H3, H4-H6, H7-H12, H13-H24** |
| Calibration: 2024-04-30 to 2024-10-31 | Calibration: **2024-05-01 to 2024-10-31** (same window) |
| Test period: 2024-11-01 to 2025-10-31 | Test: **2024-11-01 to 2025-10-31** (identical) |
| Base model: XGBoost (walk-forward monthly) | Base model: **W1_XGB_D24** (same walk-forward, CSI target) |
| Scenario generation: t-Copula | **Not in scope** for ID-H24 prob (future work) |

---

## 10. Gaps / Differences for ID-H24 Design

1. **Rolling issue-time**: DA has one issue per day; ID-H24 has one issue per hour. AgACI must be adapted to group by issue_date for the update step.
2. **Multi-horizon evaluation**: DA evaluates at daily level; ID-H24 must evaluate per lead group (H1-H3, H4-H6, H7-H12, H13-H24).
3. **CSI target**: ID-H24 predicts CSI, not raw GHI. Back-conversion must preserve quantile ordering (monotone since ghi_clear_target > 0).
4. **No scenario generation planned**: ID-H24 probabilistic output will be quantile intervals only (not full scenarios). t-Copula and k-medoids are DA-specific.
5. **AgACI group structure**: DA uses time-of-day groups; ID-H24 uses horizon lead groups. Both achieve conditional calibration in their respective contexts.
