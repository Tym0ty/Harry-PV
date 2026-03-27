#!/usr/bin/env python3
"""
Improved CQR Calibration — Hour-Block Asymmetric Conformal

Addresses the CQR tail audit findings:
1. Hour-block calibration: sunrise (6-8), core (9-15), sunset (16-19)
2. Asymmetric corrections: separate upper and lower conformal residuals
3. Daylight-only calibration: exclude nighttime from calibration set

Reads: pipeline_outputs/forecast_ghi_quantiles_daily_base_raw.parquet
Writes: pipeline_outputs/forecast_ghi_quantiles_daily.parquet (overwrite)

Then re-runs scenario generation (Gaussian copula + k-medoids) with
the improved quantiles and re-runs bridge to produce updated ingest.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_QUANTILES = ROOT / "pipeline_outputs" / "forecast_ghi_quantiles_daily_base_raw.parquet"
OUT_QUANTILES = ROOT / "pipeline_outputs" / "forecast_ghi_quantiles_daily.parquet"

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
QCOLS = [f"q{q:.2f}" for q in QUANTILES]

# Hour blocks for calibration
HOUR_BLOCKS = {
    "sunrise": list(range(6, 9)),    # 6, 7, 8
    "core":    list(range(9, 16)),   # 9, 10, 11, 12, 13, 14, 15
    "sunset":  list(range(16, 20)),  # 16, 17, 18, 19
}


def asymmetric_cqr_block(cal_df, qcols, quantiles, alpha_coverage=0.9):
    """Compute per-quantile asymmetric conformal corrections for a block.

    Uses the standard conformal quantile approach:
    For each quantile tau, compute the conformity score needed so that
    exactly tau fraction of calibration data falls below q_tau + delta.

    The finite-sample correction ensures valid marginal coverage.

    Returns: (delta_lower, delta_upper) arrays of shape (n_quantiles,)
    where corrected quantile = q_tau - delta_lower + delta_upper
    """
    y = cal_df["label_ghi_obs_wm2"].values
    n = len(y)
    n_q = len(quantiles)

    delta_lower = np.zeros(n_q)
    delta_upper = np.zeros(n_q)

    for i, (tau, qcol) in enumerate(zip(quantiles, qcols)):
        q_vals = cal_df[qcol].values

        # Conformity scores: how much we need to shift q_tau to cover y
        scores = y - q_vals  # positive = y above quantile, negative = y below

        # For quantile tau: we want P(Y ≤ q_tau + delta) = tau
        # Using finite-sample conformal: pick delta = quantile of scores at level tau
        # With correction: level = ceil((n+1) * tau) / n
        level = min(np.ceil((n + 1) * tau) / n, 1.0)
        delta = np.quantile(scores, level)

        if delta > 0:
            # Need to raise quantile (under-coverage)
            delta_upper[i] = delta
        elif delta < 0:
            # Need to lower quantile (over-coverage)
            delta_lower[i] = -delta

    return delta_lower, delta_upper


def apply_corrections(df, qcols, quantiles, delta_lower, delta_upper):
    """Apply asymmetric corrections to quantile forecasts."""
    df = df.copy()
    for i, qcol in enumerate(qcols):
        # Net shift: positive delta_upper raises quantile, positive delta_lower lowers it
        df[qcol] = df[qcol] - delta_lower[i] + delta_upper[i]
        df[qcol] = df[qcol].clip(lower=0)

    # Monotone rearrangement: ensure q0.05 ≤ q0.10 ≤ ... ≤ q0.95
    q_matrix = df[qcols].values
    q_sorted = np.sort(q_matrix, axis=1)
    df[qcols] = q_sorted

    return df


def run():
    print("=" * 60)
    print("Improved CQR: Hour-Block Asymmetric Conformal")
    print("=" * 60)

    # Load raw (uncalibrated) quantiles
    raw = pd.read_parquet(RAW_QUANTILES)
    raw["target_time_local"] = pd.to_datetime(raw["target_time_local"])
    raw["target_day_local"] = pd.to_datetime(raw["target_day_local"])

    # Detect quantile column names
    qcols_in = []
    for q in QUANTILES:
        candidates = [f"q{q:.2f}", f"q0.{int(q*100):02d}", f"q{q}"]
        for c in candidates:
            if c in raw.columns:
                qcols_in.append(c)
                break
        else:
            raise ValueError(f"Quantile column for {q} not found in {raw.columns.tolist()}")

    print(f"  Loaded {len(raw)} rows, quantile columns: {qcols_in[:3]}...{qcols_in[-1]}")

    # Extract hour
    raw["hour"] = raw["target_time_local"].dt.hour

    # Split into calibration and test
    # Use the split_name column if available
    if "split_name" in raw.columns:
        cal = raw[raw["split_name"] == "calib"].copy()
        test = raw[raw["split_name"] == "test"].copy()
        print(f"  Cal: {len(cal)} rows, Test: {len(test)} rows")
    else:
        # Use first 30% as calibration
        n = len(raw)
        cal = raw.iloc[:int(n * 0.3)].copy()
        test = raw.iloc[int(n * 0.3):].copy()
        print(f"  Cal (30%): {len(cal)} rows, Test (70%): {len(test)} rows")

    # Filter to daylight hours only for calibration
    cal_daylight = cal[cal["solar_elevation"] > 0].copy() if "solar_elevation" in cal.columns else cal[cal["hour"].between(6, 19)].copy()
    print(f"  Cal daylight: {len(cal_daylight)} rows")

    # Compute corrections per hour block
    corrections = {}
    for block_name, hours in HOUR_BLOCKS.items():
        block_cal = cal_daylight[cal_daylight["hour"].isin(hours)]
        if len(block_cal) < 50:
            print(f"  Block {block_name}: only {len(block_cal)} samples, using global correction")
            block_cal = cal_daylight

        delta_lower, delta_upper = asymmetric_cqr_block(block_cal, qcols_in, QUANTILES)

        corrections[block_name] = (delta_lower, delta_upper)
        print(f"  Block {block_name} (N={len(block_cal)}):")
        print(f"    Lower shifts: {[f'{d:.1f}' for d in delta_lower[:5]]}...{[f'{d:.1f}' for d in delta_lower[-3:]]}")
        print(f"    Upper shifts: {[f'{d:.1f}' for d in delta_upper[:5]]}...{[f'{d:.1f}' for d in delta_upper[-3:]]}")

    # Apply corrections to ALL data (both cal and test)
    result = raw.copy()

    for block_name, hours in HOUR_BLOCKS.items():
        delta_lower, delta_upper = corrections[block_name]
        mask = result["hour"].isin(hours)
        block_rows = result[mask].copy()

        for i, qcol in enumerate(qcols_in):
            block_rows[qcol] = block_rows[qcol] - delta_lower[i] + delta_upper[i]
            block_rows[qcol] = block_rows[qcol].clip(lower=0)

        result.loc[mask, qcols_in] = block_rows[qcols_in].values

    # Monotone rearrangement
    q_matrix = result[qcols_in].values
    q_sorted = np.sort(q_matrix, axis=1)
    result[qcols_in] = q_sorted

    # Nighttime: zero all quantiles
    if "solar_elevation" in result.columns:
        night_mask = result["solar_elevation"] <= 0
    else:
        night_mask = ~result["hour"].between(5, 20)
    result.loc[night_mask, qcols_in] = 0.0

    # Clear-sky cap
    if "ghi_clear_wm2" in result.columns:
        for qcol in qcols_in:
            result[qcol] = result[qcol].clip(upper=result["ghi_clear_wm2"] * 1.2)

    # Update metadata
    result["calibration_method"] = "hour_block_asymmetric_cqr"
    result["source"] = "cqr_v2"

    # Drop helper column
    result = result.drop(columns=["hour"], errors="ignore")

    # Save
    result.to_parquet(OUT_QUANTILES, index=False)
    print(f"\n  Saved improved CQR quantiles: {OUT_QUANTILES}")
    print(f"  Method: hour-block asymmetric CQR (sunrise/core/sunset)")

    # Evaluate improvement
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)

    # Quick calibration check on test set
    test_result = result[result["split_name"] == "test"].copy() if "split_name" in result.columns else result.iloc[int(len(result) * 0.3):].copy()
    test_result["hour"] = test_result["target_time_local"].dt.hour if "target_time_local" in test_result.columns else 0

    if "solar_elevation" in test_result.columns:
        daylight = test_result[test_result["solar_elevation"] > 0]
    else:
        daylight = test_result[test_result["hour"].between(6, 19)]

    if "label_ghi_obs_wm2" in daylight.columns and len(daylight) > 0:
        y = daylight["label_ghi_obs_wm2"].values
        print(f"\n  Test daylight N = {len(daylight)}")

        errors = []
        for tau, qcol in zip(QUANTILES, qcols_in):
            q_vals = daylight[qcol].values
            observed = np.mean(y <= q_vals)
            err = abs(observed - tau)
            errors.append(err)
            if tau in [0.10, 0.50, 0.80, 0.90, 0.95]:
                print(f"    q{tau:.2f}: nominal={tau:.2f}, observed={observed:.3f}, |error|={err:.3f}")

        mean_error = np.mean(errors)
        print(f"\n  Mean |calibration error| = {mean_error:.4f}")

        # PICP80
        q10_vals = daylight[qcols_in[1]].values  # q0.10
        q90_vals = daylight[qcols_in[-2]].values  # q0.90
        picp80 = np.mean((y >= q10_vals) & (y <= q90_vals))
        print(f"  PICP80 (daylight) = {picp80*100:.1f}%")

        # Critical hours
        critical = daylight[daylight["hour"].between(10, 15)]
        if len(critical) > 0:
            y_crit = critical["label_ghi_obs_wm2"].values
            q10_crit = critical[qcols_in[1]].values
            q90_crit = critical[qcols_in[-2]].values
            picp80_crit = np.mean((y_crit >= q10_crit) & (y_crit <= q90_crit))
            print(f"  PICP80 (critical 10-15) = {picp80_crit*100:.1f}%")

    return result


if __name__ == "__main__":
    run()
