#!/usr/bin/env python3
"""
Calibration Comparison: Raw vs Standard CQR vs Seasonal CQR vs Normalized CQR
===============================================================================
Applies three calibration strategies to the raw XGBQ quantile forecasts and
evaluates each on the test set. Saves a comparison reliability diagram and
metrics table.

Background
----------
The calibration set spans April–October 2024 (spring/summer/fall). The test
set spans November 2024–October 2025 (full year including winter). Standard
CQR calibrated only on warm months overcorrects when applied to cool months,
causing WORSE calibration than the raw model on the full test set.

Methods
-------
1. Raw          — uncalibrated XGBQ quantiles (baseline)
2. Standard CQR — global per-quantile additive shift (existing approach)
3. Seasonal CQR — separate calibration for warm (Apr-Oct) and cool (Nov-Mar)
                  seasons. For cool months in the test set where no cool
                  calibration data exists, falls back to raw (no correction).
4. Normalized CQR — divides calibration residuals by clear-sky GHI before
                  computing the conformal correction, then scales back by
                  clear-sky GHI at test time. Makes interval widths
                  proportional to sky brightness.

Run from repo root:
    python Project_Archive_Prediction_Final/9_calibration_comparison.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
RAW_FILE = Path("pipeline_outputs/forecast_ghi_quantiles_daily_base_raw.parquet")
OUT_DIR  = Path("Project_Archive_Prediction_Final/reports/comprehensive_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET  = "label_ghi_obs_wm2"
TAUS    = np.arange(0.05, 1.0, 0.05)          # 19 quantile levels
QCOLS   = [f"q{t:.2f}" for t in TAUS]
EPS     = 1.0                                  # small constant for normalization
DAYTIME = 10                                   # threshold W/m² for daylight filter

# Summer/warm months (calibration data only covers Apr–Oct)
WARM_MONTHS = {4, 5, 6, 7, 8, 9, 10}
COOL_MONTHS = {1, 2, 3, 11, 12}

DPI = 150

# ── Data loading ──────────────────────────────────────────────────────

def load_splits(path: Path):
    df = pd.read_parquet(path)
    df["target_time_local"] = pd.to_datetime(df["target_time_local"])
    df["month"] = df["target_time_local"].dt.month
    df["hour"]  = df["target_time_local"].dt.hour
    df["is_warm"] = df["month"].isin(WARM_MONTHS)
    cal  = df[df["split_name"] == "calib"].copy()
    test = df[df["split_name"] == "test"].copy()
    return cal, test


def daylight(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df[TARGET] > DAYTIME) | (df[QCOLS].max(axis=1) > DAYTIME)]


# ── CQR helpers ───────────────────────────────────────────────────────

def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal correction: (1-alpha)(1+1/n) quantile of scores."""
    n = len(scores)
    level = min((1 - alpha) * (1 + 1 / n), 1.0)
    return float(np.quantile(scores, level))


def apply_standard_cqr(cal: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Global per-quantile additive shift.
    For each τ: δ_τ = conformal_quantile(Y_i - q_τ^raw(X_i)) on cal set.
    Calibrated: q_τ^cal = q_τ^raw + δ_τ
    """
    out = test.copy()
    for tau, col in zip(TAUS, QCOLS):
        scores = cal[TARGET].values - cal[col].values    # Y - q (positive = under-predict)
        delta  = _finite_sample_quantile(scores, 1 - tau)
        out[col] = out[col] + delta
    out["calibration_method"] = "standard_cqr"
    return out


def apply_seasonal_cqr(cal: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Separate calibration for warm (Apr-Oct) and cool (Nov-Mar) months.
    Cool months have no calibration data → fall back to raw (no correction).
    """
    out = test.copy()
    cal_warm = cal[cal["is_warm"]]
    cal_cool = cal[~cal["is_warm"]]     # empty in this dataset

    for tau, col in zip(TAUS, QCOLS):
        delta_warm = np.nan
        delta_cool = 0.0    # default: no correction if no cool cal data

        if len(cal_warm) > 0:
            scores_warm = cal_warm[TARGET].values - cal_warm[col].values
            delta_warm  = _finite_sample_quantile(scores_warm, 1 - tau)

        if len(cal_cool) > 0:
            scores_cool = cal_cool[TARGET].values - cal_cool[col].values
            delta_cool  = _finite_sample_quantile(scores_cool, 1 - tau)
        # else: delta_cool = 0.0 (raw, no correction for cool months)

        # Apply season-matched correction
        warm_mask = out["is_warm"].values
        deltas    = np.where(warm_mask, delta_warm, delta_cool)
        out[col]  = out[col].values + deltas

    out["calibration_method"] = "seasonal_cqr"
    return out


def apply_normalized_cqr(cal: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Residuals normalized by clear-sky GHI before conformal correction.
    Score: s_i = (Y_i - q_τ^raw(X_i)) / (ghi_clear_i + EPS)
    Correction: δ_τ = conformal_quantile(s_i)
    Calibrated: q_τ^cal = q_τ^raw + δ_τ × (ghi_clear_test + EPS)
    """
    out = test.copy()
    ghi_clear_test = test["ghi_clear_wm2"].values + EPS
    ghi_clear_cal  = cal["ghi_clear_wm2"].values  + EPS

    for tau, col in zip(TAUS, QCOLS):
        residuals = cal[TARGET].values - cal[col].values      # Y - q_τ
        scores    = residuals / ghi_clear_cal                  # normalize
        delta     = _finite_sample_quantile(scores, 1 - tau)  # conformal correction
        out[col]  = out[col].values + delta * ghi_clear_test   # scale back
    out["calibration_method"] = "normalized_cqr"
    return out


# ── Evaluation ────────────────────────────────────────────────────────

def calibration_error(df: pd.DataFrame, label: str) -> dict:
    """Mean absolute calibration error and per-quantile observed coverage."""
    y  = df[TARGET].values
    obs = np.array([np.mean(y < df[c].values) for c in QCOLS])
    ace = np.abs(obs - TAUS)
    picp80 = float(np.mean((y >= df["q0.10"].values) & (y <= df["q0.90"].values)))
    picp90 = float(np.mean((y >= df["q0.05"].values) & (y <= df["q0.95"].values)))
    mpiw80 = float(np.mean(df["q0.90"].values - df["q0.10"].values))
    return {
        "method": label,
        "mean_abs_cal_error": float(ace.mean()),
        "PICP80": picp80,
        "PICP90": picp90,
        "MPIW80": mpiw80,
        "obs_freq": obs,
    }


# ── Figures ───────────────────────────────────────────────────────────

def fig_comparison(results: dict, out_path: Path):
    """4-method reliability diagram on daylight test set."""
    styles = {
        "Raw":            ("o-",  "salmon",      1.5),
        "Standard CQR":   ("s-",  "steelblue",   1.5),
        "Seasonal CQR":   ("^-",  "seagreen",    1.5),
        "Normalized CQR": ("D-",  "darkorange",  1.5),
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect calibration", zorder=0)

    # shaded ±2% band around diagonal
    band = 0.02
    ax.fill_between([0, 1], [0 - band, 1 - band], [0 + band, 1 + band],
                    color="grey", alpha=0.12, label=f"±{band:.0%} band")

    for label, info in results.items():
        marker, color, lw = styles[label]
        ace_str = f"{info['mean_abs_cal_error']:.4f}"
        ax.plot(TAUS, info["obs_freq"], marker, color=color, lw=lw,
                ms=5, label=f"{label}  (mean|err|={ace_str})")

    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage  (fraction below)")
    ax.set_title("Reliability Diagram — Calibration Methods Comparison\n(Daylight test hours)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path.name}")


def fig_seasonal_breakdown(results_warm: dict, results_cool: dict, out_path: Path):
    """Side-by-side reliability diagrams for warm vs cool months."""
    styles = {
        "Raw":            ("o-",  "salmon"),
        "Standard CQR":   ("s-",  "steelblue"),
        "Seasonal CQR":   ("^-",  "seagreen"),
        "Normalized CQR": ("D-",  "darkorange"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (title, results) in zip(axes, [
        ("Warm months (Apr–Oct)", results_warm),
        ("Cool months (Nov–Mar)", results_cool),
    ]):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect", zorder=0)
        ax.fill_between([0, 1], [-0.02, 0.98], [0.02, 1.02],
                        color="grey", alpha=0.12)
        for label, info in results.items():
            marker, color = styles[label]
            ace_str = f"{info['mean_abs_cal_error']:.4f}"
            ax.plot(TAUS, info["obs_freq"], marker, color=color,
                    lw=1.3, ms=4, label=f"{label} ({ace_str})")
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal quantile"); ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    fig.suptitle("Seasonal Calibration Breakdown", fontsize=13)
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading raw predictions ...")
    cal, test = load_splits(RAW_FILE)
    print(f"  Cal rows:  {len(cal)}  ({cal['target_time_local'].min().date()} – {cal['target_time_local'].max().date()})")
    print(f"  Test rows: {len(test)} ({test['target_time_local'].min().date()} – {test['target_time_local'].max().date()})")
    print(f"  Cal warm/cool: {cal['is_warm'].sum()}/{(~cal['is_warm']).sum()}")

    # ── Apply calibration methods ─────────────────────────────────────
    print("\nApplying calibration methods ...")
    test_raw    = test.copy(); test_raw["calibration_method"] = "raw"
    test_std    = apply_standard_cqr(cal, test)
    test_seas   = apply_seasonal_cqr(cal, test)
    test_norm   = apply_normalized_cqr(cal, test)
    print("  Done.")

    # ── Evaluate on daylight test rows ───────────────────────────────
    print("\nEvaluating (daylight) ...")
    variants = {
        "Raw":            daylight(test_raw),
        "Standard CQR":   daylight(test_std),
        "Seasonal CQR":   daylight(test_seas),
        "Normalized CQR": daylight(test_norm),
    }
    results_all = {k: calibration_error(v, k) for k, v in variants.items()}

    # Warm-month evaluation
    results_warm = {
        k: calibration_error(v[v["is_warm"]], k)
        for k, v in variants.items()
        if len(v[v["is_warm"]]) > 0
    }
    # Cool-month evaluation
    results_cool = {
        k: calibration_error(v[~v["is_warm"]], k)
        for k, v in variants.items()
        if len(v[~v["is_warm"]]) > 0
    }

    # ── Print summary table ──────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Method':<18} {'Mean|err|':>10} {'PICP80':>7} {'PICP90':>7} {'MPIW80':>8}")
    print("-"*65)
    for k, r in results_all.items():
        print(f"{k:<18} {r['mean_abs_cal_error']:>10.4f} {r['PICP80']:>7.3f} {r['PICP90']:>7.3f} {r['MPIW80']:>8.1f}")
    print("="*65)

    print("\nWarm months only:")
    print(f"{'Method':<18} {'Mean|err|':>10} {'PICP80':>7} {'PICP90':>7}")
    print("-"*40)
    for k, r in results_warm.items():
        print(f"{k:<18} {r['mean_abs_cal_error']:>10.4f} {r['PICP80']:>7.3f} {r['PICP90']:>7.3f}")

    print("\nCool months only:")
    print(f"{'Method':<18} {'Mean|err|':>10} {'PICP80':>7} {'PICP90':>7}")
    print("-"*40)
    for k, r in results_cool.items():
        print(f"{k:<18} {r['mean_abs_cal_error']:>10.4f} {r['PICP80']:>7.3f} {r['PICP90']:>7.3f}")

    # ── Save metrics CSV ─────────────────────────────────────────────
    rows = []
    for k, r in results_all.items():
        rows.append({"method": k, "scope": "all_daylight",
                     "mean_abs_cal_error": r["mean_abs_cal_error"],
                     "PICP80": r["PICP80"], "PICP90": r["PICP90"], "MPIW80": r["MPIW80"]})
    for k, r in results_warm.items():
        rows.append({"method": k, "scope": "warm_months",
                     "mean_abs_cal_error": r["mean_abs_cal_error"],
                     "PICP80": r["PICP80"], "PICP90": r["PICP90"], "MPIW80": r["MPIW80"]})
    for k, r in results_cool.items():
        rows.append({"method": k, "scope": "cool_months",
                     "mean_abs_cal_error": r["mean_abs_cal_error"],
                     "PICP80": r["PICP80"], "PICP90": r["PICP90"], "MPIW80": r["MPIW80"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "calibration_comparison.csv", index=False, float_format="%.6f")
    print(f"\n  -> calibration_comparison.csv")

    # ── Save calibrated parquets ─────────────────────────────────────
    test_seas.to_parquet("pipeline_outputs/forecast_ghi_quantiles_daily_seasonal_cqr.parquet", index=False)
    test_norm.to_parquet("pipeline_outputs/forecast_ghi_quantiles_daily_normalized_cqr.parquet", index=False)
    print("  -> forecast_ghi_quantiles_daily_seasonal_cqr.parquet")
    print("  -> forecast_ghi_quantiles_daily_normalized_cqr.parquet")

    # ── Figures ──────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    fig_comparison(results_all, OUT_DIR / "calibration_comparison.png")
    fig_seasonal_breakdown(results_warm, results_cool, OUT_DIR / "calibration_seasonal_breakdown.png")

    print("\nDone. All outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
