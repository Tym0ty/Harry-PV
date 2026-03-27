#!/usr/bin/env python3
"""
Comprehensive Probabilistic Forecast Evaluation
================================================
Produces every metric, table, and figure Harry requires for thesis Chapter 3.

Run from repo root:
    python Project_Archive_Prediction_Final/8_comprehensive_eval.py

Inputs
------
- pipeline_outputs/forecast_ghi_quantiles_daily.parquet          (CQR-calibrated)
- pipeline_outputs/forecast_ghi_quantiles_daily_base_raw.parquet (uncalibrated)

Outputs  (all under Project_Archive_Prediction_Final/reports/comprehensive_eval/)
-------
Tables:
  forecast_eval_master.csv      — full metrics for 3 scopes
  metrics_by_season.csv         — seasonal breakdown (daylight)
  metrics_by_month.csv          — monthly breakdown (daylight)
  metrics_by_hour.csv           — hourly breakdown (daylight)
  quantile_calibration.csv      — 19-quantile calibration table (2 scopes)

Figures:
  reliability_diagram.png       — raw vs CQR, daylight + critical hours
  pit_histograms.png            — 3-panel PIT with 95 % CI bands
  hourly_calibration.png        — 4-panel PICP80 / ACE80 / tail / MPIW by hour
  seasonal_calibration.png      — grouped bars PICP80 + MPIW80 by season
  quantile_fan_chart.png        — 3 representative days
  cqr_comparison.png            — reliability diagram raw vs CQR
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Configuration ────────────────────────────────────────────────────
CAL_FILE = Path("pipeline_outputs/forecast_ghi_quantiles_daily_normalized_cqr.parquet")
RAW_FILE = Path("pipeline_outputs/forecast_ghi_quantiles_daily_base_raw.parquet")
OUT_DIR  = Path("Project_Archive_Prediction_Final/reports/comprehensive_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label_ghi_obs_wm2"
QUANTILE_COLS = [f"q{tau:.2f}" for tau in np.arange(0.05, 1.0, 0.05)]
TAUS = np.arange(0.05, 1.0, 0.05)  # 0.05, 0.10, ..., 0.95  (19 levels)
CRITICAL_HOURS = list(range(10, 16))  # 10..15
DPI = 300

SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.constrained_layout.use": True,
})


# ── Helpers ──────────────────────────────────────────────────────────

def load_test(path: Path) -> pd.DataFrame:
    """Load parquet, keep only test split rows."""
    df = pd.read_parquet(path)
    df = df[df["split_name"] == "test"].copy()
    df["target_time_local"] = pd.to_datetime(df["target_time_local"])
    df["hour"] = df["target_time_local"].dt.hour
    df["month"] = df["target_time_local"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    return df


def daylight_mask(df: pd.DataFrame) -> pd.Series:
    """Daylight = observed GHI > 10  OR  q50 > 10."""
    return (df[TARGET_COL] > 10) | (df["q0.50"] > 10)


def critical_mask(df: pd.DataFrame) -> pd.Series:
    """Critical peak-PV hours (10-15) intersected with daylight."""
    return daylight_mask(df) & df["hour"].isin(CRITICAL_HOURS)


# ── Pinball / CRPS ──────────────────────────────────────────────────

def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Mean pinball (quantile) loss."""
    delta = y - q
    return np.mean(np.where(delta >= 0, tau * delta, (tau - 1) * delta))


def mean_pinball_all(df: pd.DataFrame) -> float:
    """Mean pinball averaged over all 19 quantiles (CRPS proxy)."""
    losses = [pinball_loss(df[TARGET_COL].values, df[c].values, tau)
              for c, tau in zip(QUANTILE_COLS, TAUS)]
    return float(np.mean(losses))


# ── Interval metrics ────────────────────────────────────────────────

def picp(y, lower, upper):
    return np.mean((y >= lower) & (y <= upper))

def ace(y, lower, upper, nominal):
    return picp(y, lower, upper) - nominal

def mpiw(lower, upper):
    return np.mean(upper - lower)


# ── PIT estimation (piecewise-linear across 19 quantiles) ───────────

def compute_pit(df: pd.DataFrame) -> np.ndarray:
    """Estimate PIT values using piecewise-linear interpolation."""
    y = df[TARGET_COL].values
    Q = df[QUANTILE_COLS].values  # (N, 19)
    n = len(y)
    pit = np.empty(n)
    for i in range(n):
        yi = y[i]
        qi = Q[i]
        if yi < qi[0]:
            pit[i] = 0.025  # below q0.05 -> midpoint of [0, 0.05]
        elif yi > qi[-1]:
            pit[i] = 0.975  # above q0.95 -> midpoint of [0.95, 1]
        else:
            # find bounding quantiles
            idx = np.searchsorted(qi, yi, side="right")
            idx = min(idx, len(TAUS) - 1)
            idx = max(idx, 1)
            tau_lo, tau_hi = TAUS[idx - 1], TAUS[idx]
            q_lo, q_hi = qi[idx - 1], qi[idx]
            if q_hi == q_lo:
                pit[i] = 0.5 * (tau_lo + tau_hi)
            else:
                frac = (yi - q_lo) / (q_hi - q_lo)
                pit[i] = tau_lo + frac * (tau_hi - tau_lo)
    return pit


# ── Build metrics for one scope ─────────────────────────────────────

def compute_metrics(df: pd.DataFrame, scope_name: str) -> dict:
    y = df[TARGET_COL].values
    q50 = df["q0.50"].values

    # Point
    mae = mean_absolute_error(y, q50)
    rmse = float(np.sqrt(mean_squared_error(y, q50)))
    bias = float(np.mean(q50 - y))
    r2 = r2_score(y, q50) if np.var(y) > 0 else np.nan

    # Probabilistic
    mpb = mean_pinball_all(df)
    crps_proxy = mpb  # average pinball across all quantiles

    # Per-quantile pinball
    pb10 = pinball_loss(y, df["q0.10"].values, 0.10)
    pb50 = pinball_loss(y, df["q0.50"].values, 0.50)
    pb90 = pinball_loss(y, df["q0.90"].values, 0.90)

    # Interval (80 %: q10-q90,  90 %: q05-q95)
    picp80 = picp(y, df["q0.10"].values, df["q0.90"].values)
    picp90 = picp(y, df["q0.05"].values, df["q0.95"].values)
    ace80  = ace(y, df["q0.10"].values, df["q0.90"].values, 0.80)
    ace90  = ace(y, df["q0.05"].values, df["q0.95"].values, 0.90)
    mpiw80 = mpiw(df["q0.10"].values, df["q0.90"].values)
    mpiw90 = mpiw(df["q0.05"].values, df["q0.95"].values)

    # Tail
    lower_hit = float(np.mean(y < df["q0.10"].values))
    upper_exceed = float(np.mean(y > df["q0.90"].values))

    return {
        "scope": scope_name,
        "N": len(df),
        "MAE": mae, "RMSE": rmse, "Bias": bias, "R2": r2,
        "Mean_Pinball": mpb, "CRPS_proxy": crps_proxy,
        "Pinball_P10": pb10, "Pinball_P50": pb50, "Pinball_P90": pb90,
        "PICP80": picp80, "PICP90": picp90,
        "ACE80": ace80, "ACE90": ace90,
        "MPIW80": mpiw80, "MPIW90": mpiw90,
        "Lower_tail_hit_q10": lower_hit, "Upper_tail_exceed_q90": upper_exceed,
    }


# ── Quantile-wise calibration ───────────────────────────────────────

def quantile_calibration_table(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    y = df[TARGET_COL].values
    rows = []
    for tau, col in zip(TAUS, QUANTILE_COLS):
        obs_freq = float(np.mean(y < df[col].values))
        rows.append({
            "scope": scope,
            "nominal": round(tau, 2),
            "observed_coverage": obs_freq,
            "abs_calibration_error": abs(obs_freq - tau),
        })
    return pd.DataFrame(rows)


# ── Grouped breakdown helper ─────────────────────────────────────────

def grouped_metrics(df: pd.DataFrame, groupcol: str) -> pd.DataFrame:
    rows = []
    for key, grp in df.groupby(groupcol):
        if len(grp) == 0:
            continue
        y = grp[TARGET_COL].values
        rows.append({
            groupcol: key,
            "N": len(grp),
            "PICP80": picp(y, grp["q0.10"].values, grp["q0.90"].values),
            "ACE80": ace(y, grp["q0.10"].values, grp["q0.90"].values, 0.80),
            "MPIW80": mpiw(grp["q0.10"].values, grp["q0.90"].values),
            "MAE": mean_absolute_error(y, grp["q0.50"].values),
            "Mean_Pinball": mean_pinball_all(grp),
        })
    return pd.DataFrame(rows)


def hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, grp in df.groupby("hour"):
        if len(grp) == 0:
            continue
        y = grp[TARGET_COL].values
        rows.append({
            "hour": h,
            "N": len(grp),
            "PICP80": picp(y, grp["q0.10"].values, grp["q0.90"].values),
            "ACE80": ace(y, grp["q0.10"].values, grp["q0.90"].values, 0.80),
            "MPIW80": mpiw(grp["q0.10"].values, grp["q0.90"].values),
            "Lower_tail_hit_q10": float(np.mean(y < grp["q0.10"].values)),
            "Pinball_P10": pinball_loss(y, grp["q0.10"].values, 0.10),
            "Pinball_P90": pinball_loss(y, grp["q0.90"].values, 0.90),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════

def _observed_freq(df: pd.DataFrame):
    """Observed frequency y < q for each of 19 quantiles."""
    y = df[TARGET_COL].values
    return np.array([np.mean(y < df[c].values) for c in QUANTILE_COLS])


def fig_reliability(df_cal: pd.DataFrame, df_raw: pd.DataFrame):
    """Reliability diagram: raw vs CQR for daylight and critical hours."""
    dl = df_cal[daylight_mask(df_cal)]
    cr = df_cal[critical_mask(df_cal)]
    dl_raw = df_raw[daylight_mask(df_raw)]
    cr_raw = df_raw[critical_mask(df_raw)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (cal, raw, title) in zip(axes, [
        (dl, dl_raw, "Daylight (all hours)"),
        (cr, cr_raw, "Critical hours (10-15)"),
    ]):
        obs_cal = _observed_freq(cal)
        obs_raw = _observed_freq(raw)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
        ax.plot(TAUS, obs_raw, "o-", ms=4, lw=1.2, label="Raw (uncalibrated)")
        ax.plot(TAUS, obs_cal, "s-", ms=4, lw=1.2, label="Normalized CQR")
        ax.set_xlabel("Nominal quantile level")
        ax.set_ylabel("Observed frequency (y < q)")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "reliability_diagram.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> reliability_diagram.png")


def fig_pit_histograms(df_cal: pd.DataFrame):
    """PIT histograms: all test, daylight, critical — with 95 % CI bands."""
    subsets = [
        ("All test", df_cal),
        ("Daylight only", df_cal[daylight_mask(df_cal)]),
        ("Critical hours (10-15)", df_cal[critical_mask(df_cal)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    n_bins = 10
    for ax, (label, sub) in zip(axes, subsets):
        pit = compute_pit(sub)
        ax.hist(pit, bins=n_bins, range=(0, 1), density=True,
                edgecolor="white", alpha=0.75, color="steelblue")
        # 95 % binomial CI for uniform: expected count = N/n_bins
        N = len(pit)
        p = 1.0 / n_bins
        se = np.sqrt(p * (1 - p) / N) * n_bins  # density scale
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.axhspan(1.0 - 1.96 * se, 1.0 + 1.96 * se,
                    color="grey", alpha=0.2, label="95% CI (uniform)")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}  (N={N})")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "pit_histograms.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> pit_histograms.png")


def fig_hourly_calibration(df_hourly: pd.DataFrame):
    """4-panel hourly calibration: PICP80, ACE80, tail q10, MPIW80."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    hours = df_hourly["hour"].values

    ax = axes[0, 0]
    ax.bar(hours, df_hourly["PICP80"], color="steelblue", width=0.7)
    ax.axhline(0.80, color="red", ls="--", lw=1, label="Nominal 80%")
    ax.set_ylabel("PICP80"); ax.set_title("80% Prediction Interval Coverage by Hour")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    colors = ["green" if v >= 0 else "salmon" for v in df_hourly["ACE80"]]
    ax.bar(hours, df_hourly["ACE80"], color=colors, width=0.7)
    ax.axhline(0, color="k", ls="-", lw=0.6)
    ax.set_ylabel("ACE80"); ax.set_title("Average Coverage Error (80%) by Hour")
    ax.set_xlabel("Hour"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.bar(hours, df_hourly["Lower_tail_hit_q10"], color="darkorange", width=0.7)
    ax.axhline(0.10, color="red", ls="--", lw=1, label="Nominal 10%")
    ax.set_ylabel("Fraction y < q10"); ax.set_title("Lower-Tail Hit Rate (q10) by Hour")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(hours, df_hourly["MPIW80"], color="mediumpurple", width=0.7)
    ax.set_ylabel("MPIW80 (W/m\u00b2)"); ax.set_title("Mean PI Width (80%) by Hour")
    ax.set_xlabel("Hour"); ax.grid(True, alpha=0.3)

    fig.savefig(OUT_DIR / "hourly_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> hourly_calibration.png")


def fig_seasonal(df_season: pd.DataFrame):
    """Grouped bar chart: PICP80 + MPIW80 by season."""
    df_season = df_season.set_index("season").reindex(SEASON_ORDER)
    x = np.arange(len(SEASON_ORDER))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 5))
    bars1 = ax1.bar(x - w / 2, df_season["PICP80"], w, label="PICP80", color="steelblue")
    ax1.axhline(0.80, color="red", ls="--", lw=1, label="Nominal 80%")
    ax1.set_ylabel("PICP80")
    ax1.set_xlabel("Season")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SEASON_ORDER)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, df_season["MPIW80"], w, label="MPIW80", color="mediumpurple", alpha=0.8)
    ax2.set_ylabel("MPIW80 (W/m\u00b2)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Seasonal PICP80 and MPIW80 (daylight)")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.savefig(OUT_DIR / "seasonal_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> seasonal_calibration.png")


def _pick_representative_days(df: pd.DataFrame):
    """Pick 1 clear, 1 mixed, 1 overcast day from daylight hours."""
    dl = df[daylight_mask(df)].copy()
    daily = dl.groupby(dl["target_time_local"].dt.date).agg(
        obs_mean=(TARGET_COL, "mean"),
        clear_mean=("ghi_clear_wm2", "mean"),
    )
    daily["clear_ratio"] = daily["obs_mean"] / daily["clear_mean"].clip(lower=1)

    # Sort by clear ratio
    daily = daily.sort_values("clear_ratio")
    # Overcast: lowest ratio, Clear: highest, Mixed: median
    overcast_day = daily.index[0]
    clear_day = daily.index[-1]
    mid = len(daily) // 2
    mixed_day = daily.index[mid]
    return [
        (clear_day, "Clear day"),
        (mixed_day, "Mixed day"),
        (overcast_day, "Overcast day"),
    ]


def fig_fan_chart(df_cal: pd.DataFrame):
    """Quantile fan chart for 3 representative days."""
    days = _pick_representative_days(df_cal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    band_specs = [
        ("q0.05", "q0.95", 0.15, "q05-q95"),
        ("q0.10", "q0.90", 0.25, "q10-q90"),
        ("q0.25", "q0.75", 0.35, "q25-q75"),
    ]
    for ax, (day, label) in zip(axes, days):
        mask = df_cal["target_time_local"].dt.date == day
        sub = df_cal[mask].sort_values("target_time_local")
        hours = sub["target_time_local"].dt.hour
        y = sub[TARGET_COL].values
        for lo, hi, alpha, blabel in band_specs:
            ax.fill_between(hours, sub[lo].values, sub[hi].values,
                            alpha=alpha, color="steelblue", label=blabel)
        ax.plot(hours, sub["q0.50"].values, "-", color="navy", lw=1.5, label="q50 (median)")
        ax.plot(hours, y, "o-", color="red", ms=3, lw=1, label="Observed")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("GHI (W/m\u00b2)")
        ax.set_title(f"{label}\n{day}")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(hours.min(), hours.max())
    fig.savefig(OUT_DIR / "quantile_fan_chart.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> quantile_fan_chart.png")


def fig_cqr_comparison(df_cal: pd.DataFrame, df_raw: pd.DataFrame):
    """Reliability diagram: raw vs CQR-calibrated (daylight)."""
    dl_cal = df_cal[daylight_mask(df_cal)]
    dl_raw = df_raw[daylight_mask(df_raw)]
    obs_cal = _observed_freq(dl_cal)
    obs_raw = _observed_freq(dl_raw)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.plot(TAUS, obs_raw, "o-", ms=5, lw=1.3, color="salmon", label="Raw (uncalibrated)")
    ax.plot(TAUS, obs_cal, "s-", ms=5, lw=1.3, color="steelblue", label="Normalized CQR")

    # Annotate improvement
    mae_raw = np.mean(np.abs(obs_raw - TAUS))
    mae_cal = np.mean(np.abs(obs_cal - TAUS))
    ax.text(0.05, 0.90,
            f"Mean |cal. error|:\n  Raw        = {mae_raw:.4f}\n  Norm. CQR = {mae_cal:.4f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Observed frequency (y < q)")
    ax.set_title("Calibration Comparison: Raw vs Normalized CQR (daylight)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "cqr_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> cqr_comparison.png")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("Loading data ...")
    df_cal = load_test(CAL_FILE)
    df_raw = load_test(RAW_FILE)
    print(f"  CQR test rows: {len(df_cal)}")
    print(f"  Raw test rows: {len(df_raw)}")

    # ── Scope masks (on calibrated data) ─────────────────────────────
    dl_mask = daylight_mask(df_cal)
    cr_mask = critical_mask(df_cal)

    df_dl   = df_cal[dl_mask]
    df_cr   = df_cal[cr_mask]
    df_all  = df_cal

    print(f"  Daylight rows: {len(df_dl)}")
    print(f"  Critical rows: {len(df_cr)}")

    # ── 1. Master metrics table ──────────────────────────────────────
    print("\n=== Metrics Table ===")
    rows = [
        compute_metrics(df_dl, "daylight_all"),
        compute_metrics(df_cr, "critical_10_15"),
        compute_metrics(df_all, "full_test_all"),
    ]
    master = pd.DataFrame(rows)
    master.to_csv(OUT_DIR / "forecast_eval_master.csv", index=False, float_format="%.6f")
    print(master.to_string(index=False))

    # ── 2. Season / Month breakdown ──────────────────────────────────
    print("\n=== Seasonal Breakdown (daylight) ===")
    df_season = grouped_metrics(df_dl, "season")
    # Reorder seasons
    df_season["season"] = pd.Categorical(df_season["season"], categories=SEASON_ORDER, ordered=True)
    df_season = df_season.sort_values("season")
    df_season.to_csv(OUT_DIR / "metrics_by_season.csv", index=False, float_format="%.6f")
    print(df_season.to_string(index=False))

    print("\n=== Monthly Breakdown (daylight) ===")
    df_month = grouped_metrics(df_dl, "month")
    df_month = df_month.sort_values("month")
    df_month.to_csv(OUT_DIR / "metrics_by_month.csv", index=False, float_format="%.6f")
    print(df_month.to_string(index=False))

    # ── 3. Hourly breakdown (daylight) ───────────────────────────────
    print("\n=== Hourly Breakdown (daylight) ===")
    df_hourly = hourly_metrics(df_dl)
    df_hourly = df_hourly.sort_values("hour")
    df_hourly.to_csv(OUT_DIR / "metrics_by_hour.csv", index=False, float_format="%.6f")
    print(df_hourly.to_string(index=False))

    # ── 4. Quantile-wise calibration table ───────────────────────────
    print("\n=== Quantile Calibration ===")
    cal_dl = quantile_calibration_table(df_dl, "daylight_all")
    cal_cr = quantile_calibration_table(df_cr, "critical_10_15")
    cal_all = pd.concat([cal_dl, cal_cr], ignore_index=True)
    cal_all.to_csv(OUT_DIR / "quantile_calibration.csv", index=False, float_format="%.6f")
    print(cal_all.to_string(index=False))

    # ── 5. Figures ───────────────────────────────────────────────────
    print("\n=== Generating Figures ===")
    fig_reliability(df_cal, df_raw)
    fig_pit_histograms(df_cal)
    fig_hourly_calibration(df_hourly)
    fig_seasonal(df_season)
    fig_fan_chart(df_cal)
    fig_cqr_comparison(df_cal, df_raw)

    print(f"\nAll outputs written to: {OUT_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
