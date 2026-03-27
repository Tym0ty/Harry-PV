#!/usr/bin/env python3
"""
9_scenario_diagnostics.py  –  Chapter 4 scenario-display figures for Harry's thesis.

Outputs (to docs/figures/):
  1. scenario_cloud_vs_medoids.png   – 3-panel: clear / mixed / overcast
  2. scenario_reduction_detail.png   – single-day deep dive + histogram
  3. scenario_billing_risk.png       – billing-hour PV shortfall analysis

Data consumed:
  - pipeline_outputs/scenarios_joint_pv_load_raw_500.parquet
  - pipeline_outputs/scenarios_joint_pv_load_reduced_5.parquet
  - pipeline_outputs/pv_point_forecast_caseyear.parquet
  - bridge_outputs_fullyear/full_year_replay_truth_package.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RAW_PATH   = ROOT / "pipeline_outputs" / "scenarios_joint_pv_load_raw_500.parquet"
RED_PATH   = ROOT / "pipeline_outputs" / "scenarios_joint_pv_load_reduced_5.parquet"
DET_PATH   = ROOT / "pipeline_outputs" / "pv_point_forecast_caseyear.parquet"
TRUTH_PATH = ROOT / "bridge_outputs_fullyear" / "full_year_replay_truth_package.parquet"

# ── representative days ──────────────────────────────────────────────────────
DAY_CLEAR   = pd.Timestamp("2025-06-30")   # high total PV, stable profile
DAY_MIXED   = pd.Timestamp("2025-08-14")   # high intra-day variability
DAY_OVERCAST = pd.Timestamp("2025-01-15")  # very low PV
DAY_BILLING = pd.Timestamp("2025-09-16")   # peak summer load day

MEDOID_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# PV scaling: scenarios (raw, reduced, det) are at 50 kW reference scale.
# Truth package uses full campus PV capacity (2687 kW from bridge metadata).
# Scale truth down to 50 kW reference so all plots are consistent.
PV_REF_KW   = 50.0
PV_FIXED_KW = 2687.0
TRUTH_SCALE  = PV_REF_KW / PV_FIXED_KW   # ≈ 0.0186

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data …")
raw   = pd.read_parquet(RAW_PATH)
red   = pd.read_parquet(RED_PATH)
det   = pd.read_parquet(DET_PATH)
truth = pd.read_parquet(TRUTH_PATH)

# Align hour conventions: truth uses hour_local 1-24, raw/red use hour 0-23
# We plot on 0-23 axis.  Truth hour 1 → hour 0, etc.
truth["hour"] = truth["hour_local"] - 1

# Scale truth PV to 50 kW reference
truth["pv_realized_kw"] = truth["pv_realized_kw"] * TRUTH_SCALE

# Ensure reduced has an 'hour' column (derive from target_time_local if missing)
if "hour" not in red.columns:
    red["hour"] = red["target_time_local"].dt.hour


def _slice(df, day, hour_col="hour", day_col="target_day_local"):
    """Return subset for a single day."""
    return df[df[day_col] == day].copy()


def _truth_day(day):
    t = truth[truth["calendar_day"] == day].copy()
    return t.sort_values("hour")


def _det_day(day):
    d = det[det["target_day_local"] == day].copy()
    d["hour"] = d["target_time_local"].dt.hour
    return d.sort_values("hour")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – scenario_cloud_vs_medoids.png
# ══════════════════════════════════════════════════════════════════════════════
def fig1_cloud_vs_medoids():
    days = [DAY_CLEAR, DAY_MIXED, DAY_OVERCAST]
    titles = ["(a)  Clear day — 2025-06-30",
              "(b)  Mixed-cloud day — 2025-08-14",
              "(c)  Overcast day — 2025-01-15"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    fig.subplots_adjust(wspace=0.30, left=0.06, right=0.97, top=0.82, bottom=0.13)

    for ax, day, title in zip(axes, days, titles):
        # raw cloud
        r = _slice(raw, day)
        for sid in r["scenario_id"].unique():
            s = r[r["scenario_id"] == sid].sort_values("hour")
            ax.plot(s["hour"], s["pv_available_kw"], color="0.80", lw=0.25,
                    alpha=0.35, zorder=1)

        # medoids
        rd = _slice(red, day)
        for i, sid in enumerate(sorted(rd["scenario_id"].unique())):
            s = rd[rd["scenario_id"] == sid].sort_values("hour")
            prob = s["probability_pi"].iloc[0]
            ax.plot(s["hour"], s["pv_available_kw"], color=MEDOID_COLORS[i],
                    lw=2.2, zorder=3,
                    label=f"S{sid} (π={prob:.2f})")

        # deterministic Q50
        d = _det_day(day)
        ax.plot(d["hour"], d["pv_point_kw"], color="#1f78b4", lw=2.0,
                ls="--", zorder=4, label="Det. Q50")

        # truth
        t = _truth_day(day)
        ax.scatter(t["hour"], t["pv_realized_kw"], color="red", s=28,
                   zorder=5, edgecolors="darkred", linewidths=0.5,
                   label="Realized")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of day", fontsize=10)
        ax.set_xlim(-0.5, 23.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("PV available (kW, 50 kW ref.)", fontsize=10)

    # shared legend from middle axis (has representative set of entries)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.52, 0.99), frameon=True, fancybox=True,
               edgecolor="0.7")

    fig.savefig(OUT / "scenario_cloud_vs_medoids.png", dpi=300)
    plt.close(fig)
    print(f"  [1/3] scenario_cloud_vs_medoids.png  saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – scenario_reduction_detail.png
# ══════════════════════════════════════════════════════════════════════════════
def fig2_reduction_detail():
    day = DAY_MIXED  # most informative for reduction visualization

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8),
                                          gridspec_kw={"height_ratios": [3, 2]})
    fig.subplots_adjust(hspace=0.32, left=0.11, right=0.95, top=0.93, bottom=0.08)

    # ── upper panel: hourly profiles ──
    r = _slice(raw, day)
    for sid in r["scenario_id"].unique():
        s = r[r["scenario_id"] == sid].sort_values("hour")
        ax_top.plot(s["hour"], s["pv_available_kw"], color="0.80", lw=0.3,
                    alpha=0.30, zorder=1)

    rd = _slice(red, day)
    for i, sid in enumerate(sorted(rd["scenario_id"].unique())):
        s = rd[rd["scenario_id"] == sid].sort_values("hour")
        prob = s["probability_pi"].iloc[0]
        ax_top.plot(s["hour"], s["pv_available_kw"], color=MEDOID_COLORS[i],
                    lw=2.5, zorder=3,
                    label=f"Medoid S{sid}  (π = {prob:.3f})")

    d = _det_day(day)
    ax_top.plot(d["hour"], d["pv_point_kw"], color="#1f78b4", lw=2.0,
                ls="--", zorder=4, label="Det. Q50")

    t = _truth_day(day)
    ax_top.scatter(t["hour"], t["pv_realized_kw"], color="red", s=35,
                   zorder=5, edgecolors="darkred", linewidths=0.5,
                   label="Realized")

    ax_top.set_title(f"Scenario reduction: mixed-cloud day {day.strftime('%Y-%m-%d')}",
                     fontsize=12, fontweight="bold")
    ax_top.set_xlabel("Hour of day", fontsize=10)
    ax_top.set_ylabel("PV available (kW)", fontsize=10)
    ax_top.set_xlim(-0.5, 23.5)
    ax_top.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_top.legend(fontsize=8, loc="upper right", ncol=2)
    ax_top.grid(True, alpha=0.3)

    # ── lower panel: histogram of daily total PV ──
    daily_totals = r.groupby("scenario_id")["pv_available_kw"].sum()
    ax_bot.hist(daily_totals, bins=40, color="0.75", edgecolor="0.55", zorder=1,
                label="Raw 500 scenarios")

    # mark medoid daily totals
    med_totals = rd.groupby("scenario_id")["pv_available_kw"].sum()
    for i, (sid, total) in enumerate(med_totals.items()):
        prob = rd[rd["scenario_id"] == sid]["probability_pi"].iloc[0]
        ax_bot.axvline(total, color=MEDOID_COLORS[i], lw=2.2, ls="--", zorder=3,
                       label=f"Medoid S{sid} ({total:.0f} kWh, π={prob:.3f})")

    # mark truth
    truth_total = t["pv_realized_kw"].sum()
    ax_bot.axvline(truth_total, color="red", lw=2.5, ls="-", zorder=4,
                   label=f"Realized ({truth_total:.0f} kWh)")

    # mark deterministic
    det_total = d["pv_point_kw"].sum()
    ax_bot.axvline(det_total, color="#1f78b4", lw=2.0, ls=":", zorder=4,
                   label=f"Det. Q50 ({det_total:.0f} kWh)")

    ax_bot.set_xlabel("Daily total PV (kWh)", fontsize=10)
    ax_bot.set_ylabel("Count (of 500 scenarios)", fontsize=10)
    ax_bot.set_title("Distribution of daily PV generation across raw scenarios",
                     fontsize=11, fontweight="bold")
    ax_bot.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax_bot.grid(True, alpha=0.3)

    fig.savefig(OUT / "scenario_reduction_detail.png", dpi=300)
    plt.close(fig)
    print(f"  [2/3] scenario_reduction_detail.png  saved")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – scenario_billing_risk.png
# ══════════════════════════════════════════════════════════════════════════════
def fig3_billing_risk():
    """
    For billing-critical hours (10-15) on the peak summer day,
    compare PV shortfall distributions: raw 500 vs reduced 5.
    Also show maximum downward ramp per scenario.
    """
    day = DAY_BILLING
    billing_hours = list(range(10, 16))  # hours 10-15

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.95, top=0.88, bottom=0.12)

    # ── Left panel: billing-hour PV shortfall CDF ──
    r = _slice(raw, day)
    rd = _slice(red, day)
    d = _det_day(day)

    # Deterministic PV in billing hours
    det_billing = d[d["hour"].isin(billing_hours)]["pv_point_kw"].values

    # Raw: shortfall = det_billing_mean - scenario_billing for each scenario
    # Actually more useful: total PV in billing hours per scenario
    raw_billing_totals = (
        r[r["hour"].isin(billing_hours)]
        .groupby("scenario_id")["pv_available_kw"].sum()
        .values
    )

    # Reduced: same, weighted by probability
    red_billing = rd[rd["hour"].isin(billing_hours)]
    red_billing_totals = red_billing.groupby("scenario_id").agg(
        pv_total=("pv_available_kw", "sum"),
        prob=("probability_pi", "first")
    )

    # Deterministic total
    det_billing_total = det_billing.sum()

    # Shortfall relative to deterministic
    raw_shortfall = det_billing_total - raw_billing_totals
    red_shortfall = det_billing_total - red_billing_totals["pv_total"].values
    red_probs = red_billing_totals["prob"].values

    # CDF of raw shortfall
    raw_sorted = np.sort(raw_shortfall)
    raw_cdf = np.arange(1, len(raw_sorted) + 1) / len(raw_sorted)
    ax_left.plot(raw_sorted, raw_cdf, color="0.4", lw=2.0, label="Raw 500 scenarios")

    # Step function for reduced (weighted)
    order = np.argsort(red_shortfall)
    red_sorted = red_shortfall[order]
    red_prob_sorted = red_probs[order]
    red_cdf = np.cumsum(red_prob_sorted)
    # Plot as step
    for i in range(len(red_sorted)):
        y0 = 0.0 if i == 0 else red_cdf[i - 1]
        y1 = red_cdf[i]
        ax_left.plot([red_sorted[i], red_sorted[i]], [y0, y1],
                     color=MEDOID_COLORS[order[i]], lw=2.5, zorder=3)
        if i < len(red_sorted) - 1:
            ax_left.plot([red_sorted[i], red_sorted[i + 1]], [y1, y1],
                         color="0.2", lw=1.0, ls=":", zorder=2)
        # annotate
        ax_left.annotate(f"S{order[i]} (π={red_prob_sorted[i]:.2f})",
                         xy=(red_sorted[i], (y0 + y1) / 2),
                         xytext=(8, 0), textcoords="offset points",
                         fontsize=7.5, color=MEDOID_COLORS[order[i]],
                         fontweight="bold")

    ax_left.axvline(0, color="blue", ls=":", lw=1.2, alpha=0.6, label="Det. Q50 baseline")

    # truth shortfall
    t = _truth_day(day)
    truth_billing_total = t[t["hour"].isin(billing_hours)]["pv_realized_kw"].sum()
    truth_shortfall = det_billing_total - truth_billing_total
    ax_left.axvline(truth_shortfall, color="red", lw=2, ls="--",
                    label=f"Realized shortfall ({truth_shortfall:+.0f} kWh)")

    ax_left.set_xlabel("PV shortfall vs. deterministic (kWh, billing hours 10-15)", fontsize=9)
    ax_left.set_ylabel("Cumulative probability", fontsize=10)
    ax_left.set_title(f"(a) Billing-hour PV shortfall CDF\n{day.strftime('%Y-%m-%d')} (peak summer load)",
                      fontsize=11, fontweight="bold")
    ax_left.legend(fontsize=8, loc="lower right")
    ax_left.grid(True, alpha=0.3)

    # ── Right panel: maximum downward ramp per scenario ──
    def max_down_ramp(group):
        pv = group.sort_values("hour")["pv_available_kw"].values
        diffs = np.diff(pv)
        return diffs.min() if len(diffs) > 0 else 0.0

    raw_ramps = r.groupby("scenario_id").apply(max_down_ramp).values
    red_ramps_df = rd.groupby("scenario_id").apply(max_down_ramp).reset_index()
    red_ramps_df.columns = ["scenario_id", "max_ramp"]
    red_ramps_df["prob"] = rd.groupby("scenario_id")["probability_pi"].first().values

    ax_right.hist(raw_ramps, bins=40, color="0.75", edgecolor="0.55",
                  density=True, label="Raw 500 (density)", zorder=1)

    for i, row in red_ramps_df.iterrows():
        ax_right.axvline(row["max_ramp"], color=MEDOID_COLORS[int(row["scenario_id"])],
                         lw=2.2, ls="--", zorder=3,
                         label=f"S{int(row['scenario_id'])} (π={row['prob']:.2f})")

    # Truth ramp
    t_sorted = t.sort_values("hour")
    truth_ramp = np.diff(t_sorted["pv_realized_kw"].values).min()
    ax_right.axvline(truth_ramp, color="red", lw=2.5, zorder=4,
                     label=f"Realized ({truth_ramp:.0f} kW/h)")

    ax_right.set_xlabel("Maximum downward ramp (kW/h, most negative = steepest drop)", fontsize=9)
    ax_right.set_ylabel("Density", fontsize=10)
    ax_right.set_title(f"(b) Max downward PV ramp distribution\n{day.strftime('%Y-%m-%d')}",
                       fontsize=11, fontweight="bold")
    ax_right.legend(fontsize=7.5, loc="upper left")
    ax_right.grid(True, alpha=0.3)

    fig.savefig(OUT / "scenario_billing_risk.png", dpi=300)
    plt.close(fig)
    print(f"  [3/3] scenario_billing_risk.png  saved")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Chapter 4 scenario diagnostic figures …\n")
    fig1_cloud_vs_medoids()
    fig2_reduction_detail()
    fig3_billing_risk()
    print(f"\nAll figures saved to: {OUT}")
