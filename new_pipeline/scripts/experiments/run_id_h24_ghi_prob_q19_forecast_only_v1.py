#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forecast-only Q19 extension for the operational ID-H24 GHI probabilistic line.

This script is intentionally isolated from downstream scenario generation and
scheduling. It reuses the frozen 11-quantile Q2_LG outputs and trains only the
eight missing interior quantile levels needed for a 19-quantile forecast product.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
BASE_11Q = REPO / "new_pipeline/data/output/experiments/id_h24_ghi_prob"
H24_DATA = REPO / "new_pipeline/data/output/experiments/ghi_h24_model/rolling_h24_ghi_dataset.parquet"
LEGACY_SCRIPT = REPO / "new_pipeline/scripts/experiments/exp_id_h24_ghi_probabilistic.py"
OUT = REPO / "new_pipeline/data/output/experiments/id_h24_ghi_prob_q19_forecast_only_v1"

Q19_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
Q19_COLS = [f"q{int(q * 100):02d}" for q in Q19_LEVELS]
REUSED_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
REUSED_COLS = [f"q{int(q * 100):02d}" for q in REUSED_LEVELS]
MISSING_LEVELS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
MISSING_COLS = [f"q{int(q * 100):02d}" for q in MISSING_LEVELS]

CALIB_START = pd.Timestamp("2024-05-01")
TEST_START = pd.Timestamp("2024-11-01")
SEED = 42
MIN_TRAIN = 100


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_idh24_prob", LEGACY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["figures", "audit", "tables", "snapshots"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, OUT / "run_id_h24_ghi_prob_q19_forecast_only_v1.py")


def train_missing8_q2(legacy) -> pd.DataFrame:
    """Train only the missing eight Q2 lead-group quantile levels for test rows."""
    cache = OUT / "q19_missing8_quantile_test.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        for c in ["issue_time", "target_time"]:
            df[c] = pd.to_datetime(df[c])
        return df

    from xgboost import XGBRegressor

    df = pd.read_parquet(H24_DATA).copy()
    for c in ["issue_time", "target_time"]:
        df[c] = pd.to_datetime(df[c])
    feat_cols = legacy.get_feat_cols(df)
    df["_mp"] = df["target_time"].dt.to_period("M")
    months = sorted(df[df["target_time"] >= TEST_START]["_mp"].unique())
    parts = []

    for mp in months:
        mp_s = mp.to_timestamp()
        mp_e = (mp + 1).to_timestamp()
        tr = df[df["target_time"] < mp_s]
        te = df[(df["target_time"] >= mp_s) & (df["target_time"] < mp_e)]
        if len(te) == 0:
            continue
        for lg, leads in legacy.LEAD_GROUPS.items():
            tr_lg = tr[tr["horizon_h"].isin(leads)]
            te_lg = te[te["horizon_h"].isin(leads)]
            if len(te_lg) == 0:
                continue
            if len(tr_lg) < MIN_TRAIN:
                preds_2d = np.tile(
                    te_lg["csi_lag_tau"].fillna(0).values[:, None],
                    (1, len(MISSING_LEVELS)),
                )
            else:
                params = dict(legacy.XGB_PARAMS)
                params["random_state"] = SEED
                mdl = XGBRegressor(quantile_alpha=MISSING_LEVELS, **params)
                mdl.fit(tr_lg[feat_cols].fillna(0).values, tr_lg["y_csi_true"].values)
                preds_2d = mdl.predict(te_lg[feat_cols].fillna(0).values)
            preds_2d = np.clip(np.sort(preds_2d, axis=1), 0.0, 2.0)
            chunk = te_lg[["issue_time", "target_time", "horizon_h"]].copy()
            chunk["lead_group"] = lg
            for i, qc in enumerate(MISSING_COLS):
                chunk[qc] = np.clip(preds_2d[:, i] * te_lg["ghi_clear_target"].values, 0.0, None)
            parts.append(chunk)
        print(f"[Q19] trained missing quantiles for {mp}")

    out = pd.concat(parts, ignore_index=True).sort_values(["issue_time", "target_time", "horizon_h"])
    out.to_parquet(cache, index=False)
    return out


def clip_between(series: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    return np.minimum(np.maximum(series.to_numpy(float), lo.to_numpy(float)), hi.to_numpy(float))


def build_raw_q19(missing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(BASE_11Q / "q2_lg_quantile_test.parquet").copy()
    for c in ["issue_time", "target_time"]:
        base[c] = pd.to_datetime(base[c])
    keys = ["issue_time", "target_time", "horizon_h"]
    q19 = base.merge(missing[keys + MISSING_COLS], on=keys, how="left", validate="one_to_one")

    bounds = {
        "q15": ("q10", "q20"),
        "q25": ("q20", "q30"),
        "q35": ("q30", "q40"),
        "q45": ("q40", "q50"),
        "q55": ("q50", "q60"),
        "q65": ("q60", "q70"),
        "q75": ("q70", "q80"),
        "q85": ("q80", "q90"),
    }
    for q, (lo, hi) in bounds.items():
        q19[q] = clip_between(q19[q], q19[lo], q19[hi])

    q19["model_id"] = "Q2_LG_Q19_raw"
    keep = ["issue_time", "target_time", "horizon_h", "y_ghi_true", "y_csi_true",
            "ghi_clear_target", "lead_group", "model_id"] + Q19_COLS
    q19 = q19[keep].sort_values(keys).reset_index(drop=True)

    anchor_rows = []
    for qc in ["q05", "q10", "q50", "q90", "q95"]:
        diff = (q19[qc].to_numpy(float) - base.sort_values(keys)[qc].to_numpy(float))
        anchor_rows.append({
            "anchor_quantile": qc,
            "max_abs_diff": float(np.nanmax(np.abs(diff))),
            "status": "PASS" if float(np.nanmax(np.abs(diff))) <= 1e-10 else "FAIL",
        })
    audit = pd.DataFrame(anchor_rows)
    q19.to_parquet(OUT / "q19_raw_forecasts.parquet", index=False)
    audit.to_csv(OUT / "anchor_invariance_audit.csv", index=False, encoding="utf-8-sig")
    return q19, audit


def build_postcal_q19(raw_q19: pd.DataFrame) -> pd.DataFrame:
    cal11 = pd.read_parquet(BASE_11Q / "q2_lg_calibrated.parquet").copy()
    for c in ["issue_time", "target_time"]:
        cal11[c] = pd.to_datetime(cal11[c])
    keys = ["issue_time", "target_time", "horizon_h"]
    cols = keys + ["q05_cal90", "q10_cal80", "q90_cal80", "q95_cal90",
                   "q20_cal", "q30_cal", "q40_cal", "q50_cal", "q60_cal", "q70_cal", "q80_cal"]
    pc = raw_q19.merge(cal11[cols], on=keys, how="left", validate="one_to_one", suffixes=("", "_cal_src"))

    out = pc[["issue_time", "target_time", "horizon_h", "y_ghi_true", "y_csi_true",
              "ghi_clear_target", "lead_group"]].copy()
    out["model_id"] = "Q2_LG_Q19_postcal"
    out["q05"] = pc["q05_cal90"]
    out["q10"] = pc["q10_cal80"]
    out["q15"] = pc["q15"]
    out["q20"] = pc["q20_cal"]
    out["q25"] = pc["q25"]
    out["q30"] = pc["q30_cal"]
    out["q35"] = pc["q35"]
    out["q40"] = pc["q40_cal"]
    out["q45"] = pc["q45"]
    out["q50"] = pc["q50_cal"]
    out["q55"] = pc["q55"]
    out["q60"] = pc["q60_cal"]
    out["q65"] = pc["q65"]
    out["q70"] = pc["q70_cal"]
    out["q75"] = pc["q75"]
    out["q80"] = pc["q80_cal"]
    out["q85"] = pc["q85"]
    out["q90"] = pc["q90_cal80"]
    out["q95"] = pc["q95_cal90"]

    # Do not change calibrated endpoints. Interior quantiles are clipped into the
    # calibrated central interval and made monotone if required.
    interiors = ["q15", "q20", "q25", "q30", "q35", "q40", "q45", "q50",
                 "q55", "q60", "q65", "q70", "q75", "q80", "q85"]
    lo = out["q10"].to_numpy(float)
    hi = out["q90"].to_numpy(float)
    endpoint_cross = int(np.sum(hi < lo))
    for qc in interiors:
        out[qc] = np.minimum(np.maximum(out[qc].to_numpy(float), lo), hi)
    prev = out["q10"].to_numpy(float).copy()
    for qc in interiors:
        vals = np.maximum(out[qc].to_numpy(float), prev)
        vals = np.minimum(vals, hi)
        out[qc] = vals
        prev = vals

    out["q05_cal90"] = pc["q05_cal90"]
    out["q10_cal80"] = pc["q10_cal80"]
    out["q90_cal80"] = pc["q90_cal80"]
    out["q95_cal90"] = pc["q95_cal90"]
    out.attrs["endpoint_crossing_count"] = endpoint_cross
    out.to_parquet(OUT / "q19_postcal_forecasts.parquet", index=False)
    return out


def picp(y, lo, hi):
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean())


def mpiw(lo, hi):
    m = np.isfinite(lo) & np.isfinite(hi)
    return float((hi[m] - lo[m]).mean())


def pinball(y, q, alpha):
    e = y - q
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1.0) * e)))


def metric_rows(raw: pd.DataFrame, pc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    pb_rows = []
    for label, df in [("Raw Q19", raw), ("AgACI endpoint-calibrated Q19", pc)]:
        y = df["y_ghi_true"].to_numpy(float)
        pinballs = {}
        for alpha, qc in zip(Q19_LEVELS, Q19_COLS):
            val = pinball(y, df[qc].to_numpy(float), alpha)
            pinballs[qc] = val
            pb_rows.append({"method": label, "quantile": qc, "alpha": alpha, "pinball_loss": val})
        mean_pb = float(np.mean(list(pinballs.values())))
        crps = float(2.0 * mean_pb)
        p80 = picp(y, df["q10"].to_numpy(float), df["q90"].to_numpy(float))
        p90 = picp(y, df["q05"].to_numpy(float), df["q95"].to_numpy(float))
        rows.append({
            "Method": label,
            "PICP80": p80 * 100,
            "PICP90": p90 * 100,
            "ACE80": p80 - 0.80,
            "ACE90": p90 - 0.90,
            "MPIW80": mpiw(df["q10"].to_numpy(float), df["q90"].to_numpy(float)),
            "MPIW90": mpiw(df["q05"].to_numpy(float), df["q95"].to_numpy(float)),
            "Mean_Pinball_Q19": mean_pb,
            "CRPS_Q19": crps,
            "crps_columns": ",".join(Q19_COLS),
        })
    metrics = pd.DataFrame(rows)
    pinball_df = pd.DataFrame(pb_rows)
    table = metrics[["Method", "PICP80", "PICP90", "ACE80", "ACE90", "MPIW80", "MPIW90",
                     "Mean_Pinball_Q19", "CRPS_Q19"]].copy()
    metrics.to_csv(OUT / "q19_probabilistic_metrics.csv", index=False, encoding="utf-8-sig")
    pinball_df.to_csv(OUT / "q19_pinball_by_quantile.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT / "table_4_2_q19_raw_vs_agaci_calibration_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, pinball_df, table


def make_fan_chart(pc: pd.DataFrame) -> None:
    dates = [("Winter", "2025-01-11"), ("Spring", "2025-05-23"),
             ("Summer", "2025-07-06"), ("Autumn", "2025-09-14")]
    fig_rows = []
    selected = []
    tmp = pc.copy()
    tmp["target_time"] = pd.to_datetime(tmp["target_time"])
    tmp["issue_time"] = pd.to_datetime(tmp["issue_time"])
    for season, date_s in dates:
        date = pd.Timestamp(date_s).date()
        sub = tmp[tmp["target_time"].dt.date.eq(date)].copy()
        if sub.empty:
            selected.append((season, date_s, None, sub))
            continue
        issue = sub.groupby("issue_time").size().sort_values(ascending=False).index[0]
        day = sub[sub["issue_time"].eq(issue)].sort_values("target_time").copy()
        day["hour"] = day["target_time"].dt.hour
        rmse = float(np.sqrt(np.mean((day["q50"] - day["y_ghi_true"]) ** 2)))
        mae = float(np.mean(np.abs(day["q50"] - day["y_ghi_true"])))
        for _, r in day.iterrows():
            fig_rows.append({
                "season": season,
                "date": date_s,
                "selected_issue_time": r["issue_time"],
                "target_time": r["target_time"],
                "hour": int(r["hour"]),
                "observed_ghi": float(r["y_ghi_true"]),
                "q10": float(r["q10"]),
                "q25": float(r["q25"]),
                "q50": float(r["q50"]),
                "q75": float(r["q75"]),
                "q90": float(r["q90"]),
                "rmse": rmse,
                "mae": mae,
            })
        selected.append((season, date_s, issue, day, rmse, mae))

    pd.DataFrame(fig_rows).to_csv(OUT / "figure_4_1_q19_source.csv", index=False, encoding="utf-8-sig")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharey=True)
    axes = axes.ravel()
    for ax, item in zip(axes, selected):
        season, date_s, issue, day, rmse, mae = item
        if day.empty:
            ax.set_title(f"{season}: {date_s}\\nNo data")
            continue
        x = day["hour"].to_numpy()
        ax.fill_between(x, day["q10"], day["q90"], color="#9ecae1", alpha=0.45, label="P10-P90")
        ax.fill_between(x, day["q25"], day["q75"], color="#3182bd", alpha=0.35, label="P25-P75")
        ax.plot(x, day["q50"], color="#08519c", lw=1.8, label="Median")
        ax.plot(x, day["y_ghi_true"], color="#111111", lw=1.4, marker="o", ms=2.5, label="Observed")
        ax.set_title(f"{season}: {date_s}\\nRMSE={rmse:.1f}, MAE={mae:.1f} W/m²")
        ax.set_xlabel("Hour of day")
        ax.grid(alpha=0.2, lw=0.6)
    axes[0].set_ylabel("GHI (W/m²)")
    axes[2].set_ylabel("GHI (W/m²)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "figure_4_1_q19_seasonal_fan_chart.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "figure_4_1_q19_seasonal_fan_chart.pdf", facecolor="white")
    plt.close(fig)


def degradation_audit() -> tuple[pd.DataFrame, dict]:
    import yaml

    cfg_path = REPO / "milp_v2/config.yaml"
    design_path = REPO / "milp_v2/layer_a/designs/design_BASE.json"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    b = [float(x) for x in cfg["battery"]["degradation_segments"]["b"]]
    lam = [float(x) for x in cfg["battery"]["degradation_segments"]["lambda"]]
    eb = float(design["EB_star"])
    pb = float(design["PB_star"])
    repl = float(cfg["battery"]["C_BE_ntd_per_kwh"])

    def g(dod: float) -> float:
        rem = dod
        total = 0.0
        for k, lk in enumerate(lam):
            width = b[k + 1] - b[k]
            seg = max(0.0, min(rem, width))
            total += lk * seg
            rem -= seg
            if rem <= 1e-12:
                break
        return total

    g08 = g(0.8)
    pnnl_cycles = {0.05: 192000, 0.30: 32000, 0.60: 8000, 0.80: 6000}
    rows = []
    for dod in [0.05, 0.10, 0.30, 0.60, 0.80]:
        rows.append({
            "metric": "damage_curve",
            "DOD": dod,
            "current_g_ntd_per_kwh_capacity": g(dod),
            "current_normalized_damage_vs_g08": g(dod) / g08 if g08 else math.nan,
            "PNNL_cycles": pnnl_cycles.get(dod, math.nan),
            "PNNL_normalized_damage_vs_80pct": (6000 / pnnl_cycles[dod]) if dod in pnnl_cycles else math.nan,
            "source_file": "milp_v2/config.yaml",
        })
    for cyc in [2475, 4550, 6250]:
        rows.append({
            "metric": "replacement_cost_sensitivity",
            "DOD": 0.80,
            "cycles": cyc,
            "replacement_cost_implied_ntd_per_kwh": cyc * g08,
            "source_file": "user_requested_literature_sensitivity",
        })
    n80 = repl / g08 if g08 else math.nan
    rows.append({
        "metric": "repository_replacement_cost_implied_cycles",
        "DOD": 0.80,
        "replacement_cost_ntd_per_kwh": repl,
        "N80_implied_cycles": n80,
        "source_file": "milp_v2/config.yaml",
        "source_line": "C_BE_ntd_per_kwh",
    })
    max_frac = pb / eb
    rows.append({
        "metric": "bess_physical_hourly_discharge_fraction",
        "P_dis_max_kw": pb,
        "E_B_kwh": eb,
        "max_hourly_discharge_fraction": max_frac,
        "all_segments_individually_reachable_in_one_hour": bool(max_frac >= max(np.diff(b))),
        "source_file": "milp_v2/layer_a/designs/design_BASE.json",
    })

    # Segment utilization from existing realized replay, without rerunning optimization.
    hourly_candidates = [
        REPO / "new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/scheduling/D_G1_R3/D_G1_R3_CCOR_PROB_m150_hourly_dispatch_replay.parquet",
        REPO / "new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/scheduling/D_G1_R3/D_G1_R3_MPC_PROB_hourly_dispatch_replay.parquet",
    ]
    util_status = "NO_HOURLY_REPLAY_FOUND"
    for hp in hourly_candidates:
        if not hp.exists():
            continue
        h = pd.read_parquet(hp)
        col = "p_dis_exec" if "p_dis_exec" in h.columns else "p_dis"
        if col not in h.columns:
            continue
        seg_energy = np.zeros(len(lam))
        for p in h[col].fillna(0).to_numpy(float):
            rem = p
            for k, lk in enumerate(lam):
                cap = (b[k + 1] - b[k]) * eb
                seg = min(max(rem, 0.0), cap)
                seg_energy[k] += seg
                rem -= seg
                if rem <= 0:
                    break
        seg_cost = seg_energy * np.array(lam)
        total_cost = float(seg_cost.sum())
        for k in range(len(lam)):
            rows.append({
                "metric": "realized_segment_utilization",
                "case_source": str(hp.relative_to(REPO)).replace("\\", "/"),
                "segment_k": k + 1,
                "segment_lower": b[k],
                "segment_upper": b[k + 1],
                "annual_discharge_energy_kwh_by_segment": float(seg_energy[k]),
                "annual_degradation_cost_ntd_by_segment": float(seg_cost[k]),
                "percentage_of_total_degradation_cost_by_segment": float(seg_cost[k] / total_cost * 100) if total_cost else 0.0,
            })
        util_status = "SEGMENT_UTILIZATION_COMPUTED"
        break

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "battery_degradation_parameter_defensibility_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "g_0_1": g(0.1),
        "g_0_3": g(0.3),
        "g_0_6": g(0.6),
        "g_0_8": g08,
        "N80_implied": n80,
        "max_hourly_discharge_fraction": max_frac,
        "segment_utilization_status": util_status,
    }
    return df, summary


def write_report(metrics: pd.DataFrame, anchor: pd.DataFrame, deg_summary: dict, postcal: pd.DataFrame) -> None:
    raw_row = metrics[metrics["Method"].eq("Raw Q19")].iloc[0]
    cal_row = metrics[metrics["Method"].str.contains("AgACI")].iloc[0]
    anchor_status = "PASS" if anchor["status"].eq("PASS").all() else "DOWNSTREAM_ANCHOR_INVARIANCE_FAIL"
    report = f"""# Q19 Forecast-Only Final Report

## Scope

This independent forecast-only line extends the operational ID-H24 GHI Q2_LG probabilistic forecast from the frozen 11-quantile grid to a 19-quantile grid. It does not rerun scenario generation, scenario reduction, MPC, CCOR-MPC, MILP, or realized-operation replay.

## Output Root

`{OUT.relative_to(REPO).as_posix()}`

## Quantile Handling

- Reused frozen quantiles: `{', '.join(REUSED_COLS)}`.
- Newly trained quantiles: `{', '.join(MISSING_COLS)}`.
- Q19 grid: `{', '.join(Q19_COLS)}`.
- Missing quantiles were clipped between adjacent frozen anchors.
- AgACI is kept as endpoint calibration for 80% and 90% central intervals only.
- AgACI calibrates the 80% and 90% interval endpoints; it does not independently calibrate every interior quantile.

## Anchor Invariance

{anchor.to_markdown(index=False)}

Anchor invariance status: `{anchor_status}`.

## Q19 Metrics

{metrics[['Method','PICP80','PICP90','ACE80','ACE90','MPIW80','MPIW90','Mean_Pinball_Q19','CRPS_Q19']].to_markdown(index=False)}

CRPS raw columns: `{', '.join(Q19_COLS)}`.

CRPS post-calibration columns: `{', '.join(Q19_COLS)}` from `q19_postcal_forecasts.parquet`, where q05/q10/q90/q95 are AgACI endpoint-calibrated and interior quantiles are the Q19 interior forecast product constrained into the calibrated central interval.

## Figure 4-1

The Q19 fan chart uses direct `q25` and `q75` columns. No P25/P75 interpolation is used.

- PNG: `figure_4_1_q19_seasonal_fan_chart.png`
- PDF: `figure_4_1_q19_seasonal_fan_chart.pdf`
- Source: `figure_4_1_q19_source.csv`

## Battery Degradation Defensibility Audit

- g(0.1) = {deg_summary['g_0_1']:.6f}
- g(0.3) = {deg_summary['g_0_3']:.6f}
- g(0.6) = {deg_summary['g_0_6']:.6f}
- g(0.8) = {deg_summary['g_0_8']:.6f}
- Repository BESS energy capital cost = 7738 NTD/kWh.
- Implied N80 = {deg_summary['N80_implied']:.2f} cycles.
- Max hourly discharge fraction = {deg_summary['max_hourly_discharge_fraction']:.4f}.
- Segment utilization status = `{deg_summary['segment_utilization_status']}`.

Full audit table: `battery_degradation_parameter_defensibility_audit.csv`.

## Non-Rerun Guarantees

- Scenario generation rerun: FALSE
- Scenario reduction rerun: FALSE
- Optimization rerun: FALSE
- Formal 11q outputs overwritten: FALSE
"""
    write_text(OUT / "Q19_FORECAST_ONLY_FINAL_REPORT.md", report)


def main() -> int:
    setup()
    legacy = load_legacy_module()
    missing = train_missing8_q2(legacy)
    raw_q19, anchor = build_raw_q19(missing)
    if not anchor["status"].eq("PASS").all():
        write_text(OUT / "FINAL_STATUS.txt", "FINAL_STATUS: FAILED\nANCHOR_INVARIANCE_STATUS: DOWNSTREAM_ANCHOR_INVARIANCE_FAIL\n")
        return 2
    postcal = build_postcal_q19(raw_q19)
    metrics, pinball_df, table = metric_rows(raw_q19, postcal)
    make_fan_chart(postcal)
    deg_df, deg_summary = degradation_audit()
    write_report(metrics, anchor, deg_summary, postcal)

    raw_row = metrics[metrics["Method"].eq("Raw Q19")].iloc[0]
    cal_row = metrics[metrics["Method"].str.contains("AgACI")].iloc[0]
    seg_util = deg_df[deg_df["metric"].eq("realized_segment_utilization")]
    used_segments = sorted(seg_util.loc[seg_util["annual_discharge_energy_kwh_by_segment"].fillna(0) > 0, "segment_k"].dropna().astype(int).tolist())
    status_lines = [
        "FINAL_STATUS: SUCCESS",
        f"OUTPUT_ROOT: {OUT.relative_to(REPO).as_posix()}",
        "Q19_MODEL_COUNT: 19",
        f"NEWLY_TRAINED_QUANTILES: {','.join(MISSING_COLS)}",
        f"REUSED_QUANTILES: {','.join(REUSED_COLS)}",
        "ANCHOR_INVARIANCE_STATUS: PASS",
        f"RAW_Q19_CRPS: {raw_row['CRPS_Q19']:.6f}",
        f"POSTCAL_Q19_CRPS: {cal_row['CRPS_Q19']:.6f}",
        f"PICP80_RAW: {raw_row['PICP80']:.4f}",
        f"PICP80_CAL: {cal_row['PICP80']:.4f}",
        f"PICP90_RAW: {raw_row['PICP90']:.4f}",
        f"PICP90_CAL: {cal_row['PICP90']:.4f}",
        "FIGURE_4_1_P25_P75_DIRECT: TRUE",
        "SCENARIO_GENERATION_RERUN: FALSE",
        "OPTIMIZATION_RERUN: FALSE",
        "FORMAL_OUTPUTS_OVERWRITTEN: FALSE",
        f"BATTERY_DEGRADATION_IMPLIED_CYCLE_LIFE: {deg_summary['N80_implied']:.2f}",
        f"BATTERY_SEGMENT_UTILIZATION_STATUS: {deg_summary['segment_utilization_status']}; used_segments={used_segments}",
        "PARAMETER_DEFENSIBILITY_STATUS: PARAMETER_PROVENANCE_STILL_REQUIRES_TEXTUAL_DEFENSE",
    ]
    write_text(OUT / "FINAL_STATUS.txt", "\n".join(status_lines) + "\n")
    print("\n".join(status_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
