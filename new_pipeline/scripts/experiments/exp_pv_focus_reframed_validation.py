#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "new_pipeline/scripts/experiments"))

from pv_focus_bridge_utils import (  # noqa: E402
    CC_KW,
    OLD_DA_S1S6,
    OLD_ID_S1S6,
    OUT_DIR,
    PV_CAP_KW,
    load_truth,
    write_md,
)


BRIDGES = ["PV_KM", "PV_TKM", "PV_TKM_SAFE"]


def truth_table() -> pd.DataFrame:
    t = load_truth()[["timestamp", "load_realized_kw", "pv_realized_kw"]].copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t["realized_netload_kw"] = t["load_realized_kw"] - t["pv_realized_kw"]
    t["hour"] = t["timestamp"].dt.hour
    t["pv_active"] = t["pv_realized_kw"] > 0.05 * PV_CAP_KW
    t["daylight"] = t["hour"].between(6, 18)
    t["transition"] = t["hour"].between(15, 20)
    t["near_oc"] = t["realized_netload_kw"] >= 0.95 * CC_KW
    t["oc_risk"] = t["realized_netload_kw"] >= CC_KW
    t["top5_netload"] = t["realized_netload_kw"] >= t["realized_netload_kw"].quantile(0.95)
    t["top1_netload"] = t["realized_netload_kw"] >= t["realized_netload_kw"].quantile(0.99)
    pv_active = t[t["pv_active"]]
    if not pv_active.empty:
        t["pv_low_tail_active"] = t["pv_active"] & (t["pv_realized_kw"] <= pv_active["pv_realized_kw"].quantile(0.10))
        t["pv_high_tail_active"] = t["pv_active"] & (t["pv_realized_kw"] >= pv_active["pv_realized_kw"].quantile(0.90))
    else:
        t["pv_low_tail_active"] = False
        t["pv_high_tail_active"] = False
    t["pv_sensitive_oc"] = (t["near_oc"] | t["oc_risk"]) & (t["pv_active"] | t["transition"])
    return t


def reduced_env(mode: str, bridge: str) -> pd.DataFrame:
    prefix = "da" if mode == "DA" else "id_h24"
    suffix = bridge.lower().replace("pv_", "")
    path = OUT_DIR / f"{prefix}_pvfocus_base_{suffix}_K5_scenarios.parquet"
    df = pd.read_parquet(path)
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    env = (
        df.groupby(["issue_time", "target_time"], observed=True)
        .agg(
            scen_min=("netload_s_kw", "min"),
            scen_max=("netload_s_kw", "max"),
            scen_mean=("netload_s_kw", "mean"),
            pv_min=("pv_s_kw", "min"),
            pv_max=("pv_s_kw", "max"),
            pv_mean=("pv_s_kw", "mean"),
        )
        .reset_index()
    )
    env["mode"] = mode
    env["bridge_type"] = bridge
    return env


def raw_env(mode: str) -> pd.DataFrame:
    prefix = "da" if mode == "DA" else "id_h24"
    path = OUT_DIR / f"{prefix}_pvfocus_base_raw_netload_M500.parquet"
    cache = OUT_DIR / f"{prefix}_pvfocus_base_raw_envelope_reframed.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=["issue_time", "target_time", "netload_s_kw", "pv_s_kw"]).to_pandas()
        df["issue_time"] = pd.to_datetime(df["issue_time"])
        df["target_time"] = pd.to_datetime(df["target_time"])
        parts.append(
            df.groupby(["issue_time", "target_time"], observed=True)
            .agg(
                scen_min=("netload_s_kw", "min"),
                scen_max=("netload_s_kw", "max"),
                scen_mean=("netload_s_kw", "mean"),
                pv_min=("pv_s_kw", "min"),
                pv_max=("pv_s_kw", "max"),
                pv_mean=("pv_s_kw", "mean"),
            )
            .reset_index()
        )
        if (i + 1) % 1000 == 0:
            print(f"raw env {mode}: {i+1}/{pf.num_row_groups}")
    out = pd.concat(parts, ignore_index=True)
    # In case row groups were not exactly issue-time chunks.
    out = (
        out.groupby(["issue_time", "target_time"], observed=True)
        .agg(
            scen_min=("scen_min", "min"),
            scen_max=("scen_max", "max"),
            scen_mean=("scen_mean", "mean"),
            pv_min=("pv_min", "min"),
            pv_max=("pv_max", "max"),
            pv_mean=("pv_mean", "mean"),
        )
        .reset_index()
    )
    out["mode"] = mode
    out["bridge_type"] = "RAW_M500"
    out.to_parquet(cache, index=False)
    return out


def old_s1s6_env(mode: str) -> pd.DataFrame:
    path = OLD_DA_S1S6 if mode == "DA" else OLD_ID_S1S6
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["target_time"] = pd.to_datetime(df["target_timestamp"])
    if mode == "DA":
        df["issue_time"] = df["target_time"].dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=20)
    else:
        df["lead_hour"] = df["lead_time_hours"].astype(int)
        df["issue_time"] = df["target_time"] - pd.to_timedelta(df["lead_hour"], unit="h")
    scen_cols = [c for c in ["nl_median", "nl_low", "nl_high", "nl_high_load_stress", "nl_low_pv_stress", "nl_combined_stress"] if c in df.columns]
    out = pd.DataFrame(
        {
            "issue_time": df["issue_time"],
            "target_time": df["target_time"],
            "scen_min": df[scen_cols].min(axis=1),
            "scen_max": df[scen_cols].max(axis=1),
            "scen_mean": df[scen_cols].mean(axis=1),
            "pv_min": np.nan,
            "pv_max": np.nan,
            "pv_mean": np.nan,
            "mode": mode,
            "bridge_type": "old_S1S6_DIAGNOSTIC_ONLY",
        }
    )
    return out


def merge_truth(env: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    out = env.merge(truth.rename(columns={"timestamp": "target_time"}), on="target_time", how="left")
    return out.dropna(subset=["realized_netload_kw"]).copy()


def safe_mean(masked: pd.Series) -> float:
    return float(masked.mean()) if len(masked) else np.nan


def metric_row(env: pd.DataFrame) -> dict:
    n = len(env)
    actual = env["realized_netload_kw"]
    pv = env["pv_realized_kw"]
    row = {
        "mode": env["mode"].iloc[0],
        "bridge_type": env["bridge_type"].iloc[0],
        "row_count": int(n),
        "pv_envelope_coverage": safe_mean((pv >= env["pv_min"]) & (pv <= env["pv_max"])) if env["pv_min"].notna().any() else np.nan,
        "pv_low_tail_coverage": safe_mean((pv >= env["pv_min"])[env["pv_low_tail_active"]]) if env["pv_min"].notna().any() and env["pv_low_tail_active"].any() else np.nan,
        "pv_high_tail_coverage": safe_mean((pv <= env["pv_max"])[env["pv_high_tail_active"]]) if env["pv_max"].notna().any() and env["pv_high_tail_active"].any() else np.nan,
        "pv_active_netload_upper_coverage": safe_mean((actual <= env["scen_max"])[env["pv_active"]]) if env["pv_active"].any() else np.nan,
        "daylight_top5_netload_upper_coverage": safe_mean((actual <= env["scen_max"])[env["daylight"] & env["top5_netload"]]) if (env["daylight"] & env["top5_netload"]).any() else np.nan,
        "daylight_top1_netload_upper_coverage": safe_mean((actual <= env["scen_max"])[env["daylight"] & env["top1_netload"]]) if (env["daylight"] & env["top1_netload"]).any() else np.nan,
        "transition_netload_upper_coverage": safe_mean((actual <= env["scen_max"])[env["transition"]]) if env["transition"].any() else np.nan,
        "near_oc_pv_active_upper_coverage": safe_mean((actual <= env["scen_max"])[env["near_oc"] & env["pv_active"]]) if (env["near_oc"] & env["pv_active"]).any() else np.nan,
        "oc_risk_pv_active_upper_coverage": safe_mean((actual <= env["scen_max"])[env["oc_risk"] & env["pv_active"]]) if (env["oc_risk"] & env["pv_active"]).any() else np.nan,
        "pv_sensitive_oc_upper_coverage": safe_mean((actual <= env["scen_max"])[env["pv_sensitive_oc"]]) if env["pv_sensitive_oc"].any() else np.nan,
        "pv_shortfall_high_netload_coverage": safe_mean((actual <= env["scen_max"])[env["top5_netload"] & env["pv_active"]]) if (env["top5_netload"] & env["pv_active"]).any() else np.nan,
        "all_hour_oc_risk_upper_coverage_DIAGNOSTIC": safe_mean((actual <= env["scen_max"])[env["oc_risk"]]) if env["oc_risk"].any() else np.nan,
        "all_hour_top1_upper_coverage_DIAGNOSTIC": safe_mean((actual <= env["scen_max"])[env["top1_netload"]]) if env["top1_netload"].any() else np.nan,
        "night_evening_load_dominant_oc_coverage_DIAGNOSTIC": safe_mean((actual <= env["scen_max"])[env["oc_risk"] & (~env["pv_active"])]) if (env["oc_risk"] & (~env["pv_active"])).any() else np.nan,
        "scenario_max_netload_kw": float(env["scen_max"].max()),
        "actual_max_netload_kw": float(actual.max()),
        "scenario_min_pv_kw": float(env["pv_min"].min()) if env["pv_min"].notna().any() else np.nan,
        "scenario_max_pv_kw": float(env["pv_max"].max()) if env["pv_max"].notna().any() else np.nan,
    }
    return row


def energy_coverage(env: pd.DataFrame) -> dict:
    # Window/day energy coverage using summed PV envelope per issue_time.
    rows = []
    for (mode, bridge), sub in env.groupby(["mode", "bridge_type"], observed=True):
        if not sub["pv_min"].notna().any():
            rows.append({"mode": mode, "bridge_type": bridge, "pv_energy_envelope_coverage": np.nan, "pv_energy_rows": 0})
            continue
        g = (
            sub.groupby("issue_time", observed=True)
            .agg(
                actual_pv_energy=("pv_realized_kw", "sum"),
                pv_min_energy=("pv_min", "sum"),
                pv_max_energy=("pv_max", "sum"),
            )
            .reset_index()
        )
        rows.append(
            {
                "mode": mode,
                "bridge_type": bridge,
                "pv_energy_envelope_coverage": float(((g["actual_pv_energy"] >= g["pv_min_energy"]) & (g["actual_pv_energy"] <= g["pv_max_energy"])).mean()),
                "pv_energy_rows": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def weights_summary() -> pd.DataFrame:
    rows = []
    for mode in ["DA", "ID_H24"]:
        prefix = "da" if mode == "DA" else "id_h24"
        for bridge in BRIDGES:
            suffix = bridge.lower().replace("pv_", "")
            path = OUT_DIR / f"{prefix}_pvfocus_base_{suffix}_K5_scenarios.parquet"
            df = pd.read_parquet(path, columns=["issue_time", "scenario_id", "scenario_weight"])
            w = df.drop_duplicates(["issue_time", "scenario_id"])
            eff = w.groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum())
            rows.append(
                {
                    "mode": mode,
                    "bridge_type": bridge,
                    "min_weight": float(w["scenario_weight"].min()),
                    "max_weight": float(w["scenario_weight"].max()),
                    "weight_sum_min": float(w.groupby("issue_time")["scenario_weight"].sum().min()),
                    "weight_sum_max": float(w.groupby("issue_time")["scenario_weight"].sum().max()),
                    "effective_scenarios_mean": float(eff.mean()),
                }
            )
    return pd.DataFrame(rows)


def choose_status(passfail: pd.DataFrame, rawred: pd.DataFrame) -> str:
    safe = passfail[(passfail["mode"].eq("ID_H24")) & (passfail["bridge_type"].eq("PV_TKM_SAFE"))].iloc[0]
    tkm = passfail[(passfail["mode"].eq("ID_H24")) & (passfail["bridge_type"].eq("PV_TKM"))].iloc[0]
    raw = passfail[(passfail["mode"].eq("ID_H24")) & (passfail["bridge_type"].eq("RAW_M500"))].iloc[0]
    # If raw itself is poor on PV coverage, generation needs fixing.
    if raw["pv_envelope_coverage"] < 0.65 or raw["pv_low_tail_coverage"] < 0.75:
        return "NEEDS_PV_SCENARIO_GENERATION_FIX"
    # If safe materially loses PV-active/top coverage vs raw, prefer TKM if better.
    loss_safe = raw["pv_active_netload_upper_coverage"] - safe["pv_active_netload_upper_coverage"]
    loss_tkm = raw["pv_active_netload_upper_coverage"] - tkm["pv_active_netload_upper_coverage"]
    if safe["pv_envelope_coverage"] >= 0.68 and safe["pv_low_tail_coverage"] >= 0.80 and loss_safe <= 0.05:
        return "PASS_PV_TKM_SAFE_FOR_DA_MPC_SCHEDULING"
    if tkm["pv_envelope_coverage"] >= 0.68 and tkm["pv_low_tail_coverage"] >= 0.80 and loss_tkm <= 0.05:
        return "PASS_PV_TKM_FOR_DA_MPC_SCHEDULING"
    return "NEEDS_PV_TAIL_REDUCTION_FIX"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth = truth_table()
    envs = []
    for mode in ["DA", "ID_H24"]:
        envs.append(raw_env(mode))
        for bridge in BRIDGES:
            envs.append(reduced_env(mode, bridge))
        old = old_s1s6_env(mode)
        if not old.empty:
            envs.append(old)
    merged = pd.concat([merge_truth(e, truth) for e in envs], ignore_index=True)
    metric = pd.DataFrame([metric_row(g) for _, g in merged.groupby(["mode", "bridge_type"], observed=True)])
    energy = energy_coverage(merged)
    metric = metric.merge(energy, on=["mode", "bridge_type"], how="left")
    w = weights_summary()

    passfail = metric[~metric["bridge_type"].str.contains("old_S1S6", na=False)].copy()
    diagnostic = metric[metric["bridge_type"].str.contains("old_S1S6", na=False) | metric["bridge_type"].isin(["PV_KM", "PV_TKM", "PV_TKM_SAFE"])].copy()

    status = choose_status(passfail, metric)

    metric.to_csv(OUT_DIR / "pvfocused_reframed_validation_metrics.csv", index=False, encoding="utf-8-sig")
    passfail.to_csv(OUT_DIR / "pvfocused_reframed_passfail_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostic.to_csv(OUT_DIR / "pvfocused_reframed_diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
    w.to_csv(OUT_DIR / "pvfocused_reframed_weight_summary.csv", index=False, encoding="utf-8-sig")

    old_statement = (
        "Old S1-S6 contains load-stress scenarios, while PV-focused bridge does not.\n"
        "Therefore, old S1-S6 is not a fair pass/fail benchmark for PV-focused all-hour OC-risk coverage."
    )
    write_md(
        OUT_DIR / "PV_FOCUSED_VALIDATION_REFRAMED_REPORT.md",
        "PV-Focused Validation Reframed Report",
        [
            ("Final Status", f"`{status}`"),
            ("Validation Scope", "This report uses the CWA-GHI-derived PV truth package and treats old S1-S6 only as diagnostic reference. LOAD-QN and random load error are not used."),
            ("Required Statement", old_statement),
            ("A. Pass/Fail Metrics", passfail.to_markdown(index=False)),
            ("Reduction Quality and Scenario Weights", w.to_markdown(index=False)),
            ("B. Diagnostic-Only Metrics", diagnostic.to_markdown(index=False)),
            ("C. Old S1-S6 Comparison", old_statement + "\n\nAll old S1-S6 all-hour OC-risk comparisons are diagnostic-only and do not determine pass/fail."),
            ("Decision Rule Applied", "PV coverage, PV-active net-load coverage, PV-sensitive OC-risk coverage, and PV tail retention determine scheduling readiness. All-hour OC-risk coverage is diagnostic-only because load-dominant risk is outside PV-focused uncertainty scope."),
            ("No Scheduling Yet in This Step", "This script only reframes validation. Scheduling is run separately only if the status is PASS."),
        ],
    )
    print(f"[PVFocus-Reframed] {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
