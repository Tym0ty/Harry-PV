#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_focus_bridge_utils import (
    CC_KW,
    OLD_DA_S1S6,
    OLD_ID_S1S6,
    OUT_DIR,
    PROFILES,
    ensure_out,
    load_truth,
    write_md,
)


def old_env(mode: str) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "mode": mode,
            "load_profile_name": "base",
            "bridge_type": "old_S1S6_LOAD_QN_REFERENCE",
            "issue_time": df["issue_time"],
            "target_time": df["target_time"],
            "actual": df["netload_true_kw"].astype(float),
            "scen_min": df[scen_cols].min(axis=1).astype(float),
            "scen_max": df[scen_cols].max(axis=1).astype(float),
            "scen_mean": df[scen_cols].mean(axis=1).astype(float),
            "pv_min": np.nan,
            "pv_max": np.nan,
        }
    )


def bridge_env(mode: str, profile: str, bridge: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = "da" if mode == "DA" else "id_h24"
    suffix = bridge.lower().replace("pv_", "")
    path = OUT_DIR / f"{prefix}_pvfocus_{profile}_{suffix}_K5_scenarios.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_parquet(path)
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    truth = load_truth()[["timestamp", "load_realized_kw", "pv_realized_kw"]].rename(columns={"timestamp": "target_time"})
    truth["target_time"] = pd.to_datetime(truth["target_time"])
    grp = (
        df.groupby(["issue_time", "target_time"], observed=True)
        .agg(
            scen_min=("netload_s_kw", "min"),
            scen_max=("netload_s_kw", "max"),
            scen_mean=("netload_s_kw", "mean"),
            pv_min=("pv_s_kw", "min"),
            pv_max=("pv_s_kw", "max"),
        )
        .reset_index()
    )
    env = grp.merge(truth, on="target_time", how="left")
    env["actual"] = env["load_realized_kw"] - env["pv_realized_kw"]
    env["mode"] = mode
    env["load_profile_name"] = profile
    env["bridge_type"] = bridge
    w = (
        df.drop_duplicates(["issue_time", "scenario_id"])
        .groupby("issue_time")["scenario_weight"]
        .agg(["min", "max", "sum", "count"])
        .reset_index()
    )
    return env, pd.DataFrame(
        {
            "mode": [mode],
            "load_profile_name": [profile],
            "bridge_type": [bridge],
            "min_weight": [float(w["min"].min())],
            "max_weight": [float(w["max"].max())],
            "weight_sum_min": [float(w["sum"].min())],
            "weight_sum_max": [float(w["sum"].max())],
            "effective_scenarios_mean": [float(df.drop_duplicates(["issue_time", "scenario_id"]).groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum()).mean())],
        }
    )


def metrics(env: pd.DataFrame) -> dict:
    env = env.dropna(subset=["actual"]).copy()
    cov = ((env["actual"] >= env["scen_min"]) & (env["actual"] <= env["scen_max"])).mean()
    upper = (env["actual"] <= env["scen_max"]).mean()
    high5 = env["actual"] >= env["actual"].quantile(0.95)
    high1 = env["actual"] >= env["actual"].quantile(0.99)
    risk = env["actual"] >= CC_KW
    near = env["actual"] >= 0.95 * CC_KW
    daily = env.groupby("issue_time").agg(actual_max=("actual", "max"), scen_max=("scen_max", "max"), scen_min=("scen_min", "min")).reset_index()
    d5 = daily["actual_max"] >= daily["actual_max"].quantile(0.95)
    d1 = daily["actual_max"] >= daily["actual_max"].quantile(0.99)
    return {
        "row_count": int(len(env)),
        "realized_netload_envelope_coverage": float(cov),
        "realized_netload_upper_coverage": float(upper),
        "daily_max_netload_envelope_coverage": float(((daily["actual_max"] >= daily["scen_min"]) & (daily["actual_max"] <= daily["scen_max"])).mean()),
        "top5_netload_upper_coverage": float((env.loc[high5, "actual"] <= env.loc[high5, "scen_max"]).mean()) if high5.any() else np.nan,
        "top1_netload_upper_coverage": float((env.loc[high1, "actual"] <= env.loc[high1, "scen_max"]).mean()) if high1.any() else np.nan,
        "top5_daily_max_upper_coverage": float((daily.loc[d5, "actual_max"] <= daily.loc[d5, "scen_max"]).mean()) if d5.any() else np.nan,
        "top1_daily_max_upper_coverage": float((daily.loc[d1, "actual_max"] <= daily.loc[d1, "scen_max"]).mean()) if d1.any() else np.nan,
        "oc_risk_hour_upper_coverage": float((env.loc[risk, "actual"] <= env.loc[risk, "scen_max"]).mean()) if risk.any() else np.nan,
        "near_oc_hour_upper_coverage": float((env.loc[near, "actual"] <= env.loc[near, "scen_max"]).mean()) if near.any() else np.nan,
        "monthly_peak_preservation_ratio": float(env["scen_max"].max() / env["actual"].max()) if env["actual"].max() else np.nan,
        "actual_max_netload_kw": float(env["actual"].max()),
        "scenario_max_netload_kw": float(env["scen_max"].max()),
        "mean_abs_error_scen_mean_kw": float(np.mean(np.abs(env["scen_mean"] - env["actual"]))),
        "mean_bias_scen_mean_kw": float(np.mean(env["scen_mean"] - env["actual"])),
        "pv_envelope_coverage": float(((env["pv_realized_kw"] >= env["pv_min"]) & (env["pv_realized_kw"] <= env["pv_max"])).mean()) if "pv_realized_kw" in env and env["pv_min"].notna().any() else np.nan,
    }


def main() -> int:
    ensure_out()
    envs, weights = [], []
    for mode in ["DA", "ID_H24"]:
        oe = old_env(mode)
        if not oe.empty:
            envs.append(oe)
    for profile in PROFILES:
        for mode in ["DA", "ID_H24"]:
            for bridge in ["PV_KM", "PV_TKM", "PV_TKM_SAFE"]:
                e, w = bridge_env(mode, profile, bridge)
                if not e.empty:
                    envs.append(e)
                if not w.empty:
                    weights.append(w)
    all_env = pd.concat(envs, ignore_index=True)
    all_w = pd.concat(weights, ignore_index=True) if weights else pd.DataFrame()
    rows = []
    for keys, sub in all_env.groupby(["mode", "load_profile_name", "bridge_type"], observed=True):
        m = metrics(sub)
        m.update({"mode": keys[0], "load_profile_name": keys[1], "bridge_type": keys[2]})
        rows.append(m)
    val = pd.DataFrame(rows)
    val.to_csv(OUT_DIR / "pvfocus_scenario_validation_summary.csv", index=False, encoding="utf-8-sig")
    val[["mode", "load_profile_name", "bridge_type", "top5_netload_upper_coverage", "top1_netload_upper_coverage", "top5_daily_max_upper_coverage", "top1_daily_max_upper_coverage"]].to_csv(OUT_DIR / "pvfocus_tail_coverage_comparison.csv", index=False, encoding="utf-8-sig")
    val[["mode", "load_profile_name", "bridge_type", "oc_risk_hour_upper_coverage", "near_oc_hour_upper_coverage", "monthly_peak_preservation_ratio", "actual_max_netload_kw", "scenario_max_netload_kw"]].to_csv(OUT_DIR / "pvfocus_oc_risk_coverage_comparison.csv", index=False, encoding="utf-8-sig")
    val[["mode", "load_profile_name", "bridge_type", "daily_max_netload_envelope_coverage", "top5_daily_max_upper_coverage", "top1_daily_max_upper_coverage"]].to_csv(OUT_DIR / "pvfocus_daily_max_netload_coverage.csv", index=False, encoding="utf-8-sig")
    all_w.to_csv(OUT_DIR / "pvfocus_scenario_weight_distribution.csv", index=False, encoding="utf-8-sig")
    lb = pd.read_csv(OUT_DIR / "load_bias_profile_summary.csv")
    lb.to_csv(OUT_DIR / "pvfocus_load_bias_sanity_summary.csv", index=False, encoding="utf-8-sig")
    base_safe = val[(val["load_profile_name"].eq("base")) & (val["bridge_type"].eq("PV_TKM_SAFE"))]
    pass_safe = bool((base_safe["oc_risk_hour_upper_coverage"].fillna(0) >= 0.65).all() and (base_safe["scenario_max_netload_kw"].max() < 8000))
    rec = "PASS_PV_TKM_SAFE_FOR_SCHEDULING" if pass_safe else "NEEDS_FIX_BEFORE_SCHEDULING"
    write_md(
        OUT_DIR / "PV_FOCUSED_SCENARIO_VALIDATION_REPORT.md",
        "PV-Focused Scenario Validation Report",
        [
            ("Status", f"Recommendation: `{rec}`"),
            ("Validation Summary", val.to_markdown(index=False)),
            ("Weight Summary", all_w.to_markdown(index=False) if not all_w.empty else "No weight summary."),
            ("Load-Bias Sanity", lb.to_markdown(index=False)),
            ("Safety", "Realized PV/load are used only for validation metrics. Scenarios were generated from PV quantiles and deterministic load profiles."),
        ],
    )
    print(f"[PVFocus-5] validation {rec}")
    return 0 if pass_safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
