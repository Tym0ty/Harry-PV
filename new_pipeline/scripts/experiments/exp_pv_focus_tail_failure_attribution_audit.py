#!/usr/bin/env python3
"""Tail-failure attribution audit for the PV-focused ID H24 bridge.

This script does not run scheduling and does not modify existing bridge outputs.
It determines whether ID H24 tail under-coverage is caused by raw PV scenario
under-dispersion, reduction loss, or load-dominant high-risk hours.
"""

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
    OLD_ID_S1S6,
    OUT_DIR as PVFOCUS_OUT,
    PV_CAP_KW,
    choose_medoids,
    full_id_pv_quantiles,
    is_summer_tariff,
    load_truth,
    time_group,
    write_md,
)

OUT = REPO / "new_pipeline/data/output/experiments/pv_focused_tail_failure_attribution_audit"
RAW = PVFOCUS_OUT / "id_h24_pvfocus_base_raw_netload_M500.parquet"
RED = {
    "PV_KM": PVFOCUS_OUT / "id_h24_pvfocus_base_km_K5_scenarios.parquet",
    "PV_TKM": PVFOCUS_OUT / "id_h24_pvfocus_base_tkm_K5_scenarios.parquet",
    "PV_TKM_SAFE": PVFOCUS_OUT / "id_h24_pvfocus_base_tkm_safe_K5_scenarios.parquet",
}


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def build_risk_table() -> pd.DataFrame:
    truth = load_truth().copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    truth["realized_netload_kw"] = truth["load_realized_kw"] - truth["pv_realized_kw"]
    q95 = truth["realized_netload_kw"].quantile(0.95)
    q99 = truth["realized_netload_kw"].quantile(0.99)
    truth["hour_0based"] = truth["timestamp"].dt.hour
    truth["season"] = ["summer" if is_summer_tariff(pd.Timestamp(ts)) else "nonsummer" for ts in truth["timestamp"]]
    truth["pv_active"] = truth["pv_realized_kw"] > 0.05 * PV_CAP_KW
    truth["near_oc_flag"] = truth["realized_netload_kw"] >= 0.95 * CC_KW
    truth["oc_risk_flag"] = truth["realized_netload_kw"] >= CC_KW
    truth["top5_netload_flag"] = truth["realized_netload_kw"] >= q95
    truth["top1_netload_flag"] = truth["realized_netload_kw"] >= q99
    truth["time_group"] = truth["hour_0based"].map(time_group)

    def cls(r) -> str:
        if not (r.near_oc_flag or r.top5_netload_flag or r.oc_risk_flag):
            return "non_high_risk"
        if bool(r.pv_active) and r.time_group in {"daytime", "morning"}:
            return "daylight_PV_shortfall_risk"
        if bool(r.pv_active):
            return "PV_active_risk"
        if r.time_group == "evening_peak":
            return "evening_transition_risk"
        return "PV_inactive_load_dominant_risk"

    truth["risk_hour_class"] = [cls(r) for r in truth.itertuples(index=False)]
    cols = [
        "timestamp",
        "load_realized_kw",
        "pv_realized_kw",
        "realized_netload_kw",
        "hour_0based",
        "season",
        "pv_active",
        "near_oc_flag",
        "oc_risk_flag",
        "top5_netload_flag",
        "top1_netload_flag",
        "risk_hour_class",
    ]
    out = truth[cols].copy()
    out.to_csv(OUT / "risk_hour_classification.csv", index=False, encoding="utf-8-sig")
    return out


def pv_contribution(risk: pd.DataFrame) -> pd.DataFrame:
    q = full_id_pv_quantiles()
    q = q[q["lead_hour"].eq(1)].copy()
    q = q.rename(columns={"target_time": "timestamp"})
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    # Use the latest H1 row per target timestamp if duplicates exist.
    q = q.sort_values(["timestamp", "issue_time"]).drop_duplicates("timestamp", keep="last")
    df = risk.merge(q[["timestamp", "pv_q05", "pv_q10", "pv_q50", "pv_q90", "pv_q95"]], on="timestamp", how="left")
    # Reduced scenario first-step min PV by target, using all issue-target records that target that timestamp.
    red = pd.read_parquet(RED["PV_TKM_SAFE"], columns=["target_time", "pv_s_kw", "netload_s_kw"])
    red["timestamp"] = pd.to_datetime(red["target_time"])
    rgrp = red.groupby("timestamp").agg(reduced_min_pv=("pv_s_kw", "min"), reduced_max_netload=("netload_s_kw", "max")).reset_index()
    df = df.merge(rgrp, on="timestamp", how="left")
    df["pv_q50_minus_q05_kw"] = df["pv_q50"] - df["pv_q05"]
    df["pv_q50_minus_q10_kw"] = df["pv_q50"] - df["pv_q10"]
    df["pv_reduced_range_kw"] = df["pv_q50"] - df["reduced_min_pv"]
    high = df["near_oc_flag"] | df["oc_risk_flag"] | df["top5_netload_flag"]
    df["pv_range_lt_100_kw"] = df["pv_q50_minus_q05_kw"] < 100
    df["pv_range_lt_200_kw"] = df["pv_q50_minus_q05_kw"] < 200
    df["pv_range_lt_300_kw"] = df["pv_q50_minus_q05_kw"] < 300
    df["pv_near_zero"] = df["pv_realized_kw"] <= 0.05 * PV_CAP_KW
    # A simple explainability proxy: q50->q05 PV shortfall over net-load excess above 0.95CC.
    excess = np.maximum(df["realized_netload_kw"] - 0.95 * CC_KW, 1.0)
    df["pv_uncertainty_explainability_ratio"] = np.clip(df["pv_q50_minus_q05_kw"] / excess, 0, 10)
    df.to_csv(OUT / "pv_contribution_to_netload_tail.csv", index=False, encoding="utf-8-sig")
    summary = []
    for label, mask in {
        "all_hours": pd.Series(True, index=df.index),
        "high_risk_hours": high,
        "oc_risk_hours": df["oc_risk_flag"],
        "top5_hours": df["top5_netload_flag"],
    }.items():
        sub = df[mask]
        summary.append(
            {
                "subset": label,
                "rows": len(sub),
                "share_pv_range_lt_100_kw": float(sub["pv_range_lt_100_kw"].mean()) if len(sub) else np.nan,
                "share_pv_range_lt_200_kw": float(sub["pv_range_lt_200_kw"].mean()) if len(sub) else np.nan,
                "share_pv_range_lt_300_kw": float(sub["pv_range_lt_300_kw"].mean()) if len(sub) else np.nan,
                "share_pv_near_zero": float(sub["pv_near_zero"].mean()) if len(sub) else np.nan,
                "median_pv_explainability_ratio": float(sub["pv_uncertainty_explainability_ratio"].median()) if len(sub) else np.nan,
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "pv_contribution_summary.csv", index=False, encoding="utf-8-sig")
    return df


def reduced_env(path: Path, bridge: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["issue_time", "target_time", "netload_s_kw", "pv_s_kw"])
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    return (
        df.groupby(["issue_time", "target_time"], observed=True)
        .agg(scen_max=("netload_s_kw", "max"), pv_min=("pv_s_kw", "min"))
        .reset_index()
        .assign(bridge_type=bridge)
    )


def raw_envelope_and_candidates(risk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk_lookup = risk.set_index("timestamp")
    pf = pq.ParquetFile(RAW)
    env_rows = []
    cand_rows = []
    k_rows = []
    for rg in range(pf.num_row_groups):
        raw = pf.read_row_group(rg).to_pandas()
        raw["issue_time"] = pd.to_datetime(raw["issue_time"])
        raw["target_time"] = pd.to_datetime(raw["target_time"])
        issue = raw["issue_time"].iloc[0]
        # raw envelope
        env = raw.groupby(["issue_time", "target_time"], observed=True).agg(raw_scen_max=("netload_s_kw", "max"), raw_pv_min=("pv_s_kw", "min")).reset_index()
        env_rows.append(env)

        # candidate medoids using risk-aware target subsets within this horizon.
        mat = raw.pivot_table(index="raw_scenario_id", columns="lead_hour", values="netload_s_kw", aggfunc="first").sort_index()
        pvmat = raw.pivot_table(index="raw_scenario_id", columns="lead_hour", values="pv_s_kw", aggfunc="first").sort_index()
        targets = raw.drop_duplicates("lead_hour").sort_values("lead_hour")["target_time"].to_list()
        rsub = risk_lookup.reindex(targets)
        high = (rsub["near_oc_flag"].fillna(False) | rsub["top5_netload_flag"].fillna(False) | rsub["oc_risk_flag"].fillna(False)).to_numpy(bool)
        pv_active = rsub["pv_active"].fillna(False).to_numpy(bool)
        daylight = rsub["risk_hour_class"].fillna("").astype(str).str.contains("daylight|PV_active").to_numpy(bool)
        near = rsub["near_oc_flag"].fillna(False).to_numpy(bool)
        # Candidate 1: low PV during PV-active high-risk hours.
        mask1 = high & pv_active
        if not mask1.any():
            mask1 = pv_active
        score1 = -pvmat.loc[:, np.asarray(pvmat.columns)[mask1]].sum(axis=1) if mask1.any() else -pvmat.sum(axis=1)
        sid1 = int(score1.idxmax())
        # Candidate 2: max netload during near-OC/high-risk hours.
        mask2 = near | high
        score2 = mat.loc[:, np.asarray(mat.columns)[mask2]].max(axis=1) if mask2.any() else mat.max(axis=1)
        sid2 = int(score2.idxmax())
        # Candidate 3: afternoon/evening PV shortfall ramp.
        pv_arr = pvmat.to_numpy(float)
        if pv_arr.shape[1] >= 2:
            drop = np.maximum(pv_arr[:, :-1] - pv_arr[:, 1:], 0).max(axis=1)
            sid3 = int(pvmat.index[int(np.argmax(drop))])
        else:
            sid3 = int(pvmat.index[0])
        candidates = {
            "daylight_low_pv_medoid": sid1,
            "dct_risk_low_pv_medoid": sid2,
            "pv_shortfall_ramp_medoid": sid3,
        }
        for cname, sid in candidates.items():
            sub = raw[raw["raw_scenario_id"].eq(sid)].sort_values("lead_hour")
            for r in sub.itertuples(index=False):
                cand_rows.append(
                    {
                        "issue_time": issue,
                        "target_time": r.target_time,
                        "candidate_type": cname,
                        "raw_scenario_id": sid,
                        "candidate_netload": float(r.netload_s_kw),
                        "candidate_pv": float(r.pv_s_kw),
                    }
                )
        # K10 estimate via same medoid function.
        try:
            k10 = choose_medoids(raw, k=10, bridge_type="PV_TKM_SAFE")
            for r in k10.groupby(["issue_time", "target_time"], observed=True).agg(k10_scen_max=("netload_s_kw", "max")).reset_index().itertuples(index=False):
                k_rows.append({"issue_time": r.issue_time, "target_time": r.target_time, "k10_scen_max": float(r.k10_scen_max)})
        except Exception:
            pass
        if (rg + 1) % 1000 == 0:
            print(f"processed raw row group {rg+1}/{pf.num_row_groups}")
    return pd.concat(env_rows, ignore_index=True), pd.DataFrame(cand_rows), pd.DataFrame(k_rows)


def raw_vs_reduced_coverage(risk: pd.DataFrame) -> pd.DataFrame:
    target = risk[["timestamp", "realized_netload_kw", "top5_netload_flag", "top1_netload_flag", "near_oc_flag", "oc_risk_flag"]].rename(columns={"timestamp": "target_time"})
    raw_env, candidates, k10 = raw_envelope_and_candidates(risk)
    raw_env.to_parquet(OUT / "id_h24_raw_envelope_cache.parquet", index=False)
    candidates.to_parquet(OUT / "id_h24_lowpv_candidate_cache.parquet", index=False)
    if not k10.empty:
        k10.to_parquet(OUT / "id_h24_k10_envelope_cache.parquet", index=False)
    envs = [raw_env.rename(columns={"raw_scen_max": "scen_max"}).assign(bridge_type="RAW_M500")]
    for b, p in RED.items():
        envs.append(reduced_env(p, b))
    all_env = pd.concat(envs, ignore_index=True)
    all_env = all_env.merge(target, on="target_time", how="left")
    rows = []
    for bridge, g in all_env.groupby("bridge_type", observed=True):
        rows.append(coverage_row(bridge, g, "scen_max"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "raw_vs_reduced_tail_coverage.csv", index=False, encoding="utf-8-sig")
    return out


def coverage_row(label: str, df: pd.DataFrame, col: str) -> dict:
    masks = {
        "all": pd.Series(True, index=df.index),
        "top5": df["top5_netload_flag"].fillna(False),
        "top1": df["top1_netload_flag"].fillna(False),
        "near_oc": df["near_oc_flag"].fillna(False),
        "oc_risk": df["oc_risk_flag"].fillna(False),
    }
    row = {"bridge_type": label, "rows": int(len(df)), "scenario_max_netload_kw": float(df[col].max())}
    for name, mask in masks.items():
        sub = df[mask]
        row[f"{name}_upper_coverage"] = float((sub["realized_netload_kw"] <= sub[col]).mean()) if len(sub) else np.nan
    return row


def old_s1s6_decomposition(risk: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_parquet(OLD_ID_S1S6)
    old["target_time"] = pd.to_datetime(old["target_timestamp"])
    old["lead_hour"] = old["lead_time_hours"].astype(int)
    old["issue_time"] = old["target_time"] - pd.to_timedelta(old["lead_hour"], unit="h")
    scen_cols = [c for c in ["nl_median", "nl_low", "nl_high", "nl_high_load_stress", "nl_low_pv_stress", "nl_combined_stress"] if c in old.columns]
    old["upper_source"] = old[scen_cols].idxmax(axis=1)
    merged = old.merge(risk[["timestamp", "top5_netload_flag", "top1_netload_flag", "near_oc_flag", "oc_risk_flag", "risk_hour_class"]].rename(columns={"timestamp": "target_time"}), on="target_time", how="left")
    rows = []
    for subset, mask in {
        "all": pd.Series(True, index=merged.index),
        "top5": merged["top5_netload_flag"].fillna(False),
        "top1": merged["top1_netload_flag"].fillna(False),
        "near_oc": merged["near_oc_flag"].fillna(False),
        "oc_risk": merged["oc_risk_flag"].fillna(False),
    }.items():
        sub = merged[mask]
        vc = sub["upper_source"].value_counts(normalize=True)
        for source, share in vc.items():
            rows.append({"subset": subset, "upper_source": source, "share": float(share), "count": int((sub["upper_source"] == source).sum())})
    out = pd.DataFrame(rows)
    out["source_type"] = out["upper_source"].map(
        {
            "nl_high": "balanced_or_rank_high",
            "nl_high_load_stress": "load_stress",
            "nl_low_pv_stress": "pv_stress",
            "nl_combined_stress": "combined_load_pv_stress",
            "nl_median": "median",
            "nl_low": "low_netload",
        }
    )
    out.to_csv(OUT / "old_s1_s6_tail_source_decomposition.csv", index=False, encoding="utf-8-sig")
    return out


def low_pv_and_k_sensitivity(risk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = risk[["timestamp", "realized_netload_kw", "top5_netload_flag", "top1_netload_flag", "near_oc_flag", "oc_risk_flag"]].rename(columns={"timestamp": "target_time"})
    cur = reduced_env(RED["PV_TKM_SAFE"], "PV_TKM_SAFE").merge(target, on="target_time", how="left")
    cand = pd.read_parquet(OUT / "id_h24_lowpv_candidate_cache.parquet")
    cand["target_time"] = pd.to_datetime(cand["target_time"])
    rows = []
    for cname, g in cand.groupby("candidate_type", observed=True):
        env = g.groupby(["issue_time", "target_time"], observed=True).agg(candidate_max=("candidate_netload", "max")).reset_index()
        merged = cur.merge(env, on=["issue_time", "target_time"], how="left")
        merged["augmented_max"] = np.maximum(merged["scen_max"], merged["candidate_max"].fillna(-1e9))
        base = coverage_row("PV_TKM_SAFE", merged, "scen_max")
        aug = coverage_row(cname, merged, "augmented_max")
        rows.append(
            {
                "candidate_type": cname,
                "top5_coverage_gain": aug["top5_upper_coverage"] - base["top5_upper_coverage"],
                "top1_coverage_gain": aug["top1_upper_coverage"] - base["top1_upper_coverage"],
                "oc_risk_coverage_gain": aug["oc_risk_upper_coverage"] - base["oc_risk_upper_coverage"],
                "near_oc_coverage_gain": aug["near_oc_upper_coverage"] - base["near_oc_upper_coverage"],
                "duplicates_existing_tkm_safe_rate": float((merged["candidate_max"] <= merged["scen_max"] + 1e-6).mean()),
                "k5_enough_if_gain_small": bool((aug["oc_risk_upper_coverage"] - base["oc_risk_upper_coverage"]) < 0.05),
            }
        )
    lowpv = pd.DataFrame(rows)
    lowpv.to_csv(OUT / "low_pv_medoid_feasibility.csv", index=False, encoding="utf-8-sig")
    krows = [coverage_row("PV_TKM_SAFE_K5", cur, "scen_max")]
    k10_path = OUT / "id_h24_k10_envelope_cache.parquet"
    if k10_path.exists():
        k10 = pd.read_parquet(k10_path)
        k10["target_time"] = pd.to_datetime(k10["target_time"])
        k10m = k10.merge(target, on="target_time", how="left")
        krows.append(coverage_row("PV_TKM_SAFE_K10_estimate", k10m, "k10_scen_max"))
    ksens = pd.DataFrame(krows)
    ksens.to_csv(OUT / "k_sensitivity_tail_estimate.csv", index=False, encoding="utf-8-sig")
    return lowpv, ksens


def validation_rule(risk: pd.DataFrame, rawred: pd.DataFrame, olddec: pd.DataFrame, lowpv: pd.DataFrame, ksens: pd.DataFrame) -> str:
    high = risk[risk["near_oc_flag"] | risk["oc_risk_flag"] | risk["top5_netload_flag"]]
    pv_inactive_share = float((high["risk_hour_class"].eq("PV_inactive_load_dominant_risk") | high["risk_hour_class"].eq("evening_transition_risk")).mean())
    raw = rawred[rawred["bridge_type"].eq("RAW_M500")].iloc[0]
    red = rawred[rawred["bridge_type"].eq("PV_TKM_SAFE")].iloc[0]
    old_top = olddec[olddec["subset"].eq("oc_risk")].sort_values("share", ascending=False).head(3)
    status = "LOAD_DOMINANT_RISK_REFRAME_VALIDATION"
    if float(raw["oc_risk_upper_coverage"]) < 0.65:
        status = "FIX_GENERATION_PV_UNDERDISPERSION"
    elif float(red["oc_risk_upper_coverage"]) + 0.10 < float(raw["oc_risk_upper_coverage"]):
        status = "FIX_REDUCTION_ADD_LOW_PV_MEDOID"
    if not ksens.empty and len(ksens) > 1:
        k5 = float(ksens[ksens["bridge_type"].eq("PV_TKM_SAFE_K5")]["oc_risk_upper_coverage"].iloc[0])
        k10 = float(ksens[ksens["bridge_type"].eq("PV_TKM_SAFE_K10_estimate")]["oc_risk_upper_coverage"].iloc[0])
        if k10 - k5 > 0.08:
            status = "USE_ID_K10"
    md = f"""# Recommended Validation Rule Update

Final status: `{status}`

## Fairness Of Comparing Against Old S1-S6

Direct all-hour OC-risk comparison against old S1-S6 is only partially fair. Old S1-S6 includes load-stress and combined load/PV stress scenarios, while the PV-focused bridge intentionally uses deterministic given load and PV uncertainty only.

High-risk hour load-dominant/evening-transition share: `{pv_inactive_share:.3f}`.

## Recommended Rule

Use a combination:

1. all-hour OC-risk coverage as a safety screen;
2. PV-active-hour OC-risk coverage as the PV-focused method metric;
3. PV-sensitive top-tail coverage for daylight / PV-active periods;
4. a separate disclosure for load-dominant evening or PV-inactive risk hours.

## Old S1-S6 Tail Source

Top OC-risk upper-envelope sources:

{old_top.to_markdown(index=False)}

## K Sensitivity

{ksens.to_markdown(index=False)}
"""
    (OUT / "recommended_validation_rule_update.md").write_text(md, encoding="utf-8")
    return status


def write_report(status: str, risk: pd.DataFrame, pvcontrib: pd.DataFrame, rawred: pd.DataFrame, olddec: pd.DataFrame, lowpv: pd.DataFrame, ksens: pd.DataFrame) -> None:
    high = risk[risk["near_oc_flag"] | risk["oc_risk_flag"] | risk["top5_netload_flag"]]
    class_summary = high["risk_hour_class"].value_counts(normalize=True).rename_axis("risk_class").reset_index(name="share")
    pv_summary = pd.read_csv(OUT / "pv_contribution_summary.csv")
    report = f"""# PV-Focused Tail-Failure Attribution Report

Final status: `{status}`

## Executive Summary

The audit separates three mechanisms:

1. raw PV scenario under-dispersion;
2. k-medoids reduction loss;
3. load-dominant risk that PV-only uncertainty cannot explain.

The old S1-S6 bridge includes load stress and combined load/PV stress, so it can cover high net-load hours that are not explainable by PV uncertainty alone. The PV-focused bridge should not be judged only by old S1-S6 all-hour OC-risk coverage without disclosing this difference.

## Risk-Hour Classification

{class_summary.to_markdown(index=False)}

## PV Contribution Summary

{pv_summary.to_markdown(index=False)}

## Raw Vs Reduced Tail Coverage

{rawred.to_markdown(index=False)}

## Old S1-S6 Tail Source Decomposition

{olddec.head(30).to_markdown(index=False)}

## Low-PV Medoid Feasibility

{lowpv.to_markdown(index=False)}

## K Sensitivity

{ksens.to_markdown(index=False)}

## Interpretation

- If raw M500 coverage is already weak, the issue is PV generation/quantile under-dispersion or load-dominant risk.
- If raw M500 coverage is strong but PV-TKM-safe is weak, the issue is reduction losing PV-shortfall tails.
- If old S1-S6 upper envelope is mostly `nl_high_load_stress` or `nl_combined_stress`, the old bridge is covering risk through load stress that the PV-focused bridge intentionally excludes.

No scheduling was run.
"""
    (OUT / "PV_FOCUSED_TAIL_FAILURE_ATTRIBUTION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    ensure_out()
    risk = build_risk_table()
    pvcontrib = pv_contribution(risk)
    rawred = raw_vs_reduced_coverage(risk)
    olddec = old_s1s6_decomposition(risk)
    lowpv, ksens = low_pv_and_k_sensitivity(risk)
    status = validation_rule(risk, rawred, olddec, lowpv, ksens)
    write_report(status, risk, pvcontrib, rawred, olddec, lowpv, ksens)
    print(f"[PVFocus-tail-audit] {status} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
