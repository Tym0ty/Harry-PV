#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "new_pipeline/scripts/experiments"))

from pv_focus_bridge_utils import (  # noqa: E402
    CC_KW,
    OUT_DIR,
    PV_CAP_KW,
    ensure_out,
    raw_to_matrix,
    write_md,
    write_table,
)

BRIDGE = "PV_TKM_LOWPV_SAFE"


def _scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - np.nanmean(x)) / s


def scenario_features(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mat, ids = raw_to_matrix(raw, "netload_s_kw")
    pv_mat, _ = raw_to_matrix(raw, "pv_s_kw")
    peaks = np.nanmax(mat, axis=1)
    ramps = np.nanmax(np.abs(np.diff(mat, axis=1)), axis=1) if mat.shape[1] > 1 else np.zeros(len(ids))
    exceed_area = np.maximum(mat - CC_KW, 0.0).sum(axis=1)
    near_exceed_area = np.maximum(mat - 0.95 * CC_KW, 0.0).sum(axis=1)

    lead_hours = sorted(raw["lead_hour"].dropna().astype(int).unique())
    target_hours = (
        raw.drop_duplicates("lead_hour")
        .sort_values("lead_hour")["target_time"]
        .pipe(pd.to_datetime)
        .dt.hour
        .to_numpy()
    )
    pv_median = np.nanmedian(pv_mat, axis=0)
    daylight = np.array([(6 <= h <= 18) for h in target_hours], dtype=bool)
    pv_active = pv_median > 0.05 * PV_CAP_KW
    highrisk = np.nanmedian(mat, axis=0) >= 0.90 * CC_KW
    mask = daylight | pv_active | highrisk
    if not mask.any():
        mask = daylight if daylight.any() else np.ones(mat.shape[1], dtype=bool)
    lowpv_shortfall = np.maximum(pv_median[None, :] - pv_mat, 0.0)[:, mask].sum(axis=1)
    dct_score = 0.70 * _scale(peaks) + 0.20 * _scale(near_exceed_area) + 0.10 * _scale(exceed_area)
    high_score = peaks
    lowpv_score = lowpv_shortfall

    feat = np.concatenate(
        [
            _scale(mat.reshape(-1)).reshape(mat.shape),
            0.50 * _scale(peaks)[:, None],
            0.25 * _scale(ramps)[:, None],
            0.60 * _scale(near_exceed_area)[:, None],
            0.45 * _scale(lowpv_shortfall)[:, None],
        ],
        axis=1,
    )
    return feat, ids, {
        "mat": mat,
        "pv_mat": pv_mat,
        "peaks": peaks,
        "ramps": ramps,
        "exceed_area": exceed_area,
        "near_exceed_area": near_exceed_area,
        "lowpv_shortfall": lowpv_shortfall,
        "dct_score": dct_score,
        "high_score": high_score,
        "lowpv_score": lowpv_score,
        "mask_count": np.repeat(mask.sum(), len(ids)),
    }


def pick_unique(candidates: list[int], chosen: list[int]) -> int | None:
    for idx in candidates:
        if int(idx) not in chosen:
            return int(idx)
    return None


def choose_lowpv_medoids(raw: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict]:
    feat, ids, m = scenario_features(raw)
    n = len(ids)
    chosen: list[int] = []
    roles: list[str] = []

    def add_by_score(score: np.ndarray, role: str, descending: bool = True) -> None:
        order = np.argsort(score)
        if descending:
            order = order[::-1]
        idx = pick_unique([int(x) for x in order], chosen)
        if idx is not None:
            chosen.append(idx)
            roles.append(role)

    if n <= k:
        chosen = list(range(n))
        roles = ["typical"] * n
    else:
        add_by_score(m["high_score"], "high_netload_tail", True)
        add_by_score(m["lowpv_score"], "low_pv_shortfall_tail", True)
        add_by_score(m["dct_score"], "dct_risk_tail", True)

        # Add two typical medoids for K=5, more for K=10, from non-tail-ish quantiles.
        typical_needed = max(k - len(chosen), 0)
        peak_qs = np.linspace(0.15, 0.85, typical_needed) if typical_needed else []
        for q in peak_qs:
            target = np.nanquantile(m["peaks"], q)
            order = np.argsort(np.abs(m["peaks"] - target))
            idx = pick_unique([int(x) for x in order], chosen)
            if idx is not None:
                chosen.append(idx)
                roles.append("typical")

        while len(chosen) < min(k, n):
            centers = feat[chosen]
            dist = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            mind = dist.min(axis=1)
            for c in chosen:
                mind[c] = -1
            idx = int(np.nanargmax(mind))
            chosen.append(idx)
            roles.append("typical")

    centers = feat[chosen]
    dist = ((feat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1)
    weights = np.bincount(assign, minlength=len(chosen)) / len(assign)

    rows = []
    medoid_audit = []
    for sc_id, idx in enumerate(chosen):
        sid = int(ids[idx])
        sub = raw[raw["raw_scenario_id"].eq(sid)].sort_values(["lead_hour", "target_time"])
        medoid_audit.append(
            {
                "scenario_id": sc_id,
                "raw_scenario_id": sid,
                "medoid_type": roles[sc_id],
                "scenario_weight": float(weights[sc_id]),
                "daily_max_netload_kw": float(m["peaks"][idx]),
                "lowpv_shortfall_score_kwh": float(m["lowpv_shortfall"][idx]),
                "near_exceed_area_kwh": float(m["near_exceed_area"][idx]),
                "exceed_area_kwh": float(m["exceed_area"][idx]),
            }
        )
        for r in sub.itertuples(index=False):
            rows.append(
                {
                    "issue_time": r.issue_time,
                    "target_time": r.target_time,
                    "lead_hour": int(r.lead_hour),
                    "scenario_id": sc_id,
                    "scenario_weight": float(weights[sc_id]),
                    "load_profile_name": r.load_profile_name,
                    "bridge_type": BRIDGE if k == 5 else f"{BRIDGE}_K{k}",
                    "medoid_type": roles[sc_id],
                    "load_kw": float(r.load_kw),
                    "pv_s_kw": float(r.pv_s_kw),
                    "netload_s_kw": float(r.netload_s_kw),
                    "raw_scenario_id": sid,
                }
            )
    red = pd.DataFrame(rows)
    info = {
        "issue_time": raw["issue_time"].iloc[0],
        "k": k,
        "raw_scenario_count": n,
        "max_raw_netload_kw": float(raw["netload_s_kw"].max()),
        "max_reduced_netload_kw": float(red["netload_s_kw"].max()),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "effective_scenarios": float(1.0 / np.square(weights).sum()),
        "lowpv_medoid_count": int(sum("low_pv" in x for x in roles)),
        "dct_medoid_count": int(sum("dct" in x for x in roles)),
        "high_netload_medoid_count": int(sum("high_netload" in x for x in roles)),
        "medoids": medoid_audit,
    }
    return red, info


def reduce_one(mode: str, profile: str, k: int, m_raw: int) -> tuple[dict, list[dict], list[dict]]:
    prefix = "da" if mode == "DA" else "id_h24"
    src = OUT_DIR / f"{prefix}_pvfocus_{profile}_raw_netload_M{m_raw}.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = OUT_DIR / f"{prefix}_pvfocus_{profile}_tkm_lowpv_safe_K{k}_scenarios.parquet"
    if dst.exists():
        dst.unlink()
    pf = pq.ParquetFile(src)
    writer = None
    weights, audits = [], []
    t0 = time.time()
    for rg in range(pf.num_row_groups):
        raw = pf.read_row_group(rg).to_pandas()
        red, info = choose_lowpv_medoids(raw, k=k)
        writer = write_table(writer, dst, red)
        first = red.drop_duplicates(["scenario_id"])
        for r in first.itertuples(index=False):
            weights.append(
                {
                    "mode": mode,
                    "load_profile_name": profile,
                    "bridge_type": BRIDGE if k == 5 else f"{BRIDGE}_K{k}",
                    "k": k,
                    "issue_time": r.issue_time,
                    "scenario_id": int(r.scenario_id),
                    "scenario_weight": float(r.scenario_weight),
                    "medoid_type": r.medoid_type,
                    "raw_scenario_id": int(r.raw_scenario_id),
                }
            )
        for mi in info["medoids"]:
            audits.append(
                {
                    "mode": mode,
                    "load_profile_name": profile,
                    "bridge_type": BRIDGE if k == 5 else f"{BRIDGE}_K{k}",
                    "k": k,
                    "issue_time": info["issue_time"],
                    **mi,
                    "max_raw_netload_kw": info["max_raw_netload_kw"],
                    "max_reduced_netload_kw": info["max_reduced_netload_kw"],
                    "effective_scenarios": info["effective_scenarios"],
                }
            )
        if (rg + 1) % 1000 == 0:
            print(f"{mode}-{profile}-K{k}: reduced {rg+1}/{pf.num_row_groups} issues in {time.time()-t0:.0f}s")
    if writer is not None:
        writer.close()
    return (
        {"mode": mode, "profile": profile, "bridge_type": BRIDGE if k == 5 else f"{BRIDGE}_K{k}", "k": k, "issues": pf.num_row_groups, "path": str(dst), "elapsed_sec": time.time() - t0},
        weights,
        audits,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--profile", default="base")
    parser.add_argument("--run-da", action="store_true")
    parser.add_argument("--run-id-k10", action="store_true")
    args = parser.parse_args()
    ensure_out()
    summaries, weights, audits = [], [], []

    jobs = [("ID_H24", args.profile, 5)]
    if args.run_da:
        jobs.append(("DA", args.profile, 5))
    if args.run_id_k10:
        jobs.append(("ID_H24", args.profile, 10))

    for mode, profile, k in jobs:
        s, w, a = reduce_one(mode, profile, k, args.M)
        summaries.append(s)
        weights.extend(w)
        audits.extend(a)

    summary = pd.DataFrame(summaries)
    weight_df = pd.DataFrame(weights)
    audit_df = pd.DataFrame(audits)
    weight_df.to_csv(OUT_DIR / "pvfocus_lowpv_scenario_weight_summary.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(OUT_DIR / "pvfocus_lowpv_medoid_selection_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "pvfocus_lowpv_reduction_summary.csv", index=False, encoding="utf-8-sig")

    weight_issue = (
        weight_df.groupby(["mode", "bridge_type", "k", "issue_time"], observed=True)["scenario_weight"]
        .sum()
        .reset_index()
    )
    weight_sanity = (
        weight_issue.groupby(["mode", "bridge_type", "k"], observed=True)["scenario_weight"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )
    write_md(
        OUT_DIR / "PV_FOCUSED_LOW_PV_REDUCTION_FIX_REPORT.md",
        "PV-Focused Low-PV Reduction Fix Report",
        [
            ("Status", "Generated `PV_TKM_LOWPV_SAFE` reduced scenario packages without overwriting existing PV-KM/PV-TKM/PV-TKM-safe outputs."),
            ("Medoid Roles", "K=5 uses explicit roles: high-net-load tail, low-PV/PV-shortfall tail, DCT-risk tail, and two typical medoids. K=10 keeps the same tail roles and adds typical/diversity medoids."),
            ("Weights", "Scenario weights are cluster probability masses from nearest-medoid assignment in a peak- and low-PV-aware feature space."),
            ("Generated Packages", summary.to_markdown(index=False)),
            ("Weight Sanity", weight_sanity.to_markdown(index=False)),
            ("No Scheduling", "This script only creates reduced scenario packages for validation. It does not run DA/MPC/CCOR scheduling."),
        ],
    )
    print(f"[PVFocus-LowPV] wrote reduction fix outputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
