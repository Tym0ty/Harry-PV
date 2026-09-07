#!/usr/bin/env python3
"""Utilities for PV-focused scenario bridge experiments.

The mainline here is PV probabilistic forecast + deterministic given load
profile. Structured load-bias profiles are outer robustness cases, not
probabilistic load scenarios.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge"
SCHED_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
REPLAY_TRUTH_PACKAGE_PATH = REPO / "new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet"
TRUTH = REPLAY_TRUTH_PACKAGE_PATH
PV_TRUTH_SOURCE_LABEL = "CWA_GHI_PV_REBUILT"
DA_PV_Q = REPO / "new_pipeline/data/output/stage1_aci_calibrated_da_v2.parquet"
ID_PV_Q = REPO / "new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_calibrated_quantiles.parquet"
OLD_DA_S1S6 = REPO / "new_pipeline/data/output/experiments/netload/da_netload_prob_scenarios_mainline_qn.parquet"
OLD_ID_S1S6 = REPO / "new_pipeline/data/output/experiments/netload/id_h24_netload_prob_scenarios_mainline_qn.parquet"
PV_CAP_KW = 2687.0
PV_PR = 0.80
CC_KW = 3232.0
TEST_START = pd.Timestamp("2024-11-01 00:00:00")
TEST_END = pd.Timestamp("2025-10-31 23:00:00")
PROFILES = ["base", "structured_low", "structured_high"]
OPTIONAL_PROFILES = ["uniform_low_5", "uniform_high_5"]
BRIDGES = ["PV_KM", "PV_TKM", "PV_TKM_SAFE"]
EPS = 1e-6


def ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def ensure_sched_out() -> Path:
    SCHED_OUT.mkdir(parents=True, exist_ok=True)
    return SCHED_OUT


def write_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines += [f"## {h}", "", body.strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def timestamp_from_day_hour(calendar_day, hour_local: int) -> pd.Timestamp:
    return pd.Timestamp(calendar_day) + pd.Timedelta(hours=int(hour_local) - 1)


def load_truth(path: Path | str | None = None) -> pd.DataFrame:
    truth_path = Path(path) if path is not None else REPLAY_TRUTH_PACKAGE_PATH
    df = pd.read_parquet(truth_path).copy()
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])
    df["timestamp"] = [
        timestamp_from_day_hour(cd, hl)
        for cd, hl in zip(df["calendar_day"], df["hour_local"])
    ]
    return df.sort_values("timestamp").reset_index(drop=True)


def is_summer_tariff(ts: pd.Timestamp) -> bool:
    month = int(ts.month)
    day = int(ts.day)
    if month < 5 or month > 10:
        return False
    if month == 5:
        return day >= 16
    if month == 10:
        return day <= 15
    return True


def time_group(hour: int) -> str:
    if 0 <= hour <= 6:
        return "night"
    if 7 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "daytime"
    if 17 <= hour <= 22:
        return "evening_peak"
    return "late_night"


def load_bias_matrix() -> pd.DataFrame:
    rows = []
    vals = {
        ("summer", "structured_low"): {"night": -0.03, "morning": -0.05, "daytime": -0.05, "evening_peak": -0.07, "late_night": -0.03},
        ("nonsummer", "structured_low"): {"night": -0.02, "morning": -0.04, "daytime": -0.04, "evening_peak": -0.05, "late_night": -0.02},
        ("summer", "structured_high"): {"night": 0.03, "morning": 0.05, "daytime": 0.05, "evening_peak": 0.07, "late_night": 0.03},
        ("nonsummer", "structured_high"): {"night": 0.02, "morning": 0.04, "daytime": 0.04, "evening_peak": 0.05, "late_night": 0.02},
    }
    groups = ["night", "morning", "daytime", "evening_peak", "late_night"]
    for season in ["summer", "nonsummer"]:
        rows.append({"profile": "base", "season": season, **{g: 0.0 for g in groups}})
        rows.append({"profile": "uniform_low_5", "season": season, **{g: -0.05 for g in groups}})
        rows.append({"profile": "uniform_high_5", "season": season, **{g: 0.05 for g in groups}})
        for profile in ["structured_low", "structured_high"]:
            rows.append({"profile": profile, "season": season, **vals[(season, profile)]})
    return pd.DataFrame(rows)


def build_load_profiles() -> pd.DataFrame:
    truth = load_truth()
    mat = load_bias_matrix()
    recs = []
    for r in truth.itertuples(index=False):
        ts = pd.Timestamp(r.timestamp)
        season = "summer" if is_summer_tariff(ts) else "nonsummer"
        tg = time_group(ts.hour)
        base = float(r.load_realized_kw)
        row = {
            "timestamp": ts,
            "calendar_day": pd.Timestamp(r.calendar_day),
            "day_index": int(r.day_index),
            "month_id": int(r.month_id),
            "hour_local": int(r.hour_local),
            "hour_0based": int(ts.hour),
            "season": season,
            "time_group": tg,
            "load_base_kw": base,
        }
        for profile in PROFILES + OPTIONAL_PROFILES:
            b = float(mat[(mat["profile"].eq(profile)) & (mat["season"].eq(season))][tg].iloc[0])
            row[f"load_{profile}_kw"] = max(base * (1.0 + b), 0.0)
        recs.append(row)
    return pd.DataFrame(recs)


def ghi_to_pv_kw(x) -> np.ndarray:
    return np.clip(PV_PR * np.asarray(x, dtype=float) / 1000.0 * PV_CAP_KW, 0.0, PV_CAP_KW)


def full_da_pv_quantiles() -> pd.DataFrame:
    truth = load_truth()[["timestamp"]].copy()
    truth["target_time"] = truth["timestamp"]
    q = pd.read_parquet(DA_PV_Q).copy()
    q["issue_time"] = pd.to_datetime(q["issue_day"])
    q["target_time"] = pd.to_datetime(q["target_day"]) + pd.to_timedelta(q["hour_local"].astype(int) - 1, unit="h")
    out = truth.merge(q, on="target_time", how="left")
    # DA package convention: one H24 package per target day. Use the same
    # D-1 20:00 issue gate as the RAW DA load baseline/LitErr bridge.
    out["issue_time"] = out["target_time"].dt.normalize() - pd.Timedelta(hours=4)
    out["lead_hour"] = ((out["target_time"] - out["issue_time"]) / pd.Timedelta(hours=1)).round().astype(int)
    for dst, src in [("pv_q05", "q05_aci"), ("pv_q10", "q10_aci"), ("pv_q50", "q50"), ("pv_q90", "q90_aci"), ("pv_q95", "q95_aci")]:
        out[dst] = ghi_to_pv_kw(out[src].fillna(0.0))
    return out[["issue_time", "target_time", "lead_hour", "pv_q05", "pv_q10", "pv_q50", "pv_q90", "pv_q95"]]


def full_id_pv_quantiles() -> pd.DataFrame:
    df = pd.read_parquet(ID_PV_Q).copy()
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    out = df[["issue_time", "target_time", "horizon_h", "pv_q05", "pv_q10", "pv_q50", "pv_q90", "pv_q95"]].rename(columns={"horizon_h": "lead_hour"})
    out = out[(out["target_time"] >= TEST_START) & (out["target_time"] <= TEST_END)].copy()
    # Add missing issue/target pairs with zero PV at night or unavailable edge cases.
    truth_ts = pd.date_range(TEST_START, TEST_END, freq="h")
    issues = pd.date_range(TEST_START, TEST_END - pd.Timedelta(hours=1), freq="h")
    base_rows = []
    existing = set(zip(out["issue_time"], out["target_time"]))
    for issue in issues:
        for lead in range(1, 25):
            target = issue + pd.Timedelta(hours=lead)
            if target > TEST_END:
                continue
            if (issue, target) not in existing:
                base_rows.append({"issue_time": issue, "target_time": target, "lead_hour": lead, "pv_q05": 0.0, "pv_q10": 0.0, "pv_q50": 0.0, "pv_q90": 0.0, "pv_q95": 0.0})
    if base_rows:
        out = pd.concat([out, pd.DataFrame(base_rows)], ignore_index=True)
    for c in ["pv_q05", "pv_q10", "pv_q50", "pv_q90", "pv_q95"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(0, PV_CAP_KW)
    return out.sort_values(["issue_time", "lead_hour", "target_time"]).reset_index(drop=True)


def inv_from_quantiles(u: np.ndarray, q05, q10, q50, q90, q95) -> np.ndarray:
    probs = np.array([0.05, 0.10, 0.50, 0.90, 0.95], dtype=float)
    vals = np.vstack([q05, q10, q50, q90, q95]).T
    out = np.empty(len(vals), dtype=float)
    for i, row in enumerate(vals):
        row = np.maximum.accumulate(np.asarray(row, dtype=float))
        if np.nanmax(row) - np.nanmin(row) <= 1e-9:
            out[i] = row[0]
        else:
            out[i] = np.interp(float(np.clip(u[i], EPS, 1 - EPS)), probs, row)
    return np.clip(out, 0.0, PV_CAP_KW)


def seed_for(mode: str, issue_time: pd.Timestamp, m: int = 500) -> int:
    key = f"pvfocus|{mode}|{pd.Timestamp(issue_time).isoformat()}|{m}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def simulate_pv_for_issue(g: pd.DataFrame, m: int, mode: str) -> pd.DataFrame:
    g = g.sort_values(["lead_hour", "target_time"]).reset_index(drop=True)
    h = len(g)
    rng = np.random.default_rng(seed_for(mode, pd.Timestamp(g["issue_time"].iloc[0]), m))
    u = rng.uniform(EPS, 1 - EPS, size=(m, h))
    q = [g[c].to_numpy(float) for c in ["pv_q05", "pv_q10", "pv_q50", "pv_q90", "pv_q95"]]
    pv = np.empty((m, h), dtype=np.float32)
    for i in range(m):
        pv[i, :] = inv_from_quantiles(u[i, :], *q).astype(np.float32)
    return pd.DataFrame(
        {
            "issue_time": np.repeat(g["issue_time"].iloc[0], m * h),
            "target_time": np.tile(g["target_time"].to_numpy(), m),
            "lead_hour": np.tile(g["lead_hour"].to_numpy(), m).astype(np.int16),
            "raw_scenario_id": np.repeat(np.arange(m, dtype=np.int32), h),
            "pv_s_kw": pv.reshape(-1),
            "source_quantile_package": "DA_ACI" if mode == "DA" else "ID_H24_calibrated",
            "scenario_generation_method": "quantile_inverse_uniform_sampling",
        }
    )


def raw_to_matrix(raw: pd.DataFrame, value_col: str = "netload_s_kw") -> tuple[np.ndarray, np.ndarray]:
    piv = raw.pivot_table(index="raw_scenario_id", columns="lead_hour", values=value_col, aggfunc="first").sort_index()
    return piv.to_numpy(float), piv.index.to_numpy()


def choose_medoids(raw: pd.DataFrame, k: int = 5, bridge_type: str = "PV_KM") -> pd.DataFrame:
    mat, ids = raw_to_matrix(raw, "netload_s_kw")
    peaks = np.nanmax(mat, axis=1)
    ramps = np.nanmax(np.abs(np.diff(mat, axis=1)), axis=1) if mat.shape[1] > 1 else np.zeros(len(ids))
    exceed_area = np.maximum(mat - 0.95 * CC_KW, 0.0).sum(axis=1)
    chosen: list[int] = []
    if len(ids) <= k:
        chosen = list(range(len(ids)))
    else:
        if bridge_type in {"PV_TKM", "PV_TKM_SAFE"}:
            if bridge_type == "PV_TKM_SAFE":
                # Tail-band medoid: avoid one-off absolute max dominance.
                lo, hi = np.nanquantile(peaks, 0.90), np.nanquantile(peaks, 0.975)
                cand = np.where((peaks >= lo) & (peaks <= hi))[0]
                if len(cand) == 0:
                    cand = np.argsort(peaks)[-max(10, min(50, len(peaks))):]
                idx = int(cand[np.nanargmin(np.abs(peaks[cand] - np.nanmedian(peaks[cand])))])
            else:
                idx = int(np.nanargmax(peaks))
            chosen.append(idx)
            risk_score = peaks + 0.10 * ramps + 0.05 * exceed_area
            if bridge_type == "PV_TKM_SAFE":
                lo, hi = np.nanquantile(risk_score, 0.95), np.nanquantile(risk_score, 0.99)
                cand = np.where((risk_score >= lo) & (risk_score <= hi))[0]
                if len(cand) == 0:
                    cand = np.argsort(risk_score)[-max(10, min(50, len(risk_score))):]
                idx = int(cand[np.nanargmin(np.abs(risk_score[cand] - np.nanmedian(risk_score[cand])))])
            else:
                idx = int(np.nanargmax(risk_score))
            if idx not in chosen:
                chosen.append(idx)
        for q in np.linspace(0.1, 0.9, k):
            if len(chosen) >= k:
                break
            idx = int(np.nanargmin(np.abs(peaks - np.nanquantile(peaks, q))))
            if idx not in chosen:
                chosen.append(idx)
        while len(chosen) < k:
            centers = mat[chosen]
            dist = ((mat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            mind = dist.min(axis=1)
            for c in chosen:
                mind[c] = -1
            chosen.append(int(np.nanargmax(mind)))
    centers = mat[chosen]
    dist = ((mat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1)
    weights = np.bincount(assign, minlength=len(chosen)) / len(assign)
    rows = []
    for sc_id, idx in enumerate(chosen):
        sid = ids[idx]
        sub = raw[raw["raw_scenario_id"].eq(sid)].sort_values(["lead_hour", "target_time"])
        peak = float(sub["netload_s_kw"].max())
        if bridge_type in {"PV_TKM", "PV_TKM_SAFE"} and sc_id == 0:
            medoid_type = "high_netload_tail"
        elif bridge_type in {"PV_TKM", "PV_TKM_SAFE"} and (peak >= 0.95 * CC_KW or sc_id == 1):
            medoid_type = "dct_risk_tail"
        else:
            medoid_type = "typical"
        for r in sub.itertuples(index=False):
            rows.append(
                {
                    "issue_time": r.issue_time,
                    "target_time": r.target_time,
                    "lead_hour": int(r.lead_hour),
                    "scenario_id": sc_id,
                    "scenario_weight": float(weights[sc_id]),
                    "load_profile_name": r.load_profile_name,
                    "bridge_type": bridge_type,
                    "medoid_type": medoid_type,
                    "load_kw": float(r.load_kw),
                    "pv_s_kw": float(r.pv_s_kw),
                    "netload_s_kw": float(r.netload_s_kw),
                    "raw_scenario_id": int(sid),
                }
            )
    return pd.DataFrame(rows)


def parquet_writer(path: Path, df: pd.DataFrame | None = None) -> pq.ParquetWriter:
    if df is None:
        raise ValueError("Need initial dataframe for schema.")
    return pq.ParquetWriter(path, pa.Table.from_pandas(df, preserve_index=False).schema, compression="zstd")


def write_table(writer: pq.ParquetWriter | None, path: Path, df: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def load_profile_lookup(profile: str) -> dict[pd.Timestamp, float]:
    df = pd.read_parquet(OUT_DIR / "load_bias_profiles.parquet")
    col = f"load_{profile}_kw"
    if col not in df.columns:
        raise KeyError(col)
    return {pd.Timestamp(r.timestamp): float(getattr(r, col)) for r in df.itertuples(index=False)}


def summarize_weights(df: pd.DataFrame) -> dict:
    w = df.drop_duplicates(["issue_time", "scenario_id"])["scenario_weight"].astype(float)
    return {
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
        "effective_scenarios_mean": float(df.drop_duplicates(["issue_time", "scenario_id"]).groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum()).mean()),
    }
