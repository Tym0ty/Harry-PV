#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Formal Q19 independent-model ID-H24 GHI probabilistic forecast line v4.

Forecast-only experiment. It trains one independent XGBoost quantile model per
quantile level and lead group/month, builds non-crossing raw Q19 forecasts,
applies the existing endpoint AgACI design, and propagates endpoint calibration
into a full monotone Q19 distribution with median-aware nested anchors.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
LEGACY_SCRIPT = REPO / "new_pipeline/scripts/experiments/exp_id_h24_ghi_probabilistic.py"
H24_DATA = REPO / "new_pipeline/data/output/experiments/ghi_h24_model/rolling_h24_ghi_dataset.parquet"
OLD_Q11 = REPO / "new_pipeline/data/output/experiments/id_h24_ghi_prob/q2_lg_quantile_test.parquet"
OUT = REPO / "new_pipeline/data/output/experiments/id_h24_ghi_prob_q19_independent_agaci_v4"
MODEL_DIR = OUT / "models"

CALIB_START = pd.Timestamp("2024-05-01")
TEST_START = pd.Timestamp("2024-11-01")
SEED = 42
MIN_TRAIN = 100

Q_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
Q_COLS = [f"q{int(q * 100):02d}" for q in Q_LEVELS]
Q_UNC = [f"{c}_uncorrected" for c in Q_COLS]


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_idh24_prob", LEGACY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted isotonic projection for nondecreasing sequence."""
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w, dtype=float)
    values = []
    weights = []
    starts = []
    ends = []
    for i, (val, wt) in enumerate(zip(y, w)):
        values.append(float(val))
        weights.append(float(wt))
        starts.append(i)
        ends.append(i)
        while len(values) >= 2 and values[-2] > values[-1]:
            new_w = weights[-2] + weights[-1]
            new_v = (values[-2] * weights[-2] + values[-1] * weights[-1]) / new_w
            values[-2] = new_v
            weights[-2] = new_w
            ends[-2] = ends[-1]
            values.pop(); weights.pop(); starts.pop(); ends.pop()
    out = np.empty_like(y)
    for v, s, e in zip(values, starts, ends):
        out[s:e + 1] = v
    return out


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["models", "figures", "audit", "snapshots"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, OUT / "run_id_h24_ghi_prob_q19_independent_agaci_v4.py")


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    e = y - q
    return float(np.mean(np.where(e >= 0, alpha * e, (alpha - 1.0) * e)))


def train_predict_independent(legacy) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unc_test_path = OUT / "q19_independent_uncorrected_forecasts.parquet"
    unc_calib_path = OUT / "q19_independent_uncorrected_calibration.parquet"
    registry_path = OUT / "q19_model_registry.csv"
    if unc_test_path.exists() and unc_calib_path.exists() and registry_path.exists():
        test = pd.read_parquet(unc_test_path)
        calib = pd.read_parquet(unc_calib_path)
        reg = pd.read_csv(registry_path)
        for df in [test, calib]:
            for c in ["issue_time", "target_time"]:
                df[c] = pd.to_datetime(df[c])
        return test, calib, reg

    from xgboost import XGBRegressor

    df = pd.read_parquet(H24_DATA).copy()
    for c in ["issue_time", "target_time"]:
        df[c] = pd.to_datetime(df[c])
    feat_cols = legacy.get_feat_cols(df)
    df["_mp"] = df["target_time"].dt.to_period("M")
    parts_test = []
    parts_cal = []
    registry = []

    base_params = dict(legacy.XGB_PARAMS)
    base_params.pop("quantile_alpha", None)
    base_params["random_state"] = SEED

    def fit_one(tr, te, alpha, lg, split_label, model_id):
        if len(tr) < MIN_TRAIN:
            pred_csi = te["csi_lag_tau"].fillna(0).to_numpy(float)
            val_pin = pinball(te["y_csi_true"].to_numpy(float), pred_csi, alpha)
            return pred_csi, "", np.nan, val_pin, len(tr), len(te), "persistence_fallback"
        model = XGBRegressor(quantile_alpha=float(alpha), **base_params)
        model.fit(tr[feat_cols].fillna(0).values, tr["y_csi_true"].values)
        pred_csi = model.predict(te[feat_cols].fillna(0).values)
        pred_csi = np.clip(pred_csi, 0.0, 2.0)
        model_path = MODEL_DIR / f"{model_id}.json"
        model.save_model(model_path)
        val_pin = pinball(te["y_csi_true"].to_numpy(float), pred_csi, alpha)
        best_it = getattr(model, "best_iteration", None)
        if best_it is None:
            best_it = getattr(model, "n_estimators", np.nan)
        return pred_csi, str(model_path.relative_to(REPO)).replace("\\", "/"), best_it, val_pin, len(tr), len(te), "xgboost_quantileerror"

    # Calibration split: train before CALIB_START, predict CALIB_START..TEST_START.
    cal_te = df[(df["target_time"] >= CALIB_START) & (df["target_time"] < TEST_START)]
    for lg, leads in legacy.LEAD_GROUPS.items():
        tr_lg = df[(df["target_time"] < CALIB_START) & (df["horizon_h"].isin(leads))]
        te_lg = cal_te[cal_te["horizon_h"].isin(leads)].copy()
        if len(te_lg) == 0:
            continue
        chunk = te_lg[["issue_time", "target_time", "horizon_h", "y_ghi_true", "y_csi_true", "ghi_clear_target"]].copy()
        chunk["lead_group"] = lg
        chunk["model_id"] = "Q19_IND_CALIB"
        for alpha, qc, quc in zip(Q_LEVELS, Q_COLS, Q_UNC):
            model_id = f"q{int(alpha*100):02d}_calib_{lg.replace('-', '_')}"
            pred_csi, model_file, best_it, val_pin, ntr, nte, status = fit_one(tr_lg, te_lg, alpha, lg, "calibration", model_id)
            chunk[quc] = np.clip(pred_csi * te_lg["ghi_clear_target"].to_numpy(float), 0.0, None)
            registry.append({
                "Quantile": alpha, "Model ID": model_id, "Model file": model_file,
                "Seed": SEED, "Best iteration": best_it, "Validation pinball": val_pin,
                "lead_group": lg, "split": "calibration", "train_rows": ntr,
                "prediction_rows": nte, "status": status,
            })
        parts_cal.append(chunk)
        print(f"[Q19-v2] calibration {lg}")

    # Test split: walk-forward month, train on all data before month start.
    months = sorted(df[df["target_time"] >= TEST_START]["_mp"].unique())
    for mp in months:
        mp_s = mp.to_timestamp()
        mp_e = (mp + 1).to_timestamp()
        tr = df[df["target_time"] < mp_s]
        te = df[(df["target_time"] >= mp_s) & (df["target_time"] < mp_e)]
        for lg, leads in legacy.LEAD_GROUPS.items():
            tr_lg = tr[tr["horizon_h"].isin(leads)]
            te_lg = te[te["horizon_h"].isin(leads)].copy()
            if len(te_lg) == 0:
                continue
            chunk = te_lg[["issue_time", "target_time", "horizon_h", "y_ghi_true", "y_csi_true", "ghi_clear_target"]].copy()
            chunk["lead_group"] = lg
            chunk["model_id"] = "Q19_IND_UNCORRECTED"
            for alpha, qc, quc in zip(Q_LEVELS, Q_COLS, Q_UNC):
                model_id = f"q{int(alpha*100):02d}_{mp}_{lg.replace('-', '_')}"
                pred_csi, model_file, best_it, val_pin, ntr, nte, status = fit_one(tr_lg, te_lg, alpha, lg, str(mp), model_id)
                chunk[quc] = np.clip(pred_csi * te_lg["ghi_clear_target"].to_numpy(float), 0.0, None)
                registry.append({
                    "Quantile": alpha, "Model ID": model_id, "Model file": model_file,
                    "Seed": SEED, "Best iteration": best_it, "Validation pinball": val_pin,
                    "lead_group": lg, "split": str(mp), "train_rows": ntr,
                    "prediction_rows": nte, "status": status,
                })
            parts_test.append(chunk)
        print(f"[Q19-v2] test month {mp}")

    calib = pd.concat(parts_cal, ignore_index=True).sort_values(["issue_time", "target_time", "horizon_h"])
    test = pd.concat(parts_test, ignore_index=True).sort_values(["issue_time", "target_time", "horizon_h"])
    reg = pd.DataFrame(registry)
    calib.to_parquet(unc_calib_path, index=False)
    test.to_parquet(unc_test_path, index=False)
    reg.to_csv(registry_path, index=False, encoding="utf-8-sig")
    return test, calib, reg


def crossing_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    arr = df[cols].to_numpy(float)
    dif = arr[:, 1:] - arr[:, :-1]
    cross = dif < 0
    mags = -dif[cross]
    pair_counts = {}
    for i in range(len(cols) - 1):
        pair_counts[f"{cols[i]}>{cols[i+1]}"] = int(cross[:, i].sum())
    return {
        "total_rows": int(len(df)),
        "rows_with_crossing": int(cross.any(axis=1).sum()),
        "adjacent_crossing_pairs": int(cross.sum()),
        "crossing_rate": float(cross.any(axis=1).mean()),
        "maximum_crossing_magnitude": float(mags.max()) if mags.size else 0.0,
        "mean_crossing_magnitude": float(mags.mean()) if mags.size else 0.0,
        "pair_counts": pair_counts,
    }


def noncrossing_projection(df: pd.DataFrame, input_cols: list[str], output_cols: list[str]) -> pd.DataFrame:
    out = df[["issue_time", "target_time", "horizon_h", "y_ghi_true", "y_csi_true", "ghi_clear_target", "lead_group"]].copy()
    arr = df[input_cols].to_numpy(float)
    proj = np.vstack([np.maximum(pava(row), 0.0) for row in arr])
    for i, c in enumerate(output_cols):
        out[c] = proj[:, i]
    out["model_id"] = "Q19_IND_RAW_NONCROSSING"
    return out


def build_raw_products(test_unc: pd.DataFrame, calib_unc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    before = crossing_stats(test_unc, Q_UNC)
    rows = []
    for pair, count in before["pair_counts"].items():
        rows.append({"stage": "uncorrected", "pair": pair, "crossing_count": count})
    raw_test = noncrossing_projection(test_unc, Q_UNC, Q_COLS)
    raw_cal = noncrossing_projection(calib_unc, Q_UNC, Q_COLS)
    after = crossing_stats(raw_test, Q_COLS)
    for pair, count in after["pair_counts"].items():
        rows.append({"stage": "raw_noncrossing", "pair": pair, "crossing_count": count})
    before_audit = pd.DataFrame(rows)
    before_audit.to_csv(OUT / "raw_q19_crossing_before_correction.csv", index=False, encoding="utf-8-sig")

    audit = pd.DataFrame(rows)
    for k, v in before.items():
        if k != "pair_counts":
            audit.loc[len(audit)] = {"stage": "uncorrected_summary", "pair": k, "crossing_count": v}
    for k, v in after.items():
        if k != "pair_counts":
            audit.loc[len(audit)] = {"stage": "raw_noncrossing_summary", "pair": k, "crossing_count": v}
    y = test_unc["y_ghi_true"].to_numpy(float)
    for alpha, quc, qc in zip(Q_LEVELS, Q_UNC, Q_COLS):
        before_vals = test_unc[quc].to_numpy(float)
        after_vals = raw_test[qc].to_numpy(float)
        adj = after_vals - before_vals
        audit.loc[len(audit)] = {
            "stage": "projection_adjustment_by_quantile",
            "pair": qc,
            "crossing_count": "",
            "rows_adjusted": int((np.abs(adj) > 1e-10).sum()),
            "adjustment_rate": float((np.abs(adj) > 1e-10).mean()),
            "mean_absolute_adjustment": float(np.mean(np.abs(adj))),
            "maximum_absolute_adjustment": float(np.max(np.abs(adj))),
            "pinball_before_projection": pinball(y, before_vals, alpha),
            "pinball_after_projection": pinball(y, after_vals, alpha),
        }
    audit.to_csv(OUT / "raw_q19_projection_audit.csv", index=False, encoding="utf-8-sig")
    raw_test.to_parquet(OUT / "q19_raw_noncrossing_forecasts.parquet", index=False)
    raw_cal.to_parquet(OUT / "q19_raw_noncrossing_calibration.parquet", index=False)
    return raw_test, raw_cal, {"before": before, "after": after}


def ncqr(y, q_lo, q_hi, eps=1e-6):
    return np.maximum(q_lo - y, y - q_hi) / (q_hi - q_lo + eps)


def coverage(rows, lo, hi, qd):
    width = hi - lo
    l = lo - qd * width
    h = hi + qd * width
    y = rows["y_ghi_true"].to_numpy(float)
    return float(((y >= l) & (y <= h)).mean())


def gamma_loss(mean_cov, alpha):
    return max(alpha - mean_cov, 0.0) * 3.0 + max(mean_cov - alpha, 0.0)


def select_gamma(cal, lg, alpha, lo_col, hi_col, gamma_grid):
    sub = cal[cal["lead_group"].eq(lg)].copy()
    if sub.empty:
        return gamma_grid[0], 0.0, np.nan
    y = sub["y_ghi_true"].to_numpy(float)
    lo = sub[lo_col].to_numpy(float)
    hi = sub[hi_col].to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    scores = ncqr(y[valid], lo[valid], hi[valid]) if valid.sum() else np.array([0.0])
    q0 = float(np.quantile(scores, 1.0 - alpha))
    sub["issue_date"] = sub["issue_time"].dt.normalize()
    dates = sorted(sub["issue_date"].unique())
    best_gamma, best_loss = gamma_grid[0], float("inf")
    for g in gamma_grid:
        qd = q0
        covs = []
        for d in dates:
            day = sub[sub["issue_date"].eq(d)]
            cov = coverage(day, day[lo_col].to_numpy(float), day[hi_col].to_numpy(float), qd)
            covs.append(cov)
            qd = qd + g * (alpha - cov)
        loss = gamma_loss(float(np.mean(covs)), alpha) if covs else float("inf")
        if loss < best_loss:
            best_loss, best_gamma = loss, g
    return best_gamma, q0, best_loss


def apply_agaci_endpoints(raw_test: pd.DataFrame, raw_cal: pd.DataFrame, legacy) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gamma_grid = legacy.GAMMA_GRID
    test = raw_test.copy()
    test["issue_date"] = test["issue_time"].dt.normalize()
    parts = []
    param_rows = []
    update_rows = []
    for lg in legacy.LG_KEYS:
        test_lg = test[test["lead_group"].eq(lg)].copy()
        cal_lg = raw_cal[raw_cal["lead_group"].eq(lg)].copy()
        if test_lg.empty:
            continue
        cal_lg["issue_date"] = cal_lg["issue_time"].dt.normalize()
        chunk = test_lg.copy()
        for name, alpha, lo_col, hi_col in [
            ("pi80", 0.80, "q10", "q90"),
            ("pi90", 0.90, "q05", "q95"),
        ]:
            gamma, q0, loss = select_gamma(cal_lg, lg, alpha, lo_col, hi_col, gamma_grid)
            qd = q0
            qd_by_date = {}
            dates = sorted(test_lg["issue_date"].unique())
            for idx, d in enumerate(dates):
                qd_by_date[d] = qd
                day = test_lg[test_lg["issue_date"].eq(d)]
                cov = coverage(day, day[lo_col].to_numpy(float), day[hi_col].to_numpy(float), qd)
                if idx < 100:
                    update_rows.append({"lead_group": lg, "interval": name, "step": idx, "issue_date": d, "Q_d_before": qd, "coverage": cov, "gamma": gamma})
                qd = qd + gamma * (alpha - cov)
            qd_arr = test_lg["issue_date"].map(qd_by_date).to_numpy(float)
            width = test_lg[hi_col].to_numpy(float) - test_lg[lo_col].to_numpy(float)
            lo_adj = np.clip(test_lg[lo_col].to_numpy(float) - qd_arr * width, 0.0, None)
            hi_adj = np.clip(test_lg[hi_col].to_numpy(float) + qd_arr * width, 0.0, None)
            if name == "pi80":
                chunk["q10_agaci80"] = lo_adj
                chunk["q90_agaci80"] = hi_adj
            else:
                chunk["q05_agaci90"] = lo_adj
                chunk["q95_agaci90"] = hi_adj
            param_rows.append({"lead_group": lg, "interval": name, "target_coverage": alpha, "selected_gamma": gamma, "Q0": q0, "selection_loss": loss, "aggregation_weights": "not_used_single_gamma_formal_logic"})
        parts.append(chunk)
    endpoints = pd.concat(parts, ignore_index=True).sort_values(["issue_time", "target_time", "horizon_h"])
    params = pd.DataFrame(param_rows)
    updates = pd.DataFrame(update_rows)
    endpoints.to_parquet(OUT / "q19_agaci_endpoints_before_reconciliation.parquet", index=False)
    params.to_csv(OUT / "agaci_parameter_summary.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "agaci_first100_update_audit.csv", index=False, encoding="utf-8-sig")
    return endpoints, params, updates


def nesting_reconcile(ep: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = ep.copy()
    q05_before = out["q05_agaci90"].to_numpy(float)
    q10_before = out["q10_agaci80"].to_numpy(float)
    q50_raw = out["q50"].to_numpy(float)
    q90_before = out["q90_agaci80"].to_numpy(float)
    q95_before = out["q95_agaci90"].to_numpy(float)

    q05_gt_q10 = q05_before > q10_before
    q10_gt_q50 = q10_before > q50_raw
    q50_gt_q90 = q50_raw > q90_before
    q90_gt_q95 = q90_before > q95_before

    q50_final = q50_raw
    q10_final = np.minimum(q10_before, q50_final)
    q90_final = np.maximum(q90_before, q50_final)
    q05_final = np.maximum(0.0, np.minimum(q05_before, q10_final))
    q95_final = np.maximum(q95_before, q90_final)

    out["q05_final"] = q05_final
    out["q10_final"] = q10_final
    out["q50_final"] = q50_final
    out["q90_final"] = q90_final
    out["q95_final"] = q95_final

    # Back-compatible names used downstream in this script.
    out["q05_nested"] = q05_final
    out["q10_nested"] = q10_final
    out["q90_nested"] = q90_final
    out["q95_nested"] = q95_final
    y = out["y_ghi_true"].to_numpy(float)
    p90_before = float(((y >= out["q05_agaci90"]) & (y <= out["q95_agaci90"])).mean())
    p90_after = float(((y >= out["q05_nested"]) & (y <= out["q95_nested"])).mean())
    mpiw90_before = float((out["q95_agaci90"] - out["q05_agaci90"]).mean())
    mpiw90_after = float((out["q95_nested"] - out["q05_nested"]).mean())
    adj = np.r_[
        np.abs(out["q05_final"] - q05_before),
        np.abs(out["q10_final"] - q10_before),
        np.abs(out["q90_final"] - q90_before),
        np.abs(out["q95_final"] - q95_before),
    ]
    after_fail = (
        (out["q05_final"] > out["q10_final"]) |
        (out["q10_final"] > out["q50_final"]) |
        (out["q50_final"] > out["q90_final"]) |
        (out["q90_final"] > out["q95_final"]) |
        (out["q05_final"] < 0)
    )
    summary = {
        "q05_gt_q10_before": int(q05_gt_q10.sum()),
        "q10_gt_q50_before": int(q10_gt_q50.sum()),
        "q50_gt_q90_before": int(q50_gt_q90.sum()),
        "q90_gt_q95_before": int(q90_gt_q95.sum()),
        "total_affected_rows": int((q05_gt_q10 | q10_gt_q50 | q50_gt_q90 | q90_gt_q95).sum()),
        "mean_adjustment": float(np.mean(adj)),
        "maximum_adjustment": float(np.max(adj)),
        "picp90_before": p90_before,
        "picp90_after": p90_after,
        "mpiw90_before": mpiw90_before,
        "mpiw90_after": mpiw90_after,
        "anchor_feasibility_failures_after": int(after_fail.sum()),
        "nesting_violations_after": int(after_fail.sum()),
    }
    pd.DataFrame([summary]).to_csv(OUT / "median_nested_reconciliation_audit.csv", index=False, encoding="utf-8-sig")
    return out, summary


def project_with_fixed_anchors(row: np.ndarray, anchors: dict[int, float]) -> np.ndarray | None:
    if not all(anchors[i] <= anchors[j] + 1e-9 for i, j in zip(sorted(anchors)[:-1], sorted(anchors)[1:])):
        return None
    out = row.copy()
    for idx, val in anchors.items():
        out[idx] = val
    keys = sorted(anchors)
    for left, right in zip(keys[:-1], keys[1:]):
        lo, hi = anchors[left], anchors[right]
        if left + 1 >= right:
            continue
        internal = np.clip(out[left + 1:right], lo, hi)
        internal = pava(internal)
        internal = np.clip(internal, lo, hi)
        out[left + 1:right] = internal
    return out


def full_calibrated_q19(nested: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame, dict]:
    rows = []
    failures = []
    anchor_probs = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
    probs = np.array(Q_LEVELS)
    raw = nested[Q_COLS].to_numpy(float)
    for i, (_, r) in enumerate(nested.iterrows()):
        shifts = np.array([
            r["q05_nested"] - r["q05"],
            r["q10_nested"] - r["q10"],
            0.0,
            r["q90_nested"] - r["q90"],
            r["q95_nested"] - r["q95"],
        ], dtype=float)
        pre = raw[i, :] + np.interp(probs, anchor_probs, shifts)
        anchors = {0: float(r["q05_final"]), 1: float(r["q10_final"]), 9: float(r["q50_final"]), 17: float(r["q90_final"]), 18: float(r["q95_final"])}
        projected = project_with_fixed_anchors(pre, anchors)
        if projected is None:
            failures.append({
                "issue_time": r["issue_time"], "target_time": r["target_time"], "horizon_h": r["horizon_h"],
                "q05_nested": anchors[0], "q10_nested": anchors[1], "q50_raw": anchors[9],
                "q90_nested": anchors[17], "q95_nested": anchors[18],
            })
            rows.append(pre)
        else:
            rows.append(projected)
    fail_df = pd.DataFrame(failures)
    fail_df.to_csv(OUT / "calibrated_q19_anchor_feasibility_audit.csv", index=False, encoding="utf-8-sig")
    if len(fail_df) > 0:
        return None, fail_df, {"anchor_feasibility_failures": len(fail_df)}
    arr = np.vstack(rows)
    out = nested[["issue_time", "target_time", "horizon_h", "y_ghi_true", "y_csi_true", "ghi_clear_target", "lead_group"]].copy()
    out["model_id"] = "Q19_IND_AGACI_CAL_NONCROSSING"
    for j, qc in enumerate(Q_COLS):
        out[f"{qc}_cal"] = arr[:, j]
    out.to_parquet(OUT / "q19_agaci_calibrated_noncrossing_forecasts.parquet", index=False)
    stats = crossing_stats(out.rename(columns={f"{c}_cal": c for c in Q_COLS}), Q_COLS)
    nesting_viol = int(((out["q05_cal"] > out["q10_cal"]) | (out["q10_cal"] > out["q50_cal"]) | (out["q50_cal"] > out["q90_cal"]) | (out["q90_cal"] > out["q95_cal"])).sum())
    pd.DataFrame([{
        "final_crossing_count": stats["adjacent_crossing_pairs"],
        "rows_with_crossing": stats["rows_with_crossing"],
        "calibrated_interval_nesting_violations": nesting_viol,
        "calibrated_q19_anchor_failures": nesting_viol,
    }]).to_csv(OUT / "calibrated_q19_crossing_audit.csv", index=False, encoding="utf-8-sig")
    return out, fail_df, {"anchor_feasibility_failures": 0, "crossing_stats": stats}


def r2_score(y, pred):
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst if sst else np.nan


def crps_trapezoid(y: np.ndarray, qmat: np.ndarray) -> tuple[float, float, list[dict]]:
    pin = []
    rows = []
    for j, a in enumerate(Q_LEVELS):
        val = pinball(y, qmat[:, j], a)
        pin.append(val)
        rows.append({"quantile": Q_COLS[j], "alpha": a, "pinball_loss": val})
    mean_pb = float(np.mean(pin))
    crps = float(2.0 * np.trapz(np.array(pin), np.array(Q_LEVELS)) / (Q_LEVELS[-1] - Q_LEVELS[0]))
    return mean_pb, crps, rows


def evaluate(raw: pd.DataFrame, cal: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = [("Raw independent XGBQ", raw, Q_COLS)]
    if cal is not None:
        products.append(("AgACI-calibrated independent XGBQ", cal, [f"{c}_cal" for c in Q_COLS]))
    rows = []
    pin_rows = []
    for name, df, cols in products:
        y = df["y_ghi_true"].to_numpy(float)
        qmat = df[cols].to_numpy(float)
        mean_pb, crps, pbr = crps_trapezoid(y, qmat)
        for r in pbr:
            r["Method"] = name
        pin_rows.extend(pbr)
        q05, q10, q50, q90, q95 = [qmat[:, i] for i in [0, 1, 9, 17, 18]]
        p80 = float(((y >= q10) & (y <= q90)).mean())
        p90 = float(((y >= q05) & (y <= q95)).mean())
        neg_before = int((qmat < 0).sum())
        crossing = crossing_stats(pd.DataFrame(qmat, columns=Q_COLS), Q_COLS)
        rows.append({
            "Method": name,
            "verification_rows": len(df),
            "PICP80_pct": p80 * 100,
            "PICP90_pct": p90 * 100,
            "ACE80": p80 - 0.80,
            "ACE90": p90 - 0.90,
            "MPIW80_Wm2": float(np.mean(q90 - q10)),
            "MPIW90_Wm2": float(np.mean(q95 - q05)),
            "Mean_Pinball_Q19": mean_pb,
            "CRPS_Q19": crps,
            "q50_RMSE": float(np.sqrt(np.mean((q50 - y) ** 2))),
            "q50_MAE": float(np.mean(np.abs(q50 - y))),
            "q50_R2": r2_score(y, q50),
            "final_crossing_count": crossing["adjacent_crossing_pairs"],
            "negative_prediction_count_before_clipping": neg_before,
            "negative_prediction_count_after_clipping": int((qmat < 0).sum()),
            "crps_columns": ",".join(cols),
        })
    metrics = pd.DataFrame(rows)
    pins = pd.DataFrame(pin_rows)
    metrics.to_csv(OUT / "q19_probabilistic_metrics.csv", index=False, encoding="utf-8-sig")
    pins.to_csv(OUT / "q19_pinball_by_quantile.csv", index=False, encoding="utf-8-sig")
    table = metrics[["Method", "PICP80_pct", "PICP90_pct", "ACE80", "ACE90", "MPIW80_Wm2", "MPIW90_Wm2", "Mean_Pinball_Q19", "CRPS_Q19"]].copy()
    table.to_csv(OUT / "table_4_2_q19_independent_raw_vs_agaci.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "table_4_2_q19_full_audit.csv", index=False, encoding="utf-8-sig")
    return metrics, pins, table


def downstream_anchor_compare(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    old = pd.read_parquet(OLD_Q11)
    for c in ["issue_time", "target_time"]:
        old[c] = pd.to_datetime(old[c])
    keys = ["issue_time", "target_time", "horizon_h"]
    joined = raw[keys + ["q05", "q10", "q50", "q90", "q95"]].merge(
        old[keys + ["q05", "q10", "q50", "q90", "q95"]],
        on=keys, suffixes=("_new", "_old"), validate="one_to_one"
    )
    rows = []
    for qc in ["q05", "q10", "q50", "q90", "q95"]:
        d = joined[f"{qc}_new"].to_numpy(float) - joined[f"{qc}_old"].to_numpy(float)
        rows.append({
            "anchor": qc,
            "MAE": float(np.mean(np.abs(d))),
            "RMSE": float(np.sqrt(np.mean(d ** 2))),
            "mean_signed_difference": float(np.mean(d)),
            "maximum_absolute_difference": float(np.max(np.abs(d))),
            "correlation": float(np.corrcoef(joined[f"{qc}_new"], joined[f"{qc}_old"])[0, 1]),
            "percent_identical_within_1e-10": float(np.mean(np.abs(d) <= 1e-10) * 100),
            "percent_within_1_Wm2": float(np.mean(np.abs(d) <= 1.0) * 100),
            "percent_within_5_Wm2": float(np.mean(np.abs(d) <= 5.0) * 100),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "downstream_anchor_comparison.csv", index=False, encoding="utf-8-sig")
    if comp["maximum_absolute_difference"].max() <= 1e-10:
        status = "DOWNSTREAM_ANCHORS_EXACTLY_REPRODUCED"
    elif comp["percent_within_5_Wm2"].min() >= 95:
        status = "DOWNSTREAM_ANCHORS_NUMERICALLY_CLOSE"
    else:
        status = "DOWNSTREAM_ANCHORS_CHANGED"
    return comp, status


def make_figure(cal: pd.DataFrame) -> None:
    dates = [("Winter", "2025-01-11"), ("Spring", "2025-05-23"), ("Summer", "2025-07-06"), ("Autumn", "2025-09-14")]
    fig_rows = []
    selected = []
    df = cal.copy()
    for c in ["issue_time", "target_time"]:
        df[c] = pd.to_datetime(df[c])
    for season, date_s in dates:
        sub = df[df["target_time"].dt.date.eq(pd.Timestamp(date_s).date())].copy()
        if sub.empty:
            selected.append((season, date_s, None, sub))
            continue
        issue = sub.groupby("issue_time").size().sort_values(ascending=False).index[0]
        day = sub[sub["issue_time"].eq(issue)].sort_values("target_time").copy()
        day["hour"] = day["target_time"].dt.hour
        rmse = float(np.sqrt(np.mean((day["q50_cal"] - day["y_ghi_true"]) ** 2)))
        mae = float(np.mean(np.abs(day["q50_cal"] - day["y_ghi_true"])))
        for _, r in day.iterrows():
            fig_rows.append({
                "season": season, "date": date_s, "selected_issue_time": r["issue_time"],
                "target_time": r["target_time"], "hour": int(r["hour"]),
                "observed_ghi": float(r["y_ghi_true"]), "q10_cal": float(r["q10_cal"]),
                "q25_cal": float(r["q25_cal"]), "q50_cal": float(r["q50_cal"]),
                "q75_cal": float(r["q75_cal"]), "q90_cal": float(r["q90_cal"]),
                "rmse": rmse, "mae": mae,
            })
        selected.append((season, date_s, issue, day, rmse, mae))
    pd.DataFrame(fig_rows).to_csv(OUT / "figure_4_1_q19_independent_agaci_source.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), sharey=True)
    axes = axes.ravel()
    for ax, item in zip(axes, selected):
        season, date_s, issue, day, rmse, mae = item
        if day.empty:
            ax.set_title(f"{season}: {date_s}\nNo data")
            continue
        x = day["hour"].to_numpy()
        ax.fill_between(x, day["q10_cal"], day["q90_cal"], color="#9ecae1", alpha=0.45, label="P10-P90")
        ax.fill_between(x, day["q25_cal"], day["q75_cal"], color="#3182bd", alpha=0.35, label="P25-P75")
        ax.plot(x, day["q50_cal"], color="#08519c", lw=1.8, label="P50")
        ax.plot(x, day["y_ghi_true"], color="#111111", lw=1.4, marker="o", ms=2.5, label="Observed")
        ax.set_title(f"{season}: {date_s}\nRMSE={rmse:.1f}, MAE={mae:.1f} W/m²")
        ax.set_xlabel("Local hour")
        ax.grid(alpha=0.2, lw=0.6)
    axes[0].set_ylabel("GHI (W/m²)")
    axes[2].set_ylabel("GHI (W/m²)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "figure_4_1_q19_independent_agaci.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "figure_4_1_q19_independent_agaci.pdf", facecolor="white")
    plt.close(fig)


def environment_json(reg: pd.DataFrame, legacy) -> None:
    try:
        import xgboost
        xgb_v = xgboost.__version__
    except Exception:
        xgb_v = "unknown"
    payload = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "xgboost_version": xgb_v,
        "random_seed": SEED,
        "quantile_levels": Q_LEVELS,
        "total_registry_rows": int(len(reg)),
        "train_start": "rolling: all rows before prediction month; calibration models train before 2024-05-01",
        "validation_start": "2024-05-01",
        "validation_end": "2024-10-31",
        "test_start": "2024-11-01",
        "test_end": "2025-10-31",
        "lead_groups": legacy.LEAD_GROUPS,
    }
    (OUT / "experiment_environment.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_report(status: str, metrics: pd.DataFrame | None, raw_stats: dict, nesting: dict | None, anchor_status: str | None, fail_df: pd.DataFrame | None) -> None:
    metric_md = metrics.to_markdown(index=False) if metrics is not None else "Metrics unavailable because calibrated Q19 construction did not complete."
    fail_msg = ""
    if fail_df is not None and len(fail_df) > 0:
        fail_msg = f"\nFixed-anchor feasibility failures: {len(fail_df)} rows. See `calibrated_q19_anchor_feasibility_audit.csv`.\n"
    report = f"""# Final Status

FINAL_STATUS: {status}

# Experiment Scope

Forecast-only Q19 independent XGBoost quantile-regression line. Scenario generation, scenario reduction, MPC, CCOR-MPC, MILP, and realized-operation replay were not rerun.

# Data and Information Boundary

This runner reads `new_pipeline/data/output/experiments/ghi_h24_model/rolling_h24_ghi_dataset.parquet` and follows the existing ID-H24 chronological protocol: calibration starts 2024-05-01 and test starts 2024-11-01.

# Nineteen Independent Quantile Models

One `XGBRegressor(objective='reg:quantileerror', quantile_alpha=tau)` is fit for each tau in:

`{Q_LEVELS}`

Model registry: `q19_model_registry.csv`.

# Raw Quantile-Crossing Audit

Rows with crossing before correction: {raw_stats['before']['rows_with_crossing'] if raw_stats else 'NA'}

Adjacent crossing pairs before correction: {raw_stats['before']['adjacent_crossing_pairs'] if raw_stats else 'NA'}

Raw final crossing count after isotonic projection: {raw_stats['after']['adjacent_crossing_pairs'] if raw_stats else 'NA'}

# Raw Q19 Monotonic Projection

Raw non-crossing product: `q19_raw_noncrossing_forecasts.parquet`.
Projection audit: `raw_q19_projection_audit.csv`.

# AgACI 80% and 90% Calibration

AgACI endpoint output before reconciliation: `q19_agaci_endpoints_before_reconciliation.parquet`.

# Median-Aware and Nested Reconciliation

Nesting audit: `median_nested_reconciliation_audit.csv`.

{pd.DataFrame([nesting]).to_markdown(index=False) if nesting else 'Nesting audit unavailable.'}

# Full Q19 Calibration Mapping

Endpoint shifts are propagated over probability levels with anchors (0.05, 0.10, 0.50, 0.90, 0.95), then projected segment-wise with fixed anchors.

{fail_msg}

# Final Monotonicity Audit

Crossing audit: `calibrated_q19_crossing_audit.csv` if calibrated construction completed.

# Probabilistic Forecast Results

{metric_md}

# Pinball Loss by Quantile

`q19_pinball_by_quantile.csv`

# CRPS Calculation Audit

CRPS is computed by trapezoidal integration of pinball loss over the 19 quantile levels, separately for raw and calibrated columns.

# Figure 4-1 Audit

Figure source uses direct `q25_cal` and `q75_cal`. No interpolation is used.

# Downstream Anchor Comparison

`downstream_anchor_comparison.csv`

Anchor status: `{anchor_status}`

# Reproducibility Information

`experiment_environment.json`

# Thesis-Supported Statements

- One separate XGBoost quantile-regression model was trained for each of nineteen quantile levels.
- AgACI calibrates central 80% and 90% interval endpoints.
- Endpoint corrections are propagated into a full Q19 distribution when fixed-anchor feasibility holds.

# Statements That Must Not Be Made

- Do not state that AgACI independently calibrates all nineteen quantile levels.
- Do not state that the new Q19 forecasts were used by scenario generation or scheduling.

# Output Index

See files in `{OUT.relative_to(REPO).as_posix()}`.
"""
    write_text(OUT / "Q19_INDEPENDENT_AGACI_V4_FINAL_REPORT.md", report)


def main() -> int:
    t0 = time.time()
    setup()
    legacy = load_legacy()
    test_unc, calib_unc, reg = train_predict_independent(legacy)
    environment_json(reg, legacy)
    raw_test, raw_cal, raw_stats = build_raw_products(test_unc, calib_unc)
    endpoints, params, updates = apply_agaci_endpoints(raw_test, raw_cal, legacy)
    nested, nesting = nesting_reconcile(endpoints)
    cal, fail_df, cal_meta = full_calibrated_q19(nested)
    anchor_comp, anchor_status = downstream_anchor_compare(raw_test)

    non_xgb_rows = int((reg["status"] != "xgboost_quantileerror").sum()) if "status" in reg.columns else 0

    if cal is None:
        metrics, pins, table = evaluate(raw_test, None)
        status = "PARTIAL"
        write_report(status, metrics, raw_stats, nesting, anchor_status, fail_df)
    else:
        metrics, pins, table = evaluate(raw_test, cal)
        make_figure(cal)
        crossing_audit = pd.read_csv(OUT / "calibrated_q19_crossing_audit.csv")
        cal_crossings = int(crossing_audit["final_crossing_count"].iloc[0])
        cal_nesting = int(crossing_audit["calibrated_interval_nesting_violations"].iloc[0])
        if non_xgb_rows > 0 or raw_stats["after"]["adjacent_crossing_pairs"] != 0 or nesting["anchor_feasibility_failures_after"] != 0 or cal_crossings != 0 or cal_nesting != 0:
            status = "FAIL"
        elif anchor_status == "DOWNSTREAM_ANCHORS_EXACTLY_REPRODUCED":
            status = "SUCCESS"
        else:
            status = "SUCCESS_WITH_DOWNSTREAM_VERSION_WARNING"
        write_report(status, metrics, raw_stats, nesting, anchor_status, fail_df)

    raw_row = metrics[metrics["Method"].eq("Raw independent XGBQ")].iloc[0]
    cal_rows = metrics[metrics["Method"].eq("AgACI-calibrated independent XGBQ")]
    cal_row = cal_rows.iloc[0] if len(cal_rows) else None
    crossing_final = "NA" if cal is None else int(pd.read_csv(OUT / "calibrated_q19_crossing_audit.csv")["final_crossing_count"].iloc[0])
    nesting_final = "NA" if cal is None else int(pd.read_csv(OUT / "calibrated_q19_crossing_audit.csv")["calibrated_interval_nesting_violations"].iloc[0])
    status_lines = [
        f"FINAL_STATUS: {status}",
        f"OUTPUT_ROOT: {OUT.relative_to(REPO).as_posix()}",
        f"TOTAL_QUANTILE_LEVELS: {len(Q_LEVELS)}",
        f"TOTAL_INDEPENDENT_MODELS: {len(reg)}",
        f"ONE_MODEL_PER_QUANTILE_CONFIRMED: {str(non_xgb_rows == 0).upper()}",
        f"RAW_ROWS_WITH_CROSSING_BEFORE: {raw_stats['before']['rows_with_crossing']}",
        f"RAW_ADJACENT_CROSSINGS_BEFORE: {raw_stats['before']['adjacent_crossing_pairs']}",
        f"RAW_Q19_CROSSING_COUNT_AFTER: {raw_stats['after']['adjacent_crossing_pairs']}",
        f"Q05_GT_Q10_BEFORE: {nesting['q05_gt_q10_before']}",
        f"Q10_GT_Q50_BEFORE: {nesting['q10_gt_q50_before']}",
        f"Q50_GT_Q90_BEFORE: {nesting['q50_gt_q90_before']}",
        f"Q90_GT_Q95_BEFORE: {nesting['q90_gt_q95_before']}",
        f"ANCHOR_FEASIBILITY_FAILURES_AFTER: {nesting['anchor_feasibility_failures_after']}",
        f"CALIBRATED_Q19_CROSSING_COUNT: {crossing_final}",
        f"CALIBRATED_NESTING_VIOLATIONS: {nesting_final}",
        f"RAW_PICP80: {raw_row['PICP80_pct']:.6f}",
        f"CAL_PICP80: {cal_row['PICP80_pct']:.6f}" if cal_row is not None else "CAL_PICP80: NA",
        f"RAW_PICP90: {raw_row['PICP90_pct']:.6f}",
        f"CAL_PICP90: {cal_row['PICP90_pct']:.6f}" if cal_row is not None else "CAL_PICP90: NA",
        f"RAW_ACE80: {raw_row['ACE80']:.6f}",
        f"CAL_ACE80: {cal_row['ACE80']:.6f}" if cal_row is not None else "CAL_ACE80: NA",
        f"RAW_ACE90: {raw_row['ACE90']:.6f}",
        f"CAL_ACE90: {cal_row['ACE90']:.6f}" if cal_row is not None else "CAL_ACE90: NA",
        f"RAW_MPIW80: {raw_row['MPIW80_Wm2']:.6f}",
        f"CAL_MPIW80: {cal_row['MPIW80_Wm2']:.6f}" if cal_row is not None else "CAL_MPIW80: NA",
        f"RAW_MPIW90: {raw_row['MPIW90_Wm2']:.6f}",
        f"CAL_MPIW90: {cal_row['MPIW90_Wm2']:.6f}" if cal_row is not None else "CAL_MPIW90: NA",
        f"RAW_MEAN_PINBALL_Q19: {raw_row['Mean_Pinball_Q19']:.6f}",
        f"CAL_MEAN_PINBALL_Q19: {cal_row['Mean_Pinball_Q19']:.6f}" if cal_row is not None else "CAL_MEAN_PINBALL_Q19: NA",
        f"RAW_CRPS_Q19: {raw_row['CRPS_Q19']:.6f}",
        f"CAL_CRPS_Q19: {cal_row['CRPS_Q19']:.6f}" if cal_row is not None else "CAL_CRPS_Q19: NA",
        f"P25_DIRECT_INDEPENDENT_MODEL: {str(cal is not None).upper()}",
        f"P75_DIRECT_INDEPENDENT_MODEL: {str(cal is not None).upper()}",
        f"DOWNSTREAM_ANCHOR_STATUS: {anchor_status}",
        "SCENARIO_GENERATION_RERUN: FALSE",
        "SCENARIO_REDUCTION_RERUN: FALSE",
        "OPTIMIZATION_RERUN: FALSE",
        "FORMAL_OUTPUTS_OVERWRITTEN: FALSE",
        "SOURCE_EXPERIMENTS_MODIFIED: FALSE",
        f"RUNTIME_SECONDS: {time.time() - t0:.1f}",
    ]
    write_text(OUT / "FINAL_STATUS.txt", "\n".join(status_lines) + "\n")
    print("\n".join(status_lines))
    return 0 if status.startswith("SUCCESS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
