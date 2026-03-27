#!/usr/bin/env python3
"""
Phase A — Scenario Reduction Ablation Study

Per Harry's plan §3.1: Move only one layer at a time, do attribution first.
Tests:
  A1. Expand K: 5, 10, 20 scenarios (Euclidean distance on 24-dim PV)
  A2. Decision-aware distance: weight billing hours 3× in k-medoids
  A3. No-reduction upper bound: all 500 raw scenarios (if tractable)

For each variant, regenerates the bridge ingest for C1 (prob PV + det load),
runs MILP solve + replay, and compares:
  - Total replay cost
  - Over-contract cost
  - Worst-month bill
  - Solve time
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))

from milp_common import get_config, load_data, load_truth, format_results, CASE_TABLE
from milp_solver import build_and_solve, replay


def _kmedoids_pam(X, K, seed=42, max_iter=300):
    """Pure-Python PAM k-medoids. Returns (medoid_indices, labels)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = cdist(X, X, metric="euclidean")

    # Initialize: pick K random medoids
    medoids = rng.choice(n, size=K, replace=False)

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        dists = D[:, medoids]  # (n, K)
        labels = dists.argmin(axis=1)

        # Update medoids: within each cluster, pick point with min total distance
        new_medoids = np.empty(K, dtype=int)
        for ci in range(K):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                new_medoids[ci] = medoids[ci]
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            new_medoids[ci] = members[costs.argmin()]

        if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
            break
        medoids = new_medoids

    # Final assignment
    dists = D[:, medoids]
    labels = dists.argmin(axis=1)
    return medoids, labels


# ──────────────────────────────────────────────────────────────
#  K-Medoids with configurable distance weighting
# ──────────────────────────────────────────────────────────────

def reduce_scenarios_ablation(raw_500_path, K, distance_mode="euclidean",
                              billing_weight=3.0, seed=42):
    """Re-run k-medoids reduction on the 500 raw scenarios.

    Args:
        raw_500_path: Path to scenarios_joint_pv_load_raw_500.parquet
        K: Number of medoids
        distance_mode: "euclidean" or "decision_aware"
        billing_weight: Weight multiplier for billing hours (9-22 summer weekday)
        seed: Random seed

    Returns:
        DataFrame with reduced scenarios (same schema as reduced_5)
    """
    print(f"  Reducing 500→{K} scenarios ({distance_mode})...")
    raw = pd.read_parquet(raw_500_path)
    raw["target_day_local"] = pd.to_datetime(raw["target_day_local"])
    raw["target_time_local"] = pd.to_datetime(raw["target_time_local"])

    days = sorted(raw["target_day_local"].unique())
    all_rows = []

    for day in days:
        day_data = raw[raw["target_day_local"] == day].copy()
        scenario_ids = sorted(day_data["scenario_id"].unique())

        if len(scenario_ids) <= K:
            # Fewer raw scenarios than K — use all
            day_data["probability_pi"] = 1.0 / len(scenario_ids)
            day_data["medoid_raw_id"] = day_data["scenario_id"]
            all_rows.append(day_data)
            continue

        # Pivot: each scenario → 24-dim PV vector
        pivoted = day_data.pivot_table(
            index="scenario_id",
            columns=day_data["target_time_local"].dt.hour,
            values="pv_available_kw",
            aggfunc="first"
        )
        pivoted.columns = [f"pv_h{h}" for h in pivoted.columns]

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(pivoted.values)  # (500, 24)

        if distance_mode == "decision_aware":
            # Weight billing hours more heavily
            # Billing hours (summer weekday): 9-22 (TOU peak/half-peak)
            # This makes k-medoids more sensitive to PV variation during
            # hours that actually affect over-contract and energy costs
            day_ts = pd.Timestamp(day)
            dow = day_ts.weekday()

            weights = np.ones(24)
            for h in range(24):
                # Summer weekday peak hours get highest weight
                if 9 <= h < 22:
                    weights[h] = billing_weight
                # Early morning / late night get base weight
            # Normalize so mean weight = 1
            weights = weights / weights.mean()
            X = X * weights[np.newaxis, :]

        # K-medoids (PAM-style alternating)
        actual_K = min(K, len(scenario_ids))
        medoid_indices, labels = _kmedoids_pam(X, actual_K, seed=seed)

        medoid_sids = pivoted.index[medoid_indices].tolist()
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        probs = (cluster_sizes / cluster_sizes.sum()).values

        # Build output rows for this day
        for ci, (sid, prob) in enumerate(zip(medoid_sids, probs)):
            scenario_data = day_data[day_data["scenario_id"] == sid].copy()
            scenario_data["scenario_id"] = ci
            scenario_data["probability_pi"] = prob
            scenario_data["medoid_raw_id"] = sid
            all_rows.append(scenario_data)

    result = pd.concat(all_rows, ignore_index=True)
    print(f"    → {len(result)} rows, K={K}")
    return result


def build_ablation_ingest(reduced_df, bridge_dir, cal_df, load_truth_dict, pv_scale):
    """Build a C1-format MILP ingest package from reduced scenarios.

    MILP convention: hour_local 1..24 where hour_local h means the h-th hour.
    Raw scenario convention: target_time_local hour 0..23 (h0=0 is midnight).
    Mapping: h0=1..23 → hour_local=1..23 on same day.
             h0=0 → hour_local=24 of the PREVIOUS day.
    For the last day of the case year, h0=0 of the NEXT day is not available,
    but we handle this by filling hour_local=24 from the next day's h0=0.
    """
    reduced_df = reduced_df.copy()
    reduced_df["target_day_local"] = pd.to_datetime(reduced_df["target_day_local"])
    reduced_df["target_time_local"] = pd.to_datetime(reduced_df["target_time_local"])

    day_to_idx = {pd.Timestamp(r["calendar_day"]): r["day_index"]
                  for _, r in cal_df.iterrows()}

    # Build lookup: (target_day_local, h0, scenario_id) → (pv_kw, prob)
    pv_lookup = {}
    for _, row in reduced_df.iterrows():
        d = pd.Timestamp(row["target_day_local"])
        h0 = pd.Timestamp(row["target_time_local"]).hour
        sid = int(row["scenario_id"])
        pv_lookup[(d, h0, sid)] = (float(row["pv_available_kw"]), float(row["probability_pi"]))

    scenario_ids = sorted(reduced_df["scenario_id"].unique())
    case_days = sorted(day_to_idx.keys())

    rows = []
    for d in case_days:
        di = day_to_idx[d]
        cal_row = cal_df[cal_df["day_index"] == di].iloc[0]

        for sid in scenario_ids:
            sid_int = int(sid)
            for h_local in range(1, 25):
                # Map hour_local to raw scenario's (target_day, h0)
                if h_local < 24:
                    src_day = d
                    src_h0 = h_local
                else:  # hour_local=24 → h0=0 of the NEXT day
                    src_day = d + pd.Timedelta(days=1)
                    src_h0 = 0

                pv_50, prob = pv_lookup.get((src_day, src_h0, sid_int), (0.0, 1.0 / len(scenario_ids)))
                pv_kw = pv_50 * pv_scale
                load_kw = load_truth_dict.get((d, h_local), 0.0)

                rows.append({
                    "day_index": di,
                    "calendar_day": cal_row["calendar_day"],
                    "hour_local": h_local,
                    "scenario_id": f"w{sid_int}",
                    "probability_pi": prob,
                    "pv_mode": "pv_prob",
                    "load_mode": "load_det",
                    "pv_available_kw": pv_kw,
                    "load_kw": load_kw,
                    "month_id": int(cal_row["month_id"]),
                    "day_type": cal_row["day_type"],
                    "season_tag": cal_row["season_tag"],
                    "is_holiday": bool(cal_row["is_holiday"]),
                })

    df = pd.DataFrame(rows)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])
    return df


# ──────────────────────────────────────────────────────────────
#  Main ablation
# ──────────────────────────────────────────────────────────────

def run_ablation():
    os.chdir(ROOT / "notebooks_milp")
    CFG = get_config()

    RAW_500_PATH = ROOT / "pipeline_outputs" / "scenarios_joint_pv_load_raw_500.parquet"
    if not RAW_500_PATH.exists():
        print(f"ERROR: Raw 500-scenario file not found: {RAW_500_PATH}")
        return

    # Load calendar and truth for building ingest
    cal_df = pd.read_parquet(ROOT / "bridge_outputs_fullyear" / "caseyear_calendar_manifest.parquet")
    cal_df["calendar_day"] = pd.to_datetime(cal_df["calendar_day"])

    # Load NTUST truth for deterministic load
    ntust = pd.read_csv(ROOT / "NTUST_Load_PV.csv").dropna(subset=["Date", "Time"])
    ntust["Date"] = pd.to_datetime(ntust["Date"])
    ntust["hour_0"] = ntust["Time"].str[:2].astype(int)

    day_to_idx = {pd.Timestamp(r["calendar_day"]): r["day_index"]
                  for _, r in cal_df.iterrows()}

    load_truth_dict = {}
    for _, row in ntust.iterrows():
        d = row["Date"]
        h0 = row["hour_0"]
        if h0 == 0:
            d_assign = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            d_assign = d
            h_local = h0
        if d_assign in day_to_idx:
            load_truth_dict[(d_assign, h_local)] = float(row["Load_kWh"])

    PV_SCALE = 2687.0 / 50.0  # 53.74

    # ── Define ablation variants ──────────────────────────────
    variants = [
        {"name": "K5_euclidean",    "K": 5,  "distance": "euclidean"},
        {"name": "K10_euclidean",   "K": 10, "distance": "euclidean"},
        {"name": "K20_euclidean",   "K": 20, "distance": "euclidean"},
        {"name": "K5_decision",     "K": 5,  "distance": "decision_aware"},
        {"name": "K10_decision",    "K": 10, "distance": "decision_aware"},
    ]

    # Also get baseline C0 for comparison
    print("\n" + "=" * 60)
    print("BASELINE: Solving C0 (deterministic)")
    print("=" * 60)
    c0_case = CASE_TABLE[0]  # C0
    day_data_c0, day_idx_c0, sc_ids_c0 = load_data(CFG, c0_case)
    r_c0 = build_and_solve(day_data_c0, day_idx_c0, sc_ids_c0, CFG, "C0")

    # Replay C0
    truth_df, calendar_replay = load_truth(CFG)
    sizing_c0 = {"CC": r_c0["CC"], "P_B": r_c0["P_B"], "E_B": r_c0["E_B"]}
    rr_c0 = replay(sizing_c0, truth_df, calendar_replay, CFG, "C0")

    print(f"\nC0 baseline: solve={r_c0['obj_val']/1e6:.2f}M, replay={rr_c0['replay_total_M']:.2f}M, "
          f"over={rr_c0['replay_over_M']:.2f}M")

    # ── Run ablation variants ─────────────────────────────────
    results = []

    for var in variants:
        print(f"\n{'=' * 60}")
        print(f"VARIANT: {var['name']} (K={var['K']}, dist={var['distance']})")
        print("=" * 60)

        t0 = time.time()

        # Step 1: Reduce scenarios
        reduced = reduce_scenarios_ablation(
            RAW_500_PATH, K=var["K"],
            distance_mode=var["distance"],
            seed=42,
        )

        # Step 2: Build MILP ingest
        ingest_df = build_ablation_ingest(
            reduced,
            ROOT / "bridge_outputs_fullyear",
            cal_df, load_truth_dict, PV_SCALE,
        )

        # Step 3: Save temporary ingest
        tmp_path = ROOT / "bridge_outputs_fullyear" / f"temp_ablation_{var['name']}.parquet"
        ingest_df.to_parquet(tmp_path, index=False)

        # Step 4: Load and solve
        case_info = {
            "case_id": f"A_{var['name']}",
            "ingest_file": f"temp_ablation_{var['name']}.parquet",
            "pv_mode": "prob",
            "load_mode": "det",
            "label": f"Ablation {var['name']}",
        }

        try:
            day_data, day_indices, scenario_ids = load_data(CFG, case_info)

            # Apply sizing bounds (same as C1: CC×1.015, PB×0.97)
            CC_SCALE = 1.015
            PB_SCALE = 0.97

            r = build_and_solve(day_data, day_indices, scenario_ids, CFG, case_info["case_id"])

            if r:
                # Scale sizing (mimicking the notebook's approach)
                sizing = {"CC": r["CC"], "P_B": r["P_B"], "E_B": r["E_B"]}
                rr = replay(sizing, truth_df, calendar_replay, CFG, case_info["case_id"])

                t_total = time.time() - t0
                results.append({
                    "variant": var["name"],
                    "K": var["K"],
                    "distance": var["distance"],
                    "CC": round(r["CC"], 1),
                    "P_B": round(r["P_B"], 1),
                    "E_B": round(r["E_B"], 1),
                    "solve_M": round(r["obj_val"] / 1e6, 2),
                    "replay_total_M": rr["replay_total_M"],
                    "replay_over_M": rr["replay_over_M"],
                    "replay_ene_M": rr["replay_ene_M"],
                    "replay_inv_M": rr["replay_inv_M"],
                    "over_months": rr["over_months"],
                    "worst_bill_M": rr["worst_bill_M"],
                    "RE_pct": rr["RE_pct"],
                    "solve_time_s": round(r["solve_time"], 1),
                    "total_time_s": round(t_total, 1),
                })
                print(f"  → replay={rr['replay_total_M']:.2f}M, over={rr['replay_over_M']:.2f}M")
            else:
                print(f"  → FAILED")
        except Exception as e:
            print(f"  → ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)

    # Add C0 baseline to results
    baseline_row = {
        "variant": "C0_baseline",
        "K": 1,
        "distance": "n/a",
        "CC": round(r_c0["CC"], 1),
        "P_B": round(r_c0["P_B"], 1),
        "E_B": round(r_c0["E_B"], 1),
        "solve_M": round(r_c0["obj_val"] / 1e6, 2),
        "replay_total_M": rr_c0["replay_total_M"],
        "replay_over_M": rr_c0["replay_over_M"],
        "replay_ene_M": rr_c0["replay_ene_M"],
        "replay_inv_M": rr_c0["replay_inv_M"],
        "over_months": rr_c0["over_months"],
        "worst_bill_M": rr_c0["worst_bill_M"],
        "RE_pct": rr_c0["RE_pct"],
        "solve_time_s": round(r_c0["solve_time"], 1),
        "total_time_s": 0,
    }

    all_results = [baseline_row] + results
    df = pd.DataFrame(all_results)

    print(df[["variant", "K", "distance", "CC", "P_B", "E_B",
              "replay_total_M", "replay_over_M", "worst_bill_M", "RE_pct",
              "solve_time_s"]].to_string(index=False))

    # Save
    out_dir = ROOT / "milp_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scenario_ablation_results.csv", index=False)
    print(f"\n→ Saved to {out_dir / 'scenario_ablation_results.csv'}")

    # Attribution analysis
    print("\n" + "=" * 60)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 60)
    c0_total = rr_c0["replay_total_M"]
    c0_over = rr_c0["replay_over_M"]

    for r in results:
        delta_total = r["replay_total_M"] - c0_total
        delta_over = r["replay_over_M"] - c0_over
        print(f"  {r['variant']:20s}: ΔTotal={delta_total:+.2f}M  ΔOver={delta_over:+.2f}M  "
              f"({'WINS' if delta_total < 0 else 'loses'} on total, "
              f"{'WINS' if delta_over < 0 else 'loses'} on over-contract)")

    return all_results


if __name__ == "__main__":
    # Set Gurobi license
    if "GRB_LICENSE_FILE" not in os.environ:
        lic_path = Path.home() / "gurobi.lic"
        if lic_path.exists():
            os.environ["GRB_LICENSE_FILE"] = str(lic_path)

    run_ablation()
