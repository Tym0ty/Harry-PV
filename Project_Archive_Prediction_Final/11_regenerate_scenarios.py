#!/usr/bin/env python3
"""
Regenerate Gaussian copula scenarios from improved CQR quantiles.

Reads: pipeline_outputs/forecast_ghi_quantiles_daily.parquet (improved CQR v2)
       pipeline_outputs/load_deterministic_hourly.parquet
Writes: pipeline_outputs/scenarios_joint_pv_load_raw_500.parquet
        pipeline_outputs/scenarios_joint_pv_load_reduced_5.parquet

Then re-runs bridge_full_year.py to update MILP ingest packages.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline_outputs"

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
QCOLS = [f"q{q:.2f}" for q in QUANTILES]
N_SCENARIOS = 500
K_REDUCED = 5
RANDOM_SEED = 42
COPULA_DECAY = 0.3

# PVWatts parameters
PV_DC_RATED = 50.0
GAMMA_PDC = -0.0047
INV_EFF = 0.96
DC_AC_RATIO = 1.2
PV_AC_RATED = PV_DC_RATED / DC_AC_RATIO
INOCT = 49.0
MONTHLY_TEMP = {1:16, 2:17, 3:19, 4:23, 5:26, 6:28, 7:30, 8:30, 9:28, 10:25, 11:21, 12:18}


def quantile_function(quantiles, q_values):
    taus = np.array([0.0] + list(quantiles) + [1.0])
    vals = np.concatenate([[q_values[0]], q_values, [q_values[-1]]])
    return interp1d(taus, vals, kind='linear', bounds_error=False,
                    fill_value=(vals[0], vals[-1]))


def kmedoids_pam(X, K, seed=42, max_iter=300):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = cdist(X, X, metric="euclidean")
    medoids = rng.choice(n, size=K, replace=False)
    for _ in range(max_iter):
        dists = D[:, medoids]
        labels = dists.argmin(axis=1)
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
    dists = D[:, medoids]
    labels = dists.argmin(axis=1)
    return medoids, labels


def ghi_to_pv(ghi_wm2, month, hour, solar_elev=None):
    """PVWatts-style GHI → PV conversion."""
    if solar_elev is not None and solar_elev <= 0:
        return 0.0
    ghi = max(0, ghi_wm2)
    if ghi < 1:
        return 0.0
    poa = ghi
    t_air = MONTHLY_TEMP.get(month, 25)
    t_cell = t_air + poa / 800.0 * (INOCT - 20)
    p_dc = PV_DC_RATED * (poa / 1000.0) * (1 + GAMMA_PDC * (t_cell - 25))
    p_ac = min(PV_AC_RATED, p_dc * INV_EFF)
    return max(0, p_ac)


def run():
    print("=" * 60)
    print("Regenerate Scenarios from Improved CQR Quantiles")
    print("=" * 60)

    # Load improved quantiles
    fc = pd.read_parquet(PIPELINE / "forecast_ghi_quantiles_daily.parquet")
    fc["target_time_local"] = pd.to_datetime(fc["target_time_local"])
    fc["target_day_local"] = pd.to_datetime(fc["target_day_local"])

    # Use test split only
    fc_test = fc[fc["split_name"] == "test"].copy()
    print(f"  Test set: {len(fc_test)} rows")

    # Load deterministic load
    load_det = pd.read_parquet(PIPELINE / "load_deterministic_hourly.parquet")
    load_det["target_time_local"] = pd.to_datetime(load_det["target_time_local"])
    load_lookup = load_det[["target_time_local", "load_kw"]].drop_duplicates("target_time_local")
    if hasattr(load_lookup["target_time_local"].dtype, "tz") and load_lookup["target_time_local"].dt.tz is not None:
        load_lookup["target_time_local"] = load_lookup["target_time_local"].dt.tz_localize(None)
    load_map = dict(zip(load_lookup["target_time_local"], load_lookup["load_kw"]))

    # Temporal correlation matrix
    dim = 24
    Sigma = np.eye(dim)
    for i in range(24):
        for j in range(24):
            Sigma[i, j] = np.exp(-COPULA_DECAY * abs(i - j))

    # Generate raw scenarios
    print(f"  Generating {N_SCENARIOS} raw GHI scenarios per day...")
    target_days = sorted(fc_test["target_day_local"].unique())
    print(f"  Days: {len(target_days)}")

    all_raw = []
    for day in tqdm(target_days, desc="GHI copula"):
        grp = fc_test[fc_test["target_day_local"] == day].sort_values("target_time_local")
        if len(grp) < 24:
            continue
        grp = grp.head(24)

        # Build quantile functions
        ghi_qfuncs = []
        for _, row in grp.iterrows():
            q_vals = row[QCOLS].values.astype(float)
            ghi_qfuncs.append(quantile_function(QUANTILES, q_vals))

        # Sample via Gaussian copula
        rng = np.random.default_rng(RANDOM_SEED + hash(str(day)) % 10000)
        z = rng.multivariate_normal(np.zeros(dim), Sigma, size=N_SCENARIOS)
        u = stats.norm.cdf(z)

        issue_day = day - pd.Timedelta(days=1)
        horizon_hours = grp["horizon_hour"].values
        solar_elevs = grp["solar_elevation"].values if "solar_elevation" in grp.columns else [None] * 24

        for s in range(N_SCENARIOS):
            for h in range(24):
                ghi_val = max(0.0, float(ghi_qfuncs[h](u[s, h])))
                target_time = pd.Timestamp(day) + pd.Timedelta(hours=h)
                month = target_time.month

                pv_kw = ghi_to_pv(ghi_val, month, h,
                                  solar_elevs[h] if solar_elevs[h] is not None else None)

                load_kw = load_map.get(target_time, 0.0)

                all_raw.append({
                    "issue_day_local": issue_day,
                    "target_day_local": day,
                    "target_time_local": target_time,
                    "horizon_hour": horizon_hours[h],
                    "scenario_id": s,
                    "ghi_sample_wm2": ghi_val,
                    "load_kw": load_kw,
                    "ghi_wm2": ghi_val,
                    "poa_effective": ghi_val,
                    "hour": h,
                    "month": month,
                    "temp_air": MONTHLY_TEMP.get(month, 25),
                    "wind_speed_est": 2.0,
                    "t_cell": MONTHLY_TEMP.get(month, 25) + ghi_val / 800.0 * (INOCT - 20),
                    "p_dc": PV_DC_RATED * (ghi_val / 1000.0) * (1 + GAMMA_PDC * (MONTHLY_TEMP.get(month, 25) + ghi_val / 800.0 * (INOCT - 20) - 25)),
                    "pv_available_kw": pv_kw,
                })

    raw_df = pd.DataFrame(all_raw)
    print(f"  Raw scenarios: {raw_df.shape}")

    # Save raw
    raw_df.to_parquet(PIPELINE / f"scenarios_joint_pv_load_raw_{N_SCENARIOS}.parquet", index=False)
    print(f"  → scenarios_joint_pv_load_raw_{N_SCENARIOS}.parquet")

    # K-medoids reduction
    print(f"\n  Reducing {N_SCENARIOS} → {K_REDUCED} via k-medoids...")
    all_reduced = []

    for day in tqdm(target_days, desc="k-medoids"):
        day_data = raw_df[raw_df["target_day_local"] == day]
        if len(day_data) == 0:
            continue

        # Pivot PV to (n_scenarios, 24)
        pivoted = day_data.pivot_table(
            index="scenario_id",
            columns=day_data["target_time_local"].dt.hour,
            values="pv_available_kw",
            aggfunc="first"
        ).fillna(0)

        if len(pivoted) < K_REDUCED:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(pivoted.values)

        medoids, labels = kmedoids_pam(X, K_REDUCED, seed=RANDOM_SEED)
        medoid_sids = pivoted.index[medoids].tolist()
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        probs = (cluster_sizes / cluster_sizes.sum()).values

        for k_idx, (sid, prob) in enumerate(zip(medoid_sids, probs)):
            medoid_rows = day_data[day_data["scenario_id"] == sid].copy()
            medoid_rows = medoid_rows.assign(
                scenario_id=k_idx,
                probability_pi=prob,
                medoid_raw_id=sid,
            )
            all_reduced.append(medoid_rows)

    reduced_df = pd.concat(all_reduced, ignore_index=True)

    # Keep standard columns
    keep_cols = ["issue_day_local", "target_day_local", "scenario_id",
                 "target_time_local", "horizon_hour",
                 "pv_available_kw", "load_kw", "probability_pi", "medoid_raw_id"]
    reduced_df = reduced_df[[c for c in keep_cols if c in reduced_df.columns]]

    reduced_df.to_parquet(PIPELINE / f"scenarios_joint_pv_load_reduced_{K_REDUCED}.parquet", index=False)
    print(f"  → scenarios_joint_pv_load_reduced_{K_REDUCED}.parquet: {reduced_df.shape}")

    # Verify probabilities
    pi_check = reduced_df.groupby("target_day_local").apply(
        lambda g: g.drop_duplicates("scenario_id")["probability_pi"].sum())
    print(f"  Prob sum check: min={pi_check.min():.4f}, max={pi_check.max():.4f}")

    return reduced_df


if __name__ == "__main__":
    run()

    # Re-run bridge
    print("\n" + "=" * 60)
    print("Re-running bridge with updated scenarios...")
    print("=" * 60)
    sys.path.insert(0, str(ROOT / "notebooks_bridge"))
    from bridge_full_year import run_bridge
    run_bridge()
