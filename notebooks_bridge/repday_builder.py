"""
Representative-Day Builder — F0329Bridge_Spec Batch 8, Chapter 11.

Pipeline:
  Step 1 — Day-descriptor construction (calendar-day level features)
  Step 2 — Risk-day retention (monthly peak, net-load stress, reduced PV)
  Step 3 — Body-day clustering (k-medoids, actual medoid day)
  Step 4 — Calendar-to-repday mapping
  Step 5 — Weight computation + closure check (sum == |N|)
  Step 6 — Repday metadata
  Step 7 — Build per-case repday design packages (5 cases)

Outputs written to OUT_DIR (same as bridge_outputs_fullyear):
  calendar_to_repday_map.parquet
  repday_weights.parquet
  repdays_metadata.parquet
  repday_input_pvdet_loaddet.parquet        (C0)
  repday_input_pvprob_loaddet.parquet       (C1)
  repday_input_pvdet_loadunc.parquet        (C2)
  repday_input_pvprob_loadunc.parquet       (C3)
  repday_input_pvperfect_loadperfect.parquet (C_PFI)
  repday_build_report.json
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json, datetime

# ── Configuration ─────────────────────────────────────────────
N_BODY_CLUSTERS   = 30     # k-medoids body clusters (adjust based on data)
SEED              = 425
RISK_PEAK_TOPN    = 3      # retain top-N peak-load days per month
RISK_NETLOAD_PCT  = 0.02   # retain top 2% net-load stress days
RISK_PV_PCT       = 0.02   # retain bottom 2% PV availability days (cloudy stress)

# ── Helpers ───────────────────────────────────────────────────

def _hard_fail(msg):
    raise RuntimeError(f"[REPDAY HARD FAIL] {msg}")


def _kmedoids(X, k, seed=SEED, max_iter=200):
    """
    Simple k-medoids implementation (PAM initialisation + swap).
    X: (n_samples, n_features) array
    Returns: medoid_indices (length k), labels (length n_samples)
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    # Initialise medoids randomly
    medoid_idx = rng.choice(n, size=k, replace=False)

    # Precompute pairwise L2 distances
    diff = X[:, None, :] - X[None, :, :]       # (n, n, f)
    dist = np.sqrt((diff ** 2).sum(axis=-1))    # (n, n)

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        labels = np.argmin(dist[:, medoid_idx], axis=1)

        new_medoids = medoid_idx.copy()
        changed = False
        for c in range(k):
            cluster_pts = np.where(labels == c)[0]
            if len(cluster_pts) == 0:
                continue
            # Find point in cluster that minimises total distance to cluster
            intra = dist[np.ix_(cluster_pts, cluster_pts)].sum(axis=1)
            best_local = cluster_pts[np.argmin(intra)]
            if best_local != medoid_idx[c]:
                new_medoids[c] = best_local
                changed = True
        medoid_idx = new_medoids
        if not changed:
            break

    # Final assignment
    labels = np.argmin(dist[:, medoid_idx], axis=1)
    return medoid_idx, labels


# ── Main entry point ─────────────────────────────────────────

def build_repdays(out_dir: Path):
    """
    Build all representative-day artifacts.
    Reads from out_dir (same as bridge_outputs_fullyear):
      - caseyear_calendar_manifest.parquet
      - rolling_da_input_pvdet_loaddet.parquet   (for load baseline)
      - rolling_da_input_pvprob_loaddet.parquet  (for PV scenarios)
      - rolling_da_input_pvdet_loadunc.parquet   (for load uncertainty)
      - rolling_da_input_pvprob_loadunc.parquet  (for C3 joint)
      - rolling_da_input_pvperfect_loadperfect.parquet (for C_PFI)
      - full_year_replay_truth_package.parquet   (for realized PV)
    """
    print("=" * 60)
    print("Representative-Day Builder — F0329Bridge_Spec Batch 8")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────
    cal_df  = pd.read_parquet(out_dir / "caseyear_calendar_manifest.parquet")
    c0_pkg  = pd.read_parquet(out_dir / "rolling_da_input_pvdet_loaddet.parquet")
    truth_pkg = pd.read_parquet(out_dir / "full_year_replay_truth_package.parquet")

    cal_df["calendar_day"] = pd.to_datetime(cal_df["calendar_day"])
    n_days = len(cal_df)
    day_indices = sorted(cal_df["day_index"].tolist())

    # ── Step 1: Day descriptors ───────────────────────────────
    print("Step 1: Building day descriptors...")

    # Pre-aggregate C0 package to daily sums
    c0_daily = c0_pkg.groupby("day_index").agg(
        total_load_kwh = ("load_kw", "sum"),
        peak_load_kw   = ("load_kw", "max"),
    ).reset_index()

    truth_daily = truth_pkg.groupby("day_index").agg(
        total_pv_kwh   = ("pv_realized_kw", "sum"),
        peak_pv_kw     = ("pv_realized_kw", "max"),
        total_load_real = ("load_realized_kw", "sum"),
    ).reset_index()

    desc_df = cal_df.merge(c0_daily,    on="day_index", how="left")
    desc_df = desc_df.merge(truth_daily, on="day_index", how="left")
    desc_df["net_load_kwh"] = desc_df["total_load_kwh"] - desc_df["total_pv_kwh"].fillna(0)
    desc_df["pv_ratio"]     = (
        desc_df["total_pv_kwh"].fillna(0) /
        (desc_df["total_load_kwh"].replace(0, np.nan))
    ).fillna(0)

    # ── Step 2: Risk-day retention ────────────────────────────
    print("Step 2: Risk-day retention...")
    risk_flags  = pd.Series(False, index=desc_df.index)
    risk_reason = pd.Series("", index=desc_df.index)

    # 2a. Monthly peak-load risk: top RISK_PEAK_TOPN load days per month
    for m in desc_df["month_id"].unique():
        month_mask = desc_df["month_id"] == m
        top_idx = (desc_df.loc[month_mask, "peak_load_kw"]
                   .nlargest(RISK_PEAK_TOPN).index)
        risk_flags[top_idx] = True
        risk_reason[top_idx] = risk_reason[top_idx].str.cat(
            pd.Series("peak_load", index=top_idx), sep="|").str.strip("|")

    # 2b. High net-load stress
    thresh_netload = desc_df["net_load_kwh"].quantile(1 - RISK_NETLOAD_PCT)
    hi_net = desc_df.index[desc_df["net_load_kwh"] >= thresh_netload]
    risk_flags[hi_net] = True
    risk_reason[hi_net] = risk_reason[hi_net].str.cat(
        pd.Series("net_load_stress", index=hi_net), sep="|").str.strip("|")

    # 2c. Severely reduced PV (cloudy stress)
    thresh_pv = desc_df["total_pv_kwh"].quantile(RISK_PV_PCT)
    lo_pv = desc_df.index[desc_df["total_pv_kwh"].fillna(0) <= thresh_pv]
    risk_flags[lo_pv] = True
    risk_reason[lo_pv] = risk_reason[lo_pv].str.cat(
        pd.Series("low_pv", index=lo_pv), sep="|").str.strip("|")

    desc_df["is_risk_day"]     = risk_flags
    desc_df["retention_reason"] = risk_reason.where(risk_flags, "")
    n_risk = risk_flags.sum()
    n_body = n_days - n_risk
    print(f"  Risk days: {n_risk} | Body days: {n_body}")

    # ── Step 3: Body-day clustering (k-medoids) ───────────────
    print(f"Step 3: k-medoids clustering ({N_BODY_CLUSTERS} clusters)...")
    body_mask = ~desc_df["is_risk_day"].values
    body_idx  = desc_df.index[body_mask]
    body_desc = desc_df.loc[body_idx, ["total_load_kwh", "total_pv_kwh",
                                       "net_load_kwh", "pv_ratio"]].fillna(0)

    # Standardise features
    feat_mean = body_desc.mean()
    feat_std  = body_desc.std().replace(0, 1)
    X = ((body_desc - feat_mean) / feat_std).values

    k = min(N_BODY_CLUSTERS, len(body_idx))
    medoid_local_idx, labels = _kmedoids(X, k, seed=SEED)

    # Map back to desc_df row indices and day_indices
    medoid_desc_idx = body_idx[medoid_local_idx]    # desc_df indices of medoids
    medoid_day_idx  = desc_df.loc[medoid_desc_idx, "day_index"].values

    # Assign each body day to its cluster's medoid day_index
    body_repday = np.array([medoid_day_idx[c] for c in labels])

    print(f"  Clusters: {k}, body medoids selected.")

    # ── Step 4: Calendar-to-repday mapping ───────────────────
    print("Step 4: Building calendar-to-repday map...")
    map_rows = []
    for i, row in desc_df.iterrows():
        di = row["day_index"]
        if row["is_risk_day"]:
            repday_id = int(di)     # risk day maps to itself
            weight_tag = "risk"
        else:
            # Find position of this body day in body_idx
            body_pos = np.where(body_idx == i)[0][0]
            repday_id = int(body_repday[body_pos])
            weight_tag = "body_cluster"
        map_rows.append({
            "day_index":        int(di),
            "calendar_day":     row["calendar_day"],
            "repday_id":        repday_id,
            "is_risk_day":      bool(row["is_risk_day"]),
            "retention_reason": row["retention_reason"],
            "weight_source_tag": weight_tag,
            "month_id":         int(row["month_id"]),
            "season_tag":       row["season_tag"],
            "day_type":         row["day_type"],
            "is_holiday":       bool(row["is_holiday"]),
        })

    map_df = pd.DataFrame(map_rows)

    # ── Step 5: Weight computation + closure check ────────────
    print("Step 5: Weight computation + closure check...")
    weight_df = (
        map_df.groupby("repday_id")
        .size()
        .reset_index(name="annual_weight_days")
    )
    weight_df["mapped_day_count"] = weight_df["annual_weight_days"]
    weight_df["repday_type"] = weight_df["repday_id"].apply(
        lambda rid: "risk" if rid in desc_df.loc[desc_df["is_risk_day"], "day_index"].values
        else "body"
    )

    weight_sum = weight_df["annual_weight_days"].sum()
    if weight_sum != n_days:
        _hard_fail(f"Weight closure FAILED: sum={weight_sum} != n_days={n_days}")
    print(f"  Weight closure: PASS (sum={weight_sum} == n_days={n_days})")

    # ── Step 6: Repday metadata ───────────────────────────────
    print("Step 6: Building repday metadata...")
    repday_ids = sorted(weight_df["repday_id"].unique())
    meta_rows = []
    for rid in repday_ids:
        src_row = desc_df[desc_df["day_index"] == rid].iloc[0]
        rtype   = "risk" if src_row["is_risk_day"] else "body"
        meta_rows.append({
            "repday_id":          int(rid),
            "repday_type":        rtype,
            "source_calendar_day": src_row["calendar_day"],
            "month_id":           int(src_row["month_id"]),
            "season_tag":         src_row["season_tag"],
            "day_type":           src_row["day_type"],
            "is_holiday":         bool(src_row["is_holiday"]),
            "total_load_kwh":     float(src_row.get("total_load_kwh", 0)),
            "total_pv_kwh":       float(src_row.get("total_pv_kwh", 0)),
            "net_load_kwh":       float(src_row.get("net_load_kwh", 0)),
            "peak_load_kw":       float(src_row.get("peak_load_kw", 0)),
            "retention_reason":   src_row["retention_reason"],
        })
    meta_df = pd.DataFrame(meta_rows)

    # Acceptance check: every repday_id used in map_df exists in weight_df
    repday_in_map    = set(map_df["repday_id"].unique())
    repday_in_weight = set(weight_df["repday_id"].unique())
    missing_in_weight = repday_in_map - repday_in_weight
    if missing_in_weight:
        _hard_fail(f"repday_ids {missing_in_weight} in map not found in weight table.")

    # ── Step 7: Build per-case repday design packages ─────────
    print("Step 7: Building repday design packages for 5 cases...")

    # Load all rolling packages (we select repday rows from them)
    rolling = {
        "pvdet_loaddet":           pd.read_parquet(out_dir / "rolling_da_input_pvdet_loaddet.parquet"),
        "pvprob_loaddet":          pd.read_parquet(out_dir / "rolling_da_input_pvprob_loaddet.parquet"),
        "pvdet_loadunc":           pd.read_parquet(out_dir / "rolling_da_input_pvdet_loadunc.parquet"),
        "pvprob_loadunc":          pd.read_parquet(out_dir / "rolling_da_input_pvprob_loadunc.parquet"),
        "pvperfect_loadperfect":   pd.read_parquet(out_dir / "rolling_da_input_pvperfect_loadperfect.parquet"),
    }

    REPDAY_CASES = {
        "repday_input_pvdet_loaddet.parquet":         "pvdet_loaddet",
        "repday_input_pvprob_loaddet.parquet":        "pvprob_loaddet",
        "repday_input_pvdet_loadunc.parquet":         "pvdet_loadunc",
        "repday_input_pvprob_loadunc.parquet":        "pvprob_loadunc",
        "repday_input_pvperfect_loadperfect.parquet": "pvperfect_loadperfect",
    }

    for out_fname, roll_key in REPDAY_CASES.items():
        roll_df = rolling[roll_key]
        # Select only rows where day_index is a repday_id
        repday_rows = roll_df[roll_df["day_index"].isin(repday_ids)].copy()

        # Add repday metadata columns
        weight_lookup = weight_df.set_index("repday_id")["annual_weight_days"].to_dict()
        type_lookup   = weight_df.set_index("repday_id")["repday_type"].to_dict()
        repday_rows["repday_id"]     = repday_rows["day_index"]
        repday_rows["repday_type"]   = repday_rows["repday_id"].map(type_lookup)
        repday_rows["mapped_weight"] = repday_rows["repday_id"].map(weight_lookup)

        # Source calendar day info
        src_day_map = meta_df.set_index("repday_id")["source_calendar_day"].to_dict()
        repday_rows["source_calendar_day"] = repday_rows["repday_id"].map(src_day_map)

        # Acceptance: no truth columns in C0-C3 repday packages
        if "perfect" not in roll_key:
            forbidden = {"pv_realized_kw", "load_realized_kw"}
            found = forbidden & set(repday_rows.columns)
            if found:
                _hard_fail(f"Truth columns {found} in repday package '{out_fname}'")

        # Acceptance: every repday_id has a weight
        missing_weight = set(repday_rows["repday_id"].unique()) - set(weight_lookup.keys())
        if missing_weight:
            _hard_fail(f"repday_ids {missing_weight} in '{out_fname}' missing from weight table.")

        repday_rows.to_parquet(out_dir / out_fname, index=False)
        print(f"  → {out_fname}: {len(repday_rows)} rows, "
              f"{repday_rows['repday_id'].nunique()} repdays")

    # ── Write metadata artifacts ───────────────────────────────
    map_df.to_parquet(out_dir / "calendar_to_repday_map.parquet", index=False)
    weight_df.to_parquet(out_dir / "repday_weights.parquet", index=False)
    meta_df.to_parquet(out_dir / "repdays_metadata.parquet", index=False)
    print(f"  → calendar_to_repday_map.parquet: {len(map_df)} rows")
    print(f"  → repday_weights.parquet: {len(weight_df)} repdays")
    print(f"  → repdays_metadata.parquet: {len(meta_df)} rows")

    # ── Build report ──────────────────────────────────────────
    report = {
        "spec_ref":             "F0329Bridge_Spec Batch 8 §11",
        "n_case_year_days":     n_days,
        "n_risk_days":          int(n_risk),
        "n_body_days":          int(n_body),
        "n_body_clusters":      int(k),
        "n_repdays_total":      len(repday_ids),
        "weight_closure_pass":  True,
        "weight_sum":           int(weight_sum),
        "clustering": {
            "method":        "k-medoids (PAM)",
            "distance":      "L2 on standardised descriptors",
            "features":      ["total_load_kwh", "total_pv_kwh", "net_load_kwh", "pv_ratio"],
            "seed":          SEED,
        },
        "risk_retention": {
            "peak_top_n_per_month": RISK_PEAK_TOPN,
            "net_load_top_pct":     RISK_NETLOAD_PCT,
            "pv_bottom_pct":        RISK_PV_PCT,
        },
        "created_at": datetime.datetime.now().isoformat(),
    }
    with open(out_dir / "repday_build_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RepDay builder complete!")
    print(f"  {n_risk} risk + {k} body clusters = {len(repday_ids)} total repdays")
    print(f"  Weight closure: PASS ({weight_sum} == {n_days})")
    print("=" * 60)
    return report


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "bridge_outputs_fullyear"
    build_repdays(out)
