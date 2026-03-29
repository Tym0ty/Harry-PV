"""
Case-Dependent Scenario Reduction — F0329Bridge_Spec Batch 8, Chapter 10.

Reduces pre-reduction scenarios to a smaller representative set per case:
  C0, C_PFI : no reduction (deterministic, scenario_id='det')
  C1         : PV-only space reduction
  C2         : load-only space reduction
  C3         : joint PV-load space reduction

Method: k-medoids on daily trajectory feature vectors.
Post-reduction: probabilities renormalised to sum to 1.0 per (day, hour).

Usage:
    from scenario_reduction import reduce_package
    reduced_df = reduce_package(df, case_id, n_target, seed=425)
"""
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────

def _kmedoids_1d_on_trajectories(traj_matrix, k, seed=425):
    """
    k-medoids on rows of traj_matrix (n_scenarios × n_hours).
    Returns (medoid_local_indices, labels).
    """
    rng = np.random.default_rng(seed)
    n = len(traj_matrix)
    if k >= n:
        return np.arange(n), np.arange(n)

    medoid_idx = rng.choice(n, size=k, replace=False)
    diff = traj_matrix[:, None, :] - traj_matrix[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    for _ in range(200):
        labels = np.argmin(dist[:, medoid_idx], axis=1)
        new_medoids = medoid_idx.copy()
        changed = False
        for c in range(k):
            pts = np.where(labels == c)[0]
            if len(pts) == 0:
                continue
            best = pts[np.argmin(dist[np.ix_(pts, pts)].sum(axis=1))]
            if best != medoid_idx[c]:
                new_medoids[c] = best
                changed = True
        medoid_idx = new_medoids
        if not changed:
            break

    labels = np.argmin(dist[:, medoid_idx], axis=1)
    return medoid_idx, labels


def _build_trajectory_matrix(day_df, value_col):
    """
    Build (n_scenarios × 24) matrix for a single day.
    day_df: rows for one day_index, sorted by scenario_id then hour_local.
    """
    scenarios = sorted(day_df["scenario_id"].unique())
    traj = []
    for sid in scenarios:
        s_rows = day_df[day_df["scenario_id"] == sid].sort_values("hour_local")
        traj.append(s_rows[value_col].values)
    return np.array(traj), scenarios


def reduce_package(df, case_id, n_target, seed=425):
    """
    Reduce a rolling DA input package to n_target scenarios per day.

    Parameters
    ----------
    df        : full pre-reduction package DataFrame
    case_id   : 'C0','C1','C2','C3','C_PFI'
    n_target  : target number of scenarios after reduction
    seed      : random seed for k-medoids

    Returns
    -------
    reduced_df : DataFrame with same schema, probabilities renormalised
    """
    if case_id in ("C0", "C_PFI"):
        # Already deterministic — no reduction needed
        assert df["scenario_id"].nunique() == 1, (
            f"Expected 1 scenario for {case_id}, got {df['scenario_id'].nunique()}")
        return df.copy()

    day_indices = sorted(df["day_index"].unique())
    reduced_rows = []

    for di in day_indices:
        day_df = df[df["day_index"] == di]

        if case_id == "C1":
            # PV-only reduction: cluster on pv_available_kw trajectories
            traj_matrix, scenarios = _build_trajectory_matrix(day_df, "pv_available_kw")
            medoid_idx, labels = _kmedoids_1d_on_trajectories(traj_matrix, n_target, seed)

        elif case_id == "C2":
            # Load-only reduction: cluster on load_kw trajectories
            traj_matrix, scenarios = _build_trajectory_matrix(day_df, "load_kw")
            medoid_idx, labels = _kmedoids_1d_on_trajectories(traj_matrix, n_target, seed)

        elif case_id == "C3":
            # Joint reduction: cluster on concatenated [pv, load] feature vector
            pv_traj,   sc_pv   = _build_trajectory_matrix(day_df, "pv_available_kw")
            load_traj, sc_load = _build_trajectory_matrix(day_df, "load_kw")
            assert sc_pv == sc_load, "Scenario mismatch between pv and load trajectories"
            scenarios = sc_pv
            # Standardise each half before concatenating
            pv_std   = pv_traj.std()   or 1.0
            load_std = load_traj.std() or 1.0
            joint = np.concatenate(
                [pv_traj / pv_std, load_traj / load_std], axis=1)
            medoid_idx, labels = _kmedoids_1d_on_trajectories(joint, n_target, seed)

        else:
            raise ValueError(f"Unknown case_id: {case_id}")

        # Select medoid scenarios; aggregate probability from assigned cluster
        medoid_scenarios = [scenarios[i] for i in medoid_idx]

        # Count how many original scenarios each medoid represents
        cluster_prob = np.zeros(len(medoid_idx))
        for orig_idx, c in enumerate(labels):
            orig_sid = scenarios[orig_idx]
            orig_prob = float(day_df[day_df["scenario_id"] == orig_sid]["probability_pi"].iloc[0])
            cluster_prob[c] += orig_prob

        # Renormalise so probabilities sum to 1.0
        cluster_prob = cluster_prob / cluster_prob.sum()

        for c_idx, (sid, prob) in enumerate(zip(medoid_scenarios, cluster_prob)):
            s_rows = day_df[day_df["scenario_id"] == sid].sort_values("hour_local").copy()
            s_rows["probability_pi"] = prob
            s_rows["scenario_id"]    = f"r{c_idx:03d}"   # renumber after reduction
            reduced_rows.append(s_rows)

    reduced_df = pd.concat(reduced_rows, ignore_index=True)

    # Acceptance check: probability sums per (day_index, hour_local)
    prob_check = reduced_df.groupby(["day_index", "hour_local"])["probability_pi"].sum()
    if not np.allclose(prob_check.values, 1.0, atol=1e-5):
        bad = prob_check[~np.isclose(prob_check.values, 1.0, atol=1e-5)]
        raise RuntimeError(
            f"[SCENARIO REDUCTION FAIL] Probabilities do not sum to 1.0 after reduction.\n"
            f"Examples:\n{bad.head()}"
        )

    return reduced_df


def reduce_all_packages(out_dir, n_target_c1=5, n_target_c2=5, n_target_c3=10, seed=425):
    """
    Convenience wrapper: reduce C1, C2, C3 packages in-place (overwrite).
    C0 and C_PFI are left unchanged.
    """
    print("=" * 60)
    print("Scenario Reduction — F0329Bridge_Spec Batch 8 §10")
    print("=" * 60)

    targets = {
        "C1": ("rolling_da_input_pvprob_loaddet.parquet",   n_target_c1),
        "C2": ("rolling_da_input_pvdet_loadunc.parquet",    n_target_c2),
        "C3": ("rolling_da_input_pvprob_loadunc.parquet",   n_target_c3),
    }

    for case_id, (fname, n_target) in targets.items():
        fpath = out_dir / fname
        print(f"  Reducing {case_id} ({fname}) → {n_target} scenarios/day...")
        df = pd.read_parquet(fpath)
        before = df["scenario_id"].nunique()
        reduced = reduce_package(df, case_id, n_target, seed=seed)
        after = reduced["scenario_id"].nunique()
        reduced.to_parquet(fpath, index=False)
        print(f"    {before} → {after} scenarios, {len(reduced)} rows")

    print("Scenario reduction complete.")


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "bridge_outputs_fullyear"
    reduce_all_packages(out)
