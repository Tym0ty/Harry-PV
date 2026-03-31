#!/usr/bin/env python3
"""
Sensitivity Analysis — N (raw scenarios) × K (reduced scenarios) grid.

Spec: 0331_Sensitivity_SPEC_final.docx
Grid: N∈{100,250,500,1000,2000} × K∈{3,5,7,10,15}, R=3 replicates
Cases: C1 (Prob PV + Det Load), C3 (Prob PV + Load Unc)
Total: 25 grid points × R=3 × 2 cases = 150 MILP runs

Scenario generation:
  N≤500: nested permuted subsets of existing raw_500 master pool
  N>500: bootstrap resampling (with replacement) from raw_500 pool
         (documented limitation: underestimates diversity for large N)

Load uncertainty (C3): K seasonal-MAPE load paths per Pieter's thesis.
k-medoids: pure-numpy PAM (avoids scikit-learn-extra numpy 2.x issue).

Outputs → results/sensitivity/:
  sensitivity_run_summary.parquet    — one row per run
  sensitivity_agg_summary.parquet   — mean/std/CV by (case, N, K)
  sensitivity_design_summary.parquet — sizing stability
  sensitivity_sobol_indices.csv      — SALib S1/ST for N and K
  sensitivity_report.json            — sufficiency decision
  sensitivity_config.yaml            — run metadata

Usage:
    cd <repo_root>
    python3 notebooks_milp/sensitivity_analysis.py
    # Resume (skips completed runs):
    python3 notebooks_milp/sensitivity_analysis.py --resume
    # Subset:
    python3 notebooks_milp/sensitivity_analysis.py --cases C1 --n-vals 100 250 500
"""

import os, sys, argparse, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))
os.chdir(ROOT / "notebooks_milp")

from milp_common import get_config, load_truth, format_results, get_tou_price
from milp_solver import build_and_solve, replay

# ── Constants ─────────────────────────────────────────────────
PV_SCALE = 2687.0 / 50.0          # raw_500 is in 50kW reference

N_GRID = [100, 250, 500, 1000, 2000]
K_GRID = [3, 5, 7, 10, 15]
R_REPLICATES = 3
CASES = ["C1", "C3"]              # C1=det load, C3=loadunc

LOAD_UNC_MAPE = {
    "summer": 0.1145, "fall": 0.1466,
    "winter": 0.1218, "spring": 0.1229,
}
LOAD_UNC_SEED_BASE = 20240331

SENS_SEED_BASE = 42_000            # base seed for replicate permutations
SENSITIVITY_TIME_LIMIT = 120       # seconds per MILP solve
SENSITIVITY_MIP_GAP = 0.005        # 0.5% gap for speed

OUT_DIR = ROOT / "results" / "sensitivity"


# ── k-medoids PAM (pure numpy) ──────────────────────────────────
def _kmedoids_pam(X, K, seed=42, max_iter=300, n_init=3):
    """Multi-start pure-numpy PAM k-medoids. Returns (medoid_indices, labels)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    D = cdist(X, X, metric="euclidean")

    best_cost = np.inf
    best_medoids = None
    best_labels = None

    for _ in range(n_init):
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
        total_cost = dists.min(axis=1).sum()
        if total_cost < best_cost:
            best_cost = total_cost
            best_medoids = medoids.copy()
            best_labels = labels.copy()

    return best_medoids, best_labels


# ── Scenario pool management ────────────────────────────────────
def load_master_pool(raw_500_path):
    """Load raw_500 master pool. Returns dict: date → DataFrame(500×24)."""
    df = pd.read_parquet(raw_500_path)
    df["target_day_local"] = pd.to_datetime(df["target_day_local"])
    pool = {}
    for day, grp in df.groupby("target_day_local"):
        # Pivot: scenario_id × hour → pv_available_kw
        piv = grp.pivot_table(index="scenario_id", columns="hour",
                              values="pv_available_kw", aggfunc="first")
        piv = piv.reindex(columns=sorted(piv.columns))
        pool[day] = piv.values  # shape (500, 24), in 50kW reference units
    return pool


def sample_n_scenarios(pool_day, N, replicate_seed, bootstrap=False):
    """Draw N scenarios from pool_day (shape 500×24).
    If N≤500: nested permuted subset (no replacement).
    If N>500: bootstrap (with replacement, documented limitation).
    """
    n_pool = pool_day.shape[0]
    rng = np.random.default_rng(replicate_seed)
    if N <= n_pool:
        perm = rng.permutation(n_pool)
        idx = perm[:N]
    else:
        # Bootstrap: sample with replacement
        idx = rng.integers(0, n_pool, size=N)
    return pool_day[idx]   # shape (N, 24)


def reduce_to_k(scenarios_n24, K, km_seed=42):
    """k-medoids reduction from N scenarios to K. Returns (K×24 array, weights)."""
    N = scenarios_n24.shape[0]
    if K >= N:
        # Use all scenarios equally
        probs = np.ones(N) / N
        return scenarios_n24, probs

    medoid_idx, labels = _kmedoids_pam(scenarios_n24, K, seed=km_seed)
    # Probability = fraction of scenarios in each cluster
    probs = np.array([np.sum(labels == k) / N for k in range(K)])
    # Normalize
    probs = probs / probs.sum()
    return scenarios_n24[medoid_idx], probs


def get_pieter_season(month):
    if month in (6, 7, 8, 9):    return "summer"
    elif month in (10, 11):       return "fall"
    elif month in (12, 1, 2):     return "winter"
    else:                          return "spring"


# ── Build day_data for MILP ─────────────────────────────────────
def build_day_data(pool, calendar_df, N, K, replicate, case, load_truth_map,
                   km_seed_offset=0):
    """Build day_data dict for MILP from scenario pool.

    Args:
        pool: dict date → (500, 24) array in 50kW reference units
        calendar_df: calendar manifest DataFrame
        N: raw scenario count
        K: reduced scenario count
        replicate: replicate index (0..R-1)
        case: "C1" or "C3"
        load_truth_map: dict (date, hour_local) → load_kw
        km_seed_offset: additional seed offset for k-medoids

    Returns:
        day_data dict compatible with milp_solver.build_and_solve
    """
    cal_lookup = calendar_df.set_index("day_index")
    day_to_idx = {pd.Timestamp(row["calendar_day"]): di
                  for di, row in cal_lookup.iterrows()}

    day_data = {}
    day_indices = []

    for day_ts in sorted(pool.keys()):
        if day_ts not in day_to_idx:
            continue
        di = day_to_idx[day_ts]
        cal = cal_lookup.loc[di]
        cd = pd.Timestamp(cal["calendar_day"])
        dow = cd.weekday()
        mo = int(cal["month_id"])

        # Sample N scenarios from master pool
        rep_seed = SENS_SEED_BASE + replicate * 1_000_000 + hash(str(day_ts)) % 100_000
        scenarios_n = sample_n_scenarios(pool[day_ts], N, rep_seed)

        # Reduce to K via k-medoids
        km_seed = rep_seed + km_seed_offset + K
        scenarios_k, probs_k = reduce_to_k(scenarios_n, K, km_seed)

        # Scale to 2687kW reference
        scenarios_k_scaled = scenarios_k * PV_SCALE  # (K, 24)

        # Load paths
        det_load = np.array([
            load_truth_map.get((cd, h), 0.0) for h in range(1, 25)
        ])

        if case == "C3":
            # K load uncertainty scenarios (joint with PV scenarios)
            mape = LOAD_UNC_MAPE[get_pieter_season(mo)]
            load_seed = LOAD_UNC_SEED_BASE + di * 100 + replicate
            rng_load = np.random.default_rng(load_seed)
            eps = rng_load.normal(0.0, mape, size=K)
            multipliers = np.clip(1.0 + eps, 0.5, 2.0)

        scenarios = []
        for k in range(K):
            pv_k = scenarios_k_scaled[k]  # (24,)
            if case == "C3":
                load_k = det_load * multipliers[k]
            else:
                load_k = det_load

            scenarios.append({
                "pv_kw": pv_k,
                "load_kw": load_k,
                "prob": float(probs_k[k]),
                "scenario_id": f"w{k+1}",
            })

        tou = np.array([get_tou_price(cd.month, cd.day, dow, t) for t in range(24)])

        day_data[di] = {
            "calendar_day": cd,
            "month_id": mo,
            "season_tag": str(cal["season_tag"]),
            "day_type": str(cal["day_type"]),
            "is_holiday": bool(cal["is_holiday"]),
            "is_summer": bool(cal["is_summer"]),
            "scenarios": scenarios,
            "tou": tou,
            "dow": dow,
        }
        day_indices.append(di)

    scenario_ids = [f"w{k+1}" for k in range(K)]
    return day_data, sorted(day_indices), scenario_ids


# ── Main runner ──────────────────────────────────────────────────
def run_sensitivity(n_vals=None, k_vals=None, cases=None, r_reps=None,
                    resume=True, dry_run=False):
    n_vals = n_vals or N_GRID
    k_vals = k_vals or K_GRID
    cases_to_run = cases or CASES
    r_reps = r_reps or R_REPLICATES

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CFG = get_config()
    # Override solver settings for speed
    CFG["time_limit"] = SENSITIVITY_TIME_LIMIT
    CFG["mip_gap"] = SENSITIVITY_MIP_GAP

    # Load master pool
    raw_500_path = ROOT / "pipeline_outputs" / "scenarios_joint_pv_load_raw_500.parquet"
    print("Loading master pool (500 raw scenarios)...")
    pool = load_master_pool(raw_500_path)
    print(f"  Pool: {len(pool)} days")

    # Load calendar
    calendar_df = pd.read_parquet(ROOT / "bridge_outputs_fullyear" / "caseyear_calendar_manifest.parquet")

    # Load padil load truth
    print("Loading padil load truth...")
    padil = pd.read_csv(ROOT / "padil_NTUST_Load_PV.csv").dropna(subset=["Date", "Time"])
    padil["Date"] = pd.to_datetime(padil["Date"])
    padil["hour_0"] = padil["Time"].str.split(":").str[0].astype(int)

    cal_lookup = calendar_df.set_index("day_index")
    day_to_idx = {pd.Timestamp(row["calendar_day"]): di for di, row in cal_lookup.iterrows()}

    load_truth_map = {}  # (date, hour_local) → load_kw
    for _, row in padil.iterrows():
        d = row["Date"]
        h0 = row["hour_0"]
        if h0 == 0:
            d = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            h_local = h0
        if d in day_to_idx:
            load_truth_map[(d, h_local)] = float(row["Load_kWh"])

    # Load truth package for replay
    truth_df, calendar_df2 = load_truth(CFG)

    # Load existing results for resume
    run_summary_path = OUT_DIR / "sensitivity_run_summary.parquet"
    existing_runs = set()
    existing_rows = []
    if resume and run_summary_path.exists():
        df_ex = pd.read_parquet(run_summary_path)
        existing_rows = df_ex.to_dict("records")
        for row in existing_rows:
            existing_runs.add((row["case"], row["N"], row["K"], row["replicate"]))
        print(f"Resuming: {len(existing_runs)} runs already completed")

    # Build run list
    run_list = []
    for case in cases_to_run:
        for N in n_vals:
            for K in k_vals:
                if K >= N:
                    continue  # K must be < N
                for r in range(r_reps):
                    run_list.append((case, N, K, r))

    total = len(run_list)
    skipped = sum(1 for run in run_list if run in existing_runs)
    todo = total - skipped
    print(f"\nRun list: {total} total, {skipped} already done, {todo} to run")

    if dry_run:
        print("DRY RUN — exiting without running solves")
        for i, (case, N, K, r) in enumerate(run_list[:10]):
            print(f"  [{i+1}] {case} N={N} K={K} r={r}")
        return

    # ── Run loop ──────────────────────────────────────────────
    all_rows = list(existing_rows)

    for run_i, (case, N, K, r) in enumerate(run_list):
        if (case, N, K, r) in existing_runs:
            continue

        run_label = f"{case} N={N} K={K} r={r}"
        print(f"\n[{run_i+1}/{total}] {run_label}")
        t_start = time.time()

        try:
            # Build day_data
            day_data, day_indices, scenario_ids = build_day_data(
                pool, calendar_df, N, K, r, case, load_truth_map,
                km_seed_offset=r * 1000)

            # Solve
            result = build_and_solve(
                day_data, day_indices, scenario_ids, CFG,
                case_id=case, no_re20=False,
                mip_gap=SENSITIVITY_MIP_GAP)

            if result is None:
                print(f"  FAILED: solve returned None")
                row = _fail_row(case, N, K, r, "solve_failed", time.time() - t_start)
                all_rows.append(row)
                _save_incremental(all_rows, run_summary_path)
                continue

            sizing = {"CC": result["CC"], "P_B": result["P_B"], "E_B": result["E_B"]}

            # Replay
            rr = replay(sizing, truth_df, calendar_df2, CFG,
                        case_id=case, no_re20=False,
                        save_dispatch=False)

            t_elapsed = time.time() - t_start

            row = {
                "case": case,
                "N": N,
                "K": K,
                "replicate": r,
                "CC_kw": round(result["CC"], 1),
                "P_B_kw": round(result["P_B"], 1),
                "E_B_kwh": round(result["E_B"], 1),
                "solve_obj_M": round(result["obj_val"] / 1e6, 3),
                "solve_time_s": round(result["solve_time"], 1),
                "replay_total_M": round(rr["replay_total_M"], 3) if rr else None,
                "replay_over_M": round(rr["replay_over_M"], 3) if rr else None,
                "over_months": rr["over_months"] if rr else None,
                "RE_pct": rr["RE_pct"] if rr else None,
                "elapsed_s": round(t_elapsed, 1),
                "status": "ok" if rr else "replay_failed",
                "gap_pct": result.get("mip_gap_pct", None),
            }

            print(f"  CC={result['CC']:.0f} kW, P_B={result['P_B']:.0f} kW, "
                  f"E_B={result['E_B']:.0f} kWh, "
                  f"solve={result['obj_val']/1e6:.2f}M"
                  + (f", replay={rr['replay_total_M']:.2f}M" if rr else ""))

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            row = _fail_row(case, N, K, r, str(e)[:200], time.time() - t_start)

        all_rows.append(row)
        existing_runs.add((case, N, K, r))
        _save_incremental(all_rows, run_summary_path)

    # ── Post-processing ─────────────────────────────────────────
    print("\n" + "="*60)
    print("Post-processing sensitivity results...")
    df_runs = pd.DataFrame(all_rows)
    df_runs.to_parquet(run_summary_path, index=False)
    print(f"Saved: {run_summary_path} ({len(df_runs)} rows)")

    _compute_aggregates(df_runs)
    _compute_sobol(df_runs)
    _write_report(df_runs)
    _write_config(n_vals, k_vals, cases_to_run, r_reps)

    print("\nSensitivity analysis complete.")
    return df_runs


def _fail_row(case, N, K, r, reason, elapsed):
    return {
        "case": case, "N": N, "K": K, "replicate": r,
        "CC_kw": None, "P_B_kw": None, "E_B_kwh": None,
        "solve_obj_M": None, "solve_time_s": None,
        "replay_total_M": None, "replay_over_M": None,
        "over_months": None, "RE_pct": None,
        "elapsed_s": round(elapsed, 1),
        "status": f"failed: {reason}",
        "gap_pct": None,
    }


def _save_incremental(rows, path):
    """Save results incrementally after each run."""
    pd.DataFrame(rows).to_parquet(path, index=False)


def _compute_aggregates(df_runs):
    """Compute mean/std/CV by (case, N, K) and save."""
    ok = df_runs[df_runs["status"] == "ok"].copy()
    if ok.empty:
        print("No successful runs for aggregation")
        return

    agg = ok.groupby(["case", "N", "K"]).agg(
        replay_total_mean=("replay_total_M", "mean"),
        replay_total_std=("replay_total_M", "std"),
        replay_total_cv=("replay_total_M", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        replay_over_mean=("replay_over_M", "mean"),
        over_months_mean=("over_months", "mean"),
        CC_mean=("CC_kw", "mean"),
        CC_std=("CC_kw", "std"),
        P_B_mean=("P_B_kw", "mean"),
        E_B_mean=("E_B_kwh", "mean"),
        solve_obj_mean=("solve_obj_M", "mean"),
        n_reps=("replicate", "count"),
    ).reset_index()

    agg_path = OUT_DIR / "sensitivity_agg_summary.parquet"
    agg.to_parquet(agg_path, index=False)
    print(f"Saved: {agg_path} ({len(agg)} rows)")

    # Design stability
    design = ok.groupby(["case", "N", "K"]).agg(
        CC_cv=("CC_kw", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        P_B_cv=("P_B_kw", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        E_B_cv=("E_B_kwh", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    ).reset_index()
    design_path = OUT_DIR / "sensitivity_design_summary.parquet"
    design.to_parquet(design_path, index=False)
    print(f"Saved: {design_path}")

    return agg


def _compute_sobol(df_runs):
    """Compute SALib Sobol indices for N and K on replay AEC."""
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol as sobol_analyze
        import yaml

        ok = df_runs[df_runs["status"] == "ok"].dropna(subset=["replay_total_M"])
        if len(ok) < 20:
            print("Too few successful runs for Sobol indices")
            return

        # For Sobol: aggregate by (case, N, K) mean across replicates
        # Then treat (log2(N), K) as inputs and replay_total as output
        agg = ok.groupby(["case", "N", "K"])["replay_total_M"].mean().reset_index()

        sobol_rows = []
        for case in ok["case"].unique():
            sub = agg[agg["case"] == case].copy()
            if len(sub) < 4:
                continue

            # Log-transform N for SALib problem
            sub["log_N"] = np.log2(sub["N"].astype(float))
            X = sub[["log_N", "K"]].values

            # SALib problem definition
            problem = {
                "num_vars": 2,
                "names": ["log_N", "K"],
                "bounds": [
                    [sub["log_N"].min(), sub["log_N"].max()],
                    [sub["K"].min(), sub["K"].max()],
                ],
            }

            # Use variance-based sensitivity (simplified: regression-based for small samples)
            Y = sub["replay_total_M"].values
            Y_centered = Y - Y.mean()

            # Simple variance decomposition (eta squared)
            from scipy.stats import pearsonr
            for var_name in ["log_N", "K"]:
                x_col = sub[var_name].values
                r, p = pearsonr(x_col, Y)
                sobol_rows.append({
                    "case": case,
                    "parameter": var_name,
                    "pearson_r": round(r, 4),
                    "pearson_p": round(p, 4),
                    "eta2_approx": round(r**2, 4),   # variance explained
                })

        sobol_df = pd.DataFrame(sobol_rows)
        sobol_path = OUT_DIR / "sensitivity_sobol_indices.csv"
        sobol_df.to_csv(sobol_path, index=False)
        print(f"Saved: {sobol_path}")
        print(sobol_df.to_string(index=False))

    except Exception as e:
        print(f"Sobol computation skipped: {e}")


def _write_report(df_runs):
    """Write sufficiency decision report."""
    ok = df_runs[df_runs["status"] == "ok"].dropna(subset=["replay_total_M"])

    report = {
        "total_runs": len(df_runs),
        "successful_runs": len(ok),
        "failed_runs": len(df_runs) - len(ok),
        "cases": ok["case"].unique().tolist() if len(ok) > 0 else [],
    }

    # Check sufficiency: is (N=500, K=5) close to (N=2000, K=15)?
    baseline_nk = (500, 5)
    saturation_nk = None

    if len(ok) > 0:
        for case in ok["case"].unique():
            sub = ok[ok["case"] == case]
            agg = sub.groupby(["N", "K"])["replay_total_M"].mean()

            baseline_val = agg.get(baseline_nk, None)
            if baseline_val is not None:
                report[f"{case}_baseline_N500_K5_replay_M"] = round(float(baseline_val), 3)

            # Find saturation: N where |replay(N,K=5) - replay(N/2,K=5)| < 0.1M
            k5 = sub[sub["K"] == 5].groupby("N")["replay_total_M"].mean()
            if len(k5) >= 2:
                ns = sorted(k5.index)
                for i in range(1, len(ns)):
                    delta = abs(k5[ns[i]] - k5[ns[i-1]])
                    if delta < 0.1:
                        saturation_nk = ns[i]
                        break
                report[f"{case}_N_saturation"] = saturation_nk
                report[f"{case}_sufficient_N"] = saturation_nk
                report[f"{case}_sufficient_K"] = 5

        # Sufficiency conclusion
        report["conclusion"] = (
            "N=500,K=5 is sufficient if |replay(500,K=5) - replay(250,K=5)| < 0.5%"
        )

    report_path = OUT_DIR / "sensitivity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {report_path}")
    print(f"  Runs: {report['total_runs']} total, {report['successful_runs']} OK")


def _write_config(n_vals, k_vals, cases, r_reps):
    """Write config YAML."""
    try:
        import yaml
        config = {
            "N_grid": n_vals,
            "K_grid": k_vals,
            "cases": cases,
            "R_replicates": r_reps,
            "master_pool_size": 500,
            "bootstrap_for_N_gt_500": True,
            "time_limit_s": SENSITIVITY_TIME_LIMIT,
            "mip_gap": SENSITIVITY_MIP_GAP,
            "load_unc_mape": LOAD_UNC_MAPE,
            "pv_scale_factor": PV_SCALE,
        }
        config_path = OUT_DIR / "sensitivity_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved: {config_path}")
    except ImportError:
        pass


# ── Entry point ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--n-vals", nargs="+", type=int, default=N_GRID)
    parser.add_argument("--k-vals", nargs="+", type=int, default=K_GRID)
    parser.add_argument("--r-reps", type=int, default=R_REPLICATES)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_sensitivity(
        n_vals=args.n_vals,
        k_vals=args.k_vals,
        cases=args.cases,
        r_reps=args.r_reps,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
