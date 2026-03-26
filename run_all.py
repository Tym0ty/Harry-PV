#!/usr/bin/env python3
"""
Run-All Pipeline: Forecast artifacts → Bridge → MILP Solve → Replay → Results.

Generates forecast artifacts from NTUST_Load_PV.csv (actual PV as proxy),
runs bridge layer, then runs MILP C0–C3 with replay on truth data.

Usage:
    python3 run_all.py

Requires: pandas, numpy, pyarrow, scipy, scikit-learn, scikit-learn-extra, gurobipy
Set GRB_LICENSE_FILE env var to your Gurobi license file path.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ──────────────────────────────────────────────────────────────
#  Stage 0: Generate forecast artifacts from NTUST actual data
# ──────────────────────────────────────────────────────────────

def stage0_generate_forecast_artifacts():
    """Generate pv_point_forecast and pv_scenarios from NTUST_Load_PV.csv.

    Uses actual PV as basis for deterministic forecast (with slight smoothing),
    and generates 5 scenarios per day by perturbing the actual PV with
    realistic variability (lognormal noise scaled by hour).
    """
    print("=" * 60)
    print("STAGE 0: Generate Forecast Artifacts")
    print("=" * 60)

    CASE_YEAR_START = pd.Timestamp("2024-11-01")
    CASE_YEAR_END   = pd.Timestamp("2025-10-31")
    PV_REF_KW  = 50.0
    PV_FIXED_KW = 2687.0
    NTUST_PV_KW = 379.0

    # Load NTUST data
    ntust = pd.read_csv(ROOT / "NTUST_Load_PV.csv").dropna(subset=["Date", "Time"])
    ntust["Date"] = pd.to_datetime(ntust["Date"])
    ntust["hour_0"] = ntust["Time"].str[:2].astype(int)
    print(f"  NTUST rows: {len(ntust)}")

    date_range = pd.date_range(CASE_YEAR_START, CASE_YEAR_END, freq="D")

    # Build actual PV dict (already in kWh at NTUST scale ~379kW)
    pv_actual = {}  # (date, hour_0) -> kWh
    for _, row in ntust.iterrows():
        d = row["Date"]
        h0 = row["hour_0"]
        if CASE_YEAR_START <= d <= CASE_YEAR_END + pd.Timedelta(days=1):
            pv_actual[(d, h0)] = max(0.0, float(row["Solar_kWh"]))

    # ── Deterministic PV point forecast (at 50kW reference) ────
    # Use actual PV scaled down from NTUST ~379kW to 50kW ref
    print("  Building deterministic PV point forecast...")
    pv_point_rows = []
    for d in date_range:
        for h0 in range(24):
            # Target time: the hour starting at h0:00
            target_time = d + pd.Timedelta(hours=h0)
            pv_ntust = pv_actual.get((d, h0), 0.0)
            if h0 == 0:
                # Check if midnight belongs to previous day
                pv_ntust = pv_actual.get((d, 0), 0.0)

            # Scale NTUST → 50kW reference
            pv_50kw = pv_ntust * (PV_REF_KW / NTUST_PV_KW)

            pv_point_rows.append({
                "target_day_local": d,
                "target_time_local": target_time,
                "pv_point_kw": pv_50kw,
            })

    pv_point_df = pd.DataFrame(pv_point_rows)
    out_dir = ROOT / "pipeline_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pv_point_df.to_parquet(out_dir / "pv_point_forecast_caseyear.parquet", index=False)
    print(f"  → pv_point_forecast_caseyear.parquet: {len(pv_point_df)} rows")

    # ── Probabilistic PV scenarios (5 per day, at 50kW ref) ────
    print("  Building probabilistic PV scenarios (5/day)...")
    rng = np.random.RandomState(42)
    K = 5
    scenario_rows = []

    for d in date_range:
        # Get actual PV profile for this day (24 hours)
        pv_day = np.array([pv_actual.get((d, h), 0.0) for h in range(24)])
        pv_day_50 = pv_day * (PV_REF_KW / NTUST_PV_KW)

        # Generate K scenarios with correlated perturbations
        # Use multiplicative noise: scenario = actual * exp(noise)
        # Higher variance during midday (when PV is high and uncertain)
        for k in range(K):
            # Day-level shift + hour-level noise
            day_shift = rng.normal(0, 0.15)
            hour_noise = rng.normal(0, 0.10, size=24)

            pv_scenario = np.zeros(24)
            for h in range(24):
                if pv_day_50[h] > 0.1:
                    mult = np.exp(day_shift + hour_noise[h])
                    pv_scenario[h] = max(0.0, pv_day_50[h] * mult)
                else:
                    pv_scenario[h] = 0.0

            for h in range(24):
                target_time = d + pd.Timedelta(hours=h)
                scenario_rows.append({
                    "target_day_local": d,
                    "target_time_local": target_time,
                    "scenario_id": k,
                    "pv_available_kw": pv_scenario[h],
                    "probability_pi": 1.0 / K,
                })

    sc_df = pd.DataFrame(scenario_rows)

    # Save to bridge_outputs (where bridge expects it)
    bridge_out = ROOT / "bridge_outputs"
    bridge_out.mkdir(parents=True, exist_ok=True)
    sc_df.to_parquet(bridge_out / "scenarios_fullyear_reduced_5.parquet", index=False)
    print(f"  → scenarios_fullyear_reduced_5.parquet: {len(sc_df)} rows")

    return True


# ──────────────────────────────────────────────────────────────
#  Stage 1: Bridge Layer
# ──────────────────────────────────────────────────────────────

def stage1_bridge():
    print("\n" + "=" * 60)
    print("STAGE 1: Bridge Layer")
    print("=" * 60)

    sys.path.insert(0, str(ROOT / "notebooks_bridge"))
    from bridge_full_year import run_bridge
    report = run_bridge()
    return report


# ──────────────────────────────────────────────────────────────
#  Stage 2: MILP Solve (C0–C3) + Replay
# ──────────────────────────────────────────────────────────────

def stage2_milp():
    print("\n" + "=" * 60)
    print("STAGE 2: MILP Solve (C0–C3) + Replay")
    print("=" * 60)

    # milp_common uses relative paths like '../bridge_outputs_fullyear'
    os.chdir(ROOT / "notebooks_milp")
    sys.path.insert(0, str(ROOT / "notebooks_milp"))
    from milp_common import get_config, CASE_TABLE, load_data, load_truth, format_results
    from milp_solver import build_and_solve, replay

    CFG = get_config()

    # Solve all 4 cases
    results = []
    for case in CASE_TABLE:
        print(f"\n{'=' * 60}")
        print(f"Case {case['case_id']}: {case['label']}")
        print("=" * 60)
        day_data, day_indices, scenario_ids = load_data(CFG, case)
        r = build_and_solve(day_data, day_indices, scenario_ids, CFG, case['case_id'])
        if r:
            results.append(r)
        else:
            print(f"  FAILED: {case['case_id']}")

    # Summary table
    rows = []
    for r in results:
        rows.append(format_results(
            r['case_id'], r['P_B'], r['E_B'], r['CC'],
            r['obj_val'], r['re_pct'], r['cost_breakdown'], r['solve_time']))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("SOLVE RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save
    Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{CFG['output_dir']}/case_summary_fullyear.csv", index=False)
    with open(f"{CFG['output_dir']}/case_results_fullyear.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Replay all cases with truth data
    print("\n" + "=" * 60)
    print("REPLAY ON TRUTH DATA")
    print("=" * 60)
    truth_df, calendar_df = load_truth(CFG)

    replay_results = []
    for r in results:
        sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
        rr = replay(sizing, truth_df, calendar_df, CFG, r['case_id'])
        if rr:
            rr['solve_obj_M'] = round(r['obj_val'] / 1e6, 2)
            rr['gap_pct'] = round(
                (rr['replay_total_M'] - rr['solve_obj_M']) / rr['solve_obj_M'] * 100, 1)
            replay_results.append(rr)

    replay_df = pd.DataFrame(replay_results)
    print("\n" + "=" * 60)
    print("REPLAY RESULTS")
    print("=" * 60)
    print(replay_df[['case_id', 'solve_obj_M', 'replay_total_M', 'gap_pct',
                      'RE_pct', 'over_months', 'worst_bill_M',
                      'replay_over_M']].to_string(index=False))
    replay_df.to_csv(f"{CFG['output_dir']}/replay_summary_fullyear.csv", index=False)

    # Gap analysis
    print("\n" + "=" * 60)
    print("GAP ANALYSIS: Does probabilistic PV outperform deterministic?")
    print("=" * 60)

    if len(replay_results) >= 2:
        c0 = next(r for r in replay_results if r['case_id'] == 'C0')
        c1 = next(r for r in replay_results if r['case_id'] == 'C1')
        diff = c1['replay_total_M'] - c0['replay_total_M']
        print(f"C0 (det) replay:  {c0['replay_total_M']}M  (over-contract: {c0['replay_over_M']}M, months: {c0['over_months']})")
        print(f"C1 (prob) replay: {c1['replay_total_M']}M  (over-contract: {c1['replay_over_M']}M, months: {c1['over_months']})")
        print(f"Difference: {diff:+.2f}M ({'*** Prob WINS ***' if diff < 0 else 'Det wins'})")

    if len(replay_results) >= 4:
        c2 = next(r for r in replay_results if r['case_id'] == 'C2')
        c3 = next(r for r in replay_results if r['case_id'] == 'C3')
        diff2 = c3['replay_total_M'] - c2['replay_total_M']
        print(f"\nC2 (det+pert) replay:  {c2['replay_total_M']}M  (over-contract: {c2['replay_over_M']}M, months: {c2['over_months']})")
        print(f"C3 (prob+pert) replay: {c3['replay_total_M']}M  (over-contract: {c3['replay_over_M']}M, months: {c3['over_months']})")
        print(f"Difference: {diff2:+.2f}M ({'*** Prob WINS ***' if diff2 < 0 else 'Det wins'})")

    # Full cost breakdown comparison
    print("\n" + "=" * 60)
    print("FULL COST BREAKDOWN (Solve)")
    print("=" * 60)
    for r in rows:
        print(f"\n{r['case']}:")
        print(f"  BESS: {r['bess_p_kw']:.0f} kW / {r['bess_e_kwh']:.0f} kWh (E/P={r['ep_ratio']})")
        print(f"  CC: {r['contract_kw']:.0f} kW")
        print(f"  AEC_inv={r['AEC_inv_M']}M  AEC_ene={r['AEC_ene_M']}M  "
              f"AEC_basic={r['AEC_basic_M']}M  AEC_over={r['AEC_over_M']}M  "
              f"AEC_green={r['AEC_green_M']}M  AEC_deg={r['AEC_deg_M']}M")
        print(f"  Total: {r['total_cost_M']}M  RE: {r['re_pct']}%")

    return results, replay_results


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    # Set Gurobi license if not already set
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / "gurobi.lic"
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)
            print(f"Using Gurobi license: {lic_path}")

    stage0_generate_forecast_artifacts()
    stage1_bridge()
    results, replay_results = stage2_milp()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Total time: {elapsed:.1f}s")
    print("=" * 60)
