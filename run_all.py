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
    """Link real Gaussian copula pipeline scenarios into bridge_outputs.

    Uses the official pipeline outputs:
    - pipeline_outputs/pv_point_forecast_caseyear.parquet (deterministic PV)
    - pipeline_outputs/scenarios_joint_pv_load_reduced_5.parquet (5 k-medoids scenarios)

    These come from the CQR-calibrated 19-quantile Gaussian copula (S6-S8)
    with temporal correlation decay=0.3, 500 raw → 5 k-medoids reduced.
    """
    print("=" * 60)
    print("STAGE 0: Link Forecast Artifacts")
    print("=" * 60)

    pipeline_dir = ROOT / "pipeline_outputs"
    bridge_out = ROOT / "bridge_outputs"
    bridge_out.mkdir(parents=True, exist_ok=True)

    # Check that real pipeline outputs exist
    pv_det_path = pipeline_dir / "pv_point_forecast_caseyear.parquet"
    sc_path = pipeline_dir / "scenarios_joint_pv_load_reduced_5.parquet"

    if not pv_det_path.exists() or not sc_path.exists():
        raise FileNotFoundError(
            f"Pipeline outputs not found. Run the forecast pipeline first.\n"
            f"  Expected: {pv_det_path}\n"
            f"  Expected: {sc_path}"
        )

    # Copy reduced scenarios to bridge_outputs (where bridge_full_year.py expects them)
    sc_df = pd.read_parquet(sc_path)
    sc_df.to_parquet(bridge_out / "scenarios_fullyear_reduced_5.parquet", index=False)

    pv_det = pd.read_parquet(pv_det_path)
    print(f"  Det PV: {len(pv_det)} rows")
    print(f"  Scenarios: {len(sc_df)} rows, {sc_df['scenario_id'].nunique()} scenarios/day")
    print(f"  Probabilities (day 1): {sc_df[sc_df['target_day_local'] == sc_df['target_day_local'].iloc[0]].groupby('scenario_id')['probability_pi'].first().values}")
    print(f"  Source: Gaussian copula (19Q CQR, decay=0.3, 500→5 k-medoids)")

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
