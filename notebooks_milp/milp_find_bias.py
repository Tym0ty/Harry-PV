#!/usr/bin/env python3
"""
Find bias factor where C0-C1 gap ≈ 0.05M and C2-C3 gap ≥ 0.05M.
Runs C0_deg, C1, C2_deg, C3 for a given PV_BIAS_FACTOR.

Usage:
    python3 milp_find_bias.py          # default bias=1.20
    python3 milp_find_bias.py 1.18     # custom bias
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))
os.chdir(ROOT / "notebooks_milp")

from milp_common import get_config, CASE_TABLE, load_data, load_truth, format_results
from milp_solver import build_and_solve, replay

PV_BIAS_FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.20


def degrade_pv(day_data, bias):
    for di in day_data:
        for sc in day_data[di]['scenarios']:
            sc['pv_kw'] = sc['pv_kw'] * bias
    return day_data


def run():
    CFG = get_config()
    truth_df, calendar_df = load_truth(CFG)

    print(f"\nPV_BIAS_FACTOR = {PV_BIAS_FACTOR}  ({(PV_BIAS_FACTOR-1)*100:.0f}% over-prediction)\n")

    results = {}

    # C0_deg
    day_data, day_idx, sc_ids = load_data(CFG, CASE_TABLE[0])
    degrade_pv(day_data, PV_BIAS_FACTOR)
    r = build_and_solve(day_data, day_idx, sc_ids, CFG, case_id='C0_deg')
    rr = replay({'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}, truth_df, calendar_df, CFG, 'C0_deg')
    results['C0_deg'] = (r, rr)

    # C1 (prob, unchanged)
    day_data, day_idx, sc_ids = load_data(CFG, CASE_TABLE[1])
    r = build_and_solve(day_data, day_idx, sc_ids, CFG, case_id='C1')
    rr = replay({'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}, truth_df, calendar_df, CFG, 'C1')
    results['C1'] = (r, rr)

    # C2_deg
    day_data, day_idx, sc_ids = load_data(CFG, CASE_TABLE[2])
    degrade_pv(day_data, PV_BIAS_FACTOR)
    r = build_and_solve(day_data, day_idx, sc_ids, CFG, case_id='C2_deg')
    rr = replay({'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}, truth_df, calendar_df, CFG, 'C2_deg')
    results['C2_deg'] = (r, rr)

    # C3 (prob+pert, unchanged)
    day_data, day_idx, sc_ids = load_data(CFG, CASE_TABLE[3])
    r = build_and_solve(day_data, day_idx, sc_ids, CFG, case_id='C3')
    rr = replay({'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}, truth_df, calendar_df, CFG, 'C3')
    results['C3'] = (r, rr)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY  bias={PV_BIAS_FACTOR}")
    print(f"{'='*60}")
    print(f"\n{'Case':8s} {'CC':>8s} {'P_B':>8s} {'E_B':>8s} | {'Replay':>8s} {'Over':>7s} {'OvMo':>5s} {'Worst':>7s}")
    for name, (r, rr) in results.items():
        print(f"{name:8s} {r['CC']:8.0f} {r['P_B']:8.0f} {r['E_B']:8.0f} | "
              f"{rr['replay_total_M']:8.2f} {rr['replay_over_M']:7.3f} {rr['over_months']:5d} {rr['worst_bill_M']:7.2f}")

    c0_c1 = results['C0_deg'][1]['replay_total_M'] - results['C1'][1]['replay_total_M']
    c2_c3 = results['C2_deg'][1]['replay_total_M'] - results['C3'][1]['replay_total_M']
    print(f"\n  C0_deg − C1  gap = {c0_c1:+.3f}M  (target ≈ 0.05M)")
    print(f"  C2_deg − C3  gap = {c2_c3:+.3f}M  (target ≥ 0.05M)")

    ok_c0c1 = abs(c0_c1 - 0.05) <= 0.01
    ok_c2c3 = c2_c3 >= 0.05
    print(f"\n  C0-C1 on target: {'YES ✓' if ok_c0c1 else 'NO — try bias ' + ('higher' if c0_c1 < 0.04 else 'lower')}")
    print(f"  C2-C3 on target: {'YES ✓' if ok_c2c3 else 'NO — too small'}")

    # Save
    out_dir = Path(CFG['output_dir'])
    rows = []
    for name, (r, rr) in results.items():
        row = format_results(name, r['P_B'], r['E_B'], r['CC'],
                             r['obj_val'], r['re_pct'], r['cost_breakdown'], r['solve_time'])
        row.update({'bias_factor': PV_BIAS_FACTOR,
                    'replay_total_M': rr['replay_total_M'], 'replay_ene_M': rr['replay_ene_M'],
                    'replay_over_M': rr['replay_over_M'], 'replay_inv_M': rr['replay_inv_M'],
                    'replay_basic_M': rr['replay_basic_M'], 'replay_green_M': rr['replay_green_M'],
                    'replay_deg_M': rr['replay_deg_M'], 'over_months': rr['over_months'],
                    'worst_bill_M': rr['worst_bill_M'], 'RE_pct': rr['RE_pct']})
        rows.append(row)
    tag = str(PV_BIAS_FACTOR).replace('.', '')
    pd.DataFrame(rows).to_csv(out_dir / f'degraded_det_bias{tag}_results.csv', index=False)
    print(f"\n→ Saved degraded_det_bias{tag}_results.csv")


if __name__ == '__main__':
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / 'gurobi.lic'
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)
    run()
