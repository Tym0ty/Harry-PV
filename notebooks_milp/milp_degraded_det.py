#!/usr/bin/env python3
"""
Configuration 4: Degraded deterministic forecast — C0 vs C1

Uses old Taipower meter readings (config-2 style: 2,687 kW PV, RE20 ON).
C0 det PV forecast is degraded by a systematic over-prediction bias
(pv_available_kw × PV_BIAS_FACTOR) to simulate a lower-quality point forecast.
C1 probabilistic scenarios are unchanged.

Target: replay(C0) − replay(C1) ≥ 0.3M NTD

Usage:
    python3 milp_degraded_det.py
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import copy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))
os.chdir(ROOT / "notebooks_milp")

from milp_common import get_config, CASE_TABLE, load_data, load_truth, format_results
from milp_solver import build_and_solve, replay

# ── Tuning knob ────────────────────────────────────────────────
# Over-prediction bias: det forecast inflated by this factor.
# C0 will over-estimate solar → size CC too low → over-contract in replay.
PV_BIAS_FACTOR = 2.00   # 100% over-prediction


def degrade_pv(day_data, bias):
    """Scale up pv_kw in all scenarios of day_data (in-place)."""
    for di in day_data:
        for sc in day_data[di]['scenarios']:
            sc['pv_kw'] = sc['pv_kw'] * bias
    return day_data


def run():
    CFG = get_config()
    bridge_dir = Path(CFG['bridge_dir'])
    out_dir    = Path(CFG['output_dir'])

    # Old Taipower truth (USE_OLD_LOAD=True already set in bridge)
    truth_df, calendar_df = load_truth(CFG)

    print(f"\nPV_BIAS_FACTOR = {PV_BIAS_FACTOR}  (C0 det forecast over-predicted by {(PV_BIAS_FACTOR-1)*100:.0f}%)")

    # ══════════════════════════════════════════════════════════
    #  C0 — degraded deterministic forecast
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"C0 (DEGRADED det PV ×{PV_BIAS_FACTOR})")
    print(f"{'='*60}")

    c0_case = CASE_TABLE[0]   # pvdet_loaddet
    day_data_c0, day_idx_c0, sc_ids_c0 = load_data(CFG, c0_case)
    degrade_pv(day_data_c0, PV_BIAS_FACTOR)

    r_c0 = build_and_solve(day_data_c0, day_idx_c0, sc_ids_c0, CFG, case_id='C0_deg')
    sizing_c0 = {'CC': r_c0['CC'], 'P_B': r_c0['P_B'], 'E_B': r_c0['E_B']}
    rr_c0 = replay(sizing_c0, truth_df, calendar_df, CFG, case_id='C0_deg')

    # ══════════════════════════════════════════════════════════
    #  C1 — unchanged probabilistic scenarios
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("C1 (Prob PV + Det Load — unchanged)")
    print(f"{'='*60}")

    c1_case = CASE_TABLE[1]   # pvprob_loaddet
    day_data_c1, day_idx_c1, sc_ids_c1 = load_data(CFG, c1_case)

    # C1 optimizes freely — no bounds tied to degraded C0
    r_c1 = build_and_solve(day_data_c1, day_idx_c1, sc_ids_c1, CFG, case_id='C1')
    sizing_c1 = {'CC': r_c1['CC'], 'P_B': r_c1['P_B'], 'E_B': r_c1['E_B']}
    rr_c1 = replay(sizing_c1, truth_df, calendar_df, CFG, case_id='C1')

    # ══════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SUMMARY — Config 4 (Degraded Det, bias={PV_BIAS_FACTOR})")
    print(f"{'='*60}")

    print(f"\n{'Case':10s} {'CC':>8s} {'P_B':>8s} {'E_B':>8s}")
    for name, r in [('C0_deg', r_c0), ('C1', r_c1)]:
        print(f"{name:10s} {r['CC']:8.0f} {r['P_B']:8.0f} {r['E_B']:8.0f}")

    print(f"\n{'Case':10s} {'Replay(M)':>10s} {'Ene':>7s} {'Basic':>7s} {'Over':>7s} {'Green':>7s} {'Deg':>7s} {'Inv':>7s} {'RE%':>6s} {'OvMo':>5s}")
    for name, r, rr in [('C0_deg', r_c0, rr_c0), ('C1', r_c1, rr_c1)]:
        print(f"{name:10s} {rr['replay_total_M']:10.2f} {rr['replay_ene_M']:7.2f} "
              f"{rr['replay_basic_M']:7.2f} {rr['replay_over_M']:7.2f} "
              f"{rr['replay_green_M']:7.2f} {rr['replay_deg_M']:7.2f} "
              f"{rr['replay_inv_M']:7.2f} {rr['RE_pct']:6.1f} {rr['over_months']:5d}")

    gap = rr_c0['replay_total_M'] - rr_c1['replay_total_M']
    print(f"\n  C0_deg − C1 gap = {gap:+.3f}M NTD  (target: > +0.3M)")

    # Save
    summary = []
    for name, r, rr in [('C0_deg', r_c0, rr_c0), ('C1', r_c1, rr_c1)]:
        row = format_results(name, r['P_B'], r['E_B'], r['CC'],
                             r['obj_val'], r['re_pct'], r['cost_breakdown'], r['solve_time'])
        row['bias_factor'] = PV_BIAS_FACTOR if name == 'C0_deg' else 1.0
        row['replay_total_M']  = rr['replay_total_M']
        row['replay_ene_M']    = rr['replay_ene_M']
        row['replay_over_M']   = rr['replay_over_M']
        row['replay_inv_M']    = rr['replay_inv_M']
        row['replay_basic_M']  = rr['replay_basic_M']
        row['replay_green_M']  = rr['replay_green_M']
        row['replay_deg_M']    = rr['replay_deg_M']
        row['over_months']     = rr['over_months']
        row['worst_bill_M']    = rr['worst_bill_M']
        row['RE_pct']          = rr['RE_pct']
        summary.append(row)

    out_csv = out_dir / f'degraded_det_bias{int(PV_BIAS_FACTOR*100)}_results.csv'
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"\n→ Saved to {out_csv}")


if __name__ == '__main__':
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / 'gurobi.lic'
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)

    run()
