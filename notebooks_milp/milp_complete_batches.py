#!/usr/bin/env python3
"""
Complete batch runs for the 4-table comparison matrix:

  Table 1 (no cheat, RE20, PV=2687, old load):  C0-C3+PI
    → C0, C1, PI already in harry_comparison_results.csv
    → THIS SCRIPT adds: C2, C3

  Table 2 (no cheat, no RE20, PV=515, old load): C0-C3+PI+BASE
    → ALREADY COMPLETE in batch_no_re20_515kw_results.csv

  Table 3 (cheat/bias, RE20, PV=2687, old load): C0_deg-C3+PI
    → C0_deg, C1 already in degraded_det_bias200_results.csv
    → PI from harry_comparison_results.csv (unchanged)
    → C1, C3 same as Table 1 (prob PV not degraded)
    → THIS SCRIPT adds: C2_deg

  Table 4 (cheat/bias, no RE20, PV=515, old load): C0_deg-C3+PI+BASE
    → THIS SCRIPT runs: all cases

Saves:
  milp_outputs/re20_2687_normal_c2c3.csv       (new C2+C3 for Table 1)
  milp_outputs/re20_2687_biased_c2deg.csv       (new C2_deg for Table 3)
  milp_outputs/no_re20_515_biased_full.csv      (full Table 4)

Usage:
    python3 milp_complete_batches.py
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

PV_BIAS_FACTOR = 2.00
PV_515_KW      = 515.0
PV_MODEL_KW    = 2_687.0
PV_SCALE_515   = PV_515_KW / PV_MODEL_KW   # ≈ 0.1917


# ── Helpers ────────────────────────────────────────────────────

def degrade_pv(day_data, bias):
    """Scale up pv_kw in all det scenarios (in-place) to simulate over-prediction."""
    for di in day_data:
        for sc in day_data[di]['scenarios']:
            sc['pv_kw'] = sc['pv_kw'] * bias
    return day_data


def scale_pv(day_data, scale):
    """Scale pv_kw arrays in day_data in-place."""
    for di in day_data:
        for sc in day_data[di]['scenarios']:
            sc['pv_kw'] = sc['pv_kw'] * scale
    return day_data


def scale_truth_pv(truth_df, scale):
    """Return a copy of truth_df with pv_realized_kw scaled."""
    df = truth_df.copy()
    df['pv_realized_kw'] = df['pv_realized_kw'] * scale
    return df


def build_row(name, r, rr, pv_kw=2687, bias=1.0, no_re20=False):
    row = format_results(name, r['P_B'], r['E_B'], r['CC'],
                         r['obj_val'], r['re_pct'], r['cost_breakdown'], r['solve_time'])
    row['pv_kw']         = int(pv_kw)
    row['bias_factor']   = bias
    row['no_re20']       = no_re20
    row.update({
        'replay_total_M':  rr['replay_total_M'],
        'replay_ene_M':    rr['replay_ene_M'],
        'replay_over_M':   rr['replay_over_M'],
        'replay_inv_M':    rr['replay_inv_M'],
        'replay_basic_M':  rr['replay_basic_M'],
        'replay_green_M':  rr['replay_green_M'],
        'replay_deg_M':    rr['replay_deg_M'],
        'over_months':     rr['over_months'],
        'worst_bill_M':    rr['worst_bill_M'],
        'RE_pct':          rr['RE_pct'],
    })
    return row


def build_pi_ingest_515(CFG):
    """Build perfect-information ingest scaled to 515 kW (uses current truth package)."""
    bridge_dir = Path(CFG['bridge_dir'])
    truth = pd.read_parquet(bridge_dir / 'full_year_replay_truth_package.parquet')
    calendar = pd.read_parquet(bridge_dir / 'caseyear_calendar_manifest.parquet')

    rows = []
    for _, row in truth.iterrows():
        cal = calendar[calendar['day_index'] == row['day_index']].iloc[0]
        rows.append({
            'day_index':       int(row['day_index']),
            'calendar_day':    cal['calendar_day'],
            'hour_local':      int(row['hour_local']),
            'scenario_id':     'det',
            'probability_pi':  1.0,
            'pv_mode':         'pv_perfect',
            'load_mode':       'load_perfect',
            'pv_available_kw': float(row['pv_realized_kw']) * PV_SCALE_515,
            'load_kw':         float(row['load_realized_kw']),
            'month_id':        int(row['month_id']),
            'day_type':        row['day_type'],
            'season_tag':      row['season_tag'],
            'is_holiday':      bool(row['is_holiday']),
        })

    df = pd.DataFrame(rows)
    df['calendar_day'] = pd.to_datetime(df['calendar_day'])
    out_path = bridge_dir / 'full_year_milp_ingest_pi_515kw_v2.parquet'
    df.to_parquet(out_path, index=False)
    print(f"  Built PI 515 kW ingest: {len(df)} rows, PV max={df['pv_available_kw'].max():.0f} kW")
    return out_path


# ── Main ───────────────────────────────────────────────────────

def run():
    CFG = get_config()
    out_dir = Path(CFG['output_dir'])
    truth_df, calendar_df = load_truth(CFG)

    # ══════════════════════════════════════════════════════════
    #  TABLE 1 additions: C2 + C3 (normal, RE20, PV=2687, old load)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TABLE 1 ADDITIONS: C2 + C3 (normal, RE20, PV=2687, old load)")
    print("=" * 60)

    t1_rows = []
    for case in [CASE_TABLE[2], CASE_TABLE[3]]:   # C2, C3
        cid = case['case_id']
        print(f"\n{'='*40}")
        print(f"CASE {cid}: {case['label']}")
        print(f"{'='*40}")
        day_data, day_indices, scenario_ids = load_data(CFG, case)
        r = build_and_solve(day_data, day_indices, scenario_ids, CFG, case_id=cid)
        sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
        rr = replay(sizing, truth_df, calendar_df, CFG, case_id=cid)
        t1_rows.append(build_row(cid, r, rr, pv_kw=2687, bias=1.0, no_re20=False))
        print(f"  → {cid}: replay={rr['replay_total_M']:.2f}M, over={rr['replay_over_M']:.2f}M, RE%={rr['RE_pct']:.1f}")

    pd.DataFrame(t1_rows).to_csv(out_dir / 're20_2687_normal_c2c3.csv', index=False)
    print(f"\n→ Saved re20_2687_normal_c2c3.csv")

    # ══════════════════════════════════════════════════════════
    #  TABLE 3 addition: C2_deg (biased det+pert, RE20, PV=2687, old load)
    #  Note: C0_deg from degraded_det_bias200, C1/C3/PI from Table 1 results
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TABLE 3 ADDITION: C2_deg (biased det+pert, RE20, PV=2687, old load)")
    print("=" * 60)

    print(f"\n{'='*40}")
    print("CASE C2_deg: Det PV (×2 bias) + Pert Load")
    print(f"{'='*40}")
    c2_case = CASE_TABLE[2]   # pvdet_loadpert
    day_data_c2, day_idx_c2, sc_ids_c2 = load_data(CFG, c2_case)
    degrade_pv(day_data_c2, PV_BIAS_FACTOR)
    r_c2deg = build_and_solve(day_data_c2, day_idx_c2, sc_ids_c2, CFG, case_id='C2_deg')
    sizing_c2deg = {'CC': r_c2deg['CC'], 'P_B': r_c2deg['P_B'], 'E_B': r_c2deg['E_B']}
    rr_c2deg = replay(sizing_c2deg, truth_df, calendar_df, CFG, case_id='C2_deg')
    c2deg_row = build_row('C2_deg', r_c2deg, rr_c2deg, pv_kw=2687, bias=PV_BIAS_FACTOR, no_re20=False)
    print(f"  → C2_deg: replay={rr_c2deg['replay_total_M']:.2f}M, over={rr_c2deg['replay_over_M']:.2f}M")

    pd.DataFrame([c2deg_row]).to_csv(out_dir / 're20_2687_biased_c2deg.csv', index=False)
    print(f"\n→ Saved re20_2687_biased_c2deg.csv")

    # ══════════════════════════════════════════════════════════
    #  TABLE 4: No RE20, PV=515, old load, biased (all cases)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TABLE 4: No RE20, PV=515 kW, old load, bias=2× (all cases)")
    print("=" * 60)

    truth_515 = scale_truth_pv(truth_df, PV_SCALE_515)
    t4_results = []

    # C0_deg: biased det PV + det load
    print(f"\n{'='*40}")
    print("CASE C0_deg: Det PV (×2 bias) + Det Load, 515 kW, no RE20")
    print(f"{'='*40}")
    day_data, day_indices, scenario_ids = load_data(CFG, CASE_TABLE[0])
    scale_pv(day_data, PV_SCALE_515)
    degrade_pv(day_data, PV_BIAS_FACTOR)
    r = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                        case_id='C0_deg', no_re20=True)
    sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
    rr = replay(sizing, truth_515, calendar_df, CFG, case_id='C0_deg', no_re20=True)
    t4_results.append(('C0_deg', r, rr))
    print(f"  → C0_deg: replay={rr['replay_total_M']:.2f}M, over={rr['replay_over_M']:.2f}M")

    # C1: prob PV (unchanged) + det load
    print(f"\n{'='*40}")
    print("CASE C1: Prob PV + Det Load, 515 kW, no RE20")
    print(f"{'='*40}")
    day_data, day_indices, scenario_ids = load_data(CFG, CASE_TABLE[1])
    scale_pv(day_data, PV_SCALE_515)
    r = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                        case_id='C1', no_re20=True)
    sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
    rr = replay(sizing, truth_515, calendar_df, CFG, case_id='C1', no_re20=True)
    t4_results.append(('C1', r, rr))
    print(f"  → C1: replay={rr['replay_total_M']:.2f}M, over={rr['replay_over_M']:.2f}M")

    # C2_deg: biased det PV + pert load
    print(f"\n{'='*40}")
    print("CASE C2_deg: Det PV (×2 bias) + Pert Load, 515 kW, no RE20")
    print(f"{'='*40}")
    day_data, day_indices, scenario_ids = load_data(CFG, CASE_TABLE[2])
    scale_pv(day_data, PV_SCALE_515)
    degrade_pv(day_data, PV_BIAS_FACTOR)
    r = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                        case_id='C2_deg', no_re20=True)
    sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
    rr = replay(sizing, truth_515, calendar_df, CFG, case_id='C2_deg', no_re20=True)
    t4_results.append(('C2_deg', r, rr))
    print(f"  → C2_deg: replay={rr['replay_total_M']:.2f}M, over={rr['replay_over_M']:.2f}M")

    # C3: prob PV (unchanged) + pert load
    print(f"\n{'='*40}")
    print("CASE C3: Prob PV + Pert Load, 515 kW, no RE20")
    print(f"{'='*40}")
    day_data, day_indices, scenario_ids = load_data(CFG, CASE_TABLE[3])
    scale_pv(day_data, PV_SCALE_515)
    r = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                        case_id='C3', no_re20=True)
    sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
    rr = replay(sizing, truth_515, calendar_df, CFG, case_id='C3', no_re20=True)
    t4_results.append(('C3', r, rr))
    print(f"  → C3: replay={rr['replay_total_M']:.2f}M, over={rr['replay_over_M']:.2f}M")

    # PI: perfect information, 515 kW, no RE20
    print(f"\n{'='*40}")
    print("CASE PI: Perfect Information, 515 kW, no RE20")
    print(f"{'='*40}")
    pi_ingest_path = build_pi_ingest_515(CFG)
    pi_case = {
        'case_id':    'PI',
        'ingest_file': pi_ingest_path.name,
        'pv_mode':    'perfect',
        'load_mode':  'perfect',
        'label':      'Perfect Information 515 kW',
    }
    day_data_pi, day_idx_pi, sc_ids_pi = load_data(CFG, pi_case)
    r_pi = build_and_solve(day_data_pi, day_idx_pi, sc_ids_pi, CFG,
                           case_id='PI', no_re20=True)
    sizing_pi = {'CC': r_pi['CC'], 'P_B': r_pi['P_B'], 'E_B': r_pi['E_B']}
    rr_pi = replay(sizing_pi, truth_515, calendar_df, CFG, case_id='PI', no_re20=True)
    t4_results.append(('PI', r_pi, rr_pi))
    print(f"  → PI: replay={rr_pi['replay_total_M']:.2f}M, over={rr_pi['replay_over_M']:.2f}M")

    # BASE: BESS=0, CC=5000, 515 kW (school status quo)
    print(f"\n{'='*40}")
    print("CASE BASE: BESS=0, CC=5000, 515 kW, no RE20")
    print(f"{'='*40}")
    sizing_base = {'CC': 5000.0, 'P_B': 0.0, 'E_B': 0.0}
    rr_base = replay(sizing_base, truth_515, calendar_df, CFG,
                     case_id='BASE', no_re20=True)
    r_base = {
        'CC': 5000.0, 'P_B': 0.0, 'E_B': 0.0,
        'obj_val': rr_base['replay_total_M'] * 1e6,
        're_pct': rr_base['RE_pct'],
        'cost_breakdown': {
            'AEC_inv':   rr_base['replay_inv_M']   * 1e6,
            'AEC_ene':   rr_base['replay_ene_M']   * 1e6,
            'AEC_basic': rr_base['replay_basic_M'] * 1e6,
            'AEC_over':  rr_base['replay_over_M']  * 1e6,
            'AEC_green': rr_base['replay_green_M'] * 1e6,
            'AEC_deg':   rr_base['replay_deg_M']   * 1e6,
        },
        'solve_time': 0.0,
    }
    t4_results.append(('BASE', r_base, rr_base))
    print(f"  → BASE: replay={rr_base['replay_total_M']:.2f}M")

    # ── Summary ────────────────────────────────────────────────
    pi_total_t4 = rr_pi['replay_total_M']
    print(f"\n{'='*60}")
    print("TABLE 4 SUMMARY (No RE20, PV=515 kW, bias=2×)")
    print(f"{'='*60}")
    print(f"\nSizing:")
    print(f"{'Case':8s} {'CC':>8s} {'P_B':>8s} {'E_B':>8s} {'E/P':>6s}")
    for name, r, rr in t4_results:
        ep = r['E_B'] / max(r['P_B'], 0.01)
        print(f"{name:8s} {r['CC']:8.0f} {r['P_B']:8.0f} {r['E_B']:8.0f} {ep:6.1f}")

    print(f"\nReplay breakdown:")
    print(f"{'Case':8s} {'Total':>8s} {'Ene':>7s} {'Basic':>7s} {'Over':>7s} {'Green':>7s} {'Deg':>7s} {'Inv':>7s} {'RE%':>6s} {'OvMo':>5s} {'Worst':>7s}")
    for name, r, rr in t4_results:
        print(f"{name:8s} {rr['replay_total_M']:8.2f} {rr['replay_ene_M']:7.2f} "
              f"{rr['replay_basic_M']:7.2f} {rr['replay_over_M']:7.2f} "
              f"{rr['replay_green_M']:7.2f} {rr['replay_deg_M']:7.2f} "
              f"{rr['replay_inv_M']:7.2f} {rr['RE_pct']:6.1f} "
              f"{rr['over_months']:5d} {rr['worst_bill_M']:7.2f}")

    print(f"\nGap to PI (total={pi_total_t4:.2f}M):")
    for name, r, rr in t4_results:
        delta = rr['replay_total_M'] - pi_total_t4
        print(f"  {name:8s}: {rr['replay_total_M']:.2f}M  gap={delta:+.2f}M")

    # Save Table 4
    summary_t4 = []
    for name, r, rr in t4_results:
        bias = PV_BIAS_FACTOR if '_deg' in name else 1.0
        summary_t4.append(build_row(name, r, rr, pv_kw=515, bias=bias, no_re20=True))
    pd.DataFrame(summary_t4).to_csv(out_dir / 'no_re20_515_biased_full.csv', index=False)
    print(f"\n→ Saved no_re20_515_biased_full.csv")

    print("\n" + "=" * 60)
    print("ALL BATCH RUNS COMPLETE")
    print("  re20_2687_normal_c2c3.csv   → Table 1 C2+C3")
    print("  re20_2687_biased_c2deg.csv  → Table 3 C2_deg")
    print("  no_re20_515_biased_full.csv → Table 4 all cases")
    print("=" * 60)


if __name__ == '__main__':
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / 'gurobi.lic'
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)

    run()
