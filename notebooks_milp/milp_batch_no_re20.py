#!/usr/bin/env python3
"""
Batch MILP experiment: No RE20, PV = 515 kW (current school conditions)

Scenarios compared:
  C0   — Det PV + Det Load
  C1   — Prob PV + Det Load
  C2   — Det PV + Pert Load
  C3   — Prob PV + Pert Load
  PI   — Perfect Information (realized PV + realized Load)
  BASE — Baseline (School Status): BESS=0, CC=5000

Changes vs standard runs:
  - RE20 constraint excluded (no_re20=True)
  - All PV data scaled to 515 kW  (PV_SCALE = 515 / 2687)
  - BASE is replay-only with fixed BESS=0, CC=5000

Usage:
    python3 milp_batch_no_re20.py
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

# ── PV scaling factor ──────────────────────────────────────────
PV_INSTALLED_KW = 515.0      # current school installation
PV_MODEL_KW     = 2_687.0    # PV capacity in existing ingest files
PV_SCALE        = PV_INSTALLED_KW / PV_MODEL_KW   # ≈ 0.1917


# ── Helpers ────────────────────────────────────────────────────

def scale_day_data_pv(day_data, scale):
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


def build_pi_ingest(CFG, pv_scale):
    """Build perfect-information ingest scaled to pv_scale."""
    bridge_dir = Path(CFG['bridge_dir'])
    truth = pd.read_parquet(bridge_dir / 'full_year_replay_truth_package.parquet')
    calendar = pd.read_parquet(bridge_dir / 'caseyear_calendar_manifest.parquet')

    rows = []
    for _, row in truth.iterrows():
        cal = calendar[calendar['day_index'] == row['day_index']].iloc[0]
        rows.append({
            'day_index':        int(row['day_index']),
            'calendar_day':     cal['calendar_day'],
            'hour_local':       int(row['hour_local']),
            'scenario_id':      'det',
            'probability_pi':   1.0,
            'pv_mode':          'pv_perfect',
            'load_mode':        'load_perfect',
            'pv_available_kw':  float(row['pv_realized_kw']) * pv_scale,
            'load_kw':          float(row['load_realized_kw']),
            'month_id':         int(row['month_id']),
            'day_type':         row['day_type'],
            'season_tag':       row['season_tag'],
            'is_holiday':       bool(row['is_holiday']),
        })

    df = pd.DataFrame(rows)
    df['calendar_day'] = pd.to_datetime(df['calendar_day'])

    out_path = bridge_dir / 'full_year_milp_ingest_pi_515kw.parquet'
    df.to_parquet(out_path, index=False)
    print(f"  Built PI 515 kW ingest: {len(df)} rows, "
          f"PV max={df['pv_available_kw'].max():.0f} kW")
    return out_path


# ── Main ───────────────────────────────────────────────────────

def run():
    CFG = get_config()
    bridge_dir = Path(CFG['bridge_dir'])
    out_dir    = Path(CFG['output_dir'])

    truth_df_raw, calendar_df = load_truth(CFG)
    truth_df = scale_truth_pv(truth_df_raw, PV_SCALE)

    all_results = []

    # ══════════════════════════════════════════════════════════
    #  C0, C1, C2, C3
    # ══════════════════════════════════════════════════════════
    for case in CASE_TABLE:
        cid = case['case_id']
        print(f"\n{'='*60}")
        print(f"CASE {cid}: {case['label']}")
        print(f"{'='*60}")

        day_data, day_indices, scenario_ids = load_data(CFG, case)
        scale_day_data_pv(day_data, PV_SCALE)

        r = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                            case_id=cid, no_re20=True)
        if r is None:
            print(f"  SKIPPED — no feasible solution for {cid}")
            continue

        sizing = {'CC': r['CC'], 'P_B': r['P_B'], 'E_B': r['E_B']}
        rr = replay(sizing, truth_df, calendar_df, CFG,
                    case_id=cid, no_re20=True)

        all_results.append((cid, r, rr))

    # ══════════════════════════════════════════════════════════
    #  PI — Perfect Information (515 kW PV, realized load)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CASE PI: Perfect Information (realized PV + Load, 515 kW)")
    print(f"{'='*60}")

    pi_ingest_path = build_pi_ingest(CFG, PV_SCALE)
    pi_case = {
        'case_id':    'PI',
        'ingest_file': pi_ingest_path.name,
        'pv_mode':    'perfect',
        'load_mode':  'perfect',
        'label':      'Perfect Information 515 kW',
    }

    day_data_pi, day_idx_pi, sc_ids_pi = load_data(CFG, pi_case)
    # PV already at 515 kW (built into ingest), no extra scaling needed

    r_pi = build_and_solve(day_data_pi, day_idx_pi, sc_ids_pi, CFG,
                           case_id='PI', no_re20=True)
    if r_pi is not None:
        sizing_pi = {'CC': r_pi['CC'], 'P_B': r_pi['P_B'], 'E_B': r_pi['E_B']}
        rr_pi = replay(sizing_pi, truth_df, calendar_df, CFG,
                       case_id='PI', no_re20=True)
        all_results.append(('PI', r_pi, rr_pi))

    # ══════════════════════════════════════════════════════════
    #  BASE — Baseline: BESS=0, CC=5000 (school status quo)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("CASE BASE: Baseline (BESS=0, CC=5000, 515 kW PV)")
    print(f"{'='*60}")

    sizing_base = {'CC': 5000.0, 'P_B': 0.0, 'E_B': 0.0}
    rr_base = replay(sizing_base, truth_df, calendar_df, CFG,
                     case_id='BASE', no_re20=True)

    # Construct a minimal "r" dict for BASE (no optimization was run)
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
    all_results.append(('BASE', r_base, rr_base))

    # ══════════════════════════════════════════════════════════
    #  Summary tables
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("SUMMARY (No RE20, PV=515 kW)")
    print(f"{'='*60}")

    print(f"\n{'Case':6s} {'CC':>8s} {'P_B':>8s} {'E_B':>8s} {'E/P':>6s}")
    for name, r, rr in all_results:
        ep = r['E_B'] / max(r['P_B'], 0.01)
        print(f"{name:6s} {r['CC']:8.0f} {r['P_B']:8.0f} {r['E_B']:8.0f} {ep:6.1f}")

    print(f"\n--- REPLAY COST BREAKDOWN (M NTD) ---")
    hdr = f"{'Case':6s} {'Total':>8s} {'Inv':>7s} {'Ene':>7s} {'Basic':>7s} {'Over':>7s} {'Green':>7s} {'Deg':>7s} {'RE%':>6s} {'OverMo':>7s} {'Worst':>7s}"
    print(hdr)
    for name, r, rr in all_results:
        print(f"{name:6s} {rr['replay_total_M']:8.2f} {rr['replay_inv_M']:7.2f} "
              f"{rr['replay_ene_M']:7.2f} {rr['replay_basic_M']:7.2f} "
              f"{rr['replay_over_M']:7.2f} {rr['replay_green_M']:7.2f} "
              f"{rr['replay_deg_M']:7.2f} {rr['RE_pct']:6.1f} "
              f"{rr['over_months']:7d} {rr['worst_bill_M']:7.2f}")

    # Gap to PI
    pi_entry = next((rr for name, r, rr in all_results if name == 'PI'), None)
    if pi_entry:
        print(f"\n--- GAP TO PI ---")
        for name, r, rr in all_results:
            delta = rr['replay_total_M'] - pi_entry['replay_total_M']
            print(f"  {name:6s}: {rr['replay_total_M']:.2f}M  gap={delta:+.2f}M")

    # ── Save CSV ───────────────────────────────────────────────
    summary = []
    for name, r, rr in all_results:
        row = format_results(name, r['P_B'], r['E_B'], r['CC'],
                             r['obj_val'], r['re_pct'], r['cost_breakdown'],
                             r['solve_time'])
        row['pv_kw'] = int(PV_INSTALLED_KW)
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
        row['no_re20']         = True
        summary.append(row)

    out_csv = out_dir / 'batch_no_re20_515kw_results.csv'
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"\n→ Results saved to {out_csv}")


if __name__ == '__main__':
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / 'gurobi.lic'
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)

    run()
