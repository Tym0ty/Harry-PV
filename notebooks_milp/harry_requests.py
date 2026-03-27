#!/usr/bin/env python3
"""
Harry's three requests:
1. Try net load (Load_kWh - Solar_kWh) as billing load
2. Run perfect information case (real PV + real load)
3. Compare all cases with cost component breakdown

Usage: python3 harry_requests.py
"""

import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))
os.chdir(ROOT / "notebooks_milp")

from milp_common import get_config, CASE_TABLE, load_data, load_truth, format_results
from milp_solver import build_and_solve, replay


def build_perfect_info_ingest(CFG):
    """Build a perfect information ingest (real PV + real load).

    This is the "oracle" case: the optimizer knows the actual realized
    PV and load for every hour of the year. This provides the theoretical
    lower bound on cost — no forecast can do better than this.
    """
    bridge_dir = Path(CFG['bridge_dir'])
    truth = pd.read_parquet(bridge_dir / 'full_year_replay_truth_package.parquet')
    calendar = pd.read_parquet(bridge_dir / 'caseyear_calendar_manifest.parquet')

    rows = []
    for _, row in truth.iterrows():
        cal = calendar[calendar['day_index'] == row['day_index']].iloc[0]
        rows.append({
            'day_index': int(row['day_index']),
            'calendar_day': cal['calendar_day'],
            'hour_local': int(row['hour_local']),
            'scenario_id': 'det',
            'probability_pi': 1.0,
            'pv_mode': 'pv_perfect',
            'load_mode': 'load_perfect',
            'pv_available_kw': float(row['pv_realized_kw']),
            'load_kw': float(row['load_realized_kw']),
            'month_id': int(row['month_id']),
            'day_type': row['day_type'],
            'season_tag': row['season_tag'],
            'is_holiday': bool(row['is_holiday']),
        })

    df = pd.DataFrame(rows)
    df['calendar_day'] = pd.to_datetime(df['calendar_day'])

    # Save
    out_path = bridge_dir / 'full_year_milp_ingest_perfect_info.parquet'
    df.to_parquet(out_path, index=False)
    print(f"  Built perfect info ingest: {len(df)} rows, "
          f"PV range: 0-{df['pv_available_kw'].max():.0f} kW, "
          f"Load range: {df['load_kw'].min():.0f}-{df['load_kw'].max():.0f} kW")
    return out_path


def build_net_load_ingest(CFG):
    """Build ingest packages using net load (Load - Solar) instead of gross load.

    This represents the Taipower meter reading: what the grid actually sees
    after existing rooftop PV offsets consumption.
    """
    ntust = pd.read_csv(ROOT / 'NTUST_Load_PV.csv').dropna(subset=['Date', 'Time'])
    ntust['Date'] = pd.to_datetime(ntust['Date'])
    ntust['hour_0'] = ntust['Time'].str[:2].astype(int)

    bridge_dir = Path(CFG['bridge_dir'])
    calendar = pd.read_parquet(bridge_dir / 'caseyear_calendar_manifest.parquet')
    calendar['calendar_day'] = pd.to_datetime(calendar['calendar_day'])
    date_range = sorted(calendar['calendar_day'].unique())

    day_to_idx = {pd.Timestamp(r['calendar_day']): r['day_index']
                  for _, r in calendar.iterrows()}

    # Build net load dict
    net_load = {}  # (date, hour_local) -> net_load_kw
    pv_realized = {}
    PV_SCALE = 2687.0 / 379.0

    for _, row in ntust.iterrows():
        d = row['Date']
        h0 = row['hour_0']
        if h0 == 0:
            d_assign = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            d_assign = d
            h_local = h0

        if d_assign in day_to_idx:
            load_kw = float(row['Load_kWh'])
            solar_kw = float(row['Solar_kWh'])
            net_load[(d_assign, h_local)] = max(0, load_kw - solar_kw)
            pv_realized[(d_assign, h_local)] = solar_kw * PV_SCALE

    # Build C0-net (deterministic PV + net load)
    # Read existing det PV ingest to get pv_available_kw
    det_ingest = pd.read_parquet(bridge_dir / 'full_year_milp_ingest_pvdet_loaddet.parquet')

    rows = []
    for _, row in det_ingest.iterrows():
        d = pd.Timestamp(row['calendar_day'])
        h = int(row['hour_local'])
        nl = net_load.get((d, h), 0.0)
        rows.append({
            'day_index': int(row['day_index']),
            'calendar_day': row['calendar_day'],
            'hour_local': h,
            'scenario_id': 'det',
            'probability_pi': 1.0,
            'pv_mode': 'pv_det',
            'load_mode': 'load_net',
            'pv_available_kw': float(row['pv_available_kw']),
            'load_kw': nl,
            'month_id': int(row['month_id']),
            'day_type': row['day_type'],
            'season_tag': row['season_tag'],
            'is_holiday': bool(row['is_holiday']),
        })

    df = pd.DataFrame(rows)
    df['calendar_day'] = pd.to_datetime(df['calendar_day'])

    out_path = bridge_dir / 'full_year_milp_ingest_pvdet_loadnet.parquet'
    df.to_parquet(out_path, index=False)
    print(f"  Built net-load det ingest: {len(df)} rows, "
          f"Load range: {df['load_kw'].min():.0f}-{df['load_kw'].max():.0f} kW")

    # Also build truth replay with net load
    truth = pd.read_parquet(bridge_dir / 'full_year_replay_truth_package.parquet')
    truth_net = truth.copy()
    for i, row in truth_net.iterrows():
        d = pd.Timestamp(row['calendar_day'])
        h = int(row['hour_local'])
        truth_net.at[i, 'load_realized_kw'] = net_load.get((d, h), 0.0)
    truth_net.to_parquet(bridge_dir / 'full_year_replay_truth_netload.parquet', index=False)
    print(f"  Built net-load truth: {len(truth_net)} rows")

    return out_path


def run():
    CFG = get_config()
    bridge_dir = Path(CFG['bridge_dir'])

    # ══════════════════════════════════════════════════════════════
    #  Request 1: Net load (Load - Solar)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("REQUEST 1: Net Load (Load − Solar) as billing load")
    print("=" * 60)
    build_net_load_ingest(CFG)

    case_netload = {
        'case_id': 'C0_net',
        'ingest_file': 'full_year_milp_ingest_pvdet_loadnet.parquet',
        'pv_mode': 'det', 'load_mode': 'net',
        'label': 'Det PV + Net Load (Load−Solar)',
    }

    day_data, day_indices, scenario_ids = load_data(CFG, case_netload)
    r_net = build_and_solve(day_data, day_indices, scenario_ids, CFG, 'C0_net')

    # Replay with net load truth
    truth_net = pd.read_parquet(bridge_dir / 'full_year_replay_truth_netload.parquet')
    calendar = pd.read_parquet(bridge_dir / 'caseyear_calendar_manifest.parquet')
    sizing_net = {'CC': r_net['CC'], 'P_B': r_net['P_B'], 'E_B': r_net['E_B']}
    rr_net = replay(sizing_net, truth_net, calendar, CFG, 'C0_net')

    # ══════════════════════════════════════════════════════════════
    #  Request 2: Perfect Information case
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("REQUEST 2: Perfect Information (real PV + real Load)")
    print("=" * 60)
    build_perfect_info_ingest(CFG)

    case_pi = {
        'case_id': 'C_PI',
        'ingest_file': 'full_year_milp_ingest_perfect_info.parquet',
        'pv_mode': 'det', 'load_mode': 'det',
        'label': 'Perfect Information (realized PV + Load)',
    }

    day_data_pi, day_idx_pi, sc_ids_pi = load_data(CFG, case_pi)
    r_pi = build_and_solve(day_data_pi, day_idx_pi, sc_ids_pi, CFG, 'C_PI')

    # Replay PI (should be identical to solve since same data)
    truth_df, calendar_df = load_truth(CFG)
    sizing_pi = {'CC': r_pi['CC'], 'P_B': r_pi['P_B'], 'E_B': r_pi['E_B']}
    rr_pi = replay(sizing_pi, truth_df, calendar_df, CFG, 'C_PI')

    # ══════════════════════════════════════════════════════════════
    #  Request 3: Re-run C0 and C1 for comparison
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("REQUEST 3: Re-run C0 + C1 for comparison")
    print("=" * 60)

    # C0
    c0_case = CASE_TABLE[0]
    day_data_c0, day_idx_c0, sc_ids_c0 = load_data(CFG, c0_case)
    r_c0 = build_and_solve(day_data_c0, day_idx_c0, sc_ids_c0, CFG, 'C0')
    sizing_c0 = {'CC': r_c0['CC'], 'P_B': r_c0['P_B'], 'E_B': r_c0['E_B']}
    rr_c0 = replay(sizing_c0, truth_df, calendar_df, CFG, 'C0')

    # C1 with sizing bounds
    c1_case = CASE_TABLE[1]
    day_data_c1, day_idx_c1, sc_ids_c1 = load_data(CFG, c1_case)
    CC_SCALE, PB_SCALE = 1.015, 0.97
    r_c1 = build_and_solve(day_data_c1, day_idx_c1, sc_ids_c1, CFG, 'C1',
                           cc_lb=r_c0['CC'] * CC_SCALE,
                           pb_ub=r_c0['P_B'] * PB_SCALE,
                           eb_ub=r_c0['E_B'], eb_lb=r_c0['E_B'],
                           mip_gap=1e-4)
    sizing_c1 = {'CC': r_c1['CC'], 'P_B': r_c1['P_B'], 'E_B': r_c1['E_B']}
    rr_c1 = replay(sizing_c1, truth_df, calendar_df, CFG, 'C1')

    # ══════════════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FULL COMPARISON TABLE")
    print("=" * 60)

    all_results = [
        ('C_PI', r_pi, rr_pi),
        ('C0', r_c0, rr_c0),
        ('C1', r_c1, rr_c1),
        ('C0_net', r_net, rr_net),
    ]

    # Sizing comparison
    print("\n--- SIZING ---")
    print(f"{'Case':10s} {'CC':>8s} {'P_B':>8s} {'E_B':>8s} {'E/P':>6s}")
    for name, r, rr in all_results:
        ep = r['E_B'] / max(r['P_B'], 0.01)
        print(f"{name:10s} {r['CC']:8.0f} {r['P_B']:8.0f} {r['E_B']:8.0f} {ep:6.1f}")

    # Solve cost breakdown
    print("\n--- SOLVE COST BREAKDOWN (M NTD) ---")
    print(f"{'Case':10s} {'Total':>8s} {'Inv':>7s} {'Ene':>7s} {'Basic':>7s} {'Over':>7s} {'Green':>7s} {'Deg':>7s} {'RE%':>6s}")
    for name, r, rr in all_results:
        cb = r['cost_breakdown']
        total = r['obj_val'] / 1e6
        print(f"{name:10s} {total:8.2f} {cb.get('AEC_inv',0)/1e6:7.2f} {cb.get('AEC_ene',0)/1e6:7.2f} "
              f"{cb.get('AEC_basic',0)/1e6:7.2f} {cb.get('AEC_over',0)/1e6:7.2f} "
              f"{cb.get('AEC_green',0)/1e6:7.2f} {cb.get('AEC_deg',0)/1e6:7.2f} {r['re_pct']:6.1f}")

    # Replay cost breakdown
    print("\n--- REPLAY COST BREAKDOWN (M NTD) ---")
    print(f"{'Case':10s} {'Total':>8s} {'Inv':>7s} {'Ene':>7s} {'Basic':>7s} {'Over':>7s} {'Green':>7s} {'Deg':>7s} {'RE%':>6s} {'Gap%':>7s} {'OverMo':>7s} {'Worst':>7s}")
    for name, r, rr in all_results:
        gap = (rr['replay_total_M'] - r['obj_val']/1e6) / (r['obj_val']/1e6) * 100
        print(f"{name:10s} {rr['replay_total_M']:8.2f} {rr['replay_inv_M']:7.2f} {rr['replay_ene_M']:7.2f} "
              f"{rr['replay_basic_M']:7.2f} {rr['replay_over_M']:7.2f} "
              f"{rr['replay_green_M']:7.2f} {rr['replay_deg_M']:7.2f} {rr['RE_pct']:6.1f} "
              f"{gap:7.1f} {rr['over_months']:7d} {rr['worst_bill_M']:7.2f}")

    # Gap analysis
    print("\n--- ATTRIBUTION vs C_PI (PERFECT INFO) ---")
    pi_total = rr_pi['replay_total_M']
    for name, r, rr in all_results:
        delta = rr['replay_total_M'] - pi_total
        print(f"  {name:10s}: replay={rr['replay_total_M']:.2f}M, gap to PI = {delta:+.2f}M")

    # C0 vs C1 detailed
    print("\n--- C0 vs C1 DETAILED ---")
    for comp in ['replay_total_M', 'replay_inv_M', 'replay_ene_M', 'replay_basic_M',
                 'replay_over_M', 'replay_green_M', 'replay_deg_M']:
        v0, v1 = rr_c0[comp], rr_c1[comp]
        diff = v1 - v0
        pct = diff / v0 * 100 if v0 != 0 else 0
        print(f"  {comp:20s}: C0={v0:.3f}  C1={v1:.3f}  Δ={diff:+.3f} ({pct:+.1f}%)")

    # Save results
    out_dir = Path(CFG['output_dir'])
    summary = []
    for name, r, rr in all_results:
        row = format_results(name, r['P_B'], r['E_B'], r['CC'],
                             r['obj_val'], r['re_pct'], r['cost_breakdown'], r['solve_time'])
        row['replay_total_M'] = rr['replay_total_M']
        row['replay_over_M'] = rr['replay_over_M']
        row['replay_ene_M'] = rr['replay_ene_M']
        row['replay_inv_M'] = rr['replay_inv_M']
        row['replay_basic_M'] = rr['replay_basic_M']
        row['replay_green_M'] = rr['replay_green_M']
        row['replay_deg_M'] = rr['replay_deg_M']
        row['over_months'] = rr['over_months']
        row['worst_bill_M'] = rr['worst_bill_M']
        row['RE_pct'] = rr['RE_pct']
        gap = (rr['replay_total_M'] - r['obj_val']/1e6) / (r['obj_val']/1e6) * 100
        row['gap_pct'] = round(gap, 1)
        summary.append(row)

    pd.DataFrame(summary).to_csv(out_dir / 'harry_comparison_results.csv', index=False)
    print(f"\n→ Saved to {out_dir / 'harry_comparison_results.csv'}")


if __name__ == '__main__':
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / 'gurobi.lic'
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)

    run()
