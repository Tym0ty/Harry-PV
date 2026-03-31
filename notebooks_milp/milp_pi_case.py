#!/usr/bin/env python3
"""
Perfect-Information (PI) MILP case.

Uses full_year_milp_ingest_perfect_info.parquet — single scenario with
realized PV and load (truth known at planning time). Provides PI lower bound.

Appends/updates PI row in:
  milp_outputs/case_summary_fullyear.csv
  milp_outputs/replay_summary_fullyear.csv

Usage:
    cd notebooks_milp
    python3 milp_pi_case.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))
os.chdir(ROOT / "notebooks_milp")

from milp_common import get_config, load_truth, format_results
from milp_solver import build_and_solve, replay


PI_CASE = {
    "case_id": "PI",
    "ingest_file": "full_year_milp_ingest_perfect_info.parquet",
    "pv_mode": "det",
    "load_mode": "det",
    "label": "Perfect Information",
}


def load_pi_data(CFG, case):
    """Load PI ingest package (same schema as regular cases)."""
    from milp_common import get_tou_price
    bridge = Path(CFG['bridge_dir'])
    ingest = pd.read_parquet(bridge / case['ingest_file'])
    calendar = pd.read_parquet(bridge / 'caseyear_calendar_manifest.parquet')

    day_indices = sorted(ingest['day_index'].unique())
    scenario_ids = sorted(ingest['scenario_id'].unique())
    n_hours = 24

    print(f"  Loading PI: {case['label']}")
    print(f"  Days: {len(day_indices)}, Scenarios: {len(scenario_ids)}, Hours: {n_hours}")

    cal_lookup = calendar.set_index('day_index')
    day_data = {}

    for di in day_indices:
        cal = cal_lookup.loc[di]
        d_ingest = ingest[ingest['day_index'] == di]
        cd = pd.Timestamp(cal['calendar_day'])
        dow = cd.weekday()

        scenarios = []
        for sid in scenario_ids:
            s_data = d_ingest[d_ingest['scenario_id'] == sid].sort_values('hour_local')
            if len(s_data) == 0:
                continue
            scenarios.append({
                'pv_kw': s_data['pv_available_kw'].values,
                'load_kw': s_data['load_kw'].values,
                'prob': float(s_data['probability_pi'].iloc[0]),
                'scenario_id': sid,
            })

        tou = np.zeros(n_hours)
        for t in range(n_hours):
            tou[t] = get_tou_price(cd.month, cd.day, dow, t)

        day_data[di] = {
            'calendar_day': cd,
            'month_id': int(cal['month_id']),
            'season_tag': cal['season_tag'],
            'day_type': cal['day_type'],
            'is_holiday': bool(cal['is_holiday']),
            'is_summer': bool(cal['is_summer']),
            'scenarios': scenarios,
            'tou': tou,
            'dow': dow,
        }

    return day_data, day_indices, scenario_ids


def main():
    CFG = get_config()
    output_dir = Path(CFG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    case_csv = output_dir / "case_summary_fullyear.csv"
    replay_csv = output_dir / "replay_summary_fullyear.csv"

    # Load PI ingest
    day_data, day_indices, scenario_ids = load_pi_data(CFG, PI_CASE)

    print(f"\n{'='*60}")
    print(f"Running PI: Perfect Information")
    print(f"{'='*60}")

    result = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                              case_id="PI", no_re20=False)
    if result is None:
        print("FAILED: PI case")
        return

    # Format sizing row
    pi_row = format_results(
        "PI",
        result['P_B'], result['E_B'], result['CC'],
        result['obj_val'], result['re_pct'],
        result['cost_breakdown'], result['solve_time']
    )
    print(f"PI sizing: CC={result['CC']:.0f} kW, P_B={result['P_B']:.0f} kW, "
          f"E_B={result['E_B']:.0f} kWh, total={result['obj_val']/1e6:.2f}M")

    # Replay
    truth_df, calendar_df = load_truth(CFG)
    sizing = {'CC': result['CC'], 'P_B': result['P_B'], 'E_B': result['E_B']}

    print("Running PI replay...")
    rr = replay(sizing, truth_df, calendar_df, CFG,
                case_id="PI", no_re20=False,
                save_dispatch=True, output_dir=str(output_dir))

    if rr is None:
        print("Replay failed for PI")
        return

    rr['case_id'] = "PI"
    rr['solve_obj_M'] = pi_row['total_cost_M']
    rr['gap_pct'] = round(
        (rr['replay_total_M'] - pi_row['total_cost_M']) / pi_row['total_cost_M'] * 100, 1)

    # Update case CSV
    if case_csv.exists():
        df_c = pd.read_csv(case_csv)
        df_c = df_c[df_c['case'] != 'PI']
    else:
        df_c = pd.DataFrame()
    df_c = pd.concat([df_c, pd.DataFrame([pi_row])], ignore_index=True)
    # Reorder: C0, C1, C2, C3, PI
    order = ['C0', 'C1', 'C2', 'C3', 'PI']
    df_c['_order'] = df_c['case'].map({c: i for i, c in enumerate(order)})
    df_c = df_c.sort_values('_order').drop(columns='_order')
    df_c.to_csv(case_csv, index=False)
    print(f"Updated: {case_csv}")

    # Update replay CSV
    if replay_csv.exists():
        df_r = pd.read_csv(replay_csv)
        df_r = df_r[df_r['case_id'] != 'PI']
    else:
        df_r = pd.DataFrame()

    pi_replay_row = dict(rr)
    if isinstance(pi_replay_row.get('monthly_bills'), dict):
        pi_replay_row['monthly_bills'] = json.dumps(
            {int(k): round(v, 4) for k, v in pi_replay_row['monthly_bills'].items()})

    df_r = pd.concat([df_r, pd.DataFrame([pi_replay_row])], ignore_index=True)
    order_map = {c: i for i, c in enumerate(['C0', 'C1', 'C2', 'C3', 'PI'])}
    df_r['_order'] = df_r['case_id'].map(order_map)
    df_r = df_r.sort_values('_order').drop(columns='_order')
    df_r.to_csv(replay_csv, index=False)
    print(f"Updated: {replay_csv}")

    print(f"\nPI Results:")
    print(f"  CC={result['CC']:.0f} kW, P_B={result['P_B']:.0f} kW, E_B={result['E_B']:.0f} kWh")
    print(f"  Solve={result['obj_val']/1e6:.2f}M, Replay={rr['replay_total_M']:.2f}M")
    print(f"  RE={rr['RE_pct']:.1f}%, over_months={rr['over_months']}")


if __name__ == "__main__":
    main()
