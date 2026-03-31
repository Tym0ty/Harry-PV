#!/usr/bin/env python3
"""
Full-Year MILP Runner — C0 through C3 with loadunc (spec-compliant).

Runs all 4 cases, saves dispatch parquets, and updates:
  milp_outputs/case_summary_fullyear.csv
  milp_outputs/replay_summary_fullyear.csv

Previous (loadpert) results are preserved in:
  milp_outputs/case_summary_fullyear_loadpert.csv
  milp_outputs/replay_summary_fullyear_loadpert.csv

Usage:
    cd notebooks_milp
    python3 milp_run_fullyear.py
    # or run specific cases:
    python3 milp_run_fullyear.py --cases C0 C1
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))
os.chdir(ROOT / "notebooks_milp")

from milp_common import get_config, CASE_TABLE, load_data, load_truth, format_results
from milp_solver import build_and_solve, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["C0", "C1", "C2", "C3"],
                        help="Cases to run (default: all)")
    parser.add_argument("--no-re20", action="store_true", help="Disable RE20 constraint")
    parser.add_argument("--skip-replay", action="store_true", help="Skip replay step")
    parser.add_argument("--save-dispatch", action="store_true", default=True,
                        help="Save dispatch parquets (default: True)")
    args = parser.parse_args()

    CFG = get_config()
    output_dir = Path(CFG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing CSVs as loadpert-legacy if not already done
    case_csv = output_dir / "case_summary_fullyear.csv"
    replay_csv = output_dir / "replay_summary_fullyear.csv"
    case_csv_bak = output_dir / "case_summary_fullyear_loadpert.csv"
    replay_csv_bak = output_dir / "replay_summary_fullyear_loadpert.csv"

    if case_csv.exists() and not case_csv_bak.exists():
        import shutil
        shutil.copy(case_csv, case_csv_bak)
        print(f"Backed up previous case_summary → {case_csv_bak.name}")
    if replay_csv.exists() and not replay_csv_bak.exists():
        import shutil
        shutil.copy(replay_csv, replay_csv_bak)
        print(f"Backed up previous replay_summary → {replay_csv_bak.name}")

    truth_df, calendar_df = load_truth(CFG)

    # Load existing results if available (to preserve results not being re-run)
    existing_case = {}
    existing_replay = {}
    if case_csv.exists():
        df_c = pd.read_csv(case_csv)
        for _, row in df_c.iterrows():
            existing_case[row['case']] = row.to_dict()
    if replay_csv.exists():
        df_r = pd.read_csv(replay_csv)
        for _, row in df_r.iterrows():
            existing_replay[row['case_id']] = row.to_dict()

    case_rows = dict(existing_case)
    replay_rows = dict(existing_replay)

    for case in CASE_TABLE:
        if case['case_id'] not in args.cases:
            print(f"Skipping {case['case_id']} (not in --cases)")
            continue

        print(f"\n{'='*60}")
        print(f"Running {case['case_id']}: {case['label']}")
        print(f"{'='*60}")

        day_data, day_indices, scenario_ids = load_data(CFG, case)

        # Solve
        result = build_and_solve(day_data, day_indices, scenario_ids, CFG,
                                  case_id=case['case_id'],
                                  no_re20=args.no_re20)
        if result is None:
            print(f"  FAILED: {case['case_id']}")
            continue

        # Format sizing row
        row = format_results(
            case['case_id'],
            result['P_B'], result['E_B'], result['CC'],
            result['obj_val'], result['re_pct'],
            result['cost_breakdown'], result['solve_time']
        )
        case_rows[case['case_id']] = row

        sizing = {
            'CC': result['CC'],
            'P_B': result['P_B'],
            'E_B': result['E_B'],
        }

        if not args.skip_replay:
            print(f"  Running replay for {case['case_id']}...")
            rr = replay(sizing, truth_df, calendar_df, CFG,
                        case_id=case['case_id'],
                        no_re20=args.no_re20,
                        save_dispatch=args.save_dispatch,
                        output_dir=str(output_dir))
            if rr is not None:
                rr['case_id'] = case['case_id']
                rr['solve_obj_M'] = row['total_cost_M']
                rr['gap_pct'] = round(
                    (rr['replay_total_M'] - row['total_cost_M']) / row['total_cost_M'] * 100, 1)
                replay_rows[case['case_id']] = rr

    # Write updated CSVs
    # Case summary
    case_order = ["C0", "C1", "C2", "C3", "PI"]
    case_list = [case_rows[c] for c in case_order if c in case_rows]
    df_case = pd.DataFrame(case_list)
    df_case.to_csv(case_csv, index=False)
    print(f"\nSaved: {case_csv}")

    # Replay summary — flatten monthly_bills before saving
    replay_list = []
    for c in case_order:
        if c not in replay_rows:
            continue
        row = dict(replay_rows[c])
        # Convert monthly_bills dict to JSON string for CSV storage
        if isinstance(row.get('monthly_bills'), dict):
            import json
            row['monthly_bills'] = json.dumps(
                {int(k): round(v, 4) for k, v in row['monthly_bills'].items()})
        replay_list.append(row)

    df_replay = pd.DataFrame(replay_list)
    # Reorder columns for readability
    cols_first = ['case_id', 'replay_total_M', 'replay_ene_M', 'replay_over_M',
                  'replay_green_M', 'replay_deg_M', 'replay_inv_M', 'replay_basic_M',
                  'RE_pct', 'TREC_kWh', 'over_months', 'worst_bill_M',
                  'solve_obj_M', 'gap_pct']
    cols_rest = [c for c in df_replay.columns if c not in cols_first]
    df_replay = df_replay[[c for c in cols_first if c in df_replay.columns] + cols_rest]
    df_replay.to_csv(replay_csv, index=False)
    print(f"Saved: {replay_csv}")

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(df_case[['case', 'bess_p_kw', 'bess_e_kwh', 'contract_kw',
                   'total_cost_M']].to_string(index=False))
    if replay_list:
        print("\nREPLAY SUMMARY")
        print(df_replay[['case_id', 'replay_total_M', 'replay_over_M',
                         'over_months', 'RE_pct']].to_string(index=False))


if __name__ == "__main__":
    main()
