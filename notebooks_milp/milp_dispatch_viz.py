"""
Dispatch Comparison: Deterministic (C0) vs Probabilistic (C1)

Solves fixed-sizing dispatch for a few selected days and creates
side-by-side visualizations showing how deterministic vs probabilistic
PV inputs lead to different BESS scheduling strategies.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from milp_common import get_config, CASE_TABLE, load_data, get_tou_price

import gurobipy as gp
from gurobipy import GRB


def solve_dispatch_fixed_sizing(day_data, day_indices, scenario_ids, CFG,
                                 sizing, case_id="C0"):
    """Solve dispatch LP with fixed sizing for a subset of days.

    Returns DataFrame with dispatch variables per (day, scenario, hour).
    """
    n_days = len(day_indices)
    n_scenarios = len(scenario_ids)
    n_hours = 24

    CC = sizing['CC']
    P_B = sizing['P_B']
    E_B = sizing['E_B']

    eta_ch = CFG['eff_charge']
    eta_dis = CFG['eff_discharge']
    soc_min = CFG['soc_min']
    soc_max = CFG['soc_max']
    soc_init = CFG['soc_init']
    kappa = CFG['kappa']

    m = gp.Model(f"Dispatch_{case_id}")
    m.Params.OutputFlag = 0

    # Indices
    keys = [(di, s, t) for di in day_indices
            for s in range(n_scenarios) for t in range(n_hours)]

    # Variables
    P_grid_load = m.addVars(keys, lb=0, name="Pgl")
    P_grid_ch   = m.addVars(keys, lb=0, name="Pgc")
    P_pv_load   = m.addVars(keys, lb=0, name="Ppvl")
    P_pv_ch     = m.addVars(keys, lb=0, name="Ppvc")
    P_pv_curt   = m.addVars(keys, lb=0, name="Pcurt")
    P_ch        = m.addVars(keys, lb=0, ub=P_B, name="Pch")
    P_dis       = m.addVars(keys, lb=0, ub=P_B, name="Pdis")
    E_soc       = m.addVars(keys, lb=soc_min * E_B, ub=soc_max * E_B, name="E")

    # Constraints per (day, scenario, hour)
    for di in day_indices:
        dd = day_data[di]
        for s_idx in range(n_scenarios):
            sc = dd['scenarios'][s_idx]
            pv = sc['pv_kw']
            load = sc['load_kw']

            for t in range(n_hours):
                key = (di, s_idx, t)

                # C1: load balance
                m.addConstr(
                    P_grid_load[key] + P_pv_load[key] + P_dis[key]
                    == load[t],
                    name=f"LB_{di}_{s_idx}_{t}"
                )

                # C2: PV split
                m.addConstr(
                    P_pv_load[key] + P_pv_ch[key] + P_pv_curt[key]
                    == pv[t],
                    name=f"PV_{di}_{s_idx}_{t}"
                )

                # C3: charge balance
                m.addConstr(
                    P_ch[key] == P_grid_ch[key] + P_pv_ch[key],
                    name=f"CH_{di}_{s_idx}_{t}"
                )

                # C4: SOC dynamics
                if t == 0:
                    E_prev = soc_init * E_B
                else:
                    E_prev = E_soc[(di, s_idx, t - 1)]

                m.addConstr(
                    E_soc[key] == E_prev
                    + eta_ch * P_ch[key]
                    - P_dis[key] / eta_dis,
                    name=f"SOC_{di}_{s_idx}_{t}"
                )

    # Objective: minimize energy cost + grid draw (simplified for dispatch)
    obj = gp.LinExpr()
    for di in day_indices:
        dd = day_data[di]
        for s_idx in range(n_scenarios):
            prob = dd['scenarios'][s_idx]['prob']
            for t in range(n_hours):
                key = (di, s_idx, t)
                tou = dd['tou'][t]
                obj += prob * tou * (P_grid_load[key] + P_grid_ch[key])

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        print(f"  Warning: solve status {m.status}")
        return None

    # Extract dispatch
    rows = []
    for di in day_indices:
        dd = day_data[di]
        for s_idx in range(n_scenarios):
            sc = dd['scenarios'][s_idx]
            for t in range(n_hours):
                key = (di, s_idx, t)
                rows.append({
                    'day_index': di,
                    'scenario_idx': s_idx,
                    'scenario_id': sc['scenario_id'],
                    'prob': sc['prob'],
                    'hour': t + 1,
                    'pv_avail': sc['pv_kw'][t],
                    'load': sc['load_kw'][t],
                    'P_grid_load': P_grid_load[key].X,
                    'P_grid_ch': P_grid_ch[key].X,
                    'P_grid_total': P_grid_load[key].X + P_grid_ch[key].X,
                    'P_pv_load': P_pv_load[key].X,
                    'P_pv_ch': P_pv_ch[key].X,
                    'P_pv_curt': P_pv_curt[key].X,
                    'P_ch': P_ch[key].X,
                    'P_dis': P_dis[key].X,
                    'E_soc': E_soc[key].X,
                })
    return pd.DataFrame(rows)


def find_interesting_days(day_data, day_indices, n=3):
    """Find days with high PV variability across scenarios (summer weekdays)."""
    scores = []
    for di in day_indices:
        dd = day_data[di]
        if len(dd['scenarios']) <= 1:
            continue
        # Only summer weekdays for interesting behavior
        if not dd['is_summer'] or dd['day_type'] != 'weekday':
            continue

        # PV variability: std of total daily PV across scenarios
        daily_pvs = [np.sum(sc['pv_kw']) for sc in dd['scenarios']]
        if max(daily_pvs) < 1000:
            continue  # skip very low PV days
        cv = np.std(daily_pvs) / (np.mean(daily_pvs) + 1)
        scores.append((di, cv, np.mean(daily_pvs)))

    # Sort by coefficient of variation (highest spread)
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in scores[:n]]
    print(f"Selected days (high PV variability): {selected}")
    for di, cv, mean_pv in scores[:n]:
        dd = day_data[di]
        print(f"  Day {di} ({dd['calendar_day'].strftime('%Y-%m-%d')} {dd['day_type']}): "
              f"CV={cv:.3f}, mean daily PV={mean_pv:.0f} kWh")
    return selected


def plot_dispatch_comparison(disp_c0, disp_c1, day_data_c0, day_data_c1,
                              selected_days, sizing_c0, sizing_c1, out_dir):
    """Create multi-panel dispatch comparison plots."""

    for di in selected_days:
        d0 = disp_c0[disp_c0['day_index'] == di]
        d1 = disp_c1[disp_c1['day_index'] == di]
        dd0 = day_data_c0[di]
        dd1 = day_data_c1[di]
        date_str = dd0['calendar_day'].strftime('%Y-%m-%d (%a)')

        n_sc_c1 = d1['scenario_idx'].nunique()

        fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True)
        hours = np.arange(1, 25)

        # ── Column 0: C0 (Deterministic) ──────────────────────
        ax = axes[0, 0]
        ax.set_title(f'C0 Deterministic — {date_str}', fontweight='bold', fontsize=11)
        # Single scenario
        s0 = d0[d0['scenario_idx'] == 0].sort_values('hour')
        ax.fill_between(hours, 0, s0['load'].values, alpha=0.3, color='gray', label='Load')
        ax.fill_between(hours, 0, s0['pv_avail'].values, alpha=0.4, color='gold', label='PV Available')
        ax.plot(hours, s0['P_pv_load'].values, color='orange', lw=2, label='PV→Load')
        ax.plot(hours, s0['P_grid_total'].values, color='red', lw=2, label='Grid Draw')
        ax.axhline(y=sizing_c0['CC'], color='crimson', ls='--', lw=1, alpha=0.7, label=f"CC={sizing_c0['CC']:.0f}")
        ax.set_ylabel('Power (kW)')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(bottom=0)

        # ── Column 1: C1 (Probabilistic) ─────────────────────
        ax = axes[0, 1]
        ax.set_title(f'C1 Probabilistic ({n_sc_c1} scenarios) — {date_str}', fontweight='bold', fontsize=11)
        # Plot all scenarios with transparency
        for s_idx in range(n_sc_c1):
            s1 = d1[d1['scenario_idx'] == s_idx].sort_values('hour')
            alpha = 0.15
            ax.fill_between(hours, 0, s1['pv_avail'].values, alpha=alpha, color='gold')
            ax.plot(hours, s1['P_grid_total'].values, color='red', lw=1, alpha=0.4)

        # Overlay expected (probability-weighted) values
        d1_exp = d1.groupby('hour').apply(
            lambda g: pd.Series({
                'load': (g['load'] * g['prob']).sum(),
                'pv_avail': (g['pv_avail'] * g['prob']).sum(),
                'P_pv_load': (g['P_pv_load'] * g['prob']).sum(),
                'P_grid_total': (g['P_grid_total'] * g['prob']).sum(),
            })
        ).reset_index()

        ax.fill_between(hours, 0, d1_exp['load'].values, alpha=0.3, color='gray', label='Load (E)')
        ax.plot(hours, d1_exp['P_pv_load'].values, color='orange', lw=2, label='PV→Load (E)')
        ax.plot(hours, d1_exp['P_grid_total'].values, color='darkred', lw=2.5, label='Grid Draw (E)')
        ax.axhline(y=sizing_c1['CC'], color='crimson', ls='--', lw=1, alpha=0.7, label=f"CC={sizing_c1['CC']:.0f}")
        ax.set_ylabel('Power (kW)')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(bottom=0)

        # ── Row 1: BESS Charge/Discharge ──────────────────────
        # C0
        ax = axes[1, 0]
        s0 = d0[d0['scenario_idx'] == 0].sort_values('hour')
        ax.bar(hours, s0['P_ch'].values, color='steelblue', alpha=0.7, label='Charge')
        ax.bar(hours, -s0['P_dis'].values, color='coral', alpha=0.7, label='Discharge')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.axhline(y=sizing_c0['P_B'], color='steelblue', ls=':', lw=1, alpha=0.5)
        ax.axhline(y=-sizing_c0['P_B'], color='coral', ls=':', lw=1, alpha=0.5)
        ax.set_ylabel('BESS Power (kW)')
        ax.set_title('C0: BESS Charge (+) / Discharge (−)', fontsize=10)
        ax.legend(fontsize=8)

        # C1 — show all scenario traces
        ax = axes[1, 1]
        for s_idx in range(n_sc_c1):
            s1 = d1[d1['scenario_idx'] == s_idx].sort_values('hour')
            ax.plot(hours, s1['P_ch'].values, color='steelblue', alpha=0.3, lw=1)
            ax.plot(hours, -s1['P_dis'].values, color='coral', alpha=0.3, lw=1)

        # Expected
        d1_bess = d1.groupby('hour').apply(
            lambda g: pd.Series({
                'P_ch': (g['P_ch'] * g['prob']).sum(),
                'P_dis': (g['P_dis'] * g['prob']).sum(),
            })
        ).reset_index()
        ax.plot(hours, d1_bess['P_ch'].values, color='steelblue', lw=2.5, label='Charge (E)')
        ax.plot(hours, -d1_bess['P_dis'].values, color='coral', lw=2.5, label='Discharge (E)')
        ax.axhline(y=0, color='black', lw=0.5)
        ax.axhline(y=sizing_c1['P_B'], color='steelblue', ls=':', lw=1, alpha=0.5)
        ax.axhline(y=-sizing_c1['P_B'], color='coral', ls=':', lw=1, alpha=0.5)
        ax.set_ylabel('BESS Power (kW)')
        ax.set_title(f'C1: BESS Charge/Discharge ({n_sc_c1} scenario traces)', fontsize=10)
        ax.legend(fontsize=8)

        # ── Row 2: SOC ────────────────────────────────────────
        # C0
        ax = axes[2, 0]
        s0 = d0[d0['scenario_idx'] == 0].sort_values('hour')
        soc_pct = s0['E_soc'].values / sizing_c0['E_B'] * 100
        ax.plot(hours, soc_pct, color='green', lw=2.5, label='SOC')
        ax.axhline(y=10, color='gray', ls='--', lw=1, alpha=0.5, label='SOC min (10%)')
        ax.axhline(y=90, color='gray', ls='--', lw=1, alpha=0.5, label='SOC max (90%)')
        ax.fill_between(hours, 10, 90, alpha=0.05, color='green')
        ax.set_ylabel('SOC (%)')
        ax.set_xlabel('Hour of Day')
        ax.set_title('C0: Battery State of Charge', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

        # C1 — show all scenario SOC traces
        ax = axes[2, 1]
        for s_idx in range(n_sc_c1):
            s1 = d1[d1['scenario_idx'] == s_idx].sort_values('hour')
            soc_pct = s1['E_soc'].values / sizing_c1['E_B'] * 100
            ax.plot(hours, soc_pct, color='green', alpha=0.3, lw=1)

        # Expected SOC
        d1_soc = d1.groupby('hour').apply(
            lambda g: pd.Series({
                'E_soc': (g['E_soc'] * g['prob']).sum(),
            })
        ).reset_index()
        soc_exp_pct = d1_soc['E_soc'].values / sizing_c1['E_B'] * 100
        ax.plot(hours, soc_exp_pct, color='darkgreen', lw=2.5, label='SOC (E)')
        ax.axhline(y=10, color='gray', ls='--', lw=1, alpha=0.5)
        ax.axhline(y=90, color='gray', ls='--', lw=1, alpha=0.5)
        ax.fill_between(hours, 10, 90, alpha=0.05, color='green')
        ax.set_ylabel('SOC (%)')
        ax.set_xlabel('Hour of Day')
        ax.set_title(f'C1: Battery SOC ({n_sc_c1} scenario traces)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

        fig.suptitle(f'Dispatch Comparison: Deterministic vs Probabilistic\n{date_str}',
                     fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / f'dispatch_compare_day{di}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved dispatch_compare_day{di}.png")


def main():
    CFG = get_config()
    out_dir = Path(CFG['output_dir']) / 'figures_fullyear'
    out_dir.mkdir(exist_ok=True)

    # Load sizing from solved results
    summary = pd.read_csv(Path(CFG['output_dir']) / 'case_summary_fullyear.csv')
    sizing_c0 = {
        'CC': summary[summary['case'] == 'C0']['contract_kw'].values[0],
        'P_B': summary[summary['case'] == 'C0']['bess_p_kw'].values[0],
        'E_B': summary[summary['case'] == 'C0']['bess_e_kwh'].values[0],
    }
    sizing_c1 = {
        'CC': summary[summary['case'] == 'C1']['contract_kw'].values[0],
        'P_B': summary[summary['case'] == 'C1']['bess_p_kw'].values[0],
        'E_B': summary[summary['case'] == 'C1']['bess_e_kwh'].values[0],
    }
    print(f"C0 sizing: CC={sizing_c0['CC']:.0f}, P_B={sizing_c0['P_B']:.0f}, E_B={sizing_c0['E_B']:.0f}")
    print(f"C1 sizing: CC={sizing_c1['CC']:.0f}, P_B={sizing_c1['P_B']:.0f}, E_B={sizing_c1['E_B']:.0f}")

    # Load full data for C0 and C1
    case_c0 = CASE_TABLE[0]  # C0
    case_c1 = CASE_TABLE[1]  # C1
    print("\nLoading C0 data...")
    day_data_c0, day_indices_c0, scen_c0 = load_data(CFG, case_c0)
    print("Loading C1 data...")
    day_data_c1, day_indices_c1, scen_c1 = load_data(CFG, case_c1)

    # Find interesting days (high PV variability in C1)
    print("\nFinding days with high PV scenario variability...")
    selected = find_interesting_days(day_data_c1, day_indices_c1, n=3)

    # Solve dispatch for selected days only
    print("\nSolving C0 dispatch (fixed sizing)...")
    sub_data_c0 = {di: day_data_c0[di] for di in selected}
    disp_c0 = solve_dispatch_fixed_sizing(sub_data_c0, selected, scen_c0, CFG,
                                           sizing_c0, case_id="C0")

    print("Solving C1 dispatch (fixed sizing)...")
    sub_data_c1 = {di: day_data_c1[di] for di in selected}
    disp_c1 = solve_dispatch_fixed_sizing(sub_data_c1, selected, scen_c1, CFG,
                                           sizing_c1, case_id="C1")

    if disp_c0 is None or disp_c1 is None:
        print("ERROR: dispatch solve failed")
        return

    # Generate plots
    print("\nGenerating dispatch comparison plots...")
    plot_dispatch_comparison(disp_c0, disp_c1, day_data_c0, day_data_c1,
                              selected, sizing_c0, sizing_c1, out_dir)

    # Print summary stats
    print("\n" + "=" * 60)
    print("DISPATCH BEHAVIOR COMPARISON")
    print("=" * 60)
    for di in selected:
        dd = day_data_c1[di]
        date_str = dd['calendar_day'].strftime('%Y-%m-%d (%a)')
        print(f"\nDay {di}: {date_str}")

        # C0 stats
        d0 = disp_c0[disp_c0['day_index'] == di]
        s0 = d0[d0['scenario_idx'] == 0]
        peak_grid_c0 = s0['P_grid_total'].max()
        total_grid_c0 = s0['P_grid_total'].sum()
        total_pv_used_c0 = s0['P_pv_load'].sum() + s0['P_pv_ch'].sum()
        total_curt_c0 = s0['P_pv_curt'].sum()

        print(f"  C0 (det): peak_grid={peak_grid_c0:.0f}kW, "
              f"total_grid={total_grid_c0:.0f}kWh, "
              f"PV_used={total_pv_used_c0:.0f}kWh, curt={total_curt_c0:.0f}kWh")

        # C1 stats (expected across scenarios)
        d1 = disp_c1[disp_c1['day_index'] == di]
        # Worst-case peak grid across scenarios
        peak_grids = []
        for s_idx in d1['scenario_idx'].unique():
            s1 = d1[d1['scenario_idx'] == s_idx]
            peak_grids.append(s1['P_grid_total'].max())

        exp_grid = (d1.groupby('hour')
                    .apply(lambda g: (g['P_grid_total'] * g['prob']).sum())
                    .sum())
        exp_pv = (d1.groupby('hour')
                  .apply(lambda g: ((g['P_pv_load'] + g['P_pv_ch']) * g['prob']).sum())
                  .sum())

        print(f"  C1 (prob): worst_peak_grid={max(peak_grids):.0f}kW, "
              f"E[total_grid]={exp_grid:.0f}kWh, "
              f"E[PV_used]={exp_pv:.0f}kWh")
        print(f"     peak_grid range: [{min(peak_grids):.0f}, {max(peak_grids):.0f}]kW")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    main()
