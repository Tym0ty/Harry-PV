"""
Shared data loading, config, and results reporting for MILP notebooks.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json


def get_config():
    CFG = dict(
        bridge_dir    = '../bridge_outputs',
        load_csv      = '../NTUST_Load_PV.csv',
        output_dir    = '../milp_outputs',

        capex_pv_per_kw         = 40_000,
        capex_bess_power_per_kw = 8_000,
        capex_bess_energy_per_kwh = 10_000,
        fom_bess_rate           = 0.01,

        discount_rate  = 0.03,
        lifetime_pv    = 20,
        lifetime_bess  = 10,

        pv_max_kw      = 15000,
        bess_p_max_kw  = 5000,
        bess_e_max_kwh = 20000,

        contract_price_per_kw_month = 223.6,
        over_contract_penalty_mult  = 2.0,
        feed_in_tariff = 2.0,

        eff_charge    = 0.95,
        eff_discharge = 0.95,
        soc_min       = 0.10,
        soc_max       = 0.90,
        soc_init      = 0.50,
        bess_degradation_cost = 0.05,

        re_target     = 0.30,
        time_limit    = 600,
    )

    Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)

    def crf(r, n):
        return r * (1 + r)**n / ((1 + r)**n - 1)

    CFG['crf_pv'] = crf(CFG['discount_rate'], CFG['lifetime_pv'])
    CFG['crf_bess'] = crf(CFG['discount_rate'], CFG['lifetime_bess'])
    CFG['penalty_per_kw'] = CFG['contract_price_per_kw_month'] * CFG['over_contract_penalty_mult']

    print(f"CRF PV ({CFG['lifetime_pv']}yr): {CFG['crf_pv']:.4f}")
    print(f"CRF BESS ({CFG['lifetime_bess']}yr): {CFG['crf_bess']:.4f}")
    print(f"Annual contract rate: {CFG['contract_price_per_kw_month'] * 12:.0f} TWD/kW/yr")
    return CFG


def load_data(CFG):
    scenarios = pd.read_parquet(f"{CFG['bridge_dir']}/scenarios_repdays_pv_reduced.parquet")
    meta = pd.read_parquet(f"{CFG['bridge_dir']}/repdays_metadata.parquet")

    print(f"\nScenarios: {scenarios.shape[0]} rows")
    print(f"  {meta.shape[0]} repdays, {scenarios.scenario_id.nunique()} scenarios/day, 24 hours")
    print(f"  Weight sum: {meta.weight.sum()} (should be 365)")
    print(f"  PV range: {scenarios.pv_available_kw.min():.1f} – {scenarios.pv_available_kw.max():.1f} kW")
    print(f"  Load range: {scenarios.load_kw.min():.0f} – {scenarios.load_kw.max():.0f} kW")

    # TOU tariff
    tariff_df = pd.read_csv(CFG['load_csv'])
    tariff_df = tariff_df.dropna(subset=['Date'])
    tariff_df['ts'] = pd.to_datetime(tariff_df['Date'] + ' ' + tariff_df['Time'])
    tariff_df['month'] = tariff_df.ts.dt.month
    tariff_df['hour'] = tariff_df.ts.dt.hour
    tou_lookup = tariff_df.groupby(['month', 'hour'])['Price_NTD_kWh'].first().to_dict()

    def get_tou(month, hour_1based):
        return tou_lookup.get((month, hour_1based), 2.0)

    # Prepare repday data
    repday_ids = meta['repday_id'].tolist()
    n_repdays = len(repday_ids)
    n_scenarios = scenarios.scenario_id.nunique()
    n_hours = 24

    repday_data = {}
    for _, row in meta.iterrows():
        rid = row['repday_id']
        d_idx = repday_ids.index(rid)
        sc_day = scenarios[scenarios.repday_id == rid].sort_values(['scenario_id', 'hour'])
        month = row['month_tag']
        weight = row['weight']

        sc_list = []
        for s_id in range(n_scenarios):
            sc_s = sc_day[sc_day.scenario_id == s_id].sort_values('hour')
            sc_list.append({
                'pv_kw': sc_s['pv_available_kw'].values,
                'load_kw': sc_s['load_kw'].values,
                'prob': float(sc_s['probability_pi'].iloc[0]),
            })

        hours = sc_day[sc_day.scenario_id == 0].sort_values('hour')['hour'].values
        tou_prices = np.array([get_tou(month, int(h)) for h in hours])

        repday_data[d_idx] = {
            'rid': rid, 'month': month, 'weight': weight,
            'scenarios': sc_list, 'tou': tou_prices,
        }

    print(f"\nPrepared {n_repdays} repdays x {n_scenarios} scenarios x {n_hours} hours")
    print(f"Total hourly decision points: {n_repdays * n_scenarios * n_hours:,}")

    return scenarios, meta, repday_data, n_repdays, n_scenarios, n_hours


def print_results(capacities, obj_val, CFG, re_val, load_val, solve_time,
                  repday_data, n_repdays, n_scenarios, n_hours, scenarios):
    pv_opt, bess_p_opt, bess_e_opt, contract_opt = capacities

    print('=' * 60)
    print('  OPTIMAL CAMPUS MICROGRID SIZING')
    print('=' * 60)
    print(f'\n  PV Capacity:        {pv_opt:,.1f} kW')
    print(f'  BESS Power:         {bess_p_opt:,.1f} kW')
    print(f'  BESS Energy:        {bess_e_opt:,.1f} kWh  (E/P = {bess_e_opt/max(bess_p_opt,0.01):.1f}h)')
    print(f'  Contract Demand:    {contract_opt:,.1f} kW')

    inv_pv = pv_opt * CFG['capex_pv_per_kw'] * CFG['crf_pv']
    inv_bess_p = bess_p_opt * CFG['capex_bess_power_per_kw'] * CFG['crf_bess']
    inv_bess_e = bess_e_opt * CFG['capex_bess_energy_per_kwh'] * CFG['crf_bess']
    inv_fom = bess_e_opt * CFG['capex_bess_energy_per_kwh'] * CFG['fom_bess_rate']
    inv_contract = contract_opt * CFG['contract_price_per_kw_month'] * 12
    inv_total = inv_pv + inv_bess_p + inv_bess_e + inv_fom + inv_contract
    opex_total = obj_val - inv_total

    print(f'\n  --- Annual Cost Breakdown ---')
    print(f'  PV annuity:         {inv_pv:>14,.0f} TWD')
    print(f'  BESS power annuity: {inv_bess_p:>14,.0f} TWD')
    print(f'  BESS energy annuity:{inv_bess_e:>14,.0f} TWD')
    print(f'  BESS O&M:          {inv_fom:>14,.0f} TWD')
    print(f'  Contract demand:    {inv_contract:>14,.0f} TWD')
    print(f'  ────────────────────────────────')
    print(f'  Investment subtotal:{inv_total:>14,.0f} TWD')
    print(f'  Operating cost:     {opex_total:>14,.0f} TWD')
    print(f'  ════════════════════════════════')
    print(f'  TOTAL ANNUAL COST:  {obj_val:>14,.0f} TWD')
    print(f'  (= {obj_val/1e6:,.2f} M TWD)')
    print(f'\n  RE share: {re_val/load_val*100:.1f}% (target: {CFG["re_target"]*100:.0f}%)')

    # Baseline
    baseline_grid_cost = 0
    baseline_load_total = 0
    baseline_pv_total = 0
    for d in range(n_repdays):
        dd = repday_data[d]
        w = dd['weight']
        tou = dd['tou']
        for s in range(n_scenarios):
            sc = dd['scenarios'][s]
            pw = sc['prob'] * w
            for t in range(n_hours):
                pv_self = min(sc['pv_kw'][t], sc['load_kw'][t])
                baseline_grid_cost += pw * (sc['load_kw'][t] - pv_self) * tou[t]
                baseline_load_total += pw * sc['load_kw'][t]
                baseline_pv_total += pw * pv_self

    peak_demand = scenarios.load_kw.max()
    baseline_contract = peak_demand * CFG['contract_price_per_kw_month'] * 12
    baseline_total = baseline_grid_cost + baseline_contract
    savings = baseline_total - obj_val

    print(f'\n  --- Baseline (50kW PV, No BESS) ---')
    print(f'  Baseline annual cost: {baseline_total:>12,.0f} TWD ({baseline_total/1e6:,.2f} M)')
    print(f'  MILP annual cost:     {obj_val:>12,.0f} TWD ({obj_val/1e6:,.2f} M)')
    print(f'  Annual savings:       {savings:>12,.0f} TWD ({savings/baseline_total*100:.1f}%)')
    print(f'\n  Solve time: {solve_time:.1f}s')
    print('=' * 60)

    # Export
    results = {
        'status': 'optimal',
        'objective_TWD': obj_val,
        'solve_time_s': solve_time,
        'capacities': {
            'pv_kw': round(pv_opt, 1),
            'bess_power_kw': round(bess_p_opt, 1),
            'bess_energy_kwh': round(bess_e_opt, 1),
            'contract_demand_kw': round(contract_opt, 1),
        },
        'annual_cost_breakdown_TWD': {
            'pv_annuity': round(inv_pv),
            'bess_power_annuity': round(inv_bess_p),
            'bess_energy_annuity': round(inv_bess_e),
            'bess_om': round(inv_fom),
            'contract_demand': round(inv_contract),
            'investment_subtotal': round(inv_total),
            'operating_cost': round(opex_total),
        },
        're_share_pct': round(re_val / load_val * 100, 1),
        'baseline_cost_TWD': round(baseline_total),
        'annual_savings_TWD': round(savings),
    }

    out_path = f"{CFG['output_dir']}/milp_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')
    return results
