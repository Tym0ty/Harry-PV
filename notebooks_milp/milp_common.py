"""
STO-MILP v10 — Shared config, data loading, case definitions, and results.

Per STO_MILP_Engineering_Spec_Final_v10.
Changes from v1: inter-day SOC (Method 1), Green SOC, RE20 + T-REC,
PWL degradation, PV routing (no export), segmented billing, kappa proxy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json


# ──────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────

def get_config():
    CFG = dict(
        bridge_dir    = '../bridge_outputs',
        load_csv      = '../NTUST_Load_PV.csv',
        output_dir    = '../milp_outputs',

        # Investment costs (spec §3.2)
        capex_pv_per_kw           = 40_000,
        capex_bess_power_per_kw   = 11_944,    # v10 change (was 8000)
        capex_bess_energy_per_kwh = 7_738,     # v10 change (was 10000)
        fom_bess_rate             = 0.01,      # 1% of energy CAPEX

        # Economic (spec §3.2)
        discount_rate  = 0.05,    # v10 change (was 0.03)
        lifetime_pv    = 20,
        lifetime_bess  = 15,      # v10 change (was 10)

        # Capacity limits
        pv_max_kw      = 15_000,
        bess_p_max_kw  = 5_000,
        bess_e_max_kwh = 20_000,

        # Billing (spec §3.1 C10-C11)
        contract_price_per_kw_month = 223.6,
        kappa          = 1.0035,   # hourly→15-min demand proxy
        oc_within_10pct_mult = 2.0,   # penalty for within 10% over-contract
        oc_beyond_10pct_mult = 3.0,   # penalty for beyond 10% over-contract

        # BESS technical (spec §3.2)
        eff_charge    = 0.95,
        eff_discharge = 0.95,
        soc_min       = 0.10,
        soc_max       = 0.90,
        soc_init      = 0.50,
        epsilon_term  = 0.05,     # terminal SOC band (fraction of E_B)

        # RE accounting (spec §3.1 C12-C13)
        re_target     = 0.20,     # v10 change (was 0.30)
        trec_cost_per_kwh = 4.63, # T-REC top-up cost

        # PWL degradation breakpoints: (throughput_fraction_of_EB, cost_per_kwh)
        # throughput = total charge+discharge per cycle relative to E_B
        pwl_deg_breakpoints = [
            (0.0, 0.0),
            (500, 0.03),     # first 500 equivalent cycles: 0.03 TWD/kWh
            (1500, 0.05),    # 500-1500 cycles: 0.05 TWD/kWh
            (3000, 0.10),    # 1500-3000: 0.10 TWD/kWh
        ],

        # Solver
        time_limit    = 600,

        # PV rating for normalization
        pv_rating_kw  = 50.0,
    )

    Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)

    def crf(r, n):
        return r * (1 + r)**n / ((1 + r)**n - 1)

    CFG['crf_pv'] = crf(CFG['discount_rate'], CFG['lifetime_pv'])
    CFG['crf_bess'] = crf(CFG['discount_rate'], CFG['lifetime_bess'])

    print(f"CRF PV ({CFG['lifetime_pv']}yr, r={CFG['discount_rate']}): {CFG['crf_pv']:.4f}")
    print(f"CRF BESS ({CFG['lifetime_bess']}yr, r={CFG['discount_rate']}): {CFG['crf_bess']:.4f}")
    return CFG


# ──────────────────────────────────────────────────────────────
#  Master Case Table (spec §4)
# ──────────────────────────────────────────────────────────────

CASE_TABLE = [
    {"name": "M0_I0_R0", "method1": False, "risk_days": False, "prob_pv": False, "uplift": None},
    {"name": "M1_I0_R0", "method1": True,  "risk_days": False, "prob_pv": False, "uplift": None},
    {"name": "M2_I0_R0", "method1": True,  "risk_days": True,  "prob_pv": False, "uplift": None},
    {"name": "M2_I1_R0", "method1": True,  "risk_days": True,  "prob_pv": True,  "uplift": None},
    {"name": "M2_I1_R1_p3", "method1": True, "risk_days": True, "prob_pv": True,
     "uplift": ("all_day", 0.03)},
    {"name": "M2_I1_R1_p5", "method1": True, "risk_days": True, "prob_pv": True,
     "uplift": ("all_day", 0.05)},
    {"name": "M2_I1_R2_p3", "method1": True, "risk_days": True, "prob_pv": True,
     "uplift": ("peak_hour", 0.03)},
    {"name": "M2_I1_R2_p5", "method1": True, "risk_days": True, "prob_pv": True,
     "uplift": ("peak_hour", 0.05)},
]


# ──────────────────────────────────────────────────────────────
#  Data Loading (case-aware)
# ──────────────────────────────────────────────────────────────

def load_data(CFG, case_flags):
    """Load bridge data with case-specific filtering.

    case_flags: dict with keys:
        risk_days (bool): include risk repdays
        prob_pv (bool): if False, collapse to P50 single scenario
        uplift: None or (mode, pct) where mode='all_day'|'peak_hour'
        method1 (bool): whether to prepare inter-day ordering
    """
    scenarios = pd.read_parquet(f"{CFG['bridge_dir']}/scenarios_repdays_pv_reduced.parquet")
    meta = pd.read_parquet(f"{CFG['bridge_dir']}/repdays_metadata.parquet")
    calendar_map = pd.read_parquet(f"{CFG['bridge_dir']}/calendar_map.parquet")

    # Normalize column names (bridge v1 uses 'hour', v7 uses 'hour_local')
    if 'hour' in scenarios.columns and 'hour_local' not in scenarios.columns:
        scenarios = scenarios.rename(columns={'hour': 'hour_local'})

    # Filter repdays based on case
    if not case_flags.get('risk_days', True):
        body_ids = set(meta[meta['repday_type'] == 'body']['repday_id'])
        meta = meta[meta['repday_type'] == 'body'].copy()
        scenarios = scenarios[scenarios['repday_id'].isin(body_ids)].copy()
        calendar_map = calendar_map[calendar_map['repday_id'].isin(body_ids)].copy()
        # Re-normalize weights to sum to 365
        old_sum = meta['weight'].sum()
        meta['weight'] = (meta['weight'] / old_sum * 365).round().astype(int)
        # Fix rounding to exactly 365
        diff = 365 - meta['weight'].sum()
        if diff != 0:
            idx = meta['weight'].idxmax()
            meta.loc[idx, 'weight'] += diff

    # Collapse to deterministic PV (P50) if needed
    if not case_flags.get('prob_pv', True):
        # For each repday, compute probability-weighted mean PV across scenarios
        new_rows = []
        for rid in meta['repday_id']:
            rsc = scenarios[scenarios['repday_id'] == rid]
            hours = sorted(rsc['hour_local'].unique())
            for h in hours:
                h_data = rsc[rsc['hour_local'] == h]
                pv_p50 = np.average(h_data['pv_available_kw'].values,
                                     weights=h_data['probability_pi'].values)
                load_kw = h_data['load_kw'].iloc[0]
                source_date = h_data['source_date'].iloc[0]
                new_rows.append({
                    'repday_id': rid, 'scenario_id': 0,
                    'probability_pi': 1.0, 'hour_local': h,
                    'pv_available_kw': pv_p50, 'load_kw': load_kw,
                    'source_date': source_date,
                })
        scenarios = pd.DataFrame(new_rows)

    # Apply load uplift
    uplift = case_flags.get('uplift')
    if uplift is not None:
        mode, pct = uplift
        if mode == 'all_day':
            scenarios['load_kw'] = scenarios['load_kw'] * (1 + pct)
        elif mode == 'peak_hour':
            # Peak hours: 10-17 (hours 10-17 in 1-based = 10,11,...,17)
            peak_mask = scenarios['hour_local'].between(10, 17)
            scenarios.loc[peak_mask, 'load_kw'] *= (1 + pct)

    # TOU tariff
    tariff_df = pd.read_csv(CFG['load_csv'])
    tariff_df = tariff_df.dropna(subset=['Date'])
    tariff_df['ts'] = pd.to_datetime(tariff_df['Date'] + ' ' + tariff_df['Time'])
    tariff_df['month'] = tariff_df.ts.dt.month
    tariff_df['hour'] = tariff_df.ts.dt.hour
    tou_lookup = tariff_df.groupby(['month', 'hour'])['Price_NTD_kWh'].first().to_dict()

    def get_tou(month, hour_1based):
        return tou_lookup.get((month, hour_1based), 2.0)

    # Prepare repday data structures
    repday_ids = meta['repday_id'].tolist()
    n_repdays = len(repday_ids)
    n_scenarios = scenarios['scenario_id'].nunique()
    n_hours = 24

    repday_data = {}
    for d_idx, (_, row) in enumerate(meta.iterrows()):
        rid = row['repday_id']
        sc_day = scenarios[scenarios['repday_id'] == rid].sort_values(['scenario_id', 'hour_local'])
        month = row['month_tag']
        weight = row['weight']

        sc_list = []
        for s_id in sorted(sc_day['scenario_id'].unique()):
            sc_s = sc_day[sc_day['scenario_id'] == s_id].sort_values('hour_local')
            sc_list.append({
                'pv_kw': sc_s['pv_available_kw'].values,
                'load_kw': sc_s['load_kw'].values,
                'prob': float(sc_s['probability_pi'].iloc[0]),
            })

        hours = sc_day[sc_day['scenario_id'] == sc_day['scenario_id'].min()].sort_values('hour_local')['hour_local'].values
        tou_prices = np.array([get_tou(month, int(h)) for h in hours])

        repday_data[d_idx] = {
            'rid': rid, 'month': month, 'weight': weight,
            'scenarios': sc_list, 'tou': tou_prices,
        }

    # Calendar ordering for Method 1
    calendar_order = None
    if case_flags.get('method1', True):
        cal = calendar_map.copy()
        cal['calendar_day'] = pd.to_datetime(cal['calendar_day'])
        cal = cal.sort_values('calendar_day').reset_index(drop=True)
        # Map repday_id to d_idx
        rid_to_idx = {rid: i for i, rid in enumerate(repday_ids)}
        cal['d_idx'] = cal['repday_id'].map(rid_to_idx)
        cal['month_id'] = cal['calendar_day'].dt.month
        calendar_order = cal[['calendar_day', 'repday_id', 'd_idx', 'month_id']].to_dict('records')

    n_scenarios = max(len(repday_data[d]['scenarios']) for d in repday_data)

    info = {
        'n_repdays': n_repdays, 'n_scenarios': n_scenarios, 'n_hours': n_hours,
        'repday_ids': repday_ids, 'meta': meta,
        'weight_sum': meta['weight'].sum(),
    }

    print(f"  Repdays: {n_repdays} ({meta[meta['repday_type']=='body'].shape[0]} body + "
          f"{meta[meta['repday_type']=='risk'].shape[0] if 'risk' in meta['repday_type'].values else 0} risk)")
    print(f"  Scenarios: {n_scenarios}/repday, {n_hours} hours")
    print(f"  Weight sum: {info['weight_sum']}")
    print(f"  Calendar days: {len(calendar_order) if calendar_order else 'N/A (no Method 1)'}")

    return repday_data, calendar_order, info


# ──────────────────────────────────────────────────────────────
#  Results formatting
# ──────────────────────────────────────────────────────────────

def format_results(case_name, cap_pv, cap_bess_p, cap_bess_e, cap_contract,
                   obj_val, re_pct, trec_cost, solve_time, CFG):
    """Format a single case result as a dict."""
    inv_pv = cap_pv * CFG['capex_pv_per_kw'] * CFG['crf_pv']
    inv_bess_p = cap_bess_p * CFG['capex_bess_power_per_kw'] * CFG['crf_bess']
    inv_bess_e = cap_bess_e * CFG['capex_bess_energy_per_kwh'] * CFG['crf_bess']
    inv_fom = cap_bess_e * CFG['capex_bess_energy_per_kwh'] * CFG['fom_bess_rate']
    inv_contract = cap_contract * CFG['contract_price_per_kw_month'] * 12
    inv_total = inv_pv + inv_bess_p + inv_bess_e + inv_fom + inv_contract
    opex = obj_val - inv_total

    return {
        'case': case_name,
        'pv_kw': round(cap_pv, 1),
        'bess_p_kw': round(cap_bess_p, 1),
        'bess_e_kwh': round(cap_bess_e, 1),
        'ep_ratio': round(cap_bess_e / max(cap_bess_p, 0.01), 1),
        'contract_kw': round(cap_contract, 1),
        'total_cost_M': round(obj_val / 1e6, 2),
        'invest_M': round(inv_total / 1e6, 2),
        'opex_M': round(opex / 1e6, 2),
        'trec_M': round(trec_cost / 1e6, 2),
        're_pct': round(re_pct, 1),
        'solve_s': round(solve_time, 1),
    }
