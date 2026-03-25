"""
STO-MILP v10 — Shared config, data loading, case definitions, and results.

Per BH_STO-MILP_Engineering_Spec_Final_v2.3.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json


# ──────────────────────────────────────────────────────────────
#  Spec §6.2: TOU_FixedPeak Baseline Table
# ──────────────────────────────────────────────────────────────

def _is_summer(month, day):
    """Summer period: May 16 – Oct 15 (spec §6 CP_009/CP_010)."""
    if month < 5 or month > 10:
        return False
    if month == 5:
        return day >= 16
    if month == 10:
        return day <= 15
    return True


def _build_tou_table():
    """Build the spec §6.2 TOU_FixedPeak tariff table.

    Returns a dict: (season, day_type, hour_0based) -> price NTD/kWh
    where season = 'summer'|'nonsummer', day_type = 'weekday'|'saturday'|'sunday'.
    """
    tou = {}

    # Summer Mon-Fri
    for h in range(0, 9):
        tou[('summer', 'weekday', h)] = 2.53   # Off-peak
    for h in range(9, 16):
        tou[('summer', 'weekday', h)] = 5.85   # Half-peak
    for h in range(16, 22):
        tou[('summer', 'weekday', h)] = 9.39   # Peak
    for h in range(22, 24):
        tou[('summer', 'weekday', h)] = 5.85   # Half-peak

    # Non-summer Mon-Fri
    for h in range(0, 6):
        tou[('nonsummer', 'weekday', h)] = 2.32   # Off-peak
    for h in range(6, 11):
        tou[('nonsummer', 'weekday', h)] = 5.47   # Half-peak
    for h in range(11, 14):
        tou[('nonsummer', 'weekday', h)] = 2.32   # Off-peak
    for h in range(14, 24):
        tou[('nonsummer', 'weekday', h)] = 5.47   # Half-peak

    # Summer Saturday
    for h in range(0, 9):
        tou[('summer', 'saturday', h)] = 2.53   # Off-peak
    for h in range(9, 24):
        tou[('summer', 'saturday', h)] = 2.60   # Half-peak

    # Non-summer Saturday
    for h in range(0, 6):
        tou[('nonsummer', 'saturday', h)] = 2.32   # Off-peak
    for h in range(6, 11):
        tou[('nonsummer', 'saturday', h)] = 2.41   # Half-peak
    for h in range(11, 14):
        tou[('nonsummer', 'saturday', h)] = 2.32   # Off-peak
    for h in range(14, 24):
        tou[('nonsummer', 'saturday', h)] = 2.41   # Half-peak

    # Summer Sunday/Holiday
    for h in range(24):
        tou[('summer', 'sunday', h)] = 2.53   # Off-peak all day

    # Non-summer Sunday/Holiday
    for h in range(24):
        tou[('nonsummer', 'sunday', h)] = 2.32   # Off-peak all day

    return tou


TOU_TABLE = _build_tou_table()


def get_tou_price(month, day, dow, hour_0based):
    """Get TOU price for a specific timestamp.

    Args:
        month: 1-12
        day: day of month (1-31)
        dow: day of week (0=Mon, 5=Sat, 6=Sun)
        hour_0based: 0-23
    """
    season = 'summer' if _is_summer(month, day) else 'nonsummer'
    if dow < 5:
        day_type = 'weekday'
    elif dow == 5:
        day_type = 'saturday'
    else:
        day_type = 'sunday'
    return TOU_TABLE[(season, day_type, hour_0based)]


# ──────────────────────────────────────────────────────────────
#  Configuration (spec §6 parameters)
# ──────────────────────────────────────────────────────────────

def get_config():
    CFG = dict(
        bridge_dir    = '../bridge_outputs',
        load_csv      = '../NTUST_Load_PV.csv',
        output_dir    = '../milp_outputs',

        # Investment costs
        capex_pv_per_kw           = 40_000,   # PV CAPEX
        capex_bess_power_per_kw   = 11_944,   # C_B_P (spec §6 BESS_006)
        capex_bess_energy_per_kwh = 7_738,    # C_B_E (spec §6 BESS_007)
        fom_bess_rate             = 0.01,     # 1% of energy CAPEX

        # Economic (spec §6 FIN_001/002/003)
        discount_rate  = 0.05,     # r
        lifetime_pv    = 20,
        lifetime_bess  = 15,       # N_B

        # Capacity limits
        pv_max_kw      = 15_000,
        bess_p_max_kw  = 5_000,
        bess_e_max_kwh = 20_000,

        # PV rating for normalization (bridge data is for 50 kW reference system)
        pv_rating_kw   = 50.0,

        # Billing (spec §6 CP_001-CP_010)
        basic_charge_summer    = 223.6,   # c_basic_s  NTD/kW-month
        basic_charge_nonsummer = 166.9,   # c_basic_ns NTD/kW-month
        kappa          = 1.0035,          # CP_006: hourly→15-min demand proxy
        oc_within_10pct_mult = 2.0,       # CP_004: m_over_10
        oc_beyond_10pct_mult = 3.0,       # CP_005: m_over_gt10

        # BESS technical (spec §6 BESS_001-004)
        eff_charge    = 0.95,    # η_ch
        eff_discharge = 0.95,    # η_dis
        soc_min       = 0.10,    # SOC_min
        soc_max       = 0.90,    # SOC_max
        soc_init      = 0.50,
        epsilon_term  = 0.05,    # RE_003: terminal band (fraction of E_B)
        epsilon_g_term = 0.05,   # RE_004: green terminal band (≤ epsilon_term)

        # RE accounting (spec §6 SYS_004, RE_002)
        re_target     = 0.20,    # RE20
        trec_cost_per_kwh = 4.63,  # T-REC top-up cost

        # PWL degradation (spec §6 DEG_001-007, C14)
        # N_cyc=6000, DoD=0.80, C_replace_E = C_B_E = 7738
        # λ_base = C_replace_E / (N_cyc * DoD) = 7738 / (6000*0.80) = 1.612 NTD/kWh
        pwl_deg_b_k = [0.0, 0.1, 0.3, 0.6, 0.8],     # breakpoints (fraction of E_B)
        pwl_deg_mu_k = [0.6, 1.0, 1.6, 2.4],          # convex multipliers
        pwl_deg_lambda_base = 1.612,                    # NTD/kWh
        pwl_deg_lambda_k = [0.97, 1.61, 2.58, 3.87],  # segment slopes NTD/kWh

        # Solver
        time_limit    = 600,
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
#  Master Case Table (spec §12)
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
            peak_mask = scenarios['hour_local'].between(10, 17)
            scenarios.loc[peak_mask, 'load_kw'] *= (1 + pct)

    # Prepare repday data structures with spec TOU
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

        # Source date for TOU day-type determination
        src = pd.to_datetime(row['source_date'])
        dow = src.dayofweek  # 0=Mon, 5=Sat, 6=Sun

        sc_list = []
        for s_id in sorted(sc_day['scenario_id'].unique()):
            sc_s = sc_day[sc_day['scenario_id'] == s_id].sort_values('hour_local')
            sc_list.append({
                'pv_kw': sc_s['pv_available_kw'].values,
                'load_kw': sc_s['load_kw'].values,
                'prob': float(sc_s['probability_pi'].iloc[0]),
            })

        # TOU from spec §6.2 table (using source_date for season + day type)
        hours = sc_day[sc_day['scenario_id'] == sc_day['scenario_id'].min()].sort_values('hour_local')['hour_local'].values
        tou_prices = np.array([
            get_tou_price(src.month, src.day, dow, int(h) % 24) for h in hours
        ])

        # Season flag for basic charge determination
        is_summer = _is_summer(src.month, src.day)

        repday_data[d_idx] = {
            'rid': rid, 'month': month, 'weight': weight,
            'scenarios': sc_list, 'tou': tou_prices,
            'is_summer': is_summer, 'dow': dow,
        }

    # Calendar ordering for Method 1
    calendar_order = None
    if case_flags.get('method1', True):
        cal = calendar_map.copy()
        cal['calendar_day'] = pd.to_datetime(cal['calendar_day'])
        cal = cal.sort_values('calendar_day').reset_index(drop=True)
        rid_to_idx = {rid: i for i, rid in enumerate(repday_ids)}
        cal['d_idx'] = cal['repday_id'].map(rid_to_idx)
        cal['month_id'] = cal['calendar_day'].dt.month
        cal['day_of_month'] = cal['calendar_day'].dt.day
        calendar_order = cal[['calendar_day', 'repday_id', 'd_idx', 'month_id',
                              'day_of_month']].to_dict('records')

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
#  Monthly season helper (for basic charge)
# ──────────────────────────────────────────────────────────────

def get_monthly_basic_charge(month, CFG):
    """Return basic charge rate for a month.

    Summer months: Jun, Jul, Aug, Sep (fully within May 16–Oct 15).
    Non-summer: Nov, Dec, Jan, Feb, Mar, Apr.
    Mixed months (May, Oct): use summer rate conservatively.
    """
    if month in (6, 7, 8, 9):
        return CFG['basic_charge_summer']
    elif month in (5, 10):
        return CFG['basic_charge_summer']  # mixed → summer rate (conservative)
    else:
        return CFG['basic_charge_nonsummer']


# ──────────────────────────────────────────────────────────────
#  Results formatting
# ──────────────────────────────────────────────────────────────

def format_results(case_name, cap_pv, cap_bess_p, cap_bess_e, cap_contract,
                   obj_val, re_pct, trec_cost, solve_time, cost_breakdown, CFG):
    """Format a single case result as a dict."""
    return {
        'case': case_name,
        'pv_kw': round(cap_pv, 1),
        'bess_p_kw': round(cap_bess_p, 1),
        'bess_e_kwh': round(cap_bess_e, 1),
        'ep_ratio': round(cap_bess_e / max(cap_bess_p, 0.01), 1),
        'contract_kw': round(cap_contract, 1),
        'total_cost_M': round(obj_val / 1e6, 2),
        'AEC_inv_M': round(cost_breakdown.get('AEC_inv', 0) / 1e6, 2),
        'AEC_ene_M': round(cost_breakdown.get('AEC_ene', 0) / 1e6, 2),
        'AEC_basic_M': round(cost_breakdown.get('AEC_basic', 0) / 1e6, 2),
        'AEC_over_M': round(cost_breakdown.get('AEC_over', 0) / 1e6, 2),
        'AEC_green_M': round(cost_breakdown.get('AEC_green', 0) / 1e6, 2),
        'AEC_deg_M': round(cost_breakdown.get('AEC_deg', 0) / 1e6, 2),
        'trec_M': round(trec_cost / 1e6, 2),
        're_pct': round(re_pct, 1),
        'solve_s': round(solve_time, 1),
    }
