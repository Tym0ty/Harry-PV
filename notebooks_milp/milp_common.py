"""
Full-Year Direct Solve MILP — Shared config, data loading, TOU, results.

Per FF0326Harry_MILP_Engineering_Spec_FullYear_Formal_vfinal.
Cases: C0 (det PV + det load), C1 (prob PV + det load),
       C2 (det PV + pert load), C3 (prob PV + pert load).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json


# ──────────────────────────────────────────────────────────────
#  TOU_FixedPeak Baseline Table (spec §5.1)
# ──────────────────────────────────────────────────────────────

def _is_summer(month, day):
    """Summer: May 16 – Oct 15."""
    if month < 5 or month > 10:
        return False
    if month == 5:
        return day >= 16
    if month == 10:
        return day <= 15
    return True


def _build_tou_table():
    """Build TOU_FixedPeak tariff: (season, day_type, hour_0based) -> NTD/kWh."""
    tou = {}
    # Summer Mon-Fri
    for h in range(0, 9):   tou[('summer', 'weekday', h)] = 2.53
    for h in range(9, 16):  tou[('summer', 'weekday', h)] = 5.85
    for h in range(16, 22): tou[('summer', 'weekday', h)] = 9.39
    for h in range(22, 24): tou[('summer', 'weekday', h)] = 5.85
    # Non-summer Mon-Fri
    for h in range(0, 6):   tou[('nonsummer', 'weekday', h)] = 2.32
    for h in range(6, 11):  tou[('nonsummer', 'weekday', h)] = 5.47
    for h in range(11, 14): tou[('nonsummer', 'weekday', h)] = 2.32
    for h in range(14, 24): tou[('nonsummer', 'weekday', h)] = 5.47
    # Summer Saturday
    for h in range(0, 9):   tou[('summer', 'saturday', h)] = 2.53
    for h in range(9, 24):  tou[('summer', 'saturday', h)] = 2.60
    # Non-summer Saturday
    for h in range(0, 6):   tou[('nonsummer', 'saturday', h)] = 2.32
    for h in range(6, 11):  tou[('nonsummer', 'saturday', h)] = 2.41
    for h in range(11, 14): tou[('nonsummer', 'saturday', h)] = 2.32
    for h in range(14, 24): tou[('nonsummer', 'saturday', h)] = 2.41
    # Summer Sunday/Holiday
    for h in range(24):     tou[('summer', 'sunday', h)] = 2.53
    # Non-summer Sunday/Holiday
    for h in range(24):     tou[('nonsummer', 'sunday', h)] = 2.32
    return tou

TOU_TABLE = _build_tou_table()


def get_tou_price(month, day, dow, hour_0based):
    """TOU price lookup. dow: 0=Mon..6=Sun. hour_0based: 0-23."""
    season = 'summer' if _is_summer(month, day) else 'nonsummer'
    if dow < 5:
        day_type = 'weekday'
    elif dow == 5:
        day_type = 'saturday'
    else:
        day_type = 'sunday'
    return TOU_TABLE[(season, day_type, hour_0based)]


# ──────────────────────────────────────────────────────────────
#  Configuration (spec §5 frozen baseline)
# ──────────────────────────────────────────────────────────────

def get_config():
    CFG = dict(
        bridge_dir    = '../bridge_outputs_fullyear',
        output_dir    = '../milp_outputs',

        # PV (fixed — not a decision variable, spec PV_001)
        pv_fixed_kw   = 2_687,

        # BESS investment (spec BESS_006/007)
        capex_bess_power_per_kw   = 11_944,   # C_B_P
        capex_bess_energy_per_kwh = 7_738,     # C_B_E

        # Economic (spec FIN_001/002/003)
        discount_rate  = 0.05,
        lifetime_bess  = 15,       # N_B

        # Capacity limits
        bess_p_max_kw  = 5_000,
        bess_e_max_kwh = 20_000,

        # Billing (spec CP_001-CP_010)
        basic_charge_summer    = 223.6,
        basic_charge_nonsummer = 166.9,
        kappa          = 1.0035,
        oc_within_10pct_mult = 2.0,
        oc_beyond_10pct_mult = 3.0,

        # BESS technical (spec BESS_001-005)
        eff_charge    = 0.95,
        eff_discharge = 0.95,
        soc_min       = 0.10,
        soc_max       = 0.90,
        soc_init      = 0.50,
        epsilon_term  = 0.05,     # RE_003
        epsilon_g_term = 0.05,    # RE_004

        # RE accounting (spec SYS_004, RE_002)
        re_target     = 0.20,
        trec_cost_per_kwh = 4.63,

        # PWL degradation (spec DEG_001-007)
        pwl_deg_b_k = [0.0, 0.1, 0.3, 0.6, 0.8],
        pwl_deg_mu_k = [0.6, 1.0, 1.6, 2.4],
        pwl_deg_lambda_base = 1.612,
        pwl_deg_lambda_k = [0.97, 1.61, 2.58, 3.87],

        # Solver
        time_limit    = 600,
        mip_gap       = 1e-3,
    )

    Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)

    def crf(r, n):
        return r * (1 + r)**n / ((1 + r)**n - 1)

    CFG['crf_bess'] = crf(CFG['discount_rate'], CFG['lifetime_bess'])
    print(f"CRF BESS ({CFG['lifetime_bess']}yr, r={CFG['discount_rate']}): {CFG['crf_bess']:.4f}")
    return CFG


# ──────────────────────────────────────────────────────────────
#  Case Table (spec §4)
# ──────────────────────────────────────────────────────────────

CASE_TABLE = [
    {"case_id": "C0", "ingest_file": "full_year_milp_ingest_pvdet_loaddet.parquet",
     "pv_mode": "det", "load_mode": "det", "label": "Det PV + Det Load"},
    {"case_id": "C1", "ingest_file": "full_year_milp_ingest_pvprob_loaddet.parquet",
     "pv_mode": "prob", "load_mode": "det", "label": "Prob PV + Det Load"},
    {"case_id": "C2", "ingest_file": "full_year_milp_ingest_pvdet_loadpert.parquet",
     "pv_mode": "det", "load_mode": "pert", "label": "Det PV + Pert Load"},
    {"case_id": "C3", "ingest_file": "full_year_milp_ingest_pvprob_loadpert.parquet",
     "pv_mode": "prob", "load_mode": "pert", "label": "Prob PV + Pert Load"},
]


# ──────────────────────────────────────────────────────────────
#  Data Loading (full-year ingest)
# ──────────────────────────────────────────────────────────────

def load_data(CFG, case):
    """Load a full-year MILP ingest package.

    Returns:
        day_data: dict[day_index] -> {
            'calendar_day', 'month_id', 'season_tag', 'day_type', 'is_holiday',
            'scenarios': list of {pv_kw[24], load_kw[24], prob}
            'tou': ndarray[24] TOU prices
            'is_summer': bool
        }
        n_days, n_hours, scenario_ids, day_indices
    """
    bridge = Path(CFG['bridge_dir'])
    ingest = pd.read_parquet(bridge / case['ingest_file'])
    calendar = pd.read_parquet(bridge / 'caseyear_calendar_manifest.parquet')

    day_indices = sorted(ingest['day_index'].unique())
    scenario_ids = sorted(ingest['scenario_id'].unique())
    n_days = len(day_indices)
    n_scenarios = len(scenario_ids)
    n_hours = 24

    print(f"  Loading {case['case_id']}: {case['label']}")
    print(f"  Days: {n_days}, Scenarios: {n_scenarios}, Hours: {n_hours}")

    # Build day_data
    day_data = {}
    cal_lookup = calendar.set_index('day_index')

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

        # TOU prices for this day's 24 hours
        # t=0..23 maps directly to TOU hour_0based=0..23
        # (hour_local=1 is hour-ending 01:00, i.e. the 00:00-01:00 interval)
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


def load_truth(CFG):
    """Load truth replay package."""
    bridge = Path(CFG['bridge_dir'])
    truth = pd.read_parquet(bridge / 'full_year_replay_truth_package.parquet')
    calendar = pd.read_parquet(bridge / 'caseyear_calendar_manifest.parquet')
    return truth, calendar


# ──────────────────────────────────────────────────────────────
#  Monthly basic charge helper
# ──────────────────────────────────────────────────────────────

def get_monthly_basic_charge(month, CFG):
    """Return basic charge rate. Summer months: Jun-Sep + May,Oct (mixed)."""
    if month in (5, 6, 7, 8, 9, 10):
        return CFG['basic_charge_summer']
    else:
        return CFG['basic_charge_nonsummer']


# ──────────────────────────────────────────────────────────────
#  Results formatting
# ──────────────────────────────────────────────────────────────

def format_results(case_id, cap_bess_p, cap_bess_e, cap_contract,
                   obj_val, re_pct, cost_breakdown, solve_time):
    """Format a single case result."""
    return {
        'case': case_id,
        'pv_kw': 2687,
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
        're_pct': round(re_pct, 1),
        'solve_s': round(solve_time, 1),
    }
