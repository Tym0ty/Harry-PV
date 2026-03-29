"""
MILP Layer — Shared config, data loading, TOU, results.

Per F0329MILP_Spec.pdf (Batch 17) + F0329Bridge_Spec.pdf (Batch 8).
Cases: C0 (det PV + det load), C1 (prob PV + det load),
       C2 (det PV + load unc), C3 (prob PV + load unc), C_PFI (perfect).

One-parser rule: load_data() dispatches on pv_mode / load_mode fields,
NOT on filename or case_id. All 5 cases share the same loading code path.
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
        bridge_dir    = '../bridge_outputs_fullyear',   # F0329 bridge output dir
        output_dir    = '../milp_outputs',
        spec_version  = 'F0329MILP_Spec Batch 17',

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

# ── Canonical filename lookup (spec §F0329Bridge Batch 8) ─────
# Alias fallback: try canonical name first, then legacy alias if file missing.
CANONICAL_ROLLING_FILES = {
    "C0":    "rolling_da_input_pvdet_loaddet.parquet",
    "C1":    "rolling_da_input_pvprob_loaddet.parquet",
    "C2":    "rolling_da_input_pvdet_loadunc.parquet",
    "C3":    "rolling_da_input_pvprob_loadunc.parquet",
    "C_PFI": "rolling_da_input_pvperfect_loadperfect.parquet",
}
LEGACY_ALIAS_ROLLING = {
    "C0": "full_year_milp_ingest_pvdet_loaddet.parquet",
    "C1": "full_year_milp_ingest_pvprob_loaddet.parquet",
    "C2": "full_year_milp_ingest_pvdet_loadpert.parquet",
    "C3": "full_year_milp_ingest_pvprob_loadpert.parquet",
}

CASE_TABLE = [
    {"case_id": "C0",    "pv_mode": "pv_det",     "load_mode": "load_det",
     "label": "Det PV + Det Load (Baseline)"},
    {"case_id": "C1",    "pv_mode": "pv_prob",    "load_mode": "load_det",
     "label": "Prob PV + Det Load"},
    {"case_id": "C2",    "pv_mode": "pv_det",     "load_mode": "load_unc",
     "label": "Det PV + Load Uncertainty"},
    {"case_id": "C3",    "pv_mode": "pv_prob",    "load_mode": "load_unc",
     "label": "Prob PV + Load Uncertainty"},
    {"case_id": "C_PFI", "pv_mode": "pv_pfi",     "load_mode": "load_pfi",
     "label": "Perfect Forecast (Upper Bound)"},
]


# ──────────────────────────────────────────────────────────────
#  Data Loading (full-year ingest)
# ──────────────────────────────────────────────────────────────

def load_data(CFG, case):
    """Load a rolling DA input package for any of the 5 formal cases.

    One-parser rule (spec §2.3): dispatches on pv_mode / load_mode fields
    from the parquet data itself — NOT on filename or case_id branching.

    Tries canonical filename first; falls back to legacy alias if missing.

    Returns:
        day_data: dict[day_index] -> {
            'calendar_day', 'month_id', 'season_tag', 'day_type', 'is_holiday',
            'scenarios': list of {pv_kw[24], load_kw[24], prob, scenario_id}
            'tou': ndarray[24] TOU prices
            'is_summer': bool, 'pv_mode': str, 'load_mode': str
        }
        day_indices, scenario_ids
    """
    bridge = Path(CFG['bridge_dir'])
    case_id = case['case_id']

    # Canonical filename with legacy alias fallback
    canonical = bridge / CANONICAL_ROLLING_FILES[case_id]
    legacy = bridge / LEGACY_ALIAS_ROLLING.get(case_id, "")
    if canonical.exists():
        ingest_path = canonical
    elif legacy.exists():
        print(f"  [WARN] Using legacy alias: {legacy.name}")
        ingest_path = legacy
    else:
        raise FileNotFoundError(
            f"Rolling package not found for {case_id}.\n"
            f"  Tried: {canonical}\n"
            f"  Tried: {legacy}"
        )

    ingest = pd.read_parquet(ingest_path)
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

        # One-parser rule: dispatch on pv_mode/load_mode from data, not from case_id
        sample_row = d_ingest.iloc[0]
        day_pv_mode   = sample_row['pv_mode']
        day_load_mode = sample_row['load_mode']

        scenarios = []
        for sid in scenario_ids:
            s_data = d_ingest[d_ingest['scenario_id'] == sid].sort_values('hour_local')
            if len(s_data) == 0:
                continue
            scenarios.append({
                'pv_kw':       s_data['pv_available_kw'].values,
                'load_kw':     s_data['load_kw'].values,
                'prob':        float(s_data['probability_pi'].iloc[0]),
                'scenario_id': sid,
            })

        # TOU prices: t=0..23 maps to hour_0based=0..23
        tou = np.zeros(n_hours)
        for t in range(n_hours):
            tou[t] = get_tou_price(cd.month, cd.day, dow, t)

        day_data[di] = {
            'calendar_day': cd,
            'month_id':     int(cal['month_id']),
            'season_tag':   cal['season_tag'],
            'day_type':     cal['day_type'],
            'is_holiday':   bool(cal['is_holiday']),
            'is_summer':    bool(cal['is_summer']),
            'pv_mode':      day_pv_mode,
            'load_mode':    day_load_mode,
            'scenarios':    scenarios,
            'tou':          tou,
            'dow':          dow,
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
