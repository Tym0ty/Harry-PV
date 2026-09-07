"""
milp_v2/layer_b/run_layer_b_mpc.py
Hourly rolling MPC with v4 intraday forecast — Phase 2.

Hour convention (critical):
  MILP step t (0-indexed, midnight=0, 6am=6, 23=11pm)
  hour_local     = t + 1  (1-indexed; matches load_truth, replay_truth, stage4)
  v4 target_hour = t + 1  (same as hour_local; daytime = 7..17)
  v4 issue_hour  = h      (MILP step at decision time; fm.index.hour)

At MPC decision step h (deciding battery for hour h):
  issue_hour = h (features available at h:00, causal)
  For remaining MILP step t in {h, h+1, ..., 23}:
    v4_lead_time = t + 1 - h
    if v4_lead_time in {1..6} and (t+1) in {7..17}: use v4
    elif (t+1) in {7..17} and v4_lead_time > 6:      persistence (repeat lead_time=6)
    else (nighttime):                                  pv = 0

K=5 scenarios: pv_q10, pv_q30, pv_q50, pv_q70, pv_q90
Weights: 0.15, 0.20, 0.30, 0.20, 0.15

Usage:
    python milp_v2/layer_b/run_layer_b_mpc.py --case DP --lam 0.0
    python milp_v2/layer_b/run_layer_b_mpc.py --case DP --lam 0.5
"""

import argparse, json, time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_basic_charge, get_tou_price
from layer_b.milp_mpc import solve_mpc_milp

CFG     = load_config()
DES_DIR = ROOT / CFG["paths"]["designs_dir"]
LB_DIR  = ROOT / CFG["paths"]["layerb_results_dir"]
LB_DIR.mkdir(parents=True, exist_ok=True)

# ── Scenario weights for K=5 PV quantiles ────────────────────────────────────
K_SCENARIOS   = 5
SC_QUANTILES  = ["pv_q10", "pv_q30", "pv_q50", "pv_q70", "pv_q90"]
SC_WEIGHTS    = np.array([0.15, 0.20, 0.30, 0.20, 0.15], dtype=float)

# Daytime hours in v4 target_hour convention (= hour_local = MILP t + 1)
V4_DAY_HOURS  = set(range(7, 18))   # 7..17


# ── Data loading ──────────────────────────────────────────────────────────────

def build_v4_lookup(cfg: dict) -> dict:
    """
    Build (issue_date, issue_hour, v4_target_hour) -> {pv_q10..q90} dict.
    issue_date  : datetime.date
    issue_hour  : int (0-23), hour component of fm.index at that row
    v4_target_hour : int = issue_hour + lead_time
    """
    print("Building v4 intraday forecast lookup ...")
    v4_path = Path(__file__).resolve().parents[2] / "new_pipeline/data/final_forecast/id_forecast_v4_final.parquet"
    fm_path = Path(__file__).resolve().parents[2] / "new_pipeline/data/intraday/feature_matrix_id.parquet"

    fm = pd.read_parquet(fm_path, columns=[])   # only need the index
    v4 = pd.read_parquet(v4_path, columns=["lead_time"] + SC_QUANTILES)

    issue_dts   = fm.index[v4.index]            # DatetimeIndex (may have duplicates)
    issue_dates = issue_dts.date
    issue_hours = issue_dts.hour
    target_hours = issue_hours + v4["lead_time"].values

    lookup = {}
    for i, idx in enumerate(v4.index):
        key = (issue_dates[i], issue_hours[i], int(target_hours[i]))
        row = v4.loc[idx]
        lookup[key] = np.maximum(
            np.array([row[q] for q in SC_QUANTILES], dtype=float),
            0.0
        )

    print(f"  v4 lookup: {len(lookup)} entries")
    return lookup


def build_load_lookup(cfg: dict) -> dict:
    """
    Build (date, hour_local) -> load_kw dict.
    hour_local = t + 1 (1-indexed, same as MILP t=0..23).
    """
    lt_path = Path(__file__).resolve().parents[2] / "new_pipeline/data/input/load_truth.parquet"
    lt = pd.read_parquet(lt_path)
    lookup = {}
    for _, row in lt.iterrows():
        key = (row["calendar_day"].date(), int(row["hour_local"]))
        lookup[key] = float(row["load_kw"])
    print(f"  load lookup: {len(lookup)} entries")
    return lookup


def build_replay_lookup(cfg: dict) -> dict:
    """
    Build (date, hour_local) -> pv_realized_kw dict from replay_truth.
    """
    rt_path = ROOT / cfg["paths"]["upstream_truth"].replace("../", "")
    if not rt_path.exists():
        rt_path = ROOT.parent / "bridge_outputs_fullyear/full_year_replay_truth_package.parquet"
    rt = pd.read_parquet(rt_path)
    lookup = {}
    for _, row in rt.iterrows():
        key = (pd.Timestamp(row["calendar_day"]).date(), int(row["hour_local"]))
        lookup[key] = float(row["pv_realized_kw"])
    print(f"  replay PV lookup: {len(lookup)} entries")
    return lookup


def get_test_dates(cfg: dict) -> list:
    """Return all calendar dates in the test period (2024-11-01..2025-10-31)."""
    start = pd.Timestamp(cfg["horizon"]["case_year_start"]).date()
    end   = (pd.Timestamp(cfg["horizon"]["case_year_end"]) - pd.Timedelta(days=1)).date()
    import datetime
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += datetime.timedelta(days=1)
    return dates


# ── V4 forecast builder for one MPC step ─────────────────────────────────────

def build_pv_scenarios(
    date,
    current_hour: int,
    H_remain: int,
    v4_lookup: dict,
) -> np.ndarray:
    """
    Build (K=5, H_remain) PV scenario matrix for MPC step at current_hour.
    Returns array clipped to >= 0.
    """
    pv = np.zeros((K_SCENARIOS, H_remain), dtype=float)

    # Persistence fallback: lead_time=6 from current_hour
    persist_target = current_hour + 6
    persist_key    = (date, current_hour, persist_target)
    persist_vals   = v4_lookup.get(persist_key, None)

    for j in range(H_remain):
        milp_t    = current_hour + j
        tgt       = milp_t + 1          # v4_target_hour = t + 1
        lead      = tgt - current_hour  # = j + 1

        if tgt not in V4_DAY_HOURS:
            # Nighttime: PV = 0
            continue

        if 1 <= lead <= 6:
            key = (date, current_hour, tgt)
            if key in v4_lookup:
                pv[:, j] = v4_lookup[key]
        else:
            # Persistence: repeat lead=6 forecast
            if persist_vals is not None and persist_target in V4_DAY_HOURS:
                pv[:, j] = persist_vals

    return np.maximum(pv, 0.0)


def build_load_scenarios(
    date,
    current_hour: int,
    H_remain: int,
    load_lookup: dict,
) -> np.ndarray:
    """
    Build (K=5, H_remain) deterministic load matrix (same for all scenarios).
    load_truth hour_local = MILP t + 1.
    """
    load = np.zeros((K_SCENARIOS, H_remain), dtype=float)
    for j in range(H_remain):
        milp_t    = current_hour + j
        hl        = milp_t + 1            # hour_local (1-indexed)
        load_kw   = load_lookup.get((date, hl), 0.0)
        load[:, j] = load_kw
    return load


# ── MPC day loop ──────────────────────────────────────────────────────────────

def run_mpc_day(
    date,
    design: dict,
    state_start: dict,
    month_id: int,
    v4_lookup: dict,
    load_lookup: dict,
    replay_lookup: dict,
    lam: float,
    alpha: float,
    cfg: dict,
    is_final_day_of_year: bool = False,
) -> dict:
    """
    Run 24-hour hourly MPC loop for a single calendar day.
    Returns {hourly_rows, state_end, D_mth_end}.
    """
    CC = design["CC"]
    PB = design["PB"]
    EB = design["EB"]

    cd   = pd.Timestamp(date)
    dow  = cd.weekday()
    tou  = np.array([get_tou_price(cd.month, cd.day, dow, h) for h in range(24)])
    c_basic = get_basic_charge(month_id, cfg)

    E_soc  = float(state_start["E_soc"])
    D_mth  = float(state_start["D_mth"])

    hourly_rows = []
    p_grid_hour = []     # realized grid imports per hour (for D_mth update)

    for h in range(24):
        H_remain = 24 - h
        is_last  = (h == 23) and is_final_day_of_year

        pv_sc   = build_pv_scenarios(date, h, H_remain, v4_lookup)
        ld_sc   = build_load_scenarios(date, h, H_remain, load_lookup)
        tou_rem = tou[h:]                # remaining TOU prices

        res = solve_mpc_milp(
            pv_scenarios        = pv_sc,
            load_scenarios      = ld_sc,
            pi                  = SC_WEIGHTS,
            cc                  = CC,
            p_b                 = PB,
            e_b                 = EB,
            initial_soc         = E_soc,
            peak_demand_running = D_mth,
            current_hour        = h,
            lam                 = lam,
            alpha               = alpha,
            cfg                 = cfg,
            is_last_hour        = is_last,
            month_id            = month_id,
            tou_remaining       = tou_rem,
            time_limit          = 30,
            mip_gap             = 0.005,
            verbose             = False,
        )

        p_ch_exec  = res["p_ch_exec"]
        p_dis_exec = res["p_dis_exec"]
        soc_next   = res["soc_next"]

        # Realized PV and load (for actual power balance)
        hl              = h + 1   # hour_local = t + 1
        pv_realized     = replay_lookup.get((date, hl), 0.0)
        load_realized   = load_lookup.get((date, hl), 0.0)

        # Actual grid import this hour (kappa × net grid draw)
        p_grid_realized = max(load_realized - pv_realized - p_dis_exec + p_ch_exec, 0.0)

        # Update SOC with realized values
        bat = cfg["battery"]
        E_soc_new = E_soc + bat["eta_ch"] * p_ch_exec - (1.0 / bat["eta_dis"]) * p_dis_exec
        E_soc_new = float(np.clip(E_soc_new,
                                  bat["soc_min"] * EB,
                                  bat["soc_max"] * EB))

        # Demand proxy for billing: kappa × p_grid
        d_this_hour = cfg["demand"]["kappa"] * p_grid_realized
        D_mth = max(D_mth, d_this_hour)

        hourly_rows.append({
            "hour":            h,
            "hour_local":      hl,
            "p_ch_exec":       p_ch_exec,
            "p_dis_exec":      p_dis_exec,
            "soc_start":       E_soc,
            "soc_end":         E_soc_new,
            "pv_realized":     pv_realized,
            "load_realized":   load_realized,
            "p_grid_realized": p_grid_realized,
            "d_this_hour":     d_this_hour,
            "D_mth_running":   D_mth,
            "E_coc_val":       res["E_coc_val"],
            "cvar_val":        res["cvar_val"],
            "solve_time":      res["solve_time"],
            "milp_status":     res["status"],
        })

        E_soc = E_soc_new

    return {
        "hourly_rows": hourly_rows,
        "E_soc_end":   E_soc,
        "D_mth_end":   D_mth,
    }


# ── Full-year runner ──────────────────────────────────────────────────────────

def run_case_mpc(case_id: str, lam: float, alpha: float) -> tuple:
    """
    Run hourly MPC for one case over the full test year.
    Returns (daily_df, hourly_df).
    """
    design_path = DES_DIR / f"design_{case_id}.json"
    if not design_path.exists():
        raise FileNotFoundError(f"Design not found: {design_path}  (run Layer A first)")
    with open(design_path) as f:
        des = json.load(f)
    design = {"CC": des["CC_star"], "PB": des["PB_star"], "EB": des["EB_star"]}

    lam_tag   = f"{lam:.2f}".replace(".", "p")
    alpha_tag = f"{int(alpha*100):02d}"
    run_tag   = f"mpc_lam{lam_tag}_a{alpha_tag}"

    print(f"\n[Layer B MPC] case={case_id}  lam={lam}  alpha={alpha}")
    print(f"  CC={design['CC']:.1f}  PB={design['PB']:.1f}  EB={design['EB']:.1f}")

    v4_lookup     = build_v4_lookup(CFG)
    load_lookup   = build_load_lookup(CFG)
    replay_lookup = build_replay_lookup(CFG)

    test_dates = get_test_dates(CFG)
    n_days     = len(test_dates)
    print(f"  Test period: {test_dates[0]} to {test_dates[-1]}  ({n_days} days)")

    soc_init = CFG["battery"]["soc_init"]
    state = {
        "E_soc": soc_init * design["EB"],
        "D_mth": 0.0,
    }

    daily_rows   = []
    all_hourly   = []
    t0_case      = time.time()

    # Build a month_id lookup (date → month_id) from replay_truth
    rt_path = ROOT / CFG["paths"]["upstream_truth"].replace("../", "")
    if not rt_path.exists():
        rt_path = ROOT.parent / "bridge_outputs_fullyear/full_year_replay_truth_package.parquet"
    rt_cal = pd.read_parquet(rt_path, columns=["calendar_day", "month_id"]).drop_duplicates()
    month_id_lookup = {
        pd.Timestamp(r["calendar_day"]).date(): int(r["month_id"])
        for _, r in rt_cal.iterrows()
    }

    cur_month_id = None
    prev_month_id = None

    for idx, date in enumerate(test_dates):
        month_id = month_id_lookup.get(date, date.month)

        # Month roll: reset D_mth at month boundary
        if prev_month_id is not None and month_id != prev_month_id:
            state["D_mth"] = 0.0
        prev_month_id = month_id

        day_result = run_mpc_day(
            date                  = date,
            design                = design,
            state_start           = state,
            month_id              = month_id,
            v4_lookup             = v4_lookup,
            load_lookup           = load_lookup,
            replay_lookup         = replay_lookup,
            lam                   = lam,
            alpha                 = alpha,
            cfg                   = CFG,
            is_final_day_of_year  = (idx == n_days - 1),
        )

        D_mth_end = day_result["D_mth_end"]
        E_soc_end = day_result["E_soc_end"]

        # Over-contract cost for the day's final peak
        dem = CFG["demand"]
        CC  = design["CC"]
        c_b = get_basic_charge(month_id, CFG)
        over = max(D_mth_end - CC, 0.0)
        o1   = min(over, dem["over_contract_tier1_ratio"] * CC)
        o2   = max(over - dem["over_contract_tier1_ratio"] * CC, 0.0)
        C_oc = c_b * (dem["over_contract_m1"] * o1 + dem["over_contract_m2"] * o2)

        daily_rows.append({
            "day_index":    idx,
            "calendar_day": str(date),
            "month_id":     month_id,
            "D_mth_end":    D_mth_end,
            "C_oc":         C_oc,
            "E_soc_end":    E_soc_end,
        })

        # Append hourly rows with day context
        for hr in day_result["hourly_rows"]:
            hr["day_index"]    = idx
            hr["calendar_day"] = str(date)
            hr["month_id"]     = month_id
            all_hourly.append(hr)

        state = {"E_soc": E_soc_end, "D_mth": D_mth_end}

        if (idx + 1) % 30 == 0 or (idx + 1) == n_days:
            elapsed = time.time() - t0_case
            print(f"  Day {idx+1:3d}/{n_days}  date={date}  "
                  f"D_mth={D_mth_end:.0f}  C_oc={C_oc:.0f}  "
                  f"E_soc={E_soc_end:.0f}  t={elapsed:.0f}s")

    t_total = time.time() - t0_case
    df_daily  = pd.DataFrame(daily_rows)
    df_hourly = pd.DataFrame(all_hourly)

    out_daily  = LB_DIR / f"layerb_{case_id}_{run_tag}_daily.parquet"
    out_hourly = LB_DIR / f"layerb_{case_id}_{run_tag}_hourly.parquet"
    df_daily.to_parquet(out_daily,  index=False)
    df_hourly.to_parquet(out_hourly, index=False)

    print(f"  Saved {len(df_daily)} days   -> {out_daily.name}  [{t_total:.0f}s]")
    print(f"  Saved {len(df_hourly)} hours  -> {out_hourly.name}")

    return df_daily, df_hourly


# ── Validation: single-day 24-hour loop ──────────────────────────────────────

def validate_single_day(case_id: str = "DP", lam: float = 0.0, alpha: float = 0.90,
                        target_date_str: str = "2024-11-01") -> bool:
    """
    V1-V5 checks for a single day's 24-hour MPC loop.
    """
    import datetime

    print("\n" + "=" * 60)
    print(f"Phase 2 Validation: single-day MPC ({target_date_str}, case={case_id})")
    print("=" * 60)

    design_path = DES_DIR / f"design_{case_id}.json"
    if not design_path.exists():
        print(f"  [SKIP] design file not found: {design_path}")
        return False
    with open(design_path) as f:
        des = json.load(f)
    design = {"CC": des["CC_star"], "PB": des["PB_star"], "EB": des["EB_star"]}

    v4_lookup     = build_v4_lookup(CFG)
    load_lookup   = build_load_lookup(CFG)
    replay_lookup = build_replay_lookup(CFG)

    date = pd.Timestamp(target_date_str).date()
    month_id = date.month

    print(f"\n  Design: CC={design['CC']:.0f}  PB={design['PB']:.0f}  EB={design['EB']:.0f}")
    print(f"  Date: {date}  lam={lam}  alpha={alpha}")

    t0 = time.time()
    result = run_mpc_day(
        date                  = date,
        design                = design,
        state_start           = {"E_soc": CFG["battery"]["soc_init"] * design["EB"], "D_mth": 0.0},
        month_id              = month_id,
        v4_lookup             = v4_lookup,
        load_lookup           = load_lookup,
        replay_lookup         = replay_lookup,
        lam                   = lam,
        alpha                 = alpha,
        cfg                   = CFG,
        is_final_day_of_year  = False,
    )
    elapsed = time.time() - t0

    hrs     = result["hourly_rows"]
    n_hours = len(hrs)
    n_fail  = sum(1 for r in hrs if r["milp_status"] < 0)

    all_ok  = True

    def check(label, cond, detail=""):
        tag = "[PASS]" if cond else "[FAIL]"
        print(f"  {tag}  {label}" + (f"  ({detail})" if detail else ""))
        return cond

    # V1: 24 hourly rows produced
    all_ok &= check("V1: 24 hourly rows produced", n_hours == 24,
                    f"n_hours={n_hours}")

    # V2: all MILP solves succeeded
    all_ok &= check("V2: all MILP solves optimal/feasible", n_fail == 0,
                    f"fails={n_fail}")

    # V3: SOC continuity (each hour's soc_end = next hour's soc_start)
    soc_ok = True
    for i in range(n_hours - 1):
        diff = abs(hrs[i]["soc_end"] - hrs[i+1]["soc_start"])
        if diff > 0.1:
            soc_ok = False
            break
    all_ok &= check("V3: SOC continuity across hours", soc_ok)

    # V4: SOC within physical bounds
    soc_min_v = CFG["battery"]["soc_min"] * design["EB"]
    soc_max_v = CFG["battery"]["soc_max"] * design["EB"]
    soc_bound_ok = all(soc_min_v - 1 <= r["soc_end"] <= soc_max_v + 1 for r in hrs)
    all_ok &= check("V4: SOC within [soc_min*EB, soc_max*EB]", soc_bound_ok,
                    f"bounds=[{soc_min_v:.0f},{soc_max_v:.0f}]")

    # V5: D_mth is non-decreasing within the day
    d_vals  = [r["D_mth_running"] for r in hrs]
    mono_ok = all(d_vals[i] <= d_vals[i+1] + 0.1 for i in range(len(d_vals)-1))
    all_ok &= check("V5: D_mth non-decreasing within day", mono_ok,
                    f"D_mth_final={d_vals[-1]:.1f}")

    # Summary
    print(f"\n  Total wall time: {elapsed:.1f}s  ({elapsed/24:.1f}s/hour)")
    print(f"  D_mth_end={result['D_mth_end']:.1f} kW  E_soc_end={result['E_soc_end']:.1f} kWh")
    print(f"\n  Phase 2 validation: {'ALL PASS' if all_ok else 'SOME FAIL'}")

    return all_ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case",     choices=["DD","DP","PD","PP","PI"], default="DP")
    ap.add_argument("--lam",      type=float, default=0.0)
    ap.add_argument("--alpha",    type=float, default=0.90)
    ap.add_argument("--validate", action="store_true",
                    help="Run single-day validation only (faster)")
    ap.add_argument("--date",     default="2024-11-01",
                    help="Validation date (YYYY-MM-DD)")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("milp_v2 Layer B — Hourly MPC (Phase 2)")
    print("=" * 60)

    if args.validate:
        validate_single_day(args.case, args.lam, args.alpha, args.date)
    else:
        try:
            run_case_mpc(args.case, args.lam, args.alpha)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
