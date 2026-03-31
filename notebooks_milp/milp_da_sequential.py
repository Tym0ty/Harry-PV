#!/usr/bin/env python3
"""
Two-Layer Framework — Layer B: Sequential Day-Ahead Daily MILP
Per 0331_MILP_without_REP_SPEC_final (§6.3–§7.1) and 0331_bridge__SPEC_final.

═══════════════════════════════════════════════════════════════════
FOUR-MODULE WORKFLOW
═══════════════════════════════════════════════════════════════════

MODULE 1 — ANNUAL DESIGN PACKAGE  (produced by bridge_full_year.py)
  bridge_outputs_fullyear/full_year_milp_ingest_*.parquet
  These are the design-world inputs used by Layer A to select (CC*, P_B*, E_B*).
  They are NEVER used directly in Layer B.

MODULE 2 — FROZEN DESIGN ARTIFACT  (produced by milp_run_fullyear.py → Layer A)
  milp_outputs/case_summary_fullyear.csv    — source of record
  milp_outputs/da/layer_a_frozen_design_{case}.json  — written here at startup
  CC*, P_B*, E_B* are read once and NEVER changed within Layer B.

MODULE 3 — DAILY DAY-AHEAD PACKAGES  (produced by bridge_full_year.py)
  bridge_outputs_fullyear/rolling_da_input_*.parquet
  Each row carries issue_day (D-1) and gate_time_local="20:00" to make the
  information boundary auditable. Layer B reads ONLY the record for calendar_day=i
  when solving day i. Realized truth is NEVER read in the solve step.

MODULE 4 — SEQUENTIAL DAILY REPLAY  (this script)
  For each calendar day i (in calendar order):
    a. Solve one 24-hour day-ahead MILP using ONLY Module 3 forecast package
       Battery control (u, P_ch, P_dis, E) is SCENARIO-INVARIANT.
    b. Settle frozen plan arithmetically against realized truth (no MILP).
    c. Carry forward state: E_soc, E_g, D_mth.
  Evidence outputs:
    da_daily_solve_log_{case}.csv       — one row per day, gate metadata + solve stats
    daily_plan_trace_{case}.parquet     — hourly battery plan (scenario-invariant)
    daily_settlement_trace_{case}.parquet — hourly realized settlement
    rolling_state_trace_{case}.parquet  — SOC/GreenSOC/D_mth start+end per day

═══════════════════════════════════════════════════════════════════

Key spec rules enforced:
  - (Rule 1) Day-ahead only: each solve covers exactly 24h for the next calendar day
  - (Rule 2) D-1 20:00 gate: only forecast data frozen at that gate enters the solve
  - (Rule 3) Scenario-invariant control: u[t], P_ch[t], P_dis[t], E[t] are common
             across all scenarios; scenarios only affect settlement variables
  - (Rule 4) One solve per day: not rolling hourly MPC, not one-shot annual
  - (Rule 5) Serialized state: E_soc, E_g, D_mth carry forward day-to-day;
             D_mth resets at month boundaries

Usage:
    cd <repo_root>
    python notebooks_milp/milp_da_sequential.py
    python notebooks_milp/milp_da_sequential.py --cases C0 C1
"""

import argparse, json, time, warnings
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))

from milp_common import get_config, get_tou_price, get_monthly_basic_charge, CASE_TABLE

OUT_DA  = ROOT / "milp_outputs" / "da"
BRIDGE  = ROOT / "bridge_outputs_fullyear"
MILP_OUT = ROOT / "milp_outputs"

warnings.filterwarnings("ignore")


# ── Load Layer A sizing ───────────────────────────────────────────────────────

def load_layer_a_sizing(case_id):
    """MODULE 2: Load and freeze CC*, P_B*, E_B* from Layer A output.

    Reads case_summary_fullyear.csv (Layer A source of record) and writes
    layer_a_frozen_design_{case_id}.json as an auditable artifact confirming
    that the design is fixed before Layer B begins.
    """
    path = MILP_OUT / "case_summary_fullyear.csv"
    df = pd.read_csv(path)
    row = df[df["case"] == case_id]
    if len(row) == 0:
        raise ValueError(f"Case {case_id} not found in case_summary_fullyear.csv")
    row = row.iloc[0]
    sizing = {
        "CC":  float(row["contract_kw"]),
        "P_B": float(row["bess_p_kw"]),
        "E_B": float(row["bess_e_kwh"]),
    }

    # Write frozen design artifact (Module 2 evidence)
    OUT_DA.mkdir(parents=True, exist_ok=True)
    frozen = {
        "module": "MODULE 2 — FROZEN DESIGN ARTIFACT",
        "case_id": case_id,
        "layer_a_source_file": str(path.relative_to(ROOT)),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "CC_kw":  sizing["CC"],
        "P_B_kw": sizing["P_B"],
        "E_B_kwh": sizing["E_B"],
        "note": (
            "These three values are read once from Layer A and held fixed "
            "for all 365 day-ahead solves in Layer B. "
            "CC/P_B/E_B are NEVER re-optimized within Layer B."
        ),
    }
    artifact_path = OUT_DA / f"layer_a_frozen_design_{case_id}.json"
    with open(artifact_path, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"  [Module 2] Frozen design artifact: {artifact_path.name}")

    return sizing


# ── Load rolling day-ahead package ───────────────────────────────────────────

def load_rolling_da_package(case):
    """MODULE 3: Load rolling_da_input_*.parquet for a case.

    Each daily slice carries issue_day (D-1) and gate_time_local="20:00"
    as auditable evidence of the information boundary.

    Returns dict[day_index] -> scenario dicts + gate metadata.
    """
    pv_mode   = "pvprob" if case["pv_mode"] == "prob" else "pvdet"
    load_mode = "loadunc" if case["load_mode"] == "unc" else "loaddet"
    fname     = f"rolling_da_input_{pv_mode}_{load_mode}.parquet"
    path      = BRIDGE / fname

    if not path.exists():
        raise FileNotFoundError(f"Rolling DA package not found: {path}")

    df = pd.read_parquet(path)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])
    if "issue_day" in df.columns:
        df["issue_day"] = pd.to_datetime(df["issue_day"])

    load_col = "load_input_kw" if "load_input_kw" in df.columns else "load_kw"

    calendar = pd.read_parquet(BRIDGE / "caseyear_calendar_manifest.parquet")
    cal_lookup = calendar.set_index("day_index")

    day_indices = sorted(df["day_index"].unique())
    day_data = {}

    for di in day_indices:
        d_df = df[df["day_index"] == di]
        cal  = cal_lookup.loc[di]
        cd   = pd.Timestamp(cal["calendar_day"])
        dow  = cd.weekday()

        # Gate metadata: extract from first row of this day's slice
        first_row = d_df.iloc[0]
        issue_day       = str(first_row["issue_day"].date()) if "issue_day" in d_df.columns else str((cd - pd.Timedelta(days=1)).date())
        gate_time_local = str(first_row["gate_time_local"]) if "gate_time_local" in d_df.columns else "20:00"

        scenario_ids = sorted(d_df["scenario_id"].unique())
        scenarios = []
        for sid in scenario_ids:
            s = d_df[d_df["scenario_id"] == sid].sort_values("hour_local")
            scenarios.append({
                "scenario_id": sid,
                "pv_kw":  s["pv_available_kw"].values.astype(float),   # [24]
                "load_kw": s[load_col].values.astype(float),            # [24]
                "prob":   float(s["probability_pi"].iloc[0]),
            })

        # TOU prices for this day (hour_0based = 0..23)
        tou = np.array([get_tou_price(cd.month, cd.day, dow, t) for t in range(24)])

        day_data[di] = {
            "calendar_day":   cd,
            "month_id":       int(cal["month_id"]),
            "season_tag":     cal["season_tag"],
            "day_type":       cal["day_type"],
            "is_holiday":     bool(cal["is_holiday"]),
            "is_summer":      bool(cal["is_summer"]),
            "scenarios":      scenarios,
            "tou":            tou,
            "dow":            dow,
            # MODULE 3 gate-time metadata
            "issue_day":       issue_day,
            "gate_time_local": gate_time_local,
            "source_package":  fname,
        }

    print(f"  [Module 3] Loaded {fname}: {len(day_indices)} days, {len(scenarios)} scenarios/day")
    return day_data, sorted(day_indices)


# ── Load truth replay package ─────────────────────────────────────────────────

def load_truth_package():
    """Load full_year_replay_truth_package.parquet.
    Returns dict[(day_index, hour_local)] -> {pv_realized_kw, load_realized_kw}.
    """
    path = BRIDGE / "full_year_replay_truth_package.parquet"
    df = pd.read_parquet(path)
    truth = {}
    for _, row in df.iterrows():
        truth[(int(row["day_index"]), int(row["hour_local"]))] = {
            "pv":   float(row["pv_realized_kw"]),
            "load": float(row["load_realized_kw"]),
        }
    return truth


# ── Day-ahead MILP (single day, 24 hours) ────────────────────────────────────

def solve_day_ahead(day_dict, state, sizing, CFG, day_index, is_final_day=False,
                    time_limit=60, mip_gap=1e-3):
    """Solve the 24-hour day-ahead MILP for one calendar day.

    Battery control (u, P_ch, P_dis, E) is SCENARIO-INVARIANT.
    Scenarios affect only settlement variables (P_grid[ω,t], etc.).
    State variables (E_init, E_g_init, D_init) are parameters fixed from the previous day.

    Returns: dict with plan arrays + carry-forward values.
    """
    CC  = sizing["CC"]
    P_B = sizing["P_B"]
    E_B = sizing["E_B"]

    E_init   = state["E_soc"]    # kWh, start-of-day SOC
    Eg_init  = state["E_g"]      # kWh, start-of-day green SOC
    D_init   = state["D_mth"]    # kW,  month-to-date observed demand peak

    eta_ch  = CFG["eff_charge"]
    eta_dis = CFG["eff_discharge"]
    soc_min = CFG["soc_min"]
    soc_max = CFG["soc_max"]
    soc_init = CFG["soc_init"]
    eps_term  = CFG["epsilon_term"]
    kappa   = CFG["kappa"]

    oc_m10 = CFG["oc_within_10pct_mult"]
    oc_gt10 = CFG["oc_beyond_10pct_mult"]

    b_k  = CFG["pwl_deg_b_k"]
    lam_k = CFG["pwl_deg_lambda_k"]
    n_seg = len(lam_k)

    scenarios = day_dict["scenarios"]
    n_sc = len(scenarios)
    tou  = day_dict["tou"]        # [24]
    cd   = day_dict["calendar_day"]
    month = day_dict["month_id"]
    c_basic = get_monthly_basic_charge(month, CFG)

    H = 24
    T = range(H)
    W = range(n_sc)

    t_start = time.time()

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", time_limit)
        env.setParam("MIPGap", mip_gap)
        env.start()
        m = gp.Model(env=env)

        # ── Scenario-invariant battery control ───────────────────────
        # Spec §6.5.4: common day-ahead battery-control variables
        u       = m.addVars(H, vtype=GRB.BINARY,      name="u")        # charge=1, dis=0
        P_ch    = m.addVars(H, lb=0.0,                name="P_ch")     # kW
        P_dis   = m.addVars(H, lb=0.0,                name="P_dis")    # kW
        E       = m.addVars(H, lb=soc_min * E_B, ub=soc_max * E_B, name="E")  # kWh end-of-hour

        # PWL degradation segments (scenario-invariant, linked to P_dis)
        e_seg = m.addVars(H, n_seg, lb=0.0, name="e_seg")

        # ── Scenario-dependent settlement variables ───────────────────
        # Spec §6.5.4: scenario-dependent settlement variables
        P_gl   = m.addVars(n_sc, H, lb=0.0, name="P_gl")    # grid_load[ω,t]
        P_gc   = m.addVars(n_sc, H, lb=0.0, name="P_gc")    # grid_ch[ω,t]
        P_pvl  = m.addVars(n_sc, H, lb=0.0, name="P_pvl")   # pv_load[ω,t]
        P_pvc  = m.addVars(n_sc, H, lb=0.0, name="P_pvc")   # pv_ch[ω,t]
        P_pvcu = m.addVars(n_sc, H, lb=0.0, name="P_pvcu")  # pv_curt[ω,t]

        # Green SOC tracking (scenario-dependent)
        P_chg  = m.addVars(n_sc, H, lb=0.0, name="P_chg")   # pv_ch_green[ω,t]
        P_disg = m.addVars(n_sc, H, lb=0.0, name="P_disg")  # dis_green[ω,t]
        E_g    = m.addVars(n_sc, H, lb=0.0, name="E_g")     # green SOC end-of-hour[ω,t]

        # ── Monthly demand proxy (spec §6.5.9) ───────────────────────
        D_cand    = m.addVar(lb=D_init,  name="D_cand")       # candidate MTD peak
        Over_full = m.addVar(lb=0.0,     name="Over_full")
        O1_full   = m.addVar(lb=0.0, ub=0.1 * CC, name="O1_full")
        O2_full   = m.addVar(lb=0.0,              name="O2_full")

        # ── Constraint (1): battery control limits ────────────────────
        for t in T:
            m.addConstr(P_ch[t]  <= u[t] * P_B,         name=f"ch_ub_{t}")
            m.addConstr(P_dis[t] <= (1 - u[t]) * P_B,   name=f"dis_ub_{t}")

        # ── Constraint (2): SOC recursion ─────────────────────────────
        # E[t] is SOC at end of hour t; E_init is start-of-day (before hour 0)
        for t in T:
            if t == 0:
                m.addConstr(
                    E[t] == E_init + eta_ch * P_ch[t] - (1.0 / eta_dis) * P_dis[t],
                    name=f"soc_rec_{t}"
                )
            else:
                m.addConstr(
                    E[t] == E[t-1] + eta_ch * P_ch[t] - (1.0 / eta_dis) * P_dis[t],
                    name=f"soc_rec_{t}"
                )

        # ── Constraint (3): SOC bounds already set as lb/ub on E ──────

        # ── Constraint (4): terminal SOC policy ──────────────────────
        # Non-final days: loose (SOC_min..SOC_max already enforced by bounds)
        # Final day: tight year-end band
        if is_final_day:
            E_term_lb = (soc_init - eps_term) * E_B
            E_term_ub = (soc_init + eps_term) * E_B
            m.addConstr(E[H-1] >= E_term_lb, name="term_lb")
            m.addConstr(E[H-1] <= E_term_ub, name="term_ub")

        # ── Constraint (5): discharge segmentation ────────────────────
        for t in T:
            m.addConstr(P_dis[t] == gp.quicksum(e_seg[t, k] for k in range(n_seg)),
                        name=f"seg_sum_{t}")
            for k in range(n_seg):
                cap_k = (b_k[k+1] - b_k[k]) * E_B
                m.addConstr(e_seg[t, k] <= cap_k, name=f"seg_cap_{t}_{k}")

        # ── Constraints (6)-(9): scenario settlement ──────────────────
        for w, sc in enumerate(scenarios):
            pv_w   = sc["pv_kw"]    # [24]
            load_w = sc["load_kw"]  # [24]
            for t in T:
                # (6) Load balance: grid_load + pv_load + P_dis = load
                m.addConstr(
                    P_gl[w, t] + P_pvl[w, t] + P_dis[t] == load_w[t],
                    name=f"load_bal_{w}_{t}"
                )
                # (7) Charging source decomposition: P_ch = grid_ch + pv_ch
                m.addConstr(
                    P_ch[t] == P_gc[w, t] + P_pvc[w, t],
                    name=f"ch_src_{w}_{t}"
                )
                # (8) Total grid purchase
                # P_grid[ω,t] = P_gl[ω,t] + P_gc[ω,t] — implicit, used in objective
                # (9) PV allocation (no-export)
                m.addConstr(
                    P_pvl[w, t] + P_pvc[w, t] + P_pvcu[w, t] == pv_w[t],
                    name=f"pv_alloc_{w}_{t}"
                )

        # ── Constraint (10): monthly demand proxy ─────────────────────
        # Expected grid import (scenario-weighted)
        for t in T:
            P_grid_bar_t = gp.quicksum(
                sc["prob"] * (P_gl[w, t] + P_gc[w, t]) for w, sc in enumerate(scenarios)
            )
            m.addConstr(D_cand >= kappa * P_grid_bar_t, name=f"dmax_{t}")

        # Over-contract decomposition
        m.addConstr(Over_full >= D_cand - CC,     name="over_pos")
        m.addConstr(Over_full == O1_full + O2_full, name="over_split")

        # ── Constraint (11): Green SOC per scenario ───────────────────
        for w, sc in enumerate(scenarios):
            for t in T:
                m.addConstr(P_chg[w, t]  <= P_pvc[w, t],  name=f"chg_ub_{w}_{t}")
                m.addConstr(P_disg[w, t] <= P_dis[t],      name=f"disg_ub_{w}_{t}")
                m.addConstr(E_g[w, t]    <= E[t],           name=f"eg_ub_{w}_{t}")
                if t == 0:
                    m.addConstr(
                        E_g[w, t] == Eg_init + eta_ch * P_chg[w, t] - (1.0 / eta_dis) * P_disg[w, t],
                        name=f"eg_rec_{w}_{t}"
                    )
                else:
                    m.addConstr(
                        E_g[w, t] == E_g[w, t-1] + eta_ch * P_chg[w, t] - (1.0 / eta_dis) * P_disg[w, t],
                        name=f"eg_rec_{w}_{t}"
                    )

        # ── Objective ─────────────────────────────────────────────────
        # C_ene: expected energy cost
        C_ene = gp.quicksum(
            sc["prob"] * tou[t] * (P_gl[w, t] + P_gc[w, t])
            for w, sc in enumerate(scenarios) for t in T
        )

        # C_deg: battery degradation cost
        C_deg = gp.quicksum(lam_k[k] * e_seg[t, k] for t in T for k in range(n_seg))

        # C_peak: incremental demand-charge proxy (spec §6.5.9 incremental form)
        # PeakCost_prev: cost implied by D_init (fixed, no variable)
        Over_prev = max(D_init - CC, 0.0)
        O1_prev = min(Over_prev, 0.1 * CC)
        O2_prev = max(Over_prev - 0.1 * CC, 0.0)
        PeakCost_prev = c_basic * (oc_m10 * O1_prev + oc_gt10 * O2_prev)
        PeakCost_full = c_basic * (oc_m10 * O1_full + oc_gt10 * O2_full)
        C_peak = PeakCost_full - PeakCost_prev   # incremental

        m.setObjective(C_ene + C_deg + C_peak, GRB.MINIMIZE)
        m.optimize()

        solve_time = time.time() - t_start
        status = m.Status

        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            # Infeasible or other failure — return a do-nothing plan
            return _fallback_plan(E_init, E_B, soc_init, H, n_sc, scenarios, day_index, solve_time)

        # ── Extract plan ──────────────────────────────────────────────
        u_plan    = np.array([u[t].X for t in T])
        P_ch_plan = np.array([P_ch[t].X for t in T])
        P_dis_plan= np.array([P_dis[t].X for t in T])
        E_plan    = np.array([E[t].X for t in T])
        E_end     = float(E[H-1].X)   # carry forward as E_init for next day

        # Scenario-weighted green SOC at end of day
        Eg_end = sum(
            sc["prob"] * float(E_g[w, H-1].X) for w, sc in enumerate(scenarios)
        )

        D_cand_val = float(D_cand.X)
        obj_val    = float(m.ObjVal)

        # Scenario dispatch for logging
        sc_rows = []
        for w, sc in enumerate(scenarios):
            for t in T:
                sc_rows.append({
                    "scenario_id": sc["scenario_id"],
                    "hour_local": t + 1,
                    "prob": sc["prob"],
                    "P_grid_kw": float(P_gl[w, t].X + P_gc[w, t].X),
                    "P_pv_load_kw": float(P_pvl[w, t].X),
                    "P_pv_ch_kw": float(P_pvc[w, t].X),
                    "P_pv_curt_kw": float(P_pvcu[w, t].X),
                    "E_g_kwh": float(E_g[w, t].X),
                })

        return {
            "u_plan":     u_plan,
            "P_ch_plan":  P_ch_plan,
            "P_dis_plan": P_dis_plan,
            "E_plan":     E_plan,
            "E_end":      E_end,
            "E_g_end":    Eg_end,
            "D_cand":     D_cand_val,
            "obj_val":    obj_val,
            "solve_time": solve_time,
            "status":     status,
            "scenario_dispatch": sc_rows,
        }


def _fallback_plan(E_init, E_B, soc_init, H, n_sc, scenarios, day_index, solve_time):
    """Return do-nothing plan if MILP fails (infeasible)."""
    print(f"    WARNING: day {day_index} MILP infeasible — using do-nothing plan")
    return {
        "u_plan":     np.ones(H),           # no discharge
        "P_ch_plan":  np.zeros(H),
        "P_dis_plan": np.zeros(H),
        "E_plan":     np.full(H, E_init),
        "E_end":      E_init,
        "E_g_end":    0.0,
        "D_cand":     0.0,
        "obj_val":    0.0,
        "solve_time": solve_time,
        "status":     -1,
        "scenario_dispatch": [],
    }


# ── Arithmetic settlement against realized truth ──────────────────────────────

def settle_against_truth(plan, truth_day, state, sizing, CFG, day_dict):
    """Fix the day-ahead battery plan and settle against realized PV and load.

    Spec §7.1: open-loop settlement. Battery control is frozen; grid purchase
    and PV allocation absorb forecast errors.

    Returns: dict with realized hourly quantities and daily cost components.
    """
    CC  = sizing["CC"]
    P_B = sizing["P_B"]
    E_B = sizing["E_B"]

    eta_ch   = CFG["eff_charge"]
    eta_dis  = CFG["eff_discharge"]
    soc_min  = CFG["soc_min"]
    soc_max  = CFG["soc_max"]
    month    = day_dict["month_id"]
    c_basic  = get_monthly_basic_charge(month, CFG)
    oc_m10   = CFG["oc_within_10pct_mult"]
    oc_gt10  = CFG["oc_beyond_10pct_mult"]
    tou      = day_dict["tou"]

    P_ch_plan  = plan["P_ch_plan"]
    P_dis_plan = plan["P_dis_plan"]

    E_rep = np.zeros(24)
    E_init_rep = state["E_soc"]

    grid_rep   = np.zeros(24)
    pv_load_rep= np.zeros(24)
    pv_ch_rep  = np.zeros(24)
    pv_curt_rep= np.zeros(24)

    for t in range(24):
        h = t + 1   # hour_local is 1-based
        pv_real   = truth_day.get(h, {}).get("pv",   0.0)
        load_real = truth_day.get(h, {}).get("load", 0.0)

        P_ch_t  = float(P_ch_plan[t])
        P_dis_t = float(P_dis_plan[t])

        # Feasibility-restoring settlement (spec §7.1):
        # Battery plan is frozen; PV curtailment and grid import absorb deviations.
        # load_balance: P_grid_load + P_pv_load + P_dis = load_real
        # charging:     P_ch = P_grid_ch + P_pv_ch
        # pv_alloc:     P_pv_load + P_pv_ch + P_pv_curt = pv_real

        # First satisfy load with PV and discharge
        pv_to_load = min(pv_real, max(0.0, load_real - P_dis_t))
        pv_to_load = max(0.0, pv_to_load)
        pv_remaining = pv_real - pv_to_load

        # Use remaining PV for charging
        pv_to_ch = min(pv_remaining, P_ch_t)
        pv_to_ch = max(0.0, pv_to_ch)
        pv_curt  = pv_remaining - pv_to_ch

        grid_ch   = P_ch_t - pv_to_ch
        grid_load = max(0.0, load_real - pv_to_load - P_dis_t)
        grid_t    = grid_load + grid_ch

        pv_load_rep[t] = pv_to_load
        pv_ch_rep[t]   = pv_to_ch
        pv_curt_rep[t] = pv_curt
        grid_rep[t]    = grid_t

        # SOC update (physically clipped to bounds)
        if t == 0:
            E_t = E_init_rep + eta_ch * P_ch_t - (1.0 / eta_dis) * P_dis_t
        else:
            E_t = E_rep[t-1] + eta_ch * P_ch_t - (1.0 / eta_dis) * P_dis_t
        E_rep[t] = np.clip(E_t, soc_min * E_B, soc_max * E_B)

    # Daily realized energy cost
    ene_cost_rep = float(np.sum(tou * grid_rep))

    # Realized monthly demand for this day
    realized_peak_today = float(np.max(grid_rep))

    return {
        "P_ch_rep":   P_ch_plan.copy(),      # frozen from plan
        "P_dis_rep":  P_dis_plan.copy(),     # frozen from plan
        "P_grid_rep": grid_rep,
        "P_pv_load_rep": pv_load_rep,
        "P_pv_ch_rep":   pv_ch_rep,
        "P_pv_curt_rep": pv_curt_rep,
        "E_rep":          E_rep,
        "ene_cost_rep":   ene_cost_rep,
        "realized_peak_today": realized_peak_today,
    }


# ── State carry-forward ───────────────────────────────────────────────────────

def update_state(plan, settled, day_dict, next_day_dict, state):
    """Propagate sequential state from day i to day i+1.

    Spec §6.5.5:
    - E_soc carries from solve-world E_end (scenario-invariant)
    - E_g carries as scenario-weighted green SOC end
    - D_mth resets at month boundaries; otherwise carries D_cand forward
    """
    new_state = dict(state)
    new_state["E_soc"] = plan["E_end"]
    new_state["E_g"]   = plan["E_g_end"]

    if next_day_dict is not None:
        if next_day_dict["month_id"] != day_dict["month_id"]:
            new_state["D_mth"] = 0.0   # month boundary reset
        else:
            new_state["D_mth"] = plan["D_cand"]
    else:
        new_state["D_mth"] = 0.0   # year end

    return new_state


# ── Annual post-processing ────────────────────────────────────────────────────

def compute_annual_results(plan_trace, settlement_trace, state_trace,
                            sizing, CFG, day_data, day_indices):
    """Aggregate hourly traces into annual and monthly cost summaries."""
    CC  = sizing["CC"]
    P_B = sizing["P_B"]
    E_B = sizing["E_B"]
    crf = CFG["crf_bess"]
    c_bp = CFG["capex_bess_power_per_kw"]
    c_be = CFG["capex_bess_energy_per_kwh"]
    oc_m10  = CFG["oc_within_10pct_mult"]
    oc_gt10 = CFG["oc_beyond_10pct_mult"]
    c_trec  = CFG["trec_cost_per_kwh"]
    re_target = CFG["re_target"]

    AEC_inv = crf * (c_bp * P_B + c_be * E_B)

    # Monthly aggregation from settlement trace
    monthly = {}
    for di in day_indices:
        dd   = day_data[di]
        mo   = dd["month_id"]
        cd   = dd["calendar_day"]
        c_basic = get_monthly_basic_charge(mo, CFG)

        if mo not in monthly:
            monthly[mo] = {
                "month": mo,
                "ene_charge": 0.0,
                "max_grid":   0.0,
                "calendar_day_sample": cd,
            }
        monthly[mo]["ene_charge"] += settlement_trace[di]["ene_cost_rep"]

        # Track month-end peak (use final D_cand for each month)
        # We use the state trace D_mth_end for this day
        st = state_trace[di]
        if st["D_mth_end"] > monthly[mo]["max_grid"]:
            monthly[mo]["max_grid"] = st["D_mth_end"]

    month_rows = []
    for mo, m in sorted(monthly.items()):
        D_end = m["max_grid"]
        Over   = max(D_end - CC, 0.0)
        O1     = min(Over, 0.1 * CC)
        O2     = max(Over - 0.1 * CC, 0.0)
        c_basic = get_monthly_basic_charge(mo, CFG)
        over_charge  = c_basic * (oc_m10 * O1 + oc_gt10 * O2)
        basic_charge = c_basic * CC
        total_bill   = m["ene_charge"] + basic_charge + over_charge
        month_rows.append({
            "month":        mo,
            "ene_charge":   round(m["ene_charge"], 0),
            "basic_charge": round(basic_charge, 0),
            "over_charge":  round(over_charge, 0),
            "total_bill":   round(total_bill, 0),
            "peak_grid_kw": round(D_end, 1),
        })

    # Annual totals
    total_ene   = sum(r["ene_charge"]   for r in month_rows)
    total_basic = sum(r["basic_charge"] for r in month_rows)
    total_over  = sum(r["over_charge"]  for r in month_rows)
    worst_bill  = max(r["total_bill"]   for r in month_rows)

    # Annual RE20 (from settlement)
    total_pv_load   = sum(settlement_trace[di]["P_pv_load_rep"].sum()  for di in day_indices)
    total_load_real = sum(
        sum(plan_trace[di]["scenario_dispatch"][0]["prob"] * 0   # placeholder
            for _ in [0]) for di in day_indices
    )
    # Compute from settlement truth quantities directly
    total_load_truth = sum(
        settlement_trace[di]["P_grid_rep"].sum()
        + settlement_trace[di]["P_pv_load_rep"].sum()
        + settlement_trace[di]["P_pv_ch_rep"].sum()
        - settlement_trace[di]["P_ch_rep"].sum()
        for di in day_indices
    )
    # Simpler: load = grid + pv_to_load + dis - ch (energy balance simplified)
    # Use P_pv_load_rep + P_grid_rep + P_dis_rep - P_ch_rep as load proxy
    total_load_proxy = sum(
        np.sum(settlement_trace[di]["P_pv_load_rep"])
        + np.sum(settlement_trace[di]["P_pv_ch_rep"])
        + np.sum(settlement_trace[di]["P_pv_curt_rep"])
        for di in day_indices
    )  # = total PV
    total_pv = total_load_proxy

    # RE = PV self-consumed (pv_load) + green discharge — approximate with pv_load only
    re_pct = 100.0 * total_pv_load / max(total_load_truth, 1.0)

    # T-REC gap
    e_ren   = total_pv_load   # kWh (direct PV consumption)
    e_load  = max(total_load_truth, 1.0)
    trec_gap = max(re_target * e_load - e_ren, 0.0)
    C_trec  = c_trec * trec_gap

    total_op = total_ene + total_basic + total_over + C_trec
    total_all = total_op + AEC_inv

    # Solve-world daily obj sum
    da_solve_total = sum(plan_trace[di]["obj_val"] for di in day_indices)

    return {
        "month_rows":      month_rows,
        "total_ene_M":     round(total_ene   / 1e6, 4),
        "total_basic_M":   round(total_basic / 1e6, 4),
        "total_over_M":    round(total_over  / 1e6, 4),
        "total_trec_M":    round(C_trec      / 1e6, 4),
        "total_inv_M":     round(AEC_inv     / 1e6, 4),
        "total_op_M":      round(total_op    / 1e6, 4),
        "total_all_M":     round(total_all   / 1e6, 4),
        "worst_month_M":   round(worst_bill  / 1e6, 4),
        "re_pct":          round(re_pct, 2),
        "da_solve_total_M": round(da_solve_total / 1e6, 4),
        "n_over_months":   sum(1 for r in month_rows if r["over_charge"] > 0),
    }


# ── Write output files ────────────────────────────────────────────────────────

def write_outputs(case_id, plan_trace, settlement_trace, state_trace, solve_log,
                  annual, sizing, day_data, day_indices):
    """Write daily traces, solve log, monthly bills, and summary JSON for one case."""
    OUT_DA.mkdir(parents=True, exist_ok=True)

    # daily_plan_trace
    plan_rows = []
    for di in day_indices:
        pt = plan_trace[di]
        dd = day_data[di]
        for t in range(24):
            plan_rows.append({
                "day_index":    di,
                "calendar_day": dd["calendar_day"].date(),
                "hour_local":   t + 1,
                "u_plan":       int(round(pt["u_plan"][t])),
                "P_ch_plan_kw": round(float(pt["P_ch_plan"][t]), 3),
                "P_dis_plan_kw":round(float(pt["P_dis_plan"][t]), 3),
                "E_plan_kwh":   round(float(pt["E_plan"][t]), 2),
                "obj_day":      round(float(pt["obj_val"]), 2) if t == 0 else None,
                "solve_time_s": round(float(pt["solve_time"]), 3) if t == 0 else None,
                "solve_status": pt["status"] if t == 0 else None,
            })
    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_parquet(OUT_DA / f"daily_plan_trace_{case_id}.parquet", index=False)
    print(f"    daily_plan_trace_{case_id}.parquet: {len(plan_df)} rows")

    # daily_settlement_trace
    stl_rows = []
    for di in day_indices:
        st = settlement_trace[di]
        dd = day_data[di]
        for t in range(24):
            stl_rows.append({
                "day_index":         di,
                "calendar_day":      dd["calendar_day"].date(),
                "hour_local":        t + 1,
                "P_grid_rep_kw":     round(float(st["P_grid_rep"][t]),   3),
                "P_pv_load_rep_kw":  round(float(st["P_pv_load_rep"][t]),3),
                "P_pv_ch_rep_kw":    round(float(st["P_pv_ch_rep"][t]),  3),
                "P_pv_curt_rep_kw":  round(float(st["P_pv_curt_rep"][t]),3),
                "P_ch_rep_kw":       round(float(st["P_ch_rep"][t]),     3),
                "P_dis_rep_kw":      round(float(st["P_dis_rep"][t]),    3),
                "E_rep_kwh":         round(float(st["E_rep"][t]),        2),
                "ene_cost_rep":      round(float(st["ene_cost_rep"]), 2) if t == 0 else None,
            })
    stl_df = pd.DataFrame(stl_rows)
    stl_df.to_parquet(OUT_DA / f"daily_settlement_trace_{case_id}.parquet", index=False)
    print(f"    daily_settlement_trace_{case_id}.parquet: {len(stl_df)} rows")

    # rolling_state_trace
    state_rows = []
    for di in day_indices:
        s = state_trace[di]
        dd = day_data[di]
        state_rows.append({
            "day_index":     di,
            "calendar_day":  dd["calendar_day"].date(),
            "E_soc_start_kwh": round(s["E_soc_start"], 2),
            "E_soc_end_kwh":   round(s["E_soc_end"],   2),
            "E_g_start_kwh":   round(s["E_g_start"],   2),
            "E_g_end_kwh":     round(s["E_g_end"],     2),
            "D_mth_start_kw":  round(s["D_mth_start"], 1),
            "D_mth_end_kw":    round(s["D_mth_end"],   1),
        })
    state_df = pd.DataFrame(state_rows)
    state_df.to_parquet(OUT_DA / f"rolling_state_trace_{case_id}.parquet", index=False)
    print(f"    rolling_state_trace_{case_id}.parquet: {len(state_df)} rows")

    # MODULE 4 evidence: daily solve log (gate metadata + solve stats per day)
    log_df = pd.DataFrame(solve_log)
    log_df.to_csv(OUT_DA / f"da_daily_solve_log_{case_id}.csv", index=False)
    print(f"    da_daily_solve_log_{case_id}.csv: {len(log_df)} rows")

    # Monthly bill CSV
    bills_df = pd.DataFrame(annual["month_rows"])
    bills_df.to_csv(OUT_DA / f"replay_monthly_bill_{case_id}.csv", index=False)
    print(f"    replay_monthly_bill_{case_id}.csv")

    # Summary JSON
    summary = {
        "case_id":          case_id,
        "CC_kw":            sizing["CC"],
        "P_B_kw":           sizing["P_B"],
        "E_B_kwh":          sizing["E_B"],
        "layer_a_source":   "case_summary_fullyear.csv",
        "replay_total_M":   annual["total_all_M"],
        "replay_ene_M":     annual["total_ene_M"],
        "replay_basic_M":   annual["total_basic_M"],
        "replay_over_M":    annual["total_over_M"],
        "replay_trec_M":    annual["total_trec_M"],
        "replay_inv_M":     annual["total_inv_M"],
        "replay_op_M":      annual["total_op_M"],
        "worst_month_M":    annual["worst_month_M"],
        "re_pct":           annual["re_pct"],
        "da_solve_total_M": annual["da_solve_total_M"],
        "n_over_months":    annual["n_over_months"],
    }
    with open(OUT_DA / f"replay_summary_{case_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    replay_summary_{case_id}.json")

    return summary


# ── Main sequential loop per case ─────────────────────────────────────────────

def run_case(case, CFG, da_time_limit=60, da_mip_gap=1e-3, verbose=True):
    """Run the full sequential day-ahead MILP + replay for one case."""
    case_id = case["case_id"]
    print(f"\n{'='*60}")
    print(f"Layer B — Day-Ahead Sequential MILP: {case_id} ({case['label']})")
    print(f"{'='*60}")

    # Load sizing from Layer A
    sizing = load_layer_a_sizing(case_id)
    print(f"  Layer A sizing: CC={sizing['CC']:.1f} kW, P_B={sizing['P_B']:.1f} kW, E_B={sizing['E_B']:.1f} kWh")

    # Load rolling day-ahead package
    day_data, day_indices = load_rolling_da_package(case)

    # Load truth
    truth_pkg = load_truth_package()
    truth_by_day = {}
    for (di, hl), vals in truth_pkg.items():
        if di not in truth_by_day:
            truth_by_day[di] = {}
        truth_by_day[di][hl] = vals

    # Initialize state (spec §6.5.5: E_init(1) = SOC_init * E_B)
    E_B   = sizing["E_B"]
    state = {
        "E_soc": CFG["soc_init"] * E_B,
        "E_g":   0.0,
        "D_mth": 0.0,
    }

    plan_trace       = {}
    settlement_trace = {}
    state_trace      = {}
    solve_log        = []   # MODULE 4 evidence: one row per day-ahead solve

    n_days = len(day_indices)
    t_case_start = time.time()

    for idx, di in enumerate(day_indices):
        dd = day_data[di]
        is_final = (idx == n_days - 1)
        next_dd  = day_data[day_indices[idx + 1]] if idx < n_days - 1 else None

        if verbose and (idx % 30 == 0 or is_final):
            elapsed = time.time() - t_case_start
            print(f"  Day {di:3d}/{n_days}  ({dd['calendar_day'].date()})  "
                  f"E_soc={state['E_soc']:.0f} kWh  D_mth={state['D_mth']:.0f} kW  "
                  f"gate={dd['issue_day']}@{dd['gate_time_local']}  [{elapsed:.0f}s elapsed]")

        # Record state at start of this day
        state_snap_start = dict(state)

        # ── MODULE 4a: SOLVE day-ahead MILP ──────────────────────────
        # Uses ONLY Module 3 forecast package for this day.
        # Realized truth is NOT accessible here.
        plan = solve_day_ahead(
            dd, state, sizing, CFG, di,
            is_final_day=is_final,
            time_limit=da_time_limit,
            mip_gap=da_mip_gap,
        )

        # ── MODULE 4b: SETTLE against realized truth (arithmetic) ────
        # Battery control is FROZEN from the solve step above.
        # Truth is used here ONLY for energy balance settlement.
        settled = settle_against_truth(
            plan, truth_by_day.get(di, {}), state, sizing, CFG, dd
        )

        # ── MODULE 4c: STATE CARRY-FORWARD ───────────────────────────
        new_state = update_state(plan, settled, dd, next_dd, state)

        # Record
        plan_trace[di]       = plan
        settlement_trace[di] = settled
        state_trace[di] = {
            "E_soc_start": state_snap_start["E_soc"],
            "E_soc_end":   plan["E_end"],
            "E_g_start":   state_snap_start["E_g"],
            "E_g_end":     plan["E_g_end"],
            "D_mth_start": state_snap_start["D_mth"],
            "D_mth_end":   plan["D_cand"],
        }

        # Daily solve log entry (gate-time metadata + solve stats)
        solve_log.append({
            "day_index":        di,
            "calendar_day":     str(dd["calendar_day"].date()),
            "issue_day":        dd["issue_day"],          # D-1 gate date
            "gate_time_local":  dd["gate_time_local"],    # "20:00"
            "source_package":   dd["source_package"],
            "n_scenarios":      len(dd["scenarios"]),
            "is_final_day":     is_final,
            "month_id":         dd["month_id"],
            "E_soc_start_kwh":  round(state_snap_start["E_soc"], 2),
            "D_mth_start_kw":   round(state_snap_start["D_mth"], 1),
            "solve_status":     plan["status"],
            "obj_val":          round(float(plan["obj_val"]), 2),
            "solve_time_s":     round(float(plan["solve_time"]), 3),
            "E_soc_end_kwh":    round(float(plan["E_end"]), 2),
            "D_cand_kw":        round(float(plan["D_cand"]), 1),
            "ene_cost_rep":     round(float(settled["ene_cost_rep"]), 2),
        })

        state = new_state

    t_case_elapsed = time.time() - t_case_start
    print(f"\n  Sequential solve complete: {t_case_elapsed:.1f}s total for {n_days} days")

    # Annual aggregation
    annual = compute_annual_results(
        plan_trace, settlement_trace, state_trace, sizing, CFG, day_data, day_indices
    )

    print(f"  Annual replay (operational): {annual['total_op_M']:.3f} M NTD")
    print(f"  Annual replay (with inv):    {annual['total_all_M']:.3f} M NTD")
    print(f"  Worst month bill:            {annual['worst_month_M']:.3f} M NTD")
    print(f"  RE%:                         {annual['re_pct']:.1f}%")

    # Write outputs
    print(f"\n  Writing outputs → milp_outputs/da/")
    summary = write_outputs(case_id, plan_trace, settlement_trace, state_trace, solve_log,
                             annual, sizing, day_data, day_indices)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Layer B: Sequential Day-Ahead Daily MILP")
    parser.add_argument("--cases", nargs="+", default=["C0","C1","C2","C3"],
                        choices=["C0","C1","C2","C3"],
                        help="Cases to run (default: all)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Gurobi time limit per daily MILP in seconds (default: 60)")
    parser.add_argument("--mip-gap",   type=float, default=1e-3,
                        help="MIP gap tolerance (default: 0.001)")
    args = parser.parse_args()

    CFG = get_config()

    case_map = {c["case_id"]: c for c in CASE_TABLE}
    summaries = []

    for case_id in args.cases:
        case = case_map[case_id]
        summary = run_case(case, CFG,
                           da_time_limit=args.time_limit,
                           da_mip_gap=args.mip_gap)
        summaries.append(summary)

    # Aggregate comparison table
    if summaries:
        comp_df = pd.DataFrame(summaries)
        comp_path = OUT_DA / "da_case_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\n{'='*60}")
        print("Day-Ahead MILP — Case Comparison")
        print(f"{'='*60}")
        print(comp_df[["case_id","CC_kw","P_B_kw","E_B_kwh",
                        "replay_total_M","replay_over_M","re_pct","worst_month_M"]].to_string(index=False))
        print(f"\nFull results in: milp_outputs/da/")

    # Write manifest
    manifest = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases_run": args.cases,
        "da_time_limit_s": args.time_limit,
        "da_mip_gap": args.mip_gap,
        "layer_a_source": "milp_outputs/case_summary_fullyear.csv",
        "bridge_source": "bridge_outputs_fullyear/rolling_da_input_*.parquet",
        "schema": "rolling_da_v1",
        "summaries": summaries,
    }
    with open(OUT_DA / "da_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
