"""
milp_v2/layer_b/milp_daily.py
Sequential day-ahead MILP for Layer B.

Logic preserved from thesis_pipeline (milp_da_sequential.py):
  - 24h day-ahead MILP with frozen design (CC*, PB*, EB*)
  - Scenario-invariant battery control (u, P_ch, P_dis shared across scenarios)
  - Incremental peak-cost proxy (D_init, D_cand updated daily)
  - Year-end terminal SOC policy (final day only — return to soc_init band)
  - Green SOC cross-day carry: E_g_init = weighted-average E_g[end-of-day]
  - Monthly demand state accumulated across days

Input source is controlled by the caller via the day_dict["scenarios"] structure.
"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_basic_charge

CFG = load_config()


def solve_day_ahead(
    day_index:    int,
    day_dict:     dict,
    state:        dict,
    design:       dict,
    cfg:          dict = None,
    is_final_day: bool = False,
    time_limit:   int = 60,
    mip_gap:      float = 0.005,
    verbose:      bool = False,
) -> dict:
    """
    Solve a single day-ahead MILP for Layer B.

    Args:
        day_index:    calendar day index (for logging)
        day_dict:     {scenarios, tou, month_id, calendar_day}
                      scenarios: list of {id, pi, pv[24], load[24]}
        state:        {E_soc: float, E_g: float, D_mth: float}
                      E_soc  — carry-in SOC [kWh]
                      E_g    — carry-in green SOC [kWh] (weighted avg from prev day)
                      D_mth  — month-to-date demand peak [kW]
        design:       {CC: float, PB: float, EB: float} — frozen from Layer A
        cfg:          config dict (uses global CFG if None)
        is_final_day: if True, apply tight year-end terminal SOC constraint
        time_limit:   Gurobi time limit [s]
        mip_gap:      Gurobi MIP gap

    Returns:
        plan dict with P_ch_plan, P_dis_plan, D_cand, E_end, E_g_end, etc.
    """
    if cfg is None:
        cfg = CFG

    bat = cfg["battery"]
    dem = cfg["demand"]

    CC     = design["CC"]
    PB     = design["PB"]
    EB     = design["EB"]

    E_init  = float(state["E_soc"])
    Eg_init = float(state["E_g"])
    D_init  = float(state["D_mth"])

    eta_ch   = bat["eta_ch"]
    eta_dis  = bat["eta_dis"]
    soc_min  = bat["soc_min"]
    soc_max  = bat["soc_max"]
    soc_init = bat["soc_init"]
    eps_term = bat["epsilon_term"]
    b_k      = bat["degradation_segments"]["b"]
    lam_k    = bat["degradation_segments"]["lambda"]
    n_seg    = len(lam_k)

    kappa = dem["kappa"]
    m1    = dem["over_contract_m1"]
    m2    = dem["over_contract_m2"]
    tier1 = dem["over_contract_tier1_ratio"]

    scenarios = day_dict["scenarios"]
    tou       = day_dict["tou"]       # [24] NTD/kWh
    month_id  = day_dict["month_id"]
    c_basic   = get_basic_charge(month_id, cfg)
    n_sc      = len(scenarios)
    H         = 24
    T         = range(H)
    W         = range(n_sc)

    t_start = time.time()

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag",  1 if verbose else 0)
        env.setParam("TimeLimit",   time_limit)
        env.setParam("MIPGap",      mip_gap)
        env.start()
        mdl = gp.Model(env=env)

        # ── Scenario-invariant battery control ────────────────────────────────
        u     = mdl.addVars(H, vtype=GRB.BINARY,                  name="u")
        P_ch  = mdl.addVars(H, lb=0.0,                            name="P_ch")
        P_dis = mdl.addVars(H, lb=0.0,                            name="P_dis")
        E     = mdl.addVars(H, lb=soc_min*EB, ub=soc_max*EB,     name="E")
        e_seg = mdl.addVars(H, n_seg, lb=0.0,                     name="e_seg")

        # ── Scenario-dependent settlement ─────────────────────────────────────
        P_gl   = mdl.addVars(n_sc, H, lb=0.0, name="P_gl")    # grid -> load
        P_gc   = mdl.addVars(n_sc, H, lb=0.0, name="P_gc")    # grid -> charge
        P_pvl  = mdl.addVars(n_sc, H, lb=0.0, name="P_pvl")   # pv -> load
        P_pvc  = mdl.addVars(n_sc, H, lb=0.0, name="P_pvc")   # pv -> charge
        P_pvcu = mdl.addVars(n_sc, H, lb=0.0, name="P_pvcu")  # pv curtailed

        # ── Green SOC per scenario ─────────────────────────────────────────────
        E_g    = mdl.addVars(n_sc, H, lb=0.0, ub=EB, name="E_g")
        P_chg  = mdl.addVars(n_sc, H, lb=0.0,        name="P_chg")   # green charge
        P_disg = mdl.addVars(n_sc, H, lb=0.0,        name="P_disg")  # green discharge

        # ── Monthly demand proxy (incremental form) ────────────────────────────
        # D_cand >= D_init enforced by lb
        D_cand    = mdl.addVar(lb=D_init,    name="D_cand")
        Over_full = mdl.addVar(lb=0.0,       name="Over_full")
        O1_full   = mdl.addVar(lb=0.0, ub=tier1 * CC, name="O1_full")
        O2_full   = mdl.addVar(lb=0.0,       name="O2_full")

        # ── Constraint (1): battery limits ────────────────────────────────────
        for t in T:
            mdl.addConstr(P_ch[t]  <= u[t] * PB,        name=f"ch_ub_{t}")
            mdl.addConstr(P_dis[t] <= (1-u[t]) * PB,    name=f"dis_ub_{t}")

        # ── Constraint (2): SOC recursion ─────────────────────────────────────
        for t in T:
            E_prev = E_init if t == 0 else E[t-1]
            mdl.addConstr(
                E[t] == E_prev + eta_ch * P_ch[t] - (1.0/eta_dis) * P_dis[t],
                name=f"soc_{t}"
            )
        # SOC bounds already set as lb/ub on E

        # ── Constraint (3): year-end terminal SOC (final day only) ──────────
        # Only applied on the last day of the year so the battery returns to
        # the initial SOC band.  Applying every day was over-constraining the
        # prob-scheduling cases (DP/PP) and inflating over-contract costs.
        if is_final_day:
            E_term_lb = (soc_init - eps_term) * EB
            E_term_ub = (soc_init + eps_term) * EB
            mdl.addConstr(E[H-1] >= E_term_lb, name="term_lb")
            mdl.addConstr(E[H-1] <= E_term_ub, name="term_ub")

        # ── Constraint (4): degradation segments ──────────────────────────────
        for t in T:
            mdl.addConstr(
                P_dis[t] == gp.quicksum(e_seg[t,k] for k in range(n_seg)),
                name=f"seg_sum_{t}"
            )
            for k in range(n_seg):
                cap_k = (b_k[k+1] - b_k[k]) * EB
                mdl.addConstr(e_seg[t,k] <= cap_k, name=f"seg_cap_{t}_{k}")

        # ── Constraint (5): scenario settlement ───────────────────────────────
        for w, sc in enumerate(scenarios):
            pv_w   = sc["pv"]    # [24]
            load_w = sc["load"]  # [24]
            for t in T:
                # Load balance
                mdl.addConstr(
                    P_gl[w,t] + P_pvl[w,t] + P_dis[t] == load_w[t],
                    name=f"Lbal_{w}_{t}"
                )
                # Charge balance
                mdl.addConstr(
                    P_ch[t] == P_gc[w,t] + P_pvc[w,t],
                    name=f"Cbal_{w}_{t}"
                )
                # PV balance
                mdl.addConstr(
                    P_pvl[w,t] + P_pvc[w,t] + P_pvcu[w,t] == pv_w[t],
                    name=f"PVbal_{w}_{t}"
                )

        # ── Constraint (6): monthly demand proxy — per-scenario robust ────────
        # For K=1 (det scheduling): per-scenario == expectation, no difference.
        # For K>1 (prob scheduling): D_cand must cover every scenario's grid
        # peak, not just the weighted average, so probabilistic scheduling
        # explicitly protects against all forecast scenarios.
        for w in W:
            for t in T:
                mdl.addConstr(
                    D_cand >= kappa * (P_gl[w,t] + P_gc[w,t]),
                    name=f"Dcand_{w}_{t}"
                )
        mdl.addConstr(Over_full >= D_cand - CC,            name="over_pos")
        mdl.addConstr(Over_full == O1_full + O2_full,      name="over_split")
        # O1_full ub already set via ub=tier1*CC

        # ── Constraint (7): Green SOC per scenario (式 39) ────────────────────
        for w, sc in enumerate(scenarios):
            for t in T:
                Eg_prev = Eg_init if t == 0 else E_g[w, t-1]
                mdl.addConstr(
                    E_g[w,t] == Eg_prev
                        + eta_ch * P_chg[w,t]
                        - (1.0/eta_dis) * P_disg[w,t],
                    name=f"Eg_rec_{w}_{t}"
                )
                mdl.addConstr(E_g[w,t]   <= E[t],         name=f"Eg_ub_{w}_{t}")
                mdl.addConstr(P_chg[w,t]  <= P_pvc[w,t],  name=f"Pcg_ub_{w}_{t}")
                mdl.addConstr(P_disg[w,t] <= P_dis[t],    name=f"Pdg_ub_{w}_{t}")

        # ── Objective ──────────────────────────────────────────────────────────
        # Energy cost (expected)
        C_ene = gp.quicksum(
            sc["pi"] * tou[t] * (P_gl[w,t] + P_gc[w,t])
            for w, sc in enumerate(scenarios) for t in T
        )

        # Degradation cost
        C_deg = gp.quicksum(
            lam_k[k] * e_seg[t,k] for t in T for k in range(n_seg)
        )

        # Incremental demand-charge proxy (spec §6.5.9)
        Over_prev = max(D_init - CC, 0.0)
        O1_prev   = min(Over_prev, tier1 * CC)
        O2_prev   = max(Over_prev - tier1 * CC, 0.0)
        PeakCost_prev = c_basic * (m1 * O1_prev + m2 * O2_prev)
        PeakCost_obj  = c_basic * (m1 * O1_full + m2 * O2_full)
        C_peak = PeakCost_obj - PeakCost_prev   # incremental cost delta

        mdl.setObjective(C_ene + C_deg + C_peak, GRB.MINIMIZE)
        mdl.optimize()

        solve_time = time.time() - t_start
        status     = mdl.Status

        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mdl.SolCount == 0:
            return _fallback_plan(E_init, EB, soc_init, H, n_sc,
                                  scenarios, day_index, solve_time)

        # ── Extract plan ──────────────────────────────────────────────────────
        P_ch_plan  = np.array([P_ch[t].X  for t in T])
        P_dis_plan = np.array([P_dis[t].X for t in T])
        E_plan     = np.array([E[t].X     for t in T])
        u_plan     = np.array([u[t].X     for t in T])
        E_end      = float(E[H-1].X)

        # Green SOC end-of-day: scenario-weighted average (for cross-day carry)
        E_g_end = float(sum(
            sc["pi"] * float(E_g[w, H-1].X)
            for w, sc in enumerate(scenarios)
        ))

        D_cand_val = float(D_cand.X)
        obj_val    = float(mdl.ObjVal)

        # Scenario dispatch log (for downstream settle)
        sc_rows = []
        for w, sc in enumerate(scenarios):
            for t in T:
                sc_rows.append({
                    "scenario_id":   sc["id"],
                    "hour_local":    t + 1,
                    "pi":            sc["pi"],
                    "P_grid_kw":     float(P_gl[w,t].X + P_gc[w,t].X),
                    "P_pv_load_kw":  float(P_pvl[w,t].X),
                    "P_pv_ch_kw":    float(P_pvc[w,t].X),
                    "P_pv_curt_kw":  float(P_pvcu[w,t].X),
                    "E_g_kwh":       float(E_g[w,t].X),
                })

        return {
            "u_plan":            u_plan,
            "P_ch_plan":         P_ch_plan,
            "P_dis_plan":        P_dis_plan,
            "E_plan":            E_plan,
            "E_end":             E_end,
            "E_g_end":           E_g_end,
            "D_cand":            D_cand_val,
            "obj_val":           obj_val,
            "solve_time":        solve_time,
            "status":            status,
            "scenario_dispatch": sc_rows,
        }


def _fallback_plan(E_init, EB, soc_init, H, n_sc, scenarios, day_index, solve_time):
    """Do-nothing plan returned on MILP failure."""
    print(f"  WARNING: day {day_index} MILP failed — using do-nothing plan")
    return {
        "u_plan":            np.ones(H),
        "P_ch_plan":         np.zeros(H),
        "P_dis_plan":        np.zeros(H),
        "E_plan":            np.full(H, E_init),
        "E_end":             E_init,
        "E_g_end":           0.0,
        "D_cand":            0.0,
        "obj_val":           0.0,
        "solve_time":        solve_time,
        "status":            -1,
        "scenario_dispatch": [],
    }
