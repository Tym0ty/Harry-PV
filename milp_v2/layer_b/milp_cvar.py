"""
milp_v2/layer_b/milp_cvar.py
CVaR-augmented Layer B MILP — Phase 1B extension of milp_daily.py.

Changes from milp_daily.py:
  - D_cand, Over_full, O1_full, O2_full become per-scenario variables
  - CVaR linearization (Rockafellar-Uryasev) added over scenario-level C_oc
  - Objective: E[C_oc] + lambda * CVaR_alpha(C_oc) (+ C_ene + C_deg unchanged)
  - lam=0.0 reduces to pure expected-cost (≈ milp_daily for K=1 exactly)
  - Return dict extended: D_cand_by_scenario, C_oc_by_scenario, eta_val, cvar_val, E_coc_val
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
    time_limit:   int  = 60,
    mip_gap:      float = 0.005,
    verbose:      bool = False,
    lam:          float = 0.0,   # CVaR penalty weight (0 = expected cost only)
    alpha:        float = 0.90,  # CVaR tail probability
) -> dict:
    """
    Solve a single day-ahead MILP for Layer B with CVaR risk penalty.

    Args:
        day_index:    calendar day index (for logging)
        day_dict:     {scenarios, tou, month_id, calendar_day}
        state:        {E_soc, E_g, D_mth}
        design:       {CC, PB, EB}
        cfg:          config dict (uses global CFG if None)
        is_final_day: apply year-end terminal SOC constraint
        time_limit:   Gurobi time limit [s]
        mip_gap:      Gurobi MIP gap
        lam:          CVaR penalty weight lambda >= 0
        alpha:        CVaR confidence level (e.g. 0.90 = 90th percentile)

    Returns:
        plan dict (superset of milp_daily return dict, with CVaR fields appended)
    """
    if cfg is None:
        cfg = CFG

    bat = cfg["battery"]
    dem = cfg["demand"]

    CC = design["CC"]
    PB = design["PB"]
    EB = design["EB"]

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
    tou       = day_dict["tou"]
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
        u     = mdl.addVars(H, vtype=GRB.BINARY,              name="u")
        P_ch  = mdl.addVars(H, lb=0.0,                        name="P_ch")
        P_dis = mdl.addVars(H, lb=0.0,                        name="P_dis")
        E     = mdl.addVars(H, lb=soc_min*EB, ub=soc_max*EB, name="E")
        e_seg = mdl.addVars(H, n_seg, lb=0.0,                 name="e_seg")

        # ── Scenario-dependent settlement ─────────────────────────────────────
        P_gl   = mdl.addVars(n_sc, H, lb=0.0, name="P_gl")
        P_gc   = mdl.addVars(n_sc, H, lb=0.0, name="P_gc")
        P_pvl  = mdl.addVars(n_sc, H, lb=0.0, name="P_pvl")
        P_pvc  = mdl.addVars(n_sc, H, lb=0.0, name="P_pvc")
        P_pvcu = mdl.addVars(n_sc, H, lb=0.0, name="P_pvcu")

        # ── Green SOC per scenario ─────────────────────────────────────────────
        E_g   = mdl.addVars(n_sc, H, lb=0.0, ub=EB, name="E_g")
        P_chg = mdl.addVars(n_sc, H, lb=0.0,        name="P_chg")
        P_disg= mdl.addVars(n_sc, H, lb=0.0,        name="P_disg")

        # ── Per-scenario demand proxy (CVaR extension) ────────────────────────
        # Each scenario has its own D_cand, Over_full, O1_full, O2_full.
        # lam=0 + K=1: equivalent to milp_daily (single scenario).
        D_cand    = mdl.addVars(n_sc, lb=D_init,             name="D_cand")
        Over_full = mdl.addVars(n_sc, lb=0.0,                name="Over_full")
        O1_full   = mdl.addVars(n_sc, lb=0.0, ub=tier1*CC,  name="O1_full")
        O2_full   = mdl.addVars(n_sc, lb=0.0,                name="O2_full")

        # ── CVaR variables (Rockafellar-Uryasev, 2000) ───────────────────────
        # CVaR_alpha(C_oc) = eta + 1/(1-alpha) * E[max(C_oc - eta, 0)]
        # Linearized: xi[w] >= C_oc[w] - eta, xi[w] >= 0
        eta = mdl.addVar(lb=-1e8, ub=1e8, name="eta")
        xi  = mdl.addVars(n_sc, lb=0.0,  name="xi")

        # ── Constraint (1): battery limits ────────────────────────────────────
        for t in T:
            mdl.addConstr(P_ch[t]  <= u[t] * PB,     name=f"ch_ub_{t}")
            mdl.addConstr(P_dis[t] <= (1-u[t]) * PB, name=f"dis_ub_{t}")

        # ── Constraint (2): SOC recursion ─────────────────────────────────────
        for t in T:
            E_prev = E_init if t == 0 else E[t-1]
            mdl.addConstr(
                E[t] == E_prev + eta_ch * P_ch[t] - (1.0/eta_dis) * P_dis[t],
                name=f"soc_{t}"
            )

        # ── Constraint (3): year-end terminal SOC ────────────────────────────
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
            pv_w   = sc["pv"]
            load_w = sc["load"]
            for t in T:
                mdl.addConstr(
                    P_gl[w,t] + P_pvl[w,t] + P_dis[t] == load_w[t],
                    name=f"Lbal_{w}_{t}"
                )
                mdl.addConstr(
                    P_ch[t] == P_gc[w,t] + P_pvc[w,t],
                    name=f"Cbal_{w}_{t}"
                )
                mdl.addConstr(
                    P_pvl[w,t] + P_pvc[w,t] + P_pvcu[w,t] == pv_w[t],
                    name=f"PVbal_{w}_{t}"
                )

        # ── Constraint (6): per-scenario demand proxy ─────────────────────────
        for w in W:
            for t in T:
                mdl.addConstr(
                    D_cand[w] >= kappa * (P_gl[w,t] + P_gc[w,t]),
                    name=f"Dcand_{w}_{t}"
                )
            mdl.addConstr(Over_full[w] >= D_cand[w] - CC,       name=f"over_pos_{w}")
            mdl.addConstr(Over_full[w] == O1_full[w] + O2_full[w], name=f"over_split_{w}")
            # O1_full[w] ub already set via ub=tier1*CC

        # ── Constraint (7): Green SOC per scenario ────────────────────────────
        for w, sc in enumerate(scenarios):
            for t in T:
                Eg_prev = Eg_init if t == 0 else E_g[w, t-1]
                mdl.addConstr(
                    E_g[w,t] == Eg_prev
                        + eta_ch * P_chg[w,t]
                        - (1.0/eta_dis) * P_disg[w,t],
                    name=f"Eg_rec_{w}_{t}"
                )
                mdl.addConstr(E_g[w,t]   <= E[t],        name=f"Eg_ub_{w}_{t}")
                mdl.addConstr(P_chg[w,t]  <= P_pvc[w,t], name=f"Pcg_ub_{w}_{t}")
                mdl.addConstr(P_disg[w,t] <= P_dis[t],   name=f"Pdg_ub_{w}_{t}")

        # ── Constraint (8): CVaR linearization ───────────────────────────────
        # xi[w] >= C_oc[w] - eta for each scenario w
        # C_oc[w] = c_basic * (m1*O1_full[w] + m2*O2_full[w])
        for w in W:
            C_oc_w = c_basic * (m1 * O1_full[w] + m2 * O2_full[w])
            mdl.addConstr(xi[w] >= C_oc_w - eta, name=f"cvar_xi_{w}")

        # ── Objective ──────────────────────────────────────────────────────────
        C_ene = gp.quicksum(
            sc["pi"] * tou[t] * (P_gl[w,t] + P_gc[w,t])
            for w, sc in enumerate(scenarios) for t in T
        )

        C_deg = gp.quicksum(
            lam_k[k] * e_seg[t,k] for t in T for k in range(n_seg)
        )

        # Incremental over-contract cost (expected over scenarios)
        Over_prev    = max(D_init - CC, 0.0)
        O1_prev      = min(Over_prev, tier1 * CC)
        O2_prev      = max(Over_prev - tier1 * CC, 0.0)
        PeakCost_prev = c_basic * (m1 * O1_prev + m2 * O2_prev)

        E_C_oc = gp.quicksum(
            sc["pi"] * c_basic * (m1 * O1_full[w] + m2 * O2_full[w])
            for w, sc in enumerate(scenarios)
        )
        C_peak_exp = E_C_oc - PeakCost_prev

        # CVaR term: eta + 1/(1-alpha) * sum_w pi_w * xi_w
        C_cvar = eta + (1.0 / (1.0 - alpha)) * gp.quicksum(
            sc["pi"] * xi[w] for w, sc in enumerate(scenarios)
        )

        mdl.setObjective(C_ene + C_deg + C_peak_exp + lam * C_cvar, GRB.MINIMIZE)
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

        E_g_end = float(sum(
            sc["pi"] * float(E_g[w, H-1].X)
            for w, sc in enumerate(scenarios)
        ))

        # Per-scenario D_cand and C_oc
        D_cand_by_sc = [float(D_cand[w].X) for w in W]
        C_oc_by_sc   = [
            c_basic * (m1 * float(O1_full[w].X) + m2 * float(O2_full[w].X))
            for w in W
        ]

        # Expected D_cand (pi-weighted) — used for state carry-forward
        D_cand_exp = float(sum(
            sc["pi"] * D_cand_by_sc[w]
            for w, sc in enumerate(scenarios)
        ))

        eta_val  = float(eta.X)
        xi_vals  = [float(xi[w].X) for w in W]
        E_coc_val = float(sum(sc["pi"] * C_oc_by_sc[w] for w, sc in enumerate(scenarios)))
        cvar_val = eta_val + (1.0 / (1.0 - alpha)) * sum(
            sc["pi"] * xi_vals[w] for w, sc in enumerate(scenarios)
        )

        obj_val = float(mdl.ObjVal)

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
            "u_plan":              u_plan,
            "P_ch_plan":           P_ch_plan,
            "P_dis_plan":          P_dis_plan,
            "E_plan":              E_plan,
            "E_end":               E_end,
            "E_g_end":             E_g_end,
            "D_cand":              D_cand_exp,        # pi-weighted expected (for state carry)
            "D_cand_by_scenario":  D_cand_by_sc,      # per-scenario list
            "C_oc_by_scenario":    C_oc_by_sc,        # per-scenario over-contract cost [NTD]
            "eta_val":             eta_val,            # CVaR VaR threshold
            "cvar_val":            cvar_val,           # CVaR_alpha(C_oc)
            "E_coc_val":           E_coc_val,          # E[C_oc]
            "obj_val":             obj_val,
            "solve_time":          solve_time,
            "status":              status,
            "scenario_dispatch":   sc_rows,
        }


def _fallback_plan(E_init, EB, soc_init, H, n_sc, scenarios, day_index, solve_time):
    """Do-nothing plan returned on MILP failure."""
    print(f"  WARNING: day {day_index} MILP failed — using do-nothing plan")
    return {
        "u_plan":              np.ones(H),
        "P_ch_plan":           np.zeros(H),
        "P_dis_plan":          np.zeros(H),
        "E_plan":              np.full(H, E_init),
        "E_end":               E_init,
        "E_g_end":             0.0,
        "D_cand":              0.0,
        "D_cand_by_scenario":  [0.0] * n_sc,
        "C_oc_by_scenario":    [0.0] * n_sc,
        "eta_val":             0.0,
        "cvar_val":            0.0,
        "E_coc_val":           0.0,
        "obj_val":             0.0,
        "solve_time":          solve_time,
        "status":              -1,
        "scenario_dispatch":   [],
    }
