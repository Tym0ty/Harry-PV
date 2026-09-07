"""
milp_v2/layer_b/milp_mpc.py
Shrinking-horizon MPC MILP for intraday rolling control — Phase 2.

Key differences from milp_cvar.py:
  - Accepts explicit pv_scenarios/load_scenarios arrays (shape [K, H])
  - H is dynamic: 24 - current_hour  (shrinking horizon)
  - Battery control is scenario-invariant (same as milp_cvar)
  - CVaR over per-scenario over-contract cost (same Rockafellar-Uryasev formulation)
  - Returns only the t=0 decisions; caller executes one step and re-solves next hour

Signature:
    solve_mpc_milp(pv_scenarios, load_scenarios, pi, cc, p_b, e_b,
                   initial_soc, peak_demand_running, current_hour,
                   lam, alpha, cfg, is_last_hour, month_id,
                   time_limit, mip_gap, verbose)
"""

import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_basic_charge

CFG = load_config()


def solve_mpc_milp(
    pv_scenarios:         np.ndarray,   # shape (K, H)  — PV forecast [kW]
    load_scenarios:       np.ndarray,   # shape (K, H)  — load forecast [kW]
    pi:                   np.ndarray,   # shape (K,)    — scenario probabilities
    cc:                   float,        # contracted capacity [kW]
    p_b:                  float,        # battery power rating [kW]
    e_b:                  float,        # battery energy capacity [kWh]
    initial_soc:          float,        # E_soc at start of horizon [kWh]
    peak_demand_running:  float,        # D_mth carry-in [kW] (month-to-date peak)
    current_hour:         int,          # 0-indexed hour of day (0..23)
    lam:                  float = 0.0,  # CVaR penalty weight
    alpha:                float = 0.90, # CVaR tail probability
    cfg:                  dict  = None,
    is_last_hour:         bool  = False,
    month_id:             int   = 1,
    tou_remaining:        np.ndarray = None,  # shape (H,) TOU prices for remaining hours
    time_limit:           int   = 30,
    mip_gap:              float = 0.005,
    verbose:              bool  = False,
    # ── M6: Reference-Guided MPC ─────────────────────────────────────────────
    m6b_grid_ref:         np.ndarray = None,  # shape (H,): M2 reference P_grid [kW]
    m6b_weight:           float = 0.0,        # λ_B: NTD/kW penalty for excess grid import
    m6c_d_ref:            float = None,       # M2 monthly D_peak reference [kW]
    m6c_weight:           float = 0.0,        # λ_C: NTD/kW penalty for peak over reference
    # ── M7-A: Risk-hour-only DA-guided grid cap ───────────────────────────────
    m7a_risk_mask:        np.ndarray = None,  # shape (H,): bool, True = risk hour
    m7a_grid_ref:         np.ndarray = None,  # shape (H,): M2 reference P_grid [kW]
    m7a_epsilon:          float = 0.0,        # [kW] slack above M2 reference (allowed band)
    m7a_slack_penalty:    float = 0.0,        # [NTD/kW] penalty for exceeding risk-hour cap
    # ── M7-D: DA-tail grid cap (lead > 6 solar hours) ────────────────────────
    m7d_tail_mask:        np.ndarray = None,  # shape (H,): bool, True = DA-tail slot
    m7d_tail_ref:         np.ndarray = None,  # shape (H,): M2 DA reference for tail [kW]
    m7d_tail_epsilon:     float = 0.0,        # [kW] slack above tail reference
    m7d_tail_penalty:     float = 0.0,        # [NTD/kW] penalty for tail hour excess
    # ── M7.1-B: Charging gate (scenario-invariant P_ch constraint) ───────────
    m71b_charge_gate_mask: np.ndarray = None, # shape (H,): bool, True = gate active
    m71b_charge_gate_mode: str = "off",       # "ref_plus_margin" | "absolute_cap"
    m71b_ch_ref:           np.ndarray = None, # shape (H,): M2 P_ch reference [kW]
    m71b_margin:           float = 0.0,       # [kW] allowed margin above M2 P_ch ref
    m71b_cap:              float = 0.0,       # [kW] absolute charging cap
    m71b_penalty:          float = 0.0,       # [NTD/kW] penalty per kW of gate slack
    # ── DA-aware MPC ablation: terminal SOC reference soft band ──────────────
    da_soc_terminal_ref:    float = None,      # [kWh] DA SOC reference at horizon end
    da_soc_terminal_band:   float = 0.0,       # [kWh] no-penalty band around reference
    da_soc_terminal_weight: float = 0.0,       # [NTD/kWh] penalty outside band
    # ── Standalone MPC terminal treatment / SOC-reference ablations ──────────
    terminal_soc_value_ntd_per_kwh: float = 0.0, # [NTD/kWh] salvage value of terminal SOC
    soc_ref_trajectory:    np.ndarray = None,    # shape (H,), optional SOC reference [kWh]
    soc_ref_band:          float = 0.0,           # [kWh] no-penalty band around SOC reference
    soc_ref_weight:        float = 0.0,           # [NTD/kWh] absolute deviation penalty
    peak_increment_weight_ntd_per_kw: float = 0.0, # [NTD/kW] penalty for new monthly peak
    current_peak_increment_weight_ntd_per_kw: float = 0.0, # [NTD/kW] executed-step peak increment
    first_step_min_discharge_kw: float = 0.0,       # [kW] soft minimum discharge for action-level shaving
    first_step_discharge_slack_penalty: float = 0.0,# [NTD/kW] slack penalty for action-level shaving
) -> dict:
    """
    Solve a single shrinking-horizon MPC step.

    H = pv_scenarios.shape[1]  = 24 - current_hour

    Returns a dict with:
        p_ch_exec   : float  [kW]  — charging power to execute this hour
        p_dis_exec  : float  [kW]  — discharging power to execute this hour
        soc_next    : float  [kWh] — predicted SOC after this hour's action
        P_ch_plan   : (H,)   full plan (for MPC preview / logging)
        P_dis_plan  : (H,)   full plan
        D_cand      : float  expected peak demand (pi-weighted)
        E_coc_val   : float  expected over-contract cost [NTD]
        cvar_val    : float  CVaR of over-contract cost
        eta_val     : float
        obj_val     : float
        solve_time  : float
        status      : int
    """
    if cfg is None:
        cfg = CFG

    bat = cfg["battery"]
    dem = cfg["demand"]

    K, H = pv_scenarios.shape
    assert load_scenarios.shape == (K, H), "pv/load shape mismatch"
    assert len(pi) == K, "pi length mismatch"

    CC    = cc
    PB    = p_b
    EB    = e_b
    D_init = peak_demand_running

    eta_ch  = bat["eta_ch"]
    eta_dis = bat["eta_dis"]
    soc_min = bat["soc_min"]
    soc_max = bat["soc_max"]
    soc_init_frac = bat["soc_init"]
    eps_term = bat["epsilon_term"]
    b_k     = bat["degradation_segments"]["b"]
    lam_k   = bat["degradation_segments"]["lambda"]
    n_seg   = len(lam_k)

    kappa = dem["kappa"]
    m1    = dem["over_contract_m1"]
    m2    = dem["over_contract_m2"]
    tier1 = dem["over_contract_tier1_ratio"]

    c_basic = get_basic_charge(month_id, cfg)

    # TOU for remaining hours (H-length array)
    if tou_remaining is None:
        tou_remaining = np.zeros(H)

    T = range(H)
    W = range(K)

    t_start = time.time()

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1 if verbose else 0)
        env.setParam("TimeLimit",  time_limit)
        env.setParam("MIPGap",     mip_gap)
        env.start()
        mdl = gp.Model(env=env)

        # ── Scenario-invariant battery control ────────────────────────────────
        u     = mdl.addVars(H, vtype=GRB.BINARY,              name="u")
        P_ch  = mdl.addVars(H, lb=0.0,                        name="P_ch")
        P_dis = mdl.addVars(H, lb=0.0,                        name="P_dis")
        E     = mdl.addVars(H, lb=soc_min*EB, ub=soc_max*EB, name="E")
        e_seg = mdl.addVars(H, n_seg, lb=0.0,                 name="e_seg")

        # ── Scenario-dependent flows ──────────────────────────────────────────
        P_gl  = mdl.addVars(K, H, lb=0.0, name="P_gl")
        P_gc  = mdl.addVars(K, H, lb=0.0, name="P_gc")
        P_pvl = mdl.addVars(K, H, lb=0.0, name="P_pvl")
        P_pvc = mdl.addVars(K, H, lb=0.0, name="P_pvc")
        P_pvcu= mdl.addVars(K, H, lb=0.0, name="P_pvcu")

        # ── Per-scenario demand proxy ─────────────────────────────────────────
        D_cand    = mdl.addVars(K, lb=D_init,             name="D_cand")
        D_now     = mdl.addVars(K, lb=D_init,             name="D_now")
        Over_full = mdl.addVars(K, lb=0.0,                name="Over_full")
        O1_full   = mdl.addVars(K, lb=0.0, ub=tier1*CC,  name="O1_full")
        O2_full   = mdl.addVars(K, lb=0.0,                name="O2_full")

        # ── CVaR variables (Rockafellar-Uryasev) ─────────────────────────────
        eta = mdl.addVar(lb=-1e8, ub=1e8, name="eta")
        xi  = mdl.addVars(K, lb=0.0,     name="xi")

        # ── Constraints ───────────────────────────────────────────────────────

        # (1) Battery charge/discharge limits
        for t in T:
            mdl.addConstr(P_ch[t]  <= u[t] * PB,     name=f"ch_ub_{t}")
            mdl.addConstr(P_dis[t] <= (1-u[t]) * PB, name=f"dis_ub_{t}")

        # (2) SOC recursion
        for t in T:
            E_prev = initial_soc if t == 0 else E[t-1]
            mdl.addConstr(
                E[t] == E_prev + eta_ch * P_ch[t] - (1.0/eta_dis) * P_dis[t],
                name=f"soc_{t}"
            )

        # (3) Terminal SOC anchor at end of day (only if last hour of day)
        if is_last_hour:
            E_term_lb = (soc_init_frac - eps_term) * EB
            E_term_ub = (soc_init_frac + eps_term) * EB
            mdl.addConstr(E[H-1] >= E_term_lb, name="term_lb")
            mdl.addConstr(E[H-1] <= E_term_ub, name="term_ub")

        # (3b) Optional terminal SOC soft reference for DA-aware MPC ablations.
        use_da_soc_ref = (
            da_soc_terminal_ref is not None
            and da_soc_terminal_weight > 0.0
        )
        C_da_soc_ref = 0.0
        if use_da_soc_ref:
            s_soc_pos = mdl.addVar(lb=0.0, name="s_da_soc_pos")
            s_soc_neg = mdl.addVar(lb=0.0, name="s_da_soc_neg")
            ref_e = float(da_soc_terminal_ref)
            band_e = max(float(da_soc_terminal_band), 0.0)
            mdl.addConstr(
                E[H-1] <= ref_e + band_e + s_soc_pos,
                name="da_soc_ref_ub"
            )
            mdl.addConstr(
                E[H-1] >= ref_e - band_e - s_soc_neg,
                name="da_soc_ref_lb"
            )
            C_da_soc_ref = da_soc_terminal_weight * (s_soc_pos + s_soc_neg)

        # (3c) Optional full-horizon SOC reference soft band. This is used only
        # for standalone MPC fix experiments; defaults are off.
        use_soc_ref = (
            soc_ref_trajectory is not None
            and soc_ref_weight > 0.0
        )
        C_soc_ref = 0.0
        if use_soc_ref:
            ref_arr = np.asarray(soc_ref_trajectory, dtype=float)
            if len(ref_arr) != H:
                raise ValueError("soc_ref_trajectory length must match H")
            band_e = max(float(soc_ref_band), 0.0)
            s_ref_pos = mdl.addVars(H, lb=0.0, name="s_soc_ref_pos")
            s_ref_neg = mdl.addVars(H, lb=0.0, name="s_soc_ref_neg")
            for t in T:
                mdl.addConstr(
                    E[t] <= float(ref_arr[t]) + band_e + s_ref_pos[t],
                    name=f"soc_ref_ub_{t}"
                )
                mdl.addConstr(
                    E[t] >= float(ref_arr[t]) - band_e - s_ref_neg[t],
                    name=f"soc_ref_lb_{t}"
                )
            C_soc_ref = soc_ref_weight * quicksum(
                s_ref_pos[t] + s_ref_neg[t] for t in T
            )

        C_first_step_shave = 0.0
        if first_step_min_discharge_kw > 0.0 and first_step_discharge_slack_penalty > 0.0:
            s_first_dis = mdl.addVar(lb=0.0, name="s_first_step_discharge")
            mdl.addConstr(
                P_dis[0] + s_first_dis >= float(first_step_min_discharge_kw),
                name="first_step_min_discharge"
            )
            C_first_step_shave = float(first_step_discharge_slack_penalty) * s_first_dis

        # (4) Degradation segments
        for t in T:
            mdl.addConstr(
                P_dis[t] == quicksum(e_seg[t, k] for k in range(n_seg)),
                name=f"seg_sum_{t}"
            )
            for k in range(n_seg):
                cap_k = (b_k[k+1] - b_k[k]) * EB
                mdl.addConstr(e_seg[t, k] <= cap_k, name=f"seg_cap_{t}_{k}")

        # (5) Scenario settlement (power balance)
        for w in W:
            pv_w   = pv_scenarios[w]
            load_w = load_scenarios[w]
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

        # (6) Per-scenario demand proxy
        for w in W:
            for t in T:
                mdl.addConstr(
                    D_cand[w] >= kappa * (P_gl[w,t] + P_gc[w,t]),
                    name=f"Dcand_{w}_{t}"
                )
            mdl.addConstr(
                D_now[w] >= kappa * (P_gl[w,0] + P_gc[w,0]),
                name=f"Dnow_{w}"
            )
            mdl.addConstr(Over_full[w] >= D_cand[w] - CC,            name=f"over_pos_{w}")
            mdl.addConstr(Over_full[w] == O1_full[w] + O2_full[w],   name=f"over_split_{w}")

        # (7) CVaR linearization
        for w in W:
            C_oc_w = c_basic * (m1 * O1_full[w] + m2 * O2_full[w])
            mdl.addConstr(xi[w] >= C_oc_w - eta, name=f"cvar_xi_{w}")

        # ── M6-B: one-sided grid import tracking ─────────────────────────────
        use_m6b = (m6b_grid_ref is not None) and (m6b_weight > 0.0)
        if use_m6b:
            s_grid = mdl.addVars(K, H, lb=0.0, name="s_grid")
            for w in W:
                for t in T:
                    ref_t = float(m6b_grid_ref[t])
                    mdl.addConstr(
                        s_grid[w, t] >= P_gl[w, t] + P_gc[w, t] - ref_t,
                        name=f"m6b_{w}_{t}"
                    )

        # ── M6-C: monthly peak guard ──────────────────────────────────────────
        use_m6c = (m6c_d_ref is not None) and (m6c_weight > 0.0)
        if use_m6c:
            s_peak = mdl.addVars(K, lb=0.0, name="s_peak")
            for w in W:
                mdl.addConstr(
                    s_peak[w] >= D_cand[w] - float(m6c_d_ref),
                    name=f"m6c_{w}"
                )

        # ── M7-A: risk-hour-only grid cap ────────────────────────────────────
        use_m7a = (m7a_risk_mask is not None) and (m7a_slack_penalty > 0.0)
        C_m7a = 0.0
        m7a_total_slack = 0.0   # filled after solve
        m7a_active_pairs = []
        if use_m7a:
            m7a_active_pairs = [(w, t) for w in W for t in T if m7a_risk_mask[t]]
            if m7a_active_pairs:
                s_m7a = {(w, t): mdl.addVar(lb=0.0, name=f"sm7a_{w}_{t}")
                         for (w, t) in m7a_active_pairs}
                for (w, t) in m7a_active_pairs:
                    ref_t = float(m7a_grid_ref[t])
                    mdl.addConstr(
                        P_gl[w, t] + P_gc[w, t] <= ref_t + m7a_epsilon + s_m7a[(w, t)],
                        name=f"m7a_{w}_{t}"
                    )
                C_m7a = m7a_slack_penalty * quicksum(
                    pi[w] * s_m7a[(w, t)] for (w, t) in m7a_active_pairs
                )

        # ── M7-D: DA-tail grid cap (lead > 6 solar hours) ────────────────────
        use_m7d = (m7d_tail_mask is not None) and (m7d_tail_penalty > 0.0)
        C_m7d = 0.0
        m7d_active_pairs = []
        if use_m7d:
            m7d_active_pairs = [(w, t) for w in W for t in T if m7d_tail_mask[t]]
            if m7d_active_pairs:
                s_m7d = {(w, t): mdl.addVar(lb=0.0, name=f"sm7d_{w}_{t}")
                         for (w, t) in m7d_active_pairs}
                for (w, t) in m7d_active_pairs:
                    ref_t = float(m7d_tail_ref[t])
                    mdl.addConstr(
                        P_gl[w, t] + P_gc[w, t] <= ref_t + m7d_tail_epsilon + s_m7d[(w, t)],
                        name=f"m7d_{w}_{t}"
                    )
                C_m7d = m7d_tail_penalty * quicksum(
                    pi[w] * s_m7d[(w, t)] for (w, t) in m7d_active_pairs
                )

        # ── M7.1-B: Charging gate ────────────────────────────────────────────
        use_m71b = (
            m71b_charge_gate_mask is not None
            and m71b_penalty > 0.0
            and m71b_charge_gate_mode in ("ref_plus_margin", "absolute_cap")
        )
        C_m71b = 0.0
        m71b_active_t = []
        s_cg = {}
        if use_m71b:
            m71b_active_t = [t for t in T if m71b_charge_gate_mask[t]]
            if m71b_active_t:
                s_cg = {t: mdl.addVar(lb=0.0, name=f"scg_{t}")
                        for t in m71b_active_t}
                for t in m71b_active_t:
                    if m71b_charge_gate_mode == "ref_plus_margin":
                        ref_ch = float(m71b_ch_ref[t]) if m71b_ch_ref is not None else 0.0
                        mdl.addConstr(
                            P_ch[t] <= ref_ch + m71b_margin + s_cg[t],
                            name=f"cg_{t}"
                        )
                    else:  # absolute_cap
                        mdl.addConstr(
                            P_ch[t] <= m71b_cap + s_cg[t],
                            name=f"cg_{t}"
                        )
                C_m71b = m71b_penalty * quicksum(s_cg[t] for t in m71b_active_t)

        # ── Objective ─────────────────────────────────────────────────────────
        C_ene = quicksum(
            pi[w] * tou_remaining[t] * (P_gl[w,t] + P_gc[w,t])
            for w in W for t in T
        )

        C_deg = quicksum(
            lam_k[k] * e_seg[t, k] for t in T for k in range(n_seg)
        )

        Over_prev    = max(D_init - CC, 0.0)
        O1_prev      = min(Over_prev, tier1 * CC)
        O2_prev      = max(Over_prev - tier1 * CC, 0.0)
        PeakCost_prev = c_basic * (m1 * O1_prev + m2 * O2_prev)

        E_C_oc = quicksum(
            pi[w] * c_basic * (m1 * O1_full[w] + m2 * O2_full[w])
            for w in W
        )
        C_peak_exp = E_C_oc - PeakCost_prev
        C_peak_increment = (
            float(peak_increment_weight_ntd_per_kw)
            * quicksum(pi[w] * (D_cand[w] - D_init) for w in W)
            if peak_increment_weight_ntd_per_kw > 0.0 else 0.0
        )
        C_current_peak_increment = (
            float(current_peak_increment_weight_ntd_per_kw)
            * quicksum(pi[w] * (D_now[w] - D_init) for w in W)
            if current_peak_increment_weight_ntd_per_kw > 0.0 else 0.0
        )

        C_cvar = eta + (1.0 / (1.0 - alpha)) * quicksum(
            pi[w] * xi[w] for w in W
        )

        C_m6b = (m6b_weight * quicksum(pi[w] * s_grid[w, t] for w in W for t in T)
                 if use_m6b else 0.0)
        C_m6c = (m6c_weight * quicksum(pi[w] * s_peak[w] for w in W)
                 if use_m6c else 0.0)

        mdl.setObjective(
            C_ene + C_deg + C_peak_exp + lam * C_cvar
            + C_m6b + C_m6c + C_m7a + C_m7d + C_m71b + C_da_soc_ref + C_soc_ref
            + C_peak_increment + C_current_peak_increment + C_first_step_shave
            - float(terminal_soc_value_ntd_per_kwh) * E[H-1],
            GRB.MINIMIZE
        )
        mdl.optimize()

        solve_time = time.time() - t_start
        status = mdl.Status

        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mdl.SolCount == 0:
            return _fallback_mpc(initial_soc, EB, soc_init_frac, eta_ch, H, solve_time)

        P_ch_plan  = np.array([P_ch[t].X  for t in T])
        P_dis_plan = np.array([P_dis[t].X for t in T])
        E_plan     = np.array([E[t].X     for t in T])

        # Only t=0 is executed
        p_ch_exec  = float(P_ch_plan[0])
        p_dis_exec = float(P_dis_plan[0])
        soc_next   = float(E_plan[0])

        D_cand_by_sc = [float(D_cand[w].X) for w in W]
        C_oc_by_sc   = [
            c_basic * (m1 * float(O1_full[w].X) + m2 * float(O2_full[w].X))
            for w in W
        ]
        D_cand_exp = float(sum(pi[w] * D_cand_by_sc[w] for w in W))
        E_coc_val  = float(sum(pi[w] * C_oc_by_sc[w]   for w in W))
        eta_val    = float(eta.X)
        xi_vals    = [float(xi[w].X) for w in W]
        cvar_val   = eta_val + (1.0 / (1.0 - alpha)) * sum(pi[w] * xi_vals[w] for w in W)

        m6b_excess = (float(sum(pi[w] * s_grid[w, t].X for w in W for t in T))
                      if use_m6b else 0.0)
        m6c_excess = (float(sum(pi[w] * s_peak[w].X for w in W))
                      if use_m6c else 0.0)
        m7a_slack_val = (
            float(sum(pi[w] * s_m7a[(w, t)].X for (w, t) in m7a_active_pairs))
            if (use_m7a and m7a_active_pairs) else 0.0
        )
        m7d_slack_val = (
            float(sum(pi[w] * s_m7d[(w, t)].X for (w, t) in m7d_active_pairs))
            if (use_m7d and m7d_active_pairs) else 0.0
        )
        da_soc_slack_val = (
            float(s_soc_pos.X + s_soc_neg.X) if use_da_soc_ref else 0.0
        )

        return {
            "p_ch_exec":           p_ch_exec,
            "p_dis_exec":          p_dis_exec,
            "soc_next":            soc_next,
            "P_ch_plan":           P_ch_plan,
            "P_dis_plan":          P_dis_plan,
            "D_cand":              D_cand_exp,
            "D_cand_by_scenario":  D_cand_by_sc,
            "C_oc_by_scenario":    C_oc_by_sc,
            "E_coc_val":           E_coc_val,
            "cvar_val":            cvar_val,
            "eta_val":             eta_val,
            "obj_val":             float(mdl.ObjVal),
            "solve_time":          solve_time,
            "status":              status,
            "m6b_excess":          m6b_excess,
            "m6c_excess":          m6c_excess,
            "m7a_slack_kw":        m7a_slack_val,
            "m7d_slack_kw":        m7d_slack_val,
            "m71b_slack_kw":       (float(sum(s_cg[t].X for t in m71b_active_t))
                                    if (use_m71b and m71b_active_t) else 0.0),
            "da_soc_ref_slack_kwh": da_soc_slack_val,
        }


def _fallback_mpc(initial_soc, EB, soc_init_frac, eta_ch, H, solve_time):
    """Do-nothing plan on MILP failure: no charge/discharge."""
    return {
        "p_ch_exec":           0.0,
        "p_dis_exec":          0.0,
        "soc_next":            initial_soc,
        "P_ch_plan":           np.zeros(H),
        "P_dis_plan":          np.zeros(H),
        "D_cand":              0.0,
        "D_cand_by_scenario":  [],
        "C_oc_by_scenario":    [],
        "E_coc_val":           0.0,
        "cvar_val":            0.0,
        "eta_val":             0.0,
        "obj_val":             0.0,
        "solve_time":          solve_time,
        "status":              -1,
        "m6b_excess":          0.0,
        "m6c_excess":          0.0,
        "m7a_slack_kw":        0.0,
        "m7d_slack_kw":        0.0,
        "m71b_slack_kw":       0.0,
        "da_soc_ref_slack_kwh": 0.0,
    }
