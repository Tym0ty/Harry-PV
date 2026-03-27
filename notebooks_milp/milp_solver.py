"""
Full-Year Direct Solve MILP — Core solver.

Per FF0326Harry_MILP_Engineering_Spec_FullYear_Formal_vfinal.
Implements C0–C3 full-year sizing with:
  - P_grid split into P_grid_load + P_grid_ch (spec C1/C2)
  - Green SOC accounting (spec C11/C12)
  - PWL battery degradation (spec C13)
  - Expected inter-day SOC for probabilistic (spec C6)
  - No export / No CPPA (spec C14)
"""
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB
from milp_common import get_monthly_basic_charge


def build_and_solve(day_data, day_indices, scenario_ids, CFG, case_id="C0", cc_ub=None, cc_lb=None, pb_lb=None, pb_ub=None, eb_ub=None, eb_lb=None, mip_gap=None, no_re20=False):
    """Build and solve the full-year MILP for one case.

    Args:
        day_data: dict from load_data()
        day_indices: list of day indices (1..N)
        scenario_ids: list of scenario ids
        CFG: config dict
        case_id: "C0"/"C1"/"C2"/"C3"

    Returns:
        result dict with sizing, cost breakdown, solve time
    """
    t0 = time.time()
    n_days = len(day_indices)
    n_scenarios = len(scenario_ids)
    n_hours = 24
    is_prob = n_scenarios > 1  # probabilistic case

    # Parameters
    eta_ch = CFG['eff_charge']
    eta_dis = CFG['eff_discharge']
    soc_min = CFG['soc_min']
    soc_max = CFG['soc_max']
    soc_init = CFG['soc_init']
    kappa = CFG['kappa']
    crf_bess = CFG['crf_bess']
    c_bp = CFG['capex_bess_power_per_kw']
    c_be = CFG['capex_bess_energy_per_kwh']
    re_target = CFG['re_target']
    c_trec = CFG['trec_cost_per_kwh']
    eps_term = CFG['epsilon_term']
    eps_g_term = CFG['epsilon_g_term']
    oc_m1 = CFG['oc_within_10pct_mult']
    oc_m2 = CFG['oc_beyond_10pct_mult']

    # PWL degradation
    b_k = CFG['pwl_deg_b_k']
    lam_k = CFG['pwl_deg_lambda_k']
    n_seg = len(lam_k)

    # Month info
    all_months = sorted(set(day_data[di]['month_id'] for di in day_indices))

    # Map day_index to sequential position (0-based)
    di_to_pos = {di: pos for pos, di in enumerate(day_indices)}
    # Map (day_index) -> list of (scenario_idx, prob)
    # For deterministic: one scenario with prob=1
    # For probabilistic: K scenarios with their probs

    print(f"Building model {case_id}: {n_days} days × {n_scenarios} scenarios × {n_hours} hours")
    total_vars_est = n_days * n_scenarios * n_hours * 16
    print(f"  Estimated variables: ~{total_vars_est:,}")

    # ── Model ────────────────────────────────────────────────
    m = gp.Model(f"MILP_{case_id}")
    m.Params.TimeLimit = CFG['time_limit']
    m.Params.MIPGap = mip_gap if mip_gap is not None else CFG['mip_gap']
    m.Params.OutputFlag = 1

    # ── Stage 1: Sizing variables ────────────────────────────
    cc_lower = cc_lb if cc_lb is not None else 0
    CC = m.addVar(lb=cc_lower, name="CC")          # contract capacity (kW)
    if cc_ub is not None:
        CC.ub = cc_ub
        print(f"  CC upper bound set to {cc_ub:.1f} kW")
    if cc_lb is not None:
        print(f"  CC lower bound set to {cc_lb:.1f} kW")
    pb_lower = pb_lb if pb_lb is not None else 0
    pb_upper = pb_ub if pb_ub is not None else CFG['bess_p_max_kw']
    P_B = m.addVar(lb=pb_lower, ub=pb_upper, name="P_B")    # BESS power
    if pb_lb is not None:
        print(f"  P_B lower bound set to {pb_lb:.1f} kW")
    if pb_ub is not None:
        print(f"  P_B upper bound set to {pb_ub:.1f} kW")
    eb_lower = eb_lb if eb_lb is not None else 0
    eb_upper = eb_ub if eb_ub is not None else CFG['bess_e_max_kwh']
    E_B = m.addVar(lb=eb_lower, ub=eb_upper, name="E_B")   # BESS energy
    if eb_lb is not None:
        print(f"  E_B lower bound set to {eb_lb:.1f} kWh")
    if eb_ub is not None:
        print(f"  E_B upper bound set to {eb_ub:.1f} kWh")

    # ── Stage 2: Operational variables ───────────────────────
    # Indexed by (day_pos, scenario_idx, hour)
    # For efficiency, use addVars with tuples

    idx_dsh = [(di, s, t) for di in day_indices
               for s in range(n_scenarios) for t in range(n_hours)]

    P_grid_load = m.addVars(idx_dsh, lb=0, name="Pgl")
    P_grid_ch   = m.addVars(idx_dsh, lb=0, name="Pgc")
    P_pv_load   = m.addVars(idx_dsh, lb=0, name="Ppl")
    P_pv_ch     = m.addVars(idx_dsh, lb=0, name="Ppc")
    P_pv_curt   = m.addVars(idx_dsh, lb=0, name="Pcurt")
    P_ch        = m.addVars(idx_dsh, lb=0, name="Pch")
    P_dis       = m.addVars(idx_dsh, lb=0, name="Pdis")
    u           = m.addVars(idx_dsh, vtype=GRB.BINARY, name="u")
    E_soc       = m.addVars(idx_dsh, lb=0, name="E")

    # Green SOC
    P_ch_g  = m.addVars(idx_dsh, lb=0, name="Pchg")
    P_dis_g = m.addVars(idx_dsh, lb=0, name="Pdisg")
    E_g     = m.addVars(idx_dsh, lb=0, name="Eg")

    # PWL degradation segments
    idx_seg = [(di, s, t, k) for di in day_indices
               for s in range(n_scenarios) for t in range(n_hours)
               for k in range(n_seg)]
    e_dis_seg = m.addVars(idx_seg, lb=0, name="eseg")

    # Start-of-day SOC (shared across scenarios for a given day)
    E_daystart = m.addVars(day_indices, lb=0, name="Eds")
    E_g_daystart = m.addVars(day_indices, lb=0, name="Egds")

    # Monthly demand proxy: expected-value for prob, direct for det
    Dmax = m.addVars(all_months, lb=0, name="Dmax")

    # Over-contract segments
    O1 = m.addVars(all_months, lb=0, name="O1")
    O2 = m.addVars(all_months, lb=0, name="O2")

    # T-REC gap filler (single — spec eq 20 uses expected RE)
    E_TREC = m.addVar(lb=0, name="Etrec")

    m.update()

    # ── Constraints ──────────────────────────────────────────
    print("  Adding constraints...")

    total_load_per_scenario = {s: 0.0 for s in range(n_scenarios)}

    for di in day_indices:
        dd = day_data[di]
        scenarios = dd['scenarios']
        tou = dd['tou']

        for s_idx in range(n_scenarios):
            sc = scenarios[s_idx]
            pv_arr = sc['pv_kw']
            load_arr = sc['load_kw']

            for t in range(n_hours):
                key = (di, s_idx, t)
                pv_t = float(pv_arr[t])
                load_t = float(load_arr[t])
                total_load_per_scenario[s_idx] += load_t

                # C1: Load balance
                m.addConstr(
                    P_grid_load[key] + P_pv_load[key] + P_dis[key] == load_t,
                    name=f"C1_{di}_{s_idx}_{t}")

                # C2: Charging source decomposition
                m.addConstr(
                    P_ch[key] == P_grid_ch[key] + P_pv_ch[key],
                    name=f"C2_{di}_{s_idx}_{t}")

                # C3: PV flow split
                m.addConstr(
                    P_pv_load[key] + P_pv_ch[key] + P_pv_curt[key] <= pv_t,
                    name=f"C3_{di}_{s_idx}_{t}")

                # C4: Charge/discharge mutual exclusivity
                m.addConstr(P_ch[key] <= P_B * u[key],
                            name=f"C4a_{di}_{s_idx}_{t}")
                m.addConstr(P_dis[key] <= P_B * (1 - u[key]),
                            name=f"C4b_{di}_{s_idx}_{t}")

                # C5: SOC recursion
                if t == 0:
                    E_prev = E_daystart[di]
                else:
                    E_prev = E_soc[di, s_idx, t - 1]

                m.addConstr(
                    E_soc[key] == E_prev + eta_ch * P_ch[key] - (1.0 / eta_dis) * P_dis[key],
                    name=f"C5_{di}_{s_idx}_{t}")

                # SOC bounds
                m.addConstr(E_soc[key] >= soc_min * E_B,
                            name=f"C5lo_{di}_{s_idx}_{t}")
                m.addConstr(E_soc[key] <= soc_max * E_B,
                            name=f"C5hi_{di}_{s_idx}_{t}")

                # C8: Monthly demand proxy
                # For deterministic: direct constraint (single scenario)
                # For probabilistic: skip here, add expected-value below
                if not is_prob:
                    mo = dd['month_id']
                    m.addConstr(
                        Dmax[mo] >= kappa * (P_grid_load[key] + P_grid_ch[key]),
                        name=f"C8_{di}_{s_idx}_{t}")

                # C11: Green SOC accounting
                # P_ch_g <= P_pv_ch (green charging only from PV)
                m.addConstr(P_ch_g[key] <= P_pv_ch[key],
                            name=f"C11a_{di}_{s_idx}_{t}")
                # P_dis_g <= P_dis
                m.addConstr(P_dis_g[key] <= P_dis[key],
                            name=f"C11b_{di}_{s_idx}_{t}")

                # Green SOC recursion
                if t == 0:
                    Eg_prev = E_g_daystart[di]
                else:
                    Eg_prev = E_g[di, s_idx, t - 1]

                m.addConstr(
                    E_g[key] == Eg_prev + eta_ch * P_ch_g[key] - (1.0 / eta_dis) * P_dis_g[key],
                    name=f"C11c_{di}_{s_idx}_{t}")

                # E_g <= E
                m.addConstr(E_g[key] <= E_soc[key],
                            name=f"C11d_{di}_{s_idx}_{t}")
                # E_g >= 0 (already lb=0)

                # C13: PWL degradation
                # P_dis * Δt = Σ_k e_dis_seg(k)
                m.addConstr(
                    P_dis[key] == gp.quicksum(e_dis_seg[di, s_idx, t, k] for k in range(n_seg)),
                    name=f"C13a_{di}_{s_idx}_{t}")

                # Segment upper bounds: (b_{k+1} - b_k) * E_B
                for k in range(n_seg):
                    seg_width = b_k[k + 1] - b_k[k]
                    m.addConstr(
                        e_dis_seg[di, s_idx, t, k] <= seg_width * E_B,
                        name=f"C13b_{di}_{s_idx}_{t}_{k}")

    # C8 (probabilistic): Expected-value demand proxy
    # Dmax[mo] >= kappa * Σ_ω π_ω * (P_grid_load + P_grid_ch) per (day, hour)
    if is_prob:
        print("  Adding expected-value demand proxy (C8)...")
        for di in day_indices:
            dd = day_data[di]
            mo = dd['month_id']
            probs = [dd['scenarios'][s]['prob'] for s in range(n_scenarios)]
            for t in range(n_hours):
                m.addConstr(
                    Dmax[mo] >= kappa * gp.quicksum(
                        probs[s] * (P_grid_load[di, s, t] + P_grid_ch[di, s, t])
                        for s in range(n_scenarios)),
                    name=f"C8exp_{di}_{t}")

    # C6: Cross-day SOC linkage
    print("  Adding cross-day SOC linkage...")
    # First day: E_daystart[day_indices[0]] = soc_init * E_B
    m.addConstr(E_daystart[day_indices[0]] == soc_init * E_B, name="C6_init")
    m.addConstr(E_g_daystart[day_indices[0]] == 0, name="C6g_init")

    for pos in range(len(day_indices) - 1):
        di_curr = day_indices[pos]
        di_next = day_indices[pos + 1]

        if not is_prob:
            # Deterministic: direct linkage
            m.addConstr(
                E_daystart[di_next] == E_soc[di_curr, 0, n_hours - 1],
                name=f"C6_{di_curr}")
            m.addConstr(
                E_g_daystart[di_next] == E_g[di_curr, 0, n_hours - 1],
                name=f"C6g_{di_curr}")
        else:
            # Probabilistic: expected carry-over
            dd = day_data[di_curr]
            probs = [dd['scenarios'][s]['prob'] for s in range(n_scenarios)]
            m.addConstr(
                E_daystart[di_next] == gp.quicksum(
                    probs[s] * E_soc[di_curr, s, n_hours - 1]
                    for s in range(n_scenarios)),
                name=f"C6_{di_curr}")
            m.addConstr(
                E_g_daystart[di_next] == gp.quicksum(
                    probs[s] * E_g[di_curr, s, n_hours - 1]
                    for s in range(n_scenarios)),
                name=f"C6g_{di_curr}")

    # C7: Year-end terminal band
    last_di = day_indices[-1]
    first_di = day_indices[0]
    if not is_prob:
        for s in range(n_scenarios):
            m.addConstr(
                E_soc[last_di, s, n_hours - 1] >= soc_init * E_B - eps_term * E_B,
                name=f"C7lo_{s}")
            m.addConstr(
                E_soc[last_di, s, n_hours - 1] <= soc_init * E_B + eps_term * E_B,
                name=f"C7hi_{s}")
    else:
        # Expected terminal band — constrains probability-weighted end SOC
        dd_last = day_data[last_di]
        probs_last = [dd_last['scenarios'][s]['prob'] for s in range(n_scenarios)]
        E_soc_exp = gp.quicksum(
            probs_last[s] * E_soc[last_di, s, n_hours - 1]
            for s in range(n_scenarios))
        m.addConstr(E_soc_exp >= soc_init * E_B - eps_term * E_B, name="C7lo_exp")
        m.addConstr(E_soc_exp <= soc_init * E_B + eps_term * E_B, name="C7hi_exp")

    # C9: Over-contract linearization (robust — scenario-independent)
    for mo in all_months:
        m.addConstr(Dmax[mo] - CC == O1[mo] + O2[mo],
                    name=f"C9a_{mo}")
        m.addConstr(O1[mo] <= 0.10 * CC,
                    name=f"C9b_{mo}")

    # C10: RE20 and T-REC gap filler (spec eq 17-20: expected accounting)
    # Expected RE20 constraint — prevents BESS oversizing for worst-case scenario
    # while maintaining robust over-contract hedging via Dmax
    if not no_re20:
        E_pv_self_yr = gp.quicksum(
            day_data[di]['scenarios'][s]['prob'] * P_pv_load[di, s, t]
            for di in day_indices for s in range(n_scenarios) for t in range(n_hours))
        E_dis_g_yr = gp.quicksum(
            day_data[di]['scenarios'][s]['prob'] * P_dis_g[di, s, t]
            for di in day_indices for s in range(n_scenarios) for t in range(n_hours))
        E_load_yr = sum(
            day_data[di]['scenarios'][s]['prob'] * float(day_data[di]['scenarios'][s]['load_kw'][t])
            for di in day_indices for s in range(n_scenarios) for t in range(n_hours))

        m.addConstr(
            E_pv_self_yr + E_dis_g_yr + E_TREC >= re_target * E_load_yr,
            name="C10")

    # C12: Green SOC year-end
    if not is_prob:
        for s in range(n_scenarios):
            m.addConstr(E_g[last_di, s, n_hours - 1] <= eps_g_term * E_B,
                    name=f"C12_{s}")
    else:
        E_g_exp = gp.quicksum(
            probs_last[s] * E_g[last_di, s, n_hours - 1]
            for s in range(n_scenarios))
        m.addConstr(E_g_exp <= eps_g_term * E_B, name="C12_exp")

    # ── Objective ────────────────────────────────────────────
    print("  Building objective...")

    # AEC_inv: BESS investment annualized
    AEC_inv = crf_bess * (c_bp * P_B + c_be * E_B)

    # AEC_ene: energy purchase cost
    if not is_prob:
        AEC_ene = gp.quicksum(
            day_data[di]['tou'][t] * (P_grid_load[di, 0, t] + P_grid_ch[di, 0, t])
            for di in day_indices for t in range(n_hours))
    else:
        AEC_ene = gp.quicksum(
            day_data[di]['scenarios'][s]['prob'] *
            day_data[di]['tou'][t] * (P_grid_load[di, s, t] + P_grid_ch[di, s, t])
            for di in day_indices for s in range(n_scenarios) for t in range(n_hours))

    # AEC_basic: monthly basic charge
    AEC_basic = gp.quicksum(
        get_monthly_basic_charge(mo, CFG) * CC for mo in all_months)

    # AEC_over: over-contract penalties (expected-value demand proxy for prob)
    AEC_over = gp.quicksum(
        get_monthly_basic_charge(mo, CFG) * (oc_m1 * O1[mo] + oc_m2 * O2[mo])
        for mo in all_months)

    # AEC_green: T-REC gap filler cost (spec eq 72)
    AEC_green = c_trec * E_TREC

    # AEC_deg: PWL degradation cost
    if not is_prob:
        AEC_deg = gp.quicksum(
            lam_k[k] * e_dis_seg[di, 0, t, k]
            for di in day_indices for t in range(n_hours) for k in range(n_seg))
    else:
        AEC_deg = gp.quicksum(
            day_data[di]['scenarios'][s]['prob'] * lam_k[k] * e_dis_seg[di, s, t, k]
            for di in day_indices for s in range(n_scenarios)
            for t in range(n_hours) for k in range(n_seg))

    m.setObjective(AEC_inv + AEC_ene + AEC_basic + AEC_over + AEC_green + AEC_deg,
                   GRB.MINIMIZE)

    # ── Solve ────────────────────────────────────────────────
    print("  Solving...")
    m.optimize()
    solve_time = time.time() - t0

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        if m.SolCount == 0:
            print(f"  No solution found within time limit!")
            return None

        # Extract results
        cc_val = CC.X
        pb_val = P_B.X
        eb_val = E_B.X

        inv_val = crf_bess * (c_bp * pb_val + c_be * eb_val)
        ene_val = AEC_ene.getValue()
        basic_val = AEC_basic.getValue()
        over_val = AEC_over.getValue()
        green_val = AEC_green.getValue()
        deg_val = AEC_deg.getValue()
        obj_val = m.ObjVal

        # RE percentage (including T-REC — constraint C10 guarantees ≥ 20%)
        trec_val = E_TREC.X
        if not is_prob:
            pv_self = sum(P_pv_load[di, 0, t].X
                         for di in day_indices for t in range(n_hours))
            dis_green = sum(P_dis_g[di, 0, t].X
                           for di in day_indices for t in range(n_hours))
            total_load = total_load_per_scenario[0]
            re_pct = (pv_self + dis_green + trec_val) / total_load * 100 if total_load > 0 else 0
        else:
            # Expected RE across scenarios (matches C10 expected formulation)
            pv_self_exp = sum(
                day_data[di]['scenarios'][s]['prob'] * P_pv_load[di, s, t].X
                for di in day_indices for s in range(n_scenarios) for t in range(n_hours))
            dis_green_exp = sum(
                day_data[di]['scenarios'][s]['prob'] * P_dis_g[di, s, t].X
                for di in day_indices for s in range(n_scenarios) for t in range(n_hours))
            total_load = sum(
                day_data[di]['scenarios'][s]['prob'] * total_load_per_scenario[s]
                for s in range(n_scenarios))
            re_pct = (pv_self_exp + dis_green_exp + trec_val) / total_load * 100 if total_load > 0 else 0

        cost_breakdown = {
            'AEC_inv': inv_val,
            'AEC_ene': ene_val,
            'AEC_basic': basic_val,
            'AEC_over': over_val,
            'AEC_green': green_val,
            'AEC_deg': deg_val,
        }

        print(f"\n  === {case_id} Results ===")
        print(f"  CC = {cc_val:.0f} kW")
        print(f"  P_B = {pb_val:.0f} kW, E_B = {eb_val:.0f} kWh (E/P = {eb_val/max(pb_val,0.01):.1f})")
        print(f"  Total AEC = {obj_val/1e6:.2f} M NTD")
        print(f"  AEC_inv={inv_val/1e6:.2f}M  AEC_ene={ene_val/1e6:.2f}M  "
              f"AEC_basic={basic_val/1e6:.2f}M  AEC_over={over_val/1e6:.2f}M  "
              f"AEC_green={green_val/1e6:.2f}M  AEC_deg={deg_val/1e6:.2f}M")
        print(f"  RE = {re_pct:.1f}%")
        print(f"  Solve time: {solve_time:.1f}s")

        result = {
            'case_id': case_id,
            'CC': cc_val,
            'P_B': pb_val,
            'E_B': eb_val,
            'obj_val': obj_val,
            're_pct': re_pct,
            'cost_breakdown': cost_breakdown,
            'solve_time': solve_time,
            'mip_gap': m.MIPGap if hasattr(m, 'MIPGap') else None,
            'status': m.status,
        }

        # Extract dispatch for replay comparison
        dispatch = []
        for di in day_indices:
            dd = day_data[di]
            for s in range(n_scenarios):
                for t in range(n_hours):
                    key = (di, s, t)
                    dispatch.append({
                        'day_index': di,
                        'scenario_idx': s,
                        'hour_local': t + 1,
                        'P_grid_load': P_grid_load[key].X,
                        'P_grid_ch': P_grid_ch[key].X,
                        'P_pv_load': P_pv_load[key].X,
                        'P_pv_ch': P_pv_ch[key].X,
                        'P_pv_curt': P_pv_curt[key].X,
                        'P_ch': P_ch[key].X,
                        'P_dis': P_dis[key].X,
                        'E_soc': E_soc[key].X,
                        'E_g': E_g[key].X,
                        'P_dis_g': P_dis_g[key].X,
                    })
        result['dispatch'] = dispatch

        return result
    else:
        print(f"  Solve failed with status {m.status}")
        return None


def replay(sizing, truth_df, calendar_df, CFG, case_id="C0", no_re20=False):
    """Fixed-design replay using truth data.

    Args:
        sizing: dict with CC, P_B, E_B from solve
        truth_df: truth replay package DataFrame
        calendar_df: calendar manifest DataFrame
        CFG: config dict
        case_id: case identifier

    Returns:
        replay result dict
    """
    from milp_common import get_tou_price, _is_summer

    cc_val = sizing['CC']
    pb_val = sizing['P_B']
    eb_val = sizing['E_B']
    eta_ch = CFG['eff_charge']
    eta_dis = CFG['eff_discharge']
    soc_min = CFG['soc_min']
    soc_max = CFG['soc_max']
    soc_init = CFG['soc_init']
    kappa = CFG['kappa']
    crf_bess = CFG['crf_bess']
    c_bp = CFG['capex_bess_power_per_kw']
    c_be = CFG['capex_bess_energy_per_kwh']
    re_target = CFG['re_target']
    c_trec = CFG['trec_cost_per_kwh']
    lam_k = CFG['pwl_deg_lambda_k']
    b_k = CFG['pwl_deg_b_k']
    n_seg = len(lam_k)

    all_months = sorted(calendar_df['month_id'].unique())
    day_indices = sorted(truth_df['day_index'].unique())

    print(f"  Replay {case_id}: CC={cc_val:.0f}, P_B={pb_val:.0f}, E_B={eb_val:.0f}")

    # Build model (single scenario, fixed sizing)
    m = gp.Model(f"Replay_{case_id}")
    m.Params.TimeLimit = CFG['time_limit']
    m.Params.MIPGap = CFG['mip_gap']
    m.Params.OutputFlag = 0

    n_hours = 24
    idx = [(di, t) for di in day_indices for t in range(n_hours)]

    P_grid_load = m.addVars(idx, lb=0, name="Pgl")
    P_grid_ch   = m.addVars(idx, lb=0, name="Pgc")
    P_pv_load   = m.addVars(idx, lb=0, name="Ppl")
    P_pv_ch     = m.addVars(idx, lb=0, name="Ppc")
    P_pv_curt   = m.addVars(idx, lb=0, name="Pcurt")
    P_ch        = m.addVars(idx, lb=0, name="Pch")
    P_dis       = m.addVars(idx, lb=0, name="Pdis")
    u_var       = m.addVars(idx, vtype=GRB.BINARY, name="u")
    E_soc       = m.addVars(idx, lb=0, name="E")
    P_ch_g      = m.addVars(idx, lb=0, name="Pchg")
    P_dis_g     = m.addVars(idx, lb=0, name="Pdisg")
    E_g         = m.addVars(idx, lb=0, name="Eg")

    idx_seg = [(di, t, k) for di in day_indices for t in range(n_hours) for k in range(n_seg)]
    e_dis_seg = m.addVars(idx_seg, lb=0, name="eseg")

    E_daystart = m.addVars(day_indices, lb=0, name="Eds")
    E_g_daystart = m.addVars(day_indices, lb=0, name="Egds")

    Dmax = m.addVars(all_months, lb=0, name="Dmax")
    O1 = m.addVars(all_months, lb=0, name="O1")
    O2 = m.addVars(all_months, lb=0, name="O2")
    E_TREC = m.addVar(lb=0, name="Etrec")

    m.update()

    # Precompute truth data lookup
    truth_lookup = {}
    for _, row in truth_df.iterrows():
        truth_lookup[(int(row['day_index']), int(row['hour_local']))] = (
            float(row['pv_realized_kw']),
            float(row['load_realized_kw']),
        )

    cal_lookup = calendar_df.set_index('day_index')
    total_load = 0.0

    for di in day_indices:
        cal = cal_lookup.loc[di]
        cd = pd.Timestamp(cal['calendar_day'])
        dow = cd.weekday()
        mo = int(cal['month_id'])

        for t in range(n_hours):
            h_local = t + 1
            pv_t, load_t = truth_lookup.get((di, h_local), (0.0, 0.0))
            total_load += load_t
            key = (di, t)

            # Same constraints as solve but single scenario, fixed sizing
            m.addConstr(P_grid_load[key] + P_pv_load[key] + P_dis[key] == load_t)
            m.addConstr(P_ch[key] == P_grid_ch[key] + P_pv_ch[key])
            m.addConstr(P_pv_load[key] + P_pv_ch[key] + P_pv_curt[key] <= pv_t)
            m.addConstr(P_ch[key] <= pb_val * u_var[key])
            m.addConstr(P_dis[key] <= pb_val * (1 - u_var[key]))

            if t == 0:
                E_prev = E_daystart[di]
            else:
                E_prev = E_soc[di, t - 1]

            m.addConstr(E_soc[key] == E_prev + eta_ch * P_ch[key] - (1/eta_dis) * P_dis[key])
            m.addConstr(E_soc[key] >= soc_min * eb_val)
            m.addConstr(E_soc[key] <= soc_max * eb_val)

            m.addConstr(Dmax[mo] >= kappa * (P_grid_load[key] + P_grid_ch[key]))

            m.addConstr(P_ch_g[key] <= P_pv_ch[key])
            m.addConstr(P_dis_g[key] <= P_dis[key])
            if t == 0:
                Eg_prev = E_g_daystart[di]
            else:
                Eg_prev = E_g[di, t - 1]
            m.addConstr(E_g[key] == Eg_prev + eta_ch * P_ch_g[key] - (1/eta_dis) * P_dis_g[key])
            m.addConstr(E_g[key] <= E_soc[key])

            m.addConstr(P_dis[key] == gp.quicksum(e_dis_seg[di, t, k] for k in range(n_seg)))
            for k in range(n_seg):
                m.addConstr(e_dis_seg[di, t, k] <= (b_k[k+1] - b_k[k]) * eb_val)

    # Cross-day SOC
    m.addConstr(E_daystart[day_indices[0]] == soc_init * eb_val)
    m.addConstr(E_g_daystart[day_indices[0]] == 0)
    for pos in range(len(day_indices) - 1):
        m.addConstr(E_daystart[day_indices[pos+1]] == E_soc[day_indices[pos], n_hours-1])
        m.addConstr(E_g_daystart[day_indices[pos+1]] == E_g[day_indices[pos], n_hours-1])

    # Over-contract
    for mo in all_months:
        m.addConstr(Dmax[mo] - cc_val == O1[mo] + O2[mo])
        m.addConstr(O1[mo] <= 0.10 * cc_val)

    # RE20
    if not no_re20:
        E_pv_self = gp.quicksum(P_pv_load[di, t] for di in day_indices for t in range(n_hours))
        E_dis_green = gp.quicksum(P_dis_g[di, t] for di in day_indices for t in range(n_hours))
        m.addConstr(E_pv_self + E_dis_green + E_TREC >= re_target * total_load)

    # Objective (operational only, investment is sunk)
    AEC_ene = gp.quicksum(
        get_tou_price(pd.Timestamp(cal_lookup.loc[di, 'calendar_day']).month,
                      pd.Timestamp(cal_lookup.loc[di, 'calendar_day']).day,
                      pd.Timestamp(cal_lookup.loc[di, 'calendar_day']).weekday(),
                      t) *
        (P_grid_load[di, t] + P_grid_ch[di, t])
        for di in day_indices for t in range(n_hours))

    AEC_basic = sum(get_monthly_basic_charge(mo, CFG) * cc_val for mo in all_months)

    AEC_over = gp.quicksum(
        get_monthly_basic_charge(mo, CFG) * (CFG['oc_within_10pct_mult'] * O1[mo] +
                                              CFG['oc_beyond_10pct_mult'] * O2[mo])
        for mo in all_months)

    AEC_green = c_trec * E_TREC

    AEC_deg = gp.quicksum(
        lam_k[k] * e_dis_seg[di, t, k]
        for di in day_indices for t in range(n_hours) for k in range(n_seg))

    AEC_inv = crf_bess * (c_bp * pb_val + c_be * eb_val)

    m.setObjective(AEC_ene + AEC_over + AEC_green + AEC_deg, GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        print(f"  Replay failed!")
        return None

    # Total replay cost includes investment (fixed)
    replay_ene = AEC_ene.getValue()
    replay_over = AEC_over.getValue()
    replay_green = AEC_green.getValue()
    replay_deg = AEC_deg.getValue()
    replay_total = AEC_inv + AEC_basic + replay_ene + replay_over + replay_green + replay_deg

    pv_self_val = sum(P_pv_load[di, t].X for di in day_indices for t in range(n_hours))
    dis_green_val = sum(P_dis_g[di, t].X for di in day_indices for t in range(n_hours))
    trec_val = E_TREC.X
    re_pct = (pv_self_val + dis_green_val + trec_val) / total_load * 100 if total_load > 0 else 0

    # Monthly bills
    monthly_bills = {}
    for mo in all_months:
        basic = get_monthly_basic_charge(mo, CFG) * cc_val
        over = get_monthly_basic_charge(mo, CFG) * (
            CFG['oc_within_10pct_mult'] * O1[mo].X +
            CFG['oc_beyond_10pct_mult'] * O2[mo].X)
        # Energy for this month
        mo_days = [di for di in day_indices if int(cal_lookup.loc[di, 'month_id']) == mo]
        ene_mo = sum(
            get_tou_price(pd.Timestamp(cal_lookup.loc[di, 'calendar_day']).month,
                          pd.Timestamp(cal_lookup.loc[di, 'calendar_day']).day,
                          pd.Timestamp(cal_lookup.loc[di, 'calendar_day']).weekday(),
                          t) *
            (P_grid_load[di, t].X + P_grid_ch[di, t].X)
            for di in mo_days for t in range(n_hours))
        monthly_bills[mo] = basic + over + ene_mo

    over_months = sum(1 for mo in all_months if O1[mo].X + O2[mo].X > 1)
    worst_bill = max(monthly_bills.values()) if monthly_bills else 0

    result = {
        'case_id': case_id,
        'replay_total_M': round(replay_total / 1e6, 2),
        'replay_ene_M': round(replay_ene / 1e6, 2),
        'replay_over_M': round(replay_over / 1e6, 2),
        'replay_green_M': round(replay_green / 1e6, 2),
        'replay_deg_M': round(replay_deg / 1e6, 2),
        'replay_inv_M': round(AEC_inv / 1e6, 2),
        'replay_basic_M': round(AEC_basic / 1e6, 2),
        'RE_pct': round(re_pct, 1),
        'TREC_kWh': round(E_TREC.X, 0),
        'over_months': over_months,
        'worst_bill_M': round(worst_bill / 1e6, 2),
        'monthly_bills': {mo: round(v / 1e6, 4) for mo, v in monthly_bills.items()},
    }

    print(f"  Replay {case_id}: total={replay_total/1e6:.2f}M, RE={re_pct:.1f}%, "
          f"over_months={over_months}, worst_bill={worst_bill/1e6:.2f}M")

    return result
