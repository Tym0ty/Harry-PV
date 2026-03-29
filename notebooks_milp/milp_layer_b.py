"""
MILP Layer B — Daily Sequential Day-Ahead Solve + Fixed-Design Replay
Per F0329MILP_Spec.pdf (Batch 17), §6.5.4-6.5.13 + §7.

For each calendar day n (sequential):
  1. Load rolling DA input slice for day n (info frozen at day-ahead gate)
  2. Solve 24-hour MILP: u(n,t), P_ch(n,t), P_dis(n,t) shared;
     scenario-dependent settlement for probabilistic cases
  3. Record end-of-day state: SOC, Green SOC, month max-demand, RE accumulators

Replay mode (§7):
  - Fix (CC*, P_B*, E_B*) and u_plan from solve
  - Settle against realized pv_realized_kw / load_realized_kw from truth package
  - Apply fixed-mode clipping: P_ch_rep = min(P_ch_plan, SOC headroom / eta_ch)
  - Output: daily_settlement_trace, replay_monthly_bill, replay_summary

Layer B must NOT:
  - Reselect annual design
  - Use truth data during solve
  - Perform intra-day MPC reforecasting

Usage:
    python milp_layer_b.py --case C0 --design design_results_C0.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from milp_common import get_config, CASE_TABLE, load_data, load_truth, get_tou_price


def _hard_fail(msg):
    raise RuntimeError(f"[LAYER_B HARD FAIL] {msg}")


def _crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


# ── Single-day 24-hour MILP solve ─────────────────────────────

def solve_day(di, day_data, design, CFG, prev_state):
    """
    Solve the 24-hour day-ahead MILP for calendar day n.

    Parameters
    ----------
    di        : day_index (1-based)
    day_data  : dict from load_data() for this day
    design    : {"CC": float, "P_B": float, "E_B": float}
    CFG       : config dict
    prev_state: {"E_init": float, "E_g_init": float,
                 "month_demand": float, "month_id_prev": int,
                 "E_load_yr": float, "E_pv_self_yr": float, "E_dis_g_yr": float}

    Returns
    -------
    result dict or None on failure
    """
    if not GUROBI_AVAILABLE:
        return None

    CC, P_B, E_B = design["CC"], design["P_B"], design["E_B"]
    eta_ch   = CFG['eff_charge']
    eta_dis  = CFG['eff_discharge']
    soc_min  = CFG['soc_min']
    soc_max  = CFG['soc_max']
    kappa    = CFG['kappa']
    re_tgt   = CFG['re_target']
    c_trec   = CFG['trec_cost_per_kwh']
    b_k      = CFG['pwl_deg_b_k']
    lam_k    = CFG['pwl_deg_lambda_k']
    c_basic_s  = CFG['basic_charge_summer']
    c_basic_ns = CFG['basic_charge_nonsummer']
    m_oc10   = CFG['oc_within_10pct_mult']
    m_oc10p  = CFG['oc_beyond_10pct_mult']
    n_seg    = len(lam_k)

    E_MAX = soc_max * E_B
    E_MIN = soc_min * E_B
    E_init   = prev_state["E_init"]
    E_g_init = prev_state["E_g_init"]
    D_month  = prev_state["month_demand"]    # running month max-demand
    is_summer = day_data["is_summer"]
    c_basic   = c_basic_s if is_summer else c_basic_ns
    tou       = day_data["tou"]
    scenarios = day_data["scenarios"]

    m_model = gp.Model(f"B_d{di}")
    m_model.setParam("OutputFlag", 0)
    m_model.setParam("MIPGap", CFG.get("mip_gap", 1e-3))
    m_model.setParam("TimeLimit", CFG.get("time_limit", 60))

    # Shared controls (day-ahead decision, common across scenarios)
    u      = [m_model.addVar(vtype=GRB.BINARY, name=f"u_{t}") for t in range(24)]
    P_ch   = [m_model.addVar(lb=0, ub=P_B, name=f"Pch_{t}") for t in range(24)]
    P_dis  = [m_model.addVar(lb=0, ub=P_B, name=f"Pdis_{t}") for t in range(24)]

    # Scenario-dependent variables
    P_pv_load  = {}
    P_pv_ch    = {}
    P_pv_curt  = {}
    P_grid_load = {}
    P_grid_ch   = {}
    E_s         = {}
    E_g_s       = {}

    # PWL degradation
    d_seg = [[m_model.addVar(lb=0) for _ in range(n_seg)] for t in range(24)]

    for sc in scenarios:
        sid = sc["scenario_id"]
        pv_arr = sc["pv_kw"]
        ld_arr = sc["load_kw"]
        for t in range(24):
            P_pv_load[(sid, t)]   = m_model.addVar(lb=0, ub=pv_arr[t])
            P_pv_ch[(sid, t)]     = m_model.addVar(lb=0, ub=pv_arr[t])
            P_pv_curt[(sid, t)]   = m_model.addVar(lb=0, ub=pv_arr[t])
            P_grid_load[(sid, t)] = m_model.addVar(lb=0, ub=ld_arr[t])
            P_grid_ch[(sid, t)]   = m_model.addVar(lb=0, ub=P_B)
            E_s[(sid, t)]         = m_model.addVar(lb=E_MIN, ub=E_MAX)
            E_g_s[(sid, t)]       = m_model.addVar(lb=0, ub=E_MAX)

    m_model.update()

    # ── Constraints ───────────────────────────────────────────
    for sc in scenarios:
        sid    = sc["scenario_id"]
        pv_arr = sc["pv_kw"]
        ld_arr = sc["load_kw"]
        for t in range(24):
            # Power balance
            m_model.addConstr(
                ld_arr[t] == P_grid_load[(sid, t)] + P_pv_load[(sid, t)]
                            + P_dis[t] - P_ch[t])
            # PV allocation
            m_model.addConstr(
                P_pv_load[(sid, t)] + P_pv_ch[(sid, t)] + P_pv_curt[(sid, t)]
                == pv_arr[t])
            # Charge source
            m_model.addConstr(
                P_ch[t] == P_grid_ch[(sid, t)] + P_pv_ch[(sid, t)])
            # Battery mode
            m_model.addConstr(P_ch[t]  <= P_B * u[t])
            m_model.addConstr(P_dis[t] <= P_B * (1 - u[t]))
            # SOC dynamics
            E_prev  = E_init  if t == 0 else E_s[(sid, t - 1)]
            Eg_prev = E_g_init if t == 0 else E_g_s[(sid, t - 1)]
            m_model.addConstr(
                E_s[(sid, t)] == E_prev + eta_ch * P_ch[t] - (1 / eta_dis) * P_dis[t])
            m_model.addConstr(
                E_g_s[(sid, t)] == Eg_prev + eta_ch * P_pv_ch[(sid, t)]
                                  - (1 / eta_dis) * P_dis[t])
            m_model.addConstr(E_g_s[(sid, t)] >= 0)
            m_model.addConstr(E_g_s[(sid, t)] <= E_s[(sid, t)])
            # Contract constraint
            m_model.addConstr(
                P_grid_load[(sid, t)] + P_grid_ch[(sid, t)] <= CC * kappa)

    # PWL degradation
    for t in range(24):
        m_model.addConstr(P_dis[t] == gp.quicksum(d_seg[t]))
        for k in range(n_seg):
            seg_ub = (b_k[k + 1] - b_k[k]) * E_B if k + 1 < len(b_k) else E_B
            m_model.addConstr(d_seg[t][k] <= seg_ub)

    # ── Objective: minimise day operational cost ───────────────
    obj = gp.LinExpr()
    for sc in scenarios:
        sid = sc["scenario_id"]
        pi  = sc["prob"]
        for t in range(24):
            obj.addTerms(pi * tou[t], P_grid_load[(sid, t)])
            obj.addTerms(pi * tou[t], P_grid_ch[(sid, t)])
        for t in range(24):
            for k in range(n_seg):
                obj.addTerms(pi * lam_k[k], d_seg[t][k])

    m_model.setObjective(obj, GRB.MINIMIZE)

    t0 = time.time()
    m_model.optimize()
    solve_time = time.time() - t0

    if m_model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None

    # ── Extract results ────────────────────────────────────────
    u_plan   = [u[t].X for t in range(24)]
    P_ch_plan  = [P_ch[t].X for t in range(24)]
    P_dis_plan = [P_dis[t].X for t in range(24)]

    # Expected end-of-day SOC across scenarios
    E_end_exp  = sum(sc["prob"] * E_s[(sc["scenario_id"], 23)].X for sc in scenarios)
    Eg_end_exp = sum(sc["prob"] * E_g_s[(sc["scenario_id"], 23)].X for sc in scenarios)

    # Expected grid import for max-demand tracking
    exp_grid   = [
        sum(sc["prob"] * (P_grid_load[(sc["scenario_id"], t)].X
                          + P_grid_ch[(sc["scenario_id"], t)].X)
            for sc in scenarios)
        for t in range(24)
    ]
    new_month_demand = max(D_month, max(exp_grid) * kappa)

    return {
        "day_index":    di,
        "calendar_day": day_data["calendar_day"],
        "month_id":     day_data["month_id"],
        "u_plan":       u_plan,
        "P_ch_plan":    P_ch_plan,
        "P_dis_plan":   P_dis_plan,
        "E_end":        E_end_exp,
        "Eg_end":       Eg_end_exp,
        "month_demand": new_month_demand,
        "obj_val":      m_model.ObjVal,
        "solve_time":   solve_time,
    }


# ── Fixed-design replay for one day ───────────────────────────

def replay_day(di, truth_day, design, plan, state, CFG):
    """
    Replay one day against realized truth using fixed u_plan.
    Fixed-mode clipping: P_ch_rep = min(P_ch_plan, SOC_headroom/eta_ch)
    """
    CC, P_B, E_B = design["CC"], design["P_B"], design["E_B"]
    eta_ch  = CFG['eff_charge']
    eta_dis = CFG['eff_discharge']
    soc_max = CFG['soc_max']
    soc_min = CFG['soc_min']
    kappa   = CFG['kappa']
    lam_k   = CFG['pwl_deg_lambda_k']
    b_k     = CFG['pwl_deg_b_k']

    E   = state["E_init"]
    E_g = state["E_g_init"]
    is_summer = truth_day["is_summer"]
    tou = truth_day["tou"]

    rows = []
    for t in range(24):
        pv_real  = float(truth_day["pv_kw"][t])
        load_real = float(truth_day["load_kw"][t])
        u_t      = int(plan["u_plan"][t])
        Pch_plan = float(plan["P_ch_plan"][t])
        Pdis_plan= float(plan["P_dis_plan"][t])

        # Fixed-mode clipping (SOC limits)
        if u_t == 1:   # charge mode
            headroom = (soc_max * E_B - E) / eta_ch
            Pch_rep  = min(Pch_plan, max(0.0, headroom), P_B)
            Pdis_rep = 0.0
        else:           # discharge mode
            avail    = (E - soc_min * E_B) * eta_dis
            Pdis_rep = min(Pdis_plan, max(0.0, avail), P_B)
            Pch_rep  = 0.0

        # PV first-priority to load, rest to charge or curtailment
        pv_to_load = min(pv_real, load_real)
        pv_to_ch   = min(max(0.0, pv_real - pv_to_load), Pch_rep)
        pv_curt    = max(0.0, pv_real - pv_to_load - pv_to_ch)
        pch_grid   = max(0.0, Pch_rep - pv_to_ch)
        grid_load  = max(0.0, load_real - pv_to_load - Pdis_rep + Pch_rep)
        grid_total = grid_load + pch_grid

        # SOC update
        E   = E   + eta_ch * Pch_rep - (1 / eta_dis) * Pdis_rep
        E   = max(soc_min * E_B, min(soc_max * E_B, E))
        E_g = E_g + eta_ch * pv_to_ch - (1 / eta_dis) * Pdis_rep
        E_g = max(0.0, min(E, E_g))

        # PWL degradation cost
        deg_cost = 0.0
        remaining = Pdis_rep
        for k in range(len(lam_k)):
            seg_cap = (b_k[k + 1] - b_k[k]) * E_B if k + 1 < len(b_k) else E_B
            in_seg  = min(remaining, seg_cap)
            deg_cost += in_seg * lam_k[k]
            remaining -= in_seg

        rows.append({
            "day_index":   di,
            "hour_local":  t + 1,
            "pv_realized": pv_real,
            "load_realized": load_real,
            "u_plan":      u_t,
            "P_ch_rep":    Pch_rep,
            "P_dis_rep":   Pdis_rep,
            "grid_total":  grid_total,
            "energy_cost": grid_total * tou[t],
            "deg_cost":    deg_cost,
            "E_soc":       E,
            "E_green":     E_g,
        })

    return rows, E, E_g


# ── Full-year Layer B run ──────────────────────────────────────

def run_layer_b(case_id, design, CFG, output_dir=None):
    """
    Sequential daily solve + replay for a full case year.
    design: {"CC": float, "P_B": float, "E_B": float}
    """
    print("=" * 60)
    print(f"Layer B — Sequential Day-Ahead Solve  [{case_id}]")
    print("=" * 60)

    out_dir = Path(output_dir or CFG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    case = next(c for c in CASE_TABLE if c["case_id"] == case_id)
    day_data, day_indices, scenario_ids = load_data(CFG, case)
    truth_df, calendar_df = load_truth(CFG)

    # Acceptance: no truth columns in C0-C3 solve packages (spec truth isolation rule)
    from milp_common import CANONICAL_ROLLING_FILES
    solve_pkg_path = Path(CFG['bridge_dir']) / CANONICAL_ROLLING_FILES[case_id]
    if solve_pkg_path.exists() and case_id != "C_PFI":
        sample_cols = set(pd.read_parquet(solve_pkg_path, columns=None).columns)
        forbidden = {"pv_realized_kw", "load_realized_kw"}
        if forbidden & sample_cols:
            _hard_fail(f"Truth columns {forbidden & sample_cols} found in solve package for {case_id}.")

    # State initialisation
    soc_init = CFG['soc_init']
    E_B = design["E_B"]
    state = {
        "E_init":        soc_init * E_B,
        "E_g_init":      0.0,
        "month_demand":  0.0,
        "month_id_prev": None,
        "E_load_yr":     0.0,
        "E_pv_self_yr":  0.0,
        "E_dis_g_yr":    0.0,
    }

    plan_rows    = []
    state_rows   = []
    replay_all   = []
    monthly_bills = {}

    for di in day_indices:
        dd = day_data[di]
        m  = dd["month_id"]

        # Month boundary reset for demand accumulator
        if state["month_id_prev"] is not None and m != state["month_id_prev"]:
            state["month_demand"] = 0.0
        state["month_id_prev"] = m

        # ── Layer B solve ──────────────────────────────────────
        result = solve_day(di, dd, design, CFG, state)
        if result is None:
            print(f"  [WARN] Day {di} solve failed — using zero dispatch")
            result = {
                "day_index": di, "calendar_day": dd["calendar_day"],
                "month_id": m, "u_plan": [0]*24,
                "P_ch_plan": [0.0]*24, "P_dis_plan": [0.0]*24,
                "E_end": state["E_init"], "Eg_end": state["E_g_init"],
                "month_demand": state["month_demand"],
                "obj_val": 0.0, "solve_time": 0.0,
            }

        # ── Replay ────────────────────────────────────────────
        truth_day_rows = truth_df[truth_df["day_index"] == di].sort_values("hour_local")
        if len(truth_day_rows) == 0:
            print(f"  [WARN] No truth data for day {di} — using plan SOC")
            state["E_init"]   = result["E_end"]
            state["E_g_init"] = result["Eg_end"]
        else:
            truth_day_info = {
                "is_summer": bool(dd["is_summer"]),
                "tou":       dd["tou"],
                "pv_kw":     truth_day_rows["pv_realized_kw"].values,
                "load_kw":   truth_day_rows["load_realized_kw"].values,
            }
            rep_rows, E_new, Eg_new = replay_day(
                di, truth_day_info, design, result, state, CFG)
            replay_all.extend(rep_rows)

            # Monthly bill accumulators
            if m not in monthly_bills:
                monthly_bills[m] = {
                    "month_id": m,
                    "is_summer": bool(dd["is_summer"]),
                    "energy_cost": 0.0,
                    "deg_cost": 0.0,
                    "peak_grid": 0.0,
                    "total_pv_self": 0.0,
                    "total_dis_g": 0.0,
                    "total_load": 0.0,
                }
            for r in rep_rows:
                monthly_bills[m]["energy_cost"]  += r["energy_cost"]
                monthly_bills[m]["deg_cost"]      += r["deg_cost"]
                monthly_bills[m]["peak_grid"]      = max(
                    monthly_bills[m]["peak_grid"], r["grid_total"])
                monthly_bills[m]["total_load"]     += r["load_realized"]
                monthly_bills[m]["total_pv_self"]  += r["pv_realized"]
                monthly_bills[m]["total_dis_g"]    += r["P_dis_rep"]

            # State carry-over from replay (truth-settled SOC)
            state["E_init"]   = E_new
            state["E_g_init"] = Eg_new

        state["month_demand"] = result["month_demand"]

        plan_rows.append({
            "day_index":    di,
            "calendar_day": dd["calendar_day"],
            "month_id":     m,
            "obj_val":      result["obj_val"],
            "solve_time":   result["solve_time"],
        })
        state_rows.append({
            "day_index":    di,
            "calendar_day": dd["calendar_day"],
            "month_id":     m,
            "E_start":      state["E_init"],
            "Eg_start":     state["E_g_init"],
            "month_demand": state["month_demand"],
        })

        if di % 30 == 0:
            print(f"  Day {di}/{len(day_indices)} done")

    # ── Compute monthly bills ──────────────────────────────────
    kappa = CFG['kappa']
    c_basic_s  = CFG['basic_charge_summer']
    c_basic_ns = CFG['basic_charge_nonsummer']
    m_oc10  = CFG['oc_within_10pct_mult']
    m_oc10p = CFG['oc_beyond_10pct_mult']
    c_trec  = CFG['trec_cost_per_kwh']
    re_tgt  = CFG['re_target']
    CC = design["CC"]

    bill_rows = []
    for m_id, mb in monthly_bills.items():
        c_basic = c_basic_s if mb["is_summer"] else c_basic_ns
        D_m = mb["peak_grid"] * kappa
        basic = c_basic * D_m
        oc_band = CC * 0.1
        oc = 0.0
        if D_m > CC:
            exc = D_m - CC
            if exc <= oc_band:
                oc = m_oc10 * c_basic * exc
            else:
                oc = m_oc10 * c_basic * oc_band + m_oc10p * c_basic * (exc - oc_band)
        bill_rows.append({
            "month_id":    m_id,
            "is_summer":   mb["is_summer"],
            "basic_NTD":   basic,
            "overcontract_NTD": oc,
            "energy_NTD":  mb["energy_cost"],
            "deg_NTD":     mb["deg_cost"],
            "total_NTD":   basic + oc + mb["energy_cost"] + mb["deg_cost"],
            "peak_grid_kw": D_m,
        })

    bill_df = pd.DataFrame(bill_rows).sort_values("month_id")

    # Annual RE20 / T-REC
    total_load = sum(mb["total_load"] for mb in monthly_bills.values())
    total_pv_self = sum(mb["total_pv_self"] for mb in monthly_bills.values())
    total_dis_g   = sum(mb["total_dis_g"] for mb in monthly_bills.values())
    re_kwh   = total_pv_self + total_dis_g
    re_pct   = re_kwh / max(total_load, 1.0) * 100
    trec_shortfall = max(0.0, re_tgt * total_load - re_kwh)
    trec_cost = trec_shortfall * c_trec

    total_replay = bill_df["total_NTD"].sum() + trec_cost
    crf_bess = CFG['crf_bess'] if 'crf_bess' in CFG else _crf(CFG['discount_rate'], CFG['lifetime_bess'])
    aec_inv  = crf_bess * (CFG['capex_bess_power_per_kw'] * design["P_B"]
                           + CFG['capex_bess_energy_per_kwh'] * E_B)

    summary = {
        "case_id":         case_id,
        "CC":              design["CC"],
        "P_B":             design["P_B"],
        "E_B":             design["E_B"],
        "replay_total_NTD": total_replay + aec_inv,
        "replay_opex_NTD":  total_replay,
        "AEC_inv_NTD":      aec_inv,
        "basic_NTD":        float(bill_df["basic_NTD"].sum()),
        "overcontract_NTD": float(bill_df["overcontract_NTD"].sum()),
        "energy_NTD":       float(bill_df["energy_NTD"].sum()),
        "deg_NTD":          float(bill_df["deg_NTD"].sum()),
        "trec_NTD":         trec_cost,
        "RE_pct":           re_pct,
        "trec_shortfall_kwh": trec_shortfall,
        "n_overcontract_months": int((bill_df["overcontract_NTD"] > 0).sum()),
    }

    # ── Save outputs ───────────────────────────────────────────
    pd.DataFrame(replay_all).to_parquet(
        out_dir / f"daily_settlement_trace_{case_id}.parquet", index=False)
    pd.DataFrame(plan_rows).to_parquet(
        out_dir / f"daily_plan_trace_{case_id}.parquet", index=False)
    pd.DataFrame(state_rows).to_parquet(
        out_dir / f"rolling_state_trace_{case_id}.parquet", index=False)
    bill_df.to_csv(out_dir / f"replay_monthly_bill_{case_id}.csv", index=False)
    with open(out_dir / f"replay_summary_{case_id}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Replay total: {total_replay/1e6:.3f} M NTD "
          f"(incl. CAPEX: {(total_replay+aec_inv)/1e6:.3f} M)")
    print(f"  RE: {re_pct:.1f}%  |  T-REC shortfall: {trec_shortfall:.0f} kWh")
    print(f"  Over-contract months: {summary['n_overcontract_months']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer B — Sequential Daily Solve + Replay")
    parser.add_argument("--case",    required=True, choices=["C0","C1","C2","C3","C_PFI"])
    parser.add_argument("--design",  required=True,
                        help="Path to design_results_CASE.json from Layer A")
    parser.add_argument("--bridge_dir", default="../bridge_outputs_fullyear")
    parser.add_argument("--output_dir", default="../milp_outputs")
    args = parser.parse_args()

    with open(args.design) as f:
        design = json.load(f)

    CFG = get_config()
    CFG['bridge_dir'] = args.bridge_dir
    CFG['output_dir'] = args.output_dir

    run_layer_b(args.case, design, CFG, args.output_dir)
