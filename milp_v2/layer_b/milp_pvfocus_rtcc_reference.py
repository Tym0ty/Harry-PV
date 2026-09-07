"""RTCC reference-trajectory extensions for the PV-focused revised-fullspec MILP.

This module is intentionally separate from ``milp_pvfocus_revised_fullspec.py``.
It reuses the same configuration, physical constraints, tariff functions, and
replay-compatible output shape, but adds only the RTCC controller terms:

* optional reference-peak OC uplift,
* optional scenario-wise terminal SOC equality.
"""

from __future__ import annotations

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from common import get_basic_charge
from layer_b.milp_pvfocus_revised_fullspec import (
    CFG,
    GOOD_STATUSES,
    _as_float_array,
    _month_pre_peaks,
    _normalize_scenarios,
    _over_contract_cost_expr,
    _over_contract_cost_value,
    _safe_expr_value,
)


def solve_pvfocus_rtcc_milp(
    *,
    scenarios: list[dict],
    state: dict,
    design: dict,
    cfg: dict | None = None,
    issue_time=None,
    day_index: int | None = None,
    month_ids: list[int] | None = None,
    tou: np.ndarray | list[float] | None = None,
    horizon: int = 24,
    dt: float = 1.0,
    time_limit: int = 60,
    mip_gap: float = 0.005,
    verbose: bool = False,
    reference_peak_by_month: dict[int, float] | None = None,
    use_reference_peak_uplift: bool = False,
    terminal_soc_target_kwh: float | None = None,
    reference_midnight_floor_kwh: float | None = None,
    is_reference_stage: bool = False,
    is_final_step: bool = False,
) -> dict:
    """Solve one RTCC reference or main-stage rolling MILP.

    ``is_reference_stage`` is used only for diagnostics and for the optional
    midnight SOC floor.  Passing a single scenario with weight 1 implements the
    nominal reference stage; passing K scenarios implements the stochastic main
    stage.  No CCOR shield variables are created in either case.
    """
    if cfg is None:
        cfg = CFG

    scenarios = _normalize_scenarios(scenarios, horizon)
    weight_sum = sum(sc["pi"] for sc in scenarios)
    if abs(weight_sum - 1.0) > 1e-8:
        raise ValueError(f"Scenario weights do not sum to 1: {weight_sum}")

    bat = cfg["battery"]
    dem = cfg["demand"]
    re20 = cfg["re20"]

    cc = float(design["CC"])
    pb = float(design["PB"])
    eb = float(design["EB"])

    e_init = float(state["E_soc"])
    eg_init = float(state.get("E_g", 0.0))
    current_month = int(state.get("current_month_id", month_ids[0] if month_ids else 1))

    eta_ch = float(bat["eta_ch"])
    eta_dis = float(bat["eta_dis"])
    soc_min = float(bat["soc_min"])
    soc_max = float(bat["soc_max"])
    soc_init = float(bat["soc_init"])
    eps_term = float(bat.get("epsilon_term", 0.05))
    b_k = [float(x) for x in bat["degradation_segments"]["b"]]
    lam_k = [float(x) for x in bat["degradation_segments"]["lambda"]]
    n_seg = len(lam_k)

    kappa = float(dem["kappa"])
    tier1 = float(dem["over_contract_tier1_ratio"])
    m1 = float(dem["over_contract_m1"])
    m2 = float(dem["over_contract_m2"])
    alpha_re = float(re20["re_target_ratio"])
    c_trec = float(re20["c_trec_ntd_per_kwh"])

    if tou is None:
        raise ValueError("tou path is required")
    tou_arr = _as_float_array(tou, horizon, "tou")

    if month_ids is None:
        month_ids = [current_month] * horizon
    if len(month_ids) != horizon:
        raise ValueError("month_ids length must equal horizon")
    month_ids = [int(m) for m in month_ids]
    months = sorted(set(month_ids))
    d_pre = _month_pre_peaks(state, month_ids, current_month)
    reference_peak_by_month = {int(k): float(v) for k, v in (reference_peak_by_month or {}).items()}

    l_cum = float(state.get("L_cum", 0.0))
    r_cum = float(state.get("R_cum", 0.0))
    l_rem_hat = float(state.get("L_rem_hat", 0.0))
    r_rem_hat = float(state.get("R_rem_hat", 0.0))
    y_trec_pre = float(state.get("Y_TREC_pre", 0.0))

    t_start = time.time()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1 if verbose else 0)
        env.setParam("TimeLimit", time_limit)
        env.setParam("MIPGap", mip_gap)
        env.start()
        mdl = gp.Model("pvfocus_rtcc_reference", env=env)

        W = range(len(scenarios))
        T = range(horizon)
        M = months

        u = mdl.addVars(len(scenarios), horizon, vtype=GRB.BINARY, name="u")
        p_ch = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_ch")
        p_dis = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_dis")
        e = mdl.addVars(len(scenarios), horizon, lb=soc_min * eb, ub=soc_max * eb, name="E")
        e_seg = mdl.addVars(len(scenarios), horizon, n_seg, lb=0.0, name="e_seg")

        p_gl = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_grid_load")
        p_gc = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_grid_ch")
        p_pvl = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_pv_load")
        p_pvc = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_pv_ch")
        p_pvcu = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_pv_curt")
        p_grid = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_grid")

        e_g = mdl.addVars(len(scenarios), horizon, lb=0.0, ub=eb, name="E_g")
        p_dis_g = mdl.addVars(len(scenarios), horizon, lb=0.0, name="P_dis_g")

        d_max = mdl.addVars(len(scenarios), M, lb=0.0, name="D_max")
        over = mdl.addVars(len(scenarios), M, lb=0.0, name="Over")
        o1 = mdl.addVars(len(scenarios), M, lb=0.0, ub=tier1 * cc, name="O1")
        o2 = mdl.addVars(len(scenarios), M, lb=0.0, name="O2")

        d_refmax = None
        over_ref = o1_ref = o2_ref = None
        if use_reference_peak_uplift:
            d_refmax = mdl.addVars(len(scenarios), M, lb=0.0, name="D_max_ref_uplift")
            over_ref = mdl.addVars(len(scenarios), M, lb=0.0, name="Over_ref_uplift")
            o1_ref = mdl.addVars(len(scenarios), M, lb=0.0, ub=tier1 * cc, name="O1_ref_uplift")
            o2_ref = mdl.addVars(len(scenarios), M, lb=0.0, name="O2_ref_uplift")

        y_trec = mdl.addVars(len(scenarios), lb=0.0, name="Y_TREC")

        p_ch_exe = mdl.addVar(lb=0.0, name="p_ch_exe")
        p_dis_exe = mdl.addVar(lb=0.0, name="p_dis_exe")
        u_exe = mdl.addVar(vtype=GRB.BINARY, name="u_exe")

        for w, sc in enumerate(scenarios):
            for t in T:
                mdl.addConstr(p_ch[w, t] <= pb * u[w, t], name=f"ch_ub_{w}_{t}")
                mdl.addConstr(p_dis[w, t] <= pb * (1 - u[w, t]), name=f"dis_ub_{w}_{t}")

                e_prev = e_init if t == 0 else e[w, t - 1]
                mdl.addConstr(
                    e[w, t] == e_prev + eta_ch * p_ch[w, t] * dt - (1.0 / eta_dis) * p_dis[w, t] * dt,
                    name=f"soc_{w}_{t}",
                )
                mdl.addConstr(p_dis[w, t] * dt == gp.quicksum(e_seg[w, t, k] for k in range(n_seg)), name=f"seg_sum_{w}_{t}")
                for k in range(n_seg):
                    mdl.addConstr(e_seg[w, t, k] <= (b_k[k + 1] - b_k[k]) * eb, name=f"seg_cap_{w}_{t}_{k}")

                mdl.addConstr(p_gl[w, t] + p_pvl[w, t] + p_dis[w, t] == float(sc["load"][t]), name=f"Lbal_{w}_{t}")
                mdl.addConstr(p_ch[w, t] == p_gc[w, t] + p_pvc[w, t], name=f"Cbal_{w}_{t}")
                mdl.addConstr(p_pvl[w, t] + p_pvc[w, t] + p_pvcu[w, t] == float(sc["pv"][t]), name=f"PVbal_{w}_{t}")
                mdl.addConstr(p_grid[w, t] == p_gl[w, t] + p_gc[w, t], name=f"Grid_{w}_{t}")

                eg_prev = eg_init if t == 0 else e_g[w, t - 1]
                mdl.addConstr(e_g[w, t] == eg_prev + eta_ch * p_pvc[w, t] * dt - (1.0 / eta_dis) * p_dis_g[w, t] * dt, name=f"Eg_rec_{w}_{t}")
                mdl.addConstr(e_g[w, t] <= e[w, t], name=f"Eg_ub_{w}_{t}")
                mdl.addConstr(p_dis_g[w, t] <= p_dis[w, t], name=f"Pdg_dis_ub_{w}_{t}")

                mo = month_ids[t]
                mdl.addConstr(d_max[w, mo] >= kappa * p_grid[w, t], name=f"Dproxy_{w}_{t}")

            mdl.addConstr(p_ch[w, 0] == p_ch_exe, name=f"na_ch0_{w}")
            mdl.addConstr(p_dis[w, 0] == p_dis_exe, name=f"na_dis0_{w}")
            mdl.addConstr(u[w, 0] == u_exe, name=f"na_u0_{w}")

            if terminal_soc_target_kwh is not None:
                mdl.addConstr(e[w, horizon - 1] == float(terminal_soc_target_kwh), name=f"rtcc_terminal_soc_{w}")
            elif is_final_step:
                mdl.addConstr(e[w, horizon - 1] >= (soc_init - eps_term) * eb, name=f"term_lb_{w}")
                mdl.addConstr(e[w, horizon - 1] <= (soc_init + eps_term) * eb, name=f"term_ub_{w}")

            if is_reference_stage and reference_midnight_floor_kwh is not None:
                mdl.addConstr(e[w, horizon - 1] >= float(reference_midnight_floor_kwh), name=f"reference_midnight_floor_{w}")

            for mo in M:
                mdl.addConstr(d_max[w, mo] >= d_pre.get(mo, 0.0), name=f"Dpre_{w}_{mo}")
                mdl.addConstr(over[w, mo] >= d_max[w, mo] - cc, name=f"over_pos_{w}_{mo}")
                mdl.addConstr(over[w, mo] == o1[w, mo] + o2[w, mo], name=f"over_split_{w}_{mo}")

                if use_reference_peak_uplift:
                    ref_peak = reference_peak_by_month.get(mo, d_pre.get(mo, 0.0))
                    mdl.addConstr(d_refmax[w, mo] >= d_max[w, mo], name=f"refmax_mpc_{w}_{mo}")
                    mdl.addConstr(d_refmax[w, mo] >= ref_peak, name=f"refmax_ref_{w}_{mo}")
                    mdl.addConstr(over_ref[w, mo] >= d_refmax[w, mo] - cc, name=f"over_ref_pos_{w}_{mo}")
                    mdl.addConstr(over_ref[w, mo] == o1_ref[w, mo] + o2_ref[w, mo], name=f"over_ref_split_{w}_{mo}")

            r_onsite = gp.quicksum((p_pvl[w, t] + p_dis_g[w, t]) * dt for t in T)
            l_horizon = float(np.sum(sc["load"]) * dt)
            mdl.addConstr(
                y_trec[w] >= alpha_re * (l_cum + l_horizon + l_rem_hat) - (r_cum + r_onsite + r_rem_hat),
                name=f"trec_shortfall_{w}",
            )

        c_tou = gp.quicksum(sc["pi"] * tou_arr[t] * p_grid[w, t] * dt for w, sc in enumerate(scenarios) for t in T)
        c_deg = gp.quicksum(sc["pi"] * lam_k[k] * e_seg[w, t, k] for w, sc in enumerate(scenarios) for t in T for k in range(n_seg))

        c_oc_pre = 0.0
        c_oc_terms = []
        c_oc_standard_terms = []
        for mo in M:
            c_basic = get_basic_charge(mo, cfg)
            c_oc_pre += _over_contract_cost_value(d_pre.get(mo, 0.0), cc, cfg, mo)
            for w, sc in enumerate(scenarios):
                c_oc_standard_terms.append(sc["pi"] * _over_contract_cost_expr(c_basic, m1, m2, o1[w, mo], o2[w, mo]))
                if use_reference_peak_uplift:
                    c_oc_terms.append(sc["pi"] * _over_contract_cost_expr(c_basic, m1, m2, o1_ref[w, mo], o2_ref[w, mo]))
                else:
                    c_oc_terms.append(sc["pi"] * _over_contract_cost_expr(c_basic, m1, m2, o1[w, mo], o2[w, mo]))
        c_oc_standard = gp.quicksum(c_oc_standard_terms) - c_oc_pre
        c_oc = gp.quicksum(c_oc_terms) - c_oc_pre
        c_ref_uplift = c_oc - c_oc_standard

        c_trec_expr = gp.quicksum(sc["pi"] * c_trec * (y_trec[w] - y_trec_pre) for w, sc in enumerate(scenarios))

        mdl.setObjective(c_tou + c_oc + c_deg + c_trec_expr, GRB.MINIMIZE)
        mdl.optimize()

        solve_time = time.time() - t_start
        status = int(mdl.Status)
        if status not in GOOD_STATUSES or mdl.SolCount == 0:
            return {
                "status": status,
                "has_solution": False,
                "solve_time": solve_time,
                "message": "No feasible incumbent from RTCC MILP",
            }

        scenario_solution = []
        d_by_scenario = []
        d_refmax_by_scenario = []
        y_by_scenario = []
        for w, sc in enumerate(scenarios):
            dvals = {mo: float(d_max[w, mo].X) for mo in M}
            d_by_scenario.append(dvals)
            if use_reference_peak_uplift:
                d_refmax_by_scenario.append({mo: float(d_refmax[w, mo].X) for mo in M})
            else:
                d_refmax_by_scenario.append(dict(dvals))
            y_by_scenario.append(float(y_trec[w].X))
            for t in T:
                scenario_solution.append(
                    {
                        "scenario_id": sc["id"],
                        "scenario_index": w,
                        "tau": t,
                        "pi": float(sc["pi"]),
                        "load_kw": float(sc["load"][t]),
                        "pv_kw": float(sc["pv"][t]),
                        "P_ch_kw": float(p_ch[w, t].X),
                        "P_dis_kw": float(p_dis[w, t].X),
                        "E_kwh": float(e[w, t].X),
                        "u": float(u[w, t].X),
                        "P_grid_load_kw": float(p_gl[w, t].X),
                        "P_grid_ch_kw": float(p_gc[w, t].X),
                        "P_grid_kw": float(p_grid[w, t].X),
                        "P_pv_load_kw": float(p_pvl[w, t].X),
                        "P_pv_ch_kw": float(p_pvc[w, t].X),
                        "P_pv_curt_kw": float(p_pvcu[w, t].X),
                        "E_g_kwh": float(e_g[w, t].X),
                        "P_dis_g_kw": float(p_dis_g[w, t].X),
                        "D_proxy_kw": float(kappa * p_grid[w, t].X),
                    }
                )

        expected_e_end = float(sum(sc["pi"] * float(e[w, horizon - 1].X) for w, sc in enumerate(scenarios)))
        expected_eg_end = float(sum(sc["pi"] * float(e_g[w, horizon - 1].X) for w, sc in enumerate(scenarios)))
        p_ch_plan = np.array([sum(sc["pi"] * float(p_ch[w, t].X) for w, sc in enumerate(scenarios)) for t in T])
        p_dis_plan = np.array([sum(sc["pi"] * float(p_dis[w, t].X) for w, sc in enumerate(scenarios)) for t in T])
        e_plan = np.array([sum(sc["pi"] * float(e[w, t].X) for w, sc in enumerate(scenarios)) for t in T])

        obj_tou = _safe_expr_value(c_tou)
        obj_oc = _safe_expr_value(c_oc)
        obj_oc_standard = _safe_expr_value(c_oc_standard)
        obj_ref_uplift = _safe_expr_value(c_ref_uplift)
        obj_deg = _safe_expr_value(c_deg)
        obj_trec = _safe_expr_value(c_trec_expr)

        return {
            "status": status,
            "has_solution": True,
            "objective_total": float(mdl.ObjVal),
            "objective_tou": obj_tou,
            "objective_oc": obj_oc,
            "objective_oc_standard": obj_oc_standard,
            "objective_ref_uplift": obj_ref_uplift,
            "objective_deg": obj_deg,
            "objective_trec": obj_trec,
            "objective_shield": 0.0,
            "p_ch_exe": float(p_ch_exe.X),
            "p_dis_exe": float(p_dis_exe.X),
            "u_exe": float(u_exe.X),
            "P_ch_plan": p_ch_plan,
            "P_dis_plan": p_dis_plan,
            "E_plan": e_plan,
            "E_end": expected_e_end,
            "E_g_end": expected_eg_end,
            "D_max_by_scenario": d_by_scenario,
            "D_refmax_by_scenario": d_refmax_by_scenario,
            "Y_TREC_by_scenario": y_by_scenario,
            "z_shield_by_scenario": [0.0 for _ in scenarios],
            "scenario_solution": scenario_solution,
            "diagnostics": {
                "mode": "RTCC_REF" if is_reference_stage else "RTCC_MAIN",
                "issue_time": str(issue_time) if issue_time is not None else "",
                "day_index": day_index,
                "scenario_count": len(scenarios),
                "scenario_weight_sum": float(weight_sum),
                "month_ids": ";".join(str(x) for x in month_ids),
                "D_pre_current": float(state.get("D_mth", 0.0)),
                "reference_peak_by_month": ";".join(f"{k}:{v:.6f}" for k, v in sorted(reference_peak_by_month.items())),
                "terminal_soc_target_kwh": float(terminal_soc_target_kwh) if terminal_soc_target_kwh is not None else np.nan,
                "reference_midnight_floor_kwh": float(reference_midnight_floor_kwh) if reference_midnight_floor_kwh is not None else np.nan,
                "reference_peak_uplift_enabled": bool(use_reference_peak_uplift),
                "expected_reference_uplift": float(obj_ref_uplift),
                "L_cum": float(l_cum),
                "R_cum": float(r_cum),
                "L_rem_hat": float(l_rem_hat),
                "R_rem_hat": float(r_rem_hat),
                "Y_TREC_pre": float(y_trec_pre),
                "TREC_incremental": True,
            },
            "solve_time": float(solve_time),
        }
