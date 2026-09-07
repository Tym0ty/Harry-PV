#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MILP_DIR = REPO / "milp_v2"
sys.path.insert(0, str(MILP_DIR))
sys.path.insert(0, str(MILP_DIR / "experiments/mpc_fixed_rolling_horizon"))
sys.path.insert(0, str(MILP_DIR / "experiments/load_forecast_rerun"))
sys.path.insert(0, str(REPO / "new_pipeline/scripts/experiments"))

from common import get_basic_charge, get_tou_price, load_config  # noqa: E402
from mpc_fixed_horizon_utils import corrected_kpis  # noqa: E402
from pv_focus_bridge_utils import (  # noqa: E402
    OUT_DIR as BRIDGE_OUT,
    PV_TRUTH_SOURCE_LABEL,
    REPLAY_TRUTH_PACKAGE_PATH,
    full_da_pv_quantiles,
    full_id_pv_quantiles,
    load_profile_lookup,
    load_truth,
    write_md,
)
from run_da_formulation_rolling_controlled_experiments import load_design  # noqa: E402

CFG = load_config()
DESIGN = load_design()
CC = float(DESIGN["CC"])
PB = float(DESIGN["PB"])
EB = float(DESIGN["EB"])
H = 24
GOOD = {GRB.OPTIMAL, GRB.TIME_LIMIT, 13}
OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_mpc"
STD_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
PROB_SCENARIO = BRIDGE_OUT / "id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet"

CCOR_PARAMS = {
    "q_reserve": 0.80,
    "reserve_scale": 0.50,
    "reserve_cap_frac_usable": 0.30,
    "rho_reserve_ntd_per_kwh": 250.0,
    "rho_peak0_multiplier": 3.0,
    "cvar_lam": 0.0,
    "alpha": 0.90,
}


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def tou_path(start: pd.Timestamp) -> np.ndarray:
    return np.asarray(
        [
            get_tou_price(
                (start + pd.Timedelta(hours=j)).month,
                (start + pd.Timedelta(hours=j)).day,
                (start + pd.Timedelta(hours=j)).weekday(),
                (start + pd.Timedelta(hours=j)).hour,
            )
            for j in range(H)
        ],
        dtype=float,
    )


def state0() -> dict:
    bat = CFG["battery"]
    return {"E_soc": bat["soc_init"] * EB, "E_g": 0.0, "D_mth": 0.0}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w) / max(w.sum(), 1e-12)
    return float(v[np.searchsorted(c, q, side="left").clip(0, len(v) - 1)])


def reserve_metrics(scenarios: list[dict], state: dict) -> dict:
    bat = CFG["battery"]
    kappa = CFG["demand"]["kappa"]
    d_guard = max(CC, float(state["D_mth"]))
    grid_guard = d_guard / kappa
    areas, weights = [], []
    max_first = []
    for sc in scenarios:
        nl = np.asarray(sc["load"], dtype=float) - np.asarray(sc["pv"], dtype=float)
        areas.append(float(np.maximum(nl - grid_guard, 0.0).sum()))
        weights.append(float(sc["pi"]))
        max_first.append(float(nl[0]))
    areas = np.asarray(areas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    q_area = weighted_quantile(areas, weights, CCOR_PARAMS["q_reserve"])
    expected_area = float(np.sum(weights * areas))
    usable = (bat["soc_max"] - bat["soc_min"]) * EB
    cap = CCOR_PARAMS["reserve_cap_frac_usable"] * usable
    r_target = min(CCOR_PARAMS["reserve_scale"] * q_area, cap)
    reserve_soc_target = bat["soc_min"] * EB + r_target / bat["eta_dis"]
    reserve_soc_target = min(reserve_soc_target, bat["soc_max"] * EB)
    return {
        "D_guard_kw": d_guard,
        "grid_guard_kw": grid_guard,
        "R_expected_kwh": expected_area,
        "R_q80_kwh": q_area,
        "R_target_kwh": r_target,
        "reserve_soc_target_kwh": reserve_soc_target,
        "first_step_stress_netload_kw": float(max(max_first) if max_first else 0.0),
    }


def solve_ccor(day_index: int, day_dict: dict, state: dict, is_final_day: bool = False) -> dict:
    bat = CFG["battery"]
    dem = CFG["demand"]
    scenarios = day_dict["scenarios"]
    tou = day_dict["tou"]
    month_id = int(day_dict["month_id"])
    c_basic = get_basic_charge(month_id, CFG)
    n_sc = len(scenarios)
    T = range(H)
    W = range(n_sc)

    E_init = float(state["E_soc"])
    Eg_init = float(state["E_g"])
    D_init = float(state["D_mth"])
    rmeta = reserve_metrics(scenarios, state)

    eta_ch = bat["eta_ch"]
    eta_dis = bat["eta_dis"]
    soc_min = bat["soc_min"]
    soc_max = bat["soc_max"]
    soc_init = bat["soc_init"]
    eps_term = bat["epsilon_term"]
    b_k = bat["degradation_segments"]["b"]
    lam_k = bat["degradation_segments"]["lambda"]
    n_seg = len(lam_k)
    kappa = dem["kappa"]
    m1 = dem["over_contract_m1"]
    m2 = dem["over_contract_m2"]
    tier1 = dem["over_contract_tier1_ratio"]
    rho_peak0 = CCOR_PARAMS["rho_peak0_multiplier"] * c_basic * m2

    t_start = time.time()
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", 60)
        env.setParam("MIPGap", 0.005)
        env.start()
        mdl = gp.Model(env=env)

        u = mdl.addVars(H, vtype=GRB.BINARY, name="u")
        P_ch = mdl.addVars(H, lb=0.0, name="P_ch")
        P_dis = mdl.addVars(H, lb=0.0, name="P_dis")
        E = mdl.addVars(H, lb=soc_min * EB, ub=soc_max * EB, name="E")
        e_seg = mdl.addVars(H, n_seg, lb=0.0, name="e_seg")

        P_gl = mdl.addVars(n_sc, H, lb=0.0, name="P_gl")
        P_gc = mdl.addVars(n_sc, H, lb=0.0, name="P_gc")
        P_pvl = mdl.addVars(n_sc, H, lb=0.0, name="P_pvl")
        P_pvc = mdl.addVars(n_sc, H, lb=0.0, name="P_pvc")
        P_pvcu = mdl.addVars(n_sc, H, lb=0.0, name="P_pvcu")

        E_g = mdl.addVars(n_sc, H, lb=0.0, ub=EB, name="E_g")
        P_chg = mdl.addVars(n_sc, H, lb=0.0, name="P_chg")
        P_disg = mdl.addVars(n_sc, H, lb=0.0, name="P_disg")

        D_cand = mdl.addVars(n_sc, lb=D_init, name="D_cand")
        Over_full = mdl.addVars(n_sc, lb=0.0, name="Over_full")
        O1_full = mdl.addVars(n_sc, lb=0.0, ub=tier1 * CC, name="O1_full")
        O2_full = mdl.addVars(n_sc, lb=0.0, name="O2_full")
        eta = mdl.addVar(lb=-1e8, ub=1e8, name="eta")
        xi = mdl.addVars(n_sc, lb=0.0, name="xi")

        z_peak0 = mdl.addVars(n_sc, lb=0.0, name="z_peak0")
        slack_reserve = mdl.addVar(lb=0.0, name="slack_reserve")

        for t in T:
            mdl.addConstr(P_ch[t] <= u[t] * PB, name=f"ch_ub_{t}")
            mdl.addConstr(P_dis[t] <= (1 - u[t]) * PB, name=f"dis_ub_{t}")
            E_prev = E_init if t == 0 else E[t - 1]
            mdl.addConstr(E[t] == E_prev + eta_ch * P_ch[t] - (1.0 / eta_dis) * P_dis[t], name=f"soc_{t}")
            mdl.addConstr(P_dis[t] == gp.quicksum(e_seg[t, k] for k in range(n_seg)), name=f"seg_sum_{t}")
            for k in range(n_seg):
                mdl.addConstr(e_seg[t, k] <= (b_k[k + 1] - b_k[k]) * EB, name=f"seg_cap_{t}_{k}")

        if is_final_day:
            mdl.addConstr(E[H - 1] >= (soc_init - eps_term) * EB, name="term_lb")
            mdl.addConstr(E[H - 1] <= (soc_init + eps_term) * EB, name="term_ub")

        mdl.addConstr(E[H - 1] + slack_reserve >= rmeta["reserve_soc_target_kwh"], name="ccor_terminal_reserve")

        for w, sc in enumerate(scenarios):
            pv_w = sc["pv"]
            load_w = sc["load"]
            for t in T:
                mdl.addConstr(P_gl[w, t] + P_pvl[w, t] + P_dis[t] == load_w[t], name=f"Lbal_{w}_{t}")
                mdl.addConstr(P_ch[t] == P_gc[w, t] + P_pvc[w, t], name=f"Cbal_{w}_{t}")
                mdl.addConstr(P_pvl[w, t] + P_pvc[w, t] + P_pvcu[w, t] == pv_w[t], name=f"PVbal_{w}_{t}")
                mdl.addConstr(D_cand[w] >= kappa * (P_gl[w, t] + P_gc[w, t]), name=f"Dcand_{w}_{t}")
            mdl.addConstr(Over_full[w] >= D_cand[w] - CC, name=f"over_pos_{w}")
            mdl.addConstr(Over_full[w] == O1_full[w] + O2_full[w], name=f"over_split_{w}")
            mdl.addConstr(z_peak0[w] >= kappa * (P_gl[w, 0] + P_gc[w, 0]) - rmeta["D_guard_kw"], name=f"ccor_peak0_{w}")

            for t in T:
                Eg_prev = Eg_init if t == 0 else E_g[w, t - 1]
                mdl.addConstr(E_g[w, t] == Eg_prev + eta_ch * P_chg[w, t] - (1.0 / eta_dis) * P_disg[w, t], name=f"Eg_rec_{w}_{t}")
                mdl.addConstr(E_g[w, t] <= E[t], name=f"Eg_ub_{w}_{t}")
                mdl.addConstr(P_chg[w, t] <= P_pvc[w, t], name=f"Pcg_ub_{w}_{t}")
                mdl.addConstr(P_disg[w, t] <= P_dis[t], name=f"Pdg_ub_{w}_{t}")
            C_oc_w = c_basic * (m1 * O1_full[w] + m2 * O2_full[w])
            mdl.addConstr(xi[w] >= C_oc_w - eta, name=f"cvar_xi_{w}")

        C_ene = gp.quicksum(sc["pi"] * tou[t] * (P_gl[w, t] + P_gc[w, t]) for w, sc in enumerate(scenarios) for t in T)
        C_deg = gp.quicksum(lam_k[k] * e_seg[t, k] for t in T for k in range(n_seg))
        Over_prev = max(D_init - CC, 0.0)
        O1_prev = min(Over_prev, tier1 * CC)
        O2_prev = max(Over_prev - tier1 * CC, 0.0)
        PeakCost_prev = c_basic * (m1 * O1_prev + m2 * O2_prev)
        E_C_oc = gp.quicksum(sc["pi"] * c_basic * (m1 * O1_full[w] + m2 * O2_full[w]) for w, sc in enumerate(scenarios))
        C_peak_exp = E_C_oc - PeakCost_prev
        C_cvar = eta + (1.0 / (1.0 - CCOR_PARAMS["alpha"])) * gp.quicksum(sc["pi"] * xi[w] for w, sc in enumerate(scenarios))
        C_peak0 = gp.quicksum(sc["pi"] * rho_peak0 * z_peak0[w] for w, sc in enumerate(scenarios))
        C_reserve = CCOR_PARAMS["rho_reserve_ntd_per_kwh"] * slack_reserve
        mdl.setObjective(C_ene + C_deg + C_peak_exp + CCOR_PARAMS["cvar_lam"] * C_cvar + C_peak0 + C_reserve, GRB.MINIMIZE)
        mdl.optimize()

        solve_time = time.time() - t_start
        if mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mdl.SolCount == 0:
            return fallback_plan(E_init, H, n_sc, day_index, solve_time, rmeta)

        P_ch_plan = np.array([P_ch[t].X for t in T])
        P_dis_plan = np.array([P_dis[t].X for t in T])
        E_plan = np.array([E[t].X for t in T])
        D_cand_by_sc = [float(D_cand[w].X) for w in W]
        z_vals = [float(z_peak0[w].X) for w in W]
        C_oc_by_sc = [c_basic * (m1 * float(O1_full[w].X) + m2 * float(O2_full[w].X)) for w in W]
        weights = [float(sc["pi"]) for sc in scenarios]
        first_grid_by_sc = [float(P_gl[w, 0].X + P_gc[w, 0].X) for w in W]
        first_demand_by_sc = [kappa * g for g in first_grid_by_sc]
        E_g_end = float(sum(weights[w] * float(E_g[w, H - 1].X) for w in W))
        return {
            "u_plan": np.array([u[t].X for t in T]),
            "P_ch_plan": P_ch_plan,
            "P_dis_plan": P_dis_plan,
            "E_plan": E_plan,
            "E_end": float(E[H - 1].X),
            "E_g_end": E_g_end,
            "D_cand": float(sum(weights[w] * D_cand_by_sc[w] for w in W)),
            "D_cand_by_scenario": D_cand_by_sc,
            "C_oc_by_scenario": C_oc_by_sc,
            "obj_val": float(mdl.ObjVal),
            "solve_time": solve_time,
            "status": int(mdl.Status),
            "z_peak0_by_scenario": z_vals,
            "z_peak0_expected": float(sum(weights[w] * z_vals[w] for w in W)),
            "z_peak0_max": float(max(z_vals) if z_vals else 0.0),
            "first_step_grid_import_expected": float(sum(weights[w] * first_grid_by_sc[w] for w in W)),
            "first_step_demand_expected": float(sum(weights[w] * first_demand_by_sc[w] for w in W)),
            "first_step_demand_max": float(max(first_demand_by_sc) if first_demand_by_sc else 0.0),
            "slack_reserve": float(slack_reserve.X),
            "terminal_soc": float(E[H - 1].X),
            "reserve_binding": bool(float(slack_reserve.X) > 1e-6 or abs(float(E[H - 1].X) - rmeta["reserve_soc_target_kwh"]) < 1e-3),
            **rmeta,
        }


def fallback_plan(E_init, H, n_sc, day_index, solve_time, rmeta):
    print(f"WARNING CCOR day/step {day_index} failed; using do-nothing fallback")
    return {
        "u_plan": np.ones(H),
        "P_ch_plan": np.zeros(H),
        "P_dis_plan": np.zeros(H),
        "E_plan": np.full(H, E_init),
        "E_end": E_init,
        "E_g_end": 0.0,
        "D_cand": 0.0,
        "D_cand_by_scenario": [0.0] * n_sc,
        "C_oc_by_scenario": [0.0] * n_sc,
        "obj_val": 0.0,
        "solve_time": solve_time,
        "status": -1,
        "z_peak0_by_scenario": [0.0] * n_sc,
        "z_peak0_expected": 0.0,
        "z_peak0_max": 0.0,
        "first_step_grid_import_expected": 0.0,
        "first_step_demand_expected": 0.0,
        "first_step_demand_max": 0.0,
        "slack_reserve": 0.0,
        "terminal_soc": E_init,
        "reserve_binding": False,
        **rmeta,
    }


def replay(row, plan, state):
    bat = CFG["battery"]
    p_ch = max(float(plan["P_ch_plan"][0]), 0.0)
    p_dis = max(float(plan["P_dis_plan"][0]), 0.0)
    pv = float(row.pv_realized_kw)
    load = float(row.load_realized_kw)
    p_grid = max(load - pv - p_dis + p_ch, 0.0)
    e = float(np.clip(state["E_soc"] + bat["eta_ch"] * p_ch - (1.0 / bat["eta_dis"]) * p_dis, bat["soc_min"] * EB, bat["soc_max"] * EB))
    d = max(float(state["D_mth"]), CFG["demand"]["kappa"] * p_grid)
    ts = pd.Timestamp(row.timestamp)
    return {
        "p_ch_exec": p_ch,
        "p_dis_exec": p_dis,
        "pv_realized": pv,
        "load_realized": load,
        "p_grid_realized": p_grid,
        "tou_ene_hour": get_tou_price(ts.month, ts.day, ts.weekday(), ts.hour) * p_grid,
        "E_soc": e,
        "D_mth_running": d,
    }


def day_dict_point(start: pd.Timestamp, load_lookup: dict, pv_lookup: dict, row) -> tuple[dict, dict]:
    loads, pvs = [], []
    for j in range(H):
        ts = start + pd.Timedelta(hours=j)
        loads.append(float(load_lookup.get(ts, 0.0)))
        pvs.append(float(pv_lookup.get(ts, 0.0)))
    load = np.asarray(loads)
    pv = np.asarray(pvs)
    nl = load - pv
    return {
        "scenarios": [{"id": "k1", "pi": 1.0, "pv": np.maximum(pv, 0.0), "load": np.maximum(load, 0.0)}],
        "tou": tou_path(start),
        "month_id": int(row.month_id),
        "calendar_day": pd.Timestamp(row.calendar_day),
    }, {"scenario_count": 1, "scenario_weight_sum": 1.0, "forecast_netload_first": float(nl[0]), "max_stress_netload_H24": float(np.max(nl)), "forecast_peak_lead": int(np.argmax(nl) + 1)}


def day_dict_prob(start: pd.Timestamp, scen_rows: pd.DataFrame, row) -> tuple[dict, dict]:
    scenarios = []
    for sid, g in scen_rows.groupby("scenario_id", sort=True):
        gg = g.sort_values(["lead_hour", "target_time"])
        if len(gg) != H:
            raise ValueError(f"{start} scenario {sid} len {len(gg)}")
        scenarios.append({"id": f"s{int(sid)}", "pi": float(gg["scenario_weight"].iloc[0]), "pv": gg["pv_s_kw"].to_numpy(float), "load": gg["load_kw"].to_numpy(float)})
    wsum = sum(sc["pi"] for sc in scenarios)
    for sc in scenarios:
        sc["pi"] = float(sc["pi"] / wsum)
    nls = [sc["load"] - sc["pv"] for sc in scenarios]
    weights = [sc["pi"] for sc in scenarios]
    return {
        "scenarios": scenarios,
        "tou": tou_path(start),
        "month_id": int(row.month_id),
        "calendar_day": pd.Timestamp(row.calendar_day),
    }, {"scenario_count": len(scenarios), "scenario_weight_sum": float(sum(weights)), "forecast_netload_first": float(np.average([nl[0] for nl in nls], weights=weights)), "max_stress_netload_H24": float(np.max(np.vstack(nls))), "forecast_peak_lead": int(np.argmax(np.max(np.vstack(nls), axis=0)) + 1)}


def daily_summary(h):
    dem = CFG["demand"]
    rows = []
    for (di, cd, mo), g in h.groupby(["day_index", "calendar_day", "month_id"], sort=True):
        peak = float(g["D_mth_running"].max())
        c_b = get_basic_charge(int(mo), CFG)
        over = max(peak - CC, 0.0)
        o1 = min(over, dem["over_contract_tier1_ratio"] * CC)
        o2 = max(over - dem["over_contract_tier1_ratio"] * CC, 0.0)
        rows.append({
            "day_index": int(di),
            "calendar_day": str(pd.Timestamp(cd).date()),
            "month_id": int(mo),
            "D_mth_end": peak,
            "C_oc": c_b * (dem["over_contract_m1"] * o1 + dem["over_contract_m2"] * o2),
            "E_soc_end": float(g["E_soc"].iloc[-1]),
            "p_ch_sum": float(g["p_ch_exec"].sum()),
            "p_dis_sum": float(g["p_dis_exec"].sum()),
            "tou_ene_day": float(g["tou_ene_hour"].sum()),
            "n_milp_fail": int((~g["milp_status"].isin(GOOD)).sum()),
        })
    return pd.DataFrame(rows)


def run_ccor(case: str, prob: bool):
    hp = OUT / f"{case}_hourly.parquet"
    dp = OUT / f"{case}_daily.parquet"
    if hp.exists() and dp.exists():
        h = pd.read_parquet(hp)
        d = pd.read_parquet(dp)
        return h, d, corrected_kpis(case, h, d, DESIGN)

    truth = load_truth()
    load_lookup = load_profile_lookup("base")
    da_pvq = full_da_pv_quantiles()
    da_pv_lookup = {pd.Timestamp(r.target_time): float(r.pv_q50) for r in da_pvq.itertuples(index=False)}
    id_pvq = full_id_pv_quantiles()
    id_pv_lookup = {(pd.Timestamp(r.issue_time), pd.Timestamp(r.target_time)): float(r.pv_q50) for r in id_pvq.itertuples(index=False)}
    scen_groups = {}
    if prob:
        scen = pd.read_parquet(PROB_SCENARIO)
        scen["issue_time"] = pd.to_datetime(scen["issue_time"])
        scen["target_time"] = pd.to_datetime(scen["target_time"])
        scen_groups = {k: g.copy() for k, g in scen.groupby("issue_time")}

    state = state0()
    prev_month = None
    rows = []
    t0 = time.time()
    for i, rr in enumerate(truth.itertuples(index=False)):
        ts = pd.Timestamp(rr.timestamp)
        issue = ts - pd.Timedelta(hours=1)
        if prev_month is not None and int(rr.month_id) != prev_month:
            state["D_mth"] = 0.0
        prev_month = int(rr.month_id)

        if prob and issue in scen_groups:
            g = scen_groups[issue]
            if g["target_time"].nunique() < H:
                extras = []
                for sid, sg in g.groupby("scenario_id"):
                    for j in range(H):
                        target = ts + pd.Timedelta(hours=j)
                        if not sg["target_time"].eq(target).any():
                            load = load_lookup.get(target, 0.0)
                            pv = da_pv_lookup.get(target, 0.0)
                            extras.append({"issue_time": issue, "target_time": target, "lead_hour": j + 1, "scenario_id": sid, "scenario_weight": float(sg["scenario_weight"].iloc[0]), "load_kw": load, "pv_s_kw": pv, "netload_s_kw": load - pv})
                if extras:
                    g = pd.concat([g, pd.DataFrame(extras)], ignore_index=True)
            dd, meta = day_dict_prob(ts, g, rr)
        else:
            pv_path = {}
            for j in range(H):
                target = ts + pd.Timedelta(hours=j)
                pv_path[target] = id_pv_lookup.get((issue, target), da_pv_lookup.get(target, 0.0))
            dd, meta = day_dict_point(ts, load_lookup, pv_path, rr)

        d_mtd_before = float(state["D_mth"])
        plan = solve_ccor(int(rr.day_index), dd, state, is_final_day=(i == len(truth) - 1))
        rep = replay(rr, plan, state)
        rows.append({
            "timestamp": ts,
            "day_index": int(rr.day_index),
            "month_id": int(rr.month_id),
            "calendar_day": str(pd.Timestamp(rr.calendar_day).date()),
            "hour": int(rr.hour_local) - 1,
            "hour_local": int(rr.hour_local),
            "case_name": case,
            "solve_time": float(plan.get("solve_time", 0.0)),
            "milp_status": int(plan.get("status", -999)),
            "D_mtd_before": d_mtd_before,
            **meta,
            **rep,
            "z_peak0_expected": float(plan["z_peak0_expected"]),
            "z_peak0_max": float(plan["z_peak0_max"]),
            "first_step_grid_import_expected": float(plan["first_step_grid_import_expected"]),
            "first_step_demand_expected": float(plan["first_step_demand_expected"]),
            "first_step_demand_max": float(plan["first_step_demand_max"]),
            "D_guard_kw": float(plan["D_guard_kw"]),
            "grid_guard_kw": float(plan["grid_guard_kw"]),
            "R_expected_kwh": float(plan["R_expected_kwh"]),
            "R_q80_kwh": float(plan["R_q80_kwh"]),
            "R_target_kwh": float(plan["R_target_kwh"]),
            "reserve_soc_target_kwh": float(plan["reserve_soc_target_kwh"]),
            "slack_reserve": float(plan["slack_reserve"]),
            "terminal_soc": float(plan["terminal_soc"]),
            "reserve_binding": bool(plan["reserve_binding"]),
            "first_step_stress_netload_kw": float(plan["first_step_stress_netload_kw"]),
        })
        state = {"E_soc": rep["E_soc"], "E_g": float(plan.get("E_g_end", state["E_g"])), "D_mth": rep["D_mth_running"]}
        if (i + 1) % 1000 == 0:
            print(f"{case}: {i+1}/8760 hours {time.time()-t0:.0f}s")

    h = pd.DataFrame(rows)
    d = daily_summary(h)
    h.to_parquet(hp, index=False)
    d.to_parquet(dp, index=False)
    return h, d, corrected_kpis(case, h, d, DESIGN)


def summary_row(case, h, d, k):
    dem = CFG["demand"]
    return {
        "case": case,
        "full_total_m_ntd": k["full_total_NTD"] / 1e6,
        "capex_m_ntd": k["capex_NTD"] / 1e6,
        "basic_m_ntd": k["basic_NTD"] / 1e6,
        "TOU_m_ntd": k["tou_NTD"] / 1e6,
        "OC_DCT_m_ntd": k["oc_total_NTD"] / 1e6,
        "Deg_m_ntd": k["deg_NTD"] / 1e6,
        "TREC_RE20_m_ntd": k["trec_NTD"] / 1e6,
        "monthly_peak_max_kw": max(k["monthly_d"].values()),
        "oc_hours": int((h["p_grid_realized"] * dem["kappa"] > CC).sum()),
        "total_charge_kwh": float(h["p_ch_exec"].sum()),
        "total_discharge_kwh": float(h["p_dis_exec"].sum()),
        "soc_min_kwh": float(h["E_soc"].min()),
        "soc_max_kwh": float(h["E_soc"].max()),
        "z_peak0_positive_hours": int((h["z_peak0_max"] > 1e-6).sum()),
        "reserve_slack_positive_hours": int((h["slack_reserve"] > 1e-6).sum()),
        "first_step_charge_near_dct_risk_hours": int(((h["p_ch_exec"] > 1e-6) & (h["first_step_demand_max"] >= 0.95 * h["D_guard_kw"])).sum()),
        "infeasible_steps": int((~h["milp_status"].isin(GOOD)).sum()),
        "fallback_steps": int((h["milp_status"] == -1).sum()),
        "solver_status_summary": json.dumps({str(k): int(v) for k, v in h["milp_status"].value_counts().to_dict().items()}),
        "solve_time_total_sec": float(h["solve_time"].sum()),
    }


def collect(results: dict):
    rows, monthly = [], []
    for case, (h, d, k) in results.items():
        rows.append(summary_row(case, h, d, k))
        for mo in range(1, 13):
            hm = h[h["month_id"].eq(mo)]
            if hm.empty:
                continue
            monthly.append({
                "case": case,
                "month_id": mo,
                "TOU_m_ntd": hm["tou_ene_hour"].sum() / 1e6,
                "OC_DCT_m_ntd": k["monthly_oc"].get(mo, 0) / 1e6,
                "monthly_peak_kw": k["monthly_d"].get(mo, np.nan),
                "oc_hours": int((hm["p_grid_realized"] * CFG["demand"]["kappa"] > CC).sum()),
                "z_peak0_positive_hours": int((hm["z_peak0_max"] > 1e-6).sum()),
                "reserve_slack_positive_hours": int((hm["slack_reserve"] > 1e-6).sum()),
                "avg_R_target_kwh": float(hm["R_target_kwh"].mean()),
                "avg_terminal_soc_kwh": float(hm["terminal_soc"].mean()),
            })
    annual = pd.DataFrame(rows)
    mon = pd.DataFrame(monthly)
    annual.to_csv(OUT / "annual_cost_summary_realized_basis.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "cost_component_breakdown.csv", index=False, encoding="utf-8-sig")
    mon.to_csv(OUT / "monthly_cost_summary.csv", index=False, encoding="utf-8-sig")
    mon[["case", "month_id", "monthly_peak_kw"]].to_csv(OUT / "monthly_peak_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "oc_hours", "monthly_peak_max_kw", "OC_DCT_m_ntd"]].to_csv(OUT / "oc_event_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "infeasible_steps", "fallback_steps", "solver_status_summary", "solve_time_total_sec"]].to_csv(OUT / "solver_status_summary.csv", index=False, encoding="utf-8-sig")
    write_audits(results, annual, mon)


def scenario_usage_audit() -> pd.DataFrame:
    df = pd.read_parquet(PROB_SCENARIO)
    w = df.drop_duplicates(["issue_time", "scenario_id"])
    scen_count = w.groupby("issue_time")["scenario_id"].nunique()
    wsum = w.groupby("issue_time")["scenario_weight"].sum()
    eff = w.groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum())
    truth_cols = [c for c in df.columns if any(tok in c.lower() for tok in ["realized", "actual", "truth"])]
    out = pd.DataFrame([{
        "case": "CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5",
        "scenario_path": str(PROB_SCENARIO.relative_to(REPO)),
        "bridge_type": str(df["bridge_type"].iloc[0]),
        "k_min": int(scen_count.min()),
        "k_max": int(scen_count.max()),
        "weight_sum_min": float(wsum.min()),
        "weight_sum_max": float(wsum.max()),
        "effective_scenarios_min": float(eff.min()),
        "effective_scenarios_mean": float(eff.mean()),
        "effective_scenarios_max": float(eff.max()),
        "has_realized_or_truth_columns": bool(truth_cols),
        "truth_like_columns": ";".join(truth_cols),
    }])
    out.to_csv(OUT / "scenario_usage_audit.csv", index=False, encoding="utf-8-sig")
    return out


def replay_truth_audit() -> pd.DataFrame:
    truth = load_truth()
    out = pd.DataFrame([{
        "component": "CCOR replay/KPI truth",
        "truth_path": str(REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)),
        "pv_truth_source": PV_TRUTH_SOURCE_LABEL,
        "uses_old_invalid_ntust_solar_kwh": False,
        "row_count": len(truth),
        "pv_max_kw": float(truth["pv_realized_kw"].max()),
        "pv_mean_kw": float(truth["pv_realized_kw"].mean()),
        "source_marker_values": ";".join(sorted(truth["pv_truth_source"].dropna().astype(str).unique())) if "pv_truth_source" in truth.columns else "",
    }])
    out.to_csv(OUT / "replay_truth_usage_audit.csv", index=False, encoding="utf-8-sig")
    return out


def mechanism_audit(results: dict) -> pd.DataFrame:
    rows = []
    for case, (h, _, _) in results.items():
        for mo, g in h.groupby("month_id", observed=True):
            rows.append({
                "case": case,
                "month_id": int(mo),
                "z_peak0_positive_hours": int((g["z_peak0_max"] > 1e-6).sum()),
                "z_peak0_expected_sum_kw": float(g["z_peak0_expected"].sum()),
                "z_peak0_max_kw": float(g["z_peak0_max"].max()),
                "first_step_demand_max_kw": float(g["first_step_demand_max"].max()),
                "D_guard_max_kw": float(g["D_guard_kw"].max()),
                "reserve_slack_positive_hours": int((g["slack_reserve"] > 1e-6).sum()),
                "reserve_slack_sum_kwh": float(g["slack_reserve"].sum()),
                "R_target_mean_kwh": float(g["R_target_kwh"].mean()),
                "R_target_max_kwh": float(g["R_target_kwh"].max()),
                "terminal_soc_mean_kwh": float(g["terminal_soc"].mean()),
                "terminal_soc_min_kwh": float(g["terminal_soc"].min()),
                "charge_near_dct_risk_hours": int(((g["p_ch_exec"] > 1e-6) & (g["first_step_demand_max"] >= 0.95 * g["D_guard_kw"])).sum()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ccor_mechanism_audit.csv", index=False, encoding="utf-8-sig")
    return out


def first_step_peak_shield_audit(results: dict) -> pd.DataFrame:
    rows = []
    for case, (h, _, _) in results.items():
        cols = [
            "timestamp", "month_id", "D_mtd_before", "D_guard_kw", "p_ch_exec", "p_dis_exec",
            "p_grid_realized", "D_mth_running", "first_step_grid_import_expected",
            "first_step_demand_expected", "first_step_demand_max", "z_peak0_expected",
            "z_peak0_max", "R_target_kwh", "slack_reserve", "terminal_soc",
        ]
        sub = h[cols].copy()
        sub["case"] = case
        sub["first_step_charging_near_dct_risk"] = (sub["p_ch_exec"] > 1e-6) & (sub["first_step_demand_max"] >= 0.95 * sub["D_guard_kw"])
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "first_step_peak_shield_audit.csv", index=False, encoding="utf-8-sig")
    return out


def reserve_target_audit(results: dict) -> pd.DataFrame:
    rows = []
    for case, (h, _, _) in results.items():
        cols = [
            "timestamp", "month_id", "D_guard_kw", "grid_guard_kw", "R_expected_kwh", "R_q80_kwh",
            "R_target_kwh", "reserve_soc_target_kwh", "terminal_soc", "slack_reserve",
            "reserve_binding", "max_stress_netload_H24", "forecast_peak_lead",
        ]
        sub = h[cols].copy()
        sub["case"] = case
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "reserve_target_audit.csv", index=False, encoding="utf-8-sig")
    return out


def soc_peak_events(results: dict) -> pd.DataFrame:
    rows = []
    for case, (h, _, _) in results.items():
        h = h.copy()
        h["demand_kw"] = CFG["demand"]["kappa"] * h["p_grid_realized"]
        h["E_soc_before"] = h["E_soc"].shift(1).fillna(0.50 * EB)
        for mo, g in h.groupby("month_id", observed=True):
            idx = g["demand_kw"].idxmax()
            ts = pd.Timestamp(h.loc[idx, "timestamp"])
            w = h[(h["timestamp"] >= ts - pd.Timedelta(hours=6)) & (h["timestamp"] <= ts + pd.Timedelta(hours=6))]
            for r in w.itertuples(index=False):
                rows.append({
                    "case": case,
                    "event_month_id": int(mo),
                    "event_peak_timestamp": ts,
                    "timestamp": pd.Timestamp(r.timestamp),
                    "relative_hour": int((pd.Timestamp(r.timestamp) - ts) / pd.Timedelta(hours=1)),
                    "demand_kw": float(r.demand_kw),
                    "grid_import_kw": float(r.p_grid_realized),
                    "E_soc_before_kwh": float(r.E_soc_before),
                    "E_soc_after_kwh": float(r.E_soc),
                    "p_ch_kw": float(r.p_ch_exec),
                    "p_dis_kw": float(r.p_dis_exec),
                    "z_peak0_max_kw": float(r.z_peak0_max),
                    "R_target_kwh": float(r.R_target_kwh),
                    "slack_reserve_kwh": float(r.slack_reserve),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "soc_behavior_around_peak_events.csv", index=False, encoding="utf-8-sig")
    return out


def comparison_table(annual: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(STD_OUT / "annual_cost_summary_realized_basis.csv")
    all_rows = pd.concat([base, annual], ignore_index=True)
    idx = {r.case: r for r in all_rows.itertuples(index=False)}
    pairs = [
        ("MPC_DET_PVFOCUS_BASE", "CCOR_MPC_DET_PVFOCUS_BASE", "CCOR vs standard deterministic MPC"),
        ("MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR vs standard probabilistic MPC"),
        ("CCOR_MPC_DET_PVFOCUS_BASE", "CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "PV probabilistic value under CCOR"),
        ("DA_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR-PROB vs strongest basic DA-PROB"),
    ]
    rows = []
    for a, b, label in pairs:
        if a not in idx or b not in idx:
            continue
        ar = pd.Series(idx[a]._asdict())
        br = pd.Series(idx[b]._asdict())
        rows.append({
            "comparison": label,
            "case_a": a,
            "case_b": b,
            "delta_full_total_m_ntd_b_minus_a": float(br["full_total_m_ntd"] - ar["full_total_m_ntd"]),
            "delta_TOU_m_ntd_b_minus_a": float(br["TOU_m_ntd"] - ar["TOU_m_ntd"]),
            "delta_OC_DCT_m_ntd_b_minus_a": float(br["OC_DCT_m_ntd"] - ar["OC_DCT_m_ntd"]),
            "delta_Deg_m_ntd_b_minus_a": float(br["Deg_m_ntd"] - ar["Deg_m_ntd"]),
            "delta_monthly_peak_kw_b_minus_a": float(br["monthly_peak_max_kw"] - ar["monthly_peak_max_kw"]),
            "delta_oc_hours_b_minus_a": int(br["oc_hours"] - ar["oc_hours"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "comparison_vs_basic_four_cases.csv", index=False, encoding="utf-8-sig")
    return out


def final_status(annual: pd.DataFrame, comp: pd.DataFrame) -> str:
    if (annual["infeasible_steps"] > 0).any() or (annual["fallback_steps"] > 0).any():
        return "NEEDS_SOLVER_OR_MODEL_FIX"
    det = comp[comp["comparison"].eq("CCOR vs standard deterministic MPC")]
    prob = comp[comp["comparison"].eq("CCOR vs standard probabilistic MPC")]
    improved = False
    tradeoff = False
    for df in [det, prob]:
        if df.empty:
            continue
        r = df.iloc[0]
        if r["delta_full_total_m_ntd_b_minus_a"] < -1e-6 and r["delta_OC_DCT_m_ntd_b_minus_a"] < -1e-6:
            improved = True
        if r["delta_full_total_m_ntd_b_minus_a"] <= 0.20 and (r["delta_OC_DCT_m_ntd_b_minus_a"] < -0.10 or r["delta_monthly_peak_kw_b_minus_a"] < -50):
            tradeoff = True
    if improved:
        return "CCOR_MAIN_SUCCESS"
    if tradeoff:
        return "CCOR_RISK_COST_TRADEOFF"
    if (annual["TOU_m_ntd"].max() - annual["TOU_m_ntd"].min()) > 2.0:
        return "CCOR_OVER_CONSERVATIVE"
    return "CCOR_NO_IMPROVEMENT"


def write_report(annual, mon, mech, scen, truth, comp, status):
    write_md(
        OUT / "CCOR_MPC_PV_FOCUS_FINAL_REPORT.md",
        "CCOR-MPC PV-Focused Scheduling Report",
        [
            ("Final Status", f"`{status}`"),
            ("Method", "CCOR-MPC is a minimum-change rolling MPC extension: same PV-focused deterministic load/PV forecast or low-PV-safe K5 scenarios, with a first-step DCT soft shield and DCT-state-aware terminal SOC reserve. CVaR/risk-weighted OC is disabled in this first run."),
            ("Truth and Scenario Inputs", f"Replay/KPI uses `{REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)}` (`{PV_TRUTH_SOURCE_LABEL}`). PROB uses `{PROB_SCENARIO.relative_to(REPO)}`. LOAD-QN, random load error, old S1-S6, and old invalid PV truth are not used."),
            ("Annual Cost", annual.to_markdown(index=False)),
            ("Monthly Cost", mon.to_markdown(index=False)),
            ("Comparisons", comp.to_markdown(index=False)),
            ("CCOR Mechanism Audit", mech.to_markdown(index=False)),
            ("Scenario Usage", scen.to_markdown(index=False)),
            ("Replay Truth", truth.to_markdown(index=False)),
            ("Interpretation", "Success is judged against standard MPC. If total and OC/DCT both decrease, CCOR succeeds as the next proposed method. If OC/DCT decreases but TOU rises enough to offset it, report the result as a risk-cost tradeoff and tune reserve/peak penalties later."),
        ],
    )


def write_audits(results, annual, mon):
    scen = scenario_usage_audit()
    truth = replay_truth_audit()
    mech = mechanism_audit(results)
    first_step_peak_shield_audit(results)
    reserve_target_audit(results)
    soc_peak_events(results)
    comp = comparison_table(annual)
    status = final_status(annual, comp)
    write_report(annual, mon, mech, scen, truth, comp, status)


def main() -> int:
    ensure_out()
    results = {
        "CCOR_MPC_DET_PVFOCUS_BASE": run_ccor("CCOR_MPC_DET_PVFOCUS_BASE", prob=False),
        "CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5": run_ccor("CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", prob=True),
    }
    collect(results)
    print(f"[CCOR PVFocus] complete under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
