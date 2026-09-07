#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MILP_DIR = REPO / "milp_v2"
sys.path.insert(0, str(MILP_DIR))
sys.path.insert(0, str(MILP_DIR / "experiments/load_forecast_rerun"))
sys.path.insert(0, str(REPO / "new_pipeline/scripts/experiments"))

from common import get_basic_charge, get_tou_price, load_config  # noqa: E402
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
KAPPA = float(CFG["demand"]["kappa"])
OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_smoke"
STD_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
PROB_SCENARIO = BRIDGE_OUT / "id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet"
SMOKE_MONTHS = [(2024, 11), (2025, 5), (2025, 6)]


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    margin_kw: float = 0.0
    embargo_cap_kw: float | None = None
    use_charge_penalty: bool = False


VARIANTS = [
    Variant("CCOR_V2_SHIELD_ONLY_m50", "shield", margin_kw=50.0),
    Variant("CCOR_V2_SHIELD_ONLY_m100", "shield", margin_kw=100.0),
    Variant("CCOR_V2_SHIELD_ONLY_m150", "shield", margin_kw=150.0),
    Variant("CCOR_V2_CHARGE_EMBARGO_cap0", "embargo", embargo_cap_kw=0.0),
    Variant("CCOR_V2_CHARGE_PENALTY", "charge_penalty", use_charge_penalty=True),
    Variant("CCOR_V2_COMBINED_m100_embargo0", "combined", margin_kw=100.0, embargo_cap_kw=0.0),
]

PARAMS = {
    "rho_shield_multiplier": 20.0,
    "rho_charge_risk_ntd_per_kwh": 500.0,
    "risk_threshold_frac": 0.95,
    "risk0_prob_logic": "any_scenario_first_step_netload",
    "terminal_reserve_enabled": False,
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


def state0(e_soc: float | None = None) -> dict:
    bat = CFG["battery"]
    return {
        "E_soc": float(bat["soc_init"] * EB if e_soc is None else e_soc),
        "E_g": 0.0,
        "D_mth": 0.0,
    }


def oc_cost_for_peak(month_id: int, peak_kw: float) -> float:
    dem = CFG["demand"]
    over = max(float(peak_kw) - CC, 0.0)
    o1 = min(over, dem["over_contract_tier1_ratio"] * CC)
    o2 = max(over - dem["over_contract_tier1_ratio"] * CC, 0.0)
    return get_basic_charge(int(month_id), CFG) * (dem["over_contract_m1"] * o1 + dem["over_contract_m2"] * o2)


def deg_cost_for_discharge(p_dis: float) -> float:
    bat = CFG["battery"]
    b_k = bat["degradation_segments"]["b"]
    lam_k = bat["degradation_segments"]["lambda"]
    remaining = max(float(p_dis), 0.0)
    cost = 0.0
    for k, lam in enumerate(lam_k):
        width = max((b_k[k + 1] - b_k[k]) * EB, 0.0)
        take = min(remaining, width)
        cost += float(lam) * take
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        cost += float(lam_k[-1]) * remaining
    return float(cost)


def fallback_plan(E_init, n_sc, solve_time, risk_meta):
    return {
        "P_ch_plan": np.zeros(H),
        "P_dis_plan": np.zeros(H),
        "E_plan": np.full(H, E_init),
        "E_end": E_init,
        "E_g_end": 0.0,
        "status": -1,
        "solve_time": solve_time,
        "obj_val": 0.0,
        "z_shield_expected": 0.0,
        "z_shield_max": 0.0,
        "z_shield_by_scenario": [0.0] * n_sc,
        "first_step_grid_import_expected": 0.0,
        "first_step_demand_expected": 0.0,
        "first_step_demand_max": 0.0,
        "D_cand_by_scenario": [0.0] * n_sc,
        "terminal_reserve_enabled": False,
        **risk_meta,
    }


def risk0_meta(scenarios: list[dict], state: dict) -> dict:
    d_guard = max(CC, float(state["D_mth"]))
    grid_guard = d_guard / KAPPA
    threshold_grid = PARAMS["risk_threshold_frac"] * grid_guard
    nls0 = np.asarray([float(sc["load"][0] - sc["pv"][0]) for sc in scenarios], dtype=float)
    weights = np.asarray([float(sc["pi"]) for sc in scenarios], dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    risk_any = bool(np.max(nls0) >= threshold_grid)
    risk_expected = bool(np.sum(weights * nls0) >= threshold_grid)
    return {
        "D_guard_kw": float(d_guard),
        "D_cert_kw": float(d_guard),
        "grid_guard_kw": float(grid_guard),
        "risk_threshold_grid_kw": float(threshold_grid),
        "first_step_netload_expected_kw": float(np.sum(weights * nls0)),
        "first_step_netload_max_kw": float(np.max(nls0)),
        "risk0": risk_any,
        "risk0_expected_logic": risk_expected,
    }


def solve_v2(day_index: int, day_dict: dict, state: dict, variant: Variant) -> dict:
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
    rmeta = risk0_meta(scenarios, state)
    rmeta["D_cert_kw"] = float(max(CC, D_init) - max(variant.margin_kw, 0.0))

    eta_ch = bat["eta_ch"]
    eta_dis = bat["eta_dis"]
    soc_min = bat["soc_min"]
    soc_max = bat["soc_max"]
    b_k = bat["degradation_segments"]["b"]
    lam_k = bat["degradation_segments"]["lambda"]
    n_seg = len(lam_k)

    m1 = dem["over_contract_m1"]
    m2 = dem["over_contract_m2"]
    tier1 = dem["over_contract_tier1_ratio"]
    rho_shield = PARAMS["rho_shield_multiplier"] * c_basic * m2
    rho_charge = PARAMS["rho_charge_risk_ntd_per_kwh"]

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

        z_shield = mdl.addVars(n_sc, lb=0.0, name="z_shield")

        for t in T:
            mdl.addConstr(P_ch[t] <= u[t] * PB, name=f"ch_ub_{t}")
            mdl.addConstr(P_dis[t] <= (1 - u[t]) * PB, name=f"dis_ub_{t}")
            E_prev = E_init if t == 0 else E[t - 1]
            mdl.addConstr(E[t] == E_prev + eta_ch * P_ch[t] - (1.0 / eta_dis) * P_dis[t], name=f"soc_{t}")
            mdl.addConstr(P_dis[t] == gp.quicksum(e_seg[t, k] for k in range(n_seg)), name=f"seg_sum_{t}")
            for k in range(n_seg):
                mdl.addConstr(e_seg[t, k] <= (b_k[k + 1] - b_k[k]) * EB, name=f"seg_cap_{t}_{k}")

        if variant.embargo_cap_kw is not None and rmeta["risk0"]:
            mdl.addConstr(P_ch[0] <= float(variant.embargo_cap_kw), name="ccor_v2_first_step_charge_embargo")

        for w, sc in enumerate(scenarios):
            pv_w = sc["pv"]
            load_w = sc["load"]
            for t in T:
                mdl.addConstr(P_gl[w, t] + P_pvl[w, t] + P_dis[t] == load_w[t], name=f"Lbal_{w}_{t}")
                mdl.addConstr(P_ch[t] == P_gc[w, t] + P_pvc[w, t], name=f"Cbal_{w}_{t}")
                mdl.addConstr(P_pvl[w, t] + P_pvc[w, t] + P_pvcu[w, t] == pv_w[t], name=f"PVbal_{w}_{t}")
                mdl.addConstr(D_cand[w] >= KAPPA * (P_gl[w, t] + P_gc[w, t]), name=f"Dcand_{w}_{t}")

            mdl.addConstr(Over_full[w] >= D_cand[w] - CC, name=f"over_pos_{w}")
            mdl.addConstr(Over_full[w] == O1_full[w] + O2_full[w], name=f"over_split_{w}")
            if variant.mode in {"shield", "combined"}:
                mdl.addConstr(
                    z_shield[w] >= KAPPA * (P_gl[w, 0] + P_gc[w, 0]) - rmeta["D_cert_kw"],
                    name=f"ccor_v2_peak_shield0_{w}",
                )

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
        C_cvar = eta + (1.0 / (1.0 - 0.90)) * gp.quicksum(sc["pi"] * xi[w] for w, sc in enumerate(scenarios))
        C_shield = gp.quicksum(sc["pi"] * rho_shield * z_shield[w] for w, sc in enumerate(scenarios))
        C_charge_risk = rho_charge * P_ch[0] if variant.use_charge_penalty and rmeta["risk0"] else 0.0
        mdl.setObjective(C_ene + C_deg + C_peak_exp + 0.0 * C_cvar + C_shield + C_charge_risk, GRB.MINIMIZE)
        mdl.optimize()

        solve_time = time.time() - t_start
        if mdl.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mdl.SolCount == 0:
            return fallback_plan(E_init, n_sc, solve_time, rmeta)

        P_ch_plan = np.array([P_ch[t].X for t in T])
        P_dis_plan = np.array([P_dis[t].X for t in T])
        E_plan = np.array([E[t].X for t in T])
        weights = [float(sc["pi"]) for sc in scenarios]
        z_vals = [float(z_shield[w].X) for w in W]
        first_grid_by_sc = [float(P_gl[w, 0].X + P_gc[w, 0].X) for w in W]
        first_demand_by_sc = [KAPPA * g for g in first_grid_by_sc]
        D_cand_by_sc = [float(D_cand[w].X) for w in W]
        E_g_end = float(sum(weights[w] * float(E_g[w, H - 1].X) for w in W))
        return {
            "P_ch_plan": P_ch_plan,
            "P_dis_plan": P_dis_plan,
            "E_plan": E_plan,
            "E_end": float(E[H - 1].X),
            "E_g_end": E_g_end,
            "status": int(mdl.Status),
            "solve_time": solve_time,
            "obj_val": float(mdl.ObjVal),
            "z_shield_expected": float(sum(weights[w] * z_vals[w] for w in W)),
            "z_shield_max": float(max(z_vals) if z_vals else 0.0),
            "z_shield_by_scenario": z_vals,
            "first_step_grid_import_expected": float(sum(weights[w] * first_grid_by_sc[w] for w in W)),
            "first_step_demand_expected": float(sum(weights[w] * first_demand_by_sc[w] for w in W)),
            "first_step_demand_max": float(max(first_demand_by_sc) if first_demand_by_sc else 0.0),
            "D_cand_by_scenario": D_cand_by_sc,
            "terminal_reserve_enabled": False,
            **rmeta,
        }


def replay(row, plan: dict, state: dict) -> dict:
    bat = CFG["battery"]
    p_ch = max(float(plan["P_ch_plan"][0]), 0.0)
    p_dis = max(float(plan["P_dis_plan"][0]), 0.0)
    pv = float(row.pv_realized_kw)
    load = float(row.load_realized_kw)
    p_grid = max(load - pv - p_dis + p_ch, 0.0)
    e = float(np.clip(state["E_soc"] + bat["eta_ch"] * p_ch - (1.0 / bat["eta_dis"]) * p_dis, bat["soc_min"] * EB, bat["soc_max"] * EB))
    d = max(float(state["D_mth"]), KAPPA * p_grid)
    ts = pd.Timestamp(row.timestamp)
    return {
        "p_ch_exec": p_ch,
        "p_dis_exec": p_dis,
        "pv_realized": pv,
        "load_realized": load,
        "p_grid_realized": p_grid,
        "tou_ene_hour": get_tou_price(ts.month, ts.day, ts.weekday(), ts.hour) * p_grid,
        "deg_hour": deg_cost_for_discharge(p_dis),
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
    }, {
        "scenario_count": 1,
        "scenario_weight_sum": 1.0,
        "forecast_netload_first": float(nl[0]),
        "max_stress_netload_H24": float(np.max(nl)),
        "forecast_peak_lead": int(np.argmax(nl) + 1),
    }


def day_dict_prob(start: pd.Timestamp, scen_rows: pd.DataFrame, row) -> tuple[dict, dict]:
    scenarios = []
    for sid, g in scen_rows.groupby("scenario_id", sort=True):
        gg = g.sort_values(["lead_hour", "target_time"])
        if len(gg) != H:
            raise ValueError(f"{start} scenario {sid} len {len(gg)}")
        scenarios.append({
            "id": f"s{int(sid)}",
            "pi": float(gg["scenario_weight"].iloc[0]),
            "pv": gg["pv_s_kw"].to_numpy(float),
            "load": gg["load_kw"].to_numpy(float),
        })
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
    }, {
        "scenario_count": len(scenarios),
        "scenario_weight_sum": float(sum(weights)),
        "forecast_netload_first": float(np.average([nl[0] for nl in nls], weights=weights)),
        "max_stress_netload_H24": float(np.max(np.vstack(nls))),
        "forecast_peak_lead": int(np.argmax(np.max(np.vstack(nls), axis=0)) + 1),
    }


def load_standard_start_soc(case_name: str, month_start: pd.Timestamp) -> float:
    path = STD_OUT / f"{case_name}_hourly.parquet"
    if not path.exists():
        return float(CFG["battery"]["soc_init"] * EB)
    h = pd.read_parquet(path, columns=["timestamp", "E_soc"])
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    prev = h[h["timestamp"] < month_start].sort_values("timestamp").tail(1)
    if prev.empty:
        return float(CFG["battery"]["soc_init"] * EB)
    return float(prev["E_soc"].iloc[0])


def selected_truth() -> pd.DataFrame:
    truth = load_truth().copy()
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    mask = pd.Series(False, index=truth.index)
    for year, month in SMOKE_MONTHS:
        mask |= (truth["timestamp"].dt.year == year) & (truth["timestamp"].dt.month == month)
    return truth[mask].sort_values("timestamp").reset_index(drop=True)


def prepare_scenario_groups() -> dict:
    scen = pd.read_parquet(PROB_SCENARIO)
    scen["issue_time"] = pd.to_datetime(scen["issue_time"])
    scen["target_time"] = pd.to_datetime(scen["target_time"])
    return {k: g.copy() for k, g in scen.groupby("issue_time")}


def run_variant(case_root: str, variant: Variant, prob: bool) -> pd.DataFrame:
    truth = selected_truth()
    load_lookup = load_profile_lookup("base")
    da_pvq = full_da_pv_quantiles()
    da_pv_lookup = {pd.Timestamp(r.target_time): float(r.pv_q50) for r in da_pvq.itertuples(index=False)}
    id_pvq = full_id_pv_quantiles()
    id_pv_lookup = {(pd.Timestamp(r.issue_time), pd.Timestamp(r.target_time)): float(r.pv_q50) for r in id_pvq.itertuples(index=False)}
    scen_groups = prepare_scenario_groups() if prob else {}

    std_case = "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5" if prob else "MPC_DET_PVFOCUS_BASE"
    rows = []
    t0 = time.time()
    for (year, month), mt in truth.groupby([truth["timestamp"].dt.year, truth["timestamp"].dt.month], sort=True):
        month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
        state = state0(load_standard_start_soc(std_case, month_start))
        for i, rr in enumerate(mt.itertuples(index=False)):
            ts = pd.Timestamp(rr.timestamp)
            issue = ts - pd.Timedelta(hours=1)
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
                                extras.append({
                                    "issue_time": issue,
                                    "target_time": target,
                                    "lead_hour": j + 1,
                                    "scenario_id": sid,
                                    "scenario_weight": float(sg["scenario_weight"].iloc[0]),
                                    "load_kw": load,
                                    "pv_s_kw": pv,
                                    "netload_s_kw": load - pv,
                                })
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
            e_soc_before = float(state["E_soc"])
            plan = solve_v2(int(rr.day_index), dd, state, variant)
            rep = replay(rr, plan, state)
            rows.append({
                "timestamp": ts,
                "day_index": int(rr.day_index),
                "month_id": int(rr.month_id),
                "calendar_day": str(pd.Timestamp(rr.calendar_day).date()),
                "hour": int(rr.hour_local) - 1,
                "hour_local": int(rr.hour_local),
                "case_root": case_root,
                "variant": variant.name,
                "case_name": f"{case_root}_{variant.name}",
                "probabilistic": bool(prob),
                "margin_kw": float(variant.margin_kw),
                "embargo_cap_kw": np.nan if variant.embargo_cap_kw is None else float(variant.embargo_cap_kw),
                "charge_penalty_enabled": bool(variant.use_charge_penalty),
                "solve_time": float(plan.get("solve_time", 0.0)),
                "milp_status": int(plan.get("status", -999)),
                "D_mtd_before": d_mtd_before,
                "E_soc_before": e_soc_before,
                **meta,
                **rep,
                "D_guard_kw": float(plan["D_guard_kw"]),
                "D_cert_kw": float(plan["D_cert_kw"]),
                "grid_guard_kw": float(plan["grid_guard_kw"]),
                "risk_threshold_grid_kw": float(plan["risk_threshold_grid_kw"]),
                "risk0": bool(plan["risk0"]),
                "risk0_expected_logic": bool(plan["risk0_expected_logic"]),
                "first_step_netload_expected_kw": float(plan["first_step_netload_expected_kw"]),
                "first_step_netload_max_kw": float(plan["first_step_netload_max_kw"]),
                "first_step_grid_import_expected": float(plan["first_step_grid_import_expected"]),
                "first_step_demand_expected": float(plan["first_step_demand_expected"]),
                "first_step_demand_max": float(plan["first_step_demand_max"]),
                "z_shield_expected": float(plan["z_shield_expected"]),
                "z_shield_max": float(plan["z_shield_max"]),
            })
            state = {"E_soc": rep["E_soc"], "E_g": float(plan.get("E_g_end", state["E_g"])), "D_mth": rep["D_mth_running"]}
        print(f"{case_root} {variant.name} {year}-{month:02d}: {len(mt)} h, elapsed {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


def cost_summary_for_hourly(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    mon_rows = []
    for case, gcase in h.groupby("case_name", sort=True):
        total_tou = float(gcase["tou_ene_hour"].sum())
        total_deg = float(gcase["deg_hour"].sum())
        total_basic = 0.0
        total_oc = 0.0
        max_peak = 0.0
        oc_hours = 0
        for mo, gm in gcase.groupby("month_id", sort=True):
            peak = float(gm["D_mth_running"].max())
            oc = oc_cost_for_peak(int(mo), peak)
            basic = get_basic_charge(int(mo), CFG) * CC
            tou = float(gm["tou_ene_hour"].sum())
            deg = float(gm["deg_hour"].sum())
            total_oc += oc
            total_basic += basic
            max_peak = max(max_peak, peak)
            oc_h = int((KAPPA * gm["p_grid_realized"] > CC).sum())
            oc_hours += oc_h
            mon_rows.append({
                "case": case,
                "variant": str(gm["variant"].iloc[0]),
                "case_root": str(gm["case_root"].iloc[0]),
                "month_id": int(mo),
                "basic_m_ntd": basic / 1e6,
                "TOU_m_ntd": tou / 1e6,
                "OC_DCT_m_ntd": oc / 1e6,
                "Deg_m_ntd": deg / 1e6,
                "smoke_total_m_ntd": (basic + tou + oc + deg) / 1e6,
                "monthly_peak_kw": peak,
                "oc_hours": oc_h,
                "risk0_hours": int(gm["risk0"].sum()),
                "z_shield_positive_hours": int((gm["z_shield_max"] > 1e-6).sum()),
                "first_step_charging_near_risk_hours": int(((gm["risk0"]) & (gm["p_ch_exec"] > 1e-6)).sum()),
            })
        rows.append({
            "case": case,
            "variant": str(gcase["variant"].iloc[0]),
            "case_root": str(gcase["case_root"].iloc[0]),
            "months": ";".join(str(int(x)) for x in sorted(gcase["month_id"].unique())),
            "basic_m_ntd": total_basic / 1e6,
            "TOU_m_ntd": total_tou / 1e6,
            "OC_DCT_m_ntd": total_oc / 1e6,
            "Deg_m_ntd": total_deg / 1e6,
            "smoke_total_m_ntd": (total_basic + total_tou + total_oc + total_deg) / 1e6,
            "monthly_peak_max_kw": max_peak,
            "oc_hours": oc_hours,
            "risk0_hours": int(gcase["risk0"].sum()),
            "z_shield_positive_hours": int((gcase["z_shield_max"] > 1e-6).sum()),
            "p_ch_during_risk0_mean_kw": float(gcase.loc[gcase["risk0"], "p_ch_exec"].mean()) if bool(gcase["risk0"].any()) else 0.0,
            "first_step_charging_near_risk_hours": int(((gcase["risk0"]) & (gcase["p_ch_exec"] > 1e-6)).sum()),
            "soc_min_kwh": float(gcase["E_soc"].min()),
            "soc_max_kwh": float(gcase["E_soc"].max()),
            "infeasible_steps": int((~gcase["milp_status"].isin(GOOD)).sum()),
            "fallback_steps": int((gcase["milp_status"] == -1).sum()),
            "solver_status_summary": json.dumps({str(k): int(v) for k, v in gcase["milp_status"].value_counts().to_dict().items()}),
            "solve_time_total_sec": float(gcase["solve_time"].sum()),
        })
    return pd.DataFrame(rows), pd.DataFrame(mon_rows)


def baseline_subset_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cases = [
        "MPC_DET_PVFOCUS_BASE",
        "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5",
        "DA_PROB_PVFOCUS_LOWPV_SAFE_K5",
    ]
    rows = []
    hourly_parts = []
    for case in base_cases:
        hp = STD_OUT / f"{case}_hourly.parquet"
        if not hp.exists():
            continue
        h = pd.read_parquet(hp)
        h["timestamp"] = pd.to_datetime(h["timestamp"])
        mask = pd.Series(False, index=h.index)
        for year, month in SMOKE_MONTHS:
            mask |= (h["timestamp"].dt.year == year) & (h["timestamp"].dt.month == month)
        sub = h[mask].copy()
        sub["case_name"] = case
        sub["case_root"] = case
        sub["variant"] = "BASELINE"
        if "deg_hour" not in sub.columns:
            sub["deg_hour"] = sub["p_dis_exec"].apply(deg_cost_for_discharge)
        sub["risk0"] = KAPPA * sub["p_grid_realized"] >= 0.95 * np.maximum(CC, sub.get("D_mtd_before", sub["D_mth_running"]))
        sub["z_shield_max"] = 0.0
        sub["milp_status"] = sub.get("milp_status", 2)
        sub["solve_time"] = sub.get("solve_time", 0.0)
        sub["D_guard_kw"] = np.maximum(CC, sub.get("D_mtd_before", sub["D_mth_running"]))
        hourly_parts.append(sub)
    if not hourly_parts:
        return pd.DataFrame(), pd.DataFrame()
    bh = pd.concat(hourly_parts, ignore_index=True)
    return cost_summary_for_hourly(bh)


def write_diagnostics(all_h: pd.DataFrame, summary: pd.DataFrame, monthly: pd.DataFrame) -> None:
    all_h.to_parquet(OUT / "CCOR_V2_ALL_HOURLY.parquet", index=False)
    summary.to_csv(OUT / "ccor_v2_variant_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "monthly_cost_summary.csv", index=False, encoding="utf-8-sig")

    risk_cols = ["case", "variant", "case_root", "risk0_hours", "p_ch_during_risk0_mean_kw", "first_step_charging_near_risk_hours"]
    summary[risk_cols].to_csv(OUT / "first_step_charging_near_risk_summary.csv", index=False, encoding="utf-8-sig")
    summary[["case", "variant", "case_root", "z_shield_positive_hours", "infeasible_steps", "fallback_steps"]].to_csv(
        OUT / "shield_slack_summary.csv", index=False, encoding="utf-8-sig"
    )
    risk = all_h.groupby(["case_name", "variant", "case_root"], as_index=False).agg(
        risk0_hours=("risk0", "sum"),
        mean_p_ch_risk0_kw=("p_ch_exec", lambda x: float(x[all_h.loc[x.index, "risk0"]].mean()) if bool(all_h.loc[x.index, "risk0"].any()) else 0.0),
        max_p_ch_risk0_kw=("p_ch_exec", lambda x: float(x[all_h.loc[x.index, "risk0"]].max()) if bool(all_h.loc[x.index, "risk0"].any()) else 0.0),
    )
    risk.to_csv(OUT / "risk0_event_summary.csv", index=False, encoding="utf-8-sig")
    summary[["case", "variant", "case_root", "infeasible_steps", "fallback_steps", "solver_status_summary", "solve_time_total_sec"]].to_csv(
        OUT / "solver_status_summary.csv", index=False, encoding="utf-8-sig"
    )

    peak_rows = []
    soc_rows = []
    h = all_h.copy()
    h["demand_kw"] = KAPPA * h["p_grid_realized"]
    for case, gc in h.groupby("case_name", sort=True):
        for mo, gm in gc.groupby("month_id", sort=True):
            idx = gm["demand_kw"].idxmax()
            r = h.loc[idx]
            peak_rows.append({
                "case": case,
                "variant": r["variant"],
                "case_root": r["case_root"],
                "month_id": int(mo),
                "peak_timestamp": r["timestamp"],
                "peak_demand_kw": float(r["demand_kw"]),
                "grid_import_kw": float(r["p_grid_realized"]),
                "soc_before_kwh": float(r.get("E_soc_before", np.nan)),
                "soc_after_kwh": float(r["E_soc"]),
                "p_ch_kw": float(r["p_ch_exec"]),
                "p_dis_kw": float(r["p_dis_exec"]),
                "risk0": bool(r.get("risk0", False)),
                "z_shield_max_kw": float(r.get("z_shield_max", 0.0)),
            })
            ts = pd.Timestamp(r["timestamp"])
            win = gc[(gc["timestamp"] >= ts - pd.Timedelta(hours=6)) & (gc["timestamp"] <= ts + pd.Timedelta(hours=6))]
            for wr in win.itertuples(index=False):
                soc_rows.append({
                    "case": case,
                    "variant": wr.variant,
                    "case_root": wr.case_root,
                    "event_month_id": int(mo),
                    "event_peak_timestamp": ts,
                    "timestamp": pd.Timestamp(wr.timestamp),
                    "relative_hour": int((pd.Timestamp(wr.timestamp) - ts) / pd.Timedelta(hours=1)),
                    "demand_kw": float(wr.demand_kw),
                    "grid_import_kw": float(wr.p_grid_realized),
                    "soc_before_kwh": float(getattr(wr, "E_soc_before", np.nan)),
                    "soc_after_kwh": float(wr.E_soc),
                    "p_ch_kw": float(wr.p_ch_exec),
                    "p_dis_kw": float(wr.p_dis_exec),
                    "risk0": bool(getattr(wr, "risk0", False)),
                    "z_shield_max_kw": float(getattr(wr, "z_shield_max", 0.0)),
                })
    pd.DataFrame(peak_rows).to_csv(OUT / "top_peak_event_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(soc_rows).to_csv(OUT / "soc_behavior_around_peak_events.csv", index=False, encoding="utf-8-sig")


def choose_recommendation(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty or (summary["infeasible_steps"] > 0).any() or (summary["fallback_steps"] > 0).any():
        return "CCOR_V2_NEEDS_PARAMETER_TUNING", "Some variants had infeasible/fallback steps or no valid summary."
    base = summary[summary["variant"].eq("BASELINE")]
    cand = summary[~summary["variant"].eq("BASELINE")]
    prob_base = base[base["case_root"].eq("MPC_PROB_PVFOCUS_LOWPV_SAFE_K5")]
    prob_cand = cand[cand["case_root"].eq("CCOR_V2_PROB_PVFOCUS_LOWPV_SAFE_K5")]
    if prob_base.empty or prob_cand.empty:
        return "CCOR_V2_NEEDS_PARAMETER_TUNING", "Baseline or probabilistic candidate summary is missing."
    b = prob_base.iloc[0]
    c = prob_cand.copy()
    c["delta_total"] = c["smoke_total_m_ntd"] - float(b["smoke_total_m_ntd"])
    c["delta_oc"] = c["OC_DCT_m_ntd"] - float(b["OC_DCT_m_ntd"])
    c["delta_peak"] = c["monthly_peak_max_kw"] - float(b["monthly_peak_max_kw"])
    feasible = c[(c["delta_oc"] < -1e-6) | (c["delta_total"] <= 0.0) | (c["delta_peak"] < -1e-6)].copy()
    if feasible.empty:
        return "CCOR_V2_NOT_PROMISING", "No probabilistic v2 variant reduced OC, total smoke cost, or peak versus standard MPC-PROB."
    feasible["score"] = feasible["delta_total"] + 0.5 * feasible["delta_oc"] + 0.0001 * feasible["delta_peak"]
    best = feasible.sort_values("score").iloc[0]
    if best["delta_total"] <= 0.0 and best["delta_oc"] <= 0.0:
        return "BEST_CCOR_V2_VARIANT_FOR_FULL_YEAR", f"Best candidate `{best['variant']}` reduced selected-month smoke total and OC versus MPC-PROB."
    return "CCOR_V2_NEEDS_PARAMETER_TUNING", f"Best candidate `{best['variant']}` improved at least one risk metric but not both total and OC."


def write_report(summary: pd.DataFrame, monthly: pd.DataFrame, recommendation: str, reason: str) -> None:
    prob = summary[summary["case_root"].str.contains("PROB", na=False)].sort_values("smoke_total_m_ntd")
    det = summary[summary["case_root"].str.contains("DET", na=False)].sort_values("smoke_total_m_ntd")
    write_md(
        OUT / "CCOR_V2_SMOKE_TEST_REPORT.md",
        "CCOR-MPC v2 First-Step Peak-Certified Smoke Test",
        [
            ("Final Recommendation", f"`{recommendation}`\n\n{reason}"),
            ("Scope", "This is a selected-month smoke test only for 2024-11, 2025-05, and 2025-06. Full-year scheduling was not run."),
            ("Truth and Inputs", f"Replay uses `{REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)}` (`{PV_TRUTH_SOURCE_LABEL}`). PROB uses `{PROB_SCENARIO.relative_to(REPO)}`. LOAD-QN, random load error, old S1-S6, and old invalid PV truth are not used."),
            ("CCOR-v2 Variants", "Terminal reserve is disabled in all variants. Tested variants: shield-only margins 50/100/150 kW, charge embargo, charge risk penalty, and combined margin-100 + embargo."),
            ("Parameter Settings", json.dumps(PARAMS, indent=2)),
            ("Probabilistic Ranking", prob.to_markdown(index=False)),
            ("Deterministic Ranking", det.to_markdown(index=False)),
            ("Monthly Summary", monthly.to_markdown(index=False)),
            ("Interpretation", "The v2 mechanism is judged against standard MPC for the same selected months. A useful variant should reduce first-step charging during DCT-risk hours and lower OC/DCT or monthly peaks without a large TOU penalty."),
        ],
    )


def main() -> int:
    ensure_out()
    hourly_parts = []
    for prob, root in [
        (False, "CCOR_V2_DET_PVFOCUS_BASE"),
        (True, "CCOR_V2_PROB_PVFOCUS_LOWPV_SAFE_K5"),
    ]:
        for variant in VARIANTS:
            hp = OUT / f"{root}_{variant.name}_hourly.parquet"
            if hp.exists():
                h = pd.read_parquet(hp)
                h["timestamp"] = pd.to_datetime(h["timestamp"])
                print(f"cached {root} {variant.name}: {len(h)} rows")
            else:
                h = run_variant(root, variant, prob)
                h.to_parquet(hp, index=False)
            hourly_parts.append(h)
    v2_hourly = pd.concat(hourly_parts, ignore_index=True)
    v2_summary, v2_monthly = cost_summary_for_hourly(v2_hourly)
    base_summary, base_monthly = baseline_subset_summary()
    summary = pd.concat([base_summary, v2_summary], ignore_index=True)
    monthly = pd.concat([base_monthly, v2_monthly], ignore_index=True)
    all_h = v2_hourly
    write_diagnostics(all_h, summary, monthly)
    recommendation, reason = choose_recommendation(summary)
    write_report(summary, monthly, recommendation, reason)
    print(f"[CCOR v2 smoke] {recommendation} under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
