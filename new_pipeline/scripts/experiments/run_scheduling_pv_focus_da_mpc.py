#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MILP_DIR = REPO / "milp_v2"
sys.path.insert(0, str(MILP_DIR))
sys.path.insert(0, str(MILP_DIR / "experiments/mpc_fixed_rolling_horizon"))
sys.path.insert(0, str(MILP_DIR / "experiments/load_forecast_rerun"))
sys.path.insert(0, str(REPO / "new_pipeline/scripts/experiments"))

from common import get_basic_charge, get_tou_price, load_config  # noqa: E402
from layer_b.milp_cvar import solve_day_ahead  # noqa: E402
from mpc_fixed_horizon_utils import corrected_kpis  # noqa: E402
from pv_focus_bridge_utils import OUT_DIR as BRIDGE_OUT, PV_TRUTH_SOURCE_LABEL, REPLAY_TRUTH_PACKAGE_PATH, SCHED_OUT, ensure_sched_out, full_da_pv_quantiles, full_id_pv_quantiles, load_profile_lookup, load_truth, write_md  # noqa: E402
from run_da_formulation_rolling_controlled_experiments import load_design  # noqa: E402

CFG = load_config()
DESIGN = load_design()
CC = float(DESIGN["CC"])
H = 24
GOOD = {2, 9, 13}
DA_PROB_SCENARIO = BRIDGE_OUT / "da_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet"
MPC_PROB_SCENARIO = BRIDGE_OUT / "id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet"
VALIDATION_REPORT = BRIDGE_OUT / "PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md"


def tou_path(start: pd.Timestamp) -> np.ndarray:
    return np.asarray([get_tou_price((start + pd.Timedelta(hours=j)).month, (start + pd.Timedelta(hours=j)).day, (start + pd.Timedelta(hours=j)).weekday(), (start + pd.Timedelta(hours=j)).hour) for j in range(H)], dtype=float)


def state0() -> dict:
    bat = CFG["battery"]
    return {"E_soc": bat["soc_init"] * DESIGN["EB"], "E_g": 0.0, "D_mth": 0.0}


def replay(row, plan, state, idx=0):
    bat = CFG["battery"]; dem = CFG["demand"]; EB = DESIGN["EB"]
    p_ch = max(float(plan["P_ch_plan"][idx]), 0.0)
    p_dis = max(float(plan["P_dis_plan"][idx]), 0.0)
    pv = float(row.pv_realized_kw); load = float(row.load_realized_kw)
    p_grid = max(load - pv - p_dis + p_ch, 0.0)
    e = float(np.clip(state["E_soc"] + bat["eta_ch"] * p_ch - (1.0 / bat["eta_dis"]) * p_dis, bat["soc_min"] * EB, bat["soc_max"] * EB))
    d = max(float(state["D_mth"]), dem["kappa"] * p_grid)
    ts = pd.Timestamp(row.timestamp)
    return {"p_ch_exec": p_ch, "p_dis_exec": p_dis, "pv_realized": pv, "load_realized": load, "p_grid_realized": p_grid, "tou_ene_hour": get_tou_price(ts.month, ts.day, ts.weekday(), ts.hour) * p_grid, "E_soc": e, "D_mth_running": d}


def day_dict_point(start: pd.Timestamp, load_lookup: dict, pv_lookup: dict, row) -> tuple[dict, dict, int]:
    loads, pvs, fallback = [], [], 0
    for j in range(H):
        ts = start + pd.Timedelta(hours=j)
        loads.append(float(load_lookup.get(ts, 0.0)))
        pv = pv_lookup.get(ts)
        if pv is None:
            pv = 0.0; fallback += 1
        pvs.append(float(pv))
    load = np.asarray(loads); pv = np.asarray(pvs); nl = load - pv
    return {
        "scenarios": [{"id": "k1", "pi": 1.0, "pv": np.maximum(pv, 0.0), "load": np.maximum(load, 0.0)}],
        "tou": tou_path(start),
        "month_id": int(row.month_id),
        "calendar_day": pd.Timestamp(row.calendar_day),
    }, {"scenario_count": 1, "scenario_weight_sum": 1.0, "forecast_load_first": float(load[0]), "forecast_pv_first": float(pv[0]), "forecast_netload_first": float(nl[0]), "max_stress_netload_H24": float(np.max(nl)), "forecast_peak_lead": int(np.argmax(nl)+1)}, fallback


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
    }, {"scenario_count": len(scenarios), "scenario_weight_sum": float(sum(weights)), "forecast_load_first": float(np.average([sc["load"][0] for sc in scenarios], weights=weights)), "forecast_pv_first": float(np.average([sc["pv"][0] for sc in scenarios], weights=weights)), "forecast_netload_first": float(np.average([nl[0] for nl in nls], weights=weights)), "max_stress_netload_H24": float(np.max(np.vstack(nls))), "forecast_peak_lead": int(np.argmax(np.max(np.vstack(nls), axis=0))+1)}


def solve(dd, state, row, final=False):
    return solve_day_ahead(int(row.day_index), dd, state, DESIGN, CFG, is_final_day=final, time_limit=60, mip_gap=0.005, lam=0.0, alpha=0.90, verbose=False)


def daily_summary(h):
    dem = CFG["demand"]; rows = []
    for (di, cd, mo), g in h.groupby(["day_index", "calendar_day", "month_id"], sort=True):
        peak = float(g["D_mth_running"].max())
        c_b = get_basic_charge(int(mo), CFG)
        over = max(peak - CC, 0.0)
        o1 = min(over, dem["over_contract_tier1_ratio"] * CC); o2 = max(over - dem["over_contract_tier1_ratio"] * CC, 0.0)
        rows.append({"day_index": int(di), "calendar_day": str(pd.Timestamp(cd).date()), "month_id": int(mo), "D_mth_end": peak, "C_oc": c_b * (dem["over_contract_m1"] * o1 + dem["over_contract_m2"] * o2), "E_soc_end": float(g["E_soc"].iloc[-1]), "p_ch_sum": float(g["p_ch_exec"].sum()), "p_dis_sum": float(g["p_dis_exec"].sum()), "tou_ene_day": float(g["tou_ene_hour"].sum()), "n_milp_fail": int((~g["milp_status"].isin(GOOD)).sum())})
    return pd.DataFrame(rows)


def summary_row(case, h, d, k):
    dem = CFG["demand"]
    return {"case": case, "full_total_m_ntd": k["full_total_NTD"]/1e6, "capex_m_ntd": k["capex_NTD"]/1e6, "basic_m_ntd": k["basic_NTD"]/1e6, "TOU_m_ntd": k["tou_NTD"]/1e6, "OC_DCT_m_ntd": k["oc_total_NTD"]/1e6, "Deg_m_ntd": k["deg_NTD"]/1e6, "TREC_RE20_m_ntd": k["trec_NTD"]/1e6, "monthly_peak_max_kw": max(k["monthly_d"].values()), "oc_hours": int((h["p_grid_realized"] * dem["kappa"] > CC).sum()), "total_charge_kwh": float(h["p_ch_exec"].sum()), "total_discharge_kwh": float(h["p_dis_exec"].sum()), "soc_min_kwh": float(h["E_soc"].min()), "soc_max_kwh": float(h["E_soc"].max()), "infeasible_steps": int((~h["milp_status"].isin(GOOD)).sum()), "fallback_steps": 0, "solver_status_summary": json.dumps({str(k): int(v) for k, v in h["milp_status"].value_counts().to_dict().items()}), "solve_time_total_sec": float(h["solve_time"].sum())}


def run_da(case: str, prob: bool):
    ensure_sched_out()
    hp = SCHED_OUT / f"{case}_hourly.parquet"; dp = SCHED_OUT / f"{case}_daily.parquet"
    if hp.exists() and dp.exists():
        h = pd.read_parquet(hp); d = pd.read_parquet(dp); return h, d, corrected_kpis(case, h, d, DESIGN)
    truth = load_truth(); load_lookup = load_profile_lookup("base")
    pvq = full_da_pv_quantiles(); pv_lookup = {pd.Timestamp(r.target_time): float(r.pv_q50) for r in pvq.itertuples(index=False)}
    scen = pd.read_parquet(DA_PROB_SCENARIO) if prob else None
    if scen is not None:
        scen["target_time"] = pd.to_datetime(scen["target_time"]); scen["target_date"] = scen["target_time"].dt.date
        scen_by_day = {k: g.copy() for k, g in scen.groupby("target_date")}
    state = state0(); prev_month = None; rows=[]; t0=time.time()
    days = sorted(pd.Timestamp(x).date() for x in truth["calendar_day"].unique())
    for i, day in enumerate(days):
        gt = truth[pd.to_datetime(truth["calendar_day"]).dt.date.eq(day)].sort_values("timestamp")
        row0 = gt.iloc[0]
        if prev_month is not None and int(row0.month_id) != prev_month: state["D_mth"] = 0.0
        prev_month = int(row0.month_id)
        start = pd.Timestamp(day)
        if prob:
            dd, meta = day_dict_prob(start, scen_by_day[day], row0)
        else:
            dd, meta, _ = day_dict_point(start, load_lookup, pv_lookup, row0)
        plan = solve(dd, state, row0, final=(i == len(days)-1))
        for idx, rr in enumerate(gt.itertuples(index=False)):
            rep = replay(rr, plan, state, idx)
            state["E_soc"] = rep["E_soc"]; state["D_mth"] = rep["D_mth_running"]
            rows.append({"timestamp": pd.Timestamp(rr.timestamp), "day_index": int(rr.day_index), "month_id": int(rr.month_id), "calendar_day": str(pd.Timestamp(rr.calendar_day).date()), "hour": int(rr.hour_local)-1, "hour_local": int(rr.hour_local), "case_name": case, "solve_time": float(plan.get("solve_time", 0.0)) if idx == 0 else 0.0, "milp_status": int(plan.get("status", -999)), **meta, **rep})
        state["E_g"] = float(plan.get("E_g_end", state["E_g"]))
        if (i+1) % 50 == 0: print(f"{case}: {i+1}/365 days {time.time()-t0:.0f}s")
    h = pd.DataFrame(rows); d = daily_summary(h); h.to_parquet(hp, index=False); d.to_parquet(dp, index=False)
    return h, d, corrected_kpis(case, h, d, DESIGN)


def run_mpc(case: str, prob: bool):
    ensure_sched_out()
    hp = SCHED_OUT / f"{case}_hourly.parquet"; dp = SCHED_OUT / f"{case}_daily.parquet"
    if hp.exists() and dp.exists():
        h = pd.read_parquet(hp); d = pd.read_parquet(dp); return h, d, corrected_kpis(case, h, d, DESIGN)
    truth = load_truth(); load_lookup = load_profile_lookup("base")
    da_pvq = full_da_pv_quantiles(); da_pv_lookup = {pd.Timestamp(r.target_time): float(r.pv_q50) for r in da_pvq.itertuples(index=False)}
    id_pvq = full_id_pv_quantiles(); id_pv_lookup = {(pd.Timestamp(r.issue_time), pd.Timestamp(r.target_time)): float(r.pv_q50) for r in id_pvq.itertuples(index=False)}
    scen_groups = {}
    if prob:
        scen = pd.read_parquet(MPC_PROB_SCENARIO)
        scen["issue_time"] = pd.to_datetime(scen["issue_time"]); scen["target_time"] = pd.to_datetime(scen["target_time"])
        scen_groups = {k: g.copy() for k, g in scen.groupby("issue_time")}
    state = state0(); prev_month=None; rows=[]; t0=time.time()
    for i, rr in enumerate(truth.itertuples(index=False)):
        ts = pd.Timestamp(rr.timestamp); issue = ts - pd.Timedelta(hours=1)
        if prev_month is not None and int(rr.month_id) != prev_month: state["D_mth"] = 0.0
        prev_month = int(rr.month_id)
        if prob and issue in scen_groups:
            g = scen_groups[issue]
            # pad end if necessary
            if g["target_time"].nunique() < H:
                extras = []
                for sid, sg in g.groupby("scenario_id"):
                    for j in range(H):
                        target = ts + pd.Timedelta(hours=j)
                        if not sg["target_time"].eq(target).any():
                            extras.append({"issue_time": issue, "target_time": target, "lead_hour": j+1, "scenario_id": sid, "scenario_weight": float(sg["scenario_weight"].iloc[0]), "load_kw": load_lookup.get(target, 0.0), "pv_s_kw": da_pv_lookup.get(target, 0.0), "netload_s_kw": load_lookup.get(target, 0.0)-da_pv_lookup.get(target, 0.0)})
                if extras: g = pd.concat([g, pd.DataFrame(extras)], ignore_index=True)
            dd, meta = day_dict_prob(ts, g, rr)
        else:
            pv_path = {}
            for j in range(H):
                target = ts + pd.Timedelta(hours=j)
                pv_path[target] = id_pv_lookup.get((issue, target), da_pv_lookup.get(target, 0.0))
            dd, meta, _ = day_dict_point(ts, load_lookup, pv_path, rr)
        plan = solve(dd, state, rr, final=(i == len(truth)-1))
        rep = replay(rr, plan, state, 0)
        state = {"E_soc": rep["E_soc"], "E_g": float(plan.get("E_g_end", state["E_g"])), "D_mth": rep["D_mth_running"]}
        rows.append({"timestamp": ts, "day_index": int(rr.day_index), "month_id": int(rr.month_id), "calendar_day": str(pd.Timestamp(rr.calendar_day).date()), "hour": int(rr.hour_local)-1, "hour_local": int(rr.hour_local), "case_name": case, "solve_time": float(plan.get("solve_time", 0.0)), "milp_status": int(plan.get("status", -999)), **meta, **rep})
        if (i+1) % 1000 == 0: print(f"{case}: {i+1}/8760 hours {time.time()-t0:.0f}s")
    h = pd.DataFrame(rows); d = daily_summary(h); h.to_parquet(hp, index=False); d.to_parquet(dp, index=False)
    return h, d, corrected_kpis(case, h, d, DESIGN)


def collect(results):
    rows=[]; monthly=[]
    for case, (h,d,k) in results.items():
        rows.append(summary_row(case,h,d,k))
        for mo in range(1,13):
            hm=h[h["month_id"].eq(mo)]
            if hm.empty: continue
            monthly.append({"case":case,"month_id":mo,"TOU_m_ntd":hm["tou_ene_hour"].sum()/1e6,"OC_DCT_m_ntd":k["monthly_oc"].get(mo,0)/1e6,"monthly_peak_kw":k["monthly_d"].get(mo,np.nan),"oc_hours":int((hm["p_grid_realized"]*CFG["demand"]["kappa"]>CC).sum())})
    annual=pd.DataFrame(rows); mon=pd.DataFrame(monthly)
    annual.to_csv(SCHED_OUT/"annual_cost_summary_realized_basis.csv",index=False); annual.to_csv(SCHED_OUT/"cost_component_breakdown.csv",index=False)
    mon.to_csv(SCHED_OUT/"monthly_cost_summary.csv",index=False); mon[["case","month_id","monthly_peak_kw"]].to_csv(SCHED_OUT/"monthly_peak_summary.csv",index=False)
    annual[["case","oc_hours","monthly_peak_max_kw","OC_DCT_m_ntd"]].to_csv(SCHED_OUT/"oc_event_summary.csv",index=False)
    annual[["case","infeasible_steps","fallback_steps","solver_status_summary","solve_time_total_sec"]].to_csv(SCHED_OUT/"solver_status_summary.csv",index=False)
    write_usage_audits(annual)
    write_final_report(annual, mon)
    write_md(SCHED_OUT/"CCOR_MPC_NEXT_EXPERIMENT_PLAN.md","CCOR-MPC Next Experiment Plan",[
        ("Do Not Run Yet", "This batch intentionally stops before CCOR-MPC."),
        ("Planned Cases", "1. CCOR-MPC-DET-PVFOCUS_BASE\n2. CCOR-MPC-PROB-PVFOCUS_TKM_SAFE_BASE\n3. Optional CCOR-MPC-PROB-PVFOCUS_KM_BASE\n4. Structured-high robustness for MPC/CCOR-MPC."),
    ])


def scenario_audit() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("DA_PROB_PVFOCUS_LOWPV_SAFE_K5", DA_PROB_SCENARIO),
        ("MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", MPC_PROB_SCENARIO),
    ]:
        df = pd.read_parquet(path)
        df["issue_time"] = pd.to_datetime(df["issue_time"])
        w = df.drop_duplicates(["issue_time", "scenario_id"])
        scen_count = w.groupby("issue_time")["scenario_id"].nunique()
        wsum = w.groupby("issue_time")["scenario_weight"].sum()
        eff = w.groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum())
        truth_cols = [c for c in df.columns if any(tok in c.lower() for tok in ["realized", "actual", "truth"])]
        rows.append(
            {
                "case": label,
                "scenario_path": str(path.relative_to(REPO)),
                "bridge_type": str(df["bridge_type"].iloc[0]) if "bridge_type" in df.columns and len(df) else "",
                "k_min": int(scen_count.min()),
                "k_max": int(scen_count.max()),
                "weight_sum_min": float(wsum.min()),
                "weight_sum_max": float(wsum.max()),
                "effective_scenarios_min": float(eff.min()),
                "effective_scenarios_mean": float(eff.mean()),
                "effective_scenarios_max": float(eff.max()),
                "min_weight": float(w["scenario_weight"].min()),
                "max_weight": float(w["scenario_weight"].max()),
                "has_realized_or_truth_columns": bool(truth_cols),
                "truth_like_columns": ";".join(truth_cols),
                "max_netload_kw": float(df["netload_s_kw"].max()),
                "min_pv_kw": float(df["pv_s_kw"].min()),
                "max_pv_kw": float(df["pv_s_kw"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SCHED_OUT / "scenario_usage_audit.csv", index=False, encoding="utf-8-sig")
    return out


def replay_truth_audit() -> pd.DataFrame:
    truth = load_truth()
    rows = [
        {
            "component": "PV-focused scheduling replay/KPI truth",
            "truth_path": str(REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)),
            "pv_truth_source": PV_TRUTH_SOURCE_LABEL,
            "uses_old_invalid_ntust_solar_kwh": False,
            "row_count": len(truth),
            "pv_max_kw": float(truth["pv_realized_kw"].max()),
            "pv_mean_kw": float(truth["pv_realized_kw"].mean()),
            "load_max_kw": float(truth["load_realized_kw"].max()),
            "load_mean_kw": float(truth["load_realized_kw"].mean()),
            "source_marker_values": ";".join(sorted(truth["pv_truth_source"].dropna().astype(str).unique())) if "pv_truth_source" in truth.columns else "",
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(SCHED_OUT / "replay_truth_usage_audit.csv", index=False, encoding="utf-8-sig")
    return out


def write_usage_audits(annual: pd.DataFrame) -> None:
    scen = scenario_audit()
    truth = replay_truth_audit()
    _ = annual
    write_md(
        SCHED_OUT / "SCENARIO_AND_REPLAY_USAGE_AUDIT.md",
        "Scenario and Replay Usage Audit",
        [
            ("PV Truth Source", f"`PV truth source = {PV_TRUTH_SOURCE_LABEL}`; replay/KPI uses `{REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)}`."),
            ("Scenario Source", scen.to_markdown(index=False)),
            ("Replay Truth", truth.to_markdown(index=False)),
            ("Optimization Input Safety", "The probabilistic scenario files do not include realized/actual/truth columns. LOAD-QN and random load error are not used in this batch."),
        ],
    )


def diff_row(a: pd.Series, b: pd.Series, label: str) -> dict:
    fields = ["full_total_m_ntd", "TOU_m_ntd", "OC_DCT_m_ntd", "Deg_m_ntd", "monthly_peak_max_kw", "oc_hours"]
    out = {"comparison": label, "case_a": a["case"], "case_b": b["case"]}
    for f in fields:
        out[f"delta_{f}_b_minus_a"] = float(b[f] - a[f])
    return out


def write_final_report(annual: pd.DataFrame, mon: pd.DataFrame) -> None:
    idx = {r.case: r for r in annual.itertuples(index=False)}
    comps = []
    pairs = [
        ("DA_DET_PVFOCUS_BASE", "DA_PROB_PVFOCUS_LOWPV_SAFE_K5", "PV probabilistic value under DA"),
        ("MPC_DET_PVFOCUS_BASE", "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "PV probabilistic value under MPC"),
        ("DA_DET_PVFOCUS_BASE", "MPC_DET_PVFOCUS_BASE", "Rolling MPC value under point forecast"),
        ("DA_PROB_PVFOCUS_LOWPV_SAFE_K5", "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "Rolling MPC value under PV probabilistic forecast"),
    ]
    for a, b, label in pairs:
        if a in idx and b in idx:
            comps.append(diff_row(pd.Series(idx[a]._asdict()), pd.Series(idx[b]._asdict()), label))
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(SCHED_OUT / "pairwise_comparison_summary.csv", index=False, encoding="utf-8-sig")

    any_bad = bool((annual["infeasible_steps"].astype(int) > 0).any())
    status = "NEEDS_SOLVER_FIX" if any_bad else "BASIC_FOUR_CASES_READY_FOR_CCOR_NEXT"
    interp = []
    if not comp_df.empty:
        for r in comp_df.itertuples(index=False):
            interp.append(
                f"- {r.comparison}: {r.case_b} minus {r.case_a} = "
                f"{r.delta_full_total_m_ntd_b_minus_a:+.4f} M NTD total, "
                f"TOU {r.delta_TOU_m_ntd_b_minus_a:+.4f}, "
                f"OC/DCT {r.delta_OC_DCT_m_ntd_b_minus_a:+.4f}, "
                f"Deg {r.delta_Deg_m_ntd_b_minus_a:+.4f}, "
                f"monthly peak {r.delta_monthly_peak_max_kw_b_minus_a:+.2f} kW."
            )
    write_md(
        SCHED_OUT / "DA_MPC_PV_FOCUS_LOWPV_SAFE_FINAL_REPORT.md",
        "DA/MPC PV-Focused LowPV-Safe Scheduling Report",
        [
            ("Final Decision", f"`{status}`"),
            ("Case Definitions", "\n".join([
                "- `DA_DET_PVFOCUS_BASE`: deterministic given load + PV q50, DA MILP.",
                "- `DA_PROB_PVFOCUS_LOWPV_SAFE_K5`: deterministic given load + DA PV-focused K5 low-PV-safe scenarios, stochastic DA MILP.",
                "- `MPC_DET_PVFOCUS_BASE`: deterministic given load + issue-time PV q50/DA tail, hourly rolling MPC.",
                "- `MPC_PROB_PVFOCUS_LOWPV_SAFE_K5`: deterministic given load + ID H24 PV-focused K5 low-PV-safe scenarios, hourly stochastic MPC.",
            ])),
            ("PV Truth and Scenario Inputs", f"Replay/KPI uses `{REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)}` (`{PV_TRUTH_SOURCE_LABEL}`). DA-PROB uses `{DA_PROB_SCENARIO.relative_to(REPO)}`. MPC-PROB uses `{MPC_PROB_SCENARIO.relative_to(REPO)}`. Old S1-S6, LOAD-QN, random load error, and old invalid NTUST `Solar_kWh` PV truth are not used."),
            ("Annual Realized-Basis Cost", annual.to_markdown(index=False)),
            ("Monthly Cost Summary", mon.to_markdown(index=False)),
            ("Pairwise Comparisons", "\n".join(interp) if interp else "No comparisons available."),
            ("Interpretation", "Use the pairwise deltas to judge whether PV probabilistic scenarios reduce OC/DCT at the expense of TOU, whether rolling MPC improves over DA, and whether standard MPC still motivates CCOR-MPC."),
            ("Diagnostics", "Solver status, infeasible step counts, scenario weights, and replay truth source are reported in the required CSV audit files."),
            ("Next Batch Recommendation", "If the final decision is `BASIC_FOUR_CASES_READY_FOR_CCOR_NEXT`, the next batch should run `CCOR_MPC_DET_PVFOCUS_BASE`, `CCOR_MPC_PROB_PVFOCUS_LOWPV_SAFE_K5`, optional structured-high robustness, and optional ID K10 sensitivity. These were not run in this batch."),
        ],
    )


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    val=VALIDATION_REPORT
    if not val.exists() or "PASS_PV_TKM_LOWPV_SAFE_K5_FOR_DA_MPC_SCHEDULING" not in val.read_text(encoding="utf-8"):
        raise SystemExit("PV-focused validation has not passed; stop before scheduling.")
    ensure_sched_out()
    scenario_audit()
    replay_truth_audit()
    results={}
    results["DA_DET_PVFOCUS_BASE"]=run_da("DA_DET_PVFOCUS_BASE",False)
    results["DA_PROB_PVFOCUS_LOWPV_SAFE_K5"]=run_da("DA_PROB_PVFOCUS_LOWPV_SAFE_K5",True)
    results["MPC_DET_PVFOCUS_BASE"]=run_mpc("MPC_DET_PVFOCUS_BASE",False)
    results["MPC_PROB_PVFOCUS_LOWPV_SAFE_K5"]=run_mpc("MPC_PROB_PVFOCUS_LOWPV_SAFE_K5",True)
    collect(results)
    print(f"[PVFocus-6] scheduling complete under {SCHED_OUT}")


if __name__=="__main__":
    main()
