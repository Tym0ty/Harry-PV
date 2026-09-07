#!/usr/bin/env python3
from __future__ import annotations

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
from run_scheduling_pv_focus_ccor_v2_smoke import (  # noqa: E402
    CC,
    EB,
    GOOD,
    H,
    KAPPA,
    Variant,
    deg_cost_for_discharge,
    solve_v2,
    state0,
)
from run_da_formulation_rolling_controlled_experiments import load_design  # noqa: E402


CFG = load_config()
DESIGN = load_design()
OUT = REPO / "new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus"
BASE_MPC_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
BASE_CCOR_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150"
LOAD_PROFILE_PATH = BRIDGE_OUT / "load_bias_profiles.parquet"
CCOR_VARIANT = Variant("CCOR_V2_SHIELD_ONLY_m150", "shield", margin_kw=150.0)
PROFILES = ["structured_low", "structured_high"]


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def scenario_path(profile: str) -> Path:
    return BRIDGE_OUT / f"id_h24_pvfocus_{profile}_tkm_lowpv_safe_K5_scenarios.parquet"


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


def replay_profile(row, plan: dict, state: dict, load_kw: float) -> dict:
    bat = CFG["battery"]
    p_ch = max(float(plan["P_ch_plan"][0]), 0.0)
    p_dis = max(float(plan["P_dis_plan"][0]), 0.0)
    pv = float(row.pv_realized_kw)
    load = float(load_kw)
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
        "forecast_load_first": float(load[0]),
        "forecast_pv_first": float(pv[0]),
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
        "forecast_load_first": float(np.average([sc["load"][0] for sc in scenarios], weights=weights)),
        "forecast_pv_first": float(np.average([sc["pv"][0] for sc in scenarios], weights=weights)),
        "forecast_netload_first": float(np.average([nl[0] for nl in nls], weights=weights)),
        "max_stress_netload_H24": float(np.max(np.vstack(nls))),
        "forecast_peak_lead": int(np.argmax(np.max(np.vstack(nls), axis=0)) + 1),
    }


def solve_standard(dd: dict, state: dict, row) -> dict:
    return solve_day_ahead(
        int(row.day_index),
        dd,
        state,
        DESIGN,
        CFG,
        is_final_day=False,
        time_limit=60,
        mip_gap=0.005,
        lam=0.0,
        alpha=0.90,
        verbose=False,
    )


def run_case(case: str, profile: str, prob: bool, ccor: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hp = OUT / f"{case}_hourly.parquet"
    dp = OUT / f"{case}_daily.parquet"
    if hp.exists() and dp.exists():
        h = pd.read_parquet(hp)
        h["timestamp"] = pd.to_datetime(h["timestamp"])
        d = pd.read_parquet(dp)
        return h, d, corrected_kpis(case, h, d, DESIGN)

    truth = load_truth()
    load_lookup = load_profile_lookup(profile)
    da_pvq = full_da_pv_quantiles()
    da_pv_lookup = {pd.Timestamp(r.target_time): float(r.pv_q50) for r in da_pvq.itertuples(index=False)}
    id_pvq = full_id_pv_quantiles()
    id_pv_lookup = {(pd.Timestamp(r.issue_time), pd.Timestamp(r.target_time)): float(r.pv_q50) for r in id_pvq.itertuples(index=False)}
    scen_groups = {}
    if prob:
        scen = pd.read_parquet(scenario_path(profile))
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
                            extras.append({
                                "issue_time": issue,
                                "target_time": target,
                                "lead_hour": j + 1,
                                "scenario_id": sid,
                                "scenario_weight": float(sg["scenario_weight"].iloc[0]),
                                "load_profile_name": profile,
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

        e_before = float(state["E_soc"])
        d_before = float(state["D_mth"])
        plan = solve_v2(int(rr.day_index), dd, state, CCOR_VARIANT) if ccor else solve_standard(dd, state, rr)
        rep = replay_profile(rr, plan, state, load_lookup[ts])
        row = {
            "timestamp": ts,
            "day_index": int(rr.day_index),
            "month_id": int(rr.month_id),
            "calendar_day": str(pd.Timestamp(rr.calendar_day).date()),
            "hour": int(rr.hour_local) - 1,
            "hour_local": int(rr.hour_local),
            "case_name": case,
            "load_profile_name": profile,
            "probabilistic": bool(prob),
            "ccor_v2": bool(ccor),
            "solve_time": float(plan.get("solve_time", 0.0)),
            "milp_status": int(plan.get("status", -999)),
            "E_soc_before": e_before,
            "D_mtd_before": d_before,
            **meta,
            **rep,
        }
        if ccor:
            row.update({
                "D_guard_kw": float(plan["D_guard_kw"]),
                "D_cert_kw": float(plan["D_cert_kw"]),
                "risk0": bool(plan["risk0"]),
                "first_step_netload_max_kw": float(plan["first_step_netload_max_kw"]),
                "first_step_demand_max": float(plan["first_step_demand_max"]),
                "z_shield_expected": float(plan["z_shield_expected"]),
                "z_shield_max": float(plan["z_shield_max"]),
            })
        else:
            d_guard = max(CC, d_before)
            row.update({
                "D_guard_kw": d_guard,
                "D_cert_kw": d_guard,
                "risk0": bool(row["max_stress_netload_H24"] >= 0.95 * d_guard / KAPPA),
                "first_step_netload_max_kw": float(row["forecast_netload_first"]),
                "first_step_demand_max": float(KAPPA * max(row["forecast_netload_first"] + rep["p_ch_exec"] - rep["p_dis_exec"], 0.0)),
                "z_shield_expected": 0.0,
                "z_shield_max": 0.0,
            })
        rows.append(row)
        state = {"E_soc": rep["E_soc"], "E_g": float(plan.get("E_g_end", state.get("E_g", 0.0))), "D_mth": rep["D_mth_running"]}
        if (i + 1) % 1000 == 0:
            print(f"{case}: {i+1}/8760 h, elapsed {time.time()-t0:.0f}s")
    h = pd.DataFrame(rows)
    d = daily_summary(h)
    h.to_parquet(hp, index=False)
    d.to_parquet(dp, index=False)
    return h, d, corrected_kpis(case, h, d, DESIGN)


def daily_summary(h: pd.DataFrame) -> pd.DataFrame:
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


def summary_row(case: str, profile: str, h: pd.DataFrame, k: dict) -> dict:
    return {
        "case": case,
        "load_profile_name": profile,
        "full_total_m_ntd": k["full_total_NTD"] / 1e6,
        "capex_m_ntd": k["capex_NTD"] / 1e6,
        "basic_m_ntd": k["basic_NTD"] / 1e6,
        "TOU_m_ntd": k["tou_NTD"] / 1e6,
        "OC_DCT_m_ntd": k["oc_total_NTD"] / 1e6,
        "Deg_m_ntd": k["deg_NTD"] / 1e6,
        "TREC_RE20_m_ntd": k["trec_NTD"] / 1e6,
        "monthly_peak_max_kw": max(k["monthly_d"].values()),
        "oc_hours": int((h["p_grid_realized"] * KAPPA > CC).sum()),
        "first_step_charging_near_risk_hours": int(((h["risk0"]) & (h["p_ch_exec"] > 1e-6)).sum()),
        "shield_slack_positive_hours": int((h["z_shield_max"] > 1e-6).sum()),
        "risk0_hours": int(h["risk0"].sum()),
        "soc_min_kwh": float(h["E_soc"].min()),
        "soc_max_kwh": float(h["E_soc"].max()),
        "infeasible_steps": int((~h["milp_status"].isin(GOOD)).sum()),
        "fallback_steps": int((h["milp_status"] == -1).sum()),
        "solver_status_summary": json.dumps({str(k): int(v) for k, v in h["milp_status"].value_counts().to_dict().items()}),
        "solve_time_total_sec": float(h["solve_time"].sum()),
    }


def collect(results: dict[str, tuple[str, pd.DataFrame, pd.DataFrame, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, mon = [], []
    for case, (profile, h, _, k) in results.items():
        rows.append(summary_row(case, profile, h, k))
        for mo in range(1, 13):
            hm = h[h["month_id"].eq(mo)]
            if hm.empty:
                continue
            mon.append({
                "case": case,
                "load_profile_name": profile,
                "month_id": mo,
                "TOU_m_ntd": hm["tou_ene_hour"].sum() / 1e6,
                "OC_DCT_m_ntd": k["monthly_oc"].get(mo, 0) / 1e6,
                "Deg_m_ntd": hm["deg_hour"].sum() / 1e6,
                "monthly_peak_kw": k["monthly_d"].get(mo, np.nan),
                "oc_hours": int((hm["p_grid_realized"] * KAPPA > CC).sum()),
                "risk0_hours": int(hm["risk0"].sum()),
                "shield_slack_positive_hours": int((hm["z_shield_max"] > 1e-6).sum()),
                "first_step_charging_near_risk_hours": int(((hm["risk0"]) & (hm["p_ch_exec"] > 1e-6)).sum()),
            })
    annual = pd.DataFrame(rows)
    monthly = pd.DataFrame(mon)
    annual.to_csv(OUT / "annual_cost_summary_realized_basis.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "cost_component_breakdown.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "monthly_cost_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "load_profile_name", "infeasible_steps", "fallback_steps", "solver_status_summary", "solve_time_total_sec"]].to_csv(
        OUT / "solver_status_summary.csv", index=False, encoding="utf-8-sig"
    )
    annual[["case", "load_profile_name", "oc_hours", "monthly_peak_max_kw", "OC_DCT_m_ntd"]].to_csv(
        OUT / "oc_event_summary.csv", index=False, encoding="utf-8-sig"
    )
    monthly[["case", "load_profile_name", "month_id", "monthly_peak_kw"]].to_csv(
        OUT / "monthly_peak_summary.csv", index=False, encoding="utf-8-sig"
    )
    return annual, monthly


def load_profile_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    lp = pd.read_parquet(LOAD_PROFILE_PATH)
    rows = []
    month_rows = []
    for profile in ["base", "structured_low", "structured_high"]:
        col = f"load_{profile}_kw"
        rows.append({
            "load_profile_name": profile,
            "column": col,
            "exists": col in lp.columns,
            "annual_kwh": float(lp[col].sum()),
            "annual_pct_vs_base": float((lp[col].sum() / lp["load_base_kw"].sum() - 1.0) * 100.0),
            "max_load_kw": float(lp[col].max()),
            "min_load_kw": float(lp[col].min()),
            "negative_load_count": int((lp[col] < 0).sum()),
            "oc_risk_hours_gt_CC": int((lp[col] * KAPPA > CC).sum()),
            "near_oc_hours_gt_095CC": int((lp[col] * KAPPA > 0.95 * CC).sum()),
            "load_qn_used": False,
            "random_load_error_used": False,
        })
        for mo, g in lp.groupby("month_id", sort=True):
            month_rows.append({
                "load_profile_name": profile,
                "month_id": int(mo),
                "monthly_kwh": float(g[col].sum()),
                "monthly_peak_kw": float(g[col].max()),
                "oc_risk_hours_gt_CC": int((g[col] * KAPPA > CC).sum()),
                "near_oc_hours_gt_095CC": int((g[col] * KAPPA > 0.95 * CC).sum()),
            })
    audit = pd.DataFrame(rows)
    mon = pd.DataFrame(month_rows)
    audit.to_csv(OUT / "load_profile_usage_audit.csv", index=False, encoding="utf-8-sig")
    mon.to_csv(OUT / "structured_load_bias_profile_check.csv", index=False, encoding="utf-8-sig")
    return audit, mon


def scenario_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, weights = [], []
    for profile in PROFILES:
        path = scenario_path(profile)
        df = pd.read_parquet(path)
        df["issue_time"] = pd.to_datetime(df["issue_time"])
        w = df.drop_duplicates(["issue_time", "scenario_id"])
        cnt = w.groupby("issue_time")["scenario_id"].nunique()
        wsum = w.groupby("issue_time")["scenario_weight"].sum()
        eff = w.groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum())
        truth_cols = [c for c in df.columns if any(tok in c.lower() for tok in ["realized", "actual", "truth"])]
        profile_values = sorted(df["load_profile_name"].dropna().astype(str).unique()) if "load_profile_name" in df.columns else []
        rows.append({
            "load_profile_name": profile,
            "scenario_file": str(path.relative_to(REPO)),
            "exists": path.exists(),
            "k_min": int(cnt.min()),
            "k_max": int(cnt.max()),
            "weight_sum_min": float(wsum.min()),
            "weight_sum_max": float(wsum.max()),
            "effective_scenarios_min": float(eff.min()),
            "effective_scenarios_mean": float(eff.mean()),
            "effective_scenarios_max": float(eff.max()),
            "profile_values": ";".join(profile_values),
            "has_realized_or_truth_columns": bool(truth_cols),
            "truth_like_columns": ";".join(truth_cols),
            "bridge_type_values": ";".join(sorted(df["bridge_type"].dropna().astype(str).unique())) if "bridge_type" in df.columns else "",
        })
        tmp = eff.reset_index(name="effective_scenarios")
        tmp["load_profile_name"] = profile
        weights.append(tmp)
    audit = pd.DataFrame(rows)
    effdf = pd.concat(weights, ignore_index=True)
    audit.to_csv(OUT / "scenario_usage_audit.csv", index=False, encoding="utf-8-sig")
    effdf.to_csv(OUT / "structured_low_high_scenario_weight_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "structured_low_high_scenario_validation_summary.csv", index=False, encoding="utf-8-sig")
    return audit, effdf, audit.copy()


def replay_truth_audit() -> pd.DataFrame:
    truth = load_truth()
    out = pd.DataFrame([{
        "component": "load-bias robustness replay/KPI truth",
        "truth_path": str(REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)),
        "pv_truth_source": PV_TRUTH_SOURCE_LABEL,
        "uses_old_invalid_ntust_solar_kwh": False,
        "uses_LOAD_QN": False,
        "uses_random_load_error": False,
        "row_count": len(truth),
        "pv_max_kw": float(truth["pv_realized_kw"].max()),
        "pv_mean_kw": float(truth["pv_realized_kw"].mean()),
    }])
    out.to_csv(OUT / "replay_truth_usage_audit.csv", index=False, encoding="utf-8-sig")
    return out


def ccor_shield_audit(results) -> pd.DataFrame:
    rows = []
    for case, (profile, h, _, _) in results.items():
        if not bool(h["ccor_v2"].iloc[0]):
            continue
        for mo, g in h.groupby("month_id", sort=True):
            rows.append({
                "case": case,
                "load_profile_name": profile,
                "month_id": int(mo),
                "shield_slack_positive_hours": int((g["z_shield_max"] > 1e-6).sum()),
                "z_shield_max_kw": float(g["z_shield_max"].max()),
                "first_step_charging_near_risk_hours": int(((g["risk0"]) & (g["p_ch_exec"] > 1e-6)).sum()),
                "risk0_hours": int(g["risk0"].sum()),
                "monthly_peak_kw": float(g["D_mth_running"].max()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ccor_shield_audit.csv", index=False, encoding="utf-8-sig")
    return out


def base_rows_for_comparison() -> pd.DataFrame:
    rows = []
    base_mpc = pd.read_csv(BASE_MPC_OUT / "annual_cost_summary_realized_basis.csv")
    base_ccor = pd.read_csv(BASE_CCOR_OUT / "annual_cost_summary_realized_basis.csv")
    mapping = {
        "MPC_DET_PVFOCUS_BASE": "MPC_DET",
        "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5": "MPC_PROB",
        "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE": "CCOR_DET",
        "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5": "CCOR_PROB",
    }
    for df in [base_mpc, base_ccor]:
        for r in df.itertuples(index=False):
            if r.case in mapping:
                d = r._asdict()
                d["method_key"] = mapping[r.case]
                d["load_profile_name"] = "base"
                rows.append(d)
    return pd.DataFrame(rows)


def comparisons(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_map = {
        "MPC_DET_PVFOCUS_STRUCTURED_LOW": "MPC_DET",
        "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_LOW": "MPC_PROB",
        "CCOR_V2_DET_SHIELD_m150_STRUCTURED_LOW": "CCOR_DET",
        "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_LOW": "CCOR_PROB",
        "MPC_DET_PVFOCUS_STRUCTURED_HIGH": "MPC_DET",
        "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_HIGH": "MPC_PROB",
        "CCOR_V2_DET_SHIELD_m150_STRUCTURED_HIGH": "CCOR_DET",
        "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5_STRUCTURED_HIGH": "CCOR_PROB",
    }
    ann = annual.copy()
    ann["method_key"] = ann["case"].map(method_map)
    all_profiles = pd.concat([base_rows_for_comparison(), ann], ignore_index=True, sort=False)
    pairs = []
    for profile in PROFILES:
        sub = ann[ann["load_profile_name"].eq(profile)]
        idx = {r.method_key: r for r in sub.itertuples(index=False)}
        for a, b, label in [
            ("MPC_DET", "MPC_PROB", "MPC_PROB_vs_MPC_DET"),
            ("CCOR_DET", "CCOR_PROB", "CCOR_PROB_vs_CCOR_DET"),
            ("MPC_DET", "CCOR_DET", "CCOR_DET_vs_MPC_DET"),
            ("MPC_PROB", "CCOR_PROB", "CCOR_PROB_vs_MPC_PROB"),
        ]:
            if a in idx and b in idx:
                ar = pd.Series(idx[a]._asdict())
                br = pd.Series(idx[b]._asdict())
                pairs.append({
                    "load_profile_name": profile,
                    "comparison": label,
                    "case_a": ar["case"],
                    "case_b": br["case"],
                    "delta_full_total_m_ntd_b_minus_a": float(br["full_total_m_ntd"] - ar["full_total_m_ntd"]),
                    "delta_TOU_m_ntd_b_minus_a": float(br["TOU_m_ntd"] - ar["TOU_m_ntd"]),
                    "delta_OC_DCT_m_ntd_b_minus_a": float(br["OC_DCT_m_ntd"] - ar["OC_DCT_m_ntd"]),
                    "delta_Deg_m_ntd_b_minus_a": float(br["Deg_m_ntd"] - ar["Deg_m_ntd"]),
                    "delta_monthly_peak_kw_b_minus_a": float(br["monthly_peak_max_kw"] - ar["monthly_peak_max_kw"]),
                    "delta_oc_hours_b_minus_a": int(br["oc_hours"] - ar["oc_hours"]),
                })
    within = pd.DataFrame(pairs)
    within.to_csv(OUT / "robustness_delta_comparison.csv", index=False, encoding="utf-8-sig")

    cross_rows = []
    for method, g in all_profiles.groupby("method_key"):
        idx = {r.load_profile_name: r for r in g.itertuples(index=False)}
        for prof in ["structured_low", "structured_high"]:
            if "base" in idx and prof in idx:
                ar = pd.Series(idx["base"]._asdict())
                br = pd.Series(idx[prof]._asdict())
                cross_rows.append({
                    "method_key": method,
                    "load_profile_a": "base",
                    "load_profile_b": prof,
                    "case_b": br["case"],
                    "delta_full_total_m_ntd_b_minus_base": float(br["full_total_m_ntd"] - ar["full_total_m_ntd"]),
                    "delta_TOU_m_ntd_b_minus_base": float(br["TOU_m_ntd"] - ar["TOU_m_ntd"]),
                    "delta_OC_DCT_m_ntd_b_minus_base": float(br["OC_DCT_m_ntd"] - ar["OC_DCT_m_ntd"]),
                    "delta_Deg_m_ntd_b_minus_base": float(br["Deg_m_ntd"] - ar["Deg_m_ntd"]),
                    "delta_monthly_peak_kw_b_minus_base": float(br["monthly_peak_max_kw"] - ar["monthly_peak_max_kw"]),
                    "delta_oc_hours_b_minus_base": int(br["oc_hours"] - ar["oc_hours"]),
                })
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(OUT / "structured_load_bias_comparison.csv", index=False, encoding="utf-8-sig")
    return within, cross


def final_status(annual: pd.DataFrame, within: pd.DataFrame) -> str:
    if (annual["infeasible_steps"] > 0).any() or (annual["fallback_steps"] > 0).any():
        return "NEEDS_MODEL_OR_DATA_REVIEW"
    checks = {}
    for profile in PROFILES:
        sub = within[within["load_profile_name"].eq(profile)]
        def delta(name):
            r = sub[sub["comparison"].eq(name)]
            return None if r.empty else float(r.iloc[0]["delta_full_total_m_ntd_b_minus_a"])
        checks[(profile, "ccor_prob_vs_mpc_prob")] = delta("CCOR_PROB_vs_MPC_PROB")
        checks[(profile, "mpc_prob_vs_det")] = delta("MPC_PROB_vs_MPC_DET")
        checks[(profile, "ccor_prob_vs_det")] = delta("CCOR_PROB_vs_CCOR_DET")
    stable_ccor = all(v is not None and v < 0 for k, v in checks.items() if k[1] == "ccor_prob_vs_mpc_prob")
    stable_prob = all(v is not None and v < 0 for k, v in checks.items() if k[1] in {"mpc_prob_vs_det", "ccor_prob_vs_det"})
    if stable_ccor and stable_prob:
        return "LOAD_ROBUSTNESS_PASS_MAIN_CONCLUSION_STABLE"
    if stable_ccor or stable_prob:
        return "LOAD_ROBUSTNESS_PARTIAL_PASS"
    return "LOAD_ROBUSTNESS_FAIL_MAIN_CONCLUSION_SENSITIVE"


def write_report(annual, monthly, load_audit, scenario_usage, replay_audit, shield, within, cross, status) -> None:
    write_md(
        OUT / "FINAL_FOUR_LOAD_BIAS_ROBUSTNESS_REPORT.md",
        "Final Four Structured Load-Bias Robustness Report",
        [
            ("Final Status", f"`{status}`"),
            ("Scope", "Ran exactly the final four methods under `structured_low` and `structured_high` load profiles. DA, K10, LitErr random load error, LOAD-QN, old S1-S6, and old invalid PV truth were not used."),
            ("Methodological Framing", "Structured load bias is an outer robustness setting, not a probabilistic load forecast. Base, structured-low, and structured-high profiles are separate deterministic cases and are not mixed or assigned probabilities."),
            ("Annual Cost", annual.to_markdown(index=False)),
            ("Within-Profile Comparisons", within.to_markdown(index=False)),
            ("Across-Profile Deltas", cross.to_markdown(index=False)),
            ("Load Profile Audit", load_audit.to_markdown(index=False)),
            ("Scenario Usage Audit", scenario_usage.to_markdown(index=False)),
            ("Replay Truth Audit", replay_audit.to_markdown(index=False)),
            ("CCOR Shield Audit", shield.to_markdown(index=False)),
            ("Monthly Cost", monthly.to_markdown(index=False)),
        ],
    )


def main() -> int:
    ensure_out()
    load_audit, _ = load_profile_audit()
    scenario_usage, _, _ = scenario_audit()
    replay_audit = replay_truth_audit()
    results = {}
    for profile, suffix in [("structured_low", "STRUCTURED_LOW"), ("structured_high", "STRUCTURED_HIGH")]:
        cases = [
            (f"MPC_DET_PVFOCUS_{suffix}", False, False),
            (f"MPC_PROB_PVFOCUS_LOWPV_SAFE_K5_{suffix}", True, False),
            (f"CCOR_V2_DET_SHIELD_m150_{suffix}", False, True),
            (f"CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5_{suffix}", True, True),
        ]
        for case, prob, ccor in cases:
            h, d, k = run_case(case, profile, prob=prob, ccor=ccor)
            results[case] = (profile, h, d, k)
    annual, monthly = collect(results)
    shield = ccor_shield_audit(results)
    within, cross = comparisons(annual)
    status = final_status(annual, within)
    write_report(annual, monthly, load_audit, scenario_usage, replay_audit, shield, within, cross, status)
    print(f"[Load-bias robustness] {status} under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
