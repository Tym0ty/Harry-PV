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
OUT = REPO / "new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm"
BASE_MPC_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
BASE_CCOR_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150"
KM_SCENARIO = BRIDGE_OUT / "id_h24_pvfocus_base_km_K5_scenarios.parquet"
TKM_SCENARIO = BRIDGE_OUT / "id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet"
CCOR_VARIANT = Variant("CCOR_V2_SHIELD_ONLY_m150", "shield", margin_kw=150.0)


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def tou_path(start: pd.Timestamp) -> np.ndarray:
    return np.asarray([get_tou_price((start + pd.Timedelta(hours=j)).month, (start + pd.Timedelta(hours=j)).day, (start + pd.Timedelta(hours=j)).weekday(), (start + pd.Timedelta(hours=j)).hour) for j in range(H)], dtype=float)


def replay_base(row, plan: dict, state: dict, load_kw: float) -> dict:
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
    return solve_day_ahead(int(row.day_index), dd, state, DESIGN, CFG, is_final_day=False, time_limit=60, mip_gap=0.005, lam=0.0, alpha=0.90, verbose=False)


def daily_summary(h: pd.DataFrame) -> pd.DataFrame:
    dem = CFG["demand"]
    rows = []
    for (di, cd, mo), g in h.groupby(["day_index", "calendar_day", "month_id"], sort=True):
        peak = float(g["D_mth_running"].max())
        c_b = get_basic_charge(int(mo), CFG)
        over = max(peak - CC, 0.0)
        o1 = min(over, dem["over_contract_tier1_ratio"] * CC)
        o2 = max(over - dem["over_contract_tier1_ratio"] * CC, 0.0)
        rows.append({"day_index": int(di), "calendar_day": str(pd.Timestamp(cd).date()), "month_id": int(mo), "D_mth_end": peak, "C_oc": c_b * (dem["over_contract_m1"] * o1 + dem["over_contract_m2"] * o2), "E_soc_end": float(g["E_soc"].iloc[-1]), "p_ch_sum": float(g["p_ch_exec"].sum()), "p_dis_sum": float(g["p_dis_exec"].sum()), "tou_ene_day": float(g["tou_ene_hour"].sum()), "n_milp_fail": int((~g["milp_status"].isin(GOOD)).sum())})
    return pd.DataFrame(rows)


def run_case(case: str, ccor: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hp = OUT / f"{case}_hourly.parquet"
    dp = OUT / f"{case}_daily.parquet"
    if hp.exists() and dp.exists():
        h = pd.read_parquet(hp); h["timestamp"] = pd.to_datetime(h["timestamp"])
        d = pd.read_parquet(dp)
        return h, d, corrected_kpis(case, h, d, DESIGN)

    truth = load_truth()
    load_lookup = load_profile_lookup("base")
    da_pvq = full_da_pv_quantiles()
    da_pv_lookup = {pd.Timestamp(r.target_time): float(r.pv_q50) for r in da_pvq.itertuples(index=False)}
    scen = pd.read_parquet(KM_SCENARIO)
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
        g = scen_groups[issue]
        if g["target_time"].nunique() < H:
            extras = []
            for sid, sg in g.groupby("scenario_id"):
                for j in range(H):
                    target = ts + pd.Timedelta(hours=j)
                    if not sg["target_time"].eq(target).any():
                        load = load_lookup.get(target, 0.0)
                        pv = da_pv_lookup.get(target, 0.0)
                        extras.append({"issue_time": issue, "target_time": target, "lead_hour": j + 1, "scenario_id": sid, "scenario_weight": float(sg["scenario_weight"].iloc[0]), "load_profile_name": "base", "load_kw": load, "pv_s_kw": pv, "netload_s_kw": load - pv})
            if extras:
                g = pd.concat([g, pd.DataFrame(extras)], ignore_index=True)
        dd, meta = day_dict_prob(ts, g, rr)
        e_before = float(state["E_soc"])
        d_before = float(state["D_mth"])
        plan = solve_v2(int(rr.day_index), dd, state, CCOR_VARIANT) if ccor else solve_standard(dd, state, rr)
        rep = replay_base(rr, plan, state, load_lookup[ts])
        row = {"timestamp": ts, "day_index": int(rr.day_index), "month_id": int(rr.month_id), "calendar_day": str(pd.Timestamp(rr.calendar_day).date()), "hour": int(rr.hour_local) - 1, "hour_local": int(rr.hour_local), "case_name": case, "scenario_reduction": "PV_KM_K5", "solve_time": float(plan.get("solve_time", 0.0)), "milp_status": int(plan.get("status", -999)), "E_soc_before": e_before, "D_mtd_before": d_before, **meta, **rep}
        if ccor:
            row.update({"D_guard_kw": float(plan["D_guard_kw"]), "D_cert_kw": float(plan["D_cert_kw"]), "risk0": bool(plan["risk0"]), "z_shield_expected": float(plan["z_shield_expected"]), "z_shield_max": float(plan["z_shield_max"])})
        else:
            d_guard = max(CC, d_before)
            row.update({"D_guard_kw": d_guard, "D_cert_kw": d_guard, "risk0": bool(row["max_stress_netload_H24"] >= 0.95 * d_guard / KAPPA), "z_shield_expected": 0.0, "z_shield_max": 0.0})
        rows.append(row)
        state = {"E_soc": rep["E_soc"], "E_g": float(plan.get("E_g_end", state.get("E_g", 0.0))), "D_mth": rep["D_mth_running"]}
        if (i + 1) % 1000 == 0:
            print(f"{case}: {i+1}/8760 h, elapsed {time.time()-t0:.0f}s")
    h = pd.DataFrame(rows)
    d = daily_summary(h)
    h.to_parquet(hp, index=False)
    d.to_parquet(dp, index=False)
    return h, d, corrected_kpis(case, h, d, DESIGN)


def summary_row(case: str, h: pd.DataFrame, k: dict) -> dict:
    return {"case": case, "scenario_reduction": "PV_KM_K5", "full_total_m_ntd": k["full_total_NTD"] / 1e6, "capex_m_ntd": k["capex_NTD"] / 1e6, "basic_m_ntd": k["basic_NTD"] / 1e6, "TOU_m_ntd": k["tou_NTD"] / 1e6, "OC_DCT_m_ntd": k["oc_total_NTD"] / 1e6, "Deg_m_ntd": k["deg_NTD"] / 1e6, "TREC_RE20_m_ntd": k["trec_NTD"] / 1e6, "monthly_peak_max_kw": max(k["monthly_d"].values()), "oc_hours": int((h["p_grid_realized"] * KAPPA > CC).sum()), "risk0_hours": int(h["risk0"].sum()), "shield_slack_positive_hours": int((h["z_shield_max"] > 1e-6).sum()), "infeasible_steps": int((~h["milp_status"].isin(GOOD)).sum()), "fallback_steps": int((h["milp_status"] == -1).sum()), "solver_status_summary": json.dumps({str(k): int(v) for k, v in h["milp_status"].value_counts().to_dict().items()}), "solve_time_total_sec": float(h["solve_time"].sum())}


def scenario_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(KM_SCENARIO)
    df["issue_time"] = pd.to_datetime(df["issue_time"])
    w = df.drop_duplicates(["issue_time", "scenario_id"])
    cnt = w.groupby("issue_time")["scenario_id"].nunique()
    wsum = w.groupby("issue_time")["scenario_weight"].sum()
    eff = w.groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum())
    truth_cols = [c for c in df.columns if any(tok in c.lower() for tok in ["realized", "actual", "truth"])]
    audit = pd.DataFrame([{"case": "KM_K5_ABLATION", "scenario_file": str(KM_SCENARIO.relative_to(REPO)), "bridge_type": ";".join(sorted(df["bridge_type"].dropna().astype(str).unique())), "tail_aware": False, "low_pv_safe": False, "K_min": int(cnt.min()), "K_max": int(cnt.max()), "weights_type": "cluster_probability", "weight_sum_min": float(wsum.min()), "weight_sum_max": float(wsum.max()), "effective_scenarios_min": float(eff.min()), "effective_scenarios_mean": float(eff.mean()), "effective_scenarios_max": float(eff.max()), "load_profile_name": ";".join(sorted(df["load_profile_name"].dropna().astype(str).unique())), "has_realized_or_truth_columns": bool(truth_cols), "truth_like_columns": ";".join(truth_cols), "LOAD_QN_used": False, "random_load_error_used": False, "old_S1S6_used": False, "structured_load_bias_used": False}])
    audit.to_csv(OUT / "scenario_usage_audit.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "km_scenario_usage_audit.csv", index=False, encoding="utf-8-sig")
    eff.reset_index(name="effective_scenarios").to_csv(OUT / "km_scenario_weight_summary.csv", index=False, encoding="utf-8-sig")
    write_md(OUT / "KM_SCENARIO_PREPARATION_REPORT.md", "KM Scenario Preparation Report", [("Status", "Ordinary PV-KM K=5 scenario package exists and was audited."), ("Audit", audit.to_markdown(index=False))])
    return audit, eff.reset_index(name="effective_scenarios")


def replay_truth_audit() -> pd.DataFrame:
    truth = load_truth()
    out = pd.DataFrame([{"component": "tail-aware ablation replay/KPI truth", "truth_path": str(REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)), "pv_truth_source": PV_TRUTH_SOURCE_LABEL, "uses_old_invalid_ntust_solar_kwh": False, "uses_LOAD_QN": False, "uses_random_load_error": False, "uses_old_S1S6": False, "row_count": len(truth), "pv_max_kw": float(truth["pv_realized_kw"].max()), "pv_mean_kw": float(truth["pv_realized_kw"].mean())}])
    out.to_csv(OUT / "replay_truth_usage_audit.csv", index=False, encoding="utf-8-sig")
    return out


def collect(results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, mon = [], []
    for case, (h, _, k) in results.items():
        rows.append(summary_row(case, h, k))
        for mo in range(1, 13):
            hm = h[h["month_id"].eq(mo)]
            if hm.empty:
                continue
            mon.append({"case": case, "month_id": mo, "TOU_m_ntd": hm["tou_ene_hour"].sum() / 1e6, "OC_DCT_m_ntd": k["monthly_oc"].get(mo, 0) / 1e6, "Deg_m_ntd": hm["deg_hour"].sum() / 1e6, "monthly_peak_kw": k["monthly_d"].get(mo, np.nan), "oc_hours": int((hm["p_grid_realized"] * KAPPA > CC).sum())})
    annual = pd.DataFrame(rows)
    monthly = pd.DataFrame(mon)
    annual.to_csv(OUT / "annual_cost_summary_realized_basis.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "cost_component_breakdown.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "monthly_cost_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "infeasible_steps", "fallback_steps", "solver_status_summary", "solve_time_total_sec"]].to_csv(OUT / "solver_status_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "oc_hours", "monthly_peak_max_kw", "OC_DCT_m_ntd"]].to_csv(OUT / "oc_event_summary.csv", index=False, encoding="utf-8-sig")
    monthly[["case", "month_id", "monthly_peak_kw"]].to_csv(OUT / "monthly_peak_summary.csv", index=False, encoding="utf-8-sig")
    return annual, monthly


def comparisons(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    base = pd.concat([
        pd.read_csv(BASE_MPC_OUT / "annual_cost_summary_realized_basis.csv"),
        pd.read_csv(BASE_CCOR_OUT / "annual_cost_summary_realized_basis.csv"),
    ], ignore_index=True)
    all_rows = pd.concat([base, annual], ignore_index=True)
    idx = {r.case: r for r in all_rows.itertuples(index=False)}
    pairs = [
        ("MPC_PROB_PVFOCUS_KM_K5", "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "MPC tail-aware TKM vs ordinary KM"),
        ("CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5", "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5", "CCOR tail-aware TKM vs ordinary KM"),
    ]
    rows = []
    for km, tkm, label in pairs:
        ar = pd.Series(idx[km]._asdict())
        br = pd.Series(idx[tkm]._asdict())
        rows.append({"comparison": label, "ordinary_km_case": km, "tailaware_tkm_case": tkm, "delta_full_total_m_ntd_tkm_minus_km": float(br["full_total_m_ntd"] - ar["full_total_m_ntd"]), "delta_TOU_m_ntd_tkm_minus_km": float(br["TOU_m_ntd"] - ar["TOU_m_ntd"]), "delta_OC_DCT_m_ntd_tkm_minus_km": float(br["OC_DCT_m_ntd"] - ar["OC_DCT_m_ntd"]), "delta_Deg_m_ntd_tkm_minus_km": float(br["Deg_m_ntd"] - ar["Deg_m_ntd"]), "delta_monthly_peak_kw_tkm_minus_km": float(br["monthly_peak_max_kw"] - ar["monthly_peak_max_kw"]), "delta_oc_hours_tkm_minus_km": int(br["oc_hours"] - ar["oc_hours"]), "tailaware_improves_total": bool(br["full_total_m_ntd"] < ar["full_total_m_ntd"]), "tailaware_reduces_OC_DCT": bool(br["OC_DCT_m_ntd"] < ar["OC_DCT_m_ntd"]), "tailaware_increases_TOU": bool(br["TOU_m_ntd"] > ar["TOU_m_ntd"])})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "km_vs_tkm_delta_comparison.csv", index=False, encoding="utf-8-sig")
    interp = comp[["comparison", "tailaware_improves_total", "tailaware_reduces_OC_DCT", "tailaware_increases_TOU"]].copy()
    if comp["tailaware_improves_total"].any() or comp["tailaware_reduces_OC_DCT"].any():
        status = "TAILAWARE_ABLATION_SUPPORTS_TKM_MAIN"
    elif (comp["tailaware_increases_TOU"] & comp["tailaware_reduces_OC_DCT"]).any():
        status = "TAILAWARE_ABLATION_TKM_RISK_COST_TRADEOFF"
    elif (comp["delta_full_total_m_ntd_tkm_minus_km"] > 0).all() and (~comp["tailaware_reduces_OC_DCT"]).all():
        status = "TAILAWARE_ABLATION_KM_BETTER"
    else:
        status = "TAILAWARE_ABLATION_NEEDS_REVIEW"
    interp["final_status"] = status
    interp.to_csv(OUT / "tailaware_ablation_interpretation.csv", index=False, encoding="utf-8-sig")
    return comp, interp, status


def write_report(annual, monthly, scen_audit, truth_audit, comp, interp, status) -> None:
    write_md(OUT / "TAILAWARE_ABLATION_REPORT.md", "PV-Focused Tail-Aware Reduction Ablation", [
        ("Final Status", f"`{status}`"),
        ("Scope", "Ran exactly two ordinary PV-KM K=5 probabilistic cases under base load: standard MPC-PROB and CCOR-v2 PROB m150. No DA, DET, K10, structured load bias, LitErr, LOAD-QN, old S1-S6, or old invalid PV truth were used."),
        ("Annual Cost", annual.to_markdown(index=False)),
        ("KM vs Tail-Aware TKM Comparison", comp.to_markdown(index=False)),
        ("Interpretation", interp.to_markdown(index=False)),
        ("Scenario Audit", scen_audit.to_markdown(index=False)),
        ("Replay Truth Audit", truth_audit.to_markdown(index=False)),
        ("Monthly Cost", monthly.to_markdown(index=False)),
    ])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scen_audit, _ = scenario_audit()
    truth_audit = replay_truth_audit()
    results = {
        "MPC_PROB_PVFOCUS_KM_K5": run_case("MPC_PROB_PVFOCUS_KM_K5", ccor=False),
        "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5": run_case("CCOR_V2_PROB_SHIELD_m150_PVFOCUS_KM_K5", ccor=True),
    }
    annual, monthly = collect(results)
    comp, interp, status = comparisons(annual)
    write_report(annual, monthly, scen_audit, truth_audit, comp, interp, status)
    print(f"[Tail-aware ablation] {status} under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
