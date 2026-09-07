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

from common import get_basic_charge, load_config  # noqa: E402
from mpc_fixed_horizon_utils import corrected_kpis  # noqa: E402
from pv_focus_bridge_utils import (  # noqa: E402
    PV_TRUTH_SOURCE_LABEL,
    REPLAY_TRUTH_PACKAGE_PATH,
    load_truth,
    write_md,
)
from run_scheduling_pv_focus_ccor_v2_smoke import (  # noqa: E402
    CC,
    EB,
    GOOD,
    H,
    KAPPA,
    PROB_SCENARIO,
    Variant,
    cost_summary_for_hourly,
    day_dict_point,
    day_dict_prob,
    deg_cost_for_discharge,
    full_da_pv_quantiles,
    full_id_pv_quantiles,
    load_design,
    load_profile_lookup,
    replay,
    solve_v2,
    state0,
)

CFG = load_config()
DESIGN = load_design()
OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150"
STD_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
VARIANT = Variant("CCOR_V2_SHIELD_ONLY_m150", "shield", margin_kw=150.0)


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


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


def run_full_year(case: str, prob: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hp = OUT / f"{case}_hourly.parquet"
    dp = OUT / f"{case}_daily.parquet"
    if hp.exists() and dp.exists():
        h = pd.read_parquet(hp)
        h["timestamp"] = pd.to_datetime(h["timestamp"])
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
        plan = solve_v2(int(rr.day_index), dd, state, VARIANT)
        rep = replay(rr, plan, state)
        rows.append({
            "timestamp": ts,
            "day_index": int(rr.day_index),
            "month_id": int(rr.month_id),
            "calendar_day": str(pd.Timestamp(rr.calendar_day).date()),
            "hour": int(rr.hour_local) - 1,
            "hour_local": int(rr.hour_local),
            "case_name": case,
            "variant": VARIANT.name,
            "probabilistic": bool(prob),
            "margin_kw": 150.0,
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
        if (i + 1) % 1000 == 0:
            print(f"{case}: {i+1}/8760 hours, elapsed {time.time()-t0:.0f}s")

    h = pd.DataFrame(rows)
    d = daily_summary(h)
    h.to_parquet(hp, index=False)
    d.to_parquet(dp, index=False)
    return h, d, corrected_kpis(case, h, d, DESIGN)


def summary_row(case: str, h: pd.DataFrame, k: dict) -> dict:
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


def collect(results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, monthly = [], []
    for case, (h, _, k) in results.items():
        rows.append(summary_row(case, h, k))
        for mo in range(1, 13):
            hm = h[h["month_id"].eq(mo)]
            if hm.empty:
                continue
            monthly.append({
                "case": case,
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
    mon = pd.DataFrame(monthly)
    annual.to_csv(OUT / "annual_cost_summary_realized_basis.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(OUT / "cost_component_breakdown.csv", index=False, encoding="utf-8-sig")
    mon.to_csv(OUT / "monthly_cost_summary.csv", index=False, encoding="utf-8-sig")
    mon[["case", "month_id", "monthly_peak_kw"]].to_csv(OUT / "monthly_peak_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "oc_hours", "monthly_peak_max_kw", "OC_DCT_m_ntd"]].to_csv(OUT / "oc_event_summary.csv", index=False, encoding="utf-8-sig")
    annual[["case", "infeasible_steps", "fallback_steps", "solver_status_summary", "solve_time_total_sec"]].to_csv(OUT / "solver_status_summary.csv", index=False, encoding="utf-8-sig")
    return annual, mon


def usage_audits() -> tuple[pd.DataFrame, pd.DataFrame]:
    scen = pd.read_parquet(PROB_SCENARIO)
    scen["issue_time"] = pd.to_datetime(scen["issue_time"])
    w = scen.drop_duplicates(["issue_time", "scenario_id"])
    scen_count = w.groupby("issue_time")["scenario_id"].nunique()
    wsum = w.groupby("issue_time")["scenario_weight"].sum()
    eff = w.groupby("issue_time")["scenario_weight"].apply(lambda x: 1.0 / np.square(x.astype(float)).sum())
    truth_cols = [c for c in scen.columns if any(tok in c.lower() for tok in ["realized", "actual", "truth"])]
    scen_audit = pd.DataFrame([{
        "case": "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5",
        "scenario_path": str(PROB_SCENARIO.relative_to(REPO)),
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
    scen_audit.to_csv(OUT / "scenario_usage_audit.csv", index=False, encoding="utf-8-sig")

    truth = load_truth()
    truth_audit = pd.DataFrame([{
        "component": "CCOR-v2 full-year replay/KPI truth",
        "truth_path": str(REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)),
        "pv_truth_source": PV_TRUTH_SOURCE_LABEL,
        "uses_old_invalid_ntust_solar_kwh": False,
        "row_count": len(truth),
        "pv_max_kw": float(truth["pv_realized_kw"].max()),
        "pv_mean_kw": float(truth["pv_realized_kw"].mean()),
        "source_marker_values": ";".join(sorted(truth["pv_truth_source"].dropna().astype(str).unique())) if "pv_truth_source" in truth.columns else "",
    }])
    truth_audit.to_csv(OUT / "replay_truth_usage_audit.csv", index=False, encoding="utf-8-sig")
    return scen_audit, truth_audit


def write_mechanism_audits(results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]]) -> None:
    h = pd.concat([v[0] for v in results.values()], ignore_index=True)
    h.to_parquet(OUT / "CCOR_V2_FULL_YEAR_ALL_HOURLY.parquet", index=False)

    h[[
        "timestamp", "case_name", "month_id", "D_mtd_before", "D_guard_kw", "D_cert_kw",
        "risk0", "first_step_netload_expected_kw", "first_step_netload_max_kw",
        "first_step_demand_expected", "first_step_demand_max", "p_ch_exec", "p_dis_exec",
        "p_grid_realized", "D_mth_running", "z_shield_expected", "z_shield_max", "E_soc_before", "E_soc",
    ]].to_csv(OUT / "first_step_peak_shield_audit.csv", index=False, encoding="utf-8-sig")

    rows = []
    for case, g in h.groupby("case_name"):
        rows.append({
            "case": case,
            "risk0_hours": int(g["risk0"].sum()),
            "first_step_charging_near_risk_hours": int(((g["risk0"]) & (g["p_ch_exec"] > 1e-6)).sum()),
            "mean_p_ch_risk0_kw": float(g.loc[g["risk0"], "p_ch_exec"].mean()) if bool(g["risk0"].any()) else 0.0,
            "max_p_ch_risk0_kw": float(g.loc[g["risk0"], "p_ch_exec"].max()) if bool(g["risk0"].any()) else 0.0,
        })
    pd.DataFrame(rows).to_csv(OUT / "first_step_charging_near_risk_summary.csv", index=False, encoding="utf-8-sig")

    rows = []
    for case, g in h.groupby("case_name"):
        rows.append({
            "case": case,
            "shield_slack_positive_hours": int((g["z_shield_max"] > 1e-6).sum()),
            "z_shield_expected_sum_kw": float(g["z_shield_expected"].sum()),
            "z_shield_max_kw": float(g["z_shield_max"].max()),
            "infeasible_steps": int((~g["milp_status"].isin(GOOD)).sum()),
            "fallback_steps": int((g["milp_status"] == -1).sum()),
        })
    pd.DataFrame(rows).to_csv(OUT / "shield_slack_summary.csv", index=False, encoding="utf-8-sig")

    peak_rows, soc_rows = [], []
    h["demand_kw"] = KAPPA * h["p_grid_realized"]
    for case, gc in h.groupby("case_name"):
        for mo, gm in gc.groupby("month_id"):
            idx = gm["demand_kw"].idxmax()
            r = h.loc[idx]
            peak_rows.append({
                "case": case,
                "month_id": int(mo),
                "peak_timestamp": r["timestamp"],
                "peak_demand_kw": float(r["demand_kw"]),
                "grid_import_kw": float(r["p_grid_realized"]),
                "soc_before_kwh": float(r["E_soc_before"]),
                "soc_after_kwh": float(r["E_soc"]),
                "p_ch_kw": float(r["p_ch_exec"]),
                "p_dis_kw": float(r["p_dis_exec"]),
                "D_guard_kw": float(r["D_guard_kw"]),
                "D_cert_kw": float(r["D_cert_kw"]),
                "risk0": bool(r["risk0"]),
                "z_shield_max_kw": float(r["z_shield_max"]),
                "first_step_netload_max_kw": float(r["first_step_netload_max_kw"]),
            })
            ts = pd.Timestamp(r["timestamp"])
            win = gc[(gc["timestamp"] >= ts - pd.Timedelta(hours=6)) & (gc["timestamp"] <= ts + pd.Timedelta(hours=6))]
            for wr in win.itertuples(index=False):
                soc_rows.append({
                    "case": case,
                    "event_month_id": int(mo),
                    "event_peak_timestamp": ts,
                    "timestamp": pd.Timestamp(wr.timestamp),
                    "relative_hour": int((pd.Timestamp(wr.timestamp) - ts) / pd.Timedelta(hours=1)),
                    "demand_kw": float(wr.demand_kw),
                    "grid_import_kw": float(wr.p_grid_realized),
                    "soc_before_kwh": float(wr.E_soc_before),
                    "soc_after_kwh": float(wr.E_soc),
                    "p_ch_kw": float(wr.p_ch_exec),
                    "p_dis_kw": float(wr.p_dis_exec),
                    "D_guard_kw": float(wr.D_guard_kw),
                    "z_shield_max_kw": float(wr.z_shield_max),
                    "risk0": bool(wr.risk0),
                })
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(OUT / "top_peak_event_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(soc_rows).to_csv(OUT / "soc_behavior_around_peak_events.csv", index=False, encoding="utf-8-sig")

    std_prob = pd.read_parquet(STD_OUT / "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5_hourly.parquet")
    std_prob["timestamp"] = pd.to_datetime(std_prob["timestamp"])
    ccor_prob = results["CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5"][0].copy()
    std_prob["demand_kw"] = KAPPA * std_prob["p_grid_realized"]
    ccor_prob["demand_kw"] = KAPPA * ccor_prob["p_grid_realized"]
    june_std = std_prob[std_prob["month_id"].eq(6)]
    june_cc = ccor_prob[ccor_prob["month_id"].eq(6)]
    june_peak_ts = pd.Timestamp(june_cc.loc[june_cc["demand_kw"].idxmax(), "timestamp"])
    win_std = june_std[(june_std["timestamp"] >= june_peak_ts - pd.Timedelta(hours=6)) & (june_std["timestamp"] <= june_peak_ts + pd.Timedelta(hours=6))].copy()
    win_cc = june_cc[(june_cc["timestamp"] >= june_peak_ts - pd.Timedelta(hours=6)) & (june_cc["timestamp"] <= june_peak_ts + pd.Timedelta(hours=6))].copy()
    cols_std = ["timestamp", "demand_kw", "p_grid_realized", "p_ch_exec", "p_dis_exec", "E_soc"]
    std_small = win_std[cols_std].rename(columns={c: f"standard_mpc_{c}" for c in cols_std if c != "timestamp"})
    cc_cols = ["timestamp", "demand_kw", "p_grid_realized", "p_ch_exec", "p_dis_exec", "E_soc_before", "E_soc", "D_guard_kw", "D_cert_kw", "risk0", "z_shield_max", "first_step_netload_max_kw"]
    cc_small = win_cc[cc_cols].rename(columns={c: f"ccor_v2_{c}" for c in cc_cols if c != "timestamp"})
    june = std_small.merge(cc_small, on="timestamp", how="outer").sort_values("timestamp")
    june["event_timestamp"] = june_peak_ts
    june["relative_hour"] = ((june["timestamp"] - june_peak_ts) / pd.Timedelta(hours=1)).astype(int)
    june.to_csv(OUT / "june_peak_event_audit.csv", index=False, encoding="utf-8-sig")


def benchmark_comparison(annual: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(STD_OUT / "annual_cost_summary_realized_basis.csv")
    all_rows = pd.concat([base, annual], ignore_index=True)
    idx = {r.case: r for r in all_rows.itertuples(index=False)}
    pairs = [
        ("MPC_DET_PVFOCUS_BASE", "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE", "MPC_DET vs CCOR_V2_DET"),
        ("MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5", "MPC_PROB vs CCOR_V2_PROB"),
        ("DA_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5", "DA_PROB vs CCOR_V2_PROB"),
        ("CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE", "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5", "CCOR_V2_DET vs CCOR_V2_PROB"),
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
    out.to_csv(OUT / "comparison_vs_benchmarks.csv", index=False, encoding="utf-8-sig")
    return out


def final_status(annual: pd.DataFrame, comp: pd.DataFrame) -> str:
    if (annual["infeasible_steps"] > 0).any() or (annual["fallback_steps"] > 0).any():
        return "NEEDS_MANUAL_REVIEW"
    p = comp[comp["comparison"].eq("MPC_PROB vs CCOR_V2_PROB")]
    d = comp[comp["comparison"].eq("DA_PROB vs CCOR_V2_PROB")]
    if p.empty:
        return "NEEDS_MANUAL_REVIEW"
    pr = p.iloc[0]
    if pr["delta_full_total_m_ntd_b_minus_a"] < 0 and pr["delta_OC_DCT_m_ntd_b_minus_a"] < 0:
        if not d.empty and d.iloc[0]["delta_full_total_m_ntd_b_minus_a"] > 0:
            return "CCOR_V2_NOT_BETTER_THAN_DA_PROB"
        return "CCOR_V2_FULL_YEAR_SUCCESS"
    if pr["delta_OC_DCT_m_ntd_b_minus_a"] < 0 or pr["delta_monthly_peak_kw_b_minus_a"] < 0:
        return "CCOR_V2_RISK_COST_TRADEOFF"
    if pr["delta_monthly_peak_kw_b_minus_a"] > 0:
        return "NEEDS_M200_OR_MULTISTEP_SHIELD"
    return "CCOR_V2_NOT_BETTER_THAN_STANDARD_MPC"


def write_report(annual, monthly, comp, scen_audit, truth_audit, status) -> None:
    june = pd.read_csv(OUT / "june_peak_event_audit.csv")
    write_md(
        OUT / "CCOR_V2_FULL_YEAR_M150_FINAL_REPORT.md",
        "CCOR-v2 Full-Year m150 PV-Focused Report",
        [
            ("Final Status", f"`{status}`"),
            ("Scope", "Only `CCOR_V2_SHIELD_ONLY_m150` was run for DET and PROB. No other ablations, structured-high robustness, K10 sensitivity, LOAD-QN, random load error, old S1-S6, or old invalid PV truth were used."),
            ("Truth and Scenario Inputs", f"Replay/KPI uses `{REPLAY_TRUTH_PACKAGE_PATH.relative_to(REPO)}` (`{PV_TRUTH_SOURCE_LABEL}`). PROB uses `{PROB_SCENARIO.relative_to(REPO)}`."),
            ("Annual Cost", annual.to_markdown(index=False)),
            ("Monthly Cost", monthly.to_markdown(index=False)),
            ("Benchmark Comparisons", comp.to_markdown(index=False)),
            ("Scenario Usage Audit", scen_audit.to_markdown(index=False)),
            ("Replay Truth Audit", truth_audit.to_markdown(index=False)),
            ("June Peak Event Audit", june.to_markdown(index=False)),
            ("Interpretation", "The m150 first-step shield is intended to prevent executable first-step charging from creating new monthly DCT peaks. The June audit identifies whether any remaining peak is caused by charging, insufficient discharge, SOC shortage, or unavoidable load/PV condition."),
        ],
    )


def main() -> int:
    ensure_out()
    results = {
        "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE": run_full_year("CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE", prob=False),
        "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5": run_full_year("CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5", prob=True),
    }
    annual, monthly = collect(results)
    write_mechanism_audits(results)
    scen_audit, truth_audit = usage_audits()
    comp = benchmark_comparison(annual)
    status = final_status(annual, comp)
    write_report(annual, monthly, comp, scen_audit, truth_audit, status)
    print(f"[CCOR v2 full-year m150] {status} under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
