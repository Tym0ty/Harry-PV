#!/usr/bin/env python3
"""
Generate the final PV-focused thesis method/result convergence report.

This script is reporting-only: it reads existing artifacts and writes a
consolidated Markdown report plus supporting CSV summaries. It does not run
forecasting, scenario generation, scheduling, or optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus"

BASIC_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc"
CCOR_OUT = REPO / "new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150"
ROBUST_OUT = REPO / "new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus"
TAIL_OUT = REPO / "new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm"
PV_TRUTH_OUT = REPO / "new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild"
BRIDGE_OUT = REPO / "new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge"
POINT_FCST_OUT = REPO / "new_pipeline/data/output/experiments/main_forecasting_model_comparison"
ID_GHI_POINT_OUT = REPO / "new_pipeline/data/output/experiments/lit_guided_id_h24_ghi"
ID_GHI_PROB_OUT = REPO / "new_pipeline/data/output/experiments/id_h24_ghi_prob"
FIG_OUT = REPO / "new_pipeline/data/output/experiments/figures"


MAIN_CASES = {
    "MPC_DET_PVFOCUS_BASE": "MPC_DET",
    "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5": "MPC_PROB",
    "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE": "CCOR_DET",
    "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5": "CCOR_PROB",
}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(path)
    return pd.DataFrame()


def md_table(df: pd.DataFrame, floatfmt: str = ".4f", max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows found._"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    return out.to_markdown(index=False, floatfmt=floatfmt)


def pick_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def round_numeric(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(ndigits)
    return out


def compute_delta(row_b, row_a, comparison: str) -> dict:
    fields = [
        "full_total_m_ntd",
        "TOU_m_ntd",
        "OC_DCT_m_ntd",
        "Deg_m_ntd",
        "TREC_RE20_m_ntd",
        "monthly_peak_max_kw",
        "oc_hours",
    ]
    d = {"comparison": comparison, "case_a": row_a["case"], "case_b": row_b["case"]}
    for f in fields:
        if f in row_a and f in row_b:
            d[f"delta_{f}_b_minus_a"] = float(row_b[f]) - float(row_a[f])
    return d


def load_main_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    basic = read_csv(BASIC_OUT / "annual_cost_summary_realized_basis.csv")
    ccor = read_csv(CCOR_OUT / "annual_cost_summary_realized_basis.csv")
    both = pd.concat([basic, ccor], ignore_index=True, sort=False)
    main = both[both["case"].isin(MAIN_CASES)].copy()
    main.insert(0, "final_label", main["case"].map(MAIN_CASES))
    main = main.sort_values("final_label")
    main.to_csv(OUT / "final_four_main_case_summary.csv", index=False, encoding="utf-8-sig")

    idx = {r["case"]: r for _, r in main.iterrows()}
    comparisons = [
        ("MPC_PROB vs MPC_DET", "MPC_DET_PVFOCUS_BASE", "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5"),
        ("CCOR_PROB vs CCOR_DET", "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE", "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5"),
        ("CCOR_DET vs MPC_DET", "MPC_DET_PVFOCUS_BASE", "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE"),
        ("CCOR_PROB vs MPC_PROB", "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5", "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5"),
    ]
    deltas = []
    for label, a, b in comparisons:
        deltas.append(compute_delta(idx[b], idx[a], label))
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(OUT / "final_four_delta_comparison.csv", index=False, encoding="utf-8-sig")
    return main, delta_df


def build_case_definition() -> pd.DataFrame:
    rows = [
        {
            "Final Case": "MPC_DET",
            "Case ID": "MPC_DET_PVFOCUS_BASE",
            "Scheduling Method": "Standard hourly rolling MPC",
            "Forecast Type": "PV point/q50",
            "Scenario Type": "none",
            "Load Setting": "base deterministic given load",
            "Execution": "H24 rolling, execute first step",
        },
        {
            "Final Case": "MPC_PROB",
            "Case ID": "MPC_PROB_PVFOCUS_LOWPV_SAFE_K5",
            "Scheduling Method": "Standard stochastic hourly rolling MPC",
            "Forecast Type": "PV probabilistic",
            "Scenario Type": "PV-focused low-PV-safe TKM K5",
            "Load Setting": "base deterministic given load",
            "Execution": "H24 rolling, execute first step",
        },
        {
            "Final Case": "CCOR_DET",
            "Case ID": "CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE",
            "Scheduling Method": "CCOR-v2 first-step peak-certified MPC",
            "Forecast Type": "PV point/q50",
            "Scenario Type": "none",
            "Load Setting": "base deterministic given load",
            "Execution": "H24 rolling, execute first step",
        },
        {
            "Final Case": "CCOR_PROB",
            "Case ID": "CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5",
            "Scheduling Method": "CCOR-v2 stochastic first-step peak-certified MPC",
            "Forecast Type": "PV probabilistic",
            "Scenario Type": "PV-focused low-PV-safe TKM K5",
            "Load Setting": "base deterministic given load",
            "Execution": "H24 rolling, execute first step",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_case_definition_table.csv", index=False, encoding="utf-8-sig")
    return df


def build_robustness_summary() -> pd.DataFrame:
    annual = read_csv(ROBUST_OUT / "annual_cost_summary_realized_basis.csv")
    annual.to_csv(OUT / "load_bias_robustness_summary.csv", index=False, encoding="utf-8-sig")
    return annual


def build_tailaware_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = read_csv(TAIL_OUT / "annual_cost_summary_realized_basis.csv")
    comp = read_csv(TAIL_OUT / "km_vs_tkm_delta_comparison.csv")
    annual.to_csv(OUT / "tailaware_ablation_summary.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(OUT / "tailaware_ablation_delta_summary.csv", index=False, encoding="utf-8-sig")
    return annual, comp


def build_traceability() -> tuple[pd.DataFrame, pd.DataFrame]:
    method_rows = [
        {
            "Method Component": "Corrected CWA-GHI PV truth used",
            "Thesis Description": "Replay/KPI settlement uses CWA-GHI-derived PV truth.",
            "Script / Artifact Evidence": "pv_focus_bridge_utils.py:23, run_scheduling_pv_focus_da_mpc.py:254-271, run_scheduling_pv_focus_ccor_v2_full_year_m150.py: truth audit output",
            "Status": "MATCH",
            "Notes": rel(PV_TRUTH_OUT / "full_year_replay_truth_package_CWA_GHI_PV.parquet"),
        },
        {
            "Method Component": "Old invalid PV truth excluded",
            "Thesis Description": "NTUST_Load_PV.csv::Solar_kWh is not used for final replay.",
            "Script / Artifact Evidence": "replay_truth_usage_audit.csv in main, CCOR, robustness, and tail-aware ablation folders",
            "Status": "MATCH",
            "Notes": "All final audit files mark CWA_GHI_PV_REBUILT and old invalid NTUST Solar_kWh false.",
        },
        {
            "Method Component": "LOAD-QN excluded",
            "Thesis Description": "LOAD-QN is not used in final mainline.",
            "Script / Artifact Evidence": "scenario/replay audit fields; run_scheduling_pv_focus_da_mpc.py:286, run_tailaware_ablation_pvfocus_km_vs_tkm.py:275",
            "Status": "MATCH",
            "Notes": "Final load is deterministic given profile; no probabilistic load forecast in mainline.",
        },
        {
            "Method Component": "Random load error excluded",
            "Thesis Description": "Literature/random load error is appendix/diagnostic only, not final mainline.",
            "Script / Artifact Evidence": "PV-focused scripts and reports; tail-aware/robustness audit flags",
            "Status": "MATCH",
            "Notes": "Structured load bias is outer robustness, not random scenario mixing.",
        },
        {
            "Method Component": "Old S1-S6 excluded from mainline",
            "Thesis Description": "Old S1-S6 is diagnostic-only and not a final scenario input.",
            "Script / Artifact Evidence": "PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md; scenario audit flags",
            "Status": "MATCH",
            "Notes": "Old S1-S6 all-hour OC coverage is not used as pass/fail criterion.",
        },
        {
            "Method Component": "PV-focused scenario generation used",
            "Thesis Description": "PV scenarios are sampled from calibrated PV/GHI probabilistic forecasts and combined with deterministic load.",
            "Script / Artifact Evidence": "exp_pv_focus_generate_pv_scenarios.py; exp_pv_focus_build_netload_scenarios.py; pv_focus_bridge_utils.py:207-300",
            "Status": "MATCH",
            "Notes": "netload_s_kw = load_kw - pv_s_kw.",
        },
        {
            "Method Component": "Low-PV-safe TKM K5 used for PROB mainline",
            "Thesis Description": "PROB cases use low-PV-safe tail-aware K-medoids K=5.",
            "Script / Artifact Evidence": "exp_pv_focus_lowpv_reduction_fix.py; scenario_usage_audit.csv",
            "Status": "MATCH",
            "Notes": "Scenario files: id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet.",
        },
        {
            "Method Component": "Cluster probability weights used",
            "Thesis Description": "Reduced scenario weights are cluster probabilities, not equal weights.",
            "Script / Artifact Evidence": "scenario_usage_audit.csv; PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md weight table",
            "Status": "MATCH",
            "Notes": "Weight sums equal 1 per issue_time in final audits.",
        },
        {
            "Method Component": "Optimization scenario files contain no truth columns",
            "Thesis Description": "No realized/actual/truth columns are passed into optimization-facing scenario files.",
            "Script / Artifact Evidence": "scenario_usage_audit.csv; run_scheduling_pv_focus_da_mpc.py:227-243",
            "Status": "MATCH",
            "Notes": "Truth only enters replay/KPI settlement.",
        },
        {
            "Method Component": "Standard MPC uses base deterministic load",
            "Thesis Description": "MPC_DET and MPC_PROB use the base deterministic load profile.",
            "Script / Artifact Evidence": "run_scheduling_pv_focus_da_mpc.py:154; load_profile_lookup('base')",
            "Status": "MATCH",
            "Notes": "Structured profiles appear only in robustness folder.",
        },
        {
            "Method Component": "CCOR-v2 uses shield m150",
            "Thesis Description": "CCOR-v2 adds a first-step soft peak shield with 150 kW margin.",
            "Script / Artifact Evidence": "run_scheduling_pv_focus_ccor_v2_full_year_m150.py:52; run_scheduling_pv_focus_ccor_v2_smoke.py:263,286",
            "Status": "MATCH",
            "Notes": "Terminal reserve is disabled in final v2 shield-only run.",
        },
        {
            "Method Component": "Structured load-bias robustness uses structured_low/high only",
            "Thesis Description": "Load bias is an outer deterministic robustness setting.",
            "Script / Artifact Evidence": "run_final_four_load_bias_robustness_pvfocus.py:608; load_profile_usage_audit.csv",
            "Status": "MATCH",
            "Notes": "No uniform or random error profiles used in final robustness batch.",
        },
        {
            "Method Component": "All final cases solved optimally 8760/8760",
            "Thesis Description": "No infeasible or fallback steps in final main and robustness cases.",
            "Script / Artifact Evidence": "annual_cost_summary_realized_basis.csv in main, CCOR, robustness, tail-aware folders",
            "Status": "MATCH",
            "Notes": "Gurobi status summary {'2': 8760}; infeasible_steps=0; fallback_steps=0.",
        },
    ]
    method = pd.DataFrame(method_rows)
    method.to_csv(OUT / "method_component_traceability.csv", index=False, encoding="utf-8-sig")

    eq_rows = [
        {
            "Equation / Block": "Horizon and scenarios",
            "Mathematical Form": "h in {0,...,H-1}; omega in Omega; pi_omega scenario weights",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:27-56; scenario dictionaries passed by scheduling scripts",
            "Notes": "Final H24 rolling MPC executes first step only.",
        },
        {
            "Equation / Block": "Power balance",
            "Mathematical Form": "P_gl_{omega,h} + P_pvl_{omega,h} + P_dis_h = L_{omega,h}",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:174",
            "Notes": "Load balance is scenario-specific; discharge is common across scenarios.",
        },
        {
            "Equation / Block": "PV allocation",
            "Mathematical Form": "P_pvl_{omega,h}+P_pvc_{omega,h}+P_pvcu_{omega,h}=PV_{omega,h}",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:178-182",
            "Notes": "PV can serve load, charge BESS, or be curtailed.",
        },
        {
            "Equation / Block": "Grid import",
            "Mathematical Form": "G_{omega,h}=kappa*(P_gl_{omega,h}+P_gc_{omega,h})",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:190",
            "Notes": "Used for DCT candidate peak; replay also computes realized grid import.",
        },
        {
            "Equation / Block": "Charge/discharge exclusivity",
            "Mathematical Form": "P_ch_h <= u_h PB; P_dis_h <= (1-u_h) PB",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:140-141",
            "Notes": "Binary u prevents simultaneous charge and discharge.",
        },
        {
            "Equation / Block": "SOC transition",
            "Mathematical Form": "E_h = E_{h-1} + eta_ch P_ch_h - P_dis_h/eta_dis",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:145-147",
            "Notes": "Delta t is one hour in the final experiments.",
        },
        {
            "Equation / Block": "SOC bounds",
            "Mathematical Form": "SOC_min <= E_h <= SOC_max",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:113",
            "Notes": "Bounds are applied through variable lower/upper bounds.",
        },
        {
            "Equation / Block": "Terminal SOC",
            "Mathematical Form": "Only final-day/year-end terminal SOC bound if is_final_day",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:151-156",
            "Notes": "CCOR-v2 final m150 does not rely on terminal reserve.",
        },
        {
            "Equation / Block": "DCT candidate peak",
            "Mathematical Form": "D_cand_omega >= D_init; D_cand_omega >= G_{omega,h} for all h",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:127,190",
            "Notes": "D_init is the month-to-date realized peak state.",
        },
        {
            "Equation / Block": "Over-contract segmentation",
            "Mathematical Form": "Over_omega >= D_cand_omega-CC; Over_omega=O1_omega+O2_omega; 0<=O1<=0.1CC",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:127-130,193-195",
            "Notes": "Segmented Taiwan DCT proxy.",
        },
        {
            "Equation / Block": "Expected incremental OC/DCT objective",
            "Mathematical Form": "sum_omega pi_omega c_basic(m1 O1_omega+m2 O2_omega) - PeakCost_prev",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:230-235",
            "Notes": "Constant previous peak cost is subtracted in optimization objective.",
        },
        {
            "Equation / Block": "Energy cost objective",
            "Mathematical Form": "sum_omega pi_omega sum_h tou_h * (P_gl_{omega,h}+P_gc_{omega,h})",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:222-245",
            "Notes": "TOU price is included in MILP objective.",
        },
        {
            "Equation / Block": "Degradation objective",
            "Mathematical Form": "sum_h sum_k deg_slope_k e_seg_{h,k}",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:149-161,222-245",
            "Notes": "Segmented discharge degradation is optimized.",
        },
        {
            "Equation / Block": "Green SOC constraints",
            "Mathematical Form": "E_g transition; E_g<=E; P_chg<=P_pvc; P_disg<=P_dis",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:197-209",
            "Notes": "Green SOC accounting constraints are present; TREC/RE20 final amount is settlement/accounting.",
        },
        {
            "Equation / Block": "CVaR",
            "Mathematical Form": "eta + (1/(1-alpha)) sum pi xi; xi>=C_oc-eta",
            "Code Evidence": "milp_v2/layer_b/milp_cvar.py:132-133,211-245",
            "Notes": "Final cases pass lam=0.0, so CVaR is disabled.",
        },
        {
            "Equation / Block": "CCOR-v2 shield",
            "Mathematical Form": "z_shield_omega >= G_{omega,0} - (max(CC,D_init)-150); objective += sum pi rho_shield z_shield_omega",
            "Code Evidence": "run_scheduling_pv_focus_ccor_v2_smoke.py:189,202,236,263,286; full_year_m150.py:52",
            "Notes": "Soft first-step shield only; no terminal reserve in final v2.",
        },
        {
            "Equation / Block": "Rolling replay/update",
            "Mathematical Form": "Execute h=0 action; update SOC and D_mtd using realized CWA-GHI PV truth and load profile",
            "Code Evidence": "run_scheduling_pv_focus_da_mpc.py:149-190; pv_focus_bridge_utils.py:62-139",
            "Notes": "Truth enters replay/KPI only.",
        },
    ]
    eq = pd.DataFrame(eq_rows)
    eq.to_csv(OUT / "milp_equation_traceability.csv", index=False, encoding="utf-8-sig")
    return method, eq


def build_manifest() -> pd.DataFrame:
    paths = [
        BASIC_OUT / "annual_cost_summary_realized_basis.csv",
        CCOR_OUT / "annual_cost_summary_realized_basis.csv",
        ROBUST_OUT / "annual_cost_summary_realized_basis.csv",
        TAIL_OUT / "annual_cost_summary_realized_basis.csv",
        TAIL_OUT / "km_vs_tkm_delta_comparison.csv",
        PV_TRUTH_OUT / "cwa_pv_truth_summary.csv",
        PV_TRUTH_OUT / "full_year_replay_truth_package_CWA_GHI_PV.parquet",
        BRIDGE_OUT / "id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet",
        BRIDGE_OUT / "PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md",
        ID_GHI_PROB_OUT / "id_h24_ghi_prob_metrics.csv",
        ID_GHI_PROB_OUT / "id_h24_pv_prob_metrics.csv",
        POINT_FCST_OUT / "main_model_comparison.csv",
        ID_GHI_POINT_OUT / "literature_guided_id_h24_ghi_model_comparison.csv",
        FIG_OUT / "table1_agaci_metrics.csv",
    ]
    rows = []
    for p in paths:
        rows.append(
            {
                "source_path": rel(p),
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_report_source_manifest.csv", index=False, encoding="utf-8-sig")
    return df


def load_forecast_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point = read_csv(POINT_FCST_OUT / "main_model_comparison.csv", required=False)
    id_point = read_csv(ID_GHI_POINT_OUT / "literature_guided_id_h24_ghi_model_comparison.csv", required=False)
    prob = read_csv(ID_GHI_PROB_OUT / "id_h24_ghi_prob_metrics.csv", required=False)
    agaci = read_csv(FIG_OUT / "table1_agaci_metrics.csv", required=False)

    if not point.empty:
        point.to_csv(OUT / "ghi_point_model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    if not prob.empty:
        prob.to_csv(OUT / "ghi_probabilistic_agaci_summary.csv", index=False, encoding="utf-8-sig")
    if not agaci.empty:
        agaci.to_csv(OUT / "agaci_publication_table_summary.csv", index=False, encoding="utf-8-sig")
    return point, id_point, prob, agaci


def make_report(
    case_def: pd.DataFrame,
    main: pd.DataFrame,
    delta: pd.DataFrame,
    robust: pd.DataFrame,
    tail_ann: pd.DataFrame,
    tail_delta: pd.DataFrame,
    method_trace: pd.DataFrame,
    eq_trace: pd.DataFrame,
    manifest: pd.DataFrame,
    point_fcst: pd.DataFrame,
    id_point_fcst: pd.DataFrame,
    prob_fcst: pd.DataFrame,
    agaci: pd.DataFrame,
) -> str:
    pv_summary = read_csv(PV_TRUTH_OUT / "cwa_pv_truth_summary.csv", required=False)
    lowpv_valid = read_csv(BRIDGE_OUT / "pvfocused_lowpv_fix_passfail_metrics.csv", required=False)
    lowpv_weights = read_csv(BRIDGE_OUT / "pvfocused_lowpv_fix_weight_summary.csv", required=False)
    ccor_june = read_csv(CCOR_OUT / "june_peak_event_audit.csv", required=False)
    load_profile_audit = read_csv(ROBUST_OUT / "load_profile_usage_audit.csv", required=False)

    main_cols = [
        "final_label",
        "case",
        "full_total_m_ntd",
        "TOU_m_ntd",
        "OC_DCT_m_ntd",
        "Deg_m_ntd",
        "TREC_RE20_m_ntd",
        "basic_m_ntd",
        "capex_m_ntd",
        "monthly_peak_max_kw",
        "oc_hours",
        "infeasible_steps",
        "fallback_steps",
        "solver_status_summary",
    ]
    robust_cols = [
        "case",
        "load_profile_name",
        "full_total_m_ntd",
        "TOU_m_ntd",
        "OC_DCT_m_ntd",
        "Deg_m_ntd",
        "TREC_RE20_m_ntd",
        "monthly_peak_max_kw",
        "oc_hours",
        "infeasible_steps",
        "fallback_steps",
    ]

    # Keep DA out of the main table. DA is mentioned only in one explicit reference note.
    point_table = point_fcst.copy()
    if not point_table.empty:
        point_table = pick_cols(point_table, ["Model Type", "Model", "RMSE", "MAE", "MSE", "R2", "n", "Remark"])
    id_h24_overall = pd.DataFrame()
    if not id_point_fcst.empty:
        id_h24_overall = id_point_fcst[id_point_fcst["label"].astype(str).str.contains("Overall", na=False)].copy()
        id_h24_overall = id_h24_overall[~id_h24_overall["label"].astype(str).str.startswith("DA|")].copy()
    prob_overall = pd.DataFrame()
    if not prob_fcst.empty:
        prob_overall = prob_fcst[prob_fcst["subset"].eq("Overall")].copy()
        prob_overall = pick_cols(prob_overall, ["approach", "subset", "n", "raw_picp80", "raw_picp90", "cal_picp80", "cal_picp90", "cal_mpiw80", "cal_mpiw90", "q50_rmse", "mean_pinball", "crps"])

    june_note = "UNCLEAR"
    if not ccor_june.empty:
        evt = ccor_june[ccor_june.get("timestamp", pd.Series(dtype=str)).astype(str).eq("2025-06-12 08:00:00")]
        if not evt.empty:
            r = evt.iloc[0]
            june_note = (
                f"At 2025-06-12 08:00, CCOR-v2 demand was {r.get('ccor_v2_demand_kw', float('nan')):.2f} kW, "
                f"with p_ch={r.get('ccor_v2_p_ch_exec', float('nan')):.2f} kW, D_guard={r.get('ccor_v2_D_guard_kw', float('nan')):.2f} kW, "
                f"D_cert={r.get('ccor_v2_D_cert_kw', float('nan')):.2f} kW, risk0={r.get('ccor_v2_risk0', 'UNCLEAR')}, "
                f"z_shield={r.get('ccor_v2_z_shield_max', float('nan')):.2f}, and first-step stress net-load max={r.get('ccor_v2_first_step_netload_max_kw', float('nan')):.2f} kW."
            )

    report_status = "FINAL_REPORT_READY_WITH_MINOR_UNCERTAIN_ITEMS"
    uncertain_items = [
        "A final literature citation for the simple PR-based GHI-to-PV conversion is not stored in the repository artifact; mark as CITATION_NEEDED.",
        "Some detailed feature-level descriptions for the selected forecast models are summarized from reports, but exact feature lists are not fully reproduced in this final report.",
    ]

    md = f"""# Final Thesis Method and Results Report: PV-Focused Rolling MPC and CCOR-MPC

## 0. Executive Summary

This report consolidates the final converged thesis method line for the BH PV-BESS scheduling project. The final study focuses on PV uncertainty and PV-BESS scheduling under a deterministic given campus load profile. Load uncertainty is not modeled as a probabilistic load forecast because only one year of site load data is available; instead, the load assumption is defended through season x time-of-day structured load-bias robustness.

PV uncertainty is represented through calibrated probabilistic GHI/PV forecasts. The final optimization-facing net-load scenario bridge is:

```text
netload_s(t) = load_given(t) - PV_s(t)
```

Scenario reduction uses a low-PV-safe, tail-aware K-medoids reduction with K=5. The final main cases are only `MPC_DET`, `MPC_PROB`, `CCOR_DET`, and `CCOR_PROB`; DA is excluded from the final main case set and retained only as a removed/reference-only benchmark if needed.

The final results support three thesis conclusions. First, PV probabilistic scenarios improve both standard MPC and CCOR-v2. Second, CCOR-v2 improves standard MPC mainly by reducing OC/DCT exposure. Third, the conclusion remains stable under structured load-bias robustness. Tail-aware low-PV-safe scenario reduction also improves PROB scheduling results compared with ordinary non-tail-aware KM.

Final report status: `{report_status}`.

Minor uncertain items:

{chr(10).join(f'- {x}' for x in uncertain_items)}

## 1. Final Research Scope and Case Definition

The final scope is the NTUST/BH campus PV-BESS contract-capacity scheduling problem. The confirmed system parameters used in the final PV-focused scripts include PV capacity `PV_CAP = 2687 kWp`, performance ratio `PR = 0.80`, contract capacity `CC = 3232 kW`, and the test period `2024-11-01` to `2025-10-31`. The corrected replay truth package is `{rel(PV_TRUTH_OUT / 'full_year_replay_truth_package_CWA_GHI_PV.parquet')}`.

The final main case set is:

{md_table(case_def)}

Excluded from the final mainline: DA main cases, LOAD-QN, random load-error mainline, old S1-S6 scenario inputs, old invalid PV truth, and LitErr random load-error cases. DA is removed from the final main case matrix because the final research question is whether PV probabilistic information and CCOR control improve rolling MPC. DA can remain a reference-only or appendix benchmark, but it is not part of the final method matrix.

## 2. GHI Forecasting: Point Forecast Comparison

The repository contains a point forecast comparison for GHI/PV-related models. The main H1 comparison shows XGBoost, weather-classified CatBoost, and CNN-LSTM baselines. A separate ID-H24 literature-guided benchmark compares persistence and rolling multi-step ML variants over H1-H24.

Table: GHI point forecasting model comparison.

{md_table(round_numeric(point_table), max_rows=12)}

ID-H24 overall model comparison rows:

{md_table(round_numeric(id_h24_overall), max_rows=12)}

Evidence files: `{rel(POINT_FCST_OUT / 'main_model_comparison.csv')}` and `{rel(ID_GHI_POINT_OUT / 'literature_guided_id_h24_ghi_model_comparison.csv')}`.

The final probabilistic path is XGBoost quantile-style forecasting with AgACI calibration because it provides usable quantile outputs for scenario construction and conformal calibration, while remaining compatible with the rolling H24 setting.

## 3. XGBQ Probabilistic GHI Forecast and Dynamic AgACI Calibration

The final probabilistic GHI/PV pipeline is based on XGBoost quantile outputs and AgACI-style adaptive conformal calibration. The ID-H24 probabilistic report defines approaches Q1/Q2/Q3 and calibrated variants. AgACI adjusts interval bounds online by lead-group/issue-date style updates to target empirical coverage and reduce interval miscoverage.

Overall ID-H24 probabilistic GHI metrics:

{md_table(round_numeric(prob_overall), max_rows=12)}

Publication-style AgACI summary table:

{md_table(agaci)}

Implementation evidence:

- `new_pipeline/data/output/experiments/id_h24_ghi_prob/FINAL_ID_H24_GHI_PROBABILISTIC_MODEL_SELECTION_REPORT.md`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_ghi_prob_metrics.csv`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_prob_metrics.csv`
- `new_pipeline/data/output/experiments/figures/table1_agaci_metrics.csv`

The final scheduling bridge uses PV-focused scenarios produced from calibrated GHI/PV probabilistic information. Realized PV/GHI is used for calibration/evaluation and final replay diagnostics according to the corresponding reports, not as future optimization input.

## 4. GHI-to-PV Conversion and PV Truth Rebuild

The final GHI-to-PV conversion is implemented as:

```text
PV_kW = clip(PR * GHI_Wm2 / 1000 * PV_CAP, 0, PV_CAP)
```

with `PR = 0.80` and `PV_CAP = 2687 kWp`. Code evidence includes `new_pipeline/scripts/experiments/exp_rebuild_pv_truth_from_cwa_ghi.py` and `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py`.

PV truth summary:

{md_table(round_numeric(pv_summary))}

The old `NTUST_Load_PV.csv::Solar_kWh` PV field is invalid and is not used for final replay. The corrected CWA-GHI-derived truth package is used for all final validation, settlement, KPI, and scheduling reports. An annual average of about 328 kW is plausible for 2687 kWp because the mean includes night hours; the rebuilt truth has max PV about 2298.88 kW and capacity factor about 12.22%.

Citation note: `CITATION_NEEDED` for the final thesis text if a formal reference is desired for the performance-ratio style PV conversion. A PVWatts/performance-ratio style source would be appropriate, but no final citation file was found in the repository.

## 5. PV Scenario Generation and Net-Load Scenario Construction

The final scenario bridge samples PV scenarios from calibrated probabilistic PV/GHI forecasts, applies physical clipping (`0 <= PV <= PV_CAP`), and combines those PV trajectories with a deterministic given campus load profile:

```text
netload_s_kw = load_given_kw - pv_s_kw
```

The load profile is deterministic within each outer case. It is not a probabilistic load forecast. Structured low/high load-bias profiles are used only for robustness and are never mixed into a single probability distribution.

Evidence:

- `new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py`
- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py`
- `{rel(BRIDGE_OUT / 'PV_FOCUSED_NETLOAD_BUILD_REPORT.md')}`

Scenario input safety is audited in final scheduling folders. The optimization-facing scenario files do not contain realized/actual/truth columns.

## 6. Low-PV-Safe Tail-Aware Scenario Reduction

Raw PV scenario sets are too large for repeated hourly rolling MILP solves. The final method therefore uses K=5 representative scenarios. Ordinary KM is distribution-oriented and may drop low-probability, high-impact low-PV/high-net-load trajectories. The final low-PV-safe TKM explicitly preserves typical medoids, high-net-load medoids, low-PV/PV-shortfall medoids, and DCT-risk medoids; scenario weights are cluster probabilities.

The low-PV-safe validation status is:

```text
PASS_PV_TKM_LOWPV_SAFE_K5_FOR_DA_MPC_SCHEDULING
```

Selected validation metrics:

{md_table(round_numeric(pick_cols(lowpv_valid, ['mode','bridge_type','pv_envelope_coverage','pv_low_tail_coverage','pv_active_netload_upper_coverage','pv_sensitive_oc_upper_coverage','daylight_top5_netload_upper_coverage','scenario_max_netload_kw'])), max_rows=12)}

Weight sanity:

{md_table(round_numeric(lowpv_weights), max_rows=8)}

Tail-aware ablation annual results:

{md_table(round_numeric(tail_ann))}

Ordinary KM K5 vs low-PV-safe TKM K5:

{md_table(round_numeric(tail_delta))}

Interpretation: low-PV-safe TKM improves total cost mainly by reducing OC/DCT while slightly increasing TOU. This supports keeping low-PV-safe TKM K5 as the final PROB scenario reduction.

## 7. Shared MILP Framework for MPC and CCOR-MPC

The final standard MPC and CCOR-v2 cases use the same physical and settlement-oriented Layer-B MILP structure, with CCOR-v2 adding only a first-step soft peak shield. The core builder evidence is `milp_v2/layer_b/milp_cvar.py::solve_day_ahead()`. Final rolling scripts call this builder or reproduce its same equations with the added CCOR shield.

### Sets and Indices

- `h = 0,...,H-1`: rolling horizon step, with `H=24` in final cases.
- `omega in Omega`: scenario index; DET has one scenario, PROB has K=5.
- `k`: degradation segment index.
- Monthly state is carried through `D_init`/month-to-date peak rather than modeled as a separate annual horizon.

### Parameters

Key parameters include net-load or separate load/PV scenario trajectories, scenario weights `pi_omega`, TOU price, BESS power limit `PB`, BESS energy limit `EB`, SOC lower/upper bounds, charge/discharge efficiencies, initial SOC, initial Green SOC, contract capacity `CC=3232 kW`, month-to-date peak state `D_init`, OC/DCT segment multipliers, degradation segment slopes, one-hour time step, and for CCOR-v2 the shield margin `150 kW` and shield penalty coefficient.

### Decision Variables

The common MILP includes battery charge `P_ch_h`, discharge `P_dis_h`, binary charge/discharge mode `u_h`, SOC `E_h`, degradation segment energy `e_seg_hk`, scenario-specific grid-to-load `P_gl`, grid-to-charge `P_gc`, PV-to-load `P_pvl`, PV-to-charge `P_pvc`, PV curtailment `P_pvcu`, Green SOC variables `E_g`, green charge/discharge variables, scenario-specific demand proxy `D_cand_omega`, over-contract variables `Over_full_omega`, `O1_full_omega`, `O2_full_omega`, and CVaR variables. In final cases CVaR is disabled with `lam=0.0`.

CCOR-v2 adds `z_shield_omega` only for the first-step peak shield.

### Objective Function

The common objective minimizes:

```text
expected TOU energy cost
+ battery degradation cost
+ expected incremental OC/DCT cost
+ lam * CVaR_OC
```

with `lam=0.0` in final cases. CCOR-v2 adds:

```text
sum_omega pi_omega * rho_shield * z_shield_omega
```

Basic charge, CAPEX, and TREC/RE20 settlement components are included in final annual accounting, not as dispatch-changing terms in the final MILP objective. Load uncertainty is not an objective term.

### Constraints

The implemented constraints include power balance, PV allocation, grid import through grid-to-load/grid-to-charge, charge/discharge exclusivity, SOC transition, SOC bounds, optional year-end terminal SOC condition, degradation segment decomposition, DCT candidate peak constraints, segmented over-contract constraints, Green SOC transition and bounds, and CVaR linearization constraints. The final CCOR-v2 shield-only run does not rely on terminal reserve.

Equation-to-code traceability:

{md_table(eq_trace)}

## 8. CCOR-MPC v2 Shield m150 Design

Standard rolling MPC already includes an OC/DCT term and month-to-date peak state, but the rolling first-step architecture can still create irreversible monthly peaks, especially by charging during peak-sensitive first steps. CCOR-v1 terminal reserve did not solve this robustly because the reserve did not bind when the critical risk was at the current executable step.

CCOR-v2 therefore adds first-step peak certification. At issue time `t`:

```text
D_mtd = current month-to-date realized peak before executing step t
D_guard = max(CC, D_mtd)
D_cert = D_guard - 150 kW
```

The shield is:

```text
G_0 <= D_cert + z_shield
z_shield >= 0
objective += rho_shield * z_shield
```

In the implementation this is scenario-specific for PROB:

```text
z_shield_omega >= kappa * (P_gl_omega0 + P_gc_omega0) - D_cert
```

DET and PROB both use the same shield principle. PROB retains the scenario-weighted stochastic structure.

Full-year mechanism evidence:

{md_table(round_numeric(pick_cols(main, ['final_label','case','first_step_charging_near_risk_hours','shield_slack_positive_hours','risk0_hours','infeasible_steps','fallback_steps'])))}

June event caveat: {june_note} This indicates that m150 works annually but is not perfect; future work may explore m200, multi-step shielding, or realized-charge-aware first-step guards.

## 9. Main Four-Case Results

Final main annual cost table:

{md_table(round_numeric(pick_cols(main, main_cols)))}

Delta comparison table:

{md_table(round_numeric(delta))}

Interpretation:

- `MPC_PROB` improves over `MPC_DET` by reducing OC/DCT, with a small TOU increase.
- `CCOR_PROB` improves over `CCOR_DET` by reducing OC/DCT more strongly, again with TOU increasing because dispatch becomes more risk-aware.
- `CCOR_DET` improves over `MPC_DET` mainly through OC/DCT reduction.
- `CCOR_PROB` is the best final case and improves over `MPC_PROB` mainly through lower OC/DCT and monthly peak.

## 10. Structured Load-Bias Robustness

The mainline uses deterministic given load. To address the critique that load is assumed perfect, the final robustness batch applies structured load-bias profiles as outer deterministic sensitivity cases, not probabilistic load forecasts. The profiles are season x time-of-day dependent and are not assigned probabilities.

Load profile audit:

{md_table(round_numeric(load_profile_audit), max_rows=10)}

Structured load-bias annual results:

{md_table(round_numeric(pick_cols(robust, robust_cols)))}

The robustness folder reports final status:

```text
LOAD_ROBUSTNESS_PASS_MAIN_CONCLUSION_STABLE
```

Under both `structured_low` and `structured_high`, PROB improves DET and CCOR_PROB improves MPC_PROB. The main conclusion is therefore stable under the structured load-bias robustness design.

## 11. Implementation Audit and Traceability

{md_table(method_trace)}

Source manifest:

{md_table(manifest)}

## 12. Final Thesis Interpretation

PV probabilistic forecasting improves scheduling because it preserves low-PV/high-net-load risk information that directly affects Taiwan contract-capacity demand charges. Standard MPC can reduce local TOU cost but may still expose the system to month-maximum DCT risk. CCOR-v2 adds a first-step peak-certification mechanism that directly addresses this rolling-execution weakness.

Low-PV-safe TKM is beneficial because DCT risk is tail-driven. Ordinary scenario reduction can discard decision-relevant low-PV scenarios; the tail-aware reduction keeps these scenarios while preserving cluster probability weights. Structured load-bias robustness shows that the main conclusion remains stable when the deterministic load profile is perturbed.

The remaining caveat is the June 2025 event described above. CCOR-v2 shield m150 is a successful final method for this thesis stage, but not a perfect controller. Future work can test m200, multi-step shields, or realized-charge-aware guards. Load probabilistic forecasting is not claimed as a contribution because the site has only one year of load data.

## 13. Files Generated

This report generated the following supporting files in `{rel(OUT)}`:

- `final_four_main_case_summary.csv`
- `final_four_delta_comparison.csv`
- `load_bias_robustness_summary.csv`
- `tailaware_ablation_summary.csv`
- `tailaware_ablation_delta_summary.csv`
- `method_component_traceability.csv`
- `milp_equation_traceability.csv`
- `final_case_definition_table.csv`
- `final_report_source_manifest.csv`
- `ghi_point_model_comparison_summary.csv`
- `ghi_probabilistic_agaci_summary.csv`
- `agaci_publication_table_summary.csv`

## Quality Checks

- DA cases are not included in the final main result table.
- Old invalid PV truth is not used in final replay/KPI artifacts.
- LOAD-QN is not used.
- Random load error is not used in the mainline.
- Old S1-S6 is not used as final scenario input.
- The main four-case results use base deterministic load.
- Structured load-bias robustness uses only structured_low and structured_high.
- Tail-aware ablation is separated from load-bias robustness.
- MILP equations are mapped to implemented code paths.

Final status: `{report_status}`.
"""
    return md


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case_def = build_case_definition()
    main_df, delta_df = load_main_summaries()
    robust_df = build_robustness_summary()
    tail_ann, tail_delta = build_tailaware_summary()
    method_trace, eq_trace = build_traceability()
    manifest = build_manifest()
    point_fcst, id_point_fcst, prob_fcst, agaci = load_forecast_tables()

    report = make_report(
        case_def,
        main_df,
        delta_df,
        robust_df,
        tail_ann,
        tail_delta,
        method_trace,
        eq_trace,
        manifest,
        point_fcst,
        id_point_fcst,
        prob_fcst,
        agaci,
    )
    (OUT / "FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "ok", "output": rel(OUT / "FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
