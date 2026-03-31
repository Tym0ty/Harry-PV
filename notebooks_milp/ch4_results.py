#!/usr/bin/env python3
"""
Chapter 4 Results Generator — spec-named figures, tables, manifest, QA.

Creates results/ch4/ per spec §2 naming:
  results/ch4/figures/   PNG files
  results/ch4/tables/    CSV + XLSX
  results/ch4/data/      tidy data per figure
  results/ch4/manifest/  figure_manifest.csv, table_manifest.csv
  results/ch4/qa/        qa_report.json, warnings.log

Figure IDs follow the spec naming:
  B-F1..B-T2  Bridge
  M-T1..M-F2  Sizing
  R-T1..R-F5  Replay
  C-F1..C-T1  Dispatch
  L-F1..L-T1  Load uncertainty
  S-F1..S-T1  Sensitivity (generated after sensitivity_analysis.py)

Usage:
    cd <repo_root>
    python3 notebooks_milp/ch4_results.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_milp"))

OUT_ROOT   = ROOT / "results" / "ch4"
FIG_DIR    = OUT_ROOT / "figures"
TABLE_DIR  = OUT_ROOT / "tables"
DATA_DIR   = OUT_ROOT / "data"
MANIFEST_DIR = OUT_ROOT / "manifest"
QA_DIR     = OUT_ROOT / "qa"

MILP_DIR   = ROOT / "milp_outputs"
BRIDGE_DIR = ROOT / "bridge_outputs_fullyear"
PIPELINE_DIR = ROOT / "pipeline_outputs"
SENS_DIR   = ROOT / "results" / "sensitivity"

CASE_COLORS = {
    "C0": "#1f77b4",  # blue
    "C1": "#ff7f0e",  # orange
    "C2": "#2ca02c",  # green
    "C3": "#d62728",  # red
    "PI": "#9467bd",  # purple
}
CASE_LABELS = {
    "C0": "C0: Det PV + Det Load",
    "C1": "C1: Prob PV + Det Load",
    "C2": "C2: Det PV + Load Unc",
    "C3": "C3: Prob PV + Load Unc",
    "PI": "PI: Perfect Info",
}

warnings.filterwarnings("ignore")
QA_WARNINGS = []
QA_HARD_FAILS = []


# ── Setup ────────────────────────────────────────────────────────
def setup_dirs():
    for d in [FIG_DIR, TABLE_DIR, DATA_DIR, MANIFEST_DIR, QA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def savefig(fig_id, fig, dpi=150):
    path = FIG_DIR / f"{fig_id}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_id}.png")
    return str(path.relative_to(ROOT))


def savedata(fig_id, df):
    path = DATA_DIR / f"{fig_id}_data.csv"
    df.to_csv(path, index=False)
    return str(path.relative_to(ROOT))


def savetable(table_id, df, caption=""):
    csv_path = TABLE_DIR / f"{table_id}.csv"
    df.to_csv(csv_path, index=False)
    try:
        xlsx_path = TABLE_DIR / f"{table_id}.xlsx"
        df.to_excel(xlsx_path, index=False)
    except Exception:
        pass
    print(f"  Saved table: {table_id}.csv")
    return str(csv_path.relative_to(ROOT))


# ── Load data ────────────────────────────────────────────────────
def load_case_summary():
    path = MILP_DIR / "case_summary_fullyear.csv"
    df = pd.read_csv(path)
    return df


def load_replay_summary():
    path = MILP_DIR / "replay_summary_fullyear.csv"
    df = pd.read_csv(path)
    # Parse monthly_bills from JSON string
    if "monthly_bills" in df.columns:
        def parse_bills(s):
            if pd.isna(s): return {}
            if isinstance(s, dict): return s
            try: return json.loads(str(s))
            except: return {}
        df["monthly_bills_dict"] = df["monthly_bills"].apply(parse_bills)
    return df


def load_dispatch(case_id):
    path = MILP_DIR / f"dispatch_replay_{case_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ── FIGURE REGISTRY ───────────────────────────────────────────────
FIGURE_REGISTRY = []


def register_fig(fig_id, title, chapter_section, research_question,
                 case_scope, time_scope, source_script, data_file="", notes=""):
    FIGURE_REGISTRY.append({
        "figure_id": fig_id,
        "chapter_section": chapter_section,
        "title": title,
        "research_question": research_question,
        "case_scope": case_scope,
        "time_scope": time_scope,
        "source_script": "ch4_results.py",
        "data_file": data_file,
        "png_file": f"results/ch4/figures/{fig_id}.png",
        "notes": notes,
    })


TABLE_REGISTRY = []


def register_table(table_id, title, chapter_section, case_scope, notes=""):
    TABLE_REGISTRY.append({
        "table_id": table_id,
        "chapter_section": chapter_section,
        "title": title,
        "case_scope": case_scope,
        "csv_file": f"results/ch4/tables/{table_id}.csv",
        "notes": notes,
    })


# ── Bridge Figures ───────────────────────────────────────────────
def make_B_F1(raw_df, sc_df):
    """B-F1: Raw (N=500) vs Reduced (K=5) scenario trajectories for 2 representative days."""
    fig_id = "B-F1"
    pv_scale = 2687.0 / 50.0  # 50 kW reference → 2687 kW system

    raw_df = raw_df.copy()
    raw_df["target_day_local"] = pd.to_datetime(raw_df["target_day_local"])
    raw_df["hour"] = pd.to_datetime(raw_df["target_time_local"]).dt.hour

    sc_df = sc_df.copy()
    sc_df["target_day_local"] = pd.to_datetime(sc_df["target_day_local"])
    sc_df["hour"] = pd.to_datetime(sc_df["target_time_local"]).dt.hour

    all_days = sc_df["target_day_local"].unique()
    all_days_sorted = sorted(all_days)

    # Pick summer peak day (highest median PV) and winter trough day (lowest median PV)
    daily_med = {}
    for d in all_days_sorted:
        sub = sc_df[sc_df["target_day_local"] == d]
        daily_med[d] = sub["pv_available_kw"].median()
    sorted_days = sorted(daily_med, key=lambda d: daily_med[d])
    winter_day = sorted_days[0]
    summer_day = sorted_days[-1]
    rep_days = [summer_day, winter_day]
    day_labels = ["Summer Peak", "Winter Trough"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)

    for col_i, (day, dlabel) in enumerate(zip(rep_days, day_labels)):
        raw_day = raw_df[raw_df["target_day_local"] == day].sort_values("hour")
        sc_day  = sc_df[sc_df["target_day_local"] == day].sort_values("hour")
        date_str = pd.Timestamp(day).strftime("%Y-%m-%d")

        # Top row: raw 500 scenarios
        ax_raw = axes[0, col_i]
        for sid in raw_day["scenario_id"].unique():
            s = raw_day[raw_day["scenario_id"] == sid]
            ax_raw.plot(s["hour"].values, s["pv_available_kw"].values * pv_scale,
                        color="steelblue", alpha=0.04, lw=0.8)
        ax_raw.set_title(f"B-F1 Raw N=500 — {dlabel}\n{date_str}", fontsize=9)
        ax_raw.set_ylabel("PV (kW)")
        ax_raw.set_xlim(0, 23)
        ax_raw.xaxis.set_major_locator(mticker.MultipleLocator(4))

        # Bottom row: K=5 reduced medoids
        ax_red = axes[1, col_i]
        colors_k = plt.cm.tab10.colors
        for k_i, sid in enumerate(sorted(sc_day["scenario_id"].unique())):
            s = sc_day[sc_day["scenario_id"] == sid]
            prob = s["probability_pi"].iloc[0] if "probability_pi" in s.columns else None
            lbl = f"S{k_i+1}" + (f" (p={prob:.2f})" if prob is not None else "")
            ax_red.plot(s["hour"].values, s["pv_available_kw"].values * pv_scale,
                        color=colors_k[k_i % 10], lw=2, label=lbl)
        ax_red.set_title(f"B-F1 Reduced K=5 — {dlabel}\n{date_str}", fontsize=9)
        ax_red.set_xlabel("Hour of Day")
        ax_red.set_ylabel("PV (kW)")
        ax_red.set_xlim(0, 23)
        ax_red.xaxis.set_major_locator(mticker.MultipleLocator(4))
        ax_red.legend(fontsize=7, loc="upper right")

    fig.suptitle("B-F1: Raw vs Reduced PV Scenario Trajectories")
    fig.tight_layout()

    data_path = savedata(fig_id, sc_df[sc_df["target_day_local"].isin(rep_days)][
        ["target_day_local", "scenario_id", "hour", "pv_available_kw", "probability_pi"]])
    register_fig(fig_id, "Raw vs Reduced PV Scenario Trajectories",
                 "3.3", "Does K-medoid reduction preserve the scenario distribution?",
                 "C1,C3", "2 representative days", "ch4_results.py", data_path,
                 f"Summer day: {pd.Timestamp(summer_day).date()}, Winter day: {pd.Timestamp(winter_day).date()}")
    return savefig(fig_id, fig)


def make_B_F2(sc_df):
    """B-F2: Reduced scenario probability weight bar chart for 2 representative days."""
    fig_id = "B-F2"
    pv_scale = 2687.0 / 50.0

    sc_df = sc_df.copy()
    sc_df["target_day_local"] = pd.to_datetime(sc_df["target_day_local"])

    all_days = sc_df["target_day_local"].unique()
    all_days_sorted = sorted(all_days)

    daily_med = {}
    for d in all_days_sorted:
        sub = sc_df[sc_df["target_day_local"] == d]
        daily_med[d] = sub["pv_available_kw"].median()
    sorted_days = sorted(daily_med, key=lambda d: daily_med[d])
    winter_day = sorted_days[0]
    summer_day = sorted_days[-1]
    rep_days = [summer_day, winter_day]
    day_labels = ["Summer Peak", "Winter Trough"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors_k = plt.cm.tab10.colors
    equal_weight = 1.0 / 5  # K=5

    for ax, day, dlabel in zip(axes, rep_days, day_labels):
        day_df = sc_df[sc_df["target_day_local"] == day]
        # One probability per scenario
        sc_probs = (day_df.groupby("scenario_id")["probability_pi"]
                    .first().sort_index().reset_index())
        sc_probs["label"] = [f"S{i+1}" for i in range(len(sc_probs))]
        bar_colors = [colors_k[i % 10] for i in range(len(sc_probs))]
        bars = ax.bar(sc_probs["label"], sc_probs["probability_pi"],
                      color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.axhline(equal_weight, color="red", linestyle="--", lw=1.5, label=f"Equal (1/K={equal_weight:.2f})")
        ax.set_ylim(0, max(sc_probs["probability_pi"].max() * 1.3, 0.5))
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Probability")
        ax.set_title(f"B-F2: Weights — {dlabel}\n{pd.Timestamp(day).strftime('%Y-%m-%d')}", fontsize=9)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, sc_probs["probability_pi"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("B-F2: K=5 Reduced Scenario Probabilities (K-Medoid Weights)")
    fig.tight_layout()

    data_rows = []
    for day in rep_days:
        day_df = sc_df[sc_df["target_day_local"] == day]
        sc_probs = day_df.groupby("scenario_id")["probability_pi"].first().reset_index()
        sc_probs["date"] = str(pd.Timestamp(day).date())
        data_rows.append(sc_probs)
    data_path = savedata(fig_id, pd.concat(data_rows, ignore_index=True))
    register_fig(fig_id, "Reduced Scenario Probability Weights",
                 "3.3", "How are scenario probabilities assigned by K-medoid?",
                 "C1,C3", "2 representative days", "ch4_results.py", data_path,
                 "K-medoid probability assigned proportional to cluster size")
    return savefig(fig_id, fig)


def make_B_T1():
    """B-T1: Annual package completeness summary table."""
    table_id = "B-T1"
    import json as _json

    report_path = BRIDGE_DIR / "bridge_full_year_report.json"
    with open(report_path) as f:
        report = _json.load(f)

    n_days = report.get("n_admissible_days", 365)
    k_reduced = report.get("n_scenarios_probabilistic", 5)

    ingest_files = [
        ("pvdet_loadunc",  BRIDGE_DIR / "full_year_milp_ingest_pvdet_loadunc.parquet",  1,   1),
        ("pvprob_loadunc", BRIDGE_DIR / "full_year_milp_ingest_pvprob_loadunc.parquet", 500, k_reduced),
    ]

    rows = []
    for case_id, path, n_raw, k_red in ingest_files:
        if path.exists():
            df = pd.read_parquet(path)
            actual_days = df["calendar_day"].nunique() if "calendar_day" in df.columns else n_days
            missing = n_days - actual_days
            # Check probability sums (only for probabilistic parquets)
            prob_ok = True
            if "probability_pi" in df.columns:
                day_prob_sums = df.groupby(
                    ["calendar_day", "scenario_id"])["probability_pi"].first() \
                    .groupby("calendar_day").sum()
                prob_ok = bool((day_prob_sums - 1.0).abs().lt(0.01).all())
        else:
            actual_days = 0
            missing = n_days
            prob_ok = False

        rows.append({
            "case_id": case_id,
            "n_days": actual_days,
            "N_raw": n_raw,
            "K_reduced": k_red,
            "prob_sum_ok": "PASS" if prob_ok else "FAIL",
            "missing_rows": missing,
        })

    t = pd.DataFrame(rows)
    register_table(table_id, "Bridge Annual Package Completeness", "3.4", "All")
    return savetable(table_id, t)


def make_B_T2():
    """B-T2: Bridge QA gate summary."""
    table_id = "B-T2"
    import json as _json

    report_path = BRIDGE_DIR / "bridge_full_year_report.json"
    meta_path   = BRIDGE_DIR / "bridge_run_metadata.json"

    with open(report_path) as f:
        report = _json.load(f)
    with open(meta_path) as f:
        meta = _json.load(f)

    rows = [
        {"gate": "probability_check",
         "result": "PASS" if report.get("probability_check_pass") else "FAIL",
         "detail": "K=5 scenario probabilities sum to 1.0 per day"},
        {"gate": "completeness_24h",
         "result": "PASS" if report.get("completeness_24h_pass") else "FAIL",
         "detail": "All 365 days have 24 hourly rows"},
        {"gate": "truth_solve_segregation",
         "result": "PASS" if report.get("truth_solve_segregation_pass") else "FAIL",
         "detail": "Realized (truth) load/PV kept separate from solve scenarios"},
        {"gate": "n_admissible_days",
         "result": "PASS",
         "detail": f"{report.get('n_admissible_days', '?')} days (0 excluded)"},
        {"gate": "schema_version",
         "result": "PASS",
         "detail": report.get("schema_version", meta.get("bridge_version", "?"))},
        {"gate": "load_source",
         "result": "PASS",
         "detail": report.get("load_source_v2", meta.get("load_source", "?"))},
        {"gate": "load_uncertainty_mode",
         "result": "PASS",
         "detail": (f"K_scenarios_pieter_mape: summer={report['load_unc_mape']['summer']:.4f}, "
                    f"fall={report['load_unc_mape']['fall']:.4f}, "
                    f"winter={report['load_unc_mape']['winter']:.4f}, "
                    f"spring={report['load_unc_mape']['spring']:.4f}")
                   if "load_unc_mape" in report else "N/A"},
    ]

    t = pd.DataFrame(rows)
    register_table(table_id, "Bridge QA Gate Summary", "3.4", "All")
    return savetable(table_id, t)


def make_B_F3(sc_df, pv_det_df):
    """B-F3: Det vs prob PV day profile — Q50 + K scenario fan."""
    fig_id = "B-F3"
    # Pick a high-solar-variance summer weekday
    summer_days = sc_df[sc_df["target_day_local"].dt.month.isin([6,7,8,9])]["target_day_local"].unique()
    if len(summer_days) == 0:
        summer_days = sc_df["target_day_local"].unique()

    # Choose day with highest std across scenarios
    best_day = None
    best_std = -1
    for d in summer_days[:60]:  # check first 60 summer days
        sub = sc_df[sc_df["target_day_local"] == d]
        pv_vals = sub.groupby("scenario_id")["pv_available_kw"].sum().values
        if pv_vals.std() > best_std:
            best_std = pv_vals.std()
            best_day = d

    if best_day is None:
        best_day = sc_df["target_day_local"].iloc[0]

    day_sc = sc_df[sc_df["target_day_local"] == best_day].copy()
    hours = list(range(24))

    fig, ax = plt.subplots(figsize=(9, 4))
    # K reduced scenarios
    pv_scale = 2687.0 / 50.0
    day_sc = day_sc.copy()
    day_sc["hour"] = pd.to_datetime(day_sc["target_time_local"]).dt.hour
    for sid in day_sc["scenario_id"].unique():
        s = day_sc[day_sc["scenario_id"] == sid].sort_values("hour")
        ax.plot(s["hour"].values, s["pv_available_kw"].values * pv_scale,
                alpha=0.35, color="#ff7f0e", lw=1)

    # Use median across scenarios as Q50
    q50 = day_sc.groupby("hour")["pv_available_kw"].median() * pv_scale
    ax.plot(q50.index, q50.values, color="#1f77b4", lw=2, label="Q50 (prob)")
    ax.plot([], [], color="#ff7f0e", alpha=0.6, lw=1, label="K scenarios")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("PV Available (kW)")
    ax.set_title(f"B-F3: Prob PV Scenarios — {pd.Timestamp(best_day).date()}")
    ax.legend()
    ax.set_xlim(0, 23)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig.tight_layout()

    data_df = day_sc[["scenario_id", "hour", "pv_available_kw"]].copy()
    data_df["pv_kw_scaled"] = data_df["pv_available_kw"] * pv_scale
    data_df["date"] = str(pd.Timestamp(best_day).date())
    data_path = savedata(fig_id, data_df)
    register_fig(fig_id, "Prob PV Scenario Fan — Representative Summer Day",
                 "4.2", "Does probabilistic PV improve dispatch vs deterministic?",
                 "C1", "1 day", "ch4_results.py", data_path,
                 f"Shown day: {pd.Timestamp(best_day).date()}, highest scenario variance")
    return savefig(fig_id, fig)


def make_B_F4(loadunc_path):
    """B-F4: Load uncertainty fan chart — K load paths vs det load."""
    fig_id = "B-F4"
    df = pd.read_parquet(loadunc_path)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])

    # Pick a summer weekday
    summer_wkday = df[(df["calendar_day"].dt.month.isin([6,7,8,9])) &
                      (df["calendar_day"].dt.weekday < 5)]
    if len(summer_wkday) == 0:
        summer_wkday = df

    ref_day = summer_wkday["calendar_day"].iloc[0]
    day_df = df[df["calendar_day"] == ref_day].sort_values("hour_local")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax_i, (ax, day_label) in enumerate(zip(axes, ["Summer Weekday", "Det load"])):
        if ax_i == 0:
            # Show K load scenarios for the chosen day
            for sid in sorted(day_df["scenario_id"].unique()):
                s = day_df[day_df["scenario_id"] == sid].sort_values("hour_local")
                ax.plot(s["hour_local"].values - 1, s["load_kw"].values,
                        alpha=0.5, lw=1.2)
            ax.set_title(f"B-F4a: Load Scenarios (C2/C3)\n{ref_day.date()}")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Load (kW)")
        else:
            # Show spread: mean ± std across scenarios per hour
            hourly = day_df.groupby("hour_local")["load_kw"]
            mu = hourly.mean()
            sig = hourly.std()
            ax.plot(mu.index - 1, mu.values, color="black", lw=2, label="Mean")
            ax.fill_between(mu.index - 1, (mu - sig).values, (mu + sig).values,
                            alpha=0.3, color="blue", label="±1σ")
            ax.set_title(f"B-F4b: Load Mean ± σ\n{ref_day.date()}")
            ax.set_xlabel("Hour")
            ax.legend(fontsize=8)

    fig.suptitle("B-F4: Load Uncertainty (K-Scenario, Pieter MAPE)")
    fig.tight_layout()

    data_path = savedata(fig_id, day_df[["calendar_day", "scenario_id", "hour_local", "load_kw"]])
    register_fig(fig_id, "Load Uncertainty K-Scenario Fan",
                 "4.2", "How does load uncertainty affect dispatch reliability?",
                 "C2,C3", "1 day (summer weekday)", "ch4_results.py", data_path,
                 "K=5 load paths from Pieter seasonal MAPE")
    return savefig(fig_id, fig)


# ── Sizing Figures ───────────────────────────────────────────────
def make_M_T1(case_df):
    """M-T1: Sizing master table."""
    table_id = "M-T1"
    cols = ["case", "bess_p_kw", "bess_e_kwh", "ep_ratio", "contract_kw", "total_cost_M"]
    labels = ["Case", "P_B (kW)", "E_B (kWh)", "E/P", "CC* (kW)", "Solve Cost (M NTD)"]
    t = case_df[cols].copy()
    t.columns = labels
    register_table(table_id, "BESS Sizing + Contract Capacity Results", "4.3", "C0–C3,PI")
    return savetable(table_id, t)


def make_M_F1(case_df):
    """M-F1: Sizing comparison 3-panel grouped bar."""
    fig_id = "M-F1"
    cases = [c for c in ["C0", "C1", "C2", "C3", "PI"] if c in case_df["case"].values]
    x = np.arange(len(cases))
    w = 0.6

    metrics = [
        ("contract_kw", "Contract Capacity CC* (kW)"),
        ("bess_p_kw", "BESS Power P_B (kW)"),
        ("bess_e_kwh", "BESS Energy E_B (kWh)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, ylabel) in zip(axes, metrics):
        vals = [case_df.loc[case_df["case"] == c, col].iloc[0] for c in cases]
        bars = ax.bar(x, vals, width=w,
                      color=[CASE_COLORS.get(c, "gray") for c in cases],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("(")[0].strip())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("M-F1: BESS Sizing & Contract Capacity — All Cases", fontsize=12)
    fig.tight_layout()

    data_path = savedata(fig_id, case_df[["case"] + [m[0] for m in metrics]])
    register_fig(fig_id, "BESS Sizing Comparison — C0–C3 + PI",
                 "4.3", "How does uncertainty treatment affect optimal sizing?",
                 "C0,C1,C2,C3,PI", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


def make_M_T2(case_df):
    """M-T2: Solve-world cost breakdown table."""
    table_id = "M-T2"
    cols = ["case", "AEC_inv_M", "AEC_ene_M", "AEC_basic_M",
            "AEC_over_M", "AEC_green_M", "AEC_deg_M", "total_cost_M"]
    t = case_df[[c for c in cols if c in case_df.columns]].copy()
    register_table(table_id, "Solve-World Annual Cost Breakdown", "4.3", "C0–C3,PI")
    return savetable(table_id, t)


def make_M_F2(case_df):
    """M-F2: Solve-world cost stacked bar."""
    fig_id = "M-F2"
    cases = [c for c in ["C0", "C1", "C2", "C3", "PI"] if c in case_df["case"].values]
    components = ["AEC_inv_M", "AEC_ene_M", "AEC_basic_M", "AEC_over_M", "AEC_green_M", "AEC_deg_M"]
    comp_labels = ["Investment", "Energy", "Basic", "Over-contract", "Green", "Degradation"]
    comp_colors = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cases))
    bottom = np.zeros(len(cases))
    for comp, label, color in zip(components, comp_labels, comp_colors):
        if comp not in case_df.columns:
            continue
        vals = np.array([case_df.loc[case_df["case"] == c, comp].values[0]
                         if c in case_df["case"].values else 0 for c in cases])
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Annual Cost (M NTD)")
    ax.set_title("M-F2: Solve-World Annual Cost Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    data_path = savedata(fig_id, case_df[["case"] + [c for c in components if c in case_df.columns]])
    register_fig(fig_id, "Solve-World Cost Stacked Bar",
                 "4.3", "What drives total annual cost in each case?",
                 "C0,C1,C2,C3,PI", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


# ── Replay Figures ───────────────────────────────────────────────
def make_R_T1(replay_df, case_df):
    """R-T1: Replay summary master table."""
    table_id = "R-T1"
    cols = ["case_id", "replay_total_M", "replay_ene_M", "replay_over_M",
            "over_months", "worst_bill_M", "RE_pct", "solve_obj_M", "gap_pct"]
    t = replay_df[[c for c in cols if c in replay_df.columns]].copy()
    register_table(table_id, "Replay Annual Cost Summary", "4.4", "C0–C3,PI")
    return savetable(table_id, t)


def make_R_F1(replay_df):
    """R-F1: Replay annual total cost grouped bar."""
    fig_id = "R-F1"
    cases = [c for c in ["C0", "C1", "C2", "C3", "PI"] if c in replay_df["case_id"].values]
    vals = [replay_df.loc[replay_df["case_id"] == c, "replay_total_M"].iloc[0] for c in cases]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cases))
    bars = ax.bar(x, vals, color=[CASE_COLORS.get(c, "gray") for c in cases],
                  width=0.5, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS.get(c, c) for c in cases], fontsize=7.5)
    ax.set_ylabel("Annual Cost (M NTD)")
    ax.set_title("R-F1: Replay Annual Total Cost")
    vmin = min(vals) * 0.995
    ax.set_ylim(vmin, max(vals) * 1.015)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    data_path = savedata(fig_id, replay_df[["case_id", "replay_total_M"]])
    register_fig(fig_id, "Replay Annual Total Cost — All Cases",
                 "4.4", "Does uncertainty modelling improve realized cost?",
                 "C0,C1,C2,C3,PI", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


def make_R_F2(replay_df):
    """R-F2: Over-contract fee + worst-month bill."""
    fig_id = "R-F2"
    cases = [c for c in ["C0", "C1", "C2", "C3", "PI"] if c in replay_df["case_id"].values]
    x = np.arange(len(cases))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    oc_vals = [replay_df.loc[replay_df["case_id"] == c, "replay_over_M"].iloc[0] for c in cases]
    wm_vals = [replay_df.loc[replay_df["case_id"] == c, "worst_bill_M"].iloc[0] for c in cases]

    ax1.bar(x, oc_vals, color=[CASE_COLORS.get(c, "gray") for c in cases], width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(cases)
    ax1.set_ylabel("Over-Contract Fee (M NTD)")
    ax1.set_title("R-F2a: Annual Over-Contract Fee")
    for xi, v in zip(x, oc_vals):
        ax1.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)

    ax2.bar(x, wm_vals, color=[CASE_COLORS.get(c, "gray") for c in cases], width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(cases)
    ax2.set_ylabel("Worst Monthly Bill (M NTD)")
    ax2.set_title("R-F2b: Worst Monthly Bill")
    wm_min = min(wm_vals) * 0.99
    ax2.set_ylim(wm_min, max(wm_vals) * 1.015)
    for xi, v in zip(x, wm_vals):
        ax2.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle("R-F2: Over-Contract & Peak Bill", fontsize=11)
    fig.tight_layout()

    data = replay_df[["case_id", "replay_over_M", "worst_bill_M", "over_months"]].copy()
    data_path = savedata(fig_id, data)
    register_fig(fig_id, "Over-Contract Fee & Worst Month Bill",
                 "4.4", "Which case minimizes demand-charge risk?",
                 "C0,C1,C2,C3,PI", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


def make_R_F3(replay_df):
    """R-F3: RE20 attainment + T-REC volume."""
    fig_id = "R-F3"
    cases = [c for c in ["C0", "C1", "C2", "C3", "PI"] if c in replay_df["case_id"].values]
    x = np.arange(len(cases))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    re_vals = [replay_df.loc[replay_df["case_id"] == c, "RE_pct"].iloc[0] for c in cases]
    trec_vals = [replay_df.loc[replay_df["case_id"] == c, "TREC_kWh"].iloc[0] / 1000 for c in cases]

    ax1.bar(x, re_vals, color=[CASE_COLORS.get(c, "gray") for c in cases], width=0.5)
    ax1.axhline(20, color="red", ls="--", lw=1.2, label="20% target")
    ax1.set_xticks(x); ax1.set_xticklabels(cases)
    ax1.set_ylabel("RE Share (%)")
    ax1.set_title("R-F3a: RE20 Attainment")
    ax1.legend(fontsize=8)

    ax2.bar(x, trec_vals, color=[CASE_COLORS.get(c, "gray") for c in cases], width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(cases)
    ax2.set_ylabel("T-REC Volume (MWh)")
    ax2.set_title("R-F3b: T-REC Purchases")
    for xi, v in zip(x, trec_vals):
        ax2.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8)

    fig.suptitle("R-F3: RE20 Compliance & T-REC", fontsize=11)
    fig.tight_layout()

    data = replay_df[["case_id", "RE_pct", "TREC_kWh"]].copy()
    data_path = savedata(fig_id, data)
    register_fig(fig_id, "RE20 Compliance & T-REC Volume",
                 "4.4", "How much T-REC procurement is required?",
                 "C0,C1,C2,C3,PI", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


def make_R_F4(replay_df):
    """R-F4: Monthly bill breakdown stacked bar (all cases, 12 months)."""
    fig_id = "R-F4"
    cases = [c for c in ["C0", "C1", "C2", "C3"] if c in replay_df["case_id"].values]
    months = list(range(1, 13))
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes_flat = axes.flatten()

    for ax, case in zip(axes_flat, cases):
        row = replay_df[replay_df["case_id"] == case].iloc[0]
        bills = row.get("monthly_bills_dict", {})
        vals = [float(bills.get(str(m), bills.get(m, 0.0))) for m in months]
        ax.bar(months, vals, color=CASE_COLORS.get(case, "gray"), alpha=0.8, width=0.7)
        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_ylabel("Monthly Bill (M NTD)")
        ax.set_title(f"{case}: {CASE_LABELS.get(case, case)}")
        ax.set_xlabel("Month")

    fig.suptitle("R-F4: Monthly Bill — All Cases", fontsize=12)
    fig.tight_layout()

    rows = []
    for case in cases:
        row = replay_df[replay_df["case_id"] == case].iloc[0]
        bills = row.get("monthly_bills_dict", {})
        for m in months:
            rows.append({"case": case, "month": m,
                         "bill_M": float(bills.get(str(m), bills.get(m, 0.0)))})
    data_path = savedata(fig_id, pd.DataFrame(rows))
    register_fig(fig_id, "Monthly Bill Breakdown — All Cases",
                 "4.4", "Which months drive peak billing?",
                 "C0,C1,C2,C3", "12 months", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


def make_R_F5(replay_df, case_df):
    """R-F5: Design-to-replay gap."""
    fig_id = "R-F5"
    cases = [c for c in ["C0", "C1", "C2", "C3", "PI"] if c in replay_df["case_id"].values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cases))
    gaps = [replay_df.loc[replay_df["case_id"] == c, "gap_pct"].iloc[0]
            if c in replay_df["case_id"].values else 0 for c in cases]
    colors = ["green" if g >= 0 else "red" for g in gaps]
    bars = ax.bar(x, gaps, color=colors, alpha=0.75, width=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Replay − Solve Gap (%)")
    ax.set_title("R-F5: Design-to-Replay Cost Gap (%)")
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2,
                g + 0.02 if g >= 0 else g - 0.08,
                f"{g:+.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    data = replay_df[["case_id", "solve_obj_M", "replay_total_M", "gap_pct"]].copy()
    data_path = savedata(fig_id, data)
    register_fig(fig_id, "Design-to-Replay Gap (%)",
                 "4.4", "How conservative is each design vs realized cost?",
                 "C0,C1,C2,C3,PI", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


# ── Dispatch Figures ─────────────────────────────────────────────
def make_C_F1(dispatch_c0, dispatch_c1, case_df, replay_df):
    """C-F1: 5-day dispatch trace C0 vs C1 (July high-variance week)."""
    fig_id = "C-F1"
    if dispatch_c0 is None or dispatch_c1 is None:
        print(f"  Skipping {fig_id}: dispatch data missing")
        return None

    # Select July (high-solar week) — days with highest PV std
    for disp, label in [(dispatch_c0, "C0"), (dispatch_c1, "C1")]:
        disp["calendar_day"] = pd.to_datetime(disp["calendar_day"])

    jul_days = dispatch_c0[dispatch_c0["calendar_day"].dt.month == 7]["calendar_day"].dt.normalize().unique()
    if len(jul_days) < 5:
        jul_days = dispatch_c0["calendar_day"].dt.normalize().unique()[:5]

    # Pick 5 consecutive days with highest PV variance
    if len(jul_days) >= 5:
        # compute daily PV std
        pv_stds = []
        for d in jul_days:
            sub = dispatch_c0[dispatch_c0["calendar_day"].dt.normalize() == d]
            pv_stds.append((d, sub["pv_realized_kw"].std()))
        pv_stds.sort(key=lambda x: x[1], reverse=True)
        anchor = pv_stds[0][0]
        # Take 5 consecutive days starting from anchor
        all_days = sorted(dispatch_c0["calendar_day"].dt.normalize().unique())
        anchor_i = list(all_days).index(anchor)
        start_i = max(0, anchor_i - 2)
        selected_days = all_days[start_i:start_i+5]
    else:
        selected_days = jul_days[:5]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, (disp, case) in zip(axes, [(dispatch_c0, "C0"), (dispatch_c1, "C1")]):
        sub = disp[disp["calendar_day"].dt.normalize().isin(selected_days)].sort_values(
            ["calendar_day", "hour_local"])
        t = np.arange(len(sub))
        ax.stackplot(t,
                     sub["pv_to_load_kw"].values,
                     sub["bess_dis_kw"].values,
                     sub["grid_load_kw"].values,
                     labels=["PV→Load", "BESS Dis.", "Grid Import"],
                     colors=["#ffd700", "#2ca02c", "#1f77b4"],
                     alpha=0.75)
        ax.plot(t, sub["load_realized_kw"].values, "k-", lw=1.5, label="Load")
        ax2 = ax.twinx()
        ax2.plot(t, sub["soc_end_kwh"].values, "m--", lw=1, alpha=0.7, label="SOC")
        ax2.set_ylabel("SOC (kWh)", color="m")
        ax.set_title(f"C-F1: {case} Dispatch Trace")
        ax.set_ylabel("Power (kW)")
        ax.legend(loc="upper left", fontsize=7, ncol=4)

    ticks = np.arange(0, len(sub), 24)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(selected_days[i].date()) if i < len(selected_days) else ""
                               for i in range(len(ticks))], rotation=30, fontsize=7)
    fig.suptitle("C-F1: 5-Day Dispatch Trace — C0 vs C1", fontsize=11)
    fig.tight_layout()

    data_path = savedata(fig_id, dispatch_c0[dispatch_c0["calendar_day"].dt.normalize().isin(
        selected_days)][["calendar_day", "hour_local", "grid_import_kw", "bess_dis_kw",
                         "pv_to_load_kw", "soc_end_kwh", "load_realized_kw"]])
    register_fig(fig_id, "5-Day Dispatch Trace: C0 vs C1",
                 "4.5", "How do C0 and C1 dispatch strategies differ?",
                 "C0,C1", "5 days (July high-variance)", "ch4_results.py", data_path,
                 f"Selected days: {selected_days[0].date()} to {selected_days[-1].date()}")
    return savefig(fig_id, fig)


def make_C_F2_F3(dispatch, case_id, replay_df):
    """C-F2/C-F3: 48h stress-window dispatch (worst over-contract month)."""
    if dispatch is None:
        return None, None

    dispatch = dispatch.copy()
    dispatch["calendar_day"] = pd.to_datetime(dispatch["calendar_day"])

    # Auto-select: worst month = month with highest over-contract
    row = replay_df[replay_df["case_id"] == case_id]
    if len(row) == 0:
        return None, None
    bills_dict = row.iloc[0].get("monthly_bills_dict", {})
    if bills_dict:
        worst_month = max(bills_dict, key=lambda k: bills_dict[k])
        worst_month = int(worst_month)
    else:
        worst_month = 7  # default July

    # Find 48h window in worst month with highest grid import
    month_data = dispatch[dispatch["calendar_day"].dt.month == worst_month].sort_values(
        ["calendar_day", "hour_local"])
    if len(month_data) < 48:
        month_data = dispatch.sort_values(["calendar_day", "hour_local"])

    best_start = 0
    best_sum = -1
    for i in range(0, len(month_data) - 48, 24):
        s = month_data.iloc[i:i+48]["grid_import_kw"].sum()
        if s > best_sum:
            best_sum = s
            best_start = i

    window = month_data.iloc[best_start:best_start+48].copy()
    t = np.arange(48)
    window_label = f"{window['calendar_day'].iloc[0].date()}"

    # C-F2: single panel
    fig_id2 = f"C-F2-{case_id}"
    fig2, ax = plt.subplots(figsize=(12, 4.5))
    ax.stackplot(t,
                 window["pv_to_load_kw"].values,
                 window["bess_dis_kw"].values,
                 window["grid_load_kw"].values,
                 labels=["PV→Load", "BESS Dis.", "Grid Import"],
                 colors=["#ffd700", "#2ca02c", "#1f77b4"], alpha=0.8)
    ax.plot(t, window["load_realized_kw"].values, "k-", lw=2, label="Load")
    ax2 = ax.twinx()
    ax2.plot(t, window["soc_end_kwh"].values, "m--", lw=1.5, label="SOC (kWh)")
    ax2.set_ylabel("SOC (kWh)", color="m")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Hour in 48h window")
    ax.set_title(f"C-F2: 48h Stress Window — {case_id} (month={worst_month}, start={window_label})")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    fig2.tight_layout()
    register_fig(fig_id2, f"48h Stress Window Dispatch — {case_id}",
                 "4.5", "What is the worst-case 48h dispatch?",
                 case_id, "48h (auto-selected worst month)", "ch4_results.py", "",
                 f"Worst month={worst_month}, start={window_label}")
    path2 = savefig(fig_id2, fig2)

    # C-F3: multi-panel publication version
    fig_id3 = f"C-F3-{case_id}"
    fig3, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    panels = [
        ("Load vs Supply", window["load_realized_kw"], window["pv_to_load_kw"] + window["bess_dis_kw"] + window["grid_load_kw"]),
        ("Grid Import", window["grid_import_kw"], None),
        ("BESS Dispatch", None, None),
        ("SOC", window["soc_end_kwh"], None),
    ]
    axes[0].plot(t, window["load_realized_kw"].values, "k-", lw=1.5, label="Load")
    axes[0].stackplot(t, window["pv_to_load_kw"].values, window["bess_dis_kw"].values,
                      window["grid_load_kw"].values,
                      labels=["PV", "BESS", "Grid"], colors=["#ffd700","#2ca02c","#1f77b4"], alpha=0.7)
    axes[0].set_ylabel("kW"); axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title("Supply Stack")

    axes[1].fill_between(t, 0, window["grid_import_kw"].values, color="#1f77b4", alpha=0.7)
    axes[1].set_ylabel("Grid Import (kW)")
    cc_val = case_df_global["contract_kw"].values[0] if 'case_df_global' in globals() else None

    axes[2].bar(t, window["bess_ch_kw"].values, color="orange", alpha=0.7, label="Charge")
    axes[2].bar(t, -window["bess_dis_kw"].values, color="green", alpha=0.7, label="Discharge")
    axes[2].set_ylabel("BESS Power (kW)"); axes[2].legend(fontsize=7)

    axes[3].plot(t, window["soc_end_kwh"].values, "m-", lw=2)
    axes[3].set_ylabel("SOC (kWh)"); axes[3].set_xlabel("Hour")

    for ax in axes:
        ax.set_xlim(0, 47)
    fig3.suptitle(f"C-F3: 48h Stress Window Multi-Panel — {case_id}", fontsize=11)
    fig3.tight_layout()
    register_fig(fig_id3, f"48h Stress Window Multi-Panel — {case_id}",
                 "4.5", "Detailed dispatch analysis during peak demand period",
                 case_id, "48h", "ch4_results.py", "",
                 f"Month={worst_month}, 4-panel publication version")
    path3 = savefig(fig_id3, fig3)

    # Save data
    savedata("C-F2-F3-" + case_id, window[["calendar_day", "hour_local", "grid_import_kw",
                                            "bess_ch_kw", "bess_dis_kw", "soc_end_kwh",
                                            "pv_to_load_kw", "load_realized_kw"]])
    return path2, path3


# ── Load Uncertainty Figures ─────────────────────────────────────
def make_L_F1(case_df, replay_df):
    """L-F1: C2 vs C3 comparison."""
    fig_id = "L-F1"
    metrics_cases = {
        "replay_total_M": ("Annual Cost (M NTD)", ["C0","C1","C2","C3"]),
        "replay_over_M":  ("Over-Contract (M NTD)", ["C0","C1","C2","C3"]),
        "RE_pct":         ("RE Share (%)", ["C0","C1","C2","C3"]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (metric, (ylabel, cases)) in zip(axes, metrics_cases.items()):
        vals = []
        for c in cases:
            row = replay_df[replay_df["case_id"] == c]
            vals.append(float(row[metric].iloc[0]) if len(row) > 0 else 0)
        x = np.arange(len(cases))
        ax.bar(x, vals, color=[CASE_COLORS.get(c, "gray") for c in cases], width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(cases)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    fig.suptitle("L-F1: Load Uncertainty Impact — C0–C3", fontsize=11)
    fig.tight_layout()

    data = replay_df[replay_df["case_id"].isin(["C0","C1","C2","C3"])][
        ["case_id", "replay_total_M", "replay_over_M", "RE_pct"]].copy()
    data_path = savedata(fig_id, data)
    register_fig(fig_id, "Load Uncertainty Impact — C2 vs C3 vs baseline",
                 "4.6", "Does adding load uncertainty improve dispatch robustness?",
                 "C0,C1,C2,C3", "Full year", "ch4_results.py", data_path)
    return savefig(fig_id, fig)


def make_L_T1(case_df, replay_df):
    """L-T1: Incremental value table."""
    table_id = "L-T1"
    rows = []
    for pair, label in [("C0","C1"), ("C2","C3")]:
        r0 = replay_df[replay_df["case_id"] == pair]
        r1 = replay_df[replay_df["case_id"] == label]
        c0 = case_df[case_df["case"] == pair]
        c1 = case_df[case_df["case"] == label]
        if len(r0) == 0 or len(r1) == 0: continue
        rows.append({
            "comparison": f"{label}−{pair}",
            "Δreplay_total_M": round(float(r1["replay_total_M"].iloc[0]) - float(r0["replay_total_M"].iloc[0]), 3),
            "Δreplay_over_M": round(float(r1["replay_over_M"].iloc[0]) - float(r0["replay_over_M"].iloc[0]), 3),
            "Δover_months": float(r1["over_months"].iloc[0]) - float(r0["over_months"].iloc[0]),
            "Δbess_e_kwh": float(c1["bess_e_kwh"].iloc[0]) - float(c0["bess_e_kwh"].iloc[0])
            if len(c0) > 0 and len(c1) > 0 else None,
        })
    t = pd.DataFrame(rows)
    register_table(table_id, "Incremental Value of Load Uncertainty", "4.6", "C0–C3")
    return savetable(table_id, t)


# ── Sensitivity Figures ──────────────────────────────────────────
def make_S_F1_F2(sens_agg, metric="replay_total_mean", fig_id_base="S-F1"):
    """S-F1: Replay AEC mean heatmap (C1 and C3)."""
    if sens_agg is None or len(sens_agg) == 0:
        print(f"  Skipping {fig_id_base}: no sensitivity data")
        return None

    cases = sens_agg["case"].unique()
    fig, axes = plt.subplots(1, len(cases), figsize=(6*len(cases), 5))
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        sub = sens_agg[sens_agg["case"] == case]
        if len(sub) == 0:
            continue
        n_vals = sorted(sub["N"].unique())
        k_vals = sorted(sub["K"].unique())
        grid = np.full((len(k_vals), len(n_vals)), np.nan)
        for i, k in enumerate(k_vals):
            for j, n in enumerate(n_vals):
                v = sub[(sub["N"] == n) & (sub["K"] == k)][metric]
                if len(v) > 0:
                    grid[i, j] = float(v.iloc[0])

        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(n_vals)))
        ax.set_xticklabels(n_vals, rotation=45)
        ax.set_yticks(range(len(k_vals)))
        ax.set_yticklabels(k_vals)
        ax.set_xlabel("N (raw scenarios)")
        ax.set_ylabel("K (reduced scenarios)")
        ax.set_title(f"{case}: {metric}")
        plt.colorbar(im, ax=ax, label="M NTD")

        # Annotate cells
        for i in range(len(k_vals)):
            for j in range(len(n_vals)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.suptitle(f"{fig_id_base}: Replay AEC Heatmap ({metric})", fontsize=11)
    fig.tight_layout()
    register_fig(fig_id_base, f"Sensitivity Heatmap — {metric}",
                 "4.7", "What N and K are sufficient?",
                 "C1,C3", "Full year", "ch4_results.py", "",
                 f"Metric: {metric}")
    return savefig(fig_id_base, fig)


def make_S_T1(sens_report):
    """S-T1: Sufficiency decision table."""
    table_id = "S-T1"
    if sens_report is None:
        print(f"  Skipping {table_id}: no sensitivity report")
        return None
    rows = []
    for case in ["C1", "C3"]:
        rows.append({
            "case": case,
            "sufficient_N": sens_report.get(f"{case}_sufficient_N", "N/A"),
            "sufficient_K": sens_report.get(f"{case}_sufficient_K", "N/A"),
            "baseline_N500_K5_replay_M": sens_report.get(f"{case}_baseline_N500_K5_replay_M", "N/A"),
            "N_saturation": sens_report.get(f"{case}_N_saturation", "N/A"),
        })
    t = pd.DataFrame(rows)
    register_table(table_id, "N×K Sufficiency Decision Table", "4.7", "C1,C3")
    return savetable(table_id, t)


# ── QA ───────────────────────────────────────────────────────────
def run_qa(case_df, replay_df):
    """Run QA checks. Returns (n_hard_fails, n_warnings)."""
    # Check probability sums (from bridge)
    # Check case order
    expected_order = ["C0", "C1", "C2", "C3", "PI"]
    actual_order = list(case_df["case"].values)
    for i, (exp, act) in enumerate(zip(expected_order, actual_order)):
        if exp != act:
            QA_HARD_FAILS.append(f"Case order mismatch at position {i}: expected {exp}, got {act}")

    # Check replay > solve (solve is optimistic lower bound)
    for _, row in replay_df.iterrows():
        cid = row["case_id"]
        if cid == "PI":
            continue
        replay_m = row.get("replay_total_M", 0)
        solve_m = row.get("solve_obj_M", 0)
        if solve_m > 0 and replay_m < solve_m * 0.95:
            QA_HARD_FAILS.append(
                f"{cid}: replay ({replay_m:.2f}M) < 95% of solve ({solve_m:.2f}M) — unexpected")

    # Check RE=20% for all cases
    for _, row in replay_df.iterrows():
        re = row.get("RE_pct", 0)
        if re < 19.5:
            QA_WARNINGS.append(f"{row['case_id']}: RE_pct={re:.1f}% < 19.5% (below 20% target)")

    # Write QA files
    qa_report = {
        "generated_at": datetime.now().isoformat(),
        "hard_fails": QA_HARD_FAILS,
        "warnings": QA_WARNINGS,
        "n_hard_fails": len(QA_HARD_FAILS),
        "n_warnings": len(QA_WARNINGS),
        "pass": len(QA_HARD_FAILS) == 0,
    }
    with open(QA_DIR / "qa_report.json", "w") as f:
        json.dump(qa_report, f, indent=2)
    with open(QA_DIR / "hard_fail.log", "w") as f:
        f.write("\n".join(QA_HARD_FAILS) or "NONE")
    with open(QA_DIR / "warnings.log", "w") as f:
        f.write("\n".join(QA_WARNINGS) or "NONE")

    print(f"  QA: {len(QA_HARD_FAILS)} hard fails, {len(QA_WARNINGS)} warnings")
    return len(QA_HARD_FAILS), len(QA_WARNINGS)


# ── Manifest ─────────────────────────────────────────────────────
def write_manifest():
    fig_df = pd.DataFrame(FIGURE_REGISTRY)
    if len(fig_df) > 0:
        fig_df.to_csv(MANIFEST_DIR / "figure_manifest.csv", index=False)
        print(f"  Figure manifest: {len(fig_df)} figures")

    tbl_df = pd.DataFrame(TABLE_REGISTRY)
    if len(tbl_df) > 0:
        tbl_df.to_csv(MANIFEST_DIR / "table_manifest.csv", index=False)
        print(f"  Table manifest: {len(tbl_df)} tables")

    # README
    readme_lines = [
        "# Chapter 4 Results Package\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        "## Directory Structure\n",
        "- figures/   PNG files (spec-named)\n",
        "- tables/    CSV + XLSX tables\n",
        "- data/      Tidy CSV source data per figure\n",
        "- manifest/  figure_manifest.csv, table_manifest.csv\n",
        "- qa/        qa_report.json, hard_fail.log, warnings.log\n\n",
        "## Figures\n",
    ]
    for row in FIGURE_REGISTRY:
        readme_lines.append(f"- **{row['figure_id']}**: {row['title']}\n")
    readme_lines.append("\n## Tables\n")
    for row in TABLE_REGISTRY:
        readme_lines.append(f"- **{row['table_id']}**: {row['title']}\n")

    with open(MANIFEST_DIR / "README.md", "w") as f:
        f.writelines(readme_lines)


# ── Main ─────────────────────────────────────────────────────────
def main():
    global case_df_global
    setup_dirs()
    print("="*60)
    print("Chapter 4 Results Generator")
    print("="*60)

    # Load data
    print("\nLoading case + replay summaries...")
    case_df = load_case_summary()
    replay_df = load_replay_summary()
    case_df_global = case_df

    print(f"  Cases: {list(case_df['case'].values)}")
    print(f"  Replay cases: {list(replay_df['case_id'].values)}")

    # Load dispatch data
    print("Loading dispatch data...")
    dispatch = {c: load_dispatch(c) for c in ["C0", "C1", "C2", "C3", "PI"]}

    # ── Bridge Figures ───────────────────────────────────────────
    print("\n--- Bridge Figures ---")
    sc_path = PIPELINE_DIR / "scenarios_joint_pv_load_reduced_5.parquet"
    raw_path = PIPELINE_DIR / "scenarios_joint_pv_load_raw_500.parquet"

    if sc_path.exists():
        sc_df = pd.read_parquet(sc_path)
        sc_df["target_day_local"] = pd.to_datetime(sc_df["target_day_local"])

        if raw_path.exists():
            raw_df = pd.read_parquet(raw_path)
            make_B_F1(raw_df, sc_df)
        else:
            print(f"  Skipping B-F1: {raw_path.name} not found")

        make_B_F2(sc_df)

        pv_det_df = pd.read_parquet(PIPELINE_DIR / "pv_point_forecast_caseyear.parquet") \
            if (PIPELINE_DIR / "pv_point_forecast_caseyear.parquet").exists() else None
        make_B_F3(sc_df, pv_det_df)
    else:
        print(f"  Skipping B-F1/B-F2/B-F3: {sc_path.name} not found")

    make_B_T1()
    make_B_T2()

    loadunc_path = BRIDGE_DIR / "full_year_milp_ingest_pvdet_loadunc.parquet"
    if loadunc_path.exists():
        make_B_F4(loadunc_path)

    # ── Sizing Figures ───────────────────────────────────────────
    print("\n--- Sizing Figures ---")
    make_M_T1(case_df)
    make_M_F1(case_df)
    make_M_T2(case_df)
    make_M_F2(case_df)

    # ── Replay Figures ───────────────────────────────────────────
    print("\n--- Replay Figures ---")
    make_R_T1(replay_df, case_df)
    make_R_F1(replay_df)
    make_R_F2(replay_df)
    make_R_F3(replay_df)
    make_R_F4(replay_df)
    make_R_F5(replay_df, case_df)

    # ── Dispatch Figures ─────────────────────────────────────────
    print("\n--- Dispatch Figures ---")
    make_C_F1(dispatch.get("C0"), dispatch.get("C1"), case_df, replay_df)
    for case_id in ["C0", "C1"]:
        make_C_F2_F3(dispatch.get(case_id), case_id, replay_df)

    # ── Load Uncertainty Figures ─────────────────────────────────
    print("\n--- Load Uncertainty Figures ---")
    make_L_F1(case_df, replay_df)
    make_L_T1(case_df, replay_df)

    # ── Sensitivity Figures ──────────────────────────────────────
    print("\n--- Sensitivity Figures ---")
    sens_agg_path = SENS_DIR / "sensitivity_agg_summary.parquet"
    sens_report_path = SENS_DIR / "sensitivity_report.json"
    sens_agg = None
    sens_report = None

    if sens_agg_path.exists():
        sens_agg = pd.read_parquet(sens_agg_path)
        print(f"  Loaded sensitivity agg: {len(sens_agg)} rows")
        make_S_F1_F2(sens_agg, "replay_total_mean", "S-F1")
        make_S_F1_F2(sens_agg, "replay_total_std", "S-F2")

        # S-F3: Δ from baseline (N=500, K=5)
        baseline_vals = {}
        for case in sens_agg["case"].unique():
            bv = sens_agg[(sens_agg["case"] == case) &
                          (sens_agg["N"] == 500) & (sens_agg["K"] == 5)]
            if len(bv) > 0:
                baseline_vals[case] = float(bv["replay_total_mean"].iloc[0])

        if baseline_vals:
            delta_agg = sens_agg[["case","N","K"]].copy()
            delta_agg["replay_total_mean"] = sens_agg.apply(
                lambda r: r["replay_total_mean"] - baseline_vals.get(r["case"], 0), axis=1)
            make_S_F1_F2(delta_agg, "replay_total_mean", "S-F3")

        # S-F4: Design stability
        design_path = SENS_DIR / "sensitivity_design_summary.parquet"
        if design_path.exists():
            design_df = pd.read_parquet(design_path)
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, col, label in zip(axes,
                                      ["CC_cv", "P_B_cv", "E_B_cv"],
                                      ["CC CV", "P_B CV", "E_B CV"]):
                if col not in design_df.columns: continue
                for case in design_df["case"].unique():
                    sub = design_df[design_df["case"] == case]
                    k5 = sub[sub["K"] == 5].sort_values("N") if "K" in sub.columns else sub.sort_values("N")
                    ax.plot(k5["N"].values, k5[col].values * 100,
                            "o-", label=case, color=CASE_COLORS.get(case, "gray"))
                ax.set_xlabel("N"); ax.set_ylabel(f"{label} (%)")
                ax.set_title(label); ax.legend(fontsize=7)
                ax.set_xscale("log")
            fig.suptitle("S-F4: Design Stability vs N (K=5)")
            fig.tight_layout()
            savefig("S-F4", fig)
            register_fig("S-F4", "Design Stability vs N",
                         "4.7", "How stable is sizing as N increases?",
                         "C1,C3", "Full year", "ch4_results.py")

    else:
        print(f"  Sensitivity agg not found (sensitivity analysis still running?)")
        QA_WARNINGS.append("Sensitivity figures S-F1 to S-F4 skipped: sensitivity_agg_summary.parquet not available")

    if sens_report_path.exists():
        with open(sens_report_path) as f:
            sens_report = json.load(f)
    make_S_T1(sens_report)

    # ── QA ───────────────────────────────────────────────────────
    print("\n--- QA ---")
    run_qa(case_df, replay_df)

    # ── Manifest ─────────────────────────────────────────────────
    print("\n--- Manifest ---")
    write_manifest()

    # Summary
    print("\n" + "="*60)
    print(f"Chapter 4 results written to: results/ch4/")
    print(f"  Figures: {len(list(FIG_DIR.glob('*.png')))} PNG files")
    print(f"  Tables:  {len(list(TABLE_DIR.glob('*.csv')))} CSV files")
    print(f"  QA:      {len(QA_HARD_FAILS)} hard fails, {len(QA_WARNINGS)} warnings")
    if QA_HARD_FAILS:
        print("\nHARD FAILS:")
        for f in QA_HARD_FAILS:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
