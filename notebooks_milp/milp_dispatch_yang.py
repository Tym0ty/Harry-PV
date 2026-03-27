"""
Yang-Style Stacked-Bar Dispatch Plots: C0 vs C1 Side-by-Side.

Produces publication-quality figures with:
  - Stacked bars: Grid purchase, BESS discharge, PV-to-load, BESS charge (negative)
  - Dual-axis lines: Demand (blue, left axis) + TOU price (red, right axis)
  - Horizontal axis: Period (h), 24 hours
  - Left axis: Power (kW)
  - Right axis: Price (NTD/kWh)

Two figures:
  1. dispatch_yang_summer_peak.png   — hottest summer weekday (high load + PV)
  2. dispatch_yang_overcontract.png  — billing-sensitive day (over-contract risk)

Run:  cd notebooks_milp && python milp_dispatch_yang.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from milp_common import get_config, get_tou_price

import gurobipy as gp
from gurobipy import GRB

# ── Paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = ROOT / "bridge_outputs_fullyear"
MILP_OUT = ROOT / "milp_outputs"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Selected days ────────────────────────────────────────────
# day_index 320 = 2025-09-16 (Tue), summer weekday, peak load 5179 kW
# day_index 214 = 2025-06-02 (Mon), summer weekday, load 4932 kW + PV 15541 kWh
SELECTED_DAYS = {
    "summer_peak": 320,
    "overcontract": 214,
}


# ── Load sizing ──────────────────────────────────────────────
def load_sizing():
    with open(MILP_OUT / "case_results_fullyear.json") as f:
        results = json.load(f)
    sizing = {}
    for item in results:
        cid = item["case"]
        if cid in ("C0", "C1"):
            sizing[cid] = {
                "CC": item["contract_kw"],
                "P_B": item["bess_p_kw"],
                "E_B": item["bess_e_kwh"],
            }
    return sizing


# ── Load truth data for a single day ────────────────────────
def load_truth_day(day_index):
    """Return (pv_kw[24], load_kw[24], calendar_day, month, dow)."""
    truth = pd.read_parquet(BRIDGE_DIR / "full_year_replay_truth_package.parquet")
    cal = pd.read_parquet(BRIDGE_DIR / "caseyear_calendar_manifest.parquet")

    day_truth = truth[truth["day_index"] == day_index].sort_values("hour_local")
    pv = day_truth["pv_realized_kw"].values  # length 24
    load = day_truth["load_realized_kw"].values

    cal_row = cal[cal["day_index"] == day_index].iloc[0]
    cd = pd.Timestamp(cal_row["calendar_day"])
    return pv, load, cd, cd.month, cd.day, cd.weekday()


# ── Load ingest scenarios for a day ──────────────────────────
def load_ingest_day(day_index, case_id):
    """Return list of dicts: [{pv_kw, load_kw, prob, scenario_id}, ...]."""
    if case_id == "C0":
        fname = "full_year_milp_ingest_pvdet_loaddet.parquet"
    else:
        fname = "full_year_milp_ingest_pvprob_loaddet.parquet"

    ingest = pd.read_parquet(BRIDGE_DIR / fname)
    day = ingest[ingest["day_index"] == day_index]
    scenarios = []
    for sid in sorted(day["scenario_id"].unique()):
        sdata = day[day["scenario_id"] == sid].sort_values("hour_local")
        scenarios.append({
            "pv_kw": sdata["pv_available_kw"].values,
            "load_kw": sdata["load_kw"].values,
            "prob": float(sdata["probability_pi"].iloc[0]),
            "scenario_id": sid,
        })
    return scenarios


# ── Single-day dispatch LP (fixed sizing) ────────────────────
def solve_single_day_dispatch(scenarios, tou_24, sizing, CFG):
    """Solve dispatch for one day, return probability-weighted dispatch arrays.

    Returns dict with keys:
        P_grid_load[24], P_pv_load[24], P_dis[24], P_ch[24],
        P_grid_ch[24], load[24], pv_avail[24], tou[24]
    All arrays are probability-weighted expectations.
    """
    n_sc = len(scenarios)
    n_hours = 24
    P_B = sizing["P_B"]
    E_B = sizing["E_B"]
    eta_ch = CFG["eff_charge"]
    eta_dis = CFG["eff_discharge"]
    soc_min = CFG["soc_min"]
    soc_max = CFG["soc_max"]
    soc_init = CFG["soc_init"]

    m = gp.Model("DayDispatch")
    m.Params.OutputFlag = 0

    keys = [(s, t) for s in range(n_sc) for t in range(n_hours)]

    P_gl = m.addVars(keys, lb=0, name="Pgl")
    P_gc = m.addVars(keys, lb=0, name="Pgc")
    P_pvl = m.addVars(keys, lb=0, name="Ppvl")
    P_pvc = m.addVars(keys, lb=0, name="Ppvc")
    P_curt = m.addVars(keys, lb=0, name="Pcurt")
    P_ch = m.addVars(keys, lb=0, ub=P_B, name="Pch")
    P_dis = m.addVars(keys, lb=0, ub=P_B, name="Pdis")
    E_soc = m.addVars(keys, lb=soc_min * E_B, ub=soc_max * E_B, name="E")

    for s in range(n_sc):
        sc = scenarios[s]
        pv = sc["pv_kw"]
        load = sc["load_kw"]
        for t in range(n_hours):
            k = (s, t)
            # Load balance
            m.addConstr(P_gl[k] + P_pvl[k] + P_dis[k] == load[t])
            # PV split
            m.addConstr(P_pvl[k] + P_pvc[k] + P_curt[k] == pv[t])
            # Charge balance
            m.addConstr(P_ch[k] == P_gc[k] + P_pvc[k])
            # SOC dynamics
            E_prev = soc_init * E_B if t == 0 else E_soc[(s, t - 1)]
            m.addConstr(E_soc[k] == E_prev + eta_ch * P_ch[k] - P_dis[k] / eta_dis)

    # Objective: minimize expected energy cost
    obj = gp.LinExpr()
    for s in range(n_sc):
        prob = scenarios[s]["prob"]
        for t in range(n_hours):
            k = (s, t)
            obj += prob * tou_24[t] * (P_gl[k] + P_gc[k])
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        print(f"  WARNING: solve status {m.status}")
        return None

    # Extract probability-weighted expected dispatch
    result = {
        "P_grid_load": np.zeros(n_hours),
        "P_pv_load": np.zeros(n_hours),
        "P_dis": np.zeros(n_hours),
        "P_ch": np.zeros(n_hours),
        "P_grid_ch": np.zeros(n_hours),
        "P_grid_total": np.zeros(n_hours),
        "load": np.zeros(n_hours),
        "pv_avail": np.zeros(n_hours),
        "E_soc": np.zeros(n_hours),
    }

    for s in range(n_sc):
        prob = scenarios[s]["prob"]
        for t in range(n_hours):
            k = (s, t)
            result["P_grid_load"][t] += prob * P_gl[k].X
            result["P_pv_load"][t] += prob * P_pvl[k].X
            result["P_dis"][t] += prob * P_dis[k].X
            result["P_ch"][t] += prob * P_ch[k].X
            result["P_grid_ch"][t] += prob * P_gc[k].X
            result["P_grid_total"][t] += prob * (P_gl[k].X + P_gc[k].X)
            result["load"][t] += prob * scenarios[s]["load_kw"][t]
            result["pv_avail"][t] += prob * scenarios[s]["pv_kw"][t]
            result["E_soc"][t] += prob * E_soc[k].X

    result["tou"] = tou_24.copy()
    return result


# ── Yang-style stacked bar plot ──────────────────────────────
def plot_yang_dispatch(ax, dispatch, sizing, case_label, date_str):
    """Single-panel Yang-style stacked bar + dual lines.

    Bar stack (bottom-up, positive):
        Grid purchase (gray)
        BESS discharge (teal)
        PV to load (gold)
    Bar stack (negative):
        BESS charge (steelblue, shown below zero)
    Lines:
        Demand (blue, left axis)
        TOU price (red dashed, right axis)
    """
    hours = np.arange(24)
    bar_width = 0.8

    P_grid = dispatch["P_grid_load"]
    P_pvl = dispatch["P_pv_load"]
    P_dis = dispatch["P_dis"]
    P_ch = dispatch["P_ch"]
    load = dispatch["load"]
    tou = dispatch["tou"]
    CC = sizing["CC"]

    # --- Positive stacked bars ---
    # Order: Grid purchase (bottom), BESS discharge, PV to load (top)
    ax.bar(hours, P_grid, bar_width, bottom=0,
           color="#A0A0A0", edgecolor="white", linewidth=0.3,
           label="Grid Purchase", zorder=2)
    ax.bar(hours, P_dis, bar_width, bottom=P_grid,
           color="#2CA02C", edgecolor="white", linewidth=0.3,
           label="BESS Discharge", zorder=2)
    ax.bar(hours, P_pvl, bar_width, bottom=P_grid + P_dis,
           color="#FFD700", edgecolor="white", linewidth=0.3,
           label="PV to Load", zorder=2)

    # --- Negative bars: BESS charge ---
    ax.bar(hours, -P_ch, bar_width, bottom=0,
           color="#1F77B4", edgecolor="white", linewidth=0.3,
           label="BESS Charge", zorder=2)

    # --- Demand line (left axis) ---
    ax.plot(hours, load, color="#0000CC", linewidth=2.0, marker="o",
            markersize=3, label="Demand", zorder=4)

    # --- Contract capacity reference ---
    ax.axhline(y=CC, color="crimson", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"CC = {CC:.0f} kW", zorder=3)

    # --- Zero line ---
    ax.axhline(y=0, color="black", linewidth=0.5, zorder=1)

    # --- Right axis: TOU price ---
    ax2 = ax.twinx()
    ax2.plot(hours, tou, color="red", linewidth=1.8, linestyle="--",
             marker="s", markersize=2.5, alpha=0.8, label="TOU Price", zorder=3)
    ax2.set_ylabel("Price (NTD/kWh)", fontsize=11, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 12)

    # --- Axis formatting ---
    ax.set_xlabel("Period (h)", fontsize=11)
    ax.set_ylabel("Power (kW)", fontsize=11)
    ax.set_xlim(-0.6, 23.6)
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xticklabels([f"{h}" for h in range(0, 24, 2)])

    # Auto y-limits with margin
    ymax = max(np.max(P_grid + P_dis + P_pvl), np.max(load), CC) * 1.15
    ymin = -np.max(P_ch) * 1.3 if np.max(P_ch) > 0 else -200
    ax.set_ylim(ymin, ymax)

    ax.set_title(f"{case_label}\n{date_str}", fontsize=12, fontweight="bold")

    # --- Combined legend ---
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Desired order: Grid, BESS dis, PV to load, BESS ch, Demand, CC, Price
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper left", fontsize=7.5, ncol=2,
              framealpha=0.9, edgecolor="gray")

    return ax2


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Yang-Style Dispatch Plots: C0 vs C1")
    print("=" * 60)

    CFG = get_config()
    sizing = load_sizing()
    print(f"C0 sizing: CC={sizing['C0']['CC']:.0f}, P_B={sizing['C0']['P_B']:.0f}, E_B={sizing['C0']['E_B']:.0f}")
    print(f"C1 sizing: CC={sizing['C1']['CC']:.0f}, P_B={sizing['C1']['P_B']:.0f}, E_B={sizing['C1']['E_B']:.0f}")

    for fig_key, day_index in SELECTED_DAYS.items():
        print(f"\n--- Figure: {fig_key} (day_index={day_index}) ---")

        # Load truth data for date info
        pv_truth, load_truth, cd, month, day, dow = load_truth_day(day_index)
        date_str = cd.strftime("%Y-%m-%d (%a)")
        print(f"  Date: {date_str}, peak_load={load_truth.max():.0f} kW, "
              f"total_PV={pv_truth.sum():.0f} kWh")

        # TOU prices for this day (hour_0based = 0..23)
        tou_24 = np.array([get_tou_price(month, day, dow, h) for h in range(24)])

        # Solve dispatch for C0 and C1
        dispatches = {}
        for case_id in ("C0", "C1"):
            print(f"  Solving {case_id} dispatch...")
            scenarios = load_ingest_day(day_index, case_id)
            print(f"    {len(scenarios)} scenario(s), "
                  f"probs={[s['prob'] for s in scenarios]}")
            disp = solve_single_day_dispatch(scenarios, tou_24, sizing[case_id], CFG)
            if disp is None:
                print(f"    FAILED!")
                return
            dispatches[case_id] = disp
            print(f"    peak_grid={disp['P_grid_total'].max():.0f} kW, "
                  f"total_ch={disp['P_ch'].sum():.0f} kWh, "
                  f"total_dis={disp['P_dis'].sum():.0f} kWh")

        # ── Create side-by-side figure ───────────────────────
        fig, (ax_c0, ax_c1) = plt.subplots(1, 2, figsize=(16, 6))

        plot_yang_dispatch(ax_c0, dispatches["C0"], sizing["C0"],
                           "C0: Deterministic PV", date_str)
        plot_yang_dispatch(ax_c1, dispatches["C1"], sizing["C1"],
                           "C1: Probabilistic PV (E[dispatch])", date_str)

        # Match left-axis ylim across panels
        ymin = min(ax_c0.get_ylim()[0], ax_c1.get_ylim()[0])
        ymax = max(ax_c0.get_ylim()[1], ax_c1.get_ylim()[1])
        ax_c0.set_ylim(ymin, ymax)
        ax_c1.set_ylim(ymin, ymax)

        fig.suptitle(
            f"Dispatch Comparison — {fig_key.replace('_', ' ').title()}\n"
            f"{date_str}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        out_path = FIG_DIR / f"dispatch_yang_{fig_key}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
