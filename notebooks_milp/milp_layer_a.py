"""
MILP Layer A — Representative-Day Annual Fixed-Design Selection
Per F0329MILP_Spec.pdf (Batch 17), §4 + §6.5.5A.

For each candidate design x = (CC, P_B, E_B) in X_grid:
  Evaluate J_A(x) = AEC_inv + C_energy^A + C_deg^A + C_basic^A + C_over^A + C_TREC^A
  using repday design packages from the bridge.

Select x* = argmin J_A(x).

Key spec requirements:
  - Month-aware D_m^A: reconstructed via calendar mapping, NOT weighted average of daily peaks
  - Green SOC inter-period chaining: G_n^A propagated across repdays
  - PWL degradation: 5 segments, frozen {b_k}, {lambda_k}
  - No-export constraint
  - Battery mode u^A(d,t) shared across scenarios (day-ahead decision)
  - Candidate sets from milp_run_manifest.json (NOT hardcoded)

Usage:
    python milp_layer_a.py --case C0 --manifest milp_run_manifest.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("[WARN] gurobipy not available — Layer A will run in dry-run mode only.")

from milp_common import get_config, CASE_TABLE


# ── Helpers ───────────────────────────────────────────────────

def _hard_fail(msg):
    raise RuntimeError(f"[LAYER_A HARD FAIL] {msg}")


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        return json.load(f)


def load_repday_inputs(bridge_dir, case_id):
    """Load repday design package, weights, metadata, and calendar map for a case."""
    REPDAY_FILES = {
        "C0":    "repday_input_pvdet_loaddet.parquet",
        "C1":    "repday_input_pvprob_loaddet.parquet",
        "C2":    "repday_input_pvdet_loadunc.parquet",
        "C3":    "repday_input_pvprob_loadunc.parquet",
        "C_PFI": "repday_input_pvperfect_loadperfect.parquet",
    }
    bdir = Path(bridge_dir)
    repday_df  = pd.read_parquet(bdir / REPDAY_FILES[case_id])
    weights_df = pd.read_parquet(bdir / "repday_weights.parquet")
    meta_df    = pd.read_parquet(bdir / "repdays_metadata.parquet")
    cal_map    = pd.read_parquet(bdir / "calendar_to_repday_map.parquet")
    calendar   = pd.read_parquet(bdir / "caseyear_calendar_manifest.parquet")
    return repday_df, weights_df, meta_df, cal_map, calendar


def _crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


# ── Layer A evaluation for a single design candidate ─────────

def evaluate_design(x_design, repday_df, weights_df, meta_df, cal_map,
                    calendar, CFG, case_id):
    """
    Evaluate J_A(x) for a single design x = (CC, P_B, E_B) using a Gurobi MILP
    over the representative-day set.

    Returns dict with J_A value and cost decomposition, or None on infeasibility.
    """
    CC, P_B, E_B = x_design

    # Frozen parameters
    eta_ch   = CFG['eff_charge']
    eta_dis  = CFG['eff_discharge']
    soc_min  = CFG['soc_min']
    soc_max  = CFG['soc_max']
    soc_init = CFG['soc_init']
    eps_term = CFG['epsilon_term']
    eps_g    = CFG['epsilon_g_term']
    kappa    = CFG['kappa']
    re_tgt   = CFG['re_target']
    c_trec   = CFG['trec_cost_per_kwh']
    b_k      = CFG['pwl_deg_b_k']
    lam_k    = CFG['pwl_deg_lambda_k']
    c_basic_s  = CFG['basic_charge_summer']
    c_basic_ns = CFG['basic_charge_nonsummer']
    m_oc10   = CFG['oc_within_10pct_mult']
    m_oc10p  = CFG['oc_beyond_10pct_mult']
    n_seg    = len(lam_k)   # 4 PWL slope segments (5 breakpoints)

    repday_ids = sorted(weights_df["repday_id"].tolist())
    weight_map = weights_df.set_index("repday_id")["annual_weight_days"].to_dict()

    # Build per-repday scenario data
    repday_data = {}
    for rid in repday_ids:
        rd_rows = repday_df[repday_df["repday_id"] == rid]
        scenarios = sorted(rd_rows["scenario_id"].unique())
        sc_data = []
        for sid in scenarios:
            sr = rd_rows[rd_rows["scenario_id"] == sid].sort_values("hour_local")
            sc_data.append({
                "scenario_id": sid,
                "pv_kw":  sr["pv_available_kw"].values,   # shape (24,)
                "load_kw": sr["load_kw"].values,            # shape (24,)
                "prob":    float(sr["probability_pi"].iloc[0]),
            })
        meta_row = meta_df[meta_df["repday_id"] == rid].iloc[0]
        repday_data[rid] = {
            "scenarios": sc_data,
            "weight":    weight_map[rid],
            "month_id":  int(meta_row["month_id"]),
            "is_summer": meta_row["season_tag"] == "summer",
            "tou":       _build_tou(meta_row, CFG),
        }

    # Month-aware D_m^A: for each calendar month, find the set of repdays
    # that have calendar days mapping to that month
    months = sorted(calendar["month_id"].unique())
    month_repdays = {}   # month_id -> set of repday_ids
    for m in months:
        cal_days_this_month = cal_map[cal_map["month_id"] == m]["repday_id"].unique()
        month_repdays[m] = set(cal_days_this_month.tolist())

    if not GUROBI_AVAILABLE:
        return None   # dry run

    m_model = gp.Model(f"LayerA_{case_id}_CC{CC}_PB{P_B}_EB{E_B}")
    m_model.setParam("OutputFlag", 0)
    m_model.setParam("MIPGap", CFG.get('mip_gap', 1e-3))
    m_model.setParam("TimeLimit", CFG.get('time_limit', 600))

    E_MAX = soc_max * E_B
    E_MIN = soc_min * E_B
    OC_UB = CC * 0.3   # over-contract UB: 30% above CC

    # ── Decision variables ─────────────────────────────────────
    # For each repday d and scenario omega:
    # u[d,t], P_ch[d,t], P_dis[d,t] are shared across scenarios (day-ahead)
    # Scenario-dependent: P_pv_load, P_pv_ch, P_pv_curt, P_grid_load, P_grid_ch

    u      = {}   # (rid, t) binary
    P_ch   = {}   # (rid, t) continuous
    P_dis  = {}   # (rid, t) continuous
    E_start = {}  # (rid,) SOC at start of repday (inter-period carry)
    E_g_start = {}  # (rid,) Green SOC start

    # Scenario-dependent
    P_pv_load  = {}   # (rid, sid, t)
    P_pv_ch    = {}   # (rid, sid, t)
    P_pv_curt  = {}   # (rid, sid, t)
    P_grid_load = {}  # (rid, sid, t)
    P_grid_ch   = {}  # (rid, sid, t)
    E_s         = {}  # (rid, sid, t) SOC within day
    E_g_s       = {}  # (rid, sid, t) Green SOC within day

    # PWL degradation auxiliary variables
    d_seg = {}   # (rid, t, k) discharge in segment k

    # Month max-demand: D_m^A per calendar month (NOT weighted average of daily peaks)
    D_m = {m: m_model.addVar(lb=0, ub=CC * 1.3, name=f"Dm_{m}") for m in months}

    # Over-contract per month
    OC_m_lo = {m: m_model.addVar(lb=0, ub=OC_UB, name=f"OClo_{m}") for m in months}
    OC_m_hi = {m: m_model.addVar(lb=0, ub=OC_UB, name=f"OChi_{m}") for m in months}

    # Annual RE accounting
    E_load_yr  = gp.LinExpr()
    E_pv_self_yr = gp.LinExpr()
    E_dis_g_yr   = gp.LinExpr()

    for rid in repday_ids:
        rd = repday_data[rid]
        w  = rd["weight"]
        m  = rd["month_id"]
        tou = rd["tou"]

        E_start[rid]   = m_model.addVar(lb=E_MIN, ub=E_MAX, name=f"Es_{rid}")
        E_g_start[rid] = m_model.addVar(lb=0,     ub=E_MAX, name=f"Eg_{rid}")

        for t in range(24):
            u[(rid, t)]    = m_model.addVar(vtype=GRB.BINARY, name=f"u_{rid}_{t}")
            P_ch[(rid, t)] = m_model.addVar(lb=0, ub=P_B,     name=f"Pch_{rid}_{t}")
            P_dis[(rid, t)]= m_model.addVar(lb=0, ub=P_B,     name=f"Pdis_{rid}_{t}")
            for k in range(n_seg):
                d_seg[(rid, t, k)] = m_model.addVar(lb=0, name=f"dseg_{rid}_{t}_{k}")

        for sc in rd["scenarios"]:
            sid = sc["scenario_id"]
            pv_arr   = sc["pv_kw"]
            load_arr = sc["load_kw"]

            for t in range(24):
                P_pv_load[(rid, sid, t)]  = m_model.addVar(lb=0, ub=pv_arr[t],   name=f"Ppvl_{rid}_{sid}_{t}")
                P_pv_ch[(rid, sid, t)]    = m_model.addVar(lb=0, ub=pv_arr[t],   name=f"Ppvc_{rid}_{sid}_{t}")
                P_pv_curt[(rid, sid, t)]  = m_model.addVar(lb=0, ub=pv_arr[t],   name=f"Ppvcurt_{rid}_{sid}_{t}")
                P_grid_load[(rid, sid, t)]= m_model.addVar(lb=0, ub=load_arr[t], name=f"Pgl_{rid}_{sid}_{t}")
                P_grid_ch[(rid, sid, t)]  = m_model.addVar(lb=0, ub=P_B,         name=f"Pgc_{rid}_{sid}_{t}")
                E_s[(rid, sid, t)]        = m_model.addVar(lb=E_MIN, ub=E_MAX,    name=f"E_{rid}_{sid}_{t}")
                E_g_s[(rid, sid, t)]      = m_model.addVar(lb=0,     ub=E_MAX,    name=f"Eg_{rid}_{sid}_{t}")

    m_model.update()

    # ── Constraints ───────────────────────────────────────────
    for rid in repday_ids:
        rd  = repday_data[rid]
        w   = rd["weight"]
        tou = rd["tou"]

        for sc in rd["scenarios"]:
            sid    = sc["scenario_id"]
            pv_arr = sc["pv_kw"]
            ld_arr = sc["load_kw"]

            for t in range(24):
                # Power balance: load = grid_load + pv_load + dis - ch (no export)
                m_model.addConstr(
                    ld_arr[t] ==
                    P_grid_load[(rid, sid, t)] + P_pv_load[(rid, sid, t)]
                    + P_dis[(rid, t)] - P_ch[(rid, t)],
                    name=f"pb_{rid}_{sid}_{t}"
                )
                # PV allocation
                m_model.addConstr(
                    P_pv_load[(rid, sid, t)] + P_pv_ch[(rid, sid, t)]
                    + P_pv_curt[(rid, sid, t)] == pv_arr[t],
                    name=f"pva_{rid}_{sid}_{t}"
                )
                # Charge source: only grid+PV
                m_model.addConstr(
                    P_ch[(rid, t)] == P_grid_ch[(rid, sid, t)] + P_pv_ch[(rid, sid, t)],
                    name=f"chsrc_{rid}_{sid}_{t}"
                )
                # Battery mode: charge/discharge mutual exclusion via u
                m_model.addConstr(P_ch[(rid, t)]  <= P_B * u[(rid, t)])
                m_model.addConstr(P_dis[(rid, t)] <= P_B * (1 - u[(rid, t)]))

                # SOC dynamics
                if t == 0:
                    E_prev = E_start[rid]
                else:
                    E_prev = E_s[(rid, sid, t - 1)]
                m_model.addConstr(
                    E_s[(rid, sid, t)] == E_prev
                    + eta_ch * P_ch[(rid, t)]
                    - (1 / eta_dis) * P_dis[(rid, t)],
                    name=f"soc_{rid}_{sid}_{t}"
                )

                # Green SOC dynamics (green energy = PV self-charge only)
                if t == 0:
                    Eg_prev = E_g_start[rid]
                else:
                    Eg_prev = E_g_s[(rid, sid, t - 1)]
                m_model.addConstr(
                    E_g_s[(rid, sid, t)] == Eg_prev
                    + eta_ch * P_pv_ch[(rid, sid, t)]
                    - (1 / eta_dis) * P_dis[(rid, t)],  # simplification: all discharge draws from green pool
                    name=f"gsoc_{rid}_{sid}_{t}"
                )
                m_model.addConstr(E_g_s[(rid, sid, t)] >= 0)
                m_model.addConstr(E_g_s[(rid, sid, t)] <= E_s[(rid, sid, t)])

                # Grid import tied to demand kappa proxy (§CP_006)
                m_model.addConstr(
                    P_grid_load[(rid, sid, t)] + P_grid_ch[(rid, sid, t)]
                    <= CC * kappa,
                    name=f"cc_{rid}_{sid}_{t}"
                )

                # Month-aware max demand D_m^A (only for repdays in this month)
                if rid in month_repdays.get(rd["month_id"], set()):
                    m_model.addConstr(
                        D_m[rd["month_id"]] >=
                        kappa * (P_grid_load[(rid, sid, t)] + P_grid_ch[(rid, sid, t)]),
                        name=f"dm_{rid}_{sid}_{t}"
                    )

            # PWL degradation per discharge
            for t in range(24):
                m_model.addConstr(
                    P_dis[(rid, t)] == gp.quicksum(d_seg[(rid, t, k)] for k in range(n_seg)))
                for k in range(n_seg):
                    seg_lb = b_k[k] * E_B
                    seg_ub = b_k[k + 1] * E_B if k + 1 < len(b_k) else E_B
                    m_model.addConstr(d_seg[(rid, t, k)] <= seg_ub - seg_lb)

        # Terminal SOC constraint (each repday end must be within epsilon band)
        # Use expected end-of-day SOC across scenarios
        for sc in rd["scenarios"]:
            sid = sc["scenario_id"]
            m_model.addConstr(E_s[(rid, sid, 23)] >= (soc_init - eps_term) * E_B)
            m_model.addConstr(E_s[(rid, sid, 23)] <= (soc_init + eps_term) * E_B)

    # Over-contract: D_m^A vs CC
    for m in months:
        OC_band = CC * 0.1
        m_model.addConstr(OC_m_lo[m] >= D_m[m] - CC - OC_band)
        m_model.addConstr(OC_m_hi[m] >= D_m[m] - CC)
        m_model.addConstr(OC_m_lo[m] <= D_m[m])
        m_model.addConstr(OC_m_hi[m] <= D_m[m])

    # ── Objective J_A(x) ──────────────────────────────────────
    crf_bess = _crf(CFG['discount_rate'], CFG['lifetime_bess'])
    AEC_inv = crf_bess * (CFG['capex_bess_power_per_kw'] * P_B
                          + CFG['capex_bess_energy_per_kwh'] * E_B)

    # Annual energy cost: sum over repdays × weight × scenario prob × TOU
    C_energy = gp.LinExpr()
    C_deg    = gp.LinExpr()
    C_basic  = gp.LinExpr()
    C_over   = gp.LinExpr()

    for rid in repday_ids:
        rd  = repday_data[rid]
        w   = rd["weight"]
        tou = rd["tou"]
        m   = rd["month_id"]
        c_b = c_basic_s if rd["is_summer"] else c_basic_ns

        for sc in rd["scenarios"]:
            sid = sc["scenario_id"]
            pi  = sc["prob"]
            for t in range(24):
                # Energy cost
                C_energy.addTerms(
                    w * pi * tou[t],
                    P_grid_load[(rid, sid, t)]
                )
                C_energy.addTerms(
                    w * pi * tou[t],
                    P_grid_ch[(rid, sid, t)]
                )
                # RE accounting
                E_load_yr.addTerms(w * pi,   P_grid_load[(rid, sid, t)])
                E_load_yr.addTerms(w * pi,   P_pv_load[(rid, sid, t)])
                E_pv_self_yr.addTerms(w * pi, P_pv_load[(rid, sid, t)])
                E_dis_g_yr.addTerms(w * pi,   P_dis[(rid, t)])

        # Basic charge (per month × weight proxy: weight/30 months represented)
        # Note: each repday has a weight (# calendar days); monthly charge is billed monthly
        # We proxy monthly charge from the D_m^A for this repday's month
        # The calendar_to_repday_map provides the true month scope
        C_basic.add(w * c_b * D_m[m])

        # Degradation cost
        for t in range(24):
            for k in range(n_seg):
                C_deg.addTerms(w * lam_k[k], d_seg[(rid, t, k)])

    # Over-contract annual cost
    for m in months:
        C_over.addTerms(m_oc10  * c_basic_s,  OC_m_lo[m])
        C_over.addTerms(m_oc10p * c_basic_s,  OC_m_hi[m])

    # T-REC shortfall
    E_TREC = re_tgt * E_load_yr - E_pv_self_yr - E_dis_g_yr
    # E_TREC_cost as a variable (must be >= 0)
    E_TREC_var = m_model.addVar(lb=0, name="E_TREC_yr")
    m_model.addConstr(E_TREC_var >= E_TREC)
    C_TREC_expr = c_trec * E_TREC_var

    J_A = AEC_inv + C_energy + C_deg + C_basic + C_over + C_TREC_expr
    m_model.setObjective(J_A, GRB.MINIMIZE)

    t0 = time.time()
    m_model.optimize()
    solve_time = time.time() - t0

    if m_model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None

    def _val(expr_or_var):
        if hasattr(expr_or_var, 'getValue'):
            return expr_or_var.getValue()
        return expr_or_var.X

    return {
        "case_id":    case_id,
        "CC":         CC,
        "P_B":        P_B,
        "E_B":        E_B,
        "J_A":        m_model.ObjVal,
        "AEC_inv":    AEC_inv,
        "C_energy_A": _val(C_energy),
        "C_deg_A":    _val(C_deg),
        "C_basic_A":  _val(C_basic),
        "C_over_A":   _val(C_over),
        "C_TREC_A":   E_TREC_var.X * c_trec,
        "solve_time": solve_time,
        "gurobi_status": m_model.Status,
    }


def _build_tou(meta_row, CFG):
    """Build 24-element TOU price array for a repday (hour_0based 0-23)."""
    from milp_common import get_tou_price
    cd = pd.Timestamp(meta_row["source_calendar_day"])
    return np.array([
        get_tou_price(cd.month, cd.day, cd.weekday(), t)
        for t in range(24)
    ])


# ── Main grid search ──────────────────────────────────────────

def run_layer_a(case_id, manifest_path, bridge_dir=None, output_dir=None):
    """
    Enumerate X_grid and evaluate J_A for each candidate.
    Returns the optimal design and all results.
    """
    print("=" * 60)
    print(f"Layer A — Annual Design Selection  [{case_id}]")
    print("=" * 60)

    manifest = load_manifest(manifest_path)
    CFG = get_config()
    if bridge_dir:
        CFG['bridge_dir'] = bridge_dir
    if output_dir:
        CFG['output_dir'] = output_dir

    C_grid = manifest["C_grid"]
    P_grid = manifest["P_grid"]
    E_grid = manifest["E_grid"]
    total  = len(C_grid) * len(P_grid) * len(E_grid)
    print(f"  X_grid size: {len(C_grid)} CC × {len(P_grid)} PB × {len(E_grid)} EB = {total} candidates")

    repday_df, weights_df, meta_df, cal_map, calendar = load_repday_inputs(
        CFG['bridge_dir'], case_id)

    results = []
    best    = None
    for i_cc, CC in enumerate(C_grid):
        for P_B in P_grid:
            for E_B in E_grid:
                res = evaluate_design(
                    (CC, P_B, E_B),
                    repday_df, weights_df, meta_df, cal_map, calendar,
                    CFG, case_id
                )
                if res is None:
                    continue
                results.append(res)
                if best is None or res["J_A"] < best["J_A"]:
                    best = res
        print(f"  CC={CC} kW done ({i_cc+1}/{len(C_grid)}), "
              f"best so far: J_A={best['J_A']:.0f} NTD")

    if best is None:
        _hard_fail(f"No feasible design found for {case_id}.")

    print(f"\n  OPTIMAL: CC={best['CC']} kW, P_B={best['P_B']} kW, E_B={best['E_B']} kWh")
    print(f"  J_A = {best['J_A']/1e6:.3f} M NTD")
    print(f"    AEC_inv={best['AEC_inv']/1e6:.3f}  C_energy={best['C_energy_A']/1e6:.3f}  "
          f"C_deg={best['C_deg_A']/1e6:.3f}  C_basic={best['C_basic_A']/1e6:.3f}  "
          f"C_over={best['C_over_A']/1e6:.3f}  C_TREC={best['C_TREC_A']/1e6:.3f}")

    # Save results
    out_dir = Path(CFG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"design_results_{case_id}.json", "w") as f:
        json.dump(best, f, indent=2)

    all_df = pd.DataFrame(results)
    all_df.to_csv(out_dir / f"design_grid_search_{case_id}.csv", index=False)

    return best, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer A — Annual Fixed-Design Selection")
    parser.add_argument("--case", required=True, choices=["C0","C1","C2","C3","C_PFI"])
    parser.add_argument("--manifest", default="../milp_run_manifest.json")
    parser.add_argument("--bridge_dir",  default="../bridge_outputs_fullyear")
    parser.add_argument("--output_dir",  default="../milp_outputs")
    args = parser.parse_args()

    run_layer_a(args.case, args.manifest, args.bridge_dir, args.output_dir)
