"""
milp_v2/layer_a/run_sizing_base.py
Phase 3A: Base sizing with realized PV + σ=8% load perturbation, K=1.

Package construction:
  - PV  : replay_truth.pv_realized_kw  (actual campus PV output, kW)
  - Load: replay_truth.load_realized_kw × (1 + ε_t), ε_t ~ N(0, 0.08²), seed=42
  - K   : 1 scenario (deterministic), probability_pi = 1.0

Annual MILP changes vs milp_annual.py:
  - Terminal year-end SOC constraint REMOVED (avoid over-constraint on design)
  - All other constraints/objective identical

Output:
  milp_v2/layer_a/designs/design_BASE.json
  milp_v2/layer_a/designs/sizing_result_base.json  (same content, alias)

Usage:
    cd milp_v2
    python layer_a/run_sizing_base.py [--time-limit 7200] [--mip-gap 0.005]
"""

import argparse, json, time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_basic_charge, get_tou_price

CFG     = load_config()
PKG_DIR = ROOT / CFG["paths"]["packages_dir"]
DES_DIR = ROOT / CFG["paths"]["designs_dir"]
DES_DIR.mkdir(parents=True, exist_ok=True)

LOAD_SIGMA = 0.08
LOAD_SEED  = 42
CASE_ID    = "BASE"


# ── Package builder ───────────────────────────────────────────────────────────

def build_base_package(load_sigma: float = LOAD_SIGMA,
                       load_seed: int = LOAD_SEED) -> tuple:
    """
    Load replay_truth, add Gaussian load perturbation, return structured day_data.

    ε_t ~ N(0, sigma²), seed=load_seed, clipped so load > 0.
    Returns (day_data, day_indices) in same format as milp_annual.load_annual_package.
    """
    rt_path = (ROOT / CFG["paths"]["upstream_truth"].replace("../", "")).resolve()
    if not rt_path.exists():
        # Try alternate path (relative to repo root)
        rt_path = ROOT.parent / "bridge_outputs_fullyear/full_year_replay_truth_package.parquet"
    rt = pd.read_parquet(rt_path)
    rt = rt.sort_values(["day_index", "hour_local"]).reset_index(drop=True)

    rng = np.random.default_rng(load_seed)
    n_rows = len(rt)
    eps    = rng.normal(0.0, load_sigma, size=n_rows)
    load_perturbed = rt["load_realized_kw"].values * (1.0 + eps)
    load_perturbed = np.maximum(load_perturbed, 0.0)  # clip to non-negative

    rt["load_input_kw"]    = load_perturbed
    rt["pv_available_kw"]  = rt["pv_realized_kw"].clip(lower=0.0)

    day_indices = sorted(rt["day_index"].unique().tolist())
    day_data    = {}

    for di in day_indices:
        d_di = rt[rt["day_index"] == di].sort_values("hour_local")
        assert len(d_di) == 24, f"Day {di} has {len(d_di)} rows (expected 24)"

        month_id = int(d_di["month_id"].iloc[0])
        cal_day  = pd.Timestamp(d_di["calendar_day"].iloc[0])
        dow      = cal_day.weekday()
        tou      = np.array([get_tou_price(cal_day.month, cal_day.day, dow, h)
                              for h in range(24)])

        pv_arr   = d_di["pv_available_kw"].values.astype(float)
        load_arr = d_di["load_input_kw"].values.astype(float)

        day_data[di] = {
            "scenarios":    [{"id": "det", "pi": 1.0, "pv": pv_arr, "load": load_arr}],
            "tou":          tou,
            "month_id":     month_id,
            "calendar_day": cal_day,
        }

    print(f"  Package built: {len(day_data)} days  "
          f"load_sigma={load_sigma}  seed={load_seed}")
    print(f"  PV  range: [{rt['pv_available_kw'].min():.1f}, "
          f"{rt['pv_available_kw'].max():.1f}] kW")
    print(f"  Load range: [{load_perturbed.min():.1f}, "
          f"{load_perturbed.max():.1f}] kW (perturbed)")

    return day_data, day_indices


# ── Annual MILP — no terminal SOC constraint ──────────────────────────────────

def solve_base_sizing(day_data: dict, day_indices: list,
                      time_limit: int = 7200, mip_gap: float = 0.005) -> dict:
    """
    Annual MILP sizing with realized PV + perturbed load, K=1.
    Terminal year-end SOC constraint removed.
    """
    cfg    = CFG
    bat    = cfg["battery"]
    dem    = cfg["demand"]
    re20   = cfg["re20"]
    bounds = cfg["milp_bounds"]

    eta_ch   = bat["eta_ch"]
    eta_dis  = bat["eta_dis"]
    soc_min  = bat["soc_min"]
    soc_max  = bat["soc_max"]
    soc_init = bat["soc_init"]
    C_BP     = bat["C_BP_ntd_per_kw"]
    C_BE     = bat["C_BE_ntd_per_kwh"]
    CRF      = bat["crf"]
    b_k      = bat["degradation_segments"]["b"]
    lam_k    = bat["degradation_segments"]["lambda"]
    n_seg    = len(lam_k)

    kappa  = dem["kappa"]
    m1     = dem["over_contract_m1"]
    m2     = dem["over_contract_m2"]
    tier1  = dem["over_contract_tier1_ratio"]

    re_tgt = re20["re_target_ratio"]
    c_trec = re20["c_trec_ntd_per_kwh"]

    CC_lb  = bounds["CC_lb"];  CC_ub  = bounds["CC_ub"]
    PB_lb  = bounds["PB_lb"];  PB_ub  = bounds["PB_ub"]
    EB_lb  = bounds["EB_lb"];  EB_ub  = bounds["EB_ub"]
    EP_min = bounds["EP_ratio_min"]
    EP_max = bounds["EP_ratio_max"]

    n_days = len(day_indices)
    I      = list(range(n_days))
    di_map = {i: day_indices[i] for i in I}

    months_dict: dict = {}
    for i in I:
        mo = day_data[di_map[i]]["month_id"]
        months_dict.setdefault(mo, []).append(i)
    month_ids = sorted(months_dict.keys())

    print(f"\n[Layer A BASE] K=1  days={n_days}  months={len(month_ids)}")

    t0_build = time.time()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.setParam("LogToConsole", 1)
    env.start()
    mdl = gp.Model(f"annual_{CASE_ID}", env=env)
    mdl.setParam("MIPGap",        mip_gap)
    mdl.setParam("TimeLimit",     time_limit)
    mdl.setParam("Threads",       0)
    mdl.setParam("NodefileStart", 0.5)

    # ── Design variables ──────────────────────────────────────────────────────
    CC = mdl.addVar(lb=CC_lb, ub=CC_ub, name="CC")
    PB = mdl.addVar(lb=PB_lb, ub=PB_ub, name="PB")
    EB = mdl.addVar(lb=EB_lb, ub=EB_ub, name="EB")
    mdl.addConstr(EB >= EP_min * PB, name="EP_min")
    mdl.addConstr(EB <= EP_max * PB, name="EP_max")

    # ── Battery control (K=1: no scenario index needed) ───────────────────────
    u     = mdl.addVars(n_days, 24, vtype=GRB.BINARY, name="u")
    P_ch  = mdl.addVars(n_days, 24, lb=0.0, name="P_ch")
    P_dis = mdl.addVars(n_days, 24, lb=0.0, name="P_dis")
    E     = mdl.addVars(n_days, 24, lb=0.0, ub=soc_max * EB_ub, name="E")
    e_seg = mdl.addVars(n_days, 24, n_seg, lb=0.0, name="e_seg")

    # ── Monthly demand ────────────────────────────────────────────────────────
    D_m    = mdl.addVars(month_ids, lb=0.0, name="D_m")
    Over_m = mdl.addVars(month_ids, lb=0.0, name="Over_m")
    O1_m   = mdl.addVars(month_ids, lb=0.0, name="O1_m")
    O2_m   = mdl.addVars(month_ids, lb=0.0, name="O2_m")
    E_TREC_yr = mdl.addVar(lb=0.0, name="E_TREC_yr")
    E_g_bar   = mdl.addVars(n_days, lb=0.0, ub=EB_ub, name="Egbar")

    # ── Scenario flows (K=1, w=0 only) ───────────────────────────────────────
    P_grid     = mdl.addVars(n_days, 24, lb=0.0, name="Pg")
    P_grid_load= mdl.addVars(n_days, 24, lb=0.0, name="Pgl")
    P_grid_ch  = mdl.addVars(n_days, 24, lb=0.0, name="Pgc")
    P_pv_load  = mdl.addVars(n_days, 24, lb=0.0, name="Pvl")
    P_pv_ch    = mdl.addVars(n_days, 24, lb=0.0, name="Pvc")
    P_pv_curt  = mdl.addVars(n_days, 24, lb=0.0, name="Pvcu")
    E_g        = mdl.addVars(n_days, 24, lb=0.0, ub=EB_ub, name="Eg")
    P_ch_g     = mdl.addVars(n_days, 24, lb=0.0, name="Pcg")
    P_dis_g    = mdl.addVars(n_days, 24, lb=0.0, name="Pdg")

    mdl.update()
    print(f"  Variables: {mdl.NumVars} total  ({mdl.NumBinVars} binary)")

    # ── Constraints ───────────────────────────────────────────────────────────

    # (B) Battery limits
    for i in I:
        for h in range(24):
            mdl.addConstr(P_ch[i,h]  <= PB_ub * u[i,h])
            mdl.addConstr(P_ch[i,h]  <= PB)
            mdl.addConstr(P_dis[i,h] <= PB_ub * (1 - u[i,h]))
            mdl.addConstr(P_dis[i,h] <= PB)

    # (C) SOC evolution + bounds
    for i in I:
        for h in range(24):
            E_prev = (soc_init * EB) if (i == 0 and h == 0) else \
                     (E[i-1, 23]    if h == 0               else E[i, h-1])
            mdl.addConstr(E[i,h] == E_prev + eta_ch * P_ch[i,h]
                          - (1.0/eta_dis) * P_dis[i,h])
            mdl.addConstr(E[i,h] >= soc_min * EB)
            mdl.addConstr(E[i,h] <= soc_max * EB)

    # NOTE: Year-end terminal SOC constraint deliberately OMITTED per Phase 3A spec.

    # (G) Degradation segments
    for i in I:
        for h in range(24):
            mdl.addConstr(P_dis[i,h] == quicksum(e_seg[i,h,k] for k in range(n_seg)))
            for k in range(n_seg):
                mdl.addConstr(e_seg[i,h,k] <= (b_k[k+1] - b_k[k]) * EB)

    # (A) + (E) Power balance + Green SOC — K=1 (w=0 only)
    for i in I:
        di  = di_map[i]
        sc  = day_data[di]["scenarios"][0]   # single scenario
        Eg_carry = 0.0 if i == 0 else E_g_bar[i-1]

        for h in range(24):
            pv_v   = float(sc["pv"][h])
            load_v = float(sc["load"][h])
            # Power balance
            mdl.addConstr(P_grid_load[i,h] + P_pv_load[i,h] + P_dis[i,h] == load_v)
            mdl.addConstr(P_ch[i,h] == P_grid_ch[i,h] + P_pv_ch[i,h])
            mdl.addConstr(P_grid[i,h] == P_grid_load[i,h] + P_grid_ch[i,h])
            mdl.addConstr(P_pv_load[i,h] + P_pv_ch[i,h] + P_pv_curt[i,h] == pv_v)
            # Green SOC
            Eg_prev = (Eg_carry if h == 0 else E_g[i, h-1])
            mdl.addConstr(E_g[i,h] == Eg_prev + eta_ch * P_ch_g[i,h]
                          - (1.0/eta_dis) * P_dis_g[i,h])
            mdl.addConstr(E_g[i,h]    <= E[i,h])
            mdl.addConstr(P_ch_g[i,h]  <= P_pv_ch[i,h])
            mdl.addConstr(P_dis_g[i,h] <= P_dis[i,h])

        # E_g_bar[i] = green SOC at EOD (K=1: just E_g[i, 23])
        mdl.addConstr(E_g_bar[i] == E_g[i, 23])

    # (D) Monthly demand
    for mo, day_list_m in months_dict.items():
        for i in day_list_m:
            for h in range(24):
                mdl.addConstr(D_m[mo] >= kappa * P_grid[i, h])
        mdl.addConstr(Over_m[mo] >= D_m[mo] - CC)
        mdl.addConstr(Over_m[mo] == O1_m[mo] + O2_m[mo])
        mdl.addConstr(O1_m[mo] <= tier1 * CC)

    # (F) Annual RE20
    e_ren  = gp.LinExpr()
    e_load = 0.0
    for i in I:
        di  = di_map[i]
        sc  = day_data[di]["scenarios"][0]
        e_load += float(sum(sc["load"]))
        for h in range(24):
            e_ren += P_pv_load[i,h] + P_dis_g[i,h]
    mdl.addConstr(E_TREC_yr >= re_tgt * e_load - e_ren, name="RE20")

    # ── Objective ─────────────────────────────────────────────────────────────
    AEC_inv   = CRF * (C_BP * PB + C_BE * EB)

    C_ene = gp.LinExpr()
    for i in I:
        di  = di_map[i]
        tou = day_data[di]["tou"]
        for h in range(24):
            C_ene += tou[h] * P_grid[i, h]

    C_basic = gp.LinExpr()
    for mo in month_ids:
        C_basic += get_basic_charge(mo, cfg) * CC

    C_over = gp.LinExpr()
    for mo in month_ids:
        c_b = get_basic_charge(mo, cfg)
        C_over += c_b * (m1 * O1_m[mo] + m2 * O2_m[mo])

    C_deg = quicksum(
        lam_k[k] * e_seg[i,h,k]
        for i in I for h in range(24) for k in range(n_seg)
    )

    C_TREC = c_trec * E_TREC_yr

    mdl.setObjective(AEC_inv + C_ene + C_basic + C_over + C_deg + C_TREC,
                     GRB.MINIMIZE)

    t_build = time.time() - t0_build
    print(f"  Model built: {mdl.NumVars} vars, {mdl.NumConstrs} constrs  [{t_build:.1f}s]")
    print(f"  Solving (time_limit={time_limit}s, mip_gap={mip_gap*100:.1f}%) ...")

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0_solve = time.time()
    mdl.optimize()
    t_solve  = time.time() - t0_solve

    STATUS = {GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "TimeLimit",
              GRB.INFEASIBLE: "Infeasible", GRB.UNBOUNDED: "Unbounded"}
    status = STATUS.get(mdl.Status, f"Code{mdl.Status}")

    if mdl.SolCount == 0:
        raise RuntimeError(f"No feasible solution (status={status})")

    CC_v = round(CC.X, 2)
    PB_v = round(PB.X, 2)
    EB_v = round(EB.X, 2)
    J_v  = round(mdl.ObjVal / 1e6, 4)
    gap  = round(mdl.MIPGap, 6)

    AEC_inv_v = CRF * (C_BP * PB_v + C_BE * EB_v)
    C_basic_v = sum(get_basic_charge(mo, cfg) * CC_v for mo in month_ids)
    C_over_v  = sum(
        get_basic_charge(mo, cfg) * (m1 * O1_m[mo].X + m2 * O2_m[mo].X)
        for mo in month_ids
    )
    C_TREC_v  = c_trec * E_TREC_yr.X

    print(f"\n  ========== BASE SIZING RESULT ==========")
    print(f"  CC* = {CC_v:.1f} kW")
    print(f"  P_B* = {PB_v:.1f} kW")
    print(f"  E_B* = {EB_v:.1f} kWh")
    print(f"  J*  = {J_v:.4f} M NTD")
    print(f"  Status: {status}  MIP gap: {gap*100:.3f}%  Solve: {t_solve:.0f}s")
    print(f"  CAPEX={AEC_inv_v/1e6:.4f}  basic={C_basic_v/1e6:.4f}  "
          f"over={C_over_v/1e6:.4f}  TREC={C_TREC_v/1e6:.4f}")

    artifact = {
        "case_id":      CASE_ID,
        "description":  "K=1 sizing with realized PV + 8% Gaussian load perturbation (seed=42)",
        "input_spec":   {"pv_source": "realized", "load_sigma": LOAD_SIGMA,
                         "load_seed": LOAD_SEED, "K": 1,
                         "terminal_soc_constraint": False},
        "CC_star":      CC_v,
        "PB_star":      PB_v,
        "EB_star":      EB_v,
        "J_star_MNTD":  J_v,
        "cost_breakdown_MNTD": {
            "AEC_inv": round(AEC_inv_v / 1e6, 4),
            "C_basic": round(C_basic_v / 1e6, 4),
            "C_over":  round(C_over_v  / 1e6, 4),
            "C_TREC":  round(C_TREC_v  / 1e6, 4),
        },
        "solve_metadata": {
            "solver":           "gurobi",
            "mip_gap_achieved": gap,
            "mip_gap_target":   mip_gap,
            "solve_time_sec":   round(t_solve, 1),
            "build_time_sec":   round(t_build, 1),
            "n_vars":           mdl.NumVars,
            "n_binary":         mdl.NumBinVars,
            "n_constraints":    mdl.NumConstrs,
            "status":           status,
            "obj_val_NTD":      round(mdl.ObjVal, 0),
        },
        "reference_designs": {
            "DD": {"CC": 3262, "PB": 1103, "EB": 7177},
            "PI": {"CC": 3210, "PB": 1130, "EB": 7309},
        }
    }

    # Save as design_BASE.json (for run_layer_b*.py compatibility)
    p1 = DES_DIR / f"design_{CASE_ID}.json"
    with open(p1, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n  Saved -> {p1}")

    # Also save as sizing_result_base.json (per Phase 3A spec)
    p2 = DES_DIR / "sizing_result_base.json"
    with open(p2, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Saved -> {p2}")

    mdl.dispose()
    env.dispose()
    return artifact


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=int,   default=7200)
    ap.add_argument("--mip-gap",    type=float, default=0.005)
    ap.add_argument("--load-sigma", type=float, default=LOAD_SIGMA)
    ap.add_argument("--load-seed",  type=int,   default=LOAD_SEED)
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("Phase 3A: Base Sizing (realized PV + perturbed load, K=1)")
    print("=" * 60)

    day_data, day_indices = build_base_package(args.load_sigma, args.load_seed)
    solve_base_sizing(day_data, day_indices, args.time_limit, args.mip_gap)


if __name__ == "__main__":
    main()
