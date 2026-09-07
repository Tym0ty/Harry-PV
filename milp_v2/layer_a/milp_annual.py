"""
milp_v2/layer_a/milp_annual.py
Full-year annual MILP — Layer A solver.

Simultaneously optimises design (CC, PB, EB) and operation schedule.

Model characteristics:
  - Binary u[i,h]: 365*24 = 8,760
  - Continuous vars: ~450k (prob case K=5)
  - Constraints: ~550k (prob case)
  - All bilinear terms linearised exactly via big-M (PB_ub):
      P_ch[i,h] <= PB_ub * u[i,h]   AND   P_ch[i,h] <= PB
      P_dis[i,h] <= PB_ub*(1-u[i,h]) AND   P_dis[i,h] <= PB
  - e_seg[i,h,k] <= (b[k+1]-b[k])*EB  is linear (constant*EB_var)
  - Result: pure MILP, no bilinear/quadratic terms
"""

import json, time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_basic_charge, get_tou_price

CFG      = load_config()
PKG_DIR  = ROOT / CFG["paths"]["packages_dir"]
DES_DIR  = ROOT / CFG["paths"]["designs_dir"]
DES_DIR.mkdir(parents=True, exist_ok=True)


# ── Package loader ────────────────────────────────────────────────────────────

def load_annual_package(pkg_name: str) -> tuple:
    """
    Load a Layer A annual package parquet into structured day_data.

    Returns:
        day_data: dict[day_index] -> {scenarios, tou, month_id, calendar_day}
        day_indices: sorted list of day_index values
    """
    df = pd.read_parquet(PKG_DIR / pkg_name)
    day_indices  = sorted(df["day_index"].unique().tolist())
    scenario_ids = sorted(df["scenario_id"].unique().tolist())

    day_data = {}
    for di in day_indices:
        d_di = df[df["day_index"] == di]
        month_id  = int(d_di["month_id"].iloc[0])
        cal_day   = pd.Timestamp(d_di["calendar_day"].iloc[0])
        dow       = cal_day.weekday()
        tou       = np.array([get_tou_price(cal_day.month, cal_day.day, dow, h)
                               for h in range(24)])

        scenarios = []
        for sid in scenario_ids:
            s = d_di[d_di["scenario_id"] == sid].sort_values("hour_local")
            if len(s) == 0:
                continue
            pi   = float(s["probability_pi"].iloc[0])
            pv   = s["pv_available_kw"].values.astype(float)
            load = s["load_input_kw"].values.astype(float)
            scenarios.append({"id": sid, "pi": pi, "pv": pv, "load": load})

        # Normalise probabilities (guard against floating-point drift)
        total_pi = sum(sc["pi"] for sc in scenarios)
        if abs(total_pi - 1.0) > 1e-6:
            for sc in scenarios:
                sc["pi"] /= total_pi

        day_data[di] = {
            "scenarios":    scenarios,
            "tou":          tou,
            "month_id":     month_id,
            "calendar_day": cal_day,
        }

    return day_data, day_indices


# ── Annual MILP solver ────────────────────────────────────────────────────────

def solve_annual_milp(case_id: str, pkg_name: str, cfg: dict = None) -> dict:
    """
    Build and solve the full-year annual MILP for Layer A.

    Args:
        case_id:  'DD' | 'DP' | 'PD' | 'PP' | 'PI'
        pkg_name: filename in bridge/packages/ (e.g. 'layer_a_prob.parquet')
        cfg:      config dict (loaded from config.yaml if None)

    Returns:
        design artifact dict — also saved to layer_a/designs/design_{case_id}.json
    """
    if cfg is None:
        cfg = CFG

    bat    = cfg["battery"]
    dem    = cfg["demand"]
    re20   = cfg["re20"]
    bounds = cfg["milp_bounds"]
    solver = cfg["solver"]

    eta_ch   = bat["eta_ch"]
    eta_dis  = bat["eta_dis"]
    soc_min  = bat["soc_min"]
    soc_max  = bat["soc_max"]
    soc_init = bat["soc_init"]
    eps_term = bat["epsilon_term"]
    C_BP     = bat["C_BP_ntd_per_kw"]
    C_BE     = bat["C_BE_ntd_per_kwh"]
    CRF      = bat["crf"]
    b_k      = bat["degradation_segments"]["b"]      # 5 breakpoints
    lam_k    = bat["degradation_segments"]["lambda"]  # 4 lambdas
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

    # ── Load package ──────────────────────────────────────────────────────────
    day_data, day_indices = load_annual_package(pkg_name)
    n_days = len(day_indices)
    I      = list(range(n_days))           # 0-based day index
    di_map = {i: day_indices[i] for i in I}

    # Monthly grouping: month_id -> list of 0-based day indices
    months_dict: dict = {}
    for i in I:
        mo = day_data[di_map[i]]["month_id"]
        months_dict.setdefault(mo, []).append(i)
    month_ids = sorted(months_dict.keys())

    K_max = max(len(day_data[di_map[i]]["scenarios"]) for i in I)
    print(f"\n[Layer A] case={case_id}  pkg={pkg_name}")
    print(f"  days={n_days}  months={len(month_ids)}  K_max={K_max}")

    # ── Build model ───────────────────────────────────────────────────────────
    t0_build = time.time()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.setParam("LogToConsole", 1)
    env.start()
    mdl = gp.Model(f"annual_{case_id}", env=env)

    mdl.setParam("MIPGap",        solver["mip_gap"])
    mdl.setParam("TimeLimit",     solver["time_limit"])
    mdl.setParam("Threads",       solver["threads"])
    mdl.setParam("NodefileStart", solver["node_file_gb"])

    # ── Design variables ──────────────────────────────────────────────────────
    CC = mdl.addVar(lb=CC_lb, ub=CC_ub, name="CC")
    PB = mdl.addVar(lb=PB_lb, ub=PB_ub, name="PB")
    EB = mdl.addVar(lb=EB_lb, ub=EB_ub, name="EB")

    mdl.addConstr(EB >= EP_min * PB, name="EP_min")
    mdl.addConstr(EB <= EP_max * PB, name="EP_max")

    # ── Scenario-invariant battery control ────────────────────────────────────
    u     = mdl.addVars(n_days, 24, vtype=GRB.BINARY, name="u")
    P_ch  = mdl.addVars(n_days, 24, lb=0.0, name="P_ch")
    P_dis = mdl.addVars(n_days, 24, lb=0.0, name="P_dis")
    E     = mdl.addVars(n_days, 24, lb=0.0, ub=soc_max * EB_ub, name="E")
    e_seg = mdl.addVars(n_days, 24, n_seg, lb=0.0, name="e_seg")

    # ── Monthly demand variables ───────────────────────────────────────────────
    D_m    = mdl.addVars(month_ids, lb=0.0, name="D_m")
    Over_m = mdl.addVars(month_ids, lb=0.0, name="Over_m")
    O1_m   = mdl.addVars(month_ids, lb=0.0, name="O1_m")
    O2_m   = mdl.addVars(month_ids, lb=0.0, name="O2_m")

    # ── T-REC ──────────────────────────────────────────────────────────────────
    E_TREC_yr = mdl.addVar(lb=0.0, name="E_TREC_yr")

    # ── Green SOC cross-day auxiliary ─────────────────────────────────────────
    # E_g_bar[i] = scenario-weighted average green SOC at end of day i
    E_g_bar = mdl.addVars(n_days, lb=0.0, ub=EB_ub, name="Egbar")

    # ── Scenario-dependent settlement variables ───────────────────────────────
    # Indexed as Python dicts with (i, w, h) keys for memory efficiency
    P_grid      = {}
    P_grid_load = {}
    P_grid_ch   = {}
    P_pv_load   = {}
    P_pv_ch     = {}
    P_pv_curt   = {}
    E_g         = {}
    P_ch_g      = {}
    P_dis_g     = {}

    for i in I:
        di    = di_map[i]
        scens = day_data[di]["scenarios"]
        for w in range(len(scens)):
            for h in range(24):
                P_grid     [i,w,h] = mdl.addVar(lb=0.0, name=f"Pg_{i}_{w}_{h}")
                P_grid_load[i,w,h] = mdl.addVar(lb=0.0, name=f"Pgl_{i}_{w}_{h}")
                P_grid_ch  [i,w,h] = mdl.addVar(lb=0.0, name=f"Pgc_{i}_{w}_{h}")
                P_pv_load  [i,w,h] = mdl.addVar(lb=0.0, name=f"Pvl_{i}_{w}_{h}")
                P_pv_ch    [i,w,h] = mdl.addVar(lb=0.0, name=f"Pvc_{i}_{w}_{h}")
                P_pv_curt  [i,w,h] = mdl.addVar(lb=0.0, name=f"Pvcu_{i}_{w}_{h}")
                E_g        [i,w,h] = mdl.addVar(lb=0.0, ub=EB_ub, name=f"Eg_{i}_{w}_{h}")
                P_ch_g     [i,w,h] = mdl.addVar(lb=0.0, name=f"Pcg_{i}_{w}_{h}")
                P_dis_g    [i,w,h] = mdl.addVar(lb=0.0, name=f"Pdg_{i}_{w}_{h}")

    mdl.update()
    print(f"  Variables added: {mdl.NumVars} cont + {mdl.NumBinVars} binary ...")

    # ── Constraints ───────────────────────────────────────────────────────────

    # (B) Charge/discharge limits — exact linearisation of u * PB
    for i in I:
        for h in range(24):
            mdl.addConstr(P_ch[i,h]  <= PB_ub * u[i,h],          name=f"Bch_u_{i}_{h}")
            mdl.addConstr(P_ch[i,h]  <= PB,                       name=f"Bch_pb_{i}_{h}")
            mdl.addConstr(P_dis[i,h] <= PB_ub * (1 - u[i,h]),    name=f"Bdis_u_{i}_{h}")
            mdl.addConstr(P_dis[i,h] <= PB,                       name=f"Bdis_pb_{i}_{h}")

    # (C) SOC evolution
    for i in I:
        for h in range(24):
            E_prev = (soc_init * EB) if (i == 0 and h == 0) else \
                     (E[i-1, 23]    if h == 0               else E[i, h-1])
            mdl.addConstr(
                E[i,h] == E_prev + eta_ch * P_ch[i,h] - (1.0/eta_dis) * P_dis[i,h],
                name=f"C_soc_{i}_{h}"
            )
            mdl.addConstr(E[i,h] >= soc_min * EB, name=f"C_lo_{i}_{h}")
            mdl.addConstr(E[i,h] <= soc_max * EB, name=f"C_hi_{i}_{h}")

    # (C5) Year-end terminal SOC (double-sided band)
    mdl.addConstr(E[n_days-1, 23] >= (soc_init - eps_term) * EB, name="Cterm_lo")
    mdl.addConstr(E[n_days-1, 23] <= (soc_init + eps_term) * EB, name="Cterm_hi")

    # (G) Degradation segments
    for i in I:
        for h in range(24):
            mdl.addConstr(
                P_dis[i,h] == gp.quicksum(e_seg[i,h,k] for k in range(n_seg)),
                name=f"G1_{i}_{h}"
            )
            for k in range(n_seg):
                # (b[k+1]-b[k]) * EB is linear — constant scalar × EB variable
                mdl.addConstr(
                    e_seg[i,h,k] <= (b_k[k+1] - b_k[k]) * EB,
                    name=f"G2_{i}_{h}_{k}"
                )

    # (A) Power balance & (E) Green SOC — per scenario
    for i in I:
        di    = di_map[i]
        scens = day_data[di]["scenarios"]

        for w, sc in enumerate(scens):
            # Green SOC carry-in (E3): weighted avg from prev day, or 0 for day 0
            Eg_carry = 0.0 if i == 0 else E_g_bar[i-1]

            for h in range(24):
                pv_val   = float(sc["pv"][h])
                load_val = float(sc["load"][h])

                # (A1) Load balance
                mdl.addConstr(
                    P_grid_load[i,w,h] + P_pv_load[i,w,h] + P_dis[i,h] == load_val,
                    name=f"A1_{i}_{w}_{h}"
                )
                # (A2) Charge balance
                mdl.addConstr(
                    P_ch[i,h] == P_grid_ch[i,w,h] + P_pv_ch[i,w,h],
                    name=f"A2_{i}_{w}_{h}"
                )
                # (A3) Grid import
                mdl.addConstr(
                    P_grid[i,w,h] == P_grid_load[i,w,h] + P_grid_ch[i,w,h],
                    name=f"A3_{i}_{w}_{h}"
                )
                # (A4) PV allocation
                mdl.addConstr(
                    P_pv_load[i,w,h] + P_pv_ch[i,w,h] + P_pv_curt[i,w,h] == pv_val,
                    name=f"A4_{i}_{w}_{h}"
                )

                # (E4) Green SOC evolution
                Eg_prev = (Eg_carry if h == 0 else E_g[i, w, h-1])
                mdl.addConstr(
                    E_g[i,w,h] == Eg_prev
                        + eta_ch * P_ch_g[i,w,h]
                        - (1.0/eta_dis) * P_dis_g[i,w,h],
                    name=f"E4_{i}_{w}_{h}"
                )
                # (E5) Green SOC <= total SOC (physical)
                mdl.addConstr(E_g[i,w,h] <= E[i,h],          name=f"E5_{i}_{w}_{h}")
                # (E6) Green charge <= PV charge
                mdl.addConstr(P_ch_g[i,w,h] <= P_pv_ch[i,w,h], name=f"E6_{i}_{w}_{h}")
                # (E7) Green discharge <= total discharge
                mdl.addConstr(P_dis_g[i,w,h] <= P_dis[i,h],  name=f"E7_{i}_{w}_{h}")

        # (E2) E_g_bar[i] = weighted average over scenarios
        mdl.addConstr(
            E_g_bar[i] == gp.quicksum(
                sc["pi"] * E_g[i, w, 23]
                for w, sc in enumerate(scens)
            ),
            name=f"Egbar_{i}"
        )

    # (D) Monthly demand — per-scenario robust
    # For K=1 (det sizing): per-scenario == expectation, no difference.
    # For K>1 (prob sizing): D_m must cover every scenario's hourly grid import,
    # not just the weighted average. This lets the optimizer size CC (and battery)
    # to handle the worst-case PV scenario, making probabilistic sizing meaningful.
    for mo, day_list_m in months_dict.items():
        for i in day_list_m:
            di    = di_map[i]
            scens = day_data[di]["scenarios"]
            for w, sc in enumerate(scens):
                for h in range(24):
                    mdl.addConstr(
                        D_m[mo] >= kappa * P_grid[i, w, h],
                        name=f"D1_{mo}_{i}_{w}_{h}"
                    )
        mdl.addConstr(Over_m[mo] >= D_m[mo] - CC,           name=f"D3_{mo}")
        mdl.addConstr(Over_m[mo] == O1_m[mo] + O2_m[mo],    name=f"D5_{mo}")
        mdl.addConstr(O1_m[mo] <= tier1 * CC,                name=f"D6_{mo}")

    # (F) Annual RE20 / T-REC
    e_ren  = gp.LinExpr()
    e_load = 0.0   # constant (deterministic load baseline)

    for i in I:
        di    = di_map[i]
        scens = day_data[di]["scenarios"]
        for w, sc in enumerate(scens):
            e_load += sc["pi"] * float(sum(sc["load"]))
            for h in range(24):
                e_ren += sc["pi"] * (P_pv_load[i,w,h] + P_dis_g[i,w,h])

    mdl.addConstr(E_TREC_yr >= re_tgt * e_load - e_ren, name="F4")

    # ── Objective ──────────────────────────────────────────────────────────────
    AEC_inv = CRF * (C_BP * PB + C_BE * EB)

    C_ene = gp.LinExpr()
    for i in I:
        di    = di_map[i]
        scens = day_data[di]["scenarios"]
        tou   = day_data[di]["tou"]
        for w, sc in enumerate(scens):
            for h in range(24):
                C_ene += sc["pi"] * tou[h] * P_grid[i, w, h]

    C_basic = gp.LinExpr()
    for mo in month_ids:
        C_basic += get_basic_charge(mo, cfg) * CC

    C_over = gp.LinExpr()
    for mo in month_ids:
        c_b = get_basic_charge(mo, cfg)
        C_over += c_b * (m1 * O1_m[mo] + m2 * O2_m[mo])

    C_deg = gp.quicksum(
        lam_k[k] * e_seg[i,h,k]
        for i in I for h in range(24) for k in range(n_seg)
    )

    C_TREC = c_trec * E_TREC_yr

    mdl.setObjective(AEC_inv + C_ene + C_basic + C_over + C_deg + C_TREC,
                     GRB.MINIMIZE)

    t_build = time.time() - t0_build
    print(f"  Model complete: {mdl.NumVars} vars ({mdl.NumBinVars} binary), "
          f"{mdl.NumConstrs} constrs  [{t_build:.1f}s build]")

    # ── Solve ─────────────────────────────────────────────────────────────────
    t0_solve = time.time()
    mdl.optimize()
    t_solve = time.time() - t0_solve

    STATUS = {
        GRB.OPTIMAL:    "Optimal",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.INFEASIBLE: "Infeasible",
        GRB.UNBOUNDED:  "Unbounded",
    }
    status = STATUS.get(mdl.Status, f"Code{mdl.Status}")

    if mdl.SolCount == 0:
        raise RuntimeError(
            f"[Layer A] No feasible solution for {case_id} (status={status})"
        )

    # ── Extract and save design ────────────────────────────────────────────────
    CC_v = round(CC.X, 2)
    PB_v = round(PB.X, 2)
    EB_v = round(EB.X, 2)
    J_v  = round(mdl.ObjVal / 1e6, 4)

    # Component breakdown (post-solve, for reporting)
    AEC_inv_v   = CRF * (C_BP * PB_v + C_BE * EB_v)
    C_basic_v   = sum(get_basic_charge(mo, cfg) * CC_v for mo in month_ids)
    C_over_v    = sum(
        get_basic_charge(mo, cfg) * (m1 * O1_m[mo].X + m2 * O2_m[mo].X)
        for mo in month_ids
    )
    C_TREC_v    = c_trec * E_TREC_yr.X

    print(f"\n  [Layer A result] {case_id}")
    print(f"  CC={CC_v:.1f} kW  PB={PB_v:.1f} kW  EB={EB_v:.1f} kWh")
    print(f"  J*={J_v:.4f} M NTD  status={status}  "
          f"gap={mdl.MIPGap*100:.3f}%  solve={t_solve:.0f}s")
    print(f"  CAPEX={AEC_inv_v/1e6:.4f}  basic={C_basic_v/1e6:.4f}  "
          f"over={C_over_v/1e6:.4f}  TREC={C_TREC_v/1e6:.4f}")

    artifact = {
        "case_id":     case_id,
        "CC_star":     CC_v,
        "PB_star":     PB_v,
        "EB_star":     EB_v,
        "J_star_MNTD": J_v,
        "cost_breakdown_MNTD": {
            "AEC_inv":   round(AEC_inv_v / 1e6, 4),
            "C_basic":   round(C_basic_v / 1e6, 4),
            "C_over":    round(C_over_v  / 1e6, 4),
            "C_TREC":    round(C_TREC_v  / 1e6, 4),
        },
        "solve_metadata": {
            "solver":           "gurobi",
            "mip_gap_achieved": round(mdl.MIPGap, 6),
            "mip_gap_target":   solver["mip_gap"],
            "solve_time_sec":   round(t_solve, 1),
            "build_time_sec":   round(t_build, 1),
            "n_vars":           mdl.NumVars,
            "n_binary":         mdl.NumBinVars,
            "n_constraints":    mdl.NumConstrs,
            "status":           status,
            "obj_val_NTD":      round(mdl.ObjVal, 0),
        },
    }

    out_path = DES_DIR / f"design_{case_id}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Saved -> {out_path.name}")

    mdl.dispose()
    env.dispose()
    return artifact
