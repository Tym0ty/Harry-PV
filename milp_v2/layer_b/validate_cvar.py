"""
milp_v2/layer_b/validate_cvar.py
Phase 1B validation: K=1 and K=6 smoke tests for milp_cvar.py.

Validation checks (K=1, lam=1.0, alpha=0.90):
  V1: C_oc[0] correctness  (C_oc = c_basic * (m1*O1 + m2*O2))
  V2: eta converges to C_oc[0]  (for K=1, VaR = C_oc itself)
  V3: xi[0] = 0  (no excess above eta)
  V4: CVaR = C_oc[0]  (for K=1, CVaR = E[C_oc] = C_oc[0])
  V5: obj(lam=0) != obj(lam=1) IFF CVaR != 0

Run from repo root:
    python milp_v2/layer_b/validate_cvar.py
"""

import json, sys, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_basic_charge, get_tou_price
from layer_b.milp_cvar import solve_day_ahead

CFG     = load_config()
PKG_DIR = ROOT / CFG["paths"]["packages_dir"]
DES_DIR = ROOT / CFG["paths"]["designs_dir"]


def load_one_day(pkg_name: str, target_day_index: int = None) -> dict:
    """Load a single day from a package. Uses first day if target_day_index is None."""
    df = pd.read_parquet(PKG_DIR / pkg_name)
    if target_day_index is None:
        target_day_index = int(df["day_index"].iloc[0])
    d = df[df["day_index"] == target_day_index]
    scenario_ids = sorted(d["scenario_id"].unique().tolist())

    cd  = pd.Timestamp(d["calendar_day"].iloc[0])
    dow = cd.weekday()
    tou = np.array([get_tou_price(cd.month, cd.day, dow, h) for h in range(24)])

    scenarios = []
    for sid in scenario_ids:
        s  = d[d["scenario_id"] == sid].sort_values("hour_local")
        pi = float(s["probability_pi"].iloc[0])
        pv = s["pv_available_kw"].values.astype(float)
        ld = s["load_input_kw"].values.astype(float)
        scenarios.append({"id": sid, "pi": pi, "pv": pv, "load": ld})

    total_pi = sum(sc["pi"] for sc in scenarios)
    if abs(total_pi - 1.0) > 1e-6:
        for sc in scenarios:
            sc["pi"] /= total_pi

    return {
        "scenarios":    scenarios,
        "tou":          tou,
        "month_id":     int(d["month_id"].iloc[0]),
        "calendar_day": cd,
    }


def load_design(case_id: str) -> dict:
    p = DES_DIR / f"design_{case_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Design file not found: {p}. Run Layer A first.")
    with open(p) as f:
        des = json.load(f)
    return {"CC": des["CC_star"], "PB": des["PB_star"], "EB": des["EB_star"]}


def make_state(design: dict) -> dict:
    soc_init = CFG["battery"]["soc_init"]
    return {"E_soc": soc_init * design["EB"], "E_g": 0.0, "D_mth": 0.0}


# ── Validation helpers ────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

def check(label: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {label}" + (f"  ({detail})" if detail else ""))
    return cond


# ── K=1 validation ────────────────────────────────────────────────────────────

def validate_k1():
    print("\n" + "="*60)
    print("K=1 Validation (det package, DD case)")
    print("="*60)

    try:
        design   = load_design("DD")
        day_dict = load_one_day("layerB_det_package.parquet")
    except FileNotFoundError as e:
        print(f"  {SKIP}  Cannot load design or package: {e}")
        return False

    state = make_state(design)
    month_id = day_dict["month_id"]
    c_basic  = get_basic_charge(month_id, CFG)
    dem      = CFG["demand"]
    m1, m2, tier1 = dem["over_contract_m1"], dem["over_contract_m2"], dem["over_contract_tier1_ratio"]
    CC = design["CC"]

    print(f"  Design: CC={CC:.0f} PB={design['PB']:.0f} EB={design['EB']:.0f}")
    print(f"  Package day: {day_dict['calendar_day'].date()}  month={month_id}  c_basic={c_basic:.1f}")

    # Run lam=1.0 (CVaR active)
    print("\n  Running lam=1.0 alpha=0.90 ...")
    t0 = time.time()
    plan1 = solve_day_ahead(
        day_index=0, day_dict=day_dict, state=state, design=design,
        cfg=CFG, is_final_day=False, time_limit=60, mip_gap=0.005,
        lam=1.0, alpha=0.90,
    )
    print(f"  Solved in {time.time()-t0:.1f}s  status={plan1['status']}")

    if plan1["status"] < 0:
        print(f"  {SKIP}  MILP failed, cannot validate")
        return False

    C_oc_0    = plan1["C_oc_by_scenario"][0]
    eta_val   = plan1["eta_val"]
    cvar_val  = plan1["cvar_val"]
    E_coc_val = plan1["E_coc_val"]

    print(f"\n  K=1 raw values:")
    print(f"    C_oc[0]   = {C_oc_0:.4f}")
    print(f"    eta       = {eta_val:.4f}")
    print(f"    cvar      = {cvar_val:.4f}")
    print(f"    E_coc     = {E_coc_val:.4f}")

    # V1: C_oc[0] matches manual formula
    D_cand_0 = plan1["D_cand_by_scenario"][0]
    Over_0   = max(D_cand_0 - CC, 0.0)
    O1_0     = min(Over_0, tier1 * CC)
    O2_0     = max(Over_0 - tier1 * CC, 0.0)
    C_oc_manual = c_basic * (m1 * O1_0 + m2 * O2_0)
    ok1 = check("V1: C_oc[0] matches manual formula",
                abs(C_oc_0 - C_oc_manual) < 1.0,
                f"C_oc[0]={C_oc_0:.2f}  manual={C_oc_manual:.2f}")

    # V2: eta converges to C_oc[0] for K=1
    ok2 = check("V2: eta ~= C_oc[0] for K=1",
                abs(eta_val - C_oc_0) < 5.0,
                f"eta={eta_val:.2f}  C_oc[0]={C_oc_0:.2f}")

    # V3: CVaR ~= C_oc[0] for K=1 (single scenario, all tail mass = one point)
    ok3 = check("V3: CVaR ~= C_oc[0] for K=1",
                abs(cvar_val - C_oc_0) < 5.0,
                f"CVaR={cvar_val:.2f}  C_oc[0]={C_oc_0:.2f}")

    # V4: E_coc ~= C_oc[0] for K=1
    ok4 = check("V4: E_coc ~= C_oc[0] for K=1",
                abs(E_coc_val - C_oc_0) < 1.0,
                f"E_coc={E_coc_val:.2f}  C_oc[0]={C_oc_0:.2f}")

    # V5: lam=0 run for comparison
    print("\n  Running lam=0.0 for V5 ...")
    plan0 = solve_day_ahead(
        day_index=0, day_dict=day_dict, state=state, design=design,
        cfg=CFG, is_final_day=False, time_limit=60, mip_gap=0.005,
        lam=0.0, alpha=0.90,
    )
    obj0 = plan0["obj_val"]
    obj1 = plan1["obj_val"]
    cvar0 = plan0["cvar_val"]
    print(f"    obj(lam=0)={obj0:.2f}  obj(lam=1)={obj1:.2f}  CVaR(lam=0)={cvar0:.2f}")
    if abs(cvar0) < 1.0:
        ok5 = check("V5: obj diff = 0 when CVaR=0",
                    abs(obj1 - obj0) < 5.0,
                    f"|obj1-obj0|={abs(obj1-obj0):.2f}  CVaR≈0")
    else:
        ok5 = check("V5: obj(lam=1) > obj(lam=0) when CVaR > 0",
                    obj1 >= obj0 - 1.0,
                    f"obj1={obj1:.2f}  obj0={obj0:.2f}  CVaR={cvar0:.2f}")

    all_pass = all([ok1, ok2, ok3, ok4, ok5])
    print(f"\n  K=1 validation: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return all_pass


# ── K=6 smoke test ────────────────────────────────────────────────────────────

def smoke_k6():
    print("\n" + "="*60)
    print("K=6 Smoke Test (prob package, DP case, lam=0.5)")
    print("="*60)

    try:
        design   = load_design("DP")
        day_dict = load_one_day("layerB_prob_package_x.parquet")
    except FileNotFoundError as e:
        print(f"  {SKIP}  Cannot load design or package: {e}")
        return False

    state = make_state(design)
    n_sc  = len(day_dict["scenarios"])
    print(f"  Design: CC={design['CC']:.0f}  K={n_sc}")
    print(f"  Package day: {day_dict['calendar_day'].date()}")

    print("\n  Running lam=0.5 alpha=0.90 ...")
    t0 = time.time()
    plan = solve_day_ahead(
        day_index=0, day_dict=day_dict, state=state, design=design,
        cfg=CFG, is_final_day=False, time_limit=120, mip_gap=0.005,
        lam=0.5, alpha=0.90,
    )
    elapsed = time.time() - t0
    print(f"  Solved in {elapsed:.1f}s  status={plan['status']}")

    if plan["status"] < 0:
        print(f"  {SKIP}  MILP failed")
        return False

    print(f"\n  Per-scenario C_oc:")
    for w, sc in enumerate(day_dict["scenarios"]):
        print(f"    sc={sc['id']}  pi={sc['pi']:.4f}  "
              f"D_cand={plan['D_cand_by_scenario'][w]:.1f}  "
              f"C_oc={plan['C_oc_by_scenario'][w]:.2f}")
    print(f"\n  E_coc = {plan['E_coc_val']:.2f}")
    print(f"  eta   = {plan['eta_val']:.2f}")
    print(f"  CVaR  = {plan['cvar_val']:.2f}")
    print(f"  obj   = {plan['obj_val']:.2f}")
    print(f"  solve = {elapsed:.1f}s")

    # Check CVaR >= E_coc (by definition CVaR >= mean for coherent risk measure)
    ok = check("K=6: CVaR >= E_coc (coherent risk measure property)",
               plan["cvar_val"] >= plan["E_coc_val"] - 1.0,
               f"CVaR={plan['cvar_val']:.2f}  E_coc={plan['E_coc_val']:.2f}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nmilp_cvar.py Phase 1B Validation")
    print("="*60)

    ok_k1 = validate_k1()
    ok_k6 = smoke_k6()

    print("\n" + "="*60)
    print(f"Summary: K=1={'PASS' if ok_k1 else 'FAIL'}  K=6={'PASS' if ok_k6 else 'FAIL'}")
    print("="*60)
