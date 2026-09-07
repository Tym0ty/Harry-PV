"""
milp_v2/layer_b/run_layer_b_cvar.py
Run Layer B with CVaR-augmented MILP (Phase 1B).

Usage:
    python milp_v2/layer_b/run_layer_b_cvar.py --case DP --lam 0.5 --alpha 0.90
    python milp_v2/layer_b/run_layer_b_cvar.py --case PP --lam 1.0 --alpha 0.95
    python milp_v2/layer_b/run_layer_b_cvar.py --case DD --lam 0.0   # equiv to run_layer_b

Output:
    layer_b/results/layerb_{case}_cvar_lam{lam}_a{alpha}_daily.parquet
    layer_b/results/layerb_{case}_cvar_lam{lam}_a{alpha}_cvar_log.parquet
"""

import argparse, json, time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_tou_price
from layer_b.milp_cvar import solve_day_ahead

CFG     = load_config()
PKG_DIR = ROOT / CFG["paths"]["packages_dir"]
DES_DIR = ROOT / CFG["paths"]["designs_dir"]
LB_DIR  = ROOT / CFG["paths"]["layerb_results_dir"]
LB_DIR.mkdir(parents=True, exist_ok=True)

LAYER_B_PACKAGE = {
    "DD": "layerB_det_package.parquet",
    "DP": "layerB_prob_package_x.parquet",
    "PD": "layerB_det_package.parquet",
    "PP": "layerB_prob_package_x.parquet",
    "PI": "layerB_PI_perfect.parquet",
}

ALL_CASES = ["DD", "DP", "PD", "PP", "PI"]


def load_layerb_package(pkg_name: str) -> tuple:
    df = pd.read_parquet(PKG_DIR / pkg_name)
    day_indices  = sorted(df["day_index"].unique().tolist())
    scenario_ids = sorted(df["scenario_id"].unique().tolist())

    day_data = {}
    for di in day_indices:
        d_di  = df[df["day_index"] == di]
        month = int(d_di["month_id"].iloc[0])
        cd    = pd.Timestamp(d_di["calendar_day"].iloc[0])
        dow   = cd.weekday()
        tou   = np.array([get_tou_price(cd.month, cd.day, dow, h) for h in range(24)])

        scenarios = []
        for sid in scenario_ids:
            s = d_di[d_di["scenario_id"] == sid].sort_values("hour_local")
            if len(s) == 0:
                continue
            pi   = float(s["probability_pi"].iloc[0])
            pv   = s["pv_available_kw"].values.astype(float)
            load = s["load_input_kw"].values.astype(float)
            scenarios.append({"id": sid, "pi": pi, "pv": pv, "load": load})

        total_pi = sum(sc["pi"] for sc in scenarios)
        if abs(total_pi - 1.0) > 1e-6:
            for sc in scenarios:
                sc["pi"] /= total_pi

        day_data[di] = {
            "scenarios":    scenarios,
            "tou":          tou,
            "month_id":     month,
            "calendar_day": cd,
        }

    return day_data, day_indices


def run_case_cvar(case_id: str, lam: float, alpha: float) -> tuple:
    """
    Run Layer B CVaR MILP for one case. Returns (daily_df, cvar_log_df).
    """
    design_path = DES_DIR / f"design_{case_id}.json"
    if not design_path.exists():
        raise FileNotFoundError(
            f"Design not found for {case_id}: {design_path}\n"
            "Run Layer A first."
        )
    with open(design_path) as f:
        des = json.load(f)
    design = {"CC": des["CC_star"], "PB": des["PB_star"], "EB": des["EB_star"]}

    lam_tag   = f"{lam:.2f}".replace(".", "p")
    alpha_tag = f"{int(alpha*100):02d}"
    run_tag   = f"cvar_lam{lam_tag}_a{alpha_tag}"

    print(f"\n[Layer B CVaR] case={case_id}  lam={lam}  alpha={alpha}")
    print(f"  CC={design['CC']:.1f}  PB={design['PB']:.1f}  EB={design['EB']:.1f}")

    pkg_name = LAYER_B_PACKAGE[case_id]
    day_data, day_indices = load_layerb_package(pkg_name)
    n_days = len(day_indices)
    print(f"  Package: {pkg_name}  days={n_days}")

    soc_init = CFG["battery"]["soc_init"]
    state = {
        "E_soc": soc_init * design["EB"],
        "E_g":   0.0,
        "D_mth": 0.0,
    }

    rows      = []
    cvar_rows = []
    t0_case   = time.time()

    for idx, di in enumerate(day_indices):
        dd       = day_data[di]
        mo       = dd["month_id"]
        is_final = (idx == n_days - 1)

        plan = solve_day_ahead(
            day_index    = di,
            day_dict     = dd,
            state        = state,
            design       = design,
            cfg          = CFG,
            is_final_day = is_final,
            time_limit   = 60,
            mip_gap      = 0.005,
            lam          = lam,
            alpha        = alpha,
        )

        # D_mth carry-forward uses expected D_cand (pi-weighted)
        D_new = max(state["D_mth"], plan["D_cand"])
        if idx + 1 < n_days:
            next_mo = day_data[day_indices[idx+1]]["month_id"]
            D_next  = 0.0 if next_mo != mo else D_new
        else:
            D_next = 0.0

        rows.append({
            "day_index":    di,
            "month_id":     mo,
            "calendar_day": str(dd["calendar_day"].date()),
            "is_final_day": is_final,
            "E_soc_start":  state["E_soc"],
            "E_g_start":    state["E_g"],
            "D_mth_start":  state["D_mth"],
            "E_soc_end":    plan["E_end"],
            "E_g_end":      plan["E_g_end"],
            "D_cand":       plan["D_cand"],      # expected (pi-weighted)
            "D_mth_end":    D_new,
            "obj_val":      plan["obj_val"],
            "E_coc_val":    plan["E_coc_val"],
            "cvar_val":     plan["cvar_val"],
            "eta_val":      plan["eta_val"],
            "solve_time":   plan["solve_time"],
            "status":       plan["status"],
            "P_ch_plan":    plan["P_ch_plan"].tolist(),
            "P_dis_plan":   plan["P_dis_plan"].tolist(),
        })

        # Per-scenario CVaR log
        for w, sc in enumerate(dd["scenarios"]):
            cvar_rows.append({
                "day_index":    di,
                "scenario_id":  sc["id"],
                "pi":           sc["pi"],
                "D_cand":       plan["D_cand_by_scenario"][w],
                "C_oc":         plan["C_oc_by_scenario"][w],
                "eta_val":      plan["eta_val"],
                "cvar_val":     plan["cvar_val"],
            })

        if (idx + 1) % 30 == 0 or is_final:
            elapsed = time.time() - t0_case
            print(f"  Day {idx+1:3d}/{n_days}  di={di}  "
                  f"D_cand={plan['D_cand']:.0f}  "
                  f"E_coc={plan['E_coc_val']:.1f}  "
                  f"CVaR={plan['cvar_val']:.1f}  "
                  f"t={elapsed:.0f}s")

        state = {
            "E_soc": plan["E_end"],
            "E_g":   plan["E_g_end"],
            "D_mth": D_next,
        }

    t_case = time.time() - t0_case
    df_results  = pd.DataFrame(rows)
    df_cvar_log = pd.DataFrame(cvar_rows)

    out_daily   = LB_DIR / f"layerb_{case_id}_{run_tag}_daily.parquet"
    out_cvar    = LB_DIR / f"layerb_{case_id}_{run_tag}_cvar_log.parquet"
    df_results.to_parquet(out_daily, index=False)
    df_cvar_log.to_parquet(out_cvar, index=False)

    print(f"  Saved {len(df_results)} days -> {out_daily.name}  [{t_case:.0f}s]")
    print(f"  CVaR log  -> {out_cvar.name}  ({len(df_cvar_log)} rows)")

    return df_results, df_cvar_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case",  choices=ALL_CASES, required=True,
                    help="Case ID: DD/DP/PD/PP/PI")
    ap.add_argument("--lam",   type=float, default=0.0,
                    help="CVaR penalty weight lambda (default 0.0 = expected cost only)")
    ap.add_argument("--alpha", type=float, default=0.90,
                    help="CVaR tail probability (default 0.90)")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("milp_v2 Layer B — CVaR-Augmented Sequential MILP (Phase 1B)")
    print("=" * 60)
    print(f"Case: {args.case}  lam={args.lam}  alpha={args.alpha}")

    try:
        run_case_cvar(args.case, args.lam, args.alpha)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
