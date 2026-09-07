"""
milp_v2/layer_b/run_layer_b.py
Run Layer B sequential day-ahead MILP for all cases.

Requires Layer A designs in layer_a/designs/design_{case_id}.json.

Usage:
    python milp_v2/layer_b/run_layer_b.py
    python milp_v2/layer_b/run_layer_b.py --cases DD PP
"""

import argparse, json, time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config, get_tou_price
from layer_b.milp_daily import solve_day_ahead

CFG     = load_config()
PKG_DIR = ROOT / CFG["paths"]["packages_dir"]
DES_DIR = ROOT / CFG["paths"]["designs_dir"]
LB_DIR  = ROOT / CFG["paths"]["layerb_results_dir"]
LB_DIR.mkdir(parents=True, exist_ok=True)

# Package routing (matches config.yaml)
LAYER_B_PACKAGE = {
    "DD": "layerB_det_package.parquet",
    "DP": "layerB_prob_package_x.parquet",
    "PD": "layerB_det_package.parquet",
    "PP": "layerB_prob_package_x.parquet",
    "PI": "layerB_PI_perfect.parquet",
}

ALL_CASES = ["DD", "DP", "PD", "PP", "PI"]


# ── Package loader ────────────────────────────────────────────────────────────

def load_layerb_package(pkg_name: str) -> tuple:
    """
    Load a Layer B package parquet into day_data dict.
    Returns (day_data, day_indices).
    """
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

        # Normalise
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


# ── Single-case runner ────────────────────────────────────────────────────────

def run_case(case_id: str, pkg_override: str = None,
             result_tag: str = "") -> pd.DataFrame:
    """
    Run Layer B for one case. Loads design from designs/, package from bridge/.
    Returns a DataFrame with daily results.

    Args:
        case_id:      Base case ID (DD/DP/PD/PP/PI).
        pkg_override: If set, use this package filename instead of LAYER_B_PACKAGE[case_id].
                      Used by run_variants.py for scenario variant experiments.
        result_tag:   Suffix appended to the output filename (e.g. '_A' → layerb_DP_A_daily).
    """
    # Load frozen design
    design_path = DES_DIR / f"design_{case_id}.json"
    if not design_path.exists():
        raise FileNotFoundError(
            f"Design not found for {case_id}: {design_path}\n"
            "Run Layer A first."
        )
    with open(design_path) as f:
        des = json.load(f)
    design = {"CC": des["CC_star"], "PB": des["PB_star"], "EB": des["EB_star"]}

    print(f"\n[Layer B] case={case_id}  "
          f"CC={design['CC']:.1f}  PB={design['PB']:.1f}  EB={design['EB']:.1f}")

    # Load package (variant override or default routing)
    pkg_name  = pkg_override if pkg_override else LAYER_B_PACKAGE[case_id]
    day_data, day_indices = load_layerb_package(pkg_name)
    n_days = len(day_indices)
    print(f"  Package: {pkg_name}  days={n_days}")

    # Monthly tracking state
    soc_init = CFG["battery"]["soc_init"]
    state = {
        "E_soc": soc_init * design["EB"],
        "E_g":   0.0,
        "D_mth": 0.0,
    }

    rows = []
    t0_case = time.time()

    for idx, di in enumerate(day_indices):
        dd         = day_data[di]
        mo         = dd["month_id"]
        is_final   = (idx == n_days - 1)

        plan = solve_day_ahead(
            day_index    = di,
            day_dict     = dd,
            state        = state,
            design       = design,
            cfg          = CFG,
            is_final_day = is_final,
            time_limit   = 60,
            mip_gap      = 0.005,
        )

        # Monthly demand carry-forward
        D_new = max(state["D_mth"], plan["D_cand"])

        # Month boundary reset (next day in new month → reset D_mth to 0)
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
            "D_cand":       plan["D_cand"],
            "D_mth_end":    D_new,
            "obj_val":      plan["obj_val"],
            "solve_time":   plan["solve_time"],
            "status":       plan["status"],
            # Store schedules as lists (JSON-serialisable)
            "P_ch_plan":    plan["P_ch_plan"].tolist(),
            "P_dis_plan":   plan["P_dis_plan"].tolist(),
        })

        if (idx + 1) % 30 == 0 or is_final:
            elapsed = time.time() - t0_case
            print(f"  Day {idx+1:3d}/{n_days}  di={di}  "
                  f"D_cand={plan['D_cand']:.0f}  "
                  f"E_end={plan['E_end']:.0f}  "
                  f"t={elapsed:.0f}s")

        # Update state for next day
        state = {
            "E_soc": plan["E_end"],
            "E_g":   plan["E_g_end"],
            "D_mth": D_next,
        }

    t_case = time.time() - t0_case
    df_results = pd.DataFrame(rows)

    # Save
    out_path = LB_DIR / f"layerb_{case_id}{result_tag}_daily.parquet"
    df_results.to_parquet(out_path, index=False)
    print(f"  Saved {len(df_results)} days -> {out_path.name}  [{t_case:.0f}s]")

    return df_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", choices=ALL_CASES + ["all"],
                    default=["all"])
    args = ap.parse_args()

    cases = ALL_CASES if "all" in args.cases else args.cases

    print("\n" + "=" * 60)
    print("milp_v2 Layer B — Sequential Day-Ahead MILP")
    print("=" * 60)
    print(f"Cases: {cases}")

    t0_all = time.time()
    for case_id in cases:
        try:
            run_case(case_id)
        except Exception as e:
            print(f"\n  ERROR in {case_id}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nAll Layer B done. Total: {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
