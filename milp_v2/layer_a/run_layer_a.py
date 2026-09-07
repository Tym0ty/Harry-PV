"""
milp_v2/layer_a/run_layer_a.py
Run Layer A for all five cases: DD, DP, PD, PP, PI.

Usage:
    python milp_v2/layer_a/run_layer_a.py
    python milp_v2/layer_a/run_layer_a.py --cases DD PP
"""

import argparse, json, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import load_config
from layer_a.milp_annual import solve_annual_milp

CFG = load_config()


# Package routing (matches config.yaml layer_a_package)
LAYER_A_PACKAGE = {
    "DD": "layer_a_det.parquet",
    "DP": "layer_a_det.parquet",   # DD and DP share det sizing
    "PD": "layer_a_prob.parquet",
    "PP": "layer_a_prob.parquet",  # PD and PP share prob sizing
    "PI": "layer_a_PI.parquet",
}

ALL_CASES = ["DD", "DP", "PD", "PP", "PI"]


def run_all(cases=None):
    cases = cases or ALL_CASES
    results = {}
    t0_all  = time.time()

    print("\n" + "=" * 60)
    print("milp_v2 Layer A — Annual MILP Sizing")
    print("=" * 60)
    print(f"Cases: {cases}")

    for case_id in cases:
        pkg = LAYER_A_PACKAGE[case_id]
        try:
            artifact = solve_annual_milp(case_id, pkg, CFG)
            results[case_id] = artifact
        except Exception as e:
            print(f"\n  ERROR in {case_id}: {e}")
            results[case_id] = {"error": str(e)}

    t_total = time.time() - t0_all

    # Summary table
    print("\n" + "=" * 60)
    print("LAYER A SUMMARY")
    print(f"{'Case':<6} {'CC':>8} {'PB':>8} {'EB':>8} {'J* (M NTD)':>12} {'Status':<12}")
    print("-" * 60)
    for case_id in cases:
        r = results.get(case_id, {})
        if "error" in r:
            print(f"{case_id:<6} ERROR: {r['error']}")
        else:
            status = r.get("solve_metadata", {}).get("status", "?")
            print(f"{case_id:<6} {r['CC_star']:>8.1f} {r['PB_star']:>8.1f} "
                  f"{r['EB_star']:>8.1f} {r['J_star_MNTD']:>12.4f} {status:<12}")
    print("=" * 60)
    print(f"Total wall-clock: {t_total/60:.1f} min")

    # Save summary
    out = ROOT / CFG["paths"]["designs_dir"] / "layer_a_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved -> {out.name}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", choices=ALL_CASES + ["all"],
                    default=["all"])
    args = ap.parse_args()

    cases = ALL_CASES if "all" in args.cases else args.cases
    run_all(cases)


if __name__ == "__main__":
    main()
