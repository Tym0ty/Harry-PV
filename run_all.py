#!/usr/bin/env python3
"""
Run-All Pipeline: F0329 5-Stage Orchestration.
Per F0329Bridge_Spec Batch 8 + F0329MILP_Spec Batch 17.

Stages:
  Stage 0  — Link forecast artifacts from pipeline_outputs/
  Stage 1  — Bridge: build both rolling DA + repday families (+ scenario reduction)
  Stage 2a — Layer A: annual fixed-design selection per case
  Stage 2b — Layer B: sequential daily solve per case (using Layer A design)
  Stage 2c — Replay: fixed-design replay against realized truth (within Layer B)
  Stage 3  — Reporting: aggregate tables + gap analysis

Usage:
    python3 run_all.py [--cases C0 C1 C2 C3 C_PFI] [--skip_bridge] [--skip_layer_a]
    Set GRB_LICENSE_FILE env var to your Gurobi license path.

Requires: pandas, numpy, pyarrow, scipy, scikit-learn, gurobipy
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "notebooks_bridge"))
sys.path.insert(0, str(ROOT / "notebooks_milp"))

MANIFEST_PATH = ROOT / "milp_run_manifest.json"
BRIDGE_OUT    = ROOT / "bridge_outputs_fullyear"
MILP_OUT      = ROOT / "milp_outputs"


# ──────────────────────────────────────────────────────────────
#  Stage 0: Link forecast artifacts
# ──────────────────────────────────────────────────────────────

def stage0_link_forecasts():
    print("\n" + "=" * 60)
    print("STAGE 0: Link Forecast Artifacts")
    print("=" * 60)

    pipeline_dir = ROOT / "pipeline_outputs"
    pv_det_path = pipeline_dir / "pv_point_forecast_caseyear.parquet"
    sc_path     = pipeline_dir / "scenarios_joint_pv_load_reduced_5.parquet"

    if not pv_det_path.exists() or not sc_path.exists():
        raise FileNotFoundError(
            f"Pipeline outputs not found. Run the forecast pipeline first.\n"
            f"  Expected: {pv_det_path}\n  Expected: {sc_path}"
        )

    sc_df  = pd.read_parquet(sc_path)
    pv_det = pd.read_parquet(pv_det_path)
    print(f"  Det PV: {len(pv_det)} rows")
    print(f"  Scenarios: {len(sc_df)} rows, {sc_df['scenario_id'].nunique()} PV scenarios/day")
    return True


# ──────────────────────────────────────────────────────────────
#  Stage 1: Bridge
# ──────────────────────────────────────────────────────────────

def stage1_bridge():
    print("\n" + "=" * 60)
    print("STAGE 1: Bridge Layer (F0329 Spec)")
    print("=" * 60)

    from bridge_full_year import run_bridge
    report = run_bridge()

    # 1b: Representative-day builder
    print("\n--- RepDay Builder ---")
    from repday_builder import build_repdays
    repday_report = build_repdays(BRIDGE_OUT)

    # 1c: Scenario reduction (C1, C2, C3)
    print("\n--- Scenario Reduction ---")
    from scenario_reduction import reduce_all_packages
    reduce_all_packages(BRIDGE_OUT, n_target_c1=5, n_target_c2=5, n_target_c3=10)

    return report


# ──────────────────────────────────────────────────────────────
#  Stage 2a: Layer A — Annual design selection
# ──────────────────────────────────────────────────────────────

def stage2a_layer_a(cases):
    print("\n" + "=" * 60)
    print("STAGE 2a: Layer A — Annual Fixed-Design Selection")
    print("=" * 60)

    from milp_layer_a import run_layer_a

    design_results = {}
    for case_id in cases:
        print(f"\n{'=' * 40}")
        print(f"Layer A — {case_id}")
        best, _ = run_layer_a(
            case_id,
            str(MANIFEST_PATH),
            bridge_dir=str(BRIDGE_OUT),
            output_dir=str(MILP_OUT),
        )
        design_results[case_id] = best

    # Save master design table
    rows = []
    for cid, r in design_results.items():
        if r:
            rows.append({
                "case_id": cid,
                "CC_kW":   r["CC"],
                "P_B_kW":  r["P_B"],
                "E_B_kWh": r["E_B"],
                "J_A_M":   round(r["J_A"] / 1e6, 3),
                "AEC_inv_M": round(r.get("AEC_inv", 0) / 1e6, 3),
            })
    if rows:
        pd.DataFrame(rows).to_csv(MILP_OUT / "design_results_master.csv", index=False)
        print("\n=== DESIGN RESULTS MASTER ===")
        print(pd.DataFrame(rows).to_string(index=False))

    return design_results


# ──────────────────────────────────────────────────────────────
#  Stage 2b/2c: Layer B — Sequential solve + replay
# ──────────────────────────────────────────────────────────────

def stage2bc_layer_b(cases, design_results):
    print("\n" + "=" * 60)
    print("STAGE 2b/c: Layer B — Sequential Solve + Replay")
    print("=" * 60)

    from milp_common import get_config
    from milp_layer_b import run_layer_b

    CFG = get_config()
    CFG['bridge_dir'] = str(BRIDGE_OUT)
    CFG['output_dir'] = str(MILP_OUT)

    replay_summaries = {}
    for case_id in cases:
        design = design_results.get(case_id)
        if not design:
            print(f"  [SKIP] No design found for {case_id}")
            continue
        print(f"\n{'=' * 40}")
        print(f"Layer B — {case_id}")
        summary = run_layer_b(case_id, design, CFG, str(MILP_OUT))
        replay_summaries[case_id] = summary

    return replay_summaries


# ──────────────────────────────────────────────────────────────
#  Stage 3: Reporting + Gap Analysis
# ──────────────────────────────────────────────────────────────

def stage3_reporting(design_results, replay_summaries):
    print("\n" + "=" * 60)
    print("STAGE 3: Reporting + Gap Analysis")
    print("=" * 60)

    MILP_OUT.mkdir(parents=True, exist_ok=True)

    # Replay summary master
    replay_rows = []
    for cid, s in replay_summaries.items():
        if s:
            replay_rows.append({
                "case_id":    cid,
                "CC_kW":      s["CC"],
                "P_B_kW":     s["P_B"],
                "E_B_kWh":    s["E_B"],
                "replay_total_M":   round(s["replay_total_NTD"] / 1e6, 3),
                "basic_M":          round(s["basic_NTD"] / 1e6, 3),
                "overcontract_M":   round(s["overcontract_NTD"] / 1e6, 3),
                "energy_M":         round(s["energy_NTD"] / 1e6, 3),
                "trec_M":           round(s["trec_NTD"] / 1e6, 3),
                "RE_pct":           round(s["RE_pct"], 1),
                "oc_months":        s["n_overcontract_months"],
            })

    if replay_rows:
        replay_df = pd.DataFrame(replay_rows)
        replay_df.to_csv(MILP_OUT / "replay_summary_master.csv", index=False)
        print("\n=== REPLAY SUMMARY MASTER ===")
        print(replay_df.to_string(index=False))

    # Design-to-replay gap analysis
    print("\n=== GAP ANALYSIS ===")
    for cid in design_results:
        d = design_results.get(cid)
        r = replay_summaries.get(cid)
        if d and r:
            j_a = d.get("J_A", 0)
            rep = r.get("replay_total_NTD", 0)
            gap = rep - j_a
            print(f"  {cid}: J_A={j_a/1e6:.3f}M → Replay={rep/1e6:.3f}M "
                  f"(Gap={gap/1e6:+.3f}M, {gap/j_a*100:+.1f}%)")

    # Research question: C0 vs C1
    print("\n=== RESEARCH QUESTION: Value of Probabilistic PV ===")
    c0 = replay_summaries.get("C0")
    c1 = replay_summaries.get("C1")
    cpfi = replay_summaries.get("C_PFI")
    if c0 and c1:
        diff = c1["replay_total_NTD"] - c0["replay_total_NTD"]
        pct  = diff / c0["replay_total_NTD"] * 100
        print(f"  C0 (Det PV):  {c0['replay_total_NTD']/1e6:.3f} M NTD")
        print(f"  C1 (Prob PV): {c1['replay_total_NTD']/1e6:.3f} M NTD")
        print(f"  Difference:   {diff/1e6:+.3f} M NTD ({pct:+.1f}%)")
        print(f"  → {'Probabilistic PV REDUCES annual cost' if diff < 0 else 'Deterministic PV performs as well or better'}")
    if cpfi:
        print(f"  C_PFI (Upper bound): {cpfi['replay_total_NTD']/1e6:.3f} M NTD")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BH Project F0329 Pipeline")
    parser.add_argument("--cases", nargs="+",
                        default=["C0", "C1", "C2", "C3", "C_PFI"],
                        choices=["C0", "C1", "C2", "C3", "C_PFI"])
    parser.add_argument("--skip_bridge",  action="store_true")
    parser.add_argument("--skip_layer_a", action="store_true")
    args = parser.parse_args()

    # Set Gurobi license
    if 'GRB_LICENSE_FILE' not in os.environ:
        lic_path = Path.home() / "gurobi.lic"
        if lic_path.exists():
            os.environ['GRB_LICENSE_FILE'] = str(lic_path)

    t_start = time.time()

    stage0_link_forecasts()

    if not args.skip_bridge:
        stage1_bridge()
    else:
        print("  [SKIP] Bridge (--skip_bridge)")

    if not args.skip_layer_a:
        design_results = stage2a_layer_a(args.cases)
    else:
        print("  [SKIP] Layer A (--skip_layer_a) — loading existing design results")
        design_results = {}
        for cid in args.cases:
            p = MILP_OUT / f"design_results_{cid}.json"
            if p.exists():
                with open(p) as f:
                    design_results[cid] = json.load(f)

    replay_summaries = stage2bc_layer_b(args.cases, design_results)
    stage3_reporting(design_results, replay_summaries)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Total: {elapsed:.1f}s")
    print("=" * 60)
