#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import pandas as pd
import pyarrow.parquet as pq

from pv_focus_bridge_utils import BRIDGES, OUT_DIR, PROFILES, choose_medoids, ensure_out, write_md, write_table


def reduce_file(mode: str, profile: str, bridge: str, m: int) -> tuple[dict, list[dict], list[dict]]:
    prefix = "da" if mode == "DA" else "id_h24"
    src = OUT_DIR / f"{prefix}_pvfocus_{profile}_raw_netload_M{m}.parquet"
    suffix = bridge.lower().replace("pv_", "")
    dst = OUT_DIR / f"{prefix}_pvfocus_{profile}_{suffix}_K5_scenarios.parquet"
    if dst.exists():
        dst.unlink()
    pf = pq.ParquetFile(src)
    writer = None
    weight_rows = []
    audit_rows = []
    t0 = time.time()
    for rg in range(pf.num_row_groups):
        raw = pf.read_row_group(rg).to_pandas()
        red = choose_medoids(raw, k=5, bridge_type=bridge)
        writer = write_table(writer, dst, red)
        first = red.drop_duplicates(["scenario_id"])
        for r in first.itertuples(index=False):
            weight_rows.append(
                {
                    "mode": mode,
                    "load_profile_name": profile,
                    "bridge_type": bridge,
                    "issue_time": r.issue_time,
                    "scenario_id": int(r.scenario_id),
                    "scenario_weight": float(r.scenario_weight),
                    "medoid_type": r.medoid_type,
                }
            )
        audit_rows.append(
            {
                "mode": mode,
                "load_profile_name": profile,
                "bridge_type": bridge,
                "issue_time": raw["issue_time"].iloc[0],
                "max_medoid_netload_kw": float(red["netload_s_kw"].max()),
                "min_weight": float(first["scenario_weight"].min()),
                "max_weight": float(first["scenario_weight"].max()),
                "effective_scenarios": float(1.0 / (first["scenario_weight"].astype(float) ** 2).sum()),
                "tail_medoid_count": int(first["medoid_type"].astype(str).str.contains("tail").sum()),
            }
        )
        if (rg + 1) % 1000 == 0:
            print(f"{mode}-{profile}-{bridge}: reduced {rg+1}/{pf.num_row_groups} issues in {time.time()-t0:.0f}s")
    if writer is not None:
        writer.close()
    return (
        {"mode": mode, "profile": profile, "bridge_type": bridge, "issues": pf.num_row_groups, "path": str(dst), "elapsed_sec": time.time() - t0},
        weight_rows,
        audit_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--profiles", nargs="*", default=PROFILES)
    parser.add_argument("--mode", choices=["DA", "ID_H24", "both"], default="both")
    args = parser.parse_args()
    ensure_out()
    summaries, weights, audits = [], [], []
    for profile in args.profiles:
        for mode in (["DA", "ID_H24"] if args.mode == "both" else [args.mode]):
            for bridge in BRIDGES:
                s, w, a = reduce_file(mode, profile, bridge, args.M)
                summaries.append(s)
                weights.extend(w)
                audits.extend(a)
    pd.DataFrame(weights).to_csv(OUT_DIR / "pvfocus_scenario_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(OUT_DIR / "pvfocus_medoid_selection_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT_DIR / "pvfocus_scenario_reduction_summary.csv", index=False, encoding="utf-8-sig")
    write_md(
        OUT_DIR / "PV_FOCUSED_SCENARIO_REDUCTION_REPORT.md",
        "PV-Focused Scenario Reduction Report",
        [
            ("Status", "Generated PV-KM, PV-TKM, and PV-TKM-safe reduced K=5 packages for each requested load profile."),
            ("Weights", "Scenario weights are empirical cluster masses, not equal weights."),
            ("Tail-Safe Logic", "PV-TKM-safe uses tail-band medoid selection to preserve high net-load / DCT-risk behavior without selecting an unexplained pathological extreme."),
            ("Summary", pd.DataFrame(summaries).to_markdown(index=False)),
        ],
    )
    print(f"[PVFocus-4] reduction complete under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
