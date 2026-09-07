#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import pandas as pd

from pv_focus_bridge_utils import OUT_DIR, PROFILES, OPTIONAL_PROFILES, ensure_out, load_profile_lookup, write_md, write_table


def combine_one(mode: str, profile: str, m: int) -> dict:
    src = OUT_DIR / (f"da_pv_raw_scenarios_M{m}.parquet" if mode == "DA" else f"id_h24_pv_raw_scenarios_M{m}.parquet")
    prefix = "da" if mode == "DA" else "id_h24"
    dst = OUT_DIR / f"{prefix}_pvfocus_{profile}_raw_netload_M{m}.parquet"
    if dst.exists():
        dst.unlink()
    loads = load_profile_lookup(profile)
    writer = None
    rows = 0
    t0 = time.time()
    # Read by row groups to avoid holding ID raw in memory.
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        df["target_time"] = pd.to_datetime(df["target_time"])
        df["load_profile_name"] = profile
        df["load_kw"] = df["target_time"].map(loads).astype("float32")
        df["netload_s_kw"] = (df["load_kw"].astype(float) - df["pv_s_kw"].astype(float)).astype("float32")
        out = df[["issue_time", "target_time", "lead_hour", "raw_scenario_id", "load_profile_name", "load_kw", "pv_s_kw", "netload_s_kw"]]
        writer = write_table(writer, dst, out)
        rows += len(out)
    if writer is not None:
        writer.close()
    return {"mode": mode, "profile": profile, "rows": rows, "path": str(dst), "elapsed_sec": time.time() - t0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--mode", choices=["DA", "ID_H24", "both"], default="both")
    args = parser.parse_args()
    ensure_out()
    profiles = PROFILES + (OPTIONAL_PROFILES if args.include_optional else [])
    rows = []
    for profile in profiles:
        if args.mode in {"DA", "both"}:
            rows.append(combine_one("DA", profile, args.M))
        if args.mode in {"ID_H24", "both"}:
            rows.append(combine_one("ID_H24", profile, args.M))
    pd.DataFrame(rows).to_csv(OUT_DIR / "pvfocused_netload_build_summary.csv", index=False, encoding="utf-8-sig")
    write_md(
        OUT_DIR / "PV_FOCUSED_NETLOAD_BUILD_REPORT.md",
        "PV-Focused Net-Load Build Report",
        [
            ("Status", "Built raw net-load scenario packages for each deterministic load profile separately."),
            ("Method", "`netload_s_kw = load_profile_kw - pv_s_kw`. PV is the stochastic source. Load profile is deterministic within each outer sensitivity case."),
            ("No Mixing", "Base, structured-low, and structured-high load profiles are not assigned probabilities and are not mixed into one probabilistic scenario set."),
            ("Summary", pd.DataFrame(rows).to_markdown(index=False)),
        ],
    )
    print(f"[PVFocus-3] built raw netload scenarios under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
