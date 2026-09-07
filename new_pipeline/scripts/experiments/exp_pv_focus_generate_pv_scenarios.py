#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import pandas as pd

from pv_focus_bridge_utils import (
    OUT_DIR,
    ensure_out,
    full_da_pv_quantiles,
    full_id_pv_quantiles,
    simulate_pv_for_issue,
    write_md,
    write_table,
)


def generate(mode: str, m: int) -> dict:
    data = full_da_pv_quantiles() if mode == "DA" else full_id_pv_quantiles()
    path = OUT_DIR / (f"da_pv_raw_scenarios_M{m}.parquet" if mode == "DA" else f"id_h24_pv_raw_scenarios_M{m}.parquet")
    if path.exists():
        path.unlink()
    writer = None
    t0 = time.time()
    issue_count = 0
    rows = 0
    for issue, g in data.groupby("issue_time", sort=True):
        raw = simulate_pv_for_issue(g, m, mode)
        writer = write_table(writer, path, raw)
        issue_count += 1
        rows += len(raw)
        if issue_count % 500 == 0:
            print(f"{mode}: generated {issue_count} issues in {time.time()-t0:.0f}s")
    if writer is not None:
        writer.close()
    return {"mode": mode, "M": m, "issue_count": issue_count, "rows": rows, "path": str(path), "elapsed_sec": time.time() - t0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--mode", choices=["DA", "ID_H24", "both"], default="both")
    args = parser.parse_args()
    ensure_out()
    rows = []
    if args.mode in {"DA", "both"}:
        rows.append(generate("DA", args.M))
    if args.mode in {"ID_H24", "both"}:
        rows.append(generate("ID_H24", args.M))
    pd.DataFrame(rows).to_csv(OUT_DIR / "pv_raw_scenario_generation_summary.csv", index=False, encoding="utf-8-sig")
    write_md(
        OUT_DIR / "PV_RAW_SCENARIO_GENERATION_REPORT.md",
        "PV Raw Scenario Generation Report",
        [
            ("Status", "Generated DA and ID H24 raw PV scenarios from calibrated PV quantiles using inverse-CDF uniform sampling."),
            ("Safety", "Realized PV was not used to generate scenarios. Quantile artifacts may contain truth columns, but this script reads only forecast quantile columns."),
            ("Summary", pd.DataFrame(rows).to_markdown(index=False)),
        ],
    )
    print(f"[PVFocus-2] generated raw PV scenarios under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
