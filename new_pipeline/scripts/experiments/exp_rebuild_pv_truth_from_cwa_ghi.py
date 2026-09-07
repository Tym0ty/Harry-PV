#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild"
OLD_TRUTH = REPO / "bridge_outputs_fullyear/full_year_replay_truth_package.parquet"
CWA_GHI = REPO / "new_pipeline/data/output/stage1_aci_calibrated_da_v2.parquet"
PV_CAP_KWP = 2687.0
PR = 0.80


def write_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines += [f"## {h}", "", body.strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def ghi_to_pv_kw(ghi_wm2) -> np.ndarray:
    arr = np.asarray(ghi_wm2, dtype=float)
    return np.clip(np.where(arr > 0, PR * arr / 1000.0 * PV_CAP_KWP, 0.0), 0.0, PV_CAP_KWP)


def timestamp_from_truth(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["calendar_day"]) + pd.to_timedelta(df["hour_local"].astype(int) - 1, unit="h")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    truth = pd.read_parquet(OLD_TRUTH).copy()
    truth["timestamp"] = timestamp_from_truth(truth)

    ghi = pd.read_parquet(CWA_GHI).copy()
    ghi["calendar_day"] = pd.to_datetime(ghi["target_day"])
    ghi["hour_local"] = ghi["hour_local"].astype(int)
    ghi["timestamp"] = pd.to_datetime(ghi["target_day"]) + pd.to_timedelta(ghi["hour_local"].astype(int) - 1, unit="h")
    ghi = ghi[["calendar_day", "hour_local", "timestamp", "ghi_realized", "ghi_clear_sky"]].drop_duplicates(["calendar_day", "hour_local"])
    ghi["pv_cwa_ghi_kw"] = ghi_to_pv_kw(ghi["ghi_realized"].fillna(0.0))

    rebuilt = truth.drop(columns=["pv_realized_kw"], errors="ignore").merge(
        ghi[["calendar_day", "hour_local", "ghi_realized", "ghi_clear_sky", "pv_cwa_ghi_kw"]],
        on=["calendar_day", "hour_local"],
        how="left",
    )
    rebuilt["ghi_realized"] = rebuilt["ghi_realized"].fillna(0.0)
    rebuilt["ghi_clear_sky"] = rebuilt["ghi_clear_sky"].fillna(0.0)
    rebuilt["pv_realized_kw"] = rebuilt["pv_cwa_ghi_kw"].fillna(0.0)
    rebuilt["pv_truth_source"] = "CWA_GHI_REALIZED_TO_PV"
    rebuilt["pv_truth_formula"] = f"clip({PR} * ghi_realized_Wm2 / 1000 * {PV_CAP_KWP}, 0, {PV_CAP_KWP})"
    rebuilt["pv_source_validity"] = "VALID_BY_USER_METHOD_DECISION"

    # Preserve original schema-like order with appended audit fields.
    ordered = [
        "day_index",
        "calendar_day",
        "hour_local",
        "pv_realized_kw",
        "load_realized_kw",
        "month_id",
        "day_type",
        "season_tag",
        "is_holiday",
        "ghi_realized",
        "ghi_clear_sky",
        "pv_truth_source",
        "pv_truth_formula",
        "pv_source_validity",
    ]
    rebuilt = rebuilt[ordered]
    package_path = OUT / "full_year_replay_truth_package_CWA_GHI_PV.parquet"
    rebuilt.to_parquet(package_path, index=False)

    # Comparison is against invalid old PV only to estimate magnitude; do not use it as validation.
    old = pd.read_parquet(OLD_TRUTH).copy()
    old["timestamp"] = timestamp_from_truth(old)
    comp = old[["timestamp", "calendar_day", "hour_local", "pv_realized_kw"]].rename(columns={"pv_realized_kw": "old_invalid_pv_kw"}).merge(
        rebuilt.assign(timestamp=timestamp_from_truth(rebuilt))[["timestamp", "pv_realized_kw", "ghi_realized", "ghi_clear_sky"]],
        on="timestamp",
        how="left",
    )
    comp["cwa_pv_kw"] = comp["pv_realized_kw"]
    comp["delta_cwa_minus_old_invalid_kw"] = comp["cwa_pv_kw"] - comp["old_invalid_pv_kw"]
    comp.to_csv(OUT / "cwa_pv_truth_vs_old_invalid_pv_comparison.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for label, col in [("CWA_GHI_PV_TRUTH", "cwa_pv_kw"), ("OLD_INVALID_NTUST_SOLAR_SCALED", "old_invalid_pv_kw")]:
        s = comp[col].astype(float)
        summary_rows.append(
            {
                "series": label,
                "min_kw": float(s.min()),
                "max_kw": float(s.max()),
                "mean_kw": float(s.mean()),
                "p95_kw": float(s.quantile(0.95)),
                "p99_kw": float(s.quantile(0.99)),
                "annual_energy_kwh": float(s.sum()),
                "capacity_factor": float(s.sum() / (PV_CAP_KWP * 8760.0)),
                "zero_fraction": float((s.abs() < 1e-9).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "cwa_pv_truth_summary.csv", index=False, encoding="utf-8-sig")

    monthly = rebuilt.copy()
    monthly["timestamp"] = timestamp_from_truth(monthly)
    monthly_summary = (
        monthly.groupby("month_id", observed=True)
        .agg(
            cwa_pv_energy_kwh=("pv_realized_kw", "sum"),
            cwa_pv_max_kw=("pv_realized_kw", "max"),
            cwa_pv_mean_kw=("pv_realized_kw", "mean"),
            ghi_max_wm2=("ghi_realized", "max"),
            ghi_mean_wm2=("ghi_realized", "mean"),
        )
        .reset_index()
    )
    monthly_summary.to_csv(OUT / "cwa_pv_truth_monthly_summary.csv", index=False, encoding="utf-8-sig")

    status = pd.DataFrame(
        [
            {
                "artifact": str(package_path.relative_to(REPO)),
                "role": "replacement replay truth package",
                "pv_source": "CWA realized GHI converted to PV",
                "load_source": "carried from existing full_year_replay_truth_package load_realized_kw",
                "overwrites_existing": "NO",
                "ready_for_replay_rerun": "YES",
            },
            {
                "artifact": "bridge_outputs_fullyear/full_year_replay_truth_package.parquet",
                "role": "old replay truth package",
                "pv_source": "invalid NTUST_Load_PV.csv Solar_kWh scaled",
                "load_source": "padil gross load",
                "overwrites_existing": "NO",
                "ready_for_replay_rerun": "NO - PV invalid",
            },
        ]
    )
    status.to_csv(OUT / "pv_truth_replacement_status.csv", index=False, encoding="utf-8-sig")

    write_md(
        OUT / "CWA_GHI_PV_TRUTH_REBUILD_REPORT.md",
        "CWA GHI to PV Truth Rebuild Report",
        [
            (
                "Decision",
                "Valid PV truth is now defined as CWA realized GHI converted to PV using the project fixed PV capacity and PR. "
                "`NTUST_Load_PV.csv` PV / `Solar_kWh` is ignored.",
            ),
            (
                "Formula",
                f"`pv_realized_kw = clip({PR} * ghi_realized_Wm2 / 1000 * {PV_CAP_KWP}, 0, {PV_CAP_KWP})`",
            ),
            (
                "Generated Package",
                f"`{package_path.relative_to(REPO)}`\n\nThis does not overwrite `bridge_outputs_fullyear/full_year_replay_truth_package.parquet`.",
            ),
            ("Summary", summary.to_markdown(index=False)),
            ("Monthly Summary", monthly_summary.to_markdown(index=False)),
            (
                "Important Caveat",
                "The comparison against old PV is only a magnitude check. The old PV is invalid and must not be used as ground truth.",
            ),
            (
                "Next Step",
                "Patch replay/scheduling runners to accept this replacement truth path, then rerun the required DA/MPC/M8 or same-core cases. "
                "Do not update thesis final numbers until those reruns are complete.",
            ),
        ],
    )
    print(f"CWA PV truth package written: {package_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
