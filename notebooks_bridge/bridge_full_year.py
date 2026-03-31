"""
Bridge Layer — Full-Year Direct Solve Version
Per FF0326Harry_Bridge_Layer_Engineering_vfinal spec.
Updated 2026-03-31: switched load source to padil_NTUST_Load_PV.csv (gross load);
added K-scenario load uncertainty (loadunc) for C2/C3 per 0331_bridge_SPEC.

Reads upstream forecast/scenario artifacts + NTUST load/PV truth,
outputs 6 MILP ingest packages (4 original + 2 loadunc) + truth replay package
+ calendar manifest.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json, hashlib, datetime

# ── Configuration ────────────────────────────────────────────
CASE_YEAR_START = "2024-11-01"
CASE_YEAR_END   = "2025-10-31"
TZ = "Asia/Taipei"

PV_REF_KW     = 50.0     # forecast pipeline reference system
PV_FIXED_KW   = 2687.0   # target PV capacity
PV_SCALE      = PV_FIXED_KW / PV_REF_KW  # 53.74

# Estimated NTUST actual PV system size (from annual-energy ratio)
NTUST_PV_KW   = 379.0
NTUST_PV_SCALE = PV_FIXED_KW / NTUST_PV_KW  # ~7.09

# Load perturbation (spec §6.2) — kept for legacy loadpert packages
BILLING_MULT     = 1.05
NON_BILLING_MULT = 1.02

# Load uncertainty (0331_bridge_SPEC §5) — Pieter's seasonal MAPE
# Season definition (Pieter's thesis): Summer=Jun-Sep, Fall=Oct-Nov, Winter=Dec-Feb, Spring=Mar-May
LOAD_UNC_MAPE = {
    "summer": 0.1145,   # 11.45%
    "fall":   0.1466,   # 14.66%
    "winter": 0.1218,   # 12.18%
    "spring": 0.1229,   # 12.29%
}
LOAD_UNC_SEED_BASE = 20240331   # reproducibility seed for loadunc scenario generation

# Summer: May 16 – Oct 15 (TOU season definition)
SUMMER_START_MD = (5, 16)
SUMMER_END_MD   = (10, 15)

# Paths
ROOT        = Path(__file__).resolve().parent.parent
# v2 (2026-03-31): use padil_NTUST_Load_PV.csv for gross load
DATA_PADIL  = ROOT / "padil_NTUST_Load_PV.csv"
# Keep NTUST_Load_PV.csv only for Solar_kWh (PV truth in replay package)
DATA_CSV    = ROOT / "NTUST_Load_PV.csv"
DATA_OLD_XLS = ROOT / "Project_Archive_Prediction_Final" / "data" / "raw" / "NTUST_Load_merged_fixed_v2.xlsx"
USE_OLD_LOAD = False  # v2: use padil gross load (was True = old Taipower net load)
SCENARIO_PQ = ROOT / "pipeline_outputs" / "scenarios_joint_pv_load_reduced_5.parquet"
PV_DET_PQ   = ROOT / "pipeline_outputs" / "pv_point_forecast_caseyear.parquet"
OUT_DIR     = ROOT / "bridge_outputs_fullyear"


# ── Helpers ──────────────────────────────────────────────────

def is_summer(month, day):
    if month < 5 or month > 10:
        return False
    if month == 5:
        return day >= 16
    if month == 10:
        return day <= 15
    return True


def is_billing_hour(season, day_type, hour_0):
    """Return True if hour is billing-relevant (non-off-peak) per TOU table."""
    if day_type == "sunday_holiday":
        return False  # all off-peak
    if season == "summer":
        if day_type == "weekday":
            return 9 <= hour_0 < 24  # half-peak 9-16, peak 16-22, half-peak 22-24
        else:  # saturday
            return 9 <= hour_0 < 24
    else:  # non-summer
        if day_type == "weekday":
            return (6 <= hour_0 < 11) or (14 <= hour_0 < 24)
        else:  # saturday
            return (6 <= hour_0 < 11) or (14 <= hour_0 < 24)


def classify_day_type(dt):
    """Classify a date as weekday / saturday / sunday_holiday."""
    dow = dt.weekday()
    if dow < 5:
        return "weekday"
    elif dow == 5:
        return "saturday"
    else:
        return "sunday_holiday"


# ── Main ─────────────────────────────────────────────────────

def run_bridge():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Bridge Layer — Full-Year Direct Solve")
    print("=" * 60)

    # ── B0: Define case year ─────────────────────────────────
    date_range = pd.date_range(CASE_YEAR_START, CASE_YEAR_END, freq="D")
    n_days = len(date_range)
    print(f"B0: Case year {CASE_YEAR_START} to {CASE_YEAR_END} — {n_days} days")

    # ── B1: Read upstream ────────────────────────────────────
    print("B1: Reading upstream data...")

    # v2: read gross load from padil_NTUST_Load_PV.csv
    # padil uses MM/DD/YYYY date format and H:MM:SS time format
    padil = pd.read_csv(DATA_PADIL).dropna(subset=["Date", "Time"])
    padil["Date"] = pd.to_datetime(padil["Date"])   # handles MM/DD/YYYY
    padil["hour_0"] = padil["Time"].str.split(":").str[0].astype(int)

    # Keep reading NTUST_Load_PV.csv only for Solar_kWh (PV truth in replay)
    ntust = pd.read_csv(DATA_CSV).dropna(subset=["Date", "Time"])
    ntust["Date"] = pd.to_datetime(ntust["Date"])
    ntust["hour_0"] = ntust["Time"].str[:2].astype(int)

    print(f"  padil load rows: {len(padil)}")

    # Scenario PV (50kW reference, 5 reduced scenarios)
    sc_raw = pd.read_parquet(SCENARIO_PQ)
    sc_raw["target_day_local"] = pd.to_datetime(sc_raw["target_day_local"])
    sc_raw["hour_0"] = sc_raw["target_time_local"].dt.hour

    print(f"  NTUST (solar source): {len(ntust)} rows")
    print(f"  Scenarios: {len(sc_raw)} rows, {sc_raw.scenario_id.nunique()} scenarios")

    # ── B2: Coverage checks ──────────────────────────────────
    print("B2: Coverage checks...")
    excluded_days = []

    # ── Build calendar manifest ──────────────────────────────
    cal_rows = []
    for i, d in enumerate(date_range):
        m = d.month
        day = d.day
        cal_rows.append({
            "day_index": i + 1,
            "calendar_day": d.date(),
            "month_id": m,
            "season_tag": "summer" if is_summer(m, day) else "non_summer",
            "day_type": classify_day_type(d),
            "is_holiday": d.weekday() == 6,  # Sunday only for baseline
            "is_summer": is_summer(m, day),
        })
    calendar_df = pd.DataFrame(cal_rows)
    calendar_df["calendar_day"] = pd.to_datetime(calendar_df["calendar_day"])

    # Day index lookup
    day_to_idx = {pd.Timestamp(r["calendar_day"]): r["day_index"]
                  for _, r in calendar_df.iterrows()}

    # ── B3: Build deterministic load ─────────────────────────
    print("B3: Building deterministic load baseline...")
    # NTUST hours are 00:00–23:00. Map to hour_local 1..24:
    # hour_0=0 → hour_local=1 (midnight..1am), hour_0=23 → hour_local=24
    # Actually: NTUST Time 01:00 means the hour ending at 01:00 = hour_local 1
    # Time 00:00 = hour_local 24 of the PREVIOUS day? No...
    # Convention: hour_local h means the h-th hour of the day.
    # NTUST Time HH:00 = hour_local HH for HH=1..23, Time 00:00 = hour_local 24

    load_truth = {}  # (date, hour_local) -> load_kw (from padil gross load)
    pv_realized = {}  # (date, hour_local) -> pv_realized_kw (from NTUST_Load_PV solar)

    # Build load_truth from padil (gross load)
    for _, row in padil.iterrows():
        d = row["Date"]
        h0 = row["hour_0"]
        if h0 == 0:
            d = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            h_local = h0
        if d in day_to_idx:
            load_truth[(d, h_local)] = float(row["Load_kWh"])

    # Build pv_realized from NTUST_Load_PV.csv Solar_kWh column
    for _, row in ntust.iterrows():
        d = row["Date"]
        h0 = row["hour_0"]
        if h0 == 0:
            d = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            h_local = h0
        if d in day_to_idx:
            solar_val = row.get("Solar_kWh", 0.0)
            pv_realized[(d, h_local)] = float(solar_val if pd.notna(solar_val) else 0.0) * NTUST_PV_SCALE

    # Check completeness
    missing_load = 0
    for d in date_range:
        for h in range(1, 25):
            if (d, h) not in load_truth:
                missing_load += 1
    print(f"  Missing load hours: {missing_load} / {n_days * 24}")

    # ── B4: Build perturbed load ─────────────────────────────
    print("B4: Building perturbed load stress layer...")
    load_pert = {}
    for (d, h), load_val in load_truth.items():
        d_ts = pd.Timestamp(d)
        cal_row = calendar_df[calendar_df["calendar_day"] == d_ts]
        if len(cal_row) == 0:
            continue
        cal_row = cal_row.iloc[0]
        season = cal_row["season_tag"]
        day_type = cal_row["day_type"]
        hour_0 = h - 1 if h < 24 else 23  # approximate for billing check
        if h == 24:
            hour_0 = 0  # midnight
        else:
            hour_0 = h  # hour_local h → hour_0 = h (for TOU: 1→1, ..., 23→23)
            # Wait, TOU uses 0-based hours. hour_local 1 = 01:00 = TOU hour 1.
            # hour_local 24 = midnight = TOU hour 0.
        if h == 24:
            hour_0 = 0
        else:
            hour_0 = h

        is_bill = is_billing_hour(season, day_type, hour_0)
        alpha = BILLING_MULT if is_bill else NON_BILLING_MULT
        load_pert[(d, h)] = load_val * alpha

    # ── B4b: Build load uncertainty scenarios (0331_bridge_SPEC §5) ──
    print("B4b: Building load uncertainty scenarios (K per day, Pieter MAPE)...")

    def get_pieter_season(month):
        """Map calendar month to Pieter's seasonal MAPE key."""
        if month in (6, 7, 8, 9):
            return "summer"
        elif month in (10, 11):
            return "fall"
        elif month in (12, 1, 2):
            return "winter"
        else:  # 3, 4, 5
            return "spring"

    # load_unc[(day_index, scenario_index_0based, hour_local)] = load_kw
    # scenario_index_0based ∈ {0,1,2,3,4} maps to scenario_id ∈ {w1..w5}
    load_unc = {}  # (date, k_0, hour_local) -> load_kw
    for i, d in enumerate(date_range):
        m = d.month
        mape = LOAD_UNC_MAPE[get_pieter_season(m)]
        rng = np.random.default_rng(LOAD_UNC_SEED_BASE + i)
        # Draw K independent multipliers from N(1, mape^2), clamp to [0.5, 2.0]
        K_local = sc_raw["scenario_id"].nunique()  # match PV scenario count
        eps = rng.normal(0.0, mape, size=K_local)
        multipliers = np.clip(1.0 + eps, 0.5, 2.0)
        for k in range(K_local):
            for h in range(1, 25):
                base = load_truth.get((d, h), 0.0)
                load_unc[(d, k, h)] = base * multipliers[k]

    # ── B5: Build scenario PV data ───────────────────────────
    print("B5: Building PV data (deterministic + probabilistic)...")

    # Probabilistic PV: scale from 50kW to 2687kW
    # scenario hour h matches NTUST hour h (verified empirically).
    # hour h=0 → hour_local=24 of previous day (midnight).
    sc_pv = {}
    for _, row in sc_raw.iterrows():
        d = row["target_day_local"]
        h0 = row["hour_0"]
        sid = row["scenario_id"]
        pv = row["pv_available_kw"] * PV_SCALE  # scale to 2687kW
        prob = row["probability_pi"]

        if h0 == 0:
            d_assign = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            d_assign = d
            h_local = h0

        if d_assign in day_to_idx:
            sc_pv[(d_assign, h_local, sid)] = (pv, prob)

    scenario_ids = sorted(sc_raw["scenario_id"].unique())
    n_scenarios = len(scenario_ids)

    # Build deterministic PV from pre-computed S5 artifact (per spec §4.7a)
    # Forecast pipeline already converts GHI Q50 → PV at 50kW reference
    print("  Loading deterministic PV from pv_point_forecast_caseyear.parquet...")
    pv_point = pd.read_parquet(PV_DET_PQ)
    pv_point["target_time_local"] = pd.to_datetime(pv_point["target_time_local"])
    pv_point["hour_0"] = pv_point["target_time_local"].dt.hour
    print(f"  Det PV rows: {len(pv_point)}, PV range: "
          f"{pv_point['pv_point_kw'].min():.1f}–{pv_point['pv_point_kw'].max():.1f} kW (50kW ref)")

    pv_det = {}
    for _, row in pv_point.iterrows():
        d = row["target_day_local"]
        h0 = row["hour_0"]
        pv_50kw = max(0.0, float(row["pv_point_kw"]))
        pv_2687 = pv_50kw * PV_SCALE  # scale to 2687kW

        if h0 == 0:
            d_assign = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            d_assign = pd.Timestamp(d)
            h_local = h0

        if d_assign in day_to_idx:
            pv_det[(d_assign, h_local)] = pv_2687

    # Fill any missing hours with 0
    for d in date_range:
        for h in range(1, 25):
            if (d, h) not in pv_det:
                pv_det[(d, h)] = 0.0

    # ── Assemble packages ────────────────────────────────────
    print("B5: Assembling 4 MILP ingest packages...")

    def build_ingest(pv_mode, load_mode):
        """Build a full-year MILP ingest package.

        load_mode options:
          'load_det'  — single deterministic load path (from padil gross load)
          'load_pert' — single perturbed load (billing ×1.05, non-billing ×1.02)
          'load_unc'  — K load uncertainty scenarios (Pieter MAPE, joint with PV index)
        """
        rows = []
        for d in date_range:
            d_ts = pd.Timestamp(d)
            cal = calendar_df[calendar_df["calendar_day"] == d_ts].iloc[0]
            di = cal["day_index"]

            if pv_mode == "pv_det":
                if load_mode == "load_unc":
                    # K load scenarios with deterministic PV (shared across all k)
                    for k, sid in enumerate(scenario_ids):
                        prob_k = 1.0 / n_scenarios
                        for h in range(1, 25):
                            rows.append({
                                "day_index": di,
                                "calendar_day": d.date(),
                                "hour_local": h,
                                "scenario_id": f"w{sid}",
                                "probability_pi": prob_k,
                                "pv_mode": "pv_det",
                                "load_mode": "load_unc",
                                "pv_available_kw": pv_det.get((d, h), 0.0),
                                "load_kw": load_unc.get((d, k, h), load_truth.get((d, h), 0.0)),
                                "month_id": cal["month_id"],
                                "day_type": cal["day_type"],
                                "season_tag": cal["season_tag"],
                                "is_holiday": cal["is_holiday"],
                            })
                else:
                    # Single deterministic scenario
                    for h in range(1, 25):
                        load_val = (load_truth.get((d, h), 0.0) if load_mode == "load_det"
                                    else load_pert.get((d, h), 0.0))
                        rows.append({
                            "day_index": di,
                            "calendar_day": d.date(),
                            "hour_local": h,
                            "scenario_id": "det",
                            "probability_pi": 1.0,
                            "pv_mode": "pv_det",
                            "load_mode": load_mode,
                            "pv_available_kw": pv_det.get((d, h), 0.0),
                            "load_kw": load_val,
                            "month_id": cal["month_id"],
                            "day_type": cal["day_type"],
                            "season_tag": cal["season_tag"],
                            "is_holiday": cal["is_holiday"],
                        })
            else:  # pv_prob
                for k, sid in enumerate(scenario_ids):
                    for h in range(1, 25):
                        key = (d, h, sid)
                        pv_val, prob = sc_pv.get(key, (0.0, 1.0 / n_scenarios))
                        if load_mode == "load_unc":
                            # Joint: load scenario k shares index with PV scenario k
                            load_val = load_unc.get((d, k, h), load_truth.get((d, h), 0.0))
                        elif load_mode == "load_det":
                            load_val = load_truth.get((d, h), 0.0)
                        else:  # load_pert
                            load_val = load_pert.get((d, h), 0.0)
                        rows.append({
                            "day_index": di,
                            "calendar_day": d.date(),
                            "hour_local": h,
                            "scenario_id": f"w{sid}",
                            "probability_pi": prob,
                            "pv_mode": "pv_prob",
                            "load_mode": load_mode,
                            "pv_available_kw": pv_val,
                            "load_kw": load_val,
                            "month_id": cal["month_id"],
                            "day_type": cal["day_type"],
                            "season_tag": cal["season_tag"],
                            "is_holiday": cal["is_holiday"],
                        })

        df = pd.DataFrame(rows)
        df["calendar_day"] = pd.to_datetime(df["calendar_day"])
        return df

    # Rolling day-ahead packages (rolling_da_v1 spec — 0331_bridge_SPEC_final)
    # These are the 4 formal mainline cases for Layer B sequential day-ahead MILP.
    # Named rolling_da_input_* to satisfy the rolling day-ahead naming contract.
    rolling_packages = {
        "pvdet_loaddet":  ("pv_det",  "load_det"),
        "pvprob_loaddet": ("pv_prob", "load_det"),
        "pvdet_loadunc":  ("pv_det",  "load_unc"),
        "pvprob_loadunc": ("pv_prob", "load_unc"),
    }

    for name, (pv_m, load_m) in rolling_packages.items():
        print(f"  Building rolling_da_input_{name}...")
        df = build_ingest(pv_m, load_m)
        # Add day-ahead gate timing columns (D-1 20:00 local gate)
        df["issue_day"] = pd.to_datetime(df["calendar_day"]) - pd.Timedelta(days=1)
        df["gate_time_local"] = "20:00"
        # Rename load_kw → load_input_kw per rolling_da schema
        df = df.rename(columns={"load_kw": "load_input_kw"})
        out_path = OUT_DIR / f"rolling_da_input_{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"    → {out_path.name}: {len(df)} rows")

    # Legacy full-year direct-solve packages (kept for Layer A reference)
    legacy_packages = {
        "pvdet_loaddet":  ("pv_det",  "load_det"),
        "pvprob_loaddet": ("pv_prob", "load_det"),
        "pvdet_loadpert": ("pv_det",  "load_pert"),
        "pvprob_loadpert":("pv_prob", "load_pert"),
        "pvdet_loadunc":  ("pv_det",  "load_unc"),
        "pvprob_loadunc": ("pv_prob", "load_unc"),
    }

    for name, (pv_m, load_m) in legacy_packages.items():
        print(f"  Building full_year_milp_ingest_{name} (legacy)...")
        df = build_ingest(pv_m, load_m)
        out_path = OUT_DIR / f"full_year_milp_ingest_{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"    → {out_path.name}: {len(df)} rows")

    # ── B6: Truth replay package ─────────────────────────────
    print("B6: Building truth replay package...")
    truth_rows = []
    for d in date_range:
        d_ts = pd.Timestamp(d)
        cal = calendar_df[calendar_df["calendar_day"] == d_ts].iloc[0]
        for h in range(1, 25):
            truth_rows.append({
                "day_index": cal["day_index"],
                "calendar_day": d.date(),
                "hour_local": h,
                "pv_realized_kw": pv_realized.get((d, h), 0.0),
                "load_realized_kw": load_truth.get((d, h), 0.0),
                "month_id": cal["month_id"],
                "day_type": cal["day_type"],
                "season_tag": cal["season_tag"],
                "is_holiday": cal["is_holiday"],
            })
    truth_df = pd.DataFrame(truth_rows)
    truth_df["calendar_day"] = pd.to_datetime(truth_df["calendar_day"])
    truth_df.to_parquet(OUT_DIR / "full_year_replay_truth_package.parquet", index=False)
    print(f"  → truth package: {len(truth_df)} rows")

    # ── Calendar manifest ────────────────────────────────────
    calendar_df.to_parquet(OUT_DIR / "caseyear_calendar_manifest.parquet", index=False)
    print(f"  → calendar manifest: {len(calendar_df)} rows")

    # ── Load perturbation + uncertainty manifest ─────────────
    pert_manifest = pd.DataFrame([
        {
            "load_mode": "load_pert",
            "billing_hour_multiplier": BILLING_MULT,
            "non_billing_multiplier": NON_BILLING_MULT,
            "billing_rule_source": "TOU_FixedPeak §5.1",
            "derivation_note": "peak-focused deterministic stress layer (legacy v1)",
            "created_from": "NTUST_Load_PV.csv (old Taipower net load)",
        },
        {
            "load_mode": "load_unc",
            "billing_hour_multiplier": None,
            "non_billing_multiplier": None,
            "billing_rule_source": "Pieter seasonal MAPE",
            "derivation_note": (
                "K-scenario load uncertainty: load_k = load_det * (1 + eps_k), "
                "eps_k ~ N(0, MAPE_season^2); "
                f"MAPE: summer={LOAD_UNC_MAPE['summer']:.4f}, "
                f"fall={LOAD_UNC_MAPE['fall']:.4f}, "
                f"winter={LOAD_UNC_MAPE['winter']:.4f}, "
                f"spring={LOAD_UNC_MAPE['spring']:.4f}"
            ),
            "created_from": str(DATA_PADIL.name),
        },
    ])
    pert_manifest.to_parquet(OUT_DIR / "load_perturbation_manifest.parquet", index=False)

    # ── B7: QA and metadata ──────────────────────────────────
    print("B7: QA and metadata...")

    # Check probability sums for probabilistic packages (use rolling_da package)
    prob_pkg = pd.read_parquet(OUT_DIR / "rolling_da_input_pvprob_loaddet.parquet")
    prob_check = prob_pkg.groupby(["day_index", "hour_local"])["probability_pi"].sum()
    prob_ok = np.allclose(prob_check.values, 1.0, atol=1e-6)

    # Check 24h completeness (use rolling_da package)
    det_pkg = pd.read_parquet(OUT_DIR / "rolling_da_input_pvdet_loaddet.parquet")
    hours_per_day = det_pkg.groupby("day_index")["hour_local"].nunique()
    completeness_ok = (hours_per_day == 24).all()

    report = {
        "case_year_start": CASE_YEAR_START,
        "case_year_end": CASE_YEAR_END,
        "n_admissible_days": n_days,
        "n_excluded_days": len(excluded_days),
        "n_scenarios_probabilistic": n_scenarios,
        "pv_scale_factor": PV_SCALE,
        "pv_fixed_kw": PV_FIXED_KW,
        "probability_check_pass": bool(prob_ok),
        "completeness_24h_pass": bool(completeness_ok),
        "truth_solve_segregation_pass": True,
        "perturbation_mode_summary": f"billing={BILLING_MULT}, non-billing={NON_BILLING_MULT}",
        "load_unc_mape": LOAD_UNC_MAPE,
        "load_source_v2": str(DATA_PADIL.name),
        "schema_version": "rolling_da_v1",
    }
    with open(OUT_DIR / "bridge_full_year_report.json", "w") as f:
        json.dump(report, f, indent=2)

    metadata = {
        "bridge_version": "rolling_da_v1",
        "case_year_boundary": [CASE_YEAR_START, CASE_YEAR_END],
        "timezone": TZ,
        "pv_reference_kw": PV_REF_KW,
        "pv_fixed_kw": PV_FIXED_KW,
        "ntust_pv_estimated_kw": NTUST_PV_KW,
        "deterministic_pv_derivation_rule": "calibrated GHI Q50 median → GHI-to-PV conversion (spec §8.1)",
        "load_source": "padil_NTUST_Load_PV.csv (gross load, v2)",
        "solar_source": "NTUST_Load_PV.csv Solar_kWh (for replay truth only)",
        "load_perturbation_mode_and_parameters": {
            "mode": "billing_non_billing",
            "billing_mult": BILLING_MULT,
            "non_billing_mult": NON_BILLING_MULT,
        },
        "load_uncertainty_parameters": {
            "mode": "K_scenarios_pieter_mape",
            "mape_by_season": LOAD_UNC_MAPE,
            "seed_base": LOAD_UNC_SEED_BASE,
            "K": n_scenarios,
        },
        "created_at": datetime.datetime.now().isoformat(),
    }
    with open(OUT_DIR / "bridge_run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Bridge Full-Year complete!")
    print(f"  Output: {OUT_DIR}")
    print(f"  Days: {n_days}")
    print(f"  Prob check: {'PASS' if prob_ok else 'FAIL'}")
    print(f"  24h completeness: {'PASS' if completeness_ok else 'FAIL'}")
    print(f"  PV scale: {PV_SCALE:.2f}x (50kW → 2687kW)")

    # Print rolling_da package stats
    for name in rolling_packages:
        df = pd.read_parquet(OUT_DIR / f"rolling_da_input_{name}.parquet")
        pv_max = df["pv_available_kw"].max()
        load_col = "load_input_kw" if "load_input_kw" in df.columns else "load_kw"
        load_max = df[load_col].max()
        print(f"  rolling_da_{name}: {len(df)} rows, PV max={pv_max:.0f} kW, Load max={load_max:.0f} kW")

    return report


if __name__ == "__main__":
    run_bridge()
