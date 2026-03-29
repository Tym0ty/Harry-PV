"""
Bridge Layer — F0329 Spec Compliance Version
Per F0329Bridge_Spec.pdf (Batch 8).

Reads upstream forecast/scenario artifacts + NTUST load/PV truth.
Outputs TWO coordinated artifact families:
  - Layer B rolling:  rolling_da_input_<pv>_<load>.parquet  (C0–C3 + C_PFI)
  - Layer A rep-day:  repday_input_<pv>_<load>.parquet       (via repday_builder)
Plus: caseyear_calendar_manifest, load_uncertainty_manifest,
      full_year_replay_truth_package, bridge_run_metadata.

Spec breach removed: fixed deterministic load uplift (5%/2%) → seasonal sigma
zero-mean Gaussian relative-error scenarios (spec §8).

Legacy alias map (old FF0326 filenames → new canonical names):
  full_year_milp_ingest_pvdet_loaddet    → rolling_da_input_pvdet_loaddet
  full_year_milp_ingest_pvprob_loaddet   → rolling_da_input_pvprob_loaddet
  full_year_milp_ingest_pvdet_loadpert   → rolling_da_input_pvdet_loadunc
  full_year_milp_ingest_pvprob_loadpert  → rolling_da_input_pvprob_loadunc
  load_perturbation_manifest             → load_uncertainty_manifest
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json, datetime

# ── Configuration ────────────────────────────────────────────
CASE_YEAR_START = "2024-10-01"   # spec: 2024/10 – 2025/10
CASE_YEAR_END   = "2025-10-31"
TZ = "Asia/Taipei"

PV_REF_KW     = 50.0      # forecast pipeline reference system
PV_FIXED_KW   = 2687.0    # target PV capacity (spec PV_001)
PV_SCALE      = PV_FIXED_KW / PV_REF_KW   # 53.74

NTUST_PV_KW   = 379.0
NTUST_PV_SCALE = PV_FIXED_KW / NTUST_PV_KW  # ~7.09

# ── Load uncertainty parameters (spec §8, F0329Bridge_Spec Batch 8) ──
# Seasonal MAPE → sigma via sigma = MAPE * sqrt(pi/2)
# Source: Pieter's forecast-error analysis; 1.0x baseline mainline
LOAD_UNC_SIGMA = {
    "summer": 0.1435,   # MAPE 11.45%
    "fall":   0.1837,   # MAPE 14.66%
    "winter": 0.1527,   # MAPE 12.18%
    "spring": 0.1540,   # MAPE 12.29%
}
LOAD_UNC_SIGMA_SCALE = 1.0     # 1.0x baseline; 0.5x or 1.5x for sensitivity
N_LOAD_UNC_SCENARIOS = 5       # pre-reduction scenario count for load uncertainty
LOAD_UNC_SEED = 425            # reproducibility seed

# Summer (billing calendar, spec CP_009/010): May 16 – Oct 15
SUMMER_START_MD = (5, 16)
SUMMER_END_MD   = (10, 15)

# Paths
ROOT        = Path(__file__).resolve().parent.parent
DATA_CSV    = ROOT / "NTUST_Load_PV.csv"
DATA_OLD_XLS = ROOT / "Project_Archive_Prediction_Final" / "data" / "raw" / "NTUST_Load_merged_fixed_v2.xlsx"
USE_OLD_LOAD = True
SCENARIO_PQ = ROOT / "pipeline_outputs" / "scenarios_joint_pv_load_reduced_5.parquet"
PV_DET_PQ   = ROOT / "pipeline_outputs" / "pv_point_forecast_caseyear.parquet"
OUT_DIR     = ROOT / "bridge_outputs_fullyear"

# Legacy alias map recorded in bridge_run_metadata.json
LEGACY_ALIAS_MAP = {
    "rolling_da_input_pvdet_loaddet.parquet":  "full_year_milp_ingest_pvdet_loaddet.parquet",
    "rolling_da_input_pvprob_loaddet.parquet": "full_year_milp_ingest_pvprob_loaddet.parquet",
    "rolling_da_input_pvdet_loadunc.parquet":  "full_year_milp_ingest_pvdet_loadpert.parquet",
    "rolling_da_input_pvprob_loadunc.parquet": "full_year_milp_ingest_pvprob_loadpert.parquet",
    "load_uncertainty_manifest.parquet":       "load_perturbation_manifest.parquet",
}


# ── Calendar helpers ─────────────────────────────────────────

def is_billing_summer(month, day):
    """Billing summer per Taipower: May 16 – Oct 15."""
    if month < 5 or month > 10:
        return False
    if month == 5:
        return day >= 16
    if month == 10:
        return day <= 15
    return True


def season_for_sigma(month):
    """Meteorological season for load uncertainty sigma lookup."""
    if month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "fall"
    elif month in (12, 1, 2):
        return "winter"
    else:   # 3, 4, 5
        return "spring"


def classify_day_type(dt):
    dow = dt.weekday()
    if dow < 5:
        return "weekday"
    elif dow == 5:
        return "saturday"
    return "sunday_holiday"


# ── Acceptance checks ────────────────────────────────────────

def _hard_fail(msg):
    raise RuntimeError(f"[BRIDGE HARD FAIL] {msg}")


def _check_truth_isolation(df, package_name):
    """Abort if truth columns appear in any C0-C3 solve package."""
    forbidden = {"pv_realized_kw", "load_realized_kw"}
    found = forbidden & set(df.columns)
    if found:
        _hard_fail(f"Truth columns {found} present in solve package '{package_name}'")


def _check_probability_sums(df, package_name):
    """Each (day_index, hour_local) must have probabilities summing to 1.0."""
    grp = df.groupby(["day_index", "hour_local"])["probability_pi"].sum()
    if not np.allclose(grp.values, 1.0, atol=1e-5):
        bad = grp[~np.isclose(grp.values, 1.0, atol=1e-5)]
        _hard_fail(f"Probability sums ≠ 1.0 in '{package_name}'. Examples:\n{bad.head()}")


def _check_24h_completeness(calendar_df, df, package_name):
    """Every (day_index, scenario_id) must have exactly 24 hour_local entries."""
    counts = df.groupby(["day_index", "scenario_id"])["hour_local"].count()
    bad = counts[counts != 24]
    if len(bad) > 0:
        _hard_fail(f"Incomplete 24-hour coverage in '{package_name}'. Offenders:\n{bad.head()}")


# ── Main ─────────────────────────────────────────────────────

def run_bridge():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Bridge Layer — F0329 Spec Compliance (Batch 8)")
    print("=" * 60)

    rng = np.random.default_rng(LOAD_UNC_SEED)

    # ── B0: Define case year ─────────────────────────────────
    date_range = pd.date_range(CASE_YEAR_START, CASE_YEAR_END, freq="D")
    n_days = len(date_range)
    print(f"B0: Case year {CASE_YEAR_START} → {CASE_YEAR_END} ({n_days} days)")

    # ── B1: Read upstream ────────────────────────────────────
    print("B1: Reading upstream data...")

    ntust = pd.read_csv(DATA_CSV).dropna(subset=["Date", "Time"])
    ntust["Date"] = pd.to_datetime(ntust["Date"])
    ntust["hour_0"] = ntust["Time"].str[:2].astype(int)

    if USE_OLD_LOAD:
        ntust_old = pd.read_excel(DATA_OLD_XLS)
        ntust_old["Date"] = pd.to_datetime(ntust_old["Date"])
        ntust_old["hour_0"] = ntust_old["Time"].str[:2].astype(int)
        old_load_map = {
            (row["Date"], row["hour_0"]): row["Load_kWh"]
            for _, row in ntust_old.iterrows()
        }
        print(f"  Old load: {len(ntust_old)} rows")

    # Probabilistic PV scenarios (50 kW reference)
    sc_raw = pd.read_parquet(SCENARIO_PQ)
    sc_raw["target_day_local"] = pd.to_datetime(sc_raw["target_day_local"])
    sc_raw["hour_0"] = sc_raw["target_time_local"].dt.hour
    print(f"  NTUST: {len(ntust)} rows | Scenarios: {len(sc_raw)} rows, "
          f"{sc_raw.scenario_id.nunique()} PV scenarios")

    # ── B2: Calendar manifest ────────────────────────────────
    print("B2: Building calendar manifest...")
    cal_rows = []
    for i, d in enumerate(date_range):
        m, day = d.month, d.day
        cal_rows.append({
            "day_index":    i + 1,
            "calendar_day": d.date(),
            "month_id":     m,
            "season_tag":   "summer" if is_billing_summer(m, day) else "non_summer",
            "sigma_season": season_for_sigma(m),
            "day_type":     classify_day_type(d),
            "is_holiday":   d.weekday() == 6,
            "is_summer":    is_billing_summer(m, day),
        })
    calendar_df = pd.DataFrame(cal_rows)
    calendar_df["calendar_day"] = pd.to_datetime(calendar_df["calendar_day"])
    day_to_idx = {pd.Timestamp(r["calendar_day"]): r["day_index"]
                  for _, r in calendar_df.iterrows()}

    # Acceptance check: no duplicate day_index
    if calendar_df["day_index"].duplicated().any():
        _hard_fail("Duplicate day_index entries in calendar manifest.")
    if calendar_df["calendar_day"].duplicated().any():
        _hard_fail("Duplicate calendar_day entries in calendar manifest.")

    calendar_df.to_parquet(OUT_DIR / "caseyear_calendar_manifest.parquet", index=False)
    print(f"  → caseyear_calendar_manifest.parquet: {n_days} days")

    # ── B3: Deterministic load baseline ──────────────────────
    print("B3: Building deterministic load baseline...")
    load_truth = {}     # (date, hour_local) -> kWh
    pv_realized = {}    # (date, hour_local) -> kW

    for _, row in ntust.iterrows():
        d = row["Date"]
        h0 = row["hour_0"]
        if h0 == 0:
            d = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            h_local = h0
        if d in day_to_idx:
            if USE_OLD_LOAD:
                old_val = old_load_map.get((row["Date"], row["hour_0"]))
                load_truth[(d, h_local)] = old_val if old_val is not None else row["Load_kWh"]
            else:
                load_truth[(d, h_local)] = row["Load_kWh"]
            pv_realized[(d, h_local)] = row["Solar_kWh"] * NTUST_PV_SCALE

    missing_load = sum(
        1 for d in date_range for h in range(1, 25)
        if (d, h) not in load_truth
    )
    print(f"  Missing load hours: {missing_load} / {n_days * 24}")
    if missing_load > n_days * 24 * 0.05:
        _hard_fail(f"Deterministic load baseline incomplete: {missing_load} missing hours "
                   f"({missing_load / (n_days * 24):.1%}). Threshold: 5%.")

    # ── B4: Pre-generate load uncertainty epsilons ───────────
    print("B4: Pre-generating load uncertainty scenarios "
          f"(σ_scale={LOAD_UNC_SIGMA_SCALE}×, N={N_LOAD_UNC_SCENARIOS}, seed={LOAD_UNC_SEED})...")

    # epsilon[day_idx, omega]: one daily scaling factor per (day, scenario)
    sigma_by_day = np.array([
        LOAD_UNC_SIGMA[calendar_df.iloc[i]["sigma_season"]] * LOAD_UNC_SIGMA_SCALE
        for i in range(n_days)
    ])
    # shape: (n_days, N_LOAD_UNC_SCENARIOS)
    load_unc_eps = np.zeros((n_days, N_LOAD_UNC_SCENARIOS))
    for i in range(n_days):
        load_unc_eps[i] = rng.normal(0.0, sigma_by_day[i], size=N_LOAD_UNC_SCENARIOS)

    # Clipping policy: clip epsilon to [-0.95, +2.0] to avoid negative or extreme loads
    load_unc_eps = np.clip(load_unc_eps, -0.95, 2.0)
    prob_load_unc = 1.0 / N_LOAD_UNC_SCENARIOS  # uniform pre-reduction probability

    # ── B5: PV data (deterministic + probabilistic) ───────────
    print("B5: Building PV data...")

    # Probabilistic PV
    sc_pv = {}  # (date, hour_local, scenario_id) -> (pv_kw, prob)
    for _, row in sc_raw.iterrows():
        d = row["target_day_local"]
        h0 = row["hour_0"]
        sid = row["scenario_id"]
        pv = row["pv_available_kw"] * PV_SCALE
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
    n_pv_scenarios = len(scenario_ids)

    # Deterministic PV (Q50 point forecast, pre-calibrated)
    pv_point = pd.read_parquet(PV_DET_PQ)
    pv_point["target_time_local"] = pd.to_datetime(pv_point["target_time_local"])
    pv_point["hour_0"] = pv_point["target_time_local"].dt.hour
    print(f"  Det PV rows: {len(pv_point)}")

    pv_det = {}
    for _, row in pv_point.iterrows():
        d = row["target_day_local"]
        h0 = row["hour_0"]
        pv_50kw = max(0.0, float(row["pv_point_kw"]))
        if h0 == 0:
            d_assign = d - pd.Timedelta(days=1)
            h_local = 24
        else:
            d_assign = pd.Timestamp(d)
            h_local = h0
        if d_assign in day_to_idx:
            pv_det[(d_assign, h_local)] = pv_50kw * PV_SCALE

    # Fill missing with 0
    for d in date_range:
        for h in range(1, 25):
            if (d, h) not in pv_det:
                pv_det[(d, h)] = 0.0

    # ── B6: Assemble rolling DA input packages ────────────────
    print("B6: Assembling rolling DA input packages...")

    def build_rolling_package(pv_mode, load_mode):
        """
        Build a full-year rolling day-ahead input package.

        pv_mode:   'pv_det'  | 'pv_prob'
        load_mode: 'load_det' | 'load_unc'

        Schema: day_index, calendar_day, hour_local, scenario_id,
                probability_pi, pv_mode, load_mode, pv_available_kw,
                load_kw, month_id, day_type, season_tag, is_holiday
        """
        rows = []
        for i_day, d in enumerate(date_range):
            d_ts = pd.Timestamp(d)
            cal = calendar_df[calendar_df["calendar_day"] == d_ts].iloc[0]
            di = cal["day_index"]

            # Deterministic load for this day
            load_det_arr = np.array([load_truth.get((d, h), 0.0) for h in range(1, 25)])

            if pv_mode == "pv_det" and load_mode == "load_det":
                # C0: one row per hour, scenario_id='det'
                for h in range(1, 25):
                    rows.append(_row(di, d, cal, h, "det", 1.0,
                                     pv_mode, load_mode,
                                     pv_det[(d, h)], load_det_arr[h - 1]))

            elif pv_mode == "pv_prob" and load_mode == "load_det":
                # C1: one row per (hour, pv_scenario)
                for sid in scenario_ids:
                    for h in range(1, 25):
                        pv_val, prob = sc_pv.get((d, h, sid), (0.0, 1.0 / n_pv_scenarios))
                        rows.append(_row(di, d, cal, h, f"pv_{sid}", prob,
                                         pv_mode, load_mode,
                                         pv_val, load_det_arr[h - 1]))

            elif pv_mode == "pv_det" and load_mode == "load_unc":
                # C2: one row per (hour, load_scenario)
                for omega in range(N_LOAD_UNC_SCENARIOS):
                    eps = load_unc_eps[i_day, omega]
                    for h in range(1, 25):
                        load_val = max(0.0, load_det_arr[h - 1] * (1.0 + eps))
                        sid = f"ls{omega:03d}"
                        rows.append(_row(di, d, cal, h, sid, prob_load_unc,
                                         pv_mode, load_mode,
                                         pv_det[(d, h)], load_val))

            elif pv_mode == "pv_prob" and load_mode == "load_unc":
                # C3: joint (pv_scenario × load_scenario)
                for i_pv, sid_pv in enumerate(scenario_ids):
                    for omega in range(N_LOAD_UNC_SCENARIOS):
                        eps = load_unc_eps[i_day, omega]
                        _, prob_pv = sc_pv.get((d, 12, sid_pv), (0.0, 1.0 / n_pv_scenarios))
                        joint_prob = prob_pv * prob_load_unc
                        sid_joint = f"pv_{sid_pv}_ls{omega:03d}"
                        for h in range(1, 25):
                            pv_val, _ = sc_pv.get((d, h, sid_pv), (0.0, 1.0 / n_pv_scenarios))
                            load_val = max(0.0, load_det_arr[h - 1] * (1.0 + eps))
                            rows.append(_row(di, d, cal, h, sid_joint, joint_prob,
                                             pv_mode, load_mode,
                                             pv_val, load_val))

        df = pd.DataFrame(rows)
        df["calendar_day"] = pd.to_datetime(df["calendar_day"])
        return df

    def _row(day_index, d, cal, hour_local, scenario_id, probability_pi,
             pv_mode, load_mode, pv_kw, load_kw):
        return {
            "day_index":       day_index,
            "calendar_day":    d.date(),
            "hour_local":      hour_local,
            "scenario_id":     scenario_id,
            "probability_pi":  probability_pi,
            "pv_mode":         pv_mode,
            "load_mode":       load_mode,
            "pv_available_kw": pv_kw,
            "load_kw":         load_kw,
            "month_id":        int(cal["month_id"]),
            "day_type":        cal["day_type"],
            "season_tag":      cal["season_tag"],
            "is_holiday":      bool(cal["is_holiday"]),
        }

    ROLLING_PACKAGES = {
        "rolling_da_input_pvdet_loaddet.parquet":  ("pv_det",  "load_det"),
        "rolling_da_input_pvprob_loaddet.parquet": ("pv_prob", "load_det"),
        "rolling_da_input_pvdet_loadunc.parquet":  ("pv_det",  "load_unc"),
        "rolling_da_input_pvprob_loadunc.parquet": ("pv_prob", "load_unc"),
    }

    pkg_scenario_counts = {}
    for fname, (pv_m, load_m) in ROLLING_PACKAGES.items():
        print(f"  Building {fname}...")
        df = build_rolling_package(pv_m, load_m)
        # Acceptance checks before writing
        _check_truth_isolation(df, fname)
        _check_probability_sums(df, fname)
        _check_24h_completeness(calendar_df, df, fname)
        df.to_parquet(OUT_DIR / fname, index=False)
        n_sc = df["scenario_id"].nunique()
        pkg_scenario_counts[fname] = n_sc
        print(f"    → {len(df)} rows, {n_sc} scenarios/day, "
              f"PV max={df['pv_available_kw'].max():.0f} kW, "
              f"Load max={df['load_kw'].max():.0f} kW")

    # ── B7: C_PFI rolling package ─────────────────────────────
    print("B7: Building C_PFI rolling package (perfect-forecast)...")
    cpfi_rows = []
    for d in date_range:
        d_ts = pd.Timestamp(d)
        cal = calendar_df[calendar_df["calendar_day"] == d_ts].iloc[0]
        di = cal["day_index"]
        for h in range(1, 25):
            cpfi_rows.append({
                "day_index":       di,
                "calendar_day":    d.date(),
                "hour_local":      h,
                "scenario_id":     "det",
                "probability_pi":  1.0,
                "pv_mode":         "pv_pfi",
                "load_mode":       "load_pfi",
                "pv_available_kw": pv_realized.get((d, h), 0.0),
                "load_kw":         load_truth.get((d, h), 0.0),
                "month_id":        int(cal["month_id"]),
                "day_type":        cal["day_type"],
                "season_tag":      cal["season_tag"],
                "is_holiday":      bool(cal["is_holiday"]),
            })
    cpfi_df = pd.DataFrame(cpfi_rows)
    cpfi_df["calendar_day"] = pd.to_datetime(cpfi_df["calendar_day"])
    # C_PFI uses pv_pfi/load_pfi — NOT truth columns — so no truth-isolation violation
    _check_24h_completeness(calendar_df, cpfi_df, "rolling_da_input_pvperfect_loadperfect.parquet")
    cpfi_df.to_parquet(OUT_DIR / "rolling_da_input_pvperfect_loadperfect.parquet", index=False)
    print(f"  → rolling_da_input_pvperfect_loadperfect.parquet: {len(cpfi_df)} rows")

    # ── B8: Truth replay package ──────────────────────────────
    print("B8: Building truth replay package...")
    truth_rows = []
    for d in date_range:
        d_ts = pd.Timestamp(d)
        cal = calendar_df[calendar_df["calendar_day"] == d_ts].iloc[0]
        for h in range(1, 25):
            truth_rows.append({
                "day_index":        cal["day_index"],
                "calendar_day":     d.date(),
                "hour_local":       h,
                "pv_realized_kw":   pv_realized.get((d, h), 0.0),
                "load_realized_kw": load_truth.get((d, h), 0.0),
                "month_id":         int(cal["month_id"]),
                "day_type":         cal["day_type"],
                "season_tag":       cal["season_tag"],
                "is_holiday":       bool(cal["is_holiday"]),
            })
    truth_df = pd.DataFrame(truth_rows)
    truth_df["calendar_day"] = pd.to_datetime(truth_df["calendar_day"])
    truth_df.to_parquet(OUT_DIR / "full_year_replay_truth_package.parquet", index=False)
    print(f"  → full_year_replay_truth_package.parquet: {len(truth_df)} rows")

    # ── B9: Load uncertainty manifest ────────────────────────
    unc_manifest = pd.DataFrame([{
        "load_mode":              "load_unc",
        "uncertainty_family":     "Gaussian_zero_mean",
        "sigma_scale_multiplier": LOAD_UNC_SIGMA_SCALE,
        "sigma_summer":           LOAD_UNC_SIGMA["summer"] * LOAD_UNC_SIGMA_SCALE,
        "sigma_fall":             LOAD_UNC_SIGMA["fall"]   * LOAD_UNC_SIGMA_SCALE,
        "sigma_winter":           LOAD_UNC_SIGMA["winter"] * LOAD_UNC_SIGMA_SCALE,
        "sigma_spring":           LOAD_UNC_SIGMA["spring"] * LOAD_UNC_SIGMA_SCALE,
        "mape_source_summer":     0.1145,
        "mape_source_fall":       0.1466,
        "mape_source_winter":     0.1218,
        "mape_source_spring":     0.1229,
        "sigma_conversion_rule":  "sigma = MAPE * sqrt(pi/2)",
        "n_scenarios_pre_reduction": N_LOAD_UNC_SCENARIOS,
        "epsilon_clipping_bounds":   "[-0.95, +2.0]",
        "epsilon_scope":          "daily (one epsilon per day-scenario, uniform across hours)",
        "seed":                   LOAD_UNC_SEED,
        "spec_ref":               "F0329Bridge_Spec Batch 8 §8",
    }])
    unc_manifest.to_parquet(OUT_DIR / "load_uncertainty_manifest.parquet", index=False)
    print("  → load_uncertainty_manifest.parquet")

    # ── B10: QA summary ──────────────────────────────────────
    print("B10: Running QA checks...")

    # Re-read C0 package for completeness check
    c0_pkg = pd.read_parquet(OUT_DIR / "rolling_da_input_pvdet_loaddet.parquet")
    hours_per_day = c0_pkg.groupby("day_index")["hour_local"].nunique()
    completeness_ok = bool((hours_per_day == 24).all())

    # Probability check for C1 probabilistic package
    c1_pkg = pd.read_parquet(OUT_DIR / "rolling_da_input_pvprob_loaddet.parquet")
    prob_grp = c1_pkg.groupby(["day_index", "hour_local"])["probability_pi"].sum()
    prob_ok = bool(np.allclose(prob_grp.values, 1.0, atol=1e-5))

    # C_PFI must have probability_pi=1.0 and no pv_realized/load_realized columns
    cpfi_check = pd.read_parquet(OUT_DIR / "rolling_da_input_pvperfect_loadperfect.parquet")
    cpfi_prob_ok = bool((cpfi_check["probability_pi"] == 1.0).all())
    cpfi_truth_isolation = bool(
        "pv_realized_kw" not in cpfi_check.columns and
        "load_realized_kw" not in cpfi_check.columns
    )

    report = {
        "spec_version":            "F0329Bridge_Spec Batch 8",
        "case_year_start":         CASE_YEAR_START,
        "case_year_end":           CASE_YEAR_END,
        "n_case_year_days":        n_days,
        "missing_load_hours":      int(missing_load),
        "n_pv_scenarios":          n_pv_scenarios,
        "n_load_unc_scenarios":    N_LOAD_UNC_SCENARIOS,
        "load_unc_sigma_scale":    LOAD_UNC_SIGMA_SCALE,
        "load_unc_seed":           LOAD_UNC_SEED,
        "probability_check_pass":  prob_ok,
        "completeness_24h_pass":   completeness_ok,
        "cpfi_prob_1_0_pass":      cpfi_prob_ok,
        "cpfi_truth_isolation_pass": cpfi_truth_isolation,
        "pkg_scenario_counts":     pkg_scenario_counts,
        "rolling_packages_written": list(ROLLING_PACKAGES.keys()) + [
            "rolling_da_input_pvperfect_loadperfect.parquet"
        ],
        "load_uncertainty_model":  "zero_mean_gaussian_seasonal_sigma (NOT deterministic uplift)",
        "schema_version":          "F0329_v1",
        "created_at":              datetime.datetime.now().isoformat(),
    }
    with open(OUT_DIR / "bridge_full_year_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── B11: Run metadata + legacy alias map ──────────────────
    metadata = {
        "bridge_version":          "F0329_v1",
        "spec_ref":                "F0329Bridge_Spec.pdf Batch 8",
        "case_year_boundary":      [CASE_YEAR_START, CASE_YEAR_END],
        "timezone":                TZ,
        "pv_reference_kw":         PV_REF_KW,
        "pv_fixed_kw":             PV_FIXED_KW,
        "ntust_pv_estimated_kw":   NTUST_PV_KW,
        "load_uncertainty_model": {
            "mode":         "zero_mean_gaussian_seasonal_sigma",
            "sigma_scale":  LOAD_UNC_SIGMA_SCALE,
            "seasonal_sigma": {k: v * LOAD_UNC_SIGMA_SCALE for k, v in LOAD_UNC_SIGMA.items()},
            "seasonal_mape_source": {
                "summer": 0.1145, "fall": 0.1466, "winter": 0.1218, "spring": 0.1229,
            },
            "conversion_rule": "sigma = MAPE * sqrt(pi/2)",
            "n_scenarios_pre_reduction": N_LOAD_UNC_SCENARIOS,
            "seed": LOAD_UNC_SEED,
            "epsilon_scope": "daily",
            "clipping": "[-0.95, +2.0]",
        },
        "legacy_alias_map":        LEGACY_ALIAS_MAP,
        "created_at":              datetime.datetime.now().isoformat(),
    }
    with open(OUT_DIR / "bridge_run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Bridge F0329 complete!")
    print(f"  Output: {OUT_DIR}")
    print(f"  Days: {n_days}  |  Missing load: {missing_load} hrs")
    print(f"  PV scale: {PV_SCALE:.2f}× (50 kW → 2687 kW)")
    print(f"  Load unc: seasonal σ × {LOAD_UNC_SIGMA_SCALE} | {N_LOAD_UNC_SCENARIOS} scenarios/day")
    print(f"  Prob check (C1): {'PASS' if prob_ok else 'FAIL'}")
    print(f"  24h completeness: {'PASS' if completeness_ok else 'FAIL'}")
    print(f"  C_PFI prob=1.0:   {'PASS' if cpfi_prob_ok else 'FAIL'}")
    print(f"  C_PFI isolation:  {'PASS' if cpfi_truth_isolation else 'FAIL'}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_bridge()
