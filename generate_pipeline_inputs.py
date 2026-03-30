#!/usr/bin/env python3
"""
Generate Pipeline Inputs — Synthetic GHI Forecasts + Load Parquets
===================================================================
Because raw CWA/NWP data is absent from this machine, this script uses
pvlib clear-sky irradiance (Ineichen model) at the NTUST site to synthesise
all parquet files required by scripts 9, 10, 11, and the bridge layer.

Outputs (all into pipeline_outputs/):
  forecast_ghi_quantiles_daily_base_raw.parquet  — 19-quantile synthetic GHI
  load_deterministic_hourly.parquet              — hourly load from padil CSV
  pv_point_forecast_caseyear.parquet             — PV point forecast (scaled)

Run from repo root with the project .venv active:
    python generate_pipeline_inputs.py
"""

import numpy as np
import pandas as pd
import pvlib
from pathlib import Path

# ── Site constants ───────────────────────────────────────────────────────────
LAT       = 25.0135    # NTUST campus, Taipei
LON       = 121.5432
ALT       = 9.0        # metres above sea level
TZ        = "Asia/Taipei"

# ── PVWatts parameters (must match 11_regenerate_scenarios.py) ───────────────
PV_DC_RATED  = 50.0        # kW reference system (pipeline scale)
GAMMA_PDC    = -0.0047     # temperature coefficient (1/°C)
INV_EFF      = 0.96
DC_AC_RATIO  = 1.2
PV_AC_RATED  = PV_DC_RATED / DC_AC_RATIO
PV_FIXED_KW  = 2687.0      # target installed capacity
PV_SCALE     = PV_FIXED_KW / PV_DC_RATED   # 53.74×

MONTHLY_TEMP = {1:16, 2:17, 3:19, 4:23, 5:26, 6:28, 7:30, 8:30, 9:28, 10:25, 11:21, 12:18}

# ── Data split periods ────────────────────────────────────────────────────────
CALIB_START = "2024-04-01"   # warm-season calibration data (Apr–Oct 2024)
CALIB_END   = "2024-10-31"
TEST_START  = "2024-11-01"   # formal case year: Nov 2024 – Oct 2025
TEST_END    = "2025-10-31"

# ── Noise / quantile spread parameters ───────────────────────────────────────
# Seasonal cloud-cover spread (fraction of clear-sky GHI used as std-dev)
# Represents realistic forecast uncertainty at each season.
CLOUD_SIGMA = {1: 0.35, 2: 0.35, 3: 0.30, 4: 0.25,  5: 0.22, 6: 0.20,
               7: 0.20, 8: 0.22, 9: 0.25, 10: 0.30, 11: 0.35, 12: 0.38}
QUANTILES   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SEED        = 42

ROOT       = Path(__file__).resolve().parent
PADIL_CSV  = ROOT / "padil_NTUST_Load_PV.csv"
OUT_DIR    = ROOT / "pipeline_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    """Hourly timestamps in Asia/Taipei, from start 01:00 through end 24:00."""
    # We want hours 1..24 (hour_local convention), so shift: hour 1 = 00:00 in hour-ending
    # Use standard pd.date_range (hour 0..23) and convert; scripts use dt.hour so we keep 0-based.
    idx = pd.date_range(start=start, end=end, freq="h", tz=TZ, inclusive="both")
    # Drop midnight of the first day so we start at 01:00 if needed; keep all 24h per day.
    return idx


def clear_sky_ghi(idx: pd.DatetimeIndex) -> pd.Series:
    """Compute Ineichen clear-sky GHI for the given hourly index."""
    loc  = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    # Convert to UTC for pvlib; pvlib handles tz-aware DatetimeIndex natively
    cs   = loc.get_clearsky(idx, model="ineichen")
    return cs["ghi"]


def solar_elevation(idx: pd.DatetimeIndex) -> pd.Series:
    """Compute solar elevation angle (degrees) for each timestamp."""
    loc  = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    sol  = loc.get_solarposition(idx)
    return sol["elevation"]


def pvwatts_kw(ghi_series: pd.Series, scale: float = 1.0) -> pd.Series:
    """PVWatts-style single-diode approximation: GHI → AC kW output.

    Parameters match 11_regenerate_scenarios.py exactly.
    scale: multiplier applied after PV_DC_RATED conversion (for site capacity).
    """
    results = []
    for ts, ghi_wm2 in ghi_series.items():
        ghi = max(0.0, float(ghi_wm2))
        if ghi < 1.0:
            results.append(0.0)
            continue
        month = ts.month if hasattr(ts, "month") else pd.Timestamp(ts).month
        t_cell = MONTHLY_TEMP[month] + (INOCT_APPROX - 20.0) * (ghi / 800.0)
        p_dc = PV_DC_RATED * (ghi / 1000.0) * (1 + GAMMA_PDC * (t_cell - 25.0))
        p_ac = min(p_dc * INV_EFF, PV_AC_RATED)
        results.append(max(0.0, p_ac) * scale)
    return pd.Series(results, index=ghi_series.index)

# Nominal operating cell temperature approximation
INOCT_APPROX = 49.0


def build_synthetic_quantiles(cs_ghi: pd.Series, elev: pd.Series,
                               split_name_map: dict) -> pd.DataFrame:
    """Build synthetic 19-quantile GHI forecasts.

    Strategy:
      1. Clear-sky GHI is the maximum; realistic GHI = clear-sky × cloud_factor.
      2. cloud_factor drawn from Beta(2, 1.5) to skew toward clear skies.
      3. Quantiles spread symmetrically around the median forecast using
         a seasonal sigma, then clipped to [0, clear-sky].
      4. Monotone rearrangement applied.
      5. label_ghi_obs_wm2 = realized GHI (slightly different from Q50).
    """
    rng = np.random.default_rng(SEED)

    rows = []
    dates = cs_ghi.index.date
    unique_days = sorted(set(dates))

    for day in unique_days:
        mask     = dates == day
        day_idx  = np.where(mask)[0]
        cs_day   = cs_ghi.iloc[day_idx].values
        elev_day = elev.iloc[day_idx].values
        ts_day   = cs_ghi.index[day_idx]
        month    = pd.Timestamp(day).month
        sigma    = CLOUD_SIGMA[month]
        split    = split_name_map.get(day, "test")

        # Per-day cloud factor (same for all hours of the day — realistic daily cloud)
        cloud_factor = float(rng.beta(2.5, 1.8))

        for k, (ts, cs_val, el) in enumerate(zip(ts_day, cs_day, elev_day)):
            if cs_val < 1.0 or el <= 0:
                # Night or near-zero sky
                q_vals = np.zeros(len(QUANTILES))
                obs    = 0.0
            else:
                # Forecast median = cloud_factor * clear-sky + small hourly jitter
                hourly_jitter = float(rng.normal(0, 0.05 * sigma * cs_val))
                q50_raw = max(0.0, cloud_factor * cs_val + hourly_jitter)

                # Spread quantiles around q50 using seasonal sigma
                # For the forecast model, uncertainty grows with clear-sky magnitude
                spread_std = sigma * cs_val
                # Map each quantile to a Gaussian percentile
                from scipy import stats as sp_stats
                q_offsets = np.array([sp_stats.norm.ppf(q) * spread_std for q in QUANTILES])
                q_vals = np.clip(q50_raw + q_offsets, 0.0, cs_val * 1.05)
                q_vals = np.sort(q_vals)  # monotone rearrangement

                # Observed GHI = q50 ± small independent observation noise
                obs_noise = float(rng.normal(0, 0.08 * spread_std))
                obs = float(np.clip(q50_raw + obs_noise, 0.0, cs_val * 1.1))

            # horizon_hour: hours since issue_day 12:00 UTC (~20:00 Taipei)
            # For day-ahead, hour h of target day = 24 + h (approx)
            h_local = ts.hour
            row = {
                "target_time_local": ts,
                "target_day_local":  pd.Timestamp(day),
                "split_name":        split,
                "label_ghi_obs_wm2": obs,
                "ghi_cs_wm2":        float(cs_val),
                "ghi_clear_wm2":     float(cs_val),
                "solar_elevation":   float(el),
                "horizon_hour":      h_local + 1,  # 1-indexed hour_local convention
            }
            for i, q in enumerate(QUANTILES):
                row[f"q{q:.2f}"] = float(q_vals[i])
            rows.append(row)

    df = pd.DataFrame(rows)
    df["target_time_local"] = pd.to_datetime(df["target_time_local"])
    df["target_day_local"]  = pd.to_datetime(df["target_day_local"])
    return df


def build_load_parquet() -> pd.DataFrame:
    """Read padil_NTUST_Load_PV.csv, clean and output hourly load_kw."""
    print(f"  Reading {PADIL_CSV} ...")
    df = pd.read_csv(PADIL_CSV, encoding="utf-8-sig")

    # Column cleanup — strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Keep only needed columns; handle extra blank columns
    if "Load_kWh" not in df.columns:
        raise ValueError(f"'Load_kWh' column not found. Columns: {df.columns.tolist()}")

    # Parse from Date + Time columns (DateTime column has data corruption on some rows)
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "Load_kWh"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Load_kWh for 1-hour interval ≡ average kW
    df["load_kw"] = pd.to_numeric(df["Load_kWh"], errors="coerce").fillna(0.0)

    # Shift back 1 hour: padil uses hour-ending convention (1:00 = period 0:00-1:00)
    # Script 11 looks up target_time = day + timedelta(hours=h) for h in 0..23
    df["target_time_local"] = df["timestamp"] - pd.Timedelta(hours=1)

    out = df[["target_time_local", "load_kw"]].copy()
    print(f"  Load rows: {len(out)} | Range: {out['target_time_local'].min()} → {out['target_time_local'].max()}")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generate Pipeline Inputs (pvlib synthetic)")
    print("=" * 60)

    # ── 1. Build hourly time index covering calib + test ─────────────────────
    print("\n[1/3] Building synthetic GHI quantile forecasts...")
    calib_idx = make_hourly_index(CALIB_START, CALIB_END)
    test_idx  = make_hourly_index(TEST_START, TEST_END)
    full_idx  = calib_idx.append(test_idx)

    # split_name mapping by date
    calib_dates = set(calib_idx.date)
    test_dates  = set(test_idx.date)
    split_map   = {d: "calib" for d in calib_dates}
    split_map.update({d: "test" for d in test_dates})

    # ── 2. Compute clear-sky GHI and solar elevation ──────────────────────────
    print("  Computing pvlib clear-sky GHI (Ineichen)...")
    cs_ghi = clear_sky_ghi(full_idx)
    elev   = solar_elevation(full_idx)

    # ── 3. Synthetic quantile forecasts ──────────────────────────────────────
    print("  Building synthetic quantile rows...")
    df_raw = build_synthetic_quantiles(cs_ghi, elev, split_map)

    out_path = OUT_DIR / "forecast_ghi_quantiles_daily_base_raw.parquet"
    df_raw.to_parquet(out_path, index=False)
    print(f"  ✓ Saved: {out_path} ({len(df_raw)} rows)")

    # QA check
    qcols = [f"q{q:.2f}" for q in QUANTILES]
    n_test  = (df_raw["split_name"] == "test").sum()
    n_calib = (df_raw["split_name"] == "calib").sum()
    print(f"    calib={n_calib} rows, test={n_test} rows")
    assert df_raw[qcols].isna().sum().sum() == 0, "NaN in quantile columns!"
    print("    QA: no NaN in quantile columns ✓")

    # ── 4. Load deterministic hourly ─────────────────────────────────────────
    print("\n[2/3] Building load_deterministic_hourly.parquet from padil CSV...")
    load_df = build_load_parquet()

    # Filter to case year only (test period) — use target_time_local column
    mask = (load_df["target_time_local"] >= TEST_START) & (load_df["target_time_local"] <= TEST_END)
    load_df = load_df[mask].reset_index(drop=True)
    if len(load_df) < 8700:
        print(f"  WARNING: only {len(load_df)} load rows in case year (expected ~8760)")

    load_out = OUT_DIR / "load_deterministic_hourly.parquet"
    load_df.to_parquet(load_out, index=False)
    print(f"  ✓ Saved: {load_out} ({len(load_df)} rows)")

    # ── 5. PV point forecast (case year, scaled) ─────────────────────────────
    print("\n[3/3] Building pv_point_forecast_caseyear.parquet...")
    test_cs   = cs_ghi[TEST_START:TEST_END]
    test_elev = elev[TEST_START:TEST_END]

    print(f"  Applying PVWatts conversion ({PV_DC_RATED}kW DC → ×{PV_SCALE:.2f})...")
    # PVWatts: use reference scale (50 kW), bridge applies PV_SCALE itself
    pv_kw_ref = pvwatts_kw(test_cs, scale=1.0)   # 50 kW reference system

    # Zero out night hours
    pv_kw_ref = pv_kw_ref.where(test_elev > 0, 0.0)

    # Bridge expects: target_time_local (col), target_day_local (col), pv_point_kw (50kW ref)
    pv_df = pd.DataFrame({
        "target_time_local": test_cs.index,
        "target_day_local":  test_cs.index.date,
        "pv_point_kw":       pv_kw_ref.values,
    })
    pv_df["target_time_local"] = pv_df["target_time_local"].dt.tz_localize(None)
    pv_df["target_day_local"]  = pd.to_datetime(pv_df["target_day_local"])

    pv_out = OUT_DIR / "pv_point_forecast_caseyear.parquet"
    pv_df.to_parquet(pv_out, index=False)
    print(f"  ✓ Saved: {pv_out} ({len(pv_df)} rows)")
    print(f"    Max PV (50kW ref): {pv_df['pv_point_kw'].max():.2f} kW, "
          f"Annual (scaled): {pv_df['pv_point_kw'].sum() * PV_SCALE:.0f} kWh")

    print("\n" + "=" * 60)
    print("Pipeline inputs generated successfully.")
    print(f"  {out_path.name}")
    print(f"  {load_out.name}")
    print(f"  {pv_out.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
