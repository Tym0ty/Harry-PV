"""
milp_v2/common.py — Shared utilities: config loading, TOU, monthly charge.

Standalone — does not import anything from notebooks_milp/ or thesis_pipeline/.
"""
import math
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Compute CRF at load time
    r = cfg["battery"]["discount_rate"]
    n = cfg["battery"]["lifetime_yr"]
    crf = r * (1 + r)**n / ((1 + r)**n - 1)
    cfg["battery"]["crf"] = round(crf, 6)

    return cfg


# ── TOU tariff ────────────────────────────────────────────────────────────────

def _is_summer(month: int, day: int) -> bool:
    """Summer season: May 16 – Oct 15 (TOU_FixedPeak spec §5.1)."""
    if month < 5 or month > 10:
        return False
    if month == 5:
        return day >= 16
    if month == 10:
        return day <= 15
    return True


def _build_tou_table() -> dict:
    tou = {}
    # Summer Mon-Fri
    for h in range(0, 9):   tou[("summer", "weekday", h)] = 2.53
    for h in range(9, 16):  tou[("summer", "weekday", h)] = 5.85
    for h in range(16, 22): tou[("summer", "weekday", h)] = 9.39
    for h in range(22, 24): tou[("summer", "weekday", h)] = 5.85
    # Non-summer Mon-Fri
    for h in range(0, 6):   tou[("nonsummer", "weekday", h)] = 2.32
    for h in range(6, 11):  tou[("nonsummer", "weekday", h)] = 5.47
    for h in range(11, 14): tou[("nonsummer", "weekday", h)] = 2.32
    for h in range(14, 24): tou[("nonsummer", "weekday", h)] = 5.47
    # Summer Saturday
    for h in range(0, 9):   tou[("summer", "saturday", h)] = 2.53
    for h in range(9, 24):  tou[("summer", "saturday", h)] = 2.60
    # Non-summer Saturday
    for h in range(0, 6):   tou[("nonsummer", "saturday", h)] = 2.32
    for h in range(6, 11):  tou[("nonsummer", "saturday", h)] = 2.41
    for h in range(11, 14): tou[("nonsummer", "saturday", h)] = 2.32
    for h in range(14, 24): tou[("nonsummer", "saturday", h)] = 2.41
    # Summer Sunday/holiday
    for h in range(24):     tou[("summer", "sunday", h)] = 2.53
    # Non-summer Sunday/holiday
    for h in range(24):     tou[("nonsummer", "sunday", h)] = 2.32
    return tou


_TOU_TABLE = _build_tou_table()


def get_tou_price(month: int, day: int, dow: int, hour_0based: int) -> float:
    """TOU price [NTD/kWh]. dow: 0=Mon..6=Sun. hour_0based: 0-23."""
    season = "summer" if _is_summer(month, day) else "nonsummer"
    if dow < 5:
        day_type = "weekday"
    elif dow == 5:
        day_type = "saturday"
    else:
        day_type = "sunday"
    return _TOU_TABLE[(season, day_type, hour_0based)]


def get_basic_charge(month: int, cfg: dict) -> float:
    """Monthly basic charge rate [NTD/kW/month]."""
    if month in (5, 6, 7, 8, 9, 10):
        return cfg["tariff"]["basic_charge_summer"]
    return cfg["tariff"]["basic_charge_nonsummer"]


# ── Calendar helper ───────────────────────────────────────────────────────────

def load_calendar(cfg: dict) -> pd.DataFrame:
    """Load caseyear calendar manifest, indexed by day_index."""
    cal_path = Path(cfg["paths"]["upstream_calendar"])
    if not cal_path.is_absolute():
        cal_path = ROOT / cal_path
    df = pd.read_parquet(cal_path)
    return df.set_index("day_index").sort_index()


def build_day_meta(cal: pd.DataFrame, cfg: dict) -> dict:
    """Return dict[day_index] -> {calendar_day, month_id, tou[24], dow}."""
    day_meta = {}
    for di in cal.index:
        row = cal.loc[di]
        cd  = pd.Timestamp(row["calendar_day"])
        dow = cd.weekday()
        mo  = int(row["month_id"])
        tou = np.array([get_tou_price(cd.month, cd.day, dow, t) for t in range(24)])
        day_meta[di] = {
            "calendar_day": cd,
            "month_id":     mo,
            "tou":          tou,
            "dow":          dow,
        }
    return day_meta
