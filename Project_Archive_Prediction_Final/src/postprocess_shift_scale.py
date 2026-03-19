# src/postprocess_shift_scale.py
# 以「桶」為單位，先做中位數位移校正，再做區間放寬/收窄。
# 支援 q 欄位別名、自動辨識 ts / ts_local、夜間=0、晴空上限、非交叉修正。

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pvlib.location import Location
from pvlib import solarposition

def parse_args():
    p = argparse.ArgumentParser("Bucket-wise shift + scale for quantile forecasts")
    p.add_argument("--in",  dest="parquet_in",  required=True, help="input parquet")
    p.add_argument("--out", dest="parquet_out", required=True, help="output parquet")
    p.add_argument("--tz",  default="Asia/Taipei")
    p.add_argument("--lat", type=float, default=25.0377)
    p.add_argument("--lon", type=float, default=121.5149)

    # 分桶設定
    p.add_argument("--use-sky-regime", action="store_true",
                   help="若資料內已有 sky_regime 欄位則一併分桶")
    p.add_argument("--min-samples", type=int, default=60,
                   help="每桶最少樣本；不足則退回較粗分桶（只用 hour）")
    p.add_argument("--elev-bin-deg", type=int, default=10,
                   help="太陽高度分箱寬度（度）")

    # 位移（shift）
    p.add_argument("--do-shift", action="store_true",
                   help="啟用位移式校正：以各桶 y_true - q50 的中位數作為位移量")
    p.add_argument("--shift-metric", choices=["median", "mean"], default="median",
                   help="位移量統計量（預設 median）")
    p.add_argument("--shift-cap", type=float, default=150.0,
                   help="單點位移量上限（W/m²），避免極端值")

    # 區間縮放（scale）
    g = p.add_mutually_exclusive_group()
    g.add_argument("--target-coverage", type=float, default=0.0,
                   help="以各桶 coverage 目標→估計縮放倍率（如 0.80/0.90）。0 表示不用")
    g.add_argument("--fixed-scale", type=float, default=0.0,
                   help="固定縮放倍率（>1 放寬，<1 收窄）；0 表示不用")
    p.add_argument("--max-boost", type=float, default=1.6,
                   help="各桶縮放倍率上限（避免過度放寬）")
    p.add_argument("--min-boost", type=float, default=0.8,
                   help="各桶縮放倍率下限（避免過度收窄）")

    # 物理限制
    p.add_argument("--twilight-deg", type=float, default=0.0,
                   help="<=此高度視為夜間，q 全設 0（預設 0 度）")
    p.add_argument("--cap-scale", type=float, default=1.10,
                   help="晴空上限倍率（e.g. 1.10 * clearsky_ghi）")

    # 附加輸出
    p.add_argument("--write-summary", action="store_true",
                   help="輸出 bucket 前後指標到同資料夾（CSV）")
    return p.parse_args()

def _pick_ts(df, tz):
    if "ts_local" in df:  # 已是本地時間且 Naive
        ts = pd.to_datetime(df["ts_local"], errors="coerce")
        return pd.DatetimeIndex(ts), ts.notna().to_numpy()
    elif "ts" in df:      # UTC→本地
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(tz)
        return pd.DatetimeIndex(ts), (~ts.isna()).to_numpy()
    else:
        raise ValueError("No time column: expect ts or ts_local")

def _pick_qcols(df):
    # 允許 pred_q10_cal / q_0.1 等命名
    cand = {
        "q10": ["pred_q10_cal","pred_q10","q_0.1","q10"],
        "q50": ["pred_q50_cal","pred_q50","q_0.5","q50"],
        "q90": ["pred_q90_cal","pred_q90","q_0.9","q90"],
    }
    out = {}
    for k, names in cand.items():
        for n in names:
            if n in df.columns:
                out[k] = n
                break
    if set(out) != {"q10","q50","q90"}:
        raise ValueError(f"Quantile columns not found. Got {out}")
    # y_true
    ytrue = None
    for c in df.columns:
        if "true" in c.lower():
            ytrue = c; break
    if ytrue is None:
        raise ValueError("y_true column not found.")
    return out, ytrue

def _build_buckets(df, times, lat, lon, tz, elev_bin_deg, use_sky_regime):
    # hour + elev_bin (+ sky_regime if exists & flag)
    elev = solarposition.get_solarposition(times, lat, lon)["apparent_elevation"].to_numpy()
    elev_bin = np.clip((np.maximum(elev,0)//elev_bin_deg*elev_bin_deg).astype(int), 0, 80)
    df["hour"] = pd.Series(times.hour, index=df.index, dtype="int16")
    df["elev_bin"] = pd.Series(elev_bin, index=df.index, dtype="int16")

    keys = ["hour","elev_bin"]
    if use_sky_regime and ("sky_regime" in df.columns):
        keys.append("sky_regime")
    return elev, keys

def _coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean()) if len(y) else np.nan

def main():
    args = parse_args()
    inp, outp = Path(args.parquet_in), Path(args.parquet_out)
    df = pd.read_parquet(inp)
    tz, lat, lon = args.tz, args.lat, args.lon

    # 時間與欄位
    times, mask = _pick_ts(df, tz)
    df = df.loc[mask].reset_index(drop=True)
    times = times[mask]
    qmap, ytrue = _pick_qcols(df)

    # 分桶鍵與物理量
    elev, keys = _build_buckets(df, times, lat, lon, tz, args.elev_bin_deg, args.use_sky_regime)
    loc = Location(lat, lon, tz=tz)
    cs = loc.get_clearsky(times, model="ineichen")["ghi"].to_numpy()
    night = elev <= max(args.twilight_deg, 0.0)
    cap = args.cap_scale * cs

    q10, q50, q90 = qmap["q10"], qmap["q50"], qmap["q90"]
    y   = df[ytrue].to_numpy()
    lo  = df[q10].to_numpy().astype(float)
    mid = df[q50].to_numpy().astype(float)
    hi  = df[q90].to_numpy().astype(float)

    # ===== 事前統計（可選輸出）=====
    if args.write_summary:
        before = (
            df.groupby(keys, observed=True)
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "cov10_90": _coverage(g[ytrue], g[q10], g[q90]),
                  "bias_q50": float(np.median(g[ytrue]-g[q50])) if len(g)>0 else np.nan
              }))
              .reset_index()
        )

    # ===== 1) 按桶位移（shift）=====
    if args.do_shift:
        # 計算各桶位移量：median(y_true - q50) 或 mean
        if args.shift_metric == "median":
            agg = df.groupby(keys, observed=True).apply(
                lambda g: np.median(g[ytrue].to_numpy() - g[q50].to_numpy())
            )
        else:
            agg = df.groupby(keys, observed=True).apply(
                lambda g: float(np.mean(g[ytrue].to_numpy() - g[q50].to_numpy()))
            )
        # 映射到每筆
        shift_map = {tuple(idx): float(np.clip(val, -args.shift_cap, args.shift_cap))
                     for idx, val in agg.items()}
        # 逐點套用位移（同量加到 q10/50/90）
        def _get_shift(i):
            key = tuple(df.loc[i, k] for k in keys)
            return shift_map.get(key, 0.0)
        shifts = np.array([_get_shift(i) for i in range(len(df))], dtype=float)

        lo += shifts; mid += shifts; hi += shifts

    # ===== 2) 區間縮放（scale：放寬>1 / 收窄<1）=====
    if args.fixed_scale and args.fixed_scale > 0:
        boost = np.full(len(df), float(args.fixed_scale), dtype=float)
    elif args.target_coverage and args.target_coverage > 0:
        # 先算各桶 coverage，再以 target/cov 做倍率
        cov = ( (df[ytrue].to_numpy() >= lo) & (df[ytrue].to_numpy() <= hi) ).astype(float)
        cov_bucket = pd.Series(cov).groupby([df[k] for k in keys], observed=True).mean()
        boost_map = {tuple(idx): float(np.clip(
            (args.target_coverage / c) if c>0 else args.max_boost,
            args.min_boost, args.max_boost))
            for idx, c in cov_bucket.items()}
        def _get_boost(i):
            key = tuple(df.loc[i, k] for k in keys)
            return boost_map.get(key, 1.0)
        boost = np.array([_get_boost(i) for i in range(len(df))], dtype=float)
    else:
        boost = np.ones(len(df), dtype=float)

    # 以中位數為中心做對稱縮放
    lo = mid - (mid - lo) * boost
    hi = mid + (hi - mid) * boost

    # ===== 3) 物理與數學約束 =====
    # 非交叉
    lo = np.minimum(lo, hi)
    mid = np.clip(mid, lo, hi)

    # 夜間 = 0
    lo[night] = 0.0; mid[night] = 0.0; hi[night] = 0.0

    # 晴空上限
    lo = np.clip(lo, 0.0, cap)
    mid = np.clip(mid, 0.0, cap)
    hi = np.clip(hi, 0.0, cap)

    # 邊界保底：若 lo>hi（極少數），拉回到 mid
    bad = lo > hi
    if bad.any():
        lo[bad] = mid[bad]
        hi[bad] = mid[bad]

    # 回填
    df[q10], df[q50], df[q90] = lo, mid, hi

    # ===== 事後統計（可選輸出）=====
    if args.write_summary:
        after = (
            df.groupby(keys, observed=True)
              .apply(lambda g: pd.Series({
                  "n": len(g),
                  "cov10_90": _coverage(g[ytrue], g[q10], g[q90]),
                  "bias_q50": float(np.median(g[ytrue]-g[q50])) if len(g)>0 else np.nan
              }))
              .reset_index()
        )
        outdir = outp.parent
        before.to_csv(outdir / (outp.stem + "_bucket_before.csv"), index=False, encoding="utf-8-sig")
        after.to_csv(outdir  / (outp.stem + "_bucket_after.csv"),  index=False, encoding="utf-8-sig")

    # 儲存
    df.to_parquet(outp, index=False)
    # 簡單回報
    cov_overall = ((df[ytrue] >= df[q10]) & (df[ytrue] <= df[q90])).mean()
    print(f"Saved → {outp} | overall coverage(Q10–Q90)={cov_overall:.3f}")
