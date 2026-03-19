# ======================================================
# train_xgbq.py (V3 多時域版)
# ======================================================

import os
import json
import argparse
import pandas as pd
from pathlib import Path
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pytz
import yaml

# ======================================================
# 主訓練函式
# ======================================================

def train_model(config_path: str):
    # --- 讀取設定檔 ---
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("exp_name", "xgbq_v3_features")
    base_model_dir = Path(cfg.get("paths", {}).get("models_dir", f"models/{exp_name}"))
    base_model_dir.mkdir(parents=True, exist_ok=True)

    feature_store_path = Path(cfg.get("paths", {}).get("base_feature_store", "data/interim/merged_feature_store.parquet"))
    print(f"--- Running train_xgbq (V3 Features) ---")
    print(f"Models base dir: {base_model_dir}")
    print(f"Loading feature store from: {feature_store_path}")

    # --- 載入資料 ---
    df = pd.read_parquet(feature_store_path)
    print(f"Feature store loaded → {df.shape}")

    # --- 時間欄位轉換 ---
    if not np.issubdtype(df["ts"].dtype, np.datetime64):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # --- 取得 splits ---
    tz = pytz.timezone(cfg.get("timezone", "Asia/Taipei"))
    train_start, train_end = [pd.Timestamp(x).tz_localize(tz) for x in cfg["splits"]["train"]]
    valid_start, valid_end = [pd.Timestamp(x).tz_localize(tz) for x in cfg["splits"]["valid"]]
    test_start, test_end = [pd.Timestamp(x).tz_localize(tz) for x in cfg["splits"]["test"]]

    print("Splitting data by time (ts)...")
    df_train_full = df.query("@train_start <= ts < @train_end").copy()
    df_valid_full = df.query("@valid_start <= ts < @valid_end").copy()

    # --- Horizon 定義（V3 固定三時域）---
    horizons = {
        "short_term": (1, 6),
        "mid_term": (7, 12),
        "long_term": (13, 24)
    }

    # --- 特徵設定 ---
    features = [
        "Air_Temperature_C",
        "Relative_Humidity_percent",
        "Wind_Speed_ms",
        "Wind_Direction_degree",
        "Precipitation_mm",
        "Sunshine_Duration_hour",
        "Total_Cloud_Amount_tenths",
        "solar_zenith",
        "solar_elevation",
        "solar_azimuth",
        "ghi_clear",
        "dni_clear",
        "dhi_clear",
        "ghi_cwa_wm2_lag24h",
        "ghi_cwa_wm2"
    ]
    target = "ghi_cwa_wm2"
    quantiles = cfg.get("quantiles", [0.1, 0.5, 0.9])

    # 儲存特徵清單
    with open(base_model_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    print(f"Feature list saved to {base_model_dir / 'feature_list.json'}")

    # ======================================================
    # 多時域訓練
    # ======================================================
    for horizon_name, (h_min, h_max) in horizons.items():
        print(f"\n--- Training Horizon: {horizon_name} (Lead {h_min}-{h_max}h) ---")

        horizon_dir = base_model_dir / horizon_name
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # 篩選該 horizon 的資料（若 lead_hour 欄位存在）
        if "lead_hour" in df.columns:
            df_train_h = df_train_full.query("@h_min <= lead_hour <= @h_max")
            df_valid_h = df_valid_full.query("@h_min <= lead_hour <= @h_max")
        else:
            print("⚠️ Warning: 找不到 lead_hour 欄位，使用全資料訓練。")
            df_train_h = df_train_full
            df_valid_h = df_valid_full

        X_train, y_train = df_train_h[features], df_train_h[target]
        X_valid, y_valid = df_valid_h[features], df_valid_h[target]
        print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

        # ======================================================
        # 量化回歸 (Quantile Regression)
        # ======================================================
        for q in quantiles:
            print(f"-- Training Q={q} --")
            params = cfg.get("xgb_params", {}).copy()
            params["alpha"] = q
            params["objective"] = "reg:quantileerror"

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="quantile",
                verbose=100
            )

            # 儲存模型
            model_path = horizon_dir / f"booster_{str(q).replace('.', 'p')}.json"
            model.save_model(model_path)
            print(f"✅ Model saved: {model_path}")

    # --- 儲存 meta ---
    meta = {
        "experiment": exp_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quantiles": quantiles,
        "features": features,
        "target": target,
        "horizons": horizons
    }
    meta_path = base_model_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Training complete. Meta saved to {meta_path}")


# ======================================================
# 主程式入口點
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Quantile Models (V3 Multi-Horizon)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    train_model(args.config)
