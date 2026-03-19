import argparse
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import shutil
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pvlib

def train_model(config_path):
    # 1. 讀取設定
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print("--- Running train_xgbq (V5.3: Strict Anti-Leakage Fix) ---")
    
    # 2. 讀取資料
    data_path = "data/interim/merged_feature_store_v4.parquet"
    print(f"Reading: {data_path}")
    df = pd.read_parquet(data_path)
    
    if 'ts' in df.columns: df['ts'] = pd.to_datetime(df['ts']); df = df.set_index('ts').sort_index()
    elif 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp']); df = df.set_index('timestamp').sort_index()

    # --- 強制資料清洗 ---
    print("--- 🧹 Cleaning Data Types ---")
    OBSERVATION_COLS = [
        'Sunshine_Duration_hour', 
        'Precipitation_mm', 
        'Total_Cloud_Amount_tenths', 
        'Visibility_km',
        'Temperature_C',
        'Humidity_percent'
    ]
    
    for col in OBSERVATION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Data Fixes ---
    print("--- 🔧 Applying Data Fixes ---")
    lat = config['site_meta']['lat']
    lon = config['site_meta']['lon']
    
    tz_index = df.index.tz_localize(None).tz_localize('Asia/Taipei')
    site = pvlib.location.Location(lat, lon, tz='Asia/Taipei')
    solpos = site.get_solarposition(tz_index, method='ephemeris')
    
    df['solar_zenith'] = solpos['zenith'].values
    df['solar_elevation'] = solpos['elevation'].values
    df['solar_azimuth'] = solpos['azimuth'].values
    
    df = df.ffill().fillna(0)

    # --- Feature Engineering V5 (Deep Lags) ---
    print("--- 🔄 Feature Engineering V5 (Deep Lags & Trends) ---")
    
    target_col = "ghi_cwa_wm2"
    if target_col not in df.columns:
        if "y_true" in df.columns: target_col = "y_true"
        elif "GHI" in df.columns: target_col = "GHI"

    # 1. GHI 自迴歸 (AR+)
    if target_col in df.columns:
        df[f'{target_col}_lag24'] = df[target_col].shift(24)
        df[f'{target_col}_lag48'] = df[target_col].shift(48)
        df[f'{target_col}_lag72'] = df[target_col].shift(72)
        df[f'{target_col}_roll3_mean'] = df[[f'{target_col}_lag24', f'{target_col}_lag48', f'{target_col}_lag72']].mean(axis=1)

    # 2. 觀測變數 Lag
    for col in OBSERVATION_COLS:
        if col in df.columns:
            df[f"{col}_lag24"] = df[col].shift(24)
            df[f"{col}_lag48"] = df[col].shift(48)
            if col in ['Sunshine_Duration_hour', 'Total_Cloud_Amount_tenths']:
                 df[f'{col}_roll2_mean'] = df[[f'{col}_lag24', f'{col}_lag48']].mean(axis=1)
    
    df = df.fillna(0)

    # 4. 定義特徵 (嚴格防洩漏)
    # [Critical Fix] 加入所有可能的觀測值變體
    DROP_COLS = [
        target_col, 'ts', 'timestamp', 
        'Global_Solar_Radiation_MJm2', 'ghi_cwa_mj', # <--- 兇手在這裡！
        'file_name', 'Station_No', 
        'h_angle', 'declination', 'dni_clear', 'dhi_clear'
    ]
    DROP_COLS.extend([c for c in OBSERVATION_COLS if c in df.columns])
    
    feature_candidates = [c for c in df.columns if c not in DROP_COLS]
    final_features = [c for c in feature_candidates if pd.api.types.is_numeric_dtype(df[c])]
            
    print(f"🧠 Training features ({len(final_features)})")
    # print(f"   (Leakage Check: 'ghi_cwa_mj' in features? {'ghi_cwa_mj' in final_features})")

    # 5. 切分資料集
    splits = config['splits']
    train_df = df.loc[splits['train'][0]:splits['train'][1]]
    valid_df = df.loc[splits['valid'][0]:splits['valid'][1]]

    # 日間過濾
    if config.get('day_training_mode') == 'day_only':
        tr_mask = train_df['solar_elevation'] > 0
        va_mask = valid_df['solar_elevation'] > 0
        X_train = train_df[tr_mask][final_features]
        y_train = train_df[tr_mask][target_col]
        X_valid = valid_df[va_mask][final_features]
        y_valid = valid_df[va_mask][target_col]
    else:
        X_train = train_df[final_features]
        y_train = train_df[target_col]
        X_valid = valid_df[final_features]
        y_valid = valid_df[target_col]

    # 6. 訓練
    quantiles = config['quantiles']
    model_dir = Path("models/xgbq_v4_multilead")
    if model_dir.exists(): shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "feature_list.json", "w") as f:
        json.dump(final_features, f)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # 刪除舊參數檔
    if Path("configs/best_params_optuna.json").exists():
        os.remove("configs/best_params_optuna.json")

    for q in quantiles:
        print(f"-- Training Q={q} --")
        # Grid Search Champion Params
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': q,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.75,
            'colsample_bytree': 0.8,
            'n_estimators': 2500,
            'n_jobs': -1,
            'random_state': 42
        }
        
        model = xgb.train(
            params, dtrain, num_boost_round=3000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50, verbose_eval=False
        )
        
        q_str = str(q).replace('.', 'p')
        model.save_model(model_dir / f"beyond_24h_booster_{q_str}.json")
        
        val_pred = model.predict(dvalid)
        mae = mean_absolute_error(y_valid, val_pred)
        print(f"   ✅ Valid MAE={mae:.3f}")

    print(f"✅ Models saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_xgbq_v4_multilead.yaml")
    args = parser.parse_args()
    train_model(args.config)