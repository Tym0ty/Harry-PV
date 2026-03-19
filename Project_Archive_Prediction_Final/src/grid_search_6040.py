import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pvlib
import itertools
from pathlib import Path
from sklearn.metrics import mean_absolute_error

def run_grid_search():
    print("--- 🕵️ Grid Search for the 60.40 Configuration ---")
    
    # 1. 準備資料
    df = pd.read_parquet("data/interim/merged_feature_store_v4.parquet")
    if 'ts' in df.columns: df['ts'] = pd.to_datetime(df['ts']); df = df.set_index('ts').sort_index()
    elif 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp']); df = df.set_index('timestamp').sort_index()

    # Physics
    lat, lon = 25.0377, 121.5149
    site = pvlib.location.Location(lat, lon, tz='Asia/Taipei')
    tz_idx = df.index.tz_localize(None).tz_localize('Asia/Taipei')
    solpos = site.get_solarposition(tz_idx, method='ephemeris')
    df['solar_elevation'] = solpos['elevation'].values
    
    # Cloud Fix
    if 'Total_Cloud_Amount_tenths' in df.columns:
        mask = df['Total_Cloud_Amount_tenths'].isna()
        df.loc[mask, 'Total_Cloud_Amount_tenths'] = ((1.0 - df['Sunshine_Duration_hour'].fillna(0)) * 10.0).clip(0, 10)
    df = df.ffill().fillna(0)

    # V5 Features
    target_col = "ghi_cwa_wm2"
    if target_col not in df.columns: target_col = "y_true"
    
    df[f'{target_col}_lag24'] = df[target_col].shift(24)
    df[f'{target_col}_lag48'] = df[target_col].shift(48)
    df[f'{target_col}_lag72'] = df[target_col].shift(72)
    df[f'{target_col}_roll3_mean'] = df[[f'{target_col}_lag24', f'{target_col}_lag48', f'{target_col}_lag72']].mean(axis=1)

    OBS_COLS = ['Sunshine_Duration_hour', 'Precipitation_mm', 'Total_Cloud_Amount_tenths', 'Visibility_km', 'Temperature_C', 'Humidity_percent']
    for c in OBS_COLS:
        if c in df.columns:
            df[f"{c}_lag24"] = df[c].shift(24)
            df[f"{c}_lag48"] = df[c].shift(48)
            if c in ['Sunshine_Duration_hour', 'Total_Cloud_Amount_tenths']:
                 df[f'{c}_roll2_mean'] = df[[f'{c}_lag24', f'{c}_lag48']].mean(axis=1)
    df = df.fillna(0)
    
    # Anti-Leakage
    DROP = [target_col, 'ts', 'timestamp', 'Global_Solar_Radiation_MJm2', 'file_name', 'Station_No', 'h_angle', 'declination', 'dni_clear', 'dhi_clear']
    DROP.extend([c for c in OBS_COLS if c in df.columns])
    feats = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    
    # Splits
    train_df = df.loc["2021-03-31":"2024-04-30"]
    valid_df = df.loc["2024-05-01":"2024-10-31"]
    
    X_train = train_df[train_df['solar_elevation'] > 0][feats]
    y_train = train_df[train_df['solar_elevation'] > 0][target_col]
    X_valid = valid_df[valid_df['solar_elevation'] > 0][feats]
    y_valid = valid_df[valid_df['solar_elevation'] > 0][target_col]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # 定義網格
    learning_rates = [0.02, 0.035, 0.05]
    subsamples = [0.65, 0.75, 0.85] # 絕對不能是 1.0
    max_depths = [6, 8]
    
    print(f"{'LR':<6} {'Sub':<6} {'Depth':<6} | {'Valid MAE':<10}")
    print("-" * 40)

    best_mae = 999
    best_params = {}

    for lr, sub, depth in itertools.product(learning_rates, subsamples, max_depths):
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.5,
            'learning_rate': lr,
            'max_depth': depth,
            'subsample': sub,
            'colsample_bytree': 0.8, # 固定這個，通常 0.8 很穩
            'n_estimators': 2500,    # 給它足夠時間收斂
            'n_jobs': -1,
            'random_state': 42
        }
        
        # 訓練
        model = xgb.train(params, dtrain, num_boost_round=2500, evals=[(dvalid, 'val')], early_stopping_rounds=30, verbose_eval=False)
        mae = mean_absolute_error(y_valid, model.predict(dvalid))
        
        print(f"{lr:<6} {sub:<6} {depth:<6} | {mae:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_params = params
            
    print("-" * 40)
    print(f"🏆 Best Found: Valid MAE = {best_mae:.4f}")
    print(f"   Params: {best_params}")

    # 存檔
    final_params = {'q0.1': best_params, 'q0.5': best_params, 'q0.9': best_params}
    with open("configs/best_params_optuna.json", "w") as f:
        json.dump(final_params, f, indent=4)
    print("💾 Saved best params. Please re-run training now.")

if __name__ == "__main__":
    run_grid_search()