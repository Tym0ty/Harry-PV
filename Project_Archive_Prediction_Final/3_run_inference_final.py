import pandas as pd
import numpy as np
import xgboost as xgb
import json
import yaml
import pvlib
from pathlib import Path
from tqdm import tqdm

def run_inference(config_path="configs/exp_xgbq_v4_multilead.yaml"):
    print("--- Running Inference (V6 Final: Matched Features) ---")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. 讀取與處理資料
    data_path = "data/interim/merged_feature_store_v4.parquet"
    print(f"Reading Data: {data_path}")
    df = pd.read_parquet(data_path)
    
    if 'ts' in df.columns: df['ts'] = pd.to_datetime(df['ts']); df = df.set_index('ts').sort_index()
    elif 'timestamp' in df.columns: df['timestamp'] = pd.to_datetime(df['timestamp']); df = df.set_index('timestamp').sort_index()

    # --- 🧹 Cleaning Data Types ---
    print("--- 🧹 Cleaning Data Types ---")
    MASTER_CWA_OBS = [
        'Station_Pressure_hPa', 'Sea_Level_Pressure_hPa', 'Air_Temperature_C', 
        'Relative_Humidity_percent', 'Wind_Speed_ms', 'Wind_Direction_degree', 
        'Gust_Speed_ms', 'Gust_Direction_degree', 'Precipitation_mm', 
        'Sunshine_Duration_hour', 'Visibility_km', 'Total_Cloud_Amount_tenths'
    ]
    
    for col in MASTER_CWA_OBS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Data Fixes (Physics) ---
    print("--- 🔧 Applying Data Fixes ---")
    lat = config['site_meta']['lat']; lon = config['site_meta']['lon']
    site = pvlib.location.Location(lat, lon, tz='Asia/Taipei')
    tz_idx = df.index.tz_localize(None).tz_localize('Asia/Taipei')
    solpos = site.get_solarposition(tz_idx, method='ephemeris')
    
    # [關鍵修正] 修正 pvlib 輸出欄位名稱
    df['solar_zenith'] = solpos['zenith'].values       # <<< 修正點
    df['solar_elevation'] = solpos['elevation'].values
    df['solar_azimuth'] = solpos['azimuth'].values
    
    df['cs_ghi'] = site.get_clearsky(tz_idx, model='ineichen')['ghi'].values
    
    df = df.ffill().fillna(0)

    # --- Feature Engineering V6 (Lag and Roll) ---
    print("--- 🔄 Feature Engineering V6 (Match Training) ---")
    target_col = "ghi_cwa_wm2"
    if target_col not in df.columns: target_col = "y_true"

    # 1. Target Lags
    if target_col in df.columns:
        df[f'{target_col}_lag24'] = df[target_col].shift(24)
        df[f'{target_col}_lag48'] = df[target_col].shift(48)
        df[f'{target_col}_lag72'] = df[target_col].shift(72)
        df[f'{target_col}_roll3_mean'] = df[[f'{target_col}_lag24', f'{target_col}_lag48', f'{target_col}_lag72']].mean(axis=1)

    # 2. CWA Observation Lags
    for col in MASTER_CWA_OBS:
        if col in df.columns:
            df[f"{col}_lag24"] = df[col].shift(24)
            df[f"{col}_lag48"] = df[col].shift(48)
            if col in ['Sunshine_Duration_hour', 'Total_Cloud_Amount_tenths']:
                 df[f'{col}_roll2_mean'] = df[[f'{col}_lag24', f'{col}_lag48']].mean(axis=1)
            
    df = df.fillna(0)

    # 3. 載入模型與特徵
    model_dir = Path("models/xgbq_v4_multilead")
    feature_list_path = model_dir / "feature_list.json"
    
    with open(feature_list_path, "r") as f:
        feature_cols = json.load(f)
    
    # 補齊缺失特徵
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
            
    X_infer = df[feature_cols]
    dtest = xgb.DMatrix(X_infer)

    # 4. 預測
    quantiles = config['quantiles']
    preds = {}
    
    for q in tqdm(quantiles, desc="Predicting Quantiles"):
        q_str = str(q).replace('.', 'p')
        model_file = model_dir / f"beyond_24h_booster_{q_str}.json"
        
        if not model_file.exists():
            print(f"❌ Final Error: Model file not found: {model_file}")
            return
            
        model = xgb.Booster()
        model.load_model(model_file)
        preds[f'pred_q{q_str}'] = model.predict(dtest)

    # 5. 輸出
    output_df = df[[target_col]].copy()
    output_df['pred_q10_cal'] = preds.get('pred_q0p1', np.zeros(len(df)))
    output_df['pred_q50_cal'] = preds.get('pred_q0p5', np.zeros(len(df)))
    output_df['pred_q90_cal'] = preds.get('pred_q0p9', np.zeros(len(df)))
    output_df['hour'] = df.index.hour
    output_df['cs_ghi'] = df['cs_ghi']
    
    output_df = output_df.iloc[72:].copy()
    
    out_path = "data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon.parquet"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(out_path)
    print(f"✅ Inference saved to: {out_path}")

if __name__ == "__main__":
    run_inference()