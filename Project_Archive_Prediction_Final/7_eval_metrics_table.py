import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

def pinball_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1.0) * error))

def evaluate_all_metrics(file_path):
    print(f"--- 📝 Comprehensive Metric Report ---")
    df = pd.read_parquet(file_path)

    # 1. 數據準備與 Test Set 鎖定
    if 'ts' not in df.columns:
        df = df.reset_index(names=['ts'])
    df['ts'] = pd.to_datetime(df['ts'])
    TEST_START_DATE = "2024-11-01"
    df_test_set = df[df['ts'] >= TEST_START_DATE].copy()

    # 2. 準備數據 (只看白晝)
    target_col = 'ghi_cwa_wm2'
    if target_col not in df_test_set.columns: target_col = 'y_true'
    
    mask = (df_test_set[target_col] > 10) | (df_test_set['pred_q50_cal'] > 10)
    df_eval = df_test_set[mask].copy()

    if len(df_eval) == 0:
        print("❌ Error: No daylight samples found in Test Set.")
        return

    # 3. 計算所有指標
    y_true = df_eval[target_col].values
    q50_pred = df_eval['pred_q50_cal'].values
    q10_pred = df_eval['pred_q10_cal'].values
    q90_pred = df_eval['pred_q90_cal'].values
    
    # A. Point Metrics
    mae = np.mean(np.abs(y_true - q50_pred))
    rmse = np.sqrt(np.mean((y_true - q50_pred)**2))
    
    # B. Probabilistic Metrics
    loss_q10 = pinball_loss(y_true, q10_pred, 0.1)
    loss_q50 = pinball_loss(y_true, q50_pred, 0.5)
    loss_q90 = pinball_loss(y_true, q90_pred, 0.9)
    avg_pinball = (loss_q10 + loss_q50 + loss_q90) / 3.0
    
    # C. Interval Metrics
    coverage = np.mean(((y_true >= q10_pred) & (y_true <= q90_pred))) * 100.0
    width = np.mean(q90_pred - q10_pred)
    
    # 輸出報告
    print("\n" + "="*50)
    print(f"📊 FINAL ACADEMIC BENCHMARK (Test Set Only)")
    print("="*50)
    print(f"🎯 Point Forecast Accuracy (P50)")
    print(f"   MAE:              {mae:.2f} W/m²")
    print(f"   RMSE:             {rmse:.2f} W/m²")
    print("-" * 50)
    print(f"🎲 Probabilistic Score")
    print(f"   CRPS Proxy (Avg Pinball): {avg_pinball:.2f} (Lower is Better)")
    print(f"   Pinball Loss q10/q50/q90: {loss_q10:.2f} / {loss_q50:.2f} / {loss_q90:.2f}")
    print("-" * 50)
    print(f"🛡️ Interval Quality (Reliability)")
    print(f"   Coverage (PICP):  {coverage:.2f}%")
    print(f"   Avg Width (MPIW): {width:.2f} W/m²")
    print("="*50)

if __name__ == "__main__":
    file_path = "data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon_V2.parquet"
    evaluate_all_metrics(file_path)