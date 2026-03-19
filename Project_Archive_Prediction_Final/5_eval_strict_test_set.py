import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_test_set(file_path):
    print(f"Loading: {file_path}")
    df = pd.read_parquet(file_path)

    # 1. 確保 Index 是時間戳 (ts)
    if 'ts' not in df.index.names and 'ts' in df.columns:
        df = df.set_index('ts')
    
    # 2. [關鍵修正] 鎖定測試集範圍 (2024-11-01 之後的數據)
    TEST_START_DATE = "2024-11-01"
    df = df[df.index >= TEST_START_DATE].copy()
    
    if df.empty:
        print("❌ Error: Test set is empty after filtering. Check index type or date.")
        return

    # 3. 準備數據 (只看白晝)
    target_col = 'ghi_cwa_wm2'
    if target_col not in df.columns: target_col = 'y_true' # Fallback

    mask = (df[target_col] > 10) | (df['pred_q50_cal'] > 10)
    df_eval = df[mask].copy()

    if len(df_eval) == 0:
        print("❌ Error: No daylight samples found in Test Set.")
        return

    # 4. 計算指標
    y_true = df_eval[target_col].values
    q50_pred = df_eval['pred_q50_cal'].values
    q10_pred = df_eval['pred_q10_cal'].values
    q90_pred = df_eval['pred_q90_cal'].values

    final_mae = np.mean(np.abs(y_true - q50_pred))
    final_rmse = np.sqrt(np.mean((y_true - q50_pred)**2))
    final_coverage = np.mean((y_true >= q10_pred) & (y_true <= q90_pred)) * 100.0
    final_width = np.mean(q90_pred - q10_pred)

    print("\n" + "="*40)
    print(f"📊 FINAL TEST SET PERFORMANCE")
    print("="*40)
    print(f"Samples evaluated: {len(df_eval)} (Strict Test Set)")
    print(f"🎯 MAE (Generalization): {final_mae:.2f} W/m²")
    print(f"RMSE:                   {final_rmse:.2f} W/m²")
    print(f"🛡️ Coverage (PICP):      {final_coverage:.2f}%")
    print(f"📏 Avg Width (MPIW):     {final_width:.2f} W/m²")
    print("="*40)

if __name__ == "__main__":
    file_path = "data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon_V2.parquet"
    evaluate_test_set(file_path)