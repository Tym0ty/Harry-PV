import pandas as pd
import os

# 設定路徑
INPUT_CSV = "NTUST_PV_Forecasting_Master_Dataset.csv" # 請確認檔名
OUTPUT_PARQUET = "Optimized_Forecast_Daylight80_OneYear.parquet"

def generate_parquet():
    print(f"🚀 Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        # 嘗試備用路徑
        INPUT_CSV_ALT = "Export_Package_for_PhD/NTUST_PV_Forecasting_Master_Dataset.csv"
        if os.path.exists(INPUT_CSV_ALT):
            INPUT_CSV = INPUT_CSV_ALT
        else:
            print("❌ CSV not found.")
            return

    df = pd.read_csv(INPUT_CSV)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()

    # 1. 鎖定時間範圍 (2024/10/31 - 2025/10/31)
    df_filtered = df.loc["2024-10-31":"2025-10-31"].copy()
    
    # 2. 應用新的最佳化係數 (0.955)
    scale_factor = 0.955
    print(f"   Applying Scale Factor: {scale_factor} (Target Daylight 80%)...")
    
    p50 = df_filtered['pred_q50_cal']
    p10_orig = df_filtered['pred_q10_cal']
    p90_orig = df_filtered['pred_q90_cal']

    lower_diff = p50 - p10_orig
    upper_diff = p90_orig - p50
    p10_opt = p50 - (lower_diff * scale_factor)
    p90_opt = p50 + (upper_diff * scale_factor)

    # 3. 輸出 Parquet
    df_export = pd.DataFrame({
        'Actual_GHI_wm2': df_filtered['ghi_cwa_wm2'],
        'Forecast_P50_wm2': p50,
        'Forecast_P10_Optimized_wm2': p10_opt,
        'Forecast_P90_Optimized_wm2': p90_opt
    })
    
    df_export.to_parquet(OUTPUT_PARQUET, index=True)
    print(f"✅ Generated: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    generate_parquet()