import pandas as pd
import os

# 檔案路徑 (請確認您的檔案位置)
INPUT_CSV = "NTUST_PV_Forecasting_Master_Dataset.csv"
OUTPUT_CSV = "Optimized_Forecast_Daylight80_OneYear.csv"

def generate_csv():
    print(f"🚀 Reading data from: {INPUT_CSV}")
    
    # 智慧路徑檢查
    if not os.path.exists(INPUT_CSV):
        # 嘗試備用路徑
        ALT_PATH = "Export_Package_for_PhD/NTUST_PV_Forecasting_Master_Dataset.csv"
        if os.path.exists(ALT_PATH):
            INPUT_CSV_PATH = ALT_PATH
        else:
            print(f"❌ Error: Source CSV not found.")
            return
    else:
        INPUT_CSV_PATH = INPUT_CSV

    df = pd.read_csv(INPUT_CSV_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp').sort_index()

    # 1. 鎖定一年期時間範圍
    start_date = "2024-10-31"
    end_date = "2025-10-31"
    print(f"   Filtering range: {start_date} to {end_date}...")
    
    df_filtered = df.loc[start_date:end_date].copy()
    
    if len(df_filtered) == 0:
        print("❌ Error: No data found in this range.")
        return

    # 2. 應用 0.955 最佳化係數 (針對日間覆蓋率 80%)
    print("   Applying optimization (Scale = 0.955)...")
    scale_factor = 0.955
    
    p50 = df_filtered['pred_q50_cal']
    p10_orig = df_filtered['pred_q10_cal']
    p90_orig = df_filtered['pred_q90_cal']

    # 計算新的寬度 (比 0.68 寬，但比原始窄)
    lower_diff = p50 - p10_orig
    upper_diff = p90_orig - p50
    p10_opt = p50 - (lower_diff * scale_factor)
    p90_opt = p50 + (upper_diff * scale_factor)

    # 3. 整理輸出欄位
    df_export = pd.DataFrame({
        'Actual_GHI_wm2': df_filtered['ghi_cwa_wm2'],
        'Forecast_P50_wm2': p50,
        'Forecast_P10_Optimized_wm2': p10_opt,
        'Forecast_P90_Optimized_wm2': p90_opt
    })

    # 四捨五入至小數點後 2 位 (方便 Excel 閱讀)
    df_export = df_export.round(2)

    # 4. 存檔
    print(f"   Saving to {OUTPUT_CSV}...")
    try:
        df_export.to_csv(OUTPUT_CSV)
        print(f"✅ Success! CSV file generated: {os.path.abspath(OUTPUT_CSV)}")
    except PermissionError:
        print(f"❌ Error: Permission denied. Please close '{OUTPUT_CSV}' first.")

if __name__ == "__main__":
    generate_csv()