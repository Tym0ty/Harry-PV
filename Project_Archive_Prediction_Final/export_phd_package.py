import pandas as pd
import numpy as np
from pathlib import Path
import os

# ==========================================
# ⚙️ 設定區域
# ==========================================
SOURCE_FILE = "Export_Package_for_PhD/Strict_Model_Data/Model_X_Y_Only.csv"
OUTPUT_DIR = "Export_Package_for_PhD/Strict_Model_Data"
OUTPUT_FILENAME = "Model_X_Y_Colored.xlsx" # 輸出成 Excel

def export_colored_excel():
    print("🚀 Starting Colored Excel Export...")
    
    # 1. 讀取已經篩選好的 CSV (如果您還沒跑上一歲，請先跑 export_strict_model_data.py)
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Source file not found: {SOURCE_FILE}")
        print("   👉 Please run 'export_strict_model_data.py' first!")
        return
        
    df = pd.read_csv(SOURCE_FILE, index_col="Timestamp", parse_dates=True)
    print(f"   📖 Loaded Data: {df.shape}")

    # =========================================================
    # 2. 定義欄位群組 (用於著色)
    # =========================================================
    
    # [Y] Outputs (紅色系)
    y_cols = [
        'ghi_cwa_wm2', 'pred_q10_cal', 'pred_q50_cal', 'pred_q90_cal'
    ]
    
    # [X] Inputs (藍色系) - 包含 NWP, Physics, Time, Obs
    # 只要不是 Y 的，全部歸類為 X
    x_cols = [c for c in df.columns if c not in y_cols]

    # =========================================================
    # 3. 建立 Excel 並上色
    # =========================================================
    output_path = f"{OUTPUT_DIR}/{OUTPUT_FILENAME}"
    print(f"   🎨 Creating Excel with colored headers...")

    # 使用 xlsxwriter 引擎來支援格式化
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # 寫入資料
        df.to_excel(writer, sheet_name='Model_Data', float_format='%.4f')
        
        # 取得 workbook 和 worksheet 物件
        workbook  = writer.book
        worksheet = writer.sheets['Model_Data']
        
        # 定義格式 (Format)
        # 🔴 Y 欄位格式: 橘紅底色 + 粗體
        fmt_y = workbook.add_format({
            'bold': True,
            'font_color': '#9C0006', # 深紅字
            'bg_color':   '#FFC7CE', # 淺紅底
            'border': 1,
            'align': 'center'
        })
        
        # 🔵 X 欄位格式: 藍色底色 + 粗體
        fmt_x = workbook.add_format({
            'bold': True,
            'font_color': '#006100', # 深綠字 (或深藍)
            'bg_color':   '#C6EFCE', # 淺綠底 (或是淺藍 #BDD7EE)
            'border': 1,
            'align': 'center'
        })

        # ⚪ Timestamp 格式 (第一欄)
        fmt_idx = workbook.add_format({'bold': True, 'border': 1})

        # --- 開始對 Header 進行著色 ---
        # enumerate 從 0 開始，但 Excel 第一欄是 Timestamp (index)
        # 所以 df 的第 0 欄資料，實際上是 Excel 的第 1 欄 (B欄)
        
        # 先寫入 Timestamp 標頭 (A1)
        worksheet.write(0, 0, "Timestamp", fmt_idx)
        
        for i, col_name in enumerate(df.columns):
            # 判斷是 X 還是 Y
            if col_name in y_cols:
                current_fmt = fmt_y
            else:
                current_fmt = fmt_x
            
            # 寫入標頭 (列=0, 欄=i+1 因為要跳過 index)
            worksheet.write(0, i + 1, col_name, current_fmt)

    print(f"   💾 Saved Colored Excel: {output_path}")
    print("\n✅ Done! The headers are now colored (Red=Y, Green/Blue=X).")

if __name__ == "__main__":
    export_colored_excel()