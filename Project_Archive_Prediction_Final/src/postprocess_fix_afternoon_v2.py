import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 忽略 Pandas 的 SettingWithCopyWarning
warnings.filterwarnings('ignore')

def apply_physics_aware_correction(df_path, output_path):
    """
    針對下午 (14-17h) 進行物理感知的後處理修正 (Physics-aware Scaling)。
    重點解決：下午時段 Coverage 不足、區間過窄、以及低估高值風險的問題。
    """
    print(f"READING: {df_path}")
    df = pd.read_parquet(df_path)

    # 1. 確保有必要的物理特徵 (若無則需從 ts_local 計算，這邊假設已有或可透過 Hour 近似)
    # 為了更精準，我們將 Hour 映射到一個「衰減權重」，模擬太陽下山
    # 下午 14:00 -> 權重 1.0, 17:00 -> 權重 2.5 (越晚越不穩，區間要越大)
    decay_map = {14: 1.1, 15: 1.3, 16: 1.6, 17: 2.2} 
    
    # 2. 定義修正邏輯
    def adjust_quantiles(row):
        h = row['hour']
        q10 = row['pred_q10_cal']
        q50 = row['pred_q50_cal']
        q90 = row['pred_q90_cal']
        
        # -------------------------------------------------------
        # 策略 A: 只針對下午 (14, 15, 16, 17) 且是白天 (q50 > 0)
        # -------------------------------------------------------
        if h in [14, 15, 16, 17] and q50 > 0:
            
            # --- 步驟 1: 基礎寬度計算 ---
            width = q90 - q10
            
            # --- 步驟 2: 物理權重放大 (Elevation-proxy) ---
            # 取得放大倍率 (Scale Factor)
            scale_factor = decay_map.get(h, 1.0)
            
            # --- 步驟 3: 低輻射保護 (Low-GHI Protection) ---
            # 在傍晚(16-17點)，如果預測值很小(例如 50W)，區間常縮到 <5W，導致 Coverage 0%。
            # 強制給一個最小物理波動寬度 (Minimum Physical Volatility)
            min_volatility = 10.0 if h >= 16 else 5.0
            
            # 如果原本寬度太窄，強迫撐開
            if width < min_volatility:
                width = min_volatility
            
            # --- 步驟 4: 不對稱擴展 (Asymmetric Boosting) ---
            # 下午特徵：GHI 容易突然變高 (雲散開)，模型常低估。
            # 作法：q90 往上推多一點，q10 往下推少一點
            # 讓 q50 也稍微往上 Shift (修正 Bias)
            
            bias_shift = width * 0.1 * scale_factor # 中位數稍微上修
            new_q50 = q50 + bias_shift
            
            # 計算新的半寬度，並乘上放大倍率
            half_width = (width / 2) * scale_factor
            
            # 擴張邊界
            new_q90 = new_q50 + half_width * 1.2  # 上界放寬更多 (捕捉突然的大太陽)
            new_q10 = new_q50 - half_width * 0.9  # 下界放寬稍少 (下面有 0 的物理底限)
            
            return new_q10, new_q50, new_q90

        else:
            return q10, q50, q90

    # 3. 應用修正
    print("APPLYING: Physics-aware Asymmetric Scaling...")
    
    # 使用 apply 逐行處理 (雖然慢一點點，但邏輯最清晰，反正推論資料量不大)
    res = df.apply(adjust_quantiles, axis=1, result_type='expand')
    df['pred_q10_fix'] = res[0]
    df['pred_q50_fix'] = res[1]
    df['pred_q90_fix'] = res[2]

    # 4. 物理約束 (Post-Constraint)
    # 絕對不能 < 0
    df['pred_q10_fix'] = df['pred_q10_fix'].clip(lower=0)
    df['pred_q50_fix'] = df['pred_q50_fix'].clip(lower=0)
    df['pred_q90_fix'] = df['pred_q90_fix'].clip(lower=0)
    
    # 絕對不能超過 Clearsky (若你有 cs_ghi 欄位)
    if 'cs_ghi' in df.columns:
        # 寬容度給 1.1 倍，避免 Clearsky 預測本身有誤差時切到真值
        df['pred_q90_fix'] = np.minimum(df['pred_q90_fix'], df['cs_ghi'] * 1.15)
        df['pred_q50_fix'] = np.minimum(df['pred_q50_fix'], df['cs_ghi'] * 1.10)
        df['pred_q10_fix'] = np.minimum(df['pred_q10_fix'], df['cs_ghi'] * 1.05)

    # 5. 替換回原始欄位名稱 (以便直接丟進評估腳本)
    # 為了保留原始 cal 版本做對照，我們新建一個 parquet
    final_df = df.copy()
    final_df['pred_q10_cal'] = final_df['pred_q10_fix']
    final_df['pred_q50_cal'] = final_df['pred_q50_fix']
    final_df['pred_q90_cal'] = final_df['pred_q90_fix']
    
    # 清理暫存欄位
    final_df.drop(columns=['pred_q10_fix', 'pred_q50_fix', 'pred_q90_fix'], inplace=True)

    print(f"SAVING: {output_path}")
    final_df.to_parquet(output_path)
    print("DONE. Physics-aware fix applied.")

# --- 執行區 ---
if __name__ == "__main__":
    # 設定你的檔案路徑 (請確認路徑是否正確)
    input_file = r"data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon.parquet"
    output_file = r"data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon_V2.parquet"
    
    # 檢查檔案是否存在
    if Path(input_file).exists():
        apply_physics_aware_correction(input_file, output_file)
    else:
        print(f"Error: Input file not found: {input_file}")