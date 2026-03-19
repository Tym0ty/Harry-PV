import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def apply_v15(df_path, output_path): # 統一函數名稱
    print(f"READING: {df_path}")
    df = pd.read_parquet(df_path)

    # ==========================================
    # V15: 80% 覆蓋率微調 (Global Scale 調整)
    # ==========================================
    # [關鍵修正] 將壓縮因子 0.95 提升至 1.00，給予區間所需的微小彈性
    GLOBAL_SCALE = 1.35
    # 維持 V14 的 Time-Adaptive 結構
    HOURLY_SCALE = {
        6: 1.20, 7: 1.15, 8: 1.10, 9: 1.05, 10: 1.00,
        11: 0.95, 12: 0.95, 13: 0.95, 14: 1.00,
        15: 1.05, 16: 1.15, 17: 1.25, 18: 1.35
    }
    DEFAULT_SCALE = 1.05
    decay_map = {12: 1.02, 13: 1.05, 14: 1.10, 15: 1.20, 16: 1.30, 17: 1.40, 18: 1.50}

    def adjust_quantiles_v15(row):
        h = row['hour']
        q10, q50, q90 = row['pred_q10_cal'], row['pred_q50_cal'], row['pred_q90_cal']
        cs = row['cs_ghi']
        
        width = q90 - q10
        half_width = width / 2
        
        current_scale = HOURLY_SCALE.get(h, DEFAULT_SCALE)
        time_scale = decay_map.get(h, 1.0)
        kt = q50 / (cs + 1.0)
        
        scale_lower = 1.0
        scale_upper = 1.0
        
        if kt > 0.7:
            scale_lower = 0.95
            scale_upper = 1.18 
        elif kt < 0.3:
            scale_lower = 0.9
            scale_upper = 1.4
        else:
            scale_lower = 1.0
            scale_upper = 1.15
            
        final_lower_width = half_width * time_scale * current_scale * scale_lower * GLOBAL_SCALE
        final_upper_width = half_width * time_scale * current_scale * scale_upper * GLOBAL_SCALE
        
        min_w = 10.0 if h >= 15 else 5.0
        if final_lower_width < min_w / 2: final_lower_width = min_w / 2
        if final_upper_width < min_w / 2: final_upper_width = min_w / 2

        bias = width * 0.03 if h >= 15 else 0
        new_q50 = q50 + bias
        
        return new_q50 - final_lower_width, new_q50, new_q50 + final_upper_width

    print(f"APPLYING: V15 80percent Fix (Global Scale {GLOBAL_SCALE})...")
    res = df.apply(adjust_quantiles_v15, axis=1, result_type='expand')
    
    df['pred_q10_fix'] = res[0].clip(lower=0)
    df['pred_q50_fix'] = res[1].clip(lower=0)
    df['pred_q90_fix'] = res[2].clip(lower=0)
    
    # 物理封頂 (Physics Cap)
    if 'cs_ghi' in df.columns:
        df['pred_q90_fix'] = np.minimum(df['pred_q90_fix'], df['cs_ghi'] * 1.30)
        df['pred_q50_fix'] = np.minimum(df['pred_q50_fix'], df['cs_ghi'] * 1.10)
        df['pred_q10_fix'] = np.minimum(df['pred_q10_fix'], df['cs_ghi'] * 1.02)

    final_df = df.copy()
    final_df['pred_q10_cal'] = final_df['pred_q10_fix']
    final_df['pred_q50_cal'] = final_df['pred_q50_fix']
    final_df['pred_q90_cal'] = final_df['pred_q90_fix']
    final_df.drop(columns=['pred_q10_fix', 'pred_q50_fix', 'pred_q90_fix'], inplace=True)

    print(f"SAVING: {output_path}")
    final_df.to_parquet(output_path)

if __name__ == "__main__":
    input_file = r"data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon.parquet"
    output_file = r"data/outputs/predictions_xgbq_v4_cal_shiftScale_fixAfternoon_V2.parquet"
    
    if Path(input_file).exists():
        apply_v15(input_file, output_file)
    else:
        print(f"Error: Input file not found: {input_file}")