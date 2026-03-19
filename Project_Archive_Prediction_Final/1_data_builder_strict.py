import pandas as pd
import numpy as np
import yaml
import warnings
import os
from pathlib import Path
from tqdm import tqdm
from pvlib import solarposition, location

warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ 設定區域 (您的專屬路徑)
# ==========================================
# 使用 r"..." 原始字串格式，確保 Windows 路徑讀取正確
CWA_DIR = r"data/raw/CWA_hourly"
NWP_DIR = r"data/raw/nwp_grib_extracted"
OUTPUT_DIR = "data/interim"

# 站點資訊 (台北)
SITE_LAT = 25.0377
SITE_LON = 121.5149
SITE_ELEV = 10.0
# ==========================================

def load_cwa(data_dir):
    print(f"📖 Loading CWA Data from: {data_dir}")
    path_obj = Path(data_dir)
    if not path_obj.exists():
        raise FileNotFoundError(f"❌ CWA folder not found: {data_dir}")
        
    all_files = sorted(path_obj.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"❌ No CSV files found in {data_dir}")

    dfs = []
    for f in tqdm(all_files, desc="Reading CWA"):
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns] # 去除空白
            
            # 時間解析 (支援兩種常見格式)
            if 'YYYYMM' in df.columns: 
                df['Year'] = df['YYYYMM'].astype(str).str.slice(0, 4)
                df['Month'] = df['YYYYMM'].astype(str).str.slice(4, 6)
                df['ts'] = pd.to_datetime(df['Year'] + '-' + df['Month'] + '-' + 
                                        df['DD'].astype(str).str.zfill(2) + ' ' + 
                                        df['HH'].astype(str).str.zfill(2) + ':00')
            elif '年' in df.columns:
                df['ts'] = pd.to_datetime(df['年'].astype(str) + '-' + 
                                        df['月'].astype(str).str.zfill(2) + '-' + 
                                        df['日'].astype(str).str.zfill(2) + ' ' + 
                                        df['時'].astype(str).str.zfill(2) + ':00')
            
            # 欄位改名映射
            rename_map = {
                'Global_Solar_Radiation_MJm2': 'ghi_cwa_mj',
                'Precp_mm': 'Precipitation_mm',
                'Temperature_C': 'Temperature_C',
                'Sunshine_Duration_hour': 'Sunshine_Duration_hour',
                'Global_Solar_Radiation_Accumulation_MJm2': 'Global_Solar_Radiation_MJm2',
                'Visb_km': 'Visibility_km',
                'Cloud_Amount_10': 'Total_Cloud_Amount_tenths'
            }
            df = df.rename(columns=rename_map)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f.name}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True).sort_values('ts').reset_index(drop=True)
    
    # 單位轉換
    if 'Global_Solar_Radiation_MJm2' in full_df.columns:
        full_df['ghi_cwa_wm2'] = full_df['Global_Solar_Radiation_MJm2'] * 1e6 / 3600
        full_df['ghi_cwa_wm2'] = full_df['ghi_cwa_wm2'].clip(lower=0)
    elif 'ghi_cwa_mj' in full_df.columns:
        full_df['ghi_cwa_wm2'] = full_df['ghi_cwa_mj'] * 1e6 / 3600
        full_df['ghi_cwa_wm2'] = full_df['ghi_cwa_wm2'].clip(lower=0)
        
    # 去除重複時間點
    full_df = full_df.drop_duplicates(subset=['ts'], keep='last')
    return full_df.set_index('ts')

def load_nwp_strict(nwp_dir):
    print(f"📖 Loading NWP Data from: {nwp_dir} (Strict Day-Ahead Filter)...")
    path_obj = Path(nwp_dir)
    if not path_obj.exists():
        raise FileNotFoundError(f"❌ NWP folder not found: {nwp_dir}")

    all_files = sorted(path_obj.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"❌ No CSV files found in {nwp_dir}")
    
    merged_nwp = None
    
    for f in tqdm(all_files, desc="Merging NWP Vars"):
        # 檔名範例: gfs_t2m_raw_database.csv -> t2m
        var_name = f.stem.replace("gfs_", "").replace("_raw_database", "")
        df = pd.read_csv(f)
        
        df['initial_time'] = pd.to_datetime(df['initial_time'])
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        df = df.rename(columns={'value': var_name})
        
        # 只取需要的欄位
        cols = ['initial_time', 'valid_time', 'forecast_hour', var_name]
        df = df[cols]
        
        if merged_nwp is None:
            merged_nwp = df
        else:
            merged_nwp = pd.merge(merged_nwp, df, on=['initial_time', 'valid_time', 'forecast_hour'], how='outer')
            
    print(f"   Raw NWP rows: {len(merged_nwp)}")
    
    # 核心邏輯：Strict Vintage Selection
    # Deadline = 前一天 (D-1) 的 12:00 UTC (因為這是日前決策的時間點)
    merged_nwp['deadline'] = (merged_nwp['valid_time'].dt.floor('D') - pd.Timedelta(days=1)) + pd.Timedelta(hours=12)
    
    # 1. 剔除所有 "未來" 發布的預報 (initial_time > deadline)
    valid_forecasts = merged_nwp[merged_nwp['initial_time'] <= merged_nwp['deadline']].copy()
    
    # 2. 對於每個 valid_time，選 "最新" 的一筆 (最接近 deadline)
    valid_forecasts = valid_forecasts.sort_values(by=['valid_time', 'initial_time'], ascending=[True, False])
    final_nwp = valid_forecasts.drop_duplicates(subset=['valid_time'], keep='first')
    
    print(f"   Filtered NWP rows (Day-Ahead): {len(final_nwp)}")

    # =========================================================================
    # 🚑 [FIX START] 關鍵修正：強制升頻至 1 小時並線性插值
    # =========================================================================
    print("   📈 Upsampling NWP from 3H to 1H resolution...")
    
    # 1. 設定 Index 並排序
    final_nwp = final_nwp.set_index('valid_time').sort_index()
    
    # 2. 建立完整的每小時索引 (填補中間消失的小時)
    if not final_nwp.empty:
        full_idx = pd.date_range(start=final_nwp.index.min(), end=final_nwp.index.max(), freq='1H')
        final_nwp = final_nwp.reindex(full_idx)
        
        # 3. 線性插值 (只針對數值欄位)
        # numeric_only=True 確保只插值氣象變數，避免對 deadline 等時間欄位報錯
        # limit_direction='both' 確保頭尾如果剛好缺值也能補上
        final_nwp = final_nwp.interpolate(method='linear', limit_direction='both')
        
        # 4. 對於非數值欄位 (如 initial_time)，使用 ffill 補齊
        # (雖然我們主要只用氣象數值，但補齊比較保險)
        final_nwp['initial_time'] = final_nwp['initial_time'].ffill().bfill()
    else:
        print("   ⚠️ Warning: NWP dataframe is empty after filtering!")
    # =========================================================================
    # 🚑 [FIX END]
    # =========================================================================
    
    return final_nwp

def build_strict_feature_store():
    # 1. 載入資料
    df_cwa = load_cwa(CWA_DIR)
    df_nwp = load_nwp_strict(NWP_DIR)
    
    # 2. 合併
    print("🔄 Merging CWA and NWP...")
    if df_cwa.index.tz is not None: df_cwa.index = df_cwa.index.tz_localize(None)
    if df_nwp.index.tz is not None: df_nwp.index = df_nwp.index.tz_localize(None)
    
    # Inner join: 因為 NWP 已經插值成 1 小時了，現在做 Inner Join 不會掉資料
    merged = df_cwa.join(df_nwp, how='inner')
    
    # 3. 補上物理特徵
    print("☀️ Adding Solar Geometry...")
    site = location.Location(SITE_LAT, SITE_LON, tz='Asia/Taipei')
    tz_idx = merged.index.tz_localize('Asia/Taipei')
    solpos = site.get_solarposition(tz_idx, method='ephemeris')
    
    merged['solar_zenith'] = solpos['zenith'].values
    merged['solar_elevation'] = solpos['elevation'].values
    merged['solar_azimuth'] = solpos['azimuth'].values
    
    cs = site.get_clearsky(tz_idx, model='ineichen')
    merged['dni_clear'] = cs['dni'].values
    merged['dhi_clear'] = cs['dhi'].values
    merged['ghi_clear'] = cs['ghi'].values
    
    # 4. 存檔
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/merged_feature_store_v4.parquet"
    print(f"💾 Saving to {output_path}...")
    merged.to_parquet(output_path)
    print(f"✅ Feature Store Rebuilt! Shape: {merged.shape}")
    
    # 檢查
    nwp_cols = [c for c in merged.columns if c in ['t2m', 'dswrf', 'lcc', 'mcc', 'hcc', 'tcc']]
    print(f"   NWP Features included: {nwp_cols}")
    
    # 簡單驗證頻率
    freq_check = pd.infer_freq(merged.index)
    print(f"   Inferred Frequency: {freq_check} (Should be 'H' or '1H')")

if __name__ == "__main__":
    build_strict_feature_store()