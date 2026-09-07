# Main Forecasting Model Comparison Summary

## XGBoost 主方法結果
- Run path: `new_pipeline\data\output\experiments\main_xgboost_h1_bh\run_20260601_154609`
- Script path: `new_pipeline/scripts/experiments/exp_main_xgboost_h1_bh.py`
- Input data path: `new_pipeline/data/interim/ghi_hourly_clean.parquet`
- Target definition: `ghi_cwa_wm2[t+1h]`
- Train period: `2021-03-22 06:00:00+08:00` to `2024-06-17 15:00:00+08:00`; n=13464
- Validation period: `2024-06-17 16:00:00+08:00` to `2024-12-01 08:00:00+08:00`; n=1923
- Test period: `2024-12-01 09:00:00+08:00` to `2025-10-31 15:00:00+08:00`; n=3846
- Leakage audit: `new_pipeline\data\output\experiments\main_xgboost_h1_bh\run_20260601_154609\LEAKAGE_AUDIT.md`; status=`BH_SAFE_OPERATIONAL_BOUNDARY`
- Uses NWP forecast features: `True`

### XGBoost Metrics
| model_name      |    RMSE |     MAE |     MSE |   NRMSE_day |      MBE |       R2 |   MAPE_daytime_safe |    n |
|:----------------|--------:|--------:|--------:|------------:|---------:|---------:|--------------------:|-----:|
| Main XGBoost H1 | 95.6715 | 63.3208 | 9153.04 |     28.9203 | -8.48504 | 0.892627 |              26.114 | 3846 |

### XGBoost Top 10 Feature Importance
| feature                   |   importance |
|:--------------------------|-------------:|
| ghi_now                   |    0.316821  |
| k_t_now                   |    0.113937  |
| nwp_dswrf_h1              |    0.105848  |
| nwp_csi_nwp_h1            |    0.0864613 |
| sin_hour                  |    0.0740682 |
| clear_sky_ghi_target      |    0.0555117 |
| solar_zenith_angle_target |    0.041571  |
| hour                      |    0.033084  |
| nwp_forecast_hour_h1      |    0.0296596 |
| cloud_cover_now           |    0.0117519 |

## 精簡比較表
| Model Type   | Model                               | Literature basis                                 |    RMSE |     MAE |       MSE |   NRMSE_day |     R2 |    n | Remark                                                        |
|:-------------|:------------------------------------|:-------------------------------------------------|--------:|--------:|----------:|------------:|-------:|-----:|:--------------------------------------------------------------|
| XGBoost      | Main XGBoost H1                     | BH main method / gradient boosting               | 95.6715 | 63.3208 | 9153.0409 |     28.9203 | 0.8926 | 3846 | BH-safe H1 run; chronological split; NWP h1 features used     |
| CatBoost     | Weather-Classified CatBoost (WC-CB) | Ahmed et al. (2024)-style WC-CB                  | 96.6837 | 63.2140 | 9347.7468 |     29.2262 | 0.8903 | 3846 | BH-safe literature-style baseline                             |
| CNN-LSTM     | CNN-LSTM hybrid                     | Zang et al. (2020)-style pseudo-spatial CNN-LSTM | 98.0615 | 68.3478 | 9616.0673 |     29.6394 | 0.8872 | 3844 | BH-safe deterministic deep-learning baseline; PyTorch backend |

## Extended Table
| Model Type   | Model                               | Literature basis                                        |     RMSE |      MAE |        MSE |   NRMSE_day |     R2 |    n | Remark                                                        |
|:-------------|:------------------------------------|:--------------------------------------------------------|---------:|---------:|-----------:|------------:|-------:|-----:|:--------------------------------------------------------------|
| XGBoost      | Main XGBoost H1                     | BH main method / gradient boosting                      |  95.6715 |  63.3208 |  9153.0409 |     28.9203 | 0.8926 | 3846 | BH-safe H1 run; chronological split; NWP h1 features used     |
| CatBoost     | Weather-Classified CatBoost (WC-CB) | Ahmed et al. (2024)-style WC-CB                         |  96.6837 |  63.2140 |  9347.7468 |     29.2262 | 0.8903 | 3846 | BH-safe literature-style baseline                             |
| CNN-LSTM     | CNN-LSTM hybrid                     | Zang et al. (2020)-style pseudo-spatial CNN-LSTM        |  98.0615 |  68.3478 |  9616.0673 |     29.6394 | 0.8872 | 3844 | BH-safe deterministic deep-learning baseline; PyTorch backend |
| CatBoost     | Conventional CatBoost               | Ahmed et al. (2024) baseline without weather clustering |  97.3687 |  64.3610 |  9480.6555 |     29.4333 | 0.8888 | 3846 | Global CatBoost baseline                                      |
| Ablation     | CNN-only                            | Zang-style ablation                                     |  97.9241 |  68.7565 |  9589.1296 |     29.5978 | 0.8875 | 3844 | Extended reference                                            |
| Ablation     | LSTM-only                           | Zang-style ablation                                     | 100.4000 |  69.3501 | 10080.1693 |     30.3462 | 0.8818 | 3844 | Extended reference                                            |
| Naive        | Persistence                         | Persistence baseline                                    | 151.8571 | 113.7559 | 23060.5936 |     45.8993 | 0.7295 | 3844 | Extended reference                                            |

## Aligned Test-Set Comparison
- Alignment key: common target timestamp. Ahmed timestamp was saved as UTC-naive issue timestamp, so aligned target timestamp is reconstructed as `timestamp(UTC)+8h+1h`.
- Common rows: 3844
- Max y_true discrepancy after alignment: 0.00006667 W/m2
| Model                               |    RMSE |     MAE |     R2 |   n_aligned |   max_ytrue_diff_max |
|:------------------------------------|--------:|--------:|-------:|------------:|---------------------:|
| Main XGBoost H1                     | 95.6772 | 63.3177 | 0.8926 |        3844 |               0.0001 |
| Weather-Classified CatBoost (WC-CB) | 96.6962 | 63.2209 | 0.8903 |        3844 |               0.0001 |
| CNN-LSTM hybrid                     | 98.0615 | 68.3478 | 0.8872 |        3844 |               0.0001 |

## 論文建議寫法
在相同 BH-safe H1 operational boundary 與 chronological split 下，Main XGBoost H1 的 RMSE 最低（95.67 W/m2），略優於 Ahmed-style WC-CB（96.68 W/m2）與 Zang-style CNN-LSTM hybrid（98.06 W/m2）。此結果支持 XGBoost 作為 deterministic point-forecasting main model；但差距相對有限，論文中應避免宣稱 XGBoost 在一般意義上優於所有 deep-learning 或 weather-classified methods。Ahmed-style WC-CB 可作為主文精簡比較或 appendix robustness；Zang-style CNN-LSTM 建議作為 appendix deep-learning reference baseline。

## Files
- `main_model_comparison.csv`
- `extended_model_comparison.csv`
- `aligned_main_model_comparison.csv`