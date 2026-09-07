# Forecasting DA + ID Model Comparison Final Report

稽核日期：2026-05-30  
原則：只引用 repository 中可由 scripts、CSV、JSON、parquet、checkpoint、log 或正式實驗報告追溯的結果。`raw_19q` 的 DA RMSE 約 124.92 W/m2 已確認含 day-ahead context leakage，不列為正式 DA baseline。

## 1. Executive Summary

目前 production forecasting backbone 仍以 XGBoost-based probabilistic pipeline 為主。對 DA forecasting，正式 no-leakage reference 是 `da_v2 XGBoost` q50 RMSE 140.88 W/m2；另外 DA-Exp1/2 boosting point baselines 約 140.18-141.01 W/m2。新完成的 DA-LSTM、DA-CNN-LSTM 與 DA-TFT 都是 point forecast baseline，不是 probabilistic backbone。

DA deep learning 結論：`DA-LSTM` RMSE 143.73，`DA-CNN-LSTM` RMSE 145.27，均差於 DA boosting baseline。`DA-TFT` 是 true `pytorch-forecasting` TFT attempt，但 RMSE 377.66，應只作 caveated exploratory result，不可寫成「TFT 一般性失敗」。

ID 結論：已驗證 ID production baseline `v4 XGBoost + AgACI` RMSE 116.45、MAE 71.93；`CatBoost Exp5 + AgACI` 幾乎相同，RMSE 116.44、MAE 71.90。ID CNN-LSTM 與 Multi-Task LSTM Exp-K 約 124.9 RMSE，僅能作 point-forecast exploratory baseline。

整體而言，deep-learning baselines 沒有改變 thesis model-selection conclusion：XGBoost 仍是較可重現、較準確、且已具備 quantile / calibration / scheduling integration artifacts 的 production choice。

## 2. Day-Ahead Model Comparison Table

| Model | Type | Forecast type | Test period | Daylight filter | RMSE | MAE | R2 | MBE | Leakage status | Safe to cite? | Evidence path | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| DA-Exp1 XGBoost Optuna | XGBoost | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 140.18 | 96.37 | 0.7636 | N/A | DA-valid 55 features | Yes | `new_pipeline/data/output/experiments/da_boosting_benchmark/predictions_da_boosting.parquet` | Best verified DA point baseline |
| DA-Exp1 LightGBM Optuna | LightGBM | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 140.44 | 96.41 | 0.7627 | N/A | DA-valid 55 features | Yes | same as above | Comparable point baseline |
| DA-Exp2 CatBoost raw GHI | CatBoost | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 140.63 | 97.26 | 0.7620 | N/A | DA-valid 55 features | Yes | `predictions_da_catboost_v4like.parquet` | CatBoost Exp A |
| da_v2 XGBoost q50 | XGBoost | quantile median / probabilistic artifact | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 140.88 | 94.23 | 0.7612 | +0.97 | DA leakage-free | Yes | `new_pipeline/data/input/da_v2_quantiles.parquet` | Production DA probabilistic artifact reference |
| DA-Exp1 CatBoost Optuna | CatBoost | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 141.01 | 97.62 | 0.7607 | N/A | DA-valid 55 features | Yes | `predictions_da_boosting.parquet` | Comparable point baseline |
| DA-LSTM | LSTM | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 143.73 | 103.56 | 0.7514 | +9.50 | No leakage found | Yes, point only | `new_pipeline/data/output/experiments/da_dl_comparison/da_lstm_metrics.csv` | Not probabilistic |
| DA-CNN-LSTM | CNN-LSTM | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 145.27 | 104.23 | 0.7461 | +18.68 | No leakage found | Yes, point only | `da_cnn_lstm_metrics.csv` | Not probabilistic |
| DA-TFT | TFT | point | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 377.66 | 294.55 | -0.7162 | +3.82 | No direct leakage found; implementation caveated | Caveat | `da_tft_metrics.csv`, `da_tft_meta.json` | True TFT attempt, but poor and exploratory |
| `raw_19q` q50 | XGBoost archive | quantile median | 2024-11-01 to 2025-10-31 | `ghi_clear > 0` | 124.92 | 81.14 | N/A | N/A | **DA context leakage** | No | `raw_19q_quantiles.parquet`; leakage audit docs | Discarded result |

## 3. Intraday Model Comparison Table

| Model | Type | Forecast type | Horizon setting | Test period | RMSE | MAE | Other metrics | Leakage status | Safe to cite? | Evidence path | Notes |
|---|---|---|---|---|---:|---:|---|---|---|---|---|
| CatBoost Exp5 + AgACI | CatBoost | quantile / calibrated | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 116.44 | 71.90 | R2 0.8363, MBE +1.97 | ID-valid | Yes | `new_pipeline/data/output/experiments/p1_analysis/id_metrics_with_mbe.csv` | Comparable to v4, not selected as main backbone |
| v4 XGBoost + AgACI | XGBoost | quantile / calibrated | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 116.45 | 71.93 | R2 0.8362, MBE +0.83 | ID-valid; not DA | Yes | `id_metrics_with_mbe.csv`, `metrics_overall_boosting.csv` | Production ID baseline |
| Ridge stacking XGB+CatBoost+LightGBM | stacking | point / stacked q50 | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 116.08 | 74.86 | R2 0.8373, MBE -0.06 | Caveat: calibration/selection risk | Caveat | `metrics_overall_stack.csv` | Lower RMSE but not automatic production choice |
| LightGBM per-H benchmark | LightGBM | point | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 124.07 | 82.53 | R2 0.8141 | ID-valid | Yes | `metrics_overall_boosting.csv` | Weaker boosting baseline |
| plain LightGBM | LightGBM | point | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 124.26 | 82.55 | R2 0.8135 | ID-valid | Yes | `plain_boosting_family/metrics_plain_overall.csv` | Plain family benchmark |
| Multi-Task LSTM Exp-K | LSTM | point | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 124.86 | N/A | per-H RMSE saved | ID-valid but no row predictions/MAE | Caveat | `dl_comparison/lstm_expk_metrics.csv`, `lstm_expk_results.json` | Point only; blend results not thesis-safe |
| CNN-LSTM multi-task | CNN-LSTM | point | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 124.89 | 84.27 | per-H metrics saved | ID-valid; log provenance caveat | Caveat | `dl_comparison/cnn_lstm_metrics.csv`, `cnn_lstm_predictions.parquet` | Point only |
| Climatology | baseline | point | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 206.89 | 156.75 | R2 0.4830 | ID-valid | Yes | `metrics_overall.csv` | Simple reference |
| CSI persistence | baseline | point | H=1-6 pooled daytime | 2024-11-01 to 2025-10-31 | 267.90 | 179.15 | R2 0.1332 | ID-valid | Yes | `metrics_overall.csv` | Simple reference |

## 4. Model Selection Explanation

The final XGBoost-based pipeline remains defensible. For DA forecasting, the newly completed leakage-free deep-learning baselines are valid as point-forecast comparisons but do not outperform the verified DA boosting models: DA-LSTM and DA-CNN-LSTM are roughly 3-5 W/m2 RMSE worse than da_v2 / DA-Exp1 boosting, and the TFT attempt is exploratory with poor performance. For ID forecasting, the saved CNN-LSTM and LSTM Exp-K results are also weaker than v4 XGBoost + AgACI. More importantly, the production XGBoost pipeline has saved quantile outputs, calibration artifacts, leakage audits, and direct downstream PV-BESS scheduling integration, while the deep-learning baselines here produce point forecasts only.

## 5. Thesis Text Paragraph

For the forecasting model comparison, both gradient-boosting and deep-learning baselines were evaluated under chronological and leakage-controlled splits. In the day-ahead setting, the verified leakage-free boosting baselines achieved RMSE values around 140.18-141.01 W/m2, with the production `da_v2` XGBoost quantile artifact yielding a q50 RMSE of 140.88 W/m2. Newly implemented day-ahead LSTM and CNN-LSTM point-forecast baselines achieved RMSE values of 143.73 and 145.27 W/m2, respectively, while a true TFT implementation was only exploratory and performed substantially worse. In the intraday setting, the production v4 XGBoost + AgACI model achieved RMSE/MAE of 116.45/71.93 W/m2, outperforming the saved CNN-LSTM and multi-task LSTM point baselines. Therefore, the XGBoost-based probabilistic pipeline was retained because it provides the strongest verified accuracy, complete quantile and calibration artifacts, and the most reliable integration with the downstream PV-BESS scheduling workflow.

## 6. Professor Slide Summary

- DA 無洩漏比較：boosting 約 140.18-141.01 W/m2；`da_v2` q50 = 140.88 W/m2。
- 新增 DA-LSTM = 143.73、DA-CNN-LSTM = 145.27，皆為 point forecast，未超越 XGBoost/LightGBM/CatBoost。
- DA-TFT 為 true TFT attempt，但目前設定表現很差，只能列 caveated exploratory result。
- ID 無洩漏比較：`v4 XGBoost + AgACI` = 116.45 / 71.93；`CatBoost Exp5 + AgACI` 幾乎相同；ID CNN-LSTM/LSTM 約 124.9。
- 結論：XGBoost pipeline 仍是 thesis production backbone，因為 accuracy、reproducibility、probabilistic calibration 與 PV-BESS scheduling integration 最完整。

## Commands Used

```powershell
rg -n -i "exp_da_lstm|DA LSTM|Attention MTL|CNN-LSTM|CNNLSTM|LSTM|TFT|Temporal Fusion Transformer|Transformer|GRU|BiLSTM|DeepAR|pytorch.forecasting|darts|gluonts" .
python new_pipeline/scripts/experiments/exp_da_dl_baselines.py --models lstm cnn_lstm tft --epochs 80 --patience 12 --tft-epochs 20 --tft-patience 5
```

Additional Python one-liners were used to verify package availability, inspect parquet/CSV schemas, and recompute RMSE/MAE/R2/MBE from saved prediction files.

## Files Generated

- `new_pipeline/scripts/experiments/exp_da_dl_baselines.py`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_lstm_metrics.csv`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_lstm_predictions.parquet`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_lstm_meta.json`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_lstm_scaler.pkl`
- `new_pipeline/models_da_dl/da_lstm_seed42.pt`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_cnn_lstm_metrics.csv`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_cnn_lstm_predictions.parquet`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_cnn_lstm_meta.json`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_cnn_lstm_scaler.pkl`
- `new_pipeline/models_da_dl/da_cnn_lstm_seed42.pt`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_tft_metrics.csv`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_tft_predictions.parquet`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_tft_meta.json`
- `new_pipeline/models_da_dl/da_tft_seed42-v1.ckpt`
- `new_pipeline/data/output/experiments/da_dl_comparison/da_dl_train.log`
- `DA_DEEP_LEARNING_MODEL_COMPARISON_TABLE.md`
- `FORECASTING_DA_ID_MODEL_COMPARISON_FINAL.md`
