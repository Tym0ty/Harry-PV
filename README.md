# Harry-PV: Day-Ahead Solar Irradiance Probabilistic Forecasting

Day-ahead GHI (Global Horizontal Irradiance) probabilistic forecasting pipeline for the NTUST site in Taipei, Taiwan. Uses CQR-XGBQ (Conformal Quantile Regression with XGBoost Quantile) to produce 19 calibrated quantile forecasts (P05–P95), then converts to PV power and generates reduced joint scenarios for stochastic optimization.

## Pipeline Overview

```
CWA Hourly Obs + GFS NWP → Feature Engineering → 19-Quantile Models → CQR Calibration → GHI→PV → Scenario Reduction
```

- **Gate-compliant**: All NWP data respects D-1 12:00 UTC deadline (no future data leakage)
- **Strict anti-leakage**: No contemporaneous CWA observations used as features — only lagged (≥24h) observations
- **Split-conformal CQR**: Calibrated prediction intervals with finite-sample correction

## Iteration Results (Test Set — Last 365 Days, Daytime)

| Version | Description | MAE (W/m²) | R² | Notes |
|---------|-------------|-------------|------|-------|
| V1 | Baseline XGBoost, default params | 83.93 | 0.8049 | Single model, lr=0.05, depth=8 |
| V2 | Tuned XGBoost + 5-seed ensemble | 83.01 | 0.8071 | 162-combo grid search |
| V3 | Tuned XGBoost + 10-seed ensemble | 83.01 | 0.8071 | 720-combo fine grid, minimal gain over V2 |
| V4 | LightGBM 10-seed ensemble | ~82.09 | ~0.81 | Comparable to XGBoost |
| V5 | CatBoost 10-seed ensemble | ~82.32 | ~0.81 | Comparable to XGBoost |
| **V6** | **3-Model Ensemble (XGB+LGB+CB)** | **83.00** | **0.8062** | Equal-weight average of 3 tree models |

**Best leak-free result: MAE ≈ 82–83 W/m², R² ≈ 0.81**

### Comparison with Harry's Original Pipeline

| Metric | Harry's Pipeline | Our Pipeline (V6) |
|--------|------------------|--------------------|
| MAE | 75.14 W/m² | 83.00 W/m² |
| R² | 0.8618 | 0.8062 |
| Data Leakage | Yes (contemporaneous obs) | No (strict gate compliance) |
| Postprocessing | Bucket shift+scale + afternoon fix | CQR calibration only |

Harry's lower MAE is explained by his model inadvertently using contemporaneous CWA weather observations (real-time temperature, cloud cover, etc.) as features — a form of data leakage that would not be available at forecast time in a real day-ahead operational setting.

## Project Structure

```
Harry-PV/
├── forecasting_pipeline.ipynb    # Main pipeline (V6 — 3-model ensemble)
├── notebooks/
│   ├── v1_baseline.ipynb         # Iteration 1: Baseline
│   ├── v2_tuned_xgb_5seed.ipynb  # Iteration 2: Tuned + 5-seed
│   ├── v3_xgb_10seed.ipynb       # Iteration 3: 10-seed fine-tuned
│   ├── v4_lightgbm.ipynb         # Iteration 4: LightGBM
│   ├── v5_catboost.ipynb         # Iteration 5: CatBoost
│   └── v6_3model_ensemble.ipynb  # Iteration 6: 3-model ensemble
├── pipeline_outputs/             # Generated artifacts (parquet, JSON)
├── Project_Archive_Prediction_Final/  # Harry's original code (reference)
└── README.md
```

## Data Sources

- **CWA Hourly**: Central Weather Administration hourly station observations (2021–2025)
- **GFS NWP**: NOAA GFS 0.25° forecasts — t2m, dswrf, cloud cover (lcc/mcc/hcc/tcc), wind, precipitation, humidity

## Requirements

```
python >= 3.10
xgboost, lightgbm, catboost
pvlib, scikit-learn, scipy
pandas, numpy, pyarrow
```

## Site

- **Location**: NTUST (National Taiwan University of Science and Technology)
- **Coordinates**: 25.0377°N, 121.5149°E
- **PV System**: 50 kW DC rated, DC/AC ratio 1.2, inverter efficiency 96%
