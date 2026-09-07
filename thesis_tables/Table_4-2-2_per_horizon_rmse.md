## Per-horizon RMSE (W/m²) for intraday GHI forecast on test set (2024-11-01 to 2025-10-31).

| Lead Time      |   v4 XGBoost + AgACI |   CatBoost (bench.) |   LightGBM (bench.) |   Ridge Stack |
|:---------------|---------------------:|--------------------:|--------------------:|--------------:|
| H=1            |               109.41 |              110.58 |              112.09 |        107.4  |
| H=2            |               116.35 |              119.31 |              119.85 |        114.54 |
| H=3            |               118.6  |              123.74 |              124.62 |        117.88 |
| H=4            |               117.58 |              126.3  |              127.39 |        118.28 |
| H=5            |               117.01 |              127.4  |              130.48 |        118.45 |
| H=6            |               119.47 |              127.02 |              129.02 |        119.5  |
| Pooled (H=1~6) |               116.45 |              122.54 |              124.07 |        116.08 |

*Note: Benchmark CatBoost and LightGBM are per-H models without clear/cloudy grouping (71 features). Ridge Stack combines v4 XGBoost, CatBoost, and LightGBM via ridge meta-learner. Source: metrics\_by\_horizon.csv; metrics\_overall\_stack.csv.*