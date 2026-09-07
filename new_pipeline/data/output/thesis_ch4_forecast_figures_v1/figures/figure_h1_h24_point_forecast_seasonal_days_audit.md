# H1/H24 Point Forecast Seasonal Days Audit

- SOURCE_FORECAST_FILE: `new_pipeline/data/output/experiments/id_h24_ghi_prob_q19_independent_agaci_v4/q19_agaci_calibrated_noncrossing_forecasts.parquet`
- Point forecast column: `q50_cal` from Q19 independent-model AgACI v4.
- Observed GHI column: `y_ghi_true`.
- Rows are the available H1 and H24 forecast target rows in the v4 artifact; no rows were synthesized or filled.
- MODEL_RERUN: FALSE
- AGACI_RERUN: FALSE
- SCENARIO_RERUN: FALSE
- OPTIMIZATION_RERUN: FALSE

## Panel QA
| season   | representative_date   |   h1_rows |   h24_rows |   observed_unique_target_times |   h1_hour_min |   h1_hour_max |   h24_hour_min |   h24_hour_max |   h1_rmse |   h1_mae |   h24_rmse |   h24_mae |
|:---------|:----------------------|----------:|-----------:|-------------------------------:|--------------:|--------------:|---------------:|---------------:|----------:|---------:|-----------:|----------:|
| Winter   | 2025-01-11            |        11 |         11 |                             11 |             7 |            17 |              7 |             17 |   22.9928 |  18.0423 |    20.6762 |   14.3576 |
| Spring   | 2025-05-23            |        13 |         13 |                             13 |             6 |            18 |              6 |             18 |   94.9157 |  68.312  |   111.406  |   89.1141 |
| Summer   | 2025-07-06            |        13 |         13 |                             13 |             6 |            18 |              6 |             18 |   66.854  |  52.9298 |   212.082  |  145.054  |
| Autumn   | 2025-09-14            |        12 |         12 |                             12 |             6 |            17 |              6 |             17 |  171.244  |  94.9595 |   118.78   |   72.0192 |
