# H1/H24 Point Forecast Seasonal Days Audit

- SOURCE_FORECAST_FILE: `new_pipeline/data/output/experiments/id_h24_ghi_prob_q19_independent_agaci_v4/q19_agaci_calibrated_noncrossing_forecasts.parquet`
- Point forecast column: `q50_cal` from Q19 independent-model AgACI v4.
- Observed GHI column: `y_ghi_true`.
- The plotted trajectory is a complete 00:00-23:00 local-day trajectory.
- Missing rows in the v4 artifact are zero-filled for observed GHI and q50 point forecasts to show the full physical nighttime-safe trajectory.
- Metrics are computed on the zero-filled 24-hour trajectories.
- MODEL_RERUN: FALSE
- AGACI_RERUN: FALSE
- SCENARIO_RERUN: FALSE
- OPTIMIZATION_RERUN: FALSE

## Panel QA
| season   | representative_date   |   h1_rows |   h24_rows |   h1_available_rows_before_zero_fill |   h24_available_rows_before_zero_fill |   h1_zero_filled_rows |   h24_zero_filled_rows |   observed_unique_target_times |   h1_hour_min |   h1_hour_max |   h24_hour_min |   h24_hour_max |   h1_rmse |   h1_mae |   h24_rmse |   h24_mae |
|:---------|:----------------------|----------:|-----------:|-------------------------------------:|--------------------------------------:|----------------------:|-----------------------:|-------------------------------:|--------------:|--------------:|---------------:|---------------:|----------:|---------:|-----------:|----------:|
| Winter   | 2025-01-11            |        24 |         24 |                                   11 |                                    11 |                    13 |                     13 |                             24 |             0 |            23 |              0 |             23 |   15.5662 |  8.26939 |    13.9979 |   6.58055 |
| Spring   | 2025-05-23            |        24 |         24 |                                   13 |                                    13 |                    11 |                     11 |                             24 |             0 |            23 |              0 |             23 |   69.8561 | 37.0023  |    81.9928 |  48.2701  |
| Summer   | 2025-07-06            |        24 |         24 |                                   13 |                                    13 |                    11 |                     11 |                             24 |             0 |            23 |              0 |             23 |   49.2032 | 28.6703  |   156.088  |  78.5708  |
| Autumn   | 2025-09-14            |        24 |         24 |                                   12 |                                    12 |                    12 |                     12 |                             24 |             0 |            23 |              0 |             23 |  121.088  | 47.4797  |    83.9901 |  36.0096  |
