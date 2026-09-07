# Without-NWP H01/H24 ID-H24 Ablation

This independent supplemental run trained only XGBoost Direct-24 H01 and H24 models with all `nwp_*` features removed.

No formal experiment output was modified.

## Comparison

| Feature setting      | Model                     | Lead   |   n_all |     RMSE |     MAE |       MBE |       R2 |
|:---------------------|:--------------------------|:-------|--------:|---------:|--------:|----------:|---------:|
| with NWP features    | W1_XGB_D24 with NWP       | H01    |    8753 |  62.8329 | 26.7629 |  0.222083 | 0.939169 |
| with NWP features    | W1_XGB_D24 with NWP       | H24    |    8753 | 101.293  | 48.0925 | -0.799093 | 0.841909 |
| without NWP features | XGB direct-24 without NWP | H01    |    8753 |  64.2971 | 27.5508 |  0.638777 | 0.936301 |
| without NWP features | XGB direct-24 without NWP | H24    |    8753 | 131.783  | 66.3979 |  2.28034  | 0.732411 |

## Files
- `without_nwp_h01_h24/xgb_direct24_without_nwp_h01_h24_daylight_predictions.parquet`
- `without_nwp_h01_h24/xgb_direct24_without_nwp_h01_h24_allhours_predictions.parquet`
- `without_nwp_h01_h24/xgb_direct24_without_nwp_h01_h24_allhours_metrics.csv`
- `figures/figure_4_3_with_without_nwp_all_hours_comparison.png`
- `figures/figure_4_3_with_without_nwp_all_hours_comparison.pdf`