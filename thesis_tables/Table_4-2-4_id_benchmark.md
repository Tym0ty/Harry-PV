## Intraday GHI forecast benchmark — point prediction (H=1\textasciitilde{}6 pooled, test set, n=25{,}686).

| Model                                  | Features   | Grouping   |   RMSE (W/m²) |   MAE (W/m²) |     R² | \(\Delta\)RMSE vs REF   | 95\% Bootstrap CI      | Source                      |
|:---------------------------------------|:-----------|:-----------|--------------:|-------------:|-------:|:------------------------|:-----------------------|:----------------------------|
| Ridge Stack                            | 74         | per-H      |        116.08 |        74.86 | 0.8373 | \(-0.37\)               | \([{-1.21}, {+0.51}]\) | metrics\_overall\_stack.csv |
| CatBoost Exp5 + AgACI                  | 74         | per-H/grp  |        116.44 |        71.9  | 0.8363 | \(-0.02\)               | \([{-0.74}, {+0.75}]\) | metrics\_exp5\_overall.csv  |
| v4 XGBoost + AgACI \textsuperscript{†} | 74         | per-H/grp  |        116.45 |        71.93 | 0.8362 | REF                     | REF                    | metrics\_overall.csv        |
| CatBoost Exp5 raw                      | 74         | per-H/grp  |        121.77 |        76.79 | 0.8209 | +5.32                   | —                      | metrics\_exp5\_overall.csv  |
| CatBoost (benchmark)                   | 71         | per-H      |        122.54 |        81.95 | 0.8187 | +6.08                   | —                      | metrics\_overall\_stack.csv |
| LightGBM (benchmark)                   | 71         | per-H      |        124.07 |        82.53 | 0.8141 | +7.62                   | \([{+5.6}, {+9.7}]\)   | metrics\_overall\_stack.csv |
| CSI Persistence                        | —          | —          |        267.9  |       179.15 | 0.1332 | +151.45                 | —                      | metrics\_overall.csv        |
| Climatology                            | —          | —          |        206.89 |       156.75 | 0.483  | +90.44                  | —                      | metrics\_overall.csv        |

*Note: \(\dagger\) Selected as production model. per-H/grp = separate model per lead time \(\times\) sky condition (clear/cloudy). Bootstrap 95\% CI from 1{,}000 block-bootstrap iterations (block size = 1 day). Hyperparameter budget: v4 XGBoost 25 Optuna trials per H/group; CatBoost Exp5 inherits Exp3 tuning; benchmark models 25 trials (no grouping). Comparison not fully controlled; see Appendix for fairness caveats. LSTM experiments (Exp-J, Exp-K) have no saved test results and are excluded.*