## Point forecast skill metrics on the test period (2024-11-01 to 2025-10-31, daytime slots).

| Model                 | Task           |   RMSE (W/m²) |   MAE (W/m²) |     R² | SS vs Pers.   | SS vs Clim.   |     n |
|:----------------------|:---------------|--------------:|-------------:|-------:|:--------------|:--------------|------:|
| ID v4 XGBoost + AgACI | Intraday H=1~6 |        116.45 |        71.93 | 0.8362 | 0.565         | 0.437         | 25686 |
| DA v2 XGBoost         | Day-Ahead      |        140.88 |        94.23 | 0.7612 | —             | —             |  4399 |
| CSI Persistence       | Intraday H=1~6 |        267.9  |       179.15 | 0.1332 | 0.000         | —             | 25686 |
| Climatology           | Intraday H=1~6 |        206.89 |       156.75 | 0.483  | —             | 0.000         | 25686 |

*Note: ID model evaluated on H=1\textasciitilde{}6 pooled (n=25{,}686 daytime slots). DA model evaluated on daytime slots with GHI\_clear\_sky\,$>$\,0 (n=4{,}399). Skill score (SS) = $1 - \text{RMSE}_\text{model} / \text{RMSE}_\text{ref}$. Source: metrics\_overall.csv; da\_baseline\_summary.json (da\_test entry).*