## M8 no-regret arbitration: filter ablation at the EOD horizon. Values from FY2 reference run (test period 2024-11-01 to 2025-10-31).

| Configuration                      | Enabled Filters                 | SOC Band   | Accept\%   |   F1 Rej. |   F2 Rej. |   F3 Rej. |   F4 Rej. |   Full Total (M NTD) |   \(\Delta\) vs M2 (kNTD) |
|:-----------------------------------|:--------------------------------|:-----------|:-----------|----------:|----------:|----------:|----------:|---------------------:|--------------------------:|
| Baseline (M2 replay)               | None (alpha=0, exact M2 replay) | 10%        | 100.0\%    |         0 |         0 |         0 |         0 |             100.486  |                       0   |
| Core-A: F1+F2                      | F1+F2                           | 10%        | 99.2\%     |        21 |        48 |         0 |         0 |             100.049  |                    -436   |
| Core-B: F1+F2+F4                   | F1+F2+F4                        | 10%        | 97.8\%     |        21 |        42 |         0 |       134 |             100.059  |                    -426.6 |
| FY1: All filters (10\% band)       | F1+F2+F3+F4                     | 10%        | 71.5\%     |        40 |       435 |      1981 |        43 |              99.8872 |                    -598.3 |
| FY2: All filters (15\% band) [REF] | F1+F2+F3+F4                     | 15%        | 74.3\%     |        50 |       100 |      2058 |        43 |              99.43   |                   -1055.5 |

*Note: REF = M8-FY2-EOD (Full Total = 99.4300 M NTD). F1 = peak cap (D\textsubscript{mth} \(\leq\) D\textsubscript{ref}); F2 = SOC corridor; F3 = cost consistency; F4 = charge headroom. \(\Delta\) vs M2 = saving relative to frozen DA master plan (negative = saving). Source: M8\_CORE\_FILTER\_ABLATION\_ANNUAL\_RESULTS.csv (FY2\_ref row for REF).*