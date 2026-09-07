# Structured Load-Bias Profile Report

## Status

Generated deterministic given load profiles for 2024-11-01 to 2025-10-31.

## Methodological Label

`load_base` is `deterministic_given_load_profile` from realized campus load used by replay/KPI. Structured-low/high are robustness sensitivities, not probabilistic load forecasts.

## Rationale

The main stochastic source is PV uncertainty. Structured load-bias profiles are kept as outer sensitivity cases to avoid weak random load-error modeling dominating the thesis focus.

## Bias Matrix

| profile         | season    |   night |   morning |   daytime |   evening_peak |   late_night |
|:----------------|:----------|--------:|----------:|----------:|---------------:|-------------:|
| base            | summer    |    0    |      0    |      0    |           0    |         0    |
| uniform_low_5   | summer    |   -0.05 |     -0.05 |     -0.05 |          -0.05 |        -0.05 |
| uniform_high_5  | summer    |    0.05 |      0.05 |      0.05 |           0.05 |         0.05 |
| structured_low  | summer    |   -0.03 |     -0.05 |     -0.05 |          -0.07 |        -0.03 |
| structured_high | summer    |    0.03 |      0.05 |      0.05 |           0.07 |         0.03 |
| base            | nonsummer |    0    |      0    |      0    |           0    |         0    |
| uniform_low_5   | nonsummer |   -0.05 |     -0.05 |     -0.05 |          -0.05 |        -0.05 |
| uniform_high_5  | nonsummer |    0.05 |      0.05 |      0.05 |           0.05 |         0.05 |
| structured_low  | nonsummer |   -0.02 |     -0.04 |     -0.04 |          -0.05 |        -0.02 |
| structured_high | nonsummer |    0.02 |      0.04 |      0.04 |           0.05 |         0.02 |

## Summary

| profile         |   annual_kwh |   max_load_kw |   min_load_kw |   negative_load_count |   oc_risk_hours_gt_CC |   near_oc_hours_gt_0p95CC |
|:----------------|-------------:|--------------:|--------------:|----------------------:|----------------------:|--------------------------:|
| base            |  2.12684e+07 |       5179    |             0 |                     0 |                  1503 |                      1741 |
| structured_low  |  2.03357e+07 |       4920.05 |             0 |                     0 |                  1246 |                      1495 |
| structured_high |  2.22012e+07 |       5437.95 |             0 |                     0 |                  1733 |                      2057 |
| uniform_low_5   |  2.0205e+07  |       4920.05 |             0 |                     0 |                  1255 |                      1503 |
| uniform_high_5  |  2.23319e+07 |       5437.95 |             0 |                     0 |                  1727 |                      2039 |
