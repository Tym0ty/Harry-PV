## Probabilistic calibration metrics before and after AgACI post-processing (test set).

| Case       | Calibration Method   |   PICP@80% | PICP@90%   |   MPIW@80% (W/m²) | MPIW@90% (W/m²)   |   CRPS (W/m²) |
|:-----------|:---------------------|-----------:|:-----------|------------------:|:------------------|--------------:|
| ID raw     | —                    |      0.742 | —          |             259.7 | —                 |         61.46 |
| ID + AgACI | RC-Conformal(4)      |      0.807 | 0.905      |             283   | 358.0             |         61.46 |
| DA raw     | —                    |      0.7   | —          |             291.6 | —                 |         70.15 |
| DA + AgACI | RC-Conformal(4)      |      0.803 | 0.9        |             323.2 | 412.0             |         70.15 |

*Note: Nominal coverage targets: PICP\,$\geq$\,0.80 at 80\% and PICP\,$\geq$\,0.90 at 90\%. AgACI = Risk-Conditional Adaptive Gamma Interval (RC-Conformal, 4 risk classes). CRPS computed on raw 19-quantile forecasts (q05\textasciitilde{}q95); AgACI adjusts boundary quantiles only, so raw and AgACI rows share the same CRPS. Source: FORECASTING\_PIPELINE\_FINAL\_STATUS\_2026-05-22.md (recorded results, confirmed by user).*