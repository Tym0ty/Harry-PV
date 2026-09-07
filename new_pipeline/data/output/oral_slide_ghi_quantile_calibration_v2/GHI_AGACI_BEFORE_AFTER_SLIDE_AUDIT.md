# GHI Quantile Calibration Slide Audit

## Final Status

Confirmed: the slide table corresponds to the identified experiment outputs.

## A. Source Identification

Most credible source for the target slide table:

- Raw quantile input: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/input/raw_19q_quantiles.parquet`
- AgACI endpoint output currently present: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/stage1_aci_calibrated.parquet`
- Exact raw+AgACI Stage-5 snapshot used for recomputation and plotting: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/stage1_aci_calibrated_raw19q.parquet`
- Stage-5 recomputed metric CSV: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/stage5_forecast_comparison.csv`
- Thesis-facing table already present: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/agaci_publication_table_summary.csv`
- Evaluation script formula source: `new_pipeline/scripts/stage5_evaluation_comparison.py`

The Stage-5 evaluator uses daytime rows with `ghi_clear_sky > 0` and the peak
window `hour_local` 10-15. The complete `stage1_aci_calibrated_raw19q.parquet`
snapshot is the most trustworthy source for this slide because it contains the
raw 19 quantiles and the Stage-5 AgACI endpoints for all 8760 test rows. The
smaller `stage1_aci_calibrated.parquet` currently present is daytime-only and
does not reproduce the target table by itself. AgACI columns directly available
in the Stage-5 snapshot are q05/q10/q90/q95 endpoints; intermediate quantiles
fall back to raw values for the Stage-5 CRPS approximation.


A newer ID-H24 operational table also exists at `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/thesis_figures_tables_ch3_ch4_v2/tables/chapter4/table_4_2_raw_vs_agaci_calibration_summary.csv`. Its values differ from the target slide table because it belongs to a later operational ID-H24/Q19 line; it is therefore not used for this slide-table reconciliation.

## B. Reconciliation Table

| Method             | Metric                 |   Target value from slide |   Recomputed / located value |   Rounded target |   Rounded recomputed | Match?   | Notes                                                                 |
|:-------------------|:-----------------------|--------------------------:|-----------------------------:|-----------------:|---------------------:|:---------|:----------------------------------------------------------------------|
| Raw (uncalibrated) | PICP80_all_daytime_pct |                    73.9   |                  73.858      |           73.9   |               73.9   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| Raw (uncalibrated) | PICP90_all_daytime_pct |                    84.4   |                  84.4188     |           84.4   |               84.4   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| Raw (uncalibrated) | ACE80_all_daytime      |                    -0.061 |                  -0.0614202  |           -0.061 |               -0.061 | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| Raw (uncalibrated) | MPIW80_Wm2             |                   252.4   |                 252.44       |          252.4   |              252.4   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| Raw (uncalibrated) | CRPS_all_daytime       |                    59.91  |                  59.9147     |           59.91  |               59.91  | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| Raw (uncalibrated) | PICP80_peak_10_15_pct  |                    73.2   |                  73.1507     |           73.2   |               73.2   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| AgACI (calibrated) | PICP80_all_daytime_pct |                    80.3   |                  80.3256     |           80.3   |               80.3   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| AgACI (calibrated) | PICP90_all_daytime_pct |                    90.1   |                  90.1402     |           90.1   |               90.1   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| AgACI (calibrated) | ACE80_all_daytime      |                     0.003 |                   0.00325645 |            0.003 |                0.003 | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| AgACI (calibrated) | MPIW80_Wm2             |                   275.1   |                 275.054      |          275.1   |              275.1   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| AgACI (calibrated) | CRPS_all_daytime       |                    59.87  |                  59.8744     |           59.87  |               59.87  | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |
| AgACI (calibrated) | PICP80_peak_10_15_pct  |                    80.3   |                  80.3196     |           80.3   |               80.3   | YES      | Stage-5 recomputation from raw_19q_quantiles + stage1_aci_calibrated. |

## Verified Slide Table

| Method             | PICP80 (all daytime)   | PICP90 (all daytime)   |   ACE80 (all daytime) |   MPIW80 (W/m2) |   CRPS (all daytime) | PICP80 (peak 10-15h)   |
|:-------------------|:-----------------------|:-----------------------|----------------------:|----------------:|---------------------:|:-----------------------|
| Raw (uncalibrated) | 73.9%                  | 84.4%                  |                -0.061 |         252.44  |              59.9147 | 73.2%                  |
| AgACI (calibrated) | 80.3%                  | 90.1%                  |                 0.003 |         275.054 |              59.8744 | 80.3%                  |

## Representative Figure Case Selection

Selected target day: `2025-03-17`.

Selection rule: rank test-set target days by the number of daytime 80% interval
misses fixed by AgACI, penalize newly introduced misses, and use observed GHI
variability plus interval spread as tie-breakers. This keeps the example
data-driven rather than manually selected.

Top 10 candidate days:

| target_date   |   n_daytime_rows |   raw_picp80_day |   cal_picp80_day |   raw_misses_fixed_by_cal |   raw_hits_lost_after_cal |   mean_abs_hourly_ghi_change |   mean_raw_mpiw80 |   selection_score |
|:--------------|-----------------:|-----------------:|-----------------:|--------------------------:|--------------------------:|-----------------------------:|------------------:|------------------:|
| 2025-03-17    |               12 |         0.333333 |         0.916667 |                         7 |                         0 |                      28.5354 |           223.728 |           71.6893 |
| 2025-05-16    |               13 |         0.538462 |         1        |                         6 |                         0 |                     158.102  |           311.733 |           64.7207 |
| 2024-12-14    |               11 |         0.363636 |         0.909091 |                         6 |                         0 |                      10.8333 |           141.925 |           60.9263 |
| 2025-09-27    |               12 |         0.583333 |         1        |                         5 |                         0 |                     150.758  |           226.451 |           54.1474 |
| 2025-01-26    |               11 |         0.272727 |         0.727273 |                         5 |                         0 |                      29.7222 |           248.888 |           51.8389 |
| 2025-09-06    |               13 |         0.692308 |         1        |                         4 |                         0 |                     157.176  |           247.133 |           44.3792 |
| 2025-05-02    |               13 |         0.692308 |         1        |                         4 |                         0 |                     137.037  |           273.311 |           44.1073 |
| 2024-11-11    |               11 |         0.636364 |         1        |                         4 |                         0 |                      72.7778 |           279.711 |           42.8541 |
| 2025-07-09    |               13 |         0.615385 |         0.923077 |                         4 |                         0 |                      39.1204 |           273.882 |           42.1518 |
| 2025-04-24    |               13 |         0.615385 |         0.923077 |                         4 |                         0 |                      21.7593 |           310.869 |           41.9895 |

## Figure Data Checks

| product   |   rows |   nan_or_inf_count |   monotonic_violations_q05_q10_q50_q90_q95 |   min_value |   max_value |
|:----------|-------:|-------------------:|-------------------------------------------:|------------:|------------:|
| raw       |     13 |                  0 |                                          0 |           0 |     577.209 |
| agaci     |     13 |                  0 |                                          0 |           0 |     621.022 |

## Slide Text

Title: **Calibrate the GHI Quantiles**

Subtitle: **AgACI improves interval reliability before PV conversion**

Key takeaway: **Dynamic AgACI moves the 80% and 90% GHI prediction intervals
toward their nominal coverages before PV conversion; the improved reliability is
obtained with a modest increase in interval width.**

Method note for the Stage-5 slide artifact:

- AgACI directly calibrates q05/q95 and q10/q90 endpoints.
- Interior quantiles are not independently AgACI-updated in this Stage-5 metric artifact.
- q50 remains unchanged as the raw median.
- In the later Q19 product, endpoint shifts are propagated by a piecewise-linear mapping and fixed-anchor monotonic projection; use that wording only when referring to that newer Q19 artifact.

## Output Index

- Main figure PNG: `figures/ghi_agaci_before_after_quantile_bands.png`
- Main figure PDF: `figures/ghi_agaci_before_after_quantile_bands.pdf`
- Main figure SVG: `figures/ghi_agaci_before_after_quantile_bands.svg`
- Source CSV: `tables/ghi_agaci_before_after_fan_source.csv`
- Reconciliation CSV: `tables/table_4_2_reconciliation_audit.csv`
- Recomputed metrics: `tables/stage5_recomputed_metrics.csv`
- Candidate days: `tables/representative_day_candidates.csv`
- Reproducible script copy: `scripts/create_ghi_agaci_before_after_slide_v1.py`
