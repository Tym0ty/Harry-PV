# Slide 9 Build Report

## Outputs

- Editable PPTX: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/slide9_calibrate_ghi_quantiles/slide9_calibrate_ghi_quantiles.pptx`
- PNG preview: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/slide9_calibrate_ghi_quantiles/slide9_calibrate_ghi_quantiles_preview.png`
- Slide-optimized main figure PNG: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/slide9_calibrate_ghi_quantiles/figures/slide9_ghi_calibration_before_after.png`
- Slide-optimized main figure PDF: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/slide9_calibrate_ghi_quantiles/figures/slide9_ghi_calibration_before_after.pdf`
- Source figure data: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/tables/ghi_agaci_before_after_fan_source.csv`
- Reconciliation audit: `C:/Users/Harry/Downloads/Harry-PV/Harry-PV/Harry-PV/new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/tables/table_4_2_reconciliation_audit.csv`

## Design Choices

- Slide title changed to **Calibrate the GHI Quantiles**.
- The main figure removes internal title, internal representative-date subtitle,
  and internal method note. These are now handled by editable slide text.
- The right-side table is a PPT table object, not a screenshot.
- The table uses the verified Stage-5 / Table 4-2 values.
- The implementation note explicitly states that q05/q95 and q10/q90 endpoints
  are directly calibrated, interior quantiles are propagated through
  piecewise-linear mapping and fixed-anchor monotonic projection, and q50 remains
  unchanged as the raw median.

## QA

- No numeric values were changed from the verified Table 4-2 target.
- The representative-day figure remains based on the audited source CSV.
- The slide separates the illustrative representative day from aggregate
  calibration metrics.
- No internal labels such as "Stage-5 artifact" are shown on the slide itself.
