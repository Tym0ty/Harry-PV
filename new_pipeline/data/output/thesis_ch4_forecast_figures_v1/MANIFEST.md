# Chapter 4 Forecasting-to-Scheduling Bridge Figures

## Figure 4-1: Representative quantile fan charts of the GHI forecasting for four representative seasonal days
- Files: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_1_representative_quantile_fan_charts.png; new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_1_representative_quantile_fan_charts.pdf
- Source data: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_1_representative_quantile_fan_charts_source.csv
- Source artifact: `new_pipeline/data/output/experiments/id_h24_ghi_prob/q2_lg_calibrated.parquet`
- Suggested caption: Representative seasonal-day GHI fan charts from the operational ID-H24 calibrated quantile forecast. The shaded bands show the P10-P90 and interpolated P25-P75 intervals, with the calibrated median and observed GHI overlaid.

## Figure 4-2: Relative importance of the major predictors in the day-ahead forecasting model
- Files: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_2_feature_importance_day_ahead_model.png; new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_2_feature_importance_day_ahead_model.pdf
- Source data: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_2_feature_importance_day_ahead_model_source.csv
- Source artifact: `new_pipeline/data/output/experiments/da_three_model_comparison_bh/feature_importance_xgboost_da.csv`
- Suggested caption: Top predictors in the XGBoost day-ahead forecasting model, grouped by NWP, solar-geometry/calendar, lagged-observation, and meteorological feature families.

## Figure 4-3: Forecast performance with and without NWP features under the spec-compliant year-long test setting
- Files: Not generated / existing figure only
- Source data: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/audit/DATA_GAP_FIGURE_4_3_WITH_WITHOUT_NWP.md
- Source artifact: `DATA_GAP`
- Suggested caption: Not generated because no paired with-NWP versus without-NWP all-hours artifact was found.

## Figure 4-4: Raw daily PV scenario clouds and retained k-medoids representative trajectories for three different sky conditions under the Gaussian-Copula-based scenario generation framework
- Files: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_4_raw_vs_kmedoids_scenarios_three_sky_conditions.png; new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_4_raw_vs_kmedoids_scenarios_three_sky_conditions.pdf
- Source data: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_4_raw_vs_kmedoids_scenarios_three_sky_conditions_source.csv
- Source artifact: `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/raw/G1_temporal_gaussian/id_h24_pv_raw_scenarios_M500.parquet; new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/reduced/G1_R3_lowpv_safe/id_h24_K5_scenarios.parquet`
- Suggested caption: PV scenario clouds from the G1 temporal Gaussian copula and the retained R3 K=5 representative trajectories for objectively selected clear, variable-sky, and overcast issue times. The representative trajectories retain cluster probabilities and are overlaid with realized CWA-GHI-derived PV.

## Figure 4-5: Shield-margin sensitivity
- Files: Not generated / existing figure only
- Source data: n/a
- Source artifact: `new_pipeline/data/output/experiments/pvfocus_ccor_shield_sensitivity_g1_v1/annual/SHIELD_SENSITIVITY_ANNUAL_RESULTS.csv`
- Suggested caption: Existing shield-margin sensitivity figure is not redrawn in this generation pass; the authoritative 18-case source table is recorded here.

## Optional appendix figure: Reliability of predictive intervals before and after AgACI calibration
- Files: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/appendix_figure_calibration_reliability_before_after_agaci.png; new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/appendix_figure_calibration_reliability_before_after_agaci.pdf
- Source data: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/appendix_figure_calibration_reliability_before_after_agaci_source.csv
- Source artifact: `new_pipeline/data/output/thesis_figures_tables_ch3_ch4_v2/figures/chapter4/figure_4_2_reliability_diagram_source.csv`
- Suggested caption: Reliability diagram comparing raw and AgACI-calibrated ID-H24 predictive intervals against the ideal nominal-equals-empirical coverage line.

## Notes
- Figure 4-3 was not generated because no paired all-hours with-NWP versus without-NWP artifact was found.
- No existing experiment output was modified.


## Figure 4-3 update: Forecast performance with and without NWP features under the spec-compliant year-long test setting
- Files: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_3_with_without_nwp_all_hours_comparison.png; new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_3_with_without_nwp_all_hours_comparison.pdf
- Source data: new_pipeline/data/output/thesis_ch4_forecast_figures_v1/figures/figure_4_3_with_without_nwp_all_hours_comparison_source.csv
- Source artifact: with-NWP from `new_pipeline/data/output/thesis_figures_tables_ch3_ch4_v2_allhours/appendix_candidates/id_h24_operational_models_all_hours_leadwise_complete.csv`; without-NWP generated by `new_pipeline/scripts/experiments/run_without_nwp_h01_h24_and_update_ch4_figure_v1.py`
- Suggested caption: All-hours H01 and H24 forecast performance with and without NWP features under the same ID-H24 Direct-24 XGBoost training pattern. The without-NWP case removes `nwp_*` predictors while retaining lagged observations, solar geometry, calendar, and contemporaneous station features.
