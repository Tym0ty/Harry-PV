# Final File Selection for GitHub Packaging

This file records the current final-use judgement for the Harry-PV project.
The latest final line is the PV-focused rolling MPC / CCOR-MPC line, not the
older full-year DA/Layer-A package described in earlier README sections.

## Final Basis

Primary source of truth:

- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS.md`
- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/final_report_source_manifest.csv`

Final main cases:

- `MPC_DET`
- `MPC_PROB`
- `CCOR_DET`
- `CCOR_PROB`

Excluded from the final mainline:

- DA main cases
- LOAD-QN
- random load-error mainline
- old S1-S6 scenario inputs
- old invalid NTUST Solar_kWh PV truth
- LitErr random load-error cases

## Must Keep: Project Shell

- `.gitignore`
- `README.md`
- `requirements.txt`
- `run_all.py`

## Must Keep: Final Code

Core final PV-focused bridge and scheduling code:

- `new_pipeline/scripts/experiments/pv_focus_bridge_utils.py`
- `new_pipeline/scripts/experiments/exp_rebuild_pv_truth_from_cwa_ghi.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_reduce_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_lowpv_reduction_fix.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_validate_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_reframed_validation.py`
- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_da_mpc.py`
- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_v2_full_year_m150.py`
- `new_pipeline/scripts/experiments/run_final_four_load_bias_robustness_pvfocus.py`
- `new_pipeline/scripts/experiments/run_tailaware_ablation_pvfocus_km_vs_tkm.py`
- `new_pipeline/scripts/experiments/write_final_thesis_method_result_report_pvfocus.py`

Final/near-final GHI probabilistic forecast code:

- `new_pipeline/scripts/experiments/run_id_h24_ghi_prob_q19_forecast_only_v1.py`
- `new_pipeline/scripts/experiments/run_id_h24_ghi_prob_q19_independent_agaci_v4.py`

MILP implementation evidence:

- `milp_v2/common.py`
- `milp_v2/config.yaml`
- `milp_v2/layer_a/`
- `milp_v2/layer_b/milp_cvar.py`
- `milp_v2/layer_b/milp_mpc.py`
- `milp_v2/layer_b/milp_pvfocus_revised_fullspec.py`
- `milp_v2/layer_b/run_layer_b_mpc.py`

## Must Keep: Final Summary Tables and Reports

Keep the whole folder:

- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/`

This folder contains the final thesis report, final case definitions, annual
summary tables, deltas, traceability tables, and source manifest.

Direct evidence artifacts referenced by the final manifest:

- `new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/km_vs_tkm_delta_comparison.csv`
- `new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/cwa_pv_truth_summary.csv`
- `new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/PV_FOCUSED_VALIDATION_REFRAMED_LOW_PV_FIX_REPORT.md`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_ghi_prob_metrics.csv`
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/id_h24_pv_prob_metrics.csv`
- `new_pipeline/data/output/experiments/main_forecasting_model_comparison/main_model_comparison.csv`
- `new_pipeline/data/output/experiments/lit_guided_id_h24_ghi/literature_guided_id_h24_ghi_model_comparison.csv`
- `new_pipeline/data/output/experiments/figures/table1_agaci_metrics.csv`

Also keep the compact reports/audits in the same final evidence folders where
they explain the CSVs above. Avoid keeping all hourly parquet outputs unless a
reproduction package, not a GitHub source package, is required.

## Keep for Thesis Figures and Tables

These are useful final thesis assets and are small:

- `thesis_figures/`
- `thesis_tables/`
- `FIGURES_TABLES_INDEX.md`

Keep newer generated thesis/slide figure packages only if they are used in the
submitted thesis or oral defense:

- `new_pipeline/data/output/thesis_ch4_forecast_figures_v1/`
- `new_pipeline/data/output/thesis_schedule_visualization_prob_ccor_m150_v2/`
- `new_pipeline/data/output/thesis_mechanism_figures_48h/`
- `new_pipeline/data/output/thesis_table_4_3_panel_b_v1/`
- `new_pipeline/data/output/thesis_table_4_4_six_case_bar_figure_v1/`
- `new_pipeline/data/output/oral_slide_ghi_quantile_calibration_v2/`

## Keep as Reference, Not Final Mainline

These may be useful for history, comparison, or appendix, but they are not the
latest final mainline:

- `Project_Archive_Prediction_Final/`
- `notebooks/`
- `notebooks_bridge/`
- `notebooks_experiments/`
- `notebooks_forecast_fixed/`
- `notebooks_milp/`
- `docs/figures/`
- `results/ch4/`
- `results/sensitivity/`
- `pipeline_standalone/`
- `milp_v3_horizon/`
- root audit and decision markdown files, especially:
  - `RESEARCH_PIPELINE_SUMMARY.md`
  - `DECISION_LOG.md`
  - `EXPERIMENT_PROGRESS_AND_METHOD_LOG.md`
  - `SCENARIO_GENERATION_TCOPULA_AUDIT.md`
  - `FORECASTING_DA_ID_MODEL_COMPARISON_FINAL.md`
  - `FINAL_METHOD_LOGIC_AUDIT_AND_RERUN_RECOMMENDATION.md`

## Exclude from GitHub Package

Do not include these in the GitHub-ready final package:

- `.git/`
- `.venv/`
- `.idea/`
- `.claude/scheduled_tasks.lock`
- `__pycache__/`
- `*.pyc`
- `catboost_info/`
- `lightning_logs/`
- `audit_temp/`
- `output/`
- `outputs/`
- `final_results/`
- root `*.zip`
- root `*.7z`
- duplicated handover folders and packages:
  - `BH_NEXT_CHAT_HANDOVER_20260606/`
  - `BH_NEXT_CHAT_HANDOVER_20260606.zip`
  - `BH_THESIS_HANDOVER_2026_05_FINAL/`
  - `BH_THESIS_HANDOVER_2026_05_FINAL.zip`
  - `forecasting_handover_for_thesis_text/`
  - `forecasting_handover_for_thesis_text.zip`
- old bridge/MILP regenerated outputs unless specifically needed for appendix:
  - `bridge_outputs/`
  - `bridge_outputs_v1/`
  - `bridge_outputs_fullyear/`
  - `milp_outputs/`
- massive raw/intermediate outputs:
  - `new_pipeline/data/output/experiments/**/raw/`
  - `new_pipeline/data/output/experiments/**/*raw*_M500*.parquet`
  - `new_pipeline/data/output/experiments/**/*hourly_dispatch*.parquet`
  - `new_pipeline/data/output/experiments/**/*_hourly.parquet`
  - `new_pipeline/data/output/experiments/**/*_daily.parquet` unless a compact final evidence archive is required

## GitHub Risk Notes

- No currently tracked file appears to exceed GitHub's 100 MB hard limit.
- The largest final evidence file in the final manifest is
  `id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`, about 36 MB.
- `new_pipeline/` is about 12 GB because of generated output data. Do not add
  the whole folder recursively to GitHub without updated ignore rules.
- The final report status is
  `FINAL_REPORT_READY_WITH_MINOR_UNCERTAIN_ITEMS`; the remaining caveat is a
  missing formal citation for the simple PR-based GHI-to-PV conversion.

