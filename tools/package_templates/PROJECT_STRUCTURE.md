# Project Structure

This repository is a curated final package, not the full exploratory local
workspace.

## Root Files

- `README.md`: final GitHub-facing overview.
- `index.html`: static dashboard for final cases, results, figures, and file links.
- `FINAL_FILE_SELECTION.md`: inclusion and exclusion judgement used for packaging.
- `PACKAGE_README.md`: package generation notes.
- `PACKAGE_MANIFEST.csv`: inventory of packaged files.
- `PACKAGE_MISSING.csv`: missing-file audit from package generation.
- `REPRODUCIBILITY.md`: reproduction levels and practical rerun notes.
- `CITATION.cff`: citation metadata.
- `LICENSE`: MIT license for code unless a file states otherwise.
- `requirements.txt`: Python dependencies.

## Final Method Code

- `new_pipeline/scripts/experiments/`: final PV-focused scenario generation,
  low-PV-safe scenario reduction, scheduling runs, robustness runs, and final
  report generation.
- `milp_v2/layer_b/`: MILP and rolling MPC implementation evidence used by the
  final MPC and CCOR-MPC cases.
- `milp_v2/layer_a/`: retained sizing/design support code and compact design
  artifacts.

## Final Reports and Evidence

- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/`:
  consolidated final report, final case definitions, annual summaries, deltas,
  and traceability tables.
- `new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/`:
  corrected CWA-GHI-derived PV truth and summary audits.
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/`:
  PV-focused scenario bridge, deterministic load profiles, low-PV-safe K5
  scenario package, and validation reports.
- `new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/`:
  final standard MPC annual summaries and audits.
- `new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150/`:
  final CCOR-v2 m150 annual summaries and audits.
- `new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus/`:
  structured low/high load-bias robustness summaries.
- `new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/`:
  ordinary KM vs low-PV-safe TKM ablation summaries.
- `new_pipeline/data/output/experiments/id_h24_ghi_prob/`:
  final probabilistic GHI/PV forecast metrics and readiness reports.

## Thesis Assets

- `thesis_figures/`: compact thesis figure exports.
- `thesis_tables/`: compact thesis table exports.
- selected `new_pipeline/data/output/thesis_*` folders: newer thesis and defense
  figure/table packages that are small enough for GitHub.

## Deliberately Excluded

The package excludes the large original exploratory output workspace, including
raw scenario pools, full hourly dispatch outputs, logs, virtual environments,
old handover archives, and non-final experiment branches.
