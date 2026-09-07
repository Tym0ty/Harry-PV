# Reproducibility

This package supports three practical reproduction levels.

## Level 1: Inspect Final Results

Start with:

- `README.md`
- `FINAL_FILE_SELECTION.md`
- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS.md`
- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/final_four_main_case_summary.csv`
- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/final_four_delta_comparison.csv`

This level requires no solver.

## Level 2: Verify Final Evidence Tables

Use `PACKAGE_MANIFEST.csv` to locate included evidence files. The most important
final evidence artifacts are:

- `new_pipeline/data/output/experiments/cwa_ghi_pv_truth_rebuild/full_year_replay_truth_package_CWA_GHI_PV.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/load_bias_profiles.parquet`
- `new_pipeline/data/output/experiments/pv_focused_structured_load_bias_bridge/id_h24_pvfocus_base_tkm_lowpv_safe_K5_scenarios.parquet`
- `new_pipeline/data/output/experiments/scheduling_pv_focus_lowpv_safe_da_mpc/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/scheduling_pv_focus_ccor_v2_full_year_m150/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/final_four_load_bias_robustness_pvfocus/annual_cost_summary_realized_basis.csv`
- `new_pipeline/data/output/experiments/tailaware_ablation_pvfocus_km_vs_tkm/annual_cost_summary_realized_basis.csv`

This level requires Python with `pandas` and parquet support. It does not require
rerunning the MILP.

## Level 3: Rerun Final Scripts

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The final scheduling scripts use Gurobi, so a working Gurobi installation and
license are required for full MILP reruns.

Core scripts:

- `new_pipeline/scripts/experiments/exp_rebuild_pv_truth_from_cwa_ghi.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_generate_pv_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_build_netload_scenarios.py`
- `new_pipeline/scripts/experiments/exp_pv_focus_lowpv_reduction_fix.py`
- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_da_mpc.py`
- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_v2_full_year_m150.py`
- `new_pipeline/scripts/experiments/run_final_four_load_bias_robustness_pvfocus.py`
- `new_pipeline/scripts/experiments/run_tailaware_ablation_pvfocus_km_vs_tkm.py`
- `new_pipeline/scripts/experiments/write_final_thesis_method_result_report_pvfocus.py`

Important caveat: this curated GitHub package does not include every raw input
or every intermediate file from the original local workspace. Full end-to-end
reruns from raw observations may require restoring the larger local data archive.

## Final Data Policy

Included:

- compact final reports
- compact final CSV summaries
- thesis figures and tables
- selected final parquet evidence needed by the final report
- final method scripts

Excluded:

- virtual environments
- old handover archives
- raw scenario pools
- full hourly dispatch parquet outputs
- exploratory logs
- old/non-final DA, LOAD-QN, random load-error, and S1-S6 artifacts

## Known Caveat

The final report marks one remaining item as `CITATION_NEEDED`: a formal
literature citation for the simple performance-ratio GHI-to-PV conversion.
