# Chapter 4 Results Package
Generated: 2026-03-31 19:36

## Directory Structure
- figures/   PNG files (spec-named)
- tables/    CSV + XLSX tables
- data/      Tidy CSV source data per figure
- manifest/  figure_manifest.csv, table_manifest.csv
- qa/        qa_report.json, hard_fail.log, warnings.log

## Figures
- **B-F1**: Raw vs Reduced PV Scenario Trajectories
- **B-F2**: Reduced Scenario Probability Weights
- **B-F3**: Prob PV Scenario Fan — Representative Summer Day
- **B-F4**: Load Uncertainty K-Scenario Fan
- **M-F1**: BESS Sizing Comparison — C0–C3 + PI
- **M-F2**: Solve-World Cost Stacked Bar
- **R-F1**: Replay Annual Total Cost — All Cases
- **R-F2**: Over-Contract Fee & Worst Month Bill
- **R-F3**: RE20 Compliance & T-REC Volume
- **R-F4**: Monthly Bill Breakdown — All Cases
- **R-F5**: Design-to-Replay Gap (%)
- **C-F1**: 5-Day Dispatch Trace: C0 vs C1
- **C-F2-C0**: 48h Stress Window Dispatch — C0
- **C-F3-C0**: 48h Stress Window Multi-Panel — C0
- **C-F2-C1**: 48h Stress Window Dispatch — C1
- **C-F3-C1**: 48h Stress Window Multi-Panel — C1
- **L-F1**: Load Uncertainty Impact — C2 vs C3 vs baseline
- **S-F1**: Sensitivity Heatmap — replay_total_mean
- **S-F2**: Sensitivity Heatmap — replay_total_std
- **S-F3**: Sensitivity Heatmap — replay_total_mean
- **S-F4**: Design Stability vs N

## Tables
- **B-T1**: Bridge Annual Package Completeness
- **B-T2**: Bridge QA Gate Summary
- **M-T1**: BESS Sizing + Contract Capacity Results
- **M-T2**: Solve-World Annual Cost Breakdown
- **R-T1**: Replay Annual Cost Summary
- **L-T1**: Incremental Value of Load Uncertainty
- **S-T1**: N×K Sufficiency Decision Table
