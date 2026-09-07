# FIGURES_TABLES_INDEX.md
Generated: 2026-05-28

> **Source policy**: All numerical values read directly from existing CSV/JSON files.
> No recalculation or fabrication. Estimated values explicitly marked (est.).

---

## Tables

| Stem | Caption | Source Files |
|------|---------|-------------|
| Table_3-8-1_case_definitions | Case definitions | text only |
| Table_3-8-2_horizon_sensitivity | Horizon sensitivity | text only |
| Table_4-2-1_point_forecast_skill | Point forecast skill | metrics_overall.csv; da_baseline_summary.json (da_test) |
| Table_4-2-2_per_horizon_rmse | Per-horizon RMSE | metrics_by_horizon.csv; metrics_overall_stack.csv |
| Table_4-2-3_probabilistic_calibration | Probabilistic calibration | FORECASTING_PIPELINE_FINAL_STATUS_2026-05-22.md (recorded, confirmed by user) |
| Table_4-2-4_id_benchmark | ID model benchmark | metrics_overall.csv; metrics_overall_stack.csv; metrics_exp5_overall.csv; FORECAST_INVENTORY_REPORT bootstrap CIs |
| Table_4-2-5_da_benchmark | DA model benchmark | da_baseline_summary.json (da_test); da_exp1_best_params.json; da_exp2_best_params.json |
| Table_4-4-1_main_cost_comparison | Main cost comparison | H24_cost_component_raw.csv; M8_Det_H24_kpis.json; M8_CORRECTED_COST_TABLES.csv (M1 est.) |
| Table_4-4-3_m8_det_vs_prob | M8-Det vs M8-Prob | H24_cost_component_raw.csv; M8_Det_H24_kpis.json |
| Table_4-5-1_horizon_robustness | Horizon robustness | EOD_H24_H48_COMPARISON_TABLE.csv |
| Table_4-6-1_m8_ablation_eod | M8 filter ablation (EOD) | M8_CORE_FILTER_ABLATION_ANNUAL_RESULTS.csv (FY2_ref) |

## Figures

| Stem | Caption | Source Files |
|------|---------|-------------|
| Figure_3-1-1_architecture | Three-layer DA–ID architecture | No data (matplotlib diagram) |
| Figure_4-2-1_id_benchmark | ID benchmark bar chart | metrics_overall.csv; metrics_overall_stack.csv; metrics_exp5_overall.csv |
| Figure_4-4-1_cost_comparison | Stacked cost comparison | H24_cost_component_raw.csv; M8_Det_H24_kpis.json; M8_CORRECTED_COST_TABLES.csv (M1 est.) |
| Figure_4-4-2_waterfall | M8-Det → M8-Prob waterfall | H24_cost_component_raw.csv; M8_Det_H24_kpis.json |
| Figure_4-5-1_horizon_robustness | Horizon robustness line chart | EOD_H24_H48_COMPARISON_TABLE.csv |
| Figure_4-6-1_m8_filter_stats | M8 filter statistics (two subplots) | M8_CORE_FILTER_ABLATION_ANNUAL_RESULTS.csv (FY2_ref, EOD); M8_H24_FY2_15pct/hourly.parquet (monthly acceptance) |

---

## Missing / Estimated Data

| Item | Status | Details |
|------|--------|---------|
| DA-Det (M1) TREC | **Estimated (5.265 M NTD)** | M1 row in M8_CORRECTED_COST_TABLES.csv has TREC_rev=19.6946 (old wrong revenue); TREC purchase estimated from FY2 values |
| DA-Det (M1) Full_total | **Estimated (~102.47 M NTD)** | Sum of confirmed components + estimated TREC |
| DA-Exp2 MAE/R² | **Not in JSON** | Only RMSE available from da_exp2_best_params.json; shown as "—" |
| ID Table 4.2-4 bootstrap CIs | **From FORECAST_INVENTORY_REPORT_2026-05-21.md** | Not in CSV; sourced from handover markdown |
| PICP/MPIW/CRPS (Table 4.2-3) | **From FORECASTING_PIPELINE_FINAL_STATUS_2026-05-22.md** | Recorded result, confirmed by user Q2 |
| H24 filter-level rejection counts | **Not available** | H24 hourly parquet has decision=candidate/fallback_m2 but no per-filter breakdown. Figure 4.6-1(a) uses EOD ablation data (FY2_ref). |

---

## Key Numbers Cross-Reference

| Value | Source | Note |
|-------|--------|------|
| ID v4 RMSE = 116.45 W/m² | metrics_overall.csv | H=1~6 pooled, n=25686, AgACI corrected |
| DA v2 RMSE = 140.88 W/m² | da_baseline_summary.json (da_test) | Clean; raw_19q=124.92 has DA leakage |
| M8-FY2-H24 = 99.6088 M NTD | H24_cost_component_raw.csv | Main proposed result |
| M8-FY2-EOD = 99.4300 M NTD | EOD_H24_H48_COMPARISON_TABLE.csv | EOD REF for Table 4.6-1 |
| M8-Det-H24 = 101.6968 M NTD | M8_Det_H24_kpis.json | full_total_NTD / 1e6 |
| H24 MPC acceptance = 70.5% | M8_H24_FY2_15pct/hourly.parquet | candidate=6172/8760; EOD acceptance=74.3% |
| EOD acceptance = 74.3% | M8_CORE_FILTER_ABLATION_ANNUAL_RESULTS.csv | FY2_ref row |

> **Warning on acceptance rate**: The thesis may have cited 74.3% as the M8-Prob-H24 acceptance rate.
> The correct H24 rate is **70.5%** (6172/8760). 74.3% is the **EOD** horizon rate from ablation CSV.
> Please correct thesis text accordingly.