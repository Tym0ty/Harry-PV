# Harry-PV: Day-Ahead Solar Irradiance Probabilistic Forecasting & Bridge Layer

Day-ahead GHI (Global Horizontal Irradiance) probabilistic forecasting pipeline for the NTUST site in Taipei, Taiwan. Uses CQR-XGBQ (Conformal Quantile Regression with XGBoost Quantile) to produce 19 calibrated quantile forecasts (P05–P95), converts to PV power, generates reduced scenarios, then transforms them via a bridge layer into representative-day tables for the annual sizing MILP.

## Pipeline Overview

```
Forecast Layer:
  CWA Hourly Obs + GFS NWP → Feature Engineering → 19-Quantile Models → CQR Calibration
  → GHI Scenario Generation → GHI→PV Conversion → k-Medoids Reduction → Bridge-Ready Package

Bridge Layer:
  Daily PV Packages → Day Descriptors → Risk-Day Tagging → Body-Day Clustering
  → Medoid Selection → Calendar Map + Weights → Repday Scenario Tables → Annual MILP Ingest
```

- **Gate-compliant**: All NWP data respects D-1 12:00 UTC deadline (no future data leakage)
- **Strict anti-leakage**: No contemporaneous CWA observations used as features — only lagged (>=24h) observations
- **Split-conformal CQR**: Calibrated prediction intervals with finite-sample correction
- **GHI-only stochastic**: Load is deterministic; only GHI/PV drives scenario uncertainty
- **PV-only reduction**: k-medoids clusters on PV trajectories (24-dim), not PV+Load

## Forecast Results

### Original Notebooks (Test Set — Last 365 Days, Daytime)

| Version | Description | MAE (W/m²) | R² | Notes |
|---------|-------------|------------|------|-------|
| V1 | Baseline XGBoost, default params | 83.93 | 0.8049 | Single model, lr=0.05, depth=8 |
| V2 | Tuned XGBoost + 5-seed ensemble | 83.01 | 0.8071 | 162-combo grid search |
| V3 | Tuned XGBoost + 10-seed ensemble | 83.01 | 0.8071 | 720-combo fine grid |
| V4 | LightGBM 10-seed ensemble | ~82.09 | ~0.81 | |
| V5 | CatBoost 10-seed ensemble | ~82.32 | ~0.81 | |
| V6 | 3-Model Ensemble (XGB+LGB+CB) | 83.00 | 0.8062 | Equal-weight average |

### Spec-Compliant Fixed Notebooks (Test Set — Last 365 Days)

Per Forecast Engineering Spec Final v2 (2024-03-24). Load data sourced from `NTUST_Load_PV.csv` (true gross load, not net load).

| Version | Description | MAE All (W/m²) | MAE Daytime (W/m²) | R² All | R² Daytime | CRPS |
|---------|-------------|----------------|---------------------|--------|------------|------|
| V1 Fixed | Baseline XGBoost | 42.56 | 80.44 | 0.8759 | 0.8145 | 31.64 |
| V2 Fixed | Tuned XGBoost + 5-seed | 42.10 | 79.60 | 0.8773 | 0.8165 | 31.24 |

The forecast model (S0–S4b) is identical between original and fixed notebooks. Small metric differences are due to XGBoost non-determinism across runs. The fixes only affect downstream stages:

| Stage | Original | Fixed (Spec-Compliant) |
|-------|----------|----------------------|
| S5 | Load 19Q via residual quantiles | Deterministic load profile |
| S6 | Gaussian Copula joint GHI+Load | GHI-only scenarios + deterministic load appended |
| S8 | k-medoids on PV+Load (48-dim) | k-medoids on PV only (24-dim) |

### Comparison with Harry's Original Pipeline

| Metric | Harry's Pipeline | Our Pipeline |
|--------|------------------|-------------|
| MAE | 75.14 W/m² | ~80 W/m² |
| R² | 0.8618 | ~0.81 |
| Data Leakage | Yes (contemporaneous obs) | No (strict gate compliance) |
| Postprocessing | Bucket shift+scale + afternoon fix | CQR calibration only |

Harry's lower MAE is explained by his model using contemporaneous CWA weather observations as features — data leakage that would not be available at forecast time in a real day-ahead setting.

## NWP Contribution Analysis

Ablation experiment comparing forecast performance with and without the 14 GFS-derived NWP features. See `notebooks_experiments/nwp_contribution.ipynb`.

| Setting | MAE All (W/m²) | MAE Daytime (W/m²) | R² All | R² Daytime | CRPS |
|---------|----------------|---------------------|--------|------------|------|
| V1 With NWP | 42.56 | 84.04 | 0.8759 | 0.8046 | 31.64 |
| V1 Without NWP | 66.86 | 132.31 | 0.7305 | 0.5759 | 47.73 |
| V2 With NWP (5-seed) | 42.10 | 83.16 | 0.8773 | 0.8067 | 31.24 |
| V2 Without NWP (5-seed) | 65.98 | 130.57 | 0.7322 | 0.5785 | 47.07 |

**NWP Impact (V1)**: MAE reduced by 48.27 W/m² (36.5%), R² improved by +0.2287, CRPS reduced by 33.7%

**NWP Impact (V2)**: MAE reduced by 47.40 W/m² (36.3%), R² improved by +0.2282, CRPS reduced by 33.6%

NWP is critical — without GFS forecast data, the model cannot predict next-day cloud cover, which is the dominant source of GHI variability in Taipei's subtropical climate.

## Comparison with Pieter's RNN-LSTM (Per-Season)

Re-evaluation using Pieter Hernando's methodology (per-season, all hours including nighttime). See `notebooks_experiments/pieter_comparison.ipynb`.

| Season | MAE Ours (W/m²) | MAE Pieter (W/m²) | Improvement | RMSE Ours | RMSE Pieter | Improvement |
|--------|-----------------|-------------------|-------------|-----------|-------------|-------------|
| Summer | 54.71 | 105.13 | -48.0% | 108.00 | 154.65 | -30.2% |
| Fall | 30.68 | 93.57 | -67.2% | 69.59 | 136.06 | -48.9% |
| Winter | 32.39 | 99.81 | -67.5% | 72.62 | 142.04 | -48.9% |
| Spring | 47.56 | 118.77 | -60.0% | 95.07 | 158.13 | -39.9% |
| **Avg** | **41.34** | **104.32** | **-60.4%** | **86.32** | **147.72** | **-41.6%** |

Caveats: different test years (2024–25 vs 2019), our model uses NWP (GFS) which Pieter's LSTM did not, and Pieter trains per-season while we train one model on all data.

## Bridge Layer Results

Per Bridge Layer Engineering Spec v3 (2024-03-24). Case year: 2024-11-01 to 2025-10-31.

| Metric | Value |
|--------|-------|
| Calendar days | 365 |
| Total repdays | 95 (20 body + 75 risk) |
| Body clusters | 20 (stratified by month, k-medoids on day descriptors) |
| Risk days | 75 (top 10% by stress/peak-load/low-PV, union + monthly supplement) |
| Scenarios per repday | 5 (per-repday, inherited from source date) |
| Calendar map coverage | 100% |
| Weight sum | 365 (= n_days) |
| Pi sum per repday | 1.0000 |
| All months have risk days | Yes (12/12) |

### Bridge Output Artifacts (in `bridge_outputs/`)

| File | Spec Ref | Purpose |
|------|----------|---------|
| `repdays_metadata.parquet` | §7.1 | Master index of representative days and risk days |
| `calendar_map.parquet` | §7.2 | Maps each calendar day to a repday (for SOC linkage + monthly max-demand) |
| `scenarios_repdays_pv_reduced.parquet` | §7.3 | Repday-level PV scenarios for annual sizing MILP |
| `risk_day_tags.parquet` | §2 | Risk tags and scores for all dates |
| `bridge_report.json` | §8 | Bridge parameters, thresholds, and QA diagnostics |
| `bridge_run_metadata.json` | §8 | Reproducibility (config hash, seed, version) |

## Thesis Figures

Publication-quality visualizations generated from `notebooks_experiments/thesis_figures.ipynb`. Both V1 (baseline XGBoost) and V2 (tuned + 5-seed ensemble) results are shown where applicable.

### Fig 1 — Quantile Fan Chart (3 Representative Days)

![Quantile Fan Chart](docs/figures/fig1_quantile_fan_chart.png)

Shows the 19-quantile probabilistic forecast (P05–P95) as nested prediction bands for three representative days: a clear-sky day, a mixed/partly-cloudy day, and an overcast day. The median forecast (P50) is drawn as a solid line, with progressively lighter shading toward the tails. This demonstrates that the model produces well-calibrated uncertainty — narrow bands on clear days (high confidence) and wide bands on cloudy days (appropriate uncertainty).

### Fig 2 — Reliability Diagram (CQR Calibration Proof)

![Reliability Diagram](docs/figures/fig2_reliability_diagram.png)

Plots nominal quantile level (x-axis) vs observed empirical coverage (y-axis) for both raw XGBQ output and post-CQR calibrated output. A perfectly calibrated model falls on the diagonal. The raw model shows systematic under-coverage at the tails; after split-conformal CQR calibration, all 19 quantiles align closely with the diagonal — proving that our prediction intervals have valid finite-sample coverage guarantees.

### Fig 3 — NWP Ablation (Impact of Weather Forecasts)

![NWP Ablation](docs/figures/fig3_nwp_ablation.png)

Grouped bar chart comparing forecast performance with and without the 14 GFS-derived NWP features, across both V1 and V2 model versions. Metrics shown: MAE (daytime), R² (daytime), and CRPS. Removing NWP degrades MAE by ~36% and R² by ~0.23 — confirming that numerical weather prediction data is the single most important input for day-ahead solar forecasting in Taipei's cloud-dominated climate.

### Fig 4 — Per-Season Comparison with Pieter's RNN-LSTM

![Pieter Comparison](docs/figures/fig4_pieter_comparison.png)

Side-by-side per-season MAE comparison between our CQR-XGBQ pipeline and Pieter Hernando's dual-layer RNN-LSTM (2023 thesis). Uses Pieter's exact methodology: per-season evaluation, all hours including nighttime. Our model achieves 48–68% lower MAE across all four seasons, with the largest gains in Fall and Winter where NWP cloud-cover forecasts provide the most value.

### Fig 5 — Feature Importance (Top 20)

![Feature Importance](docs/figures/fig5_feature_importance.png)

Top 20 features ranked by XGBoost gain, with NWP-derived features highlighted in blue and non-NWP features in gray. NWP features (especially `dswrf` — downward shortwave radiation flux, and cloud cover variables) dominate the top ranks, visually confirming the ablation study results. The lagged CWA observation features (`ghi_lag24`, `temp_lag24`) also contribute meaningfully as persistence baselines.

### Fig 6 — Error Heatmap (Hour × Month)

![Error Heatmap](docs/figures/fig6_error_heatmap.png)

MAE heatmap with hour-of-day on the y-axis and month on the x-axis (daylight hours only). Reveals the spatiotemporal error structure: highest errors occur during midday hours in summer months (Jun–Aug) when convective cloud development is most unpredictable. Winter and shoulder-season mornings/evenings show the lowest errors. This pattern is consistent with Taipei's subtropical monsoon climate.

### Fig 7 — Scatter: Predicted vs Actual GHI

![Scatter Predicted vs Actual](docs/figures/fig7_scatter_pred_vs_actual.png)

Hexbin density scatter plot of predicted (P50 median) vs actual GHI for the full test set. The diagonal line represents perfect prediction. Point density is shown via color intensity. The R² value is annotated directly on the plot. The model tracks well across the full GHI range, with the expected increase in scatter at high irradiance values where cloud transients create the most variability.

## MILP Optimization Results

Two-stage stochastic MILP for optimal campus microgrid sizing: PV, BESS, and Taipower contract demand. Stage 1 (here-and-now) sizes the equipment; Stage 2 (recourse) dispatches hourly across 95 representative days × 5 PV scenarios. Both Gurobi and HiGHS solvers produce identical results.

### Optimal Capacities

| Asset | Optimal Size |
|-------|-------------|
| PV Capacity | 13,966 kW |
| BESS Power | 1,551 kW |
| BESS Energy | 7,151 kWh (E/P = 4.6h) |
| Contract Demand | 3,081 kW |

### Annual Cost Breakdown

| Component | Cost (TWD) |
|-----------|-----------|
| PV annuity | 37,550,128 |
| BESS power annuity | 1,454,583 |
| BESS energy annuity | 8,383,675 |
| BESS O&M | 715,144 |
| Contract demand | 8,266,976 |
| **Investment subtotal** | **56,370,507** |
| Operating cost | 21,221,889 |
| **Total annual cost** | **77,592,395 (77.59 M)** |

### Key Metrics

| Metric | Value |
|--------|-------|
| RE share | 51.0% (target: 30%) |
| Baseline annual cost (no BESS) | 95.34 M TWD |
| Annual savings | 17.75 M TWD (18.6%) |
| Solve time (HiGHS) | 10.2s |
| Solve time (Gurobi) | 1.1s |

### Solver Comparison: Gurobi vs HiGHS

Both solvers produce identical optimal solutions for this problem. The key differences:

| | Gurobi | HiGHS |
|---|---|---|
| **License** | Commercial (free academic license via [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/)) | Open-source (MIT), no license needed |
| **Python API** | Native `gurobipy` — rich API with callbacks, lazy constraints, solution pools | Via `PuLP` wrapper — simpler but less control |
| **Algorithm used** | Barrier (interior-point) + crossover | Dual simplex |
| **Solve time** | 1.1s | 10.2s |
| **Iterations** | 9,105 | 60,468 |
| **When it matters** | Large MILPs with binary/integer variables (unit commitment, on/off decisions) — advanced branch-and-cut with heuristics | Small-medium LPs and MILPs where license cost or availability is a constraint |

For the current formulation (pure LP, ~80K continuous variables), HiGHS is perfectly adequate. Gurobi becomes significantly advantageous if the model grows to include integer variables (e.g., binary on/off for BESS, discrete PV panel counts, unit commitment constraints).

### MILP Configuration

| Parameter | Value |
|-----------|-------|
| PV CAPEX | 40,000 TWD/kW |
| BESS power CAPEX | 8,000 TWD/kW |
| BESS energy CAPEX | 10,000 TWD/kWh |
| Discount rate | 3% |
| PV lifetime | 20 yr (CRF 0.0672) |
| BESS lifetime | 10 yr (CRF 0.1172) |
| Charge/discharge efficiency | 95% / 95% |
| SOC limits | 10%–90% |
| RE target | 30% |
| Feed-in tariff | 2.0 TWD/kWh |

## Project Structure

```
Harry-PV/
├── notebooks_forecast_fixed/          # Spec-compliant forecast notebooks
│   ├── v1_fixed_baseline.ipynb        #   Baseline XGBoost (single model)
│   └── v2_fixed_tuned_xgb_5seed.ipynb #   Tuned XGBoost + 5-seed ensemble
├── notebooks_bridge/                  # Bridge layer notebook
│   └── bridge_v1.ipynb                #   Bridge v1 (per Spec v3)
├── notebooks_milp/                    # MILP optimization notebooks
│   ├── milp_common.py                 #   Shared config, data loading, results
│   ├── milp_v1_gurobi.ipynb           #   Gurobi solver (academic license)
│   └── milp_v1_highs.ipynb            #   PuLP + HiGHS solver (open-source)
├── notebooks_experiments/              # Experiment notebooks
│   ├── nwp_contribution.ipynb         #   NWP ablation study (with vs without GFS)
│   ├── pieter_comparison.ipynb        #   Per-season comparison with Pieter's RNN-LSTM
│   └── thesis_figures.ipynb           #   Publication-quality thesis visualizations
├── notebooks/                         # Original iteration notebooks (archived)
│   ├── v1_baseline.ipynb
│   ├── v2_tuned_xgb_5seed.ipynb
│   ├── v3_xgb_10seed.ipynb
│   ├── v4_lightgbm.ipynb
│   ├── v5_catboost.ipynb
│   └── v6_3model_ensemble.ipynb
├── pipeline_outputs/                  # Forecast pipeline artifacts (gitignored)
├── bridge_outputs/                    # Bridge layer artifacts (gitignored)
├── milp_outputs/                      # MILP results (gitignored)
├── docs/figures/                      # Thesis figures (PNG, tracked in git)
├── Project_Archive_Prediction_Final/  # Harry's original code & data (reference)
└── README.md
```

## Forecast Output Artifacts (in `pipeline_outputs/`)

| File | Purpose |
|------|---------|
| `features_hourly.parquet` | Merged CWA+NWP hourly features |
| `nwp_gate_manifest.parquet` | NWP gate traceability |
| `dataset_issue_target.parquet` | Supervised training dataset |
| `split_manifest.parquet` | Chronological split record |
| `forecast_ghi_quantiles_daily_base_raw.parquet` | Raw XGBQ 19Q (before CQR) |
| `forecast_ghi_quantiles_daily.parquet` | Official CQR-calibrated 19Q |
| `load_deterministic_hourly.parquet` | Deterministic campus load profile |
| `scenarios_ghi_raw_N.parquet` | Raw stochastic GHI trajectories |
| `scenarios_joint_ghi_load_raw_N.parquet` | Joint-compatible raw GHI+Load |
| `scenarios_joint_pv_load_raw_N.parquet` | After GHI→PV conversion |
| `scenarios_joint_pv_load_reduced_K.parquet` | Bridge-ready reduced PV scenarios |
| `qa_report.json` | QA summary |
| `run_metadata.json` | Reproducibility metadata |

## Data Sources

- **NTUST Load & PV**: `NTUST_Load_PV.csv` — Hourly campus load (Load_kWh) and rooftop PV generation (Solar_kWh) with electricity price. This is the corrected dataset with true gross load separated from PV; the previous dataset (`NTUST_Load_merged_fixed_v2.xlsx`) only contained net load (load − PV) as metered by Taipower. Date range: 2024-11-01 to 2025-11-01.
- **CWA Hourly**: Central Weather Administration hourly station observations (2021–2025)
- **GFS NWP**: NOAA GFS 0.25° forecasts — t2m, dswrf, cloud cover (lcc/mcc/hcc/tcc), wind, precipitation, humidity

## Requirements

```
python >= 3.10
xgboost, lightgbm, catboost
pvlib, scikit-learn, scipy
pandas, numpy, pyarrow
```

## Site

- **Location**: NTUST (National Taiwan University of Science and Technology)
- **Coordinates**: 25.0377°N, 121.5149°E
- **PV System**: 50 kW DC rated, DC/AC ratio 1.2, inverter efficiency 96%
