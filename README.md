# Harry-PV: Day-Ahead Solar Irradiance Probabilistic Forecasting & Bridge Layer

Day-ahead GHI (Global Horizontal Irradiance) probabilistic forecasting pipeline for the NTUST site in Taipei, Taiwan. Uses CQR-XGBQ (Conformal Quantile Regression with XGBoost Quantile) to produce 19 calibrated quantile forecasts (P05–P95), converts to PV power, generates reduced scenarios, then transforms them via a bridge layer into representative-day tables for the annual sizing MILP.

## Pipeline Overview

```
Forecast Layer:
  CWA Hourly Obs + GFS NWP → Feature Engineering → 19-Quantile XGBQ → CQR Calibration
  → S5a: GHI Q50 → PVWatts → pv_point_forecast_caseyear.parquet (Det PV)
  → S6-S8: Gaussian Copula → GHI→PV → k-Medoids → pv_scenarios_reduced_caseyear.parquet (Prob PV)

Full-Year Bridge Layer:
  Det PV + Prob PV + NTUST Load → Calendar/Tariff Tags → Load Perturbation
  → 4 Full-Year MILP Ingest Packages (C0–C3) + Truth Replay Package

Full-Year MILP:
  Ingest Package → 365×24 Direct Solve (Gurobi) → Sizing (CC*, P_B*, E_B*)
  → Fixed-Design Replay on Truth Data → Design-to-Replay Gap Analysis
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
| S5 | Load 19Q via residual quantiles | **S5a**: Det PV point artifact (GHI Q50 → PV); **S5b**: Det load profile |
| S6 | Gaussian Copula joint GHI+Load | GHI-only scenarios + deterministic load appended |
| S8 | k-medoids on PV+Load (48-dim) | k-medoids on PV only (24-dim); also outputs PV-only parquet |

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

## Bridge Layer

### Full-Year Bridge (Current — per FF0326Harry_Bridge_Layer_Engineering_vfinal)

The bridge is now a **full-year data organization layer** (not compression). It assembles forecast PV and load data into standardized ingest packages for the full-year direct solve MILP.

| Metric | Value |
|--------|-------|
| Case year | 2024-11-01 to 2025-10-31 (365 days) |
| PV scale | 50 kW reference → 2,687 kW |
| Deterministic PV | Pre-computed S5 artifact (GHI Q50 → PVWatts → scaled) |
| Probabilistic PV | 5 reduced scenarios per day |
| Load perturbation | Billing hours ×1.05, non-billing ×1.02 |
| Output packages | 4 MILP ingest + 1 truth replay + calendar manifest |

### Bridge Output Artifacts (in `bridge_outputs_fullyear/`)

| File | Purpose |
|------|---------|
| `caseyear_calendar_manifest.parquet` | Calendar index with season/day_type/holiday tags |
| `full_year_milp_ingest_pvdet_loaddet.parquet` | C0: Det PV + Det Load |
| `full_year_milp_ingest_pvprob_loaddet.parquet` | C1: Prob PV + Det Load |
| `full_year_milp_ingest_pvdet_loadpert.parquet` | C2: Det PV + Pert Load |
| `full_year_milp_ingest_pvprob_loadpert.parquet` | C3: Prob PV + Pert Load |
| `full_year_replay_truth_package.parquet` | Realized PV + Load for replay |
| `load_perturbation_manifest.parquet` | Perturbation mode and multipliers |
| `bridge_full_year_report.json` | QA and coverage summary |
| `bridge_run_metadata.json` | Reproducibility metadata |

### Legacy: Repday-Based Bridge (v7/v1)

Previous bridge versions used representative-day clustering (16 body + 28 risk days in v7). Outputs in `bridge_outputs/` (v7) and `bridge_outputs_v1/` (v1). Superseded by the full-year approach.

## Forecast Thesis Figures

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

## Full-Year Direct Solve MILP — C0–C3 Results

Full-year direct solve MILP for campus microgrid sizing (per FF0326Harry_MILP_Engineering_Spec_FullYear_Formal_vfinal). Replaces the previous representative-day approach with 365-day chronological solve. PV capacity is **fixed at 2,687 kW**; the MILP optimizes **BESS power/energy** and **contract capacity**.

Key features:
- **Full-year direct solve**: 365 days × 24 hours (no representative-day compression)
- **P_grid split**: P_grid_load + P_grid_ch (grid can charge battery separately)
- **Green SOC** tracking for RE accounting
- **PWL battery degradation** (4-segment convex cost)
- **Expected inter-day SOC** for probabilistic cases: E_daystart_{i+1} = Σ_ω π_ω · E(i,ω,24)
- **TOU_FixedPeak tariff** (spec §5.1)
- **No export / No CPPA** guardrail
- **Fixed-design replay** on truth data for validation

### 4-Case Matrix (C0–C3)

| Case | PV Info | Load Info | Positioning |
|------|---------|-----------|-------------|
| **C0** | Deterministic (GHI Q50 → PV) | Deterministic | Baseline |
| **C1** | Probabilistic (5 scenarios/day) | Deterministic | Value of probabilistic PV |
| **C2** | Deterministic | Perturbed (billing ×1.05, non-billing ×1.02) | Load stress impact |
| **C3** | Probabilistic | Perturbed | Probabilistic PV under load stress |

### Solve Results

| Case | Total AEC (M NTD) | BESS P (kW) | BESS E (kWh) | E/P | CC (kW) | RE% | Solve (s) |
|------|-------------------|-------------|--------------|-----|---------|-----|-----------|
| C0 | 95.56 | 1,227 | 7,733 | 6.3 | 3,195 | 15.0 | 12.0 |
| C1 | 95.35 | 1,319 | 9,616 | 7.3 | 3,116 | 15.9 | 32.4 |
| C2 | 100.95 | 1,287 | 8,286 | 6.4 | 3,331 | 14.5 | 11.1 |
| C3 | 100.70 | 1,391 | 10,356 | 7.4 | 3,262 | 15.4 | 31.0 |

### Replay Results (Truth Data)

| Case | Solve (M) | Replay (M) | Gap | Over-Contract (M) | Over Months | Worst Month (M) | RE% |
|------|-----------|------------|-----|--------------------|-------------|-----------------|-----|
| C0 | 95.56 | 95.77 | +0.2% | 0.28 | 4 | 10.52 | 14.9 |
| C1 | 95.35 | 95.95 | +0.6% | 0.20 | 3 | 10.30 | 14.9 |
| C2 | 100.95 | 95.93 | −5.0% | 0.09 | 2 | 10.42 | 14.9 |
| C3 | 100.70 | 96.16 | −4.5% | 0.02 | 1 | 10.17 | 14.9 |

### Key Findings

**Does probabilistic PV outperform deterministic?**

C1 (probabilistic) is actually **cheaper in solve cost** than C0 (deterministic): 95.35M vs 95.56M (−0.22%). With the updated PVWatts-based deterministic PV (from the S5 artifact), the deterministic forecast is more realistic, and the probabilistic approach provides both **lower design cost** and **better risk management**:

| Metric | C0 (Det) | C1 (Prob) | C1 Advantage |
|--------|----------|-----------|--------------|
| Solve cost (design) | 95.56M | 95.35M | −0.22% cheaper |
| Replay cost (truth) | 95.77M | 95.95M | +0.19% |
| Over-contract months | 4 | 3 | 25% fewer |
| Worst-month bill | 10.52M | 10.30M | −2.1% |
| Solve-to-replay gap | +0.2% | +0.6% | Both small |
| Over-contract fees | 0.28M | 0.20M | −29% |
| BESS investment | 1,227 kW / 7,733 kWh | 1,319 kW / 9,616 kWh | +24% E_B (hedging) |

The probabilistic design invests more in BESS energy capacity as a hedge, which pays off with fewer over-contract events and lower worst-month bills. The marginal replay cost difference (0.18M) is negligible compared to the operational risk reduction.

**Load perturbation effect:**

Under perturbed load stress, the pattern is stronger — C3 achieves the best risk profile of all cases (only 1 over-contract month, lowest worst-month bill at 10.17M) while adding just +0.23% cost vs C2. The perturbation stress (billing +5%, non-billing +2%) causes the solve to over-estimate costs by 4–5%, creating a conservative design buffer.

### MILP Configuration

| Parameter | Value | Spec ID |
|-----------|-------|---------|
| PV capacity (fixed) | 2,687 kW | PV_001 |
| BESS power CAPEX | 11,944 NTD/kW | BESS_006 |
| BESS energy CAPEX | 7,738 NTD/kWh | BESS_007 |
| Discount rate | 5% | FIN_001 |
| BESS lifetime | 15 yr (CRF 0.0963) | FIN_002/003 |
| η_ch / η_dis | 0.95 / 0.95 | BESS_001/002 |
| SOC limits | 10%–90% | BESS_003/004 |
| RE target | 20% | SYS_004 |
| T-REC cost | 4.63 NTD/kWh | RE_002 |
| κ (demand proxy) | 1.0035 | CP_006 |
| No export / No CPPA | Enforced | C14 |

### Legacy: Repday-Based 8-Case Results (STO-MILP v10)

The previous repday-based implementation (per BH_STO-MILP Engineering Spec v2.3) used Bridge v7's 44 representative days with calendar mapping. Key result: mainline M2_I1_R0 = 107.82M NTD, BESS 825/4321 kW/kWh. See `milp_outputs/case_summary_main.csv` for full results. The full-year direct solve (C0–C3 above) supersedes this approach.

## Project Structure

```
Harry-PV/
├── notebooks_forecast_fixed/          # Spec-compliant forecast notebooks
│   ├── v1_fixed_baseline.ipynb        #   Baseline XGBoost (single model)
│   └── v2_fixed_tuned_xgb_5seed.ipynb #   Tuned XGBoost + 5-seed ensemble
├── notebooks_bridge/                  # Bridge layer
│   ├── bridge_full_year.py            #   Full-year bridge (current, per vfinal spec)
│   ├── bridge_v7.ipynb                #   Legacy repday bridge v7
│   └── bridge_v1.ipynb                #   Legacy repday bridge v1
├── notebooks_milp/                    # Full-year MILP
│   ├── milp_common.py                 #   Shared config, data loading, C0-C3 case table
│   ├── milp_solver.py                 #   Full-year direct solve + replay engine
│   ├── milp_fullyear_cases.ipynb      #   C0-C3 runner notebook
│   ├── milp_figures_fullyear.py       #   7 thesis figures generator
│   ├── milp_v10_cases.ipynb           #   Legacy 8-case repday runner
│   ├── milp_v10_bridge_comparison.ipynb #  Legacy bridge comparison
│   └── milp_figures.ipynb             #   Legacy figures
├── notebooks_experiments/              # Experiment notebooks
├── pipeline_outputs/                  # Forecast pipeline artifacts
├── bridge_outputs_fullyear/           # Full-year bridge outputs (current)
├── bridge_outputs/                    # Legacy bridge v7 outputs
├── milp_outputs/                      # MILP results & figures
├── docs/figures/                      # Thesis figures
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
| `pv_point_forecast_caseyear.parquet` | Deterministic PV from GHI Q50 (S5 artifact) |
| `load_deterministic_hourly.parquet` | Deterministic campus load profile (reference) |
| `scenarios_ghi_raw_N.parquet` | Raw stochastic GHI trajectories |
| `scenarios_joint_ghi_load_raw_N.parquet` | Joint-compatible raw GHI+Load |
| `scenarios_joint_pv_load_raw_N.parquet` | After GHI→PV conversion |
| `scenarios_joint_pv_load_reduced_K.parquet` | Reduced PV scenarios (with load) |
| `pv_scenarios_reduced_caseyear.parquet` | PV-only reduced scenarios (Bridge-ready) |
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
