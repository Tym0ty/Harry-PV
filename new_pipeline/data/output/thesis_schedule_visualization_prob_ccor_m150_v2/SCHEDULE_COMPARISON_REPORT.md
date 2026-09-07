# Schedule Comparison Report

## Formal Experiment Roots

- `new_pipeline/data/output/experiments/pvfocus_milp_revised_conformant_v1/01_main_four_cases/`
- `new_pipeline/data/output/experiments/pvfocus_scenario_bridge_ablation_v1/scheduling/D_G1_R3/`

## Annual Result Check

| case                     |   annual_total_M_NTD |   annual_peak_kw_summary |   annual_peak_kw_hourly_D_mth_running |   max_executed_grid_import_kw |   rows |   missing_hours |   duplicated_timestamps |   infeasible_steps |   fallback_steps | full_year_pass   | hourly_file                                                                                                                                        | annual_file                                                                                                                        |
|:-------------------------|---------------------:|-------------------------:|--------------------------------------:|------------------------------:|-------:|----------------:|------------------------:|-------------------:|-----------------:|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|
| MPC_DET_REVISED_FULLSPEC |              107.018 |                  5215.85 |                               5215.85 |                       5184.75 |   8760 |               0 |                       0 |                  0 |                0 | True             | new_pipeline\data\output\experiments\pvfocus_milp_revised_conformant_v1\01_main_four_cases\MPC_DET_REVISED_FULLSPEC_hourly_dispatch_replay.parquet | new_pipeline\data\output\experiments\pvfocus_milp_revised_conformant_v1\01_main_four_cases\annual_cost_summary_realized_basis.csv  |
| D_G1_R3_MPC_PROB         |              105.34  |                  4787.47 |                               4787.47 |                       4758.92 |   8760 |               0 |                       0 |                  0 |                0 | True             | new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\scheduling\D_G1_R3\D_G1_R3_MPC_PROB_hourly_dispatch_replay.parquet        | new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\scheduling\D_G1_R3\annual_cost_summary_realized_basis.csv |
| D_G1_R3_CCOR_PROB_m150   |              103.687 |                  4695.42 |                               4695.42 |                       4667.42 |   8760 |               0 |                       0 |                  0 |                0 | True             | new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\scheduling\D_G1_R3\D_G1_R3_CCOR_PROB_m150_hourly_dispatch_replay.parquet  | new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\scheduling\D_G1_R3\annual_cost_summary_realized_basis.csv |

## Selected 48-hour Windows

- Figure 1: 2025-07-05 00:00:00 to 2025-07-06 23:00:00
- Figure 2: 2025-05-05 00:00:00 to 2025-05-06 23:00:00

## Event Selection Reasons

Figure 1 was selected by lexicographic ranking over complete non-cross-month
48-hour windows: avoided locked-in month-to-date peak, grid-import peak
reduction, then executed BESS dispatch separation.

- Avoided locked-in MTD peak: 762.271 kW
- Grid-import peak reduction: 958.027 kW
- Executed dispatch separation: 8668.005 kWh
- Local TOU cost difference: -10137.906 NTD

Figure 2 additionally ranks by reduction in exposure above the CCOR operational
shield threshold.

- Avoided locked-in MTD peak: 415.467 kW
- Shield exposure reduction: 3613.028 kW-h
- Grid-import peak reduction: 442.151 kW
- Executed dispatch separation: 13956.897 kWh
- Shield-active hours in the improved case: 2

## Objective Observations

### Figure 1

- The probabilistic case changes executed BESS net power by 8668.0 kWh over the selected window.
- The maximum realized grid import is lower by 958.0 kW.
- The month-to-date peak increment is lower by 762.3 kW.
- SOC differs by -37.0 percentage points at the baseline peak hour.

### Figure 2

- CCOR-MPC changes executed BESS net power by 13956.9 kWh over the selected window.
- The maximum realized grid import is lower by 442.2 kW.
- Exposure above the operational shield threshold is lower by 3613.0 kW-h.
- The month-to-date peak increment is lower by 415.5 kW.
- The red threshold in the figure is a controller-side operational shield trigger, not a Taipower tariff boundary or realized cost item.

## Captions

**Figure 1.** Executed PV-BESS schedules under standard MPC with deterministic
and contract-risk-preserving probabilistic PV information from
2025-07-05 00:00 to 2025-07-06 23:00
local time. Both cases use the same realized load, PV generation, tariff, and
controller structure; only the forecast representation differs.

**Figure 2.** Executed PV-BESS schedules under standard MPC and CCOR-MPC using
the same contract-risk-preserving probabilistic PV scenarios from
2025-05-05 00:00 to 2025-05-06 23:00
local time. CCOR-MPC applies a 150 kW operational shield margin below the
binding demand guard.

## Data Basis

All plotted dispatch quantities are realized-operation, executed first-step
records. No future-horizon decisions, scenario recourse trajectories, expected
objectives, or optimizer-internal non-executed plans are plotted.
