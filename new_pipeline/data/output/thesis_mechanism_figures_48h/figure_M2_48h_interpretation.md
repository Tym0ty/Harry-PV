# Figure M2 48h Interpretation

## Source and Event
- MPC replay source: `C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\scheduling\D_G1_R3\D_G1_R3_MPC_PROB_hourly_dispatch_replay.parquet`
- CCOR replay source: `C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\scheduling\D_G1_R3\D_G1_R3_CCOR_PROB_m150_hourly_dispatch_replay.parquet`
- Scenario sources: `C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\raw\G1_temporal_gaussian\id_h24_pv_raw_scenarios_M500.parquet` and `C:\Users\Harry\Downloads\Harry-PV\Harry-PV\Harry-PV\new_pipeline\data\output\experiments\pvfocus_scenario_bridge_ablation_v1\reduced\G1_R3_lowpv_safe\id_h24_K5_scenarios.parquet`
- Window: `2025-03-02 00:00:00` to `2025-03-03 23:00:00`
- Baseline critical hour: `2025-03-02 17:00:00`
- No optimization, AgACI, scenario generation, scenario reduction, or replay was rerun.

## Critical-Hour Check
- MPC 17:00 charge: `1467.368` kW.
- CCOR 17:00 charge: `0.000` kW.
- MPC 17:00 grid import: `3437.368` kW.
- CCOR 17:00 grid import: `1970.000` kW.
- Critical-hour grid-import difference (CCOR minus MPC): `-1467.368` kW.
- Grid-to-BESS charging difference at 17:00: `1467.368` kW.
- PV-to-BESS charging difference at 17:00: `0.000` kW.
- Difference explained by charge: `True`.

## 48-Hour Consequences
- MPC 48-h maximum grid import: `3512.037` kW at `2025-03-03 11:00:00`.
- CCOR 48-h maximum grid import: `3138.288` kW at `2025-03-03 11:00:00`.
- 48-h maximum grid-import difference (CCOR minus MPC): `-373.749` kW.
- MPC end-of-window running MTD peak: `3533.110` kW.
- CCOR end-of-window running MTD peak: `3157.118` kW.
- End running-MTD difference (CCOR minus MPC): `-375.992` kW.
- MPC total charge/discharge over 48h: `10544.748` / `9516.635` kWh.
- CCOR total charge/discharge over 48h: `9368.714` / `8455.264` kWh.
- MPC end SOC: `10.000`%; CCOR end SOC: `10.000`%.
- MPC 48-h TOU energy cost: `247924.724` NTD.
- CCOR 48-h TOU energy cost: `251002.022` NTD.
- TOU energy cost difference (CCOR minus MPC): `3077.298` NTD.
- Realized hourly degradation cost was not present as a settlement column in the replay parquet; controller `objective_deg` is retained only as a diagnostic source column.

## Monday Battery Use
- MPC first Monday major discharge: `2025-03-03 06:00:00` at `752.583` kW.
- CCOR first Monday major discharge: `2025-03-03 06:00:00` at `752.583` kW.
- MPC Monday max discharge: `1079.997` kW at `2025-03-03 07:00:00`.
- CCOR Monday max discharge: `1079.997` kW at `2025-03-03 07:00:00`.

## Event Type
- Classified as `Type A`.
- Interpretation: CCOR suppresses the Sunday 17:00 peak-creating grid-to-BESS charge. Within the 48-hour window, both cases later charge and discharge, but CCOR keeps both the 48-hour maximum grid import and the end-of-window running MTD peak below the Standard MPC case.
- The figure supports a realized-replay mechanism observation, not an identical-initial-state one-step causal ablation.

## Missing or Non-Plotted Fields
- `degradation_cost_NTD`

## QA
- 48 rows per case: `{'CCOR-PROB-Risk': 48, 'MPC-PROB-Risk': 48}`.
- Energy-balance audit file: `figure_M2_48h_energy_balance_audit.csv`.
- Summary metrics file: `figure_M2_48h_summary_metrics.csv`.