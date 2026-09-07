# Figure M2 48h Peak Reconciliation

## Finding
The apparent differences between `3,533` and `3,512`, and between `3,157` and `3,138`, are not off-by-one-hour errors. The replay's stored `D_mth_running` is a demand-charge proxy based on `kappa * p_grid_realized`, while the figure's `48-h max grid import` is the raw realized grid import.

Implementation evidence in the repository:
- `new_pipeline/scripts/experiments/run_pvfocus_milp_revised_conformant_v1.py:297` updates `D_mth = max(state['D_mth'], dem['kappa'] * p_grid)`.
- `new_pipeline/scripts/experiments/run_scheduling_pv_focus_ccor_mpc.py:318` applies the same running demand peak update.

## Final KPI Definitions
- `17:00 grid import`: raw `p_grid_realized` at `2025-03-02 17:00`.
- `48-h max grid`: maximum raw `p_grid_realized` within `2025-03-02 00:00` to `2025-03-03 23:00`.
- `End MTD demand peak`: stored `D_mth_running` at `2025-03-03 23:00`, which is kappa-adjusted.

## Case Summary
### MPC-PROB-Risk
- Incoming raw grid-import MTD peak before window: `1843.000` kW.
- Incoming stored demand MTD peak before window: `1854.058` kW.
- 17:00 raw grid import: `3437.368` kW.
- 17:00 stored running MTD demand peak: `3457.992` kW.
- 17:00 creates new raw-grid audit MTD peak: `True`.
- 17:00 creates new stored MTD demand peak: `True`.
- 48-hour maximum raw grid import: `3512.037` kW at `2025-03-03 11:00:00`.
- End-of-window stored MTD demand peak: `3533.110` kW.
- Difference at end, stored demand peak minus raw grid audit peak: `21.072` kW.

### CCOR-PROB-Risk
- Incoming raw grid-import MTD peak before window: `1843.000` kW.
- Incoming stored demand MTD peak before window: `1854.058` kW.
- 17:00 raw grid import: `1970.000` kW.
- 17:00 stored running MTD demand peak: `2601.916` kW.
- 17:00 creates new raw-grid audit MTD peak: `False`.
- 17:00 creates new stored MTD demand peak: `False`.
- 48-hour maximum raw grid import: `3138.288` kW at `2025-03-03 11:00:00`.
- End-of-window stored MTD demand peak: `3157.118` kW.
- Difference at end, stored demand peak minus raw grid audit peak: `18.830` kW.

## Labeling Decision
- The final figure labels the black line peak as `48-h max grid`.
- The KPI box labels the running peak as `End MTD demand peak` to avoid mixing it with raw grid import.
- For MPC, 17:00 is correctly labeled as a `New MTD demand peak`; it is later superseded by the larger 2025-03-03 11:00 demand peak.
- For CCOR, 17:00 is correctly labeled as `No MTD increase at 17:00`.
- `is_new_mtd_peak` in the reconciliation CSV refers to the raw-grid audit running peak. `is_new_stored_mtd_peak` refers to the stored kappa-adjusted demand peak.