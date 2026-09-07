# PV Truth Package Usage Audit

## Status

`PV truth source = CWA_GHI_PV_REBUILT` for new PV-focused validation and future scheduling utilities.

## Configured Truth Package

`new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet`

## Old Truth Package

`bridge_outputs_fullyear\full_year_replay_truth_package.parquet` is not overwritten and is not used by the PV-focused utility.

## Audit Table

| component                        | configured_truth_path                                                                                           | resolved_TRUTH_constant                                                                                         | pv_truth_source             | uses_old_ntust_solar_kwh   | validation_uses_cwa_ghi_pv_truth   | scheduling_will_use_cwa_ghi_pv_truth   | old_truth_overwritten   |   row_count |   pv_max_kw |   pv_mean_kw | has_source_marker   | source_marker_values   |
|:---------------------------------|:----------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|:----------------------------|:---------------------------|:-----------------------------------|:---------------------------------------|:------------------------|------------:|------------:|-------------:|:--------------------|:-----------------------|
| pv_focus_bridge_utils.load_truth | new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet | new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet | CWA_GHI_PV_REBUILT          | False                      | True                               | True                                   | False                   |        8760 |     2298.88 |      328.374 | True                | CWA_GHI_REALIZED_TO_PV |
| old_truth_package                | bridge_outputs_fullyear\full_year_replay_truth_package.parquet                                                  |                                                                                                                 | OLD_INVALID_NTUST_SOLAR_KWH | True                       | False                              | False                                  | False                   |        8760 |     2531.03 |      357.512 | False               |                        |

## Conclusion

Validation uses CWA-GHI-derived PV truth; future PV-focused scheduling runners importing `load_truth()` will use the same package. The old NTUST `Solar_kWh` PV truth is not used for new formal PV-focused validation/replay/KPI outputs.
