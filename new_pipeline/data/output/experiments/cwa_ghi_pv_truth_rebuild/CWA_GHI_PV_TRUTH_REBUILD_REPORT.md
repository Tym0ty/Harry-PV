# CWA GHI to PV Truth Rebuild Report

## Decision

Valid PV truth is now defined as CWA realized GHI converted to PV using the project fixed PV capacity and PR. `NTUST_Load_PV.csv` PV / `Solar_kWh` is ignored.

## Formula

`pv_realized_kw = clip(0.8 * ghi_realized_Wm2 / 1000 * 2687.0, 0, 2687.0)`

## Generated Package

`new_pipeline\data\output\experiments\cwa_ghi_pv_truth_rebuild\full_year_replay_truth_package_CWA_GHI_PV.parquet`

This does not overwrite `bridge_outputs_fullyear/full_year_replay_truth_package.parquet`.

## Summary

| series                         |   min_kw |   max_kw |   mean_kw |   p95_kw |   p99_kw |   annual_energy_kwh |   capacity_factor |   zero_fraction |
|:-------------------------------|---------:|---------:|----------:|---------:|---------:|--------------------:|------------------:|----------------:|
| CWA_GHI_PV_TRUTH               |        0 |  2298.88 |   328.374 |  1671.91 |  2054.06 |         2.87656e+06 |          0.122209 |        0.505936 |
| OLD_INVALID_NTUST_SOLAR_SCALED |        0 |  2531.03 |   357.512 |  1843.68 |  2240.35 |         3.13181e+06 |          0.133053 |        0.499658 |

## Monthly Summary

|   month_id |   cwa_pv_energy_kwh |   cwa_pv_max_kw |   cwa_pv_mean_kw |   ghi_max_wm2 |   ghi_mean_wm2 |
|-----------:|--------------------:|----------------:|-----------------:|--------------:|---------------:|
|          1 |              159714 |         1654    |          214.669 |       769.444 |        99.8647 |
|          2 |              143710 |         1862.99 |          213.854 |       866.667 |        99.4854 |
|          3 |              225404 |         2137.66 |          302.962 |       994.444 |       140.939  |
|          4 |              257405 |         2161.54 |          357.507 |      1005.56  |       166.313  |
|          5 |              285279 |         2269.02 |          383.439 |      1055.56  |       178.377  |
|          6 |              318388 |         2298.88 |          442.206 |      1069.44  |       205.715  |
|          7 |              299061 |         2257.08 |          401.963 |      1050     |       186.995  |
|          8 |              373784 |         2239.17 |          502.398 |      1041.67  |       233.717  |
|          9 |              316705 |         2089.89 |          439.868 |       972.222 |       204.628  |
|         10 |              239394 |         2018.24 |          321.766 |       938.889 |       149.687  |
|         11 |              138642 |         1510.69 |          192.559 |       702.778 |        89.5788 |
|         12 |              119074 |         1498.75 |          160.046 |       697.222 |        74.4538 |

## Important Caveat

The comparison against old PV is only a magnitude check. The old PV is invalid and must not be used as ground truth.

## Next Step

Patch replay/scheduling runners to accept this replacement truth path, then rerun the required DA/MPC/M8 or same-core cases. Do not update thesis final numbers until those reruns are complete.
