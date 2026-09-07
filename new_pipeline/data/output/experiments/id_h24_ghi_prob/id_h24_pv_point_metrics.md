# ID-H24 PV Point Forecast Metrics

**Generated**: 2026-06-05  
**Model**: W1_XGB_D24 GHI point forecast → PV (formula: clip(0.80 × GHI/1000 × 2687, 0, 2687))  
**Primary truth**: formula-derived PV from y_ghi_true  
**Secondary truth**: pv_realized_kw from truth package (where matched)

---

## 1. Overall and Lead-Group Metrics (vs Formula Truth)

| Subset | n | RMSE (kW) | MAE (kW) | MBE (kW) | R² |
|--------|---|----------|---------|---------|----|
| Overall | 101688 | 292.73 | 197.07 | -2.46 | 0.7757 |
| H01-H03 | 12711 | 234.24 | 150.55 | -0.07 | 0.8564 |
| H04-H06 | 12711 | 283.85 | 193.29 | -4.55 | 0.7891 |
| H07-H12 | 25422 | 297.64 | 201.58 | -2.43 | 0.7681 |
| H13-H24 | 50844 | 305.33 | 207.39 | -2.55 | 0.7560 |
| Summer_Day | 46464 | 327.13 | 226.58 | -10.80 | 0.7552 |
| TOU_1621 | 18912 | 183.64 | 120.19 | -11.83 | 0.6527 |
| HighPV_Top10pct | 10248 | 423.81 | 347.94 | -343.36 | -7.0988 |
| HighPV_Top5pct | 5184 | 456.63 | 384.70 | -382.99 | -23.1737 |
| LowPV_lt100kW | 18168 | 167.82 | 72.68 | 67.66 | -31.5249 |
| HighRamp_P90 | 10176 | 379.65 | 278.29 | -40.68 | 0.4745 |

---

## 2. Comparison vs Realized PV (truth package)

| Subset | n_matched | RMSE (kW) | MAE (kW) | MBE (kW) | R² |
|--------|----------|----------|---------|---------|----|
| Overall | 101688 | 390.61 | 266.62 | -57.71 | 0.6757 |
| H01-H03 | 12711 | 351.66 | 229.40 | -55.32 | 0.7372 |
| H04-H06 | 12711 | 383.33 | 262.14 | -59.80 | 0.6877 |
| H07-H12 | 25422 | 392.96 | 269.14 | -57.68 | 0.6718 |
| H13-H24 | 50844 | 400.37 | 275.78 | -57.80 | 0.6593 |
| Summer_Day | 46464 | 364.17 | 259.79 | -37.19 | 0.7419 |
| TOU_1621 | 18912 | 241.06 | 166.11 | -45.38 | 0.4913 |
| HighPV_Top10pct | 10248 | 531.65 | 467.24 | -427.90 | -1.9302 |
| HighPV_Top5pct | 5184 | 563.62 | 510.02 | -478.77 | -3.2426 |
| LowPV_lt100kW | 18168 | 196.16 | 95.45 | 46.06 | -1.0993 |
| HighRamp_P90 | 10176 | 417.06 | 321.94 | -84.53 | 0.4566 |

**Note on MBE vs realized**: Positive MBE means formula under-predicts actual PV. PR=0.80 is a conservative planning value; actual system often outperforms slightly.