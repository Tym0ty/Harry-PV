# Weather Regime Interpretation

**Generated**: 2026-06-04  
**Method**: K-means K=3 on [csi_lag_tau, csi_roll_mean_3h, csi_roll_std_3h]  
**Training data**: Pre-test daytime rows only (causal)  
**Regime assignment**: Regime 0=Clear (highest CSI centroid), 1=Partly Cloudy, 2=Overcast (lowest CSI centroid)

---

## 1. Regime Distribution and Sky Condition Statistics (Test Period)

| Regime | Label | n | % of Test | Mean CSI | Std CSI | Mean GHI (W/m²) | High-Ramp % | Cloudy % |
|--------|-------|---|-----------|---------|---------|-----------------|------------|----------|
| 0 | Clear (CSI≥0.7) | 28212 | 27.7% | 0.690 | 0.350 | 396.1 | 31.4% | 0.5% |
| 1 | Partly Cloudy | 66906 | 65.8% | 0.496 | 0.360 | 277.1 | 0.4% | 91.5% |
| 2 | Overcast (CSI<0.3) | 6570 | 6.5% | 0.637 | 0.359 | 363.9 | 15.9% | 98.9% |

---

## 2. Model RMSE by Regime (W/m²)

| Model | R0 (Clear (CSI≥0.7)) RMSE | R1 (Partly Cloudy) RMSE | R2 (Overcast (CSI<0.3)) RMSE |
|-------|---------|---------|---------|
| W1_XGB_D24 | 139.93 | 133.58 | 145.72 |
| W5_XGB_LG_Reg | 140.27 | 134.02 | 146.06 |
| P2_CB_LG | 140.03 | 134.44 | 146.02 |
| W6_CB_LG_Reg | 140.26 | 134.52 | 145.94 |
| G2_LGBM_D24 | 140.13 | 134.36 | 146.35 |
| P1_XGB_LG | 140.72 | 134.36 | 146.39 |
| W3_CB_D24 | 140.26 | 134.63 | 147.13 |
| G1_LGBM_LG | 141.00 | 135.08 | 146.94 |
| W2_XGB_Pool | 142.22 | 135.94 | 146.91 |
| G3_LGBM_Pool | 143.09 | 136.64 | 147.96 |
| W4_CB_Pool | 143.57 | 136.99 | 148.40 |
| PERS | 238.82 | 363.95 | 465.22 |
| DA | 394.95 | 311.37 | 372.05 |

---

## 3. Regime Improvement: Regime vs Base LEADGROUP

| Regime Model | Base Model | Regime | Regime RMSE | Base RMSE | Delta |
|-------------|-----------|--------|------------|----------|-------|
| W5_XGB_LG_Reg | P1_XGB_LG | Clear (CSI≥0.7) | 140.27 | 140.72 | -0.45 |
| W5_XGB_LG_Reg | P1_XGB_LG | Partly Cloudy | 134.02 | 134.36 | -0.34 |
| W5_XGB_LG_Reg | P1_XGB_LG | Overcast (CSI<0.3) | 146.06 | 146.39 | -0.33 |
| W6_CB_LG_Reg | P2_CB_LG | Clear (CSI≥0.7) | 140.26 | 140.03 | +0.23 |
| W6_CB_LG_Reg | P2_CB_LG | Partly Cloudy | 134.52 | 134.44 | +0.08 |
| W6_CB_LG_Reg | P2_CB_LG | Overcast (CSI<0.3) | 145.94 | 146.02 | -0.08 |
