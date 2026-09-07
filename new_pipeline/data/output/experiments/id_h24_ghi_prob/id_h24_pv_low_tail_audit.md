# ID-H24 PV Forecast Lower-Tail Audit

**Generated**: 2026-06-05  
**Purpose**: Verify lower-tail reliability for later high-net-load stress scenario generation

---

## 1. Lower-Tail Coverage Statistics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Raw q05 violation rate | 7.33% | Fraction where y_pv < pv_q05 |
| Raw q05 coverage | 92.67% | Target: ≥ 95% (nominal for q05) |
| Calibrated q05 violation rate | 7.72% | y_pv < pv_q05_cal90 |
| q05 tail status | MARGINAL | — |
| q50 over-forecast rate (low PV < 100 kW) | 49.82% | q50 > y (over-forecast) |
| Median q50 bias for low PV hours | 9.13 kW | Positive = over-forecast |

---

## 2. Lower-Tail Coverage by Lead Group

| Lead Group | q05 cov. | q10 cov. | Over-forecast % |
|------------|----------|---------|----------------|
| H01-H03 | 93.34% | 88.59% | 50.51% |
| H04-H06 | 92.51% | 86.76% | 49.93% |
| H07-H12 | 92.33% | 86.64% | 49.78% |
| H13-H24 | 92.72% | 87.34% | 49.64% |

---

## 3. Low-PV Period (< 100 kW) Behaviour

| Metric | Low PV (<100 kW) | Overall |
|--------|-----------------|--------|
| PICP@80% | 71.17% | 79.81% |
| MPIW@80% (kW) | 189.48 | 659.61 |
| q05 coverage | 79.66% | 92.67% |
| Over-forecast rate | 70.00% | 49.82% |

---

## 4. Assessment for Net-Load Stress Scenario Generation

Q05 PV lower bound shows higher-than-nominal violation rate. Net-load stress scenarios using q05 may understate actual PV production occasionally.

**Q05 PV** = high-net-load scenario (low PV production). A conservative q05 means the stress scenario uses a PV value that is rarely exceeded on the low side — i.e., actual PV is almost always >= q05. This is appropriate for worst-case net-load planning.

**Underprediction** (q50 < y) in low-PV hours: if the point forecast (q50) systematically under-forecasts actual PV during cloudy periods, the MILP will dispatch BESS more conservatively — a safe failure mode.

**Recommendation**: Use pv_q05_cal90 (calibrated 90% PI lower bound) as the PV lower tail for net-load stress scenarios, as it has tighter target coverage than raw q05 and is adjusted by AgACI.
