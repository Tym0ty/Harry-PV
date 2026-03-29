# BH Project Pipeline — Technical Explanation
*For: 黃博鴻 (Harry), NTUST Master's Thesis*
*Thesis: "Quantifying the Decision Value of Probabilistic PV Information for Annual Fixed Design of Contract Capacity and BESS under RE20"*

---

## What the Pipeline Does

The pipeline answers one central research question:

> **Does using a probabilistic PV forecast (instead of a simple point forecast) lead to a better annual sizing decision for Contract Capacity (CC) and a BESS — measured by what the electricity bill actually turns out to be?**

To answer this, the pipeline runs the same sizing optimization under five different information assumptions (C0–C3 + C_PFI), then evaluates each sizing decision against the truth (realized annual PV and load), and compares the results.

---

## The Three-Layer Architecture

```
Calibrated PV Forecast Artifacts
(CQR-calibrated Gaussian copula scenarios)
             │
    ┌────────▼──────────────────────────────────────────────┐
    │  BRIDGE LAYER                                          │
    │  bridge_full_year.py + repday_builder.py               │
    │  + scenario_reduction.py                               │
    │                                                        │
    │  Transforms: forecast → two MILP input families        │
    │    • Layer A: Representative-day design packages       │
    │    • Layer B: Rolling day-ahead packages (full year)   │
    └────────────────────┬──────────────────────────────────┘
                         │
          ┌──────────────┴───────────────────┐
          │                                   │
   ┌──────▼──────┐                   ┌────────▼────────┐
   │  MILP        │                   │  MILP           │
   │  Layer A     │                   │  Layer B        │
   │  milp_       │   Design:         │  milp_          │
   │  layer_a.py  │  (CC*, P_B*, E_B*)│  layer_b.py     │
   │              │──────────────────►│                 │
   │  Annual      │                   │  Sequential     │
   │  fixed-design│                   │  daily solve    │
   │  selection   │                   │  + Replay on    │
   └─────────────┘                   │  realized truth │
                                      └─────────────────┘
```

---

## The Five Formal Cases

Each case represents a different information assumption at the day-ahead gate time:

| Case | PV Input | Load Input | Research Purpose |
|------|----------|------------|-----------------|
| **C0** | Deterministic (Q50 point forecast) | Deterministic baseline | Baseline — current industry standard |
| **C1** | Probabilistic scenarios (5 reduced) | Deterministic baseline | **Main comparison**: does prob PV help? |
| **C2** | Deterministic (Q50) | Load forecast-error scenarios | Measure effect of load uncertainty under det PV |
| **C3** | Probabilistic PV + load scenarios | Joint PV-load scenarios | Full stochastic case |
| **C_PFI** | Perfect realized next-day PV | Perfect realized next-day load | **Upper bound**: if we had a crystal ball |

The thesis research question is answered by comparing C0 vs. C1 in the replay results. C_PFI gives the theoretical upper bound.

---

## The Load Uncertainty Model — What Changed and Why

### Old model (WRONG — now removed)
The old code added a fixed deterministic uplift to load:
- Billing hours: load × 1.05 (+5%)
- Non-billing hours: load × 1.02 (+2%)

**Why this is wrong**: This is not uncertainty — it is a systematic bias. It assumes load is always higher than forecast, which is not what the thesis models. It would make C2/C3 look like "high-load stress" cases rather than genuine forecast-uncertainty cases.

### New model (CORRECT — as in thesis §3.7)
```
L_scen(n, t, ω) = L̂(n, t) × (1 + ε_ω)
```

Where:
- `L̂(n, t)` = deterministic load forecast for day n, hour t
- `ε_ω` ~ N(0, σ²_season) — drawn once per (day, scenario ω)
- `σ_season` = seasonal standard deviation derived from Pieter's historical load forecast MAPE

| Season | Source MAPE | Standard Deviation σ |
|--------|------------|----------------------|
| Summer (Jun–Aug) | 11.45% | **14.35%** |
| Fall (Sep–Nov) | 14.66% | **18.37%** |
| Winter (Dec–Feb) | 12.18% | **15.27%** |
| Spring (Mar–May) | 12.29% | **15.40%** |

*(σ = MAPE × √(π/2), derived from the half-normal identity)*

**Key property**: The model is **zero-mean** — scenarios are equally likely to be above or below the forecast. This correctly represents forecast uncertainty, not a systematic bias.

---

## The Two-Layer MILP — What Changed and Why

### Old architecture (now deprecated)
One giant optimization: send all 365 days × 24 hours = 8,760 hours to Gurobi at once. This is computationally expensive and forces a single solution to simultaneously decide:
1. What size CC, P_B, E_B to install (annual design decision)
2. How to operate the battery every hour (operational decision)

### New architecture (Layer A + Layer B)

**Layer A — Annual Fixed Design** (`milp_layer_a.py`)

Instead of 365 days, uses a compressed set of **representative days** (e.g., 30–50 days that cover the diversity of the full year). For each candidate design `(CC, P_B, E_B)` from a pre-defined grid `X_grid`, evaluates:

```
J_A(x) = AEC_inv(x) + C_energy^A + C_deg^A + C_basic^A + C_over^A + C_TREC^A
```

Where:
- `AEC_inv` = annualized BESS investment cost (via Capital Recovery Factor)
- `C_energy^A` = annual energy cost from TOU pricing
- `C_deg^A` = battery degradation cost (PWL model, 5 segments)
- `C_basic^A` = monthly basic charge × monthly max-demand proxy
- `C_over^A` = over-contract penalty (×2 within 10%, ×3 beyond 10%)
- `C_TREC^A` = T-REC shortfall cost for RE20 compliance

Selects `x* = argmin J_A(x)` — the design that minimizes total annual cost in the representative-day world.

**Critical subtlety**: The monthly maximum demand `D_m^A` is NOT computed as a weighted average of daily peaks. It is reconstructed by mapping each representative day back to its source calendar months and finding the maximum peak within each month. This is essential for correctly modeling the over-contract penalty structure.

**Layer B — Sequential Day-Ahead Operation** (`milp_layer_b.py`)

Takes the fixed design `(CC*, P_B*, E_B*)` from Layer A and runs 365 individual daily 24-hour solves — one per calendar day, in chronological order. Each solve:
- Uses only information available at the day-ahead gate (no future information leakage)
- Carries state across days: SOC, Green SOC, monthly max-demand accumulator, annual RE totals
- **Replay mode**: After solving, settles the day's planned schedule against realized PV and load

**Why two layers?** The design decision (CC/BESS size) is made once a year; it must be robust to a full year of conditions. The operational decision (daily battery schedule) adapts to each day's forecast. Separating them correctly models the actual decision timeline.

---

## Representative Day Builder

The `repday_builder.py` module compresses 365 calendar days down to ~30–50 representative days:

### Step 1 — Risk-Day Retention
Before clustering, identify and retain "risk days" individually:
- **Peak-load risk days**: Days that drive the monthly maximum demand (these determine the basic charge)
- **High net-load stress days**: High PV curtailment risk or grid import stress
- **Severely reduced PV days**: Important for battery dispatch decisions

These days cannot be merged into a cluster average without losing the critical peak information.

### Step 2 — Body-Day Clustering (k-medoids)
Remaining "body days" are clustered by their daily characteristics:
- Total daily load, total daily PV, net-load proxy, ramp indicators
- Uses k-medoids (not k-means) — the representative is an actual calendar day, not a synthetic average
- Each cluster maps to its medoid day as the representative

### Step 3 — Calendar Mapping + Weight Closure
Every calendar day → exactly one `repday_id`. The weight of each representative day = number of calendar days it represents. Hard check:

```
sum(annual_weight_days) == total case-year days  # ← MUST hold exactly
```

This ensures the annual cost computed over representative days correctly scales to a full year.

---

## Scenario Reduction

For probabilistic cases, the number of scenarios is reduced to keep the MILP tractable:

| Case | Reduction Space | Method |
|------|----------------|--------|
| C0, C_PFI | None (deterministic) | No reduction needed |
| C1 | PV scenario space only | k-medoids on PV trajectory profiles |
| C2 | Load scenario space only | k-medoids on load scenario profiles |
| C3 | Joint (PV, load) space | k-medoids on combined feature vectors |

After reduction: `sum(probability_pi) == 1.0` for each day. This is a hard acceptance check.

---

## Truth Isolation — A Critical Design Constraint

The pipeline enforces a strict separation between:
- **Solve world**: Uses forecast-based inputs (C0–C3) or perfect-forecast inputs (C_PFI)
- **Replay world**: Uses realized actual PV and load

Rules:
1. `pv_realized_kw` and `load_realized_kw` columns must **never** appear in any C0–C3 solve package
2. C_PFI's perfect-forecast inputs are pre-packaged by the bridge separately — they are "perfect next-day forecasts" in the information sense, not realized actuals used in hindsight
3. Replay truth package (`full_year_replay_truth_package.parquet`) is only read during replay mode

Violation of these rules is a **hard acceptance failure** — the pipeline aborts immediately.

---

## The C_PFI Case — What It Means

"C_PFI" = **Perfect Forecast Information** case.

This simulates: *"What if your day-ahead PV and load forecasts were always exactly correct?"*

It is NOT about cheating with actual realized data during the solve. It is about using the same day-ahead framework as C0–C3, but with the realized values pre-loaded as if you had predicted them perfectly.

The C_PFI result gives the **theoretical upper bound** on performance achievable within the two-layer framework. If the difference between C1 (prob) and C_PFI is small, it means probabilistic PV gets you most of the way to the theoretical optimum.

---

## Frozen Parameters (from spec §5)

These parameters are fixed for the formal mainline — do not change for sensitivity analysis without explicit documentation.

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| η_ch | 0.95 | — | BESS_001 |
| η_dis | 0.95 | — | BESS_002 |
| SOC_min | 0.10 | — | BESS_003 |
| SOC_max | 0.90 | — | BESS_004 |
| SOC_init | 0.50 | — | BESS_005 |
| C_B_P | 11,944 | NTD/kW | BESS_006 |
| C_B_E | 7,738 | NTD/kWh | BESS_007 |
| c_basic,s | 223.6 | NTD/kW-month | CP_001 |
| c_basic,ns | 166.9 | NTD/kW-month | CP_002 |
| κ (kappa) | 1.0035 | — | CP_006 |
| m_over,≤10% | 2 | × | CP_004 |
| m_over,>10% | 3 | × | CP_005 |
| RE_target | 0.20 | — | SYS_004 |
| c_TREC | 4.63 | NTD/kWh | RE_002 |
| r | 0.05 | — | FIN_001 |
| N_B | 15 | years | FIN_002 |
| CRF_BESS | 0.09634 | — | FIN_003 |
| N_cyc | 6,000 | cycles | DEG_001 |
| DoD_ref | 0.80 | — | DEG_002 |
| λ_base | 1.612 | NTD/kWh | DEG_004 |
| b_k | [0, 0.1, 0.3, 0.6, 0.8] | — | DEG_005 |
| μ_k | [0.6, 1.0, 1.6, 2.4] | — | DEG_006 |
| λ_k | [0.97, 1.61, 2.58, 3.87] | NTD/kWh | DEG_007 |
| PV_UB | 2,687 | kW | PV_001 |

---

## Output Files (per case)

For each CASE ∈ {C0, C1, C2, C3, C_PFI}:

| File | What it contains |
|------|-----------------|
| `design_results_CASE.json` | Selected (CC*, P_B*, E_B*) + J_A cost decomposition |
| `replay_monthly_bill_CASE.csv` | Month-by-month actual electricity bill |
| `replay_summary_CASE.json` | Annual replay totals, RE20 compliance |
| `rolling_state_trace_CASE.parquet` | Daily SOC, Green SOC, demand, RE accumulators |

Aggregate comparison files:
- `design_results_master.csv` — all 5 cases side by side
- `replay_summary_master.csv` — replay results for thesis tables
- `thesis_figures_bundle/` — 9 required figures

---

## Design-to-Replay Gap

Because the Layer A optimization uses compressed representative days (not the full year), the actual replay cost will differ from J_A. These gaps are computed and reported:

```
Gap_cost   = replay_annual_cost    - J_A(x*)        ← how optimistic was Layer A?
Gap_over   = replay_overcontract   - overcontract_A  ← did we underestimate penalties?
Gap_oper   = replay_annual_cost    - solve_openloop  ← operational sub-optimality
```

Understanding these gaps is part of validating the two-layer methodology.

---

## File Naming Convention

| Prefix | Family | Example |
|--------|--------|---------|
| `rolling_da_input_` | Layer B rolling packages | `rolling_da_input_pvdet_loaddet.parquet` |
| `repday_input_` | Layer A design packages | `repday_input_pvprob_loaddet.parquet` |
| `repday_weights` / `repdays_metadata` / `calendar_to_repday_map` | Rep-day metadata | — |
| `full_year_replay_truth_package` | Replay truth | — |
| `caseyear_calendar_manifest` | Day calendar | — |
| `load_uncertainty_manifest` | Load σ params | — |

**Legacy alias note**: Old filenames (`full_year_milp_ingest_pvdet_loaddet.parquet`, `load_perturbation_manifest.parquet`) are recorded in `bridge_run_metadata.json` as aliases but are no longer written.

---

*Spec references: F0329Bridge_Spec.pdf (Batch 8), F0329MILP_Spec.pdf (Batch 17)*
*Confirmed aligned with thesis methodology (§3.7 scenario reduction, §3.8 bridge layer, §3.9 two-layer planning)*
