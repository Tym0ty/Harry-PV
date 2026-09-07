# Harry-PV Final Package

Final thesis package for PV-focused rolling MPC and CCOR-MPC scheduling for a
campus PV-BESS system.

This branch is a curated GitHub-ready snapshot. It intentionally excludes the
large exploratory workspace, virtual environments, raw scenario pools, old
handover archives, and regenerated intermediate outputs from the original local
project.

## Final Study Line

The final method line focuses on PV uncertainty and PV-BESS scheduling under a
deterministic given campus load profile.

Final optimization-facing net load:

```text
netload_s(t) = load_given(t) - PV_s(t)
```

Final system settings:

- PV capacity: `2687 kWp`
- Performance ratio for GHI-to-PV conversion: `0.80`
- Contract capacity: `3232 kW`
- Test period: `2024-11-01` to `2025-10-31`
- Rolling horizon: `H24`, execute first step only
- Probabilistic scenario reduction: low-PV-safe tail-aware K-medoids, `K=5`

The final report is:

`new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/FINAL_THESIS_METHOD_AND_RESULTS_REPORT_PVFOCUS.md`

## Final Cases

| Final case | Case ID | Method | Forecast | Scenario | Load |
|---|---|---|---|---|---|
| `MPC_DET` | `MPC_DET_PVFOCUS_BASE` | Standard rolling MPC | PV q50 point | none | base deterministic |
| `MPC_PROB` | `MPC_PROB_PVFOCUS_LOWPV_SAFE_K5` | Stochastic rolling MPC | PV probabilistic | low-PV-safe TKM K5 | base deterministic |
| `CCOR_DET` | `CCOR_V2_DET_SHIELD_m150_PVFOCUS_BASE` | CCOR-v2 rolling MPC | PV q50 point | none | base deterministic |
| `CCOR_PROB` | `CCOR_V2_PROB_SHIELD_m150_PVFOCUS_LOWPV_SAFE_K5` | CCOR-v2 stochastic rolling MPC | PV probabilistic | low-PV-safe TKM K5 | base deterministic |

Excluded from the final mainline: DA main cases, LOAD-QN, random load-error
mainline, old S1-S6 scenario inputs, old invalid NTUST `Solar_kWh` PV truth,
and LitErr random load-error cases.

## Final Results

Annual realized-basis cost summary:

| Case | Total cost (M NTD) | TOU (M NTD) | OC/DCT (M NTD) | Degradation (M NTD) | Max monthly peak (kW) | OC hours |
|---|---:|---:|---:|---:|---:|---:|
| `MPC_DET` | 106.486 | 75.557 | 7.239 | 2.147 | 5133.697 | 1021 |
| `MPC_PROB` | 104.442 | 75.877 | 4.974 | 2.048 | 4810.122 | 1007 |
| `CCOR_DET` | 105.768 | 75.642 | 6.452 | 2.126 | 4977.686 | 970 |
| `CCOR_PROB` | 103.503 | 76.217 | 3.773 | 1.971 | 4695.421 | 850 |

Key deltas:

| Comparison | Total cost change (M NTD) | OC/DCT change (M NTD) | Peak change (kW) | OC-hour change |
|---|---:|---:|---:|---:|
| `MPC_PROB` vs `MPC_DET` | -2.043 | -2.265 | -323.575 | -14 |
| `CCOR_PROB` vs `CCOR_DET` | -2.265 | -2.679 | -282.265 | -120 |
| `CCOR_DET` vs `MPC_DET` | -0.717 | -0.787 | -156.011 | -51 |
| `CCOR_PROB` vs `MPC_PROB` | -0.939 | -1.201 | -114.701 | -157 |

Interpretation: probabilistic PV scenarios improve both standard MPC and
CCOR-v2. CCOR-v2 further improves results mainly by reducing OC/DCT exposure
and monthly peaks.

## Where to Start

- `FINAL_FILE_SELECTION.md`: explains what was considered final, reference, or excluded.
- `PACKAGE_README.md`: records how this curated package was generated.
- `PACKAGE_MANIFEST.csv`: file-level package inventory.
- `REPRODUCIBILITY.md`: practical reproduction levels and required inputs.
- `PROJECT_STRUCTURE.md`: directory map for this curated package.
- `new_pipeline/data/output/experiments/final_thesis_method_result_report_pvfocus/`: final consolidated report and summary tables.

## Installation

Create a Python environment and install the listed dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The optimization scripts require a working Gurobi installation and license for
full MILP reruns.

## Reproducibility Scope

This package is optimized for inspection, citation, and compact reproduction of
final reported results. It includes selected final evidence artifacts, summary
tables, thesis figures/tables, and the final method code.

It does not include the full 12 GB exploratory `new_pipeline/data/output`
workspace from the local project.

## License and Citation

Code is released under the MIT License unless a file states otherwise. See
`LICENSE`.

For citation metadata, see `CITATION.cff`.
