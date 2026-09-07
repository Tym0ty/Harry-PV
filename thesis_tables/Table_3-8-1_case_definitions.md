## Case definitions for scheduling horizon comparison study.

| Case ID           | Description                  | Horizon Rule        | Forecast Input       | Purpose                                                |
|:------------------|:-----------------------------|:--------------------|:---------------------|:-------------------------------------------------------|
| DA-Det (M1)       | DA-only deterministic        | None (DA-only)      | Det. point forecast  | Conceptual lower bound; isolates intraday update value |
| DA-Prob (M2)      | DA probabilistic master      | None (DA-only)      | Prob. DA forecast    | Frozen DA master plan; reference for M8 baseline       |
| MPC-Det-H24 (M3)  | Rolling-H24 deterministic    | Fixed H=24          | Det. point forecast  | Deterministic rolling horizon; upper bound on OC       |
| MPC-Prob-H24 (M5) | Rolling-H24 probabilistic    | Fixed H=24          | Prob. DA+ID forecast | Probabilistic rolling MPC; no-arbitration baseline     |
| M8-Det-H24        | No-regret arbitration (Det)  | H=24, F1–F4 filters | Det. point forecast  | Arbitration with deterministic candidate               |
| M8-FY2-H24        | No-regret arbitration (Prob) | H=24, F1–F4 filters | Prob. DA+ID forecast | Proposed method; probabilistic candidate               |

*Note: All cases share identical BESS sizing, CC, and tariff parameters. Forecast inputs are generated from the same pre-specified pipeline (v4 XGBoost + AgACI for ID; da\_v2 for DA). M8 cases employ the no-regret arbitration rule with safety filters F1–F4.*