# ICLR 2027 execution gates

These gates protect against the concrete failure mode of starting expensive
experiments before data identity, output contracts and leakage boundaries are
stable. They are review checkpoints, not claims that a result is good merely
because a command completed.

| Gate | Scope | Status | Pass condition |
|---|---|---|---|
| G0 | Git/bootstrap | IN_PROGRESS | One Git root, safe ignore policy, v1 provenance, portable package skeleton |
| G1 | Data registry | PENDING | Dataset/environment/biological-instance IDs and External-B rubric are explicit |
| G2 | Predictor contract | PENDING | Simple, strong linear and official GEARS outputs share a prediction contract |
| G3 | Uncertainty contract | PENDING | Native, ensemble and normalized UQ are separated and traceable |
| G4 | PTL pilot | PENDING | Raw UQ → scalar calibration → RF/GBDT → PTL-Context is leakage-safe on a small surface |
| G5 | Predictor expansion | PENDING | CPA/scGPT/PRESCRIBE are isolated and added only where scientifically supported |
| G6 | Full experiments | PENDING | RQ1-RQ4, grouped cross-fitting, macro-averaging and robustness are reproducible |
| G7 | Paper | PENDING | ICLR paper claims, figures, tables and citations trace to released artifacts |

G0 must pass before new data reprocessing or scientific experiments begin.
