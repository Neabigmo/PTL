# ICLR 2027 execution gates

These gates protect against the concrete failure mode of starting expensive
experiments before data identity, output contracts and leakage boundaries are
stable. They are review checkpoints, not claims that a result is good merely
because a command completed.

| Gate | Scope | Status | Pass condition |
|---|---|---|---|
| G0 | Git/bootstrap | PASS | One Git root, safe ignore policy, v1 provenance, portable package skeleton |
| G1 | Data registry | PASS | Dataset/environment/biological-instance IDs and External-B rubric are explicit |
| G2 | Predictor contract | PARTIAL | Simple and linear pilot outputs share a contract; official GEARS is explicitly blocked pending its supported environment |
| G3 | Uncertainty contract | PASS | Three-seed ensemble UQ, predictor-relative normalization and legacy confidence separation are traceable |
| G4 | PTL pilot | PASS (provisional) | Raw UQ → scalar calibration → RF/GBDT → PTL-Context is leakage-safe on a small replay surface |
| G5 | Predictor expansion | PENDING | CPA/scGPT/PRESCRIBE are isolated and added only where scientifically supported |
| G6 | Full experiments | PENDING | RQ1-RQ4, grouped cross-fitting, macro-averaging and robustness are reproducible |
| G7 | Paper | PENDING | ICLR paper claims, figures, tables and citations trace to released artifacts |

G0 must pass before new data reprocessing or scientific experiments begin.

G2 remains partial because the official GEARS package is only importable through
the project’s mixed `pytorch-clean` plus `sw_mgli` environment and its v2 data
adapter has not yet been validated; the official path is contract-only and has
no self-loop or zero-vector fallback. G4 is provisional because the pilot uses existing three-seed legacy
baseline arrays, the K562 essential surface rather than the selected GWPS
surface, and a strict cosine-fidelity pilot outcome. See
`docs/G1_G4_EXECUTION_REPORT.md` for scope and next gates.
