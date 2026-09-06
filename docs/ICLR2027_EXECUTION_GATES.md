# ICLR 2027 execution gates

These gates protect against the concrete failure mode of starting expensive
experiments before data identity, output contracts and leakage boundaries are
stable. They are review checkpoints, not claims that a result is good merely
because a command completed.

| Gate | Scope | Status | Pass condition |
|---|---|---|---|
| G0 | Git/bootstrap | PASS | One Git root, safe ignore policy, v1 provenance, portable package skeleton |
| G1 | Data registry | PASS | Dataset/environment/biological-instance IDs and External-B rubric are explicit |
| G2 | Predictor contract | PASS | Simple, linear, and official GEARS pilot outputs share a validated contract; final learned-model performance remains a later experiment |
| G3 | Uncertainty contract | PASS (engineering) | Aligned 2–3-member ensemble UQ, train-only predictor-relative normalization and legacy confidence separation are traceable |
| G4 | PTL pilot | PASS (provisional) | Random-anchor target, train-derived support/novelty, grouped cross-fitting and environment-first paired bootstrap are reproducible on the replay surface |
| G5 | Predictor expansion | PENDING | CPA/scGPT/PRESCRIBE are isolated and added only where scientifically supported |
| G6 | Full experiments | PENDING | RQ1-RQ4, grouped cross-fitting, macro-averaging and robustness are reproducible |
| G7 | Paper | PENDING | ICLR paper claims, figures, tables and citations trace to released artifacts |

G0 must pass before new data reprocessing or scientific experiments begin.

G2 passes at predictor-contract scope after the official GEARS adapter completed
the audited condition-holdout pilot: absolute GEARS predictions are converted
to matched-reference deltas, train/validation/test perturbation conditions are
disjoint, the gene panel is frozen per dataset, native uncertainty direction is
explicit, and official graph resources are used without self-loop or zero-vector
fallbacks. Final learned-model performance is intentionally outside this gate.
G4 is provisional because the PTL replay uses an anchor-relative
cosine-fidelity pilot outcome and the K562 essential surface, while K562 GWPS is
kept as a separate GEARS pilot surface. See
`docs/G1_G4_EXECUTION_REPORT.md` for scope and next gates.
