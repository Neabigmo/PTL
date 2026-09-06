# G1-G4 execution report

Date: 2026-09-06
Workspace: `H:\2026try\9.5PTL`
Pilot ID: `ptl_v2_g4_baseline_replay_2026-09-06`

## Outcome

The first PTL-v2 chain is implemented and locally reproducible. G1 and G3
pass; G2 is partial because official GEARS dependencies are unavailable; G4
passes provisionally on a small replay surface. No raw or processed matrices
were modified.

## G1 registry and provenance

- The metadata-only audit inspected all 54 raw H5AD files without loading `X`.
- Stable `environment_id`, `biological_instance_id` and `prediction_id`
  functions, a path resolver, feature-contract validation and the provisional
  6,257-gene legacy evaluation panel are implemented.
- Wessels/Satija 2023 was frozen as External-B using a metadata-only rubric
  before PTL-v2 performance evaluation; the decision is recorded in
  `docs/EXTERNAL_B_FREEZE.md`.
- Adamson 10X001, 10X005 and 10X010 are retained as separate accessions. They
  are not silently treated as technical replicates; see `docs/ADAMSON_AUDIT.md`.

## G2 predictor contract

The pilot replays three independent seeds for global mean, perturbation-matched
mean and Ridge. Each output is connected to a prediction array, gene list,
split, predictor and ensemble-member declaration. The official GEARS contract
is present in configuration. The official package imports only in the project’s
mixed `pytorch-clean` plus `sw_mgli` environment; the v2 data adapter has not
yet been validated against that stack. Therefore no GEARS headline result was
fabricated and no legacy fallback was used. Strong linear and CPA/scGPT/PRESCRIBE
remain explicitly scoped for later expansion.

## G3 uncertainty

Ensemble uncertainty is computed from the three prediction members using gene
variance, median gene variance, top-effect variance, cosine disagreement and
effect-norm variance. Predictor-relative quantiles are computed separately
within each pilot predictor. Legacy v1 confidence is retained only as an audit
field and is not used as the v2 headline UQ.

## G4 pilot scope and leakage boundary

The pilot contains 20,652 prediction rows, 866 biological instances, three
environments and five grouped folds. The selected K562 GWPS dataset is the
planned primary surface, but compatible local prediction arrays currently
exist only for the Replogle K562 essential legacy surface; the pilot labels it
accordingly. Reliability models are cross-fitted by
`biological_instance_id`; deployment features exclude outcomes, target labels,
split-family fields and other answer-key columns.

The pilot writes separate prediction, deployment-feature, outcome and fold
tables, plus RQ1, G4 comparison, figure-source and paired hierarchical
bootstrap-CI tables. The bootstrap uses the same biological-instance draws for
all methods and macro-averages predictor/environment strata.

## Provisional pilot readout

Lower AURC is better. The values below are environment-first macro averages;
they are not final paper claims.

| Method | AURC | Brier | Log loss |
|---|---:|---:|---:|
| raw_normalized_uq | 0.4313 | 0.2321 | 0.8880 |
| platt_logistic | 0.4302 | 0.0431 | 0.2129 |
| isotonic | 0.4305 | 0.0457 | 0.2368 |
| ptl_context | 0.4266 | 0.0425 | 0.1985 |
| ptl_rf | 0.4206 | 0.0427 | 0.2059 |
| calibrated_gbdt | 0.4198 | 0.0399 | 0.1793 |
| support_novelty_only | 0.4184 | 0.0424 | 0.1868 |

PTL-Context improves macro AURC over raw UQ and both scalar calibrators in
this replay, but it is not the best pilot method; support/novelty-only and
tree-based models are stronger on this surface. `ptl_no_id` is identical to
`ptl_context` in this pilot because both use the same context-only feature
block, so it is a robustness control rather than an independent architecture.
The high FTR values and strict cosine threshold also make this a diagnostic
pilot, not evidence for a final reliability claim.

## Reproducibility and next action

Primary commands:

```text
F:\\anaconda3\\python.exe scripts\\audit_dataset_metadata.py --root H:\\2026try\\9.5PTL --output artifacts\\manifests\\dataset_audit.csv
F:\\anaconda3\\python.exe scripts\\build_evaluation_gene_space.py
F:\\anaconda3\\python.exe scripts\\run_ptl_v2_pilot.py --root H:\\2026try\\9.5PTL
F:\\anaconda3\\python.exe -m pytest -q tests\\test_path_portability.py tests\\test_ptl_v2_contracts.py tests\\test_ptl_v2_metrics.py
```

Before scaling, validate a clean official GEARS data adapter without the legacy
self-loop/zero-vector fallbacks, produce compatible GWPS arrays, then rerun the
same contracts and grouped folds. Only after that should G5 predictor expansion
and G6 full RQ1-RQ4 experiments begin.
