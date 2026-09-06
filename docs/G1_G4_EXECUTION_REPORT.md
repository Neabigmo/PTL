# G1-G4 execution report

Date: 2026-09-06
Workspace: `H:\2026try\9.5PTL`
Pilot ID: `ptl_v2_g4_baseline_replay_2026-09-06`

## Outcome

The reviewed PTL-v2 chain is implemented and locally reproducible. G1 passes,
G2 remains partial because the official GEARS adapter is not validated, G3 is
an engineering pass, and G4 passes provisionally on a baseline replay surface.
No raw or processed matrices were modified.

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

The pilot replays global mean, perturbation-matched mean and Ridge outputs.
Seed artifacts are aligned by `signature_id`; each signature retains only the
available 2–3 seed members, with the actual membership recorded in provenance.
Training-derived support/novelty uses the intersection of all three seed
training manifests, so a target held out by any member cannot become a support
feature through another member.
The official GEARS contract is present in configuration. The official package
imports only in the project’s mixed `pytorch-clean` plus `sw_mgli` environment;
the v2 data adapter has not yet been validated against that stack. Therefore no
GEARS headline result was fabricated and no legacy fallback was used. Strong
linear and CPA/scGPT/PRESCRIBE remain explicitly scoped for later expansion.

## G3 uncertainty

Ensemble uncertainty is computed from aligned 2–3 prediction members using gene
variance, median gene variance, top-effect variance, cosine disagreement and
effect-norm variance. Predictor-relative mid-rank UQ normalization is fitted
inside each biological-instance outer fold using training rows only; degenerate
references are explicitly flagged. Legacy v1 confidence is retained only as an
audit field and is not used as the v2 headline UQ.

## G4 pilot scope and leakage boundary

The pilot contains 23,226 prediction rows, 6,884 signature-level biological
instances, three environments, three split families and five grouped folds.
The selected K562 GWPS dataset is the
planned primary surface, but compatible local prediction arrays currently
exist only for the Replogle K562 essential legacy surface; the pilot labels it
accordingly. Reliability models are cross-fitted by
`biological_instance_id`; deployment features exclude outcomes, target labels,
split-family fields and other answer-key columns.

The pilot writes separate prediction, deployment-feature, outcome and fold
tables, plus RQ1, G4 comparison, figure-source and paired hierarchical
bootstrap-CI tables. The bootstrap resamples environments first and biological
instances second, with paired method draws and explicit method-minus-raw-UQ
delta intervals. The transportability target is
`I[fidelity >= 0.8 * median(random-anchor fidelity)]`; τ sensitivity for 0.6,
0.7, 0.8 and 0.9 is tracked separately.

## Provisional pilot readout

Lower AURC is better. The values below are environment-first macro averages;
they are not final paper claims.

| Method | AURC | Brier | Log loss |
|---|---:|---:|---:|
| raw_normalized_uq | 0.3790 | 0.3644 | 1.2457 |
| platt_logistic | 0.3785 | 0.2777 | 0.7755 |
| isotonic | 0.3739 | 0.2801 | 0.7988 |
| ptl_context | 0.3619 | 0.2206 | 0.6555 |
| ptl_full | 0.3599 | 0.2171 | 0.6511 |
| ptl_no_id | 0.3591 | 0.2196 | 0.6523 |
| gbdt_uncalibrated | 0.3505 | 0.1938 | 0.5885 |
| ptl_rf | 0.3450 | 0.1737 | 0.5286 |
| support_novelty_only | 0.3627 | 0.2552 | 0.7193 |

PTL-Context improves macro AURC over raw UQ and both scalar calibrators in
this replay, but it is not the best pilot method; tree-based models are
stronger on this surface. `ptl_full` uses the same contextual architecture with
explicit environment/predictor IDs, while `ptl_no_id` removes the predictor
family/context block and uses the strict P+S+N control, so the ablations are
now distinct. These are diagnostic pilot
results, not final paper claims.

## Verification

- PTL-v2 targeted tests: 9 passed in the default F environment.
- The full default-environment suite is blocked during collection because
  `anndata` is not installed there.
- The full `pytorch-clean` suite ran 45 tests and had 3 environment/history
  failures: one historical raw-path assertion and two GEARS imports that need
  the mixed package path. With `gears` exposed through the documented mixed
  environment, the five GEARS adapter tests pass. These failures do not touch
  the active PTL-v2 pilot code.

## Reproducibility and next action

Primary commands:

```text
F:\\anaconda3\\python.exe scripts\\audit_dataset_metadata.py --root H:\\2026try\\9.5PTL --output artifacts\\manifests\\dataset_audit.csv
F:\\anaconda3\\python.exe scripts\\build_evaluation_gene_space.py
F:\\anaconda3\\python.exe scripts\\run_ptl_v2_pilot.py --root H:\\2026try\\9.5PTL
F:\\anaconda3\\python.exe -m pytest -q tests\\test_path_portability.py tests\\test_ptl_v2_contracts.py tests\\test_ptl_v2_metrics.py
```

The pilot now parallelizes the five source-disjoint outer folds in separate
processes and merges them by fold in the parent process. The default is five
workers; set `PTL_V2_WORKERS=1` for a serial reproducibility check or choose a
smaller value when CPU or memory is constrained. This is process-level
parallelism within the existing protocol, not a bypass of operating-system or
account limits; data, model settings, splits and metrics are unchanged.

Before scaling, validate a clean official GEARS data adapter without the legacy
self-loop/zero-vector fallbacks, produce compatible GWPS arrays, then rerun the
same contracts and grouped folds. Only after that should G5 predictor expansion
and G6 full RQ1-RQ4 experiments begin.
