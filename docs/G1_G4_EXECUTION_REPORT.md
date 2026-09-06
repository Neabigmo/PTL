# G1-G4 execution report

Date: 2026-09-06
Workspace: `H:\2026try\9.5PTL`
Pilot ID: `ptl_v2_g4_baseline_replay_2026-09-06`

## Outcome

The reviewed PTL-v2 chain is implemented and locally reproducible. G1 passes,
G2 passes at predictor-contract scope after independent review of the completed
official GEARS condition-holdout scientific pilot, G3 is an engineering pass,
and G4 passes provisionally on a baseline replay surface.
Raw inputs were not modified; the GWPS processed artifacts and pilot outputs
were generated under the project results/data directories.

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
imports in the project’s documented mixed `pytorch-clean` plus `sw_mgli`
environment; the v2 adapter completed a nine-run random condition-holdout
scientific pilot on the official Norman, Replogle K562 GWPS, and Replogle RPE1
surfaces.
The adapter uses official GEARS graph resources and native log-variance UQ,
with no self-loop or zero-vector graph fallback. Strong linear and
CPA/scGPT/PRESCRIBE remain explicitly scoped for later expansion.

The scientific pilot contains the nine random condition-holdout runs below
(three seeds per surface, one CPU epoch, and one frozen 4,096-gene panel per
dataset). The run matrix records `split_protocol=random_condition_holdout` and
`perturbation_overlap=zero`; these are not the familiar/random PTL split.
The split audit verifies zero train/validation/test condition overlap. These
are adapter-validation metrics, not final PTL-v2 claims.

| Surface | Seeds | Test non-control / control | Genes | MSE range |
|---|---:|---:|---:|---:|
| Norman | 0--2 | 346--363 / 0 | 4,096 | 0.011972--0.014306 |
| K562 GWPS | 0--2 | 16--26 / 0 | 4,096 | 0.021858--0.030393 |
| RPE1 | 0--2 | 233--418 / 0 | 4,096 | 0.020417--0.024318 |

Each run writes `test_predictions.npz`, `test_metadata.parquet`, `genes.txt`
and `run_metrics.json`; the prediction archive includes
`native_confidence_gears_exp_neg_mean_logvar` and
`native_uncertainty_gears_mean_logvar`, with higher uncertainty meaning lower
native confidence. The matched-reference delta conversion and all run-level
provenance are recorded in
`artifacts/manifests/gears_pilot_audit.csv`.

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
The selected K562 GWPS dataset is represented by the separate official GEARS
pilot surface; the PTL baseline replay continues to use its documented legacy
K562 surface. Reliability models are cross-fitted by
`biological_instance_id`; deployment features exclude outcomes, target labels,
split-family fields and other answer-key columns.

The pilot writes separate prediction, deployment-feature, outcome and fold
tables, plus RQ1, G4 comparison, figure-source and paired hierarchical
bootstrap-CI tables. The bootstrap resamples environments first and biological
instances second, with paired method draws and explicit method-minus-raw-UQ
delta intervals. The transportability target is
`I[fidelity >= 0.8 * median(random-anchor fidelity)]`; τ sensitivity for 0.6,
0.7, 0.8 and 0.9 is tracked separately.

Iteration 3 freezes the familiar/random anchor in
`artifacts/manifests/reliability_anchor_registry.csv` before grouped PTL
cross-fitting. Context categories are one-hot encoded from each training fold
only; unseen categories map to an all-zero block. The P block uses a
train-only PCA response-manifold nearest-neighbor distance rather than the
previous `1 - concentration` proxy. The continuous-risk association metric is
tie-aware Spearman correlation under the name
`spearman_predicted_risk_realized_risk`.

## Provisional pilot readout

Lower AURC is better. The values below are environment-first macro averages;
they are not final paper claims.

| Method | AURC | Brier | Log loss |
|---|---:|---:|---:|
| raw_normalized_uq | 0.3790 | 0.3644 | 1.2457 |
| platt_logistic | 0.3785 | 0.2777 | 0.7755 |
| isotonic | 0.3739 | 0.2801 | 0.7988 |
| ptl_context | 0.3654 | 0.2258 | 0.6622 |
| ptl_no_context | 0.3633 | 0.2238 | 0.6581 |
| ptl_full_id | 0.3625 | 0.2102 | 0.6227 |
| gbdt_uncalibrated | 0.3514 | 0.1914 | 0.5790 |
| gbdt_full_id | 0.3545 | 0.1924 | 0.5818 |
| ptl_rf | 0.3480 | 0.1754 | 0.5321 |
| ptl_rf_full_id | 0.3480 | 0.1698 | 0.5184 |
| support_novelty_only | 0.3656 | 0.2675 | 0.7467 |

PTL-Context improves macro AURC over raw UQ and both scalar calibrators in
this replay, but it is not the best pilot method; tree-based models are
stronger on this surface. `ptl_full_id` and the `_full_id` tree variants are
identity-rich diagnostic upper bounds; `ptl_rf` and `gbdt_uncalibrated` are
identity-free headline tree methods. `ptl_no_context` removes biological
context while retaining prediction, support and novelty blocks. These are
diagnostic pilot results, not final paper claims.

## Verification

- PTL-v2 targeted tests: 9 passed in the default F environment; GEARS adapter
  tests: 8 passed in the documented mixed environment.
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
F:\\anaconda3\\python.exe scripts\\build_gears_scientific_pilot_splits.py
E:\\anaconda3\\envs\\sw_mgli\\python.exe src\\models\\run_perturbation_model.py build-run-matrix --split-audit results\\tables\\gears_scientific_pilot_split_audit.csv --output results\\tables\\perturbation_model_run_matrix.csv --gene-space-policy fixed_dataset_train_variance_4096
E:\\anaconda3\\envs\\sw_mgli\\python.exe src\\models\\run_perturbation_model.py run-all --model gears --run-matrix results\\tables\\perturbation_model_run_matrix.csv --registry results\\tables\\experiment_registry.csv --epochs 1 --max-genes 4096 --device cpu --force
F:\\anaconda3\\python.exe scripts\\build_gears_pilot_audit.py
```

The pilot now parallelizes the five biological-instance-disjoint grouped
cross-fitting folds in separate processes and merges them by fold in the
parent process. The default is five
workers; set `PTL_V2_WORKERS=1` for a serial reproducibility check or choose a
smaller value when CPU or memory is constrained. This is process-level
parallelism within the existing protocol, not a bypass of operating-system or
account limits; data, model settings, splits and metrics are unchanged.

The current next action is independent C2C review of the completed GEARS audit
before any G2 status change or commit. The next scientific step after review is
G5 predictor expansion and then G6 full RQ1--RQ4 experiments. The current GEARS
results remain an adapter-validation pilot: full PTL-v2 claims still require
the planned learned-model roster, native UQ comparison, and final multi-seed
training budget.
