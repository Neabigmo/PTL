# Codex accelerated execution packet: G1-G4 in one sprint

Date: 2026-09-06
Branch: `iclr2027-restructure`

## Why this sprint is accelerated

G0 is mostly complete locally, but the project is time constrained. Do not stop after every small subtask. Complete G1-G4 as one continuous engineering/scientific sprint, while preserving explicit checkpoints in logs and commits. The objective is to produce the first leakage-safe PTL-v2 pilot end-to-end before expanding to CPA/scGPT/PRESCRIBE.

## 0. Immediate G0 corrections before G1

1. The local branch currently has an unpushed commit and no upstream. Configure `origin/iclr2027-restructure`, reconcile with the remote branch, preserve the full remote `docs/ICLR2027_REFACTOR_MASTERPLAN.md`, then push the local bootstrap commit without force.
2. Do not replace the detailed remote masterplan with the shorter local summary; keep the detailed design contract and, if useful, keep the short local text under a different `*_SUMMARY.md` filename.
3. Run and record at minimum:
   - `python -m pytest tests/test_path_portability.py`
   - package import check for `ptl`
   - a Git tracked-file audit showing no raw `.h5ad`, large result tables/checkpoints, submission bundles or generated PDFs are tracked unintentionally.
4. Confirm/record whether a v1 provenance tag exists and exactly which commit it points to. If the root Git history cannot represent a true pre-refactor source state, state that limitation explicitly rather than inventing provenance.
5. Update the root README to distinguish legacy CBAC/v1 code from the active PTL-v2/ICLR package and point to the masterplan and execution packet.

Proceed directly to G1 after these checks unless a destructive Git conflict appears.

# G1 — Data registry, environments and folds

## G1.1 Audit all local raw datasets, but process only the selected surface

Use backed/read-only AnnData inspection. Build `artifacts/manifests/dataset_audit.csv` with at least:

- dataset_id / filename
- source study
- shape
- candidate perturbation columns
- candidate control labels
- cell line/type
- perturbation modality
- condition/timepoint/batch columns
- number of controls
- number of unique perturbations
- combination support
- metadata completeness flags
- eligibility for `common_core`, `context_expansion`, `external_candidate`

Do not reprocess all 54 datasets.

## G1.2 Selected surfaces

Common Core target:

- NormanWeissman2019
- ReplogleWeissman2022 K562 GWPS as primary large K562 screen
- ReplogleWeissman2022 RPE1
- fully audited Adamson 2016

Retain K562 essential as useful existing/extended evidence rather than discarding it.

Context Expansion candidates:

- Tian 2021 CRISPRa
- Tian 2021 CRISPRi
- Frangieh 2021 RNA, split into biological environments only when metadata justify it
- Papalexi 2021 RNA
- XuCao2023 only after exact paper/version mapping is verified
- Datlinger only as supplementary context if it contributes a distinct environment

External-A: GSE284197, marked `previously_used_external=true`.

External-B: score WesselsSatija2023 and JoungZhang2023 using a pre-result rubric: perturbation/control completeness, size, metadata clarity, contextual distinctness, predictor compatibility. Freeze one winner in `configs/datasets.yaml` before any PTL-v2 performance evaluation. Record the rubric and decision in `docs/EXTERNAL_B_FREEZE.md`.

## G1.3 Adamson resolution

Inspect all three local Adamson files (`10X001`, `10X005`, `10X010`) and the source metadata. Determine whether they are replicates, assays or distinct experimental subsets before merging. The current 10-signature processed Adamson is not eligible for headline results. Produce `docs/ADAMSON_AUDIT.md` and one reproducible preprocessing configuration for the adopted full surface.

## G1.4 Stable IDs

Implement and test:

- `environment_id`
- `biological_instance_id`
- `prediction_id`

`biological_instance_id` must group the same target response across predictors and seeds. Generate `artifacts/manifests/reliability_folds.parquet` locally (ignored) plus a compact CSV/JSON summary that is tracked.

PTL cross-fitting must group by `biological_instance_id`, never by `run_id`.

## G1.5 Path and gene-space contracts

Create portable configs and a path resolver. V2 source/config cannot depend on historical `H:\2026try\4.24` paths.

Create an evaluation-gene-space manifest with checksum and build provenance. Separate `model_gene_space` from `evaluation_gene_space`; do not force predictor training into the intersection unless required by the model.

# G2 — Predictor/output contract and first credible predictors

## G2.1 Standard output schema

Implement a typed contract (`dataclass`, pydantic or validated schema) containing:

- prediction_id / biological_instance_id / environment_id
- predictor name/version/commit
- predictor split ID
- seed
- prediction array path
- gene list/mapping path
- native UQ fields when available
- ensemble/bootstrap membership
- training configuration/provenance

Predictions must not contain target fidelity metrics.

## G2.2 Baseline roster for the pilot

Implement first:

1. non-control/global response baseline
2. matching/perturbation mean where scientifically defined
3. current Ridge as a transparent baseline
4. a strong modern linear/bilinear baseline following the Ahlmann-Eltze-style formulation where reproducible from the paper/code
5. official GEARS on its supported common-core tasks

The existing GEARS adapter remains legacy only. Headline GEARS may not use self-loop graph replacement or zero-vector fallback.

Do not wait for CPA/scGPT before G4. Their integration belongs to G5.

## G2.3 Common vs extended matrix

Write `configs/tasks.yaml` specifying which predictor-task-environment cells are scientifically valid. Do not force every model into every context. Produce a common matrix for fair comparison and an extended matrix for generality.

# G3 — Real uncertainty contract

## G3.1 Retire legacy confidence for headline use

Rename/label the existing heuristic `confidence` fields as `legacy_confidence_proxy`. They may be retained only as v1/supplementary comparisons.

## G3.2 UQ generation

For learned predictors in the pilot, use at least three independent seeds where feasible. Extract common ensemble UQ summaries:

- mean gene-wise predictive variance
- median variance
- top-effect-gene variance
- pairwise cosine disagreement
- predicted effect-norm variance

For Ridge/simple estimators, implement computationally cheap grouped/bootstrap UQ rather than mixing support/context directly into the UQ value.

Keep model-native UQ separate from ensemble UQ.

## G3.3 Predictor-relative normalization

Fit the empirical UQ normalization using source/training data only:

`q = F_source,predictor(u)` and `c = 1-q`.

No target fidelity labels may enter this normalization. Fit scalar Platt/logistic and isotonic calibration inside each PTL training fold only.

## G3.4 RQ1 pilot output

Before PTL training, generate a compact RQ1 table showing whether raw/normalized UQ relates to realized fidelity across support, novelty and environment. Do not over-interpret significance yet; this is an early sanity check that the core paper question is measurable.

# G4 — Leakage-safe PTL-v2 pilot

## G4.1 Four separate tables

Implement explicit schemas/loaders for:

- predictions
- deployment_features
- outcomes
- folds

Automated tests must fail if outcome columns enter deployment feature specifications.

## G4.2 Feature blocks

Use explicit groups:

- U: predictor uncertainty descriptors
- P: prediction geometry (norm, sparsity/concentration, training-response manifold/proximity descriptors where target-free)
- S: support/reference availability
- N: perturbation/combination novelty
- C: biological/experimental context

Headline PTL must not use benchmark labels such as `split_family`, `stress_family_flag` or `heldout_target` as direct features. Dataset/predictor identity may be included only in `PTL-Full`; keep `PTL-Context` as the primary method and `PTL-no-ID` as robustness.

## G4.3 Reliability methods

Implement and evaluate under the same grouped folds:

- raw normalized UQ
- logistic/Platt calibration
- isotonic calibration
- support/novelty-only model
- legacy/current PTL-RF adapted to the new feature contract
- calibrated GBDT meta-estimator (PertEMA-style baseline)
- PTL-Context
- PTL-Full and PTL-no-ID ablations

Primary PTL-Context formulation:

`p = sigmoid(exp(a(z)) * logit(p0) + b(z))`

with zero/identity initialization and regularization toward `a=0, b=0`. Start with a small MLP or linear block encoders. Do not add a large network unless the small model demonstrably underfits.

Implement the optional continuous-risk head, but keep it only if cross-validated selective metrics improve.

## G4.4 Pilot surface

Run the full G4 pilot on a small but credible surface before scaling:

- Norman
- Replogle K562 (use the adopted primary K562 surface)
- Replogle RPE1

Predictors: strong mean/matching baseline, Ridge/strong linear, official GEARS where supported.

Use five-fold biological-instance grouped cross-fitting for the reliability layer. The same biological target across predictor/seed variants must remain in one fold.

## G4.5 Metrics and aggregation

Primary:

- AURC / excess AURC
- realized risk at 50% and 80% coverage
- FTR at 50% and 80% coverage
- Brier score / log loss

Secondary:

- AUROC / AUPRC
- Spearman predicted vs realized continuous risk

Compute per `(predictor, environment)` first, then macro-average. Add paired hierarchical bootstrap CIs where implementation is ready; otherwise create the interface and flag the pilot CI as provisional.

## G4.6 Pilot decision rule

Do not require universal zero-shot success. The pilot passes if:

- no leakage/ID grouping violations are found;
- raw UQ is demonstrably non-identical to contextual features;
- PTL-Context has a reproducible selective-risk/calibration advantage over raw UQ and ordinary scalar calibration on a meaningful subset of model-environment cells, without being entirely explained by explicit dataset/predictor identity;
- results are generated from reproducible manifests.

If PTL-Context fails, diagnose feature/target/UQ design first; do not immediately add more datasets or larger networks.

# Required deliverables at the end of this sprint

Code/config:
- `configs/datasets.yaml`
- `configs/environments.yaml`
- `configs/predictors.yaml`
- `configs/tasks.yaml`
- `configs/reliability.yaml`
- functional modules under `src/ptl/data`, `predictors`, `uncertainty`, `reliability`, `evaluation`
- leakage/grouping/contract tests

Audits/docs:
- `docs/ADAMSON_AUDIT.md`
- `docs/EXTERNAL_B_FREEZE.md`
- `docs/G1_G4_EXECUTION_REPORT.md`
- exact Xu mapping status
- Git synchronization/push status

Artifacts/source-data summaries:
- dataset audit
- environment registry summary
- prediction/UQ manifest summary
- fold summary
- RQ1 raw-UQ summary
- G4 method comparison summary
- preliminary Figure-2/3 source data (not final styling)

Do not spend time polishing manuscript prose or final figures before these deliverables exist. Once G4 passes, immediately begin G5-G6 with CPA/scGPT and PRESCRIBE as the highest-value expansion.