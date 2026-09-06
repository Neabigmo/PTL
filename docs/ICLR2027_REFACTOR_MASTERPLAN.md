# PTL ICLR 2027 Restructuring Master Plan

Status: design freeze candidate, 2026-09-06.

## 0. Scientific thesis

PTL-v2 is not a larger perturbation benchmark and not a universal zero-shot reliability system. The central question is:

> Predictor uncertainty is context-dependent. Can prediction-, support-, novelty-, and biological-context information repair uncertainty well enough to improve prediction-level selective decisions under realistic perturbation shift?

The intended paper is a machine-learning reliability paper using single-cell perturbation prediction as the testbed.

## 1. Git and source-of-truth architecture

The active local workspace is currently not a Git repository. The public repository has historical minimal branches `methods-submission-minimal` and `cbac-submission-minimal`; this new integration line is `iclr2027-restructure`.

Before scientific refactoring:

1. Freeze the current CBAC code/results as immutable v1 provenance. Do not overwrite v1 result files.
2. Make the current workspace root the single active source tree; do not maintain `_github_ptl_work` as a second editable codebase.
3. Initialize or connect the local root to Git safely without adding raw data, processed matrices, model checkpoints, large result tables, compiled PDFs, caches, or third-party clones.
4. Add a root `.gitignore`, `.gitattributes`, `pyproject.toml`, `README.md`, and environment manifests.
5. Create one long-lived integration branch `iclr2027-restructure`; use short feature branches (`feat/data-registry`, `feat/predictor-contract`, `feat/uncertainty`, `feat/ptl-context`, `feat/evaluation`, `feat/figures-paper`) and merge only after tests.
6. Third-party code must be pinned by repository URL + commit SHA in a tracked lock file; do not vendor nested editable copies into the main source tree.
7. Track compact manifests and figure source data, not large generated artifacts. Every untracked array/model artifact must have a tracked manifest row containing checksum, provenance, configuration and generation command.

Recommended package layout:

```text
configs/
  datasets.yaml
  environments.yaml
  predictors.yaml
  tasks.yaml
  reliability.yaml
  third_party.lock.yaml
envs/
  core.yml
  systema.yml
  scgpt.yml
  prescribe.yml
src/ptl/
  data/
  predictors/
  uncertainty/
  reliability/
  evaluation/
  experiments/
  plotting/
tests/
paper/iclr2027/
artifacts/manifests/
artifacts/source_data/
scripts/setup/
scripts/run/
```

Raw data stay under local `data/` and are ignored. Native model outputs and large arrays stay under ignored `cache/` or `artifacts/local/` paths. Historical phase code is preserved by the v1 freeze/tag rather than copied indefinitely into another active namespace.

## 2. Data architecture

### 2.1 Main standardized surface

Use the published Systema-compatible genetic-perturbation surface as the principal common benchmark where possible, because it already supports CPA, GEARS and scGPT under one pipeline and explicitly addresses systematic variation. Candidate environments:

- Adamson 2016 (full dataset; replace the current 10-signature partial surface)
- Norman 2019
- Replogle 2022 K562 GWPS
- Replogle 2022 RPE1
- Tian CRISPRa
- Tian CRISPRi
- Xu dataset used by Systema (verify exact mapping/accession against local `XuCao2023` before adoption)
- Frangieh 2021 contexts, represented as separate biological environments when metadata support this.

Do not choose datasets merely to maximize count. Each retained environment must add a distinct perturbation, cell-state, experimental-condition, or platform context.

### 2.2 Existing PTL surface

Retain Norman, Replogle K562 essential, Replogle RPE1, Papalexi, Datlinger and GSE284197 as legacy/extended evidence when compatible. Do not allow tiny partial processed datasets (e.g. current Adamson 10 signatures or Dixit 11 signatures) to carry dataset-level headline claims.

### 2.3 External validation

- GSE284197 remains a useful independent public-screen environment but is no longer treated as an untouched lockbox because it has already informed project development.
- Select exactly one new frozen External-B dataset from currently unused local raw data (priority audit: WesselsSatija2023 and JoungZhang2023; select based only on metadata completeness, controls, perturbation count and contextual distinctness, never on PTL performance). Freeze its role before running PTL-v2.

### 2.4 Identifiers

Define and test three stable IDs:

- `environment_id = dataset + cell_context + modality + condition`
- `biological_instance_id = environment + perturbation + dose + timepoint` (fields omitted only when genuinely absent)
- `prediction_id = biological_instance + predictor + predictor_split + model_seed`

All predictions for the same biological ground-truth instance across predictors and seeds must share the same PTL cross-fitting fold.

### 2.5 Gene spaces

Separate predictor-native gene space from evaluation gene space. Each predictor may train/infer in its valid native vocabulary. Common metrics are calculated only after mapping prediction and target to a declared shared evaluation panel. Never force every model to train on the global intersection merely for convenience.

## 3. Predictor roster

PTL is not a predictor-ranking paper, so use a compact but credible roster.

### 3.1 Mandatory simple/strong baselines

- non-control/mean response baseline
- matching/context mean where defined
- Ahlmann-Eltze et al. bilinear/linear perturbation baseline (mandatory modern strong baseline)
- additive baseline for Norman combinatorial prediction
- current ridge/global-delta baselines retained as legacy or supplementary comparisons

### 3.2 Learned predictors

Mandatory target roster on compatible tasks:

- official GEARS, no self-loop replacement and no zero-vector fallback in headline results
- official/current CPA only on tasks its representation supports
- scGPT/Systema implementation as pretrained/foundation-style family

Strong optional additions:

- PRESCRIBE on compatible Norman/Replogle surfaces as the most important uncertainty-aware competitor
- SLIM as a 2026 strong lightweight linear/biological-prior comparator if reproduction cost is low

Do not require every predictor to fill every task cell. Maintain a `common_matrix` for apples-to-apples comparisons and an `extended_matrix` where scientifically appropriate models contribute additional reliability evidence.

## 4. Prediction and uncertainty contracts

The existing project mixes contextual support/similarity information into a variable called confidence. PTL-v2 must separate predictor uncertainty from context.

Each predictor run writes a standard metadata record plus array artifact:

- prediction mean/signature
- native uncertainty if the predictor provides one
- ensemble/bootstrap members or their sufficient statistics
- predictor commit/version
- environment and split IDs
- gene-space mapping
- seed and training configuration

For learned models, prefer at least three independent training seeds per fixed predictor split. Derive a model-agnostic ensemble uncertainty vector, e.g. mean gene variance, top-k variance, pairwise cosine disagreement and predicted-effect-norm variance. For simple/linear models use cheap grouped/bootstrap resampling where meaningful.

Normalize uncertainty predictor-relatively using source/training distributions, e.g. empirical quantile `q = F_train(u)` and confidence-like score `c = 1-q`. This normalization must not use target fidelity labels.

## 5. Outcome and feature separation

Never construct one giant PTL dataframe. Maintain four logically separate tables:

1. `predictions`: prediction identifiers, predictor provenance and UQ only.
2. `deployment_features`: prediction geometry, support, novelty and context available before observing the target outcome.
3. `outcomes`: fidelity, biological metrics, continuous risk and operational transportability labels.
4. `folds`: biological-instance grouped cross-fitting and robustness holdout assignments.

Join by `prediction_id` only inside training/evaluation code. Add automated leakage tests that fail if outcome-derived columns enter feature specifications.

## 6. PTL-v2 method

### 6.1 Baseline calibration

First map normalized predictor uncertainty to a scalar baseline reliability probability `p0` using a scalar calibrator fitted only on PTL-training folds.

### 6.2 Context-conditioned correction

Let `z` contain structured deployment information. Use block encoders for:

- U: uncertainty descriptors
- P: prediction geometry
- S: support/availability
- N: perturbation/combination novelty
- C: biological and experimental context

The main constrained calibrator is:

`p = sigmoid(exp(a_phi(z)) * logit(p0) + b_phi(z))`

Initialize the context heads at zero so the initial mapping is the identity (`exp(0)=1`, `b=0`). Add an identity-preserving regularizer on `a` and `b` so the model changes baseline confidence only when context is informative. Because the slope is positive, the method preserves monotonicity in baseline confidence for fixed context.

A small continuous-risk head may share the context representation and predict normalized continuous prediction risk as an auxiliary task. Keep this auxiliary head only if cross-validation shows stable benefit; it is not required for the core claim.

### 6.3 PTL variants

- `PTL-Context`: primary method; uses real deployment context, not benchmark split names.
- `PTL-Full`: adds explicit dataset/predictor identity as an all-information upper/full version.
- `PTL-no-ID`: removes dataset/predictor identity as a robustness ablation.
- `PTL-RF`: preserve the current random-forest model as a strong nonlinear baseline, not the new method.
- `PTL-GBDT`: PertEMA-style strong meta-estimator baseline with proper calibration.

Never feed `split_family=external_holdout`, `unseen_perturbation_split`, or equivalent benchmark labels to the headline method. Encode the underlying measurable state (seen fraction, support, context distance, modality, etc.) instead.

## 7. Theory and ML framing

The paper should contain modest but exact theory, not decorative theorem statements.

Let `Y` be the binary operational reliability outcome, `C` the baseline calibrated confidence and `Z` context.

1. Under Brier loss, the Bayes risk using `(C,Z)` cannot exceed the Bayes risk using `C` alone. The improvement equals the reduction in conditional variance:

`E Var(Y|C) - E Var(Y|C,Z) = E Var(E[Y|C,Z] | C) >= 0`.

2. Under log loss, the ideal gain from context is `I(Y; Z | C) >= 0`. This gives the paper a precise question: how much conditional reliability information remains in context after predictor uncertainty is known?

3. For fixed coverage, ranking/accepting examples by the true conditional reliability probability minimizes expected false acceptance. This connects contextual calibration to selective prediction.

4. Positive context-conditioned slope guarantees that the proposed calibrator does not invert the ordering of baseline confidence within an identical context.

State clearly that these are population/Bayes statements; finite-sample contextual models can overfit, motivating grouped cross-fitting, regularization and held-environment robustness tests.

## 8. Fidelity targets and metric policy

Do not rely on one metric or one binary threshold.

- Maintain the current matched-control delta cosine as a legacy/interpretability metric.
- Add Systema perturbation-specific evaluation (perturbed-centroid reference / centroid accuracy) on the common genetic-perturbation surface.
- Use scPertEval to audit candidate protocols and select a small predefined metric panel based on task alignment and positive/negative-control discrimination, not on which metric makes PTL look best.
- Include one gene-level biological diagnostic (e.g. DEG direction/overlap) and perturbation retrieval.

For PTL outcomes:

- Primary selective-risk analysis should use continuous realized risk and AURC/eAURC, which does not require committing the entire story to one binary threshold.
- Keep the relative transportability label as an operational secondary endpoint with the existing threshold sensitivity analysis.
- Explicitly acknowledge that a model-relative random-split anchor answers 'transport relative to that model's familiar-setting fidelity', not absolute biological correctness.

Experimental perturbation reliability/noise should be an evaluation stratum, not a PTL deployment feature, unless it is demonstrably available at prediction time. Use the 2026 perturbation-reliability literature to test whether gains persist on high-quality perturbations.

## 9. Reliability baselines and selective metrics

Mandatory reliability comparisons:

- raw normalized UQ
- scalar logistic/Platt calibration
- isotonic calibration
- support/novelty-only model
- current PTL-RF
- PertEMA-style calibrated GBDT
- PTL-Context
- PTL-Full/no-ID ablations
- PRESCRIBE confidence and PRESCRIBE+PTL on compatible subsets if reproducible

Primary reporting:

- AURC and excess AURC
- continuous risk at 50% and 80% coverage
- false-transportability rate at 50% and 80% coverage
- Brier score / log loss

Secondary:

- AUROC / AUPRC
- Spearman correlation between predicted and realized continuous risk

Report environment-level metrics first, then macro-average across environments. Use paired hierarchical bootstrap confidence intervals so large Replogle datasets do not dominate simply because they contain more signatures.

## 10. Experimental questions

RQ1: Does predictor uncertainty become conditionally unreliable under perturbation/context shift?

Analyze raw/native/ensemble UQ calibration as a function of support, novelty, biological context and environment. This empirical finding must precede the PTL result.

RQ2: Can available context repair uncertainty?

Compare raw UQ, scalar calibration, strong meta-estimator baselines and PTL-Context using selective-risk/calibration metrics.

RQ3: Which information contributes to reliability?

Use cumulative information experiments:

`U -> U+P -> U+P+S -> U+P+S+N -> U+P+S+N+C`,

plus critical controls `context-only`, `no predictor ID`, `no dataset ID`, and `no IDs`.

RQ4: How robust is the correction?

Use biological-instance grouped cross-fitting as the default. Add leave-one-screen/environment-out and leave-one-predictor-out as robustness rather than defining the entire paper around double-OOD zero-shot transfer. Repeat core conclusions under at least one Systema/scPertEval-aligned fidelity protocol and on a high-experimental-reliability subset.

## 11. Figure plan

Main text should use four dense ICLR-style figures, not a long journal benchmark atlas.

Figure 1 — Problem, model and evaluation surface.
- Same nominal confidence under familiar/high-support, novel/low-support and shifted context.
- PTL block diagram and core contextual-calibration equation.
- Compact environment x predictor/task matrix.

Figure 2 — Confidence is context-dependent.
- reliability/calibration curves in representative regimes
- predictor x environment heatmap of raw-UQ selective risk/calibration
- UQ vs realized fidelity scatter with high-confidence failures
- subgroup trend versus support/novelty/context shift

Figure 3 — Context repairs uncertainty.
- risk-coverage curves for raw UQ, scalar calibration, GBDT/PertEMA-style and PTL-Context
- forest/dot-whisker plot of paired PTL improvement per model-environment
- improvement heatmap
- FTR/risk at a fixed high-coverage point

Figure 4 — Where the information comes from and robustness.
- cumulative U/P/S/N/C information curve
- learned context correction surface/partial dependence showing up/down correction
- identity and held-environment/predictor robustness
- compact biological consequence panel (DEG direction and perturbation retrieval)

Move the existing benchmark-composition, model-rank-instability, full failure atlas, threshold sensitivity, full per-screen calibration, case atlas and additional screen tables to supplementary figures.

## 12. Paper narrative

Candidate title direction: `Confidence Is Contextual: Reliability Calibration for Single-Cell Perturbation Prediction`.

Nine-page ICLR main-text story:

1. Introduction: uncertainty/confidence has no context-free meaning under biological shift; benchmark ranking instability is motivation, not the contribution.
2. Problem setup: predictor, UQ, context, fidelity and selective decision.
3. PTL: baseline calibration, block-structured contextual correction, identity-preserving regularization and optional continuous-risk head.
4. Experimental setup: standardized common surface, predictors, grouped cross-fitting and metric policy.
5. Results organized strictly by RQ1-RQ4.
6. Related work and limitations: short and explicit.

Do not claim first reliability filtering, first model-agnostic post-hoc reliability, or first OOD perturbation benchmark.

## 13. Citation backbone requiring update

The existing literature map predates several directly relevant 2025-2026 works. At minimum add and correctly position:

- Ahlmann-Eltze, Huber & Anders, Nature Methods 2025: simple linear baselines challenge deep perturbation predictors.
- Wei et al., Nature Methods 2026: 27 methods, 29 datasets, generalization benchmark.
- Vinas Torne et al., Systema, Nature Biotechnology 2026: systematic variation and perturbation-specific evaluation.
- Cheng et al., PRESCRIBE, NeurIPS 2025: epistemic/aleatoric uncertainty and filtering in single-cell perturbation prediction.
- Radig et al., scArchon, Genome Biology 2026: reproducible multi-tool perturbation benchmarking and biological hallucination concerns.
- Schäfer et al., scPertEval, bioRxiv 2026: evaluation-protocol taxonomy and reference implementations.
- Wang et al., Reliable single-cell perturbations..., bioRxiv 2026: perturbation-level experimental reliability affects benchmarking.
- PertEMA software 2026: closest post-hoc per-prediction reliability system; explicitly note it has no accompanying paper and reports that a frozen estimator does not transfer across screens.
- Angelopoulos et al., Conformal Risk Control, ICLR 2024, plus non-exchangeable CRC where relevant.
- Recent ICLR selective-prediction/contextual calibration work should be cited to delimit generic post-hoc calibration novelty.

## 14. Execution gates

G0 Git/source-of-truth: root Git architecture, ignores, package skeleton, v1 freeze, remote integration branch.

G1 Data registry: exact Systema/local dataset mapping, environment IDs, biological IDs, frozen External-B selection, no model runs yet.

G2 Predictor contract: mean/linear baselines plus official GEARS first; output/provenance contract and tests pass.

G3 UQ contract: true predictor-native/ensemble/bootstrap uncertainty separated from contextual features. Run the first RQ1 pilot.

G4 PTL pilot: scalar calibration, PTL-RF, GBDT and contextual calibrator under biological-instance grouped cross-fitting on a small subset. Stop and inspect whether context adds held-out information before scaling.

G5 Full model roster: add CPA, scGPT and PRESCRIBE/SLIM where feasible; do not expand datasets until the pipeline is stable.

G6 Full experiments: RQ1-RQ4, hierarchical bootstrap, alternative metric robustness, final External-B opening.

G7 Figures/paper: regenerate four main figures from tracked source-data tables; write ICLR paper from results, not by editing the CBAC manuscript in place.

## 15. Non-negotiable ambiguities to resolve before full runs

- Verify whether local `XuCao2023.h5ad` is exactly the Xu dataset/version used by Systema.
- Determine the correct full Adamson preprocessing source; the current 10-signature processed file cannot be the headline Adamson surface.
- Decide task-specific predictor eligibility rather than forcing CPA/GEARS/scGPT into unsupported settings.
- Define the primary fidelity protocol before large PTL runs using a metric audit on training/source environments only.
- Define a real predictor UQ per model; retire the current heuristic `confidence` as the headline raw-UQ baseline.
- Freeze External-B before PTL-v2 feature/architecture selection.
- Separate the old 6,257/7,880 global-intersection naming/history and regenerate a single auditable evaluation-gene-space manifest.

This document is the design contract. Any later experiment or feature must map to one of RQ1-RQ4 or be explicitly labeled supplementary.
