# Current Positioning Statement

Last synchronized: 2026-09-05 Asia/Shanghai

## Working Positioning

Virtual-cell and single-cell foundation models are rapidly expanding the scope of in-silico perturbation prediction. However, for biological use, the crucial question is not only whether a model can predict a response on average, but whether that response is transportable to an unseen biological context. This project positions Perturbation Transportability Lens (PTL) as a benchmark-first, model-agnostic reliability layer for stress-testing and calibrating virtual-cell perturbation predictions.

## Core Argument

Current virtual-cell models are increasingly powerful, but the field lacks a systematic and model-agnostic way to define, quantify, and predict when perturbation responses are transportable across biological contexts.

PTL addresses that gap by treating transportability as an empirical reliability problem:
- define the context shift,
- measure fidelity under that shift,
- predict failure risk before trusting the output,
- and expose recurring failure modes.

## Manuscript Contributions

1. Define perturbation transportability as the empirical reliability of a predicted perturbation response under explicit biological context shift.
2. Build context-stress benchmark splits for single-cell perturbation prediction, including unseen perturbations, unseen combinations, unseen cell states or types, dataset holdout, and low-support settings.
3. Evaluate baseline and existing perturbation prediction models under random versus stress splits using expression, perturbation-effect, distributional, and ranking metrics.
4. Introduce PTL, a model-agnostic calibration layer that predicts a transportability score and supports keep-or-abstain selective prediction.
5. Produce a failure-mode atlas that links prediction failures to context distance, perturbation novelty, support level, and biological state shift.

## Reader Takeaway

The manuscript should make the reader believe that random-split performance is insufficient evidence for virtual-cell reliability. The correct unit of evaluation is a model's behavior under biologically meaningful context shifts, and the correct deployment question is whether a given prediction should be trusted, abstained from, or flagged for follow-up.

## Active target journal

The current submission line targets Computational Biology and Chemistry (CBAC).
The manuscript is framed as a public-data computational biology study of
reliability modeling for perturbation-response prediction, with deterministic
figures, bounded biological claims, and a single-anonymized submission format.

## Alternative journal angles

### EAAI

Emphasize the AI contribution: a benchmark-first reliability framework for virtual-cell perturbation prediction, with model-agnostic calibration, selective prediction, and interpretable failure modes.

### KBS

Emphasize knowledge-based systems: PTL connects biological context knowledge, perturbation novelty, support evidence, and model uncertainty into a structured reliability layer for decision support.

### AIME

Emphasize medical AI methodology: the paper stress-tests AI predictions before use in biomedical reasoning and provides calibrated abstention to avoid overconfident predictions under unseen biological contexts.

### JBI

Emphasize biomedical informatics: the work operationalizes external validity and transportability for single-cell perturbation predictions, using harmonized public data, reproducible splits, and transparent reliability scores.

### CMPB

Emphasize computational methods in biomedicine: the contribution is a reproducible benchmark and calibration pipeline for assessing perturbation response models across context shifts.

## Proposed Manuscript Frame

Title direction:
Context governs perturbation transfer: a calibrated benchmark for stress-testing virtual-cell models across unseen cellular states

Narrative flow:
1. Virtual-cell models promise scalable in-silico perturbation prediction.
2. Random split performance does not establish biological transportability.
3. Perturbation responses are context-dependent and require explicit stress testing.
4. PTL provides a model-agnostic reliability layer for predicting when outputs are likely to fail.
5. Selective prediction and failure-mode analysis make virtual-cell predictions more cautious, interpretable, and reusable.

## Terminology Defaults

- Use "perturbation transportability" for response reliability across unseen biological contexts.
- Use "context-stress split" for benchmark splits designed to expose specific generalization failures.
- Use "transportability score" for PTL's estimated probability or continuous reliability score.
- Use "selective prediction" for keeping high-confidence predictions and abstaining on high-risk predictions.
- Use "failure-mode atlas" for the cross-model summary of recurring failure patterns.

## Bounded Claims

Strong claims to make:
- PTL is a reliability layer, not a new perturbation predictor.
- Context-stress evaluation reveals failure modes hidden by random splits.
- Selective prediction can improve fidelity among retained predictions.
- Model rankings may be unstable across stress families.

Claims to avoid:
- PTL proves biological causality.
- PTL replaces wet-lab validation.
- Any model is universally best.
- Predictions are clinically actionable.
