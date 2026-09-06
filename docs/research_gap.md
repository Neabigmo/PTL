# Phase 01 Research Gap

Last updated: 2026-04-24 21:38 Asia/Shanghai

## Working Gap Statement

Current virtual-cell models are increasingly powerful, but the field lacks a systematic and model-agnostic way to define, quantify, and predict when perturbation responses are transportable across biological contexts.

## Why This Gap Exists

### 1. Virtual-cell ambition has outpaced reliability evaluation

Recent AI virtual-cell work argues for models that can represent, simulate, and steer cellular behavior across many molecular and cellular scales. Single-cell foundation models such as scGPT, Geneformer, CellFM, and related systems make this ambition technically plausible. Yet scale, pretraining breadth, and flexible model interfaces do not by themselves answer whether a predicted perturbation response remains valid when the perturbation, cell state, cell type, combination, dataset, assay, or support level changes.

### 2. Perturbation prediction benchmarks often report performance, not transportability

Existing perturbation-prediction studies and benchmarks compare models across tasks, datasets, and metrics. This is valuable, but the central question is often "which model performs better?" rather than "under which context shift should a prediction be trusted?" Random splits can mix related biological states across train and test and therefore overstate apparent generalization. Stress splits need to make the context shift explicit.

### 3. Perturbation response is context-dependent by construction

Perturbation effects depend on basal cell state, cell type or lineage, pathway activation, perturbation mechanism, dose, combination structure, replicate support, and dataset or batch context. A model can be accurate for a perturbation in one context and fail in another. This is not merely noise; it is a biological property that should be measured.

### 4. Reliability methods exist, but are not specialized to perturbation transportability

Calibration, selective prediction, conformal prediction, OOD detection, and domain generalization provide useful concepts. However, they typically calibrate generic classification or regression predictions. Single-cell perturbation prediction needs reliability features tied to biological shift: perturbation novelty, context distance, cell-state novelty, combination novelty, support counts, pathway support, and model uncertainty.

### 5. Model-agnostic failure maps are missing

Model papers often explain their own failures, but the field lacks a cross-model failure-mode atlas. A useful benchmark should identify where failures concentrate, such as unseen cell states, unseen perturbation combinations, low-support perturbations, strong distribution shifts, and perturbations with weak measurable effects.

## Project-Level Gap

This project fills the gap by treating perturbation transportability as the primary object of study. The benchmark will compare random splits to context-stress splits, evaluate multiple predictors and baselines under the same stress families, and add PTL as a lightweight reliability layer that predicts whether a model output is likely to be transportable.

## Claim Boundary

This project should not claim:
- Clinical recommendation.
- Wet-lab validation.
- Universal best model.
- De novo drug discovery.
- Final biological truth.

Allowed claim:
- This work defines and evaluates perturbation transportability, builds a context-stress benchmark, and provides a calibrated reliability layer for virtual-cell perturbation predictions.

## Research Questions

1. How much do perturbation prediction models degrade from random splits to context-stress splits?
2. Which stress axes cause the largest failures: unseen perturbations, unseen combinations, unseen cell states, unseen cell types, dataset holdout, or low support?
3. Do model rankings change across stress families?
4. Can model-agnostic features predict whether an individual prediction is transportable?
5. Does selective abstention improve fidelity among retained predictions?
6. What recurrent biological or statistical failure modes explain non-transportable predictions?

## Implication for Study Design

The main study should be benchmark-first:
- Use simple and strong baselines before complex models.
- Report both random-split and stress-split results.
- Separate expression fidelity from perturbation-effect fidelity and distributional fidelity.
- Evaluate selective prediction curves, not only full-coverage error.
- Write limitations explicitly: PTL predicts empirical reliability under available data and metrics; it does not prove biological causality or guarantee wet-lab validity.

