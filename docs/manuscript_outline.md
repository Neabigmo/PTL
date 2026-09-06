# Manuscript Outline

Last synchronized: 2026-09-05 Asia/Shanghai

Working title: Reliability modeling for transportable perturbation-response predictions in public single-cell functional-genomics screens

## Core message

Single-cell perturbation models can look reliable on random splits while degrading under biologically meaningful context shifts. This manuscript defines a context-stress benchmark, evaluates simple and learned signature-track baselines across perturbation, combination, support, dataset, and external-holdout regimes, and introduces a model-agnostic Perturbation Transportability Layer (PTL) that filters false transportability better than naive model confidence.

## Target venue

The active submission line targets Computational Biology and Chemistry (CBAC)
with a single-anonymized manuscript, deterministic figures, and a separate
submission package.

## Manuscript structure

1. Introduction
   - Motivation: virtual-cell and perturbation prediction models need evaluation beyond random splits.
   - Gap: current evaluations often mix interpolation and transport across context boundaries.
   - Contribution 1: context-stress benchmark over harmonized single-cell perturbation signatures.
   - Contribution 2: empirical model-ranking analysis under random, perturbation, combination, support, dataset, and external transfer.
   - Contribution 3: PTL as a model-agnostic reliability layer for false-transportability filtering.
   - Contribution 4: failure-mode atlas for novelty, support, and transfer-boundary settings.

2. Methods
   - Data sources: scPerturb-derived internal datasets plus active external holdout.
   - Signature-track modeling surface and reference-aware delta signatures.
   - Split families and leakage controls.
   - Baselines: control mean, global delta, perturbation mean delta, cell-context kNN delta, ridge regression.
   - Metrics: primary non-control cosine, transportability label, selective risk, false transportability, paired comparisons.
   - PTL: feature groups, estimators, ablations, and evaluation design.
   - Failure-mode atlas construction.

3. Results
   - Figures 1-5: CBAC study design, benchmark audit, ranking instability, PTL reliability filtering, and failure/case atlas.
   - Supplementary Figures S1-S4: split audit, threshold/calibration sensitivity, GEARS contract status, and case-level diagnostics.
   - Main result anchors:
     - Dataset-heldout transfer is the hardest measured split family by average non-control cosine.
     - Ridge leads random, low-support, and unseen-perturbation regimes, while global delta leads dataset-heldout and external-holdout regimes.
     - The primary deployment-feature random forest reduces mean false-transportability rate from 0.709 for naive confidence to 0.236.
     - Retained signatures have transportable rate 0.907 versus 0.174 among rejected high-risk signatures.

4. Discussion
   - Random-split performance should not be treated as a complete proxy for transfer.
   - A simple model can be robust in some stress regimes; there is no universal winner across contexts.
   - PTL is useful as a reliability and abstention layer, especially for false-transportability filtering.
   - Failure-mode reporting gives a practical audit surface for future perturbation models.

5. Limitations
   - Signature-level benchmark does not replace cell-level distribution modeling.
   - The external holdout is one independent source, not a universal transfer census.
   - PTL uses local benchmark artifacts and does not prove future experimental confirmation.
   - No claim is made about care decisions, universal biological mechanism, or final model dominance.

6. Conclusion
   - Context-stress benchmarking plus PTL provides a reproducible way to ask when perturbation predictions should be trusted, filtered, or studied as failures.

## Required tables and figures

- Main figures: `results/figures_cbac/Figure_1.png` through `results/figures_cbac/Figure_5.png`.
- Graphical abstract: `results/figures_cbac/Graphical_Abstract.png`.
- Supplementary figures: `results/figures_cbac/Supplementary_Figure_S1.png` through `Supplementary_Figure_S4.png`.
- Evidence tables:
  - `results/tables/main_findings.csv`
  - `results/tables/transfer_decay_summary.csv`
  - `results/tables/selective_prediction_summary.csv`
  - `results/tables/failure_mode_atlas.csv`
  - `results/tables/model_pairwise_comparisons.csv`
  - `results/tables/additional_public_screen_validation_summary.csv`
  - `results/tables/external_candidate_contract_audit.csv`
  - `results/tables/gears_validation_summary.csv`
