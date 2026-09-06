# Phase 06 Metrics

Last updated: 2026-04-26 11:36 Asia/Shanghai

## Scope
- Continued the project from the completed Phase 05 baseline campaign.
- Kept Phase 06 limited to rigorous evaluation, confidence, transportability, and statistical comparison.
- Did not start Phase 07 PTL modeling, Phase 08 analysis synthesis, or manuscript drafting.

## Skills and environment
- Used the `scikit-learn` skill for evaluation design, deterministic metrics, PCA-based distributional comparison, and validation-minded implementation.
- Used the `statsmodels` skill for FDR correction and Wilson confidence intervals.
- Executed in `E:\anaconda3\envs\sw_mgli`.

## Recovery context
- Phase 06 code and tests already existed but the first full run had been interrupted.
- Before rerunning, checked for residual `python.exe` processes targeting `evaluate_phase06.py`; none remained active.
- Re-ran the full evaluator with `login:false` because PowerShell login/profile mode had previously caused avoidable hangs in this environment.

## Implementation and execution summary
- Confirmed the canonical Phase 06 universe remains `results/tables/baseline_run_matrix.csv`.
- Rebuilt `results/tables/run_selection_audit.csv` using latest-status deduplication from `results/tables/experiment_registry.csv`.
- Included only manifest-linked runs whose latest status was `success` or `cached`.
- Excluded the `5` non-manifest registry extras / smoke runs from the formal evaluation universe.
- Re-ran:
  - `src/evaluation/evaluate_phase06.py`
- Verified the evaluator completed all `210` formal runs and wrote:
  - `results/tables/per_signature_metrics.parquet`
  - `results/tables/all_metrics.csv`
  - `results/tables/phase06_summary_by_family.csv`
  - `results/tables/model_pairwise_comparisons.csv`
  - `results/logs/evaluation/phase06_metrics.log`
- Re-ran unit tests for the evaluation module:
  - `tests/evaluation/test_metrics.py`
  - `tests/evaluation/test_run_selection.py`
  - Result: `6 passed`

## Outputs produced
- `src/evaluation/__init__.py`
- `src/evaluation/metrics.py`
- `src/evaluation/confidence.py`
- `src/evaluation/evaluate_phase06.py`
- `tests/evaluation/test_metrics.py`
- `tests/evaluation/test_run_selection.py`
- `results/tables/run_selection_audit.csv`
- `results/tables/per_signature_metrics.parquet`
- `results/tables/all_metrics.csv`
- `results/tables/phase06_summary_by_family.csv`
- `results/tables/model_pairwise_comparisons.csv`
- `results/logs/evaluation/phase06_metrics.log`
- `docs/metric_dictionary.md`

## Acceptance summary
- Formal manifest runs: `210`
- Included in Phase 06: `210`
- Excluded registry extras: `5`
- Per-signature rows written: `139,705`
- Run-level metric rows written: `210`
- Family summary rows written: `30`
- Pairwise comparison rows written: `20`
- Evaluation tests: `6 passed`

## High-level findings
- Overall mean `mean_cosine_non_control` by model:
  1. `ridge_regression_baseline`: `0.316333`
  2. `perturbation_mean_delta_baseline`: `0.296470`
  3. `cell_context_knn_delta_baseline`: `0.275280`
  4. `global_delta_baseline`: `0.251902`
  5. `control_mean_baseline`: `0.000000`
- Family-level leaders:
  - `random_split`: `ridge_regression_baseline`
  - `low_support_split`: `ridge_regression_baseline`
  - `unseen_perturbation_split`: `ridge_regression_baseline`
  - `unseen_combination_split`: `perturbation_mean_delta_baseline`
  - `dataset_heldout_split`: `global_delta_baseline`
  - `external_holdout`: `global_delta_baseline`
- This sharpens the Phase 05 hint:
  - ridge is strongest inside familiar regimes
  - the simpler global-delta baseline is more robust under cross-dataset and external transfer

## Statistical comparison highlights
- Pairwise comparisons were written to `results/tables/model_pairwise_comparisons.csv`.
- For `mean_cosine_non_control`, all non-trivial models significantly outperform `control_mean_baseline` after FDR correction.
- `ridge_regression_baseline` significantly improves `mean_cosine_non_control` over `global_delta_baseline` on matched contexts overall.
- For `random_to_stress_drop`, `ridge_regression_baseline` has significantly lower degradation than the other learned or heuristic baselines in several matched comparisons, even though `global_delta_baseline` still leads in the hardest transfer families on raw cosine.

## Distributional metric note
- In the final completed run, all `210` run rows have `distributional_status = ready`.
- No run required an `unsupported` fallback for the implemented distributional metrics.

## Engineering notes
- The main execution risk carried over from the interrupted run was repeated heavy split preparation.
- The cached split-handling fix already present in `src/evaluation/evaluate_phase06.py` was preserved and validated in the successful full rerun.
- No further code changes were needed after the clean rerun and test pass.

## Acceptance checklist
- [x] `run_selection_audit.csv` exists and reflects canonical-manifest selection.
- [x] `per_signature_metrics.parquet` exists.
- [x] `all_metrics.csv` exists.
- [x] `phase06_summary_by_family.csv` exists.
- [x] `model_pairwise_comparisons.csv` exists.
- [x] `metric_dictionary.md` exists.
- [x] Evaluation tests pass.
- [x] No Phase 07 / PTL / manuscript work was started.
