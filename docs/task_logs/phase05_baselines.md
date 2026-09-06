# Phase 05 Baselines

Last updated: 2026-04-26 03:40 Asia/Shanghai

## Scope
- Reviewed Phase 01 to Phase 04 artifacts before implementation and confirmed no mandatory backfill was required.
- Implemented the full Phase 05 signature-track baseline benchmark matrix.
- Kept Phase 05 limited to baseline execution and compact metrics; no Phase 06 metric expansion, PTL modeling, or manuscript drafting was started.

## Skill and environment notes
- Used the `scikit-learn` skill for the baseline design and evaluation workflow.
- Executed in `E:\anaconda3\envs\sw_mgli`.
- Verified `numpy`, `pandas`, `pyarrow`, and `scikit-learn` were available before baseline execution.

## Implementation summary
- Added `src/baselines/run_baseline.py` as the single Phase 05 CLI with:
  - `build-run-matrix`
  - `run-one`
  - `run-all`
  - `summarize`
- Implemented the 5 core baselines:
  - `control_mean_baseline`
  - `global_delta_baseline`
  - `perturbation_mean_delta_baseline`
  - `cell_context_knn_delta_baseline`
  - `ridge_regression_baseline`
- Built `results/tables/baseline_run_matrix.csv` directly from `results/tables/split_audit.csv`.
- Restricted scheduling to `track == signature` and `status == ready`.
- Enforced the agreed gene-space policy:
  - within-dataset splits use native genes
  - `dataset_heldout_split` and `external_holdout` use the fixed global 7,880-gene intersection
- Saved the global intersection to `results/tables/phase05_global_gene_intersection_7880.txt`.

## Engineering notes
- The first full campaign attempt exposed two real bottlenecks:
  - repeated materialization of the same split for each of the 5 models
  - slow and initially unstable cross-dataset ridge solves
- Fixed both without reducing the scientific target:
  - added split-level caching so one prepared split is reused across all 5 models
  - replaced the repeated ridge solve path with a dual-kernel eigensystem workflow for all alphas in the same run
- This kept the full-gene formulation intact while making the 210-run campaign finishable.

## Outputs produced
- `results/tables/baseline_run_matrix.csv`
- `results/tables/baseline_metrics_by_split.csv`
- `results/tables/model_ranking_instability.csv`
- `results/tables/experiment_registry.csv`
- `results/tables/phase05_global_gene_intersection_7880.txt`
- `results/logs/training/phase05_campaign.log`
- `results/baselines/` prediction bundles for all scheduled runs

## Campaign result summary
- Planned runs: `210`
- Manifest rows with final outputs: `210 / 210`
- Latest manifest-linked registry states:
  - `196` success
  - `14` cached
  - `0` failed
- `baseline_metrics_by_split.csv` rows: `210`
- `model_ranking_instability.csv` rows: `5`

## Ranking highlights
- Overall mean-rank ordering:
  1. `perturbation_mean_delta_baseline` (`mean_rank = 2.1190`)
  2. `ridge_regression_baseline` (`mean_rank = 2.2619`)
  3. `global_delta_baseline` (`mean_rank = 2.4524`)
  4. `cell_context_knn_delta_baseline` (`mean_rank = 2.5000`)
  5. `control_mean_baseline` (`mean_rank = 4.9524`)
- Split-level leader counts across the 42 split contexts:
  - `ridge_regression_baseline`: `21`
  - `global_delta_baseline`: `12`
  - `perturbation_mean_delta_baseline`: `9`
- Family-level patterns:
  - `random_split`: ridge had the strongest mean performance
  - `low_support_split`: ridge had the strongest mean performance
  - `unseen_perturbation_split`: ridge had the strongest mean performance
  - `unseen_combination_split` on Norman: perturbation-mean was clearly strongest
  - `dataset_heldout_split` and `external_holdout`: `global_delta_baseline` slightly led on mean cosine, while ridge degraded sharply on the external regime

## External-holdout note
- `GSE284197_screen` remained active only in signature families that were marked ready in Phase 04.
- On `external_holdout`, ridge produced much lower non-control cosine than the simpler global-delta baseline, which is an important carry-forward observation for Phase 06.

## Acceptance checklist
- [x] `results/tables/baseline_run_matrix.csv` exists and enumerates `210` planned runs.
- [x] All 5 core baselines are implemented from a single CLI.
- [x] Each run produced logs, registry entries, compact prediction bundles, and metrics.
- [x] Only ready signature splits were scheduled.
- [x] Within-dataset runs used native genes.
- [x] Cross-dataset runs used the fixed 7,880-gene intersection.
- [x] Validation was used for `k` and `alpha` selection.
- [x] `baseline_metrics_by_split.csv` and `model_ranking_instability.csv` were produced.
- [x] No Phase 06 metric expansion, PTL modeling, or manuscript work was started.
