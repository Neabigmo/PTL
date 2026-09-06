# Metric Dictionary

Last synchronized: 2026-09-05 Asia/Shanghai

## Scope
- This file defines the current evaluation metrics for the expanded signature-track baseline campaign.
- The canonical evaluation universe is the `525` manifest-linked runs in `results/tables/baseline_run_matrix.csv` and `results/tables/all_metrics.csv`.
- `experiment_registry.csv` is used only to recover the latest status per `run_id`; smoke runs and non-manifest extras are excluded from formal evaluation.
- The current cross-dataset gene space is the fixed `6,257`-gene intersection. The older `7,880`-gene Phase 05 artifact is historical provenance only.

## Evaluation units

### Per-signature
- One row from `test_metadata.parquet` aligned to one row in `test_predictions.npz`.
- These metrics are computed on the true and predicted delta-expression vectors for that signature.

### Per-run aggregate
- One formal `run_id`.
- Per-signature values are aggregated separately for:
  - `non_control`
  - `control`
  - `overall`
- The primary ranking slice is always `non_control`.

### Per-run distributional
- One formal `run_id`.
- The object of comparison is the full test-set predicted distribution versus the full test-set true distribution on the `non_control` slice.
- The train-set non-control truth matrix defines the shared `StandardScaler + PCA` embedding used before computing the distributional metrics.

### Cross-run aggregate
- Collections of runs compared by model, split family, heldout target, and seed.
- Used for stress-drop, transportability, selective-risk, and pairwise model comparison summaries.

## Edge-case rules
- Two constant vectors that are exactly equal:
  - `pearson_r = 1.0`
  - `spearman_r = 1.0`
- Any constant-vector mismatch:
  - `pearson_r = 0.0`
  - `spearman_r = 0.0`
- Empty top-K overlap:
  - overlap and direction metrics are recorded as `0.0`
- `NaN` or `inf` inputs:
  - must not be silently ignored
  - should be surfaced through audit or fail-fast behavior

## Per-signature deterministic metrics

### `cosine_similarity`
- Cosine similarity between predicted and true delta vectors.
- This is the main atomic fidelity metric carried forward into most Phase 06 summaries.

### `pearson_r`
- Pearson correlation between predicted and true delta vectors.
- Captures linear agreement in signed gene-level deviations.

### `spearman_r`
- Spearman rank correlation between predicted and true delta vectors.
- Captures monotonic agreement even when exact amplitudes differ.

### `rmse`
- Root mean squared error across all genes in the current gene space.

### `mae`
- Mean absolute error across all genes in the current gene space.

### `delta_norm_true`
- Euclidean norm of the true delta vector.
- Used to keep effect-size context visible.

### `delta_norm_pred`
- Euclidean norm of the predicted delta vector.
- Useful for spotting shrinkage or over-amplification.

## Top-K and ranking metrics
- Default `K` values: `20`, `50`, `100`
- Ranking basis: `abs(delta)`

### `top_deg_overlap_at_k`
- Jaccard overlap between the true and predicted top-K absolute-delta gene sets.

### `deg_direction_consistency_at_k`
- On the top-K overlap set, proportion of genes where `sign(y_pred) == sign(y_true)`.

### `recall_at_k`
- Treat the true top-K absolute-delta genes as relevant items and measure how many appear in the predicted top-K ranking.

### `ndcg_at_k`
- Normalized discounted cumulative gain using the predicted ranking against the binary relevance defined by the true top-K gene set.

### `map_at_k`
- Mean average precision at K under the same binary relevance definition.

## Per-run aggregate metrics

### Primary run-level ranking metric
- `mean_cosine_non_control`
- This remains the main ranking metric for benchmark summaries.

### Additional required run-level aggregates
- `median_cosine_non_control`
- `mean_pearson_non_control`
- `mean_spearman_non_control`
- `rmse_non_control_mean`
- `mae_non_control_mean`
- `mean_top_deg_overlap_at_{20,50,100}_non_control`
- `mean_deg_direction_consistency_at_{20,50,100}_non_control`
- `mean_recall_at_{20,50,100}_non_control`
- `mean_ndcg_at_{20,50,100}_non_control`
- `mean_map_at_{20,50,100}_non_control`
- Control-slice counterparts of the same metrics
- `n_test_non_control`
- `n_test_control`

## Distributional metrics
- These are run-level metrics computed only on the `non_control` test slice.
- The embedding is fitted on `train_non_control y_true` and then reused for `test y_true` and `test y_pred`.

### `mmd_linear`
- Linear-kernel maximum mean discrepancy between embedded true and predicted test distributions.
- Lower is better.

### `energy_distance`
- Energy distance between embedded true and predicted test distributions.
- Lower is better.

### `sliced_wasserstein_128`
- Sliced Wasserstein distance estimated over `128` random projections in the common embedding.
- Lower is better.

### `distributional_status`
- `ready` when the distributional metrics were computed.
- `unsupported` only when sample-size or embedding preconditions fail.

## Stress-drop and transportability

### Anchor rule
- Within-dataset stress families:
  - anchor = same `dataset_id`, same `seed`, same `model`, `random_split`
- Cross-dataset families (`dataset_heldout_split`, `external_holdout`):
  - anchor = same heldout dataset, same `seed`, same `model`, `random_split`
  - comparison must happen in the fixed global `6,257`-gene space

### `random_to_stress_drop`
- Defined as:
  - `anchor_mean_cosine_non_control - stress_mean_cosine_non_control`
- Larger values mean larger degradation relative to the random anchor.

### `stress_retention`
- Defined as:
  - `stress_mean_cosine_non_control / max(anchor_mean_cosine_non_control, 1e-8)`
- Higher values mean better retention under stress.

### `transportable`
- Per-signature binary label for stress-run non-control signatures.
- A signature is marked transportable when:
  - `cosine_similarity >= 0.8 * anchor_signature_median_cosine`

### `false_transportability_rate_at_tau`
- At confidence threshold `tau`, among accepted stress signatures with known transportability labels, the proportion that are actually non-transportable.

## Confidence and selective-risk metrics
- Confidence is model-specific and always normalized to `[0, 1]`.
- Main risk definition:
  - `risk = 1 - cosine_similarity`
- Main selective-risk analysis is performed on `non_control` signatures.

### `coverage_at_tau`
- Fraction of non-control signatures with `confidence >= tau`.

### `acceptance_rate_at_tau`
- Same operational meaning as coverage for the accepted set, retained for reporting clarity.

### `selective_risk_at_tau`
- Mean risk over accepted non-control signatures at threshold `tau`.

### `abstention_gain_at_tau`
- Difference between full-coverage risk and selective risk at threshold `tau`.
- At full coverage it should return to approximately `0`.

### `area_under_selective_risk_curve`
- Trapezoidal area under the selective-risk values over the default threshold grid.
- Lower is better.

### `area_under_coverage_curve`
- Trapezoidal area under the coverage curve over the default threshold grid.
- Higher means the model retains more accepted signatures across thresholds.

## Confidence model definitions

### `control_mean_baseline`
- Confidence is fixed at `0.0`.
- Used only as a degenerate reference.

### `global_delta_baseline`
- Confidence is based on reference-profile similarity to the training reference centroid, multiplied by a support scaling factor.

### `perturbation_mean_delta_baseline`
- Exact perturbation seen in train: highest confidence.
- Combination synthesized from seen components: intermediate confidence.
- Global fallback: low confidence.
- Then adjusted by support and reference similarity.

### `cell_context_knn_delta_baseline`
- Confidence follows neighbor similarity and effective support.
- Global fallback is explicitly down-weighted.

### `ridge_regression_baseline`
- Confidence is based on encoded feature-space similarity to the training design plus an `alpha`-dependent shrink adjustment.

## Statistical comparison outputs

### `model_pairwise_comparisons.csv`
- Pairwise matched-model comparisons across common contexts.
- Main compared quantities:
  - `mean_cosine_non_control`
  - `random_to_stress_drop`

### P-values and correction
- Raw paired nonparametric test p-values are recorded.
- `statsmodels.stats.multitest.multipletests(..., method="fdr_bh")` is used for FDR correction.

### Interval estimates
- Wilson intervals are used for:
  - acceptance rate
  - coverage
  - false transportability rate

## Interpretation guardrails
- `mean_cosine_non_control` remains the main benchmark ranking metric, but it is not the whole story.
- Cross-dataset and external-holdout analyses should always be read together with:
  - `random_to_stress_drop`
  - `stress_retention`
  - `false_transportability_rate_at_tau`
  - distributional metrics
- Control-slice values are sanity checks, not the basis of model ranking.
