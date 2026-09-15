"""Prospective source-only predictability metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_source_only_predictability(
    frame: pd.DataFrame,
    *,
    outcome: str = "identifiable_divergence",
    group_columns: tuple[str, ...] = ("source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget_label"),
) -> pd.DataFrame:
    """Compare source-U-only, linear ridge, and frozen PTL feature sets.

    The outer split is leave-one-perturbation-label-out.  Features are
    source-context quantities only; target risk is used exclusively as the
    held-out evaluation value.
    """

    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_candidates = [
        "uq_cosine_disagreement", "uq_mean_gene_variance", "uq_median_gene_variance", "uq_top_effect_variance", "uq_effect_norm_variance",
        "prediction_norm", "prediction_sparsity", "prediction_concentration", "prediction_manifold_distance",
        "exact_training_support_fraction", "component_training_support_fraction", "training_neighborhood_density",
        "perturbation_seen_fraction", "component_seen_fraction", "combination_novelty", "perturbation_novelty",
    ]
    feature_columns = [column for column in feature_candidates if column in frame.columns and frame[column].notna().mean() >= 0.80]
    required = set(group_columns) | {"perturbation_label", outcome} | set(feature_columns)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"predictability input missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_columns), sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        group = group.groupby("perturbation_label", sort=True)[feature_columns + [outcome]].mean().dropna()
        if len(group) < 6:
            rows.append({**dict(zip(group_columns, key)), "status": "too_few_independent_labels", "model": "unavailable", "n_labels": int(len(group))})
            continue
        y = group[outcome].to_numpy(dtype=float)
        feature_sets = {
            "source_u_only": ["uq_cosine_disagreement"] if "uq_cosine_disagreement" in feature_columns else feature_columns[:1],
            "linear_source_features": feature_columns,
        }
        for model_name, selected_features in feature_sets.items():
            prediction = np.full(len(group), np.nan, dtype=float)
            for heldout in range(len(group)):
                train = np.arange(len(group)) != heldout
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(group.iloc[train][selected_features], y[train])
                prediction[heldout] = float(model.predict(group.iloc[[heldout]][selected_features])[0])
            # Every held-out score is scaled and thresholded from the
            # training fold only.  There is deliberately no post-hoc Brier
            # score: a global min/max or target median would leak the held-out
            # outcome into calibration.
            scores = np.full(len(group), np.nan, dtype=float)
            observed_high = np.full(len(group), np.nan, dtype=float)
            for heldout in range(len(group)):
                train = np.arange(len(group)) != heldout
                train_pred = prediction[train]
                lo, hi = float(np.min(train_pred)), float(np.max(train_pred))
                scores[heldout] = (prediction[heldout] - lo) / (hi - lo or 1.0)
                observed_high[heldout] = float(y[heldout] >= np.median(y[train]))
            valid_auc = np.isfinite(scores) & np.isfinite(observed_high)
            auc = float(roc_auc_score(observed_high[valid_auc], scores[valid_auc])) if len(np.unique(observed_high[valid_auc])) == 2 else float("nan")
            rows.append({
                **dict(zip(group_columns, key)),
                "model": model_name,
                "status": "executed",
                "n_labels": int(len(group)),
                "spearman": float(pd.Series(prediction).corr(pd.Series(y), method="spearman")),
                "auroc_high_identifiable": auc,
                "calibration_brier": float("nan"),
                "calibration_threshold": float("nan"),
                "calibration_status": "not_reported_train_fold_only_no_posthoc_target_calibration",
                "mean_absolute_error": float(np.mean(np.abs(prediction - y))),
                "source_only_features": ";".join(selected_features),
                "target_outcome_used_only_for_evaluation": True,
            })
    return pd.DataFrame(rows)
