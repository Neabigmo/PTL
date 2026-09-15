"""Low-capacity source-only heterogeneity and predictability summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def source_only_features(frame: pd.DataFrame) -> list[str]:
    candidates = (
        "uq_mean_gene_variance", "uq_median_gene_variance", "uq_top_effect_variance",
        "uq_cosine_disagreement", "uq_effect_norm_variance", "prediction_norm",
        "prediction_sparsity", "prediction_concentration", "prediction_manifold_distance",
        "exact_training_support_fraction", "component_training_support_fraction",
        "training_neighborhood_density", "perturbation_seen_fraction", "component_seen_fraction",
        "combination_novelty", "perturbation_novelty",
    )
    # A feature is eligible only when it is observed for the bulk of the
    # source-only universe.  This prevents a partially materialized legacy
    # feature table from silently shrinking the analysis to a hand-picked
    # subset of labels.
    return [column for column in candidates if column in frame.columns and frame[column].notna().mean() >= 0.80]


def _rank_correlation(predicted: np.ndarray, observed: np.ndarray) -> float:
    if len(predicted) < 3 or np.std(predicted) == 0.0 or np.std(observed) == 0.0:
        return float("nan")
    return float(pd.Series(predicted).corr(pd.Series(observed), method="spearman"))


def fit_source_only_models(
    data: pd.DataFrame,
    *,
    outcome: str = "identifiable_divergence",
    group_columns: tuple[str, ...] = ("source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget_label"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ridge and source-U-only baselines with leave-one-label-out folds."""

    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = source_only_features(data)
    required = set(group_columns) | {"perturbation_label", outcome} | set(features)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"heterogeneity data missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for key, group in data.groupby(list(group_columns), sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        group = group.dropna(subset=[outcome] + features).copy()
        if len(group) < 6 or group["perturbation_label"].nunique() < 4:
            rows.append({**dict(zip(group_columns, key)), "model": "unavailable", "status": "too_few_independent_labels", "n_labels": int(group["perturbation_label"].nunique())})
            continue
        group = group.groupby("perturbation_label", sort=True)[features + [outcome]].mean().reset_index()
        y = group[outcome].to_numpy(dtype=float)
        model_specs = {
            "source_u_only": ["uq_cosine_disagreement"] if "uq_cosine_disagreement" in features else features[:1],
            "ridge_source_features": features,
        }
        for model_name, model_features in model_specs.items():
            predictions = np.full(len(group), np.nan, dtype=float)
            for heldout in range(len(group)):
                train = np.arange(len(group)) != heldout
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(group.loc[train, model_features], y[train])
                predictions[heldout] = float(model.predict(group.loc[[heldout], model_features])[0])
            high = y >= np.median(y)
            auc = float(roc_auc_score(high.astype(int), predictions)) if len(np.unique(high)) == 2 else float("nan")
            record = {
                **dict(zip(group_columns, key)),
                "model": model_name,
                "status": "executed",
                "n_labels": int(len(group)),
                "spearman": _rank_correlation(predictions, y),
                "auroc_high_identifiable": auc,
                "mean_absolute_error": float(np.mean(np.abs(predictions - y))),
                "observed_positive_fraction": float(np.mean(y > 0.0)),
                "predicted_mean": float(np.mean(predictions)),
                "outcome_not_used_as_feature": True,
            }
            rows.append(record)
            fitted = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(group[model_features], y)
            coefficients.extend(
                {
                    **dict(zip(group_columns, key)),
                    "model": model_name,
                    "feature": feature,
                    "coefficient": float(fitted[-1].coef_[index]),
                    "n_labels": int(len(group)),
                }
                for index, feature in enumerate(model_features)
            )
    return pd.DataFrame(rows), pd.DataFrame(coefficients)
