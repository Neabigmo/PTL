"""Evaluate a source-only feature ladder against held-out reordering burden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reordering_predictability import _prediction_features  # noqa: E402


FEATURE_GROUPS = {
    "U": ["uq_cosine_disagreement", "uq_mean_gene_variance", "uq_effect_norm_variance"],
    "G": ["prediction_norm", "prediction_sparsity", "prediction_concentration", "prediction_manifold_distance"],
    "S": ["exact_training_support_fraction", "component_training_support_fraction", "training_neighborhood_density", "perturbation_seen_fraction", "component_seen_fraction"],
    "N": ["combination_novelty", "perturbation_novelty"],
}
LADDERS = ("U", "U+G", "U+G+S", "U+G+S+N")


def _folds(n: int, *, repeat: int, folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260912 + repeat)
    order = rng.permutation(n)
    groups = np.array_split(order, folds)
    all_index = np.arange(n)
    return [(np.setdiff1d(all_index, test, assume_unique=False), test) for test in groups if len(test)]


def _select_alpha(x: np.ndarray, y: np.ndarray, *, repeat: int) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = (.1, 1.0, 10.0)
    scores: dict[float, list[float]] = {alpha: [] for alpha in candidates}
    for train, test in _folds(len(y), repeat=repeat + 97, folds=4):
        for alpha in candidates:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[train], y[train])
            scores[alpha].append(mean_absolute_error(y[test], model.predict(x[test])))
    return min(candidates, key=lambda alpha: float(np.mean(scores[alpha])))


def _evaluate(frame: pd.DataFrame, features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score, mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = frame[features + ["burden"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    features = [feature for feature in features if data[feature].nunique(dropna=True) > 1]
    if not features:
        return {"spearman": float("nan"), "mae": float("nan")}, {"auprc": float("nan"), "positive_prevalence": float("nan")}
    data = data[features + ["burden"]]
    x = data[features].to_numpy(dtype=float)
    y = data["burden"].to_numpy(dtype=float)
    continuous: list[dict[str, float]] = []
    classification: list[dict[str, float]] = []
    for repeat in range(10):
        for train, test in _folds(len(y), repeat=repeat):
            alpha = _select_alpha(x[train], y[train], repeat=repeat)
            regression = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            regression.fit(x[train], y[train])
            prediction = regression.predict(x[test])
            correlation = float(pd.Series(prediction).corr(pd.Series(y[test]), method="spearman")) if np.unique(prediction).size > 1 and np.unique(y[test]).size > 1 else float("nan")
            continuous.append({"spearman": correlation, "mae": float(mean_absolute_error(y[test], prediction))})
            threshold = float(np.quantile(y[train], 2.0 / 3.0))
            train_binary = y[train] >= threshold
            test_binary = y[test] >= threshold
            if train_binary.min() == train_binary.max() or test_binary.min() == test_binary.max():
                continue
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=500))
            classifier.fit(x[train], train_binary)
            probability = classifier.predict_proba(x[test])[:, 1]
            classification.append({"auprc": float(average_precision_score(test_binary, probability)), "positive_prevalence": float(np.mean(test_binary))})
    continuous_out = {key: float(np.nanmean([row[key] for row in continuous])) if np.isfinite([row[key] for row in continuous]).any() else float("nan") for key in ("spearman", "mae")}
    classification_out = {
        "auprc": float(np.nanmean([row["auprc"] for row in classification])) if classification else float("nan"),
        "positive_prevalence": float(np.nanmean([row["positive_prevalence"] for row in classification])) if classification else float("nan"),
    }
    return continuous_out, classification_out


def run(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    items = pd.read_csv(manifests / "reliability_transport_measurement_depth_matched_fixed_items.csv", dtype={"cell_budget_label": "string"})
    items = items.loc[items["cell_budget_label"].eq("full")].copy()
    items = items.loc[
        items["source_environment_id"].eq(items["left_target_environment_id"])
        | items["source_environment_id"].eq(items["right_target_environment_id"])
    ].copy()
    burden = items.groupby(
        ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "perturbation_label"], as_index=False
    )["identifiable_divergence"].mean().rename(columns={"identifiable_divergence": "burden"})
    features = pd.read_csv(root / "artifacts/source_data/formal_v2_reliability_predictions.csv")
    features = features.loc[features["scenario"].eq("in_domain") & features["predictor"].eq("mean_matching")].copy()
    features = features.groupby(["environment_key", "perturbation_label"], as_index=False)[sum(FEATURE_GROUPS.values(), [])].mean(numeric_only=True)
    derived = _prediction_features(root)
    derived = derived.rename(columns={
        "prediction_norm_derived": "prediction_norm", "prediction_sparsity_derived": "prediction_sparsity",
        "prediction_concentration_derived": "prediction_concentration", "uq_mean_gene_variance_derived": "uq_mean_gene_variance",
        "uq_cosine_disagreement_derived": "uq_cosine_disagreement", "uq_effect_norm_variance_derived": "uq_effect_norm_variance",
    })
    derived = derived[["environment_key", "perturbation_label", "prediction_norm", "prediction_sparsity", "prediction_concentration", "uq_mean_gene_variance", "uq_cosine_disagreement", "uq_effect_norm_variance"]]
    data = burden.merge(features, left_on=["source_environment_id", "perturbation_label"], right_on=["environment_key", "perturbation_label"], how="inner")
    data = data.merge(derived, left_on=["source_environment_id", "perturbation_label"], right_on=["environment_key", "perturbation_label"], how="left", suffixes=("_legacy", ""))
    rows: list[dict[str, Any]] = []
    for key, group in data.groupby(["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"], sort=True):
        for ladder in LADDERS:
            groups = ladder.split("+")
            selected = [feature for group_name in groups for feature in FEATURE_GROUPS[group_name] if feature in group.columns and group[feature].nunique(dropna=True) > 1]
            usable = group[selected + ["burden"]].dropna()
            if len(usable) < 25:
                continue
            continuous, classification = _evaluate(usable, selected)
            common = {
                "source_environment_id": key[0], "left_target_environment_id": key[1], "right_target_environment_id": key[2], "metric": key[3],
                "feature_ladder": ladder, "feature_columns": ";".join(selected), "n_labels": int(len(usable)),
                "outer_protocol": "10 repeats × 5 label-disjoint folds; nested ridge alpha selection", "target_feature_leakage": False,
            }
            rows.append({**common, "task": "continuous_burden", "spearman": continuous["spearman"], "mae": continuous["mae"], "auprc": float("nan"), "positive_prevalence": float("nan"), "auprc_lift": float("nan")})
            rows.append({**common, "task": "high_burden_top_tercile", "spearman": float("nan"), "mae": float("nan"), "auprc": classification["auprc"], "positive_prevalence": classification["positive_prevalence"], "auprc_lift": classification["auprc"] - classification["positive_prevalence"]})
    output = manifests / "reviewer_prospective_feature_ladder.csv"
    report_path = manifests / "reviewer_prospective_feature_ladder.json"
    result = pd.DataFrame(rows).sort_values(["task", "feature_ladder", "source_environment_id", "metric"], kind="stable").reset_index(drop=True)
    result.to_csv(output, index=False)
    report = {
        "status": "executed", "rows": int(len(result)), "feature_ladders": list(LADDERS),
        "tasks": ["continuous_burden", "high_burden_top_tercile"],
        "source_only_protocol": "features are joined only on the frozen source environment and perturbation label; target burden is evaluation-only",
        "outer_validation": "10 repeats × 5 label-disjoint folds with nested ridge selection",
        "output": output.relative_to(root).as_posix(),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
