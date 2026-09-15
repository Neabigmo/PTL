"""Probe environment/dataset identity from deployment features only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_reliability import (  # noqa: E402
    C_CATEGORICAL_COLUMNS, FEATURE_COLUMNS, N_COLUMNS, P_COLUMNS, S_COLUMNS, UQ_COLUMNS,
    validate_deployment_features,
)


def _family(dataset_id: str) -> str:
    return dataset_id.split("Weissman", 1)[0].split("Kampmann", 1)[0].split("Izar", 1)[0]


def _probe(frame: pd.DataFrame, feature_columns: list[str], target: str) -> tuple[dict[str, Any], pd.DataFrame]:
    x = frame[feature_columns].copy()
    categorical = [column for column in feature_columns if column in C_CATEGORICAL_COLUMNS]
    numeric = [column for column in feature_columns if column not in categorical]
    for column in categorical:
        x[column] = x[column].astype(str)
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)),
    ])
    groups = frame["biological_instance_id"].astype(str)
    split_count = min(5, int(groups.nunique()))
    if split_count < 2:
        raise ValueError("identity probe requires at least two biological-instance groups")
    predictions = pd.Series(index=frame.index, dtype="object")
    splitter = GroupKFold(n_splits=split_count)
    for train_index, test_index in splitter.split(x, frame[target], groups):
        model.fit(x.iloc[train_index], frame[target].iloc[train_index])
        predictions.iloc[test_index] = model.predict(x.iloc[test_index])
    truth = frame[target].astype(str)
    labels = sorted(truth.unique())
    matrix = confusion_matrix(truth, predictions.astype(str), labels=labels)
    confusion_rows = []
    for row_index, actual in enumerate(labels):
        for column_index, predicted in enumerate(labels):
            confusion_rows.append({"target": target, "actual": actual, "predicted": predicted, "count": int(matrix[row_index, column_index])})
    return {
        "target": target,
        "feature_block": "+".join(feature_columns),
        "n_rows": int(len(frame)),
        "n_biological_instance_groups": int(groups.nunique()),
        "grouped_cv_folds": split_count,
        "accuracy": float(accuracy_score(truth, predictions.astype(str))),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions.astype(str))),
        "target_outcome_free": 1,
        "explicit_id_free": 1,
        "deployment_safe": 1,
    }, pd.DataFrame(confusion_rows)


def run(root: Path, metrics_path: Path, confusion_path: Path, summary_path: Path) -> dict[str, Any]:
    validate_deployment_features(FEATURE_COLUMNS)
    features = pd.read_csv(root / "artifacts/source_data/formal_v2_reliability_deployment_features.csv")
    registry = pd.read_csv(root / "artifacts/manifests/environment_registry.csv")
    mapping = registry.set_index("environment_key")["dataset_id"].astype(str).to_dict()
    features["dataset_family"] = features["environment_key"].map(mapping).map(_family)
    if features["dataset_family"].isna().any():
        raise ValueError("identity probe has environment keys absent from the canonical registry")
    rows = []
    confusion = []
    blocks = {
        "U+P+S+N": list(UQ_COLUMNS + P_COLUMNS + S_COLUMNS + N_COLUMNS),
        "U+P+S+N+C": list(UQ_COLUMNS + P_COLUMNS + S_COLUMNS + N_COLUMNS + C_CATEGORICAL_COLUMNS),
    }
    for block_name, columns in blocks.items():
        for target in ("environment_key", "dataset_family"):
            metrics, matrix = _probe(features, columns, target)
            metrics["feature_block_name"] = block_name
            rows.append(metrics)
            confusion.append(matrix.assign(feature_block_name=block_name))
    metrics_frame = pd.DataFrame(rows)
    confusion_frame = pd.concat(confusion, ignore_index=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(metrics_path, index=False)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_frame.to_csv(confusion_path, index=False)
    result = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "policy": "explicit-ID-free and target-outcome-free deployment feature probe with grouped CV by biological instance",
        "metrics_path": metrics_path.relative_to(root).as_posix(),
        "confusion_path": confusion_path.relative_to(root).as_posix(),
        "status": "identity_probe_executed_deployment_safe",
        "rows": metrics_frame.to_dict("records"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--confusion", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(
        root,
        (args.metrics or root / "artifacts/manifests/formal_v2_identity_probe_metrics.csv").resolve(),
        (args.confusion or root / "artifacts/manifests/formal_v2_identity_probe_confusion.csv").resolve(),
        (args.summary or root / "artifacts/manifests/formal_v2_identity_probe_summary.json").resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
