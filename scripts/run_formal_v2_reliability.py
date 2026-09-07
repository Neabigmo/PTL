"""Run the leakage-safe formal-v2 PTL transfer evaluation.

The runner consumes only the new formal-v2 validation/test prediction arrays.
Validation rows calibrate PTL; test rows are used once for the final selective
risk metrics. Predictor and environment IDs are retained as audit columns but
are never included in the headline deployment feature matrix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_predictors import (  # noqa: E402
    METADATA_COLUMNS,
    components,
    load_manifest,
    load_panel,
    load_registry,
    read_ground_truth,
)
from src.evaluation.metrics import safe_rowwise_cosine  # noqa: E402
from src.ptl.data.contracts import validate_feature_columns  # noqa: E402
from src.ptl.uncertainty.uq import UQNormalizer, summarize_ensemble  # noqa: E402


UQ_COLUMNS = (
    "uq_mean_gene_variance",
    "uq_median_gene_variance",
    "uq_top_effect_variance",
    "uq_cosine_disagreement",
    "uq_effect_norm_variance",
)
P_COLUMNS = (
    "prediction_norm",
    "prediction_sparsity",
    "prediction_concentration",
    "prediction_manifold_distance",
)
S_COLUMNS = ("support_cells", "support_signatures", "reference_key_overlap")
N_COLUMNS = (
    "perturbation_seen_fraction",
    "component_seen_fraction",
    "combination_novelty",
    "perturbation_novelty",
)
C_NUMERIC_COLUMNS = ("batch_distance",)
C_CATEGORICAL_COLUMNS = (
    "cell_context",
    "perturbation_modality",
    "readout_modality",
    "condition",
    "platform",
)
FEATURE_COLUMNS = UQ_COLUMNS + P_COLUMNS + S_COLUMNS + N_COLUMNS + C_NUMERIC_COLUMNS + C_CATEGORICAL_COLUMNS
PREDICTORS = ("mean_matching", "strong_linear")
SCENARIOS = ("in_domain", "leave_environment_out", "leave_predictor_out")
REFERENCE_KEY_PATTERN = re.compile(r"[,;|+]+")


def validate_deployment_features(columns: tuple[str, ...] | list[str]) -> None:
    """Validate that the formal feature surface contains no answer keys or IDs."""

    validate_feature_columns(list(columns))
    forbidden_identity = {"environment_id", "environment_key", "predictor", "predictor_id"}
    violations = forbidden_identity.intersection(columns)
    if violations:
        raise ValueError(f"formal deployment features contain identity columns: {sorted(violations)}")


def scenario_partition(frame: pd.DataFrame, scenario: str, heldout: str) -> tuple[pd.Index, pd.Index]:
    """Return validation-train and test indices for one frozen transfer scenario."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    frame = frame.copy()
    if scenario == "in_domain":
        train_mask = frame["split"].eq("validation")
        test_mask = frame["split"].eq("test")
    elif scenario == "leave_environment_out":
        if heldout not in set(frame["environment_id"].astype(str)):
            raise ValueError(f"unknown held-out environment: {heldout}")
        train_mask = frame["split"].eq("validation") & frame["environment_id"].ne(heldout)
        test_mask = frame["split"].eq("test") & frame["environment_id"].eq(heldout)
    else:
        if heldout not in set(frame["predictor"].astype(str)):
            raise ValueError(f"unknown held-out predictor: {heldout}")
        train_mask = frame["split"].eq("validation") & frame["predictor"].ne(heldout)
        test_mask = frame["split"].eq("test") & frame["predictor"].eq(heldout)
    train = frame.index[train_mask]
    test = frame.index[test_mask]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"empty validation/test partition for {scenario} heldout={heldout!r}")
    train_ids = set(frame.loc[train, "biological_instance_id"].astype(str))
    test_ids = set(frame.loc[test, "biological_instance_id"].astype(str))
    if train_ids.intersection(test_ids):
        raise AssertionError("biological instances overlap between PTL calibration and test rows")
    return train, test


def _reference_tokens(value: object) -> set[str]:
    return {token.strip() for token in REFERENCE_KEY_PATTERN.split(str(value)) if token.strip()}


def _fold_local_one_hot(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_values = np.asarray(train_values, dtype=str)
    values = np.asarray(values, dtype=str)
    if train_values.ndim == 1:
        train_values = train_values.reshape(-1, 1)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    train_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for column in range(train_values.shape[1]):
        categories = sorted(set(train_values[:, column]))
        lookup = {category: index for index, category in enumerate(categories)}
        train_encoded = np.zeros((len(train_values), len(categories)), dtype=np.float64)
        value_encoded = np.zeros((len(values), len(categories)), dtype=np.float64)
        for row, category in enumerate(train_values[:, column]):
            train_encoded[row, lookup[category]] = 1.0
        for row, category in enumerate(values[:, column]):
            index = lookup.get(category)
            if index is not None:
                value_encoded[row, index] = 1.0
        train_parts.append(train_encoded)
        value_parts.append(value_encoded)
    if not train_parts:
        return np.empty((len(train_values), 0)), np.empty((len(values), 0))
    return np.column_stack(train_parts), np.column_stack(value_parts)


def _manifold_distance(reference: np.ndarray, queries: np.ndarray, *, exclude_self: bool = False) -> np.ndarray:
    """Compute nearest train-response distance in a train-only PCA space."""

    reference = np.asarray(reference, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1]:
        raise ValueError("manifold inputs must be 2D matrices with matching feature dimensions")
    center = reference.mean(axis=0)
    scale = np.where(reference.std(axis=0) > 1e-8, reference.std(axis=0), 1.0)
    reference_scaled = (reference - center) / scale
    queries_scaled = (queries - center) / scale
    rank = min(8, reference_scaled.shape[0], reference_scaled.shape[1])
    if rank:
        _, _, components = np.linalg.svd(reference_scaled, full_matrices=False)
        reference_latent = reference_scaled @ components[:rank].T
        query_latent = queries_scaled @ components[:rank].T
    else:
        reference_latent = reference_scaled
        query_latent = queries_scaled
    distances = np.empty(len(query_latent), dtype=np.float64)
    for start in range(0, len(query_latent), 256):
        block = query_latent[start:start + 256]
        pairwise = np.sqrt(np.maximum(
            ((block[:, None, :] - reference_latent[None, :, :]) ** 2).sum(axis=2), 0.0
        ))
        if exclude_self and len(reference) == len(queries):
            rows = np.arange(len(block))
            pairwise[rows, start + rows] = np.inf
        distances[start:start + len(block)] = pairwise.min(axis=1)
    finite = np.isfinite(distances)
    fallback = float(np.median(distances[finite])) if finite.any() else 0.0
    distances[~finite] = fallback
    return distances


def _numeric_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        fill = float(series.median()) if series.notna().any() else 0.0
        values.append(series.fillna(fill).to_numpy(dtype=np.float64))
    return np.column_stack(values) if values else np.empty((len(frame), 0), dtype=np.float64)


def _normalized_uq(train_frame: pd.DataFrame, query_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each UQ field using only validation rows, by predictor when possible."""

    train_out = np.empty((len(train_frame), len(UQ_COLUMNS)), dtype=np.float64)
    query_out = np.empty((len(query_frame), len(UQ_COLUMNS)), dtype=np.float64)
    global_predictors = train_frame["predictor"].astype(str)
    for column_index, column in enumerate(UQ_COLUMNS):
        global_normalizer = UQNormalizer.fit(pd.to_numeric(train_frame[column], errors="coerce").fillna(0.0))
        for predictor in sorted(set(global_predictors)):
            train_mask = global_predictors.eq(predictor).to_numpy()
            query_mask = query_frame["predictor"].astype(str).eq(predictor).to_numpy()
            reference = pd.to_numeric(train_frame.loc[train_mask, column], errors="coerce").fillna(0.0).to_numpy()
            normalizer = UQNormalizer.fit(reference) if len(reference) else global_normalizer
            train_values = pd.to_numeric(train_frame.loc[train_mask, column], errors="coerce").fillna(0.0).to_numpy()
            query_values = pd.to_numeric(query_frame.loc[query_mask, column], errors="coerce").fillna(0.0).to_numpy()
            train_out[train_mask, column_index] = normalizer.transform(train_values)
            if query_mask.any():
                query_out[query_mask, column_index] = normalizer.transform(query_values)
        unseen_query = ~query_frame["predictor"].astype(str).isin(set(global_predictors)).to_numpy()
        if unseen_query.any():
            query_values = pd.to_numeric(query_frame.loc[unseen_query, column], errors="coerce").fillna(0.0).to_numpy()
            query_out[unseen_query, column_index] = global_normalizer.transform(query_values)
    return train_out, query_out


def _feature_matrix(
    train_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_vectors = np.stack(train_frame["prediction_vector"].to_numpy())
    query_vectors = np.stack(query_frame["prediction_vector"].to_numpy())
    train_manifold = _manifold_distance(train_vectors, train_vectors, exclude_self=True)
    query_manifold = _manifold_distance(train_vectors, query_vectors)
    train_work = train_frame.copy()
    query_work = query_frame.copy()
    train_work["prediction_manifold_distance"] = train_manifold
    query_work["prediction_manifold_distance"] = query_manifold

    train_uq, query_uq = _normalized_uq(train_work, query_work)
    numeric_columns = P_COLUMNS + S_COLUMNS + N_COLUMNS + C_NUMERIC_COLUMNS
    train_numeric = _numeric_frame(train_work, numeric_columns)
    query_numeric = _numeric_frame(query_work, numeric_columns)
    train_categories, query_categories = _fold_local_one_hot(
        train_work[list(C_CATEGORICAL_COLUMNS)].astype(str).to_numpy(),
        query_work[list(C_CATEGORICAL_COLUMNS)].astype(str).to_numpy(),
    )
    train_features = np.column_stack([train_uq, train_numeric, train_categories])
    query_features = np.column_stack([query_uq, query_numeric, query_categories])
    return train_features, query_features, train_uq, query_uq


def _fit_ptl(train_frame: pd.DataFrame, query_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, str]:
    threshold = 0.8 * float(np.median(train_frame["fidelity"].to_numpy(dtype=float)))
    train_labels = (train_frame["fidelity"].to_numpy(dtype=float) >= threshold).astype(int)
    train_features, query_features, train_uq, query_uq = _feature_matrix(train_frame, query_frame)
    raw_confidence = query_uq[:, UQ_COLUMNS.index("uq_mean_gene_variance")]
    if np.unique(train_labels).size < 2:
        ptl = np.full(len(query_frame), float(train_labels.mean()), dtype=np.float64)
        model_name = "constant_validation_label"
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=20260907,
            n_jobs=1,
        )
        model.fit(train_features, train_labels)
        ptl = model.predict_proba(query_features)[:, 1]
        model_name = "shared_identity_free_random_forest"
    return raw_confidence, ptl, threshold, model_name


def _prediction_rows(root: Path) -> pd.DataFrame:
    evaluation_genes = load_panel(root)
    registry = load_registry(root)
    manifest = load_manifest(root).fillna("")
    manifest_by_id = manifest.set_index("biological_instance_id", drop=False)
    registry_by_id = registry.set_index("environment_id", drop=False)
    frames: list[dict[str, Any]] = []
    ground_truth_by_dataset: dict[str, pd.DataFrame] = {}
    for dataset_id, registry_group in registry.groupby("dataset_id", sort=True):
        source_rows = manifest.loc[manifest["environment_id"].isin(registry_group["environment_id"])]
        source = str(source_rows["ground_truth_path"].iloc[0])
        ground_truth_by_dataset[str(dataset_id)] = read_ground_truth(root, source, evaluation_genes).set_index(
            "biological_instance_id", drop=False
        )

    for registry_row in registry.itertuples(index=False):
        environment_id = str(registry_row.environment_id)
        environment_key = str(registry_row.environment_key)
        dataset_id = str(registry_row.dataset_id)
        environment_manifest = manifest.loc[manifest["environment_id"].eq(environment_id)]
        truth = ground_truth_by_dataset[dataset_id]
        train_manifest = environment_manifest.loc[environment_manifest["split"].eq("train")]
        train_labels = set(train_manifest["perturbation_label"].astype(str))
        train_components = {
            component
            for label in train_labels
            for component in components(label)
        }
        train_reference_keys = {
            key
            for value in train_manifest["source_reference_keys"]
            for key in _reference_tokens(value)
        }
        for predictor in PREDICTORS:
            array_path = root / "results/formal_v2/predictors" / f"{environment_key}__{predictor}.npz"
            if not array_path.is_file():
                raise FileNotFoundError(f"missing formal-v2 prediction array: {array_path}")
            # These arrays are local outputs created by the formal runner; the
            # ID fields were written as object-dtype strings by NumPy.
            with np.load(array_path, allow_pickle=True) as arrays:
                validation_predictions = np.asarray(arrays["validation_predictions"], dtype=np.float32)
                test_predictions = np.asarray(arrays["test_predictions"], dtype=np.float32)
                validation_ids = arrays["validation_biological_instance_ids"].astype(str)
                test_ids = arrays["test_biological_instance_ids"].astype(str)
                model_seeds = arrays["model_seeds"].astype(int).tolist()
                stored_genes = arrays["genes"].astype(str).tolist()
            if stored_genes != evaluation_genes or model_seeds != [0, 1, 2]:
                raise ValueError(f"prediction array contract mismatch: {array_path}")
            if validation_predictions.shape[0] != 3 or test_predictions.shape[0] != 3:
                raise ValueError(f"formal-v2 ensemble must contain three model seeds: {array_path}")
            for split, ids, predictions in (
                ("validation", validation_ids, validation_predictions),
                ("test", test_ids, test_predictions),
            ):
                if len(set(ids)) != len(ids):
                    raise ValueError(f"duplicate biological IDs in {array_path} {split}")
                ensemble = summarize_ensemble(predictions)
                mean_prediction = predictions.mean(axis=0)
                for row_index, biological_id in enumerate(ids):
                    if biological_id not in manifest_by_id.index or biological_id not in truth.index:
                        raise ValueError(f"prediction ID is absent from manifest/truth: {biological_id}")
                    metadata = manifest_by_id.loc[biological_id]
                    if str(metadata["environment_id"]) != environment_id or str(metadata["split"]) != split:
                        raise ValueError(f"prediction split/environment mismatch for {biological_id}")
                    truth_vector = truth.loc[biological_id, evaluation_genes].to_numpy(dtype=np.float32)
                    prediction_vector = mean_prediction[row_index]
                    label = str(metadata["perturbation_label"])
                    parts = components(label)
                    known_parts = [part for part in parts if part in train_components]
                    reference_overlap = bool(
                        _reference_tokens(metadata["source_reference_keys"]).intersection(train_reference_keys)
                    )
                    abs_prediction = np.abs(prediction_vector.astype(np.float64))
                    total_abs = float(abs_prediction.sum())
                    top_k = min(50, len(abs_prediction))
                    concentration = float(np.sort(abs_prediction)[-top_k:].sum() / total_abs) if total_abs > 1e-12 else 0.0
                    uq = ensemble
                    def numeric_metadata(name: str) -> float:
                        value = pd.to_numeric(metadata[name], errors="coerce")
                        return float(value) if pd.notna(value) else 0.0

                    frames.append({
                        "biological_instance_id": biological_id,
                        "environment_id": environment_id,
                        "environment_key": environment_key,
                        "dataset_id": dataset_id,
                        "predictor": predictor,
                        "predictor_version": "formal_v2_exact_linear_20260907",
                        "split": split,
                        "perturbation_label": label,
                        "fidelity": float(safe_rowwise_cosine(truth_vector[None, :], prediction_vector[None, :])[0]),
                        "continuous_risk": float(1.0 - safe_rowwise_cosine(truth_vector[None, :], prediction_vector[None, :])[0]),
                        "support_cells": numeric_metadata("total_cells"),
                        "support_signatures": numeric_metadata("n_reference_groups"),
                        "reference_key_overlap": float(reference_overlap),
                        "perturbation_seen_fraction": float(label in train_labels),
                        "component_seen_fraction": float(len(known_parts) / len(parts)) if parts else 1.0,
                        "combination_novelty": float(len(parts) > 1 and label not in train_labels),
                        "perturbation_novelty": float(label not in train_labels),
                        "batch_distance": float(not reference_overlap),
                        "cell_context": str(registry_by_id.loc[environment_id, "cell_context"]),
                        "perturbation_modality": str(registry_by_id.loc[environment_id, "perturbation_modality"]),
                        "readout_modality": str(registry_by_id.loc[environment_id, "readout_modality"]),
                        "condition": str(registry_by_id.loc[environment_id, "condition"]),
                        "platform": str(registry_by_id.loc[environment_id, "platform"]),
                        "prediction_norm": float(np.linalg.norm(prediction_vector)),
                        "prediction_sparsity": float(np.mean(abs_prediction <= 1e-8)),
                        "prediction_concentration": concentration,
                        "prediction_manifold_distance": 0.0,
                        "uq_mean_gene_variance": float(uq.mean_gene_variance[row_index]),
                        "uq_median_gene_variance": float(uq.median_gene_variance[row_index]),
                        "uq_top_effect_variance": float(uq.top_effect_variance[row_index]),
                        "uq_cosine_disagreement": float(uq.cosine_disagreement[row_index]),
                        "uq_effect_norm_variance": float(uq.effect_norm_variance[row_index]),
                        "prediction_vector": prediction_vector,
                    })
    frame = pd.DataFrame(frames)
    if frame.empty:
        raise ValueError("no formal-v2 reliability rows were materialized")
    return frame.reset_index(drop=True)


def _selective_metrics(risk: np.ndarray, reliable: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    risk = np.asarray(risk, dtype=float)
    reliable = np.asarray(reliable, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    if len(risk) == 0:
        raise ValueError("cannot score an empty test group")
    order = np.argsort(-confidence, kind="mergesort")
    cumulative_risk = np.cumsum(risk[order]) / np.arange(1, len(risk) + 1)
    optimal_order = np.argsort(risk, kind="mergesort")
    optimal_cumulative = np.cumsum(risk[optimal_order]) / np.arange(1, len(risk) + 1)
    row: dict[str, float] = {
        "n": float(len(risk)),
        "aurc": float(cumulative_risk.mean()),
        "optimal_aurc": float(optimal_cumulative.mean()),
        "excess_aurc": float(cumulative_risk.mean() - optimal_cumulative.mean()),
        "brier": float(np.mean((confidence - reliable) ** 2)),
        "mean_continuous_risk": float(risk.mean()),
        "risk_at_50": float(cumulative_risk[max(1, int(np.ceil(0.50 * len(risk)))) - 1]),
        "risk_at_80": float(cumulative_risk[max(1, int(np.ceil(0.80 * len(risk)))) - 1]),
        "ftr_at_50": float(1.0 - reliable[order][:max(1, int(np.ceil(0.50 * len(risk))))].mean()),
        "ftr_at_80": float(1.0 - reliable[order][:max(1, int(np.ceil(0.80 * len(risk))))].mean()),
    }
    try:
        row["log_loss"] = float(log_loss(reliable, np.clip(confidence, 1e-5, 1.0 - 1e-5), labels=[0, 1]))
    except ValueError:
        row["log_loss"] = float("nan")
    if np.unique(reliable).size == 2:
        row["auroc"] = float(roc_auc_score(reliable, confidence))
        row["auprc"] = float(average_precision_score(reliable, confidence))
    else:
        row["auroc"] = float("nan")
        row["auprc"] = float("nan")
    return row


def run(root: Path) -> dict[str, Any]:
    validate_deployment_features(FEATURE_COLUMNS)
    frame = _prediction_rows(root)
    score_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    def execute(scenario: str, heldout: str, subset: pd.DataFrame) -> None:
        train, test = scenario_partition(subset, scenario, heldout)
        train_frame = subset.loc[train].copy()
        test_frame = subset.loc[test].copy()
        raw_confidence, ptl_score, threshold, model_name = _fit_ptl(train_frame, test_frame)
        train_ids = set(train_frame["biological_instance_id"].astype(str))
        if train_ids.intersection(set(test_frame["biological_instance_id"].astype(str))):
            raise AssertionError("PTL train/test biological instance leakage")
        local_score_rows: list[dict[str, Any]] = []
        for row_index, raw, score in zip(test_frame.index, raw_confidence, ptl_score):
            row = test_frame.loc[row_index].drop(labels=["prediction_vector"])
            payload = row.to_dict()
            payload.update({
                "scenario": scenario,
                "heldout_environment_id": heldout if scenario == "leave_environment_out" else "",
                "heldout_predictor": heldout if scenario == "leave_predictor_out" else "",
                "raw_normalized_uq": float(raw),
                "ptl_rf": float(score),
                "reliable_label": int(float(row["fidelity"]) >= threshold),
                "calibration_threshold": float(threshold),
                "calibration_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "ptl_model": model_name,
            })
            score_rows.append(payload)
            local_score_rows.append(payload)
        local_output = pd.DataFrame(local_score_rows)
        for environment_id, group_output in local_output.groupby("environment_id", sort=True):
            for method in ("raw_normalized_uq", "ptl_rf"):
                metrics = _selective_metrics(
                    group_output["continuous_risk"].to_numpy(),
                    group_output["reliable_label"].to_numpy(),
                    group_output[method].to_numpy(),
                )
                metric_rows.append({
                    "aggregation_level": "environment_predictor_test",
                    "scenario": scenario,
                    "heldout_environment_id": heldout if scenario == "leave_environment_out" else "",
                    "heldout_predictor": heldout if scenario == "leave_predictor_out" else "",
                    "environment_id": str(environment_id),
                    "environment_key": str(group_output["environment_key"].iloc[0]),
                    "predictor": (
                        str(group_output["predictor"].iloc[0])
                        if scenario == "leave_predictor_out"
                        else "identity_free_shared"
                    ),
                    "method": method,
                    **metrics,
                })

    execute("in_domain", "", frame)
    for heldout_environment in sorted(frame["environment_id"].unique()):
        execute("leave_environment_out", str(heldout_environment), frame)
    for heldout_predictor in PREDICTORS:
        execute("leave_predictor_out", heldout_predictor, frame)

    score_frame = pd.DataFrame(score_rows).sort_values(
        ["scenario", "environment_id", "predictor", "biological_instance_id"], kind="stable"
    )
    metric_frame = pd.DataFrame(metric_rows)
    metric_columns = [
        "n", "aurc", "optimal_aurc", "excess_aurc", "brier", "mean_continuous_risk",
        "risk_at_50", "risk_at_80", "ftr_at_50", "ftr_at_80", "log_loss", "auroc", "auprc",
    ]
    environment_macro = metric_frame.groupby(["scenario", "method", "environment_id"], as_index=False)[metric_columns].mean()
    macro = environment_macro.groupby(["scenario", "method"], as_index=False)[metric_columns].mean()
    macro.insert(0, "aggregation_level", "environment_macro")
    macro["environment_id"] = "environment_macro"
    macro["environment_key"] = "environment_macro"
    macro["predictor"] = "identity_free_shared"
    macro["heldout_environment_id"] = ""
    macro["heldout_predictor"] = ""
    metric_frame = pd.concat([metric_frame, macro[metric_frame.columns]], ignore_index=True)

    source_dir = root / "artifacts/source_data"
    manifest_dir = root / "artifacts/manifests"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    score_path = source_dir / "formal_v2_reliability_predictions.csv"
    feature_path = source_dir / "formal_v2_reliability_deployment_features.csv"
    metrics_path = manifest_dir / "formal_v2_reliability_metrics.csv"
    summary_path = manifest_dir / "formal_v2_reliability_summary.json"
    score_frame.to_csv(score_path, index=False)
    feature_columns = [
        "prediction_id", "biological_instance_id", "environment_id", "environment_key", "predictor",
        "scenario", "heldout_environment_id", "heldout_predictor", *FEATURE_COLUMNS,
    ]
    feature_output = score_frame.copy()
    feature_output["prediction_id"] = feature_output.apply(
        lambda row: f"formal_v2:{row['biological_instance_id']}:{row['predictor']}:{row['scenario']}", axis=1
    )
    for column in feature_columns:
        if column not in feature_output:
            feature_output[column] = ""
    feature_output[feature_columns].to_csv(feature_path, index=False)
    metric_frame.sort_values(["aggregation_level", "scenario", "method", "environment_id"], kind="stable").to_csv(metrics_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "prediction_source": "formal_v2_exact_linear_20260907_validation_test_only",
        "split_id": "ptl_biological_instance_split_v1",
        "calibration_split": "validation",
        "final_evaluation_split": "test",
        "scenarios": list(SCENARIOS),
        "predictors_executed": sorted(frame["predictor"].unique().tolist()),
        "predictors_not_executed": ["official_gears", "cpa", "scgpt", "prescribe"],
        "feature_blocks": {"U": list(UQ_COLUMNS), "P": list(P_COLUMNS), "S": list(S_COLUMNS), "N": list(N_COLUMNS), "C": list(C_NUMERIC_COLUMNS + C_CATEGORICAL_COLUMNS)},
        "feature_policy": "identity_free_shared_estimator_train_only_uq_normalization",
        "forbidden_headline_features": ["environment_id", "environment_key", "predictor", "fidelity", "continuous_risk", "reliable_label"],
        "rows": int(len(score_frame)),
        "metric_rows": int(len(metric_frame)),
        "score_path": score_path.relative_to(root).as_posix(),
        "feature_path": feature_path.relative_to(root).as_posix(),
        "metrics_path": metrics_path.relative_to(root).as_posix(),
        "status": "formal_v2_reliability_executed",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
