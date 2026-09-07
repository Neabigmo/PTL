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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_predictors import (  # noqa: E402
    METADATA_COLUMNS,
    PREDICTOR_VERSIONS,
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
S_COLUMNS = (
    "exact_training_support_fraction",
    "component_training_support_fraction",
    "training_neighborhood_density",
)
N_COLUMNS = (
    "perturbation_seen_fraction",
    "component_seen_fraction",
    "combination_novelty",
    "perturbation_novelty",
)
C_NUMERIC_COLUMNS: tuple[str, ...] = ()
C_CATEGORICAL_COLUMNS = (
    "cell_context",
    "perturbation_modality",
    "readout_modality",
    "condition",
    "platform",
)
FEATURE_COLUMNS = UQ_COLUMNS + P_COLUMNS + S_COLUMNS + N_COLUMNS + C_NUMERIC_COLUMNS + C_CATEGORICAL_COLUMNS
PREDICTORS = ("mean_matching", "strong_linear", "slim_string")
SCENARIOS = ("in_domain", "leave_environment_out", "leave_predictor_out")
METHODS = (
    "raw_normalized_uq",
    "best_scalar_uq",
    "platt_logistic",
    "isotonic",
    "u_only_rf",
    "u_p_rf",
    "u_ps_rf",
    "u_psn_rf",
    "ptl_rf",
)
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


def _numeric_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    fill_values: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    values = []
    fills: dict[str, float] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        fill = (
            float(fill_values[column])
            if fill_values is not None and column in fill_values
            else float(series.median()) if series.notna().any() else 0.0
        )
        fills[column] = fill
        values.append(series.fillna(fill).to_numpy(dtype=np.float64))
    matrix = np.column_stack(values) if values else np.empty((len(frame), 0), dtype=np.float64)
    return matrix, fills


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
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame, pd.DataFrame]:
    train_vectors = np.stack(train_frame["prediction_vector"].to_numpy())
    query_vectors = np.stack(query_frame["prediction_vector"].to_numpy())
    train_manifold = _manifold_distance(train_vectors, train_vectors, exclude_self=True)
    query_manifold = _manifold_distance(train_vectors, query_vectors)
    train_work = train_frame.copy()
    query_work = query_frame.copy()
    train_work["prediction_manifold_distance"] = train_manifold
    query_work["prediction_manifold_distance"] = query_manifold

    train_uq, query_uq = _normalized_uq(train_work, query_work)
    train_p, p_fills = _numeric_frame(train_work, P_COLUMNS)
    query_p, _ = _numeric_frame(query_work, P_COLUMNS, p_fills)
    train_s, s_fills = _numeric_frame(train_work, S_COLUMNS)
    query_s, _ = _numeric_frame(query_work, S_COLUMNS, s_fills)
    train_n, n_fills = _numeric_frame(train_work, N_COLUMNS)
    query_n, _ = _numeric_frame(query_work, N_COLUMNS, n_fills)
    train_c_numeric, c_fills = _numeric_frame(train_work, C_NUMERIC_COLUMNS)
    query_c_numeric, _ = _numeric_frame(query_work, C_NUMERIC_COLUMNS, c_fills)
    train_categories, query_categories = _fold_local_one_hot(
        train_work[list(C_CATEGORICAL_COLUMNS)].astype(str).to_numpy(),
        query_work[list(C_CATEGORICAL_COLUMNS)].astype(str).to_numpy(),
    )
    train_blocks = {
        "U": train_uq,
        "P": train_p,
        "S": train_s,
        "N": train_n,
        "C": np.column_stack([train_c_numeric, train_categories]),
    }
    query_blocks = {
        "U": query_uq,
        "P": query_p,
        "S": query_s,
        "N": query_n,
        "C": np.column_stack([query_c_numeric, query_categories]),
    }
    block_sets = {
        "U": ("U",),
        "U+P": ("U", "P"),
        "U+P+S": ("U", "P", "S"),
        "U+P+S+N": ("U", "P", "S", "N"),
        "U+P+S+N+C": ("U", "P", "S", "N", "C"),
    }
    matrices = {
        name: (
            np.column_stack([train_blocks[block] for block in blocks]),
            np.column_stack([query_blocks[block] for block in blocks]),
        )
        for name, blocks in block_sets.items()
    }
    materialized_train = pd.DataFrame(index=train_work.index)
    materialized_query = pd.DataFrame(index=query_work.index)
    for index, column in enumerate(UQ_COLUMNS):
        materialized_train[column] = train_uq[:, index]
        materialized_query[column] = query_uq[:, index]
    for matrix, columns, output in (
        (train_p, P_COLUMNS, materialized_train),
        (query_p, P_COLUMNS, materialized_query),
        (train_s, S_COLUMNS, materialized_train),
        (query_s, S_COLUMNS, materialized_query),
        (train_n, N_COLUMNS, materialized_train),
        (query_n, N_COLUMNS, materialized_query),
    ):
        for index, column in enumerate(columns):
            output[column] = matrix[:, index]
    for column in C_NUMERIC_COLUMNS + C_CATEGORICAL_COLUMNS:
        materialized_train[column] = train_work[column].astype(str).to_numpy()
        materialized_query[column] = query_work[column].astype(str).to_numpy()
    return matrices, materialized_train, materialized_query


def _fit_random_forest(train_features: np.ndarray, query_features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if np.unique(labels).size < 2:
        return np.full(len(query_features), float(labels.mean()), dtype=np.float64)
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=20260907,
        n_jobs=1,
    )
    model.fit(train_features, labels)
    return model.predict_proba(query_features)[:, 1]


def _fit_ptl(
    train_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], float, str, pd.DataFrame, pd.DataFrame]:
    threshold = 0.8 * float(np.median(train_frame["fidelity"].to_numpy(dtype=float)))
    train_labels = (train_frame["fidelity"].to_numpy(dtype=float) >= threshold).astype(int)
    matrices, materialized_train, materialized_query = _feature_matrix(train_frame, query_frame)
    train_uq = matrices["U"][0]
    query_uq = matrices["U"][1]
    train_risk = train_frame["continuous_risk"].to_numpy(dtype=float)
    scalar_scores = {
        f"uq_{column}": train_uq[:, index]
        for index, column in enumerate(UQ_COLUMNS)
    }
    query_scalar_scores = {
        f"uq_{column}": query_uq[:, index]
        for index, column in enumerate(UQ_COLUMNS)
    }
    scalar_aurcs = {
        name: _selective_metrics(train_risk, train_labels, values)["aurc"]
        for name, values in scalar_scores.items()
    }
    best_scalar_name = min(scalar_aurcs, key=lambda name: (scalar_aurcs[name], name))
    best_scalar_train = scalar_scores[best_scalar_name]
    best_scalar_query = query_scalar_scores[best_scalar_name]
    if np.unique(train_labels).size < 2:
        platt = np.full(len(query_frame), float(train_labels.mean()), dtype=np.float64)
        isotonic = platt.copy()
        model_name = "constant_validation_label"
    else:
        logistic = LogisticRegression(C=1.0, max_iter=1000, random_state=20260907)
        logistic.fit(best_scalar_train.reshape(-1, 1), train_labels)
        platt = logistic.predict_proba(best_scalar_query.reshape(-1, 1))[:, 1]
        isotonic_model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        isotonic_model.fit(best_scalar_train, train_labels)
        isotonic = isotonic_model.predict(best_scalar_query)
        model_name = "shared_identity_free_random_forest"
    scores = {
        "raw_normalized_uq": query_uq[:, UQ_COLUMNS.index("uq_mean_gene_variance")],
        "best_scalar_uq": best_scalar_query,
        "platt_logistic": platt,
        "isotonic": isotonic,
        "u_only_rf": _fit_random_forest(*matrices["U"], train_labels),
        "u_p_rf": _fit_random_forest(*matrices["U+P"], train_labels),
        "u_ps_rf": _fit_random_forest(*matrices["U+P+S"], train_labels),
        "u_psn_rf": _fit_random_forest(*matrices["U+P+S+N"], train_labels),
        "ptl_rf": _fit_random_forest(*matrices["U+P+S+N+C"], train_labels),
    }
    materialized_train.attrs["best_scalar_uq"] = best_scalar_name
    materialized_query.attrs["best_scalar_uq"] = best_scalar_name
    return scores, threshold, model_name, materialized_train, materialized_query


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
        train_count = max(1, len(train_manifest))
        train_label_counts = train_manifest["perturbation_label"].astype(str).value_counts().to_dict()
        train_label_components = {
            str(label): set(components(label))
            for label in train_label_counts
        }
        train_labels_by_component: dict[str, set[str]] = {}
        for label, label_parts in train_label_components.items():
            for part in label_parts:
                train_labels_by_component.setdefault(part, set()).add(label)
        n_unique_train_labels = max(1, len(train_label_components))
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
                    exact_support = float(train_label_counts.get(label, 0) / train_count)
                    supported_labels = set().union(
                        *(train_labels_by_component.get(part, set()) for part in parts)
                    ) if parts else set()
                    component_support = float(
                        sum(train_label_counts.get(name, 0) for name in supported_labels) / train_count
                    )
                    neighborhood_density = float(len(supported_labels) / n_unique_train_labels)
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
                        "predictor_version": PREDICTOR_VERSIONS[predictor],
                        "split": split,
                        "perturbation_label": label,
                        "fidelity": float(safe_rowwise_cosine(truth_vector[None, :], prediction_vector[None, :])[0]),
                        "continuous_risk": float(1.0 - safe_rowwise_cosine(truth_vector[None, :], prediction_vector[None, :])[0]),
                        "exact_training_support_fraction": exact_support,
                        "component_training_support_fraction": component_support,
                        "training_neighborhood_density": neighborhood_density,
                        "perturbation_seen_fraction": float(label in train_labels),
                        "component_seen_fraction": float(len(known_parts) / len(parts)) if parts else 1.0,
                        "combination_novelty": float(len(parts) > 1 and label not in train_labels),
                        "perturbation_novelty": float(label not in train_labels),
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


def _aurc_only(risk: np.ndarray, confidence: np.ndarray) -> float:
    """Fast AURC-only path used inside bootstrap resampling."""

    order = np.argsort(-np.asarray(confidence, dtype=float), kind="mergesort")
    ordered_risk = np.asarray(risk, dtype=float)[order]
    return float((np.cumsum(ordered_risk) / np.arange(1, len(ordered_risk) + 1)).mean())


def _paired_hierarchical_bootstrap(
    score_frame: pd.DataFrame,
    scenario: str,
    method: str,
    *,
    baseline: str = "raw_normalized_uq",
    replicates: int = 500,
) -> dict[str, Any]:
    """Bootstrap paired selective-risk deltas by environment and row.

    The outer resample preserves environment-level dependence; the inner
    resample preserves the fact that rows within an environment share a
    calibration/data context. Both methods use the same sampled rows, so the
    interval is paired rather than a difference of independent estimates.
    """

    selected = score_frame.loc[score_frame["scenario"].eq(scenario)].copy()
    cluster_columns = ["environment_id"]
    if scenario == "leave_predictor_out":
        cluster_columns.append("heldout_predictor")
    groups = [group for _, group in selected.groupby(cluster_columns, sort=True)]
    if not groups:
        raise ValueError(f"no score groups available for bootstrap scenario {scenario}")

    def grouped_aurc(column: str) -> float:
        return float(np.mean([
            _aurc_only(
                group["continuous_risk"].to_numpy(dtype=float),
                group[column].to_numpy(dtype=float),
            )
            for group in groups
        ]))

    observed_baseline = grouped_aurc(baseline)
    observed_method = grouped_aurc(method)
    rng = np.random.default_rng(20260907 + sum(ord(char) for char in f"{scenario}:{method}"))
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        baseline_values: list[float] = []
        method_values: list[float] = []
        for group_index in sampled_groups:
            group = groups[int(group_index)]
            row_indices = rng.integers(0, len(group), size=len(group))
            sampled = group.iloc[row_indices]
            baseline_values.append(_aurc_only(
                sampled["continuous_risk"].to_numpy(dtype=float),
                sampled[baseline].to_numpy(dtype=float),
            ))
            method_values.append(_aurc_only(
                sampled["continuous_risk"].to_numpy(dtype=float),
                sampled[method].to_numpy(dtype=float),
            ))
        deltas[replicate] = float(np.mean(method_values) - np.mean(baseline_values))
    return {
        "aggregation_level": "paired_hierarchical_macro_ci",
        "scenario": scenario,
        "method": method,
        "baseline": baseline,
        "estimate_delta_aurc": observed_method - observed_baseline,
        "baseline_aurc": observed_baseline,
        "method_aurc": observed_method,
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
        "n_clusters": len(groups),
        "bootstrap_replicates": replicates,
    }


def run(root: Path) -> dict[str, Any]:
    validate_deployment_features(FEATURE_COLUMNS)
    frame = _prediction_rows(root)
    score_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    def execute(scenario: str, heldout: str, subset: pd.DataFrame) -> None:
        train, test = scenario_partition(subset, scenario, heldout)
        train_frame = subset.loc[train].copy()
        test_frame = subset.loc[test].copy()
        method_scores, threshold, model_name, materialized_train, materialized_test = _fit_ptl(
            train_frame, test_frame
        )
        train_ids = set(train_frame["biological_instance_id"].astype(str))
        if train_ids.intersection(set(test_frame["biological_instance_id"].astype(str))):
            raise AssertionError("PTL train/test biological instance leakage")
        local_score_rows: list[dict[str, Any]] = []
        for row_index in test_frame.index:
            row = test_frame.loc[row_index].drop(labels=["prediction_vector"])
            payload = row.to_dict()
            native_uq = {
                f"native_{column}": float(row[column])
                for column in UQ_COLUMNS
            }
            transformed_features = materialized_test.loc[row_index].to_dict()
            payload.update({
                "scenario": scenario,
                "heldout_environment_id": heldout if scenario == "leave_environment_out" else "",
                "heldout_predictor": heldout if scenario == "leave_predictor_out" else "",
                "reliable_label": int(float(row["fidelity"]) >= threshold),
                "calibration_threshold": float(threshold),
                "calibration_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "ptl_model": model_name,
            })
            payload.update(native_uq)
            # The formal feature artifact must contain the values consumed by
            # the estimator: train-only normalized UQ and the scenario-specific
            # train/query transforms, not the raw target-side metadata.
            payload.update(transformed_features)
            for method, values in method_scores.items():
                payload[method] = float(values[list(test_frame.index).index(row_index)])
            score_rows.append(payload)
            local_score_rows.append(payload)
        local_output = pd.DataFrame(local_score_rows)
        for environment_id, group_output in local_output.groupby("environment_id", sort=True):
            for method in METHODS:
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

    bootstrap_rows = [
        _paired_hierarchical_bootstrap(score_frame, scenario, method)
        for scenario in SCENARIOS
        for method in METHODS
        if method != "raw_normalized_uq"
    ]
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    source_dir = root / "artifacts/source_data"
    manifest_dir = root / "artifacts/manifests"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    score_path = source_dir / "formal_v2_reliability_predictions.csv"
    feature_path = source_dir / "formal_v2_reliability_deployment_features.csv"
    metrics_path = manifest_dir / "formal_v2_reliability_metrics.csv"
    bootstrap_path = manifest_dir / "formal_v2_reliability_bootstrap_ci.csv"
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
    bootstrap_frame.sort_values(["scenario", "method"], kind="stable").to_csv(bootstrap_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "prediction_source": "formal_v2_predictors_validation_test_only",
        "split_id": "ptl_biological_instance_split_v1",
        "calibration_split": "validation",
        "final_evaluation_split": "test",
        "scenarios": list(SCENARIOS),
        "predictors_executed": sorted(frame["predictor"].unique().tolist()),
        "predictors_not_executed": ["official_gears", "cpa", "scgpt", "prescribe"],
        "feature_blocks": {"U": list(UQ_COLUMNS), "P": list(P_COLUMNS), "S": list(S_COLUMNS), "N": list(N_COLUMNS), "C": list(C_NUMERIC_COLUMNS + C_CATEGORICAL_COLUMNS)},
        "feature_policy": "identity_free_shared_estimator_train_only_uq_normalization_deployment_only_support",
        "feature_transform": {
            "uq": "predictor-relative empirical confidence fitted on calibration rows only",
            "prediction_manifold_distance": "nearest calibration prediction in calibration-only standardized PCA space",
            "numeric_imputation": "calibration-row median",
            "categorical_encoding": "calibration-local one-hot with unknown-zero fallback",
        },
        "native_uq_audit_columns": [f"native_{column}" for column in UQ_COLUMNS],
        "methods": list(METHODS),
        "forbidden_headline_features": ["environment_id", "environment_key", "predictor", "fidelity", "continuous_risk", "reliable_label", "total_cells", "n_reference_groups", "source_reference_keys", "batch_distance"],
        "rows": int(len(score_frame)),
        "metric_rows": int(len(metric_frame)),
        "score_path": score_path.relative_to(root).as_posix(),
        "feature_path": feature_path.relative_to(root).as_posix(),
        "metrics_path": metrics_path.relative_to(root).as_posix(),
        "bootstrap_ci_path": bootstrap_path.relative_to(root).as_posix(),
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
