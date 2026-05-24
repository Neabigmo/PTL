from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.run_baseline import (
    PreparedSplitData,
    build_ridge_features,
    normalize_rows,
    parse_combination_components,
)
from src.evaluation.metrics import EPS, safe_rowwise_cosine


@dataclass
class ConfidenceContext:
    prepared_data: PreparedSplitData
    model_details: dict[str, Any]
    reference_gene_space: list[str]


def _support_scale(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    max_log = np.log1p(np.maximum(reference_values, 0.0)).max(initial=0.0)
    if max_log <= EPS:
        return np.ones_like(values, dtype=np.float64)
    return np.clip(np.log1p(np.maximum(values, 0.0)) / max_log, 0.0, 1.0)


def _reference_centroid_similarity(ref_train: np.ndarray, ref_query: np.ndarray) -> np.ndarray:
    if ref_train.shape[0] == 0 or ref_query.shape[0] == 0:
        return np.zeros(ref_query.shape[0], dtype=np.float64)
    centroid = ref_train.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    cosine = safe_rowwise_cosine(ref_query.astype(np.float32, copy=False), np.repeat(centroid, ref_query.shape[0], axis=0))
    return np.clip(cosine, 0.0, 1.0)


def _reference_key_support_map(train_frame: pd.DataFrame) -> dict[str, float]:
    grouped = train_frame.groupby("reference_key", sort=False)["n_cells"].sum()
    return {str(key): float(value) for key, value in grouped.items()}


def _perturbation_support_map(train_frame: pd.DataFrame) -> dict[str, float]:
    grouped = train_frame.groupby("perturbation_label", sort=False)["n_cells"].sum()
    return {str(key): float(value) for key, value in grouped.items()}


def control_mean_confidence(context: ConfidenceContext) -> np.ndarray:
    return np.zeros(len(context.prepared_data.test_frame), dtype=np.float64)


def global_delta_confidence(context: ConfidenceContext) -> np.ndarray:
    train_frame = context.prepared_data.train_frame
    test_frame = context.prepared_data.test_frame
    ref_similarity = _reference_centroid_similarity(context.prepared_data.ref_train, context.prepared_data.ref_test)
    reference_support = _reference_key_support_map(train_frame)
    reference_values = train_frame["n_cells"].to_numpy(dtype=np.float64)
    test_support = np.array([reference_support.get(str(key), 0.0) for key in test_frame["reference_key"].astype(str)], dtype=np.float64)
    support_scale = _support_scale(test_support, reference_values)
    return np.clip(ref_similarity * support_scale, 0.0, 1.0)


def perturbation_mean_confidence(context: ConfidenceContext) -> np.ndarray:
    train_frame = context.prepared_data.train_frame
    test_frame = context.prepared_data.test_frame.reset_index(drop=True)
    train_non_control = train_frame.loc[~train_frame["is_control"]].copy()
    ref_similarity = _reference_centroid_similarity(context.prepared_data.ref_train, context.prepared_data.ref_test)
    perturb_support = _perturbation_support_map(train_non_control)
    train_support_values = train_non_control["n_cells"].to_numpy(dtype=np.float64)
    singleton_support = {
        label: support
        for label, support in perturb_support.items()
        if not parse_combination_components(label)
    }
    output = np.zeros(len(test_frame), dtype=np.float64)
    for idx, row in enumerate(test_frame.itertuples(index=False)):
        if bool(row.is_control):
            output[idx] = min(1.0, 0.5 + 0.5 * ref_similarity[idx])
            continue
        perturbation = str(row.perturbation_label)
        if perturbation in perturb_support:
            base = 1.0
            support = perturb_support[perturbation]
        else:
            components = parse_combination_components(perturbation)
            if components and all(component in singleton_support for component in components):
                base = 0.6
                support = float(np.mean([singleton_support[component] for component in components]))
            else:
                base = 0.2
                support = 0.0
        support_term = _support_scale(np.array([support], dtype=np.float64), train_support_values)[0]
        output[idx] = base * (0.5 * ref_similarity[idx] + 0.5 * support_term)
    return np.clip(output, 0.0, 1.0)


def cell_context_knn_confidence(context: ConfidenceContext) -> np.ndarray:
    prepared = context.prepared_data
    train_frame = prepared.train_frame.loc[~prepared.train_frame["is_control"]].reset_index(drop=True)
    test_frame = prepared.test_frame.reset_index(drop=True)
    train_ref = prepared.ref_train[~prepared.train_frame["is_control"].to_numpy(dtype=bool)]
    query_ref = prepared.ref_test
    if train_ref.shape[0] == 0:
        return np.zeros(len(test_frame), dtype=np.float64)
    selected_k = int(context.model_details.get("selected_k", 5))
    normalized_train = normalize_rows(train_ref.astype(np.float32, copy=False))
    normalized_query = normalize_rows(query_ref.astype(np.float32, copy=False))
    train_support = train_frame["n_cells"].to_numpy(dtype=np.float64)
    support_scaled = _support_scale(train_support, train_support)
    train_perturbations = train_frame["perturbation_label"].astype(str).to_numpy()
    output = np.zeros(len(test_frame), dtype=np.float64)
    for idx, row in enumerate(test_frame.itertuples(index=False)):
        if bool(row.is_control):
            global_sim = float(np.max(normalized_train @ normalized_query[idx]))
            output[idx] = max(0.0, global_sim)
            continue
        perturbation = str(row.perturbation_label)
        candidate_mask = train_perturbations == perturbation
        if not candidate_mask.any():
            global_sim = float(np.max(normalized_train @ normalized_query[idx]))
            output[idx] = 0.1 * max(0.0, global_sim)
            continue
        sims = normalized_train[candidate_mask] @ normalized_query[idx]
        k = min(selected_k, sims.shape[0])
        nearest = np.argpartition(-sims, k - 1)[:k]
        nearest_sims = np.clip(sims[nearest], 0.0, 1.0)
        nearest_support = support_scaled[candidate_mask][nearest]
        weighted = np.average(nearest_sims, weights=np.maximum(nearest_support, EPS))
        output[idx] = weighted
    return np.clip(output, 0.0, 1.0)


def ridge_confidence(context: ConfidenceContext) -> np.ndarray:
    prepared = context.prepared_data
    train_frame = prepared.train_frame.loc[~prepared.train_frame["is_control"]].reset_index(drop=True)
    if train_frame.empty:
        return np.zeros(len(prepared.test_frame), dtype=np.float64)
    ref_train = prepared.ref_train[~prepared.train_frame["is_control"].to_numpy(dtype=bool)]
    ref_test = prepared.ref_test
    train_x, _, test_x, _, _ = build_ridge_features(
        train_frame=train_frame,
        val_frame=prepared.test_frame.reset_index(drop=True),
        test_frame=prepared.test_frame.reset_index(drop=True),
        train_ref=ref_train,
        val_ref=ref_test,
        test_ref=ref_test,
    )
    normalized_train = normalize_rows(train_x.astype(np.float32, copy=False))
    normalized_test = normalize_rows(test_x.astype(np.float32, copy=False))
    similarities = normalized_test @ normalized_train.T
    max_similarity = np.clip(np.max(similarities, axis=1), 0.0, 1.0)
    alpha = float(context.model_details.get("selected_alpha", 1.0))
    shrink = 1.0 / (1.0 + 0.5 * np.log10(alpha + 1.0))
    return np.clip(max_similarity * shrink, 0.0, 1.0)


def compute_model_specific_confidence(model_name: str, context: ConfidenceContext) -> np.ndarray:
    if model_name == "control_mean_baseline":
        return control_mean_confidence(context)
    if model_name == "global_delta_baseline":
        return global_delta_confidence(context)
    if model_name == "perturbation_mean_delta_baseline":
        return perturbation_mean_confidence(context)
    if model_name == "cell_context_knn_delta_baseline":
        return cell_context_knn_confidence(context)
    if model_name == "ridge_regression_baseline":
        return ridge_confidence(context)
    raise ValueError(f"Unsupported model_name '{model_name}' for confidence computation.")
