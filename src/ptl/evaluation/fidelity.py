"""Centralized, reproducible fidelity definitions for formal-v2 evaluation."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import safe_rowwise_cosine, rowwise_spearman


def delta_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Primary delta-response cosine used by PTL."""

    return safe_rowwise_cosine(y_true, y_pred)


def centered_delta_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Local centered-delta cosine diagnostic; not an external protocol claim."""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return safe_rowwise_cosine(true - true.mean(axis=1, keepdims=True), pred - pred.mean(axis=1, keepdims=True))


def environment_centroid_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Local environment-centroid cosine diagnostic."""

    return float(safe_rowwise_cosine(np.mean(y_true, axis=0, keepdims=True), np.mean(y_pred, axis=0, keepdims=True))[0])


def absolute_effect_rank_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Local absolute-effect rank agreement diagnostic."""

    return rowwise_spearman(np.abs(np.asarray(y_true)), np.abs(np.asarray(y_pred)))


def evaluate_fidelity_definitions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate all declared metrics without refitting a predictor."""

    primary = delta_cosine(y_true, y_pred)
    centered = centered_delta_cosine(y_true, y_pred)
    rank = absolute_effect_rank_agreement(y_true, y_pred)
    return {
        "delta_cosine": float(np.mean(primary)),
        "centered_delta_cosine": float(np.mean(centered)),
        "environment_centroid_cosine": environment_centroid_cosine(y_true, y_pred),
        "absolute_effect_rank_agreement": float(np.mean(rank)),
    }
